"""
Vultr Serverless Inference client + prompt-based structured output.

TWO-MODEL ARCHITECTURE. Sentinel Clinical uses two different model
families on Vultr Serverless Inference, for two different jobs:

  1. VultronRetriever (packages/tools/vultron_rerank_tool.py) — evidence
     reranking, via `/v1/rerank`. Confirmed against Vultr's own docs to
     be a visual document retrieval / reranking model family with NO
     chat-completion capability — it cannot plan, reason, or generate
     text. It is not used by this module.
  2. A chat-completion model (this module) — the actual reasoning
     engine behind every agent node that needs judgment: planner,
     hypothesis, reporter, reviewer. This is a standard
     `/v1/chat/completions` call against Vultr Serverless Inference,
     same endpoint and API key as VultronRetriever, different model ID.

Vultr's chat-completion docs state tool calling is currently supported
ONLY on `kimi-k2-instruct`:
https://docs.vultr.com/products/serverless/inference/management/usage/chat

LangChain's `with_structured_output()` relies on function calling under
the hood, so it cannot be trusted against the default chat model on this
endpoint without confirming tool-calling support first. Instead, every
LLM node in this project uses `llm_json_call()`: a system prompt
embedding the target JSON schema, a plain chat completion, then manual
extraction + Pydantic validation, with one retry (schema feedback
appended to the prompt) before falling back to None. This works against
any OpenAI-compatible endpoint regardless of native function-calling
support — so it's correct regardless of which chat model is configured.

MODEL ID — STILL UNVERIFIED AGAINST THE LIVE ENDPOINT.
The default below (`moonshotai/kimi-k2-instruct`) is chosen because
Vultr's own docs confirm it exists on Serverless Inference and supports
tool calling (linked above), and Vultr's VultronRetriever guide's
multimodal RAG example pairs VultronRetriever with `moonshotai/Kimi-K2.6`
as a working chat model on the same endpoint. Neither the exact current
model ID string nor its availability under your specific subscription
has been confirmed via a live `GET /v1/models` call in this session.
Vultr's model naming has changed before. DO NOT treat this as ground
truth. Run `list_models()` below with a real VULTR_API_KEY and set
VULTR_CHAT_MODEL to the confirmed ID before relying on this for anything
beyond local dry-runs with the rule-based fallback active.
"""

from __future__ import annotations

import json
import os
import re

import httpx
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, APITimeoutError
from pydantic import BaseModel, ValidationError

VULTR_BASE_URL = "https://api.vultrinference.com/v1"

# UNVERIFIED default — see module docstring. Override via VULTR_CHAT_MODEL.
_DEFAULT_CHAT_MODEL = "moonshotai/kimi-k2-instruct"

# Process-wide cached client — see get_llm()'s docstring for why this
# isn't rebuilt on every call. Deliberately module-level, not a
# functools.lru_cache: get_llm() takes temperature/max_tokens args, but
# every real call site uses the defaults, so a plain "build once" cache
# is simpler and avoids a subtle bug where lru_cache would build a
# SEPARATE client (and separate httpx connection pool) per distinct
# argument combination rather than one truly shared pool.
_llm_client: ChatOpenAI | None = None


def llm_available() -> bool:
    """True if a Vultr API key is configured. Agents use this to decide
    between the LLM path and the deterministic rule-based fallback."""
    return bool(os.getenv("VULTR_API_KEY"))


def get_llm(temperature: float = 0.3, max_tokens: int = 16000) -> ChatOpenAI:
    """
    Build a ChatOpenAI client pointed at Vultr Serverless Inference,
    using the chat-completion reasoning model (NOT VultronRetriever —
    see module docstring).

    Low temperature (0.3 default) is deliberate for clinical reasoning —
    determinism over creative variance.

    WHY `max_tokens=16000`, NOT the previous 4096 — this is the actual
    root cause of what first looked like a hang. `moonshotai/Kimi-K2.6`
    is a reasoning ("thinking") model: it spends completion tokens on a
    hidden chain-of-thought (surfaced as `message.reasoning` in the raw
    API response) BEFORE emitting the actual answer in `message.content`.
    Measured directly against this project's own Hypothesis Agent
    prompt: at `max_tokens=4096` (the old default), the model was cut off
    mid-`reasoning` at `finish_reason="length"` with `completion_tokens=
    4096` and `content=None` — every single time, not intermittently. It
    never once reached the JSON answer. Vultr's endpoint does NOT honor
    the OpenAI-compatible `chat_template_kwargs: {"enable_thinking":
    false}` override either (confirmed empirically — same token usage,
    same `content=None`, matching a known open vLLM issue where Kimi's
    reasoning parser ignores that flag). So the fix is not to suppress
    reasoning, it's to give the model enough room to finish reasoning
    AND write the answer: the same prompt completes in ~20s with
    `finish_reason="stop"` and real JSON content once `max_tokens` is
    raised to the 12000-16000 range (reasoning + JSON together measured
    at ~4300 completion tokens; 16000 leaves comfortable headroom for
    longer prompts/responses elsewhere in the graph, e.g. the Reviewer's
    multi-issue rejections). This was being masked as "the LLM call
    hangs" because `llm_json_call`'s retry loop would silently retry the
    same doomed request against the same too-small budget every time.

    `timeout=30` was the first fix attempted for the apparent hang and
    is still useful as a genuine network-level safety net (a truly
    stuck connection, not a slow-but-progressing one), but it was NOT
    sufficient on its own — a request that takes ~20-25s to actually
    complete at the correct `max_tokens` value would have kept getting
    cut off and retried at 30s if the completion took slightly longer
    under load, so the timeout is widened to 60s here to give a
    real-but-slow response room to land instead of being killed and
    retried right as it was about to finish.

    Separately: `llm_json_call()` previously called this function fresh
    on every single invocation, and every LLM-backed agent node
    (planner, hypothesis, reporter, reviewer) calls `llm_json_call()`
    independently — so a single multi-pass investigation (reject ->
    re-investigate can run up to `_MAX_RETRIES` times) created and never
    closed up to ~16 separate `ChatOpenAI`/httpx client instances.
    Observed directly on the deployed Vultr VM: `ss -tnp` showed 10
    connections to Vultr's inference endpoint stuck in CLOSE-WAIT (the
    remote side closed, our side never called `close()`) while the
    FastAPI worker sat idle in `epoll_pwait` with zero events and its
    thread pool blocked on `futex` waits — a genuinely dead background
    task, not a slow one, on an investigation that had made real
    progress (one reject/retry cycle already completed) before going
    silent. A single float `timeout=` also collapses connect/read/write/
    pool timeouts into one value via the OpenAI SDK's httpx wrapper,
    which is coarser than intended; explicit `httpx.Timeout` below
    separates "can't establish a connection" (10s — should be near-
    instant or clearly broken) from "connected but no response body yet"
    (60s — the actual slow-reasoning-model case this function's docstring
    describes above). `get_llm()` is now memoized (one client per
    process, reused across every node and every retry pass) so
    connections are pooled and reused by httpx's own keep-alive logic
    instead of a fresh, never-closed client being created per call.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    api_key = os.getenv("VULTR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VULTR_API_KEY is not set. Callers should check "
            "`llm_available()` and fall back to the rule-based stub "
            "instead of calling get_llm() / llm_json_call() directly."
        )

    _llm_client = ChatOpenAI(
        model=os.getenv("VULTR_CHAT_MODEL", _DEFAULT_CHAT_MODEL),
        base_url=VULTR_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=httpx.Timeout(60.0, connect=10.0),
        max_retries=1,
    )
    return _llm_client


def llm_json_call(
    system_prompt: str,
    user_prompt: str,
    output_model: type[BaseModel],
    retries: int = 2,
) -> BaseModel | None:
    """
    Call the LLM, extract JSON from its response, and parse it into
    `output_model`. No function calling — prompt-based structured output
    only, since tool-calling support on Vultr Serverless Inference is
    restricted to specific models (kimi-k2-instruct, per Vultr's docs)
    and this project doesn't want to depend on the configured chat
    model supporting it.

    Returns None if all retries fail; callers MUST handle this by falling
    back to their rule-based implementation, not by propagating an
    exception into the graph.
    """
    llm = get_llm()

    schema = output_model.model_json_schema()
    full_system = f"""{system_prompt}

You MUST respond with valid JSON only. No markdown, no code fences, no commentary.
The JSON must conform to this schema:

{json.dumps(schema, indent=2)}

Return ONLY the JSON object."""

    current_user_prompt = user_prompt

    for attempt in range(retries + 1):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=full_system),
                    HumanMessage(content=current_user_prompt),
                ]
            )
            # Defense in depth against the exact failure mode get_llm()'s
            # docstring describes: a reasoning model (Kimi-K2.6) that
            # spends its entire token budget on hidden chain-of-thought
            # and never emits an answer returns `content=None` (not an
            # empty string) with `finish_reason="length"`. The real fix
            # is get_llm()'s max_tokens headroom, but `None.strip()`
            # would raise AttributeError here — an exception this
            # function's own docstring promises callers never have to
            # handle — if that budget is ever exhausted anyway (a longer
            # prompt than anticipated, a particularly verbose reasoning
            # pass, etc). Treat it exactly like a parse failure: retry,
            # then fall back to None.
            if response.content is None:
                if attempt < retries:
                    current_user_prompt = f"""{user_prompt}

Your previous response was truncated before producing an answer. Please respond with valid JSON matching the schema, more concisely."""
                    continue
                return None

            raw = response.content.strip()

            # Strip markdown code fences if present (models sometimes
            # add them despite being told not to).
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)
            return output_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < retries:
                current_user_prompt = f"""{user_prompt}

Your previous response failed to parse: {e}
Please respond with valid JSON matching the schema."""
            else:
                return None
        except (APITimeoutError, APIConnectionError, APIError) as e:
            # Network-level failure (the 30s timeout on get_llm() firing,
            # a connection error, or a non-2xx from Vultr) is a DIFFERENT
            # failure mode from a malformed JSON response, and was
            # previously NOT caught here — it propagated straight out of
            # llm_json_call and crashed the calling agent node, taking
            # the whole graph run down with it (observed in practice:
            # hypothesis_node's call timed out and killed
            # `python -m packages.graph` with an unhandled
            # openai.APITimeoutError, exactly the "propagating an
            # exception into the graph" this function's own docstring
            # says callers must never have to deal with). Treat it the
            # same as a parse failure: retry with the same prompt (no
            # schema-feedback rewrite needed, nothing was wrong with the
            # prompt), then return None so every caller's existing
            # `if result is None: fall back to rule-based` path handles
            # it exactly like any other LLM failure.
            if attempt >= retries:
                return None

    return None


def list_models(api_key: str | None = None) -> list[dict]:
    """
    Hit the real Vultr Serverless Inference models endpoint and return
    the raw model list. Run this once with a real key to confirm the
    exact chat-model ID string before trusting `_DEFAULT_CHAT_MODEL` /
    `VULTR_CHAT_MODEL` for anything beyond local dry-runs, and to
    confirm the VultronRetriever tier IDs
    (packages/tools/vultron_rerank_tool.py) are available under your
    subscription.

    Usage:
        python -m packages.llm
    """
    key = api_key or os.getenv("VULTR_API_KEY")
    if not key:
        raise RuntimeError("No API key provided and VULTR_API_KEY is not set.")

    response = requests.get(
        f"{VULTR_BASE_URL}/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    # Response shape isn't pinned down without a live call — handle both
    # a bare list and a {"data": [...]} / {"models": [...]} envelope.
    if isinstance(payload, list):
        return payload
    return payload.get("data") or payload.get("models") or []


def print_model_catalog(models: list[dict]) -> None:
    """Prints the chat-model candidates and VultronRetriever tiers found
    in the live catalog, so a real deployment can pick a confirmed
    VULTR_CHAT_MODEL and confirm the rerank tier IDs before relying on
    either."""
    vultron_matches = [m for m in models if "vultronretriever" in str(m).lower()]
    kimi_matches = [m for m in models if "kimi" in str(m).lower()]

    print("VultronRetriever tiers found in the live catalog:")
    if vultron_matches:
        for m in vultron_matches:
            print(f"  - {m}")
    else:
        print("  (none found)")

    print("kimi-k2 chat model candidates found in the live catalog:")
    if kimi_matches:
        for m in kimi_matches:
            print(f"  - {m}")
    else:
        print("  (none found — pick another chat-capable model from the full list below)")

    print("\nFull model list:")
    for m in models:
        print(f"  - {m}")


if __name__ == "__main__":
    print_model_catalog(list_models())
