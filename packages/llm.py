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

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

VULTR_BASE_URL = "https://api.vultrinference.com/v1"

# UNVERIFIED default — see module docstring. Override via VULTR_CHAT_MODEL.
_DEFAULT_CHAT_MODEL = "moonshotai/kimi-k2-instruct"


def llm_available() -> bool:
    """True if a Vultr API key is configured. Agents use this to decide
    between the LLM path and the deterministic rule-based fallback."""
    return bool(os.getenv("VULTR_API_KEY"))


def get_llm(temperature: float = 0.3, max_tokens: int = 4096) -> ChatOpenAI:
    """
    Build a ChatOpenAI client pointed at Vultr Serverless Inference,
    using the chat-completion reasoning model (NOT VultronRetriever —
    see module docstring).

    Low temperature (0.3 default) is deliberate for clinical reasoning —
    determinism over creative variance.
    """
    api_key = os.getenv("VULTR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VULTR_API_KEY is not set. Callers should check "
            "`llm_available()` and fall back to the rule-based stub "
            "instead of calling get_llm() / llm_json_call() directly."
        )

    return ChatOpenAI(
        model=os.getenv("VULTR_CHAT_MODEL", _DEFAULT_CHAT_MODEL),
        base_url=VULTR_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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
