"""
Vultr Serverless Inference client + prompt-based structured output.

Vultr's chat-completion docs state tool calling is currently supported
ONLY on `kimi-k2-instruct`:
https://docs.vultr.com/products/serverless/inference/management/usage/chat

LangChain's `with_structured_output()` relies on function calling under
the hood, so it cannot be trusted against Nemotron on this endpoint.
Instead, every LLM node in this project uses `llm_json_call()`: a system
prompt embedding the target JSON schema, a plain chat completion, then
manual extraction + Pydantic validation, with one retry (schema
feedback appended to the prompt) before falling back to None. This
works against any OpenAI-compatible endpoint regardless of native
function-calling support.

MODEL ID — STILL UNVERIFIED AGAINST THE LIVE ENDPOINT.
Two candidate strings have surfaced from different sources, and neither
has been confirmed via a real `GET /v1/models` call:
  - "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8" — found directly in
    Vultr's own "Model Guides" docs, but in the context of self-hosting
    via vLLM on a Vultr GPU instance (Inference Cookbook), not
    confirmed as the managed Serverless Inference API's `model` string.
  - "nvidia/nemotron-3-nano-omni" — the marketing/product name for
    "Nemotron 3 Nano Omni" on Vultr Serverless Inference; the exact API
    string was not independently confirmed against docs during this
    session.
Vultr's model naming has changed before. DO NOT treat either as ground
truth. Run `list_models()` below with a real VULTR_API_KEY and correct
VULTR_MODEL before relying on this for anything beyond local dry-runs
with the rule-based fallback active.
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

# UNVERIFIED default — see module docstring. Override via VULTR_MODEL.
_DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"


def llm_available() -> bool:
    """True if a Vultr API key is configured. Agents use this to decide
    between the LLM path and the deterministic rule-based fallback."""
    return bool(os.getenv("VULTR_API_KEY"))


def get_llm(temperature: float = 0.3, max_tokens: int = 4096) -> ChatOpenAI:
    """
    Build a ChatOpenAI client pointed at Vultr Serverless Inference.

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
        model=os.getenv("VULTR_MODEL", _DEFAULT_MODEL),
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
    only, since Vultr's tool-calling support is restricted to
    kimi-k2-instruct and Nemotron is the model this project is committed
    to using.

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
    exact Nemotron model ID string before trusting `_DEFAULT_MODEL` /
    `VULTR_MODEL` for anything beyond local dry-runs.

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


def print_nemotron_models(models: list[dict]) -> None:
    matches = [m for m in models if "nemotron" in str(m).lower()]
    if not matches:
        print("No Nemotron models found in the live catalog.")
        return
    print("Nemotron models found in the live Vultr Serverless Inference catalog:")
    for m in matches:
        print(f"  - {m}")


if __name__ == "__main__":
    print_nemotron_models(list_models())
