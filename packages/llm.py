"""
Vultr Serverless Inference client factory.

Vultr Serverless Inference exposes an OpenAI-compatible API at
https://api.vultrinference.com/v1, so `langchain_openai.ChatOpenAI` works
directly against it with a custom `base_url`.

IMPORTANT — model ID is not yet verified against the live endpoint.
Vultr's own docs are explicit that the API `model` string must be read
from `GET /v1/models` with a real key — it does not necessarily match the
marketing name or the self-hosted vLLM `--model` flag shown in their
Inference Cookbook (that cookbook documents deploying on your own
NVIDIA HGX B200 cluster, not the managed Serverless Inference product).
The default below (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`) is the
best-evidenced candidate from Vultr's public docs as of this writing —
run `list_models()` against a real `VULTR_API_KEY` and correct
`VULTR_MODEL` before relying on this for anything beyond local dry-runs.
"""

from __future__ import annotations

import os

import requests
from langchain_openai import ChatOpenAI

VULTR_BASE_URL = "https://api.vultrinference.com/v1"

# Best-evidenced candidate, NOT yet confirmed live. See module docstring.
_DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"


def get_llm(temperature: float = 0.3, max_tokens: int = 4096) -> ChatOpenAI:
    """
    Build a ChatOpenAI client pointed at Vultr Serverless Inference.

    Low temperature (0.3 default) is deliberate for clinical reasoning —
    we want determinism, not creative variance, in a system whose core
    selling point is epistemic honesty over confident guessing.
    """
    api_key = os.getenv("VULTR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VULTR_API_KEY is not set. Callers should check "
            "`llm_available()` and fall back to the rule-based stub "
            "instead of calling get_llm() directly."
        )

    return ChatOpenAI(
        model=os.getenv("VULTR_MODEL", _DEFAULT_MODEL),
        base_url=VULTR_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def llm_available() -> bool:
    """True if a Vultr API key is configured. Agents use this to decide
    between the LLM path and the deterministic rule-based fallback."""
    return bool(os.getenv("VULTR_API_KEY"))


def list_models(api_key: str | None = None) -> list[dict]:
    """
    Hit the real Vultr Serverless Inference models endpoint and return the
    raw model list. Run this once with a real key to confirm the exact
    Nemotron model ID string before trusting `_DEFAULT_MODEL` /
    `VULTR_MODEL` for anything beyond local dry-runs.

    Usage:
        python -c "from packages.llm import list_models, print_nemotron_models; \\
                    print_nemotron_models(list_models())"
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
    # Response shape isn't 100% pinned down without a live call — handle
    # both a bare list and a {"data": [...]} / {"models": [...]} envelope.
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
