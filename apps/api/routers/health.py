"""
Health check router.

Reports whether the process is up, and which optional cloud backends are
currently configured (Vultr LLM, Azure Document Intelligence/Search/
Cosmos). None of these being configured is not itself unhealthy — every
one of them has a deterministic/local fallback (see packages/llm.py and
packages/config.py) — this endpoint just gives the frontend/ops a way to
see which mode each subsystem is running in.
"""

from __future__ import annotations

from fastapi import APIRouter

from packages.config import cosmos_available, doc_intel_available, search_available
from packages.llm import llm_available

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "backends": {
            "llm": "vultr" if llm_available() else "rule-based-fallback",
            "document_intelligence": "azure" if doc_intel_available() else "local-fallback",
            "evidence_search": "azure" if search_available() else "local-fallback",
            "investigation_storage": "cosmos" if cosmos_available() else "local-json-fallback",
        },
    }
