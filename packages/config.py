"""
Multi-cloud availability checks.

Sentinel's design pattern (established by `packages/llm.py::llm_available`)
extends here: every external cloud dependency is optional. Each tool/
client module checks the relevant `*_available()` helper and falls back
to a deterministic/local implementation if credentials are absent. The
demo MUST work end-to-end with zero cloud credentials configured.

This module centralizes the Azure-side checks (Document Intelligence,
AI Search, Cosmos DB) and the VultronRetriever rerank check, so callers
don't re-implement `os.getenv(...) and os.getenv(...)` boilerplate in
every tool file.
"""

from __future__ import annotations

import os
import re

# Matches the placeholder tokens shipped in .env.example (e.g.
# "https://<your-resource>.cognitiveservices.azure.com/",
# "https://<your-account>.documents.azure.com:443/"). A `*_available()`
# check that only tests "is this env var non-empty" is not enough —
# copying .env.example to .env and filling in only the *_KEY values
# (while leaving the endpoint placeholder untouched) makes every
# `os.getenv(...)` check above return a non-empty string, so the
# service reports "available", every live call then fails against a
# literal "<your-resource>" hostname, and the tool silently degrades to
# its local fallback. That's worse than reporting unavailable: the
# health endpoint (apps/api/routers/health.py) would claim "azure" is
# the active backend when it never actually worked.
_PLACEHOLDER_PATTERN = re.compile(r"<|your-|example|placeholder", re.IGNORECASE)


def _is_real_endpoint(var_name: str) -> bool:
    """True if `var_name` is set to something other than an empty string
    or an obvious copy-pasted placeholder from .env.example."""
    val = os.getenv(var_name, "")
    if not val:
        return False
    if _PLACEHOLDER_PATTERN.search(val):
        return False
    return True


def rerank_available() -> bool:
    """True if a Vultr API key is configured for VultronRetriever
    reranking (packages/tools/vultron_rerank_tool.py). Uses the same
    VULTR_API_KEY as the chat-completion reasoning model
    (packages/llm.py) — both are Vultr Serverless Inference, same key,
    same endpoint, different model IDs."""
    return bool(os.getenv("VULTR_API_KEY"))


def doc_intel_available() -> bool:
    """True if Azure AI Document Intelligence credentials are configured
    with a real (non-placeholder) endpoint."""
    return _is_real_endpoint("AZURE_DOC_INTEL_ENDPOINT") and bool(
        os.getenv("AZURE_DOC_INTEL_KEY")
    )


def search_available() -> bool:
    """True if Azure AI Search credentials are configured with a real
    (non-placeholder) endpoint."""
    return _is_real_endpoint("AZURE_SEARCH_ENDPOINT") and bool(os.getenv("AZURE_SEARCH_KEY"))


def cosmos_available() -> bool:
    """True if Azure Cosmos DB credentials are configured with a real
    (non-placeholder) endpoint."""
    return _is_real_endpoint("AZURE_COSMOS_ENDPOINT") and bool(os.getenv("AZURE_COSMOS_KEY"))


def azure_available() -> bool:
    """True if ANY Azure service is configured. Individual tools should
    prefer their specific `*_available()` check above — this exists for
    a single "is Azure wired up at all" signal (e.g. for a health-check
    endpoint or a startup log line)."""
    return doc_intel_available() or search_available() or cosmos_available()
