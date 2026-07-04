"""
Multi-cloud availability checks.

Sentinel's design pattern (established by `packages/llm.py::llm_available`)
extends here: every external cloud dependency is optional. Each tool/
client module checks the relevant `*_available()` helper and falls back
to a deterministic/local implementation if credentials are absent. The
demo MUST work end-to-end with zero cloud credentials configured.

This module centralizes the Azure-side checks (Document Intelligence,
AI Search, Cosmos DB) so callers don't re-implement `os.getenv(...) and
os.getenv(...)` boilerplate in three different tool files.
"""

from __future__ import annotations

import os


def doc_intel_available() -> bool:
    """True if Azure AI Document Intelligence credentials are configured."""
    return bool(os.getenv("AZURE_DOC_INTEL_ENDPOINT")) and bool(
        os.getenv("AZURE_DOC_INTEL_KEY")
    )


def search_available() -> bool:
    """True if Azure AI Search credentials are configured."""
    return bool(os.getenv("AZURE_SEARCH_ENDPOINT")) and bool(os.getenv("AZURE_SEARCH_KEY"))


def cosmos_available() -> bool:
    """True if Azure Cosmos DB credentials are configured."""
    return bool(os.getenv("AZURE_COSMOS_ENDPOINT")) and bool(os.getenv("AZURE_COSMOS_KEY"))


def azure_available() -> bool:
    """True if ANY Azure service is configured. Individual tools should
    prefer their specific `*_available()` check above — this exists for
    a single "is Azure wired up at all" signal (e.g. for a health-check
    endpoint or a startup log line)."""
    return doc_intel_available() or search_available() or cosmos_available()
