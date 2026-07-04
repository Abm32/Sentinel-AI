"""
Tool Agent / Retrieval Agent adapter for Azure AI Search.

Backs semantic evidence retrieval: given a query (a task from the
Planner, e.g. "DPYD phenotype fluorouracil"), find relevant evidence
chunks from the indexed clinical evidence base (documents previously
extracted by `doc_intel_tool.py` and indexed into Azure AI Search).

Same fallback pattern as every other cloud dependency in this project
(see `packages/config.py`): if Azure AI Search credentials are not
configured, or the live call fails, this falls back to a simple
keyword-overlap match over a small hardcoded evidence corpus — enough
to keep the Retrieval Agent node runnable end-to-end with zero cloud
credentials.
"""

from __future__ import annotations

import os
import re
from typing import Any

from packages.config import search_available

_INDEX_NAME = "sentinel-evidence"

# Hardcoded evidence corpus for the fallback path. Mirrors the project's
# fluorouracil/DPYD demo narrative — matches what tool_agent.py's
# hardcoded lab stub and pgx_tool.py's real DPYD/fluorouracil mapping
# already assume, so a fallback-mode Retrieval Agent surfaces evidence
# consistent with the rest of the pipeline.
_FALLBACK_CORPUS: list[dict[str, Any]] = [
    {
        "content": (
            "CPIC guideline: DPYD poor metabolizers are at significantly "
            "increased risk of severe, life-threatening fluoropyrimidine "
            "(5-FU, capecitabine) toxicity including neutropenia, "
            "mucositis, and diarrhea. Avoid fluorouracil or reduce dose "
            "per activity score."
        ),
        "source": "CPIC Guideline DPYD-Fluoropyrimidines",
        "doc_type": "cpic_guideline",
    },
    {
        "content": (
            "Lab trend: ANC declining to 0.4 x10^9/L by Day 5 post-"
            "infusion, consistent with cytotoxic marrow suppression. "
            "eGFR 92 mL/min/1.73m^2, within normal range."
        ),
        "source": "lab_trends",
        "doc_type": "lab_report",
    },
    {
        "content": (
            "FDA label, fluorouracil: Patients with DPD (DPYD) deficiency "
            "may experience severe, life-threatening toxicity. Testing for "
            "DPD deficiency prior to initiation should be considered."
        ),
        "source": "FDA Label - Fluorouracil",
        "doc_type": "fda_label",
    },
    {
        "content": (
            "Drug interaction check: no clinically significant "
            "interactions identified between fluorouracil and other "
            "medications in the current medication list."
        ),
        "source": "drug_interaction_check",
        "doc_type": "interaction_check",
    },
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _fallback_search(query: str, top_k: int) -> list[dict[str, Any]]:
    """Simple keyword-overlap ranking over the hardcoded corpus — no
    embeddings, no external calls. Deterministic and dependency-free."""
    query_terms = _tokenize(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in _FALLBACK_CORPUS:
        doc_terms = _tokenize(doc["content"] + " " + doc["source"])
        overlap = len(query_terms & doc_terms)
        score = overlap / max(len(query_terms), 1)
        scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        if score <= 0:
            continue
        results.append(
            {
                "content": doc["content"],
                "source": doc["source"],
                "score": round(score, 3),
                "doc_type": doc.get("doc_type", ""),
            }
        )
    return results


def _azure_search(query: str, top_k: int) -> list[dict[str, Any]]:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=_INDEX_NAME,
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
    )

    results = client.search(
        search_text=query,
        top=top_k,
        query_type="semantic",
        semantic_configuration_name="default",
    )

    return [
        {
            "content": doc.get("content", ""),
            "source": doc.get("source", ""),
            "score": doc.get("@search.score", 0.0),
            "doc_type": doc.get("doc_type", ""),
        }
        for doc in results
    ]


def search_evidence(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Semantic search across the clinical evidence index.

    Called by the Retrieval Agent to find evidence relevant to a
    Planner task or hypothesis. Tries Azure AI Search first if
    credentials are configured (`search_available()`); falls back to
    keyword-overlap matching over a small hardcoded evidence corpus
    otherwise, or if the live call fails.

    Args:
        query: Free-text search query (e.g. a task description or
            hypothesis title).
        top_k: Maximum number of evidence chunks to return.

    Returns:
        A list of evidence dicts: `content`, `source`, `score`,
        `doc_type`.
    """
    if search_available():
        try:
            return _azure_search(query, top_k)
        except Exception:
            return _fallback_search(query, top_k)
    return _fallback_search(query, top_k)


def index_document(document: dict[str, Any]) -> bool:
    """
    Index an extracted document (output of
    `doc_intel_tool.extract_clinical_document`) into Azure AI Search as
    an evidence chunk, so future `search_evidence()` calls can retrieve
    it.

    No-op (returns False) when Azure AI Search is not configured — in
    fallback mode, evidence retrieval reads from the hardcoded corpus
    instead, so there's nothing to index into.
    """
    if not search_available():
        return False

    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=_INDEX_NAME,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
        )
        doc_id = re.sub(r"[^A-Za-z0-9_\-=]", "_", str(document.get("file") or document.get("source", "doc")))
        client.upload_documents(
            documents=[
                {
                    "id": doc_id,
                    "content": document.get("content", ""),
                    "source": document.get("source", document.get("file", "unknown")),
                    "doc_type": document.get("tool", "document"),
                }
            ]
        )
        return True
    except Exception:
        return False
