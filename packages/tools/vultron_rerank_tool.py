"""
VultronRetriever adapter — evidence reranking via Vultr Serverless
Inference's `/v1/rerank` endpoint.

VultronRetriever (Flash-0.8B / Core-4.5B / Prime-8B, all Qwen3.5-based)
is a visual document retrieval model family — #1 on the ViDoRe V3
benchmark. It reads a document the way a person does: given a query and
a set of documents (plain text, or page images passed as OpenAI-style
`image_url` content parts), it scores how relevant each one is to the
query, taking in full page layout — tables, charts, scans — with no OCR
step required for the image path.

IMPORTANT — confirmed against Vultr's own documentation
(docs.vultr.com/how-to-rank-documents-with-vultronretriever-on-vultr-
serverless-inference, fetched 2026-07-04): VultronRetriever is exposed
ONLY via `POST /v1/rerank`. It has no chat-completion capability and no
embeddings endpoint (`/v1/embeddings` returns 404 for these models). It
cannot generate text, JSON, or plan/reason about anything — it can only
score relevance. This is why Sentinel's reasoning nodes (planner,
hypothesis, reporter, reviewer — see packages/llm.py) use a separate
Vultr-hosted chat-completion model, not VultronRetriever. VultronRetriever
is Sentinel's evidence-retrieval engine; the chat model is its reasoning
engine. Both run on Vultr Serverless Inference, same endpoint, same key.

This module currently reranks plain-text evidence chunks (the shape
produced by packages/tools/retrieval_tool.py and
packages/tools/doc_intel_tool.py). Passing document *page images*
straight to VultronRetriever (its primary designed use case — scoring
relevance directly from rendered layout, catching lab tables and charts
that text extraction misses) is a natural next step once the Retrieval
Agent has access to rendered page images rather than only extracted
text; the request/response shape here already accepts that path (see
`_to_rerank_document`), it's just not yet wired to a page-image source.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from packages.config import rerank_available

VULTR_RERANK_URL = "https://api.vultrinference.com/v1/rerank"

# The three VultronRetriever tiers, confirmed model ID strings (Vultr
# docs, fetched 2026-07-04). Flash is the documented default: "fast and
# cost-efficient, with strong quality" — Core/Prime trade latency for
# accuracy on denser or more cluttered documents.
_DEFAULT_RERANK_MODEL = "vultr/VultronRetrieverFlash-Qwen3.5-0.8B"

# Vultr's rerank endpoint request body is capped at ~1MB; this module
# only sends short text chunks (not page images), so this limit is not
# expected to bind in practice, but is documented here since it's a real
# constraint of the API this module calls.
_MAX_REQUEST_BYTES = 900_000


def _to_rerank_document(evidence: dict[str, Any]) -> str | dict[str, Any]:
    """Convert one evidence dict to the shape /v1/rerank expects for a
    single document. Text documents are plain strings. Page-image
    documents (not yet produced anywhere in this codebase) would use
    `{"content": [{"type": "image_url", "image_url": {"url": "data:..."}}]}`
    instead — see module docstring."""
    return evidence.get("content", "")


def _fallback_passthrough(query: str, documents: list[dict]) -> list[dict]:
    """No API key configured, or the live call failed: return the
    candidates in their original order, tagged so callers/logs can see
    reranking was skipped rather than silently assuming it happened."""
    result = []
    for doc in documents:
        entry = dict(doc)
        entry["reranked_by"] = "none (pass-through fallback)"
        result.append(entry)
    return result


def rerank_evidence(
    query: str,
    documents: list[dict[str, Any]],
    model: str | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank candidate evidence chunks by relevance to `query` using
    VultronRetriever.

    Args:
        query: The investigation question / search query (e.g. "DPYD
            deficiency symptoms neutropenia").
        documents: List of evidence dicts, each expected to carry a
            `content` field (plain text). Any other fields are passed
            through unchanged into the result.
        model: VultronRetriever model ID to use. Defaults to
            `VULTR_RERANK_MODEL` env var, then `_DEFAULT_RERANK_MODEL`
            (Flash tier).
        top_n: If given, caps the number of results returned to the
            `top_n` highest-scoring documents (matches the API's own
            `top_n` parameter).

    Returns:
        `documents`, each augmented with `relevance_score` (float) and
        `reranked_by` (which model did it, or the fallback marker),
        sorted highest score first. If VultronRetriever isn't
        configured or the call fails, returns the original order
        unscored (`reranked_by` marks this explicitly) rather than
        raising into the graph.
    """
    if not documents:
        return []

    if not rerank_available():
        return _fallback_passthrough(query, documents)

    model_id = model or os.getenv("VULTR_RERANK_MODEL", _DEFAULT_RERANK_MODEL)
    payload: dict[str, Any] = {
        "model": model_id,
        "query": query,
        "documents": [_to_rerank_document(d) for d in documents],
    }
    if top_n is not None:
        payload["top_n"] = top_n

    try:
        response = requests.post(
            VULTR_RERANK_URL,
            headers={
                "Authorization": f"Bearer {os.getenv('VULTR_API_KEY')}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError):
        # Network failure, timeout, non-2xx, or unparseable JSON — degrade
        # to pass-through rather than breaking the Retrieval Agent node.
        return _fallback_passthrough(query, documents)

    scored: list[dict[str, Any]] = []
    for item in body.get("results", []):
        idx = item.get("index")
        if idx is None or not (0 <= idx < len(documents)):
            continue
        entry = dict(documents[idx])
        entry["relevance_score"] = item.get("relevance_score")
        entry["reranked_by"] = model_id
        scored.append(entry)

    if not scored:
        # Malformed/empty response body — same reasoning as the
        # exception path above.
        return _fallback_passthrough(query, documents)

    return sorted(scored, key=lambda d: d.get("relevance_score", 0.0), reverse=True)
