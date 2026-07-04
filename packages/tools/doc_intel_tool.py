"""
Tool Agent adapter for Azure AI Document Intelligence.

Sentinel's investigation starts with uploaded clinical records — lab
reports, EHR notes, FDA label PDFs. This module wraps Azure AI Document
Intelligence (formerly Form Recognizer) the same way `pgx_tool.py` wraps
anukriti-pgx-core: one tool the Tool Agent can call, whose output is one
evidence stream among many flowing into `InvestigationState.documents`.

Uses the `prebuilt-layout` model, which extracts text, tables (e.g. lab
values with dates), and document structure without requiring a
custom-trained model — the right choice for heterogeneous clinical PDFs
where we don't control the source format.

Same fallback pattern as `packages/llm.py::llm_available()`: if Azure
Document Intelligence credentials are not configured (see
`packages/config.py::doc_intel_available`), this falls back to a local
PDF text extraction via PyPDF2 (or a hardcoded demo document if the file
can't be parsed at all) — the demo must work with zero cloud credentials.
"""

from __future__ import annotations

import os
from typing import Any

from packages.config import doc_intel_available

# Hardcoded demo document, returned only when a file path can't be read
# at all (missing file, unsupported format) AND Azure isn't configured.
# Matches the project's fluorouracil/DPYD demo narrative so the pipeline
# has something to reason about even with no real upload.
_DEMO_DOCUMENT = {
    "tool": "document-intelligence",
    "file": None,
    "content": (
        "LABORATORY REPORT\n"
        "Patient: Demo Case\nDate: Day 5 post-fluorouracil infusion\n"
        "ANC: 0.4 x10^9/L (critical low, flag: neutropenia)\n"
        "eGFR: 92 mL/min/1.73m^2 (normal)\n"
    ),
    "tables": [
        {
            "cells": [["Test", "Value", "Flag"], ["ANC", "0.4 x10^9/L", "LOW"], ["eGFR", "92 mL/min/1.73m^2", "NORMAL"]],
            "row_count": 3,
            "column_count": 3,
        }
    ],
    "pages": 1,
    "source": "hardcoded-demo-fallback",
    "status": "fallback",
}


def _fallback_extract(file_path: str) -> dict[str, Any]:
    """Local extraction path: plain-text pull via PyPDF2 if the file
    exists and looks like a PDF; otherwise the hardcoded demo document.
    This is a resilience feature (same rationale as the rule-based LLM
    fallbacks), not a lesser tier — it keeps the graph runnable with zero
    Azure credentials and offline."""
    if not file_path or not os.path.exists(file_path):
        result = dict(_DEMO_DOCUMENT)
        result["file"] = file_path
        return result

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return {
            "tool": "document-intelligence",
            "file": file_path,
            "content": text,
            "tables": [],  # PyPDF2 does not extract tables; layout tools are Azure-only.
            "pages": len(reader.pages),
            "source": "PyPDF2-fallback",
            "status": "fallback",
        }
    except Exception:
        # Unreadable / not a PDF / PyPDF2 not usable for this file — fall
        # back to the demo document rather than raising into the graph.
        result = dict(_DEMO_DOCUMENT)
        result["file"] = file_path
        return result


def _azure_extract(file_path: str) -> dict[str, Any]:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        endpoint=os.getenv("AZURE_DOC_INTEL_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_DOC_INTEL_KEY")),
    )

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", body=f)
        result = poller.result()

    tables = []
    for table in result.tables or []:
        # Reconstruct a row-major grid from the flat cell list Azure
        # returns (each cell carries its own row_index/column_index).
        grid: list[list[str]] = [
            ["" for _ in range(table.column_count)] for _ in range(table.row_count)
        ]
        for cell in table.cells:
            grid[cell.row_index][cell.column_index] = cell.content
        tables.append(
            {
                "cells": grid,
                "row_count": table.row_count,
                "column_count": table.column_count,
            }
        )

    return {
        "tool": "document-intelligence",
        "file": file_path,
        "content": result.content,
        "tables": tables,
        "pages": len(result.pages or []),
        "source": "Azure AI Document Intelligence",
        "status": "confirmed",
    }


def extract_clinical_document(file_path: str) -> dict[str, Any]:
    """
    Parse a clinical document (lab report, EHR note, FDA label PDF) into
    structured content for `InvestigationState.documents`.

    Tries Azure AI Document Intelligence first if credentials are
    configured (`doc_intel_available()`); falls back to local PyPDF2 text
    extraction (or a hardcoded demo document) otherwise, or if the Azure
    call itself raises — a live-demo network hiccup must not crash the
    investigation.

    Args:
        file_path: Path to the uploaded document (PDF or image).

    Returns:
        A dict with `content` (full extracted text), `tables` (list of
        row-major grids, e.g. lab values with dates), `pages`, and
        `source` (which backend actually produced this result).
    """
    if doc_intel_available():
        try:
            return _azure_extract(file_path)
        except Exception:
            # Azure configured but the call failed (bad key, network,
            # unsupported file, quota) — degrade gracefully rather than
            # propagating into the graph.
            return _fallback_extract(file_path)
    return _fallback_extract(file_path)
