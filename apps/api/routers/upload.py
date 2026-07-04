"""
Upload router — attach clinical documents to an investigation.

POST /api/investigations/{case_id}/upload accepts a file (PDF, image),
runs it through Azure AI Document Intelligence (or the local fallback —
see packages/tools/doc_intel_tool.py::extract_clinical_document), and
appends the structured result to that investigation's `documents` list
in storage.

Current scope note: `InvestigationState.documents` is not yet read by
any graph node — no agent currently consumes uploaded documents. This
endpoint exists so the upload pipeline (ingest -> extract -> persist)
is real and testable ahead of the Retrieval Agent (Priority 3), which
is the intended future consumer of `documents` (feeding extracted text/
tables into evidence retrieval). Uploading to a case_id does not itself
trigger or restart the investigation graph.

Uploaded files are processed in-memory and written to a temp file only
for the duration of the Document Intelligence / PyPDF2 call — nothing
is persisted to disk beyond the extracted structured content already
stored via cosmos_client.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from packages.database.cosmos_client import load_investigation, save_investigation
from packages.tools.doc_intel_tool import extract_clinical_document

router = APIRouter(prefix="/investigations", tags=["upload"])

# Guard against accidentally loading huge files into memory. Clinical
# PDFs/scans are typically well under this; tune if real-world uploads
# need more headroom.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("/{case_id}/upload")
async def upload_document(case_id: str, file: UploadFile = File(...)) -> dict:
    """Upload a clinical document (PDF or image) for an investigation.
    Extracts structured content via Document Intelligence (or its local
    fallback) and appends it to the investigation's `documents` list."""
    state = load_investigation(case_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Investigation '{case_id}' not found.")

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = Path(file.filename or "").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        extracted = extract_clinical_document(tmp.file.name)

    extracted["original_filename"] = file.filename

    documents = list(state.get("documents", []))
    documents.append(extracted)
    state["documents"] = documents
    save_investigation(state)

    return {
        "case_id": case_id,
        "filename": file.filename,
        "extraction_source": extracted.get("source"),
        "pages": extracted.get("pages"),
        "tables_found": len(extracted.get("tables", [])),
    }
