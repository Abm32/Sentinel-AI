"""
Investigations router — create, list, and fetch investigations.

POST /api/investigations kicks off the LangGraph investigation pipeline
as a FastAPI background task and returns the case_id immediately (the
graph run itself takes multiple LLM round-trips and should never block
the HTTP response). The running state is persisted to storage
(Cosmos DB, or local JSON fallback — see packages/database/cosmos_client.py)
after every graph step, so GET /api/investigations/{case_id} always
reflects current progress, and a client can poll it instead of holding
a connection open. For live/streaming updates instead of polling, see
the WebSocket endpoint (added in a later commit).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from packages.database.cosmos_client import list_investigations, load_investigation, save_investigation
from packages.graph import run_investigation
from packages.schemas.investigation_state import new_investigation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigations", tags=["investigations"])


class CreateInvestigationRequest(BaseModel):
    case_id: str = Field(description="Unique identifier for this investigation")
    incident: str = Field(description="Free-text clinical presentation / incident description")


class CreateInvestigationResponse(BaseModel):
    case_id: str
    status: str


def _execute_investigation(case_id: str, incident: str) -> None:
    """Runs synchronously inside a BackgroundTask (i.e. after the HTTP
    response has already been sent). Persists the state after every
    graph step so GET reflects live progress, and persists a final
    snapshot even if the graph raises partway through (best-effort —
    an investigation that crashed mid-run should still be inspectable,
    not silently vanish)."""
    last_state: dict = new_investigation(case_id=case_id, incident=incident)
    save_investigation(dict(last_state))

    try:
        for _node_name, state_update in run_investigation(case_id=case_id, incident=incident):
            last_state.update(state_update)
            save_investigation(dict(last_state))
    except Exception:
        logger.exception("Investigation %s failed during graph execution", case_id)
        last_state["status"] = "failed"
        save_investigation(dict(last_state))


@router.post("", response_model=CreateInvestigationResponse, status_code=202)
def create_investigation(
    request: CreateInvestigationRequest, background_tasks: BackgroundTasks
) -> CreateInvestigationResponse:
    """Start a new investigation. Returns immediately with the case_id;
    the graph runs in the background. Poll GET /investigations/{case_id}
    for progress, or use the WebSocket stream endpoint."""
    if load_investigation(request.case_id) is not None:
        raise HTTPException(
            status_code=409, detail=f"Investigation '{request.case_id}' already exists."
        )

    background_tasks.add_task(_execute_investigation, request.case_id, request.incident)

    return CreateInvestigationResponse(case_id=request.case_id, status="planning")


@router.get("")
def list_all_investigations() -> list[dict]:
    """List all investigations (across whichever backend is active)."""
    return list_investigations()


@router.get("/{case_id}")
def get_investigation(case_id: str) -> dict:
    """Fetch full investigation state (progress + final report, once
    available) by case_id."""
    state = load_investigation(case_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Investigation '{case_id}' not found.")
    return state
