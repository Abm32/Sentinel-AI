"""
Investigations router — create, list, fetch, and stream investigations.

POST /api/investigations kicks off the LangGraph investigation pipeline
as a FastAPI background task and returns the case_id immediately (the
graph run itself takes multiple LLM round-trips and should never block
the HTTP response). The running state is persisted to storage
(Cosmos DB, or local JSON fallback — see packages/database/cosmos_client.py)
after every graph step, so GET /api/investigations/{case_id} always
reflects current progress, and a client can poll it instead of holding
a connection open.

For live updates without polling, WS /api/investigations/{case_id}/stream
subscribes to the same background run's node-by-node events via
apps/api/events.py's in-memory broadcaster and forwards them to the
client as they happen — this is the demo centerpiece: watching agents
work in real time instead of refreshing a GET endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from apps.api import events
from packages.database.cosmos_client import list_investigations, load_investigation, save_investigation
from packages.graph import run_investigation
from packages.schemas.investigation_state import new_investigation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigations", tags=["investigations"])


class CreateInvestigationRequest(BaseModel):
    case_id: str = Field(description="Unique identifier for this investigation")
    incident: str = Field(description="Free-text clinical presentation / incident description")
    retrieved_evidence: list[dict] | None = Field(
        default=None,
        description=(
            "Optional structured evidence to seed before the graph starts "
            "(e.g. a genomic report already on file: {'source': "
            "'genomic_report', 'gene': 'DPYD', 'phenotype': 'Poor "
            "Metabolizer'}). Free text in `incident` is NOT parsed into "
            "structured evidence by any node — this is currently the only "
            "way to make a patient-specific phenotype visible to "
            "tool_agent.py::_find_phenotype via the API."
        ),
    )


class CreateInvestigationResponse(BaseModel):
    case_id: str
    status: str


def _execute_investigation(
    case_id: str, incident: str, retrieved_evidence: list[dict] | None = None
) -> None:
    """Runs synchronously inside a BackgroundTask (i.e. after the HTTP
    response has already been sent). Persists the state after every
    graph step so GET reflects live progress, and publishes each step
    to apps/api/events.py so any connected WebSocket client sees it
    live. Persists a final snapshot even if the graph raises partway
    through (best-effort — an investigation that crashed mid-run should
    still be inspectable, not silently vanish).

    The POST handler persists an initial placeholder state before
    scheduling this task so WebSocket clients can subscribe immediately;
    load that snapshot here rather than overwriting it."""
    existing = load_investigation(case_id)
    if existing is not None:
        last_state = dict(existing)
    else:
        last_state = new_investigation(case_id=case_id, incident=incident)
        if retrieved_evidence:
            last_state["retrieved_evidence"] = list(retrieved_evidence)
        save_investigation(dict(last_state))

    try:
        for node_name, state_update in run_investigation(
            case_id=case_id, incident=incident, retrieved_evidence=retrieved_evidence
        ):
            last_state.update(state_update)
            save_investigation(dict(last_state))
            events.publish(case_id, node_name, state_update)
    except Exception:
        logger.exception("Investigation %s failed during graph execution", case_id)
        last_state["status"] = "failed"
        save_investigation(dict(last_state))
    finally:
        events.finish(case_id)


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

    # Persist a placeholder immediately so a WebSocket client that
    # connects right after this 202 response does not race the
    # BackgroundTask's first save and get closed with 4404.
    initial_state = new_investigation(case_id=request.case_id, incident=request.incident)
    if request.retrieved_evidence:
        initial_state["retrieved_evidence"] = list(request.retrieved_evidence)
    save_investigation(dict(initial_state))

    background_tasks.add_task(
        _execute_investigation, request.case_id, request.incident, request.retrieved_evidence
    )

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


@router.websocket("/{case_id}/stream")
async def stream_investigation(websocket: WebSocket, case_id: str) -> None:
    """
    Stream live investigation progress over a WebSocket.

    Each message sent to the client has the shape:
        {"node": "<node_name>", "state_update": {...}, "done": false}
    and a final message:
        {"done": true, "state": {...full final state...}}

    If the investigation has already finished (or failed) by the time
    the client connects, the current state is sent immediately and the
    connection closes — there is no backlog replay of intermediate
    steps, only the current/final snapshot (GET /investigations/{case_id}
    is the source of truth for full history via review_history etc.).

    If the investigation doesn't exist at all, the connection is closed
    with code 4404 immediately (WebSocket close codes can't reuse HTTP
    status codes directly, so 4404 is used as a recognizable custom
    application-level code mirroring the HTTP 404 used elsewhere in this
    router).
    """
    await websocket.accept()

    state = load_investigation(case_id)
    if state is None:
        await websocket.close(code=4404, reason=f"Investigation '{case_id}' not found.")
        return

    if state.get("status") in ("completed", "failed"):
        await websocket.send_json({"done": True, "state": state})
        await websocket.close()
        return

    queue = events.subscribe(case_id)

    # Re-check state after subscribing: the background task may have
    # finished (and called events.finish()) in the gap between the
    # initial load_investigation() above and subscribe() just now — the
    # rule-based fallback path in particular runs fast enough for this
    # race to be real, not theoretical. If it already finished, drain
    # nothing and report done immediately rather than waiting forever
    # on a queue that will never receive another event.
    current_state = load_investigation(case_id)
    if current_state and current_state.get("status") in ("completed", "failed"):
        events.unsubscribe(case_id, queue)
        await websocket.send_json({"done": True, "state": current_state})
        await websocket.close()
        return

    try:
        while True:
            item = await queue.get()
            if item is events.DONE:
                final_state = load_investigation(case_id) or state
                await websocket.send_json({"done": True, "state": final_state})
                break
            node_name, state_update = item
            await websocket.send_json({"node": node_name, "state_update": state_update, "done": False})
    except WebSocketDisconnect:
        return
    finally:
        events.unsubscribe(case_id, queue)

    await websocket.close()
