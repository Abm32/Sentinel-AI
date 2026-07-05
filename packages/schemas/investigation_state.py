"""
InvestigationState — the LangGraph state channel threaded through every
node of the investigation graph.

This is a TypedDict, not a Pydantic BaseModel. LangGraph's reducer
mechanism (`Annotated[list, operator.add]`) only composes correctly with
TypedDict state channels: each node returns a partial dict of the fields
it changed, and LangGraph merges that into the running state — for
`Annotated[..., operator.add]` fields, by concatenation; for plain fields,
by overwrite. Nodes must NOT mutate state in place (e.g.
`state["tool_outputs"].append(...)`) — return the new items instead and
let the reducer merge them. See `packages/agents/*.py` for the pattern.

For structured *output* validation (e.g. the final report, or a single
tool result before it's stored), use a separate Pydantic model — don't
validate the LangGraph state channel itself.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict


class InvestigationStatus(str, Enum):
    PLANNING = "planning"
    INVESTIGATING = "investigating"
    REVIEWING = "reviewing"
    COMPLETED = "completed"


class InvestigationState(TypedDict):
    case_id: str
    incident: str  # patient presentation / incident description, free text

    documents: list[dict]  # uploaded records; replaced wholesale, not appended
    tasks: Annotated[list[dict], operator.add]  # Planner's investigation plan
    retrieved_evidence: Annotated[list[dict], operator.add]
    timeline: Annotated[list[dict], operator.add]
    hypotheses: Annotated[list[dict], operator.add]
    tool_outputs: Annotated[list[dict], operator.add]
    verified_facts: Annotated[list[dict], operator.add]
    contradictions: Annotated[list[dict], operator.add]
    review_history: Annotated[list[dict], operator.add]
    review_issues: Annotated[list[dict], operator.add]  # reviewer -> tool_agent feedback

    retry_count: int
    confidence: Optional[float]
    report: Optional[dict[str, Any]]
    status: InvestigationStatus

    # Gates retrieval_2 (packages/agents/retrieval_2.py) to run exactly
    # once per investigation, not once per Reviewer reject ->
    # re-investigate pass. Without this, every retry loop would re-run
    # a full VultronRetriever hypothesis-validation pass even though the
    # top hypothesis rarely changes shape between retries (only its
    # supporting evidence grows) -- one targeted validation pass per
    # investigation is the intended design, not one per retry.
    hypothesis_validated: bool


def new_investigation(case_id: str, incident: str) -> InvestigationState:
    """Construct a fresh, fully-initialized InvestigationState.

    LangGraph does not require every key to be present up front, but
    reducer fields (Annotated[list, operator.add]) need a starting list to
    concatenate onto — passing an InvestigationState missing those keys
    works for `operator.add` (LangGraph treats a missing key as its type's
    identity for the reducer on first write) but initializing explicitly
    here avoids relying on that and keeps `graph.invoke(...)` inputs
    self-documenting.
    """
    return InvestigationState(
        case_id=case_id,
        incident=incident,
        documents=[],
        tasks=[],
        retrieved_evidence=[],
        timeline=[],
        hypotheses=[],
        tool_outputs=[],
        verified_facts=[],
        contradictions=[],
        review_history=[],
        review_issues=[],
        retry_count=0,
        confidence=None,
        report=None,
        status=InvestigationStatus.PLANNING,
        hypothesis_validated=False,
    )
