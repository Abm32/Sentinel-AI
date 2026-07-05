"""
The investigation graph: START -> planner -> retrieval -> tool_agent ->
hypothesis -> retrieval_2 -> hypothesis -> reporter -> reviewer ->
(approved: END | rejected: back to tool_agent).

Run directly for a smoke test covering all three demo scenarios:

    python -m packages.graph

Scenario 1 (Path A, first pass): DPYD Poor Metabolizer phenotype
available -> pgx-core AVOID -> 76% hypothesis -> draft report -> Reviewer
REJECTS (confidence_inflated, single evidence source) -> loops back to
tool_agent.

Scenario 2 (Path A, full loop): same as above, but the graph runs to
completion — rejection triggers a second tool_agent pass that adds the
lab_trends stub (because the Reviewer's issue requested it), hypothesis
re-scores with the added evidence, Reporter regenerates, Reviewer
APPROVES. This is the full demo arc: investigate -> report -> reject ->
re-investigate -> approve -> finalize.

Scenario 3 (Path B, the refusal): no phenotype evidence -> pgx-core
returns insufficient_evidence -> zero-confidence UNCONFIRMED hypothesis ->
draft report says "cannot conclude without genotype" -> Reviewer APPROVES
the refusal on the first pass (no retry needed) — approving honest
uncertainty, not rejecting it.

Retrieval Agent (packages/agents/retrieval.py) sits between planner and
tool_agent: it runs the Planner's evidence-search tasks (lab trends, FDA
label, CPIC guidelines, drug interactions, medication history) through
`search_evidence()` and writes results to `retrieved_evidence`. It does
NOT re-run on the reject -> reinvestigate loop back into tool_agent (see
build_graph()'s edges below) — only tool_agent and downstream nodes
re-run on that path, matching the existing re-investigation contract
(pgx-core/lab_trends dedup logic in tool_agent.py, `_already_retrieved`
dedup in retrieval.py guards it regardless if that ever changes).

Retrieval Agent, round 2 (packages/agents/retrieval_2.py) sits between
the Hypothesis Agent's first pass and the Reporter: once a leading
hypothesis exists, it builds hypothesis-specific supporting/
contradicting queries and reranks evidence against them via
VultronRetriever, then the graph loops back into `hypothesis` to
re-score with that targeted evidence before continuing to the Reporter.
This is the "retrieves more than once when it needs to" behavior the
Retrieval Agent's round-1 pass alone cannot provide — round 1 runs
before any hypothesis exists, so it can only gather general evidence,
never evidence targeted at a specific proposed explanation.
`hypothesis_validated` gates this to run once per investigation, not
once per Reviewer reject -> re-investigate pass (see
InvestigationState and retrieval_2.py's module docstrings).
"""

from __future__ import annotations

import json

from langgraph.graph import END, StateGraph

from packages.agents.hypothesis import hypothesis_node
from packages.agents.planner import planner_node
from packages.agents.reporter import reporter_node
from packages.agents.retrieval import retrieval_node
from packages.agents.retrieval_2 import retrieval_2_node
from packages.agents.reviewer import reviewer_node
from packages.agents.tool_agent import tool_agent_node
from packages.schemas.investigation_state import InvestigationState, new_investigation

_MAX_RETRIES = 3


def review_router(state: InvestigationState) -> str:
    """Conditional edge out of the reviewer node."""
    latest = state["review_history"][-1]
    if latest["verdict"] == "approved":
        return "end"
    if state["retry_count"] >= _MAX_RETRIES:
        # Safety cap — don't loop forever if evidence can never satisfy
        # the Reviewer. Ends the graph with the last rejected state
        # (report will be None; caller can inspect review_history to see
        # why the investigation stalled).
        return "end"
    return "reinvestigate"


def hypothesis_validation_router(state: InvestigationState) -> str:
    """Conditional edge out of the first hypothesis pass. Routes to
    retrieval_2 exactly once per investigation (see
    InvestigationState.hypothesis_validated); every subsequent pass
    through this node (i.e. every Reviewer reject -> re-investigate ->
    hypothesis re-score) skips straight to the Reporter instead of
    re-running the validation retrieval pass."""
    if state.get("hypothesis_validated"):
        return "reporter"
    return "retrieval_2"


def build_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("retrieval_2", retrieval_2_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "tool_agent")
    graph.add_edge("tool_agent", "hypothesis")

    graph.add_conditional_edges(
        "hypothesis",
        hypothesis_validation_router,
        {
            "retrieval_2": "retrieval_2",
            "reporter": "reporter",
        },
    )
    # retrieval_2 always loops back into hypothesis to re-score with the
    # newly targeted evidence. hypothesis_validated is already True by
    # this point (retrieval_2_node sets it before returning), so the
    # second pass through hypothesis_validation_router above routes
    # straight to reporter rather than back into retrieval_2 again.
    graph.add_edge("retrieval_2", "hypothesis")

    graph.add_edge("reporter", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "end": END,
            "reinvestigate": "tool_agent",
        },
    )

    return graph.compile()


_INCIDENT_TEXT = (
    "Patient admitted after fluorouracil therapy. Symptoms: neutropenia, "
    "mucositis, diarrhea, fever."
)


def run_path_a() -> InvestigationState:
    """Confirmed path, full loop: reject (thin evidence) -> re-investigate
    -> approve -> finalize. This is Scenario 2, the full demo arc."""
    app = build_graph()
    state = new_investigation(case_id="demo-case-A", incident=_INCIDENT_TEXT)
    state["retrieved_evidence"] = [
        {
            "source": "genomic_report",
            "gene": "DPYD",
            "phenotype": "Poor Metabolizer",
        }
    ]
    return app.invoke(state)


def run_path_b() -> InvestigationState:
    """Refusal path: no genomic-phenotype evidence retrieved. Reviewer
    approves the refusal on the first pass — no retry needed."""
    app = build_graph()
    state = new_investigation(case_id="demo-case-B", incident=_INCIDENT_TEXT)
    return app.invoke(state)


def run_investigation(
    case_id: str,
    incident: str,
    documents: list | None = None,
    retrieved_evidence: list | None = None,
):
    """
    Generator that runs the investigation graph and yields a state
    snapshot after each node completes.

    Used by the API layer (apps/api/routers/investigations.py) to drive
    both the synchronous CRUD create path (consume the generator fully,
    persist the final state) and the WebSocket streaming endpoint (yield
    each snapshot to the client as it arrives, so the dashboard can show
    agents working live).

    Each yielded item is `(node_name, state_update)` — LangGraph's
    `graph.stream()` event shape: a dict of `{node_name:
    partial_state_update}` per step. We unpack it to a single tuple per
    step since exactly one node runs per step in this graph (no parallel
    branches).

    Args:
        case_id: Unique investigation identifier.
        incident: Free-text clinical presentation.
        documents: Optional pre-loaded documents (e.g. output of
            `doc_intel_tool.extract_clinical_document`) to seed
            `InvestigationState.documents` before the graph starts.
        retrieved_evidence: Optional pre-seeded evidence entries (e.g. a
            structured genomic report already on file:
            `{"source": "genomic_report", "gene": "DPYD", "phenotype":
            "Poor Metabolizer"}`) to seed `InvestigationState.
            retrieved_evidence` before the graph starts. This is the
            only way `tool_agent.py::_find_phenotype` can see a
            phenotype today — free text in `incident` is NOT parsed
            into structured evidence by any node. Mirrors how
            `packages/graph.py`'s own demo scenarios (`run_path_a`)
            seed this field directly, so the API-driven path can
            reproduce the same confirmed-root-cause / reject ->
            re-investigate -> approve loop instead of only the refusal
            path.
    """
    app = build_graph()
    initial_state = new_investigation(case_id=case_id, incident=incident)
    if documents:
        initial_state["documents"] = documents
    if retrieved_evidence:
        initial_state["retrieved_evidence"] = retrieved_evidence

    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            yield node_name, state_update


def _print_result(label: str, result: InvestigationState) -> None:
    print(f"=== {label} ===")
    print(f"status: {result['status']}")
    print(f"retry_count: {result['retry_count']}")
    print("review_history:")
    for r in result["review_history"]:
        print(f"  - {json.dumps(r)}")
    print("report:")
    print(json.dumps(result["report"], indent=2))
    print()


if __name__ == "__main__":
    _print_result("Path A — full loop: reject -> re-investigate -> approve", run_path_a())
    _print_result("Path B — refusal, approved on first pass", run_path_b())
