"""
Retrieval Agent node.

Reads the Planner's task list and, for each retrieval-shaped task, calls
`packages.tools.retrieval_tool.search_evidence()` to find relevant
evidence chunks — writing them to `state["retrieved_evidence"]`.

Sits between the Planner and the Tool Agent in the graph:

    planner -> retrieval -> tool_agent -> hypothesis -> reporter -> reviewer

Division of labor with the Tool Agent (packages/agents/tool_agent.py):
  - Retrieval Agent owns *evidence search* tasks — anything answerable
    by semantic/keyword lookup over the indexed evidence base (lab
    trends, FDA label text, CPIC guideline text, drug-interaction
    notes, medication history). These map onto `search_evidence()`
    calls (Azure AI Search, or the local keyword-overlap fallback —
    see packages/tools/retrieval_tool.py).
  - `retrieve_pharmacogenomics` stays a Tool Agent job: pgx-core is a
    deterministic clinical-decision engine, not a search index — you
    don't "search" for a CPIC recommendation, you compute it from a
    resolved phenotype. The Retrieval Agent does NOT call pgx-core.
  - `build_timeline` is not yet implemented by either agent — it stays
    a `not_implemented` stub, produced by the Tool Agent as before.

Genomic phenotype evidence (the `{"source": "genomic_report", "gene":
"DPYD", "phenotype": ...}` shape that `tool_agent.py::_find_phenotype`
reads) is untouched by this node — it is either seeded directly onto
`state["retrieved_evidence"]` before the graph starts (as the
Path A/B demo scenarios in packages/graph.py do) or, in a future
iteration, extracted from an uploaded genomic report document. This
node only ever *adds* general evidence chunks; it does not overwrite
or filter what's already there, since `retrieved_evidence` is an
`operator.add` reducer field.

No LLM call in this node yet — it deterministically maps each task name
to a search query and runs it. LLM-generated/refined queries (e.g.
tailoring the query to the specific incident text rather than a fixed
per-task-type string) can be added later without changing this node's
contract with the rest of the graph.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState
from packages.tools.retrieval_tool import search_evidence

# Maps a Planner task name to the search query used to find corroborating
# evidence for it. Deliberately excludes "retrieve_pharmacogenomics" (Tool
# Agent's job, not a search) and "build_timeline" (not implemented by
# either agent yet). Keys are a subset of planner.py's _TASK_VOCABULARY —
# any task not present here is simply not handled by this node.
_TASK_TO_QUERY: dict[str, str] = {
    "retrieve_medication_history": "medication history drug interaction",
    "retrieve_lab_trends": "lab trends neutropenia ANC eGFR",
    "retrieve_fda_label": "FDA label DPD deficiency fluorouracil toxicity",
    "retrieve_cpic_guidelines": "CPIC guideline DPYD fluoropyrimidine dose",
    "check_drug_interactions": "drug interaction fluorouracil",
}


def _already_retrieved(state: InvestigationState, doc_type: str) -> bool:
    """True if evidence of this doc_type has already been retrieved —
    guards against duplicate entries piling up under the operator.add
    reducer on re-investigation passes (the graph can route back into
    earlier nodes after a Reviewer rejection)."""
    return any(
        evidence.get("doc_type") == doc_type
        for evidence in state.get("retrieved_evidence", [])
    )


# Maps each handled task to the doc_type its search results are tagged
# with in the fallback corpus (packages/tools/retrieval_tool.py), so
# `_already_retrieved` can de-duplicate per task rather than per exact
# query string.
_TASK_TO_DOC_TYPE: dict[str, str] = {
    "retrieve_medication_history": "interaction_check",
    "retrieve_lab_trends": "lab_report",
    "retrieve_fda_label": "fda_label",
    "retrieve_cpic_guidelines": "cpic_guideline",
    "check_drug_interactions": "interaction_check",
}


def retrieval_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed:
    `retrieved_evidence` (merged via the `operator.add` reducer).
    """
    tasks = state.get("tasks", [])
    retrieved: list[dict] = []

    for task in tasks:
        task_name = task.get("task")
        query = _TASK_TO_QUERY.get(task_name)
        if query is None:
            continue

        doc_type = _TASK_TO_DOC_TYPE.get(task_name)
        if doc_type and _already_retrieved(state, doc_type):
            continue

        for result in search_evidence(query, top_k=3):
            entry = dict(result)
            entry["retrieved_for_task"] = task_name
            retrieved.append(entry)

    return {"retrieved_evidence": retrieved}
