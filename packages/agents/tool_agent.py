"""
Tool Agent node — reads the Planner's task list and executes tools.

Only `retrieve_pharmacogenomics` is wired to a real tool call (pgx-core)
right now. Every other task in the plan is a deliberate no-op stub —
`{"task": ..., "status": "not_implemented"}` — so the chain's shape is
honest about what's real vs. plumbing-only.

The phenotype input for the pgx-core call is read from
`state["retrieved_evidence"]` (a "genotype" evidence entry, if present)
rather than hardcoded, so the same node can produce both investigation
outcomes:

  Path A (confirmed):    a phenotype evidence entry is present  -> AVOID,
                          76%-confidence DPYD hypothesis.
  Path B (the refusal):  no phenotype evidence entry             -> pgx-core
                          returns {} -> insufficient_evidence -> the
                          Hypothesis Agent produces an explicit
                          "UNCONFIRMED... need genotype" hypothesis
                          instead of guessing.

This mirrors how a real Retrieval Agent would surface (or fail to
surface) a genomic report before the Tool Agent runs — Sentinel doesn't
invent a phenotype; it reports honestly on what evidence was retrieved.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState
from packages.tools.registry import call_tool

# Demo case constants. In a real Retrieval Agent these would come from
# parsed documents; hardcoded here since Retrieval isn't built yet.
_DEMO_GENE = "DPYD"
_DEMO_DRUG = "fluorouracil"


def _find_phenotype(state: InvestigationState) -> str | None:
    """Look for a genomic-phenotype evidence entry in retrieved_evidence.

    Expected shape: {"source": "genomic_report", "gene": "DPYD",
    "phenotype": "Poor Metabolizer"}. Returns None if no such evidence
    exists — this is what drives Path B (the refusal).
    """
    for evidence in state.get("retrieved_evidence", []):
        if evidence.get("source") == "genomic_report" and evidence.get("gene") == _DEMO_GENE:
            return evidence.get("phenotype")
    return None


def tool_agent_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed: `tool_outputs`
    (merged via the `operator.add` reducer).
    """
    tasks = state.get("tasks", [])
    task_names = {t["task"] for t in tasks}
    outputs: list[dict] = []

    if "retrieve_pharmacogenomics" in task_names:
        phenotype = _find_phenotype(state)
        result = call_tool(
            "pgx-core",
            gene=_DEMO_GENE,
            drug=_DEMO_DRUG,
            phenotype=phenotype,
        )
        outputs.append(result)

    # Every other planned task is a stub for now — no retrieval/tool
    # backend exists yet. Recorded explicitly so the investigation state
    # never silently implies work was done that wasn't.
    for task in tasks:
        if task["task"] == "retrieve_pharmacogenomics":
            continue
        outputs.append({"task": task["task"], "status": "not_implemented"})

    return {"tool_outputs": outputs}
