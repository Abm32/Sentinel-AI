"""
Tool Agent node — reads the Planner's task list and executes tools.

Three tools are wired for real (well, "real" relative to this project's
current stage):

  - retrieve_pharmacogenomics -> pgx-core (packages/tools/pgx_tool.py),
    a genuine third-party call.
  - confirm_pharmacogenomic_genotype -> the genotype-confirmation stub
    (packages/tools/genotype_tool.py). NOT run on the first pass — see
    `_genotype_requested` below. This tool exists specifically to
    answer the Reviewer's objection that a CPIC guideline recommendation
    is population-level evidence, not confirmation that *this patient*
    was actually tested and found to carry the phenotype. Confirmed
    against a live Reviewer run (Kimi-K2.6): this is exactly the
    objection it raises when pgx-core's guideline-level "AVOID" is the
    only evidence in the report.
  - retrieve_lab_trends -> a hardcoded stub returning fixed lab evidence
    (neutropenia trend, normal eGFR). Not a real lab-retrieval backend —
    exists to prove the Reviewer's reject -> re-investigate -> approve
    loop without needing the Retrieval Agent built first.

Every other planned task remains an explicit not_implemented stub.

Re-investigation behavior: this node runs once per pass through the
graph, and the graph can route back into it after a Reviewer rejection
(see packages/graph.py's conditional edge). Three things matter for that:

  1. pgx-core is NOT re-called if a confirmed/insufficient-evidence result
     already exists in tool_outputs from a prior pass — it's deterministic
     given the same inputs, so re-running it would just duplicate the same
     entry under the operator.add reducer.
  2. retrieve_lab_trends only runs when the Reviewer has actually asked
     for it (state["review_issues"] contains an issue with
     action == "retrieve_labs") and it hasn't already been supplied —
     otherwise every pass would silently start fabricating evidence that
     wasn't requested.
  3. genotype-confirmation only runs when the Reviewer's rejection issues
     mention genotype/phenotype confirmation specifically (in either the
     `action` or `description` field — the live LLM Reviewer phrases the
     action differently pass to pass, e.g. "Retrieve DPYD genotype
     results..." vs. "Obtain confirmatory pharmacogenomic testing...", so
     this checks free text rather than a single fixed action string) and
     hasn't already been supplied.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState
from packages.tools.registry import call_tool

# Demo case constants. In a real Retrieval Agent these would come from
# parsed documents; hardcoded here since Retrieval isn't built yet.
_DEMO_GENE = "DPYD"
_DEMO_DRUG = "fluorouracil"

# Hardcoded lab-evidence stub, returned only when the Reviewer requests
# `retrieve_labs`. Matches the demo narrative: neutropenia trend
# corroborates the DPYD hypothesis; normal eGFR rules out the renal
# alternative.
_LAB_STUB_RESULT = {
    "tool": "lab_trends",
    "status": "confirmed",
    "findings": [
        {"test": "ANC", "trend": "declining", "day5_value": "0.4 x10^9/L", "flag": "neutropenia"},
        {"test": "eGFR", "value": "92 mL/min/1.73m^2", "flag": "normal"},
    ],
    "interpretation": (
        "Neutropenia trend (Day 5 ANC 0.4) corroborates cytotoxic marrow "
        "suppression consistent with fluoropyrimidine toxicity. Normal "
        "eGFR argues against renal impairment as the primary driver."
    ),
    "citations": [],
}


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


def _already_ran(state: InvestigationState, tool_name: str) -> bool:
    return any(o.get("tool") == tool_name for o in state.get("tool_outputs", []))


def _labs_requested_and_missing(state: InvestigationState) -> bool:
    requested = any(
        issue.get("action") == "retrieve_labs"
        for issue in state.get("review_issues", [])
    )
    return requested and not _already_ran(state, "lab_trends")


# Free-text markers the Reviewer's rejection issues use when demanding
# patient-specific genotype/phenotype confirmation. Checked against both
# `action` and `description` rather than a single fixed action string:
# the live LLM Reviewer (Kimi-K2.6) phrases the requested action
# differently pass to pass ("Retrieve DPYD genotype results...",
# "Obtain confirmatory pharmacogenomic testing...", "revise the root
# cause to 'Suspected DPYD Poor Metabolizer'...") — a single hardcoded
# action string like the lab stub's "retrieve_labs" would miss most of
# these in practice.
_GENOTYPE_REQUEST_MARKERS = ("genotype", "phenotype")


def _genotype_requested_and_missing(state: InvestigationState) -> bool:
    requested = any(
        any(marker in str(issue.get(field, "")).lower() for field in ("action", "description"))
        for issue in state.get("review_issues", [])
        for marker in _GENOTYPE_REQUEST_MARKERS
    )
    return requested and not _already_ran(state, "genotype-confirmation")


def tool_agent_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed: `tool_outputs`
    (merged via the `operator.add` reducer).
    """
    tasks = state.get("tasks", [])
    task_names = {t["task"] for t in tasks}
    outputs: list[dict] = []

    if "retrieve_pharmacogenomics" in task_names and not _already_ran(state, "pgx-core"):
        phenotype = _find_phenotype(state)
        result = call_tool(
            "pgx-core",
            gene=_DEMO_GENE,
            drug=_DEMO_DRUG,
            phenotype=phenotype,
        )
        outputs.append(result)

    if _genotype_requested_and_missing(state):
        outputs.append(call_tool("genotype-confirmation", gene=_DEMO_GENE))

    if _labs_requested_and_missing(state):
        outputs.append(dict(_LAB_STUB_RESULT))

    # Every other planned task that hasn't been specifically handled above
    # is a stub — no retrieval/tool backend exists yet. Recorded explicitly
    # (and only on the first pass, to avoid duplicate not_implemented
    # entries piling up under the operator.add reducer on every retry).
    if not state.get("tool_outputs"):
        for task in tasks:
            if task["task"] in (
                "retrieve_pharmacogenomics",
                "retrieve_lab_trends",
                "confirm_pharmacogenomic_genotype",
            ):
                continue
            outputs.append({"task": task["task"], "status": "not_implemented"})

    return {"tool_outputs": outputs}
