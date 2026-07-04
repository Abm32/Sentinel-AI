"""
Planner Agent — rule-based stub.

Input: `state["incident"]` (free-text clinical presentation).
Output: a deterministic investigation task list, written to `state["tasks"]`.

This is intentionally NOT an LLM call yet. The task list is fixed given a
5-FU/chemotherapy-shaped incident, which is enough to prove the graph
shape (Planner -> Tool Agent -> Hypothesis). Swapping this for a Nemotron
call later means changing only this function's body — the node's
input/output contract (incident in, tasks out) does not change.

Planner never answers, never retrieves, never scores. It only plans.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState, InvestigationStatus

# Fixed plan for the fluorouracil / DPYD-shaped demo incident. A real
# (LLM-backed) planner would branch on incident content; this stub does
# not need to, since proving the graph shape doesn't require branching.
_DEFAULT_TASK_LIST: list[dict] = [
    {"task": "retrieve_medication_history", "priority": "high"},
    {"task": "retrieve_lab_trends", "priority": "high"},
    {"task": "retrieve_pharmacogenomics", "priority": "high"},
    {"task": "retrieve_fda_label", "priority": "medium"},
    {"task": "retrieve_cpic_guidelines", "priority": "medium"},
    {"task": "check_drug_interactions", "priority": "medium"},
    {"task": "build_timeline", "priority": "high"},
]


def planner_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed — `tasks` (merged
    via the `operator.add` reducer) and `status`.
    """
    return {
        "tasks": list(_DEFAULT_TASK_LIST),
        "status": InvestigationStatus.INVESTIGATING,
    }
