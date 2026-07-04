"""
Planner Agent.

Input: `state["incident"]` (free-text clinical presentation).
Output: an investigation task list, written to `state["tasks"]`.

Two implementations, selected automatically:

  - LLM path (Nemotron via Vultr Serverless Inference), used when
    VULTR_API_KEY is set. Structured output constrained to
    InvestigationPlan so the result is always a real, typed task list.
  - Rule-based fallback (deterministic), used otherwise. This keeps the
    graph runnable with no API key at all — useful for tests, CI, and
    demos without network access.

Downstream contract that MUST be preserved by both paths: `tool_agent.py`
dispatches on the literal string in `task["task"]` (e.g.
"retrieve_pharmacogenomics", "retrieve_lab_trends") to decide what to
execute. The LLM prompt is deliberately constrained to the same fixed
vocabulary the rule-based planner uses — an LLM that invents novel task
names would silently produce not_implemented stubs for everything,
since the Tool Agent doesn't know how to route them.

Planner never answers, never retrieves, never scores. It only plans.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from packages.llm import get_llm, llm_available
from packages.schemas.investigation_state import InvestigationState, InvestigationStatus

# The fixed task vocabulary. Both the rule-based planner and the LLM
# prompt operate within this set — see module docstring for why.
_TASK_VOCABULARY = (
    "retrieve_medication_history",
    "retrieve_lab_trends",
    "retrieve_pharmacogenomics",
    "retrieve_fda_label",
    "retrieve_cpic_guidelines",
    "check_drug_interactions",
    "build_timeline",
)

# Fixed plan for the fluorouracil / DPYD-shaped demo incident. The
# rule-based fallback doesn't branch on incident content.
_DEFAULT_TASK_LIST: list[dict] = [
    {"task": "retrieve_medication_history", "priority": "high"},
    {"task": "retrieve_lab_trends", "priority": "high"},
    {"task": "retrieve_pharmacogenomics", "priority": "high"},
    {"task": "retrieve_fda_label", "priority": "medium"},
    {"task": "retrieve_cpic_guidelines", "priority": "medium"},
    {"task": "check_drug_interactions", "priority": "medium"},
    {"task": "build_timeline", "priority": "high"},
]


class InvestigationTask(BaseModel):
    task: str = Field(
        description=(
            "One of the following fixed task names — do not invent new "
            f"ones: {', '.join(_TASK_VOCABULARY)}"
        )
    )
    priority: str = Field(description="high | medium | low")
    rationale: str = Field(description="Why this task matters for this case")


class InvestigationPlan(BaseModel):
    tasks: List[InvestigationTask]


_PLANNER_PROMPT = """You are the Planner in a clinical adverse drug event investigation engine.

Given a clinical incident, produce a structured investigation plan — a list of tasks
the investigation must perform. Do NOT investigate. Do NOT diagnose. Only plan.

You MUST choose task names only from this fixed set (do not invent others):
{task_vocabulary}

Cover as many of medication history, lab trends, pharmacogenomics, FDA labeling,
clinical guidelines (CPIC), drug interactions, and timeline reconstruction as are
relevant to this incident.

Incident: {incident}
"""


def _rule_based_planner(state: InvestigationState) -> dict:
    return {
        "tasks": list(_DEFAULT_TASK_LIST),
        "status": InvestigationStatus.INVESTIGATING,
    }


def _llm_planner(state: InvestigationState) -> dict:
    llm = get_llm().with_structured_output(InvestigationPlan)
    prompt = _PLANNER_PROMPT.format(
        task_vocabulary=", ".join(_TASK_VOCABULARY),
        incident=state["incident"],
    )
    plan = llm.invoke(prompt)

    tasks = [t.model_dump() for t in plan.tasks]
    # Defensive: constrain to the known vocabulary even if the model
    # drifts. Anything outside it would just become a silent
    # not_implemented stub downstream, but better to filter here and fall
    # back to the rule-based plan than to run an investigation with an
    # empty/malformed task list.
    valid_tasks = [t for t in tasks if t["task"] in _TASK_VOCABULARY]
    if not valid_tasks:
        return _rule_based_planner(state)

    return {
        "tasks": valid_tasks,
        "status": InvestigationStatus.INVESTIGATING,
    }


def planner_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed — `tasks` (merged
    via the `operator.add` reducer) and `status`.
    """
    if not llm_available():
        return _rule_based_planner(state)
    return _llm_planner(state)
