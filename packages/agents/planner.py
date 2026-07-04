"""
Planner Agent.

Input: `state["incident"]` (free-text clinical presentation).
Output: an investigation task list, written to `state["tasks"]`.

Two implementations, selected automatically:

  - LLM path (Nemotron via Vultr Serverless Inference), used when
    VULTR_API_KEY is set. Structured output via `llm_json_call`
    (prompt-based JSON + manual Pydantic parsing) — NOT
    `with_structured_output()`/function calling, since Vultr's tool
    calling is restricted to kimi-k2-instruct and this project is
    committed to Nemotron for the reasoning nodes. See packages/llm.py.
  - Rule-based fallback (deterministic), used when no key is set, or if
    the LLM call fails after retries. This keeps the graph runnable
    with no API key at all, and resilient if the API is flaky during a
    live demo — the deterministic path is a resilience feature, not a
    lesser fallback.

Downstream contract that MUST be preserved by both paths: `tool_agent.py`
dispatches on the literal string in `task["task"]` (e.g.
"retrieve_pharmacogenomics", "retrieve_lab_trends") to decide what to
execute. The LLM prompt is deliberately constrained to the same fixed
vocabulary the rule-based planner uses.

Planner never answers, never retrieves, never scores. It only plans.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from packages.llm import llm_available, llm_json_call
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
    task: str = Field(description="What to investigate")
    priority: str = Field(description="high | medium | low")
    rationale: str = Field(description="Why this task matters for this case")


class InvestigationPlan(BaseModel):
    tasks: List[InvestigationTask]


_PLANNER_SYSTEM = """You are the Planner in a clinical adverse drug event investigation engine called Sentinel Clinical.

Your job: given a clinical incident, produce a structured investigation plan — a list of specific tasks the investigation must perform.

Rules:
- Do NOT investigate. Do NOT diagnose. Do NOT draw conclusions. Only PLAN.
- You MUST choose task names ONLY from this fixed set (do not invent others): {task_vocabulary}
- Prioritize tasks: "high" for anything directly relevant to the suspected adverse event, "medium" for corroborating/contextual evidence.
- Include a rationale for each task explaining why it matters for THIS case."""


def _rule_based_planner(state: InvestigationState) -> dict:
    return {
        "tasks": list(_DEFAULT_TASK_LIST),
        "status": InvestigationStatus.INVESTIGATING,
    }


def _llm_planner(state: InvestigationState) -> dict:
    plan = llm_json_call(
        system_prompt=_PLANNER_SYSTEM.format(
            task_vocabulary=", ".join(_TASK_VOCABULARY)
        ),
        user_prompt=f"Incident: {state['incident']}",
        output_model=InvestigationPlan,
    )

    if plan is None:
        # LLM failed after retries — fall back to deterministic.
        return _rule_based_planner(state)

    tasks = [t.model_dump() for t in plan.tasks]
    # Defensive: constrain to the known vocabulary even if the model
    # drifts. Anything outside it would become a silent not_implemented
    # stub downstream, but better to filter here and fall back to the
    # rule-based plan than run an investigation with a malformed task list.
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
