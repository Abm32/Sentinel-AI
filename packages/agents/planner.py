"""
Planner Agent.

Input: `state["incident"]` (free-text clinical presentation).
Output: an investigation task list, written to `state["tasks"]`.

Two implementations, selected automatically:

  - LLM path (Vultr Serverless Inference chat-completion model — see
    packages/llm.py's module docstring for the current default and its
    unverified-model-ID caveat), used when VULTR_API_KEY is set.
    Structured output via `llm_json_call` (prompt-based JSON + manual
    Pydantic parsing) — NOT `with_structured_output()`/function
    calling, since tool-calling support on Vultr Serverless Inference
    is restricted to specific models and this project doesn't want to
    depend on the configured chat model supporting it.
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
    "confirm_pharmacogenomic_genotype",
    "retrieve_fda_label",
    "retrieve_cpic_guidelines",
    "check_drug_interactions",
    "build_timeline",
)

# Fixed plan for the fluorouracil / DPYD-shaped demo incident. The
# rule-based fallback doesn't branch on incident content.
#
# confirm_pharmacogenomic_genotype sits right after
# retrieve_pharmacogenomics: pgx-core's guideline-level "AVOID"
# recommendation is population-level evidence, not confirmation that
# this specific patient was tested and found to carry the phenotype —
# the live Reviewer (packages/agents/reviewer.py) flags exactly this
# gap when it's the only evidence in a report. Including the task in
# the plan from the start signals the investigation's intent to close
# that gap; packages/agents/tool_agent.py currently only calls the
# genotype-confirmation tool once the Reviewer actually asks for it on
# a re-investigation pass, not on the first pass (see that module's
# docstring) — the task's presence here doesn't change that timing.
_DEFAULT_TASK_LIST: list[dict] = [
    {"task": "retrieve_medication_history", "priority": "high"},
    {"task": "retrieve_lab_trends", "priority": "high"},
    {"task": "retrieve_pharmacogenomics", "priority": "high"},
    {"task": "confirm_pharmacogenomic_genotype", "priority": "high"},
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

    # Defensive: tool_agent.py hard-requires the literal task name
    # "retrieve_pharmacogenomics" to call pgx-core at all — it is not
    # interchangeable with "confirm_pharmacogenomic_genotype" from that
    # module's point of view, even though an LLM plan can reasonably
    # (if unhelpfully) treat them as covering the same ground and omit
    # one. Observed in practice: the live planner (Kimi-K2.6) sometimes
    # includes confirm_pharmacogenomic_genotype but drops
    # retrieve_pharmacogenomics, which means pgx-core never runs, the
    # Hypothesis Agent has nothing to score, and hypotheses ends up
    # empty — a silent, avoidable failure mode rather than an honest
    # refusal. Inject it defensively (mirroring the rule-based plan's
    # own "high" priority for this task) rather than let a downstream
    # node quietly get zero evidence because of a planning omission.
    if not any(t["task"] == "retrieve_pharmacogenomics" for t in valid_tasks):
        valid_tasks.append(
            {
                "task": "retrieve_pharmacogenomics",
                "priority": "high",
                "rationale": (
                    "Required for pgx-core's clinical-action lookup; "
                    "added defensively — the investigation plan omitted "
                    "this task despite the incident's pharmacogenomic "
                    "relevance."
                ),
            }
        )

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
