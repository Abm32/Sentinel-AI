"""
Reporter Agent.

Reads the full InvestigationState and produces a structured investigation
report, written to `state["report"]`. Sets status to REVIEWING (the
Reviewer runs next in the graph).

Structure-building (pulling root_cause/supporting_evidence/alternatives/
citations out of hypotheses and tool_outputs) stays deterministic in
BOTH paths — there is nothing to "reason" about there, it's a
straightforward extraction. The LLM path only replaces
`executive_summary`: a natural-language synthesis is exactly the kind of
task an LLM is suited for and a fixed template is not, whereas swapping
out the structured fields for LLM-authored JSON would just add
hallucination risk to numbers that are already known (confidence scores,
citations) with no upside.

Falls back to the deterministic summary templates if no key is set, or
if the LLM call fails after retries.

Handling accumulation across retry passes: see hypothesis.py's `round`
tagging and this module's `_latest_hypotheses` — hypotheses/tool_outputs
are operator.add state channels, so this module resolves "latest" by
round-tag filtering / de-duplication rather than assuming any fixed
count.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from packages.llm import llm_available, llm_json_call
from packages.schemas.investigation_state import InvestigationState, InvestigationStatus


def _latest_hypotheses(state: InvestigationState) -> list[dict]:
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return []
    current_round = state.get("retry_count", 0)
    latest = [h for h in hypotheses if h.get("round") == current_round]
    return latest if latest else hypotheses


def _latest_tool_outputs(state: InvestigationState) -> list[dict]:
    """De-duplicate accumulated tool_outputs by tool name, last write wins."""
    latest: dict[str, dict] = {}
    for output in state.get("tool_outputs", []):
        key = output.get("tool") or output.get("task", "unknown")
        latest[key] = output
    return list(latest.values())


class ExecutiveSummary(BaseModel):
    summary: str = Field(
        description=(
            "2-3 sentence executive summary of the investigation's "
            "findings, written for a clinical audience (MTB / quality "
            "review). Must accurately reflect whether the root cause is "
            "confirmed or unconfirmed — never state confidence the "
            "evidence does not support."
        )
    )


_REPORTER_SYSTEM = """You are the Reporter in a clinical adverse drug event investigation engine called Sentinel Clinical.

Your job: write a 2-3 sentence executive summary of the investigation for a clinical audience, given the root cause hypothesis and supporting evidence.

Rules:
- If the root cause status is "confirmed", state it plainly with its confidence level and what evidence corroborates it.
- If the root cause status is "unconfirmed" (confidence 0.0, blocked by missing evidence), you MUST say the investigation cannot conclude yet and state what evidence is missing. Do NOT write a summary that implies more certainty than the data supports. Recommend the specific next step (e.g. genotype testing) if a blocker is present.
- Be factual and concise. No hedging filler, no invented details not present in the hypothesis/evidence given to you."""


def _build_summary_context(top: dict | None, is_unconfirmed: bool) -> str:
    if top is None:
        return "No hypotheses were generated."
    lines = [
        f"Root cause candidate: {top['title']}",
        f"Confidence: {top['confidence']:.0%}",
        f"Status: {'unconfirmed' if is_unconfirmed else 'confirmed'}",
    ]
    if top.get("evidence"):
        lines.append(f"Supporting evidence: {'; '.join(top['evidence'])}")
    if top.get("blocker"):
        lines.append(f"Blocker: {top['blocker']}")
    return "\n".join(lines)


def _confirmed_summary(top: dict) -> str:
    return (
        f"Patient presented with symptoms consistent with fluoropyrimidine "
        f"toxicity. Investigation identifies {top['title']} as the most "
        f"likely root cause ({top['confidence']:.0%} confidence), "
        f"corroborated by pharmacogenomic and clinical evidence."
    )


def _unconfirmed_summary(state: InvestigationState) -> str:
    return (
        "Patient presented with symptoms consistent with fluoropyrimidine "
        "toxicity following 5-FU therapy. DPYD-mediated toxicity is "
        "suspected but cannot be confirmed: no DPYD genotype/phenotype "
        "evidence was available. Investigation recommends genotype "
        "testing before a root cause can be finalized."
    )


def _executive_summary(state: InvestigationState, top: dict | None, is_unconfirmed: bool) -> str:
    fallback = _unconfirmed_summary(state) if is_unconfirmed else _confirmed_summary(top)

    if not llm_available():
        return fallback

    result = llm_json_call(
        system_prompt=_REPORTER_SYSTEM,
        user_prompt=_build_summary_context(top, is_unconfirmed),
        output_model=ExecutiveSummary,
    )
    if result is None:
        return fallback

    # Guardrail: if the root cause is unconfirmed, the LLM's summary text
    # must not claim confidence — reject and fall back if it does
    # something like state the diagnosis as fact.
    if is_unconfirmed and "cannot" not in result.summary.lower() and "unconfirmed" not in result.summary.lower() and "unable" not in result.summary.lower():
        return fallback

    return result.summary


def reporter_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed: `report` and
    `status`.
    """
    hypotheses = sorted(
        _latest_hypotheses(state), key=lambda h: h["confidence"], reverse=True
    )
    tool_outputs = _latest_tool_outputs(state)

    top = hypotheses[0] if hypotheses else None
    alternatives = hypotheses[1:] if len(hypotheses) > 1 else []

    supporting_evidence: list[str] = []
    citations: list[str] = []
    for output in tool_outputs:
        if output.get("status") == "confirmed":
            if output.get("tool") == "pgx-core":
                supporting_evidence.append(
                    f"pgx-core: {output['action']} — {output['recommendation']}"
                )
            elif output.get("tool") == "lab_trends":
                supporting_evidence.append(f"lab_trends: {output['interpretation']}")
        citations.extend(output.get("citations", []))

    missing_evidence: list[str] = []
    if top and top.get("blocker"):
        missing_evidence.append(
            f"{top['blocker'].split('—')[0].strip().capitalize()} — "
            "genotype required to confirm"
            if "genotype" in top["blocker"]
            else top["blocker"]
        )

    is_unconfirmed = top is not None and top["confidence"] == 0.0

    report = {
        "case_id": state["case_id"],
        "incident": state["incident"],
        "executive_summary": _executive_summary(state, top, is_unconfirmed),
        "root_cause": {
            "title": top["title"] if top else "Unknown",
            "confidence": top["confidence"] if top else 0.0,
            "status": "unconfirmed" if is_unconfirmed else "confirmed",
        },
        "supporting_evidence": supporting_evidence,
        "alternative_causes": [
            {"title": h["title"], "confidence": h["confidence"]} for h in alternatives
        ],
        "missing_evidence": missing_evidence,
        "contradictions": state.get("contradictions", []),
        "citations": sorted(set(citations)),
        "report_status": "draft",
    }

    return {"report": report, "status": InvestigationStatus.REVIEWING}
