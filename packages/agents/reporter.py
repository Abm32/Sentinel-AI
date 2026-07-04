"""
Reporter Agent — rule-based template generation (not LLM, yet).

Reads the full InvestigationState and produces a structured investigation
report, written to `state["report"]`. Sets status to REVIEWING (the
Reviewer runs next in the graph).

Handling accumulation across retry passes:
`hypotheses` and `tool_outputs` are `Annotated[list, operator.add]` state
channels, so after a Reviewer-triggered re-investigation pass they contain
BOTH passes' entries concatenated, not a clean replacement (LangGraph
reducers only support additive merges). This module resolves that by
reading only the most recently produced hypothesis set (the last run of
the Hypothesis Agent always emits exactly 3 items for Path A or 1 item for
Path B, so the tail slice of that length is "latest"), and by
de-duplicating tool_outputs by `tool` name (last write wins). This is a
deliberate stopgap for the rule-based stub stage — a production version
would give the Hypothesis Agent its own "current hypothesis set" scratch
field instead of relying on slice-based recovery.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState, InvestigationStatus

_PATH_A_HYPOTHESIS_COUNT = 3
_PATH_B_HYPOTHESIS_COUNT = 1


def _latest_hypotheses(state: InvestigationState) -> list[dict]:
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return []
    # Path B always emits exactly 1 hypothesis; Path A always emits
    # exactly 3. Whatever the most recent run produced is a suffix of the
    # accumulated list matching one of those two known shapes.
    if len(hypotheses) >= _PATH_A_HYPOTHESIS_COUNT and hypotheses[-1].get("confidence") != 0.0:
        return hypotheses[-_PATH_A_HYPOTHESIS_COUNT:]
    return hypotheses[-_PATH_B_HYPOTHESIS_COUNT:]


def _latest_tool_outputs(state: InvestigationState) -> list[dict]:
    """De-duplicate accumulated tool_outputs by tool name, last write wins."""
    latest: dict[str, dict] = {}
    for output in state.get("tool_outputs", []):
        key = output.get("tool") or output.get("task", "unknown")
        latest[key] = output
    return list(latest.values())


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
        "executive_summary": (
            _unconfirmed_summary(state) if is_unconfirmed else _confirmed_summary(top)
        ),
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
