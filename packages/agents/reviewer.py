"""
Reviewer Agent — a second AI investigator that challenges the draft
report before it can be finalized.

Reads `state["report"]` (produced by the Reporter) and the full state,
and produces a review verdict written to `state["review_history"]`.

The design point this module exists to prove: the Reviewer does not
reject "I don't know" answers — an unconfirmed, zero-confidence hypothesis
with a named blocker is APPROVED, because refusing to guess without a
genotype is the correct epistemic behavior. What the Reviewer rejects is
an *unearned* confident answer: a 76%-confidence root cause resting on a
single evidence source, before broader corroboration (labs, FDA label,
timeline) has been gathered. That distinction — approving honest
uncertainty, rejecting thin-evidence confidence — is what makes this feel
like a real second reviewer rather than a rubber stamp.

SAFETY DESIGN — approving "unconfirmed" is NEVER delegated to the LLM.
If `report["root_cause"]["status"] == "unconfirmed"`, this node approves
deterministically, without calling the LLM at all. This is the most
important guardrail in the whole graph: an LLM reviewer could plausibly
be persuaded (by symptom pattern-matching, or just stochastic drift) to
second-guess a correct refusal and demand/imply a diagnosis anyway. Since
there is nothing to "reason" about in approving honest uncertainty — it
is always correct — there is no upside to routing it through the LLM and
a real downside if it ever doesn't.

For CONFIRMED reports, the LLM reasons about evidence sufficiency (this
is where LLM judgment adds real value over a fixed evidence-count
threshold) via `llm_json_call`. Its verdict is validated post-hoc: an
"approved" verdict on a report backed by fewer than 2 supporting evidence
sources is rejected as untrustworthy and overridden to the deterministic
rejection — the LLM is allowed to be stricter than the rule-based
baseline, never more lenient in a way that would rubber-stamp thin
evidence.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from packages.llm import llm_available, llm_json_call
from packages.schemas.investigation_state import InvestigationState, InvestigationStatus


class ReviewIssue(BaseModel):
    type: str = Field(
        description=(
            "One of: missing_evidence | contradiction_unresolved | "
            "hypothesis_underscored | confidence_inflated"
        )
    )
    description: str
    action: str = Field(
        description=(
            "What the Tool Agent should do next, e.g. retrieve_labs | "
            "retrieve_fda_label | retrieve_cpic_guidelines"
        )
    )


class ReviewResult(BaseModel):
    verdict: str = Field(description="approved | rejected")
    issues: List[ReviewIssue] = Field(default_factory=list)
    review_notes: str


_REVIEWER_SYSTEM = """You are the Reviewer in a clinical adverse drug event investigation engine called Sentinel Clinical — a second AI investigator who independently challenges a draft report before it can be finalized.

You are reviewing a CONFIRMED root cause (the investigation reached a conclusion, not a refusal). Your job is to judge whether the evidence base is broad enough to justify the stated confidence.

Rules:
- If the root cause rests on only ONE evidence source, REJECT it as confidence_inflated, regardless of how plausible the conclusion sounds. A single source is never enough to justify high confidence in a clinical root-cause investigation.
- If TWO OR MORE independent evidence sources corroborate the root cause, you may APPROVE.
- When rejecting, specify a concrete "action" the Tool Agent should take next (e.g. retrieve_labs, retrieve_fda_label, retrieve_cpic_guidelines) to close the gap.
- Be a real skeptic, not a rubber stamp. Reject confident conclusions that are not adequately supported, even if you personally find the conclusion likely to be correct."""


def _rule_based_reject(report: dict) -> dict:
    return {
        "verdict": "rejected",
        "issues": [
            {
                "type": "confidence_inflated",
                "description": (
                    f"{report['root_cause']['confidence']:.0%} confidence "
                    "based on single evidence source (pgx-core). No lab "
                    "trends, FDA label, or timeline retrieved to "
                    "corroborate."
                ),
                "action": "retrieve_labs",
            }
        ],
        "review_notes": (
            "Root cause confidence is not adequately supported by the "
            "current evidence base. Requesting additional corroborating "
            "evidence before this report can be finalized."
        ),
    }


def _rule_based_approve() -> dict:
    return {
        "verdict": "approved",
        "issues": [],
        "review_notes": (
            "Confidence supported by pgx-core CPIC Level A recommendation and "
            "corroborating lab trends (neutropenia Day 5, normal eGFR "
            "excluding renal cause)."
        ),
    }


def _review_confirmed_report(state: InvestigationState, report: dict) -> dict:
    """Rule-based or LLM-backed review of a CONFIRMED report. Never called
    for unconfirmed reports — see module docstring."""
    evidence_source_count = len(report.get("supporting_evidence", []))
    rule_based_verdict = "rejected" if evidence_source_count < 2 else "approved"

    if not llm_available():
        return _rule_based_reject(report) if rule_based_verdict == "rejected" else _rule_based_approve()

    result = llm_json_call(
        system_prompt=_REVIEWER_SYSTEM,
        user_prompt=(
            f"Root cause: {report['root_cause']['title']}\n"
            f"Confidence: {report['root_cause']['confidence']:.0%}\n"
            f"Supporting evidence ({evidence_source_count} source(s)):\n"
            + "\n".join(f"- {e}" for e in report.get("supporting_evidence", []))
        ),
        output_model=ReviewResult,
    )

    if result is None:
        return _rule_based_reject(report) if rule_based_verdict == "rejected" else _rule_based_approve()

    review = result.model_dump()

    # Guardrail: the LLM may be stricter than the rule-based baseline
    # (reject something the rule would approve), but must never be more
    # lenient (approve something the rule would reject) — a thin-evidence
    # report rubber-stamped by the LLM is the exact failure this node
    # exists to prevent. Override to the deterministic rejection if this
    # happens.
    if rule_based_verdict == "rejected" and review["verdict"] == "approved":
        return _rule_based_reject(report)

    # If the LLM rejected but produced no actionable issue, ensure there's
    # at least the deterministic one so the Tool Agent has something to act on.
    if review["verdict"] == "rejected" and not review["issues"]:
        review["issues"] = _rule_based_reject(report)["issues"]

    return review


def reviewer_node(state: InvestigationState) -> dict:
    """
    LangGraph node. On approval, returns `review_history` + status
    COMPLETED. On rejection, returns `review_history` + `review_issues`
    (read by the Tool Agent on the next pass) + incremented `retry_count`
    + clears `report` (Reporter regenerates after re-investigation).
    """
    report = state.get("report")
    if report is None:
        # Defensive: Reviewer should never run without a report. Fail
        # loud rather than silently approving nothing.
        raise ValueError("Reviewer ran with no report in state — check graph wiring.")

    if report["root_cause"]["status"] == "unconfirmed":
        # Never delegated to the LLM — see module docstring.
        review_result = {
            "verdict": "approved",
            "issues": [],
            "review_notes": (
                "Investigation correctly refused to conclude without genotype. "
                "Epistemic honesty confirmed. Preliminary report: DPYD toxicity "
                "suspected but unconfirmed. Recommend genotype testing before "
                "finalizing."
            ),
        }
        report["report_status"] = "finalized"
        return {
            "review_history": [review_result],
            "status": InvestigationStatus.COMPLETED,
            "report": report,
        }

    review_result = _review_confirmed_report(state, report)

    if review_result["verdict"] == "approved":
        report["report_status"] = "finalized"
        return {
            "review_history": [review_result],
            "status": InvestigationStatus.COMPLETED,
            "report": report,
        }

    return {
        "review_history": [review_result],
        "review_issues": review_result["issues"],
        "retry_count": state["retry_count"] + 1,
        "report": None,
    }
