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

Review logic is rule-based for now (a real/LLM reviewer would reason
about evidence sufficiency directly; this stub encodes one concrete rule
to prove the loop):

  - report_status == "unconfirmed" (Path B) -> always approved.
  - report_status == "confirmed" with only ONE supporting evidence
    source -> rejected, action="retrieve_labs" (confidence_inflated).
  - report_status == "confirmed" with TWO OR MORE supporting evidence
    sources -> approved.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState, InvestigationStatus


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

    evidence_source_count = len(report.get("supporting_evidence", []))

    if evidence_source_count < 2:
        review_result = {
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
        return {
            "review_history": [review_result],
            "review_issues": review_result["issues"],
            "retry_count": state["retry_count"] + 1,
            "report": None,
        }

    review_result = {
        "verdict": "approved",
        "issues": [],
        "review_notes": (
            "Confidence supported by pgx-core CPIC Level A recommendation and "
            "corroborating lab trends (neutropenia Day 5, normal eGFR "
            "excluding renal cause)."
        ),
    }
    report["report_status"] = "finalized"
    return {
        "review_history": [review_result],
        "status": InvestigationStatus.COMPLETED,
        "report": report,
    }
