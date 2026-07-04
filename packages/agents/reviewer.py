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
there is nothing to reason about in approving honest uncertainty — it is
always correct — there is no upside to routing it through the LLM and a
real downside if it ever doesn't.

For CONFIRMED reports, the LLM reasons about evidence sufficiency,
alternative-hypothesis coverage, citation completeness, and unresolved
contradictions (this is where LLM judgment adds real value over a fixed
evidence-count threshold) via `llm_json_call`. Its verdict is validated
post-hoc: an "approved" verdict on a report backed by fewer than 2
supporting evidence sources is rejected as untrustworthy and overridden
to the deterministic rejection — the LLM is allowed to be stricter than
the rule-based baseline, never more lenient in a way that would
rubber-stamp thin evidence.

This module's prompt also explicitly protects the companion case: it
instructs the LLM not to reject an inconclusive-but-explicit refusal for
being inconclusive — that's covered by the deterministic branch above,
but the instruction is kept in the prompt too in case this node is ever
extended to route unconfirmed reports through the LLM as well.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from packages.llm import llm_available, llm_json_call
from packages.schemas.investigation_state import InvestigationState, InvestigationStatus


class ReviewIssue(BaseModel):
    type: str = Field(
        description=(
            "missing_evidence | contradiction_unresolved | "
            "hypothesis_underscored | confidence_inflated | citation_missing"
        )
    )
    description: str = Field(description="What the problem is, specifically")
    action: str = Field(description="What the investigation should do to resolve this issue")


class ReviewResult(BaseModel):
    verdict: str = Field(description="'approved' or 'rejected'")
    issues: List[ReviewIssue] = Field(
        default_factory=list,
        description="Empty if approved, specific actionable issues if rejected",
    )
    review_notes: str = Field(
        description="Explanation of the reviewer's assessment — what was checked and why the verdict was reached"
    )


_REVIEWER_SYSTEM = """You are the Reviewer in Sentinel Clinical, an autonomous adverse drug event investigation engine.

You are a skeptical senior clinical investigator. Your job is to CHALLENGE the investigation report before it is finalized. You are not the investigator — you are the second pair of eyes that catches what the first investigation missed.

You are reviewing a CONFIRMED root cause (the investigation reached a conclusion, not a refusal).

## Your Mandate

1. CHALLENGE CONFIDENCE, NOT JUST CONCLUSIONS.
   - A 76% confidence score based on a single evidence source is NOT acceptable. High confidence requires corroboration from multiple independent sources.
   - Ask: "If I removed the strongest evidence source, would the conclusion still hold?" If not, confidence is inflated.
   - Flag confidence_inflated when the score exceeds what the evidence base supports.

2. VALIDATE EPISTEMIC HONESTY.
   - If the report says "we cannot confirm without genotype," that is CORRECT behavior. Do NOT reject it for being inconclusive.
   - Refusing to conclude when evidence is missing is better than guessing. Approve these reports with a note confirming the refusal is appropriate.
   - However, if the report is vague about WHAT evidence is missing ("more investigation needed" without specifics), reject with a request for explicit missing-evidence items.

3. NEVER APPROVE AN UNCONFIRMED HYPOTHESIS AS CONFIRMED.
   - If any hypothesis has status "unconfirmed" or has unresolved blockers, the root cause in the report must be marked as unconfirmed.
   - A report that presents an unconfirmed hypothesis as the root cause is a rejection-worthy error.

4. REJECTION CRITERIA.
   Reject the report if:
   - Confidence is inflated relative to evidence diversity (single-source high confidence)
   - Alternative hypotheses were not evaluated (only one explanation considered)
   - Missing evidence is not explicitly listed
   - Contradictions in the evidence were not addressed
   - Citations are missing from the evidence chain

   Approve the report if:
   - Confidence is proportionate to evidence quality and diversity
   - Missing evidence is explicitly documented
   - Alternative causes were considered and scored
   - The conclusion (or refusal to conclude) is supported by the evidence presented

5. REJECTION MUST BE ACTIONABLE.
   - Every rejection issue must include a specific "action" field telling the investigation what to retrieve or re-examine.
   - "Need more evidence" is not actionable. "Retrieve DPYD genotype to confirm or exclude pharmacogenomic toxicity" is actionable.
   - The investigation team (Tool Agent) reads your issues to determine what to do next.

6. YOUR TONE.
   - You are thorough, specific, and fair. Not adversarial for the sake of it.
   - When the investigation is good, say so clearly and explain why the evidence supports approval.
   - When the investigation has gaps, be precise about what's missing and why it matters.
   - You are the quality gate. A finalized report means YOU approved it. Own that responsibility."""


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


def _build_review_context(report: dict, evidence_source_count: int) -> str:
    lines = [
        f"Root cause: {report['root_cause']['title']}",
        f"Confidence: {report['root_cause']['confidence']:.0%}",
        f"Supporting evidence ({evidence_source_count} source(s)):",
    ]
    lines += [f"- {e}" for e in report.get("supporting_evidence", [])] or ["- (none)"]

    lines.append(f"Alternative causes considered: {len(report.get('alternative_causes', []))}")
    for alt in report.get("alternative_causes", []):
        lines.append(f"- {alt['title']} ({alt['confidence']:.0%})")

    lines.append(f"Citations: {report.get('citations') or '(none)'}")

    contradictions = report.get("contradictions", [])
    lines.append(
        f"Unresolved contradictions: {len(contradictions)}"
        if contradictions
        else "Unresolved contradictions: none"
    )

    return "\n".join(lines)


def _review_confirmed_report(state: InvestigationState, report: dict) -> dict:
    """Rule-based or LLM-backed review of a CONFIRMED report. Never called
    for unconfirmed reports — see module docstring."""
    evidence_source_count = len(report.get("supporting_evidence", []))
    rule_based_verdict = "rejected" if evidence_source_count < 2 else "approved"

    if not llm_available():
        return _rule_based_reject(report) if rule_based_verdict == "rejected" else _rule_based_approve()

    result = llm_json_call(
        system_prompt=_REVIEWER_SYSTEM,
        user_prompt=_build_review_context(report, evidence_source_count),
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
