"""
Hypothesis Agent.

Reads `state["tool_outputs"]` (and `state["incident"]`) and produces
competing hypotheses with confidence scores, written to
`state["hypotheses"]`.

Two implementations, selected automatically (same pattern as planner.py):

  - LLM path (Nemotron via Vultr Serverless Inference), used when
    VULTR_API_KEY is set. Structured output via `llm_json_call`
    (prompt-based JSON, no function calling).
  - Rule-based fallback (deterministic), used when no key is set, or the
    LLM call fails after retries, OR the LLM violates the no-guessing
    guardrail (see below) — in which case we do not trust its output at
    all and fall back rather than risk surfacing a hallucinated
    diagnosis.

THE GUARDRAIL — this is the most safety-critical prompt in the project.
If pgx-core (or any tool) reports status="insufficient_evidence" (no
phenotype available), the LLM is explicitly instructed it MUST NOT infer
or assume a phenotype, and must produce a hypothesis with status
"unconfirmed", confidence 0.0, and a named blocker instead of scoring a
confident conclusion. This is enforced twice:
  1. In the prompt itself (explicit instruction + the exact tool_outputs
     status is described in plain language, not just embedded as raw
     JSON the model might skim past; plus a hard "single source caps
     confidence at 60%" rule so the Reviewer's rejection of thin
     evidence emerges from the Hypothesis Agent's own scoring rather
     than being hardcoded).
  2. Post-hoc, in _violates_guardrail: if any tool reported
     insufficient_evidence for the demo gene and the LLM nonetheless
     returned a hypothesis naming that gene with status="confirmed" or
     confidence > 0, that response is REJECTED IN FULL and we fall back
     to the rule-based Path B — the same "never trust a guess" principle
     applies to this node's own output, not just to pgx-core's. This
     check does not rely on the LLM's self-reported status field alone
     (an LLM could write status="confirmed" while still assigning a
     nonzero confidence despite blockers) — it checks confidence
     directly, which is harder to get wrong by prompt drift.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from packages.llm import llm_available, llm_json_call
from packages.schemas.investigation_state import InvestigationState

_DEMO_GENE = "DPYD"
_DEMO_DRUG = "fluorouracil"


def _find_output(state: InvestigationState, tool_name: str) -> dict | None:
    for output in state.get("tool_outputs", []):
        if output.get("tool") == tool_name:
            return output
    return None


class Hypothesis(BaseModel):
    title: str = Field(description="Short title of the hypothesis")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    status: str = Field(
        description=(
            "'confirmed' if evidence supports it, 'unconfirmed' if "
            "blocked by missing evidence"
        )
    )
    supporting_evidence: List[str] = Field(
        description="Specific evidence items supporting this hypothesis"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list, description="Evidence that argues against this hypothesis"
    )
    blockers: List[str] = Field(
        default_factory=list,
        description="Missing evidence needed to confirm or rule out this hypothesis",
    )


class HypothesisSet(BaseModel):
    hypotheses: List[Hypothesis]
    summary: str = Field(description="One-sentence summary of the hypothesis landscape")


_HYPOTHESIS_SYSTEM = """You are the Hypothesis Agent in Sentinel Clinical, an autonomous adverse drug event investigation engine.

Your job: given a clinical incident and all collected evidence, generate COMPETING explanations for what caused the adverse event. Not one answer — multiple hypotheses, ranked by confidence.

## Core Rules

1. EPISTEMIC HONESTY IS NON-NEGOTABLE.
   - Score confidence based ONLY on evidence actually present in the investigation.
   - If evidence is missing, you MUST reduce confidence and flag the gap explicitly.
   - You may NEVER infer, assume, or fabricate evidence that was not provided.
   - "The labs probably showed X" is forbidden. Either the labs showed X or they didn't.

2. PHARMACOGENOMIC GUARDRAIL.
   - If a pharmacogenomic phenotype or genotype was NOT retrieved, you MUST NOT assign one.
   - Do not say "likely DPYD poor metabolizer" if no DPYD phenotype is in the evidence.
   - You may note "DPYD deficiency is a possible explanation but phenotype was not tested" — that is a hypothesis with a BLOCKER, not a scored conclusion.
   - A hypothesis with an unresolved blocker gets confidence 0.0 and status "unconfirmed".

3. CONFIDENCE SCORING.
   - Base confidence on: number of independent evidence sources, consistency across sources, specificity of the evidence to this hypothesis, and absence of contradictory evidence.
   - A single evidence source (e.g. only pgx-core) should NOT support confidence above 60%, no matter how strong that one source is. Corroboration is required for high confidence.
   - If evidence contradicts a hypothesis, reduce its confidence and note the contradiction.
   - Confidence scores must sum to approximately 100% across all hypotheses (they are competing explanations of the same event).

4. HYPOTHESIS STRUCTURE.
   - Generate 2-5 hypotheses, ordered by confidence (highest first).
   - Each hypothesis must include: title, confidence (0-1), supporting evidence (specific items from the investigation), contradicting evidence (if any), and blockers (missing evidence that would be needed to confirm).
   - At least one hypothesis should be an alternative explanation (not just the obvious one).

5. WHAT YOU ARE NOT.
   - You are not a diagnostic engine. You are an investigation agent.
   - You do not conclude. You propose explanations with honest uncertainty.
   - You do not recommend treatment. You identify what happened and why, with evidence."""


def _rule_based_hypothesis(state: InvestigationState) -> dict:
    pgx_output = _find_output(state, "pgx-core")

    if pgx_output is None:
        return {"hypotheses": []}

    if pgx_output.get("status") == "confirmed" and pgx_output.get("action") == "AVOID":
        top_evidence = ["pgx-core: AVOID", "symptom match"]

        lab_output = _find_output(state, "lab_trends")
        if lab_output is not None:
            top_evidence.append(f"lab_trends: {lab_output['interpretation']}")

        hypotheses = [
            {
                "title": "DPYD-mediated fluorouracil toxicity",
                "confidence": 0.76,
                "status": "confirmed",
                "supporting_evidence": top_evidence,
                "contradicting_evidence": [],
                "blockers": [],
            },
            {
                "title": "Drug interaction",
                "confidence": 0.17,
                "status": "confirmed",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "blockers": [],
            },
            {
                "title": "Renal impairment",
                "confidence": 0.07,
                "status": "confirmed",
                "supporting_evidence": (
                    [f"lab_trends: {lab_output['interpretation']}"]
                    if lab_output is not None
                    else []
                ),
                "contradicting_evidence": (
                    [f"lab_trends: {lab_output['interpretation']}"]
                    if lab_output is not None
                    else []
                ),
                "blockers": [],
            },
        ]
        return {"hypotheses": hypotheses}

    # Path B — insufficient_evidence (no phenotype, or unmapped combo).
    # Name the blocker explicitly instead of guessing.
    hypotheses = [
        {
            "title": f"{_DEMO_GENE}-mediated {_DEMO_DRUG} toxicity (UNCONFIRMED)",
            "confidence": 0.0,
            "status": "unconfirmed",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "blockers": [
                f"genotype unavailable — cannot confirm without {_DEMO_GENE} "
                "phenotype"
            ],
        },
    ]
    return {"hypotheses": hypotheses}


def _describe_tool_outputs(state: InvestigationState) -> str:
    """Render tool_outputs as plain-language lines rather than raw JSON,
    so the insufficient_evidence status can't be skimmed past inside a
    JSON blob."""
    lines = []
    for output in state.get("tool_outputs", []):
        tool = output.get("tool") or output.get("task", "unknown")
        status = output.get("status", "unknown")
        if tool == "pgx-core":
            if status == "confirmed":
                lines.append(
                    f"- pgx-core: CONFIRMED. gene={output.get('gene')} "
                    f"drug={output.get('drug')} phenotype={output.get('phenotype')} "
                    f"action={output.get('action')} "
                    f"recommendation=\"{output.get('recommendation')}\" "
                    f"citations={output.get('citations')}"
                )
            else:
                lines.append(
                    f"- pgx-core: INSUFFICIENT_EVIDENCE. gene={output.get('gene')} "
                    f"drug={output.get('drug')} phenotype={output.get('phenotype')} "
                    "-- NO PHENOTYPE WAS AVAILABLE. You must not guess one."
                )
        elif tool == "lab_trends":
            lines.append(f"- lab_trends: {status}. {output.get('interpretation', '')}")
        else:
            lines.append(f"- {tool}: {status}")
    return "\n".join(lines) if lines else "(no tool outputs yet)"


_PHENOTYPE_TERMS = (
    "poor metabolizer",
    "intermediate metabolizer",
    "normal metabolizer",
    "rapid metabolizer",
    "ultrarapid metabolizer",
)


def _mentions_phenotype_as_fact(hypothesis: dict) -> bool:
    """Scan a hypothesis's text fields for phenotype terminology stated as
    fact (in the title or supporting_evidence) rather than as a named gap
    (blockers). A hypothesis that says 'DPYD deficiency possible, phenotype
    not tested' in its blockers is fine; one that asserts 'likely DPYD
    poor metabolizer' in its title or supporting_evidence is a guess."""
    haystack = " ".join(
        [hypothesis.get("title", "")] + hypothesis.get("supporting_evidence", [])
    ).lower()
    return any(term in haystack for term in _PHENOTYPE_TERMS)


def _violates_guardrail(state: InvestigationState, hypotheses: list[dict]) -> bool:
    """Post-hoc, code-level guardrail (defense in depth alongside the
    prompt-level instruction). If pgx-core reported insufficient_evidence
    for the demo gene, no returned hypothesis may:
      (a) name that gene with confidence > 0 — checked directly on
          confidence rather than trusting the LLM's self-reported
          `status` field, since prompt drift could produce
          status="confirmed" with a nonzero confidence despite blockers
          being present; or
      (b) assert a specific phenotype as fact in its title or
          supporting_evidence (e.g. "likely DPYD poor metabolizer")
          when no phenotype was actually retrieved, even if confidence
          happens to be scored low — the mere assertion is the
          hallucination, independent of the score attached to it.
    If the LLM violates either, we do not trust ANY of its output and
    fall back entirely."""
    pgx_output = _find_output(state, "pgx-core")
    if pgx_output is None or pgx_output.get("status") != "insufficient_evidence":
        return False

    gene = pgx_output.get("gene", _DEMO_GENE)
    for h in hypotheses:
        names_gene = gene.lower() in h.get("title", "").lower()
        if names_gene and h.get("confidence", 0) > 0.0:
            return True
        if names_gene and _mentions_phenotype_as_fact(h):
            return True
    return False


def _llm_hypothesis(state: InvestigationState) -> dict:
    result = llm_json_call(
        system_prompt=_HYPOTHESIS_SYSTEM,
        user_prompt=(
            f"Incident: {state['incident']}\n\n"
            f"Tool outputs so far:\n{_describe_tool_outputs(state)}"
        ),
        output_model=HypothesisSet,
    )

    if result is None:
        return _rule_based_hypothesis(state)

    hypotheses = [h.model_dump() for h in result.hypotheses]

    if not hypotheses or _violates_guardrail(state, hypotheses):
        # Do not trust this response at all — fall back completely
        # rather than surface a hallucinated confident diagnosis.
        return _rule_based_hypothesis(state)

    return {"hypotheses": hypotheses}


def hypothesis_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed: `hypotheses`
    (merged via the `operator.add` reducer).

    Each hypothesis is tagged with `round=state["retry_count"]` before
    being returned. Because `hypotheses` accumulates via operator.add
    across re-investigation passes (LangGraph reducers only support
    additive merges, not replace), the Reporter can't rely on a fixed
    item count to recover "this pass's" hypotheses once the LLM path
    produces a variable-length list. Tagging by round lets
    reporter.py._latest_hypotheses() filter reliably regardless of how
    many hypotheses either implementation returns.
    """
    if not llm_available():
        result = _rule_based_hypothesis(state)
    else:
        result = _llm_hypothesis(state)

    round_index = state.get("retry_count", 0)
    for h in result.get("hypotheses", []):
        h["round"] = round_index
    return result
