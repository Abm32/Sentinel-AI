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
or assume a phenotype, and must name the missing evidence as a blocker
instead of scoring a confident hypothesis. This is enforced twice:
  1. In the prompt itself (explicit instruction + the exact tool_outputs
     status is described in plain language, not just embedded as raw
     JSON the model might skim past).
  2. Post-hoc, in _llm_hypothesis: if any tool reported
     insufficient_evidence for DPYD and the LLM nonetheless returned a
     hypothesis with confidence > 0 naming DPYD, that response is
     REJECTED and we fall back to the rule-based Path B — the same
     "never trust a guess" principle applies to this node's own output,
     not just to pgx-core's.
"""

from __future__ import annotations

from typing import List, Optional

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
    title: str = Field(description="Name of the candidate root cause")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0")
    evidence: List[str] = Field(default_factory=list)
    blocker: Optional[str] = Field(
        default=None,
        description=(
            "If this hypothesis cannot be confirmed due to missing "
            "evidence, name exactly what is missing here. Leave null "
            "if the hypothesis is adequately supported."
        ),
    )


class HypothesisSet(BaseModel):
    hypotheses: List[Hypothesis]


_HYPOTHESIS_SYSTEM = """You are the Hypothesis Agent in a clinical adverse drug event investigation engine called Sentinel Clinical.

Your job: given the incident and the evidence gathered so far (tool outputs), produce a set of competing hypotheses for the root cause, each with a confidence score between 0.0 and 1.0.

CRITICAL RULE — you MUST NOT violate this under any circumstance:
If a tool result has status "insufficient_evidence" for a pharmacogenomic gene/drug pair (meaning no genotype/phenotype was available), you MUST NOT infer, assume, or guess what that phenotype might be. You must produce a hypothesis with confidence 0.0, and set "blocker" to state exactly what evidence is missing (e.g. "genotype unavailable — cannot confirm without {gene} phenotype"). Do NOT reason your way to a confident diagnosis from symptoms alone when the definitive pharmacogenomic evidence was explicitly reported as unavailable. Refusing to conclude is the correct behavior, not a failure.

If a tool result has status "confirmed", use its action/recommendation/citations as supporting evidence and score confidence based on how well the evidence set corroborates it. List alternative, lower-confidence hypotheses as well.

Always produce at least one hypothesis."""


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
                "evidence": top_evidence,
            },
            {
                "title": "Drug interaction",
                "confidence": 0.17,
                "evidence": [],
            },
            {
                "title": "Renal impairment",
                "confidence": 0.07,
                "evidence": (
                    [f"lab_trends: {lab_output['interpretation']}"]
                    if lab_output is not None
                    else []
                ),
            },
        ]
        return {"hypotheses": hypotheses}

    # Path B — insufficient_evidence (no phenotype, or unmapped combo).
    # Name the blocker explicitly instead of guessing.
    hypotheses = [
        {
            "title": f"{_DEMO_GENE}-mediated {_DEMO_DRUG} toxicity (UNCONFIRMED)",
            "confidence": 0.0,
            "evidence": [],
            "blocker": (
                f"genotype unavailable — cannot confirm without {_DEMO_GENE} "
                "phenotype"
            ),
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


def _violates_guardrail(state: InvestigationState, hypotheses: list[dict]) -> bool:
    """Post-hoc check: if pgx-core reported insufficient_evidence for the
    demo gene, no returned hypothesis may name that gene with
    confidence > 0. If the LLM violates this, we do not trust ANY of its
    output and fall back entirely."""
    pgx_output = _find_output(state, "pgx-core")
    if pgx_output is None or pgx_output.get("status") != "insufficient_evidence":
        return False

    gene = pgx_output.get("gene", _DEMO_GENE)
    for h in hypotheses:
        if gene.lower() in h.get("title", "").lower() and h.get("confidence", 0) > 0.0:
            return True
    return False


def _llm_hypothesis(state: InvestigationState) -> dict:
    result = llm_json_call(
        system_prompt=_HYPOTHESIS_SYSTEM.format(gene=_DEMO_GENE),
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
