"""
Hypothesis Agent — rule-based stub.

Reads `state["tool_outputs"]` (and, loosely, `state["incident"]`) and
produces competing hypotheses with confidence scores, written to
`state["hypotheses"]`.

Two paths, both rule-based for now (an LLM/Nemotron version later branches
on the same tool_outputs, not a different contract):

  Path A — pgx-core returned status="confirmed", action="AVOID" for
           DPYD/fluorouracil: emit the three-way competing-hypothesis set
           (DPYD 76%, drug interaction 17%, renal impairment 7%) — this is
           the demo's headline moment.

  Path B — pgx-core returned status="insufficient_evidence" (no phenotype
           available): emit a single explicitly-unconfirmed, zero-
           confidence hypothesis naming the missing evidence. This is the
           more important path — it's the "cannot confirm DPYD toxicity,
           need genotype, continue investigation" behavior described in
           Phase 3 of the architecture. No hallucinated certainty.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState

_DEMO_GENE = "DPYD"
_DEMO_DRUG = "fluorouracil"


def _find_pgx_output(state: InvestigationState) -> dict | None:
    for output in state.get("tool_outputs", []):
        if output.get("tool") == "pgx-core":
            return output
    return None


def hypothesis_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed: `hypotheses`
    (merged via the `operator.add` reducer).
    """
    pgx_output = _find_pgx_output(state)

    if pgx_output is None:
        # Tool Agent never ran the pgx-core task (e.g. Planner didn't
        # request it). Nothing to reason about yet.
        return {"hypotheses": []}

    if pgx_output.get("status") == "confirmed" and pgx_output.get("action") == "AVOID":
        hypotheses = [
            {
                "title": "DPYD-mediated fluorouracil toxicity",
                "confidence": 0.76,
                "evidence": ["pgx-core: AVOID", "symptom match"],
            },
            {
                "title": "Drug interaction",
                "confidence": 0.17,
                "evidence": [],
            },
            {
                "title": "Renal impairment",
                "confidence": 0.07,
                "evidence": [],
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
