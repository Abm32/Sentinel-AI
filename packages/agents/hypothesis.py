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
           the demo's headline moment. If lab_trends evidence is also
           present (added after a Reviewer rejection asking for it), it's
           folded into the top hypothesis's evidence list — this doesn't
           change the confidence number itself (a real/LLM hypothesis
           agent would re-score; this rule-based stub keeps the same 76%
           to isolate what's being tested: evidence *breadth*, which is
           the Reviewer's actual complaint, not the number).

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


def _find_output(state: InvestigationState, tool_name: str) -> dict | None:
    for output in state.get("tool_outputs", []):
        if output.get("tool") == tool_name:
            return output
    return None


def hypothesis_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed: `hypotheses`
    (merged via the `operator.add` reducer).

    Note: because `hypotheses` accumulates via operator.add, this node
    must not be naively re-appended on every pass through the graph — the
    graph is responsible for clearing/regenerating hypotheses on retry.
    Since LangGraph reducers only support additive merges (not replace),
    the Reporter reads only the LATEST hypothesis set it needs; see
    reporter.py's `_latest_hypotheses` helper for how that's resolved.
    """
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
