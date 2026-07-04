"""
The three-node investigation chain: START -> planner -> tool_agent ->
hypothesis -> END.

Run directly for a smoke test covering both demo paths:

    python packages/graph.py

Path A (confirmed): retrieved_evidence contains a DPYD "Poor Metabolizer"
genomic report -> pgx-core returns AVOID -> Hypothesis Agent emits the
76%/17%/7% competing-hypothesis set.

Path B (the refusal — the actual demo moment): no genomic-phenotype
evidence retrieved -> pgx-core returns {} (insufficient_evidence) ->
Hypothesis Agent emits a single zero-confidence, explicitly-blocked
hypothesis naming the missing evidence, instead of guessing.
"""

from __future__ import annotations

import json

from langgraph.graph import END, StateGraph

from packages.agents.hypothesis import hypothesis_node
from packages.agents.planner import planner_node
from packages.agents.tool_agent import tool_agent_node
from packages.schemas.investigation_state import InvestigationState, new_investigation


def build_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("planner", planner_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("hypothesis", hypothesis_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_agent")
    graph.add_edge("tool_agent", "hypothesis")
    graph.add_edge("hypothesis", END)

    return graph.compile()


_INCIDENT_TEXT = (
    "Patient admitted after fluorouracil therapy. Symptoms: neutropenia, "
    "mucositis, diarrhea, fever."
)


def run_path_a() -> InvestigationState:
    """Confirmed path: a DPYD Poor Metabolizer genomic report is available."""
    app = build_graph()
    state = new_investigation(case_id="demo-case-A", incident=_INCIDENT_TEXT)
    state["retrieved_evidence"] = [
        {
            "source": "genomic_report",
            "gene": "DPYD",
            "phenotype": "Poor Metabolizer",
        }
    ]
    return app.invoke(state)


def run_path_b() -> InvestigationState:
    """Refusal path: no genomic-phenotype evidence was retrieved."""
    app = build_graph()
    state = new_investigation(case_id="demo-case-B", incident=_INCIDENT_TEXT)
    # retrieved_evidence deliberately left empty — no genotype available.
    return app.invoke(state)


def _print_result(label: str, result: InvestigationState) -> None:
    print(f"=== {label} ===")
    print("tasks:")
    for t in result["tasks"]:
        print(f"  - {t}")
    print("tool_outputs:")
    for o in result["tool_outputs"]:
        print(f"  - {json.dumps(o)}")
    print("hypotheses:")
    for h in result["hypotheses"]:
        print(f"  - {json.dumps(h)}")
    print()


if __name__ == "__main__":
    _print_result("Path A — confirmed (phenotype available)", run_path_a())
    _print_result("Path B — refusal (no phenotype / the demo moment)", run_path_b())
