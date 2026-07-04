"""
Minimal tool registry for the Tool Agent.

Not a framework — just a name -> callable map, so LangGraph nodes (or any
future dispatcher) can look up and invoke tools by name without importing
every tool module directly. Each registered callable must accept keyword
arguments and return a JSON-serializable dict, matching the envelope shape
expected by `InvestigationState.tool_outputs` (see
`packages/schemas/investigation_state.py::ToolOutput`).

Adding a new tool (drug interaction API, FHIR, PubMed, dose calculator,
etc.) later just means writing a wrapper module like `pgx_tool.py` and
registering it here.
"""

from __future__ import annotations

from typing import Any, Callable

from packages.tools.doc_intel_tool import extract_clinical_document
from packages.tools.pgx_tool import get_pharmacogenomic_recommendation
from packages.tools.retrieval_tool import search_evidence

ToolFn = Callable[..., dict[str, Any]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "pgx-core": get_pharmacogenomic_recommendation,
    "document-intelligence": extract_clinical_document,
    "evidence-search": search_evidence,
}


def call_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Look up a tool by name and invoke it. Raises KeyError with an explicit
    message if the tool isn't registered — fail loud here, since a missing
    tool is a wiring bug, not an evidence gap (that distinction matters:
    "insufficient_evidence" is a valid tool *result*; an unregistered tool
    name is a programming error).
    """
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{tool_name}' is not registered. "
            f"Available tools: {sorted(TOOL_REGISTRY)}"
        )
    return TOOL_REGISTRY[tool_name](**kwargs)
