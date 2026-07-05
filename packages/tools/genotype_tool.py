"""
Tool Agent adapter for patient-specific pharmacogenomic genotype
confirmation.

pgx_tool.py's `get_pharmacogenomic_recommendation` answers "given a
phenotype, what should we do about this drug?" — it is a
guideline/CPIC-level lookup, and its `phenotype` argument is assumed
already resolved. It does not, and should not, confirm that a specific
patient actually carries that phenotype. That gap is exactly what the
live Reviewer (packages/agents/reviewer.py, Kimi-K2.6-backed) flagged in
practice: a report whose only evidence is "CPIC guideline says DPYD
poor metabolizers should avoid fluorouracil" cites population-level
guidance, not confirmation that *this patient* has been tested and
found to be a poor metabolizer. That distinction is a legitimate
evidentiary gap, not the Reviewer being overly strict — this module
exists to let the investigation actually close it instead of routing
around it.

This is a stub for the hackathon demo, deliberately kept as simple and
honest about that as `tool_agent.py`'s `_LAB_STUB_RESULT`: in
production this would query a clinical genotyping lab's API (e.g. an
LIS/LIMS integration) or extract a pharmacogenomic test report via
Azure AI Document Intelligence (packages/tools/doc_intel_tool.py) if the
report arrived as an uploaded PDF. Neither exists yet, so this module
returns a hardcoded diplotype/phenotype result for the demo's DPYD case
and an explicit `insufficient_evidence` result for any other gene —
same "never fabricate, report the gap" contract as every other tool in
this project (see pgx_tool.py's module docstring).

Distinct from `_find_phenotype` in tool_agent.py: that function reads
phenotype evidence already present in `retrieved_evidence` (e.g. a
genomic report uploaded/seeded before the graph starts — the Path A/B
demo scenarios in packages/graph.py do this). This tool is the
opposite direction — it is *itself* the source of that confirmation,
called mid-investigation when the Reviewer specifically demands
patient-level genotype evidence that wasn't retrieved up front.
"""

from __future__ import annotations

from typing import TypedDict

# Hardcoded demo result: DPYD *2A/*2A -> Poor Metabolizer. Consistent
# with the rest of the project's fluorouracil/DPYD demo narrative
# (tool_agent.py's _DEMO_GENE, pgx_tool.py's worked example).
_DEMO_GENOTYPE: dict = {
    "tool": "genotype-confirmation",
    "patient_id": "DEMO-001",
    "gene": "DPYD",
    "allele1": "*2A",
    "allele2": "*2A",
    "diplotype": "*2A/*2A",
    "phenotype": "Poor Metabolizer",
    "test_date": "2026-06-28",
    "lab": "Molecular Pathology Lab",
    "source": "genotype_confirmation_tool",
    "status": "confirmed",
}


class GenotypeToolResult(TypedDict, total=False):
    tool: str
    patient_id: str
    gene: str
    allele1: str
    allele2: str
    diplotype: str
    phenotype: str
    test_date: str
    lab: str
    source: str
    status: str
    blocker: str


def get_genotype_confirmation(gene: str, patient_id: str = "") -> GenotypeToolResult:
    """
    Retrieve patient-specific pharmacogenomic genotype/phenotype
    confirmation for `gene`.

    Called by the Tool Agent (packages/agents/tool_agent.py) on a
    re-investigation pass when the Reviewer's rejection issues
    specifically ask for genotype/phenotype confirmation — see
    `_genotype_requested` in that module. Not called on the first pass;
    this tool exists to answer a Reviewer objection, not to pre-empt it
    (the Planner's `confirm_pharmacogenomic_genotype` task expresses the
    intent that this evidence should eventually exist, but the Tool
    Agent only calls this function in response to it being missing).

    Args:
        gene: Gene symbol, e.g. "DPYD". Case-insensitive.
        patient_id: Reserved for a future real lab-API integration
            (looking up a specific patient's test result). Unused by
            the current hardcoded stub.

    Returns:
        The confirmed diplotype/phenotype record for the demo gene, or
        an explicit `status: "insufficient_evidence"` record naming the
        gap — never a fabricated genotype for a gene with no test result
        on file, same no-guessing contract as every other tool in this
        project.
    """
    if gene.upper() == "DPYD":
        result = dict(_DEMO_GENOTYPE)
        if patient_id:
            result["patient_id"] = patient_id
        return result  # type: ignore[return-value]

    return {
        "tool": "genotype-confirmation",
        "gene": gene,
        "status": "insufficient_evidence",
        "blocker": f"No genotype test result available for {gene}",
        "source": "genotype_confirmation_tool",
    }
