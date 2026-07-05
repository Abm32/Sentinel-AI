"""
Retrieval Agent, round 2 — hypothesis-driven targeted evidence search.

Sits between the Hypothesis Agent's first pass and the Reporter:

    ... -> hypothesis (round 1: broad evidence) -> retrieval_2
        -> hypothesis (round 2: re-score with targeted evidence)
        -> reporter -> reviewer -> ...

WHY THIS NODE EXISTS: `retrieval.py` (round 1) reranks evidence against
fixed, Planner-task-derived queries decided before any hypothesis
exists — "find CPIC guideline text", "find lab trends", etc. That is
necessarily *general* evidence gathering: it cannot be targeted at a
specific explanation, because at that point in the graph there isn't
one yet. This node closes that gap: once the Hypothesis Agent has
proposed its top explanation, `retrieval_2` builds two NEW,
hypothesis-specific queries — one for evidence that would *support* it,
one for evidence that would *contradict* it — and runs each through the
same two-stage VultronRetriever pipeline (candidate search + rerank).
This is a second, genuinely different retrieval pass driven by
intermediate agent output, not a second call with the same query
repeated — i.e. exactly "retrieves more than once when it needs to",
targeted rather than reflexive.

It also makes VultronRetriever's reranking output directly load-bearing
on the investigation's conclusion in a way round-1 retrieval alone does
not: round-1 evidence feeds the Tool Agent and general context, but
round-2 evidence is reranked *specifically against the leading
hypothesis* and fed straight back into the Hypothesis Agent's re-score
— if VultronRetriever ranks a contradicting passage highly, that passage
is what the Hypothesis Agent sees on its second pass, and its confidence
can move because of it.

GATING: runs at most once per investigation (see
`InvestigationState.hypothesis_validated`), not once per Reviewer
reject -> re-investigate retry. Skipped entirely for a hypothesis set
that's already `unconfirmed` at 0% confidence (the honest-refusal path,
Path B) -- there is no leading explanation to validate evidence for,
and forcing a rerank pass there would be theater, not investigation.
"""

from __future__ import annotations

from packages.schemas.investigation_state import InvestigationState
from packages.tools.retrieval_tool import search_evidence
from packages.tools.vultron_rerank_tool import rerank_evidence

# Same shape as retrieval.py's constants — kept separate rather than
# imported, since round 2's candidate/keep counts are a deliberately
# different, narrower tuning (fewer, more targeted results per
# hypothesis) and coupling them to round 1's constants would make that
# an accident of shared code rather than a considered choice.
_CANDIDATES_PER_QUERY = 5
_KEEP_PER_QUERY = 2


def _latest_hypotheses(state: InvestigationState) -> list[dict]:
    """Selects the hypothesis set from the most recent hypothesis_node
    call — filtering on `round` alone is not enough, since retrieval_2
    itself causes hypothesis_node to run a second time within the SAME
    round (see hypothesis.py's `validation_pass` tagging). This node
    always wants the PRE-validation set (validation_pass == 0) — by the
    time retrieval_2_node runs, that is necessarily the only set that
    exists for the current round, since the post-validation call hasn't
    happened yet. Filtering explicitly rather than assuming this is
    defensive against `hypothesis_validated` state ever being reset for
    a legitimate reason (e.g. a future "re-validate on demand" feature)."""
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        return []
    current_round = state.get("retry_count", 0)
    latest = [
        h
        for h in hypotheses
        if h.get("round", 0) == current_round and h.get("validation_pass", 0) == 0
    ]
    return latest if latest else hypotheses


def _top_hypothesis(state: InvestigationState) -> dict | None:
    latest = _latest_hypotheses(state)
    if not latest:
        return None
    return max(latest, key=lambda h: h.get("confidence", 0.0))


def _build_queries(hypothesis: dict) -> tuple[str, str]:
    """Deterministic query construction from the hypothesis's own title
    — no chat-model call. Mirrors retrieval.py's fixed task->query
    mapping: the *shape* of query-building is deterministic, only the
    reranking step is a model call."""
    title = hypothesis.get("title", "")
    supporting_query = f"evidence supporting: {title}"
    contradicting_query = f"evidence against or alternative explanation for: {title}"
    return supporting_query, contradicting_query


def _pgx_core_insufficient(state: InvestigationState) -> bool:
    """True if pgx-core (packages/tools/pgx_tool.py) reported
    insufficient_evidence — i.e. no phenotype was available. When true,
    hypothesis.py's guardrail (`_violates_guardrail`) will reject ANY
    hypothesis re-score that names the demo gene with nonzero
    confidence, regardless of what retrieval_2 finds — so a validation
    pass here would do real, costly work (candidate search + a
    VultronRetriever rerank call) that gets thrown away deterministically
    on the next hypothesis_node call. Checking this directly, rather
    than only the top hypothesis's own confidence/status, catches the
    case where the LLM's first pass produces a non-zero-confidence
    hypothesis for a DIFFERENT (non-genotype-dependent) explanation
    while pgx-core still has nothing to confirm the pharmacogenomic one
    with — that combination would otherwise pass the "is there a
    leading hypothesis worth validating" check below even though the
    Path B refusal is the only place this investigation can actually
    land."""
    for output in state.get("tool_outputs", []):
        if output.get("tool") == "pgx-core" and output.get("status") == "insufficient_evidence":
            return True
    return False


def retrieval_2_node(state: InvestigationState) -> dict:
    """
    LangGraph node. Returns only the fields that changed:
    `retrieved_evidence` (merged via the `operator.add` reducer) and
    `hypothesis_validated` (set True so this node does not re-run on a
    later Reviewer reject -> re-investigate pass).
    """
    if state.get("hypothesis_validated"):
        return {}

    if _pgx_core_insufficient(state):
        return {"hypothesis_validated": True}

    top = _top_hypothesis(state)
    if top is None or top.get("status") == "unconfirmed" or top.get("confidence", 0.0) <= 0.0:
        # Nothing worth validating — the honest-refusal path (Path B)
        # or a hypothesis set the Hypothesis Agent itself scored at
        # zero confidence. See module docstring.
        return {"hypothesis_validated": True}

    supporting_query, contradicting_query = _build_queries(top)
    retrieved: list[dict] = []

    for query, label in (
        (supporting_query, "supporting"),
        (contradicting_query, "contradicting"),
    ):
        candidates = search_evidence(query, top_k=_CANDIDATES_PER_QUERY)
        if not candidates:
            continue

        reranked = rerank_evidence(query, candidates)

        for result in reranked[:_KEEP_PER_QUERY]:
            entry = dict(result)
            entry["retrieved_for_task"] = "hypothesis_validation"
            entry["validation_target"] = top.get("title")
            entry["validation_stance"] = label
            retrieved.append(entry)

    return {"retrieved_evidence": retrieved, "hypothesis_validated": True}
