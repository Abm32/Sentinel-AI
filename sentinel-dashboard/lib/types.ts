// Types mirroring packages/schemas/investigation_state.py and the
// report shape produced by packages/agents/reporter.py. Kept loose
// (optional fields, index-safe) since the backend's TypedDict/dict
// shapes aren't schema-validated on the wire — the dashboard should
// degrade gracefully rather than crash on an unexpected shape.

export type AgentName =
  | "Planner"
  | "Retrieval"
  | "Tool Agent"
  | "Hypothesis"
  | "Reporter"
  | "Reviewer";

export type AgentRunStatus = "pending" | "running" | "done";

export interface AgentStatus {
  name: AgentName;
  status: AgentRunStatus;
}

// packages/tools/retrieval_tool.py + vultron_rerank_tool.py candidate
// shape, and packages/agents/tool_agent.py tool_outputs. Both flow into
// the Evidence panel; fields vary by tool/source so this is kept wide.
export interface EvidenceItem {
  tool?: string;
  task?: string;
  source?: string;
  doc_type?: string;
  status?: string;
  gene?: string;
  drug?: string;
  phenotype?: string;
  diplotype?: string;
  action?: string;
  recommendation?: string;
  interpretation?: string;
  retrieved_for_task?: string;
  reranked_by?: string;
  citations?: string[];
  [key: string]: unknown;
}

// packages/agents/hypothesis.py::Hypothesis
export interface Hypothesis {
  title: string;
  confidence: number;
  status: "confirmed" | "unconfirmed" | string;
  supporting_evidence: string[];
  contradicting_evidence?: string[];
  blockers?: string[];
  round?: number;
}

// packages/agents/reviewer.py::ReviewResult
export interface ReviewIssue {
  type: string;
  description: string;
  action: string;
}

export interface ReviewEntry {
  verdict: "approved" | "rejected" | string;
  issues: ReviewIssue[];
  review_notes: string;
}

// packages/agents/reporter.py's report dict
export interface Report {
  case_id: string;
  incident: string;
  executive_summary: string;
  root_cause: {
    title: string;
    confidence: number;
    status: "confirmed" | "unconfirmed" | string;
  };
  supporting_evidence: string[];
  alternative_causes: { title: string; confidence: number }[];
  missing_evidence: string[];
  contradictions: unknown[];
  citations: string[];
  report_status: "draft" | "finalized" | string;
}

export interface TimelineEvent {
  day: string;
  label: string;
}

// InvestigationState, as returned by GET /api/investigations/{case_id}
// and the WebSocket's final {"done": true, "state": ...} message.
// packages/schemas/investigation_state.py::InvestigationState
export interface InvestigationState {
  case_id: string;
  incident: string;
  documents?: unknown[];
  tasks?: { task: string; priority: string; rationale?: string }[];
  retrieved_evidence?: EvidenceItem[];
  timeline?: unknown[];
  hypotheses?: Hypothesis[];
  tool_outputs?: EvidenceItem[];
  verified_facts?: unknown[];
  contradictions?: unknown[];
  review_history?: ReviewEntry[];
  review_issues?: ReviewIssue[];
  retry_count?: number;
  confidence?: number | null;
  report?: Report | null;
  status?: string;
}

// apps/api/routers/investigations.py WebSocket message shape:
//   {"node": "<node_name>", "state_update": {...}, "done": false}
//   {"done": true, "state": {...full final state...}}
export type StreamMessage =
  | { done: false; node: string; state_update: Partial<InvestigationState> }
  | { done: true; state: InvestigationState };

// Backend node_name -> display agent name
// (graph.py's node names: planner, retrieval, tool_agent, hypothesis, reporter, reviewer)
export const NODE_TO_AGENT: Record<string, AgentName> = {
  planner: "Planner",
  retrieval: "Retrieval",
  tool_agent: "Tool Agent",
  hypothesis: "Hypothesis",
  reporter: "Reporter",
  reviewer: "Reviewer",
};

export const AGENT_ORDER: AgentName[] = [
  "Planner",
  "Retrieval",
  "Tool Agent",
  "Hypothesis",
  "Reporter",
  "Reviewer",
];

export const CURRENT_ACTION_LABEL: Record<AgentName, string> = {
  Planner: "Building investigation plan...",
  Retrieval: "Searching and reranking evidence via VultronRetriever...",
  "Tool Agent": "Running pharmacogenomic and lab tools...",
  Hypothesis: "Scoring competing hypotheses...",
  Reporter: "Drafting investigation report...",
  Reviewer: "Reviewing report for evidentiary sufficiency...",
};
