"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, ArrowRight, ShieldAlert, Sparkles } from "lucide-react";
import { AgentStatusPanel } from "@/components/AgentStatusPanel";
import { HypothesesPanel } from "@/components/HypothesesPanel";
import { EvidencePanel } from "@/components/EvidencePanel";
import { ReportPanel } from "@/components/ReportPanel";
import { Timeline } from "@/components/Timeline";
import {
  AgentStatus,
  EvidenceItem,
  Hypothesis,
  InvestigationState,
  NODE_TO_AGENT,
  Report,
  ReviewEntry,
  StreamMessage,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type CaseType = "A" | "B";

const CASE_INCIDENTS: Record<CaseType, string> = {
  A: (
    "Patient admitted after fluorouracil therapy with severe neutropenia, " +
    "mucositis, diarrhea, and fever. DPYD genotype available: *2A/*2A."
  ),
  B: (
    "Patient admitted after fluorouracil therapy with severe neutropenia, " +
    "mucositis, diarrhea, and fever. No pharmacogenomic testing on file."
  ),
};

// apps/api/routers/investigations.py's CreateInvestigationRequest accepts
// an optional retrieved_evidence seed list. Free text in `incident`
// alone is NOT parsed into structured evidence by any graph node —
// packages/agents/tool_agent.py::_find_phenotype only reads this exact
// shape. Case A needs this seeded to reach the confirmed-root-cause /
// reject -> re-investigate -> approve loop instead of the refusal path;
// Case B intentionally omits it.
const CASE_EVIDENCE_SEED: Record<CaseType, Record<string, unknown>[] | undefined> = {
  A: [{ source: "genomic_report", gene: "DPYD", phenotype: "Poor Metabolizer" }],
  B: undefined,
};

function wsUrlFor(caseId: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/api/investigations/${caseId}/stream`;
}

function newCaseId(caseType: CaseType): string {
  const suffix = Date.now().toString(36);
  return `demo-case-${caseType.toLowerCase()}-${suffix}`;
}

export default function Dashboard() {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [retrievedEvidence, setRetrievedEvidence] = useState<EvidenceItem[]>([]);
  const [toolOutputs, setToolOutputs] = useState<EvidenceItem[]>([]);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [reviewHistory, setReviewHistory] = useState<ReviewEntry[]>([]);
  const [investigating, setInvestigating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const markAgentDone = useCallback((node: string) => {
    const agentName = NODE_TO_AGENT[node];
    if (!agentName) return;
    setAgents((prev) =>
      prev.some((a) => a.name === agentName) ? prev : [...prev, { name: agentName, status: "done" }]
    );
  }, []);

  const applyStateUpdate = useCallback((node: string, update: Partial<InvestigationState>) => {
    markAgentDone(node);
    if (update.retrieved_evidence) {
      setRetrievedEvidence((prev) => [...prev, ...update.retrieved_evidence!]);
    }
    if (update.tool_outputs) {
      setToolOutputs((prev) => [...prev, ...update.tool_outputs!]);
    }
    if (update.hypotheses) {
      setHypotheses((prev) => [...prev, ...update.hypotheses!]);
    }
    if (update.report !== undefined) {
      setReport(update.report ?? null);
    }
    if (update.review_history) {
      setReviewHistory((prev) => [...prev, ...update.review_history!]);
    }
    // Reviewer rejection clears tool_outputs' accumulation contract on
    // the backend? No — tool_outputs/hypotheses/etc. are operator.add
    // reducers and accumulate across retry passes; the report is the
    // only field the backend explicitly nulls out on rejection
    // (reviewer.py returns {"report": None} on reject). Handled above
    // via the `!== undefined` check so a reject-triggered null clears
    // the panel instead of being ignored.
  }, [markAgentDone]);

  const applyFinalState = useCallback((state: InvestigationState) => {
    setRetrievedEvidence(state.retrieved_evidence ?? []);
    setToolOutputs(state.tool_outputs ?? []);
    setHypotheses(state.hypotheses ?? []);
    setReport(state.report ?? null);
    setReviewHistory(state.review_history ?? []);
    setAgents((prev) => {
      // Mark every agent done on completion, since the final snapshot
      // doesn't carry a per-node breakdown.
      const done = new Set(prev.map((a) => a.name));
      const all = Object.values(NODE_TO_AGENT);
      return all.map((name) => ({
        name,
        status: done.has(name) || state.status === "completed" ? "done" : "pending",
      }));
    });
  }, []);

  const resetPanels = useCallback(() => {
    setAgents([]);
    setRetrievedEvidence([]);
    setToolOutputs([]);
    setHypotheses([]);
    setReport(null);
    setReviewHistory([]);
    setError(null);
  }, []);

  const startInvestigation = useCallback(
    async (caseType: CaseType) => {
      wsRef.current?.close();
      resetPanels();
      setInvestigating(true);

      const caseId = newCaseId(caseType);
      setActiveCaseId(caseId);

      try {
        const res = await fetch(`${API_BASE}/api/investigations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            case_id: caseId,
            incident: CASE_INCIDENTS[caseType],
            retrieved_evidence: CASE_EVIDENCE_SEED[caseType],
          }),
        });

        if (!res.ok) {
          const detail = await res.text();
          throw new Error(`Failed to start investigation (${res.status}): ${detail}`);
        }

        const ws = new WebSocket(wsUrlFor(caseId));
        wsRef.current = ws;

        ws.onmessage = (event) => {
          const data: StreamMessage = JSON.parse(event.data);
          if (data.done) {
            applyFinalState(data.state);
            setInvestigating(false);
            return;
          }
          applyStateUpdate(data.node, data.state_update);
        };

        ws.onerror = () => {
          setError("WebSocket connection error — is the API running at " + API_BASE + "?");
          setInvestigating(false);
        };

        ws.onclose = () => {
          setInvestigating(false);
        };
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to start investigation.");
        setInvestigating(false);
      }
    },
    [applyFinalState, applyStateUpdate, resetPanels]
  );

  return (
    <div className="min-h-screen text-slate-100 p-6 md:p-8 max-w-7xl mx-auto w-full">
      <motion.header
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="mb-8"
      >
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 200, damping: 15 }}
              className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500/20 to-violet-500/20 border border-white/10 flex items-center justify-center shrink-0"
            >
              <Activity size={18} className="text-sky-300" />
            </motion.div>
            <div>
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight gradient-text font-mono">
                SENTINEL CLINICAL
              </h1>
              <p className="text-sm text-slate-400 mt-0.5">
                Autonomous AI investigation engine for adverse drug events
              </p>
            </div>
          </div>
          <AnimatePresence>
            {activeCaseId && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-xs font-mono text-slate-500 bg-white/[0.03] border border-white/10 rounded-full px-3 py-1.5"
              >
                case: {activeCaseId}
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <div className="mt-5 flex gap-3 flex-wrap items-center">
          <motion.button
            whileHover={{ scale: investigating ? 1 : 1.02, y: investigating ? 0 : -1 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => startInvestigation("A")}
            disabled={investigating}
            className="group relative px-4 py-2.5 rounded-xl text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-all overflow-hidden bg-gradient-to-r from-sky-600 to-sky-500 shadow-[0_0_0_1px_rgba(56,189,248,0.3),0_8px_24px_-8px_rgba(56,189,248,0.5)] flex items-center gap-2"
          >
            <Sparkles size={14} />
            Demo Case A — Genotype Available
            <ArrowRight size={14} className="opacity-0 -ml-2 group-hover:opacity-100 group-hover:ml-0 transition-all" />
          </motion.button>
          <motion.button
            whileHover={{ scale: investigating ? 1 : 1.02, y: investigating ? 0 : -1 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => startInvestigation("B")}
            disabled={investigating}
            className="group relative px-4 py-2.5 rounded-xl text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-all overflow-hidden bg-gradient-to-r from-amber-600 to-orange-500 shadow-[0_0_0_1px_rgba(251,146,60,0.3),0_8px_24px_-8px_rgba(251,146,60,0.5)] flex items-center gap-2"
          >
            <ShieldAlert size={14} />
            Demo Case B — No Genotype (Refusal)
            <ArrowRight size={14} className="opacity-0 -ml-2 group-hover:opacity-100 group-hover:ml-0 transition-all" />
          </motion.button>
          <AnimatePresence>
            {investigating && (
              <motion.span
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center gap-2 text-xs font-mono text-sky-300 bg-sky-500/10 border border-sky-500/20 rounded-full px-3 py-1.5"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-sky-400 pulse-ring" />
                investigation running...
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <AnimatePresence>
          {error && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-3 text-sm text-rose-300 font-mono border border-rose-500/20 bg-rose-500/[0.06] rounded-xl px-4 py-2.5"
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>
      </motion.header>

      <motion.div
        initial="hidden"
        animate="visible"
        variants={{
          hidden: {},
          visible: { transition: { staggerChildren: 0.08 } },
        }}
        className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5"
      >
        {[
          <EvidencePanel key="evidence" retrievedEvidence={retrievedEvidence} toolOutputs={toolOutputs} />,
          <AgentStatusPanel key="agents" agents={agents} investigating={investigating} />,
          <HypothesesPanel key="hypotheses" hypotheses={hypotheses} />,
        ].map((panel, i) => (
          <motion.div
            key={i}
            variants={{
              hidden: { opacity: 0, y: 16 },
              visible: { opacity: 1, y: 0 },
            }}
            transition={{ duration: 0.45, ease: "easeOut" }}
          >
            {panel}
          </motion.div>
        ))}
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25, duration: 0.45, ease: "easeOut" }}
      >
        <Timeline events={[]} />
      </motion.div>
      <ReportPanel report={report} reviewHistory={reviewHistory} />
    </div>
  );
}

