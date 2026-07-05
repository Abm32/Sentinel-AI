"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ExternalLink, FileText, FlaskConical, Inbox } from "lucide-react";
import { EvidenceItem } from "@/lib/types";

interface EvidencePanelProps {
  retrievedEvidence: EvidenceItem[];
  toolOutputs: EvidenceItem[];
}

function labelFor(item: EvidenceItem): string {
  // packages/agents/retrieval_2.py's hypothesis-driven validation pass
  // -- label with the stance (supporting/contradicting) rather than
  // falling back to the generic doc source, so it's visually distinct
  // from round-1 general evidence in the panel.
  if (item.retrieved_for_task === "hypothesis_validation" && item.validation_stance) {
    return `VultronRetriever round 2 — ${item.validation_stance}`;
  }
  return (
    item.tool ??
    item.source ??
    item.doc_type ??
    item.task ??
    "evidence"
  ).toString();
}

function summaryFor(item: EvidenceItem): string {
  if (item.status === "insufficient_evidence") {
    return typeof item.blocker === "string"
      ? item.blocker
      : "Insufficient evidence — not retrieved.";
  }
  // packages/tools/retrieval_tool.py / vultron_rerank_tool.py candidate
  // shape: free-text passage in `content`, human-readable origin in
  // `source`. This is the shape most retrieved_evidence items actually
  // arrive in — checked before the tool_outputs-style fields below.
  if (typeof item.content === "string") return item.content;
  if (item.interpretation) return String(item.interpretation);
  if (item.recommendation) return String(item.recommendation);
  if (item.phenotype || item.diplotype) {
    return [item.diplotype, item.phenotype].filter(Boolean).join(" — ");
  }
  if (typeof item.title === "string") return item.title;
  return "";
}

// packages/agents/tool_agent.py writes an explicit {"task": ..., "status":
// "not_implemented"} entry for every Planner task it doesn't actually
// have a tool for (retrieve_fda_label, retrieve_cpic_guidelines, etc.)
// — deliberate internal traceability/bookkeeping, not evidence. Showing
// these in the Evidence panel makes a working investigation look
// broken, since the SAME topics (FDA label, CPIC guideline text) are
// usually already covered by real retrieved_evidence from the
// Retrieval Agent's search+rerank pass — the stub and the real
// evidence just live in two different state fields. Filtered out here
// rather than upstream, since tool_outputs' raw shape is still useful
// for debugging/logging even if it's not demo-facing.
function isBookkeepingStub(item: EvidenceItem): boolean {
  return item.status === "not_implemented";
}

// retrieval.py runs a separate search+rerank pass per Planner task, and
// several tasks map to overlapping queries (see _TASK_TO_QUERY) — the
// same CPIC/FDA passage legitimately gets pulled back for more than one
// task's top-3. De-duplicate by content (or by label+summary if content
// is absent) so the panel doesn't repeat the same passage 2-3 times.
function dedupeByContent(items: EvidenceItem[]): EvidenceItem[] {
  const seen = new Set<string>();
  const result: EvidenceItem[] = [];
  for (const item of items) {
    const key = typeof item.content === "string" ? item.content : `${labelFor(item)}::${summaryFor(item)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(item);
  }
  return result;
}

function EvidenceRow({ item, index }: { item: EvidenceItem; index: number }) {
  const failed = item.status === "insufficient_evidence";
  const citations = Array.isArray(item.citations) ? item.citations : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index * 0.04, 0.4), duration: 0.3 }}
      className="rounded-lg p-3 bg-white/[0.02] border border-white/[0.06] hover:border-white/[0.12] hover:bg-white/[0.035] transition-colors"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-mono uppercase tracking-wide text-slate-400">
          {labelFor(item)}
        </span>
        {item.status && (
          <span
            className={`text-[10px] font-mono px-1.5 py-0.5 rounded-full ${
              failed
                ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
            }`}
          >
            {String(item.status)}
          </span>
        )}
      </div>
      {summaryFor(item) && (
        <p className="text-sm text-slate-200 mt-1.5 leading-snug">{summaryFor(item)}</p>
      )}
      {citations.length > 0 && (
        <div className="flex gap-2 flex-wrap mt-2">
          {citations.map((c, i) => (
            <a
              key={i}
              href={`https://pubmed.ncbi.nlm.nih.gov/${c}/`}
              target="_blank"
              rel="noreferrer"
              className="text-[11px] text-sky-400 hover:text-sky-300 font-mono flex items-center gap-1 transition-colors"
            >
              PMID {String(c)} <ExternalLink size={10} />
            </a>
          ))}
        </div>
      )}
    </motion.div>
  );
}

export function EvidencePanel({ retrievedEvidence, toolOutputs }: EvidencePanelProps) {
  // Tool results: pgx-core, genotype-confirmation, lab_trends — real
  // computed/deterministic answers, including the honest
  // insufficient_evidence refusal. Bookkeeping stubs filtered out.
  const toolResults = toolOutputs.filter((item) => !isBookkeepingStub(item));
  // Retrieved documents: search+rerank passages (CPIC guideline, FDA
  // label, lab trend text, drug interaction checks), de-duplicated.
  const documents = dedupeByContent(retrievedEvidence);

  const isEmpty = toolResults.length === 0 && documents.length === 0;

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col">
      <h2 className="text-[11px] font-semibold tracking-[0.15em] text-slate-400 mb-4 font-mono">
        EVIDENCE
      </h2>
      <div className="space-y-5 flex-1 overflow-y-auto max-h-[26rem] pr-1 thin-scroll">
        <AnimatePresence>
          {isEmpty && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center gap-2 text-slate-600 py-8"
            >
              <Inbox size={22} className="opacity-40" />
              <p className="text-sm">No evidence retrieved yet.</p>
            </motion.div>
          )}
        </AnimatePresence>

        {toolResults.length > 0 && (
          <div>
            <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-1.5">
              <FlaskConical size={11} /> Tool Results
            </h3>
            <div className="space-y-2">
              {toolResults.map((item, i) => (
                <EvidenceRow key={`tool-${i}`} item={item} index={i} />
              ))}
            </div>
          </div>
        )}

        {documents.length > 0 && (
          <div>
            <h3 className="text-[10px] uppercase tracking-wider text-slate-500 mb-2 flex items-center gap-1.5">
              <FileText size={11} /> Retrieved Documents
            </h3>
            <div className="space-y-2">
              {documents.map((item, i) => (
                <EvidenceRow key={`doc-${i}`} item={item} index={i} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
