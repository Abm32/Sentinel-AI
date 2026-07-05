"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  FileWarning,
  ScrollText,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { Report, ReviewEntry } from "@/lib/types";

interface ReportPanelProps {
  report: Report | null;
  reviewHistory: ReviewEntry[];
}

export function ReportPanel({ report, reviewHistory }: ReportPanelProps) {
  if (!report) {
    return (
      <div className="glass-panel rounded-2xl p-6 mt-4">
        <h2 className="text-[11px] font-semibold tracking-[0.15em] text-slate-400 font-mono flex items-center gap-1.5">
          <ScrollText size={12} /> INVESTIGATION REPORT
        </h2>
        <p className="text-slate-600 text-sm mt-3">
          Report will appear here once the investigation completes.
        </p>
      </div>
    );
  }

  const unconfirmed = report.root_cause?.status === "unconfirmed";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="glass-panel rounded-2xl p-6 mt-4"
    >
      <div className="flex justify-between items-center mb-5">
        <h2 className="text-[11px] font-semibold tracking-[0.15em] text-slate-400 font-mono flex items-center gap-1.5">
          <ScrollText size={12} /> INVESTIGATION REPORT
        </h2>
        <motion.span
          key={report.report_status}
          initial={{ scale: 0.85, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className={`text-xs px-2.5 py-1 rounded-full font-mono flex items-center gap-1.5 ${
            report.report_status === "finalized"
              ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/25"
              : "bg-amber-500/10 text-amber-300 border border-amber-500/25"
          }`}
        >
          {report.report_status === "finalized" ? (
            <CheckCircle2 size={12} />
          ) : (
            <AlertCircle size={12} />
          )}
          {report.report_status?.toUpperCase()}
        </motion.span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <div className="mb-5">
            <h3 className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
              Executive Summary
            </h3>
            <p className="text-sm text-slate-200 leading-relaxed">{report.executive_summary}</p>
          </div>

          <div className="mb-5">
            <h3 className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
              Root Cause
            </h3>
            <p className="text-sm text-white font-medium">{report.root_cause?.title}</p>
            <div className="flex items-center gap-3 mt-1.5">
              <span
                className={`text-xs font-mono font-semibold ${unconfirmed ? "text-amber-400" : "text-emerald-400"}`}
              >
                {Math.round((report.root_cause?.confidence || 0) * 100)}% confidence
              </span>
              <span
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded-full ${
                  unconfirmed
                    ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                    : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                }`}
              >
                {report.root_cause?.status}
              </span>
            </div>
          </div>

          {report.supporting_evidence?.length > 0 && (
            <div className="mb-5">
              <h3 className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
                Supporting Evidence
              </h3>
              <ul className="text-sm text-slate-300 space-y-1.5">
                {report.supporting_evidence.map((e, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-sky-400/70 mt-0.5">&#8226;</span>
                    <span>{e}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {report.alternative_causes?.length > 0 && (
            <div className="mb-5">
              <h3 className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
                Alternative Causes
              </h3>
              <ul className="text-sm text-slate-400 space-y-1.5">
                {report.alternative_causes.map((a, i) => (
                  <li key={i} className="flex justify-between">
                    <span>{a.title}</span>
                    <span className="font-mono text-slate-500">{Math.round(a.confidence * 100)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {report.missing_evidence?.length > 0 && (
            <div className="mb-5">
              <h3 className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5 flex items-center gap-1.5">
                <FileWarning size={11} className="text-amber-500" /> Missing Evidence
              </h3>
              <ul className="text-sm text-amber-300/90 space-y-1.5">
                {report.missing_evidence.map((m, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="mt-0.5">&#8226;</span>
                    <span>{m}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {report.citations?.length > 0 && (
            <div>
              <h3 className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
                Citations
              </h3>
              <div className="flex gap-2 flex-wrap">
                {report.citations.map((c, i) => (
                  <a
                    key={i}
                    href={`https://pubmed.ncbi.nlm.nih.gov/${c}/`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-sky-400 hover:text-sky-300 font-mono flex items-center gap-1 transition-colors"
                  >
                    PMID {c} <ExternalLink size={10} />
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>

        <div>
          <h3 className="text-[11px] uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
            <ShieldCheck size={11} /> Review History
          </h3>
          {reviewHistory.length === 0 && (
            <p className="text-sm text-slate-600">No review yet.</p>
          )}
          <div className="space-y-2.5">
            <AnimatePresence>
              {reviewHistory.map((r, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className={`text-xs p-3.5 rounded-xl border ${
                    r.verdict === "approved"
                      ? "bg-emerald-500/[0.04] border-emerald-500/15"
                      : "bg-rose-500/[0.04] border-rose-500/15"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-slate-500 font-mono">Pass {i + 1}</span>
                    <span
                      className={`font-mono font-semibold flex items-center gap-1 ${
                        r.verdict === "approved" ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      {r.verdict === "approved" ? (
                        <CheckCircle2 size={12} />
                      ) : (
                        <XCircle size={12} />
                      )}
                      {r.verdict.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-slate-400 leading-relaxed">{r.review_notes}</p>
                  {r.issues?.length > 0 && (
                    <ul className="mt-2.5 space-y-2">
                      {r.issues.map((issue, j) => (
                        <li key={j} className="border-l-2 border-rose-500/40 pl-2.5">
                          <span className="text-rose-400 font-mono text-[10px] uppercase flex items-center gap-1">
                            <ShieldAlert size={10} /> {issue.type}
                          </span>
                          <p className="text-slate-400 mt-0.5">{issue.description}</p>
                          <p className="text-slate-500 mt-0.5">
                            <span className="text-slate-600">Action: </span>
                            {issue.action}
                          </p>
                        </li>
                      ))}
                    </ul>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
          {reviewHistory.length > 1 && (
            <div className="mt-4 flex items-center gap-2 text-[11px] font-mono text-slate-500 flex-wrap">
              {reviewHistory.map((r, i) => (
                <span key={i} className="flex items-center gap-2">
                  <span className={r.verdict === "approved" ? "text-emerald-400" : "text-rose-400"}>
                    {r.verdict === "approved" ? "Approved" : "Rejected"}
                  </span>
                  {i < reviewHistory.length - 1 && <span className="text-slate-700">&rarr;</span>}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
