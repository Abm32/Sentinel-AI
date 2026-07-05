"use client";

import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, HelpCircle } from "lucide-react";
import { Hypothesis } from "@/lib/types";

interface HypothesesPanelProps {
  hypotheses: Hypothesis[];
}

export function HypothesesPanel({ hypotheses }: HypothesesPanelProps) {
  // Mirrors packages/agents/reporter.py::_latest_hypotheses -- within
  // the latest round, prefer the post-validation set (validation_pass
  // === 1, i.e. re-scored after retrieval_2.py's targeted
  // VultronRetriever pass) over the pre-validation set from earlier in
  // the same round. `round` alone is not enough to disambiguate: both
  // the pre- and post-validation hypothesis_node calls share the same
  // round within one Reviewer retry pass.
  const latestRound = hypotheses.length
    ? Math.max(...hypotheses.map((h) => h.round ?? 0))
    : 0;
  const thisRound = hypotheses.filter((h) => (h.round ?? 0) === latestRound);
  const postValidation = thisRound.filter((h) => h.validation_pass === 1);
  const latest = postValidation.length > 0 ? postValidation : thisRound;
  const sorted = [...latest].sort((a, b) => b.confidence - a.confidence);
  const top = sorted[0];

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col">
      <h2 className="text-[11px] font-semibold tracking-[0.15em] text-slate-400 mb-4 font-mono">
        HYPOTHESES
      </h2>

      {sorted.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 text-slate-600 py-8">
          <HelpCircle size={22} className="opacity-40" />
          <p className="text-sm">Awaiting investigation...</p>
        </div>
      )}

      <div className="space-y-4 flex-1">
        <AnimatePresence>
          {sorted.map((h, i) => {
            const unconfirmed = h.status === "unconfirmed";
            return (
              <motion.div
                key={`${h.title}-${latestRound}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08, duration: 0.4, ease: "easeOut" }}
              >
                <div className="flex justify-between items-baseline text-sm gap-2">
                  <span className="text-slate-200">
                    <span className="text-slate-500 font-mono mr-1.5 text-xs">H{i + 1}</span>
                    {h.title}
                  </span>
                  <motion.span
                    className={`font-mono shrink-0 text-sm font-semibold ${
                      unconfirmed ? "text-amber-400" : "text-slate-100"
                    }`}
                    key={h.confidence}
                    initial={{ scale: 1.3, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 300, damping: 15 }}
                  >
                    {Math.round(h.confidence * 100)}%
                  </motion.span>
                </div>
                <div className="mt-2 h-2 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      unconfirmed
                        ? "bg-gradient-to-r from-amber-500 to-amber-300"
                        : "bg-gradient-to-r from-sky-400 via-violet-400 to-sky-300"
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(h.confidence * 100, unconfirmed ? 2 : 0)}%` }}
                    transition={{ type: "spring", stiffness: 60, damping: 16 }}
                  />
                </div>
                {unconfirmed && h.blockers && h.blockers.length > 0 && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="text-xs text-amber-400/90 mt-2 flex items-start gap-1.5"
                  >
                    <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                    {h.blockers.join("; ")}
                  </motion.p>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {top && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4 pt-4 border-t border-white/5 text-xs"
          >
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Status</span>
              {top.status === "unconfirmed" ? (
                <span className="flex items-center gap-1 text-amber-400 font-medium">
                  <AlertTriangle size={12} /> unconfirmed
                </span>
              ) : (
                <span className="flex items-center gap-1 text-emerald-400 font-medium">
                  <CheckCircle2 size={12} /> confirmed
                </span>
              )}
            </div>
            {top.status === "unconfirmed" && top.blockers && top.blockers.length > 0 && (
              <div className="text-slate-500 mt-1.5">
                Blocker: <span className="text-slate-300">{top.blockers[0]}</span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
