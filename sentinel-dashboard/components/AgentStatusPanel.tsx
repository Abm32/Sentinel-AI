"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Brain,
  Check,
  CircleDashed,
  ClipboardList,
  FileSearch,
  Loader2,
  ScanSearch,
  ShieldCheck,
  Wrench,
} from "lucide-react";
import { AGENT_ORDER, AgentName, AgentStatus, CURRENT_ACTION_LABEL } from "@/lib/types";

interface AgentStatusPanelProps {
  agents: AgentStatus[];
  investigating: boolean;
}

function statusFor(agents: AgentStatus[], name: AgentName, investigating: boolean, index: number) {
  const found = agents.find((a) => a.name === name);
  if (found) return found.status;
  const firstPendingIndex = AGENT_ORDER.findIndex(
    (n) => !agents.some((a) => a.name === n)
  );
  if (investigating && index === firstPendingIndex) return "running";
  return "pending";
}

const AGENT_ICON: Record<AgentName, React.ElementType> = {
  Planner: ClipboardList,
  Retrieval: ScanSearch,
  "Tool Agent": Wrench,
  Hypothesis: Brain,
  Reporter: FileSearch,
  Reviewer: ShieldCheck,
};

export function AgentStatusPanel({ agents, investigating }: AgentStatusPanelProps) {
  const currentAgent = investigating
    ? AGENT_ORDER.find(
        (name) => statusFor(agents, name, investigating, AGENT_ORDER.indexOf(name)) === "running"
      )
    : undefined;

  const allDone = AGENT_ORDER.every((name) => agents.some((a) => a.name === name));
  const doneCount = AGENT_ORDER.filter((name) => agents.some((a) => a.name === name)).length;

  let currentLabel: string;
  if (currentAgent) {
    currentLabel = CURRENT_ACTION_LABEL[currentAgent];
  } else if (investigating) {
    currentLabel = "Finalizing...";
  } else if (allDone) {
    currentLabel = "Investigation complete.";
  } else {
    currentLabel = "Idle — start an investigation.";
  }

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col relative overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-[11px] font-semibold tracking-[0.15em] text-slate-400 font-mono">
          LIVE INVESTIGATION
        </h2>
        <span className="text-[10px] font-mono text-slate-500">
          {doneCount}/{AGENT_ORDER.length}
        </span>
      </div>

      {/* progress rail */}
      <div className="relative mb-5 h-1 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-sky-400 via-violet-400 to-emerald-400"
          initial={{ width: 0 }}
          animate={{ width: `${(doneCount / AGENT_ORDER.length) * 100}%` }}
          transition={{ type: "spring", stiffness: 80, damping: 20 }}
        />
      </div>

      <div className="space-y-1.5 flex-1">
        {AGENT_ORDER.map((name, i) => {
          const status = statusFor(agents, name, investigating, i);
          const Icon = AGENT_ICON[name];
          const isRunning = status === "running";
          const isDone = status === "done";

          return (
            <motion.div
              key={name}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05, duration: 0.35, ease: "easeOut" }}
              className={`flex items-center gap-3 rounded-lg px-2.5 py-2 transition-colors ${
                isRunning ? "bg-sky-500/10" : isDone ? "bg-emerald-500/[0.04]" : ""
              }`}
            >
              <div className="relative flex items-center justify-center w-7 h-7 shrink-0">
                {isRunning && (
                  <span className="absolute inset-0 rounded-full bg-sky-400/70 pulse-ring" />
                )}
                <div
                  className={`relative flex items-center justify-center w-7 h-7 rounded-full border transition-colors ${
                    isDone
                      ? "bg-emerald-500/15 border-emerald-400/40 text-emerald-300"
                      : isRunning
                      ? "bg-sky-500/15 border-sky-400/50 text-sky-300"
                      : "bg-white/[0.03] border-white/10 text-slate-600"
                  }`}
                >
                  <AnimatePresence mode="wait" initial={false}>
                    {isDone ? (
                      <motion.span
                        key="done"
                        initial={{ scale: 0, rotate: -45 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ type: "spring", stiffness: 400, damping: 18 }}
                      >
                        <Check size={14} strokeWidth={2.5} />
                      </motion.span>
                    ) : isRunning ? (
                      <motion.span
                        key="running"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        <Loader2 size={14} className="animate-spin" strokeWidth={2.5} />
                      </motion.span>
                    ) : (
                      <motion.span key="pending" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <Icon size={13} strokeWidth={2} />
                      </motion.span>
                    )}
                  </AnimatePresence>
                </div>
              </div>

              <span
                className={`text-sm font-medium transition-colors ${
                  isDone
                    ? "text-emerald-200"
                    : isRunning
                    ? "text-sky-200"
                    : "text-slate-500"
                }`}
              >
                {name}
              </span>

              {isRunning && (
                <motion.span
                  className="ml-auto text-[10px] font-mono text-sky-400/80"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: [0.4, 1, 0.4] }}
                  transition={{ duration: 1.4, repeat: Infinity }}
                >
                  running
                </motion.span>
              )}
            </motion.div>
          );
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-white/5">
        <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">Current</p>
        <AnimatePresence mode="wait">
          <motion.p
            key={currentLabel}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.25 }}
            className="text-sm text-slate-300 min-h-[2.5rem] leading-snug"
          >
            {currentLabel}
          </motion.p>
        </AnimatePresence>
      </div>

      {!investigating && !allDone && (
        <div className="absolute top-4 right-4">
          <CircleDashed size={14} className="text-slate-700" />
        </div>
      )}
    </div>
  );
}
