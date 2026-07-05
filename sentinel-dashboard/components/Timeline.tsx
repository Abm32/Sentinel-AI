"use client";

import { motion } from "framer-motion";
import { Clock } from "lucide-react";
import { TimelineEvent } from "@/lib/types";

interface TimelineProps {
  events: TimelineEvent[];
}

// packages/agents/tool_agent.py's `build_timeline` task is currently an
// explicit not_implemented stub on the backend (see that module's
// docstring) — no real timeline data is ever populated in
// InvestigationState.timeline yet. This narrative fallback mirrors the
// project's own demo case (fluorouracil day 1 -> neutropenia day 5) so
// the panel isn't empty during a demo, and is clearly illustrative
// rather than claiming to be live backend output.
const DEMO_TIMELINE: TimelineEvent[] = [
  { day: "Day 1", label: "Fluorouracil therapy started" },
  { day: "Day 3", label: "Mucositis, diarrhea onset" },
  { day: "Day 5", label: "Neutropenia (ANC 0.4 x10^9/L), fever" },
  { day: "Day 5", label: "Admitted for suspected fluoropyrimidine toxicity" },
];

export function Timeline({ events }: TimelineProps) {
  const items = events.length > 0 ? events : DEMO_TIMELINE;

  return (
    <div className="glass-panel rounded-2xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-[11px] font-semibold tracking-[0.15em] text-slate-400 font-mono flex items-center gap-1.5">
          <Clock size={12} /> TIMELINE
        </h2>
        {events.length === 0 && (
          <span className="text-[10px] text-slate-600">illustrative — not yet backend-derived</span>
        )}
      </div>
      <div className="relative flex items-stretch gap-0 overflow-x-auto pb-1 thin-scroll">
        {items.map((ev, i) => (
          <div key={i} className="flex items-stretch">
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.12, duration: 0.4, ease: "easeOut" }}
              className="flex flex-col items-center px-4 min-w-[10rem]"
            >
              <span className="text-[10px] font-mono text-slate-500">{ev.day}</span>
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: i * 0.12 + 0.15, type: "spring", stiffness: 400, damping: 15 }}
                className="w-3 h-3 rounded-full my-2 bg-gradient-to-br from-sky-400 to-violet-400 shadow-[0_0_12px_rgba(56,189,248,0.5)]"
              />
              <span className="text-xs text-slate-300 text-center leading-snug">{ev.label}</span>
            </motion.div>
            {i < items.length - 1 && (
              <div className="relative w-10 self-start mt-[1.7rem] overflow-hidden">
                <div className="absolute inset-0 border-t border-white/10" />
                <motion.div
                  className="absolute inset-y-0 left-0 border-t border-sky-400/60"
                  style={{ width: "100%" }}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: i * 0.12 + 0.3, duration: 0.35, ease: "easeOut" }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
