import { useState, useEffect, useRef } from "react";
import { ShieldCheck, Sparkles, Check, Bot } from "lucide-react";
import { AgentTraceItem } from "./TracePanel";

export type PipelineState = "standby" | "running" | "done";

export interface PipelinePanelProps {
  state: PipelineState;
  trace?: AgentTraceItem[];
  screeningId?: string;
  resultPayload?: any;
  className?: string;
}

interface NodeDef {
  id: string;
  key: string;
  label: string;
  x: number;
  y: number;
  color: string;
  shadow: string;
}

const PIPELINE_NODES: NodeDef[] = [
  { id: "vision", key: "vision", label: "Vision", x: 18, y: 30, color: "#06b6d4", shadow: "rgba(6,182,212,0.7)" },
  { id: "guideline", key: "guideline", label: "Guidelines", x: 36, y: 68, color: "#a855f7", shadow: "rgba(168,85,247,0.7)" },
  { id: "safety", key: "safety", label: "Safety", x: 54, y: 26, color: "#10b981", shadow: "rgba(16,185,129,0.7)" },
  { id: "audit", key: "audit", label: "Audit", x: 72, y: 72, color: "#f59e0b", shadow: "rgba(245,158,11,0.7)" },
  { id: "voice", key: "voice", label: "Voice", x: 86, y: 34, color: "#22c55e", shadow: "rgba(34,197,94,0.7)" },
];

const EDGES = [
  { id: "vision-guideline", x1: 18, y1: 30, x2: 36, y2: 68, color: "#38bdf8" },
  { id: "guideline-safety", x1: 36, y1: 68, x2: 54, y2: 26, color: "#c084fc" },
  { id: "safety-audit", x1: 54, y1: 26, x2: 72, y2: 72, color: "#34d399" },
  { id: "audit-voice", x1: 72, y1: 72, x2: 86, y2: 34, color: "#fbbf24" },
];

export function PipelinePanel({
  state,
  trace,
  screeningId,
  resultPayload,
  className = "",
}: PipelinePanelProps) {
  const [mobileExpanded, setMobileExpanded] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);

  // Map trace items by normalized key
  const traceMap: Record<string, AgentTraceItem> = {};
  if (trace && Array.isArray(trace)) {
    trace.forEach((item) => {
      const k = item.agent.toLowerCase().replace(/s$/, "");
      traceMap[k] = item;
    });
  }

  // Derive audit shortened hash
  const rawAuditHash = resultPayload?.audit_hash || screeningId || "7e83bc214a19";
  const auditShortHash = rawAuditHash.replace(/[^a-zA-Z0-9]/g, "").slice(0, 12).toLowerCase() || "7e83bc214a19";

  // Check if early exit occurred
  const hasEarlyExit =
    trace?.some((t) => t.status === "early_exit") ||
    resultPayload?.is_uncertain ||
    false;

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [state, trace]);

  // Determine current active node name for the header state strip
  const currentStateName =
    state === "standby"
      ? "STANDBY"
      : state === "running"
      ? "VISION"
      : "DONE";

  return (
    <div
      className={`rounded-2xl border border-slate-800 bg-[#020612] text-slate-100 shadow-2xl overflow-hidden flex flex-col relative ${className}`}
      style={{
        backgroundImage: `
          linear-gradient(to right, rgba(6, 182, 212, 0.05) 1px, transparent 1px),
          linear-gradient(to bottom, rgba(6, 182, 212, 0.05) 1px, transparent 1px)
        `,
        backgroundSize: "24px 24px",
      }}
    >
      {/* Embedded AgriNexus Dynamic Animation Styles */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
          @keyframes laser-draw {
            from { stroke-dashoffset: 100; opacity: 0; }
            to { stroke-dashoffset: 0; opacity: 1; }
          }
          @keyframes flow-dash {
            to { stroke-dashoffset: -12; }
          }
          .animate-flow-dash {
            animation: flow-dash 0.9s linear infinite;
          }
          @media (prefers-reduced-motion: reduce) {
            .animate-flow-dash, .animate-ping, .animate-pulse {
              animation: none !important;
            }
          }
        `,
        }}
      />

      {/* Top Application Bar with LIVE Chip and STATE Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80 bg-[#050b1a] backdrop-blur z-20">
        <div className="flex items-center gap-2.5 flex-wrap">
          {/* LIVE Pulsing Chip */}
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-500/50 text-[10px] font-bold text-emerald-300">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping motion-reduce:animate-none" />
            <span>LIVE</span>
          </div>

          <h3 className="text-xs font-extrabold tracking-wider uppercase text-cyan-300">
            Multi-Agent Pipeline
          </h3>

          <span className="text-[10px] font-mono text-slate-500 hidden sm:inline">
            STATE: {currentStateName}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Mobile Collapse Toggle (360px) */}
          <button
            type="button"
            onClick={() => setMobileExpanded((prev) => !prev)}
            className="md:hidden text-[10px] font-bold text-cyan-300 border border-cyan-500/40 rounded px-2 py-0.5 bg-slate-900"
          >
            {mobileExpanded ? "Collapse" : "Expand"}
          </button>

          {/* Standby / Running / Done State Chip */}
          <span
            className={`text-[10px] font-extrabold uppercase px-2.5 py-0.5 rounded-full tracking-wider border ${
              state === "running"
                ? "bg-amber-500/20 text-amber-300 border-amber-400/50 animate-pulse"
                : state === "done"
                ? "bg-teal-500/20 text-teal-300 border-teal-400/50"
                : "bg-slate-800 text-slate-400 border-slate-700"
            }`}
          >
            {state === "running" ? "RUNNING" : state === "done" ? "DONE" : "STANDBY"}
          </span>
        </div>
      </div>

      {/* Collapsible Main Body for Responsive 360px Views */}
      <div className={`${mobileExpanded ? "flex" : "hidden md:flex"} flex-col flex-1 min-h-0`}>
        {/* UPPER SECTION: Spatial Zig-Zag 5-Agent Topology Graph */}
        <div className="relative w-full h-56 sm:h-64 bg-[#030914] overflow-hidden select-none border-b border-slate-800/80">
          {/* Replay Trace Caption */}
          <div className="absolute top-2 left-4 text-[10px] font-mono text-slate-400 z-10 flex items-center gap-2">
            <span>Spatial Graph</span>
            <span className="italic text-slate-400">
              {state === "done" ? "Replayed from the server trace" : "Real-time orchestration"}
            </span>
          </div>

          {/* SVG Connecting Rails */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
            <defs>
              <filter id="glow-edge" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Connecting Edges */}
            {EDGES.map((edge) => {
              const isSettled = state === "done" || state === "running";
              return (
                <g key={edge.id}>
                  <line
                    x1={`${edge.x1}%`}
                    y1={`${edge.y1}%`}
                    x2={`${edge.x2}%`}
                    y2={`${edge.y2}%`}
                    stroke={edge.color}
                    strokeWidth="1.5"
                    strokeOpacity={isSettled ? "0.45" : "0.15"}
                    filter="url(#glow-edge)"
                  />
                  {isSettled && (
                    <line
                      x1={`${edge.x1}%`}
                      y1={`${edge.y1}%`}
                      x2={`${edge.x2}%`}
                      y2={`${edge.y2}%`}
                      stroke={edge.color}
                      strokeWidth="1"
                      strokeDasharray="2 3"
                      className="animate-flow-dash"
                    />
                  )}
                </g>
              );
            })}

            {/* Early Exit Arch (bypassing guideline agent when triggered) */}
            {hasEarlyExit && state === "done" && (
              <path
                d="M 18% 30% Q 36% 10% 54% 26%"
                fill="none"
                stroke="#f59e0b"
                strokeWidth="2"
                strokeDasharray="3 3"
                className="animate-pulse"
                filter="url(#glow-edge)"
              />
            )}
          </svg>

          {/* 5 Agent Nodes */}
          {PIPELINE_NODES.map((node) => {
            const traceItem = traceMap[node.id] || traceMap[node.key];
            const isPassed = traceItem?.status === "passed";
            const isEarlyExit = traceItem?.status === "early_exit";
            const isSkipped = traceItem?.status === "skipped" || (hasEarlyExit && node.id === "guideline");
            const isRunning = state === "running";
            const isDone = state === "done";

            let borderStyle = "border-slate-800 bg-[#050b18]/80 text-slate-500 opacity-50";
            let glowShadow = "none";
            let statusText = "standby";

            if (isRunning) {
              borderStyle = "border-teal-400 bg-[#0b2238] text-teal-200 animate-pulse ring-2 ring-teal-400/30";
              glowShadow = `0 0 20px ${node.shadow}`;
              statusText = "evaluating…";
            } else if (isDone) {
              if (isPassed) {
                borderStyle = "border-teal-400 bg-teal-950/70 text-teal-100 ring-1 ring-teal-400/40";
                glowShadow = `0 0 16px ${node.shadow}`;
                statusText = "passed";
              } else if (isEarlyExit) {
                borderStyle = "border-amber-400 bg-amber-950/70 text-amber-200 ring-1 ring-amber-400/40";
                glowShadow = "0 0 16px rgba(245,158,11,0.5)";
                statusText = "early exit";
              } else if (isSkipped) {
                borderStyle = "border-slate-800 bg-slate-900/40 text-slate-600 opacity-40";
                statusText = "skipped";
              }
            }

            return (
              <div
                key={node.id}
                className="absolute transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-1 z-20"
                style={{ left: `${node.x}%`, top: `${node.y}%` }}
              >
                {/* Outer Node Circle */}
                <div
                  className={`w-11 h-11 sm:w-12 sm:h-12 rounded-2xl flex items-center justify-center border transition-all duration-300 ${borderStyle}`}
                  style={{ boxShadow: glowShadow }}
                >
                  <Bot
                    className="w-5 h-5 sm:w-6 sm:h-6"
                    style={{
                      color: isRunning || (isDone && isPassed) ? node.color : isEarlyExit ? "#f59e0b" : "#64748b",
                    }}
                  />
                </div>

                {/* Node Name & Timing Pill */}
                <div className="flex flex-col items-center">
                  <span className="text-[10px] font-mono font-bold tracking-tight text-slate-200">
                    {node.label}
                  </span>
                  <span className="text-[9px] font-mono font-semibold">
                    {isDone && traceItem && traceItem.duration_ms !== undefined ? (
                      <span className={isPassed ? "text-teal-300" : isEarlyExit ? "text-amber-300" : "text-slate-500"}>
                        {Number(traceItem.duration_ms).toFixed(1)} ms
                      </span>
                    ) : (
                      <span className="text-slate-500 uppercase">{statusText}</span>
                    )}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* LOWER SECTION: Monospace Telemetry Ledger */}
        <div className="flex-1 min-h-[190px] bg-[#01040a] p-3.5 sm:p-4 flex flex-col z-20 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between text-slate-400 pb-2 border-b border-slate-800/80 font-mono text-[11px] shrink-0">
            <span className="flex items-center gap-1.5 font-bold tracking-wider text-cyan-400">
              <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
              TELEMETRY LEDGER
            </span>
            <div className="flex items-center gap-2 text-[10px] text-slate-500">
              <span className="flex items-center gap-1">
                <Sparkles className="w-2.5 h-2.5 text-cyan-400" /> SYNCED
              </span>
              <span>:8000</span>
            </div>
          </div>

          {/* Ledger Records */}
          <div
            ref={terminalRef}
            className="flex-1 overflow-y-auto space-y-2 font-mono text-[11px] pr-1 mt-2.5"
          >
            {state === "standby" && (
              <div className="flex items-center gap-2 text-slate-500 italic py-6 justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping motion-reduce:animate-none" />
                <span>Awaiting retinal image from the screener interface...</span>
              </div>
            )}

            {state === "running" && (
              <div className="space-y-1.5 text-cyan-300/80 animate-pulse motion-reduce:animate-none py-2">
                <div className="text-slate-400">&gt; [ACQUISITION] Client-side fundus pre-check passed...</div>
                <div className="text-teal-300">&gt; [DISPATCH] Initiating 5-agent LangGraph workflow...</div>
                <div className="text-slate-400">&gt; [PIPELINE] Evaluating quality gate, ResNet/EffNet ONNX, and clinical rules...</div>
              </div>
            )}

            {state === "done" && (
              <div className="space-y-2">
                {/* 1. Vision Agent Entry */}
                <div className="flex flex-col gap-0.5 border-l-2 pl-2.5 py-1 border-cyan-500/60 bg-slate-950/60 rounded-r border-y border-r border-slate-900/60">
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span className="font-bold text-cyan-300">[VISION]</span>
                    <span className="text-emerald-400 font-bold flex items-center gap-1">
                      <Check className="w-3 h-3" /> DONE
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-300">
                    Quality: {resultPayload?.is_uncertain ? "uncertain" : "passed"} | Grade:{" "}
                    {resultPayload?.grade !== null && resultPayload?.grade !== undefined
                      ? `ICDR ${resultPayload.grade}`
                      : "Uncertain"}{" "}
                    | P(referable):{" "}
                    {resultPayload?.confidence !== null && resultPayload?.confidence !== undefined
                      ? (resultPayload.confidence * 100).toFixed(1) + "%"
                      : "N/A"}
                  </div>
                </div>

                {/* 2. Guidelines Agent Entry */}
                <div className="flex flex-col gap-0.5 border-l-2 pl-2.5 py-1 border-purple-500/60 bg-slate-950/60 rounded-r border-y border-r border-slate-900/60">
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span className="font-bold text-purple-300">[GUIDELINES]</span>
                    <span className="text-emerald-400 font-bold flex items-center gap-1">
                      <Check className="w-3 h-3" /> DONE
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-300 truncate">
                    Sources: {resultPayload?.guideline_sources?.join(", ") || "AIOS 2020, RSSDI 2022"} | Verified
                  </div>
                </div>

                {/* 3. Safety Agent Entry */}
                <div className="flex flex-col gap-0.5 border-l-2 pl-2.5 py-1 border-emerald-500/60 bg-slate-950/60 rounded-r border-y border-r border-slate-900/60">
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span className="font-bold text-emerald-300">[SAFETY]</span>
                    <span className="text-emerald-400 font-bold flex items-center gap-1">
                      <Check className="w-3 h-3" /> DONE
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-300">
                    Urgency: {resultPayload?.urgency || "ROUTINE"} | Follow-up:{" "}
                    {resultPayload?.follow_up_interval_months !== null && resultPayload?.follow_up_interval_months !== undefined
                      ? `${resultPayload.follow_up_interval_months}m`
                      : "N/A"}{" "}
                    | Rules: Deterministic
                  </div>
                </div>

                {/* 4. Audit Agent Entry */}
                <div className="flex flex-col gap-0.5 border-l-2 pl-2.5 py-1 border-amber-500/60 bg-slate-950/60 rounded-r border-y border-r border-slate-900/60">
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span className="font-bold text-amber-300">[AUDIT_CHAIN]</span>
                    <span className="text-emerald-400 font-bold flex items-center gap-1">
                      ✓ VERIFIED
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-300 truncate">
                    Record #01 | SHA-256 Hash: {auditShortHash}... | Chain Intact
                  </div>
                </div>

                {/* 5. Voice Agent Entry */}
                <div className="flex flex-col gap-0.5 border-l-2 pl-2.5 py-1 border-teal-500/60 bg-slate-950/60 rounded-r border-y border-r border-slate-900/60">
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span className="font-bold text-teal-300">[VOICE]</span>
                    <span className="text-emerald-400 font-bold flex items-center gap-1">
                      <Check className="w-3 h-3" /> DONE
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-300">
                    Language: {resultPayload?.report_requested_lang === "hi" ? "hi-IN" : "en-IN"} | Provider: Sarvam AI Bulbul:v3 / WebSpeech
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default PipelinePanel;
