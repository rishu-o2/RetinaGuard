export interface AgentTraceItem {
  agent: string;
  status: "passed" | "early_exit" | "skipped";
  duration_ms: number;
  note: string;
}

export function TracePanel({ trace }: { trace?: AgentTraceItem[] }) {
  if (!trace || !Array.isArray(trace) || trace.length === 0) {
    return null;
  }

  return (
    <details className="mt-3 border border-slate-200 rounded bg-slate-50 p-3 text-xs text-slate-700">
      <summary className="font-semibold text-slate-800 cursor-pointer select-none">
        How this result was made ({trace.length} pipeline steps)
      </summary>
      <div className="mt-3 space-y-2">
        {trace.map((step, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between border-b border-slate-200 pb-2 last:border-0 last:pb-0"
          >
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-slate-900 capitalize">{step.agent} Agent</span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold ${
                    step.status === "passed"
                      ? "bg-emerald-100 text-emerald-800"
                      : step.status === "early_exit"
                      ? "bg-amber-100 text-amber-800"
                      : "bg-slate-200 text-slate-600"
                  }`}
                >
                  {step.status}
                </span>
              </div>
              <p className="text-slate-500 text-[11px] mt-0.5">{step.note}</p>
            </div>
            <div className="text-right text-slate-600 font-mono text-xs whitespace-nowrap pl-4">
              {Number(step.duration_ms).toFixed(1)} ms
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
