import { useEffect, useState } from "react";
import { TracePanel } from "./TracePanel";

interface ReviewDetailProps {
  screeningId: string;
  token: string;
  onBack: () => void;
  onReviewSubmitted: () => void;
}

export function ReviewDetail({
  screeningId,
  token,
  onBack,
  onReviewSubmitted,
}: ReviewDetailProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [showOverride, setShowOverride] = useState(false);
  const [overrideGrade, setOverrideGrade] = useState<number>(2);
  const [overrideNote, setOverrideNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDetail() {
      try {
        const res = await fetch(`/screenings/${screeningId}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error("Could not fetch case details");
        const json = await res.json();
        setData(json);
      } catch (err: any) {
        setError(err.message || "Failed to load case");
      } finally {
        setLoading(false);
      }
    }
    fetchDetail();
  }, [screeningId, token]);

  async function handleConfirm() {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`/screenings/${screeningId}/review`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ decision: "confirmed" }),
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || "Confirmation failed");
      }
      onReviewSubmitted();
    } catch (err: any) {
      setError(err.message || "Review submission failed");
      setSubmitting(false);
    }
  }

  async function handleOverride() {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`/screenings/${screeningId}/review`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          decision: "overridden",
          final_grade: overrideGrade,
          note: overrideNote || undefined,
        }),
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || "Override submission failed");
      }
      onReviewSubmitted();
    } catch (err: any) {
      setError(err.message || "Review submission failed");
      setSubmitting(false);
    }
  }

  function downloadPdf(lang: "en" | "hi" = "en") {
    const url = `/screenings/${screeningId}/report.pdf?lang=${lang}`;
    window.open(url, "_blank");
  }

  if (loading) {
    return <div className="text-center py-12 text-slate-500">Loading case details...</div>;
  }

  if (error && !data) {
    return (
      <div className="p-4 bg-red-50 text-red-700 rounded-2xl border border-red-200">
        {error}
        <button onClick={onBack} className="block mt-2 text-sm underline">
          ← Back to Queue
        </button>
      </div>
    );
  }

  const result = data?.result || {};

  // Formatted human-readable metadata (no raw leaked code names)
  const eyeSideRaw = String(result?.eye_side || data?.eye_side || "").toUpperCase();
  const eyeSideDisplay =
    eyeSideRaw === "OD"
      ? "Right eye (OD)"
      : eyeSideRaw === "OS"
      ? "Left eye (OS)"
      : result?.eye_side || data?.eye_side || "Right eye (OD)";

  const cameraRaw = String(result?.camera_type || data?.camera_type || "").toUpperCase();
  const cameraTypeDisplay =
    cameraRaw === "TABLETOP"
      ? "Tabletop"
      : cameraRaw === "SMARTPHONE_PORTABLE"
      ? "Handheld / Smartphone"
      : result?.camera_type || data?.camera_type || "Tabletop";

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-6">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b pb-4">
        <div>
          <button
            onClick={onBack}
            className="text-xs font-semibold mb-1 hover:underline"
            style={{ color: "#1B7F86" }}
          >
            ← Back to Queue
          </button>
          <h2 className="text-lg font-bold" style={{ color: "#0B3C49" }}>
            Case: {data.pseudonym_id ? data.pseudonym_id.slice(0, 16) : screeningId}…
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Eye: <span className="font-medium text-slate-700">{eyeSideDisplay}</span> · Modality:{" "}
            <span className="font-medium text-slate-700">{cameraTypeDisplay}</span> · Model:{" "}
            <span className="font-mono text-slate-700">{data.model_version || "v1.0.0"}</span> · Status:{" "}
            <span className="font-semibold text-slate-700 uppercase">{data.status}</span>
          </p>
        </div>

        <div className="flex space-x-2">
          <button
            onClick={() => downloadPdf("en")}
            className="border border-slate-300 text-slate-700 hover:bg-slate-50 px-3 py-1.5 rounded-lg text-xs font-medium transition"
          >
            Download PDF (EN)
          </button>
          <button
            onClick={() => downloadPdf("hi")}
            className="border border-slate-300 text-slate-700 hover:bg-slate-50 px-3 py-1.5 rounded-lg text-xs font-medium transition"
          >
            Download PDF (HI)
          </button>
        </div>
      </div>

      {/* Prototype Banner */}
      <div className="rounded-xl border border-amber-300 bg-amber-50 px-3 py-1.5 text-xs font-bold text-amber-900 tracking-wide text-center uppercase">
        ⚠ Prototype, not clinically validated · For clinician review only
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 text-sm rounded-xl border border-red-200">
          {error}
        </div>
      )}

      {/* Structured Card: Findings Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-2xl border border-slate-200/90 bg-[#F4F7F8]">
        <div>
          <div className="text-xs text-slate-500">Predicted Grade</div>
          <div className="text-base font-bold" style={{ color: "#0B3C49" }}>
            {result.grade !== null && result.grade !== undefined
              ? `Grade ${result.grade}`
              : "UNCERTAIN"}
          </div>
          <div className="text-xs text-slate-600">{result.grade_label_en || result.grade_label || ""}</div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Urgency</div>
          <div className="text-base font-bold" style={{ color: "#0B3C49" }}>
            {result.urgency || "N/A"}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Referral</div>
          <div className="text-base font-bold" style={{ color: "#0B3C49" }}>
            {result.referral || "N/A"}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Follow-up Interval</div>
          <div className="text-base font-bold" style={{ color: "#0B3C49" }}>
            {result.follow_up_interval_months
              ? `${result.follow_up_interval_months} mo.`
              : result.follow_up_months
              ? `${result.follow_up_months} mo.`
              : "Immediate"}
          </div>
        </div>
      </div>

      {/* Structured Plain-Language Explanation */}
      {result.guideline_explanation && (
        <div className="p-4 bg-teal-50/60 rounded-2xl border border-teal-200 text-xs text-slate-800 space-y-1">
          <div className="font-bold text-[#0B3C49]">Clinical Decision Rationale:</div>
          <p className="leading-relaxed">{result.guideline_explanation}</p>
          {result.guideline_citations && result.guideline_citations.length > 0 && (
            <div className="text-[11px] text-teal-800 pt-1">
              Guidelines: {result.guideline_citations.join(", ")}
            </div>
          )}
        </div>
      )}

      {/* Full Report Text (Structured under collapsed details) */}
      {result.report_text && (
        <details className="border border-slate-200 rounded-2xl bg-[#F4F7F8] p-3 text-xs">
          <summary className="font-semibold text-slate-700 cursor-pointer select-none">
            Full report text
          </summary>
          <div className="mt-2 font-mono whitespace-pre-wrap text-slate-700 max-h-48 overflow-y-auto p-2 bg-white rounded-xl border border-slate-200">
            {result.report_text}
          </div>
        </details>
      )}

      {/* Pipeline Execution Trace */}
      <TracePanel trace={result?.trace} />

      {/* Two Big Decision Buttons */}
      <div className="border-t pt-4 space-y-4">
        <h3 className="text-sm font-bold" style={{ color: "#0B3C49" }}>
          Clinician Action
        </h3>

        {!showOverride ? (
          <div className="grid grid-cols-2 gap-4">
            {/* Confirm button — teal */}
            <button
              onClick={handleConfirm}
              disabled={submitting}
              className="py-3 px-4 font-bold rounded-2xl shadow-sm transition disabled:opacity-50 text-center text-white"
              style={{ backgroundColor: "#1B7F86" }}
              onMouseEnter={(e) => {
                if (!submitting) (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#145f65";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#1B7F86";
              }}
            >
              {submitting ? "Submitting..." : "✓ Confirm AI Assessment"}
            </button>

            {/* Override button — outlined coral/orange */}
            <button
              onClick={() => setShowOverride(true)}
              disabled={submitting}
              className="py-3 px-4 font-bold rounded-2xl transition disabled:opacity-50 text-center border-2"
              style={{ color: "#D9622B", borderColor: "#D9622B", backgroundColor: "transparent" }}
              onMouseEnter={(e) => {
                if (!submitting) {
                  (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#fff4f0";
                }
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.backgroundColor = "transparent";
              }}
            >
              ✗ Override Assessment
            </button>
          </div>
        ) : (
          <div className="bg-orange-50 p-4 rounded-2xl border border-orange-200 space-y-3">
            <h4 className="text-xs font-bold text-orange-900 uppercase">Override Details</h4>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Final Clinician Grade (0–4)
              </label>
              <select
                value={overrideGrade}
                onChange={(e) => setOverrideGrade(Number(e.target.value))}
                className="block w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm"
              >
                <option value={0}>0 — No Apparent DR</option>
                <option value={1}>1 — Mild NPDR</option>
                <option value={2}>2 — Moderate NPDR</option>
                <option value={3}>3 — Severe NPDR</option>
                <option value={4}>4 — Proliferative DR (PDR)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Clinical Note (Reason for override)
              </label>
              <textarea
                value={overrideNote}
                onChange={(e) => setOverrideNote(e.target.value)}
                placeholder="Observed neovascularization not captured by model..."
                className="block w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm"
                rows={2}
              />
            </div>

            <div className="flex space-x-3 pt-2">
              <button
                onClick={handleOverride}
                disabled={submitting}
                className="py-2 px-4 font-bold rounded-xl shadow text-xs disabled:opacity-50 text-white"
                style={{ backgroundColor: "#D9622B" }}
                onMouseEnter={(e) => {
                  if (!submitting) (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#b84e22";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#D9622B";
                }}
              >
                {submitting ? "Submitting..." : "Submit Override"}
              </button>
              <button
                type="button"
                onClick={() => setShowOverride(false)}
                className="py-2 px-3 border border-slate-300 rounded-lg text-xs text-slate-700 hover:bg-slate-100 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
