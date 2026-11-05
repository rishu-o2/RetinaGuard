import React, { useEffect, useState } from "react";

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
      <div className="p-4 bg-red-50 text-red-700 rounded-lg border border-red-200">
        {error}
        <button onClick={onBack} className="block mt-2 text-sm underline">
          ← Back to Queue
        </button>
      </div>
    );
  }

  const result = data?.result || {};

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b pb-4">
        <div>
          <button
            onClick={onBack}
            className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold mb-1"
          >
            ← Back to Queue
          </button>
          <h2 className="text-lg font-bold text-slate-900">
            Case: {data.pseudonym_id ? data.pseudonym_id.slice(0, 16) : screeningId}…
          </h2>
          <p className="text-xs text-slate-400">
            Model: {data.model_version} · Status: {data.status}
          </p>
        </div>

        <div className="flex space-x-2">
          <button
            onClick={() => downloadPdf("en")}
            className="border border-slate-300 text-slate-700 hover:bg-slate-50 px-3 py-1.5 rounded text-xs font-medium"
          >
            Download PDF (EN)
          </button>
          <button
            onClick={() => downloadPdf("hi")}
            className="border border-slate-300 text-slate-700 hover:bg-slate-50 px-3 py-1.5 rounded text-xs font-medium"
          >
            Download PDF (HI)
          </button>
        </div>
      </div>

      {/* Prototype Banner */}
      <div className="rounded border-2 border-amber-500 bg-amber-50 px-3 py-1.5 text-xs font-bold text-amber-900 tracking-wide text-center uppercase">
        ⚠ Prototype, not clinically validated · For clinician review only
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 text-sm rounded border border-red-200">
          {error}
        </div>
      )}

      {/* Case Findings Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 bg-slate-50 rounded-lg border border-slate-100">
        <div>
          <div className="text-xs text-slate-500">Predicted Grade</div>
          <div className="text-base font-bold text-slate-900">
            {result.grade !== null && result.grade !== undefined
              ? `Grade ${result.grade}`
              : "UNCERTAIN"}
          </div>
          <div className="text-xs text-slate-600">{result.grade_label_en || ""}</div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Urgency</div>
          <div className="text-base font-bold text-slate-900">{result.urgency || "N/A"}</div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Referral</div>
          <div className="text-base font-bold text-slate-900">{result.referral || "N/A"}</div>
        </div>

        <div>
          <div className="text-xs text-slate-500">Follow-up Interval</div>
          <div className="text-base font-bold text-slate-900">
            {result.follow_up_interval_months
              ? `${result.follow_up_interval_months} mo.`
              : "Immediate"}
          </div>
        </div>
      </div>

      {/* Text report preview */}
      {result.report_text && (
        <div className="p-4 bg-slate-50 rounded-lg border border-slate-100 text-xs font-mono whitespace-pre-wrap text-slate-700 max-h-48 overflow-y-auto">
          {result.report_text}
        </div>
      )}

      {/* Two Big Decision Buttons */}
      <div className="border-t pt-4 space-y-4">
        <h3 className="text-sm font-bold text-slate-800">Clinician Action</h3>

        {!showOverride ? (
          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={handleConfirm}
              disabled={submitting}
              className="py-3 px-4 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-lg shadow transition disabled:opacity-50 text-center"
            >
              {submitting ? "Submitting..." : "✓ Confirm AI Assessment"}
            </button>

            <button
              onClick={() => setShowOverride(true)}
              disabled={submitting}
              className="py-3 px-4 bg-amber-600 hover:bg-amber-700 text-white font-bold rounded-lg shadow transition disabled:opacity-50 text-center"
            >
              ✗ Override Assessment
            </button>
          </div>
        ) : (
          <div className="bg-amber-50 p-4 rounded-lg border border-amber-200 space-y-3">
            <h4 className="text-xs font-bold text-amber-900 uppercase">Override Details</h4>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Final Clinician Grade (0–4)
              </label>
              <select
                value={overrideGrade}
                onChange={(e) => setOverrideGrade(Number(e.target.value))}
                className="block w-full rounded border border-slate-300 px-3 py-1.5 text-sm"
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
                className="block w-full rounded border border-slate-300 px-3 py-1.5 text-sm"
                rows={2}
              />
            </div>

            <div className="flex space-x-3 pt-2">
              <button
                onClick={handleOverride}
                disabled={submitting}
                className="py-2 px-4 bg-red-600 hover:bg-red-700 text-white font-bold rounded shadow text-xs disabled:opacity-50"
              >
                {submitting ? "Submitting..." : "Submit Override"}
              </button>
              <button
                type="button"
                onClick={() => setShowOverride(false)}
                className="py-2 px-3 border border-slate-300 rounded text-xs text-slate-700 hover:bg-slate-100"
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
