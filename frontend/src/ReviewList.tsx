import { useEffect, useState } from "react";

export interface ScreeningItem {
  id: string;
  pseudonym_id: string;
  status: string;
  created_at: string;
  grade: number | null;
  grade_label: string | null;
  urgency: string | null;
  referral: string | null;
  is_uncertain: boolean;
  flag?: string | null;
}

interface ReviewListProps {
  token: string;
  onSelectCase: (screeningId: string) => void;
}

export function ReviewList({ token, onSelectCase }: ReviewListProps) {
  const [screenings, setScreenings] = useState<ScreeningItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function fetchQueue() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/screenings?status=pending_review", {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) {
        throw new Error(`Failed to load queue (status ${res.status})`);
      }
      const data = await res.json();
      setScreenings(data);
    } catch (err: any) {
      setError(err.message || "Failed to load review queue");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchQueue();
  }, [token]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Clinician Review Queue</h2>
          <p className="text-xs text-slate-500">
            Cases pending ophthalmologist sign-off (ADR-003: clinician review mandatory)
          </p>
        </div>
        <button
          onClick={fetchQueue}
          className="text-xs bg-slate-200 hover:bg-slate-300 text-slate-700 px-3 py-1.5 rounded font-medium"
        >
          Refresh Queue
        </button>
      </div>

      {loading && (
        <div className="text-center py-10 text-sm text-slate-500">Loading pending cases...</div>
      )}

      {error && (
        <div className="p-3 bg-red-50 text-red-700 text-sm rounded border border-red-200">
          {error}
        </div>
      )}

      {!loading && !error && screenings.length === 0 && (
        <div className="bg-white rounded-lg border border-slate-200 p-8 text-center text-slate-500">
          No cases currently pending review.
        </div>
      )}

      <div className="space-y-3">
        {screenings.map((sc) => (
          <div
            key={sc.id}
            onClick={() => onSelectCase(sc.id)}
            className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm hover:border-indigo-400 cursor-pointer transition flex items-center justify-between"
          >
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                  {sc.pseudonym_id.slice(0, 12)}…
                </span>
                {sc.is_uncertain ? (
                  <span className="bg-orange-100 text-orange-800 text-xs px-2 py-0.5 rounded font-bold">
                    {sc.flag || "needs recapture or review"}
                  </span>
                ) : (
                  <span className="bg-indigo-100 text-indigo-800 text-xs px-2 py-0.5 rounded font-medium">
                    {sc.grade_label || `Grade ${sc.grade}`}
                  </span>
                )}
                {sc.urgency && (
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-semibold ${
                      sc.urgency === "URGENT" || sc.urgency === "EMERGENCY"
                        ? "bg-red-100 text-red-800"
                        : sc.urgency === "SOON" || sc.urgency === "PRIORITY"
                        ? "bg-amber-100 text-amber-800"
                        : "bg-emerald-100 text-emerald-800"
                    }`}
                  >
                    {sc.urgency}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400">
                Created: {new Date(sc.created_at).toLocaleString()}
              </p>
            </div>
            <div className="text-sm font-semibold text-indigo-600">Review Case →</div>
          </div>
        ))}
      </div>
    </div>
  );
}

