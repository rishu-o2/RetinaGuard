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

function urgencyChipClass(urgency: string): string {
  switch (urgency) {
    case "ROUTINE":
      return "bg-emerald-100 text-emerald-800";
    case "SOON":
    case "PRIORITY":
      return "bg-amber-100 text-amber-800";
    case "URGENT":
      return "bg-orange-100 text-orange-800";
    case "EMERGENCY":
      return "bg-red-100 text-red-800";
    default:
      return "bg-slate-100 text-slate-700";
  }
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold" style={{ color: "#0B3C49" }}>
            Clinician Review Queue
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Cases pending ophthalmologist sign-off (ADR-003: clinician review mandatory)
          </p>
        </div>
        <button
          onClick={fetchQueue}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg border transition"
          style={{ backgroundColor: "#1B7F86", color: "#fff", borderColor: "#1B7F86" }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#145f65";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#1B7F86";
          }}
        >
          Refresh Queue
        </button>
      </div>

      {loading && (
        <div className="text-center py-10 text-sm text-slate-500">Loading pending cases...</div>
      )}

      {error && (
        <div className="p-3 bg-red-50 text-red-700 text-sm rounded-2xl border border-red-200">
          {error}
        </div>
      )}

      {!loading && !error && screenings.length === 0 && (
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-8 text-center text-slate-500">
          No cases currently pending review.
        </div>
      )}

      <div className="space-y-3">
        {screenings.map((sc) => (
          <div
            key={sc.id}
            onClick={() => onSelectCase(sc.id)}
            className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-4 cursor-pointer hover:border-[#1B7F86] transition flex items-center justify-between"
          >
            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center gap-2">
                {/* Pseudonym */}
                <span className="font-mono text-xs text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                  {sc.pseudonym_id.slice(0, 12)}…
                </span>

                {/* Uncertain / flag */}
                {sc.is_uncertain ? (
                  <span className="text-xs px-2 py-0.5 rounded font-bold bg-orange-100 text-[#D9622B] border border-orange-200">
                    📷 {sc.flag || "needs recapture or review"}
                  </span>
                ) : (
                  <span className="bg-slate-100 text-slate-700 text-xs px-2 py-0.5 rounded font-medium">
                    {sc.grade_label || `Grade ${sc.grade}`}
                  </span>
                )}

                {/* Urgency chip */}
                {sc.urgency && (
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-semibold ${urgencyChipClass(sc.urgency)}`}
                  >
                    {sc.urgency}
                  </span>
                )}
              </div>

              <p className="text-xs text-slate-400">
                Created: {new Date(sc.created_at).toLocaleString()}
              </p>
            </div>

            {/* CTA */}
            <div
              className="text-sm font-semibold ml-4 whitespace-nowrap"
              style={{ color: "#1B7F86" }}
            >
              Review Case →
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
