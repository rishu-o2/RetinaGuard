import { useState, useEffect } from "react";
import { X, FileText, Download, AlertCircle, Clock } from "lucide-react";

export interface StoredScreening {
  id: string;
  screening_id: string;
  created_at: string;
  pseudonym_id?: string;
  patient_pseudonym_id?: string;
  grade?: number | null;
  grade_label?: string | null;
  referral: string;
  urgency: string;
  status?: string;
  is_uncertain?: boolean;
}

interface MyScreeningsModalProps {
  isOpen: boolean;
  onClose: () => void;
  token?: string;
  localScreenings?: StoredScreening[];
}

export function MyScreeningsModal({
  isOpen,
  onClose,
  token,
  localScreenings = [],
}: MyScreeningsModalProps) {
  const [screenings, setScreenings] = useState<StoredScreening[]>([]);
  const [loading, setLoading] = useState(false);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setDownloadError(null);

    const loadScreenings = async () => {
      setLoading(true);
      let list: StoredScreening[] = [...localScreenings];

      if (token) {
        try {
          const res = await fetch("/screenings?limit=50", {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (res.ok) {
            const serverList = await res.json();
            // Merge avoiding duplicates by ID
            const existingIds = new Set(list.map((s) => s.screening_id || s.id));
            serverList.forEach((s: any) => {
              if (!existingIds.has(s.id)) {
                list.push({
                  id: s.id,
                  screening_id: s.id,
                  created_at: s.created_at,
                  pseudonym_id: s.pseudonym_id,
                  grade: s.grade,
                  grade_label: s.grade !== null ? `Grade ${s.grade}` : "Uncertain",
                  referral: s.referral,
                  urgency: s.urgency,
                  status: s.status,
                  is_uncertain: s.is_uncertain,
                });
              }
            });
          }
        } catch (e) {
          console.warn("Could not fetch remote screenings:", e);
        }
      }

      // Sort newest first
      list.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      setScreenings(list);
      setLoading(false);
    };

    loadScreenings();
  }, [isOpen, token, localScreenings]);

  if (!isOpen) return null;

  const handleDownloadPdf = async (screeningId: string, lang: "en" | "hi") => {
    setDownloadingId(`${screeningId}-${lang}`);
    setDownloadError(null);
    try {
      const headers: Record<string, string> = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const res = await fetch(`/screenings/${screeningId}/report.pdf?lang=${lang}`, {
        headers,
      });

      if (!res.ok) {
        throw new Error(`PDF download failed (HTTP ${res.status})`);
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `RetinaGuard_${screeningId.slice(0, 8)}_${lang}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();
    } catch (err: any) {
      setDownloadError(err.message || "Failed to download PDF report");
    } finally {
      setDownloadingId(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in">
      <div className="relative w-full max-w-2xl rounded-2xl bg-[#071120] border border-slate-800 text-white shadow-2xl flex flex-col max-h-[85vh] overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800 bg-[#050b1a]">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">My Screenings</h3>
              <p className="text-xs text-slate-400">Past retinal screening assessments</p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
            title="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Error Alert */}
        {downloadError && (
          <div className="px-5 py-2.5 bg-red-950/80 border-b border-red-500/40 text-red-200 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0 text-red-400" />
            <span>{downloadError}</span>
          </div>
        )}

        {/* List Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {loading ? (
            <div className="py-12 text-center text-slate-400 text-xs flex flex-col items-center gap-2">
              <div className="w-6 h-6 rounded-full border-2 border-teal-500 border-t-transparent animate-spin" />
              <span>Loading screenings...</span>
            </div>
          ) : screenings.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-xs italic">
              No screenings found. Complete a screening to see it recorded here.
            </div>
          ) : (
            screenings.map((s) => {
              const sid = s.screening_id || s.id;
              const dateStr = s.created_at ? new Date(s.created_at).toLocaleDateString() : "Recent";
              const isDownloadingEn = downloadingId === `${sid}-en`;
              const isDownloadingHi = downloadingId === `${sid}-hi`;

              return (
                <div
                  key={sid}
                  className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:border-slate-700 transition"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono text-xs font-bold text-teal-300">
                        {s.pseudonym_id ? `RG-${s.pseudonym_id.slice(0, 6)}` : `RG-${sid.slice(0, 6)}`}
                      </span>
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${
                          s.is_uncertain || s.referral === "UNCERTAIN"
                            ? "bg-slate-700/50 text-slate-300 border-slate-600"
                            : s.referral === "REFERABLE"
                            ? "bg-orange-950/60 text-orange-300 border-orange-500/40"
                            : "bg-emerald-950/60 text-emerald-300 border-emerald-500/40"
                        }`}
                      >
                        {s.is_uncertain || s.referral === "UNCERTAIN" ? "Uncertain" : s.referral || "Screened"}
                      </span>
                      {s.urgency && (
                        <span className="text-[10px] text-slate-400 font-semibold">
                          • {s.urgency}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-slate-400">
                      <span className="flex items-center gap-1 text-[11px]">
                        <Clock className="w-3 h-3 text-slate-500" />
                        {dateStr}
                      </span>
                      {s.grade !== null && s.grade !== undefined && (
                        <span className="text-[11px] text-slate-300">
                          Grade {s.grade}
                        </span>
                      )}
                      {s.status && (
                        <span className="text-[11px] text-slate-500 capitalize">
                          Status: {s.status.replace("_", " ")}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Download PDF Actions: EN and HI */}
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      disabled={isDownloadingEn}
                      onClick={() => handleDownloadPdf(sid, "en")}
                      className="px-2.5 py-1.5 rounded-lg bg-teal-500/20 border border-teal-500/40 hover:bg-teal-500/30 text-teal-300 text-xs font-bold transition flex items-center gap-1 cursor-pointer disabled:opacity-50"
                      title="Download English PDF Report"
                    >
                      <Download className="w-3 h-3" />
                      <span>{isDownloadingEn ? "..." : "PDF (EN)"}</span>
                    </button>

                    <button
                      type="button"
                      disabled={isDownloadingHi}
                      onClick={() => handleDownloadPdf(sid, "hi")}
                      className="px-2.5 py-1.5 rounded-lg bg-emerald-500/20 border border-emerald-500/40 hover:bg-emerald-500/30 text-emerald-300 text-xs font-bold transition flex items-center gap-1 cursor-pointer disabled:opacity-50"
                      title="Download Hindi PDF Report"
                    >
                      <Download className="w-3 h-3" />
                      <span>{isDownloadingHi ? "..." : "PDF (HI)"}</span>
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

export default MyScreeningsModal;
