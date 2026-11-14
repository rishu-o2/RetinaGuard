import React, { useState, useRef, useCallback, useEffect } from "react";
import { LoginPage } from "./LoginPage";
import { ReviewList } from "./ReviewList";
import { ReviewDetail } from "./ReviewDetail";
import { TracePanel, AgentTraceItem } from "./TracePanel";
import { runPrecheck, PrecheckThresholds } from "./precheck";
import {
  enqueueOfflineScreening,
  getQueuedScreenings,
  syncQueueWithServer,
} from "./offline/queue";
import {
  runOfflineInference,
  computeBufferSha256,
  preprocessRgbaToNCHW,
} from "./offline/inference";
import { ModelMetadataConfig } from "./offline/rulesEngine";
import * as ort from "onnxruntime-web";

// ─── Types ────────────────────────────────────────────────────────────────────

type Language = "en" | "hi";
type CameraType = "TABLETOP" | "SMARTPHONE_PORTABLE";
type EyeSide = "OD" | "OS";

export interface Facility {
  name: string;
  type?: string;
  address?: string;
  phone?: string;
  source_url?: string;
}

export interface ScreeningResponse {
  screening_id: string;
  grade: number | null;
  grade_label: string;
  confidence: number | null;
  referral: string;
  urgency: string;
  follow_up_months: number | null;
  is_uncertain: boolean;
  clinician_review_mandatory: boolean;
  rejection_reason?: string | null;
  report_text?: string;
  voice_text?: string;
  guideline_explanation?: string | null;
  guideline_citations?: string[];
  guideline_source_note?: string | null;
  unreviewed_translation?: boolean;
  unreviewed_translation_note?: string | null;
  trace?: AgentTraceItem[];
  facilities?: Facility[];
  is_provisional?: boolean;
  selected_district?: string;
  selected_state?: string;
}

// ─── Grade colour helpers ─────────────────────────────────────────────────────

const GRADE_COLOURS: Record<number, string> = {
  0: "bg-emerald-100 text-emerald-800 border-emerald-300",
  1: "bg-emerald-100 text-emerald-800 border-emerald-300",
  2: "bg-amber-100  text-amber-800  border-amber-300",
  3: "bg-red-100    text-red-800    border-red-300",
  4: "bg-red-100    text-red-800    border-red-300",
};

const URGENCY_COLOURS: Record<string, string> = {
  ROUTINE:   "text-emerald-700",
  PRIORITY:  "text-amber-700",
  URGENT:    "text-red-700 font-bold",
  EMERGENCY: "text-red-900 font-extrabold",
};

// ─── Sub-components ───────────────────────────────────────────────────────────

export function UncertainCard({
  result,
  language,
}: {
  result: ScreeningResponse;
  language: Language;
}) {
  return (
    <div className="rounded-lg border-2 border-orange-400 bg-orange-50 p-5 space-y-3" id="result-printable">
      {/* Prototype Warning Banner */}
      <div className="rounded border-2 border-amber-500 bg-amber-50 px-3 py-1.5 text-xs font-bold text-amber-900 tracking-wide text-center uppercase print:border-amber-600 print:text-amber-950">
        {language === "hi"
          ? "⚠️ प्रोटोटाइप — चिकित्सकीय रूप से प्रमाणित नहीं (Prototype, not clinically validated)"
          : "⚠️ Prototype, not clinically validated"}
      </div>

      {/* Provisional Offline Result Warning */}
      {result.is_provisional && (
        <div className="rounded border-2 border-amber-500 bg-amber-100 px-3 py-1 text-xs font-bold text-amber-950 tracking-wide text-center uppercase">
          {language === "hi"
            ? "🔴 अनंतिम (Provisional), ऑनलाइन होने पर पुनः जांच की जाएगी"
            : "🔴 Provisional, will be re-checked when online"}
        </div>
      )}

      {/* Unreviewed Translation Warning */}
      {result.unreviewed_translation && (
        <div className="rounded border-2 border-amber-400 bg-amber-50 px-3 py-1 text-xs font-bold text-amber-800 tracking-wide text-center uppercase">
          {language === "hi"
            ? "⚠️ अप्रमाणित अनुवाद — नैदानिक समीक्षा लंबित (Unreviewed translation)"
            : "⚠️ Unreviewed translation — pending clinical validation"}
        </div>
      )}

      <div className="flex items-center space-x-3">
        <span className="text-3xl">⚠️</span>
        <div>
          <p className="font-bold text-orange-900 text-lg">
            {language === "hi"
              ? "परिणाम अनिश्चित — UNCERTAIN"
              : "Result: UNCERTAIN"}
          </p>
          <p className="text-sm text-orange-800 mt-0.5">
            {language === "hi"
              ? "यह छवि विश्लेषण के लिए उपयुक्त नहीं है।"
              : "This image could not be reliably analysed."}
          </p>
        </div>
      </div>

      {result.rejection_reason && (
        <p className="text-sm text-orange-700 italic">
          Reason: {result.rejection_reason}
        </p>
      )}

      <div className="rounded bg-orange-100 border border-orange-300 p-3 text-sm text-orange-900 font-medium">
        {language === "hi"
          ? "कृपया छवि पुनः लें (retake the photo if possible)। एक चिकित्सक इस मामले की समीक्षा करेंगे (a clinician will review this)। इस परिणाम पर स्वतः कोई निर्णय न लें।"
          : "Please recapture the fundus image — retake the photo if possible. A clinician will review this case. Do not act on this result."}
      </div>

      <p className="text-xs text-orange-600">
        ✓ Clinician review mandatory — event logged (no patient identifiers stored).
      </p>

      {/* Nearest Eye Care Facilities */}
      {result.facilities && result.facilities.length > 0 && (
        <div className="rounded border border-teal-300 bg-teal-50 p-3 text-xs space-y-2 text-slate-800">
          <p className="font-bold text-teal-900 flex items-center gap-1.5 text-sm">
            <span>🏥</span>
            <span>{language === "hi" ? "निकटतम नेत्र देखभाल केंद्र (Nearest eye care):" : "Nearest eye care:"}</span>
          </p>
          <div className="space-y-1.5">
            {result.facilities.map((fac, idx) => (
              <div key={idx} className="border-b border-teal-200/60 pb-1.5 last:border-0 last:pb-0">
                <span className="font-bold text-teal-950">{fac.name}</span>
                {fac.type && <span className="ml-1 text-teal-700">({fac.type})</span>}
                {fac.address && <p className="text-teal-900">{fac.address}</p>}
                {fac.phone && <p className="font-mono text-teal-900 font-medium">📞 {fac.phone}</p>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trace Panel */}
      <TracePanel trace={result.trace} />
    </div>
  );
}

export function ResultCard({
  result,
  language,
  onPlayAudio,
}: {
  result: ScreeningResponse;
  language: Language;
  onPlayAudio: () => void;
}) {
  if (result.is_uncertain) {
    return <UncertainCard result={result} language={language} />;
  }

  const gradeColour =
    result.grade !== null ? (GRADE_COLOURS[result.grade] ?? "bg-slate-100 text-slate-800 border-slate-300") : "";
  const urgencyColour = URGENCY_COLOURS[result.urgency] ?? "text-slate-700";

  return (
    <div className="space-y-4 print:space-y-3" id="result-printable">
      {/* Prototype Warning Banner */}
      <div className="rounded border-2 border-amber-500 bg-amber-50 px-3 py-1.5 text-xs font-bold text-amber-900 tracking-wide text-center uppercase print:border-amber-600 print:text-amber-950">
        {language === "hi"
          ? "⚠️ प्रोटोटाइप — चिकित्सकीय रूप से प्रमाणित नहीं (Prototype, not clinically validated)"
          : "⚠️ Prototype, not clinically validated"}
      </div>

      {/* Provisional Offline Result Warning */}
      {result.is_provisional && (
        <div className="rounded border-2 border-amber-500 bg-amber-100 px-3 py-1 text-xs font-bold text-amber-950 tracking-wide text-center uppercase">
          {language === "hi"
            ? "🔴 अनंतिम (Provisional), ऑनलाइन होने पर पुनः जांच की जाएगी"
            : "🔴 Provisional, will be re-checked when online"}
        </div>
      )}

      {/* Unreviewed Translation Warning */}
      {result.unreviewed_translation && (
        <div className="rounded border-2 border-amber-400 bg-amber-50 px-3 py-1 text-xs font-bold text-amber-800 tracking-wide text-center uppercase">
          {language === "hi"
            ? "⚠️ अप्रमाणित अनुवाद — नैदानिक समीक्षा लंबित (Unreviewed translation)"
            : "⚠️ Unreviewed translation — pending clinical validation"}
        </div>
      )}

      {/* Grade + Referral row */}
      <div className="flex flex-wrap items-center gap-3">
        {result.grade !== null && (
          <span
            className={`inline-block border rounded px-3 py-1 text-sm font-bold ${gradeColour}`}
          >
            {language === "hi" ? "ICDR ग्रेड" : "ICDR Grade"} {result.grade} —{" "}
            {result.grade_label}
          </span>
        )}

        <span
          className={`inline-block rounded px-3 py-1 text-sm font-semibold border ${
            result.referral === "REFERABLE"
              ? "bg-red-100 text-red-800 border-red-300"
              : "bg-emerald-100 text-emerald-800 border-emerald-300"
          }`}
        >
          {result.referral === "REFERABLE"
            ? language === "hi"
              ? "रेफरल आवश्यक ✗"
              : "Referable ✗"
            : language === "hi"
            ? "गैर-रेफरल ✓"
            : "Non-Referable ✓"}
        </span>
      </div>

      {/* Urgency + follow-up */}
      <div className="text-sm space-y-1">
        <p>
          <span className="font-semibold text-slate-600">
            {language === "hi" ? "तत्कालता:" : "Urgency:"}
          </span>{" "}
          <span className={urgencyColour}>{result.urgency}</span>
        </p>
        {result.follow_up_months !== null && (
          <p>
            <span className="font-semibold text-slate-600">
              {language === "hi" ? "अनुवर्ती अंतराल:" : "Follow-up:"}
            </span>{" "}
            {result.follow_up_months}{" "}
            {language === "hi" ? "महीने" : "months"}
          </p>
        )}
        {result.confidence !== null && (
          <p>
            <span className="font-semibold text-slate-600">
              {language === "hi" ? "आत्मविश्वास:" : "Confidence:"}
            </span>{" "}
            {(result.confidence * 100).toFixed(1)}%
          </p>
        )}
      </div>

      {/* Guideline explanation */}
      {result.guideline_explanation && (
        <div className="rounded bg-blue-50 border border-blue-200 p-3 text-sm text-blue-900 space-y-1">
          <p className="font-semibold text-blue-800">
            {language === "hi" ? "दिशा-निर्देश व्याख्या:" : "Guideline Explanation:"}
          </p>
          <p>{result.guideline_explanation}</p>
          {result.guideline_citations && result.guideline_citations.length > 0 ? (
            <ul className="list-disc pl-4 text-xs text-blue-700 space-y-0.5">
              {result.guideline_citations.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          ) : (
            result.guideline_source_note && (
              <p className="text-xs text-blue-600 italic">
                {result.guideline_source_note}
              </p>
            )
          )}
        </div>
      )}

      {/* Nearest Eye Care Facilities */}
      {result.facilities && result.facilities.length > 0 && (
        <div className="rounded border border-teal-300 bg-teal-50 p-3 text-xs space-y-2 text-slate-800">
          <p className="font-bold text-teal-900 flex items-center gap-1.5 text-sm">
            <span>🏥</span>
            <span>{language === "hi" ? "निकटतम नेत्र देखभाल केंद्र (Nearest eye care):" : "Nearest eye care:"}</span>
          </p>
          <div className="space-y-1.5">
            {result.facilities.map((fac, idx) => (
              <div key={idx} className="border-b border-teal-200/60 pb-1.5 last:border-0 last:pb-0">
                <span className="font-bold text-teal-950">{fac.name}</span>
                {fac.type && <span className="ml-1 text-teal-700">({fac.type})</span>}
                {fac.address && <p className="text-teal-900">{fac.address}</p>}
                {fac.phone && <p className="font-mono text-teal-900 font-medium">📞 {fac.phone}</p>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Buttons row */}
      <div className="flex flex-wrap gap-2 mt-2 print:hidden items-center">
        {/* Audio play button — Web Speech API */}
        {result.voice_text && (
          <button
            onClick={onPlayAudio}
            aria-label="Play audio summary"
            className="flex items-center space-x-2 px-4 py-2 rounded bg-teal-600 hover:bg-teal-700 text-white text-sm font-medium transition"
          >
            <span>🔊</span>
            <span>{language === "hi" ? "ऑडियो चलाएँ" : "Play Audio"}</span>
          </button>
        )}

        {/* Print / PDF export */}
        <button
          onClick={() => window.print()}
          aria-label="Print or save as PDF"
          className="flex items-center space-x-2 px-4 py-2 rounded bg-slate-700 hover:bg-slate-800 text-white text-sm font-medium transition"
        >
          <span>🖨️</span>
          <span>{language === "hi" ? "प्रिंट / PDF" : "Print / PDF"}</span>
        </button>

        {/* Server-side Referral Report PDF */}
        {result.screening_id && !result.is_provisional && (
          <button
            onClick={() => {
              const qs = new URLSearchParams();
              qs.set("lang", language);
              if (result.selected_district) qs.set("district", result.selected_district);
              if (result.selected_state) qs.set("state", result.selected_state);
              window.open(`/screenings/${result.screening_id}/report.pdf?${qs.toString()}`, "_blank");
            }}
            aria-label="Download PDF report"
            className="flex items-center space-x-2 px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition"
          >
            <span>📄</span>
            <span>{language === "hi" ? "PDF डाउनलोड करें" : "Download PDF"}</span>
          </button>
        )}

        {result.is_provisional && (
          <span className="text-xs text-amber-800 bg-amber-50 px-2.5 py-1.5 rounded border border-amber-300 font-medium">
            {language === "hi"
              ? "⚠️ अनंतिम रिपोर्ट: अंतिम PDF केवल ऑनलाइन सत्यापन के बाद"
              : "⚠️ Provisional report: Final PDF available after online sync"}
          </span>
        )}
      </div>

      <p className="text-xs text-slate-400">
        ✓ Clinician review mandatory. Screening ID: {result.screening_id}
      </p>

      {/* Trace Panel */}
      <TracePanel trace={result.trace} />
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export function App() {
  // Auth & role state
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("retinaguard_token"));
  const [role, setRole] = useState<string | null>(() => localStorage.getItem("retinaguard_role"));
  const [userEmail, setUserEmail] = useState<string | null>(() => localStorage.getItem("retinaguard_email"));
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"screen" | "queue">(() => {
    const r = localStorage.getItem("retinaguard_role");
    return r === "clinician" || r === "admin" ? "queue" : "screen";
  });

  // Allowed locales fetched from backend policy
  const [availableLocales, setAvailableLocales] = useState<Array<{ code: string; name: string; unreviewed?: boolean }>>([
    { code: "en", name: "English (India)" },
    { code: "hi", name: "Hindi (हिंदी)" },
  ]);

  React.useEffect(() => {
    fetch("/auth/config")
      .then((res) => res.json())
      .then((data) => {
        if (data && Array.isArray(data.allowed_locales)) {
          setAvailableLocales(data.allowed_locales);
          if (data.allowed_locales.length > 0 && !data.allowed_locales.some((l: any) => l.code === language)) {
            setLanguage(data.allowed_locales[0].code as Language);
          }
        }
      })
      .catch(() => {});
  }, []);

  // Form state
  const [cameraType, setCameraType] = useState<CameraType>("SMARTPHONE_PORTABLE");
  const [language, setLanguage] = useState<Language>("en");
  const [eyeSide, setEyeSide] = useState<EyeSide>("OD");
  const [patientId, setPatientId] = useState("PAT-DEL-0042");
  const [abhaId, setAbhaId] = useState("");
  const [age, setAge] = useState(40);
  const [diabetesType, setDiabetesType] = useState("TYPE_2");
  const [hba1c, setHba1c] = useState<string>("");
  const [diabetesDuration, setDiabetesDuration] = useState<string>("");
  const [isPregnant, setIsPregnant] = useState(false);
  const [dpdpConsent, setDpdpConsent] = useState(false);

  // Optional district & state selector for facility directory (no GPS, not stored)
  const [district, setDistrict] = useState("");
  const [stateName, setStateName] = useState("");

  // Precheck ("bouncer") state
  const [precheckWarning, setPrecheckWarning] = useState<string | null>(null);
  const [precheckBypassed, setPrecheckBypassed] = useState(false);

  // Offline mode & IndexedDB queue state
  const [isOnline, setIsOnline] = useState<boolean>(() =>
    typeof navigator !== "undefined" ? navigator.onLine : true
  );
  const [queuedCount, setQueuedCount] = useState<number>(0);
  const [syncing, setSyncing] = useState<boolean>(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);

  // File + preview state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // API state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScreeningResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Sync & Service Worker Effects ────────────────────────────────────────

  const triggerSync = useCallback(async () => {
    setSyncing(true);
    try {
      const summary = await syncQueueWithServer(token);
      const remaining = await getQueuedScreenings();
      setQueuedCount(remaining.length);
      if (summary.synced > 0) {
        setSyncMessage(`Successfully synced ${summary.synced} offline screenings.`);
        setTimeout(() => setSyncMessage(null), 5000);
        // Replace provisional on-screen result with authoritative server outcome
        if (result?.is_provisional && summary.items.length > 0) {
          const matched = summary.items.find((it) => it.server_result);
          if (matched && matched.server_result) {
            setResult(matched.server_result);
          }
        }
      }
    } catch {
      // Retry will occur automatically on next reconnect
    } finally {
      setSyncing(false);
    }
  }, [token, result]);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      triggerSync();
    };
    const handleOffline = () => setIsOnline(false);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    // Initial check of offline queue items
    getQueuedScreenings()
      .then((items) => setQueuedCount(items.length))
      .catch(() => {});

    // Register service worker if available in browser
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [triggerSync]);

  // ── File handling & Pre-check ("Bouncer") ──────────────────────────────────

  const handleFileSelect = useCallback(
    async (file: File) => {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      setPrecheckWarning(null);
      setPrecheckBypassed(false);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);

      // Run client-side canvas pre-check
      try {
        const camParam =
          cameraType === "SMARTPHONE_PORTABLE" ? "smartphone_portable" : "tabletop";
        const thResp = await fetch(`/config/precheck?camera_type=${camParam}`);
        if (thResp.ok) {
          const thresholds: PrecheckThresholds = await thResp.json();
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement("canvas");
            const targetW = Math.min(img.naturalWidth || 256, 256);
            const targetH = Math.min(img.naturalHeight || 256, 256);
            canvas.width = targetW;
            canvas.height = targetH;
            const ctx = canvas.getContext("2d");
            if (ctx) {
              ctx.drawImage(img, 0, 0, targetW, targetH);
              const imgData = ctx.getImageData(0, 0, targetW, targetH);
              const check = runPrecheck(imgData.data, targetW, targetH, thresholds);
              if (!check.passed) {
                setPrecheckWarning(
                  "This doesn't look like a clear retinal photo. Please retake it."
                );
              }
            }
          };
          img.src = url;
        }
      } catch {
        // Fallback: If pre-check fails to run (e.g. offline/mock environment), do not block
      }
    },
    [cameraType]
  );

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  // Helper to read file as base64 string
  const readFileAsBase64 = (f: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const s = reader.result as string;
        const b64 = s.includes(",") ? s.split(",")[1] : s;
        resolve(b64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(f);
    });

  // Helper for offline inference execution
  const executeOfflineScreening = async (file: File) => {
    const metaResp = await fetch("/model/metadata");
    if (!metaResp.ok) throw new Error("Offline model metadata unavailable");
    const metadataConfig: ModelMetadataConfig = await metaResp.json();

    const modelResp = await fetch("/model/active");
    if (!modelResp.ok) throw new Error("Offline ONNX model unavailable");
    const modelBuffer = await modelResp.arrayBuffer();

    // Verify SHA-256 integrity
    const computedSha = await computeBufferSha256(modelBuffer);
    if (computedSha.toLowerCase() !== metadataConfig.onnx_sha256.toLowerCase()) {
      throw new Error("Cached model SHA-256 integrity mismatch");
    }

    const session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
      executionProviders: ["wasm"],
    });

    const b64 = await readFileAsBase64(file);
    const img = new Image();
    img.src = previewUrl || `data:${file.type};base64,${b64}`;
    await new Promise((resolve) => {
      img.onload = resolve;
      if (img.complete) resolve(null);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D unavailable");
    ctx.drawImage(img, 0, 0, 256, 256);
    const imgData = ctx.getImageData(0, 0, 256, 256);

    const nchw = preprocessRgbaToNCHW(imgData.data, 256, 256, 256);
    const offlineOut = await runOfflineInference(
      session,
      nchw,
      metadataConfig,
      {
        is_pregnant: isPregnant,
        hba1c: hba1c ? parseFloat(hba1c) : null,
        diabetes_duration_years: diabetesDuration ? parseFloat(diabetesDuration) : null,
      },
      256
    );

    const provisionalResult: ScreeningResponse = {
      screening_id: `offline-${Date.now()}`,
      grade: offlineOut.predicted_grade,
      grade_label: `Grade ${offlineOut.predicted_grade}`,
      confidence: offlineOut.confidence,
      referral: offlineOut.decision.referral,
      urgency: offlineOut.decision.urgency,
      follow_up_months: offlineOut.decision.follow_up_interval_months,
      is_uncertain: offlineOut.decision.is_uncertain,
      clinician_review_mandatory: true,
      report_text:
        language === "hi"
          ? "अनंतिम परिणाम: ऑफ़लाइन निकाला गया। ऑनलाइन होने पर सर्वर द्वारा पुनः सत्यापन किया जाएगा।"
          : "Provisional screening result: evaluated offline. Will be re-checked when online.",
      voice_text:
        language === "hi"
          ? "अनंतिम परिणाम, ऑनलाइन होने पर पुनः जांच की जाएगी।"
          : "Provisional result, will be re-checked when online.",
      is_provisional: true,
      selected_district: district || undefined,
      selected_state: stateName || undefined,
      trace: [
        {
          agent: "vision",
          status: "passed",
          duration_ms: 20.0,
          note: "Offline local ONNX inference via onnxruntime-web",
        },
        {
          agent: "safety",
          status: "passed",
          duration_ms: 1.0,
          note: "Offline client-side ClinicalRulesEngine",
        },
        { agent: "guideline", status: "skipped", duration_ms: 0.0, note: "Skipped in offline mode" },
        {
          agent: "audit",
          status: "skipped",
          duration_ms: 0.0,
          note: "Queued for server-side audit logging upon sync",
        },
        { agent: "voice", status: "passed", duration_ms: 0.5, note: "Client voice summary generated" },
      ],
    };

    // Store only pseudonymous ID in IndexedDB (no patient names or direct PHI)
    await enqueueOfflineScreening({
      patient_pseudonym_id: patientId || "ANON",
      eye_side: eyeSide,
      camera_type: cameraType,
      age: Number(age) || 40,
      diabetes_type: diabetesType,
      diabetes_duration_years: diabetesDuration ? parseFloat(diabetesDuration) : null,
      hba1c: hba1c ? parseFloat(hba1c) : null,
      is_pregnant: isPregnant,
      language: language,
      image_b64: b64,
      offline_result: provisionalResult,
    });

    const queued = await getQueuedScreenings();
    setQueuedCount(queued.length);
    setResult(provisionalResult);
  };

  // ── Submit ────────────────────────────────────────────────────────────────

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError(language === "hi" ? "कृपया एक छवि चुनें।" : "Please select an image.");
      return;
    }
    if (!dpdpConsent) {
      setError(
        language === "hi"
          ? "डेटा प्रसंस्करण के लिए सहमति आवश्यक है।"
          : "DPDP Act 2023 consent is required before screening."
      );
      return;
    }
    if (precheckWarning && !precheckBypassed) {
      setError("Please address image quality warning or click 'Send anyway'.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    // If browser is offline, directly execute offline screening flow
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      try {
        await executeOfflineScreening(selectedFile);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Offline evaluation failed");
      } finally {
        setLoading(false);
      }
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("patient_pseudonym_id", patientId || "ANON");
    formData.append("eye_side", eyeSide);
    formData.append("camera_type", cameraType);
    formData.append("age", String(age));
    formData.append("diabetes_type", diabetesType);
    formData.append("language", language);
    if (hba1c) formData.append("hba1c", hba1c);
    if (diabetesDuration) formData.append("diabetes_duration_years", diabetesDuration);
    formData.append("is_pregnant", String(isPregnant));

    try {
      const resp = await fetch("/screen/", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(detail?.detail ?? `HTTP ${resp.status}`);
      }

      const data: ScreeningResponse = await resp.json();

      // Check referral directory if district provided and result is referable or UNCERTAIN
      const isRef =
        String(data.referral).toUpperCase() === "REFERABLE" ||
        String(data.referral).toUpperCase() === "REFER" ||
        String(data.referral).toUpperCase() === "UNCERTAIN" ||
        data.is_uncertain;

      if (district && isRef) {
        try {
          const qs = new URLSearchParams();
          qs.set("district", district);
          if (stateName) qs.set("state", stateName);
          const facResp = await fetch(`/referral?${qs.toString()}`);
          if (facResp.ok) {
            const facList = await facResp.json();
            if (Array.isArray(facList) && facList.length > 0) {
              data.facilities = facList;
            }
          }
        } catch {
          // Non-blocking referral directory lookup
        }
      }

      data.selected_district = district || undefined;
      data.selected_state = stateName || undefined;
      setResult(data);
    } catch (err: unknown) {
      // If network failure occurs while online, offer seamless offline fallback
      if (err instanceof TypeError && err.message.includes("fetch")) {
        try {
          await executeOfflineScreening(selectedFile);
          return;
        } catch (offlineErr: any) {
          setError(`Offline evaluation failed: ${offlineErr.message}`);
        }
      } else {
        setError(err instanceof Error ? err.message : "Unexpected error");
      }
    } finally {
      setLoading(false);
    }
  };

  // ── Audio — Real Voice (/voice) with Web Speech API Fallback ────────────

  const handlePlayAudio = useCallback(async () => {
    if (!result?.voice_text) return;
    try {
      const res = await fetch("/voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: result.voice_text, language }),
      });
      if (res.ok && res.headers.get("content-type")?.includes("audio/")) {
        const blob = await res.blob();
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        await audio.play();
        return;
      }
    } catch {
      // Automatic fallback to browser speech synthesis
    }

    if (!("speechSynthesis" in window)) {
      alert("Your browser does not support speech synthesis.");
      return;
    }
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(result.voice_text);
    utt.lang = language === "hi" ? "hi-IN" : "en-IN";
    window.speechSynthesis.speak(utt);
  }, [result, language]);

  // ─────────────────────────────────────────────────────────────────────────

  if (!role) {
    return (
      <LoginPage
        onLogin={(newToken, newRole, newEmail) => {
          setToken(newToken);
          setRole(newRole);
          setUserEmail(newEmail);
          setActiveTab(newRole === "clinician" || newRole === "admin" ? "queue" : "screen");
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Banner */}
      <header className="bg-slate-900 text-white py-3 px-6 shadow-sm border-b border-slate-800">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-xl font-bold tracking-tight text-teal-400">RetinaGuard</span>
            <span className="text-xs bg-teal-900 text-teal-300 font-semibold px-2 py-0.5 rounded border border-teal-700">
              India Edition (CDSCO Compliant)
            </span>
          </div>

          <div className="flex items-center space-x-4">
            {/* Role navigation tabs for Clinician / Admin */}
            {(role === "clinician" || role === "admin") && (
              <div className="flex bg-slate-800 rounded p-0.5 border border-slate-700">
                <button
                  type="button"
                  onClick={() => {
                    setActiveTab("queue");
                    setSelectedCaseId(null);
                  }}
                  className={`text-xs px-3 py-1 rounded font-medium transition ${
                    activeTab === "queue"
                      ? "bg-indigo-600 text-white"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  Review Queue
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab("screen")}
                  className={`text-xs px-3 py-1 rounded font-medium transition ${
                    activeTab === "screen"
                      ? "bg-indigo-600 text-white"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  New Screening
                </button>
              </div>
            )}

            {/* Online / Offline Status & Offline Sync Pill */}
            <div className="flex items-center space-x-2 text-xs">
              {isOnline ? (
                <span className="text-emerald-400 font-medium">● Online</span>
              ) : (
                <span className="text-amber-400 font-medium">● Offline</span>
              )}
              {queuedCount > 0 && (
                <button
                  type="button"
                  onClick={triggerSync}
                  disabled={syncing || !isOnline}
                  className="bg-amber-600 hover:bg-amber-700 disabled:opacity-50 text-white px-2 py-0.5 rounded text-[11px] font-bold"
                >
                  {syncing ? "Syncing…" : `Sync (${queuedCount})`}
                </button>
              )}
            </div>

            {/* Language switch */}
            <select
              aria-label="Select Language"
              value={language}
              onChange={(e) => setLanguage(e.target.value as Language)}
              className="bg-slate-800 text-slate-200 text-sm rounded px-3 py-1 border border-slate-700"
            >
              {availableLocales.length > 0 ? (
                availableLocales.map((loc) => (
                  <option key={loc.code} value={loc.code}>
                    {loc.name} {loc.unreviewed ? "⚠️ (Unreviewed translation)" : ""}
                  </option>
                ))
              ) : (
                <option value="en" disabled>No reviewed languages available</option>
              )}
            </select>

            {/* User badge and logout */}
            <div className="flex items-center space-x-2 text-xs">
              <span className="text-slate-300 font-mono hidden sm:inline">{userEmail}</span>
              <span className="bg-slate-800 text-teal-300 px-2 py-0.5 rounded border border-slate-700 uppercase font-semibold">
                {role}
              </span>
              <button
                type="button"
                onClick={() => {
                  localStorage.clear();
                  setToken(null);
                  setRole(null);
                  setUserEmail(null);
                }}
                className="text-slate-400 hover:text-white ml-1 underline cursor-pointer"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      </header>

      {syncMessage && (
        <div className="bg-emerald-600 text-white text-xs py-1.5 px-4 text-center font-medium shadow-inner">
          {syncMessage}
        </div>
      )}

      {/* Disclaimer */}
      <div className="bg-amber-50 border-b border-amber-200 px-6 py-2 text-xs text-amber-900">
        <div className="max-w-7xl mx-auto font-medium">
          ⚠️ <strong>Clinical Decision-Support Tool:</strong> RetinaGuard assists clinicians
          and does not provide autonomous medical diagnoses. All findings require licensed
          clinical sign-off.
        </div>
      </div>

      {activeTab === "queue" ? (
        <main className="max-w-5xl mx-auto flex-1 w-full p-6">
          {selectedCaseId ? (
            <ReviewDetail
              screeningId={selectedCaseId}
              token={token || ""}
              onBack={() => setSelectedCaseId(null)}
              onReviewSubmitted={() => setSelectedCaseId(null)}
            />
          ) : (
            <ReviewList
              token={token || ""}
              onSelectCase={(id) => setSelectedCaseId(id)}
            />
          )}
        </main>
      ) : (
        /* Main workspace */
        <main className="max-w-7xl mx-auto flex-1 w-full p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* ── Left: Intake form ── */}
        <form
          onSubmit={handleSubmit}
          className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 space-y-4"
        >
          <h2 className="text-base font-semibold text-slate-800">
            {language === "hi" ? "जांच इनटेक" : "Screening Intake"}
          </h2>

          {/* Camera modality */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              {language === "hi" ? "कैमरा प्रकार" : "Camera Hardware Modality"}
            </label>
            <select
              value={cameraType}
              onChange={(e) => setCameraType(e.target.value as CameraType)}
              className="w-full text-sm border-slate-300 rounded p-2 border focus:ring-teal-500"
            >
              <option value="SMARTPHONE_PORTABLE">Handheld / Smartphone (Remidio, Forus 3nethra)</option>
              <option value="TABLETOP">Tabletop Fundus Camera (Zeiss, Topcon, Kowa)</option>
            </select>
          </div>

          {/* Eye side */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              {language === "hi" ? "आँख की तरफ (लेटरैलिटी)" : "Eye Side Laterality"}
            </label>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {(["OD", "OS"] as EyeSide[]).map((side) => (
                <button
                  key={side}
                  type="button"
                  onClick={() => setEyeSide(side)}
                  className={`py-2 border rounded font-medium transition ${
                    eyeSide === side
                      ? "bg-teal-50 text-teal-800 border-teal-300"
                      : "text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  {side === "OD" ? "OD (Right Eye)" : "OS (Left Eye)"}
                </button>
              ))}
            </div>
          </div>

          {/* Patient pseudonym ID */}
          <div className="border-t pt-4">
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              {language === "hi" ? "रोगी छद्म ID" : "Patient Pseudonym ID"}
            </label>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="e.g. PAT-DEL-042"
              className="w-full text-sm border rounded p-2"
            />
          </div>

          {/* ABHA (optional) */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              ABHA ID ({language === "hi" ? "वैकल्पिक" : "Optional"})
            </label>
            <input
              type="text"
              value={abhaId}
              onChange={(e) => setAbhaId(e.target.value)}
              placeholder="91-XXXX-XXXX-XXXX"
              className="w-full text-sm border rounded p-2"
            />
          </div>

          {/* Age */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              {language === "hi" ? "आयु (वर्ष)" : "Age (years)"}
            </label>
            <input
              type="number"
              min={0}
              max={125}
              value={age}
              onChange={(e) => setAge(Number(e.target.value))}
              className="w-full text-sm border rounded p-2"
            />
          </div>

          {/* Diabetes type */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              {language === "hi" ? "मधुमेह का प्रकार" : "Diabetes Type"}
            </label>
            <select
              value={diabetesType}
              onChange={(e) => setDiabetesType(e.target.value)}
              className="w-full text-sm border-slate-300 rounded p-2 border"
            >
              <option value="TYPE_2">Type 2</option>
              <option value="TYPE_1">Type 1</option>
              <option value="GESTATIONAL">Gestational</option>
              <option value="UNKNOWN">Unknown</option>
            </select>
          </div>

          {/* HbA1c */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              HbA1c % ({language === "hi" ? "वैकल्पिक" : "Optional"})
            </label>
            <input
              type="number"
              step="0.1"
              min={3.5}
              max={25}
              value={hba1c}
              onChange={(e) => setHba1c(e.target.value)}
              placeholder="e.g. 8.2"
              className="w-full text-sm border rounded p-2"
            />
          </div>

          {/* Diabetes duration */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              {language === "hi" ? "मधुमेह की अवधि (वर्ष)" : "Diabetes Duration (yrs)"}
            </label>
            <input
              type="number"
              step="0.5"
              min={0}
              value={diabetesDuration}
              onChange={(e) => setDiabetesDuration(e.target.value)}
              placeholder="e.g. 5"
              className="w-full text-sm border rounded p-2"
            />
          </div>

          {/* Pregnancy */}
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="pregnant"
              checked={isPregnant}
              onChange={(e) => setIsPregnant(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="pregnant" className="text-sm text-slate-700">
              {language === "hi" ? "गर्भावस्था" : "Pregnancy"}
            </label>
          </div>

          {/* Optional Collapsed District Selector for Eye Care Directory */}
          <details className="rounded border border-slate-200 bg-slate-50 p-2 text-xs">
            <summary className="font-semibold text-slate-700 cursor-pointer select-none">
              {language === "hi" ? "वैकल्पिक: रेफरल जिला (District)" : "Optional: Referral District"}
            </summary>
            <div className="grid grid-cols-2 gap-2 mt-2">
              <div>
                <label className="block text-slate-500 mb-1">State</label>
                <input
                  type="text"
                  value={stateName}
                  onChange={(e) => setStateName(e.target.value)}
                  placeholder="e.g. Maharashtra"
                  className="w-full text-xs border rounded p-1.5"
                />
              </div>
              <div>
                <label className="block text-slate-500 mb-1">District</label>
                <input
                  type="text"
                  value={district}
                  onChange={(e) => setDistrict(e.target.value)}
                  placeholder="e.g. Pune"
                  className="w-full text-xs border rounded p-1.5"
                />
              </div>
            </div>
            <p className="text-[10px] text-slate-400 mt-1 italic">
              No GPS used. Location is not stored with patient records.
            </p>
          </details>

          {/* DPDP Act 2023 consent */}
          <div className="border-t pt-4">
            <div className="flex items-start space-x-2">
              <input
                type="checkbox"
                id="dpdp-consent"
                checked={dpdpConsent}
                onChange={(e) => setDpdpConsent(e.target.checked)}
                className="mt-0.5 rounded"
              />
              <label htmlFor="dpdp-consent" className="text-xs text-slate-700 leading-relaxed">
                <strong>DPDP Act 2023</strong> — I confirm explicit informed consent has been
                obtained from the patient for processing their retinal image data for diabetic
                retinopathy screening.
              </label>
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading || !selectedFile || !dpdpConsent}
            className="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium py-2 rounded text-sm transition"
          >
            {loading
              ? language === "hi"
                ? "विश्लेषण हो रहा है…"
                : "Analysing…"
              : language === "hi"
              ? "जांच करें"
              : "Run Screening"}
          </button>

          {error && (
            <div className="rounded bg-red-50 border border-red-300 p-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </form>

        {/* ── Right two columns: drop zone + results ── */}
        <div className="md:col-span-2 flex flex-col gap-6">
          {/* Browser Pre-check Warning ("Bouncer") */}
          {precheckWarning && !precheckBypassed && (
            <div className="rounded border-2 border-amber-400 bg-amber-50 p-3 text-xs text-amber-900 flex items-center justify-between gap-3 shadow-sm">
              <div className="flex items-center gap-2">
                <span className="text-xl">⚠️</span>
                <span className="font-medium">{precheckWarning}</span>
              </div>
              <button
                type="button"
                onClick={() => setPrecheckBypassed(true)}
                className="px-3 py-1.5 bg-amber-600 hover:bg-amber-700 text-white rounded font-bold text-xs whitespace-nowrap shadow"
              >
                Send anyway
              </button>
            </div>
          )}

          {/* Drop zone / image preview */}
          <div
            role="button"
            tabIndex={0}
            aria-label="Drop zone for fundus image"
            className={`bg-white rounded-lg shadow-sm border-2 border-dashed flex flex-col items-center justify-center min-h-[300px] transition cursor-pointer ${
              isDragging
                ? "border-teal-400 bg-teal-50"
                : "border-slate-300 hover:border-teal-400"
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => { if (e.key === "Enter") fileInputRef.current?.click(); }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png"
              capture="environment"
              className="hidden"
              onChange={onFileChange}
            />

            {previewUrl ? (
              <div className="w-full h-full flex flex-col items-center p-4">
                <img
                  src={previewUrl}
                  alt="Selected fundus"
                  className="max-h-60 rounded shadow object-contain"
                />
                <p className="text-xs text-slate-500 mt-2">{selectedFile?.name}</p>
                <p className="text-xs text-teal-600 mt-1">
                  {language === "hi"
                    ? "बदलने के लिए क्लिक करें"
                    : "Click or drop to change image"}
                </p>
              </div>
            ) : (
              <div className="text-center text-slate-400 p-6">
                <div className="text-5xl mb-3">👁️</div>
                <p className="text-sm font-medium text-slate-600">
                  {language === "hi"
                    ? "रेटिनल फंडस छवि चुनें या ड्रॉप करें"
                    : "Select or drop a retinal fundus image"}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  {language === "hi"
                    ? "समर्थित प्रारूप: JPEG, PNG"
                    : "Supported: JPEG, PNG (quality & OOD checks applied)"}
                </p>
              </div>
            )}
          </div>

          {/* Loading spinner */}
          {loading && (
            <div className="bg-white rounded-lg border border-slate-200 p-6 flex items-center space-x-4 shadow-sm">
              <div className="animate-spin h-6 w-6 border-4 border-teal-500 border-t-transparent rounded-full" />
              <p className="text-sm text-slate-600">
                {language === "hi"
                  ? "विश्लेषण चल रहा है — कृपया प्रतीक्षा करें…"
                  : "Running quality gate → OOD → inference → clinical rules…"}
              </p>
            </div>
          )}

          {/* Result card */}
          {result && !loading && (
            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
              <h3 className="text-base font-semibold text-slate-800 mb-4">
                {language === "hi" ? "स्क्रीनिंग परिणाम" : "Screening Result"}
              </h3>
              <ResultCard
                result={result}
                language={language}
                onPlayAudio={handlePlayAudio}
              />
            </div>
          )}
        </div>
      </main>
      )}

      {/* Footer */}
      <footer className="bg-white border-t py-3 px-6 text-xs text-slate-500 text-center">
        RetinaGuard &copy; 2026. Aligned with AIOS 2020, RSSDI 2022, ICMR, and ICO Diabetic
        Eye Care Guidelines. All data hosted locally in India.
      </footer>
    </div>
  );
}

export default App;
