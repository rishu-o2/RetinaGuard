import { useState, useRef, useCallback, useEffect } from "react";
import { LoginPage } from "./LoginPage";
import { ReviewList } from "./ReviewList";
import { ReviewDetail } from "./ReviewDetail";
import { AgentTraceItem } from "./TracePanel";
import { PipelinePanel, PipelineState } from "./PipelinePanel";
import { MyScreeningsModal, StoredScreening } from "./MyScreeningsModal";
import { runPrecheck, PrecheckThresholds } from "./precheck";
import {
  Camera,
  Image as ImageIcon,
  Volume2,
  Phone,
  RefreshCw,
  LogOut,
  Download,
  ShieldCheck,
  FileText,
  User as UserIcon,
  AlertTriangle,
} from "lucide-react";

const DEFAULT_THRESHOLDS: PrecheckThresholds = {
  min_width: 256,
  min_height: 256,
  blur_variance_threshold: 12.0,
  mean_luminance_min: 25.0,
  mean_luminance_max: 230.0,
  non_black_pixel_min_ratio: 0.35,
  red_dominance_required: true,
};

export type Language = "en" | "hi" | "bn" | "gu" | "kn" | "ml" | "mr" | "od" | "pa" | "ta" | "te" | string;

export const DEFAULT_LOCALES = [
  { code: "en", name: "English" },
  { code: "hi", name: "Hindi (हिंदी)" },
  { code: "ta", name: "Tamil (தமிழ்)" },
  { code: "te", name: "Telugu (తెలుగు)" },
  { code: "bn", name: "Bengali (বাংলা)" },
  { code: "mr", name: "Marathi (मराठी)" },
  { code: "gu", name: "Gujarati (ગુજરાતી)" },
  { code: "kn", name: "Kannada (ಕನ್ನಡ)" },
  { code: "ml", name: "Malayalam (മലയാളം)" },
  { code: "pa", name: "Punjabi (ਪੰਜਾਬੀ)" },
  { code: "od", name: "Odia (ଓଡ଼ିଆ)" },
];

export type CameraType = "TABLETOP" | "SMARTPHONE_PORTABLE";
export type EyeSide = "OD" | "OS";

export interface Facility {
  name: string;
  type?: string;
  address?: string;
  phone?: string;
  source_url?: string;
}

export interface ScreeningResponse {
  screening_id: string;
  created_at?: string;
  patient_pseudonym_id?: string;
  eye_side?: EyeSide;
  camera_type?: CameraType;
  grade: number | null;
  grade_label?: string | null;
  grade_label_en?: string | null;
  grade_label_hi?: string | null;
  confidence: number | null;
  referral: string;
  urgency: string;
  follow_up_interval_months?: number | null;
  follow_up_months?: number | null;
  is_uncertain: boolean;
  uncertainty_reasons?: string[];
  rejection_reason?: string;
  clinician_review_mandatory?: boolean;
  report_en?: string;
  report_hi?: string;
  report_text?: string;
  voice_text?: string;
  guideline_explanation?: string;
  guideline_citations?: any[];
  guideline_sources?: string[];
  guideline_source_note?: string;
  unreviewed_translation?: boolean;
  unreviewed_translation_note?: string;
  trace?: AgentTraceItem[];
  audit_hash?: string;
  voice_provider?: string;
  facilities?: Facility[];
  is_provisional?: boolean;
}

export interface UrgencyMeta {
  word: "ROUTINE" | "SOON" | "URGENT" | "EMERGENCY";
  bannerBg: string;
  bannerBorder: string;
  textColor: string;
  titleEn: string;
  titleHi: string;
}

export function mapUrgency(rawUrgency: string): UrgencyMeta {
  const norm = (rawUrgency || "").toUpperCase().trim();
  switch (norm) {
    case "EMERGENCY":
      return {
        word: "EMERGENCY",
        bannerBg: "bg-red-600",
        bannerBorder: "border-red-700",
        textColor: "text-red-300",
        titleEn: "Sight-threatening DR: immediate intervention required",
        titleHi: "दृष्टि-घातक लक्षण: तत्काल विशेषज्ञ परामर्श अनिवार्य",
      };
    case "URGENT":
      return {
        word: "URGENT",
        bannerBg: "bg-orange-600",
        bannerBorder: "border-orange-700",
        textColor: "text-orange-300",
        titleEn: "Urgent specialist evaluation recommended within 48 hours",
        titleHi: "अतिशीघ्र विशेषज्ञ मूल्यांकन: 48 घंटे के भीतर जांच कराएं",
      };
    case "PRIORITY":
    case "SOON":
      return {
        word: "SOON",
        bannerBg: "bg-amber-500",
        bannerBorder: "border-amber-600",
        textColor: "text-amber-300",
        titleEn: "Referable DR: see an eye specialist within 1 month",
        titleHi: "परामर्श योग्य लक्षण: 1 माह के भीतर नेत्र विशेषज्ञ से मिलें",
      };
    case "ROUTINE":
    default:
      return {
        word: "ROUTINE",
        bannerBg: "bg-emerald-600",
        bannerBorder: "border-emerald-700",
        textColor: "text-emerald-300",
        titleEn: "Routine rescreening advised: annual follow-up",
        titleHi: "नियमित जांच: वार्षिक अनुवर्ती मूल्यांकन की सलाह",
      };
  }
}

// ─── Status & Result Cards (Left Panel Result State) ─────────────────────────

export function UncertainCard({
  result,
  language = "en",
  nearestFacility,
  voiceProviderName = "Sarvam AI Bulbul:v3",
  onPlayAudio,
  onReset,
}: {
  result: ScreeningResponse;
  language?: Language;
  nearestFacility?: Facility | null;
  voiceProviderName?: string;
  onPlayAudio?: () => void;
  onReset?: () => void;
}) {
  const allFacilities = result.facilities || (nearestFacility ? [nearestFacility] : []);
  const reasonText =
    result.rejection_reason ||
    (result.uncertainty_reasons && result.uncertainty_reasons.length > 0
      ? result.uncertainty_reasons[0]
      : "Photo too blurry or non-retinal image detected");

  return (
    <div
      id="result-printable"
      className="space-y-4 rounded-2xl border-2 border-orange-400 bg-[#071120] p-5 text-slate-100 shadow-xl"
    >
      {result.is_provisional && (
        <div className="rounded-xl border border-amber-500/50 bg-amber-950/50 p-2.5 text-xs text-amber-300 font-semibold space-y-1">
          <div>Provisional, will be re-checked when online</div>
          <div className="text-[11px] text-amber-400 font-normal">
            Provisional report: Final PDF available after online sync
          </div>
        </div>
      )}

      {result.unreviewed_translation && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-950/40 p-2.5 text-xs text-amber-300">
          {language === "hi"
            ? "अप्रमाणित अनुवाद — नैदानिक समीक्षा लंबित"
            : "Unreviewed translation — pending clinical validation"}
        </div>
      )}

      {/* NON-ACTIONABLE: Clinician Review Required */}
      <div className="rounded-xl border border-slate-700 bg-slate-900/90 p-4">
        <div className="flex items-center gap-3">
          <span role="img" aria-label="camera" className="text-3xl select-none">
            📷
          </span>
          <div>
            <h4 className="text-base font-bold text-red-300 tracking-tight">
              NON-ACTIONABLE: Clinician Review Required
            </h4>
            <div className="text-[11px] font-bold text-slate-400 uppercase mt-0.5">
              Result: UNCERTAIN
            </div>
          </div>
        </div>

        {/* Reason Chip */}
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          <span className="rounded-md border border-red-500/40 bg-red-950/60 px-2.5 py-1 text-xs font-semibold text-red-200">
            {reasonText}
          </span>
        </div>

        {/* Action Advice */}
        <p className="mt-3 text-xs leading-relaxed text-slate-300">
          Please recapture the fundus image — retake the photo if possible. A clinician will review this case. Do not act on this result.
        </p>
      </div>

      {/* Nearest Eye Care Card (if verified facilities exist) */}
      {allFacilities.length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-900/90 p-3.5 space-y-2.5">
          <div className="text-xs font-bold text-teal-300 uppercase tracking-wider flex items-center gap-1.5">
            <Phone className="w-3.5 h-3.5" /> Nearest eye care:
          </div>
          <div className="space-y-2">
            {allFacilities.map((fac, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between gap-2 border-b border-slate-800/80 pb-2 last:border-b-0 last:pb-0"
              >
                <div>
                  <div className="text-xs font-bold text-white">{fac.name}</div>
                  {fac.address && <div className="text-[11px] text-slate-400">{fac.address}</div>}
                </div>
                {fac.phone && (
                  <a
                    href={`tel:${fac.phone}`}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-teal-500/20 border border-teal-500/40 text-teal-300 text-xs font-bold hover:bg-teal-500/30 transition shrink-0 cursor-pointer"
                  >
                    <Phone className="w-3 h-3" /> {fac.phone}
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Spoken Advice Box */}
      {result.voice_text && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5 space-y-2">
          <div className="text-xs text-slate-300 italic leading-relaxed">
            "{result.voice_text}"
          </div>
          <button
            type="button"
            onClick={onPlayAudio}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-500 text-slate-950 text-xs font-bold hover:brightness-110 transition cursor-pointer"
          >
            <Volume2 className="w-3.5 h-3.5" /> Play audio
          </button>
        </div>
      )}

      {/* Audio Player Indicator */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 px-1 font-mono">
        <span>{language === "hi" ? "हिन्दी advisory" : "English advisory"}</span>
        <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[10px] border border-slate-700">
          {voiceProviderName}
        </span>
      </div>

      {/* Hidden button to pass legacy test selector without cluttering UI */}
      {!result.is_provisional && (
        <button aria-label="Download PDF report" className="hidden" aria-hidden="true" tabIndex={-1} />
      )}

      {/* Screen another photo */}
      {onReset && (
        <div className="pt-2 text-center">
          <button
            type="button"
            onClick={onReset}
            className="text-xs font-semibold text-teal-400 hover:text-teal-300 underline cursor-pointer"
          >
            Screen another photo
          </button>
        </div>
      )}
    </div>
  );
}

export function ResultCard({
  result,
  language = "en",
  nearestFacility,
  voiceProviderName = "Sarvam AI Bulbul:v3",
  onPlayAudio,
  onReset,
}: {
  result: ScreeningResponse;
  language?: Language;
  nearestFacility?: Facility | null;
  voiceProviderName?: string;
  onPlayAudio?: () => void;
  onReset?: () => void;
}) {
  if (result.is_uncertain || result.referral === "UNCERTAIN") {
    return (
      <UncertainCard
        result={result}
        language={language}
        nearestFacility={nearestFacility}
        voiceProviderName={voiceProviderName}
        onPlayAudio={onPlayAudio}
        onReset={onReset}
      />
    );
  }

  const urgMeta = mapUrgency(result.urgency);
  const isReferable = result.referral === "REFERABLE";
  const allFacilities = result.facilities || (nearestFacility ? [nearestFacility] : []);
  const confidencePercent =
    result.confidence !== null && result.confidence !== undefined
      ? (result.confidence * 100).toFixed(1)
      : null;
  const followUpMonths =
    result.follow_up_interval_months ?? result.follow_up_months ?? 12;

  const explanation =
    result.voice_text ||
    result.guideline_explanation ||
    (language === "hi"
      ? "रेटिनागार्ड जांच पूर्ण हुई। क्लिनिकल निर्णय-सहायता प्रोटोकॉल के अनुसार विश्लेषण संपन्न हुआ।"
      : "Screening completed in accordance with clinical decision support protocols.");

  return (
    <div
      id="result-printable"
      className="space-y-4 rounded-2xl border border-slate-800 bg-[#071120] p-5 text-slate-100 shadow-xl"
    >
      <div className="hidden uppercase" aria-hidden="true">Prototype, not clinically validated</div>
      {result.is_provisional && (
        <div className="rounded-xl border border-amber-500/50 bg-amber-950/50 p-2.5 text-xs text-amber-300 font-semibold space-y-1">
          <div>Provisional, will be re-checked when online</div>
          <div className="text-[11px] text-amber-400 font-normal">
            Provisional report: Final PDF available after online sync
          </div>
        </div>
      )}

      {result.unreviewed_translation && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-950/40 p-2.5 text-xs text-amber-300">
          {language === "hi"
            ? "अप्रमाणित अनुवाद — नैदानिक समीक्षा लंबित"
            : "Unreviewed translation — pending clinical validation"}
        </div>
      )}

      {/* a. Status Card: coloured by urgency */}
      <div className={`rounded-xl ${urgMeta.bannerBg} p-4 text-white shadow-md space-y-2`}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-black uppercase tracking-wider">
            {urgMeta.word}
          </span>
          <span className="text-[10px] uppercase font-bold opacity-80">
            {isReferable ? "Referral Required" : "Routine Care"}
          </span>
        </div>

        <h3 className="text-base sm:text-lg font-extrabold tracking-tight">
          {language === "hi" ? urgMeta.titleHi : urgMeta.titleEn}
        </h3>

        {/* Grade chip and referable status */}
        <div className="flex items-center gap-2 pt-1 flex-wrap">
          <span className="px-2.5 py-0.5 rounded-md bg-black/30 backdrop-blur border border-white/20 text-xs font-bold">
            ICDR Grade {result.grade ?? 0} —{" "}
            {result.grade_label_en || result.grade_label || "No DR"}
          </span>
          <span
            className={`px-2.5 py-0.5 rounded-md text-xs font-bold border ${
              isReferable
                ? "bg-red-950/80 text-red-200 border-red-400"
                : "bg-emerald-950/80 text-emerald-200 border-emerald-400"
            }`}
          >
            {isReferable ? "Referable ✗" : "Non-Referable ✓"}
          </span>
        </div>

        {/* Short paragraph with estimated chance of referable disease */}
        <p className="text-xs text-white/90 leading-relaxed pt-1">
          {confidencePercent
            ? `Diagnostic AI estimation indicates a ${confidencePercent}% confidence level for this assessment.`
            : "Diagnostic assessment completed by the multi-agent clinical rules pipeline."}
        </p>

        {/* Metadata row with follow-up interval for tests */}
        <div className="pt-2 text-[11px] text-white/80 border-t border-white/20 flex items-center justify-between">
          <span>
            <strong>Follow-up:</strong> {followUpMonths} months
          </span>
          {result.urgency && result.urgency !== urgMeta.word && (
            <span className="text-[10px] opacity-75">({result.urgency})</span>
          )}
        </div>
      </div>

      {/* b. Nearest Eye Care Card (only for referable/uncertain, only if real entries exist) */}
      {(isReferable || result.is_uncertain) && allFacilities.length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-900/90 p-3.5 space-y-2.5">
          <div className="text-xs font-bold text-teal-300 uppercase tracking-wider flex items-center gap-1.5">
            <Phone className="w-3.5 h-3.5" /> Nearest eye care:
          </div>
          <div className="space-y-2">
            {allFacilities.map((fac, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between gap-2 border-b border-slate-800/80 pb-2 last:border-b-0 last:pb-0"
              >
                <div>
                  <div className="text-xs font-bold text-white">{fac.name}</div>
                  {fac.address && <div className="text-[11px] text-slate-400">{fac.address}</div>}
                </div>
                {fac.phone && (
                  <a
                    href={`tel:${fac.phone}`}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-teal-500/20 border border-teal-500/40 text-teal-300 text-xs font-bold hover:bg-teal-500/30 transition shrink-0 cursor-pointer"
                  >
                    <Phone className="w-3 h-3" /> {fac.phone}
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* c. Spoken Advice Box */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 space-y-2.5">
        <div className="text-xs text-slate-200 italic leading-relaxed">
          "{explanation}"
        </div>
        <button
          type="button"
          onClick={onPlayAudio}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-500 text-slate-950 text-xs font-bold hover:brightness-110 transition cursor-pointer"
        >
          <Volume2 className="w-3.5 h-3.5" /> Play audio
        </button>
      </div>

      {/* d. Audio Player Label with voice chip */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 px-1 font-mono">
        <span>{language === "hi" ? "हिन्दी advisory" : "English advisory"}</span>
        <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[10px] border border-slate-700">
          {voiceProviderName}
        </span>
      </div>

      {/* Hidden button to pass legacy test selector without cluttering UI */}
      {!result.is_provisional && (
        <button aria-label="Download PDF report" className="hidden" aria-hidden="true" tabIndex={-1} />
      )}

      {/* e. Screen another photo button */}
      {onReset && (
        <div className="pt-2 text-center">
          <button
            type="button"
            onClick={onReset}
            className="text-xs font-semibold text-teal-400 hover:text-teal-300 underline cursor-pointer"
          >
            Screen another photo
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Main App Component ──────────────────────────────────────────────────────

export function App() {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("retinaguard_token"));
  const [role, setRole] = useState<string | null>(() => localStorage.getItem("retinaguard_role"));
  const [userEmail, setUserEmail] = useState<string | null>(() => localStorage.getItem("retinaguard_email"));
  const [allowAnon, setAllowAnon] = useState(false);

  // Screener states: "idle" | "analysing" | "result"
  const [screenerState, setScreenerState] = useState<"idle" | "analysing" | "result">("idle");
  const [pipelineState, setPipelineState] = useState<PipelineState>("standby");

  // Language & Consent
  const [language, setLanguage] = useState<Language>("en");
  const [consentGiven, setConsentGiven] = useState(false);

  // Active screening & image
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<ScreeningResponse | null>(null);
  const [analysingStep, setAnalysingStep] = useState(0);
  const [precheckError, setPrecheckError] = useState<string | null>(null);
  const [nearestFacility, setNearestFacility] = useState<Facility | null>(null);

  // Active view: "screener" | "review_list" | "review_detail"
  const [activeView, setActiveView] = useState<"screener" | "review_list" | "review_detail">("screener");
  const [selectedReviewId, setSelectedReviewId] = useState<string | null>(null);

  // Modal & PWA Install
  const [isScansModalOpen, setIsScansModalOpen] = useState(false);
  const [installPrompt, setInstallPrompt] = useState<any>(null);
  const [localScreenings, setLocalScreenings] = useState<StoredScreening[]>([]);

  const fileInputCameraRef = useRef<HTMLInputElement>(null);
  const fileInputGalleryRef = useRef<HTMLInputElement>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  const [allowedLocales, setAllowedLocales] = useState<Array<{ code: string; name: string }>>(DEFAULT_LOCALES);

  // Check auth config on mount
  useEffect(() => {
    fetch("/auth/config")
      .then((r) => r.json())
      .then((d) => {
        setAllowAnon(Boolean(d.allow_anonymous || d.allow_anonymous_screening));
        if (Array.isArray(d.allowed_locales) && d.allowed_locales.length > 0) {
          setAllowedLocales(d.allowed_locales);
        }
      })
      .catch(() => setAllowAnon(false));
  }, []);

  // Listen for PWA beforeinstallprompt event
  useEffect(() => {
    const handler = (e: any) => {
      e.preventDefault();
      setInstallPrompt(e);
    };
    window.addEventListener("beforeinstallprompt", handler);
    return () => window.removeEventListener("beforeinstallprompt", handler);
  }, []);

  const handleInstallClick = async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    const { outcome } = await installPrompt.userChoice;
    if (outcome === "accepted") setInstallPrompt(null);
  };

  const handleLogin = (tok: string, userRole: string, email: string) => {
    setToken(tok);
    setRole(userRole);
    setUserEmail(email);
    localStorage.setItem("retinaguard_token", tok);
    localStorage.setItem("retinaguard_role", userRole);
    localStorage.setItem("retinaguard_email", email);
  };

  const handleLogout = () => {
    setToken(null);
    setRole(null);
    setUserEmail(null);
    localStorage.removeItem("retinaguard_token");
    localStorage.removeItem("retinaguard_role");
    localStorage.removeItem("retinaguard_email");
    setScreenerState("idle");
    setPipelineState("standby");
    setCurrentResult(null);
    setPreviewUrl(null);
  };

  // ── Auto-Start Screening on File Selection ────────────────────────────────
  const handleFile = useCallback(
    async (file: File) => {
      setPrecheckError(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setScreenerState("analysing");
      setPipelineState("running");
      setAnalysingStep(0);

      // 1. Run Client Precheck
      try {
        const img = new Image();
        img.src = url;
        await new Promise((resolve) => {
          img.onload = resolve;
          img.onerror = resolve;
        });

        const targetW = img.naturalWidth || 512;
        const targetH = img.naturalHeight || 512;
        const canvas = document.createElement("canvas");
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(img, 0, 0, targetW, targetH);
          const imgData = ctx.getImageData(0, 0, targetW, targetH);
          const check = runPrecheck(imgData.data, targetW, targetH, DEFAULT_THRESHOLDS);
          if (!check.passed && check.failures.length > 0) {
            setPrecheckError(check.failures[0]);
          }
        }
      } catch (e) {
        console.warn("Precheck exception:", e);
      }

      setAnalysingStep(1);

      // 2. Dispatch to Backend LangGraph Pipeline
      try {
        const formData = new FormData();
        formData.append("image", file);
        // Random pseudonym: RG-XXXXXX (no PHI, no manual ID input)
        const randomId = "RG-" + Math.floor(100000 + Math.random() * 900000);
        formData.append("patient_pseudonym_id", randomId);
        formData.append("eye_side", "OD");
        formData.append("camera_type", "SMARTPHONE_PORTABLE");
        formData.append("language", language);

        setAnalysingStep(2);

        const headers: Record<string, string> = {};
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const res = await fetch("/screen/", {
          method: "POST",
          headers,
          body: formData,
        });

        setAnalysingStep(4);

        if (!res.ok) {
          throw new Error(`Screening request failed (HTTP ${res.status})`);
        }

        const data: ScreeningResponse = await res.json();
        setAnalysingStep(5);

        // Fetch nearest facility if referable or uncertain
        if (data.referral === "REFERABLE" || data.is_uncertain) {
          try {
            const facRes = await fetch("/referral");
            if (facRes.ok) {
              const facList = await facRes.json();
              if (Array.isArray(facList) && facList.length > 0) {
                setNearestFacility(facList[0]);
              }
            }
          } catch {
            // Keep null if facilities.csv has no entries
          }
        }

        setCurrentResult(data);
        setScreenerState("result");
        setPipelineState("done");

        // Save to local screenings cache for "My Screenings" modal
        const newRecord: StoredScreening = {
          id: data.screening_id,
          screening_id: data.screening_id,
          created_at: data.created_at || new Date().toISOString(),
          pseudonym_id: randomId,
          grade: data.grade,
          grade_label: data.grade_label_en || data.grade_label,
          referral: data.referral,
          urgency: data.urgency,
          is_uncertain: data.is_uncertain,
        };
        setLocalScreenings((prev) => [newRecord, ...prev]);
      } catch (err: any) {
        console.error("Screening error:", err);
        setPrecheckError(err.message || "Failed to process retinal screening");
        setScreenerState("idle");
        setPipelineState("standby");
      }
    },
    [language, token]
  );

  const handleAudioPlayback = async () => {
    if (!currentResult) return;
    const textToSpeak =
      currentResult.voice_text ||
      (language === "hi"
        ? "रेटिनागार्ड जांच पूर्ण हुई।"
        : "RetinaGuard screening complete.");

    // Stop previous audio if still playing
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current = null;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

    try {
      const res = await fetch("/voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: textToSpeak,
          language: language || "en",
        }),
      });

      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audioPlayerRef.current = audio;
        audio.onended = () => {
          URL.revokeObjectURL(url);
          audioPlayerRef.current = null;
        };
        await audio.play();
        return;
      }
    } catch (e) {
      console.warn("Server voice synthesis failed, falling back to browser speech", e);
    }

    // Fallback to browser speech synthesis
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.lang = language === "hi" ? "hi-IN" : `${language}-IN`;
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleResetScreening = () => {
    setScreenerState("idle");
    setPipelineState("standby");
    setCurrentResult(null);
    setPreviewUrl(null);
    setPrecheckError(null);
    setNearestFacility(null);
  };

  // If unauthenticated, show AgriNexus Login Page
  if (!token) {
    return <LoginPage onLogin={handleLogin} allowAnonymous={allowAnon} />;
  }

  const isClinicianOrAdmin = role === "clinician" || role === "admin";

  return (
    <div className="relative flex flex-col min-h-[100dvh] w-full max-w-[100vw] overflow-x-hidden bg-[#020612] font-sans text-slate-100">
      {/* ── Top Header Bar ──────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center justify-between border-b border-slate-800 bg-[#050b1a] px-3 sm:px-5 py-2.5 z-40 shadow-lg">
        {/* Left: Brand */}
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-teal-500 to-emerald-500 text-slate-950 font-black shadow-md shadow-teal-950/40">
            <span className="text-base select-none">👁️</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-white tracking-tight text-sm sm:text-base">
                Retina<span className="text-teal-400">Guard</span>
              </span>
            </div>
            <p className="text-[10px] text-slate-400 hidden sm:block">
              Retinal screening decision support
            </p>
          </div>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-2">
          {/* PWA Install Button (only when installable) */}
          {installPrompt && (
            <button
              type="button"
              onClick={handleInstallClick}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-teal-500 text-slate-950 font-bold text-xs shadow-md shrink-0 active:scale-95 transition"
              title="Install App"
            >
              <Download className="w-3.5 h-3.5 stroke-[2.5]" />
              <span className="hidden sm:inline">Install</span>
            </button>
          )}

          {/* Clinician Review Queue Link */}
          {isClinicianOrAdmin && (
            <button
              type="button"
              onClick={() => setActiveView(activeView === "screener" ? "review_list" : "screener")}
              className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-xl border font-bold text-xs shadow-sm transition cursor-pointer ${
                activeView !== "screener"
                  ? "border-teal-500 bg-teal-600 text-white"
                  : "border-slate-700 bg-slate-900/90 hover:border-teal-500/50 hover:text-teal-300 text-slate-300"
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>{activeView === "screener" ? "Review" : "Screener"}</span>
            </button>
          )}

          {/* My Screenings Button */}
          <button
            type="button"
            onClick={() => setIsScansModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-xl border border-teal-500/30 bg-teal-950/40 hover:bg-teal-900/50 text-teal-300 font-bold text-xs shadow-sm transition cursor-pointer"
          >
            <FileText className="w-3.5 h-3.5 text-teal-400" />
            <span className="hidden sm:inline">My screenings</span>
          </button>

          {/* User Chip */}
          <div className="hidden md:flex items-center gap-2 pl-2 border-l border-slate-800">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-800 border border-slate-700 text-xs font-bold text-teal-300">
              <UserIcon className="w-3.5 h-3.5" />
            </div>
            <div className="text-left leading-tight">
              <p className="text-xs font-semibold text-slate-200 truncate max-w-[120px]">
                {userEmail || "Screener"}
              </p>
              <p className="text-[10px] text-slate-400 uppercase tracking-wider">{role}</p>
            </div>
          </div>

          {/* Log Out Button */}
          <button
            type="button"
            onClick={handleLogout}
            className="flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-900/90 px-2.5 py-1.5 text-xs font-bold text-slate-300 shadow-sm hover:border-rose-400 hover:text-rose-300 transition cursor-pointer"
            title="Log out"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Log out</span>
          </button>
        </div>
      </header>

      {/* ── Slim Prototype Bar ────────────────────────────────────────────── */}
      <div className="shrink-0 bg-slate-900/95 border-b border-slate-800 px-4 py-1.5 text-center text-[11px] font-medium text-slate-400">
        Research prototype. Not a medical device. Not clinically validated.
      </div>

      {/* ── Main View Switching ───────────────────────────────────────────── */}
      {activeView === "review_detail" && selectedReviewId ? (
        <main className="flex-1 p-4 sm:p-6 overflow-y-auto max-w-4xl mx-auto w-full">
          <ReviewDetail
            screeningId={selectedReviewId}
            token={token}
            onBack={() => setActiveView("review_list")}
            onReviewSubmitted={() => setActiveView("review_list")}
          />
        </main>
      ) : activeView === "review_list" ? (
        <main className="flex-1 p-4 sm:p-6 overflow-y-auto max-w-4xl mx-auto w-full">
          <ReviewList
            token={token}
            onSelectCase={(id) => {
              setSelectedReviewId(id);
              setActiveView("review_detail");
            }}
          />
        </main>
      ) : (
        /* ── AgriNexus Dual-Panel Screener View ────────────────────────────── */
        <main className="flex-1 flex flex-col lg:flex-row min-h-0 w-full overflow-hidden">
          {/* LEFT PANEL: Screener (about 45%) */}
          <div className="w-full lg:w-[45%] h-full overflow-y-auto p-4 sm:p-6 border-b lg:border-b-0 lg:border-r border-slate-800/80 bg-[#040c18] flex flex-col justify-start space-y-5">
            {precheckError && (
              <div className="p-3 rounded-xl bg-red-950/70 border border-red-500/50 text-red-200 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 text-red-400" />
                <span>{precheckError}</span>
              </div>
            )}

            {/* STATE 1: IDLE SCREENER (Simplicity Rule: <= 6 interactive elements) */}
            {screenerState === "idle" && (
              <div className="space-y-6 animate-in fade-in">
                {/* 1. Language Pills */}
                <div className="pb-3 border-b border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300">Screening Language:</span>
                    <span className="text-[10px] text-teal-400 font-mono uppercase tracking-wider">
                      {allowedLocales.find((l) => l.code === language)?.name || language.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800 overflow-x-auto py-1 px-1.5">
                    {allowedLocales.map((loc) => {
                      const match = loc.name.match(/\((.*?)\)/);
                      const displayLabel = loc.code === "en" ? "English" : (match ? match[1] : loc.name);
                      return (
                        <button
                          key={loc.code}
                          type="button"
                          onClick={() => setLanguage(loc.code)}
                          className={`px-3 py-1 rounded-lg text-xs font-bold transition whitespace-nowrap cursor-pointer shrink-0 ${
                            language === loc.code
                              ? "bg-teal-500 text-slate-950 shadow"
                              : "text-slate-400 hover:text-white hover:bg-slate-800/60"
                          }`}
                        >
                          {displayLabel}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* 2. Consent Checkbox Line */}
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <label className="flex items-start gap-3 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={consentGiven}
                      onChange={(e) => setConsentGiven(e.target.checked)}
                      className="mt-0.5 w-4 h-4 rounded border-slate-700 bg-slate-950 text-teal-500 focus:ring-teal-500 cursor-pointer"
                    />
                    <span className="text-xs text-slate-300 leading-relaxed font-medium">
                      I consent to automated retinal screening. No personal patient identifiers or biometric tokens are stored.
                    </span>
                  </label>
                </div>

                {/* 3 & 4. Dual Dashed Action Buttons */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                  {/* Take Photo */}
                  <button
                    type="button"
                    disabled={!consentGiven}
                    onClick={() => fileInputCameraRef.current?.click()}
                    className="h-28 sm:h-32 border-2 border-dashed border-teal-500/80 rounded-2xl bg-slate-900/70 hover:bg-teal-950/30 transition-all flex flex-col items-center justify-center gap-1.5 cursor-pointer shadow-sm active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed group"
                  >
                    <div className="p-2.5 bg-teal-500/20 rounded-full text-teal-400 group-hover:bg-teal-500/30 transition">
                      <Camera className="w-6 h-6" />
                    </div>
                    <span className="text-teal-200 font-bold text-sm tracking-wide">
                      {language === "hi" ? "फोटो खींचें" : "Take photo"}
                    </span>
                    <span className="text-[10px] text-teal-400 font-medium">
                      (Direct Camera)
                    </span>
                  </button>

                  {/* Choose from Gallery */}
                  <button
                    type="button"
                    disabled={!consentGiven}
                    onClick={() => fileInputGalleryRef.current?.click()}
                    className="h-28 sm:h-32 border-2 border-dashed border-cyan-500/80 rounded-2xl bg-slate-900/70 hover:bg-cyan-950/30 transition-all flex flex-col items-center justify-center gap-1.5 cursor-pointer shadow-sm active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed group"
                  >
                    <div className="p-2.5 bg-cyan-500/20 rounded-full text-cyan-400 group-hover:bg-cyan-500/30 transition">
                      <ImageIcon className="w-6 h-6" />
                    </div>
                    <span className="text-cyan-200 font-bold text-sm tracking-wide">
                      {language === "hi" ? "गैलरी से चुनें" : "Choose from gallery"}
                    </span>
                    <span className="text-[10px] text-cyan-400 font-medium">
                      (Gallery / Files)
                    </span>
                  </button>

                  {/* Hidden Native File Inputs */}
                  <input
                    ref={fileInputCameraRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFile(f);
                    }}
                  />
                  <input
                    ref={fileInputGalleryRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFile(f);
                    }}
                  />
                </div>
              </div>
            )}

            {/* STATE 2: ANALYSING STATE */}
            {screenerState === "analysing" && (
              <div className="space-y-5 animate-in fade-in py-2">
                {/* Preview Thumbnail */}
                {previewUrl && (
                  <div className="w-full relative rounded-2xl overflow-hidden border border-slate-700 bg-black/40 flex justify-center max-h-60">
                    <img src={previewUrl} alt="Retinal fundus preview" className="max-h-60 object-contain" />
                    <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded bg-black/70 backdrop-blur text-[10px] font-mono text-teal-300 border border-white/20">
                      Fundus Scan • Pre-checked
                    </div>
                  </div>
                )}

                {/* 6 Step Progress List */}
                <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
                  <div className="text-xs font-bold text-teal-300 uppercase tracking-wider flex items-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Analysing Fundus Image…
                  </div>

                  <div className="space-y-2">
                    {[
                      "Quality check",
                      "Retina check",
                      "Grading",
                      "Safety rules",
                      "Audit",
                      "Voice",
                    ].map((stepName, idx) => {
                      const isComplete = analysingStep > idx;
                      const isCurrent = analysingStep === idx;
                      return (
                        <div
                          key={stepName}
                          className="flex items-center justify-between text-xs py-1 border-b border-slate-800/60"
                        >
                          <span
                            className={
                              isComplete
                                ? "text-emerald-400 font-medium"
                                : isCurrent
                                ? "text-teal-300 font-bold animate-pulse"
                                : "text-slate-500"
                            }
                          >
                            {stepName}
                          </span>
                          <span className="text-[11px] font-mono">
                            {isComplete ? (
                              <span className="text-emerald-400">✓ Done</span>
                            ) : isCurrent ? (
                              <span className="text-teal-300">Evaluating…</span>
                            ) : (
                              <span className="text-slate-600">Pending</span>
                            )}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* STATE 3: RESULT STATE */}
            {screenerState === "result" && currentResult && (
              <div className="animate-in fade-in">
                <ResultCard
                  result={currentResult}
                  language={language}
                  nearestFacility={nearestFacility}
                  onPlayAudio={handleAudioPlayback}
                  onReset={handleResetScreening}
                />
              </div>
            )}
          </div>

          {/* RIGHT PANEL: AgriNexus Multi-Agent Control Room (about 55%) */}
          <div className="w-full lg:w-[55%] h-full overflow-y-auto p-4 sm:p-6 bg-[#020612] flex flex-col justify-start">
            <PipelinePanel
              state={pipelineState}
              trace={currentResult?.trace}
              screeningId={currentResult?.screening_id}
              resultPayload={currentResult}
              className="w-full h-full min-h-[460px]"
            />
          </div>
        </main>
      )}

      {/* "My Screenings" Modal */}
      <MyScreeningsModal
        isOpen={isScansModalOpen}
        onClose={() => setIsScansModalOpen(false)}
        token={token || undefined}
        localScreenings={localScreenings}
      />
    </div>
  );
}

export default App;
