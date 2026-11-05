import React, { useState, useRef, useCallback } from "react";
import { LoginPage } from "./LoginPage";
import { ReviewList } from "./ReviewList";
import { ReviewDetail } from "./ReviewDetail";

// ─── Types ────────────────────────────────────────────────────────────────────

type Language = "en" | "hi";
type CameraType = "TABLETOP" | "SMARTPHONE_PORTABLE";
type EyeSide = "OD" | "OS";

interface ScreeningResponse {
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

function UncertainCard({
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
          ? "कृपया छवि पुनः लें (recapture) और एक प्रशिक्षित चिकित्सक से समीक्षा (clinician review) करवाएँ।"
          : "Please recapture the fundus image and request a clinician review. Do not act on this result."}
      </div>

      <p className="text-xs text-orange-600">
        ✓ Clinician review mandatory — event logged (no patient identifiers stored).
      </p>
    </div>
  );
}

function ResultCard({
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

      {/* Buttons row */}
      <div className="flex flex-wrap gap-2 mt-2 print:hidden">
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
        {result.screening_id && (
          <button
            onClick={() => window.open(`/screenings/${result.screening_id}/report.pdf?lang=${language}`, "_blank")}
            aria-label="Download PDF report"
            className="flex items-center space-x-2 px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition"
          >
            <span>📄</span>
            <span>{language === "hi" ? "PDF डाउनलोड करें" : "Download PDF"}</span>
          </button>
        )}
      </div>

      <p className="text-xs text-slate-400">
        ✓ Clinician review mandatory. Screening ID: {result.screening_id}
      </p>
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

  // File + preview state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // API state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScreeningResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── File handling ────────────────────────────────────────────────────────

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

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

    setLoading(true);
    setError(null);
    setResult(null);

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
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unexpected error");
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

            {/* Language switch — en / hi (हिंदी) */}
            <select
              aria-label="Select Language"
              value={language}
              onChange={(e) => setLanguage(e.target.value as Language)}
              className="bg-slate-800 text-slate-200 text-sm rounded px-3 py-1 border border-slate-700"
            >
              <option value="en">English (India)</option>
              <option value="hi">Hindi (हिंदी)</option>
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
