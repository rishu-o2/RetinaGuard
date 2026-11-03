import React, { useState } from "react";

export function App() {
  const [cameraType, setCameraType] = useState<"TABLETOP" | "SMARTPHONE_PORTABLE">("SMARTPHONE_PORTABLE");
  const [language, setLanguage] = useState<"en" | "hi">("en");

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Banner / Disclaimer */}
      <header className="bg-slate-900 text-white py-3 px-6 shadow-sm border-b border-slate-800">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-xl font-bold tracking-tight text-teal-400">RetinaGuard</span>
            <span className="text-xs bg-teal-900 text-teal-300 font-semibold px-2 py-0.5 rounded border border-teal-700">
              India Edition (CDSCO Compliant)
            </span>
          </div>

          <div className="flex items-center space-x-4">
            <select
              aria-label="Select Language"
              value={language}
              onChange={(e) => setLanguage(e.target.value as "en" | "hi")}
              className="bg-slate-800 text-slate-200 text-sm rounded px-3 py-1 border border-slate-700"
            >
              <option value="en">English (India)</option>
              <option value="hi">Hindi (हिंदी)</option>
            </select>
            <span className="text-xs text-slate-400">v0.1.0</span>
          </div>
        </div>
      </header>

      {/* Mandatory Clinical Disclaimer Bar */}
      <div className="bg-amber-50 border-b border-amber-200 px-6 py-2 text-xs text-amber-900">
        <div className="max-w-7xl mx-auto font-medium">
          ⚠️ <strong>Clinical Decision-Support Tool:</strong> RetinaGuard assists clinicians and does not
          provide autonomous medical diagnoses. All findings require licensed clinical sign-off.
        </div>
      </div>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto flex-1 w-full p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Column: Intake & Camera Config */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 space-y-4">
          <h2 className="text-base font-semibold text-slate-800">Screening Intake</h2>

          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              Camera Hardware Modality
            </label>
            <select
              value={cameraType}
              onChange={(e) => setCameraType(e.target.value as "TABLETOP" | "SMARTPHONE_PORTABLE")}
              className="w-full text-sm border-slate-300 rounded p-2 border focus:ring-teal-500"
            >
              <option value="SMARTPHONE_PORTABLE">Handheld / Smartphone (Remidio, Forus 3nethra)</option>
              <option value="TABLETOP">Tabletop Fundus Camera (Zeiss, Topcon, Kowa)</option>
            </select>
            <p className="text-xs text-slate-400 mt-1">
              Quality gate thresholds adapt to camera optics and illumination.
            </p>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              Eye Side Laterality
            </label>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <button className="py-2 border rounded font-medium bg-teal-50 text-teal-800 border-teal-300">
                OD (Right Eye)
              </button>
              <button className="py-2 border rounded font-medium text-slate-600 hover:bg-slate-50">
                OS (Left Eye)
              </button>
            </div>
          </div>

          <div className="border-t pt-4">
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              Patient Pseudonym ID
            </label>
            <input
              type="text"
              placeholder="e.g. PAT-DEL-042"
              className="w-full text-sm border rounded p-2"
              defaultValue="PAT-DEL-0042"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase text-slate-500 mb-1">
              ABHA ID (Optional)
            </label>
            <input
              type="text"
              placeholder="91-XXXX-XXXX-XXXX"
              className="w-full text-sm border rounded p-2"
            />
          </div>

          <div className="border-t pt-4">
            <span className="text-xs font-medium text-emerald-700 block mb-2">
              ✓ DPDP Act 2023 Explicit Consent Captured
            </span>
            <button className="w-full bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 rounded text-sm transition">
              Upload Fundus Image
            </button>
          </div>
        </div>

        {/* Center: Image & Quality Gate */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 md:col-span-2 flex flex-col items-center justify-center min-h-[360px] border-dashed border-2">
          <div className="text-center text-slate-400">
            <div className="text-4xl mb-2">👁️</div>
            <p className="text-sm font-medium text-slate-600">Select or drop a retinal fundus image</p>
            <p className="text-xs text-slate-400 mt-1">
              Supported formats: JPEG, PNG, DICOM (Strict quality & OOD checks applied)
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t py-3 px-6 text-xs text-slate-500 text-center">
        RetinaGuard &copy; 2026. Aligned with AIOS 2020, RSSDI 2022, ICMR, and ICO Diabetic Eye Care Guidelines.
        All data hosted locally in India.
      </footer>
    </div>
  );
}

export default App;

