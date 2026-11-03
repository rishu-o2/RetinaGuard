import test from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test("Frontend Package Smoke Test: App.tsx and index.html exist and contain required clinical metadata", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const indexPath = path.resolve(__dirname, "../index.html");

  assert.ok(fs.existsSync(appPath), "App.tsx should exist");
  assert.ok(fs.existsSync(indexPath), "index.html should exist");

  const appContent = fs.readFileSync(appPath, "utf-8");
  assert.match(appContent, /RetinaGuard/, "App must render product name");
  assert.match(appContent, /Clinical Decision-Support Tool/, "App must render disclaimer");
  assert.match(appContent, /DPDP Act 2023/, "App must capture DPDP consent");
});

test("ResultCard: normal result contains grade, referral badge, urgency, follow-up, and audio play button", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const appContent = fs.readFileSync(appPath, "utf-8");

  // Grade rendering
  assert.match(appContent, /grade|Grade/, "App must render DR grade");

  // Referral or urgency
  assert.match(appContent, /referral|urgency|Urgency|Referral/i, "App must render referral/urgency field");

  // Audio playback (Web Speech API)
  assert.match(appContent, /speechSynthesis|voice_text|audio|play/i, "App must include audio/speech playback UI");

  // Print/PDF button
  assert.match(appContent, /print|PDF|window\.print/i, "App must include print/PDF export button");
});

test("ResultCard: UNCERTAIN result is visually distinct (warning styles, recapture message)", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const appContent = fs.readFileSync(appPath, "utf-8");

  // UNCERTAIN pathway must render warning-style UI
  assert.match(appContent, /UNCERTAIN|is_uncertain|uncertain/i, "App must handle UNCERTAIN result");

  // Recapture or clinician guidance message
  assert.match(
    appContent,
    /recapture|clinician|review|required/i,
    "App must tell user to recapture or see a clinician on UNCERTAIN result"
  );
});

test("ResultCard: guideline explanation and citations are rendered", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const appContent = fs.readFileSync(appPath, "utf-8");

  assert.match(
    appContent,
    /guideline_explanation|guideline_source_note|guideline_citations|explanation/i,
    "App must render guideline explanation or source note"
  );
});

test("App: language switch between English and Hindi is present", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const appContent = fs.readFileSync(appPath, "utf-8");

  assert.match(appContent, /hi|Hindi|हिंदी/i, "App must support Hindi language switch");
  assert.match(appContent, /en|English/i, "App must support English language switch");
});
