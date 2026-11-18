/**
 * New UI-state tests for the two-panel layout (AgriNexus rebuild):
 *   - PipelinePanel in STANDBY / RUNNING / DONE states
 *   - mapUrgency mapping (PRIORITY → SOON, etc.)
 *   - ResultCard "Screen another photo" button
 *   - UncertainCard: no urgency chip, no follow-up interval
 */
import test from "node:test";
import assert from "node:assert";
import React from "react";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";

import { PipelinePanel } from "../src/PipelinePanel";
import { mapUrgency, ResultCard, UncertainCard } from "../src/App";
import type { AgentTraceItem } from "../src/TracePanel";

// ─── mapUrgency ───────────────────────────────────────────────────────────────

test("mapUrgency: ROUTINE → word ROUTINE, emerald banner", () => {
  const m = mapUrgency("ROUTINE");
  assert.strictEqual(m.word, "ROUTINE");
  assert.ok(m.bannerBg.includes("emerald"), "ROUTINE must use emerald colour");
});

test("mapUrgency: URGENT → word URGENT, orange banner", () => {
  const m = mapUrgency("URGENT");
  assert.strictEqual(m.word, "URGENT");
  assert.ok(m.bannerBg.includes("orange"), "URGENT must use orange colour");
});

test("mapUrgency: EMERGENCY → word EMERGENCY, red banner", () => {
  const m = mapUrgency("EMERGENCY");
  assert.strictEqual(m.word, "EMERGENCY");
  assert.ok(m.bannerBg.includes("red"), "EMERGENCY must use red colour");
});

test("mapUrgency: PRIORITY maps to word SOON (amber)", () => {
  const m = mapUrgency("PRIORITY");
  assert.strictEqual(m.word, "SOON", "PRIORITY backend value must display as SOON");
  assert.ok(m.bannerBg.includes("amber"), "PRIORITY/SOON must use amber colour");
});

test("mapUrgency: SOON maps to word SOON (amber)", () => {
  const m = mapUrgency("SOON");
  assert.strictEqual(m.word, "SOON");
  assert.ok(m.bannerBg.includes("amber"), "SOON must use amber colour");
});

test("mapUrgency: unknown value falls through to ROUTINE", () => {
  const m = mapUrgency("UNKNOWN_VALUE");
  assert.strictEqual(m.word, "ROUTINE");
});

// ─── PipelinePanel – STANDBY ──────────────────────────────────────────────────

test("PipelinePanel STANDBY: state chip reads STANDBY, telemetry shows idle message", () => {
  cleanup();
  render(<PipelinePanel state="standby" />);

  // State chip (may appear multiple times — chip + per-node status labels — just ensure at least one)
  const standbyEls = screen.getAllByText(/standby/i);
  assert.ok(standbyEls.length >= 1, "At least one STANDBY label must be in DOM");

  // Header title
  assert.ok(
    screen.getByText(/Multi-Agent Pipeline/i),
    "Panel title must be visible"
  );

  // Telemetry idle message (unique string)
  assert.ok(
    screen.getByText(/Awaiting retinal image from the screener/i),
    "Idle telemetry message must be visible in STANDBY"
  );

  // Five node names present
  for (const name of ["Vision", "Guidelines", "Safety", "Audit", "Voice"]) {
    const els = screen.getAllByText(new RegExp(`^${name}$`, "i"));
    assert.ok(els.length >= 1, `Node '${name}' must be rendered`);
  }
});

// ─── PipelinePanel – RUNNING ──────────────────────────────────────────────────

test("PipelinePanel RUNNING: state chip reads RUNNING, dispatch message visible", () => {
  cleanup();
  render(<PipelinePanel state="running" />);

  assert.ok(
    screen.getByText(/RUNNING/i),
    "State chip must read RUNNING"
  );

  // The running telemetry shows dispatch lines
  assert.ok(
    screen.getByText(/DISPATCH/i),
    "Running telemetry must include DISPATCH line"
  );

  // No real timing numbers – 'ms' must not be in running nodes
  // (timings only appear in DONE state from trace)
  const msTexts = screen.queryAllByText(/\d+\.\d+ ms/);
  assert.strictEqual(
    msTexts.length,
    0,
    "No real ms timings must appear in RUNNING state"
  );
});

// ─── PipelinePanel – DONE ─────────────────────────────────────────────────────

const SAMPLE_TRACE: AgentTraceItem[] = [
  { agent: "vision",    status: "passed", duration_ms: 412, note: "quality OK",       chain_hash: "abc123" },
  { agent: "guideline", status: "passed", duration_ms: 88,  note: "DR grade 2",       chain_hash: "def456" },
  { agent: "safety",    status: "passed", duration_ms: 32,  note: "no escalation",    chain_hash: "ghi789" },
  { agent: "audit",     status: "passed", duration_ms: 15,  note: "chain verified",   chain_hash: "jkl012" },
  { agent: "voice",     status: "passed", duration_ms: 210, note: "audio ready",      chain_hash: "mno345" },
];

test("PipelinePanel DONE: state chip reads DONE, real ms timings visible from trace", () => {
  cleanup();
  render(
    <PipelinePanel state="done" trace={SAMPLE_TRACE} screeningId="scr-test-001" />
  );

  const doneMatches = screen.getAllByText(/DONE/i);
  assert.ok(doneMatches.length >= 1, "State chip must read DONE");

  // Caption label (unique)
  assert.ok(
    screen.getByText(/Replayed from the server trace/i),
    "Must show 'Replayed from the server trace' caption in DONE state"
  );

  // Real timing from trace (412 ms vision) — may appear in node card AND ledger
  const msNodes = screen.getAllByText(/412\.0 ms/);
  assert.ok(msNodes.length >= 1, "Vision 412 ms timing must appear in DONE state");

  // Telemetry audit hash line (unique)
  assert.ok(
    screen.getByText(/AUDIT_CHAIN/i),
    "Audit chain hash line must appear in telemetry ledger"
  );

  // "✓ VERIFIED" is unique; note text "chain verified" matches /VERIFIED/i too,
  // so test the exact rendered string of the audit verification badge.
  const verifiedBadges = screen.getAllByText(/✓ VERIFIED/i);
  assert.ok(verifiedBadges.length >= 1, "Audit must show ✓ VERIFIED status in ledger");
});

test("PipelinePanel DONE: telemetry ledger lists all 5 agent names from trace", () => {
  cleanup();
  render(
    <PipelinePanel state="done" trace={SAMPLE_TRACE} screeningId="scr-test-002" />
  );

  for (const agent of ["vision", "guideline", "safety", "audit", "voice"]) {
    const upper = agent.toUpperCase();
    // Ledger renders [AGENT] tags — getAllByText since node title may also match
    const matches = screen.getAllByText(new RegExp(upper, "i"));
    assert.ok(
      matches.length >= 1,
      `Telemetry ledger must include agent [${upper}]`
    );
  }
});

// ─── UncertainCard: NO urgency chip, NO follow-up interval ───────────────────

test("UncertainCard: no follow-up interval text, no urgency colour chip", () => {
  cleanup();
  const mockUncertain = {
    screening_id: "scr-uncertain-001",
    grade: null,
    grade_label: null,
    confidence: null,
    referral: "UNCERTAIN",
    urgency: "PRIORITY",
    follow_up_months: null,
    is_uncertain: true,
    unreviewed_translation: false,
  };

  render(<UncertainCard result={mockUncertain} language="en" />);

  // Must NOT show any follow-up interval text
  const followUp = screen.queryByText(/follow[- ]?up interval/i);
  assert.strictEqual(followUp, null, "UncertainCard must not render a follow-up interval");

  // Must NOT show standalone urgency colour chip labels (SOON, URGENT, EMERGENCY, ROUTINE)
  // Note: "UNCERTAIN" itself IS shown intentionally — test only strict standalone colour-chip words
  const routineChip  = screen.queryByRole("status"); // urgency chips use role=status if set
  // Use a targeted approach: look for standalone text nodes that are exactly ROUTINE/SOON/URGENT/EMERGENCY
  for (const word of ["ROUTINE", "SOON", "URGENT", "EMERGENCY"]) {
    const exact = screen.queryAllByText(word);
    assert.strictEqual(exact.length, 0, `UncertainCard must not render standalone '${word}' urgency chip`);
  }

  // Must show retake text
  assert.ok(
    screen.getAllByText(/Retake the photo if possible/i).length >= 1,
    "UncertainCard must show retake advice"
  );
});

test("UncertainCard: retake icon and clinician review text present", () => {
  cleanup();
  const mockUncertain = {
    screening_id: "scr-uncertain-002",
    grade: null,
    grade_label: null,
    confidence: null,
    referral: "UNCERTAIN",
    urgency: "ROUTINE",
    follow_up_months: null,
    is_uncertain: true,
    unreviewed_translation: false,
  };

  render(<UncertainCard result={mockUncertain} language="en" />);

  // "A clinician will review this" appears in two DOM nodes — just check at least one
  const clinicianTexts = screen.getAllByText(/A clinician will review this/i);
  assert.ok(
    clinicianTexts.length >= 1,
    "UncertainCard must include clinician review reassurance"
  );

  // The retake icon (📷) must be in DOM — rendered with role="img"
  const icon = screen.getByRole("img", { name: /camera/i });
  assert.ok(icon, "Retake camera icon 📷 must be present in UncertainCard");
});

// ─── ResultCard: "Screen another photo" reset button ─────────────────────────

test("ResultCard: 'Screen another photo' button fires onReset callback", () => {
  cleanup();
  let resetCalled = false;

  const mockResult = {
    screening_id: "scr-reset-001",
    grade: 1,
    grade_label: "Mild NPDR",
    confidence: 0.88,
    referral: "NON_REFERABLE",
    urgency: "ROUTINE",
    follow_up_months: 12,
    is_uncertain: false,
    clinician_review_mandatory: false,
    report_text: "No sight-threatening DR changes.",
    voice_text: "Routine findings. Annual rescreening recommended.",
    guideline_explanation: "No referral indicated at this grade.",
    guideline_citations: ["RSSDI 2022"],
  };

  render(
    <ResultCard
      result={mockResult}
      language="en"
      onPlayAudio={() => {}}
      onReset={() => { resetCalled = true; }}
    />
  );

  const resetBtn = screen.getByRole("button", { name: /Screen another photo/i });
  assert.ok(resetBtn, "ResultCard must render 'Screen another photo' button when onReset is provided");

  fireEvent.click(resetBtn);
  assert.ok(resetCalled, "Clicking 'Screen another photo' must call onReset");
});

test("UncertainCard: 'Screen another photo' button fires onReset callback", () => {
  cleanup();
  let resetCalled = false;

  const mockUncertain = {
    screening_id: "scr-reset-002",
    grade: null,
    grade_label: null,
    confidence: null,
    referral: "UNCERTAIN",
    urgency: "PRIORITY",
    follow_up_months: null,
    is_uncertain: true,
    unreviewed_translation: false,
  };

  render(
    <UncertainCard
      result={mockUncertain}
      language="en"
      onReset={() => { resetCalled = true; }}
    />
  );

  const resetBtn = screen.getByRole("button", { name: /Screen another photo/i });
  assert.ok(resetBtn, "UncertainCard must render 'Screen another photo' button when onReset is provided");

  fireEvent.click(resetBtn);
  assert.ok(resetCalled, "Clicking 'Screen another photo' on UncertainCard must call onReset");
});
