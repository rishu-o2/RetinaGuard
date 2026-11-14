import test from "node:test";
import assert from "node:assert/strict";
import React from "react";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

import { ResultCard, UncertainCard, ScreeningResponse } from "../src/App.js";
import { TracePanel } from "../src/TracePanel.js";

test("Feature 1 UI: TracePanel renders 5 agents with status and duration in collapsed details", () => {
  cleanup();
  const mockTrace = [
    { agent: "vision", status: "passed" as const, duration_ms: 45.2, note: "Quality gate and inference passed" },
    { agent: "safety", status: "passed" as const, duration_ms: 1.1, note: "ClinicalRulesEngine executed" },
    { agent: "guideline", status: "skipped" as const, duration_ms: 0.0, note: "Skipped due to uncertainty" },
    { agent: "audit", status: "passed" as const, duration_ms: 3.4, note: "Audit entry recorded" },
    { agent: "voice", status: "passed" as const, duration_ms: 0.8, note: "Voice script prepared" },
  ];

  const { container } = render(<TracePanel trace={mockTrace} />);

  const details = container.querySelector("details");
  assert.ok(details, "Must render <details> element");

  // Check agent names in summary / panel
  assert.ok(screen.getByText(/How this result was made \(5 pipeline steps\)/i));
  assert.ok(screen.getByText(/Vision Agent/i));
  assert.ok(screen.getByText(/Safety Agent/i));
  assert.ok(screen.getByText(/Guideline Agent/i));
  assert.ok(screen.getByText(/Audit Agent/i));
  assert.ok(screen.getByText(/Voice Agent/i));

  // Check timings
  assert.ok(screen.getByText(/45.2 ms/i));
});

test("Feature 2 UI: Eye-facility referral directory renders on Referable and Uncertain cards", () => {
  cleanup();
  const mockFacilities = [
    {
      name: "Aundh District Hospital",
      type: "DH",
      address: "Aundh, Pune, Maharashtra",
      phone: "+91-20-27282929",
    },
    {
      name: "Sassoon General Hospital",
      type: "Tertiary",
      address: "Station Road, Pune",
      phone: "+91-20-26128000",
    },
  ];

  const referableResult: ScreeningResponse = {
    screening_id: "scr-fac-001",
    grade: 2,
    grade_label: "Moderate NPDR",
    confidence: 0.91,
    referral: "REFERABLE",
    urgency: "PRIORITY",
    follow_up_months: 3,
    is_uncertain: false,
    clinician_review_mandatory: true,
    facilities: mockFacilities,
  };

  const { container } = render(
    <ResultCard result={referableResult} language="en" onPlayAudio={() => {}} />
  );

  assert.ok(screen.getByText(/Nearest eye care:/i), "Must render Nearest eye care header");
  assert.ok(screen.getByText(/Aundh District Hospital/i), "Must render facility name");
  assert.ok(screen.getByText(/\+91-20-27282929/i), "Must render facility phone");
  assert.ok(screen.getByText(/Sassoon General Hospital/i), "Must render second facility");

  // Also verify on UncertainCard
  cleanup();
  const uncertainResult: ScreeningResponse = {
    screening_id: "scr-fac-unc-002",
    grade: null,
    grade_label: "UNCERTAIN",
    confidence: null,
    referral: "UNCERTAIN",
    urgency: "PRIORITY",
    follow_up_months: 1,
    is_uncertain: true,
    clinician_review_mandatory: true,
    facilities: mockFacilities,
  };

  render(<UncertainCard result={uncertainResult} language="en" />);
  assert.ok(screen.getByText(/Nearest eye care:/i), "UncertainCard must also render nearest eye care");
  assert.ok(screen.getByText(/Aundh District Hospital/i));
});

test("Feature 3 UI: Offline mode provisional banner and PDF export restriction", () => {
  cleanup();
  const provisionalResult: ScreeningResponse = {
    screening_id: "offline-provisional-001",
    grade: 2,
    grade_label: "Moderate NPDR",
    confidence: 0.90,
    referral: "REFERABLE",
    urgency: "PRIORITY",
    follow_up_months: 3,
    is_uncertain: false,
    clinician_review_mandatory: true,
    is_provisional: true, // Marked provisional
  };

  const { container } = render(
    <ResultCard result={provisionalResult} language="en" onPlayAudio={() => {}} />
  );

  // 1. Provisional banner displayed
  assert.ok(screen.getByText(/Provisional, will be re-checked when online/i));

  // 2. Final PDF download button must NOT be rendered for provisional results
  const pdfBtn = container.querySelector("button[aria-label='Download PDF report']");
  assert.equal(pdfBtn, null, "Final PDF export button must not be available for provisional results");

  // 3. Clear message that final PDF is unavailable until online sync
  assert.ok(screen.getByText(/Provisional report: Final PDF available after online sync/i));
});
