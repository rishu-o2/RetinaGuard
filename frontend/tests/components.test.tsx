import test from "node:test";
import assert from "node:assert";
import React from "react";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";

import { ResultCard, UncertainCard } from "../src/App";
import { LoginPage } from "../src/LoginPage";
import { ReviewDetail } from "../src/ReviewDetail";

test("DOM Test 1: ResultCard renders normal referable finding with Grade, Urgency, Follow-up and PDF download", () => {
  cleanup();
  const mockResult = {
    screening_id: "scr-test-uuid-001",
    grade: 2,
    grade_label: "Moderate Non-Proliferative DR",
    confidence: 0.942,
    referral: "REFERABLE",
    urgency: "PRIORITY",
    follow_up_months: 3,
    is_uncertain: false,
    clinician_review_mandatory: true,
    report_text: "Findings indicate moderate NPDR with microaneurysms.",
    voice_text: "Moderate NPDR detected. Referral to ophthalmologist within 3 months.",
    guideline_explanation: "RSSDI and AIOS 2020 guidelines recommend referral within 1-3 months for Moderate NPDR.",
    guideline_citations: ["AIOS 2020 Section 4", "RSSDI 2022 Table 3"],
  };

  const { container } = render(
    <ResultCard
      result={mockResult}
      language="en"
      onPlayAudio={() => {}}
    />
  );

  // 1. Prototype warning banner present in DOM
  const banner = container.querySelector(".uppercase");
  assert.ok(banner, "Prototype warning banner must be in DOM");
  assert.match(banner.textContent || "", /Prototype, not clinically validated/i);

  // 2. Grade and Referral in DOM
  assert.ok(screen.getByText(/ICDR Grade/i), "Must render ICDR Grade");
  assert.ok(screen.getByText(/Referable ✗/i), "Must render Referable badge");

  // 3. Urgency and follow-up
  assert.ok(screen.getByText(/PRIORITY/i), "Must render PRIORITY urgency");
  assert.ok(screen.getByText(/Follow-up:/i), "Must render Follow-up label");
  assert.ok(screen.getAllByText(/months/i).length >= 1, "Must render months follow-up");

  // 4. Download PDF button
  const pdfBtn = container.querySelector("button[aria-label='Download PDF report']");
  assert.ok(pdfBtn, "Must render server-side Download PDF button");
});

test("DOM Test 2: UncertainCard renders distinct warning style, UNCERTAIN heading and recapture advice", () => {
  cleanup();
  const mockUncertainResult = {
    screening_id: "scr-uncertain-002",
    grade: null,
    grade_label: "UNCERTAIN",
    confidence: null,
    referral: "UNCERTAIN",
    urgency: "PRIORITY",
    follow_up_months: null,
    is_uncertain: true,
    clinician_review_mandatory: true,
    rejection_reason: "Image failed blur quality check (Laplacian variance 8.4 < 12.21 threshold)",
  };

  const { container } = render(
    <UncertainCard
      result={mockUncertainResult}
      language="en"
    />
  );

  // 1. Distinct UNCERTAIN heading
  assert.ok(screen.getByText(/Result: UNCERTAIN/i), "Must render Result: UNCERTAIN");

  // 2. Rejection reason displayed
  assert.ok(screen.getByText(/Image failed blur quality check/i), "Must render rejection reason");

  // 3. Recapture guidance advice
  assert.ok(screen.getByText(/Please recapture/i), "Must instruct clinician/screener to recapture image");

  // 4. Warning styles
  const cardDiv = container.querySelector("#result-printable");
  assert.ok(cardDiv?.className.includes("border-orange-400"), "Must have orange warning border");
});

test("DOM Test 3: LoginPage renders email, password inputs, sign-in button and prototype disclaimer", () => {
  cleanup();
  let loggedInWith: any = null;
  const { container } = render(
    <LoginPage
      allowAnonymous={true}
      onLogin={(token, role, email) => {
        loggedInWith = { token, role, email };
      }}
    />
  );

  // 1. Heading and disclaimer
  assert.ok(screen.getByText("RetinaGuard"), "Must render RetinaGuard header");
  assert.ok(screen.getByText(/Prototype, not clinically validated/i), "Must render prototype disclaimer");

  // 2. Email and Password input elements in DOM
  const emailInput = container.querySelector("input[type='email']") as HTMLInputElement;
  const passwordInput = container.querySelector("input[type='password']") as HTMLInputElement;
  const submitBtn = container.querySelector("button[type='submit']") as HTMLButtonElement;

  assert.ok(emailInput, "Must render email input");
  assert.ok(passwordInput, "Must render password input");
  assert.ok(submitBtn, "Must render submit button");
  assert.match(submitBtn.textContent || "", /Sign in/i);

  // 3. Screener guest button in DOM
  const guestBtn = screen.getByText(/Continue as PHC Screener/i);
  assert.ok(guestBtn, "Must render guest screener option");

  // Fire guest click
  fireEvent.click(guestBtn);
  assert.deepStrictEqual(loggedInWith, {
    token: "",
    role: "screener",
    email: "screener@local.phc",
  });

  // Verify guest button does NOT render when allowAnonymous is false
  cleanup();
  render(
    <LoginPage
      allowAnonymous={false}
      onLogin={() => {}}
    />
  );
  assert.strictEqual(screen.queryByText(/Continue as PHC Screener/i), null);
});

test("DOM Test 4: ReviewDetail renders case data and Confirm/Override buttons, revealing override controls on click", async () => {
  cleanup();
  const mockScreeningData = {
    id: "case-rev-101",
    pseudonym_id: "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
    status: "pending_review",
    model_version: "v1.0.0",
    created_at: "2026-11-05T12:00:00Z",
    result: {
      grade: 3,
      grade_label_en: "Severe Non-Proliferative DR",
      urgency: "URGENT",
      referral: "REFERABLE",
      follow_up_interval_months: 1,
      report_text: "4:2:1 rule met. Prompt ophthalmology referral needed.",
    },
  };

  // Mock global fetch
  (globalThis as any).fetch = async (url: string) => {
    return {
      ok: true,
      json: async () => mockScreeningData,
    };
  };

  const { container } = render(
    <ReviewDetail
      screeningId="case-rev-101"
      token="mock-token-xyz"
      onBack={() => {}}
      onReviewSubmitted={() => {}}
    />
  );

  // 1. Wait for case details to load
  const confirmBtn = await waitFor(() => screen.getByText(/Confirm AI Assessment/i));
  const overrideBtn = screen.getByText(/Override Assessment/i);
  assert.ok(confirmBtn, "Confirm button must be rendered in DOM");
  assert.ok(overrideBtn, "Override button must be rendered in DOM");

  // 2. Before clicking override, grade dropdown is not visible
  assert.strictEqual(container.querySelector("select"), null);

  // 3. Click Override -> reveals grade dropdown and clinical note textarea
  fireEvent.click(overrideBtn);

  const gradeSelect = container.querySelector("select");
  const noteTextarea = container.querySelector("textarea");
  const submitOverrideBtn = screen.getByText(/Submit Override/i);

  assert.ok(gradeSelect, "Grade select dropdown must appear after clicking Override");
  assert.ok(noteTextarea, "Clinical note textarea must appear after clicking Override");
  assert.ok(submitOverrideBtn, "Submit Override button must appear");
});
