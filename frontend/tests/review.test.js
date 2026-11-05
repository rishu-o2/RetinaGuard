import test from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test("LoginPage: exists, contains email/password form and prototype disclaimer", () => {
  const loginPath = path.resolve(__dirname, "../src/LoginPage.tsx");
  assert.ok(fs.existsSync(loginPath), "LoginPage.tsx must exist");

  const content = fs.readFileSync(loginPath, "utf-8");
  assert.match(content, /email/i, "Login form must have email field");
  assert.match(content, /password/i, "Login form must have password field");
  assert.match(content, /Prototype, not clinically validated/i, "Login form must render prototype disclaimer");
  assert.match(content, /auth\/login/i, "Login form must post to /auth/login");
});

test("ReviewList: exists, contains queue header and case card selection", () => {
  const listPath = path.resolve(__dirname, "../src/ReviewList.tsx");
  assert.ok(fs.existsSync(listPath), "ReviewList.tsx must exist");

  const content = fs.readFileSync(listPath, "utf-8");
  assert.match(content, /Review Queue/i, "ReviewList must render Review Queue title");
  assert.match(content, /screenings\?status=pending_review/i, "ReviewList must query pending cases");
  assert.match(content, /onSelectCase/i, "ReviewList must allow selecting a case");
});

test("ReviewDetail: exists, has Confirm and Override action buttons and PDF download", () => {
  const detailPath = path.resolve(__dirname, "../src/ReviewDetail.tsx");
  assert.ok(fs.existsSync(detailPath), "ReviewDetail.tsx must exist");

  const content = fs.readFileSync(detailPath, "utf-8");
  assert.match(content, /Confirm/i, "Must have Confirm button");
  assert.match(content, /Override/i, "Must have Override button");
  assert.match(content, /final_grade/i, "Must support final grade selection on override");
  assert.match(content, /report\.pdf/i, "Must support downloading PDF report");
  assert.match(content, /Prototype, not clinically validated/i, "Must render prototype disclaimer");
});

test("App: integrates review navigation and real voice fallback", () => {
  const appPath = path.resolve(__dirname, "../src/App.tsx");
  const content = fs.readFileSync(appPath, "utf-8");

  assert.match(content, /ReviewList/i, "App must import/render ReviewList");
  assert.match(content, /ReviewDetail/i, "App must import/render ReviewDetail");
  assert.match(content, /LoginPage/i, "App must import/render LoginPage");
  assert.match(content, /\/voice/i, "App must call /voice endpoint with browser speech fallback");
  assert.match(content, /Download PDF/i, "App must have Download PDF button");
});
