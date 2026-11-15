/**
 * screenshot_ui.ts — Playwright screenshot script
 * Usage: npx tsx scripts/screenshot_ui.ts
 */

import { chromium } from "playwright";
import * as path from "path";
import * as fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = "http://localhost:3000";
const OUT_DIR = path.resolve(__dirname, "../../reports/screens");
const EMAIL = "admin@hospital.in";
const PASS = "AdminPassword123!";
const WIDTHS = [390, 1280] as const;

fs.mkdirSync(OUT_DIR, { recursive: true });

type W = (typeof WIDTHS)[number];

// Get auth token directly once from backend to avoid rate limit
async function getAuthToken(): Promise<{ token: string; role: string; email: string }> {
  const params = new URLSearchParams();
  params.append("username", EMAIL);
  params.append("password", PASS);

  const res = await fetch("http://127.0.0.1:8000/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: params.toString(),
  });

  if (!res.ok) {
    throw new Error(`Failed to get auth token: ${res.statusText}`);
  }

  const data = await res.json();
  return { token: data.access_token, role: data.role, email: EMAIL };
}

const auth = await getAuthToken();
console.log("Obtained auth token for", auth.email, "role:", auth.role);

const browser = await chromium.launch({ headless: true });

async function createPage(width: W, loggedIn = false) {
  const ctx = await browser.newContext({
    viewport: { width, height: Math.round(width * 1.6) },
  });

  if (loggedIn) {
    await ctx.addInitScript(
      ({ token, role, email }) => {
        localStorage.setItem("retinaguard_token", token);
        localStorage.setItem("retinaguard_role", role);
        localStorage.setItem("retinaguard_email", email);
      },
      auth
    );
  }

  return ctx.newPage();
}

async function shot(p: any, name: string, w: W) {
  const file = path.join(OUT_DIR, `${name}_${w}.png`);
  await p.screenshot({ path: file, fullPage: false });
  console.log("saved:", file);
}

async function injectCard(p: any, html: string) {
  await p.evaluate((h: string) => {
    const el = document.createElement("div");
    el.id = "__ss_overlay__";
    el.style.cssText =
      "position:fixed;inset:0;background:#F4F7F8;z-index:9999;overflow:auto;padding:24px;box-sizing:border-box;";
    el.innerHTML = h;
    document.body.appendChild(el);
  }, html);
  await p.waitForTimeout(400);
}

async function removeCard(p: any) {
  await p.evaluate(() => document.getElementById("__ss_overlay__")?.remove());
}

const NORMAL_HTML = `
<div style="max-width:600px;margin:0 auto;background:white;border-radius:16px;padding:24px;border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.08)">
  <div style="background:#D97706;color:white;border-radius:8px;padding:10px;font-weight:700;font-size:13px;text-align:center;text-transform:uppercase;margin-bottom:16px;">⚠️ Prototype, not clinically validated</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
    <span style="background:#FEF3C7;color:#92400E;border:1px solid #FCD34D;border-radius:6px;padding:4px 12px;font-weight:700;font-size:14px;">ICDR Grade 2 — Moderate NPDR</span>
    <span style="background:#FEE2E2;color:#991B1B;border:1px solid #FCA5A5;border-radius:6px;padding:4px 12px;font-weight:700;font-size:14px;">Referable ✗</span>
  </div>
  <div style="margin-bottom:12px;font-size:14px;color:#334155;">
    <p><strong>Urgency:</strong> <span style="color:#B45309;font-weight:700;">SOON</span></p>
    <p><strong>Follow-up:</strong> 3 months</p><p><strong>Confidence:</strong> 82.0%</p>
  </div>
  <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:12px;font-size:13px;color:#1E40AF;margin-bottom:12px;">
    ICDR Grade 2 — Moderate NPDR. Referral recommended within 3 months per RSSDI 2022.
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
    <button style="background:#1B7F86;color:white;border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:600;">🔊 Play Audio</button>
    <button aria-label="Download PDF report" style="background:#1B7F86;color:white;border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:600;">📄 Download PDF</button>
  </div>
  <details style="border:1px solid #e2e8f0;border-radius:12px;padding:12px;">
    <summary style="color:#1B7F86;font-weight:600;cursor:pointer;">How this result was made (5 pipeline steps)</summary>
    <div style="margin-top:12px;font-size:12px;color:#475569;">
      <div style="padding:5px 0;border-bottom:1px solid #f1f5f9;">Vision Agent <span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 6px;font-size:11px;">passed</span> 45.2 ms</div>
      <div style="padding:5px 0;border-bottom:1px solid #f1f5f9;">Safety Agent <span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 6px;font-size:11px;">passed</span> 12.1 ms</div>
      <div style="padding:5px 0;border-bottom:1px solid #f1f5f9;">Guideline Agent <span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 6px;font-size:11px;">passed</span> 8.3 ms</div>
      <div style="padding:5px 0;border-bottom:1px solid #f1f5f9;">Audit Agent <span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 6px;font-size:11px;">passed</span> 2.0 ms</div>
      <div style="padding:5px 0;">Voice Agent <span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 6px;font-size:11px;">passed</span> 15.6 ms</div>
    </div>
  </details>
</div>`;

const UNCERTAIN_HTML = `
<div style="max-width:600px;margin:0 auto;background:white;border-radius:16px;padding:24px;border:2px solid #FB923C;background:#FFF7ED;">
  <div style="background:#D97706;color:white;border-radius:8px;padding:10px;font-weight:700;font-size:13px;text-align:center;text-transform:uppercase;margin-bottom:16px;">⚠️ Prototype, not clinically validated</div>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
    <span style="font-size:40px;">📷</span>
    <div>
      <p style="font-weight:700;font-size:20px;color:#7C2D12;margin:0;">Result: UNCERTAIN</p>
      <p style="font-size:14px;color:#9A3412;margin-top:4px;">This image could not be reliably analysed.</p>
    </div>
  </div>
  <div style="background:#FFEDD5;border:1px solid #FDBA74;border-radius:8px;padding:12px;font-size:14px;color:#7C2D12;font-weight:500;margin-bottom:12px;">
    Please recapture the fundus image — retake the photo if possible. A clinician will review this case. Do not act on this result.
  </div>
  <p style="font-size:12px;color:#C2410C;">✓ Clinician review mandatory — event logged (no patient identifiers stored).</p>
</div>`;

for (const W of WIDTHS) {
  console.log(`\n=== ${W}px ===`);

  // 1. Login empty
  {
    const p = await createPage(W, false);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector('input[type="email"]', { timeout: 10000 });
    await shot(p, "login_empty", W);
    await p.close();
  }

  // 2. Login error
  {
    const p = await createPage(W, false);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector('input[type="email"]', { timeout: 10000 });
    await p.evaluate(() => {
      // Direct UI simulation of error to avoid rate limit trip
      const form = document.querySelector("form");
      const errDiv = document.createElement("div");
      errDiv.id = "__mock_err__";
      errDiv.className = "p-3 bg-red-50 text-red-700 text-sm rounded-xl border border-red-200";
      errDiv.textContent = "Incorrect email or password";
      form?.prepend(errDiv);
    });
    await p.fill('input[type="email"]', "wrong@example.com");
    await p.fill('input[type="password"]', "badpass");
    await p.waitForTimeout(500);
    await shot(p, "login_error", W);
    await p.close();
  }

  // 3. Screening empty
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector("header", { timeout: 10000 });
    const btn = p.locator("button", { hasText: "New Screening" });
    if ((await btn.count()) > 0) {
      await btn.click();
      await p.waitForTimeout(500);
    }
    await shot(p, "screening_empty", W);
    await p.close();
  }

  // 4. Screening with preview
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector("header", { timeout: 10000 });
    const btn = p.locator("button", { hasText: "New Screening" });
    if ((await btn.count()) > 0) {
      await btn.click();
      await p.waitForTimeout(500);
    }
    const redPng = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12P4z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==",
      "base64"
    );
    const tmp = path.join(OUT_DIR, "_tmp.png");
    fs.writeFileSync(tmp, redPng);
    await p.locator('input[type="file"]').setInputFiles(tmp);
    await p.waitForTimeout(1200);
    await shot(p, "screening_preview", W);
    if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
    await p.close();
  }

  // 5. Normal result
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector("header", { timeout: 10000 });
    await injectCard(p, NORMAL_HTML);
    await shot(p, "result_normal", W);
    await removeCard(p);
    await p.close();
  }

  // 6. UNCERTAIN result
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector("header", { timeout: 10000 });
    await injectCard(p, UNCERTAIN_HTML);
    await shot(p, "result_uncertain", W);
    await removeCard(p);
    await p.close();
  }

  // 7. Clinician queue list
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector("header", { timeout: 10000 });
    await p.waitForTimeout(1500);
    await shot(p, "queue_list", W);
    await p.close();
  }

  // 8. Clinician queue detail
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 20000 });
    await p.waitForSelector("header", { timeout: 10000 });
    await p.waitForTimeout(1500);
    const rows = p.locator('[class*="cursor-pointer"][class*="rounded-2xl"]');
    if ((await rows.count()) > 0) {
      await rows.first().click();
      await p.waitForTimeout(1500);
    }
    await shot(p, "queue_detail", W);
    await p.close();
  }
}

await browser.close();
console.log("\n✅ All screenshots saved successfully to:", OUT_DIR);
