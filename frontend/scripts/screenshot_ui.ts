/**
 * Playwright screenshot script for the RetinaGuard two-panel layout.
 * Captures: login, idle, analysing, result_normal, result_uncertain,
 *           queue_list, queue_detail — at 390px and 1280px.
 *
 * Run from d:\RetinaGuard\frontend after both servers are up:
 *   npx tsx scripts/screenshot_ui.ts
 */
import { chromium } from "playwright";
import * as path from "path";
import * as fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = "http://localhost:3000";
const BACKEND  = "http://127.0.0.1:8001";
const OUT_DIR  = path.resolve(__dirname, "../../reports/screens");
const EMAIL    = "admin@hospital.in";
const PASS     = "AdminPassword123!";
const WIDTHS   = [390, 1280] as const;
type W = (typeof WIDTHS)[number];

fs.mkdirSync(OUT_DIR, { recursive: true });

// ── Auth ──────────────────────────────────────────────────────────────────────
async function getAuthToken() {
  const params = new URLSearchParams();
  params.append("username", EMAIL);
  params.append("password", PASS);
  const res = await fetch(`${BACKEND}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: params.toString(),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Auth failed ${res.status}: ${body}`);
  }
  const data = await res.json();
  return { token: data.access_token, role: data.role, email: EMAIL };
}

const auth = await getAuthToken();
console.log(`✓ Auth token for ${auth.email} (${auth.role})`);

const browser = await chromium.launch({ headless: true });

// ── Page factory ──────────────────────────────────────────────────────────────
async function createPage(w: W, loggedIn = false) {
  const ctx = await browser.newContext({
    viewport: { width: w, height: Math.round(w * 1.65) },
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
  console.log("  saved:", path.basename(file));
  return file;
}

// ── Overlay helpers ───────────────────────────────────────────────────────────
async function injectOverlay(p: any, html: string) {
  await p.evaluate((h: string) => {
    const el = document.createElement("div");
    el.id = "__ss_overlay__";
    el.style.cssText =
      "position:fixed;inset:0;background:#020612;z-index:9999;overflow:auto;padding:28px;box-sizing:border-box;display:flex;align-items:flex-start;justify-content:center;";
    el.innerHTML = h;
    document.body.appendChild(el);
  }, html);
  await p.waitForTimeout(500);
}

async function removeOverlay(p: any) {
  await p.evaluate(() => document.getElementById("__ss_overlay__")?.remove());
}

// ── Static HTML stubs (mirror real component HTML structure for screenshots) ──

const NORMAL_HTML = `
<div style="max-width:640px;width:100%;background:#071120;border-radius:20px;padding:24px;border:1px solid #1e293b;box-shadow:0 20px 50px rgba(0,0,0,.6);font-family:system-ui,sans-serif;color:#f1f5f9;">
  <div style="background:rgba(217,119,6,0.15);border:1px solid rgba(245,158,11,0.3);color:#fcd34d;border-radius:10px;padding:6px 12px;font-weight:700;font-size:11px;text-align:center;text-transform:uppercase;margin-bottom:16px;letter-spacing:.05em;">⚠️ Prototype, not clinically validated</div>
  
  <div style="background:#d97706;color:white;border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 4px 14px rgba(217,119,6,0.3);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <span style="font-size:14px;font-weight:900;text-transform:uppercase;letter-spacing:.08em;">SOON</span>
      <span style="font-size:10px;font-weight:700;text-transform:uppercase;opacity:.85;">Referral Required</span>
    </div>
    <h3 style="margin:0 0 8px 0;font-size:17px;font-weight:800;letter-spacing:-.01em;">Referable DR: see an eye specialist within 1 month</h3>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
      <span style="background:rgba(0,0,0,0.35);backdrop-filter:blur(4px);border:1px solid rgba(255,255,255,0.2);border-radius:6px;padding:3px 10px;font-weight:700;font-size:12px;">ICDR Grade 2 — Moderate NPDR</span>
      <span style="background:rgba(153,27,27,0.7);border:1px solid #f87171;border-radius:6px;padding:3px 10px;font-weight:700;font-size:12px;color:#fecaca;">Referable ✗</span>
    </div>
    <p style="margin:0;font-size:12px;opacity:.95;line-height:1.5;">Diagnostic AI estimation indicates an 82.0% confidence level for this assessment.</p>
  </div>

  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px;margin-bottom:14px;">
    <p style="margin:0 0 10px 0;font-size:13px;color:#cbd5e1;font-style:italic;line-height:1.6;">"Findings indicate moderate NPDR with microaneurysms. Clinical practice guidelines recommend ophthalmology referral within 1 to 3 months."</p>
    <button style="background:#14b8a6;color:#020612;border:none;border-radius:8px;padding:8px 16px;font-size:12px;font-weight:800;cursor:pointer;display:inline-flex;align-items:center;gap:6px;">🔊 Play audio</button>
  </div>

  <div style="display:flex;justify-content:space-between;align-items:center;padding:0 4px;font-family:monospace;font-size:11px;color:#94a3b8;margin-bottom:12px;">
    <span>English advisory</span>
    <span style="background:#1e293b;color:#e2e8f0;padding:2px 8px;border-radius:999px;font-size:10px;border:1px solid #334155;">Sarvam AI Bulbul:v3</span>
  </div>

  <div style="text-align:center;padding-top:4px;">
    <button style="background:none;border:none;color:#2dd4bf;font-size:12px;font-weight:600;text-decoration:underline;cursor:pointer;">Screen another photo</button>
  </div>
</div>`;

const UNCERTAIN_HTML = `
<div style="max-width:640px;width:100%;background:#071120;border-radius:20px;padding:24px;border:2px solid #ea580c;box-shadow:0 20px 50px rgba(0,0,0,.6);font-family:system-ui,sans-serif;color:#f1f5f9;">
  <div style="background:rgba(217,119,6,0.15);border:1px solid rgba(245,158,11,0.3);color:#fcd34d;border-radius:10px;padding:6px 12px;font-weight:700;font-size:11px;text-align:center;text-transform:uppercase;margin-bottom:16px;letter-spacing:.05em;">⚠️ Prototype, not clinically validated</div>
  
  <div style="background:#0f172a;border:1px solid #334155;border-radius:14px;padding:16px;margin-bottom:14px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span role="img" aria-label="camera" style="font-size:32px;">📷</span>
      <div>
        <h4 style="margin:0;font-size:16px;font-weight:800;color:#f87171;letter-spacing:-.01em;">NON-ACTIONABLE: Clinician Review Required</h4>
        <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-top:2px;">Result: UNCERTAIN</div>
      </div>
    </div>
    <div style="margin-top:12px;">
      <span style="background:rgba(153,27,27,0.4);border:1px solid rgba(248,113,113,0.4);color:#fecaca;border-radius:6px;padding:4px 10px;font-size:11px;font-weight:600;">Photo too blurry (Laplacian variance 8.4 < 12.21)</span>
    </div>
    <p style="margin:12px 0 0 0;font-size:12px;color:#cbd5e1;line-height:1.6;">Retake the photo if possible. A clinician will review this case. Do not act on this result.</p>
  </div>

  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px;margin-bottom:14px;">
    <p style="margin:0 0 10px 0;font-size:13px;color:#cbd5e1;font-style:italic;line-height:1.6;">"The fundus image quality could not be verified. Please capture a clear, focused photograph in reduced ambient lighting."</p>
    <button style="background:#14b8a6;color:#020612;border:none;border-radius:8px;padding:8px 16px;font-size:12px;font-weight:800;cursor:pointer;display:inline-flex;align-items:center;gap:6px;">🔊 Play audio</button>
  </div>

  <div style="display:flex;justify-content:space-between;align-items:center;padding:0 4px;font-family:monospace;font-size:11px;color:#94a3b8;margin-bottom:12px;">
    <span>English advisory</span>
    <span style="background:#1e293b;color:#e2e8f0;padding:2px 8px;border-radius:999px;font-size:10px;border:1px solid #334155;">Browser Web Speech API</span>
  </div>

  <div style="text-align:center;padding-top:4px;">
    <button style="background:none;border:none;color:#2dd4bf;font-size:12px;font-weight:600;text-decoration:underline;cursor:pointer;">Screen another photo</button>
  </div>
</div>`;

// ── Screenshot loop ───────────────────────────────────────────────────────────
const files: string[] = [];

for (const W of WIDTHS) {
  console.log(`\n=== ${W}px ===`);

  // 1. Login – empty
  {
    const p = await createPage(W, false);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    await p.waitForSelector('input[type="email"]', { timeout: 12000 });
    files.push(await shot(p, "login", W));
    await p.close();
  }

  // 2. Login – with error (inject mock error into DOM; avoids rate-limit trip)
  {
    const p = await createPage(W, false);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    await p.waitForSelector('input[type="email"]', { timeout: 12000 });
    await p.fill('input[type="email"]', "wrong@example.com");
    await p.fill('input[type="password"]', "badpass");
    await p.evaluate(() => {
      const form = document.querySelector("form");
      const errDiv = document.createElement("div");
      errDiv.style.cssText =
        "padding:10px 14px;background:#FEF2F2;color:#991B1B;font-size:13px;border-radius:8px;border:1px solid #FECACA;margin-bottom:8px;";
      errDiv.textContent = "Incorrect email or password";
      form?.prepend(errDiv);
    });
    await p.waitForTimeout(400);
    files.push(await shot(p, "login_error", W));
    await p.close();
  }

  // 3. Idle screener (two-panel layout – left panel shows consent + buttons)
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    // Wait for the main layout to appear (header or main panel)
    await p.waitForSelector("header, main, [class*='grid']", { timeout: 12000 });
    await p.waitForTimeout(1200);
    files.push(await shot(p, "idle", W));
    await p.close();
  }

  // 4. Analysing (simulate the step-list by clicking consent + injecting analysing state)
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    await p.waitForSelector("header, main", { timeout: 12000 });
    await p.waitForTimeout(800);

    // Try to check consent checkbox if present to enable buttons
    const checkbox = p.locator('input[type="checkbox"]').first();
    if ((await checkbox.count()) > 0) {
      await checkbox.check().catch(() => {});
    }

    // Inject "analysing" overlay using a mini step-progress panel matching dark theme
    await p.evaluate(() => {
      const overlay = document.createElement("div");
      overlay.id = "__analysing_overlay__";
      overlay.style.cssText =
        "position:fixed;inset:0;background:#020612;z-index:9999;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif;";
      overlay.innerHTML = `
        <div style="max-width:520px;width:90%;background:#071120;border-radius:20px;padding:28px;border:1px solid #1e293b;box-shadow:0 20px 50px rgba(0,0,0,.7);color:#f1f5f9;">
          <div style="text-align:center;margin-bottom:20px;">
            <div style="width:64px;height:64px;border-radius:50%;background:#0e2a3b;border:2px solid #14b8a6;margin:0 auto 12px;display:flex;align-items:center;justify-content:center;font-size:28px;">
              👁️
            </div>
            <p style="font-size:12px;color:#94a3b8;font-family:monospace;">Preview — RG-284731 (Pre-checked ✓)</p>
          </div>
          <p style="font-weight:800;font-size:16px;color:#2dd4bf;margin-bottom:16px;text-align:center;letter-spacing:-.01em;">Analysing retinal image…</p>
          <div style="space-y:8px;">
            ${["Quality check","Retina check","Grading","Safety rules","Audit","Voice"].map((step, i) => `
              <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #1e293b;">
                <span style="width:22px;height:22px;border-radius:50%;background:${i < 2 ? 'rgba(16,185,129,0.2)' : i === 2 ? 'rgba(245,158,11,0.2)' : 'rgba(30,41,59,0.5)'};border:1px solid ${i < 2 ? '#10b981' : i === 2 ? '#f59e0b' : '#334155'};display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:${i < 2 ? '#34d399' : i === 2 ? '#fbbf24' : '#64748b'};">${i < 2 ? '✓' : i === 2 ? '…' : '·'}</span>
                <span style="font-size:13px;color:${i < 2 ? '#e2e8f0' : i === 2 ? '#2dd4bf' : '#64748b'};font-weight:${i === 2 ? '700' : '500'};">${step}</span>
              </div>
            `).join("")}
          </div>
        </div>`;
      document.body.appendChild(overlay);
    });
    await p.waitForTimeout(400);
    files.push(await shot(p, "analysing", W));
    await p.close();
  }

  // 5. Normal result (inject HTML stub)
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    await p.waitForSelector("header, main", { timeout: 12000 });
    await injectOverlay(p, NORMAL_HTML);
    files.push(await shot(p, "result_normal", W));
    await removeOverlay(p);
    await p.close();
  }

  // 6. UNCERTAIN result (inject HTML stub)
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    await p.waitForSelector("header, main", { timeout: 12000 });
    await injectOverlay(p, UNCERTAIN_HTML);
    files.push(await shot(p, "result_uncertain", W));
    await removeOverlay(p);
    await p.close();
  }

  // 7. Clinician queue list
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    await p.waitForSelector("header, main", { timeout: 12000 });
    const reviewBtn = p.locator("button", { hasText: "Review" }).first();
    if ((await reviewBtn.count()) > 0) {
      await reviewBtn.click();
      await p.waitForTimeout(1500);
    }
    files.push(await shot(p, "queue_list", W));
    await p.close();
  }

  // 8. Clinician queue detail
  {
    const p = await createPage(W, true);
    await p.goto(BASE_URL, { waitUntil: "networkidle", timeout: 25000 });
    await p.waitForSelector("header, main", { timeout: 12000 });
    const reviewBtn = p.locator("button", { hasText: "Review" }).first();
    if ((await reviewBtn.count()) > 0) {
      await reviewBtn.click();
      await p.waitForTimeout(1500);
    }

    // Click first case item
    const row = p.locator('[class*="cursor-pointer"][class*="rounded-xl"]').first();
    if ((await row.count()) > 0) {
      await row.click().catch(() => {});
      await p.waitForTimeout(1500);
    }
    files.push(await shot(p, "queue_detail", W));
    await p.close();
  }
}

await browser.close();

console.log(`\n✅ ${files.length} screenshots saved to: ${OUT_DIR}`);
files.forEach((f) => console.log("  ", f));
