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

