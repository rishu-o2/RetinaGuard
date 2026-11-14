import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";

import {
  computeBufferSha256,
  runOfflineInference,
} from "../src/offline/inference.js";
import { ModelMetadataConfig } from "../src/offline/rulesEngine.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const ONNX_PATH = path.join(REPO_ROOT, "ml", "exports", "retinaguard_dr_v1.onnx");
const REGISTRY_PATH = path.join(REPO_ROOT, "models", "registry.json");
const VECTORS_PATH = path.join(__dirname, "onnx_parity_20.json");

const MOCK_CONFIG: ModelMetadataConfig = {
  version: "v1.0.0",
  version_tag: "v1",
  onnx_sha256: "7d0c5d8bb1b448d09f48db586f2eff6627da5671e4abcc30c55c165d509c45aa",
  image_size: [256, 256],
  temperature: 1.296247,
  abstention: {
    lower_edge: 0.30,
    upper_edge: 0.70,
    cutoff: 0.50,
  },
  min_acceptable_confidence: 0.70,
  sight_threatening: {
    threshold_soon: 0.05,
    threshold_urgent: 0.30,
  },
  grade_table: {
    "0": { referable: false, base_urgency: "ROUTINE", base_interval_months: 12 },
    "1": { referable: false, base_urgency: "ROUTINE", base_interval_months: 12 },
    "2": { referable: true,  base_urgency: "PRIORITY", base_interval_months: 3 },
    "3": { referable: true,  base_urgency: "URGENT", base_interval_months: 1 },
    "4": { referable: true,  base_urgency: "EMERGENCY", base_interval_months: 0 },
  },
  modifiers: {
    pregnancy: { min_urgency: "PRIORITY", max_interval_months: 3 },
    high_risk_glycemic_control: { hba1c_threshold_percent: 10.0, max_interval_months: 6 },
    long_standing_diabetes: { duration_years_threshold: 15, max_interval_months: 6 },
  },
};

test("Active ONNX model SHA-256 verification against registry.json", async () => {
  assert.ok(fs.existsSync(ONNX_PATH), "ONNX model file must exist");
  assert.ok(fs.existsSync(REGISTRY_PATH), "registry.json must exist");

  const registry = JSON.parse(fs.readFileSync(REGISTRY_PATH, "utf-8"));
  const activeModel = registry.models.find((m: any) => m.active);
  assert.ok(activeModel, "Active model must exist in registry");

  const onnxBytes = fs.readFileSync(ONNX_PATH);
  const computedSha = await computeBufferSha256(onnxBytes.buffer);

  assert.equal(
    computedSha.toLowerCase(),
    activeModel.onnx_sha256.toLowerCase(),
    "Model SHA-256 must match registry.json exactly"
  );
});

test("ONNX browser runtime vs server parity on 20 real validation images (tol < 1e-4)", async () => {
  assert.ok(fs.existsSync(VECTORS_PATH), "onnx_parity_20.json must exist");
  const records = JSON.parse(fs.readFileSync(VECTORS_PATH, "utf-8"));
  assert.equal(records.length, 20, "Must contain exactly 20 real validation images");

  const onnxBytes = fs.readFileSync(ONNX_PATH);
  const session = await ort.InferenceSession.create(new Uint8Array(onnxBytes), {
    executionProviders: ["wasm"],
  });

  const TOLERANCE = 1e-4;

  for (const record of records) {
    const buf = Buffer.from(record.tensor_b64, "base64");
    const inputTensor = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const result = await runOfflineInference(
      session,
      inputTensor,
      MOCK_CONFIG,
      { is_pregnant: false, hba1c: null, diabetes_duration_years: null },
      256
    );

    // Verify calibrated output probabilities match server within 1e-4
    for (let c = 0; c < 5; c++) {
      const serverProb = record.server_probs[c];
      const browserProb = result.probabilities[c];
      const diff = Math.abs(serverProb - browserProb);
      assert.ok(
        diff < TOLERANCE,
        `Sample ${record.id_code} class ${c} probability mismatch: server=${serverProb}, browser=${browserProb}, diff=${diff}`
      );
    }
  }
});
