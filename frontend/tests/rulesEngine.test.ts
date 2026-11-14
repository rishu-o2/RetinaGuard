import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  evaluateRules,
  applyTemperatureCalibration,
  softmax,
  ModelMetadataConfig,
} from "../src/offline/rulesEngine.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Model configuration matching backend metadata exactly
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
    pregnancy: {
      min_urgency: "PRIORITY",
      max_interval_months: 3,
    },
    high_risk_glycemic_control: {
      hba1c_threshold_percent: 10.0,
      max_interval_months: 6,
    },
    long_standing_diabetes: {
      duration_years_threshold: 15,
      max_interval_months: 6,
    },
  },
};

test("Temperature scaling and softmax numerical properties", () => {
  const logits = [2.0, 1.0, 0.0, -1.0, -2.0];
  const sm = softmax(logits);
  const sum = sm.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1.0) < 1e-6);

  // Scaled with T=1.0 is identical to uncalibrated softmax
  const calibratedT1 = applyTemperatureCalibration(logits, 1.0);
  for (let i = 0; i < logits.length; i++) {
    assert.ok(Math.abs(sm[i] - calibratedT1[i]) < 1e-6);
  }

  // With higher temperature, entropy increases (max probability drops)
  const calibratedT2 = applyTemperatureCalibration(logits, 2.0);
  assert.ok(Math.max(...calibratedT2) < Math.max(...sm));
});

test("Golden Vectors Parity Test (540+ combinations from Python engine)", () => {
  const vectorsPath = path.join(__dirname, "golden_vectors.json");
  assert.ok(fs.existsSync(vectorsPath), "golden_vectors.json must exist");

  const raw = fs.readFileSync(vectorsPath, "utf-8");
  const vectors = JSON.parse(raw);

  assert.ok(vectors.length >= 300, `Expected at least 300 vectors, got ${vectors.length}`);

  let mismatches = 0;
  for (const v of vectors) {
    const out = evaluateRules(v.probabilities, v.context, MOCK_CONFIG);

    try {
      assert.equal(
        out.referral,
        v.expected.referral,
        `Vector #${v.id} referral mismatch: got ${out.referral}, expected ${v.expected.referral}`
      );
      assert.equal(
        out.urgency,
        v.expected.urgency,
        `Vector #${v.id} urgency mismatch: got ${out.urgency}, expected ${v.expected.urgency}`
      );
      assert.equal(
        out.follow_up_interval_months,
        v.expected.follow_up_interval_months,
        `Vector #${v.id} interval mismatch: got ${out.follow_up_interval_months}, expected ${v.expected.follow_up_interval_months}`
      );
      assert.equal(
        out.is_uncertain,
        v.expected.is_uncertain,
        `Vector #${v.id} uncertainty mismatch: got ${out.is_uncertain}, expected ${v.expected.is_uncertain}`
      );
    } catch (err) {
      mismatches++;
      throw err;
    }
  }

  assert.equal(mismatches, 0, "All golden parity vectors must pass with zero mismatches");
});
