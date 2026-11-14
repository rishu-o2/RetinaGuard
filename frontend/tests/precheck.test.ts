import test from "node:test";
import assert from "node:assert/strict";

import {
  checkMinSize,
  checkBrightness,
  checkNonBlackRatio,
  checkRedDominance,
  checkBlur,
  runPrecheck,
  PrecheckThresholds,
} from "../src/precheck.js";

const DEFAULT_THRESHOLDS: PrecheckThresholds = {
  min_width: 256,
  min_height: 256,
  blur_variance_threshold: 12.0,
  mean_luminance_min: 30.0,
  mean_luminance_max: 200.0,
  non_black_pixel_min_ratio: 0.35,
  red_dominance_required: true,
};

function createSyntheticImage(
  width: number,
  height: number,
  type: "clear_retina" | "blurry" | "dark" | "green_non_red"
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const cx = width / 2;
  const cy = height / 2;
  const rCircle = Math.min(width, height) * 0.45;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      const inCircle = dist <= rCircle;

      if (type === "clear_retina") {
        if (inCircle) {
          // Sharp variations (high Laplacian variance) + Red > Green > Blue
          const highFreq = ((x % 4) ^ (y % 4)) * 10;
          rgba[idx] = 180 + highFreq; // R
          rgba[idx + 1] = 60 + (highFreq > 10 ? 5 : 0); // G
          rgba[idx + 2] = 20; // B
          rgba[idx + 3] = 255;
        } else {
          // Black background
          rgba[idx] = 0;
          rgba[idx + 1] = 0;
          rgba[idx + 2] = 0;
          rgba[idx + 3] = 255;
        }
      } else if (type === "blurry") {
        // Completely flat/smooth tone without sharp edges -> very low Laplacian variance
        rgba[idx] = 120;
        rgba[idx + 1] = 50;
        rgba[idx + 2] = 20;
        rgba[idx + 3] = 255;
      } else if (type === "dark") {
        // Very low luminance across entire image
        rgba[idx] = 5;
        rgba[idx + 1] = 2;
        rgba[idx + 2] = 1;
        rgba[idx + 3] = 255;
      } else if (type === "green_non_red") {
        // Green dominant image (not retinal fundus)
        if (inCircle) {
          const highFreq = ((x % 4) ^ (y % 4)) * 10;
          rgba[idx] = 30; // R
          rgba[idx + 1] = 180 + highFreq; // G
          rgba[idx + 2] = 40; // B
          rgba[idx + 3] = 255;
        } else {
          rgba[idx] = 0;
          rgba[idx + 1] = 0;
          rgba[idx + 2] = 0;
          rgba[idx + 3] = 255;
        }
      }
    }
  }

  return rgba;
}

test("checkMinSize evaluates dimensions correctly", () => {
  assert.equal(checkMinSize(256, 256, 256, 256), true);
  assert.equal(checkMinSize(300, 300, 256, 256), true);
  assert.equal(checkMinSize(200, 256, 256, 256), false);
  assert.equal(checkMinSize(256, 180, 256, 256), false);
});

test("checkRedDominance identifies retinal spectra vs non-red", () => {
  const retina = createSyntheticImage(256, 256, "clear_retina");
  assert.equal(checkRedDominance(retina), true);

  const green = createSyntheticImage(256, 256, "green_non_red");
  assert.equal(checkRedDominance(green), false);
});

test("checkBrightness detects valid vs dark images", () => {
  const retina = createSyntheticImage(256, 256, "clear_retina");
  const retinaRes = checkBrightness(retina, 30, 200);
  assert.equal(retinaRes.passed, true);
  assert.ok(retinaRes.meanLuminance >= 30 && retinaRes.meanLuminance <= 200);

  const dark = createSyntheticImage(256, 256, "dark");
  const darkRes = checkBrightness(dark, 30, 200);
  assert.equal(darkRes.passed, false);
});

test("checkBlur differentiates sharp from flat/blurry images", () => {
  const retina = createSyntheticImage(256, 256, "clear_retina");
  const blurRes1 = runPrecheck(retina, 256, 256, DEFAULT_THRESHOLDS);
  assert.equal(blurRes1.metrics.blur_variance >= DEFAULT_THRESHOLDS.blur_variance_threshold, true);

  const blurry = createSyntheticImage(256, 256, "blurry");
  const blurRes2 = runPrecheck(blurry, 256, 256, DEFAULT_THRESHOLDS);
  assert.equal(blurRes2.passed, false);
  assert.ok(blurRes2.failures.some((f) => f.includes("blurry")));
});

test("runPrecheck passes clear retina and rejects flawed variants", () => {
  // 1. Clear retina passes all gates
  const clear = createSyntheticImage(256, 256, "clear_retina");
  const resClear = runPrecheck(clear, 256, 256, DEFAULT_THRESHOLDS);
  assert.equal(resClear.passed, true);
  assert.equal(resClear.failures.length, 0);

  // 2. Dark image fails brightness and non-black ratio
  const dark = createSyntheticImage(256, 256, "dark");
  const resDark = runPrecheck(dark, 256, 256, DEFAULT_THRESHOLDS);
  assert.equal(resDark.passed, false);
  assert.ok(resDark.failures.length > 0);

  // 3. Green image fails red dominance
  const green = createSyntheticImage(256, 256, "green_non_red");
  const resGreen = runPrecheck(green, 256, 256, DEFAULT_THRESHOLDS);
  assert.equal(resGreen.passed, false);
  assert.ok(resGreen.failures.some((f) => f.includes("red-spectrum")));

  // 4. Low resolution fails size
  const resSmall = runPrecheck(clear, 180, 180, DEFAULT_THRESHOLDS);
  assert.equal(resSmall.passed, false);
  assert.ok(resSmall.failures.some((f) => f.includes("Resolution too small")));
});
