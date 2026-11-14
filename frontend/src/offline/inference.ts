/**
 * Offline ONNX Runtime Inference in the browser.
 * Uses onnxruntime-web to execute the active exported model locally.
 * Preprocessing matches the server-side Python implementation exactly:
 * - Resize to 256x256
 * - Normalize by ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
 * - NCHW Float32Array format
 */

import * as ort from "onnxruntime-web";
import {
  ModelMetadataConfig,
  applyTemperatureCalibration,
  evaluateRules,
  PatientContextInput,
  RulesEvaluationOutput,
} from "./rulesEngine.js";

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

export interface OfflineInferenceOutput {
  predicted_grade: number;
  confidence: number;
  probabilities: number[];
  categorical_logits: number[];
  ordinal_logits?: number[];
  decision: RulesEvaluationOutput;
}

/** Preprocess RGBA image buffer (width x height) to NCHW Float32Array (1 x 3 x targetW x targetH) */
export function preprocessRgbaToNCHW(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  targetSize: number = 256
): Float32Array {
  // If already targetSize x targetSize, proceed directly.
  // Otherwise use nearest/bilinear sampling to targetSize.
  const tensor = new Float32Array(3 * targetSize * targetSize);
  const planeSize = targetSize * targetSize;

  const scaleX = width / targetSize;
  const scaleY = height / targetSize;

  for (let y = 0; y < targetSize; y++) {
    const srcY = Math.min(Math.floor(y * scaleY), height - 1);
    for (let x = 0; x < targetSize; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), width - 1);
      const srcOffset = (srcY * width + srcX) * 4;
      const dstOffset = y * targetSize + x;

      const r = rgba[srcOffset] / 255.0;
      const g = rgba[srcOffset + 1] / 255.0;
      const b = rgba[srcOffset + 2] / 255.0;

      // Channel 0: R
      tensor[dstOffset] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      // Channel 1: G
      tensor[planeSize + dstOffset] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      // Channel 2: B
      tensor[2 * planeSize + dstOffset] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }

  return tensor;
}

/** Compute SHA-256 of an ArrayBuffer in hex string format */
export async function computeBufferSha256(buffer: ArrayBuffer): Promise<string> {
  if (typeof crypto !== "undefined" && crypto.subtle) {
    const hashBuf = await crypto.subtle.digest("SHA-256", buffer);
    const hashArr = Array.from(new Uint8Array(hashBuf));
    return hashArr.map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  // Node.js fallback for unit test harnesses
  try {
    const nodeCrypto = await import("node:crypto");
    return nodeCrypto.createHash("sha256").update(Buffer.from(buffer)).digest("hex");
  } catch {
    throw new Error("No cryptographic digest provider available.");
  }
}

/** Execute offline inference using an onnxruntime-web session */
export async function runOfflineInference(
  session: ort.InferenceSession,
  inputTensorData: Float32Array,
  metadataConfig: ModelMetadataConfig,
  patientContext: PatientContextInput,
  targetSize: number = 256
): Promise<OfflineInferenceOutput> {
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor("float32", inputTensorData, [1, 3, targetSize, targetSize]);

  const feeds: Record<string, ort.Tensor> = {};
  feeds[inputName] = tensor;

  const results = await session.run(feeds);
  const outputNames = session.outputNames;

  const catLogitsTensor = results[outputNames[0]];
  const catLogits = Array.from(catLogitsTensor.data as Float32Array);

  // Calibrate with temperature
  const calibratedProbs = applyTemperatureCalibration(
    catLogits,
    metadataConfig.temperature || 1.0
  );

  let predictedGrade = 0;
  let ordLogits: number[] | undefined;

  // Check if second output is ordinal head (1, 4)
  if (outputNames.length > 1) {
    const ordTensor = results[outputNames[1]];
    if (ordTensor && ordTensor.data.length === 4) {
      ordLogits = Array.from(ordTensor.data as Float32Array);
      let ordSum = 0;
      for (const logit of ordLogits) {
        ordSum += 1.0 / (1.0 + Math.exp(-logit));
      }
      predictedGrade = Math.max(0, Math.min(4, Math.round(ordSum)));
    }
  }

  if (ordLogits === undefined) {
    let maxP = calibratedProbs[0];
    for (let i = 1; i < calibratedProbs.length; i++) {
      if (calibratedProbs[i] > maxP) {
        maxP = calibratedProbs[i];
        predictedGrade = i;
      }
    }
  }

  const confidence = calibratedProbs[predictedGrade];

  // Evaluate clinical rules engine deterministically
  const decision = evaluateRules(calibratedProbs, patientContext, metadataConfig);

  return {
    predicted_grade: predictedGrade,
    confidence,
    probabilities: calibratedProbs,
    categorical_logits: catLogits,
    ordinal_logits: ordLogits,
    decision,
  };
}
