export interface PrecheckThresholds {
  min_width: number;
  min_height: number;
  blur_variance_threshold: number;
  mean_luminance_min: number;
  mean_luminance_max: number;
  non_black_pixel_min_ratio: number;
  red_dominance_required?: boolean;
}

export interface PrecheckResult {
  passed: boolean;
  failures: string[];
  metrics: {
    width: number;
    height: number;
    blur_variance: number;
    mean_luminance: number;
    non_black_ratio: number;
    mean_r: number;
    mean_g: number;
    mean_b: number;
  };
}

/** Check if image dimensions satisfy minimum requirements */
export function checkMinSize(
  width: number,
  height: number,
  minWidth: number,
  minHeight: number
): boolean {
  return width >= minWidth && height >= minHeight;
}

/** Compute RGB means across RGBA buffer */
export function computeRgbMeans(
  rgba: Uint8ClampedArray | Uint8Array
): { r: number; g: number; b: number } {
  const pixelCount = rgba.length / 4;
  if (pixelCount === 0) return { r: 0, g: 0, b: 0 };

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;

  for (let i = 0; i < rgba.length; i += 4) {
    sumR += rgba[i];
    sumG += rgba[i + 1];
    sumB += rgba[i + 2];
  }

  return {
    r: sumR / pixelCount,
    g: sumG / pixelCount,
    b: sumB / pixelCount,
  };
}

/** Check red dominance: mean R > mean G > mean B */
export function checkRedDominance(
  rgba: Uint8ClampedArray | Uint8Array
): boolean {
  const { r, g, b } = computeRgbMeans(rgba);
  return r > g && g > b;
}

/** Compute mean luminance and check whether within [minLum, maxLum] */
export function checkBrightness(
  rgba: Uint8ClampedArray | Uint8Array,
  minLum: number,
  maxLum: number
): { passed: boolean; meanLuminance: number } {
  const pixelCount = rgba.length / 4;
  if (pixelCount === 0) return { passed: false, meanLuminance: 0 };

  let sumLum = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    // Standard ITU-R BT.601 coefficients
    const lum = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    sumLum += lum;
  }
  const meanLum = sumLum / pixelCount;
  return {
    passed: meanLum >= minLum && meanLum <= maxLum,
    meanLuminance: meanLum,
  };
}

/** Check share of non-black pixels (detect valid FOV coverage) */
export function checkNonBlackRatio(
  rgba: Uint8ClampedArray | Uint8Array,
  minRatio: number,
  blackThreshold: number = 15
): { passed: boolean; ratio: number } {
  const pixelCount = rgba.length / 4;
  if (pixelCount === 0) return { passed: false, ratio: 0 };

  let nonBlack = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const lum = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    if (lum > blackThreshold) {
      nonBlack++;
    }
  }
  const ratio = nonBlack / pixelCount;
  return {
    passed: ratio >= minRatio,
    ratio,
  };
}

/** Compute Laplacian variance on grayscale representation of 2D image */
export function checkBlur(
  grayscale: Float32Array | number[] | Uint8Array,
  width: number,
  height: number,
  threshold: number
): { passed: boolean; variance: number } {
  if (width < 3 || height < 3) return { passed: false, variance: 0 };

  const N = (width - 2) * (height - 2);
  const lap: number[] = new Array(N);
  let sum = 0;
  let idx = 0;

  // 3x3 discrete Laplacian kernel:
  // [ 0,  1,  0 ]
  // [ 1, -4,  1 ]
  // [ 0,  1,  0 ]
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const center = grayscale[y * width + x];
      const val =
        grayscale[(y - 1) * width + x] +
        grayscale[(y + 1) * width + x] +
        grayscale[y * width + (x - 1)] +
        grayscale[y * width + (x + 1)] -
        4 * center;
      lap[idx++] = val;
      sum += val;
    }
  }

  const mean = sum / N;
  let varSum = 0;
  for (let i = 0; i < N; i++) {
    const diff = lap[i] - mean;
    varSum += diff * diff;
  }
  const variance = varSum / N;

  return {
    passed: variance >= threshold,
    variance,
  };
}

/** Convert RGBA buffer to grayscale Float32Array */
export function rgbaToGrayscale(
  rgba: Uint8ClampedArray | Uint8Array
): Float32Array {
  const pixelCount = rgba.length / 4;
  const gray = new Float32Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const offset = i * 4;
    gray[i] =
      0.299 * rgba[offset] +
      0.587 * rgba[offset + 1] +
      0.114 * rgba[offset + 2];
  }
  return gray;
}

/** Run the complete pre-check suite against image data and fetched thresholds */
export function runPrecheck(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  thresholds: PrecheckThresholds
): PrecheckResult {
  const failures: string[] = [];

  // 1. Min size
  if (!checkMinSize(width, height, thresholds.min_width, thresholds.min_height)) {
    failures.push(`Resolution too small (${width}x${height} < ${thresholds.min_width}x${thresholds.min_height})`);
  }

  // 2. Brightness
  const brightRes = checkBrightness(
    rgba,
    thresholds.mean_luminance_min,
    thresholds.mean_luminance_max
  );
  if (!brightRes.passed) {
    failures.push(`Brightness out of range (${brightRes.meanLuminance.toFixed(1)})`);
  }

  // 3. Share of non-black pixels
  const nonBlackRes = checkNonBlackRatio(
    rgba,
    thresholds.non_black_pixel_min_ratio
  );
  if (!nonBlackRes.passed) {
    failures.push(`Insufficient non-black pixel area (${(nonBlackRes.ratio * 100).toFixed(1)}%)`);
  }

  // 4. Red dominance
  const rgbMeans = computeRgbMeans(rgba);
  if (thresholds.red_dominance_required !== false) {
    if (!(rgbMeans.r > rgbMeans.g && rgbMeans.g > rgbMeans.b)) {
      failures.push("Image does not exhibit expected retinal red-spectrum dominance");
    }
  }

  // 5. Blur (Laplacian variance on grayscale)
  const gray = rgbaToGrayscale(rgba);
  const blurRes = checkBlur(
    gray,
    width,
    height,
    thresholds.blur_variance_threshold
  );
  if (!blurRes.passed) {
    failures.push(`Image too blurry (variance ${blurRes.variance.toFixed(2)} < ${thresholds.blur_variance_threshold})`);
  }

  return {
    passed: failures.length === 0,
    failures,
    metrics: {
      width,
      height,
      blur_variance: blurRes.variance,
      mean_luminance: brightRes.meanLuminance,
      non_black_ratio: nonBlackRes.ratio,
      mean_r: rgbMeans.r,
      mean_g: rgbMeans.g,
      mean_b: rgbMeans.b,
    },
  };
}
