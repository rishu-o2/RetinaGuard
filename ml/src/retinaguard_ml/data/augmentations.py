"""Fundus-specific data augmentations and Indian smartphone/tabletop camera simulation.

Implements ADR-009:
- Non-destructive clinical augmentations preserving lesion colors and vascular details.
- Camera-simulating augmentations: brightness/contrast/gamma, hue/saturation/white-balance,
  blur/sharpness, JPEG compression, vignetting/uneven illumination, flips, and rotations.
- Every transformation is individually configurable, tested, and applied ONLY to the train split.
"""

from __future__ import annotations

import io
from typing import Tuple
import cv2  # type: ignore
import numpy as np


class FundusCameraAugmentor:
    """Configurable camera-simulating augmentation pipeline for retinal fundus imaging.

    Simulates optical and sensor variations observed across screening cameras in Indian
    PHCs, vision centres, and tertiary eye hospitals (Zeiss, Topcon, Forus 3nethra, Remidio NM-FOP):
      1. Geometric: horizontal/vertical flips, 360-degree rotations
      2. Photometric: brightness, contrast, and non-linear gamma curves
      3. Chromatic: subtle hue shift, saturation changes, warm/cool white-balance bias
      4. Optical: Gaussian / defocus blur and unsharp-mask sharpness enhancements
      5. Compression: lossy JPEG block artifacts from mobile storage
      6. Illumination: radial vignetting & decentered spotlight falloff from LED flash
    """

    def __init__(
        self,
        # Geometric
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rotate: float = 0.7,
        # Photometric (brightness, contrast, gamma)
        p_brightness_contrast: float = 0.5,
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        brightness_range: Tuple[float, float] = (-15.0, 15.0),
        p_gamma: float = 0.4,
        gamma_range: Tuple[float, float] = (0.85, 1.20),
        # Chromatic (hue, saturation, white-balance)
        p_hue_saturation: float = 0.4,
        hue_shift_limit: int = 8,  # Small: preserves red microaneurysms / hemorrhages
        sat_shift_limit: int = 18,
        p_white_balance: float = 0.4,
        wb_temp_range: Tuple[float, float] = (-10.0, 12.0),
        # Optical (blur & sharpness)
        p_blur: float = 0.35,
        blur_kernel_choices: Tuple[int, ...] = (3, 5),
        p_sharpness: float = 0.35,
        sharpness_strength_range: Tuple[float, float] = (0.3, 0.8),
        # Sensor / Storage Compression
        p_jpeg: float = 0.4,
        jpeg_quality_range: Tuple[int, int] = (50, 85),
        # Illumination / Vignetting
        p_vignetting: float = 0.4,
        vignette_falloff_range: Tuple[float, float] = (0.20, 0.45),
        seed: int | None = None,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rotate = p_rotate

        self.p_brightness_contrast = p_brightness_contrast
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range

        self.p_hue_saturation = p_hue_saturation
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.p_white_balance = p_white_balance
        self.wb_temp_range = wb_temp_range

        self.p_blur = p_blur
        self.blur_kernel_choices = blur_kernel_choices
        self.p_sharpness = p_sharpness
        self.sharpness_strength_range = sharpness_strength_range

        self.p_jpeg = p_jpeg
        self.jpeg_quality_range = jpeg_quality_range

        self.p_vignetting = p_vignetting
        self.vignette_falloff_range = vignette_falloff_range

        self.rng = np.random.default_rng(seed)

    # ── 1. Geometric transforms ──────────────────────────────────────────────

    def apply_flip(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self.rng.random() < self.p_hflip:
            out = cv2.flip(out, 1)
        if self.rng.random() < self.p_vflip:
            out = cv2.flip(out, 0)
        return out

    def apply_rotation(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_rotate:
            return img
        angle = float(self.rng.uniform(0, 360))
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

    # ── 2. Photometric (Brightness, Contrast, Gamma) ─────────────────────────

    def apply_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_brightness_contrast:
            return img
        alpha = float(self.rng.uniform(*self.contrast_range))
        beta = float(self.rng.uniform(*self.brightness_range))
        return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    def apply_gamma(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_gamma:
            return img
        gamma = float(self.rng.uniform(*self.gamma_range))
        inv_gamma = 1.0 / max(gamma, 1e-5)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    # ── 3. Chromatic (Hue, Saturation, White Balance) ────────────────────────

    def apply_hue_saturation(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_hue_saturation:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        h_shift = int(self.rng.integers(-self.hue_shift_limit, self.hue_shift_limit + 1))
        s_shift = int(self.rng.integers(-self.sat_shift_limit, self.sat_shift_limit + 1))

        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def apply_white_balance(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_white_balance:
            return img
        out = img.astype(np.float32)
        shift = float(self.rng.uniform(*self.wb_temp_range))
        # Warm shift: boost red, slightly reduce blue; cool shift: inverse
        out[:, :, 0] = np.clip(out[:, :, 0] + shift, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] - (shift * 0.75), 0, 255)
        return out.astype(np.uint8)

    # ── 4. Optical (Blur & Sharpness) ────────────────────────────────────────

    def apply_blur(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_blur:
            return img
        k = int(self.rng.choice(self.blur_kernel_choices))
        return cv2.GaussianBlur(img, (k, k), 0)

    def apply_sharpness(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_sharpness:
            return img
        strength = float(self.rng.uniform(*self.sharpness_strength_range))
        # Unsharp masking: sharpened = original + strength * (original - blurred)
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── 5. Compression (Lossy JPEG) ──────────────────────────────────────────

    def apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_jpeg:
            return img
        quality = int(self.rng.integers(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return img
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    # ── 6. Illumination / Vignetting ─────────────────────────────────────────

    def apply_vignetting(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p_vignetting:
            return img
        h, w = img.shape[:2]
        cx = int(w * self.rng.uniform(0.42, 0.58))
        cy = int(h * self.rng.uniform(0.42, 0.58))

        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        max_dist_sq = (w // 2) ** 2 + (h // 2) ** 2

        strength = float(self.rng.uniform(*self.vignette_falloff_range))
        mask = 1.0 - (strength * (dist_sq / max_dist_sq))
        mask = np.clip(mask, 0.50, 1.02)[:, :, np.newaxis]
        return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    # ── Master Pipeline ──────────────────────────────────────────────────────

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        out = img_rgb.copy()
        out = self.apply_flip(out)
        out = self.apply_rotation(out)
        out = self.apply_brightness_contrast(out)
        out = self.apply_gamma(out)
        out = self.apply_hue_saturation(out)
        out = self.apply_white_balance(out)
        out = self.apply_blur(out)
        out = self.apply_sharpness(out)
        out = self.apply_vignetting(out)
        out = self.apply_jpeg_compression(out)
        return out


class SmartphoneDegradationPipeline:
    """Simulates portable / smartphone fundus camera optical artifacts (ADR-009).

    Emulates non-mydriatic handheld cameras used in Indian camps & PHCs:
    - Reduced optical resolution
    - Smartphone JPEG block compression
    - Uneven LED flash illumination & peripheral vignetting
    - Mild handheld motion or focus blur
    - Minor warm/cool white balance color cast
    """

    def __init__(
        self,
        p: float = 0.5,
        jpeg_quality_range: Tuple[int, int] = (45, 75),
        downscale_range: Tuple[float, float] = (0.45, 0.75),
        seed: int | None = None,
    ):
        self.p = p
        self.jpeg_quality_range = jpeg_quality_range
        self.downscale_range = downscale_range
        self.rng = np.random.default_rng(seed)

    def apply_resolution_loss(self, img: np.ndarray) -> np.ndarray:
        scale = float(self.rng.uniform(*self.downscale_range))
        h, w = img.shape[:2]
        small_w = max(int(w * scale), 32)
        small_h = max(int(h * scale), 32)
        downscaled = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)

    def apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        quality = int(self.rng.integers(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return img
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def apply_uneven_illumination(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        cx = int(w * self.rng.uniform(0.40, 0.60))
        cy = int(h * self.rng.uniform(0.40, 0.60))
        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        max_dist_sq = (w // 2) ** 2 + (h // 2) ** 2
        falloff_strength = float(self.rng.uniform(0.25, 0.55))
        gradient = 1.0 - (falloff_strength * (dist_sq / max_dist_sq))
        gradient = np.clip(gradient, 0.40, 1.05)[:, :, np.newaxis]
        return np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    def apply_optical_blur(self, img: np.ndarray) -> np.ndarray:
        ksize = int(self.rng.choice([3, 5]))
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def apply_color_cast(self, img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32)
        r_shift = float(self.rng.uniform(-8.0, 15.0))
        g_shift = float(self.rng.uniform(-5.0, 8.0))
        b_shift = float(self.rng.uniform(-10.0, 5.0))
        out[:, :, 0] += r_shift
        out[:, :, 1] += g_shift
        out[:, :, 2] += b_shift
        return np.clip(out, 0, 255).astype(np.uint8)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.p:
            return img_rgb
        out = img_rgb.copy()
        out = self.apply_resolution_loss(out)
        out = self.apply_jpeg_compression(out)
        out = self.apply_uneven_illumination(out)
        out = self.apply_optical_blur(out)
        out = self.apply_color_cast(out)
        return out


# Backward compatibility alias
FundusClinicalAugmentor = FundusCameraAugmentor


