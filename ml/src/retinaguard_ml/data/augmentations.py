"""Fundus-specific data augmentations and Indian smartphone camera domain-shift simulation.

Implements ADR-009:
- Non-destructive clinical augmentations preserving lesion colors (flips, rotations, subtle contrast).
- Stochastic handheld/smartphone camera degradation simulating PHC/camp imaging conditions.
"""

from typing import Tuple
import cv2  # type: ignore
import numpy as np


class FundusClinicalAugmentor:
    """Safe augmentations for fundus photography that preserve diagnostic vascular features."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.7,
        brightness_contrast_prob: float = 0.3,
        seed: int | None = None,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        out = img_rgb.copy()

        # Horizontal flip
        if self.rng.random() < self.hflip_prob:
            out = cv2.flip(out, 1)

        # Vertical flip
        if self.rng.random() < self.vflip_prob:
            out = cv2.flip(out, 0)

        # Random rotation (0 to 360 degrees)
        if self.rng.random() < self.rotate_prob:
            angle = float(self.rng.uniform(0, 360))
            h, w = out.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            out = cv2.warpAffine(
                out, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )

        # Subtle contrast/brightness (preserves red lesions/hemorrhages)
        if self.rng.random() < self.brightness_contrast_prob:
            alpha = float(self.rng.uniform(0.90, 1.10))  # Contrast gain
            beta = float(self.rng.uniform(-10.0, 10.0))  # Brightness bias
            out = np.clip(alpha * out + beta, 0, 255).astype(np.uint8)

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
        """Simulate lower sensor / optical resolution through downscale and upsample."""
        scale = float(self.rng.uniform(*self.downscale_range))
        h, w = img.shape[:2]
        small_w = max(int(w * scale), 32)
        small_h = max(int(h * scale), 32)
        downscaled = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        return upscaled

    def apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Simulate lossy JPEG compression artifacts from mobile storage."""
        quality = int(self.rng.integers(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1))
        # Encode to JPEG in memory
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return img
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def apply_uneven_illumination(self, img: np.ndarray) -> np.ndarray:
        """Simulate handheld LED spotlight gradient and non-uniform illumination."""
        h, w = img.shape[:2]
        # Offset spotlight center slightly from the optical center
        cx = int(w * self.rng.uniform(0.40, 0.60))
        cy = int(h * self.rng.uniform(0.40, 0.60))

        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        max_dist_sq = (w // 2) ** 2 + (h // 2) ** 2

        # Radial falloff profile
        falloff_strength = float(self.rng.uniform(0.25, 0.55))
        gradient = 1.0 - (falloff_strength * (dist_sq / max_dist_sq))
        gradient = np.clip(gradient, 0.40, 1.05)[:, :, np.newaxis]

        illuminated = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        return illuminated

    def apply_optical_blur(self, img: np.ndarray) -> np.ndarray:
        """Simulate slight handheld motion or imperfect non-mydriatic focus."""
        ksize = int(self.rng.choice([3, 5]))
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def apply_color_cast(self, img: np.ndarray) -> np.ndarray:
        """Simulate smartphone camera white-balance shift / warm LED color temp."""
        out = img.astype(np.float32)
        # Shift red/green slightly (warm cast) or blue (cool cast)
        r_shift = float(self.rng.uniform(-8.0, 15.0))
        g_shift = float(self.rng.uniform(-5.0, 8.0))
        b_shift = float(self.rng.uniform(-10.0, 5.0))

        out[:, :, 0] += r_shift
        out[:, :, 1] += g_shift
        out[:, :, 2] += b_shift
        return np.clip(out, 0, 255).astype(np.uint8)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        """Apply smartphone camera degradation stochastically."""
        if self.rng.random() > self.p:
            return img_rgb

        out = img_rgb.copy()
        out = self.apply_resolution_loss(out)
        out = self.apply_jpeg_compression(out)
        out = self.apply_uneven_illumination(out)
        out = self.apply_optical_blur(out)
        out = self.apply_color_cast(out)
        return out

