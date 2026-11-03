"""Classical Computer Vision Image Quality Gate.

Implements ADR-004 & ADR-011:
- Circular Field-of-View (FOV) segmentation.
- Masked Laplacian focus variance (blur detection).
- Illumination percentiles (mean, 5th, 95th).
- Under-exposure and flash glare / over-exposure detection.
- Human-readable feedback in English and Hindi.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2  # type: ignore
import numpy as np
import yaml

from retinaguard.core.config import _find_repo_root
from retinaguard.core.models import CameraType


@dataclass
class QualityScores:
    """Quantitative metrics extracted inside the retinal field of view."""
    width: int
    height: int
    fov_area_ratio: float
    fov_circularity: float
    blur_variance: float
    mean_luminance: float
    p5_luminance: float
    p95_luminance: float
    underexposed_ratio: float
    overexposed_ratio: float


@dataclass
class QualityGateResult:
    """Outcome of pre-inference quality evaluation."""
    passed: bool
    scores: QualityScores
    failure_codes: List[str] = field(default_factory=list)
    human_readable_reason: str = ""
    human_readable_reason_hi: str = ""


class QualityGateService:
    """Classical Computer Vision quality gate for tabletop and smartphone fundus photos."""

    def __init__(self, config_path: Optional[Path | str] = None):
        if config_path is None:
            config_path = _find_repo_root() / "configs" / "quality_thresholds.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @staticmethod
    def segment_fundus_fov(img_rgb: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Detect the circular retinal field-of-view mask, area ratio, and circularity."""
        h, w = img_rgb.shape[:2]
        total_pixels = h * w

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Threshold dark background
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Morphological opening to eliminate dust/eyelashes outside
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((h, w), dtype=np.uint8), 0.0, 0.0

        # Largest contour is the retinal aperture
        largest_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest_contour))
        perimeter = float(cv2.arcLength(largest_contour, True))

        area_ratio = area / max(total_pixels, 1)

        # Circularity metric: 4 * pi * area / perimeter^2 (1.0 for perfect circle)
        circularity = (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Generate binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 1, -1)

        return mask, area_ratio, circularity

    @staticmethod
    def calculate_blur_variance(img_gray: np.ndarray, mask: np.ndarray) -> float:
        """Calculate Laplacian focus variance exclusively inside the fundus mask."""
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        mask_bool = (mask == 1)
        if not np.any(mask_bool):
            return 0.0
        masked_laplacian = laplacian[mask_bool]
        return float(np.var(masked_laplacian))

    @staticmethod
    def calculate_exposure_and_luminance(
        img_gray: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """Compute mean luminance, 5th/95th percentiles, and over/underexposure ratios."""
        mask_bool = (mask == 1)
        if not np.any(mask_bool):
            return 0.0, 0.0, 0.0, 1.0, 0.0

        retinal_pixels = img_gray[mask_bool].astype(np.float32)
        total_retinal_pixels = len(retinal_pixels)

        mean_lum = float(np.mean(retinal_pixels))
        p5_lum = float(np.percentile(retinal_pixels, 5))
        p95_lum = float(np.percentile(retinal_pixels, 95))

        # Pixel fractions
        underexposed_count = np.sum(retinal_pixels < 15)
        overexposed_count = np.sum(retinal_pixels > 245)

        underexposed_ratio = float(underexposed_count / total_retinal_pixels)
        overexposed_ratio = float(overexposed_count / total_retinal_pixels)

        return mean_lum, p5_lum, p95_lum, underexposed_ratio, overexposed_ratio

    def evaluate(
        self, img_rgb: np.ndarray, camera_type: CameraType = CameraType.SMARTPHONE_PORTABLE
    ) -> QualityGateResult:
        """Run full classical CV quality evaluation against camera profile thresholds."""
        profile_key = (
            "smartphone_portable"
            if camera_type == CameraType.SMARTPHONE_PORTABLE
            else "tabletop"
        )
        thresh = self.config["profiles"][profile_key]
        messages = self.config["failure_messages"]

        h, w = img_rgb.shape[:2]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Step 1: Detect circular FOV mask
        mask, fov_area_ratio, fov_circularity = self.segment_fundus_fov(img_rgb)

        # Step 2: Compute metrics inside mask
        blur_var = self.calculate_blur_variance(img_gray, mask)
        mean_lum, p5_lum, p95_lum, under_ratio, over_ratio = self.calculate_exposure_and_luminance(
            img_gray, mask
        )

        scores = QualityScores(
            width=w,
            height=h,
            fov_area_ratio=fov_area_ratio,
            fov_circularity=fov_circularity,
            blur_variance=blur_var,
            mean_luminance=mean_lum,
            p5_luminance=p5_lum,
            p95_luminance=p95_lum,
            underexposed_ratio=under_ratio,
            overexposed_ratio=over_ratio,
        )

        failure_codes: List[str] = []

        # Resolution check
        if w < thresh["min_resolution_width"] or h < thresh["min_resolution_height"]:
            failure_codes.append("RESOLUTION_TOO_LOW")

        # Field of View / Circular Mask Check
        if fov_area_ratio < thresh["fov_min_area_ratio"] or fov_circularity < thresh["fov_min_circularity"]:
            failure_codes.append("INVALID_FOV")

        # Blur Check
        if blur_var < thresh["blur_variance_threshold"]:
            failure_codes.append("BLUR")

        # Underexposure Check
        if (
            mean_lum < thresh["mean_luminance_min"]
            or p5_lum < thresh["p5_luminance_min"]
            or under_ratio > thresh["underexposure_max_ratio"]
        ):
            failure_codes.append("UNDEREXPOSED")

        # Overexposure / Flash Glare Check
        if (
            mean_lum > thresh["mean_luminance_max"]
            or p95_lum > thresh["p95_luminance_max"]
            or over_ratio > thresh["overexposure_max_ratio"]
        ):
            failure_codes.append("OVEREXPOSED")

        passed = len(failure_codes) == 0

        # Generate human-readable localized instructions
        if not passed:
            primary_code = failure_codes[0]
            msg_en = messages.get(primary_code, {}).get("en", "Image failed quality gate.")
            msg_hi = messages.get(primary_code, {}).get("hi", "छवि गुणवत्ता जांच में विफल रही।")
        else:
            msg_en = "Image meets all diagnostic quality standards."
            msg_hi = "छवि गुणवत्ता के सभी मानकों पर खरी उतरी है।"

        return QualityGateResult(
            passed=passed,
            scores=scores,
            failure_codes=failure_codes,
            human_readable_reason=msg_en,
            human_readable_reason_hi=msg_hi,
        )

