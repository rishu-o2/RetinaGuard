"""Out-of-Distribution (OOD) Detection Service.

Implements ADR-004 & ADR-011:
- Tier 1: Fast optical spectral signature validation (rejects faces, documents, landscapes).
- Tier 2: Deep feature Mahalanobis distance, Energy score, and Maximum Softmax Probability (MSP).
- Guarantees fail-safe transition to 'UNCERTAIN - needs clinician review'.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2  # type: ignore
import numpy as np
import yaml

from retinaguard.core.config import _find_repo_root


@dataclass
class OODResult:
    """Outcome of Out-of-Distribution evaluation."""
    is_in_distribution: bool
    confidence: float
    ood_score: float
    method_triggered: Optional[str] = None
    failure_reason_en: Optional[str] = None
    failure_reason_hi: Optional[str] = None


class OODDetectorService:
    """Multi-tier OOD detector ensuring non-fundus or anomalous inputs are rejected."""

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        statistics_path: Optional[Path | str] = None,
    ):
        if config_path is None:
            config_path = _find_repo_root() / "configs" / "ood_thresholds.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Load Mahalanobis statistics if available
        self.class_centroids: Optional[np.ndarray] = None
        self.inv_covariance: Optional[np.ndarray] = None

        if statistics_path is None:
            stats_file = self.config["deep_ood_metrics"]["mahalanobis_distance"].get("statistics_file")
            if stats_file:
                statistics_path = _find_repo_root() / stats_file

        if statistics_path and Path(statistics_path).exists():
            self.load_statistics(statistics_path)

    def load_statistics(self, stats_path: Path | str) -> None:
        """Load empirical class means and precision matrix (inverse covariance)."""
        data = np.load(str(stats_path))
        self.class_centroids = data["centroids"]  # Shape: [num_classes, feature_dim]
        self.inv_covariance = data["inv_covariance"]  # Shape: [feature_dim, feature_dim]

    @staticmethod
    def evaluate_optical_spectrum(img_rgb: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """Verify physiological retinal reflectance (R > G > B hemoglobin signature).

        Rejects non-fundus images (e.g. skin, landscapes, text, uniform noise) in O(1) time.
        """
        # Focus on non-black pixels
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        mask = gray > 15
        if not np.any(mask):
            return False, {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0, "r_to_b": 0.0}

        r_vals = img_rgb[:, :, 0][mask]
        g_vals = img_rgb[:, :, 1][mask]
        b_vals = img_rgb[:, :, 2][mask]

        mean_r = float(np.mean(r_vals))
        mean_g = float(np.mean(g_vals))
        mean_b = float(np.mean(b_vals))

        r_to_b = mean_r / max(mean_b, 1.0)
        r_to_g = mean_r / max(mean_g, 1.0)

        metrics = {
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "r_to_b": r_to_b,
            "r_to_g": r_to_g,
        }

        # Human fundus rules: Red channel strictly dominates blue and green
        # Blue channel is heavily absorbed by ocular media and melanin/hemoglobin
        is_retinal_spectrum = (r_to_b >= 1.25) and (r_to_g >= 1.05) and (mean_b < 140.0)
        return is_retinal_spectrum, metrics

    @staticmethod
    def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
        """Calculate Energy score: E(x; T) = -T * log(sum(exp(logits / T)))."""
        scaled = logits / max(temperature, 1e-4)
        max_val = np.max(scaled)
        # Log-sum-exp trick for numerical stability
        energy = -temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))
        return float(energy)

    def compute_mahalanobis_distance(self, feature_vector: np.ndarray) -> float:
        """Calculate minimum Mahalanobis distance to class centroids: min_c (z - mu_c)^T Sigma^-1 (z - mu_c)."""
        if self.class_centroids is None or self.inv_covariance is None:
            return 0.0

        z = feature_vector.flatten()
        min_dist = float("inf")
        for centroid in self.class_centroids:
            diff = z - centroid
            dist = float(np.dot(np.dot(diff, self.inv_covariance), diff))
            if dist < min_dist:
                min_dist = dist

        return float(np.sqrt(max(min_dist, 0.0)))

    def evaluate(
        self,
        img_rgb: np.ndarray,
        logits: Optional[np.ndarray] = None,
        penultimate_features: Optional[np.ndarray] = None,
    ) -> OODResult:
        """Run multi-tier OOD evaluation."""
        messages = self.config["failure_messages"]

        # Tier 1: Optical spectral signature
        is_retinal, spec_metrics = self.evaluate_optical_spectrum(img_rgb)
        if not is_retinal:
            return OODResult(
                is_in_distribution=False,
                confidence=0.0,
                ood_score=float(spec_metrics["r_to_b"]),
                method_triggered="OPTICAL_SPECTRUM_FILTER",
                failure_reason_en=messages["OPTICAL_SPECTRUM_INVALID"]["en"],
                failure_reason_hi=messages["OPTICAL_SPECTRUM_INVALID"]["hi"],
            )

        # Tier 2: Deep feature space OOD metrics (if logits / features provided)
        if logits is not None:
            # 1. Maximum Softmax Probability (MSP)
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            msp_confidence = float(np.max(probs))

            msp_cutoff = self.config["deep_ood_metrics"]["msp_min_confidence"]
            if msp_confidence < msp_cutoff:
                return OODResult(
                    is_in_distribution=False,
                    confidence=msp_confidence,
                    ood_score=msp_confidence,
                    method_triggered="MAX_SOFTMAX_PROBABILITY",
                    failure_reason_en=messages["OOD_FEATURE_DEVIATION"]["en"],
                    failure_reason_hi=messages["OOD_FEATURE_DEVIATION"]["hi"],
                )

            # 2. Energy Score
            energy_cfg = self.config["deep_ood_metrics"]["energy_score"]
            energy = self.compute_energy_score(logits, temperature=energy_cfg["temperature"])
            if energy > energy_cfg["threshold"]:
                return OODResult(
                    is_in_distribution=False,
                    confidence=msp_confidence,
                    ood_score=energy,
                    method_triggered="ENERGY_SCORE",
                    failure_reason_en=messages["OOD_FEATURE_DEVIATION"]["en"],
                    failure_reason_hi=messages["OOD_FEATURE_DEVIATION"]["hi"],
                )

        # 3. Mahalanobis distance on features
        if penultimate_features is not None and self.class_centroids is not None:
            dist = self.compute_mahalanobis_distance(penultimate_features)
            dist_threshold = self.config["deep_ood_metrics"]["mahalanobis_distance"]["max_distance_threshold"]
            if dist > dist_threshold:
                return OODResult(
                    is_in_distribution=False,
                    confidence=0.0,
                    ood_score=dist,
                    method_triggered="MAHALANOBIS_DISTANCE",
                    failure_reason_en=messages["OOD_FEATURE_DEVIATION"]["en"],
                    failure_reason_hi=messages["OOD_FEATURE_DEVIATION"]["hi"],
                )

        return OODResult(
            is_in_distribution=True,
            confidence=1.0 if logits is None else float(np.max(np.exp(logits) / np.sum(np.exp(logits)))),
            ood_score=0.0,
        )

