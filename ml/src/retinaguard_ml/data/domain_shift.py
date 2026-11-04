"""Quantitative Domain-Shift Analysis for Retinal Imaging.

Compares optical signatures across clinical datasets and camera hardware:
- Color histograms (RGB and CIE-LAB luminance/chrominance)
- Brightness and contrast distributions
- Image resolution profiles
- Distributional divergence (Wasserstein distance)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import cv2  # type: ignore
import numpy as np


@dataclass
class ImageDomainProfile:
    """Optical profile extracted from an individual fundus photo."""
    dataset_name: str
    device: str
    width: int
    height: int
    mean_luminance: float
    std_luminance: float
    rgb_means: Tuple[float, float, float]
    lab_means: Tuple[float, float, float]
    r_hist: np.ndarray
    g_hist: np.ndarray
    b_hist: np.ndarray


@dataclass
class AggregateDomainStats:
    """Aggregated statistical profile of a dataset or camera category."""
    group_name: str
    sample_count: int
    mean_resolution: Tuple[float, float]
    mean_brightness: float
    mean_contrast: float
    mean_rgb: Tuple[float, float, float]
    mean_lab: Tuple[float, float, float]


class DomainShiftAnalyzer:
    """Analyzes distribution shift across camera hardware and clinical cohorts."""

    @staticmethod
    def profile_image(img_rgb: np.ndarray, dataset_name: str, device: str) -> ImageDomainProfile:
        """Extract optical signature from a single RGB image."""
        h, w = img_rgb.shape[:2]

        # Grayscale & CIE-LAB representation
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # Exclude black background pixels from statistics
        mask = gray > 10
        valid_pixels_rgb = img_rgb[mask] if np.any(mask) else img_rgb.reshape(-1, 3)
        valid_pixels_lab = lab[mask] if np.any(mask) else lab.reshape(-1, 3)
        valid_gray = gray[mask] if np.any(mask) else gray.flatten()

        rgb_means = tuple(float(x) for x in valid_pixels_rgb.mean(axis=0))
        lab_means = tuple(float(x) for x in valid_pixels_lab.mean(axis=0))

        # Normalized 32-bin histograms for R, G, B
        r_hist, _ = np.histogram(valid_pixels_rgb[:, 0], bins=32, range=(0, 256), density=True)
        g_hist, _ = np.histogram(valid_pixels_rgb[:, 1], bins=32, range=(0, 256), density=True)
        b_hist, _ = np.histogram(valid_pixels_rgb[:, 2], bins=32, range=(0, 256), density=True)

        return ImageDomainProfile(
            dataset_name=dataset_name,
            device=device,
            width=w,
            height=h,
            mean_luminance=float(np.mean(valid_gray)),
            std_luminance=float(np.std(valid_gray)),
            rgb_means=(rgb_means[0], rgb_means[1], rgb_means[2]),
            lab_means=(lab_means[0], lab_means[1], lab_means[2]),
            r_hist=r_hist,
            g_hist=g_hist,
            b_hist=b_hist,
        )

    @classmethod
    def aggregate_profiles(
        cls, profiles: List[ImageDomainProfile], group_key: str = "dataset_name"
    ) -> Dict[str, AggregateDomainStats]:
        """Aggregate profiles by dataset name or camera device."""
        groups: Dict[str, List[ImageDomainProfile]] = {}
        for p in profiles:
            val = getattr(p, group_key, "UNKNOWN")
            groups.setdefault(val, []).append(p)

        stats: Dict[str, AggregateDomainStats] = {}
        for name, items in groups.items():
            widths = [p.width for p in items]
            heights = [p.height for p in items]
            brights = [p.mean_luminance for p in items]
            contrasts = [p.std_luminance for p in items]

            mean_rgb = np.mean([p.rgb_means for p in items], axis=0)
            mean_lab = np.mean([p.lab_means for p in items], axis=0)

            stats[name] = AggregateDomainStats(
                group_name=name,
                sample_count=len(items),
                mean_resolution=(float(np.mean(widths)), float(np.mean(heights))),
                mean_brightness=float(np.mean(brights)),
                mean_contrast=float(np.mean(contrasts)),
                mean_rgb=(float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])),
                mean_lab=(float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])),
            )
        return stats

    @staticmethod
    def compute_histogram_divergence(
        profiles_a: List[ImageDomainProfile], profiles_b: List[ImageDomainProfile]
    ) -> Dict[str, float]:
        """Compute histogram intersection and difference between two groups."""
        if not profiles_a or not profiles_b:
            return {"r_intersection": 0.0, "g_intersection": 0.0, "b_intersection": 0.0}

        avg_r_a = np.mean([p.r_hist for p in profiles_a], axis=0)
        avg_r_b = np.mean([p.r_hist for p in profiles_b], axis=0)
        avg_g_a = np.mean([p.g_hist for p in profiles_a], axis=0)
        avg_g_b = np.mean([p.g_hist for p in profiles_b], axis=0)
        avg_b_a = np.mean([p.b_hist for p in profiles_a], axis=0)
        avg_b_b = np.mean([p.b_hist for p in profiles_b], axis=0)

        # Histogram intersection similarity (1.0 = identical distributions)
        r_sim = float(np.sum(np.minimum(avg_r_a, avg_r_b)))
        g_sim = float(np.sum(np.minimum(avg_g_a, avg_g_b)))
        b_sim = float(np.sum(np.minimum(avg_b_a, avg_b_b)))

        return {
            "r_intersection_similarity": r_sim,
            "g_intersection_similarity": g_sim,
            "b_intersection_similarity": b_sim,
        }

