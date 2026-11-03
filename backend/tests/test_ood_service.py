"""Tests for Out-of-Distribution (OOD) Detection Service and Statistics Fitting."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from retinaguard.services.fit_ood import fit_in_distribution_statistics
from retinaguard.services.ood import OODDetectorService


@pytest.fixture
def ood_service() -> OODDetectorService:
    return OODDetectorService()


def create_synthetic_fundus(size: int = 512) -> np.ndarray:
    """Generate an image with fundus-like optical spectrum (R > G > B)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Red-dominant fundus color
    img[:, :] = (180, 80, 20)
    return img


def test_valid_fundus_passes_optical_spectrum(ood_service: OODDetectorService) -> None:
    """Verify that a valid fundus image passes the optical spectrum check."""
    fundus = create_synthetic_fundus()
    res = ood_service.evaluate(fundus)

    assert res.is_in_distribution is True
    assert res.failure_reason_en is None


def test_blue_sky_photo_rejected_as_ood(ood_service: OODDetectorService) -> None:
    """Verify that an outdoor blue sky photo (B > R) is immediately rejected as OOD."""
    # Blue sky: High blue, moderate green, low red
    blue_sky = np.zeros((512, 512, 3), dtype=np.uint8)
    blue_sky[:, :] = (60, 140, 230)

    res = ood_service.evaluate(blue_sky)

    assert res.is_in_distribution is False
    assert res.method_triggered == "OPTICAL_SPECTRUM_FILTER"
    assert "does not appear to be a human retinal fundus photo" in res.failure_reason_en


def test_green_foliage_photo_rejected_as_ood(ood_service: OODDetectorService) -> None:
    """Verify that green landscape/vegetation is rejected as OOD."""
    green_field = np.zeros((512, 512, 3), dtype=np.uint8)
    green_field[:, :] = (40, 180, 50)

    res = ood_service.evaluate(green_field)

    assert res.is_in_distribution is False
    assert res.method_triggered == "OPTICAL_SPECTRUM_FILTER"


def test_text_document_rejected_as_ood(ood_service: OODDetectorService) -> None:
    """Verify that a black and white scanned document is rejected as OOD."""
    doc = np.full((512, 512, 3), 240, dtype=np.uint8)  # White background
    # Add horizontal black text lines
    doc[100:105, 50:450] = 0
    doc[150:155, 50:450] = 0
    doc[200:205, 50:450] = 0

    res = ood_service.evaluate(doc)

    assert res.is_in_distribution is False
    assert res.method_triggered == "OPTICAL_SPECTRUM_FILTER"


def test_random_color_noise_rejected_as_ood(ood_service: OODDetectorService) -> None:
    """Verify that uniform random RGB noise is rejected as OOD."""
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)

    res = ood_service.evaluate(noise)

    assert res.is_in_distribution is False
    assert res.method_triggered == "OPTICAL_SPECTRUM_FILTER"


def test_deep_msp_confidence_cutoff(ood_service: OODDetectorService) -> None:
    """Verify that uniform/equivocal logits trigger OOD due to low max-softmax probability."""
    fundus = create_synthetic_fundus()
    # Equivocal logits across 5 classes -> max softmax ~ 0.20 < 0.60 cutoff
    equivocal_logits = np.array([1.0, 1.05, 0.95, 1.0, 1.02])

    res = ood_service.evaluate(fundus, logits=equivocal_logits)

    assert res.is_in_distribution is False
    assert res.method_triggered == "MAX_SOFTMAX_PROBABILITY"


def test_deep_energy_score_cutoff(ood_service: OODDetectorService) -> None:
    """Verify that high energy scores trigger OOD rejection."""
    fundus = create_synthetic_fundus()
    # Negative logits with low magnitude produce high energy (E = -log(sum(exp(-5))))
    low_density_logits = np.array([-5.0, -5.5, -4.8, -5.2, -6.0])

    res = ood_service.evaluate(fundus, logits=low_density_logits)

    assert res.is_in_distribution is False
    assert res.method_triggered in ("MAX_SOFTMAX_PROBABILITY", "ENERGY_SCORE")


def test_fit_and_evaluate_mahalanobis_distance() -> None:
    """Verify offline fitting of in-distribution statistics and online Mahalanobis rejection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = Path(tmpdir) / "test_ood_stats.npz"

        # Fit statistics on 100 8-dimensional feature vectors clustered around 0
        rng = np.random.default_rng(42)
        train_feats = rng.normal(loc=0.0, scale=1.0, size=(100, 8)).astype(np.float32)
        train_labels = rng.integers(0, 5, size=100)

        fit_in_distribution_statistics(train_feats, train_labels, save_path=stats_file)
        assert stats_file.exists()

        # Initialize detector with fitted statistics
        detector = OODDetectorService(statistics_path=stats_file)

        # In-distribution sample close to origin
        in_dist_feat = np.zeros(8, dtype=np.float32)
        dist_in = detector.compute_mahalanobis_distance(in_dist_feat)
        assert dist_in < 10.0

        # Extreme outlier feature vector
        outlier_feat = np.full(8, 100.0, dtype=np.float32)
        dist_out = detector.compute_mahalanobis_distance(outlier_feat)
        assert dist_out > 50.0

