"""Tests for Classical Computer Vision Image Quality Gate Service."""

import cv2  # type: ignore
import numpy as np
import pytest

from retinaguard.core.models import CameraType
from retinaguard.services.quality import QualityGateService


@pytest.fixture
def quality_service() -> QualityGateService:
    return QualityGateService()


def create_synthetic_fundus(size: int = 800) -> np.ndarray:
    """Generate a clean synthetic fundus image with circular FOV and retinal texture."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.42)

    # Orange-red retinal fundus background
    cv2.circle(img, center, radius, (180, 80, 20), -1)

    # Add optic disc (bright yellowish circle)
    cv2.circle(img, (center[0] - 120, center[1]), int(radius * 0.18), (220, 200, 100), -1)

    # Add retinal blood vessels (dark branching lines with high contrast)
    rng = np.random.default_rng(42)
    for _ in range(12):
        pt1 = (int(center[0] - 120 + rng.integers(-20, 20)), int(center[1] + rng.integers(-20, 20)))
        pt2 = (int(center[0] + rng.integers(-150, 150)), int(center[1] + rng.integers(-150, 150)))
        cv2.line(img, pt1, pt2, (100, 30, 10), 3)

    # Add high-frequency texture inside the retina to ensure adequate Laplacian variance
    noise = rng.integers(-25, 25, size=(size, size, 3), dtype=np.int16)
    mask = (img[:, :, 0] > 0)[:, :, np.newaxis]
    noisy_fundus = np.clip(img.astype(np.int16) + noise * mask, 0, 255).astype(np.uint8)

    return noisy_fundus


def test_valid_synthetic_fundus_passes_quality_gate(quality_service: QualityGateService) -> None:
    """Verify that a well-illuminated, sharp fundus image passes the quality gate."""
    fundus = create_synthetic_fundus(size=800)
    result = quality_service.evaluate(fundus, camera_type=CameraType.SMARTPHONE_PORTABLE)

    assert result.passed is True
    assert len(result.failure_codes) == 0
    assert "standards" in result.human_readable_reason


def test_blurry_image_fails_quality_gate(quality_service: QualityGateService) -> None:
    """Verify that an image degraded with severe optical/motion blur is rejected."""
    fundus = create_synthetic_fundus(size=800)
    # Apply severe Gaussian blur
    blurred = cv2.GaussianBlur(fundus, (31, 31), 15.0)

    result = quality_service.evaluate(blurred, camera_type=CameraType.SMARTPHONE_PORTABLE)

    assert result.passed is False
    assert "BLUR" in result.failure_codes
    assert "blurry" in result.human_readable_reason.lower()
    assert "धुंधली" in result.human_readable_reason_hi


def test_underexposed_dark_image_fails(quality_service: QualityGateService) -> None:
    """Verify that an excessively dark/underexposed image is rejected."""
    fundus = create_synthetic_fundus(size=1200)
    # Reduce luminance moderately so FOV is detected but illumination is below diagnostic threshold
    dark = (fundus.astype(np.float32) * 0.35).astype(np.uint8)

    # Evaluate against tabletop profile which requires mean_luminance >= 40.0 and p5 >= 15.0
    result = quality_service.evaluate(dark, camera_type=CameraType.TABLETOP)

    assert result.passed is False
    assert "UNDEREXPOSED" in result.failure_codes
    assert "dark" in result.human_readable_reason.lower()


def test_overexposed_flash_glare_fails(quality_service: QualityGateService) -> None:
    """Verify that flash reflection/glare obscuring the retina is rejected."""
    fundus = create_synthetic_fundus(size=800)
    # Paint massive saturated corneal flash glare in the center
    cv2.circle(fundus, (400, 400), 160, (255, 255, 255), -1)

    result = quality_service.evaluate(fundus, camera_type=CameraType.SMARTPHONE_PORTABLE)

    assert result.passed is False
    assert "OVEREXPOSED" in result.failure_codes
    assert "glare" in result.human_readable_reason.lower()


def test_invalid_field_of_view_fails(quality_service: QualityGateService) -> None:
    """Verify that an image without a circular retinal aperture is rejected."""
    # Flat horizontal stripe instead of a circular eye
    non_circular = np.zeros((800, 800, 3), dtype=np.uint8)
    non_circular[380:420, :] = 180

    result = quality_service.evaluate(non_circular, camera_type=CameraType.SMARTPHONE_PORTABLE)

    assert result.passed is False
    assert "INVALID_FOV" in result.failure_codes


def test_low_resolution_fails(quality_service: QualityGateService) -> None:
    """Verify that an image smaller than camera profile limits is rejected."""
    tiny_img = create_synthetic_fundus(size=300)

    result = quality_service.evaluate(tiny_img, camera_type=CameraType.SMARTPHONE_PORTABLE)

    assert result.passed is False
    assert "RESOLUTION_TOO_LOW" in result.failure_codes
