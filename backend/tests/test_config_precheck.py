"""Tests for the browser pre-check configuration endpoint."""

from fastapi.testclient import TestClient
from retinaguard.api.main import app

client = TestClient(app)


def test_get_precheck_thresholds_tabletop_default():
    resp = client.get("/config/precheck")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_type"] == "tabletop"
    assert data["min_width"] == 256
    assert data["min_height"] == 256
    assert "blur_variance_threshold" in data
    assert "mean_luminance_min" in data
    assert "mean_luminance_max" in data
    assert "non_black_pixel_min_ratio" in data
    assert data["red_dominance_required"] is True


def test_get_precheck_thresholds_smartphone():
    resp = client.get("/config/precheck?camera_type=smartphone_portable")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_type"] == "smartphone_portable"
    assert data["min_width"] == 640
    assert data["min_height"] == 640
    assert data["blur_variance_threshold"] == 8.0


def test_get_precheck_thresholds_invalid():
    resp = client.get("/config/precheck?camera_type=invalid_camera")
    assert resp.status_code == 422
