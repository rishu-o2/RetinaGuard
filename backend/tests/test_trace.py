"""Tests for the 5-agent trace panel output."""

import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from retinaguard.api.main import app

client = TestClient(app)


def _create_synthetic_fundus_bytes() -> bytes:
    # 256x256 image with reddish fundus circle
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Circle in center
    y, x = np.ogrid[:256, :256]
    mask = (x - 128) ** 2 + (y - 128) ** 2 <= 100 ** 2
    # R > G > B
    img[mask, 0] = 180
    img[mask, 1] = 60
    img[mask, 2] = 20
    # Add some noise to prevent zero blur variance
    noise = np.random.randint(0, 30, (256, 256, 3), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _create_black_image_bytes() -> bytes:
    # 256x256 pure black (fails quality gate -> early exit)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_trace_present_and_has_five_agents():
    img_bytes = _create_synthetic_fundus_bytes()
    resp = client.post(
        "/screen/",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={
            "patient_pseudonym_id": "P-TEST-TRACE-001",
            "camera_type": "TABLETOP",
            "eye_side": "OD",
            "age": "45",
            "diabetes_type": "TYPE_2",
            "language": "en",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "trace" in data
    trace = data["trace"]
    assert len(trace) == 5

    agents_in_trace = [item["agent"] for item in trace]
    assert agents_in_trace == ["vision", "safety", "guideline", "audit", "voice"]

    for item in trace:
        assert "status" in item
        assert item["status"] in ("passed", "early_exit", "skipped")
        assert "duration_ms" in item
        assert isinstance(item["duration_ms"], (int, float))
        assert "note" in item
        assert isinstance(item["note"], str)

        # PHI / Data leakage assertions: No pseudonym, no image SHA in trace notes
        assert "P-TEST-TRACE-001" not in item["note"]
        assert "image_sha256" not in item
        assert "patient_pseudonym_id" not in item


def test_trace_on_early_exit():
    # Pure black image triggers early exit in quality gate
    img_bytes = _create_black_image_bytes()
    resp = client.post(
        "/screen/",
        files={"image": ("black.png", img_bytes, "image/png")},
        data={
            "patient_pseudonym_id": "P-TEST-EARLY-EXIT",
            "camera_type": "TABLETOP",
            "eye_side": "OS",
            "age": "45",
            "diabetes_type": "TYPE_2",
            "language": "en",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "trace" in data
    trace = data["trace"]
    assert len(trace) == 5

    vision_entry = next(t for t in trace if t["agent"] == "vision")
    assert vision_entry["status"] == "early_exit"

    guideline_entry = next(t for t in trace if t["agent"] == "guideline")
    assert guideline_entry["status"] == "skipped"
    assert guideline_entry["duration_ms"] == 0.0

    # Ensure no patient data leaked in trace
    for item in trace:
        assert "P-TEST-EARLY-EXIT" not in item["note"]
