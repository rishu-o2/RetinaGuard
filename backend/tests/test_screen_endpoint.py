"""Tests for the POST /screen endpoint and report services."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from retinaguard.api.main import app
from retinaguard.core.models import (
    CameraType,
    EyeSide,
    ICDRGrade,
    InferenceOutput,
    ReferralDecision,
    ScreeningResult,
    Urgency,
)
from retinaguard.services.report import ReportService

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_png(size: int = 400, fundus: bool = True) -> bytes:
    """Create a minimal synthetic fundus-like PNG in memory."""
    try:
        import cv2
        img = np.zeros((size, size, 3), dtype=np.uint8)
        if fundus:
            # Draw a bright circle (simulated FOV) on dark background
            center = (size // 2, size // 2)
            radius = size // 2 - 10
            cv2.circle(img, center, radius, (160, 80, 60), -1)
            # Add some texture
            noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
            img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()
    except ImportError:
        # Fallback: 1-px white PNG header (enough to test decode path)
        import struct, zlib
        def png_chunk(name, data):
            c = struct.pack(">I", len(data)) + name + data
            return c + struct.pack(">I", zlib.crc32(c[4:]) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
        raw = b"".join(
            b"\x00" + b"\xff" * (size * 3) for _ in range(size)
        )
        idat = zlib.compress(raw)
        return sig + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat) + png_chunk(b"IEND", b"")


_COMMON_FORM = dict(
    patient_pseudonym_id="PAT-ANON-TEST-001",
    eye_side="OD",
    camera_type="TABLETOP",
    age="45",
    diabetes_type="TYPE_2",
    language="en",
)


# ---------------------------------------------------------------------------
# /screen endpoint tests
# ---------------------------------------------------------------------------

class TestScreenEndpoint:
    """Integration tests that exercise the full pipeline via HTTP."""

    def _post(self, png_bytes: bytes, extra_form: Optional[dict] = None, headers: Optional[dict] = None):
        form = dict(_COMMON_FORM)
        if extra_form:
            form.update(extra_form)
        resp = client.post(
            "/screen/",
            data=form,
            files={"image": ("test.png", io.BytesIO(png_bytes), "image/png")},
            headers=headers,
        )
        return resp

    def test_health_still_works(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_screen_returns_200(self):
        png = _make_synthetic_png(512)
        resp = self._post(png)
        assert resp.status_code == 200

    def test_screen_clinician_review_mandatory_always_true(self):
        png = _make_synthetic_png(512)
        resp = self._post(png)
        assert resp.status_code == 200
        data = resp.json()
        assert data["clinician_review_mandatory"] is True

    def test_screen_contains_required_fields(self):
        png = _make_synthetic_png(512)
        resp = self._post(png)
        data = resp.json()
        required = [
            "screening_id", "patient_pseudonym_id", "referral", "urgency",
            "follow_up_interval_months", "image_sha256", "report_en", "report_hi",
            "clinician_review_mandatory", "disclaimer_en",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_screen_image_sha256_is_hex64(self):
        png = _make_synthetic_png(512)
        resp = self._post(png)
        sha = resp.json()["image_sha256"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_empty_file_returns_400(self):
        resp = client.post(
            "/screen/",
            data=_COMMON_FORM,
            files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert resp.status_code == 400

    def test_corrupt_file_returns_422(self):
        resp = client.post(
            "/screen/",
            data=_COMMON_FORM,
            files={"image": ("bad.png", io.BytesIO(b"not-an-image"), "image/png")},
        )
        assert resp.status_code == 422

    def test_hindi_report_requested(self):
        png = _make_synthetic_png(512)
        resp = self._post(png, extra_form={"language": "hi"})
        data = resp.json()
        assert data["report_requested_lang"] == "hi"
        # Hindi report text should contain Devanagari
        report_text = data["report_text"]
        assert any(ord(c) > 0x0900 for c in report_text), "Hindi report should contain Devanagari characters"

    def test_audit_hash_present_and_hex64(self):
        png = _make_synthetic_png(512)
        resp = self._post(png)
        data = resp.json()
        assert data.get("audit_hash") is not None
        assert len(data["audit_hash"]) == 64

    def test_audit_verify_endpoint(self):
        """After screening, the chain verification endpoint must report intact."""
        png = _make_synthetic_png(512)
        self._post(png)
        r = client.get("/screen/audit/verify")
        assert r.status_code == 200
        assert r.json()["chain_intact"] is True

    def test_audit_tail_endpoint(self):
        r = client.get("/screen/audit/tail?n=5")
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_anonymous_screening_forbidden_when_flag_false(self):
        png = _make_synthetic_png(512)
        with patch.dict(os.environ, {"ALLOW_ANONYMOUS_SCREENING": "false"}):
            resp = self._post(png)
            assert resp.status_code == 401
            assert "Authentication required" in resp.json()["detail"]

    def test_authenticated_screening_allowed_when_flag_false(self):
        from retinaguard.auth.deps import get_optional_current_user
        from retinaguard.db.models import User, UserRole
        png = _make_synthetic_png(512)
        mock_user = User(id="user-123", email="screener@clinic.in", role=UserRole.SCREENER, is_active=True)
        app.dependency_overrides[get_optional_current_user] = lambda: mock_user
        try:
            with patch.dict(os.environ, {"ALLOW_ANONYMOUS_SCREENING": "false"}):
                resp = self._post(png, headers={"Authorization": "Bearer mock-token"})
                assert resp.status_code == 200
        finally:
            app.dependency_overrides.pop(get_optional_current_user, None)

    def test_anonymous_screening_allowed_when_flag_true(self):
        png = _make_synthetic_png(512)
        with patch.dict(os.environ, {"ALLOW_ANONYMOUS_SCREENING": "true"}):
            resp = self._post(png)
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# ReportService unit tests
# ---------------------------------------------------------------------------

def _make_screening_result(**kwargs) -> ScreeningResult:
    from datetime import datetime, timezone
    from uuid import uuid4
    defaults = dict(
        screening_id=uuid4(),
        patient_pseudonym_id="PAT-001",
        eye_side=EyeSide.RIGHT_OD,
        camera_type=CameraType.TABLETOP,
        grade=ICDRGrade.MODERATE_NPDR,
        confidence=0.82,
        referral=ReferralDecision.REFERABLE,
        urgency=Urgency.PRIORITY,
        follow_up_interval_months=3,
        is_uncertain=False,
        uncertainty_reasons=[],
        clinician_review_mandatory=True,
        report_en="placeholder",
        report_hi="placeholder",
        image_sha256="a" * 64,
        created_at=datetime(2026, 11, 2, 10, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return ScreeningResult(**defaults)


class TestReportService:
    def test_en_report_contains_grade_label(self):
        result = _make_screening_result()
        report = ReportService.build_text_report(result, lang="en")
        assert "Moderate Non-Proliferative DR" in report

    def test_hi_report_contains_devanagari(self):
        result = _make_screening_result()
        report = ReportService.build_text_report(result, lang="hi")
        assert any(ord(c) > 0x0900 for c in report)

    def test_en_report_contains_disclaimer(self):
        result = _make_screening_result()
        report = ReportService.build_text_report(result, lang="en")
        assert "NOT a diagnosis" in report

    def test_json_report_clinician_review_always_true(self):
        result = _make_screening_result()
        payload = ReportService.build_json_report(result)
        assert payload["clinician_review_mandatory"] is True

    def test_json_report_no_raw_patient_data(self):
        """Verify that the JSON report never includes PHI-like raw fields."""
        result = _make_screening_result(patient_pseudonym_id="PAT-ANON-001")
        payload = ReportService.build_json_report(result)
        payload_str = json.dumps(payload)
        # Pseudonym ID present (that's acceptable) but not raw names etc.
        assert "PAT-ANON-001" in payload_str  # pseudonym OK
        # Make sure no actual name field leaks
        assert "abha_id" not in payload_str
        assert "abha_address" not in payload_str

    def test_uncertain_result_report_mentions_uncertainty(self):
        result = _make_screening_result(
            is_uncertain=True,
            uncertainty_reasons=["Image failed diagnostic quality gate."],
            referral=ReferralDecision.UNCERTAIN,
            grade=None,
            confidence=None,
        )
        report = ReportService.build_text_report(result, lang="en")
        assert "Uncertain" in report or "uncertain" in report or "⚠" in report

    def test_grade_none_handled_gracefully(self):
        result = _make_screening_result(grade=None, confidence=None)
        report = ReportService.build_text_report(result, lang="en")
        assert report  # should not raise

    def test_all_five_grades_render(self):
        for grade_val in range(5):
            result = _make_screening_result(grade=ICDRGrade(grade_val))
            report = ReportService.build_json_report(result)
            assert report["grade"] == grade_val
            assert report["grade_label_en"] is not None

