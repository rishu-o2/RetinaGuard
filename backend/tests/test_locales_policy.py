"""Tests for clinical translation locale review policy and ALLOW_UNREVIEWED_LOCALES."""

from __future__ import annotations

import io
import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from retinaguard.api.main import app
from retinaguard.core.config import get_allowed_locales, is_locale_allowed, is_locale_reviewed


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _make_dummy_png(size: int = 512) -> bytes:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 16, (180, 90, 45), -1)
    noise = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_is_locale_reviewed_requires_human_signoff():
    """Verify that is_locale_reviewed returns False unless reviewed_by and reviewed_on are populated."""
    # en and hi currently have reviewed=false, reviewed_by=null, reviewed_on=null
    assert is_locale_reviewed("en") is False
    assert is_locale_reviewed("hi") is False
    assert is_locale_reviewed("te") is False
    assert is_locale_reviewed("nonexistent") is False


def test_is_locale_allowed_flag(monkeypatch):
    """When ALLOW_UNREVIEWED_LOCALES=false, unreviewed locales are disallowed."""
    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "false")
    allowed, unreviewed_flag = is_locale_allowed("en")
    assert allowed is False
    assert unreviewed_flag is False

    allowed_hi, _ = is_locale_allowed("hi")
    assert allowed_hi is False

    # When ALLOW_UNREVIEWED_LOCALES=true (local demo mode), en and hi are permitted with unreviewed_flag=True
    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "true")
    allowed, unreviewed_flag = is_locale_allowed("en")
    assert allowed is True
    assert unreviewed_flag is True

    allowed_hi, unreviewed_hi = is_locale_allowed("hi")
    assert allowed_hi is True
    assert unreviewed_hi is True

    # Other unreviewed languages remain disallowed even in demo mode until reviewed
    allowed_te, _ = is_locale_allowed("te")
    assert allowed_te is False


def test_get_allowed_locales_lists_only_permitted(monkeypatch):
    """Verify get_allowed_locales reflects ALLOW_UNREVIEWED_LOCALES setting."""
    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "false")
    locales_prod = get_allowed_locales()
    # In strict clinical production, no unreviewed locales may be offered
    assert len(locales_prod) == 0

    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "true")
    locales_demo = get_allowed_locales()
    codes = [loc["code"] for loc in locales_demo]
    assert "en" in codes
    assert "hi" in codes
    for loc in locales_demo:
        assert loc.get("reviewed") is False or loc.get("unreviewed") is True


@pytest.mark.asyncio
async def test_screen_endpoint_rejects_unreviewed_locale_when_flag_false(client: AsyncClient, monkeypatch):
    """When ALLOW_UNREVIEWED_LOCALES=false, POST /screen/ with unreviewed locale is rejected with HTTP 400."""
    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "false")
    monkeypatch.setenv("ALLOW_ANONYMOUS_SCREENING", "true")

    img_bytes = _make_dummy_png()
    resp = await client.post(
        "/screen/",
        data={
            "patient_pseudonym_id": "VAL-TEST-001",
            "eye_side": "OD",
            "camera_type": "TABLETOP",
            "language": "en",
        },
        files={"image": ("test.png", io.BytesIO(img_bytes), "image/png")},
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "unreviewed" in detail.lower()


@pytest.mark.asyncio
async def test_screen_endpoint_accepts_with_note_when_flag_true(client: AsyncClient, monkeypatch):
    """When ALLOW_UNREVIEWED_LOCALES=true, POST /screen/ with en/hi succeeds and includes unreviewed_translation note."""
    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "true")
    monkeypatch.setenv("ALLOW_ANONYMOUS_SCREENING", "true")

    img_bytes = _make_dummy_png()
    resp = await client.post(
        "/screen/",
        data={
            "patient_pseudonym_id": "VAL-TEST-002",
            "eye_side": "OD",
            "camera_type": "TABLETOP",
            "language": "en",
        },
        files={"image": ("test.png", io.BytesIO(img_bytes), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["unreviewed_translation"] is True
    assert "unreviewed translation" in data["unreviewed_translation_note"].lower()


@pytest.mark.asyncio
async def test_auth_config_reflects_locale_policy(client: AsyncClient, monkeypatch):
    """Verify /auth/config endpoint returns allowed_locales filtered by policy."""
    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "false")
    resp_prod = await client.get("/auth/config")
    assert resp_prod.status_code == 200
    cfg_prod = resp_prod.json()
    assert cfg_prod["allow_unreviewed_locales"] is False
    assert len(cfg_prod["allowed_locales"]) == 0

    monkeypatch.setenv("ALLOW_UNREVIEWED_LOCALES", "true")
    resp_demo = await client.get("/auth/config")
    assert resp_demo.status_code == 200
    cfg_demo = resp_demo.json()
    assert cfg_demo["allow_unreviewed_locales"] is True
    codes = [l["code"] for l in cfg_demo["allowed_locales"]]
    assert "en" in codes and "hi" in codes
