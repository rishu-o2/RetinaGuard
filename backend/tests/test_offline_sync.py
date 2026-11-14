"""Tests for offline screening sync endpoint and discrepancy logging."""

import base64
import io
import uuid
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from retinaguard.api.main import app
from retinaguard.auth.jwt import create_access_token, hash_password
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import get_sessionmaker
from retinaguard.services.audit import AuditLedger

client = TestClient(app)


@pytest.fixture
async def auth_headers():
    user_id = f"test-sync-{uuid.uuid4()}"
    email = f"sync-{uuid.uuid4()}@example.com"
    session_factory = get_sessionmaker()
    async with session_factory() as db:
        user = User(
            id=user_id,
            email=email,
            hashed_password=hash_password("Password123!"),
            role=UserRole.SCREENER,
            is_active=True,
        )
        db.add(user)
        await db.commit()

    token = create_access_token(data={"sub": user_id, "role": "screener"})
    return {"Authorization": f"Bearer {token}"}


def _create_synthetic_fundus_b64() -> str:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    mask = (x - 128) ** 2 + (y - 128) ** 2 <= 100 ** 2
    img[mask, 0] = 180
    img[mask, 1] = 60
    img[mask, 2] = 20
    noise = np.random.randint(0, 30, (256, 256, 3), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_offline_sync_requires_auth():
    resp = client.post("/offline/sync", json={"patient_pseudonym_id": "P-1", "eye_side": "OD", "camera_type": "tabletop", "image_b64": "abc", "offline_result": {}})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_offline_sync_discrepancy_logging(auth_headers):
    img_b64 = _create_synthetic_fundus_b64()

    # Pass offline_result that deliberately differs from server outcome (e.g. non-referable ROUTINE)
    fake_offline = {
        "referral": "NON_REFERABLE",
        "urgency": "ROUTINE",
        "grade": 0,
    }

    body = {
        "patient_pseudonym_id": "P-SYNC-DISCREPANCY-001",
        "eye_side": "OD",
        "camera_type": "TABLETOP",
        "language": "en",
        "image_b64": img_b64,
        "offline_result": fake_offline,
    }

    resp = client.post("/offline/sync", json=body, headers=auth_headers)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data.get("offline_sync") is True

    # Check audit ledger for OFFLINE_SYNC_DISCREPANCY event
    ledger = AuditLedger()
    entries = ledger.tail(n=20)
    discrepancy_entries = [e for e in entries if e.event_type == "OFFLINE_SYNC_DISCREPANCY"]
    assert len(discrepancy_entries) > 0
    recent = discrepancy_entries[-1]
    assert recent.patient_pseudonym_id == "P-SYNC-DISCREPANCY-001"
    assert "offline=" in recent.outcome_summary


@pytest.mark.asyncio
async def test_offline_sync_invalid_image(auth_headers):
    body = {
        "patient_pseudonym_id": "P-INVALID-IMAGE",
        "eye_side": "OD",
        "camera_type": "TABLETOP",
        "language": "en",
        "image_b64": "not_valid_base64_data!!!",
        "offline_result": {},
    }
    resp = client.post("/offline/sync", json=body, headers=auth_headers)
    assert resp.status_code == 422
