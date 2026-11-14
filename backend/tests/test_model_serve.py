"""Tests for active ONNX model serving and metadata endpoint."""

import hashlib
import json
import uuid
import pytest
from fastapi.testclient import TestClient

from retinaguard.api.main import app
from retinaguard.auth.jwt import create_access_token, hash_password
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import get_sessionmaker

client = TestClient(app)


@pytest.fixture
async def auth_headers():
    user_id = f"test-user-{uuid.uuid4()}"
    email = f"nurse-{uuid.uuid4()}@example.com"
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


@pytest.mark.asyncio
async def test_model_endpoints_require_auth():
    # 401 Unauthorized without token
    resp1 = client.get("/model/active")
    assert resp1.status_code == 401

    resp2 = client.get("/model/metadata")
    assert resp2.status_code == 401


@pytest.mark.asyncio
async def test_model_metadata_returns_complete_config(auth_headers):
    resp = client.get("/model/metadata", headers=auth_headers)
    assert resp.status_code == 200
    meta = resp.json()

    assert meta["version"] == "v1.0.0"
    assert "temperature" in meta
    assert isinstance(meta["temperature"], (int, float))
    assert meta["temperature"] > 0

    assert "abstention" in meta
    assert meta["abstention"]["lower_edge"] == 0.30
    assert meta["abstention"]["upper_edge"] == 0.70

    assert "sight_threatening" in meta
    assert meta["sight_threatening"]["threshold_soon"] == 0.05
    assert meta["sight_threatening"]["threshold_urgent"] == 0.30

    assert "grade_table" in meta
    assert "0" in meta["grade_table"]
    assert "4" in meta["grade_table"]
    assert meta["grade_table"]["0"]["referable"] is False
    assert meta["grade_table"]["2"]["referable"] is True

    assert "modifiers" in meta
    assert "pregnancy" in meta["modifiers"]
    assert "onnx_sha256" in meta


@pytest.mark.asyncio
async def test_model_active_serves_onnx_with_sha256(auth_headers):
    resp = client.get("/model/active", headers=auth_headers)
    assert resp.status_code == 200
    assert "X-Model-SHA256" in resp.headers

    # Verify SHA-256 matches actual payload bytes
    computed_sha = hashlib.sha256(resp.content).hexdigest()
    assert resp.headers["X-Model-SHA256"] == computed_sha
