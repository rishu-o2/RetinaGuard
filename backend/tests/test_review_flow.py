"""Tests for the clinician review queue and audit trail integration."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from retinaguard.api.main import app
from retinaguard.auth.jwt import create_access_token, hash_password
from retinaguard.db.base import Base
from retinaguard.db.models import Screening, ScreeningStatus, User, UserRole
from retinaguard.db.session import get_db
from retinaguard.services.audit import AuditLedger


@pytest.fixture
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def client(test_db: AsyncSession):
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_review_flow_confirm_and_audit(client: AsyncClient, test_db: AsyncSession, tmp_path: Path):
    # 1. Setup clinician user
    clinician = User(
        email="ophthalmologist@hospital.in",
        hashed_password=hash_password("OphPass123!"),
        role=UserRole.CLINICIAN,
        is_active=True,
    )
    test_db.add(clinician)
    await test_db.commit()

    token = create_access_token(data={"sub": clinician.id, "role": "clinician"})
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Seed a pending screening record
    screening_payload = {
        "screening_id": "test-scr-001",
        "grade": 2,
        "grade_label_en": "Moderate Non-Proliferative DR (Grade 2)",
        "urgency": "PRIORITY",
        "referral": "REFERABLE",
        "follow_up_interval_months": 3,
        "is_uncertain": False,
        "report_text": "Findings indicate moderate NPDR.",
    }
    screening = Screening(
        id="test-scr-001",
        pseudonym_id="a1b2c3d4e5f60718293a4b5c6d7e8f90",
        result_json=json.dumps(screening_payload),
        model_version="v1.0.0",
        audit_hash="hash-123",
        status=ScreeningStatus.PENDING_REVIEW,
    )
    test_db.add(screening)
    await test_db.commit()

    # 3. List screenings in queue
    list_resp = await client.get("/screenings", headers=headers)
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) >= 1
    found = next((item for item in items if item["id"] == "test-scr-001"), None)
    assert found is not None
    assert found["status"] == "pending_review"
    assert found["grade"] == 2
    assert found["urgency"] == "PRIORITY"

    # 4. Get screening detail
    detail_resp = await client.get("/screenings/test-scr-001", headers=headers)
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == "test-scr-001"
    assert detail["result"]["grade"] == 2

    # 5. Confirm screening
    review_resp = await client.post(
        "/screenings/test-scr-001/review",
        json={"decision": "confirmed", "note": "Verified fundus image microaneurysms."},
        headers=headers,
    )
    assert review_resp.status_code == 200
    rev_data = review_resp.json()
    assert rev_data["decision"] == "confirmed"
    assert rev_data["screening_status"] == "confirmed"

    # 6. Verify screening updated in DB
    updated_resp = await client.get("/screenings/test-scr-001", headers=headers)
    assert updated_resp.status_code == 200
    assert updated_resp.json()["status"] == "confirmed"
    assert updated_resp.json()["result"]["clinician_confirmation"] == "confirmed"


@pytest.mark.asyncio
async def test_review_flow_override_requires_grade(client: AsyncClient, test_db: AsyncSession):
    clinician = User(
        email="clinician2@hospital.in",
        hashed_password=hash_password("Pass123!"),
        role=UserRole.CLINICIAN,
        is_active=True,
    )
    test_db.add(clinician)
    await test_db.commit()

    token = create_access_token(data={"sub": clinician.id, "role": "clinician"})
    headers = {"Authorization": f"Bearer {token}"}

    screening = Screening(
        id="test-scr-002",
        pseudonym_id="hash-pseudonym-002",
        result_json=json.dumps({"grade": 1, "urgency": "ROUTINE"}),
        model_version="v1.0.0",
        audit_hash="hash-456",
        status=ScreeningStatus.PENDING_REVIEW,
    )
    test_db.add(screening)
    await test_db.commit()

    # Attempt override without final_grade -> should return 400
    bad_resp = await client.post(
        "/screenings/test-scr-002/review",
        json={"decision": "overridden", "note": "Missing final grade"},
        headers=headers,
    )
    assert bad_resp.status_code == 400
    assert "final_grade is required" in bad_resp.json()["detail"]

    # Supply final_grade -> 200
    ok_resp = await client.post(
        "/screenings/test-scr-002/review",
        json={"decision": "overridden", "final_grade": 3, "note": "Severe NPDR detected upon manual re-examination"},
        headers=headers,
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["decision"] == "overridden"
    assert ok_resp.json()["screening_status"] == "overridden"
