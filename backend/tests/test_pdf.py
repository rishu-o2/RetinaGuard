"""Tests for server-side PDF referral report generation."""

from __future__ import annotations

import json
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from retinaguard.api.main import app
from retinaguard.auth.jwt import create_access_token, hash_password
from retinaguard.db.base import Base
from retinaguard.db.models import Screening, ScreeningStatus, User, UserRole
from retinaguard.db.session import get_db
from retinaguard.services.pdf import PDFReportService


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


def test_pdf_generation_en():
    svc = PDFReportService()
    sample_result = {
        "patient_pseudonym_id": "anon-hash-12345",
        "eye_side": "OD",
        "camera_type": "TABLETOP",
        "grade": 2,
        "grade_label_en": "Moderate Non-Proliferative DR (Grade 2)",
        "urgency": "PRIORITY",
        "follow_up_interval_months": 3,
        "referral": "REFERABLE",
        "report_text": "RetinaGuard automated triage detected signs of moderate non-proliferative retinopathy.",
        "clinician_confirmation": "pending",
    }
    pdf_bytes = svc.generate(screening_id="scr-001", result_json=sample_result, language="en")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF"), "Must be a valid PDF document"
    assert len(pdf_bytes) > 1000
    # Check disclaimer in bytes
    assert b"Prototype" in pdf_bytes or b"clinically validated" in pdf_bytes


def test_pdf_generation_hi():
    svc = PDFReportService()
    sample_result = {
        "patient_pseudonym_id": "anon-hash-67890",
        "eye_side": "OS",
        "camera_type": "TABLETOP",
        "grade": 3,
        "grade_label_hi": "गंभीर नॉन-प्रोलिफ़ेरेटिव DR (ग्रेड 3)",
        "urgency": "URGENT",
        "follow_up_interval_months": 1,
        "referral": "REFERABLE",
        "report_text": "गंभीर रेटिनोपैथी के लक्षण। तत्काल नेत्र रोग विशेषज्ञ को दिखाएं।",
        "clinician_confirmation": "confirmed",
    }
    pdf_bytes = svc.generate(screening_id="scr-002", result_json=sample_result, language="hi")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_hindi_pdf_contains_devanagari_characters():
    """Verify that Hindi PDF renders and embeds real Devanagari text."""
    import io
    from pypdf import PdfReader

    svc = PDFReportService()
    sample_result = {
        "patient_pseudonym_id": "anon-hash-67890",
        "eye_side": "OS",
        "camera_type": "TABLETOP",
        "grade": 3,
        "grade_label_hi": "गंभीर नॉन-प्रोलिफ़ेरेटिव DR (ग्रेड 3)",
        "urgency": "URGENT",
        "follow_up_interval_months": 1,
        "referral": "REFERABLE",
        "report_text": "गंभीर रेटिनोपैथी के लक्षण। तत्काल नेत्र रोग विशेषज्ञ को दिखाएं।",
        "clinician_confirmation": "confirmed",
    }
    pdf_bytes = svc.generate(screening_id="scr-devanagari", result_json=sample_result, language="hi")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted = "".join((page.extract_text() or "") for page in reader.pages)
    devanagari_chars = [c for c in extracted if 0x0900 <= ord(c) <= 0x097F]
    assert len(devanagari_chars) >= 10, f"Expected Devanagari chars, got {len(devanagari_chars)}"


def test_pdf_no_phi():
    svc = PDFReportService()
    pseudonym_id = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    sample_result = {
        "patient_pseudonym_id": pseudonym_id,
        "eye_side": "OD",
        "camera_type": "TABLETOP",
        "grade": 0,
        "grade_label_en": "No Apparent DR (Grade 0)",
        "urgency": "ROUTINE",
        "follow_up_interval_months": 12,
        "referral": "NON_REFERABLE",
        "report_text": "No signs of diabetic retinopathy detected.",
    }
    pdf_bytes = svc.generate(screening_id="scr-003", result_json=sample_result, language="en")
    # Verify common PHI labels are not present
    assert b"Aadhaar" not in pdf_bytes
    assert b"ABHA" not in pdf_bytes
    assert b"Date of Birth" not in pdf_bytes


@pytest.mark.asyncio
async def test_pdf_endpoint_download(client: AsyncClient, test_db: AsyncSession):
    clinician = User(
        email="doctor.pdf@hospital.in",
        hashed_password=hash_password("Pass123!"),
        role=UserRole.CLINICIAN,
        is_active=True,
    )
    test_db.add(clinician)
    await test_db.commit()

    token = create_access_token(data={"sub": clinician.id, "role": "clinician"})
    headers = {"Authorization": f"Bearer {token}"}

    screening_data = {
        "patient_pseudonym_id": "pseudonym-hash-999",
        "eye_side": "OD",
        "camera_type": "TABLETOP",
        "grade": 1,
        "grade_label_en": "Mild Non-Proliferative DR",
        "urgency": "ROUTINE",
        "referral": "NON_REFERABLE",
        "follow_up_interval_months": 12,
        "report_text": "Annual review advised.",
    }
    screening = Screening(
        id="scr-pdf-test",
        pseudonym_id="pseudonym-hash-999",
        result_json=json.dumps(screening_data),
        model_version="v1.0.0",
        audit_hash="audit-hash-999",
        status=ScreeningStatus.PENDING_REVIEW,
    )
    test_db.add(screening)
    await test_db.commit()

    resp = await client.get("/screenings/scr-pdf-test/report.pdf", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment;" in resp.headers.get("content-disposition", "")
    assert resp.content.startswith(b"%PDF")

