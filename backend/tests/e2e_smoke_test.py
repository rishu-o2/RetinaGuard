"""End-to-End Smoke Test for RetinaGuard.

Runs the complete clinical flow against the FastAPI stack:
1. Seed admin & log in as admin
2. Admin creates a clinician user
3. Clinician logs in
4. Screen a real APTOS validation image (data/raw/aptos/train_images/0083ee8054ee.png)
5. List pending screenings and locate the new screening
6. Clinician reviews and confirms the screening
7. Download bilingual PDF report (Hindi)
8. Verify Devanagari text in PDF
9. Render page 1 to PNG for visual inspection
10. Verify audit chain integrity
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import fitz  # PyMuPDF
import pytest
from httpx import ASGITransport, AsyncClient
from pypdf import PdfReader

from retinaguard.api.main import app, lifespan
from retinaguard.core.config import _find_repo_root
from retinaguard.db.base import Base
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import get_db, get_engine, get_sessionmaker


@pytest.mark.asyncio
async def test_complete_e2e_smoke_flow():
    repo_root = _find_repo_root()
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure database tables exist
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Step 1: Check health
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"
        print("[PASS] 1. Health check succeeded")

        # Step 2: Seed and log in as Admin
        admin_email = "admin@hospital.in"
        admin_pass = "AdminSeedPassword123!"
        session_factory = get_sessionmaker()
        async with session_factory() as db:
            from retinaguard.auth.jwt import hash_password
            from sqlalchemy import select
            res = await db.execute(select(User).where(User.email == admin_email))
            existing_admin = res.scalar_one_or_none()
            if not existing_admin:
                admin_user = User(
                    email=admin_email,
                    hashed_password=hash_password(admin_pass),
                    role=UserRole.ADMIN,
                    is_active=True,
                )
                db.add(admin_user)
                await db.commit()

        login_resp = await client.post(
            "/auth/login",
            data={"username": admin_email, "password": admin_pass},
        )
        assert login_resp.status_code == 200, f"Admin login failed: {login_resp.text}"
        admin_token = login_resp.json()["access_token"]
        print(f"[PASS] 2. Logged in as seeded admin ({admin_email})")

        # Step 3: Admin creates a clinician user
        clinician_email = "clinician.e2e@hospital.in"
        clinician_pass = "DoctorPass123!"
        create_user_resp = await client.post(
            "/auth/users",
            json={
                "email": clinician_email,
                "password": clinician_pass,
                "role": "clinician",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert create_user_resp.status_code in (201, 400)  # 400 if user exists from previous run
        print(f"[PASS] 3. Created clinician user ({clinician_email})")

        # Step 4: Log in as the clinician
        clin_login_resp = await client.post(
            "/auth/login",
            data={"username": clinician_email, "password": clinician_pass},
        )
        assert clin_login_resp.status_code == 200
        clinician_token = clin_login_resp.json()["access_token"]
        clinician_headers = {"Authorization": f"Bearer {clinician_token}"}
        print("[PASS] 4. Logged in as clinician")

        # Step 5: Screen a real APTOS validation image
        aptos_img_path = repo_root / "data" / "raw" / "aptos" / "train_images" / "0083ee8054ee.png"
        assert aptos_img_path.exists(), f"APTOS validation image not found at {aptos_img_path}"
        with open(aptos_img_path, "rb") as f:
            img_bytes = f.read()

        screen_data = {
            "patient_pseudonym_id": "VAL-PAT-0083EE",
            "eye_side": "OD",
            "camera_type": "TABLETOP",
            "age": "58",
            "diabetes_type": "TYPE_2",
            "diabetes_duration_years": "14",
            "hba1c": "8.8",
            "is_pregnant": "false",
            "language": "hi",
        }
        screen_resp = await client.post(
            "/screen/",
            data=screen_data,
            files={"image": ("0083ee8054ee.png", io.BytesIO(img_bytes), "image/png")},
            headers=clinician_headers,
        )
        assert screen_resp.status_code == 200, f"Screening failed: {screen_resp.text}"
        screen_result = screen_resp.json()
        screening_id = screen_result["screening_id"]
        assert screening_id is not None
        assert screen_result["clinician_review_mandatory"] is True
        print(f"[PASS] 5. Screened APTOS val image 0083ee8054ee.png: screening_id={screening_id}, grade={screen_result.get('grade')}, referral={screen_result.get('referral')}")

        # Step 6: List pending screenings and locate the new screening
        queue_resp = await client.get("/screenings", headers=clinician_headers)
        assert queue_resp.status_code == 200
        pending_list = queue_resp.json()
        matching = [s for s in pending_list if s["id"] == str(screening_id)]
        assert len(matching) > 0, "Created screening must appear in pending review queue"
        assert matching[0]["status"] == "pending_review"
        print(f"[PASS] 6. Found screening in pending review queue (total pending: {len(pending_list)})")

        # Step 7: Clinician confirms the screening
        review_resp = await client.post(
            f"/screenings/{screening_id}/review",
            json={
                "decision": "confirmed",
                "note": "Clinician confirmed severe DR presentation with proliferative changes. Refer immediately.",
            },
            headers=clinician_headers,
        )
        assert review_resp.status_code == 200
        review_data = review_resp.json()
        assert review_data["status"] == "success"
        assert review_data["screening_status"] == "confirmed"
        print(f"[PASS] 7. Clinician reviewed and confirmed screening {screening_id}")

        # Step 8: Download the PDF report in Hindi
        pdf_resp = await client.get(
            f"/screenings/{screening_id}/report.pdf?lang=hi",
            headers=clinician_headers,
        )
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        pdf_bytes = pdf_resp.content
        assert len(pdf_bytes) > 1000

        # Save PDF to disk
        pdf_path = reports_dir / "smoke_report_hi.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"[PASS] 8. Downloaded Hindi PDF report ({len(pdf_bytes)} bytes) -> {pdf_path}")

        # Step 9: Extract text and verify Devanagari characters
        reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted_text = "".join((p.extract_text() or "") for p in reader.pages)
        devanagari_chars = [c for c in extracted_text if 0x0900 <= ord(c) <= 0x097F]
        assert len(devanagari_chars) >= 10, f"Expected Devanagari characters, found {len(devanagari_chars)}"
        print(f"[PASS] 9. Verified Devanagari Unicode characters embedded in PDF ({len(devanagari_chars)} characters)")

        # Step 10: Convert first page to PNG
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=150)
        png_path = reports_dir / "smoke_report_hi_page1.png"
        pix.save(str(png_path))
        doc.close()
        print(f"[PASS] 10. Converted first page to PNG ({pix.width}x{pix.height}) -> {png_path}")

        # Copy PNG to artifact directory for presentation
        artifact_dir = Path(r"C:\Users\hp\.gemini\antigravity\brain\0fffc3b9-96ae-4fec-8b9f-26d6505e3a54")
        if artifact_dir.exists():
            shutil.copy2(png_path, artifact_dir / "smoke_report_hi_page1.png")
            print(f"[PASS] Copied page 1 PNG to artifact directory: {artifact_dir / 'smoke_report_hi_page1.png'}")

        # Step 11: Verify audit chain
        audit_resp = await client.get("/screen/audit/verify")
        assert audit_resp.status_code == 200
        audit_data = audit_resp.json()
        assert audit_data["chain_intact"] is True, f"Audit chain broken: {audit_data}"
        print(f"[PASS] 11. Audit chain verified intact (errors: {audit_data['error_count']})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_complete_e2e_smoke_flow())
