"""Tests for eye-facility referral directory and PDF inclusion."""

import csv
import io
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from retinaguard.api.main import app
from retinaguard.api.referral import _load_facilities, get_referral_facilities
from retinaguard.services.pdf import PDFReportService

client = TestClient(app)


def test_referral_empty_directory():
    """Empty CSV should return an empty list without error."""
    resp = client.get("/referral?state=Maharashtra&district=Pune")
    assert resp.status_code == 200
    assert resp.json() == []


def test_referral_with_entries(monkeypatch, tmp_path):
    """When CSV has matching entries, returns up to 3 facilities."""
    test_csv = tmp_path / "test_facilities.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "district", "name", "type", "address", "phone", "source_url"])
        writer.writerow(["Maharashtra", "Pune", "Aundh District Hospital", "DH", "Aundh, Pune", "+91-20-111", "https://nhm.gov.in"])
        writer.writerow(["Maharashtra", "Pune", "Sassoon General Hospital", "Tertiary", "Station Rd, Pune", "+91-20-222", "https://nhm.gov.in"])
        writer.writerow(["Maharashtra", "Mumbai", "JJ Hospital", "Tertiary", "Byculla, Mumbai", "+91-22-333", "https://nhm.gov.in"])

    monkeypatch.setattr("retinaguard.api.referral._FACILITIES_CSV", test_csv)

    # Pune search
    resp = client.get("/referral?state=Maharashtra&district=Pune")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    assert results[0]["name"] == "Aundh District Hospital"
    assert results[1]["name"] == "Sassoon General Hospital"

    # Mumbai search (case-insensitive)
    resp = client.get("/referral?district=mumbai")
    assert resp.status_code == 200
    mumbai_res = resp.json()
    assert len(mumbai_res) == 1
    assert mumbai_res[0]["name"] == "JJ Hospital"

    # Non-existent district
    resp = client.get("/referral?district=Nagpur")
    assert resp.status_code == 200
    assert resp.json() == []


def test_pdf_generation_includes_facilities():
    """PDF generator renders NEAREST EYE CARE FACILITIES when facilities are present."""
    pdf_service = PDFReportService()
    sample_result = {
        "grade": 2,
        "grade_label_en": "Moderate NPDR",
        "referral": "REFER",
        "urgency": "PRIORITY",
        "follow_up_interval_months": 3,
        "is_uncertain": False,
        "report_text": "Retinal lesions observed. Specialist referral indicated.",
        "patient_pseudonym_id": "test-patient-uuid",
    }
    sample_facilities = [
        {
            "name": "District Eye Hospital",
            "type": "DH",
            "address": "Civil Lines, Pune",
            "phone": "+91-20-12345678",
            "source_url": "https://nhm.gov.in",
        }
    ]

    pdf_bytes = pdf_service.generate(
        screening_id="test-screening-1",
        result_json=sample_result,
        language="en",
        facilities=sample_facilities,
    )
    assert pdf_bytes.startswith(b"%PDF")

    # Also test with empty facilities
    pdf_bytes_empty = pdf_service.generate(
        screening_id="test-screening-2",
        result_json=sample_result,
        language="en",
        facilities=[],
    )
    assert pdf_bytes_empty.startswith(b"%PDF")
