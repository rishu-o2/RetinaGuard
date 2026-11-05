"""Pytest fixtures for RetinaGuard tests."""

import os

# Default to allowing anonymous screening for unit/integration tests that don't pass headers
os.environ.setdefault("ALLOW_ANONYMOUS_SCREENING", "true")

import pytest
from retinaguard.core.consent import ConsentPurpose, DPDPProcessingConsent
from retinaguard.core.models import (
    CameraType,
    DiabetesType,
    EyeSide,
    ImageMetadata,
    PatientContext,
)


@pytest.fixture
def valid_dpdp_consent() -> DPDPProcessingConsent:
    """Fixture providing a valid unrevoked DPDP consent artefact."""
    return DPDPProcessingConsent(
        patient_identifier_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        purpose=ConsentPurpose.RETINOPATHY_SCREENING,
        notice_version="1.0-IN",
        consent_language="hi",
    )


@pytest.fixture
def sample_patient_context(valid_dpdp_consent: DPDPProcessingConsent) -> PatientContext:
    """Fixture providing standard Indian diabetic patient context."""
    return PatientContext(
        patient_pseudonym_id="PAT-DEL-2026-0042",
        age=54,
        diabetes_type=DiabetesType.TYPE_2,
        diabetes_duration_years=11.5,
        hba1c=8.4,
        is_pregnant=False,
        abha_id="91-2345-6789-0123",
        abha_address="patient54@abdm",
        consent=valid_dpdp_consent,
    )


@pytest.fixture
def sample_image_metadata() -> ImageMetadata:
    """Fixture providing fundus image metadata from a portable camera."""
    return ImageMetadata(
        image_sha256="2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
        camera_type=CameraType.SMARTPHONE_PORTABLE,
        camera_make_model="Remidio NM-FOP",
        eye_side=EyeSide.RIGHT_OD,
        width=1920,
        height=1080,
    )

