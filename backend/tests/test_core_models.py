"""Tests for core domain models and validation."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from retinaguard.core.consent import ConsentPurpose, DPDPProcessingConsent
from retinaguard.core.models import (
    CameraType,
    DiabetesType,
    EyeSide,
    ICDRGrade,
    InferenceOutput,
    PatientContext,
    ReferralDecision,
    ScreeningResult,
    Urgency,
)


def test_patient_context_valid(sample_patient_context: PatientContext) -> None:
    """Verify that a properly constructed PatientContext passes validation."""
    assert sample_patient_context.age == 54
    assert sample_patient_context.hba1c == 8.4
    assert sample_patient_context.abha_id == "91-2345-6789-0123"
    assert sample_patient_context.abha_address == "patient54@abdm"
    assert sample_patient_context.consent.is_valid_for_processing() is True


def test_patient_context_invalid_abha_id(valid_dpdp_consent: DPDPProcessingConsent) -> None:
    """Verify that non-14-digit ABHA numbers are rejected."""
    with pytest.raises(ValidationError, match="ABHA ID must be a 14-digit identifier"):
        PatientContext(
            patient_pseudonym_id="PAT-001",
            age=45,
            consent=valid_dpdp_consent,
            abha_id="12345",  # Invalid length
        )


def test_patient_context_invalid_abha_address(valid_dpdp_consent: DPDPProcessingConsent) -> None:
    """Verify that invalid ABHA handle formats are rejected."""
    with pytest.raises(ValidationError, match="Invalid ABHA address format"):
        PatientContext(
            patient_pseudonym_id="PAT-001",
            age=45,
            consent=valid_dpdp_consent,
            abha_address="invalid_handle_without_domain",
        )


def test_patient_context_age_bounds(valid_dpdp_consent: DPDPProcessingConsent) -> None:
    """Verify age boundaries (0 to 125)."""
    with pytest.raises(ValidationError):
        PatientContext(
            patient_pseudonym_id="PAT-001",
            age=-5,
            consent=valid_dpdp_consent,
        )


def test_dpdp_consent_revocation(valid_dpdp_consent: DPDPProcessingConsent) -> None:
    """Verify right to erasure / revocation mechanics under DPDP Act 2023."""
    assert valid_dpdp_consent.is_valid_for_processing() is True

    valid_dpdp_consent.is_revoked = True
    valid_dpdp_consent.revoked_at = datetime.now(timezone.utc)

    assert valid_dpdp_consent.is_valid_for_processing() is False


def test_inference_output_bounds() -> None:
    """Verify ML inference output does not accept confidence outside [0, 1]."""
    with pytest.raises(ValidationError):
        InferenceOutput(
            predicted_grade=ICDRGrade.MILD_NPDR,
            confidence=1.45,  # Exceeds 1.0
            raw_probabilities={0: 0.1, 1: 0.9},
        )


def test_screening_result_defaults() -> None:
    """Verify screening result defaults to requiring clinician review."""
    result = ScreeningResult(
        patient_pseudonym_id="PAT-001",
        eye_side=EyeSide.RIGHT_OD,
        camera_type=CameraType.SMARTPHONE_PORTABLE,
        grade=ICDRGrade.MILD_NPDR,
        confidence=0.88,
        referral=ReferralDecision.NON_REFERABLE,
        urgency=Urgency.ROUTINE,
        follow_up_interval_months=12,
        report_en="Routine rescreening in 12 months recommended.",
        report_hi="12 महीने में नियमित जांच की सिफारिश की जाती है।",
        image_sha256="fake_sha256_hash",
    )
    assert result.clinician_review_mandatory is True
    assert result.is_uncertain is False

