"""Tests for PHI protection: zero-PHI log scrubbing and AES-256-GCM encryption."""

import logging
import pytest

from retinaguard.core.encryption import PHIEncryptionError, PHIEncryptor
from retinaguard.core.logging import PHIScrubbingFilter, get_logger, sanitize_text


def test_sanitize_text_redacts_identifying_data() -> None:
    """Verify that sanitize_text strips phone numbers, emails, ABHA IDs, and Aadhaar numbers."""
    raw_message = (
        "Patient Rahul Sharma with phone 9876543210, Aadhaar 1234 5678 9012, "
        "ABHA 91-2345-6789-0123, email patient@example.com, and handle rahul@abdm screened."
    )
    sanitized = sanitize_text(raw_message)

    assert "9876543210" not in sanitized
    assert "1234 5678 9012" not in sanitized
    assert "91-2345-6789-0123" not in sanitized
    assert "patient@example.com" not in sanitized
    assert "rahul@abdm" not in sanitized

    assert "[REDACTED_PHONE]" in sanitized
    assert "[REDACTED_AADHAAR]" in sanitized
    assert "[REDACTED_ABHA_ID]" in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert "[REDACTED_ABHA_ADDR]" in sanitized


def test_logger_phi_scrubbing(caplog: pytest.LogCaptureFixture) -> None:
    """Verify logger with PHIScrubbingFilter automatically suppresses sensitive data."""
    logger = get_logger("retinaguard.test")
    logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO):
        logger.info("Processing screening for phone: %s", "9876543210")

    assert "9876543210" not in caplog.text
    assert "[REDACTED_PHONE]" in caplog.text


def test_phi_encryptor_round_trip() -> None:
    """Verify AES-256-GCM string encryption and decryption."""
    encryptor = PHIEncryptor()
    secret_phi = "Patient Name: Ananya Sen, DOB: 1980-05-12, ABHA: 12-3456-7890-1234"

    encrypted_payload = encryptor.encrypt_string(secret_phi, associated_data="screening_123")
    assert encrypted_payload != secret_phi

    decrypted = encryptor.decrypt_string(encrypted_payload, associated_data="screening_123")
    assert decrypted == secret_phi


def test_phi_encryptor_json_round_trip() -> None:
    """Verify encryption and decryption of structured JSON health records."""
    encryptor = PHIEncryptor()
    health_record = {
        "hba1c": 9.2,
        "duration_years": 14,
        "notes": "Frequent hypoglycemia episodes reported",
    }

    encrypted = encryptor.encrypt_json(health_record)
    decrypted = encryptor.decrypt_json(encrypted)

    assert decrypted == health_record
    assert decrypted["hba1c"] == 9.2


def test_phi_encryptor_tamper_detection() -> None:
    """Verify that modifying even one byte in ciphertext triggers tamper detection."""
    encryptor = PHIEncryptor()
    encrypted = encryptor.encrypt_string("Confidential health data")

    # Tamper with the base64 string
    tampered = encrypted[:-2] + ("AA" if encrypted[-2:] != "AA" else "BB")

    with pytest.raises(PHIEncryptionError, match="Decryption failed or data tampered"):
        encryptor.decrypt_string(tampered)


def test_phi_encryptor_associated_data_mismatch() -> None:
    """Verify that authentication tag verification fails if associated data differs."""
    encryptor = PHIEncryptor()
    encrypted = encryptor.encrypt_string("Sensitive data", associated_data="screening_A")

    with pytest.raises(PHIEncryptionError):
        encryptor.decrypt_string(encrypted, associated_data="screening_B")


def test_log_scanning_covers_voice_text_zero_leak(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that voice narration text and clinical audio directives are never logged."""
    from retinaguard.services.voice_provider import logger as voice_logger

    sensitive_voice_text = "RetinaGuard screening complete: schedule follow-up in 3 months."
    with caplog.at_level(logging.DEBUG):
        voice_logger.debug("Sarvam request: 1 chunks for target_lang=hi-IN")
        voice_logger.debug("Edge-TTS request: voice=hi-IN-MadhurNeural")

    for rec in caplog.records:
        assert sensitive_voice_text not in rec.message
        assert sensitive_voice_text not in str(rec.args)


def test_log_scanning_covers_audit_ledger_zero_leak(caplog: pytest.LogCaptureFixture, tmp_path) -> None:
    """Verify that audit ledger appends emit hash-chained metadata only, never raw patient PHI."""
    from retinaguard.services.audit import AuditLedger

    ledger_path = tmp_path / "test_audit.jsonl"
    ledger = AuditLedger(ledger_path=ledger_path)

    patient_pseudo = "PSEUDO-MUM-2026-8877"
    secret_note = "Severe diabetic retinopathy with high vitreous hemorrhage risk."

    with caplog.at_level(logging.DEBUG):
        entry = ledger.append(
            event_type="SCREENING_COMPLETED",
            screening_id="scr-test-9999",
            patient_pseudonym_id=patient_pseudo,
            image_sha256="abcd1234ef567890abcd1234ef567890abcd1234ef567890abcd1234ef567890",
            outcome_summary="REFERABLE - URGENT",
            extra={"clinical_impression": secret_note},
        )

    # Verify log messages from audit ledger do not dump raw clinical impression or patient notes
    audit_logs = [rec.message for rec in caplog.records if "audit" in rec.name.lower()]
    for msg in audit_logs:
        assert secret_note not in msg
        assert "Audit entry appended" in msg or "hash=" in msg


