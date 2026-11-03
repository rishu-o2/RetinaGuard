"""Digital Personal Data Protection (DPDP) Act 2023 consent module.

Implements purpose-bound explicit consent artefacts required for processing
personal health data under Indian privacy regulations.
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class ConsentPurpose(str, Enum):
    """Permitted purposes for health data processing."""
    RETINOPATHY_SCREENING = "DIABETIC_RETINOPATHY_SCREENING"
    CLINICAL_DECISION_SUPPORT = "CLINICAL_DECISION_SUPPORT"
    AUDIT_AND_COMPLIANCE = "REGULATORY_AUDIT_AND_COMPLIANCE"


class DPDPProcessingConsent(BaseModel):
    """Explicit consent artefact compliant with DPDP Act 2023 requirements.

    Ensures that processing of retinal images and patient metabolic indicators
    is strictly purpose-limited, timestamped, and traceable.
    """
    consent_id: UUID = Field(default_factory=uuid4, description="Unique consent record identifier")
    patient_identifier_hash: str = Field(
        ...,
        description="Irreversible pseudonymized hash of patient identifier (never plaintext name/Aadhaar)",
    )
    purpose: ConsentPurpose = Field(
        default=ConsentPurpose.RETINOPATHY_SCREENING,
        description="Specific purpose for which consent was granted",
    )
    notice_version: str = Field(
        default="1.0-IN",
        description="Version of the patient privacy notice presented to the patient",
    )
    consent_language: str = Field(
        default="hi",
        description="ISO 639-1 code of language in which consent notice was provided (e.g. 'en', 'hi', 'ta')",
    )
    granted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of explicit affirmative consent",
    )
    is_revoked: bool = Field(
        default=False,
        description="Whether consent has been subsequently withdrawn by the Data Principal",
    )
    revoked_at: datetime | None = Field(
        default=None,
        description="Timestamp of revocation if applicable",
    )

    def is_valid_for_processing(self) -> bool:
        """Verify whether consent is currently active and unrevoked."""
        return not self.is_revoked

