"""Domain models and schemas for RetinaGuard.

All clinical decision attributes follow the ICDR scale (0-4), AIOS, and RSSDI
guidelines. Patient models support future ABDM/ABHA integration while enforcing
DPDP Act 2023 consent validation.
"""

from datetime import datetime, timezone
from enum import Enum, IntEnum
import re
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator

from retinaguard.core.consent import DPDPProcessingConsent


class ICDRGrade(IntEnum):
    """International Clinical Diabetic Retinopathy Scale."""
    NO_DR = 0
    MILD_NPDR = 1
    MODERATE_NPDR = 2
    SEVERE_NPDR = 3
    PROLIFERATIVE_DR = 4


class ReferralDecision(str, Enum):
    """Referral recommendation generated deterministically by the rules engine."""
    NON_REFERABLE = "NON_REFERABLE"
    REFERABLE = "REFERABLE"
    UNCERTAIN = "UNCERTAIN"


class Urgency(str, Enum):
    """Clinical urgency tiers mapped according to AIOS/RSSDI guidelines."""
    ROUTINE = "ROUTINE"
    PRIORITY = "PRIORITY"
    URGENT = "URGENT"
    EMERGENCY = "EMERGENCY"


class CameraType(str, Enum):
    """Fundus imaging hardware modality."""
    TABLETOP = "TABLETOP"
    SMARTPHONE_PORTABLE = "SMARTPHONE_PORTABLE"
    UNKNOWN = "UNKNOWN"


class EyeSide(str, Enum):
    """Laterality of retinal fundus photograph."""
    RIGHT_OD = "OD"  # Oculus Dexter
    LEFT_OS = "OS"    # Oculus Sinister


class DiabetesType(str, Enum):
    TYPE_1 = "TYPE_1"
    TYPE_2 = "TYPE_2"
    GESTATIONAL = "GESTATIONAL"
    UNKNOWN = "UNKNOWN"


class PatientContext(BaseModel):
    """Patient clinical context accompanying the retinal image.

    Contains metabolic variables impacting progression rate (e.g. HbA1c, pregnancy).
    ABHA fields are prepared for future ABDM connectivity but optional.
    """
    patient_pseudonym_id: str = Field(
        ...,
        description="Pseudonymized or anonymized internal ID (never raw Aadhaar, phone, or name)",
    )
    age: int = Field(..., ge=0, le=125, description="Patient age in years")
    diabetes_type: DiabetesType = Field(default=DiabetesType.TYPE_2)
    diabetes_duration_years: float | None = Field(
        default=None, ge=0.0, le=90.0, description="Duration since clinical diabetes diagnosis"
    )
    hba1c: float | None = Field(
        default=None, ge=3.5, le=25.0, description="Glycated hemoglobin percentage (HbA1c %)"
    )
    is_pregnant: bool = Field(
        default=False, description="Pregnancy status (high DR progression risk)"
    )
    abha_id: str | None = Field(
        default=None,
        description="Ayushman Bharat Health Account number (14-digit format XX-XXXX-XXXX-XXXX)",
    )
    abha_address: str | None = Field(
        default=None,
        description="ABHA PHR address handle (e.g. user@abdm)",
    )
    consent: DPDPProcessingConsent = Field(
        ...,
        description="Explicit DPDP Act 2023 processing consent record",
    )

    @field_validator("abha_id")
    @classmethod
    def validate_abha_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Format check: 14 digits either with or without hyphens
        cleaned = v.replace("-", "").strip()
        if not (cleaned.isdigit() and len(cleaned) == 14):
            raise ValueError("ABHA ID must be a 14-digit identifier (e.g., '12-3456-7890-1234')")
        return v

    @field_validator("abha_address")
    @classmethod
    def validate_abha_address(cls, v: str | None) -> str | None:
        if v is None:
            return None
        pattern = r"^[a-zA-Z0-9_\.\-]+@[a-zA-Z0-9\-]+$"
        if not re.match(pattern, v.strip()):
            raise ValueError("Invalid ABHA address format (expected e.g., 'username@abdm')")
        return v.strip()


class ImageMetadata(BaseModel):
    """Metadata describing the fundus photo and capture conditions."""
    image_sha256: str = Field(..., description="Cryptographic SHA-256 hash of the raw image bytes")
    camera_type: CameraType = Field(default=CameraType.UNKNOWN)
    camera_make_model: str | None = Field(
        default=None, description="e.g. 'Remidio NM-FOP', 'Forus 3nethra Classic', 'Zeiss Visucam'"
    )
    eye_side: EyeSide = Field(..., description="Right (OD) or Left (OS) eye")
    capture_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class QualityAssessment(BaseModel):
    """Result of pre-inference quality gate evaluating artifact, blur, and exposure."""
    is_acceptable: bool = Field(..., description="True if passes minimum diagnostic quality")
    blur_variance: float = Field(..., description="Laplacian focus variance metric")
    underexposed_ratio: float = Field(..., description="Fraction of underexposed pixels")
    overexposed_ratio: float = Field(..., description="Fraction of overexposed/glare pixels")
    mean_luminance: float = Field(..., description="Mean grayscale intensity (0-255)")
    failure_reasons: list[str] = Field(default_factory=list)


class InferenceOutput(BaseModel):
    """Strict ML output: grade and confidence only.

    NEVER includes referral, urgency, or follow-up decision.
    """
    predicted_grade: ICDRGrade
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_probabilities: dict[int, float] = Field(
        ..., description="Calibrated softmax probabilities per ICDR grade (0-4)"
    )


class ScreeningResult(BaseModel):
    """Final decision-support package returned to the clinician.

    Contains deterministic referral, urgency, and interval generated by the rules engine,
    plus immutable plain-language localized summaries.
    """
    screening_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    patient_pseudonym_id: str
    eye_side: EyeSide
    camera_type: CameraType

    # Diagnostic output
    grade: ICDRGrade | None = None
    confidence: float | None = None
    referral: ReferralDecision
    urgency: Urgency
    follow_up_interval_months: int = Field(
        ..., description="Recommended re-screening / referral window in months (0 = immediate)"
    )

    # Fail-safe / Uncertainty flags
    is_uncertain: bool = False
    uncertainty_reasons: list[str] = Field(default_factory=list)
    clinician_review_mandatory: bool = True  # Always True: Decision support only

    # Localized plain-language summaries (RAG / template backed)
    report_en: str
    report_hi: str
    language_requested: str = "en"

    # Audit & Integrity
    image_sha256: str
    previous_audit_hash: str | None = None
    audit_hash: str | None = None

    # Extensible metadata & Agent Additions
    guideline_sources: list[str] = Field(
        default_factory=lambda: ["AIOS 2020", "RSSDI 2022", "ICMR 2018", "ICO 2017"]
    )
    voice_text: str | None = None
    guideline_explanation: str | None = None
    guideline_citations: list[dict] = Field(default_factory=list)
    guideline_source_note: str | None = None

