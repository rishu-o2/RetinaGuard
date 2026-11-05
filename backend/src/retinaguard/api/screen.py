"""FastAPI screening endpoint — full DR pipeline.

Pipeline order (ADR-004):
  1. Image decode + SHA-256 hash
  2. Quality gate (classical CV)       → reject if failed
  3. OOD detection                     → reject if out-of-distribution
  4. ONNX inference                    → grade + confidence (or None if model absent)
  5. Clinical rules engine             → referral + urgency (deterministic)
  6. Report generation                 → bilingual text summaries
  7. Audit ledger                      → hash-chained append
  8. Return ScreeningResult JSON

All outcomes set clinician_review_mandatory = True.
"""

from __future__ import annotations

import hashlib
import io
from typing import Annotated, Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from retinaguard.agents.graph import run_screening_pipeline
from retinaguard.core.logging import get_logger
from retinaguard.core.models import (
    CameraType,
    DiabetesType,
    EyeSide,
    ICDRGrade,
    ImageMetadata,
    InferenceOutput,
    PatientContext,
    ReferralDecision,
    ScreeningResult,
    Urgency,
)
from retinaguard.core.consent import DPDPProcessingConsent
from retinaguard.rules.engine import ClinicalRulesEngine
from retinaguard.services.audit import AuditLedger
from retinaguard.services.inference import ONNXInferenceService
from retinaguard.services.ood import OODDetectorService
from retinaguard.services.quality import QualityGateService
from retinaguard.services.report import ReportService

logger = get_logger("retinaguard.api.screen")

router = APIRouter(prefix="/screen", tags=["screening"])

# ---------------------------------------------------------------------------
# Lazy singletons (initialised once at first request)
# ---------------------------------------------------------------------------
_quality_service: Optional[QualityGateService] = None
_ood_service: Optional[OODDetectorService] = None
_inference_service: Optional[ONNXInferenceService] = None
_rules_engine: Optional[ClinicalRulesEngine] = None
_audit_ledger: Optional[AuditLedger] = None


def _get_quality() -> QualityGateService:
    global _quality_service
    if _quality_service is None:
        _quality_service = QualityGateService()
    return _quality_service


def _get_ood() -> OODDetectorService:
    global _ood_service
    if _ood_service is None:
        _ood_service = OODDetectorService()
    return _ood_service


def _get_inference() -> ONNXInferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = ONNXInferenceService()
    return _inference_service


def _get_rules() -> ClinicalRulesEngine:
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = ClinicalRulesEngine()
    return _rules_engine


def _get_ledger() -> AuditLedger:
    global _audit_ledger
    if _audit_ledger is None:
        _audit_ledger = AuditLedger()
    return _audit_ledger


# ---------------------------------------------------------------------------
# Image decode helper
# ---------------------------------------------------------------------------

def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to uint8 RGB (H, W, 3) ndarray."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python required for image decoding") from exc

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Screen endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/",
    summary="Screen a fundus image for Diabetic Retinopathy",
    response_description=(
        "ScreeningResult containing grade, referral decision, urgency, and bilingual report. "
        "clinician_review_mandatory is always True."
    ),
    status_code=status.HTTP_200_OK,
)
async def screen_image(
    image: Annotated[UploadFile, File(description="Fundus photograph (PNG/JPEG, ≥256px)")],
    patient_pseudonym_id: Annotated[str, Form(description="Pseudonymized patient ID (no PHI)")],
    eye_side: Annotated[EyeSide, Form(description="OD (right) or OS (left)")],
    camera_type: Annotated[CameraType, Form()] = CameraType.UNKNOWN,
    age: Annotated[int, Form(ge=0, le=125)] = 40,
    diabetes_type: Annotated[DiabetesType, Form()] = DiabetesType.TYPE_2,
    diabetes_duration_years: Annotated[Optional[float], Form(ge=0)] = None,
    hba1c: Annotated[Optional[float], Form(ge=3.5, le=25.0)] = None,
    is_pregnant: Annotated[bool, Form()] = False,
    language: Annotated[str, Form(description="Preferred report language: 'en' or 'hi'")] = "en",
) -> JSONResponse:
    """End-to-end DR screening pipeline.

    Runs quality gate → OOD check → ONNX inference → clinical rules → audit.
    Always returns a ScreeningResult; unsafe images produce UNCERTAIN outcomes
    with human-readable rejection reasons.
    """
    # ── 1. Read & hash ────────────────────────────────────────────────────
    raw_bytes = await image.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    image_sha256 = _sha256(raw_bytes)

    try:
        img_rgb = _decode_image(raw_bytes)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Image decode failed: {exc}") from exc

    h, w = img_rgb.shape[:2]
    image_metadata = ImageMetadata(
        image_sha256=image_sha256,
        camera_type=camera_type,
        eye_side=eye_side,
        height=h,
        width=w,
    )

    consent = DPDPProcessingConsent(
        patient_identifier_hash=hashlib.sha256(patient_pseudonym_id.encode()).hexdigest(),
        purpose="DIABETIC_RETINOPATHY_SCREENING",  # type: ignore[arg-type]
        notice_version="1.0-IN",
    )
    patient_ctx = PatientContext(
        patient_pseudonym_id=patient_pseudonym_id,
        age=age,
        diabetes_type=diabetes_type,
        diabetes_duration_years=diabetes_duration_years,
        hba1c=hba1c,
        is_pregnant=is_pregnant,
        consent=consent,
    )

    logger.info(
        "Screen request: patient=%s eye=%s camera=%s size=%dx%d",
        patient_pseudonym_id, eye_side, camera_type, w, h,
    )

    # ── Run LangGraph 5-Agent Pipeline ────────────────────────────────────
    result = run_screening_pipeline(
        image_bytes=raw_bytes,
        image_metadata=image_metadata,
        patient_context=patient_ctx,
        language_requested=language,
        image_rgb=img_rgb,
    )

    # ── Return backward-compatible JSON representation ───────────────────
    payload = ReportService.build_json_report(result)
    payload["report_requested_lang"] = language
    payload["report_text"] = result.report_hi if language == "hi" else result.report_en
    payload["voice_text"] = result.voice_text
    payload["guideline_explanation"] = result.guideline_explanation
    payload["guideline_citations"] = result.guideline_citations
    payload["guideline_source_note"] = result.guideline_source_note

    # ── Persist to Screening Database (as pending review) ─────────────────
    try:
        import json as _json
        from retinaguard.db.models import Screening, ScreeningStatus
        from retinaguard.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
        async with session_factory() as db:
            screening_record = Screening(
                id=str(result.screening_id),
                pseudonym_id=hashlib.sha256(patient_pseudonym_id.encode()).hexdigest(),
                result_json=_json.dumps(payload),
                model_version=str(getattr(result, "model_version", "v1.0.0")),
                audit_hash=str(result.audit_hash or ""),
                status=ScreeningStatus.PENDING_REVIEW,
            )
            db.add(screening_record)
            await db.commit()
    except Exception as exc:
        logger.warning("Could not persist screening record to database: %s", exc)

    logger.info(
        "Screening completed: screening_id=%s grade=%s referral=%s uncertain=%s",
        result.screening_id, result.grade, result.referral, result.is_uncertain,
    )
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Audit verification endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/audit/verify",
    summary="Verify the integrity of the audit chain",
    tags=["audit"],
)
def verify_audit_chain() -> dict:
    """Re-computes and verifies the entire hash chain of the audit ledger.

    Returns True if intact; lists any corruption errors if not.
    """
    intact, errors = _get_ledger().verify_chain()
    return {
        "chain_intact": intact,
        "errors": errors,
        "error_count": len(errors),
    }


@router.get(
    "/audit/tail",
    summary="Return the last N audit entries",
    tags=["audit"],
)
def audit_tail(n: int = 20) -> dict:
    """Return the most recent audit entries (PHI-free: pseudonym IDs + hashes only)."""
    entries = _get_ledger().tail(n=min(n, 100))
    return {
        "count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
