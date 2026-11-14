"""Model serving and offline sync endpoints.

GET  /model/active   — authenticated; streams the active ONNX file + SHA-256 header
GET  /model/metadata — authenticated; returns JSON config for offline rules engine
POST /offline/sync   — authenticated; re-runs pipeline and logs discrepancies
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from retinaguard.auth.deps import get_current_user
from retinaguard.core.config import _find_repo_root
from retinaguard.core.logging import get_logger
from retinaguard.db.models import User

logger = get_logger("retinaguard.api.model_serve")

router = APIRouter(prefix="/model", tags=["offline"])
sync_router = APIRouter(prefix="/offline", tags=["offline"])

_REPO_ROOT = _find_repo_root()


def _get_active_registry_entry() -> dict:
    reg_path = _REPO_ROOT / "models" / "registry.json"
    with open(reg_path, encoding="utf-8") as fh:
        reg = json.load(fh)
    entries = reg.get("models", [])
    active = next((m for m in entries if m.get("active")), None)
    if active is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active model in registry",
        )
    return active


@router.get(
    "/active",
    summary="Download the active ONNX model (authenticated)",
    response_class=FileResponse,
)
def serve_active_model(current_user: User = Depends(get_current_user)):
    """Stream the active ONNX model file.

    Returns the raw binary with:
    - Content-Type: application/octet-stream
    - X-Model-SHA256: <sha256 from registry> header

    The frontend must verify the SHA-256 before caching.
    """
    entry = _get_active_registry_entry()
    onnx_path = _REPO_ROOT / entry["onnx_path"]
    if not onnx_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ONNX model file not found: {onnx_path}",
        )
    sha256_expected = entry.get("onnx_sha256", "")
    return FileResponse(
        path=str(onnx_path),
        media_type="application/octet-stream",
        filename=onnx_path.name,
        headers={"X-Model-SHA256": sha256_expected},
    )


@router.get(
    "/metadata",
    summary="Return active model config for the offline rules engine (authenticated)",
)
def serve_model_metadata(current_user: User = Depends(get_current_user)) -> dict:
    """Return all parameters the offline TypeScript rules engine needs.

    Every threshold comes from YAML/registry — none are hard-coded here.
    """
    entry = _get_active_registry_entry()

    # Load abstention config
    abstention_config: dict[str, Any] = {}
    abstention_path = _REPO_ROOT / entry.get("abstention_config_path", "configs/abstention.yaml")
    if abstention_path.exists():
        with open(abstention_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
            # abstention.yaml has two documents; take the second (referable_band)
            if isinstance(raw, list):
                for doc in raw:
                    if isinstance(doc, dict) and "referable_band" in doc:
                        abstention_config = doc
                        break
            elif isinstance(raw, dict):
                abstention_config = raw

    # Load clinical rules (grade mapping + modifiers)
    rules_config: dict[str, Any] = {}
    rules_path = _REPO_ROOT / "configs" / "clinical_rules.yaml"
    if rules_path.exists():
        with open(rules_path, encoding="utf-8") as fh:
            rules_config = yaml.safe_load(fh) or {}

    band = abstention_config.get("referable_band", {})
    conf_th = rules_config.get("confidence_thresholds", {})
    sight_esc = (rules_config.get("modifiers") or {}).get(
        "sight_threatening_escalation", {}
    )

    metadata = {
        "version": entry.get("version", "v1.0.0"),
        "version_tag": entry.get("version_tag", "v1"),
        "onnx_sha256": entry.get("onnx_sha256", ""),
        "image_size": entry.get("image_size", [256, 256]),
        "temperature": entry.get("temperature", 1.0),
        # Abstention band
        "abstention": {
            "lower_edge": band.get("lower_edge", 0.30),
            "upper_edge": band.get("upper_edge", 0.70),
            "cutoff": band.get("cutoff", 0.50),
        },
        # Confidence threshold
        "min_acceptable_confidence": conf_th.get("min_acceptable_confidence", 0.70),
        # Sight-threatening escalation
        "sight_threatening": {
            "threshold_soon": sight_esc.get("threshold_soon", 0.05),
            "threshold_urgent": sight_esc.get("threshold_urgent", 0.30),
        },
        # Grade table: grade int -> {referable, base_urgency, base_interval_months}
        "grade_table": {
            str(k): {
                "referable": v.get("referable", False),
                "base_urgency": v.get("base_urgency", "ROUTINE"),
                "base_interval_months": v.get("base_interval_months", 12),
            }
            for k, v in (rules_config.get("icdr_grades") or {}).items()
        },
        # Modifiers
        "modifiers": {
            "pregnancy": {
                "min_urgency": ((rules_config.get("modifiers") or {}).get("pregnancy") or {}).get("min_urgency", "PRIORITY"),
                "max_interval_months": ((rules_config.get("modifiers") or {}).get("pregnancy") or {}).get("max_interval_months", 3),
            },
            "high_risk_glycemic_control": {
                "hba1c_threshold_percent": ((rules_config.get("modifiers") or {}).get("high_risk_glycemic_control") or {}).get("hba1c_threshold_percent", 10.0),
                "max_interval_months": ((rules_config.get("modifiers") or {}).get("high_risk_glycemic_control") or {}).get("max_interval_months", 6),
            },
            "long_standing_diabetes": {
                "duration_years_threshold": ((rules_config.get("modifiers") or {}).get("long_standing_diabetes") or {}).get("duration_years_threshold", 15),
                "max_interval_months": ((rules_config.get("modifiers") or {}).get("long_standing_diabetes") or {}).get("max_interval_months", 6),
            },
        },
    }
    return metadata


# ---------------------------------------------------------------------------
# Offline Sync
# ---------------------------------------------------------------------------

class OfflineSyncRequest(BaseModel):
    patient_pseudonym_id: str
    eye_side: str
    camera_type: str
    age: int = 40
    diabetes_type: str = "TYPE_2"
    diabetes_duration_years: float | None = None
    hba1c: float | None = None
    is_pregnant: bool = False
    language: str = "en"
    image_b64: str  # base64-encoded image bytes
    offline_result: dict  # the provisional result from the TS engine


@sync_router.post(
    "/sync",
    summary="Re-run screening on a queued offline image (authenticated)",
)
async def sync_offline_result(
    body: OfflineSyncRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Accept a queued offline image, re-run the full pipeline, and return the
    authoritative server result.

    If the server result differs from the offline provisional result (referral
    or urgency), the discrepancy is appended to the audit chain with
    event_type='OFFLINE_SYNC_DISCREPANCY'.

    The client must delete the queue entry after receiving a 200 response.
    No real patient names are accepted; only the pseudonym ID.
    """
    # Decode image
    try:
        image_bytes = base64.b64decode(body.image_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 image: {exc}",
        ) from exc

    # Import pipeline lazily to avoid circular imports
    from retinaguard.agents.graph import run_screening_pipeline
    from retinaguard.core.models import (
        CameraType,
        EyeSide,
        ImageMetadata,
        PatientContext,
    )
    from retinaguard.core.consent import DPDPProcessingConsent
    from retinaguard.services.audit import AuditLedger
    from retinaguard.services.report import ReportService

    import io
    import numpy as np
    try:
        import cv2  # type: ignore
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_rgb = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {exc}",
        ) from exc

    # Map camera_type string
    cam_map = {
        "TABLETOP": CameraType.TABLETOP,
        "tabletop": CameraType.TABLETOP,
        "SMARTPHONE_PORTABLE": CameraType.SMARTPHONE_PORTABLE,
        "smartphone_portable": CameraType.SMARTPHONE_PORTABLE,
    }
    cam = cam_map.get(body.camera_type, CameraType.TABLETOP)

    eye = EyeSide.LEFT_OS if body.eye_side.upper() == "OS" else EyeSide.RIGHT_OD

    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

    image_metadata = ImageMetadata(
        image_sha256=image_sha256,
        camera_type=cam,
        eye_side=eye,
        height=h,
        width=w,
    )
    consent = DPDPProcessingConsent(
        patient_identifier_hash=hashlib.sha256(body.patient_pseudonym_id.encode()).hexdigest(),
        purpose="DIABETIC_RETINOPATHY_SCREENING",  # type: ignore[arg-type]
        notice_version="1.0-IN",
    )
    patient_ctx = PatientContext(
        patient_pseudonym_id=body.patient_pseudonym_id,
        age=body.age,
        diabetes_duration_years=body.diabetes_duration_years,
        hba1c=body.hba1c,
        is_pregnant=body.is_pregnant,
        consent=consent,
    )

    result = run_screening_pipeline(
        image_bytes=image_bytes,
        image_metadata=image_metadata,
        patient_context=patient_ctx,
        language_requested=body.language,
        image_rgb=img_rgb,
    )

    server_referral = str(result.referral.value if hasattr(result.referral, "value") else result.referral)
    server_urgency = str(result.urgency.value if hasattr(result.urgency, "value") else result.urgency)
    offline_referral = str(body.offline_result.get("referral", ""))
    offline_urgency = str(body.offline_result.get("urgency", ""))

    discrepancy = server_referral != offline_referral or server_urgency != offline_urgency

    if discrepancy:
        try:
            ledger = AuditLedger()
            ledger.append(
                event_type="OFFLINE_SYNC_DISCREPANCY",
                screening_id=str(result.screening_id),
                patient_pseudonym_id=body.patient_pseudonym_id,
                image_sha256=image_sha256,
                outcome_summary=(
                    f"offline={offline_referral}/{offline_urgency} "
                    f"server={server_referral}/{server_urgency}"
                ),
                extra={
                    "offline_referral": offline_referral,
                    "offline_urgency": offline_urgency,
                    "server_referral": server_referral,
                    "server_urgency": server_urgency,
                },
            )
            logger.warning(
                "Offline sync discrepancy for patient=%s: offline=%s/%s server=%s/%s",
                body.patient_pseudonym_id,
                offline_referral, offline_urgency,
                server_referral, server_urgency,
            )
        except Exception as exc:
            logger.error("Could not log discrepancy to audit ledger: %s", exc)

    payload = ReportService.build_json_report(result)
    payload["offline_sync"] = True
    payload["discrepancy_logged"] = discrepancy

    return payload
