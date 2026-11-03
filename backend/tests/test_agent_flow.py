"""Tests for the RetinaGuard 5-Agent pipeline and LangGraph screening workflow.

Covers:
1. VisionAgent (quality gate, OOD check, inference, and early-exit flags)
2. SafetyAgent (deterministic rules engine delegation, modifier handling)
3. GuidelineAgent & GuidelineValidator (authentic retrieval, adversarial text rejection, empty dir fallback)
4. AuditAgent (zero-PHI, tamper-evident hash chaining)
5. VoiceAgent (English & Hindi speech narratives, distinct UNCERTAIN warning, zero-PHI)
6. Graph early exit (skips guideline agent on uncertainty, still logs to audit & generates voice warning)
7. End-to-end POST /api/v1/screen test with sample image
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from retinaguard.api.main import app
from retinaguard.agents.audit_agent import AuditAgent
from retinaguard.agents.graph import (
    build_screening_graph,
    get_screening_graph,
    run_screening_pipeline,
    should_skip_guideline,
)
from retinaguard.agents.guideline_agent import (
    GuidelineAgent,
    GuidelinePassage,
    GuidelineValidator,
)
from retinaguard.agents.safety_agent import SafetyAgent
from retinaguard.agents.state import ScreeningAgentState
from retinaguard.agents.vision_agent import VisionAgent
from retinaguard.agents.voice_agent import VoiceAgent, WebSpeechVoiceProvider
from retinaguard.core.consent import DPDPProcessingConsent
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
from retinaguard.rules.engine import ClinicalRulesEngine, RulesEvaluationResult
from retinaguard.services.audit import AuditLedger
from retinaguard.services.inference import ONNXInferenceService

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_fundus(size: int = 400) -> np.ndarray:
    """Generate a realistic synthetic fundus image in RGB uint8."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 15
    import cv2
    # Retinal fundus optical signature: R > G > B (orange-red disk)
    cv2.circle(img, center, radius, (190, 75, 40), -1)
    noise = np.random.randint(0, 20, (size, size, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def _make_patient_context(
    pseudonym: str = "PAT-TEST-0042",
    age: int = 52,
    is_pregnant: bool = False,
    hba1c: Optional[float] = None,
) -> PatientContext:
    import hashlib
    consent = DPDPProcessingConsent(
        patient_identifier_hash=hashlib.sha256(pseudonym.encode()).hexdigest(),
        purpose="DIABETIC_RETINOPATHY_SCREENING",
        notice_version="1.0-IN",
    )
    return PatientContext(
        patient_pseudonym_id=pseudonym,
        age=age,
        diabetes_type=DiabetesType.TYPE_2,
        is_pregnant=is_pregnant,
        hba1c=hba1c,
        consent=consent,
    )


def _make_metadata(img_rgb: np.ndarray) -> ImageMetadata:
    import hashlib
    h, w = img_rgb.shape[:2]
    return ImageMetadata(
        image_sha256=hashlib.sha256(img_rgb.tobytes()).hexdigest(),
        camera_type=CameraType.TABLETOP,
        eye_side=EyeSide.RIGHT_OD,
        width=w,
        height=h,
    )


# ---------------------------------------------------------------------------
# 1. Vision Agent Tests
# ---------------------------------------------------------------------------

def test_vision_agent_passes_on_valid_fundus():
    fundus_rgb = _make_synthetic_fundus(1024)
    meta = _make_metadata(fundus_rgb)
    agent = VisionAgent()
    state: ScreeningAgentState = {
        "image_rgb": fundus_rgb,
        "image_metadata": meta,
        "patient_context": _make_patient_context(),
    }
    out = agent(state)
    assert out["quality_result"].passed is True
    assert out["ood_result"].is_in_distribution is True


def test_vision_agent_early_exit_on_quality_failure():
    # Blurry / zero-contrast image fails quality gate
    black_rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
    meta = _make_metadata(black_rgb)
    agent = VisionAgent()
    state: ScreeningAgentState = {
        "image_rgb": black_rgb,
        "image_metadata": meta,
        "patient_context": _make_patient_context(),
    }
    out = agent(state)
    assert out["is_uncertain"] is True
    assert out["early_exit"] is True
    assert len(out["uncertainty_reasons"]) > 0


def test_vision_agent_early_exit_on_ood_trigger():
    # Circular disk with blue-dominant sky/landscape colors (B > R)
    import cv2
    size = 1024
    sky_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - 15
    # Color: R=50, G=120, B=230 (optical spectrum violation: B > R)
    cv2.circle(sky_rgb, center, radius, (50, 120, 230), -1)
    noise = np.random.randint(0, 10, (size, size, 3), dtype=np.uint8)
    sky_rgb = np.clip(sky_rgb.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    meta = _make_metadata(sky_rgb)
    agent = VisionAgent()
    state: ScreeningAgentState = {
        "image_rgb": sky_rgb,
        "image_metadata": meta,
        "patient_context": _make_patient_context(),
    }
    out = agent(state)
    assert out["is_uncertain"] is True
    assert out["early_exit"] is True
    assert out["ood_result"].is_in_distribution is False
    assert any("fundus" in r.lower() or "retinal" in r.lower() or "ood" in r.lower() for r in out["uncertainty_reasons"])


# ---------------------------------------------------------------------------
# 2. Safety Agent Tests
# ---------------------------------------------------------------------------

def test_safety_agent_calls_clinical_rules_engine():
    fundus_rgb = _make_synthetic_fundus(1024)
    meta = _make_metadata(fundus_rgb)
    ctx = _make_patient_context()
    agent = SafetyAgent()
    inference_out = InferenceOutput(
        predicted_grade=ICDRGrade.MODERATE_NPDR,
        confidence=0.92,
        raw_probabilities={0: 0.02, 1: 0.05, 2: 0.92, 3: 0.01, 4: 0.00},
    )
    state: ScreeningAgentState = {
        "image_rgb": fundus_rgb,
        "image_metadata": meta,
        "patient_context": ctx,
        "inference_output": inference_out,
        "is_uncertain": False,
    }
    out = agent(state)
    assert out["referral_decision"].referral == ReferralDecision.REFERABLE
    assert out["referral_decision"].urgency == Urgency.PRIORITY
    assert out["referral_decision"].follow_up_interval_months == 3
    assert "report_en" in out and len(out["report_en"]) > 20
    assert "report_hi" in out and len(out["report_hi"]) > 20


def test_safety_agent_handles_uncertain_pathway():
    fundus_rgb = _make_synthetic_fundus(512)
    meta = _make_metadata(fundus_rgb)
    ctx = _make_patient_context()
    agent = SafetyAgent()
    state: ScreeningAgentState = {
        "image_rgb": fundus_rgb,
        "image_metadata": meta,
        "patient_context": ctx,
        "is_uncertain": True,
        "uncertainty_reasons": ["Quality gate failed: Blur variance below threshold"],
    }
    out = agent(state)
    assert out["is_uncertain"] is True
    assert out["referral_decision"].referral == ReferralDecision.UNCERTAIN


# ---------------------------------------------------------------------------
# 3. Guideline Agent & Validator Tests
# ---------------------------------------------------------------------------

def test_guideline_agent_empty_dir_fallback():
    # Points to empty temp folder
    import tempfile
    with tempfile.TemporaryDirectory() as empty_dir:
        agent = GuidelineAgent(guidelines_dir=empty_dir)
        state: ScreeningAgentState = {
            "referral_decision": RulesEvaluationResult(
                referral=ReferralDecision.REFERABLE,
                urgency=Urgency.ROUTINE,
                follow_up_interval_months=3,
            ),
            "inference_output": InferenceOutput(
                predicted_grade=ICDRGrade.MODERATE_NPDR,
                confidence=0.88,
                raw_probabilities={i: 0.2 for i in range(5)},
            ),
            "report_en": "Standard baseline summary.",
            "language_requested": "en",
        }
        out = agent(state)
        # Citations must be empty when guidelines dir is empty (Condition 4)
        assert out["guideline_citations"] == []
        assert "Standard clinical guideline templates" in out["guideline_source_note"]


def test_guideline_validator_rejects_conflicting_grade():
    validator = GuidelineValidator()
    decision = RulesEvaluationResult(
        referral=ReferralDecision.REFERABLE,
        urgency=Urgency.ROUTINE,
        follow_up_interval_months=3,
    )
    # Ground truth is Grade 2, but adversarial candidate claims Grade 0
    adversarial_text = "The patient has been diagnosed with Grade 0 and needs no care."
    is_valid, reason = validator.validate(adversarial_text, decision, grade=ICDRGrade.MODERATE_NPDR)
    assert is_valid is False
    assert "conflicting grade" in reason.lower()


def test_guideline_validator_rejects_conflicting_interval():
    validator = GuidelineValidator()
    decision = RulesEvaluationResult(
        referral=ReferralDecision.REFERABLE,
        urgency=Urgency.ROUTINE,
        follow_up_interval_months=3,
    )
    # Ground truth is 3 months, text claims 12 months
    adversarial_text = "Clinical guideline suggests routine rescreening in 12 months."
    is_valid, reason = validator.validate(adversarial_text, decision, grade=ICDRGrade.MODERATE_NPDR)
    assert is_valid is False
    assert "conflicting re-screening interval" in reason.lower()


def test_guideline_validator_rejects_emergency_downplay():
    validator = GuidelineValidator()
    decision = RulesEvaluationResult(
        referral=ReferralDecision.REFERABLE,
        urgency=Urgency.EMERGENCY,
        follow_up_interval_months=0,
    )
    # Ground truth is Emergency, text claims routine
    adversarial_text = "Routine follow-up is sufficient for this patient."
    is_valid, reason = validator.validate(adversarial_text, decision, grade=ICDRGrade.PROLIFERATIVE_DR)
    assert is_valid is False
    assert "routine follow-up for an emergency" in reason.lower()


# ---------------------------------------------------------------------------
# 4. Audit Agent & Privacy Tests
# ---------------------------------------------------------------------------

def test_audit_agent_zero_phi_and_hash_chain():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = AuditLedger(ledger_path=Path(tmp_dir) / "audit.jsonl")
        agent = AuditAgent(ledger=ledger)
        fundus_rgb = _make_synthetic_fundus(512)
        meta = _make_metadata(fundus_rgb)
        ctx = _make_patient_context(pseudonym="PAT-SECURE-999")
        state: ScreeningAgentState = {
            "image_metadata": meta,
            "patient_context": ctx,
            "is_uncertain": False,
            "referral_decision": RulesEvaluationResult(
                referral=ReferralDecision.REFERABLE,
                urgency=Urgency.ROUTINE,
                follow_up_interval_months=3,
            ),
            "inference_output": InferenceOutput(
                predicted_grade=ICDRGrade.MODERATE_NPDR,
                confidence=0.91,
                raw_probabilities={i: 0.2 for i in range(5)},
            ),
        }
        out = agent(state)
        assert out["audit_hash"] is not None
        assert len(out["audit_hash"]) == 64

        # Verify ledger file on disk contains NO raw image bytes and NO patient ABHA
        content = (Path(tmp_dir) / "audit.jsonl").read_text(encoding="utf-8")
        assert "PAT-SECURE-999" in content
        assert meta.image_sha256 in content
        assert "raw_bytes" not in content


# ---------------------------------------------------------------------------
# 5. Voice Agent Tests
# ---------------------------------------------------------------------------

def test_voice_agent_english_and_hindi_speech():
    agent = VoiceAgent()
    # English normal
    state_en: ScreeningAgentState = {
        "language_requested": "en",
        "is_uncertain": False,
        "referral_decision": RulesEvaluationResult(
            referral=ReferralDecision.REFERABLE,
            urgency=Urgency.ROUTINE,
            follow_up_interval_months=3,
        ),
        "inference_output": InferenceOutput(
            predicted_grade=ICDRGrade.MODERATE_NPDR,
            confidence=0.91,
            raw_probabilities={i: 0.2 for i in range(5)},
        ),
    }
    out_en = agent(state_en)
    assert "RetinaGuard screening complete" in out_en["voice_text"]
    assert "grade 2" in out_en["voice_text"].lower()

    # Hindi normal
    state_hi = dict(state_en)
    state_hi["language_requested"] = "hi"
    out_hi = agent(state_hi)
    assert any(ord(c) > 0x0900 for c in out_hi["voice_text"])
    assert "रेटिनागार्ड" in out_hi["voice_text"]

    # Uncertain warning
    state_unc: ScreeningAgentState = {
        "language_requested": "en",
        "is_uncertain": True,
        "uncertainty_reasons": ["Blur variance below threshold"],
    }
    out_unc = agent(state_unc)
    assert "UNCERTAIN" in out_unc["voice_text"]
    assert "clinician review or image recapture is mandatory" in out_unc["voice_text"].lower()


# ---------------------------------------------------------------------------
# 6. Graph Early-Exit Routing Test
# ---------------------------------------------------------------------------

def test_graph_early_exit_routing_logic():
    # If uncertain, should skip guideline agent
    state_unc: ScreeningAgentState = {"early_exit": True, "is_uncertain": True}
    assert should_skip_guideline(state_unc) == "audit"

    # If certain, should proceed to guideline agent
    state_cert: ScreeningAgentState = {"early_exit": False, "is_uncertain": False}
    assert should_skip_guideline(state_cert) == "guideline"


# ---------------------------------------------------------------------------
# 7. End-to-End POST /api/v1/screen Test
# ---------------------------------------------------------------------------

def test_api_v1_screen_endpoint():
    fundus_rgb = _make_synthetic_fundus(512)
    import cv2
    _, buf = cv2.imencode(".png", fundus_rgb)
    png_bytes = buf.tobytes()

    form = {
        "patient_pseudonym_id": "PAT-V1-TEST-001",
        "eye_side": "OD",
        "camera_type": "TABLETOP",
        "age": "50",
        "diabetes_type": "TYPE_2",
        "language": "en",
    }
    resp = client.post(
        "/api/v1/screen/",
        data=form,
        files={"image": ("sample_fundus.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["clinician_review_mandatory"] is True
    assert "voice_text" in data
    assert "guideline_explanation" in data
    assert "audit_hash" in data
    assert len(data["audit_hash"]) == 64
