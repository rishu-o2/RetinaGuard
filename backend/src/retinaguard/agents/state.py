"""State definition for RetinaGuard's 5-agent clinical screening pipeline.

Implements the shared TypedDict state passed between the 5 LangGraph agents:
  1. vision_agent    (Quality gate -> OOD check -> ONNX model)
  2. safety_agent    (Deterministic ClinicalRulesEngine)
  3. guideline_agent (RAG retrieval from docs/guidelines/ with strict validator)
  4. audit_agent     (Hash-chained tamper-evident AuditLedger)
  5. voice_agent     (Multilingual speech narrative generator)
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict
import numpy as np

from retinaguard.core.models import (
    ImageMetadata,
    InferenceOutput,
    PatientContext,
    ReferralDecision,
    ScreeningResult,
)
from retinaguard.services.ood import OODResult
from retinaguard.services.quality import QualityGateResult


class ScreeningAgentState(TypedDict, total=False):
    """Shared state dictionary passed across agents in the graph."""

    # 1. Inputs
    image_bytes: bytes
    image_rgb: Optional[np.ndarray]
    image_metadata: ImageMetadata
    patient_context: PatientContext
    language_requested: str  # "en" or "hi"

    # 2. Early-Exit & Uncertainty Flags
    is_uncertain: bool
    uncertainty_reasons: list[str]
    early_exit: bool

    # 3. Vision Agent Output
    quality_result: Optional[QualityGateResult]
    ood_result: Optional[OODResult]
    inference_output: Optional[InferenceOutput]

    # 4. Safety Agent Output (from deterministic ClinicalRulesEngine)
    referral_decision: Optional[ReferralDecision]

    # 5. Guideline Agent Output (RAG + safety validator)
    guideline_explanation: Optional[str]
    guideline_citations: list[dict[str, Any]]
    guideline_source_note: Optional[str]

    # 6. Report Summaries (English & Hindi)
    report_en: Optional[str]
    report_hi: Optional[str]

    # 7. Audit Agent Output
    audit_hash: Optional[str]
    previous_audit_hash: Optional[str]

    # 8. Voice Agent Output
    voice_text: Optional[str]

    # 9. Final Assembled Result
    final_result: Optional[ScreeningResult]
    error: Optional[str]
