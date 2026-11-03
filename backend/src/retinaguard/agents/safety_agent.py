"""Safety Agent for RetinaGuard.

ADR-003: The safety agent enforces deterministic clinical governance.
It calls the existing ClinicalRulesEngine (grounded in configs/clinical_rules.yaml)
and NEVER copies rule logic or invents clinical numbers.
Grades 3-4 are strictly protected against downgrading.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from retinaguard.agents.state import ScreeningAgentState
from retinaguard.core.logging import get_logger
from retinaguard.core.models import ScreeningResult
from retinaguard.rules.engine import ClinicalRulesEngine
from retinaguard.services.report import ReportService

logger = get_logger("retinaguard.agents.safety")


class SafetyAgent:
    """Agent delegating all referral, urgency, and interval decisions to ClinicalRulesEngine."""

    def __init__(
        self,
        rules_engine: Optional[ClinicalRulesEngine] = None,
        report_service: Optional[type[ReportService]] = None,
    ) -> None:
        self.rules_engine = rules_engine or ClinicalRulesEngine()
        self.report_service = report_service or ReportService

    def __call__(self, state: ScreeningAgentState) -> dict:
        """Evaluate clinical safety rules deterministically."""
        is_uncertain = state.get("is_uncertain", False)
        uncertainty_reasons = list(state.get("uncertainty_reasons", []))
        patient_context = state["patient_context"]
        metadata = state["image_metadata"]
        inference_out = state.get("inference_output")
        quality_res = state.get("quality_result")
        ood_res = state.get("ood_result")

        quality_passed = quality_res.passed if quality_res is not None else (not is_uncertain)
        ood_passed = ood_res.is_in_distribution if ood_res is not None else (not is_uncertain)

        # 1. Deterministic safety evaluation through existing ClinicalRulesEngine
        rules_result = self.rules_engine.evaluate(
            inference=inference_out if (not is_uncertain) else None,
            context=patient_context,
            quality_passed=quality_passed,
            ood_passed=ood_passed,
        )

        combined_reasons = list(uncertainty_reasons)
        for r in rules_result.uncertainty_reasons:
            if r not in combined_reasons:
                combined_reasons.append(r)

        grade = inference_out.predicted_grade if (inference_out and not rules_result.is_uncertain) else None
        confidence = inference_out.confidence if (inference_out and not rules_result.is_uncertain) else None

        # 2. Render verified baseline reports (EN & HI) using existing ReportService
        interim_res = ScreeningResult(
            screening_id=uuid4(),
            patient_pseudonym_id=patient_context.patient_pseudonym_id,
            eye_side=metadata.eye_side,
            camera_type=metadata.camera_type,
            grade=grade,
            confidence=confidence,
            referral=rules_result.referral,
            urgency=rules_result.urgency,
            follow_up_interval_months=rules_result.follow_up_interval_months,
            is_uncertain=rules_result.is_uncertain,
            uncertainty_reasons=combined_reasons,
            clinician_review_mandatory=True,
            report_en="",
            report_hi="",
            image_sha256=metadata.image_sha256,
        )

        report_en = self.report_service.build_text_report(interim_res, lang="en")
        report_hi = self.report_service.build_text_report(interim_res, lang="hi")

        return {
            "referral_decision": rules_result,
            "report_en": report_en,
            "report_hi": report_hi,
            "is_uncertain": rules_result.is_uncertain,
            "uncertainty_reasons": combined_reasons,
        }
