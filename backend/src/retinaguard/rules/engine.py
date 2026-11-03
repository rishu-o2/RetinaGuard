"""Deterministic Clinical Decision Rules Engine.

Follows Principle 1 & 2:
- The ML model only predicts grade and confidence. It NEVER decides referral.
- Referral, urgency, and follow-up intervals are computed deterministically
  by this pure Python engine based on AIOS, RSSDI, ICMR, and ICO guidelines.
"""

from typing import Any
from pydantic import BaseModel, Field

from retinaguard.core.config import ClinicalRulesConfig, load_clinical_rules
from retinaguard.core.models import (
    ICDRGrade,
    InferenceOutput,
    PatientContext,
    ReferralDecision,
    Urgency,
)


class RulesEvaluationResult(BaseModel):
    """Output of deterministic clinical rules evaluation."""
    referral: ReferralDecision
    urgency: Urgency
    follow_up_interval_months: int
    is_uncertain: bool = False
    uncertainty_reasons: list[str] = Field(default_factory=list)
    applied_modifiers: list[str] = Field(default_factory=list)


# Urgency ranking for comparison and escalation
URGENCY_RANKS: dict[Urgency, int] = {
    Urgency.ROUTINE: 1,
    Urgency.PRIORITY: 2,
    Urgency.URGENT: 3,
    Urgency.EMERGENCY: 4,
}


def _max_urgency(u1: Urgency, u2: Urgency) -> Urgency:
    """Return the more severe urgency tier."""
    return u1 if URGENCY_RANKS[u1] >= URGENCY_RANKS[u2] else u2


class ClinicalRulesEngine:
    """Deterministic, unit-tested clinical rules engine."""

    def __init__(self, config: ClinicalRulesConfig | None = None) -> None:
        self.config = config or load_clinical_rules()

    def evaluate(
        self,
        inference: InferenceOutput | None,
        context: PatientContext,
        quality_passed: bool = True,
        ood_passed: bool = True,
    ) -> RulesEvaluationResult:
        """Evaluate clinical decision package deterministically.

        Guarantees:
        1. If quality or OOD fails, or ML inference is absent, returns UNCERTAIN.
        2. If ML confidence < min_acceptable_confidence, returns UNCERTAIN (never a guess).
        3. For valid inference, maps grade to base referral & urgency, then applies
           patient context modifiers (pregnancy, HbA1c, duration).
        """
        uncertainty_reasons: list[str] = []

        # Gate 1: Pre-inference quality or OOD check failure
        if not quality_passed:
            uncertainty_reasons.append("Image failed diagnostic quality gate.")
        if not ood_passed:
            uncertainty_reasons.append("Image rejected as out-of-distribution (non-fundus).")
        if inference is None:
            uncertainty_reasons.append("No valid inference output available.")

        if uncertainty_reasons:
            return RulesEvaluationResult(
                referral=ReferralDecision.UNCERTAIN,
                urgency=Urgency.PRIORITY,  # Needs clinician attention to inspect
                follow_up_interval_months=1,
                is_uncertain=True,
                uncertainty_reasons=uncertainty_reasons,
            )

        assert inference is not None

        # Gate 2: Model confidence check
        min_conf = self.config.confidence_thresholds.min_acceptable_confidence
        if inference.confidence < min_conf:
            return RulesEvaluationResult(
                referral=ReferralDecision.UNCERTAIN,
                urgency=Urgency.PRIORITY,
                follow_up_interval_months=1,
                is_uncertain=True,
                uncertainty_reasons=[
                    f"Model confidence ({inference.confidence:.2f}) below acceptable threshold ({min_conf:.2f})."
                ],
            )

        # Base clinical decision from ICDR grade table
        grade_val = int(inference.predicted_grade)
        grade_rule = self.config.icdr_grades[grade_val]

        referral = (
            ReferralDecision.REFERABLE if grade_rule.referable else ReferralDecision.NON_REFERABLE
        )
        urgency = grade_rule.base_urgency
        interval_months = grade_rule.base_interval_months
        applied_modifiers: list[str] = []

        # Modifier 1: Pregnancy in diabetes (AIOS 2020 / RSSDI 2022)
        if context.is_pregnant:
            preg_mod = self.config.modifiers.pregnancy
            urgency = _max_urgency(urgency, preg_mod.min_urgency)
            interval_months = min(interval_months, preg_mod.max_interval_months)
            applied_modifiers.append("PREGNANCY_IN_DIABETES")

        # Modifier 2: High-risk glycemic control (HbA1c >= threshold)
        if (
            context.hba1c is not None
            and context.hba1c >= self.config.modifiers.high_risk_glycemic_control.hba1c_threshold_percent
        ):
            glyc_mod = self.config.modifiers.high_risk_glycemic_control
            interval_months = min(interval_months, glyc_mod.max_interval_months)
            if grade_val >= glyc_mod.upgrade_urgency_if_grade_ge:
                urgency = _max_urgency(urgency, Urgency.PRIORITY)
            applied_modifiers.append("SEVERELY_UNCONTROLLED_HBA1C")

        # Modifier 3: Long-standing diabetes (duration >= threshold)
        if (
            context.diabetes_duration_years is not None
            and context.diabetes_duration_years
            >= self.config.modifiers.long_standing_diabetes.duration_years_threshold
        ):
            dur_mod = self.config.modifiers.long_standing_diabetes
            interval_months = min(interval_months, dur_mod.max_interval_months)
            if grade_val >= dur_mod.upgrade_urgency_if_grade_ge:
                urgency = _max_urgency(urgency, Urgency.URGENT)
            applied_modifiers.append("LONG_STANDING_DIABETES")

        return RulesEvaluationResult(
            referral=referral,
            urgency=urgency,
            follow_up_interval_months=interval_months,
            is_uncertain=False,
            uncertainty_reasons=[],
            applied_modifiers=applied_modifiers,
        )

