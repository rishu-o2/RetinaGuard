"""Comprehensive unit tests for the deterministic clinical rules engine."""

import pytest
from retinaguard.core.models import (
    ICDRGrade,
    InferenceOutput,
    PatientContext,
    ReferralDecision,
    Urgency,
)
from retinaguard.rules.engine import ClinicalRulesEngine


@pytest.fixture
def rules_engine() -> ClinicalRulesEngine:
    return ClinicalRulesEngine()


def _make_inference(grade: ICDRGrade, confidence: float = 0.90) -> InferenceOutput:
    return InferenceOutput(
        predicted_grade=grade,
        confidence=confidence,
        raw_probabilities={int(g): (1.0 if g == grade else 0.0) for g in ICDRGrade},
    )


def test_grade_0_no_dr_routine(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """ICDR 0: Non-referable, routine, 12-month follow-up."""
    # Ensure patient has standard baseline context without severe modifiers
    sample_patient_context.hba1c = 7.0
    sample_patient_context.diabetes_duration_years = 5.0
    sample_patient_context.is_pregnant = False

    res = rules_engine.evaluate(_make_inference(ICDRGrade.NO_DR), sample_patient_context)
    assert res.referral == ReferralDecision.NON_REFERABLE
    assert res.urgency == Urgency.ROUTINE
    assert res.follow_up_interval_months == 12
    assert res.is_uncertain is False


def test_grade_1_mild_npdr_non_referable(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """ICDR 1: Non-referable, routine, 12-month follow-up per AIOS/ICO."""
    sample_patient_context.hba1c = 7.0
    sample_patient_context.diabetes_duration_years = 4.0
    sample_patient_context.is_pregnant = False

    res = rules_engine.evaluate(_make_inference(ICDRGrade.MILD_NPDR), sample_patient_context)
    assert res.referral == ReferralDecision.NON_REFERABLE
    assert res.urgency == Urgency.ROUTINE
    assert res.follow_up_interval_months == 12


def test_grade_2_moderate_npdr_referable(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """ICDR 2: Referable, priority urgency, 3-month window."""
    res = rules_engine.evaluate(_make_inference(ICDRGrade.MODERATE_NPDR), sample_patient_context)
    assert res.referral == ReferralDecision.REFERABLE
    assert res.urgency in (Urgency.PRIORITY, Urgency.URGENT)
    assert res.follow_up_interval_months <= 3


def test_grade_3_severe_npdr_urgent(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """ICDR 3: Referable, urgent, 1-month window."""
    res = rules_engine.evaluate(_make_inference(ICDRGrade.SEVERE_NPDR), sample_patient_context)
    assert res.referral == ReferralDecision.REFERABLE
    assert res.urgency in (Urgency.URGENT, Urgency.EMERGENCY)
    assert res.follow_up_interval_months <= 1


def test_grade_4_pdr_emergency(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """ICDR 4: Referable, emergency, immediate evaluation (0 months)."""
    res = rules_engine.evaluate(_make_inference(ICDRGrade.PROLIFERATIVE_DR), sample_patient_context)
    assert res.referral == ReferralDecision.REFERABLE
    assert res.urgency == Urgency.EMERGENCY
    assert res.follow_up_interval_months == 0


def test_low_confidence_fails_safe_to_uncertain(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """Confidence below threshold must return UNCERTAIN, never a guess."""
    low_conf_inf = _make_inference(ICDRGrade.NO_DR, confidence=0.55)
    res = rules_engine.evaluate(low_conf_inf, sample_patient_context)
    assert res.is_uncertain is True
    assert res.referral == ReferralDecision.UNCERTAIN
    assert "below acceptable threshold" in res.uncertainty_reasons[0]


def test_quality_gate_failure_returns_uncertain(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """Pre-inference quality failure must short-circuit to UNCERTAIN."""
    inf = _make_inference(ICDRGrade.NO_DR)
    res = rules_engine.evaluate(inf, sample_patient_context, quality_passed=False)
    assert res.is_uncertain is True
    assert res.referral == ReferralDecision.UNCERTAIN


def test_ood_gate_failure_returns_uncertain(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """Out-of-distribution non-fundus rejection must short-circuit to UNCERTAIN."""
    inf = _make_inference(ICDRGrade.NO_DR)
    res = rules_engine.evaluate(inf, sample_patient_context, ood_passed=False)
    assert res.is_uncertain is True
    assert res.referral == ReferralDecision.UNCERTAIN


def test_pregnancy_modifier_escalation(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """Pregnancy in diabetes must elevate urgency to PRIORITY and limit follow-up to <= 3 months."""
    sample_patient_context.is_pregnant = True
    sample_patient_context.hba1c = 6.5
    sample_patient_context.diabetes_duration_years = 3.0

    res = rules_engine.evaluate(_make_inference(ICDRGrade.NO_DR), sample_patient_context)
    assert res.urgency == Urgency.PRIORITY
    assert res.follow_up_interval_months <= 3
    assert "PREGNANCY_IN_DIABETES" in res.applied_modifiers


def test_high_hba1c_modifier(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """Severe hyperglycemia (HbA1c >= 10.0%) upgrades mild DR to PRIORITY."""
    sample_patient_context.is_pregnant = False
    sample_patient_context.hba1c = 11.2
    sample_patient_context.diabetes_duration_years = 3.0

    res = rules_engine.evaluate(_make_inference(ICDRGrade.MILD_NPDR), sample_patient_context)
    assert res.urgency == Urgency.PRIORITY
    assert res.follow_up_interval_months <= 6
    assert "SEVERELY_UNCONTROLLED_HBA1C" in res.applied_modifiers

