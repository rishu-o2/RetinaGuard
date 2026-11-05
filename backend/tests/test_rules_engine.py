"""Comprehensive unit tests for the deterministic clinical rules engine."""

import numpy as np
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
    low_conf_inf = InferenceOutput(
        predicted_grade=ICDRGrade.NO_DR,
        confidence=0.55,
        raw_probabilities={},
    )
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


# ------------------------------------------------------------------------------
# Tiered Sight-Threatening Escalation Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "predicted_grade,prob_dict,expected_urgency,expected_modifier",
    [
        # Grade 1 with P(>=3) = 0.08 (between 0.05 soon and 0.30 urgent): escalates from ROUTINE to SOON
        (
            ICDRGrade.MILD_NPDR,
            {0: 0.10, 1: 0.82, 2: 0.00, 3: 0.06, 4: 0.02},  # P(>=3) = 0.08
            Urgency.SOON,
            "SIGHT_THREATENING_DR_ESCALATION_SOON",
        ),
        # Grade 1 with P(>=3) = 0.75 (>= 0.30): escalates from ROUTINE to URGENT (outside [0.30, 0.70] band)
        (
            ICDRGrade.MILD_NPDR,
            {0: 0.05, 1: 0.20, 2: 0.00, 3: 0.50, 4: 0.25},  # P(>=3) = 0.75, P(ref) = 0.75 > 0.70
            Urgency.URGENT,
            "SIGHT_THREATENING_DR_ESCALATION_URGENT",
        ),
        # Grade 2 with P(>=3) = 0.02 (< 0.05): remains PRIORITY
        (
            ICDRGrade.MODERATE_NPDR,
            {0: 0.05, 1: 0.15, 2: 0.78, 3: 0.02, 4: 0.00},  # P(>=3) = 0.02
            Urgency.PRIORITY,
            None,
        ),
        # Grade 2 with P(>=3) = 0.35 (>= 0.30): escalates from PRIORITY to URGENT
        (
            ICDRGrade.MODERATE_NPDR,
            {0: 0.02, 1: 0.05, 2: 0.58, 3: 0.25, 4: 0.10},  # P(>=3) = 0.35
            Urgency.URGENT,
            "SIGHT_THREATENING_DR_ESCALATION_URGENT",
        ),
        # Grade 3 with P(>=3) = 0.92 (>= 0.30): stays URGENT
        (
            ICDRGrade.SEVERE_NPDR,
            {0: 0.01, 1: 0.02, 2: 0.05, 3: 0.80, 4: 0.12},  # P(>=3) = 0.92
            Urgency.URGENT,
            "SIGHT_THREATENING_DR_ESCALATION_URGENT",
        ),
        # Grade 4 with P(>=3) = 0.97 (>= 0.30): stays EMERGENCY (never downgraded to URGENT)
        (
            ICDRGrade.PROLIFERATIVE_DR,
            {0: 0.00, 1: 0.01, 2: 0.02, 3: 0.10, 4: 0.87},  # P(>=3) = 0.97
            Urgency.EMERGENCY,
            "SIGHT_THREATENING_DR_ESCALATION_URGENT",
        ),
    ],
)
def test_sight_threatening_escalation_table(
    rules_engine: ClinicalRulesEngine,
    sample_patient_context: PatientContext,
    predicted_grade: ICDRGrade,
    prob_dict: dict[int, float],
    expected_urgency: Urgency,
    expected_modifier: str | None,
) -> None:
    """Table-driven verification of tiered sight-threatening DR risk escalation."""
    sample_patient_context.is_pregnant = False
    sample_patient_context.hba1c = 7.0
    sample_patient_context.diabetes_duration_years = 4.0

    inference = InferenceOutput(
        predicted_grade=predicted_grade,
        confidence=float(max(prob_dict.values())),
        raw_probabilities=prob_dict,
    )
    res = rules_engine.evaluate(inference, sample_patient_context)
    assert res.urgency == expected_urgency
    if expected_modifier:
        assert expected_modifier in res.applied_modifiers
        if "URGENT" in expected_modifier:
            assert res.follow_up_interval_months <= 1
        elif "SOON" in expected_modifier:
            assert res.follow_up_interval_months <= 3
    else:
        assert not any("SIGHT_THREATENING" in m for m in res.applied_modifiers)


# ------------------------------------------------------------------------------
# Referable-Based Abstention Tests
# ------------------------------------------------------------------------------

def test_referable_band_abstention_triggers_uncertain(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """P(referable) within [0.30, 0.70] triggers UNCERTAIN."""
    sample_patient_context.is_pregnant = False
    sample_patient_context.hba1c = 6.0
    sample_patient_context.diabetes_duration_years = 2.0

    # P(ref) = 0.20 + 0.15 + 0.10 = 0.45 in [0.30, 0.70]
    prob_dict = {0: 0.35, 1: 0.20, 2: 0.20, 3: 0.15, 4: 0.10}
    inf = InferenceOutput(
        predicted_grade=ICDRGrade.NO_DR,
        confidence=0.35,
        raw_probabilities=prob_dict,
    )
    res = rules_engine.evaluate(inf, sample_patient_context)
    assert res.is_uncertain is True
    assert res.referral == ReferralDecision.UNCERTAIN


def test_referable_band_safety_never_confident_no_dr(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """Never return a confident No DR when P(referable) >= lower_threshold (0.30)."""
    sample_patient_context.is_pregnant = False
    sample_patient_context.hba1c = 6.0
    sample_patient_context.diabetes_duration_years = 2.0

    # Suppose model predicted grade 0, but P(ref) = 0.75 (> 0.70 upper edge)
    prob_dict = {0: 0.25, 1: 0.00, 2: 0.50, 3: 0.20, 4: 0.05}
    inf = InferenceOutput(
        predicted_grade=ICDRGrade.NO_DR,
        confidence=0.50,
        raw_probabilities=prob_dict,
    )
    res = rules_engine.evaluate(inf, sample_patient_context)
    # Must NOT be NON_REFERABLE No DR!
    assert res.referral != ReferralDecision.NON_REFERABLE


def test_property_escalation_never_lowers_urgency(
    rules_engine: ClinicalRulesEngine, sample_patient_context: PatientContext
) -> None:
    """Property test: Sight-threatening escalation rule NEVER lowers urgency under any input."""
    from retinaguard.rules.engine import URGENCY_RANKS
    import copy

    engine_disabled = copy.deepcopy(rules_engine)
    if engine_disabled.config.modifiers.sight_threatening_escalation:
        engine_disabled.config.modifiers.sight_threatening_escalation.enabled = False

    rng = np.random.default_rng(12345)
    for grade in list(ICDRGrade):
        for _ in range(50):
            # Generate random Dirichlet distribution for 5 classes
            probs = rng.dirichlet(np.ones(5))
            prob_dict = {i: float(probs[i]) for i in range(5)}
            # Max prob as confidence
            conf = float(max(probs))

            inference = InferenceOutput(
                predicted_grade=grade,
                confidence=max(conf, 0.75),  # ensure above confidence gate
                raw_probabilities=prob_dict,
            )

            # Test across multiple context combinations
            for is_preg in (False, True):
                for hba1c_val in (6.5, 11.0):
                    for dur in (3.0, 18.0):
                        ctx = copy.deepcopy(sample_patient_context)
                        ctx.is_pregnant = is_preg
                        ctx.hba1c = hba1c_val
                        ctx.diabetes_duration_years = dur
                        res_enabled = rules_engine.evaluate(inference, ctx)
                        res_disabled = engine_disabled.evaluate(inference, ctx)

                        rank_enabled = URGENCY_RANKS[res_enabled.urgency]
                        rank_disabled = URGENCY_RANKS[res_disabled.urgency]

                        # Invariant: enabled rule must be >= disabled rule
                        assert rank_enabled >= rank_disabled, (
                            f"Escalation rule lowered urgency from {res_disabled.urgency} "
                            f"to {res_enabled.urgency} for grade {grade} with probs {prob_dict}"
                        )
                        # Invariant: follow-up interval must be <= disabled rule
                        assert res_enabled.follow_up_interval_months <= res_disabled.follow_up_interval_months


