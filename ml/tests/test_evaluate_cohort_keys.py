"""Regression test: evaluate_cohort key contract.

Guarantees that every key consumed by train.py and evaluate_model.py actually
exists in the dict returned by evaluate_cohort(), and that the computed values
match expected answers for hand-crafted examples.

This test was added after the silent-0.0 bug where train.py looked up four
wrong key names ("quadratic_weighted_kappa", "referable_dr_sensitivity",
"referable_dr_specificity", "auc_roc_ovr_macro") and always received the .get()
default of 0.0.  Direct indexing is now used so a missing key raises KeyError.
"""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from retinaguard_ml.evaluate import evaluate_cohort

# ─── Keys consumed by train.py and evaluate_model.py ─────────────────────────
# These are the EXACT identifiers that appear in production code.  Any rename of
# a key in evaluate_cohort() will cause this test to fail loudly.
REQUIRED_KEYS_NO_PROBS = {
    "qwk",           # train.py:  eval_metrics["qwk"]
    "sensitivity",   # train.py:  eval_metrics["sensitivity"]
    "specificity",   # train.py:  eval_metrics["specificity"]
    # evaluate_model.py bootstrap also uses "qwk" and "sensitivity"
}

REQUIRED_KEYS_WITH_PROBS = REQUIRED_KEYS_NO_PROBS | {
    "referable_dr_auc",  # train.py:  eval_metrics.get("referable_dr_auc", 0.0)
}

# ─── Fixtures ─────────────────────────────────────────────────────────────────

# Perfect predictor: every grade is correct
Y_PERFECT = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4], dtype=int)

# All-zero predictor: model always predicts grade 0
# true has 5 referable (grade>=2) and 7 non-referable (grade 0/1)
Y_TRUE_MIXED = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=int)
Y_ALL_ZERO   = np.zeros(12, dtype=int)

# Soft probability arrays: (N, 5) — rows sum to 1
def _make_perfect_probs(y: np.ndarray) -> np.ndarray:
    """One-hot probability matrix for a perfect predictor."""
    n = len(y)
    p = np.zeros((n, 5), dtype=float)
    p[np.arange(n), y] = 1.0
    return p

def _make_uniform_probs(n: int) -> np.ndarray:
    return np.full((n, 5), 0.2, dtype=float)


# ─── Test 1: required keys present (no probs) ─────────────────────────────────

def test_evaluate_cohort_required_keys_no_probs() -> None:
    """All keys read by train.py must exist when y_probs=None."""
    metrics = evaluate_cohort(Y_TRUE_MIXED, Y_ALL_ZERO, y_probs=None)
    for key in REQUIRED_KEYS_NO_PROBS:
        assert key in metrics, (
            f"Key '{key}' missing from evaluate_cohort() output. "
            f"Keys present: {sorted(metrics.keys())}"
        )


def test_evaluate_cohort_required_keys_with_probs() -> None:
    """All keys read by train.py and evaluate_model.py must exist when y_probs supplied."""
    probs = _make_uniform_probs(len(Y_TRUE_MIXED))
    metrics = evaluate_cohort(Y_TRUE_MIXED, Y_ALL_ZERO, y_probs=probs)
    for key in REQUIRED_KEYS_WITH_PROBS:
        assert key in metrics, (
            f"Key '{key}' missing from evaluate_cohort() output. "
            f"Keys present: {sorted(metrics.keys())}"
        )


# ─── Test 2: perfect predictor ────────────────────────────────────────────────

def test_evaluate_cohort_perfect_predictor_qwk() -> None:
    """Perfect predictor must yield QWK = 1.0."""
    probs = _make_perfect_probs(Y_PERFECT)
    metrics = evaluate_cohort(Y_PERFECT, Y_PERFECT, y_probs=probs)

    assert metrics["qwk"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected QWK=1.0 for perfect predictor, got {metrics['qwk']}"
    )
    # Cross-check with sklearn
    expected_qwk = cohen_kappa_score(Y_PERFECT, Y_PERFECT, weights="quadratic")
    assert metrics["qwk"] == pytest.approx(expected_qwk, abs=1e-6)


def test_evaluate_cohort_perfect_predictor_sens_spec() -> None:
    """Perfect predictor must yield sensitivity=1.0 and specificity=1.0."""
    metrics = evaluate_cohort(Y_PERFECT, Y_PERFECT)
    assert metrics["sensitivity"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected sensitivity=1.0 for perfect predictor, got {metrics['sensitivity']}"
    )
    assert metrics["specificity"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected specificity=1.0 for perfect predictor, got {metrics['specificity']}"
    )


def test_evaluate_cohort_perfect_predictor_auc() -> None:
    """Perfect predictor with one-hot probs must yield referable_dr_auc = 1.0."""
    probs = _make_perfect_probs(Y_PERFECT)
    metrics = evaluate_cohort(Y_PERFECT, Y_PERFECT, y_probs=probs)
    assert metrics["referable_dr_auc"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected AUC=1.0 for perfect predictor, got {metrics['referable_dr_auc']}"
    )


# ─── Test 3: all-zero predictor with known ground truth ───────────────────────

def test_evaluate_cohort_all_zero_predictor_sens_spec() -> None:
    """All-zero predictor: sensitivity=0.0 (all referable missed), specificity=1.0.

    Y_TRUE_MIXED has 5 referable (grades 2-4) and 7 non-referable (grades 0-1).
    Predicting grade 0 for everything:
      TP=0, FN=5  → sensitivity = 0 / 5 = 0.0
      TN=7, FP=0  → specificity = 7 / 7 = 1.0
    """
    metrics = evaluate_cohort(Y_TRUE_MIXED, Y_ALL_ZERO, y_probs=None)

    assert metrics["sensitivity"] == pytest.approx(0.0, abs=1e-9), (
        f"Expected sensitivity=0.0 for all-zero predictor, got {metrics['sensitivity']}"
    )
    assert metrics["specificity"] == pytest.approx(1.0, abs=1e-9), (
        f"Expected specificity=1.0 for all-zero predictor, got {metrics['specificity']}"
    )


def test_evaluate_cohort_all_zero_predictor_qwk_matches_sklearn() -> None:
    """QWK from evaluate_cohort must match sklearn's cohen_kappa_score."""
    metrics = evaluate_cohort(Y_TRUE_MIXED, Y_ALL_ZERO)
    expected = cohen_kappa_score(Y_TRUE_MIXED, Y_ALL_ZERO, weights="quadratic")
    assert metrics["qwk"] == pytest.approx(expected, abs=1e-6), (
        f"QWK mismatch: evaluate_cohort={metrics['qwk']:.6f}, "
        f"sklearn={expected:.6f}"
    )


def test_evaluate_cohort_auc_uses_prob_referable_not_macro() -> None:
    """referable_dr_auc must be the binary AUC of P(grade>=2), NOT a macro OvR score.

    We construct probs so that:
      - the referable probability (sum of cols 2-4) is a near-perfect discriminator
      - but the per-class probs for grades 0 and 1 are deliberately swapped
    If the implementation used macro OvR it would produce a different (lower) value;
    the binary referable AUC must match sklearn's roc_auc_score on the binary label.
    """
    rng = np.random.default_rng(0)
    n = 60
    # 30 non-referable (grade 0 or 1), 30 referable (grade 2-4)
    y_true = np.array([0]*15 + [1]*15 + [2]*10 + [3]*10 + [4]*10)

    # Build probs where P(referable) is a near-perfect signal
    probs = np.zeros((n, 5), dtype=float)
    # Non-referable: most prob in grades 0,1; grades 2-4 sum is low (~0.05)
    probs[:30, 0] = 0.50
    probs[:30, 1] = 0.45
    probs[:30, 2] = 0.02
    probs[:30, 3] = 0.02
    probs[:30, 4] = 0.01
    # Referable: most prob in grades 2-4; sum is high (~0.90)
    probs[30:, 0] = 0.05
    probs[30:, 1] = 0.05
    probs[30:, 2] = 0.30
    probs[30:, 3] = 0.30
    probs[30:, 4] = 0.30

    # Add tiny noise so rows are not identical
    noise = rng.uniform(0, 0.002, probs.shape)
    probs = probs + noise
    probs = probs / probs.sum(axis=1, keepdims=True)  # renormalize

    metrics = evaluate_cohort(y_true, (y_true > 0).astype(int), y_probs=probs)

    # Expected: binary AUC using P(grade>=2) = sum of cols 2-4
    prob_referable = probs[:, 2:].sum(axis=1)
    y_binary = (y_true >= 2).astype(int)
    expected_binary_auc = roc_auc_score(y_binary, prob_referable)

    assert metrics["referable_dr_auc"] == pytest.approx(expected_binary_auc, abs=1e-9), (
        f"referable_dr_auc does not match binary AUC of P(grade>=2). "
        f"Got {metrics['referable_dr_auc']:.6f}, expected {expected_binary_auc:.6f}. "
        f"Implementation must use sum(probs[:, 2:]) as the score."
    )


# ─── Test 4: direct-index guard (documents the fixed contract) ────────────────

def test_wrong_keys_do_not_exist_in_output() -> None:
    """The four wrong key names from the silent-0.0 bug must NOT be present.

    If they appear, it means evaluate_cohort was accidentally changed to use the
    old names and train.py's direct indexing would break (KeyError on the right key).
    """
    metrics = evaluate_cohort(Y_TRUE_MIXED, Y_ALL_ZERO,
                              y_probs=_make_uniform_probs(len(Y_TRUE_MIXED)))
    old_wrong_keys = [
        "quadratic_weighted_kappa",
        "referable_dr_sensitivity",
        "referable_dr_specificity",
        "auc_roc_ovr_macro",
    ]
    for key in old_wrong_keys:
        assert key not in metrics, (
            f"Old wrong key '{key}' was found in evaluate_cohort() output. "
            f"Rename it back to the correct key and update train.py accordingly."
        )

