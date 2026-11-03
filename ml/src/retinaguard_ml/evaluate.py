"""Clinical Evaluation Metrics for Diabetic Retinopathy Grading.

Computes:
- Quadratic Weighted Kappa (QWK) across ordinal ICDR grades (0 to 4).
- Per-class recall / sensitivity across all 5 ICDR grades.
- Referable DR (grades >= 2) Sensitivity at 90% and 95% Specificity.
- Binary and Multi-class ROC-AUC.
- Full Confusion Matrix.
- Per-device and per-site subgroup metrics (ADR-007 / Domain Shift tracking).
"""

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # type: ignore

from retinaguard_ml.data.registry import ImageRecord


def compute_per_class_recall(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> Dict[int, float]:
    """Compute per-class sensitivity/recall across all 5 ICDR grades."""
    recalls: Dict[int, float] = {}
    for c in range(num_classes):
        mask = (y_true == c)
        total_c = int(np.sum(mask))
        if total_c > 0:
            correct_c = int(np.sum((y_true == c) & (y_pred == c)))
            recalls[c] = float(correct_c / total_c)
        else:
            recalls[c] = 0.0
    return recalls


def compute_referable_dr_sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob_referable: np.ndarray,
    target_specificities: Tuple[float, ...] = (0.90, 0.95),
) -> Dict[str, float]:
    """Calculate sensitivity at fixed operating points (90% and 95% specificity).

    Essential for clinical screening systems per British & Indian Diabetic Eye Screening standards.
    """
    y_true_binary = (y_true >= 2).astype(int)

    # Edge case: only 1 class present in batch/test set
    if len(np.unique(y_true_binary)) < 2:
        return {f"sens_at_{int(s*100)}_spec": 0.0 for s in target_specificities}

    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_referable)
    results: Dict[str, float] = {}

    for target_spec in target_specificities:
        max_allowed_fpr = 1.0 - target_spec
        # Locate operating points where FPR <= max_allowed_fpr
        valid_indices = np.where(fpr <= max_allowed_fpr)[0]
        if len(valid_indices) > 0:
            sens = float(np.max(tpr[valid_indices]))
        else:
            sens = 0.0
        results[f"sens_at_{int(target_spec * 100)}_spec"] = sens

    return results


def compute_referable_dr_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Calculate sensitivity and specificity for referable DR (grade >= 2)."""
    true_referable = (y_true >= 2).astype(int)
    pred_referable = (y_pred >= 2).astype(int)

    tp = int(np.sum((true_referable == 1) & (pred_referable == 1)))
    tn = int(np.sum((true_referable == 0) & (pred_referable == 0)))
    fp = int(np.sum((true_referable == 0) & (pred_referable == 1)))
    fn = int(np.sum((true_referable == 1) & (pred_referable == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_samples": len(y_true),
    }


def compute_quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> float:
    """Compute Quadratic Weighted Kappa (QWK) across ordinal ICDR grades (0 to 4)."""
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_mat[int(t), int(p)] += 1.0

    n = np.sum(conf_mat)
    if n == 0:
        return 0.0

    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    hist_true = np.sum(conf_mat, axis=1)
    hist_pred = np.sum(conf_mat, axis=0)
    expected = np.outer(hist_true, hist_pred) / n

    numerator = np.sum(weights * conf_mat)
    denominator = np.sum(weights * expected)

    if denominator == 0:
        return 1.0

    kappa = 1.0 - (numerator / denominator)
    return float(kappa)


def evaluate_cohort(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """Calculate full clinical metrics package: QWK, sensitivity, specificity, AUC, per-class recall, and confusion matrix."""
    metrics: Dict[str, Any] = compute_referable_dr_metrics(y_true, y_pred)
    metrics["qwk"] = compute_quadratic_weighted_kappa(y_true, y_pred, num_classes=num_classes)
    metrics["per_class_recall"] = compute_per_class_recall(y_true, y_pred, num_classes=num_classes)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

    # Probabilistic metrics (ROC-AUC, Sensitivity @ 90%/95% Specificity)
    if y_probs is not None:
        # Probability of referable DR is sum of probabilities for grades 2, 3, 4
        if y_probs.ndim == 2 and y_probs.shape[1] == num_classes:
            prob_referable = np.sum(y_probs[:, 2:], axis=1)
        elif y_probs.ndim == 1:
            prob_referable = y_probs
        else:
            prob_referable = y_probs[:, -1]

        # Referable DR AUC
        y_true_binary = (y_true >= 2).astype(int)
        if len(np.unique(y_true_binary)) >= 2:
            metrics["referable_dr_auc"] = float(roc_auc_score(y_true_binary, prob_referable))
        else:
            metrics["referable_dr_auc"] = 0.0

        # Sensitivity at 90% and 95% specificity
        sens_at_spec = compute_referable_dr_sensitivity_at_specificity(
            y_true, prob_referable, target_specificities=(0.90, 0.95)
        )
        metrics.update(sens_at_spec)

    return metrics


def evaluate_by_subgroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: Sequence[str],
    y_probs: np.ndarray | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute performance metrics broken down across a specific subgroup (device or site)."""
    subgroup_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(subgroup_labels):
        subgroup_indices[label].append(idx)

    results: Dict[str, Dict[str, Any]] = {}
    for label, indices in subgroup_indices.items():
        sub_true = y_true[indices]
        sub_pred = y_pred[indices]
        sub_probs = y_probs[indices] if y_probs is not None else None
        results[label] = evaluate_cohort(sub_true, sub_pred, y_probs=sub_probs)

    return results


def evaluate_per_device_and_site(
    records: Sequence[ImageRecord],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Evaluate performance across overall cohort, per camera device, and per clinical site."""
    overall = evaluate_cohort(y_true, y_pred, y_probs=y_probs)
    device_labels = [r.device for r in records]
    site_labels = [r.site for r in records]

    by_device = evaluate_by_subgroup(y_true, y_pred, device_labels, y_probs=y_probs)
    by_site = evaluate_by_subgroup(y_true, y_pred, site_labels, y_probs=y_probs)

    return {
        "overall": overall,
        "by_device": by_device,
        "by_site": by_site,
    }
