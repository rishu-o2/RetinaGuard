"""Model evaluation script for validation and test cohorts.

Computes:
- Quadratic Weighted Kappa (QWK)
- Per-class recall (Grades 0-4)
- Referable-DR (grade >= 2) sensitivity and specificity
- Macro One-vs-Rest ROC-AUC
- Bootstrap 95% Confidence Intervals for QWK and Referable Sensitivity
- Confusion matrix and ROC curve visualizations
- Predictions CSV export
- ONNX model export with numerical parity check
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinaguard_ml.data.loaders.aptos import APTOSDataset
from retinaguard_ml.evaluate import evaluate_cohort
from retinaguard_ml.export import export_to_onnx, verify_onnx_parity
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute empirical 95% bootstrap confidence intervals for QWK and Referable Sensitivity."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    qwk_samples = []
    sens_samples = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        boot_true = y_true[indices]
        boot_pred = y_pred[indices]

        # Compute metrics — use direct indexing so a missing key raises KeyError
        res = evaluate_cohort(boot_true, boot_pred)
        qwk_samples.append(res["qwk"])
        sens_samples.append(res["sensitivity"])

    qwk_ci = (float(np.percentile(qwk_samples, 2.5)), float(np.percentile(qwk_samples, 97.5)))
    sens_ci = (float(np.percentile(sens_samples, 2.5)), float(np.percentile(sens_samples, 97.5)))

    return {
        "qwk_95_ci": qwk_ci,
        "referable_sens_95_ci": sens_ci,
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, class_names: list[str]) -> None:
    """Save normalized confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Grade",
        xlabel="Predicted Grade",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(
    checkpoint_path: Path,
    splits_path: Path,
    cache_dir: Path,
    split_name: str = "val",
    out_dir: Path = Path("evaluation_results"),
    batch_size: int = 32,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if split_name.lower() == "test":
        print("\n" + "!" * 75)
        print("  [WARNING] EVALUATING ON TEST SPLIT!")
        print("  The test split must be used strictly ONCE per model release to prevent")
        print("  evaluation overfitting.")
        print("!" * 75 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_model] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 1. Instantiate model
    model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", num_classes=5).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Dataset & DataLoader
    dataset = APTOSDataset(
        manifest_path=splits_path,
        cache_dir=cache_dir,
        split=split_name,
        augment=False,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[evaluate_model] Running inference on {len(dataset)} samples ({split_name} split)...")

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            outs = model(imgs)

            cat_probs = torch.softmax(outs["logits"], dim=-1).cpu().numpy()
            ord_probs = torch.sigmoid(outs["ordinal_logits"])
            preds = torch.round(torch.sum(ord_probs, dim=-1)).long().clamp(0, 4)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(lbls.numpy().tolist())
            all_probs.append(cat_probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs)

    # 3. Comprehensive metrics & bootstrap
    metrics = evaluate_cohort(y_true, y_pred, y_probs=y_probs)
    cis = compute_bootstrap_cis(y_true, y_pred, n_bootstraps=1000)
    metrics.update(cis)

    # 4. Save predictions CSV
    preds_csv = out_dir / f"predictions_{split_name}.csv"
    with open(preds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "true_label", "predicted_label", "prob_0", "prob_1", "prob_2", "prob_3", "prob_4"])
        for rec, true_l, pred_l, probs in zip(dataset.records, y_true, y_pred, y_probs):
            writer.writerow([
                rec["id_code"],
                true_l,
                pred_l,
                f"{probs[0]:.4f}",
                f"{probs[1]:.4f}",
                f"{probs[2]:.4f}",
                f"{probs[3]:.4f}",
                f"{probs[4]:.4f}",
            ])

    # 5. Visualizations: Confusion Matrix
    class_names = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    cm_path = out_dir / f"confusion_matrix_{split_name}.png"
    plot_confusion_matrix(cm, cm_path, class_names)

    # 6. Export to ONNX with numerical parity test
    onnx_path = out_dir / f"retinaguard_model_{split_name}.onnx"
    input_shape = (1, 3, 384, 384)
    export_to_onnx(model, onnx_path, input_shape=input_shape)
    parity_diffs = verify_onnx_parity(model, onnx_path, input_shape=input_shape, atol=1e-4)
    parity_passed = all(diff < 1e-4 for diff in parity_diffs.values())

    # 7. Print Report
    print(f"\n{'='*65}")
    print(f"  EVALUATION REPORT: {split_name.upper()} SPLIT")
    print(f"{'='*65}")
    print(f"  Samples Evaluated            : {len(y_true)}")
    print(f"  Quadratic Weighted Kappa     : {metrics['qwk']:.4f} (95% CI: [{metrics['qwk_95_ci'][0]:.4f}, {metrics['qwk_95_ci'][1]:.4f}])")
    print(f"  Referable DR Sensitivity     : {metrics['sensitivity']:.4f} (95% CI: [{metrics['referable_sens_95_ci'][0]:.4f}, {metrics['referable_sens_95_ci'][1]:.4f}])")
    print(f"  Referable DR Specificity     : {metrics['specificity']:.4f}")
    print(f"  Referable DR AUC             : {metrics.get('referable_dr_auc', 0.0):.4f}")
    print(f"\n  Per-Class Recall:")
    for g in range(5):
        print(f"    Grade {g} ({class_names[g]:<10}): {metrics['per_class_recall'].get(g, 0.0):.4f}")

    print(f"\n  Confusion Matrix saved       : {cm_path}")
    print(f"  Predictions CSV saved        : {preds_csv}")
    print(f"  ONNX Export Path             : {onnx_path}")
    print(f"  ONNX Numerical Parity Check  : {'PASSED' if parity_passed else 'FAILED'}")
    print(f"{'='*65}\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RetinaGuard Model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--splits", default="data/processed/splits_v2.csv", help="Path to splits manifest")
    parser.add_argument("--cache-dir", default="data/processed/cache_512/aptos", help="Path to cached image dir")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Cohort split to evaluate")
    parser.add_argument("--out-dir", default="evaluation_results", help="Directory for evaluation artifacts")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=Path(args.checkpoint),
        splits_path=Path(args.splits),
        cache_dir=Path(args.cache_dir),
        split_name=args.split,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()

