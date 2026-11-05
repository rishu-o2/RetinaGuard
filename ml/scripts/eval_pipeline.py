"""Evaluation pipeline: Steps 1-6.

Steps:
  1. Validation report (VAL split) — QWK, per-class recall/precision, referable metrics,
     confusion matrix PNG, bootstrap 95% CIs.
  2. Temperature calibration on VAL — ECE before/after, reliability diagram.
  3. Abstention — coverage vs accuracy / referable-sensitivity plot,
     configs/abstention.yaml with candidate thresholds.
  4. OOD fit — penultimate features on TRAIN, fit Mahalanobis, test on VAL + non-fundus.
  5. Quality gate — run over VAL, pass rate per grade, list 12 flagged images.
  6. ONNX export — parity test, registry entry, update backend model path.

Usage:
    python ml/scripts/eval_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# --- path setup ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

import numpy as np
import torch
import torch.nn as nn
import yaml
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    cohen_kappa_score,
    roc_auc_score,
    precision_score,
)
from torch.utils.data import DataLoader

from retinaguard_ml.data.loaders.aptos import APTOSDataset
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.evaluate import evaluate_cohort

# ===========================================================================
# Constants / paths
# ===========================================================================
MANIFEST = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
CHECKPOINT = REPO_ROOT / "ml" / "runs" / "baseline_cpu_b0_256" / "best.pt"
OUT_DIR = REPO_ROOT / "ml" / "evaluation" / "val_report"
MODELS_DIR = REPO_ROOT / "models"
OOD_STATS_PATH = MODELS_DIR / "ood_stats_v1.npz"
ONNX_PATH = REPO_ROOT / "ml" / "exports" / "retinaguard_dr_v1.onnx"
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
ABSTENTION_CFG = REPO_ROOT / "configs" / "abstention.yaml"

IMAGE_SIZE = [256, 256]
BATCH_SIZE = 32
DEVICE = torch.device("cpu")
NUM_CLASSES = 5
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "ml" / "exports").mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("RetinaGuard  Evaluation Pipeline  (Steps 1-6)")
print("=" * 70)
print(f"Checkpoint : {CHECKPOINT}")
print(f"Manifest   : {MANIFEST}")
print(f"Cache      : {CACHE_DIR}")
print(f"Output dir : {OUT_DIR}")
print()

# ===========================================================================
# 1.  Load model
# ===========================================================================
print("[LOAD] Loading checkpoint ...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
best_epoch = ckpt.get("epoch", "?")
best_qwk = ckpt.get("best_qwk", "?")
print(f"       epoch={best_epoch}  best_qwk={best_qwk}")


# ===========================================================================
# Helper: run inference on a split, return arrays
# ===========================================================================
def run_inference(split: str, augment: bool = False, return_features: bool = False):
    """Run model on a split; return (y_true, y_pred, y_probs[N,5], features[N,D])."""
    ds = APTOSDataset(
        manifest_path=MANIFEST,
        cache_dir=CACHE_DIR,
        split=split,
        augment=augment,
        image_size=IMAGE_SIZE,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

    all_true, all_pred, all_probs, all_feats = [], [], [], []
    hook_feats = []

    if return_features:
        # Hook penultimate layer (dropout output = after backbone, before fc heads)
        def hook_fn(module, inp, out):
            hook_feats.append(out.detach().cpu().numpy())

        handle = model.dropout.register_forward_hook(hook_fn)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)

            # Categorical head probabilities
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

            # Ordinal-sum grade prediction
            ord_probs = torch.sigmoid(outputs["ordinal_logits"])
            grades = torch.round(ord_probs.sum(dim=-1)).long().clamp(0, 4).cpu().numpy()

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(grades.tolist())
            all_probs.append(probs)

    if return_features:
        handle.remove()
        feats_arr = np.concatenate(hook_feats, axis=0)
        return (np.array(all_true), np.array(all_pred),
                np.concatenate(all_probs, axis=0), feats_arr)

    return (np.array(all_true), np.array(all_pred),
            np.concatenate(all_probs, axis=0), None)


# ===========================================================================
# STEP 1 — Validation Report
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 1  —  Validation Report (VAL split)")
print("=" * 70)

y_true, y_pred, y_probs, _ = run_inference("val")

# Core metrics
metrics = evaluate_cohort(y_true, y_pred, y_probs=y_probs, num_classes=5)

# Per-class precision
per_class_prec = {}
for c in range(NUM_CLASSES):
    mask_pred = (y_pred == c)
    mask_true = (y_true == c)
    tp = int(np.sum(mask_pred & mask_true))
    fp = int(np.sum(mask_pred & ~mask_true))
    per_class_prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

print(f"\nQWK               : {metrics['qwk']:.4f}")
print(f"Referable Sens    : {metrics['sensitivity']:.4f}")
print(f"Referable Spec    : {metrics['specificity']:.4f}")
print(f"Referable AUC     : {metrics.get('referable_dr_auc', float('nan')):.4f}")
print(f"Sens@90spec       : {metrics.get('sens_at_90_spec', float('nan')):.4f}")
print(f"Sens@95spec       : {metrics.get('sens_at_95_spec', float('nan')):.4f}")
print()
print(f"{'Grade':<8} {'Recall':<10} {'Precision':<10} {'N (true)':<12}")
print("-" * 42)
grade_counts = {c: int(np.sum(y_true == c)) for c in range(5)}
for c in range(5):
    rec = metrics["per_class_recall"].get(c, 0.0)
    prec = per_class_prec.get(c, 0.0)
    print(f"Grade {c}  {rec:.4f}     {prec:.4f}     {grade_counts[c]}")

print()
cm = np.array(metrics["confusion_matrix"])
print("Confusion matrix (rows=true, cols=pred):")
header = "       " + "  ".join(f"P{c}" for c in range(5))
print(header)
for r in range(5):
    row_str = "  ".join(f"{cm[r, c]:4d}" for c in range(5))
    print(f"True {r}:  {row_str}")

# Most common off-diagonal confusions
off_diag = [(cm[r, c], r, c) for r in range(5) for c in range(5) if r != c]
off_diag.sort(reverse=True)
print("\nTop 5 confusions (true -> pred : count):")
for cnt, r, c in off_diag[:5]:
    print(f"  Grade {r} -> {c}: {cnt}")

# Identify weakest grades
weakest = sorted(metrics["per_class_recall"].items(), key=lambda x: x[1])
print(f"\nWeakest grade by recall : Grade {weakest[0][0]} (recall={weakest[0][1]:.4f})")
print(f"2nd weakest             : Grade {weakest[1][0]} (recall={weakest[1][1]:.4f})")

# Bootstrap CIs
rng_bs = np.random.default_rng(BOOTSTRAP_SEED)
bs_qwk, bs_sens, bs_spec = [], [], []
N = len(y_true)
for _ in range(N_BOOTSTRAP):
    idx = rng_bs.integers(0, N, size=N)
    yt_, yp_, ypr_ = y_true[idx], y_pred[idx], y_probs[idx]
    m_ = evaluate_cohort(yt_, yp_, y_probs=ypr_)
    bs_qwk.append(m_["qwk"])
    bs_sens.append(m_["sensitivity"])
    bs_spec.append(m_["specificity"])

ci_qwk   = (np.percentile(bs_qwk,   2.5), np.percentile(bs_qwk,   97.5))
ci_sens   = (np.percentile(bs_sens,   2.5), np.percentile(bs_sens,   97.5))
ci_spec   = (np.percentile(bs_spec,   2.5), np.percentile(bs_spec,   97.5))

print(f"\nBootstrap 95% CIs (N={N_BOOTSTRAP} resamples):")
print(f"  QWK               : [{ci_qwk[0]:.4f}, {ci_qwk[1]:.4f}]")
print(f"  Referable Sens    : [{ci_sens[0]:.4f}, {ci_sens[1]:.4f}]")
print(f"  Referable Spec    : [{ci_spec[0]:.4f}, {ci_spec[1]:.4f}]")

# Confusion matrix PNG
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([f"Pred {i}" for i in range(5)], rotation=45)
ax.set_yticklabels([f"True {i}" for i in range(5)])
ax.set_title("Confusion Matrix (VAL split)")
for r in range(5):
    for c in range(5):
        ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                color="white" if cm[r, c] > cm.max() * 0.6 else "black")
plt.tight_layout()
cm_png = OUT_DIR / "confusion_matrix.png"
fig.savefig(str(cm_png), dpi=150)
plt.close(fig)
print(f"\nConfusion matrix PNG saved -> {cm_png}")

# Save predictions CSV
pred_csv = OUT_DIR / "predictions_val.csv"
with open(pred_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "true_grade", "pred_grade"] + [f"prob_{i}" for i in range(5)])
    for i, (yt, yp, ypr) in enumerate(zip(y_true, y_pred, y_probs)):
        w.writerow([i, int(yt), int(yp)] + [f"{p:.6f}" for p in ypr])
print(f"Predictions CSV saved -> {pred_csv}")

# ===========================================================================
# STEP 2 — Temperature Calibration
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 2  —  Temperature Calibration")
print("=" * 70)

# Collect raw categorical logits on VAL (not cached - run again capturing logits)
print("Collecting logits on VAL for calibration...")
val_ds = APTOSDataset(
    manifest_path=MANIFEST, cache_dir=CACHE_DIR,
    split="val", augment=False, image_size=IMAGE_SIZE,
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

all_logits_np = []
all_true_cal = []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        all_logits_np.append(out["logits"].cpu().numpy())
        all_true_cal.extend(labels.numpy().tolist())

logits_np = np.concatenate(all_logits_np, axis=0)   # [N, 5]
y_true_cal = np.array(all_true_cal)                  # [N]


def nll_loss(temperature: float, logits: np.ndarray, y_true: np.ndarray) -> float:
    """Negative log-likelihood for temperature scaling."""
    if temperature <= 0:
        return 1e9
    scaled = logits / temperature
    # Stable log-softmax
    max_l = scaled.max(axis=1, keepdims=True)
    log_sm = scaled - max_l - np.log(np.sum(np.exp(scaled - max_l), axis=1, keepdims=True))
    return float(-np.mean(log_sm[np.arange(len(y_true)), y_true]))


def compute_ece(logits: np.ndarray, y_true: np.ndarray,
                temperature: float = 1.0, n_bins: int = 10) -> float:
    """Expected Calibration Error over n_bins uniform confidence bins."""
    scaled = logits / max(temperature, 1e-6)
    max_l = scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_l)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


ece_before = compute_ece(logits_np, y_true_cal, temperature=1.0)
print(f"ECE before calibration (T=1.0) : {ece_before:.4f}")

# Optimize temperature
result = minimize_scalar(
    nll_loss, bounds=(0.1, 10.0), method="bounded",
    args=(logits_np, y_true_cal),
)
T_star = float(result.x)
print(f"Optimal temperature (T*)        : {T_star:.4f}")

ece_after = compute_ece(logits_np, y_true_cal, temperature=T_star)
print(f"ECE after calibration           : {ece_after:.4f}")
print(f"ECE reduction                   : {ece_before - ece_after:.4f}")

# Reliability diagram
def reliability_diagram(logits: np.ndarray, y_true: np.ndarray,
                         temperature: float, label: str, ax, n_bins: int = 10):
    scaled = logits / max(temperature, 1e-6)
    max_l = scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_l)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confs, bin_accs, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() > 0:
            bin_confs.append(confs[mask].mean())
            bin_accs.append(correct[mask].mean())
            bin_counts.append(mask.sum())
    ax.plot(bin_confs, bin_accs, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Reliability Diagram")

fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
reliability_diagram(logits_np, y_true_cal, 1.0, f"T=1.0 (ECE={ece_before:.3f})", ax_cal)
reliability_diagram(logits_np, y_true_cal, T_star, f"T={T_star:.2f} (ECE={ece_after:.3f})", ax_cal)
fig_cal.tight_layout()
rel_diag_png = OUT_DIR / "reliability_diagram.png"
fig_cal.savefig(str(rel_diag_png), dpi=150)
plt.close(fig_cal)
print(f"Reliability diagram saved -> {rel_diag_png}")

# ===========================================================================
# STEP 3 — Abstention
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 3  —  Abstention (Coverage vs Accuracy / Referable Sensitivity)")
print("=" * 70)

# Calibrated probabilities on VAL
scaled_logits = logits_np / T_star
max_l = scaled_logits.max(axis=1, keepdims=True)
exp_s = np.exp(scaled_logits - max_l)
cal_probs = exp_s / exp_s.sum(axis=1, keepdims=True)   # [N, 5]
confidences = cal_probs.max(axis=1)                     # max probability as confidence
pred_grades = cal_probs.argmax(axis=1)

# Ordinal-sum grade (from earlier inference)
# Use calibrated softmax argmax for coverage analysis (consistent with backend)
true_referable = (y_true_cal >= 2).astype(int)

thresholds = np.linspace(0.0, 0.99, 200)
coverages, accuracies, ref_sensitivities = [], [], []

for thr in thresholds:
    keep = confidences >= thr
    cov = keep.sum() / len(confidences)
    coverages.append(cov)
    if keep.sum() > 0:
        acc = float((pred_grades[keep] == y_true_cal[keep]).mean())
        # Referable sensitivity among retained samples
        ref_keep = true_referable[keep]
        pred_ref_keep = (pred_grades[keep] >= 2).astype(int)
        if ref_keep.sum() > 0:
            ref_sens = float((pred_ref_keep[ref_keep == 1] == 1).mean())
        else:
            ref_sens = 0.0
    else:
        acc, ref_sens = 0.0, 0.0
    accuracies.append(acc)
    ref_sensitivities.append(ref_sens)

fig_abs, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(coverages, accuracies, "b-")
ax1.axhline(0.9, ls="--", color="gray", label="90% accuracy")
ax1.set_xlabel("Coverage"); ax1.set_ylabel("Accuracy")
ax1.set_title("Coverage vs Accuracy"); ax1.legend()

ax2.plot(coverages, ref_sensitivities, "r-")
ax2.axhline(0.95, ls="--", color="gray", label="95% referable sens")
ax2.set_xlabel("Coverage"); ax2.set_ylabel("Referable Sensitivity")
ax2.set_title("Coverage vs Referable Sensitivity"); ax2.legend()

abs_png = OUT_DIR / "abstention_coverage.png"
fig_abs.tight_layout()
fig_abs.savefig(str(abs_png), dpi=150)
plt.close(fig_abs)
print(f"Abstention plot saved -> {abs_png}")

# Propose candidate thresholds
def find_threshold_for_coverage(target_cov: float) -> float:
    """Return the confidence threshold that retains ~target_cov fraction."""
    return float(np.percentile(confidences, (1 - target_cov) * 100))

thr_90pct_cov = find_threshold_for_coverage(0.90)
thr_75pct_cov = find_threshold_for_coverage(0.75)

# Metrics at proposed thresholds
def metrics_at_threshold(thr: float):
    keep = confidences >= thr
    cov = float(keep.sum() / len(confidences))
    if keep.sum() == 0:
        return {"coverage": 0.0, "accuracy": 0.0, "referable_sensitivity": 0.0}
    acc = float((pred_grades[keep] == y_true_cal[keep]).mean())
    ref_true_k = true_referable[keep]
    pred_ref_k = (pred_grades[keep] >= 2).astype(int)
    if ref_true_k.sum() > 0:
        rs = float((pred_ref_k[ref_true_k == 1] == 1).mean())
    else:
        rs = 0.0
    return {"coverage": cov, "accuracy": acc, "referable_sensitivity": rs}

m_high = metrics_at_threshold(thr_90pct_cov)
m_med  = metrics_at_threshold(thr_75pct_cov)

print(f"\nProposed thresholds:")
print(f"  high (90% coverage): conf >= {thr_90pct_cov:.4f} "
      f"-> acc={m_high['accuracy']:.3f}  ref_sens={m_high['referable_sensitivity']:.3f}")
print(f"  medium (75% coverage): conf >= {thr_75pct_cov:.4f} "
      f"-> acc={m_med['accuracy']:.3f}  ref_sens={m_med['referable_sensitivity']:.3f}")

abstention_cfg = {
    "version": "1.0.0",
    "note": "NEEDS CLINICAL REVIEW before deployment",
    "policy": {
        "default": "abstain_when_below_high_confidence",
        "description": (
            "When model confidence is below the high_confidence threshold, return UNCERTAIN. "
            "Policy favours not missing referable cases: when unsure, return UNCERTAIN, "
            "never a confident 'no DR'."
        ),
    },
    "thresholds": {
        "high_confidence": {
            "value": round(float(thr_90pct_cov), 4),
            "coverage_at_val": round(m_high["coverage"], 4),
            "accuracy_at_val": round(m_high["accuracy"], 4),
            "referable_sensitivity_at_val": round(m_high["referable_sensitivity"], 4),
            "note": "NEEDS CLINICAL REVIEW",
        },
        "medium_confidence": {
            "value": round(float(thr_75pct_cov), 4),
            "coverage_at_val": round(m_med["coverage"], 4),
            "accuracy_at_val": round(m_med["accuracy"], 4),
            "referable_sensitivity_at_val": round(m_med["referable_sensitivity"], 4),
            "note": "NEEDS CLINICAL REVIEW",
        },
    },
}
ABSTENTION_CFG.parent.mkdir(parents=True, exist_ok=True)
with open(ABSTENTION_CFG, "w", encoding="utf-8") as f:
    yaml.dump(abstention_cfg, f, default_flow_style=False, sort_keys=False)
print(f"configs/abstention.yaml saved -> {ABSTENTION_CFG}")

# ===========================================================================
# STEP 4 — OOD Fit
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 4  —  OOD Fit (TRAIN features -> Mahalanobis statistics)")
print("=" * 70)

print("Extracting penultimate features on TRAIN split ...")
y_true_tr, _, _, feats_train = run_inference("train", augment=False, return_features=True)
print(f"  TRAIN features shape: {feats_train.shape}  labels: {y_true_tr.shape}")

# Fit Mahalanobis statistics
num_classes = 5
feat_dim = feats_train.shape[1]
centroids = np.zeros((num_classes, feat_dim), dtype=np.float32)
centered = np.zeros_like(feats_train, dtype=np.float32)

for c in range(num_classes):
    mask = (y_true_tr == c)
    if mask.sum() > 0:
        mu_c = feats_train[mask].mean(axis=0)
        centroids[c] = mu_c
        centered[mask] = feats_train[mask] - mu_c
    else:
        centroids[c] = feats_train.mean(axis=0)

cov = (centered.T @ centered) / float(len(feats_train))
cov += 1e-4 * np.eye(feat_dim, dtype=np.float32)
inv_cov = np.linalg.pinv(cov).astype(np.float32)

# Compute distances on TRAIN (to pick threshold)
def mahal_dist(feat_vec: np.ndarray) -> float:
    z = feat_vec.flatten()
    min_d = float("inf")
    for mu_c in centroids:
        diff = z - mu_c
        d = float(np.dot(np.dot(diff, inv_cov), diff))
        if d < min_d:
            min_d = d
    return float(np.sqrt(max(min_d, 0.0)))

print("Computing Mahalanobis distances on TRAIN (sample) ...")
train_dists = np.array([mahal_dist(f) for f in feats_train[:200]])
p99_train = float(np.percentile(train_dists, 99))
print(f"  TRAIN dist p99 (first 200): {p99_train:.2f}")

# Save statistics
checksum_data = np.concatenate([centroids.flatten(), inv_cov[:5, :5].flatten()])
sha_buf = hashlib.sha256(checksum_data.tobytes()).hexdigest()[:12]
save_path = OOD_STATS_PATH
np.savez_compressed(
    str(save_path),
    centroids=centroids,
    inv_covariance=inv_cov,
    num_classes=np.array(num_classes),
    feature_dim=np.array(feat_dim),
    p99_train_dist=np.array(p99_train),
)
print(f"OOD statistics saved -> {save_path}  (sha256_prefix={sha_buf})")

# --- Test on VAL: should NOT be rejected ----
print("\nComputing Mahalanobis distances on VAL features ...")
_, _, _, feats_val = run_inference("val", augment=False, return_features=True)
# Sample 100 val images for speed
n_sample = min(100, len(feats_val))
val_dists = np.array([mahal_dist(f) for f in feats_val[:n_sample]])
threshold = max(p99_train * 2.0, 45.0)   # keep existing config threshold, min 45
n_rejected_val = int((val_dists > threshold).sum())
print(f"  VAL distance: mean={val_dists.mean():.2f}  max={val_dists.max():.2f}")
print(f"  Threshold: {threshold:.2f}")
print(f"  VAL fundus rejected ({n_sample} sampled): {n_rejected_val}  "
      f"({100*n_rejected_val/n_sample:.1f}%)")

# --- Test on non-fundus images ---
print("\nTesting OOD on non-fundus images ...")

def make_test_image(kind: str) -> np.ndarray:
    """Create a synthetic non-fundus image for OOD testing."""
    if kind == "blank_white":
        return np.full((256, 256, 3), 240, dtype=np.uint8)
    elif kind == "black":
        return np.zeros((256, 256, 3), dtype=np.uint8)
    elif kind == "natural_scene":
        # Synthetic nature-like image: green > red > blue
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = 80   # R
        img[:, :, 1] = 130  # G  (green dominates -> not fundus)
        img[:, :, 2] = 60   # B
        # Add random variation
        rng = np.random.default_rng(1)
        img = (img + rng.integers(-20, 20, size=img.shape, dtype=np.int16)).clip(0, 255).astype(np.uint8)
        return img
    elif kind == "face_skin":
        # Skin tone: R slightly > G > B, but not the deep fundus ratio
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = 180
        img[:, :, 1] = 150
        img[:, :, 2] = 130
        return img
    elif kind == "document_text":
        img = np.full((256, 256, 3), 245, dtype=np.uint8)
        img[80:90, 20:200, :] = 20
        img[110:120, 20:180, :] = 20
        return img
    return np.zeros((256, 256, 3), dtype=np.uint8)

from retinaguard.services.ood import OODDetectorService

ood_svc = OODDetectorService()
non_fundus_kinds = ["blank_white", "black", "natural_scene", "face_skin", "document_text"]
print(f"  {'Image':<20} {'OOD detected?':<16} {'Method':<30} {'r_to_b'}")
print("  " + "-" * 80)
for kind in non_fundus_kinds:
    img = make_test_image(kind)
    res = ood_svc.evaluate(img)
    detected = "YES (rejected)" if not res.is_in_distribution else "NO (passed)"
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 15
    if np.any(mask):
        r_vals = img[:, :, 0][mask]
        b_vals = img[:, :, 2][mask]
        r_to_b = float(r_vals.mean()) / max(float(b_vals.mean()), 1.0)
    else:
        r_to_b = 0.0
    print(f"  {kind:<20} {detected:<16} {str(res.method_triggered):<30} {r_to_b:.2f}")

# ===========================================================================
# STEP 5 — Quality Gate (VAL split)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 5  —  Quality Gate (VAL split)")
print("=" * 70)

from retinaguard.services.quality import QualityGateService

quality_svc = QualityGateService()

val_ds_qg = APTOSDataset(
    manifest_path=MANIFEST, cache_dir=CACHE_DIR,
    split="val", augment=False, image_size=None,  # full 512px for quality gate
)

pass_counts = {c: 0 for c in range(5)}
fail_counts = {c: 0 for c in range(5)}
flagged = []
total_per_grade = {c: 0 for c in range(5)}

for rec in val_ds_qg.records:
    id_code = rec["id_code"]
    grade = rec["diagnosis"]
    img_path = val_ds_qg.cache_dir / f"{id_code}.png"
    if not img_path.exists():
        img_path = val_ds_qg.cache_dir / f"{id_code}.jpg"
    if not img_path.exists():
        continue

    with Image.open(img_path) as im:
        img_rgb = np.array(im.convert("RGB"))

    result = quality_svc.evaluate(img_rgb)
    total_per_grade[grade] += 1
    if result.passed:
        pass_counts[grade] += 1
    else:
        fail_counts[grade] += 1
        reason = result.human_readable_reason or "unknown"
        flagged.append({"id_code": id_code, "grade": grade, "reason": reason})

print(f"\nQuality gate pass rate per grade:")
print(f"  {'Grade':<8} {'Pass':<8} {'Fail':<8} {'Total':<8} {'Pass%'}")
print("  " + "-" * 44)
for c in range(5):
    tot = total_per_grade[c]
    pas = pass_counts[c]
    fal = fail_counts[c]
    pct = 100.0 * pas / tot if tot > 0 else 0.0
    print(f"  Grade {c}  {pas:<8} {fal:<8} {tot:<8} {pct:.1f}%")

total_all = sum(total_per_grade.values())
total_pass = sum(pass_counts.values())
print(f"\n  Overall pass rate: {total_pass}/{total_all} = {100*total_pass/total_all:.1f}%")

print(f"\n12 flagged images ({len(flagged)} total flagged):")
print(f"  {'id_code':<35} {'grade':<8} {'reason'}")
print("  " + "-" * 80)
for fi in flagged[:12]:
    print(f"  {fi['id_code']:<35} {fi['grade']:<8} {fi['reason'][:60]}")

# ===========================================================================
# STEP 6 — ONNX Export
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 6  —  ONNX Export")
print("=" * 70)

from retinaguard_ml.export import export_to_onnx, verify_onnx_parity, is_onnx_available

if not is_onnx_available():
    print("WARNING: onnx / onnxruntime not installed — skipping ONNX export.")
    print("  Install with: pip install onnx onnxruntime")
    onnx_sha256 = "SKIPPED"
    onnx_path_str = "SKIPPED"
else:
    input_shape = (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    print(f"Exporting to ONNX (input_shape={input_shape}) ...")
    t0 = time.time()
    onnx_out = export_to_onnx(model, ONNX_PATH, input_shape=input_shape, opset_version=17)
    print(f"  Export done in {time.time() - t0:.1f}s -> {onnx_out}")

    print("Running parity test ...")
    diffs = verify_onnx_parity(model, ONNX_PATH, input_shape=input_shape, rtol=1e-3, atol=1e-3)
    for head, diff in diffs.items():
        print(f"  Max abs diff ({head}): {diff:.6f}  PASS" if diff < 1e-2 else
              f"  Max abs diff ({head}): {diff:.6f}  FAIL")

    # Checksum
    with open(ONNX_PATH, "rb") as f:
        onnx_sha256 = hashlib.sha256(f.read()).hexdigest()
    onnx_path_str = str(ONNX_PATH.relative_to(REPO_ROOT).as_posix())
    print(f"  ONNX SHA-256: {onnx_sha256[:16]}...")

# Registry entry
registry_entry = {
    "version": "v1.0.0",
    "epoch": int(best_epoch) if isinstance(best_epoch, (int, float)) else best_epoch,
    "best_qwk_val": float(best_qwk) if isinstance(best_qwk, float) else best_qwk,
    "checkpoint_path": str(CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
    "onnx_path": onnx_path_str,
    "onnx_sha256": onnx_sha256,
    "config_path": "configs/baseline_cpu_b0_256.yaml",
    "image_size": IMAGE_SIZE,
    "temperature": round(T_star, 6),
    "ece_before": round(ece_before, 6),
    "ece_after": round(ece_after, 6),
    "ood_stats_path": str(OOD_STATS_PATH.relative_to(REPO_ROOT).as_posix()),
    "abstention_config_path": "configs/abstention.yaml",
    "val_metrics": {
        "qwk": round(float(metrics["qwk"]), 4),
        "referable_sensitivity": round(float(metrics["sensitivity"]), 4),
        "referable_specificity": round(float(metrics["specificity"]), 4),
        "referable_auc": round(float(metrics.get("referable_dr_auc", 0.0)), 4),
        "ci_qwk_95": [round(ci_qwk[0], 4), round(ci_qwk[1], 4)],
        "ci_sens_95": [round(ci_sens[0], 4), round(ci_sens[1], 4)],
        "ci_spec_95": [round(ci_spec[0], 4), round(ci_spec[1], 4)],
    },
}
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
    json.dump({"models": [registry_entry]}, f, indent=2)
print(f"\nModel registry saved -> {REGISTRY_PATH}")

print("\n" + "=" * 70)
print("STEPS 1-6 COMPLETE")
print("=" * 70)
print(f"  Confusion matrix PNG  : {cm_png}")
print(f"  Reliability diagram   : {rel_diag_png}")
print(f"  Abstention plot       : {abs_png}")
print(f"  Predictions CSV       : {pred_csv}")
print(f"  OOD statistics        : {OOD_STATS_PATH}")
print(f"  ONNX model            : {ONNX_PATH}")
print(f"  Registry              : {REGISTRY_PATH}")
print(f"  abstention.yaml       : {ABSTENTION_CFG}")
print()

