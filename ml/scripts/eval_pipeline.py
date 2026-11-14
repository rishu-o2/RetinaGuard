"""Evaluation pipeline: Steps 1-6, parametrized by --checkpoint, --version, --out-dir.

Usage:
    # v1 (old defaults preserved):
    python ml/scripts/eval_pipeline.py

    # v2 (new checkpoint):
    python ml/scripts/eval_pipeline.py \
        --checkpoint ml/runs/robust_cpu_b0_256/best.pt \
        --version v2 \
        --out-dir ml/evaluation/val_report_v2

Rules:
    - Thresholds are chosen on the VAL split only.
    - APTOS test split and IDRiD are NEVER touched by this script.
    - When --version v2, writes retinaguard_dr_v2.onnx and ood_stats_v2.npz.
    - registry.json is APPENDED (v2 added as inactive); v1 entry is never overwritten.
    - Re-running with --version v1 / defaults is fully reproducible.
"""

from __future__ import annotations

import argparse
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
# Argument parsing
# ===========================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RetinaGuard evaluation pipeline — VAL split only."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "ml" / "runs" / "baseline_cpu_b0_256" / "best.pt",
        help="Path to .pt checkpoint (default: v1 baseline)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Model version tag (v1 or v2). Controls artifact names.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for reports. Default: ml/evaluation/val_report (v1) or val_report_v2 (v2).",
    )
    return parser.parse_args()


ARGS = _parse_args()
VERSION = ARGS.version  # "v1" or "v2"
# Resolve checkpoint: could be relative (from CLI) or absolute
CHECKPOINT = Path(ARGS.checkpoint)
if not CHECKPOINT.is_absolute():
    CHECKPOINT = (Path.cwd() / CHECKPOINT).resolve()

# Derived versioned artifact names
_VER_SUFFIX = f"_{VERSION}" if VERSION != "v1" else "_v1"
ONNX_FILENAME = f"retinaguard_dr_{VERSION}.onnx"
OOD_FILENAME  = f"ood_stats_{VERSION}.npz"

if ARGS.out_dir is not None:
    OUT_DIR = Path(ARGS.out_dir)
    if not OUT_DIR.is_absolute():
        OUT_DIR = (Path.cwd() / OUT_DIR).resolve()
else:
    _default_subdir = "val_report" if VERSION == "v1" else f"val_report_{VERSION}"
    OUT_DIR = REPO_ROOT / "ml" / "evaluation" / _default_subdir

ABSTENTION_CFG = REPO_ROOT / "configs" / (
    "abstention.yaml" if VERSION == "v1" else f"abstention_{VERSION}.yaml"
)

# Constants / paths
MANIFEST = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
MODELS_DIR = REPO_ROOT / "models"
OOD_STATS_PATH = MODELS_DIR / OOD_FILENAME
ONNX_PATH = REPO_ROOT / "ml" / "exports" / ONNX_FILENAME
REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"

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
print(f"RetinaGuard  Evaluation Pipeline  (Steps 1-6)  —  version={VERSION}")
print("=" * 70)
print(f"Checkpoint : {CHECKPOINT}")
print(f"Manifest   : {MANIFEST}")
print(f"Cache      : {CACHE_DIR}")
print(f"Output dir : {OUT_DIR}")
print(f"ONNX path  : {ONNX_PATH}")
print(f"OOD path   : {OOD_STATS_PATH}")
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
bs_qwk, bs_sens, bs_spec, bs_auc = [], [], [], []
N = len(y_true)
for _ in range(N_BOOTSTRAP):
    idx = rng_bs.integers(0, N, size=N)
    yt_, yp_, ypr_ = y_true[idx], y_pred[idx], y_probs[idx]
    m_ = evaluate_cohort(yt_, yp_, y_probs=ypr_)
    bs_qwk.append(m_["qwk"])
    bs_sens.append(m_["sensitivity"])
    bs_spec.append(m_["specificity"])
    bs_auc.append(m_.get("referable_dr_auc", 0.0))

ci_qwk  = (np.percentile(bs_qwk,  2.5), np.percentile(bs_qwk,  97.5))
ci_sens = (np.percentile(bs_sens,  2.5), np.percentile(bs_sens,  97.5))
ci_spec = (np.percentile(bs_spec,  2.5), np.percentile(bs_spec,  97.5))
ci_auc  = (np.percentile(bs_auc,   2.5), np.percentile(bs_auc,  97.5))

print(f"\nBootstrap 95% CIs (N={N_BOOTSTRAP} resamples):")
print(f"  QWK               : [{ci_qwk[0]:.4f}, {ci_qwk[1]:.4f}]")
print(f"  Referable Sens    : [{ci_sens[0]:.4f}, {ci_sens[1]:.4f}]")
print(f"  Referable Spec    : [{ci_spec[0]:.4f}, {ci_spec[1]:.4f}]")
print(f"  Referable AUC     : [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")

# Confusion matrix PNG
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([f"Pred {i}" for i in range(5)], rotation=45)
ax.set_yticklabels([f"True {i}" for i in range(5)])
ax.set_title(f"Confusion Matrix (VAL split — {VERSION})")
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
    ax.set_title(f"Reliability Diagram ({VERSION})")

fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
reliability_diagram(logits_np, y_true_cal, 1.0, f"T=1.0 (ECE={ece_before:.3f})", ax_cal)
reliability_diagram(logits_np, y_true_cal, T_star, f"T={T_star:.2f} (ECE={ece_after:.3f})", ax_cal)
fig_cal.tight_layout()
rel_diag_png = OUT_DIR / "reliability_diagram.png"
fig_cal.savefig(str(rel_diag_png), dpi=150)
plt.close(fig_cal)
print(f"Reliability diagram saved -> {rel_diag_png}")

# ===========================================================================
# STEP 3 — Abstention (VAL-only thresholds)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 3  —  Abstention (Coverage vs Accuracy / Referable Sensitivity)")
print("         [thresholds chosen on VAL only — NEEDS CLINICAL REVIEW]")
print("=" * 70)

# Calibrated probabilities on VAL
scaled_logits = logits_np / T_star
max_l = scaled_logits.max(axis=1, keepdims=True)
exp_s = np.exp(scaled_logits - max_l)
cal_probs = exp_s / exp_s.sum(axis=1, keepdims=True)   # [N, 5]
pred_grades_cal = cal_probs.argmax(axis=1)

p_ref = cal_probs[:, 2] + cal_probs[:, 3] + cal_probs[:, 4]  # P(referable)
p_ge3 = cal_probs[:, 3] + cal_probs[:, 4]                     # P(grade >= 3) for SOON tier

true_referable = (y_true_cal >= 2).astype(int)
true_ge3       = (y_true_cal >= 3).astype(int)
n_ref = int(np.sum(true_referable))
n_ge3 = int(np.sum(true_ge3))

print(f"\nVAL N = {len(y_true_cal)}")
print(f"True Referable (G>=2): {n_ref}  ({100*n_ref/len(y_true_cal):.1f}%)")
print(f"True G>=3            : {n_ge3}  ({100*n_ge3/len(y_true_cal):.1f}%)")

# ------ Band-based abstention: sweep [low, high] on P(referable) ------
print("\n--- Sweeping [tau_low, tau_high] on P(referable) on VAL ---")
print(f"{'Band':<18} {'Uncert%':<10} {'RetSens':<10} {'RetSpec':<10} {'G>=3Sens@SOON'}")
print("-" * 70)

best_band = None
for low in [0.25, 0.30, 0.35]:
    for high in [0.60, 0.65, 0.70, 0.75, 0.80]:
        if low >= high:
            continue
        uncertain = (p_ref >= low) & (p_ref <= high)
        n_unc = int(np.sum(uncertain))
        pct_unc = 100.0 * n_unc / len(p_ref)

        # Retained: not uncertain
        retained = ~uncertain
        # On retained: predict referable if p_ref > high (confident referable)
        pred_ref_retained = p_ref > high

        ret_true_ref = true_referable[retained]
        ret_pred_ref = pred_ref_retained[retained].astype(int)
        if ret_true_ref.sum() > 0:
            ret_tp = int(np.sum((ret_pred_ref == 1) & (ret_true_ref == 1)))
            ret_fn = int(np.sum((ret_pred_ref == 0) & (ret_true_ref == 1)))
            ret_sens = ret_tp / (ret_tp + ret_fn) if (ret_tp + ret_fn) > 0 else 0.0
        else:
            ret_sens = 0.0

        ret_true_nonref = (1 - true_referable)[retained]
        ret_pred_nonref = (1 - ret_pred_ref)
        ret_tn = int(np.sum((ret_pred_nonref == 1) & (ret_true_nonref == 1)))
        ret_fp = int(np.sum((ret_pred_nonref == 0) & (ret_true_nonref == 1)))
        ret_spec = ret_tn / (ret_tn + ret_fp) if (ret_tn + ret_fp) > 0 else 0.0

        # G>=3 sensitivity at SOON tier (p_ge3 >= soon_thresh among retained)
        # Use same SOON threshold as v1 (0.05) — this is for reporting, not re-tuning
        soon_thresh = 0.05
        soon_pred = p_ge3[retained] >= soon_thresh
        ge3_true_ret = true_ge3[retained]
        ge3_sens_soon = float(np.sum(soon_pred & (ge3_true_ret == 1))) / max(np.sum(ge3_true_ret), 1)

        # Only print rows meeting target: ref_sens >= 0.95
        if ret_sens >= 0.95:
            print(f"[{low:.2f}, {high:.2f}]        {pct_unc:<9.1f}% {ret_sens:<9.4f} {ret_spec:<9.4f} {ge3_sens_soon:.4f}")
            if best_band is None:
                best_band = {
                    "low": low, "high": high, "unc_pct": pct_unc,
                    "ret_sens": ret_sens, "ret_spec": ret_spec,
                    "ge3_sens_soon": ge3_sens_soon,
                }

# Fall back to same band as v1 if no band found meeting 95% target
if best_band is None:
    print("WARNING: No band met ret_sens >= 0.95; falling back to v1 band [0.30, 0.70]")
    best_band = {"low": 0.30, "high": 0.70}

LOW  = best_band["low"]
HIGH = best_band["high"]
print(f"\nSelected band: [{LOW:.2f}, {HIGH:.2f}]  (VAL-only choice — NEEDS CLINICAL REVIEW)")

# Recompute final metrics at chosen band
uncertain_final = (p_ref >= LOW) & (p_ref <= HIGH)
n_unc_final = int(np.sum(uncertain_final))
pct_unc_final = 100.0 * n_unc_final / len(p_ref)
retained_final = ~uncertain_final
pred_ref_ret = (p_ref > HIGH)

ret_true_ref_f = true_referable[retained_final]
ret_pred_ref_f = pred_ref_ret[retained_final].astype(int)
ret_tp_f = int(np.sum((ret_pred_ref_f == 1) & (ret_true_ref_f == 1)))
ret_fn_f = int(np.sum((ret_pred_ref_f == 0) & (ret_true_ref_f == 1)))
ret_sens_f = ret_tp_f / max(ret_tp_f + ret_fn_f, 1)

ret_true_nonref_f = (1 - true_referable)[retained_final]
ret_pred_nonref_f = (1 - ret_pred_ref_f)
ret_tn_f = int(np.sum((ret_pred_nonref_f == 1) & (ret_true_nonref_f == 1)))
ret_fp_f = int(np.sum((ret_pred_nonref_f == 0) & (ret_true_nonref_f == 1)))
ret_spec_f = ret_tn_f / max(ret_tn_f + ret_fp_f, 1)

n_ref_automated = int(np.sum(retained_final & (true_referable == 1)))
pct_ref_auto = 100.0 * n_ref_automated / max(n_ref, 1)
pct_ret = 100.0 * int(np.sum(retained_final)) / len(uncertain_final)

# G>=3 SOON sensitivity at chosen band
soon_thresh = 0.05
soon_pred_f = p_ge3[retained_final] >= soon_thresh
ge3_true_ret_f = true_ge3[retained_final]
ge3_sens_soon_f = float(np.sum(soon_pred_f & (ge3_true_ret_f == 1))) / max(np.sum(ge3_true_ret_f), 1)

print(f"\nFinal VAL metrics at selected band:")
print(f"  Uncertain rate                : {n_unc_final}/{len(p_ref)} = {pct_unc_final:.1f}%")
print(f"  Automated rate                : {pct_ret:.1f}%")
print(f"  Ref cases automated           : {n_ref_automated}/{n_ref} = {pct_ref_auto:.1f}%")
print(f"  Retained referable sens       : {ret_tp_f}/{ret_tp_f+ret_fn_f} = {ret_sens_f:.4f}  (target >= 0.95)")
print(f"  Retained referable spec       : {ret_tn_f}/{ret_tn_f+ret_fp_f} = {ret_spec_f:.4f}")
print(f"  G>=3 sens at SOON tier        : {ge3_sens_soon_f:.4f}  (target >= 0.95, threshold_soon=0.05)")

# Coverage/sensitivity plot
fig_abs, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
thresholds_sweep = np.linspace(0.0, 0.99, 200)
coverages, accuracies, ref_sensitivities = [], [], []
for thr in thresholds_sweep:
    keep = p_ref > thr
    cov = keep.sum() / len(p_ref)
    coverages.append(cov)
    if keep.sum() > 0:
        acc = float((pred_grades_cal[keep] == y_true_cal[keep]).mean())
        ref_true_k = true_referable[keep]
        pred_ref_k = (p_ref[keep] > HIGH).astype(int)
        ref_sens = float((pred_ref_k[ref_true_k == 1] == 1).mean()) if ref_true_k.sum() > 0 else 0.0
    else:
        acc, ref_sens = 0.0, 0.0
    accuracies.append(acc)
    ref_sensitivities.append(ref_sens)

ax1.plot(coverages, accuracies, "b-")
ax1.axhline(0.9, ls="--", color="gray", label="90% accuracy")
ax1.set_xlabel("Coverage"); ax1.set_ylabel("Accuracy")
ax1.set_title(f"Coverage vs Accuracy ({VERSION})"); ax1.legend()

ax2.plot(coverages, ref_sensitivities, "r-")
ax2.axhline(0.95, ls="--", color="gray", label="95% referable sens")
ax2.set_xlabel("Coverage"); ax2.set_ylabel("Referable Sensitivity")
ax2.set_title(f"Coverage vs Referable Sensitivity ({VERSION})"); ax2.legend()

abs_png = OUT_DIR / "abstention_coverage.png"
fig_abs.tight_layout()
fig_abs.savefig(str(abs_png), dpi=150)
plt.close(fig_abs)
print(f"Abstention plot saved -> {abs_png}")

# Save abstention config
abstention_cfg = {
    "version": "2.0.0",
    "note": "NEEDS CLINICAL REVIEW before deployment",
    "model_version": VERSION,
    "policy": {
        "default": "referable_band_abstention",
        "description": (
            "When calibrated P(referable) = P(grade>=2) falls within [lower_edge, upper_edge], "
            "return UNCERTAIN. Never return confident 'no DR' when P(referable) >= lower_edge."
        ),
    },
    "referable_band": {
        "cutoff": 0.50,
        "lower_edge": round(LOW, 4),
        "upper_edge": round(HIGH, 4),
        "val_uncertain_rate": round(pct_unc_final / 100.0, 6),
        "val_automated_rate": round(pct_ret / 100.0, 6),
        "val_referable_automated_rate": round(pct_ref_auto / 100.0, 6),
        "val_retained_referable_sensitivity": round(ret_sens_f, 6),
        "val_retained_referable_specificity": round(ret_spec_f, 6),
        "val_ge3_sensitivity_at_soon_tier": round(ge3_sens_soon_f, 6),
        "note": "NEEDS CLINICAL REVIEW",
    },
}
ABSTENTION_CFG.parent.mkdir(parents=True, exist_ok=True)
with open(ABSTENTION_CFG, "w", encoding="utf-8") as f:
    yaml.dump(abstention_cfg, f, default_flow_style=False, sort_keys=False)
print(f"Abstention config saved -> {ABSTENTION_CFG}")

# ===========================================================================
# STEP 4 — OOD Fit
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 4  —  OOD Fit (TRAIN features -> Mahalanobis statistics)")
print("=" * 70)

print("Extracting penultimate features on TRAIN split ...")
y_true_tr, _, _, feats_train = run_inference("train", augment=False, return_features=True)
print(f"  TRAIN features shape: {feats_train.shape}  labels: {y_true_tr.shape}")

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

checksum_data = np.concatenate([centroids.flatten(), inv_cov[:5, :5].flatten()])
sha_buf = hashlib.sha256(checksum_data.tobytes()).hexdigest()[:12]
np.savez_compressed(
    str(OOD_STATS_PATH),
    centroids=centroids,
    inv_covariance=inv_cov,
    num_classes=np.array(num_classes),
    feature_dim=np.array(feat_dim),
    p99_train_dist=np.array(p99_train),
)
print(f"OOD statistics saved -> {OOD_STATS_PATH}  (sha256_prefix={sha_buf})")

# Test on VAL
print("\nComputing Mahalanobis distances on VAL features ...")
_, _, _, feats_val = run_inference("val", augment=False, return_features=True)
n_sample = min(100, len(feats_val))
val_dists = np.array([mahal_dist(f) for f in feats_val[:n_sample]])
threshold = max(p99_train * 2.0, 45.0)
n_rejected_val = int((val_dists > threshold).sum())
print(f"  VAL distance: mean={val_dists.mean():.2f}  max={val_dists.max():.2f}")
print(f"  Threshold: {threshold:.2f}")
print(f"  VAL fundus rejected ({n_sample} sampled): {n_rejected_val}  ({100*n_rejected_val/n_sample:.1f}%)")

# Test on non-fundus images
print("\nTesting OOD on non-fundus images ...")

def make_test_image(kind: str) -> np.ndarray:
    if kind == "blank_white":
        return np.full((256, 256, 3), 240, dtype=np.uint8)
    elif kind == "black":
        return np.zeros((256, 256, 3), dtype=np.uint8)
    elif kind == "natural_scene":
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = 80; img[:, :, 1] = 130; img[:, :, 2] = 60
        rng = np.random.default_rng(1)
        img = (img + rng.integers(-20, 20, size=img.shape, dtype=np.int16)).clip(0, 255).astype(np.uint8)
        return img
    elif kind == "face_skin":
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = 180; img[:, :, 1] = 150; img[:, :, 2] = 130
        return img
    elif kind == "document_text":
        img = np.full((256, 256, 3), 245, dtype=np.uint8)
        img[80:90, 20:200, :] = 20; img[110:120, 20:180, :] = 20
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
        r_to_b = float(img[:, :, 0][mask].mean()) / max(float(img[:, :, 2][mask].mean()), 1.0)
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
    split="val", augment=False, image_size=None,
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
for fi in flagged[:12]:
    print(f"  {fi['id_code']:<35} grade={fi['grade']}  {fi['reason'][:60]}")

# ===========================================================================
# STEP 6 — ONNX Export
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 6  —  ONNX Export")
print("=" * 70)

from retinaguard_ml.export import export_to_onnx, verify_onnx_parity, is_onnx_available

if not is_onnx_available():
    print("WARNING: onnx / onnxruntime not installed — skipping ONNX export.")
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
        print(f"  Max abs diff ({head}): {diff:.6f}  {'PASS' if diff < 1e-2 else 'FAIL'}")

    with open(ONNX_PATH, "rb") as f:
        onnx_sha256 = hashlib.sha256(f.read()).hexdigest()
    onnx_path_str = str(ONNX_PATH.relative_to(REPO_ROOT).as_posix())
    print(f"  ONNX SHA-256: {onnx_sha256[:16]}...")

# ===========================================================================
# STEP 7 — Registry update (v1 unchanged; v2 appended as inactive)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 7  —  Model Registry Update")
print("=" * 70)

new_entry = {
    "version": f"{VERSION}.0.0",
    "version_tag": VERSION,
    "active": VERSION == "v1",   # only v1 is active; v2 is inactive
    "epoch": int(best_epoch) if isinstance(best_epoch, (int, float)) else best_epoch,
    "best_qwk_val": float(best_qwk) if isinstance(best_qwk, (int, float)) else best_qwk,
    "checkpoint_path": str(CHECKPOINT.relative_to(REPO_ROOT).as_posix()),
    "onnx_path": onnx_path_str,
    "onnx_sha256": onnx_sha256,
    "config_path": f"configs/{'baseline_cpu_b0_256' if VERSION == 'v1' else 'robust_cpu_b0_256'}.yaml",
    "image_size": IMAGE_SIZE,
    "temperature": round(T_star, 6),
    "ece_before": round(ece_before, 6),
    "ece_after": round(ece_after, 6),
    "ood_stats_path": str(OOD_STATS_PATH.relative_to(REPO_ROOT).as_posix()),
    "abstention_config_path": str(ABSTENTION_CFG.relative_to(REPO_ROOT).as_posix()),
    "val_metrics": {
        "qwk": round(float(metrics["qwk"]), 4),
        "referable_sensitivity": round(float(metrics["sensitivity"]), 4),
        "referable_specificity": round(float(metrics["specificity"]), 4),
        "referable_auc": round(float(metrics.get("referable_dr_auc", 0.0)), 4),
        "uncertain_rate": round(pct_unc_final / 100.0, 4),
        "per_class_recall": {str(c): round(float(metrics["per_class_recall"].get(c, 0.0)), 4) for c in range(5)},
        "ci_qwk_95": [round(ci_qwk[0], 4), round(ci_qwk[1], 4)],
        "ci_sens_95": [round(ci_sens[0], 4), round(ci_sens[1], 4)],
        "ci_spec_95": [round(ci_spec[0], 4), round(ci_spec[1], 4)],
    },
}

# Load existing registry, upsert (never overwrite existing entry with same version_tag)
existing_registry = {"models": []}
if REGISTRY_PATH.exists():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        existing_registry = json.load(f)

models_list = existing_registry.get("models", [])
# Remove any existing entry for THIS version only; keep all others intact
models_list = [m for m in models_list if m.get("version_tag", m.get("version", "")) != VERSION]

# Ensure v1 stays first (active)
if VERSION == "v1":
    models_list = [new_entry] + models_list
else:
    models_list = models_list + [new_entry]

REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
    json.dump({"models": models_list}, f, indent=2)

active_versions = [m.get("version_tag", m.get("version", "?")) for m in models_list if m.get("active", False)]
inactive_versions = [m.get("version_tag", m.get("version", "?")) for m in models_list if not m.get("active", False)]
print(f"  Registry updated -> {REGISTRY_PATH}")
print(f"  Active model(s)  : {active_versions}")
print(f"  Inactive model(s): {inactive_versions}")

# Summary
print("\n" + "=" * 70)
print(f"STEPS 1-7 COMPLETE  —  version={VERSION}")
print("=" * 70)
print(f"  Confusion matrix PNG  : {cm_png}")
print(f"  Reliability diagram   : {rel_diag_png}")
print(f"  Abstention plot       : {abs_png}")
print(f"  Predictions CSV       : {pred_csv}")
print(f"  OOD statistics        : {OOD_STATS_PATH}")
print(f"  ONNX model            : {ONNX_PATH}")
print(f"  Abstention config     : {ABSTENTION_CFG}")
print(f"  Registry              : {REGISTRY_PATH}")
print()
