"""Analysis script for Tiered Escalation and Referable-Based Abstention on VAL split."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

import cv2
import numpy as np
import torch
from PIL import Image

from retinaguard_ml.data.loaders.aptos import APTOSDataset
from retinaguard_ml.models import EfficientNetDRDualHead

MANIFEST = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
CHECKPOINT = REPO_ROOT / "ml" / "runs" / "baseline_cpu_b0_256" / "best.pt"
REGISTRY = REPO_ROOT / "models" / "registry.json"

with open(REGISTRY, encoding="utf-8") as f:
    reg = json.load(f)
entry = reg["models"][0]
T_star = float(entry.get("temperature", 1.296247))

# Load model
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Run inference on VAL split (N=559)
val_ds = APTOSDataset(
    manifest_path=MANIFEST,
    cache_dir=CACHE_DIR,
    split="val",
    augment=False,
    image_size=256,
)

all_true = []
all_pred = []
all_probs = []

with torch.no_grad():
    for i in range(len(val_ds)):
        x, y = val_ds[i]
        out = model(x.unsqueeze(0))
        logits = out["logits"].squeeze(0).numpy()
        scaled = logits / T_star
        exp_l = np.exp(scaled - np.max(scaled))
        probs = exp_l / np.sum(exp_l)

        ord_probs = torch.sigmoid(out["ordinal_logits"]).squeeze(0).numpy()
        pred_grade = int(np.clip(np.round(np.sum(ord_probs)), 0, 4))

        all_true.append(int(y))
        all_pred.append(pred_grade)
        all_probs.append(probs)

y_true = np.array(all_true)
y_pred = np.array(all_pred)
probs = np.array(all_probs)

# ==============================================================================
# PART 2: TIERED ESCALATION ANALYSIS ON P(grade >= 3)
# ==============================================================================
print("=" * 70)
print("PART 2: TIERED ESCALATION ON VAL (N=559)")
print("=" * 70)

p_ge_3 = probs[:, 3] + probs[:, 4]
is_true_ge_3 = (y_true >= 3)
n_ge_3 = int(np.sum(is_true_ge_3))
print(f"Total True Grade >= 3 cases on VAL: {n_ge_3}/559 ({100*n_ge_3/len(y_true):.1f}%)")

print("\n--- Sweeping Lower Threshold (Targeting >= 95% Grade>=3 sensitivity for at least SOON) ---")
print(f"{'Thresh':<8} {'Escalated N':<12} {'Escalated %':<12} {'G>=3 Sens':<12} {'TP':<6} {'FN':<6} {'G<=2 Esc %'}")
print("-" * 65)

best_soon_th = None
for th in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
    escalated = (p_ge_3 >= th)
    n_esc = int(np.sum(escalated))
    pct_esc = 100.0 * n_esc / len(y_true)
    tp = int(np.sum(escalated & is_true_ge_3))
    fn = n_ge_3 - tp
    sens = tp / n_ge_3
    n_le_2 = int(np.sum(~is_true_ge_3))
    fp = int(np.sum(escalated & (~is_true_ge_3)))
    fp_pct = 100.0 * fp / n_le_2
    print(f"{th:<8.2f} {n_esc:<12} {pct_esc:<11.1f}% {sens:<11.3f} {tp:<6} {fn:<6} {fp_pct:.1f}%")

print("\n--- Sweeping Higher Threshold (Targeting Manageable URGENT share) ---")
print(f"{'Thresh':<8} {'URGENT N':<12} {'URGENT %':<12} {'G>=3 Sens':<12} {'TP':<6} {'FN':<6}")
print("-" * 55)
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    urgent = (p_ge_3 >= th)
    n_urg = int(np.sum(urgent))
    pct_urg = 100.0 * n_urg / len(y_true)
    tp = int(np.sum(urgent & is_true_ge_3))
    fn = n_ge_3 - tp
    sens = tp / n_ge_3
    print(f"{th:<8.2f} {n_urg:<12} {pct_urg:<11.1f}% {sens:<11.3f} {tp:<6} {fn:<6}")

# ==============================================================================
# PART 3: REFERABLE-BASED ABSTENTION ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("PART 3: REFERABLE-BASED ABSTENTION ON VAL (N=559)")
print("=" * 70)

p_ref = probs[:, 2] + probs[:, 3] + probs[:, 4]
is_true_ref = (y_true >= 2)
n_ref = int(np.sum(is_true_ref))
n_non_ref = int(np.sum(~is_true_ref))
print(f"Total True Referable (G>=2): {n_ref}/559 ({100*n_ref/len(y_true):.1f}%)")
print(f"Total True Non-Referable (G<2): {n_non_ref}/559 ({100*n_non_ref/len(y_true):.1f}%)")

# Base model without abstention (pred >= 2)
base_tp = int(np.sum((y_pred >= 2) & is_true_ref))
base_fn = n_ref - base_tp
base_tn = int(np.sum((y_pred < 2) & (~is_true_ref)))
base_fp = n_non_ref - base_tn
print(f"\nBase Model (pred >= 2 without abstention):")
print(f"  Referable Sensitivity: {base_tp}/{n_ref} = {base_tp/n_ref:.4f}")
print(f"  Referable Specificity: {base_tn}/{n_non_ref} = {base_tn/n_non_ref:.4f}")

# Band around cutoff: [low, high]
# Rule: If low <= p_ref <= high -> UNCERTAIN
# If p_ref < low -> Confident Non-referable
# If p_ref > high -> Confident Referable
# Check: Never return confident "no DR" (grade 0) when p_ref >= low!
print("\n--- Sweeping Abstention Bands [tau_low, tau_high] ---")
print(f"{'Band':<16} {'Uncertain%':<12} {'Automated%':<12} {'Ref Aut%':<12} {'Ret Sens':<10} {'Ret Spec':<10}")
print("-" * 72)

band_results = []
for low in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    for high in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        if low >= high:
            continue
        is_uncertain = (p_ref >= low) & (p_ref <= high)
        n_unc = int(np.sum(is_uncertain))
        pct_unc = 100.0 * n_unc / len(y_true)

        # Retained / Automated cases
        is_retained = ~is_uncertain
        n_ret = int(np.sum(is_retained))
        pct_ret = 100.0 * n_ret / len(y_true)

        # Share of true referable cases that get automated answer
        n_ref_auto = int(np.sum(is_retained & is_true_ref))
        pct_ref_auto = 100.0 * n_ref_auto / n_ref

        # Retained referable sensitivity & specificity
        # On retained cases:
        # predicted referable is (p_ref > high)
        # predicted non-referable is (p_ref < low)
        ret_tp = int(np.sum((p_ref > high) & is_true_ref))
        ret_fn = int(np.sum((p_ref < low) & is_true_ref))
        ret_sens = ret_tp / (ret_tp + ret_fn) if (ret_tp + ret_fn) > 0 else 0.0

        ret_tn = int(np.sum((p_ref < low) & (~is_true_ref)))
        ret_fp = int(np.sum((p_ref > high) & (~is_true_ref)))
        ret_spec = ret_tn / (ret_tn + ret_fp) if (ret_tn + ret_fp) > 0 else 0.0

        band_results.append({
            "low": low,
            "high": high,
            "unc_pct": pct_unc,
            "ret_pct": pct_ret,
            "ref_auto_pct": pct_ref_auto,
            "ret_sens": ret_sens,
            "ret_spec": ret_spec,
            "ret_tp": ret_tp,
            "ret_fn": ret_fn,
            "ret_tn": ret_tn,
            "ret_fp": ret_fp,
        })
        if ret_sens >= 0.95:
            print(f"[{low:.2f}, {high:.2f}]     {pct_unc:<11.1f}% {pct_ret:<11.1f}% {pct_ref_auto:<11.1f}% {ret_sens:<9.4f} {ret_spec:<9.4f}")
