"""Analysis script for Quality Gate calibration and Sight-Threatening Escalation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from retinaguard.core.models import CameraType
from retinaguard.services.quality import QualityGateService

MANIFEST = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
VAL_PRED_CSV = REPO_ROOT / "ml" / "evaluation" / "val_report" / "predictions_val.csv"
OUT_DIR = REPO_ROOT / "ml" / "evaluation" / "val_report"

# ==============================================================================
# 1. QUALITY GATE: TRAIN split blur distribution (N=2551)
# ==============================================================================
print("=" * 70)
print("1. QUALITY GATE CALIBRATION ON TRAIN SPLIT")
print("=" * 70)

train_records = []
val_records = []
with open(MANIFEST, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == "train":
            train_records.append(row)
        elif row["split"] == "val":
            val_records.append(row)

print(f"Loaded {len(train_records)} TRAIN records, {len(val_records)} VAL records.")

svc = QualityGateService()

def compute_retina_blur_512(img_rgb: np.ndarray) -> float:
    """Compute sharpness on a 512px fixed-size version of the cropped retina."""
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask, _, _ = svc.segment_fundus_fov(img_rgb)
    y_idx, x_idx = np.where(mask == 1)
    if len(y_idx) > 0 and len(x_idx) > 0:
        ymin, ymax = int(np.min(y_idx)), int(np.max(y_idx))
        xmin, xmax = int(np.min(x_idx)), int(np.max(x_idx))
        cropped_g = img_gray[ymin:ymax + 1, xmin:xmax + 1]
        cropped_m = mask[ymin:ymax + 1, xmin:xmax + 1]
    else:
        cropped_g = img_gray
        cropped_m = mask

    res_g = cv2.resize(cropped_g, (512, 512), interpolation=cv2.INTER_AREA)
    res_m = cv2.resize(cropped_m, (512, 512), interpolation=cv2.INTER_NEAREST)
    return svc.calculate_blur_variance(res_g, res_m)

# Compute blur for all TRAIN images
train_scores = []
for r in train_records:
    code = r["id_code"]
    p = CACHE_DIR / f"{code}.png"
    if not p.exists():
        p = CACHE_DIR / f"{code}.jpg"
    if not p.exists():
        continue
    with Image.open(p) as im:
        arr = np.array(im.convert("RGB"))
    bv = compute_retina_blur_512(arr)
    train_scores.append((bv, code, int(r["diagnosis"])))

train_scores.sort(key=lambda x: x[0])
train_bvs = np.array([x[0] for x in train_scores])

p1 = float(np.percentile(train_bvs, 1))
p3 = float(np.percentile(train_bvs, 3))
p4 = float(np.percentile(train_bvs, 4))
p5 = float(np.percentile(train_bvs, 5))
p10 = float(np.percentile(train_bvs, 10))
median = float(np.median(train_bvs))

print(f"TRAIN blur variance percentiles (N={len(train_bvs)}):")
print(f"  p1  = {p1:.2f}")
print(f"  p3  = {p3:.2f}")
print(f"  p4  = {p4:.2f}")
print(f"  p5  = {p5:.2f}")
print(f"  p10 = {p10:.2f}")
print(f"  median = {median:.2f}")

# Target roughly 3-5%: choose threshold at p4 (~4% of TRAIN)
blur_threshold = round(p4, 2)
pct_flagged_train = 100.0 * np.sum(train_bvs < blur_threshold) / len(train_bvs)
print(f"\nChosen blur threshold: {blur_threshold:.2f} (flags {pct_flagged_train:.2f}% of TRAIN)")

# Create 24 flagged TRAIN images grid (4 rows x 6 cols)
flagged_train = train_scores[:24]
fig, axes = plt.subplots(4, 6, figsize=(15, 10))
for ax, (bv, code, grade) in zip(axes.flat, flagged_train):
    p = CACHE_DIR / f"{code}.png"
    if not p.exists():
        p = CACHE_DIR / f"{code}.jpg"
    with Image.open(p) as im:
        ax.imshow(im)
    ax.set_title(f"G{grade} | blur={bv:.1f}\n{code[:8]}", fontsize=8)
    ax.axis("off")

plt.suptitle(f"24 Most Blurry TRAIN Images (Sharpness < {blur_threshold:.2f})", fontsize=14)
plt.tight_layout()
grid_png = OUT_DIR / "flagged_train_blurry_grid.png"
fig.savefig(str(grid_png), dpi=150)
plt.close(fig)
print(f"24 flagged TRAIN images grid saved -> {grid_png}")

# Now evaluate on VAL split using the new 512px blur check + FOV + resolution checks
print("\nEvaluating VAL split with new fixed-size blur check ...")
raw_dir = REPO_ROOT / "data" / "raw" / "aptos" / "train_images"

val_pass_per_grade = {c: 0 for c in range(5)}
val_total_per_grade = {c: 0 for c in range(5)}
val_fail_reasons = []

for r in val_records:
    code = r["id_code"]
    grade = int(r["diagnosis"])
    val_total_per_grade[grade] += 1

    # Load raw image for resolution and FOV checks
    raw_p = raw_dir / f"{code}.png"
    if not raw_p.exists():
        raw_p = raw_dir / f"{code}.jpg"
    if not raw_p.exists():
        continue
    with Image.open(raw_p) as im:
        raw_arr = np.array(im.convert("RGB"))

    h, w = raw_arr.shape[:2]
    mask, area_ratio, circularity = svc.segment_fundus_fov(raw_arr)

    # Resolution check (tabletop min 1024x1024)
    res_pass = (w >= 1024 and h >= 1024)
    # FOV check
    fov_pass = (area_ratio >= 0.30 and circularity >= 0.60)

    # Fixed 512px blur check
    bv = compute_retina_blur_512(raw_arr)
    blur_pass = (bv >= blur_threshold)

    passed = res_pass and fov_pass and blur_pass
    if passed:
        val_pass_per_grade[grade] += 1
    else:
        reasons = []
        if not res_pass: reasons.append("RESOLUTION_TOO_LOW")
        if not fov_pass: reasons.append("INVALID_FOV")
        if not blur_pass: reasons.append(f"BLUR (var={bv:.1f}<{blur_threshold})")
        val_fail_reasons.append({"code": code, "grade": grade, "reasons": reasons})

print("\nNew Quality Gate pass rate per grade on VAL:")
print(f"  {'Grade':<8} {'Pass':<8} {'Total':<8} {'Pass Rate'}")
print("  " + "-" * 36)
for c in range(5):
    p_cnt = val_pass_per_grade[c]
    t_cnt = val_total_per_grade[c]
    pct = 100.0 * p_cnt / t_cnt if t_cnt > 0 else 0.0
    print(f"  Grade {c}  {p_cnt:<8} {t_cnt:<8} {pct:.1f}%")

tot_p = sum(val_pass_per_grade.values())
tot_t = sum(val_total_per_grade.values())
print(f"\n  Overall VAL Pass Rate: {tot_p}/{tot_t} = {100*tot_p/tot_t:.1f}%")
print(f"  Total flagged on VAL: {len(val_fail_reasons)} ({100*len(val_fail_reasons)/tot_t:.1f}%)")


# ==============================================================================
# 2. SIGHT-THREATENING ESCALATION (VAL split only)
# ==============================================================================
print("\n" + "=" * 70)
print("2. SIGHT-THREATENING ESCALATION ANALYSIS (VAL split only)")
print("=" * 70)

# Load predictions_val.csv
val_preds = []
with open(VAL_PRED_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        val_preds.append({
            "true_grade": int(row["true_grade"]),
            "pred_grade": int(row["pred_grade"]),
            "prob_3": float(row["prob_3"]),
            "prob_4": float(row["prob_4"]),
            "prob_ge_3": float(row["prob_3"]) + float(row["prob_4"]),
        })

N_val = len(val_preds)
true_ge_3 = np.array([1 if r["true_grade"] >= 3 else 0 for r in val_preds])
prob_ge_3_arr = np.array([r["prob_ge_3"] for r in val_preds])
pred_grades = np.array([r["pred_grade"] for r in val_preds])
N_true_ge_3 = int(np.sum(true_ge_3))

print(f"Total VAL samples: {N_val}")
print(f"Total True Grade >= 3: {N_true_ge_3} (Grade 3: {np.sum([r['true_grade']==3 for r in val_preds])}, Grade 4: {np.sum([r['true_grade']==4 for r in val_preds])})")

# Evaluate thresholds
candidate_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
print(f"\n{'Threshold':<12} {'Grade>=3 Sens':<16} {'G>=3 TP/Total':<16} {'Escalated Total':<18} {'Share Escalated':<16} {'G2 Escalated':<14}")
print("-" * 92)

results_escalation = []
for thr in candidate_thresholds:
    # A case is escalated if prob_ge_3 >= thr
    # In clinical rules, if predicted grade is 3 or 4, base urgency is already URGENT or EMERGENCY.
    # The rule escalates when predicted grade is 2 (or <3) and prob_ge_3 >= thr to at least URGENT.
    escalated_mask = (prob_ge_3_arr >= thr)
    
    # Sensitivity for detecting True Grade >= 3 (either via pred >= 3 OR escalated via prob_ge_3 >= thr):
    detected_mask = (pred_grades >= 3) | escalated_mask
    tp = int(np.sum((true_ge_3 == 1) & detected_mask))
    sens = tp / N_true_ge_3

    # Share of VAL images escalated:
    # Specifically, cases where base urgency would be < URGENT (pred_grade <= 2) that get escalated to URGENT:
    g2_or_lower_escalated = np.sum((pred_grades <= 2) & escalated_mask)
    total_escalated = np.sum(escalated_mask)
    share_escalated = 100.0 * total_escalated / N_val
    share_g2_escalated = 100.0 * g2_or_lower_escalated / N_val

    results_escalation.append({
        "threshold": thr,
        "sensitivity": sens,
        "tp": tp,
        "total_escalated": int(total_escalated),
        "share_escalated": share_escalated,
        "g2_escalated": int(g2_or_lower_escalated),
        "share_g2_escalated": share_g2_escalated,
    })

    print(f"{thr:<12.2f} {sens:<16.4f} {f'{tp}/{N_true_ge_3}':<16} {f'{total_escalated}/{N_val}':<18} {f'{share_escalated:.1f}%':<16} {f'{g2_or_lower_escalated} ({share_g2_escalated:.1f}%)':<14}")

# Find threshold that achieves >= 95% sensitivity for Grade >= 3 with minimal over-escalation
best_cand = [r for r in results_escalation if r["sensitivity"] >= 0.95]
if best_cand:
    selected = best_cand[-1]  # highest threshold meeting >= 95% sens
else:
    selected = results_escalation[0]

print(f"\nRecommended Escalation Threshold: {selected['threshold']:.2f}")
print(f"  Resulting Grade >= 3 Sensitivity : {selected['sensitivity']:.4f} ({selected['tp']}/{N_true_ge_3})")
print(f"  Total Images Escalated           : {selected['total_escalated']} ({selected['share_escalated']:.1f}% of VAL)")
print(f"  Grade <= 2 Images Escalated to URGENT: {selected['g2_escalated']} ({selected['share_g2_escalated']:.1f}% of VAL)")

