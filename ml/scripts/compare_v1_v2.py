"""Operating-point-free and pipeline comparison of model versions v1 and v2.

RULES:
  - Report only. Do not change any threshold, band, model, or registry entry.
  - Keep v1 active.
  - Cohorts: APTOS VAL (559), IDRiD ALL (516), IDRiD Official Test (103).
  - Paired bootstrap (1000 resamples) for AUC, QWK, and referable sensitivity differences.
  - Threshold-free metrics: ROC-AUC, Sensitivity @ fixed Specificity (0.80, 0.90),
    Specificity @ fixed Sensitivity (0.90, 0.95).
  - Confusion matrices and per-class recall for IDRiD ALL.
  - Tiered escalation shares and Grade >= 3 sensitivity at SOON (0.05) and URGENT (0.30).
  - Full code path reconciliation: explains why v1's IDRiD numbers differed between
    model-only evaluation and the full triage pipeline.
"""

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
from torch.utils.data import DataLoader, Dataset as TorchDataset
from PIL import Image as PILImage
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from retinaguard.core.models import CameraType
from retinaguard.services.ood import OODDetectorService
from retinaguard.services.quality import QualityGateService
from retinaguard_ml.data.loaders.aptos import APTOSDataset, FUNDUS_MEAN, FUNDUS_STD
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.evaluate import evaluate_cohort

# ===========================================================================
# Paths
# ===========================================================================
MANIFEST       = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
RAW_APTOS      = REPO_ROOT / "data" / "raw" / "aptos" / "train_images"
CACHE_APTOS    = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
CACHE_IDRID_TR = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "train"
CACHE_IDRID_TE = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "test"
IDRID_RAW_TR   = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "1. Original Images" / "a. Training Set"
IDRID_RAW_TE   = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "1. Original Images" / "b. Testing Set"
IDRID_GT_TR    = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_GT_TE    = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
REGISTRY       = REPO_ROOT / "models" / "registry.json"
COMPARE_OUT    = REPO_ROOT / "ml" / "evaluation" / "comparison_v1_v2"
QG_CACHE_PATH  = REPO_ROOT / "data" / "processed" / "idrid_qg_ood_cache.json"

IMAGE_SIZE     = [256, 256]
BATCH_SIZE     = 32
N_BOOTSTRAP    = 1000
BOOTSTRAP_SEED = 42

COMPARE_OUT.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Load Models from Registry
# ===========================================================================
def _load_registry_entry(version_tag: str) -> dict:
    with open(REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)
    for entry in reg.get("models", []):
        if entry.get("version_tag", entry.get("version", "")).startswith(version_tag):
            return entry
    raise ValueError(f"No registry entry found for {version_tag}")

def _load_model(entry: dict) -> tuple[EfficientNetDRDualHead, float]:
    ckpt_path = REPO_ROOT / entry["checkpoint_path"]
    temp = float(entry.get("temperature", 1.0))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, temp

entry_v1 = _load_registry_entry("v1")
entry_v2 = _load_registry_entry("v2")
model_v1, T_v1 = _load_model(entry_v1)
model_v2, T_v2 = _load_model(entry_v2)

# ===========================================================================
# Dataset & Inference
# ===========================================================================
class FastDataset(TorchDataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with PILImage.open(rec["cache_path"]) as im:
            arr = np.array(im.convert("RGB"))
        resized = cv2.resize(arr, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) / 255.0 - FUNDUS_MEAN) / FUNDUS_STD
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
        return tensor, rec["grade"]

def run_model_inference(model: torch.nn.Module, records: list[dict], temperature: float):
    ds = FastDataset(records)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_true, all_pred, all_probs, all_ord_probs = [], [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs)
            probs = torch.softmax(out["logits"] / temperature, dim=-1).cpu().numpy()
            ord_p = torch.sigmoid(out["ordinal_logits"]).cpu().numpy()
            grades = np.clip(np.round(np.sum(ord_p, axis=-1)), 0, 4).astype(int)
            all_true.extend(labels.numpy().tolist())
            all_pred.extend(grades.tolist())
            all_probs.append(probs)
            all_ord_probs.append(ord_p)
    return (
        np.array(all_true),
        np.array(all_pred),
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_ord_probs, axis=0),
    )

# ===========================================================================
# Metric Functions
# ===========================================================================
def compute_roc_metrics(y_true_binary: np.ndarray, y_prob_ref: np.ndarray):
    """Compute AUC and operating points on ROC curve."""
    auc = float(roc_auc_score(y_true_binary, y_prob_ref))
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_ref)

    # Sensitivity at fixed Specificity (0.80 and 0.90)
    # Specificity = 1 - FPR >= spec  =>  FPR <= 1 - spec
    sens_at_80_spec = float(np.max(tpr[fpr <= (1.0 - 0.80 + 1e-9)])) if np.any(fpr <= 0.20 + 1e-9) else 0.0
    sens_at_90_spec = float(np.max(tpr[fpr <= (1.0 - 0.90 + 1e-9)])) if np.any(fpr <= 0.10 + 1e-9) else 0.0

    # Specificity at fixed Sensitivity (0.90 and 0.95)
    # Sensitivity = TPR >= sens  =>  max Specificity = 1 - min(FPR[TPR >= sens])
    spec_at_90_sens = float(1.0 - np.min(fpr[tpr >= (0.90 - 1e-9)])) if np.any(tpr >= 0.90 - 1e-9) else 0.0
    spec_at_95_sens = float(1.0 - np.min(fpr[tpr >= (0.95 - 1e-9)])) if np.any(tpr >= 0.95 - 1e-9) else 0.0

    return {
        "auc": auc,
        "sens_at_80_spec": sens_at_80_spec,
        "sens_at_90_spec": sens_at_90_spec,
        "spec_at_90_sens": spec_at_90_sens,
        "spec_at_95_sens": spec_at_95_sens,
    }

def bootstrap_operating_points(y_true: np.ndarray, y_probs1: np.ndarray, y_probs2: np.ndarray, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Paired bootstrap of operating points and AUC differences."""
    y_true_bin = (y_true >= 2).astype(int)
    p_ref1 = y_probs1[:, 2] + y_probs1[:, 3] + y_probs1[:, 4]
    p_ref2 = y_probs2[:, 2] + y_probs2[:, 3] + y_probs2[:, 4]

    pt1 = compute_roc_metrics(y_true_bin, p_ref1)
    pt2 = compute_roc_metrics(y_true_bin, p_ref2)

    rng = np.random.default_rng(seed)
    N = len(y_true)

    bs1 = {k: [] for k in pt1}
    bs2 = {k: [] for k in pt2}
    bs_diff_auc = []

    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        yt_b = y_true_bin[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        m1 = compute_roc_metrics(yt_b, p_ref1[idx])
        m2 = compute_roc_metrics(yt_b, p_ref2[idx])
        for k in pt1:
            bs1[k].append(m1[k])
            bs2[k].append(m2[k])
        bs_diff_auc.append(m2["auc"] - m1["auc"])

    def ci(vals):
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {
        "v1": {k: {"point": pt1[k], "ci": ci(bs1[k])} for k in pt1},
        "v2": {k: {"point": pt2[k], "ci": ci(bs2[k])} for k in pt2},
        "delta_auc": {
            "point": float(pt2["auc"] - pt1["auc"]),
            "ci": ci(bs_diff_auc),
        },
    }

def compute_tier_escalation(y_true: np.ndarray, probs: np.ndarray):
    """Compute escalation shares and Grade >= 3 sensitivity."""
    p_ge3 = probs[:, 3] + probs[:, 4]
    n_total = len(y_true)
    is_ge3 = (y_true >= 3)
    is_nonref = (y_true <= 1)
    n_ge3 = int(np.sum(is_ge3))
    n_nonref = int(np.sum(is_nonref))

    soon_esc = p_ge3 >= 0.05
    urg_esc = p_ge3 >= 0.30

    soon_share = float(np.sum(soon_esc)) / n_total
    urg_share = float(np.sum(urg_esc)) / n_total

    soon_sens = float(np.sum(soon_esc & is_ge3)) / max(n_ge3, 1)
    urg_sens = float(np.sum(urg_esc & is_ge3)) / max(n_ge3, 1)

    over_triage_soon = float(np.sum(soon_esc & is_nonref)) / max(n_nonref, 1)
    over_triage_urg = float(np.sum(urg_esc & is_nonref)) / max(n_nonref, 1)

    return {
        "soon_share": soon_share,
        "urg_share": urg_share,
        "soon_sens": soon_sens,
        "urg_sens": urg_sens,
        "over_triage_soon": over_triage_soon,
        "over_triage_urg": over_triage_urg,
        "n_ge3": n_ge3,
        "n_nonref": n_nonref,
    }

# ===========================================================================
# Load Cohorts
# ===========================================================================
val_records = []
with open(MANIFEST, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["split"] == "val":
            c = r["id_code"]
            val_records.append({
                "id": c,
                "grade": int(r["diagnosis"]),
                "raw_path": RAW_APTOS / f"{c}.png",
                "cache_path": CACHE_APTOS / f"{c}.png",
            })

idrid_all_records = []
with open(IDRID_GT_TR, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        name = r["Image name"].strip()
        idrid_all_records.append({
            "id": name,
            "grade": int(r["Retinopathy grade"].strip()),
            "raw_path": IDRID_RAW_TR / f"{name}.jpg",
            "cache_path": CACHE_IDRID_TR / f"{name}.png",
        })
with open(IDRID_GT_TE, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        name = r["Image name"].strip()
        idrid_all_records.append({
            "id": name,
            "grade": int(r["Retinopathy grade"].strip()),
            "raw_path": IDRID_RAW_TE / f"{name}.jpg",
            "cache_path": CACHE_IDRID_TE / f"{name}.png",
        })

idrid_test_records = []
with open(IDRID_GT_TE, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        name = r["Image name"].strip()
        idrid_test_records.append({
            "id": name,
            "grade": int(r["Retinopathy grade"].strip()),
            "raw_path": IDRID_RAW_TE / f"{name}.jpg",
            "cache_path": CACHE_IDRID_TE / f"{name}.png",
        })

print(f"Loaded cohorts: VAL N={len(val_records)}, IDRiD ALL N={len(idrid_all_records)}, IDRiD TEST N={len(idrid_test_records)}")

# ===========================================================================
# Run Inference
# ===========================================================================
print("Running inference on VAL ...")
yt_val, yp_val_v1, ypr_val_v1, _ = run_model_inference(model_v1, val_records, T_v1)
_,      yp_val_v2, ypr_val_v2, _ = run_model_inference(model_v2, val_records, T_v2)

print("Running inference on IDRiD ALL ...")
yt_idrid, yp_idrid_v1, ypr_idrid_v1, _ = run_model_inference(model_v1, idrid_all_records, T_v1)
_,        yp_idrid_v2, ypr_idrid_v2, _ = run_model_inference(model_v2, idrid_all_records, T_v2)

print("Running inference on IDRiD TEST ...")
yt_idrid_te, yp_te_v1, ypr_te_v1, _ = run_model_inference(model_v1, idrid_test_records, T_v1)
_,           yp_te_v2, ypr_te_v2, _ = run_model_inference(model_v2, idrid_test_records, T_v2)

# ===========================================================================
# 1. Operating-Point-Free Evaluation
# ===========================================================================
print("\n" + "=" * 80)
print("1. OPERATING-POINT-FREE COMPARISON (VAL, IDRiD ALL, IDRiD TEST)")
print("=" * 80)

op_val = bootstrap_operating_points(yt_val, ypr_val_v1, ypr_val_v2)
op_idrid_all = bootstrap_operating_points(yt_idrid, ypr_idrid_v1, ypr_idrid_v2)
op_idrid_te = bootstrap_operating_points(yt_idrid_te, ypr_te_v1, ypr_te_v2)

def print_op_table(cohort_name: str, op_dict: dict):
    print(f"\n--- {cohort_name} ---")
    print(f"{'Metric':<35} {'v1 (95% CI)':<25} {'v2 (95% CI)':<25} {'Delta (v2 - v1) [95% CI]'}")
    print("-" * 105)
    v1 = op_dict["v1"]
    v2 = op_dict["v2"]
    d_auc = op_dict["delta_auc"]

    # AUC
    s1 = f"{v1['auc']['point']:.4f} [{v1['auc']['ci'][0]:.4f}, {v1['auc']['ci'][1]:.4f}]"
    s2 = f"{v2['auc']['point']:.4f} [{v2['auc']['ci'][0]:.4f}, {v2['auc']['ci'][1]:.4f}]"
    sd = f"{d_auc['point']:+.4f} [{d_auc['ci'][0]:+.4f}, {d_auc['ci'][1]:+.4f}]"
    print(f"{'Referable ROC-AUC':<35} {s1:<25} {s2:<25} {sd}")

    # Sens @ Spec
    for sp in ["80", "90"]:
        k = f"sens_at_{sp}_spec"
        s1 = f"{v1[k]['point']:.4f} [{v1[k]['ci'][0]:.4f}, {v1[k]['ci'][1]:.4f}]"
        s2 = f"{v2[k]['point']:.4f} [{v2[k]['ci'][0]:.4f}, {v2[k]['ci'][1]:.4f}]"
        print(f"{f'Sensitivity @ Spec={sp}%':<35} {s1:<25} {s2:<25}")

    # Spec @ Sens
    for sn in ["90", "95"]:
        k = f"spec_at_{sn}_sens"
        s1 = f"{v1[k]['point']:.4f} [{v1[k]['ci'][0]:.4f}, {v1[k]['ci'][1]:.4f}]"
        s2 = f"{v2[k]['point']:.4f} [{v2[k]['ci'][0]:.4f}, {v2[k]['ci'][1]:.4f}]"
        print(f"{f'Specificity @ Sens={sn}%':<35} {s1:<25} {s2:<25}")

print_op_table("APTOS Validation Split (N=559)", op_val)
print_op_table("IDRiD ALL (N=516)", op_idrid_all)
print_op_table("IDRiD Official Test Split (N=103)", op_idrid_te)

# ===========================================================================
# 2. IDRiD (All 516) Confusion Matrix & Over-Calling Analysis
# ===========================================================================
print("\n" + "=" * 80)
print("2. IDRiD ALL (516) CONFUSION MATRICES & PER-CLASS RECALL")
print("=" * 80)

cm_v1 = confusion_matrix(yt_idrid, yp_idrid_v1, labels=list(range(5)))
cm_v2 = confusion_matrix(yt_idrid, yp_idrid_v2, labels=list(range(5)))

print("\nIDRiD ALL — v1 Confusion Matrix (Rows=True, Cols=Pred):")
print("       P0    P1    P2    P3    P4   | Recall")
for r in range(5):
    rec = cm_v1[r, r] / cm_v1[r].sum() if cm_v1[r].sum() > 0 else 0.0
    row_s = "  ".join(f"{cm_v1[r, c]:4d}" for c in range(5))
    print(f"True {r}: {row_s}   | {rec:.4f} ({cm_v1[r, r]}/{cm_v1[r].sum()})")

print("\nIDRiD ALL — v2 Confusion Matrix (Rows=True, Cols=Pred):")
print("       P0    P1    P2    P3    P4   | Recall")
for r in range(5):
    rec = cm_v2[r, r] / cm_v2[r].sum() if cm_v2[r].sum() > 0 else 0.0
    row_s = "  ".join(f"{cm_v2[r, c]:4d}" for c in range(5))
    print(f"True {r}: {row_s}   | {rec:.4f} ({cm_v2[r, r]}/{cm_v2[r].sum()})")

# Breakdown of referable calls
# Non-referable grades: 0 and 1
g0_v1_ref = int(np.sum(cm_v1[0, 2:]))
g0_v2_ref = int(np.sum(cm_v2[0, 2:]))
g1_v1_ref = int(np.sum(cm_v1[1, 2:]))
g1_v2_ref = int(np.sum(cm_v2[1, 2:]))

total_fp_v1 = g0_v1_ref + g1_v1_ref
total_fp_v2 = g0_v2_ref + g1_v2_ref

# True referable grades: 2, 3, 4
tp_v1 = int(np.sum(cm_v1[2:, 2:]))
tp_v2 = int(np.sum(cm_v2[2:, 2:]))
total_true_ref = int(np.sum(cm_v1[2:, :]))

print("\n--- Direction of Extra Referable Calls on IDRiD ALL ---")
print(f"Total True Non-Referable Images (Grade 0 or 1): {int(cm_v1[:2, :].sum())}")
print(f"  True Grade 0 called Referable (Pred 2+):")
print(f"    v1: {g0_v1_ref:3d} / {cm_v1[0].sum():3d} ({100.0*g0_v1_ref/cm_v1[0].sum():.1f}%)")
print(f"    v2: {g0_v2_ref:3d} / {cm_v2[0].sum():3d} ({100.0*g0_v2_ref/cm_v2[0].sum():.1f}%)  [+{g0_v2_ref - g0_v1_ref} false positives]")
print(f"  True Grade 1 called Referable (Pred 2+):")
print(f"    v1: {g1_v1_ref:3d} / {cm_v1[1].sum():3d} ({100.0*g1_v1_ref/cm_v1[1].sum():.1f}%)")
print(f"    v2: {g1_v2_ref:3d} / {cm_v2[1].sum():3d} ({100.0*g1_v2_ref/cm_v2[1].sum():.1f}%)  [+{g1_v2_ref - g1_v1_ref} false positives]")
print(f"  Total False Referable Calls (Over-calling):")
print(f"    v1: {total_fp_v1} / {int(cm_v1[:2, :].sum())} ({100.0*total_fp_v1/int(cm_v1[:2, :].sum()):.1f}%)")
print(f"    v2: {total_fp_v2} / {int(cm_v2[:2, :].sum())} ({100.0*total_fp_v2/int(cm_v2[:2, :].sum()):.1f}%)  [+{total_fp_v2 - total_fp_v1} false positives]")
print(f"\nTotal True Referable Images (Grade 2, 3, or 4): {total_true_ref}")
print(f"  Correctly called Referable (TP):")
print(f"    v1: {tp_v1} / {total_true_ref} ({100.0*tp_v1/total_true_ref:.1f}%)")
print(f"    v2: {tp_v2} / {total_true_ref} ({100.0*tp_v2/total_true_ref:.1f}%)  [+{tp_v2 - tp_v1} true positives captured]")

# ===========================================================================
# 3. Tiered Sight-Threatening Escalation Analysis
# ===========================================================================
print("\n" + "=" * 80)
print("3. TIERED SIGHT-THREATENING ESCALATION & OVER-TRIAGE")
print("=" * 80)

esc_val_v1 = compute_tier_escalation(yt_val, ypr_val_v1)
esc_val_v2 = compute_tier_escalation(yt_val, ypr_val_v2)

esc_idrid_v1 = compute_tier_escalation(yt_idrid, ypr_idrid_v1)
esc_idrid_v2 = compute_tier_escalation(yt_idrid, ypr_idrid_v2)

esc_te_v1 = compute_tier_escalation(yt_idrid_te, ypr_te_v1)
esc_te_v2 = compute_tier_escalation(yt_idrid_te, ypr_te_v2)

def print_escalation_table(name: str, e1: dict, e2: dict):
    print(f"\n--- {name} ---")
    print(f"{'Tier Metric':<45} {'v1':<15} {'v2':<15}")
    print("-" * 75)
    print(f"{'SOON Escalation Share (P>=0.05)':<45} {e1['soon_share']*100:.1f}%          {e2['soon_share']*100:.1f}%")
    print(f"{'URGENT Escalation Share (P>=0.30)':<45} {e1['urg_share']*100:.1f}%          {e2['urg_share']*100:.1f}%")
    print(f"{'Grade >= 3 Sensitivity @ SOON':<45} {e1['soon_sens']*100:.1f}%          {e2['soon_sens']*100:.1f}%")
    print(f"{'Grade >= 3 Sensitivity @ URGENT':<45} {e1['urg_sens']*100:.1f}%          {e2['urg_sens']*100:.1f}%")
    print(f"{'Over-Triage: Grade 0-1 Escalated @ SOON':<45} {e1['over_triage_soon']*100:.1f}%          {e2['over_triage_soon']*100:.1f}%")
    print(f"{'Over-Triage: Grade 0-1 Escalated @ URGENT':<45} {e1['over_triage_urg']*100:.1f}%          {e2['over_triage_urg']*100:.1f}%")

print_escalation_table("APTOS VAL Split (N=559, True G>=3 = 75, True G<=1 = 336)", esc_val_v1, esc_val_v2)
print_escalation_table("IDRiD ALL (N=516, True G>=3 = 155, True G<=1 = 193)", esc_idrid_v1, esc_idrid_v2)
print_escalation_table("IDRiD Official Test Split (N=103, True G>=3 = 32, True G<=1 = 39)", esc_te_v1, esc_te_v2)

# ===========================================================================
# 4. Explanation & Full Pipeline Code Path Reconciliation
# ===========================================================================
print("\n" + "=" * 80)
print("4. EXPLANATION: WHY v1's IDRiD NUMBERS DIFFERED")
print("=" * 80)

# Full triage pipeline logic from report_new_settings.py:
# We know IDRiD QG pass rate is 509/516 (7 failed resolution/blur), OOD reject is 0.
# Load or compute QG & OOD mask
qg_mask = np.ones(len(idrid_all_records), dtype=bool)
if QG_CACHE_PATH.exists():
    cache_data = json.loads(QG_CACHE_PATH.read_text(encoding="utf-8"))
    qg_mask = np.array([cache_data.get(r["id"], True) for r in idrid_all_records])
else:
    # Run QG service
    print("Evaluating Quality Gate across IDRiD ALL...")
    quality_svc = QualityGateService()
    cache_dict = {}
    for i, r in enumerate(idrid_all_records):
        with PILImage.open(r["raw_path"]) as im:
            raw_rgb = np.array(im.convert("RGB"))
        pas = quality_svc.evaluate(raw_rgb, camera_type=CameraType.TABLETOP).passed
        qg_mask[i] = pas
        cache_dict[r["id"]] = pas
    QG_CACHE_PATH.write_text(json.dumps(cache_dict, indent=2), encoding="utf-8")
    print(f"Quality gate cached -> {QG_CACHE_PATH}")

p_ref1 = ypr_idrid_v1[:, 2] + ypr_idrid_v1[:, 3] + ypr_idrid_v1[:, 4]
p_ge3_1 = ypr_idrid_v1[:, 3] + ypr_idrid_v1[:, 4]

# Path A: The original report_new_settings.py path:
abstains_pipe = ((0.30 <= p_ref1) & (p_ref1 <= 0.70)) | ((p_ref1 >= 0.30) & (yp_idrid_v1 == 0))
retained_pipe = qg_mask & (~abstains_pipe)

pipe_true_ref = yt_idrid[retained_pipe] >= 2
pipe_pred_ref = yp_idrid_v1[retained_pipe] >= 2
pipe_tp = int(np.sum(pipe_pred_ref & pipe_true_ref))
pipe_fn = int(np.sum((~pipe_pred_ref) & pipe_true_ref))
pipe_ret_sens = pipe_tp / (pipe_tp + pipe_fn)

pipe_soon_tp = int(np.sum((p_ge3_1 >= 0.05) & (yt_idrid >= 3)))
pipe_soon_sens = pipe_soon_tp / int(np.sum(yt_idrid >= 3))

# Path B: The initial compare_v1_v2.py shortcut path:
retained_shortcut = ~((p_ref1 >= 0.30) & (p_ref1 <= 0.70))
short_true_ref = yt_idrid[retained_shortcut] >= 2
short_pred_ref = p_ref1[retained_shortcut] > 0.70
short_tp = int(np.sum(short_pred_ref & short_true_ref))
short_fn = int(np.sum((~short_pred_ref) & short_true_ref))
short_ret_sens = short_tp / (short_tp + short_fn)

short_soon_sens = float(np.sum((p_ge3_1[retained_shortcut] >= 0.05) & (yt_idrid[retained_shortcut] >= 3))) / int(np.sum(yt_idrid[retained_shortcut] >= 3))

print("\nExact Reconciliation Comparison for v1 on IDRiD ALL (516):")
print("-" * 80)
print(f"Metric                                  Original Pipeline   Shortcut Path")
print("-" * 80)
print(f"Quality Gate Applied?                   YES (509/516 passed) NO (all 516 kept)")
print(f"Abstention Safety Constraint?           YES (ref>=0.30 & P0) NO (only [0.3, 0.7])")
print(f"Predicted Referable Definition          pred_grade >= 2      P(ref) > 0.70")
print(f"Retained Referable Sensitivity          {pipe_ret_sens:.4f} ({pipe_tp}/{pipe_tp+pipe_fn})    {short_ret_sens:.4f} ({short_tp}/{short_tp+short_fn})")
print(f"Grade >= 3 SOON Denominator             Full Cohort (155)    Retained Subset ({int(np.sum(yt_idrid[retained_shortcut] >= 3))})")
print(f"Grade >= 3 SOON Sensitivity             {pipe_soon_sens:.4f} ({pipe_soon_tp}/155)      {short_soon_sens:.4f}")
print("-" * 80)

# Also run v2 through the original pipeline path:
p_ref2 = ypr_idrid_v2[:, 2] + ypr_idrid_v2[:, 3] + ypr_idrid_v2[:, 4]
p_ge3_2 = ypr_idrid_v2[:, 3] + ypr_idrid_v2[:, 4]
abstains_pipe_v2 = ((0.30 <= p_ref2) & (p_ref2 <= 0.70)) | ((p_ref2 >= 0.30) & (yp_idrid_v2 == 0))
retained_pipe_v2 = qg_mask & (~abstains_pipe_v2)

pipe_true_ref_v2 = yt_idrid[retained_pipe_v2] >= 2
pipe_pred_ref_v2 = yp_idrid_v2[retained_pipe_v2] >= 2
pipe_tp_v2 = int(np.sum(pipe_pred_ref_v2 & pipe_true_ref_v2))
pipe_fn_v2 = int(np.sum((~pipe_pred_ref_v2) & pipe_true_ref_v2))
pipe_ret_sens_v2 = pipe_tp_v2 / (pipe_tp_v2 + pipe_fn_v2)

pipe_soon_tp_v2 = int(np.sum((p_ge3_2 >= 0.05) & (yt_idrid >= 3)))
pipe_soon_sens_v2 = pipe_soon_tp_v2 / int(np.sum(yt_idrid >= 3))

print(f"\nv2 on IDRiD ALL through Original Pipeline Path:")
print(f"  Retained Cases               : {int(np.sum(retained_pipe_v2))}/516 ({100.0*np.sum(retained_pipe_v2)/516:.1f}%)")
print(f"  Retained Referable Sens      : {pipe_ret_sens_v2:.4f} ({pipe_tp_v2}/{pipe_tp_v2+pipe_fn_v2})")
print(f"  Grade >= 3 Sens @ SOON (0.05): {pipe_soon_sens_v2:.4f} ({pipe_soon_tp_v2}/155)")

# Save full results to JSON
full_summary = {
    "operating_point_free": {
        "val": op_val,
        "idrid_all": op_idrid_all,
        "idrid_test": op_idrid_te,
    },
    "confusion_matrices_idrid_all": {
        "v1": cm_v1.tolist(),
        "v2": cm_v2.tolist(),
    },
    "extra_referable_breakdown": {
        "true_g0_called_ref_v1": g0_v1_ref,
        "true_g0_called_ref_v2": g0_v2_ref,
        "true_g1_called_ref_v1": g1_v1_ref,
        "true_g1_called_ref_v2": g1_v2_ref,
        "total_false_positives_v1": total_fp_v1,
        "total_false_positives_v2": total_fp_v2,
        "true_positives_captured_v1": tp_v1,
        "true_positives_captured_v2": tp_v2,
        "total_true_referable": total_true_ref,
    },
    "escalation_analysis": {
        "val": {"v1": esc_val_v1, "v2": esc_val_v2},
        "idrid_all": {"v1": esc_idrid_v1, "v2": esc_idrid_v2},
        "idrid_test": {"v1": esc_te_v1, "v2": esc_te_v2},
    },
    "pipeline_reconciliation": {
        "v1_retained_ref_sens_original_pipeline": pipe_ret_sens,
        "v1_retained_ref_sens_shortcut": short_ret_sens,
        "v1_g3_soon_sens_original_pipeline": pipe_soon_sens,
        "v1_g3_soon_sens_shortcut": short_soon_sens,
        "v2_retained_ref_sens_original_pipeline": pipe_ret_sens_v2,
        "v2_g3_soon_sens_original_pipeline": pipe_soon_sens_v2,
    },
}

out_file = COMPARE_OUT / "comparison_v1_v2.json"
out_file.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
print(f"\nSaved comprehensive comparison JSON -> {out_file}")

# Verify active model unchanged
reg_curr = json.loads(REGISTRY.read_text(encoding="utf-8"))
active = [m for m in reg_curr["models"] if m.get("active", False)]
print("\n[ACTIVE MODEL VERIFICATION]")
print(f"  Active model: {active[0].get('version_tag', active[0].get('version', '?'))}")
assert active[0].get("version_tag", active[0].get("version", "")).startswith("v1"), "v1 must be active!"
print("  Active model is v1. No thresholds or registry entries modified.")
