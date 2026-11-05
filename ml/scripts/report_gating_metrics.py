"""Reporting script for Step 3: Gating and Abstention Metrics across splits.

Computes:
- OOD rejection rate
- Quality gate pass rate (new fixed 512px blur check at 12.21 + FOV + resolution)
- Abstention (UNCERTAIN) rate (calibrated confidence < 0.5024)
- Retained images referable sensitivity (true grade >= 2 sensitivity among retained)
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
from PIL import Image

from retinaguard.core.models import CameraType
from retinaguard.services.ood import OODDetectorService
from retinaguard.services.quality import QualityGateService
from retinaguard_ml.data.loaders.aptos import FUNDUS_MEAN, FUNDUS_STD
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead

MANIFEST = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
RAW_APTOS = REPO_ROOT / "data" / "raw" / "aptos" / "train_images"
CACHE_APTOS = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
CACHE_IDRID_TR = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "train"
CACHE_IDRID_TE = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "test"
IDRID_GT_TR = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_GT_TE = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
IDRID_RAW_TR = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "1. Original Images" / "a. Training Set"
IDRID_RAW_TE = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "1. Original Images" / "b. Testing Set"
CHECKPOINT = REPO_ROOT / "ml" / "runs" / "baseline_cpu_b0_256" / "best.pt"
REGISTRY = REPO_ROOT / "models" / "registry.json"

# Load registry metadata
with open(REGISTRY, encoding="utf-8") as f:
    reg = json.load(f)
entry = reg["models"][0]
T_star = float(entry.get("temperature", 1.2962))

CONF_THRESHOLD = 0.5024  # High confidence threshold from configs/abstention.yaml

# Load model
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Services
quality_svc = QualityGateService()
ood_svc = OODDetectorService()

def run_cohort_evaluation(records, cohort_name: str, is_idrid: bool = False):
    print(f"\nEvaluating cohort: {cohort_name} (N={len(records)}) ...")
    n_total = len(records)
    n_qg_pass = 0
    n_ood_reject = 0
    n_abstain = 0

    retained_true = []
    retained_pred = []

    with torch.no_grad():
        for r in records:
            # 1. Load image
            raw_p = r["raw_path"]
            cache_p = r["cache_path"]
            true_grade = r["grade"]

            with Image.open(raw_p) as im:
                raw_rgb = np.array(im.convert("RGB"))

            # Quality gate evaluation
            qg_res = quality_svc.evaluate(raw_rgb, camera_type=CameraType.TABLETOP)
            qg_passed = qg_res.passed
            if qg_passed:
                n_qg_pass += 1

            # OOD evaluation (optical spectrum filter on raw/cache fundus)
            ood_res = ood_svc.evaluate(raw_rgb)
            ood_rejected = not ood_res.is_in_distribution
            if ood_rejected:
                n_ood_reject += 1

            # Inference for calibrated probabilities & grade
            with Image.open(cache_p) as im:
                cached_rgb = np.array(im.convert("RGB"))
            resized = cv2.resize(cached_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            norm = (resized.astype(np.float32) / 255.0 - FUNDUS_MEAN) / FUNDUS_STD
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()

            out = model(tensor)
            logits = out["logits"].squeeze(0).numpy()
            scaled = logits / T_star
            exp_l = np.exp(scaled - np.max(scaled))
            probs = exp_l / np.sum(exp_l)
            confidence = float(np.max(probs))

            # Ordinal grade prediction
            ord_probs = torch.sigmoid(out["ordinal_logits"]).squeeze(0).numpy()
            pred_grade = int(np.clip(np.round(np.sum(ord_probs)), 0, 4))

            # Abstention check
            is_abstain = (confidence < CONF_THRESHOLD)
            if is_abstain:
                n_abstain += 1

            # Retention condition: Passed QG AND Passed OOD AND Not abstained
            is_retained = qg_passed and (not ood_rejected) and (not is_abstain)
            if is_retained:
                retained_true.append(true_grade)
                retained_pred.append(pred_grade)

    qg_pass_rate = 100.0 * n_qg_pass / n_total
    ood_reject_rate = 100.0 * n_ood_reject / n_total
    abstain_rate = 100.0 * n_abstain / n_total
    retention_rate = 100.0 * len(retained_true) / n_total

    # Referable sensitivity among retained
    if len(retained_true) > 0:
        ret_yt = np.array(retained_true)
        ret_yp = np.array(retained_pred)
        true_ref = (ret_yt >= 2).astype(int)
        pred_ref = (ret_yp >= 2).astype(int)
        tp = int(np.sum((true_ref == 1) & (pred_ref == 1)))
        fn = int(np.sum((true_ref == 1) & (pred_ref == 0)))
        ref_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ref_spec = int(np.sum((true_ref == 0) & (pred_ref == 0))) / int(np.sum(true_ref == 0)) if np.sum(true_ref == 0) > 0 else 0.0
    else:
        ref_sens = 0.0
        ref_spec = 0.0

    print(f"Results for {cohort_name}:")
    print(f"  Total Images                 : {n_total}")
    print(f"  Quality Gate Pass Rate       : {n_qg_pass}/{n_total} = {qg_pass_rate:.1f}%")
    print(f"  OOD Rejection Rate           : {n_ood_reject}/{n_total} = {ood_reject_rate:.1f}%")
    print(f"  Abstention (UNCERTAIN) Rate  : {n_abstain}/{n_total} = {abstain_rate:.1f}%")
    print(f"  Retained (Screened) Cohort   : {len(retained_true)}/{n_total} = {retention_rate:.1f}%")
    print(f"  Retained Referable Sens      : {ref_sens:.4f} (TP={tp if len(retained_true)>0 else 0}, FN={fn if len(retained_true)>0 else 0})")
    print(f"  Retained Referable Spec      : {ref_spec:.4f}")

    return {
        "cohort": cohort_name,
        "n_total": n_total,
        "qg_pass_rate": qg_pass_rate,
        "ood_reject_rate": ood_reject_rate,
        "abstain_rate": abstain_rate,
        "n_retained": len(retained_true),
        "retention_rate": retention_rate,
        "retained_referable_sens": ref_sens,
        "retained_referable_spec": ref_spec,
    }

# 1. VAL Cohort
val_recs = []
with open(MANIFEST, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == "val":
            code = row["id_code"]
            raw_p = RAW_APTOS / f"{code}.png"
            if not raw_p.exists(): raw_p = RAW_APTOS / f"{code}.jpg"
            cache_p = CACHE_APTOS / f"{code}.png"
            if raw_p.exists() and cache_p.exists():
                val_recs.append({"id": code, "grade": int(row["diagnosis"]),
                                 "raw_path": raw_p, "cache_path": cache_p})

# 2. APTOS TEST Cohort
test_recs = []
with open(MANIFEST, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == "test":
            code = row["id_code"]
            raw_p = RAW_APTOS / f"{code}.png"
            if not raw_p.exists(): raw_p = RAW_APTOS / f"{code}.jpg"
            cache_p = CACHE_APTOS / f"{code}.png"
            if raw_p.exists() and cache_p.exists():
                test_recs.append({"id": code, "grade": int(row["diagnosis"]),
                                  "raw_path": raw_p, "cache_path": cache_p})

# 3. IDRiD Cohorts
idrid_tr_recs = []
with open(IDRID_GT_TR, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        img_name = (row.get("Image name") or row.get("image_id") or "").strip()
        grade = int((row.get("Retinopathy grade") or "0").strip())
        raw_p = IDRID_RAW_TR / f"{img_name}.jpg"
        cache_p = CACHE_IDRID_TR / f"{img_name}.png"
        if raw_p.exists() and cache_p.exists():
            idrid_tr_recs.append({"id": img_name, "grade": grade, "raw_path": raw_p, "cache_path": cache_p})

idrid_te_recs = []
with open(IDRID_GT_TE, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        img_name = (row.get("Image name") or row.get("image_id") or "").strip()
        grade = int((row.get("Retinopathy grade") or "0").strip())
        raw_p = IDRID_RAW_TE / f"{img_name}.jpg"
        cache_p = CACHE_IDRID_TE / f"{img_name}.png"
        if raw_p.exists() and cache_p.exists():
            idrid_te_recs.append({"id": img_name, "grade": grade, "raw_path": raw_p, "cache_path": cache_p})

idrid_all_recs = idrid_tr_recs + idrid_te_recs

all_summaries = []
all_summaries.append(run_cohort_evaluation(val_recs, "APTOS Validation Split"))
all_summaries.append(run_cohort_evaluation(test_recs, "APTOS Test Split"))
all_summaries.append(run_cohort_evaluation(idrid_all_recs, "IDRiD All (516 images)"))
all_summaries.append(run_cohort_evaluation(idrid_te_recs, "IDRiD Official Test Split (103 images)"))

# Save JSON summary
summary_p = REPO_ROOT / "ml" / "evaluation" / "gating_and_abstention_report.json"
with open(summary_p, "w", encoding="utf-8") as f:
    json.dump(all_summaries, f, indent=2)
print(f"\nSaved gating and abstention report -> {summary_p}")

