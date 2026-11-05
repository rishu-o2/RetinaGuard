"""Reporting script for new settings on VAL, APTOS Test, and IDRiD cohorts.

REPORTING ONLY - No thresholds or hyperparameters are tuned using test or IDRiD.
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
from retinaguard_ml.models import EfficientNetDRDualHead

MANIFEST = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
RAW_APTOS = REPO_ROOT / "data" / "raw" / "aptos" / "train_images"
CACHE_APTOS = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
CACHE_IDRID_TR = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "train"
CACHE_IDRID_TE = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "test"
IDRID_GT_TR = (
    REPO_ROOT
    / "data"
    / "raw"
    / "idrid_download"
    / "B. Disease Grading"
    / "2. Groundtruths"
    / "a. IDRiD_Disease Grading_Training Labels.csv"
)
IDRID_GT_TE = (
    REPO_ROOT
    / "data"
    / "raw"
    / "idrid_download"
    / "B. Disease Grading"
    / "2. Groundtruths"
    / "b. IDRiD_Disease Grading_Testing Labels.csv"
)
IDRID_RAW_TR = (
    REPO_ROOT
    / "data"
    / "raw"
    / "idrid_download"
    / "B. Disease Grading"
    / "1. Original Images"
    / "a. Training Set"
)
IDRID_RAW_TE = (
    REPO_ROOT
    / "data"
    / "raw"
    / "idrid_download"
    / "B. Disease Grading"
    / "1. Original Images"
    / "b. Testing Set"
)
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

quality_svc = QualityGateService()
ood_svc = OODDetectorService()

# Band thresholds chosen on VAL
BAND_LOW = 0.30
BAND_HIGH = 0.70
SOON_THRESH = 0.05
URGENT_THRESH = 0.30


def evaluate_cohort(records, cohort_name: str):
    print(f"\n=======================================================")
    print(f"Evaluating: {cohort_name} (N={len(records)})")
    print(f"=======================================================")

    n_total = len(records)
    n_qg_pass = 0
    n_ood_reject = 0
    n_abstain = 0

    true_grades = []
    pred_grades = []
    p_refs = []
    p_ge_3s = []
    qg_passes = []
    ood_rejects = []
    abstains = []

    with torch.no_grad():
        for r in records:
            raw_p = r["raw_path"]
            cache_p = r["cache_path"]
            true_g = int(r["grade"])
            true_grades.append(true_g)

            with Image.open(raw_p) as im:
                raw_rgb = np.array(im.convert("RGB"))

            # Quality Gate
            qg_res = quality_svc.evaluate(raw_rgb, camera_type=CameraType.TABLETOP)
            qg_pass = qg_res.passed
            qg_passes.append(qg_pass)
            if qg_pass:
                n_qg_pass += 1

            # OOD
            ood_res = ood_svc.evaluate(raw_rgb)
            ood_reject = not ood_res.is_in_distribution
            ood_rejects.append(ood_reject)
            if ood_reject:
                n_ood_reject += 1

            # Model inference
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

            ord_probs = torch.sigmoid(out["ordinal_logits"]).squeeze(0).numpy()
            pred_g = int(np.clip(np.round(np.sum(ord_probs)), 0, 4))
            pred_grades.append(pred_g)

            # Probabilities
            p_ref = float(probs[2] + probs[3] + probs[4])
            p_ge_3 = float(probs[3] + probs[4])
            p_refs.append(p_ref)
            p_ge_3s.append(p_ge_3)

            # Referable band abstention
            # UNCERTAIN if in [0.30, 0.70] OR (p_ref >= 0.30 and pred_g == 0)
            is_abstain = (BAND_LOW <= p_ref <= BAND_HIGH) or (p_ref >= BAND_LOW and pred_g == 0)
            abstains.append(is_abstain)
            if is_abstain:
                n_abstain += 1

    true_grades = np.array(true_grades)
    pred_grades = np.array(pred_grades)
    p_refs = np.array(p_refs)
    p_ge_3s = np.array(p_ge_3s)
    qg_passes = np.array(qg_passes)
    ood_rejects = np.array(ood_rejects)
    abstains = np.array(abstains)

    # Retained mask: passed QG AND passed OOD AND not abstained
    retained_mask = qg_passes & (~ood_rejects) & (~abstains)
    n_retained = int(np.sum(retained_mask))

    # Overall Referable Ground Truth
    is_true_ref = (true_grades >= 2)
    n_true_ref = int(np.sum(is_true_ref))
    n_true_non_ref = int(np.sum(~is_true_ref))

    # Retained metrics
    ret_true = true_grades[retained_mask]
    ret_pred = pred_grades[retained_mask]
    ret_true_ref = (ret_true >= 2)
    ret_pred_ref = (ret_pred >= 2)

    ret_tp = int(np.sum(ret_pred_ref & ret_true_ref))
    ret_fn = int(np.sum((~ret_pred_ref) & ret_true_ref))
    ret_tn = int(np.sum((~ret_pred_ref) & (~ret_true_ref)))
    ret_fp = int(np.sum(ret_pred_ref & (~ret_true_ref)))

    ret_sens = ret_tp / (ret_tp + ret_fn) if (ret_tp + ret_fn) > 0 else 0.0
    ret_spec = ret_tn / (ret_tn + ret_fp) if (ret_tn + ret_fp) > 0 else 0.0

    # Grade >= 3 Sensitivity
    is_true_ge_3 = (true_grades >= 3)
    n_true_ge_3 = int(np.sum(is_true_ge_3))
    # Sensitivity at SOON tier (P(>=3) >= 0.05)
    g3_soon_tp = int(np.sum((p_ge_3s >= SOON_THRESH) & is_true_ge_3))
    g3_soon_sens = g3_soon_tp / n_true_ge_3 if n_true_ge_3 > 0 else 0.0
    # Sensitivity at URGENT tier (P(>=3) >= 0.30)
    g3_urg_tp = int(np.sum((p_ge_3s >= URGENT_THRESH) & is_true_ge_3))
    g3_urg_sens = g3_urg_tp / n_true_ge_3 if n_true_ge_3 > 0 else 0.0

    # On retained cohort
    ret_true_ge_3 = (ret_true >= 3)
    n_ret_true_ge_3 = int(np.sum(ret_true_ge_3))
    ret_g3_soon_tp = int(np.sum((p_ge_3s[retained_mask] >= SOON_THRESH) & ret_true_ge_3))
    ret_g3_soon_sens = ret_g3_soon_tp / n_ret_true_ge_3 if n_ret_true_ge_3 > 0 else 0.0

    print(f"Results for {cohort_name}:")
    print(f"  Total Images                 : {n_total}")
    print(f"  Quality Gate Pass Rate       : {n_qg_pass}/{n_total} = {100.0*n_qg_pass/n_total:.2f}%")
    print(f"  OOD Rejection Rate           : {n_ood_reject}/{n_total} = {100.0*n_ood_reject/n_total:.2f}%")
    print(f"  Abstention (UNCERTAIN) Rate  : {n_abstain}/{n_total} = {100.0*n_abstain/n_total:.2f}%")
    print(f"  Retained (Automated) Cohort  : {n_retained}/{n_total} = {100.0*n_retained/n_total:.2f}%")
    print(f"  Retained Referable Sens      : {ret_sens:.4f} (TP={ret_tp}, FN={ret_fn})")
    print(f"  Retained Referable Spec      : {ret_spec:.4f} (TN={ret_tn}, FP={ret_fp})")
    print(f"  Grade >= 3 Sensitivity (SOON >= 0.05)   : {g3_soon_sens:.4f} ({g3_soon_tp}/{n_true_ge_3})")
    print(f"  Grade >= 3 Sensitivity (URGENT >= 0.30) : {g3_urg_sens:.4f} ({g3_urg_tp}/{n_true_ge_3})")
    if n_ret_true_ge_3 > 0:
        print(f"  Retained Grade >= 3 Sens (SOON >= 0.05) : {ret_g3_soon_sens:.4f} ({ret_g3_soon_tp}/{n_ret_true_ge_3})")

    return {
        "cohort": cohort_name,
        "n_total": n_total,
        "qg_pass_rate": 100.0 * n_qg_pass / n_total,
        "qg_pass_count": n_qg_pass,
        "ood_reject_rate": 100.0 * n_ood_reject / n_total,
        "ood_reject_count": n_ood_reject,
        "abstain_rate": 100.0 * n_abstain / n_total,
        "abstain_count": n_abstain,
        "n_retained": n_retained,
        "retention_rate": 100.0 * n_retained / n_total,
        "retained_referable_sens": ret_sens,
        "retained_referable_spec": ret_spec,
        "grade_ge_3_soon_sens": g3_soon_sens,
        "grade_ge_3_urgent_sens": g3_urg_sens,
        "retained_grade_ge_3_soon_sens": ret_g3_soon_sens,
    }


def main():
    # 1. Build APTOS VAL records
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

    # 2. Build APTOS Test records
    test_records = []
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                c = r["id_code"]
                test_records.append({
                    "id": c,
                    "grade": int(r["diagnosis"]),
                    "raw_path": RAW_APTOS / f"{c}.png",
                    "cache_path": CACHE_APTOS / f"{c}.png",
                })

    # 3. Build IDRiD All records
    idrid_all_records = []
    with open(IDRID_GT_TR, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r["Image name"].strip()
            grade = int(r["Retinopathy grade"].strip())
            idrid_all_records.append({
                "id": name,
                "grade": grade,
                "raw_path": IDRID_RAW_TR / f"{name}.jpg",
                "cache_path": CACHE_IDRID_TR / f"{name}.png",
            })
    with open(IDRID_GT_TE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r["Image name"].strip()
            grade = int(r["Retinopathy grade"].strip())
            idrid_all_records.append({
                "id": name,
                "grade": grade,
                "raw_path": IDRID_RAW_TE / f"{name}.jpg",
                "cache_path": CACHE_IDRID_TE / f"{name}.png",
            })

    # 4. Build IDRiD Official Test records
    idrid_test_records = []
    with open(IDRID_GT_TE, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r["Image name"].strip()
            grade = int(r["Retinopathy grade"].strip())
            idrid_test_records.append({
                "id": name,
                "grade": grade,
                "raw_path": IDRID_RAW_TE / f"{name}.jpg",
                "cache_path": CACHE_IDRID_TE / f"{name}.png",
            })

    results = []
    results.append(evaluate_cohort(val_records, "APTOS Validation Split"))
    results.append(evaluate_cohort(test_records, "APTOS Held-out Test Split"))
    results.append(evaluate_cohort(idrid_all_records, "IDRiD All (516 images)"))
    results.append(evaluate_cohort(idrid_test_records, "IDRiD Official Test Split (103 images)"))

    out_json = REPO_ROOT / "ml" / "evaluation" / "new_settings_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved new settings report -> {out_json}")


if __name__ == "__main__":
    main()

