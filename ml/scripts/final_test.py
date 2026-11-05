
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from PIL import Image as PILImage

from retinaguard_ml.data.loaders.aptos import APTOSDataset, FUNDUS_MEAN, FUNDUS_STD
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.evaluate import evaluate_cohort

MANIFEST        = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
CACHE_APTOS     = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
CACHE_IDRID_TR  = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "train"
CACHE_IDRID_TE  = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "test"
CHECKPOINT      = REPO_ROOT / "ml" / "runs" / "baseline_cpu_b0_256" / "best.pt"
REGISTRY        = REPO_ROOT / "models" / "registry.json"
OUT_DIR         = REPO_ROOT / "ml" / "evaluation" / "test_report"

IDRID_GT_TR = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_GT_TE = REPO_ROOT / "data" / "raw" / "idrid_download" / "B. Disease Grading" / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"

IMAGE_SIZE  = [256, 256]
BATCH_SIZE  = 32
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 7


def run_inference_on_loader(model, loader, device, temperature: float = 1.0):
    """Return (y_true, y_pred_ordinal, y_probs_calibrated)."""
    all_true, all_pred, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)

            # Calibrated categorical probs
            probs = torch.softmax(out["logits"] / temperature, dim=-1).cpu().numpy()

            # Ordinal-sum grade
            ord_probs = torch.sigmoid(out["ordinal_logits"])
            grades = torch.round(ord_probs.sum(dim=-1)).long().clamp(0, 4).cpu().numpy()

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(grades.tolist())
            all_probs.append(probs)

    return (np.array(all_true),
            np.array(all_pred),
            np.concatenate(all_probs, axis=0))


def bootstrap_metrics(y_true, y_pred, y_probs, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Return dict with point estimates and 95% CIs for key metrics."""
    point = evaluate_cohort(y_true, y_pred, y_probs=y_probs)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    bs_qwk, bs_sens, bs_spec, bs_auc = [], [], [], []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        m = evaluate_cohort(y_true[idx], y_pred[idx], y_probs=y_probs[idx])
        bs_qwk.append(m["qwk"])
        bs_sens.append(m["sensitivity"])
        bs_spec.append(m["specificity"])
        bs_auc.append(m.get("referable_dr_auc", 0.0))

    # Per-class precision
    per_class_prec = {}
    for c in range(5):
        mask_pred = (y_pred == c)
        mask_true = (y_true == c)
        tp = int(np.sum(mask_pred & mask_true))
        fp = int(np.sum(mask_pred & ~mask_true))
        per_class_prec[c] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "qwk":           float(point["qwk"]),
        "sensitivity":   float(point["sensitivity"]),
        "specificity":   float(point["specificity"]),
        "referable_auc": float(point.get("referable_dr_auc", 0.0)),
        "per_class_recall": {int(k): float(v) for k, v in point["per_class_recall"].items()},
        "per_class_precision": per_class_prec,
        "confusion_matrix": [[int(cell) for cell in row] for row in point["confusion_matrix"]],
        "ci_qwk":   [float(np.percentile(bs_qwk, 2.5)), float(np.percentile(bs_qwk, 97.5))],
        "ci_sens":  [float(np.percentile(bs_sens, 2.5)), float(np.percentile(bs_sens, 97.5))],
        "ci_spec":  [float(np.percentile(bs_spec, 2.5)), float(np.percentile(bs_spec, 97.5))],
        "ci_auc":   [float(np.percentile(bs_auc, 2.5)), float(np.percentile(bs_auc, 97.5))],
    }


def print_report(title: str, m: dict, n: int):
    print(f"\n--- {title} (N={n}) ---")
    print(f"  QWK               : {m['qwk']:.4f}  "
          f"95% CI [{m['ci_qwk'][0]:.4f}, {m['ci_qwk'][1]:.4f}]")
    print(f"  Referable Sens    : {m['sensitivity']:.4f}  "
          f"95% CI [{m['ci_sens'][0]:.4f}, {m['ci_sens'][1]:.4f}]")
    print(f"  Referable Spec    : {m['specificity']:.4f}  "
          f"95% CI [{m['ci_spec'][0]:.4f}, {m['ci_spec'][1]:.4f}]")
    print(f"  Referable AUC     : {m['referable_auc']:.4f}  "
          f"95% CI [{m['ci_auc'][0]:.4f}, {m['ci_auc'][1]:.4f}]")
    print(f"  Per-class metrics:")
    for c in range(5):
        rec = m["per_class_recall"].get(c, 0.0)
        prec = m.get("per_class_precision", {}).get(c, 0.0)
        print(f"    Grade {c}: recall={rec:.4f}  precision={prec:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    cm = np.array(m["confusion_matrix"])
    hdr = "         " + "  ".join(f"P{c}" for c in range(5))
    print(f"  {hdr}")
    for r in range(5):
        row = "  ".join(f"{cm[r, c]:4d}" for c in range(5))
        print(f"  True {r}:  {row}")


def save_predictions(path: Path, y_true, y_pred, y_probs, id_codes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_code", "true_grade", "pred_grade"] +
                   [f"prob_{i}" for i in range(5)])
        for i, (yt, yp, ypr) in enumerate(zip(y_true, y_pred, y_probs)):
            code = id_codes[i] if id_codes is not None else str(i)
            w.writerow([code, int(yt), int(yp)] + [f"{p:.6f}" for p in ypr])
    print(f"  Predictions saved -> {path}")


class IDRiDInferenceDataset(TorchDataset):
    def __init__(self, records, image_size=(256, 256)):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        with PILImage.open(rec["path"]) as im:
            arr = np.array(im.convert("RGB"))
        arr = cv2.resize(arr, (self.image_size[1], self.image_size[0]),
                         interpolation=cv2.INTER_LINEAR)
        arr = arr.astype(np.float32) / 255.0
        arr = (arr - FUNDUS_MEAN) / FUNDUS_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        return tensor, rec["grade"]


def main():
    parser = argparse.ArgumentParser(description="Final held-out test evaluation.")
    parser.add_argument("--confirm", action="store_true",
                        help="Required flag to confirm one-time-use test execution.")
    args = parser.parse_args()

    if not args.confirm:
        print(
            "\n[SAFEGUARD] This script evaluates on HELD-OUT TEST DATA (APTOS test split + IDRiD).\n"
            "Run it EXACTLY ONCE, only after all calibration and gating decisions are finalized.\n"
            "Do NOT tune anything after viewing these results.\n\n"
            "To proceed, re-run with:  python ml/scripts/final_test.py --confirm\n"
        )
        sys.exit(1)

    print("=" * 70)
    print("RetinaGuard  Final Test Evaluation  (Step 7)")
    print("ONE-TIME USE — do not tune anything after these results.")
    print("=" * 70)

    # --- Load model & temperature ---
    device = torch.device("cpu")
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  best_qwk={ckpt.get('best_qwk','?')}")

    # Read temperature from registry
    temperature = 1.0
    if REGISTRY.exists():
        with open(REGISTRY, encoding="utf-8") as f:
            reg = json.load(f)
        entry = reg["models"][0]
        temperature = float(entry.get("temperature", 1.0))
        print(f"  Temperature from registry: {temperature:.4f}")
    else:
        print("  WARNING: registry.json not found — using T=1.0")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # =========================================================
    # 7a — APTOS TEST split
    # =========================================================
    print("\n" + "=" * 70)
    print("7a — APTOS TEST split")
    print("=" * 70)

    test_ds = APTOSDataset(
        manifest_path=MANIFEST, cache_dir=CACHE_APTOS,
        split="test", augment=False, image_size=IMAGE_SIZE,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)
    print(f"  Samples: {len(test_ds)}")

    y_true_t, y_pred_t, y_probs_t = run_inference_on_loader(
        model, test_loader, device, temperature=temperature)
    m_test = bootstrap_metrics(y_true_t, y_pred_t, y_probs_t)
    print_report("APTOS TEST", m_test, len(y_true_t))
    test_ids = [r["id_code"] for r in test_ds.records]
    save_predictions(OUT_DIR / "predictions_test.csv", y_true_t, y_pred_t, y_probs_t, id_codes=test_ids)
    all_results["aptos_test"] = m_test

    # =========================================================
    # 7b — IDRiD evaluation
    # =========================================================
    print("\n" + "=" * 70)
    print("7b — IDRiD evaluation")
    print("=" * 70)
    print("Note: IDRiD grade 1 has only ~25 images; its CI will be wide.")

    # Parse IDRiD Training and Testing records
    def load_idrid_records(gt_csv: Path, cache_dir: Path, tag: str):
        records = []
        with open(gt_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = (row.get("Image name") or row.get("image_id") or row.get("Image_name", "")).strip()
                grade_raw = (row.get("Retinopathy grade") or row.get("diagnosis") or row.get("grade", "0")).strip()
                img_p = cache_dir / f"{img_name}.png"
                if not img_p.exists():
                    img_p = cache_dir / f"{img_name}.jpg"
                if img_p.exists():
                    records.append({"id_code": f"{tag}_{img_name}", "raw_id": img_name,
                                    "grade": int(grade_raw), "path": img_p, "tag": tag})
                else:
                    print(f"  WARNING: missing {img_p}")
        return records

    tr_records = load_idrid_records(IDRID_GT_TR, CACHE_IDRID_TR, "train")
    te_records = load_idrid_records(IDRID_GT_TE, CACHE_IDRID_TE, "test")
    all_idrid_records = tr_records + te_records

    print(f"  IDRiD train subset cached: {len(tr_records)}")
    print(f"  IDRiD test subset cached:  {len(te_records)}")
    print(f"  Total IDRiD images cached: {len(all_idrid_records)}")

    # Grade distribution in IDRiD
    counts_all = {c: sum(1 for r in all_idrid_records if r["grade"] == c) for c in range(5)}
    counts_te  = {c: sum(1 for r in te_records if r["grade"] == c) for c in range(5)}
    print(f"  All IDRiD grade counts:  {counts_all}")
    print(f"  Test IDRiD grade counts: {counts_te}")
    print(f"  Notice Grade 1 count in IDRiD all: {counts_all.get(1, 0)} (wide CI expected)")

    def eval_subset(records, label: str):
        ds = IDRiDInferenceDataset(records, image_size=tuple(IMAGE_SIZE))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
        yt, yp, ypr = run_inference_on_loader(model, loader, device, temperature=temperature)
        m = bootstrap_metrics(yt, yp, ypr, n=N_BOOTSTRAP)
        print_report(label, m, len(yt))
        ids = [r["id_code"] for r in records]
        return yt, yp, ypr, m, ids

    # 1. All 516 IDRiD images
    yt_all, yp_all, ypr_all, m_all, ids_all = eval_subset(all_idrid_records, "IDRiD ALL (516 images)")
    save_predictions(OUT_DIR / "predictions_idrid_all.csv", yt_all, yp_all, ypr_all, id_codes=ids_all)
    all_results["idrid_all"] = m_all

    # 2. Official 103-image test split
    yt_te, yp_te, ypr_te, m_te, ids_te = eval_subset(te_records, "IDRiD OFFICIAL TEST (103 images)")
    save_predictions(OUT_DIR / "predictions_idrid_test.csv", yt_te, yp_te, ypr_te, id_codes=ids_te)
    all_results["idrid_official_test"] = m_te

    # Save combined results JSON
    results_json = OUT_DIR / "final_test_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal test results saved -> {results_json}")

    print("\n" + "=" * 70)
    print("STEP 7 COMPLETE — do not tune anything after these results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
