"""Compare v1 vs v2 on VAL and IDRiD (reporting only).

RULES:
  - VAL is used for threshold selection (already done in eval_pipeline.py).
  - IDRiD (all 516 + official test 103) is reporting only: no threshold choices.
  - APTOS test split is NOT touched.
  - Paired bootstrap (1000 resamples) of v2-v1 differences on VAL and IDRiD.

Usage:
    python ml/scripts/compare_v1_v2.py

Requires:
    - ml/evaluation/val_report/predictions_val.csv        (v1 VAL preds)
    - ml/evaluation/val_report_v2/predictions_val.csv     (v2 VAL preds)
    - models/registry.json                                (T* for each version)
    - IDRiD cached images in data/processed/cache_512/idrid/{train,test}/
    - IDRiD ground truth CSVs
"""

from __future__ import annotations

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

# ===========================================================================
# Paths
# ===========================================================================
MANIFEST       = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
CACHE_APTOS    = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
CACHE_IDRID_TR = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "train"
CACHE_IDRID_TE = REPO_ROOT / "data" / "processed" / "cache_512" / "idrid" / "test"
REGISTRY       = REPO_ROOT / "models" / "registry.json"
OUT_DIR_V1     = REPO_ROOT / "ml" / "evaluation" / "val_report"
OUT_DIR_V2     = REPO_ROOT / "ml" / "evaluation" / "val_report_v2"
COMPARE_OUT    = REPO_ROOT / "ml" / "evaluation" / "comparison_v1_v2"

IDRID_GT_TR = (REPO_ROOT / "data" / "raw" / "idrid_download" /
               "B. Disease Grading" / "2. Groundtruths" /
               "a. IDRiD_Disease Grading_Training Labels.csv")
IDRID_GT_TE = (REPO_ROOT / "data" / "raw" / "idrid_download" /
               "B. Disease Grading" / "2. Groundtruths" /
               "b. IDRiD_Disease Grading_Testing Labels.csv")

IMAGE_SIZE    = [256, 256]
BATCH_SIZE    = 32
N_BOOTSTRAP   = 1000
BOOTSTRAP_SEED = 42

COMPARE_OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("RetinaGuard  v1 vs v2 Comparison")
print("=" * 70)
print("RULES: IDRiD = reporting only. No threshold changes. No APTOS test.")
print()

# ===========================================================================
# Registry: load T* and checkpoint for each version
# ===========================================================================
def _load_registry_entry(version_tag: str) -> dict:
    """Return registry entry for the given version_tag."""
    if not REGISTRY.exists():
        raise FileNotFoundError(f"registry.json not found at {REGISTRY}")
    with open(REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)
    for entry in reg.get("models", []):
        if entry.get("version_tag", entry.get("version", "")).startswith(version_tag):
            return entry
    raise ValueError(f"No registry entry found for version_tag={version_tag!r}")


def _load_model_from_entry(entry: dict) -> tuple:
    """Return (model, temperature) from a registry entry."""
    ckpt_path = REPO_ROOT / entry["checkpoint_path"]
    temperature = float(entry.get("temperature", 1.0))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, temperature


try:
    entry_v1 = _load_registry_entry("v1")
    entry_v2 = _load_registry_entry("v2")
except (FileNotFoundError, ValueError) as exc:
    print(f"ERROR loading registry: {exc}")
    print("Run eval_pipeline.py --version v1 and --version v2 first.")
    sys.exit(1)

T_v1 = float(entry_v1.get("temperature", 1.0))
T_v2 = float(entry_v2.get("temperature", 1.0))
print(f"v1  checkpoint: {entry_v1['checkpoint_path']}")
print(f"v1  T*        : {T_v1:.4f}  ECE_after={entry_v1.get('ece_after', '?')}")
print(f"v2  checkpoint: {entry_v2['checkpoint_path']}")
print(f"v2  T*        : {T_v2:.4f}  ECE_after={entry_v2.get('ece_after', '?')}")
print()

print("Loading model weights ...")
model_v1, _t1 = _load_model_from_entry(entry_v1)
model_v2, _t2 = _load_model_from_entry(entry_v2)
print("  Done.\n")

# ===========================================================================
# IDRiD inference dataset
# ===========================================================================
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


def load_idrid_records(gt_csv: Path, cache_dir: Path, tag: str) -> list:
    if not gt_csv.exists():
        print(f"  WARNING: IDRiD GT CSV not found: {gt_csv}")
        return []
    records = []
    with open(gt_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = (
                row.get("Image name") or row.get("image_id") or
                row.get("Image_name", "")
            ).strip()
            grade_raw = (
                row.get("Retinopathy grade") or row.get("diagnosis") or
                row.get("grade", "0")
            ).strip()
            img_p = cache_dir / f"{img_name}.png"
            if not img_p.exists():
                img_p = cache_dir / f"{img_name}.jpg"
            if img_p.exists():
                records.append({
                    "id_code": f"{tag}_{img_name}",
                    "raw_id": img_name,
                    "grade": int(grade_raw),
                    "path": img_p,
                    "tag": tag,
                })
            else:
                print(f"  WARNING: missing {img_p}")
    return records


# ===========================================================================
# Inference helpers
# ===========================================================================
def run_inference_on_loader(model, loader, temperature: float):
    """Return (y_true, y_pred_ordinal, y_probs_calibrated)."""
    all_true, all_pred, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs)
            probs = torch.softmax(out["logits"] / temperature, dim=-1).cpu().numpy()
            ord_probs = torch.sigmoid(out["ordinal_logits"])
            grades = torch.round(ord_probs.sum(dim=-1)).long().clamp(0, 4).cpu().numpy()
            all_true.extend(labels.numpy().tolist())
            all_pred.extend(grades.tolist())
            all_probs.append(probs)
    return np.array(all_true), np.array(all_pred), np.concatenate(all_probs, axis=0)


def run_inference_split(model, split: str, temperature: float):
    ds = APTOSDataset(
        manifest_path=MANIFEST, cache_dir=CACHE_APTOS,
        split=split, augment=False, image_size=IMAGE_SIZE,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return run_inference_on_loader(model, loader, temperature)


def run_inference_idrid(model, records, temperature: float):
    if not records:
        return np.array([]), np.array([]), np.empty((0, 5))
    ds = IDRiDInferenceDataset(records, image_size=tuple(IMAGE_SIZE))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return run_inference_on_loader(model, loader, temperature)


# ===========================================================================
# Bootstrap helpers
# ===========================================================================
def _bootstrap_metrics(y_true, y_pred, y_probs, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Return dict with point estimates + 95% CIs."""
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
    return {
        "qwk": float(point["qwk"]),
        "sensitivity": float(point["sensitivity"]),
        "specificity": float(point["specificity"]),
        "auc": float(point.get("referable_dr_auc", 0.0)),
        "per_class_recall": {int(k): float(v) for k, v in point["per_class_recall"].items()},
        "ece_after": None,  # filled in from registry
        "ci_qwk":  [float(np.percentile(bs_qwk,  2.5)), float(np.percentile(bs_qwk,  97.5))],
        "ci_sens": [float(np.percentile(bs_sens, 2.5)), float(np.percentile(bs_sens, 97.5))],
        "ci_spec": [float(np.percentile(bs_spec, 2.5)), float(np.percentile(bs_spec, 97.5))],
        "ci_auc":  [float(np.percentile(bs_auc,  2.5)), float(np.percentile(bs_auc,  97.5))],
    }


def _uncertain_rate(y_probs, low: float, high: float) -> float:
    """Return fraction of samples with P(referable) in [low, high]."""
    p_ref = y_probs[:, 2] + y_probs[:, 3] + y_probs[:, 4]
    uncertain = (p_ref >= low) & (p_ref <= high)
    return float(np.sum(uncertain)) / max(len(uncertain), 1)


def _retained_referable_sens(y_true, y_probs, low: float, high: float) -> float:
    """Retained referable sensitivity at the abstention band."""
    p_ref = y_probs[:, 2] + y_probs[:, 3] + y_probs[:, 4]
    retained = ~((p_ref >= low) & (p_ref <= high))
    true_ref = y_true[retained] >= 2
    pred_ref = p_ref[retained] > high
    if true_ref.sum() == 0:
        return float("nan")
    return float(np.sum(pred_ref & true_ref)) / float(np.sum(true_ref))


def _ge3_sens_at_soon(y_true, y_probs, soon_thresh=0.05, low=0.0, high=1.0) -> float:
    """Grade>=3 sensitivity at SOON tier (p_ge3 >= soon_thresh) among retained."""
    p_ref = y_probs[:, 2] + y_probs[:, 3] + y_probs[:, 4]
    retained = ~((p_ref >= low) & (p_ref <= high))
    p_ge3 = y_probs[:, 3] + y_probs[:, 4]
    soon_pred = p_ge3[retained] >= soon_thresh
    ge3_true  = (y_true[retained] >= 3)
    if ge3_true.sum() == 0:
        return float("nan")
    return float(np.sum(soon_pred & ge3_true)) / float(np.sum(ge3_true))


def _paired_bootstrap_diff(yt, yp1, ypr1, yp2, ypr2, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Paired bootstrap (1000 resamples) of v2-v1 differences in QWK and referable sensitivity.
    Returns dict: {metric: {delta, ci_lo, ci_hi}}.
    """
    rng = np.random.default_rng(seed)
    N = len(yt)
    bs_dqwk, bs_dsens = [], []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        m1 = evaluate_cohort(yt[idx], yp1[idx], y_probs=ypr1[idx])
        m2 = evaluate_cohort(yt[idx], yp2[idx], y_probs=ypr2[idx])
        bs_dqwk.append(m2["qwk"] - m1["qwk"])
        bs_dsens.append(m2["sensitivity"] - m1["sensitivity"])
    return {
        "delta_qwk": {
            "delta": float(np.mean(bs_dqwk)),
            "ci_lo": float(np.percentile(bs_dqwk, 2.5)),
            "ci_hi": float(np.percentile(bs_dqwk, 97.5)),
        },
        "delta_referable_sens": {
            "delta": float(np.mean(bs_dsens)),
            "ci_lo": float(np.percentile(bs_dsens, 2.5)),
            "ci_hi": float(np.percentile(bs_dsens, 97.5)),
        },
    }


# ===========================================================================
# SECTION 1 — VAL split
# ===========================================================================
print("=" * 70)
print("SECTION 1  —  VAL split inference")
print("=" * 70)

print("Running v1 on VAL ...")
yt_val, yp_val_v1, ypr_val_v1 = run_inference_split(model_v1, "val", T_v1)
print(f"  v1 VAL N = {len(yt_val)}")

print("Running v2 on VAL ...")
yt_val_v2, yp_val_v2, ypr_val_v2 = run_inference_split(model_v2, "val", T_v2)
assert len(yt_val) == len(yt_val_v2), "VAL size mismatch between v1 and v2!"
assert np.array_equal(yt_val, yt_val_v2), "VAL labels differ between v1 and v2 runs!"

print("  Bootstrap metrics v1 ...")
mv1_val = _bootstrap_metrics(yt_val, yp_val_v1, ypr_val_v1)
print("  Bootstrap metrics v2 ...")
mv2_val = _bootstrap_metrics(yt_val, yp_val_v2, ypr_val_v2)

# Fill in ECE from registry
mv1_val["ece_after"] = entry_v1.get("ece_after", float("nan"))
mv1_val["temperature"] = T_v1
mv2_val["ece_after"] = entry_v2.get("ece_after", float("nan"))
mv2_val["temperature"] = T_v2

# Abstention band per version (from registry / abstention config)
def _load_band(version_tag: str) -> tuple:
    """Return (low, high) from the entry's abstention config, or v1 defaults."""
    # Try to read from abstention yaml
    cfg_rel = None
    try:
        entry = _load_registry_entry(version_tag)
        cfg_rel = entry.get("abstention_config_path")
    except Exception:
        pass
    if cfg_rel:
        try:
            import yaml
            cfg_path = REPO_ROOT / cfg_rel
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            band = cfg.get("referable_band", {})
            return float(band.get("lower_edge", 0.30)), float(band.get("upper_edge", 0.70))
        except Exception:
            pass
    return 0.30, 0.70  # v1 fallback

LOW_V1, HIGH_V1 = _load_band("v1")
LOW_V2, HIGH_V2 = _load_band("v2")

unc_rate_v1_val = _uncertain_rate(ypr_val_v1, LOW_V1, HIGH_V1)
unc_rate_v2_val = _uncertain_rate(ypr_val_v2, LOW_V2, HIGH_V2)

ret_ref_sens_v1_val = _retained_referable_sens(yt_val, ypr_val_v1, LOW_V1, HIGH_V1)
ret_ref_sens_v2_val = _retained_referable_sens(yt_val, ypr_val_v2, LOW_V2, HIGH_V2)

ge3_soon_v1_val = _ge3_sens_at_soon(yt_val, ypr_val_v1, 0.05, LOW_V1, HIGH_V1)
ge3_soon_v2_val = _ge3_sens_at_soon(yt_val, ypr_val_v2, 0.05, LOW_V2, HIGH_V2)

print(f"\nv1 VAL abstention band: [{LOW_V1:.2f}, {HIGH_V1:.2f}]")
print(f"v2 VAL abstention band: [{LOW_V2:.2f}, {HIGH_V2:.2f}]")

print("  Paired bootstrap VAL (v2 - v1) ...")
diff_val = _paired_bootstrap_diff(yt_val, yp_val_v1, ypr_val_v1, yp_val_v2, ypr_val_v2)

# ===========================================================================
# SECTION 2 — IDRiD inference (reporting only)
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 2  —  IDRiD (reporting only — no thresholds changed)")
print("=" * 70)

tr_records = load_idrid_records(IDRID_GT_TR, CACHE_IDRID_TR, "train")
te_records = load_idrid_records(IDRID_GT_TE, CACHE_IDRID_TE, "test")
all_records = tr_records + te_records

print(f"IDRiD train subset cached : {len(tr_records)}")
print(f"IDRiD test  subset cached : {len(te_records)}")
print(f"IDRiD all   subset cached : {len(all_records)}")

idrid_available = len(all_records) > 0

if idrid_available:
    print("\nRunning v1 on IDRiD all ...")
    yt_idrid_all, yp_all_v1, ypr_all_v1 = run_inference_idrid(model_v1, all_records, T_v1)
    print("Running v2 on IDRiD all ...")
    yt_idrid_all_v2, yp_all_v2, ypr_all_v2 = run_inference_idrid(model_v2, all_records, T_v2)
    assert np.array_equal(yt_idrid_all, yt_idrid_all_v2), "IDRiD all label mismatch!"

    print("Running v1 on IDRiD test (103) ...")
    yt_idrid_te, yp_te_v1, ypr_te_v1 = run_inference_idrid(model_v1, te_records, T_v1)
    print("Running v2 on IDRiD test (103) ...")
    yt_idrid_te_v2, yp_te_v2, ypr_te_v2 = run_inference_idrid(model_v2, te_records, T_v2)

    print("  Bootstrap IDRiD all v1 ...")
    mv1_idrid_all = _bootstrap_metrics(yt_idrid_all, yp_all_v1, ypr_all_v1)
    print("  Bootstrap IDRiD all v2 ...")
    mv2_idrid_all = _bootstrap_metrics(yt_idrid_all, yp_all_v2, ypr_all_v2)

    print("  Bootstrap IDRiD test v1 ...")
    mv1_idrid_te = _bootstrap_metrics(yt_idrid_te, yp_te_v1, ypr_te_v1)
    print("  Bootstrap IDRiD test v2 ...")
    mv2_idrid_te = _bootstrap_metrics(yt_idrid_te, yp_te_v2, ypr_te_v2)

    # Uncertain rates on IDRiD (use each model's band)
    unc_rate_v1_idrid = _uncertain_rate(ypr_all_v1, LOW_V1, HIGH_V1)
    unc_rate_v2_idrid = _uncertain_rate(ypr_all_v2, LOW_V2, HIGH_V2)
    ret_ref_sens_v1_idrid = _retained_referable_sens(yt_idrid_all, ypr_all_v1, LOW_V1, HIGH_V1)
    ret_ref_sens_v2_idrid = _retained_referable_sens(yt_idrid_all, ypr_all_v2, LOW_V2, HIGH_V2)
    ge3_soon_v1_idrid = _ge3_sens_at_soon(yt_idrid_all, ypr_all_v1, 0.05, LOW_V1, HIGH_V1)
    ge3_soon_v2_idrid = _ge3_sens_at_soon(yt_idrid_all, ypr_all_v2, 0.05, LOW_V2, HIGH_V2)

    print("  Paired bootstrap IDRiD all (v2 - v1) ...")
    diff_idrid = _paired_bootstrap_diff(
        yt_idrid_all, yp_all_v1, ypr_all_v1, yp_all_v2, ypr_all_v2
    )

    # Save IDRiD predictions
    def _save_preds(path, y_true, y_pred, y_probs, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id_code", "true_grade", "pred_grade"] + [f"prob_{i}" for i in range(5)])
            for i, (yt, yp, ypr) in enumerate(zip(y_true, y_pred, y_probs)):
                code = records[i]["id_code"] if i < len(records) else str(i)
                w.writerow([code, int(yt), int(yp)] + [f"{p:.6f}" for p in ypr])
        print(f"  Saved -> {path}")

    _save_preds(COMPARE_OUT / "predictions_idrid_all_v1.csv", yt_idrid_all, yp_all_v1, ypr_all_v1, all_records)
    _save_preds(COMPARE_OUT / "predictions_idrid_all_v2.csv", yt_idrid_all, yp_all_v2, ypr_all_v2, all_records)
    _save_preds(COMPARE_OUT / "predictions_idrid_test_v1.csv", yt_idrid_te, yp_te_v1, ypr_te_v1, te_records)
    _save_preds(COMPARE_OUT / "predictions_idrid_test_v2.csv", yt_idrid_te, yp_te_v2, ypr_te_v2, te_records)
else:
    print("  IDRiD data not found — skipping IDRiD evaluation.")
    print("  (Download IDRiD and cache with ml/scripts/prepare_idrid.py)")

# ===========================================================================
# SECTION 3 — Comparison Table
# ===========================================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE — v1 vs v2")
print("=" * 70)

W_L = 42   # label col width
W_V = 12   # value col width

def _fmt(val, prec=4):
    if val is None or (isinstance(val, float) and (val != val)):
        return "n/a"
    return f"{val:.{prec}f}"

def _fmt_ci(lo, hi, prec=4):
    return f"[{lo:.{prec}f}, {hi:.{prec}f}]"

SEP = "=" * (W_L + W_V * 2 + 10)
ROW = f"  {{:<{W_L}}} {{:>{W_V}}} {{:>{W_V}}}"

def hdr(title):
    print(f"\n  {'─' * (W_L + W_V*2 + 6)}")
    print(f"  {title}")
    print(f"  {'─' * (W_L + W_V*2 + 6)}")
    print(ROW.format("Metric", "v1", "v2"))
    print(f"  {'─' * (W_L + W_V*2 + 6)}")

def row(label, v1, v2):
    print(ROW.format(label, v1, v2))

print(SEP)

# --- VAL ---
hdr("VAL split (N=559 expected)")
row("QWK",                      _fmt(mv1_val["qwk"]),         _fmt(mv2_val["qwk"]))
row("  95% CI QWK",             _fmt_ci(*mv1_val["ci_qwk"]),  _fmt_ci(*mv2_val["ci_qwk"]))
row("Referable Sensitivity",    _fmt(mv1_val["sensitivity"]),  _fmt(mv2_val["sensitivity"]))
row("  95% CI ref sens",        _fmt_ci(*mv1_val["ci_sens"]),  _fmt_ci(*mv2_val["ci_sens"]))
row("Referable Specificity",    _fmt(mv1_val["specificity"]),  _fmt(mv2_val["specificity"]))
row("  95% CI ref spec",        _fmt_ci(*mv1_val["ci_spec"]),  _fmt_ci(*mv2_val["ci_spec"]))
row("Referable AUC",            _fmt(mv1_val["auc"]),          _fmt(mv2_val["auc"]))
print()
for c in range(5):
    r1 = mv1_val["per_class_recall"].get(c, 0.0)
    r2 = mv2_val["per_class_recall"].get(c, 0.0)
    row(f"Grade-{c} Recall (VAL)", _fmt(r1), _fmt(r2))
print()
row("ECE after calibration (T*)",  _fmt(mv1_val.get("ece_after"), 4),  _fmt(mv2_val.get("ece_after"), 4))
row("Temperature T*",               _fmt(mv1_val.get("temperature"), 4), _fmt(mv2_val.get("temperature"), 4))
row("Abstention band (P-ref)",
    f"[{LOW_V1:.2f},{HIGH_V1:.2f}]",
    f"[{LOW_V2:.2f},{HIGH_V2:.2f}]")
row("UNCERTAIN rate (VAL)",         _fmt(unc_rate_v1_val, 4),   _fmt(unc_rate_v2_val, 4))
row("Retained ref-sens (VAL)",      _fmt(ret_ref_sens_v1_val, 4), _fmt(ret_ref_sens_v2_val, 4))
row("G>=3 sens at SOON (VAL)",      _fmt(ge3_soon_v1_val, 4),   _fmt(ge3_soon_v2_val, 4))

# --- VAL paired bootstrap ---
print()
hdr("Paired bootstrap (1000 resamples) — v2 minus v1 on VAL")
dv = diff_val
row("Δ QWK (v2-v1)",
    f"Δ={_fmt(dv['delta_qwk']['delta'])}",
    f"95%CI [{_fmt(dv['delta_qwk']['ci_lo'])}, {_fmt(dv['delta_qwk']['ci_hi'])}]")
row("Δ Ref-Sens (v2-v1)",
    f"Δ={_fmt(dv['delta_referable_sens']['delta'])}",
    f"95%CI [{_fmt(dv['delta_referable_sens']['ci_lo'])}, {_fmt(dv['delta_referable_sens']['ci_hi'])}]")

# --- IDRiD ---
if idrid_available:
    hdr(f"IDRiD ALL ({len(all_records)} images) — reporting only")
    row("QWK",                      _fmt(mv1_idrid_all["qwk"]),         _fmt(mv2_idrid_all["qwk"]))
    row("  95% CI QWK",             _fmt_ci(*mv1_idrid_all["ci_qwk"]),  _fmt_ci(*mv2_idrid_all["ci_qwk"]))
    row("Referable Sensitivity",    _fmt(mv1_idrid_all["sensitivity"]),  _fmt(mv2_idrid_all["sensitivity"]))
    row("  95% CI ref sens",        _fmt_ci(*mv1_idrid_all["ci_sens"]),  _fmt_ci(*mv2_idrid_all["ci_sens"]))
    row("Referable Specificity",    _fmt(mv1_idrid_all["specificity"]),  _fmt(mv2_idrid_all["specificity"]))
    row("UNCERTAIN rate (IDRiD)",   _fmt(unc_rate_v1_idrid, 4),    _fmt(unc_rate_v2_idrid, 4))
    row("Retained ref-sens (IDRiD)", _fmt(ret_ref_sens_v1_idrid, 4), _fmt(ret_ref_sens_v2_idrid, 4))
    row("G>=3 sens at SOON (IDRiD)", _fmt(ge3_soon_v1_idrid, 4),   _fmt(ge3_soon_v2_idrid, 4))

    hdr(f"IDRiD TEST ({len(te_records)} images, official split) — reporting only")
    row("QWK",                      _fmt(mv1_idrid_te["qwk"]),         _fmt(mv2_idrid_te["qwk"]))
    row("  95% CI QWK",             _fmt_ci(*mv1_idrid_te["ci_qwk"]),  _fmt_ci(*mv2_idrid_te["ci_qwk"]))
    row("Referable Sensitivity",    _fmt(mv1_idrid_te["sensitivity"]),  _fmt(mv2_idrid_te["sensitivity"]))
    row("  95% CI ref sens",        _fmt_ci(*mv1_idrid_te["ci_sens"]),  _fmt_ci(*mv2_idrid_te["ci_sens"]))
    row("Referable Specificity",    _fmt(mv1_idrid_te["specificity"]),  _fmt(mv2_idrid_te["specificity"]))

    hdr("Paired bootstrap (1000 resamples) — v2 minus v1 on IDRiD ALL")
    di = diff_idrid
    row("Δ QWK (v2-v1)",
        f"Δ={_fmt(di['delta_qwk']['delta'])}",
        f"95%CI [{_fmt(di['delta_qwk']['ci_lo'])}, {_fmt(di['delta_qwk']['ci_hi'])}]")
    row("Δ Ref-Sens (v2-v1)",
        f"Δ={_fmt(di['delta_referable_sens']['delta'])}",
        f"95%CI [{_fmt(di['delta_referable_sens']['ci_lo'])}, {_fmt(di['delta_referable_sens']['ci_hi'])}]")

print(SEP)

# ===========================================================================
# Save JSON summary
# ===========================================================================
summary = {
    "note": "reporting only — no thresholds changed, v1 remains active model",
    "val": {
        "N": int(len(yt_val)),
        "v1": {k: mv1_val[k] for k in mv1_val},
        "v2": {k: mv2_val[k] for k in mv2_val},
        "v1_uncertain_rate": unc_rate_v1_val,
        "v2_uncertain_rate": unc_rate_v2_val,
        "v1_retained_ref_sens": ret_ref_sens_v1_val,
        "v2_retained_ref_sens": ret_ref_sens_v2_val,
        "v1_ge3_soon": ge3_soon_v1_val,
        "v2_ge3_soon": ge3_soon_v2_val,
        "paired_bootstrap_v2_minus_v1": diff_val,
    },
}
if idrid_available:
    summary["idrid_all"] = {
        "N": int(len(yt_idrid_all)),
        "v1": {k: mv1_idrid_all[k] for k in mv1_idrid_all},
        "v2": {k: mv2_idrid_all[k] for k in mv2_idrid_all},
        "v1_uncertain_rate": unc_rate_v1_idrid,
        "v2_uncertain_rate": unc_rate_v2_idrid,
        "v1_retained_ref_sens": ret_ref_sens_v1_idrid,
        "v2_retained_ref_sens": ret_ref_sens_v2_idrid,
        "v1_ge3_soon": ge3_soon_v1_idrid,
        "v2_ge3_soon": ge3_soon_v2_idrid,
        "paired_bootstrap_v2_minus_v1": diff_idrid,
    }
    summary["idrid_test"] = {
        "N": int(len(yt_idrid_te)),
        "v1": {k: mv1_idrid_te[k] for k in mv1_idrid_te},
        "v2": {k: mv2_idrid_te[k] for k in mv2_idrid_te},
    }

out_json = COMPARE_OUT / "comparison_v1_v2.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nComparison JSON saved -> {out_json}")

# Active model
print("\n[REGISTRY CHECK]")
active = [m for m in json.loads(REGISTRY.read_text())["models"] if m.get("active", False)]
inactive = [m for m in json.loads(REGISTRY.read_text())["models"] if not m.get("active", False)]
print(f"  Active model(s)  : {[m.get('version_tag', '?') for m in active]}")
print(f"  Inactive model(s): {[m.get('version_tag', '?') for m in inactive]}")
print("  v1 is the active model — backend unchanged.")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
