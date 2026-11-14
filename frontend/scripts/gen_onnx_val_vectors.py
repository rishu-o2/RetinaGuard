#!/usr/bin/env python3
"""Extract 20 real validation images and compute server ONNX outputs for parity testing."""

import base64
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

from retinaguard.core.config import _find_repo_root

REPO_ROOT = _find_repo_root()


def main():
    splits_csv = REPO_ROOT / "data" / "processed" / "splits_v2.csv"
    cache_dir = REPO_ROOT / "data" / "processed" / "cache_512" / "aptos"
    onnx_path = REPO_ROOT / "ml" / "exports" / "retinaguard_dr_v1.onnx"
    out_json = REPO_ROOT / "frontend" / "tests" / "onnx_parity_20.json"

    df = pd.read_csv(splits_csv)
    val_df = df[df["split"] == "val"]

    # Select 20 real validation images spanning different grades if possible
    selected_samples = []
    for grade in range(5):
        grade_samples = val_df[val_df["diagnosis"] == grade]["id_code"].tolist()
        for sample_id in grade_samples:
            img_path = cache_dir / f"{sample_id}.png"
            if img_path.exists():
                selected_samples.append((sample_id, grade, img_path))
                if len([s for s in selected_samples if s[1] == grade]) >= 4:
                    break
        if len(selected_samples) >= 20:
            break

    # If less than 20, fill with whatever is available
    if len(selected_samples) < 20:
        for sample_id in val_df["id_code"].tolist():
            img_path = cache_dir / f"{sample_id}.png"
            if img_path.exists() and sample_id not in [s[0] for s in selected_samples]:
                selected_samples.append((sample_id, int(val_df[val_df["id_code"] == sample_id]["diagnosis"].iloc[0]), img_path))
            if len(selected_samples) >= 20:
                break

    print(f"Selected {len(selected_samples)} validation images.")

    # Load ONNX model
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    val_records = []
    for sample_id, grade, img_path in selected_samples[:20]:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

        # Preprocessing matching server exactly
        arr = resized.astype(np.float32) / 255.0
        norm = (arr - mean) / std
        nchw = np.transpose(norm, (2, 0, 1))[np.newaxis]

        outputs = sess.run(None, {input_name: nchw})
        cat_logits = outputs[0][0].tolist()
        ord_logits = outputs[1][0].tolist() if len(outputs) > 1 else None

        # RGBA uint8 for TS preprocessor
        h, w = resized.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = resized
        rgba[:, :, 3] = 255

        # Calibrated probabilities with active temperature
        temp = 1.296247
        cat_scaled = outputs[0][0] / temp
        exp_l = np.exp(cat_scaled - np.max(cat_scaled))
        probs = (exp_l / exp_l.sum()).tolist()

        val_records.append({
            "id_code": sample_id,
            "ground_truth_grade": grade,
            "width": w,
            "height": h,
            "tensor_b64": base64.b64encode(nchw.tobytes()).decode("ascii"),
            "server_cat_logits": cat_logits,
            "server_ord_logits": ord_logits,
            "server_probs": probs,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(val_records, f)

    print(f"Saved {len(val_records)} validation records to {out_json}")


if __name__ == "__main__":
    main()
