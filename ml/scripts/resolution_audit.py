"""Generate 12-image grid of resolution-only failures and compute new VAL pass rates."""

import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml

from retinaguard.core.models import CameraType
from retinaguard.services.quality import QualityGateService


def run_resolution_audit():
    df = pd.read_csv("data/processed/splits_v2.csv")
    val_df = df[df["split"] == "val"]
    qg = QualityGateService()

    # Load new config with tabletop min_resolution lowered to 256
    with open("configs/quality_thresholds.yaml", "r", encoding="utf-8") as f:
        cfg_new = yaml.safe_load(f)
    cfg_new["profiles"]["tabletop"]["min_resolution_width"] = 256
    cfg_new["profiles"]["tabletop"]["min_resolution_height"] = 256

    qg_new = QualityGateService()
    qg_new.config = cfg_new

    res_only_12 = []
    val_results_old = []
    val_results_new = []

    for idx, row in val_df.iterrows():
        img_id = row["id_code"]
        grade = int(row["diagnosis"])
        img_path = Path("data/raw/aptos/train_images") / f"{img_id}.png"
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        r_old = qg.evaluate(img_rgb, CameraType.TABLETOP)
        r_new = qg_new.evaluate(img_rgb, CameraType.TABLETOP)

        val_results_old.append({"grade": grade, "passed": r_old.passed})
        val_results_new.append({"grade": grade, "passed": r_new.passed})

        if r_old.failure_codes == ["RESOLUTION_TOO_LOW"] and len(res_only_12) < 12:
            mask, _, _ = qg.segment_fundus_fov(img_rgb)
            y_idx, x_idx = np.where(mask == 1)
            h, w = img_rgb.shape[:2]
            if len(y_idx) > 0 and len(x_idx) > 0:
                ymin, ymax = int(np.min(y_idx)), int(np.max(y_idx))
                xmin, xmax = int(np.min(x_idx)), int(np.max(x_idx))
                retina_diam = min(xmax - xmin + 1, ymax - ymin + 1)
            else:
                retina_diam = min(w, h)
            res_only_12.append((img_id, grade, w, h, retina_diam, img_bgr))

    # Build 4x3 Grid (4 columns, 3 rows)
    cols, rows = 4, 3
    thumb_w, thumb_h = 320, 320
    cell_h = thumb_h + 60
    grid_img = np.zeros((rows * cell_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, (img_id, grade, w, h, retina_diam, bgr) in enumerate(res_only_12):
        r_idx = i // cols
        c_idx = i % cols
        resized = cv2.resize(bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y_start = r_idx * cell_h
        x_start = c_idx * thumb_w
        grid_img[y_start : y_start + thumb_h, x_start : x_start + thumb_w] = resized

        # Captions
        caption1 = f"{img_id[:10]} (Grade {grade})"
        caption2 = f"Dim: {w}x{h} | Retina: {retina_diam}px"
        cv2.putText(
            grid_img,
            caption1,
            (x_start + 10, y_start + thumb_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            grid_img,
            caption2,
            (x_start + 10, y_start + thumb_h + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    grid_path = reports_dir / "resolution_failed_grid.png"
    cv2.imwrite(str(grid_path), grid_img)
    print(f"Saved 12-image grid to {grid_path}")

    artifact_dir = Path(r"C:\Users\hp\.gemini\antigravity\brain\0fffc3b9-96ae-4fec-8b9f-26d6505e3a54")
    if artifact_dir.exists():
        shutil.copy2(grid_path, artifact_dir / "resolution_failed_grid.png")
        print(f"Copied grid to artifact dir: {artifact_dir / 'resolution_failed_grid.png'}")

    old_df = pd.DataFrame(val_results_old)
    new_df = pd.DataFrame(val_results_new)

    old_pass = old_df["passed"].sum()
    new_pass = new_df["passed"].sum()
    total = len(old_df)

    print("\n================ QUALITY GATE PASS RATES (VAL N=559) ================")
    print(f"OLD Threshold (1024x1024): {old_pass}/{total} ({old_pass/total*100:.2f}%) pass")
    print(f"NEW Threshold (256x256):   {new_pass}/{total} ({new_pass/total*100:.2f}%) pass")
    print(f"Recovered Images:          +{new_pass - old_pass} (+{(new_pass - old_pass)/total*100:.2f}%)")
    print("\n--- Pass Rate per Grade ---")
    for g in range(5):
        og = old_df[old_df["grade"] == g]["passed"]
        ng = new_df[new_df["grade"] == g]["passed"]
        print(
            f"Grade {g} (N={len(og)}): "
            f"OLD {og.sum()}/{len(og)} ({og.mean()*100:.2f}%) -> "
            f"NEW {ng.sum()}/{len(ng)} ({ng.mean()*100:.2f}%) "
            f"(+{ng.sum() - og.sum()})"
        )
    print("===================================================================\n")


if __name__ == "__main__":
    run_resolution_audit()

