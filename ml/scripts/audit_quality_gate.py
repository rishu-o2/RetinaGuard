"""Audit script for Quality Gate on APTOS TRAIN and VAL splits.

Evaluates all images using the official QualityGateService (Tabletop profile)
and breaks down failure rates by check and grade.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
RAW_DIR = REPO_ROOT / "data" / "raw" / "aptos" / "train_images"
OUT_DIR = REPO_ROOT / "ml" / "evaluation" / "val_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

qg_svc = QualityGateService()


def audit_split(split_name: str):
    print(f"\n=======================================================")
    print(f"Auditing split: {split_name.upper()} ...")
    print(f"=======================================================")

    records = []
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split_name:
                records.append(r)

    print(f"Total {split_name} records: {len(records)}")

    # Failure counters: grade -> check -> count
    grade_totals = defaultdict(int)
    pass_counts = defaultdict(int)
    fail_combined = defaultdict(int)
    fail_blur = defaultdict(int)
    fail_fov = defaultdict(int)
    fail_exposure = defaultdict(int)
    fail_res = defaultdict(int)

    all_results = []
    non_blur_failures = []

    for idx, r in enumerate(records):
        code = r["id_code"]
        grade = int(r["diagnosis"])
        grade_totals[grade] += 1

        img_path = RAW_DIR / f"{code}.png"
        if not img_path.exists():
            img_path = RAW_DIR / f"{code}.jpg"

        with Image.open(img_path) as im:
            img_rgb = np.array(im.convert("RGB"))

        res = qg_svc.evaluate(img_rgb, camera_type=CameraType.TABLETOP)
        passed = res.passed
        codes = res.failure_codes

        has_blur = "BLUR" in codes
        has_fov = "INVALID_FOV" in codes
        has_exposure = ("UNDEREXPOSED" in codes) or ("OVEREXPOSED" in codes)
        has_res = "RESOLUTION_TOO_LOW" in codes

        if passed:
            pass_counts[grade] += 1
        else:
            fail_combined[grade] += 1
            if not has_blur:
                non_blur_failures.append({
                    "code": code,
                    "grade": grade,
                    "codes": codes,
                    "scores": res.scores,
                    "path": img_path,
                })

        if has_blur:
            fail_blur[grade] += 1
        if has_fov:
            fail_fov[grade] += 1
        if has_exposure:
            fail_exposure[grade] += 1
        if has_res:
            fail_res[grade] += 1

        all_results.append({
            "code": code,
            "grade": grade,
            "passed": passed,
            "codes": codes,
        })

    # Summary table
    print(f"\nBreakdown for {split_name.upper()} (Total N={len(records)}):")
    print(f"{'Grade':<8} {'Total':<6} {'Passed':<8} {'Pass%':<8} {'Blur Fail':<10} {'FOV Fail':<10} {'Exp Fail':<10} {'Res Fail':<10} {'Total Fail':<10}")
    print("-" * 84)

    total_n = len(records)
    total_pass = sum(pass_counts.values())
    total_fail = sum(fail_combined.values())
    total_blur = sum(fail_blur.values())
    total_fov = sum(fail_fov.values())
    total_exp = sum(fail_exposure.values())
    total_res = sum(fail_res.values())

    for g in range(5):
        tot = grade_totals[g]
        p = pass_counts[g]
        p_pct = 100.0 * p / tot if tot else 0.0
        fb = fail_blur[g]
        ff = fail_fov[g]
        fe = fail_exposure[g]
        fr = fail_res[g]
        fc = fail_combined[g]
        print(f"Grade {g:<2} {tot:<6} {p:<8} {p_pct:<7.1f}% {fb:<10} {ff:<10} {fe:<10} {fr:<10} {fc:<10}")

    print("-" * 84)
    tot_pct = 100.0 * total_pass / total_n
    print(f"{'Overall':<8} {total_n:<6} {total_pass:<8} {tot_pct:<7.1f}% {total_blur:<10} {total_fov:<10} {total_exp:<10} {total_res:<10} {total_fail:<10}")
    print(f"Failure percentages across all {split_name.upper()} images:")
    print(f"  Blur Fail:       {total_blur}/{total_n} = {100.0*total_blur/total_n:.2f}%")
    print(f"  FOV Fail:        {total_fov}/{total_n} = {100.0*total_fov/total_n:.2f}%")
    print(f"  Exposure Fail:   {total_exp}/{total_n} = {100.0*total_exp/total_n:.2f}%")
    print(f"  Resolution Fail: {total_res}/{total_n} = {100.0*total_res/total_n:.2f}%")
    print(f"  Combined Fail:   {total_fail}/{total_n} = {100.0*total_fail/total_n:.2f}%")
    print(f"  Combined Pass:   {total_pass}/{total_n} = {100.0*total_pass/total_n:.2f}%")

    return {
        "records": records,
        "non_blur_failures": non_blur_failures,
        "totals": {
            "n": total_n,
            "pass": total_pass,
            "fail": total_fail,
            "blur": total_blur,
            "fov": total_fov,
            "exp": total_exp,
            "res": total_res,
        },
        "by_grade": {
            g: {
                "total": grade_totals[g],
                "pass": pass_counts[g],
                "fail": fail_combined[g],
                "blur": fail_blur[g],
                "fov": fail_fov[g],
                "exp": fail_exposure[g],
                "res": fail_res[g],
            }
            for g in range(5)
        }
    }


def main():
    train_res = audit_split("train")
    val_res = audit_split("val")

    # Generate 24 images failing for reasons OTHER than blur
    non_blur_val = val_res["non_blur_failures"]
    print(f"\nTotal non-blur failures on VAL: {len(non_blur_val)}")
    # If fewer than 24 on VAL, top up from TRAIN non-blur failures
    candidates = list(non_blur_val)
    if len(candidates) < 24:
        print(f"Topping up with {24 - len(candidates)} non-blur failures from TRAIN...")
        candidates.extend(train_res["non_blur_failures"][: 24 - len(candidates)])

    grid_images = candidates[:24]

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for ax, item in zip(axes.flat, grid_images):
        with Image.open(item["path"]) as im:
            ax.imshow(im)
        short_codes = ",".join(c.replace("RESOLUTION_TOO_LOW", "RES").replace("INVALID_FOV", "FOV").replace("UNDEREXPOSED", "UNDER").replace("OVEREXPOSED", "OVER") for c in item["codes"])
        ax.set_title(f"G{item['grade']} | {short_codes}\n{item['code'][:8]}", fontsize=8)
        ax.axis("off")

    plt.suptitle("24 Images Failing Quality Gate for Reasons OTHER than Blur (FOV, Exposure, Resolution)", fontsize=14)
    plt.tight_layout()
    grid_path = OUT_DIR / "flagged_non_blur_failures_grid.png"
    fig.savefig(str(grid_path), dpi=150)
    plt.close(fig)
    print(f"\nSaved non-blur failure grid -> {grid_path}")

    # Save full audit results to JSON
    audit_json = OUT_DIR / "quality_gate_audit.json"
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump({
            "train": train_res["totals"],
            "train_by_grade": train_res["by_grade"],
            "val": val_res["totals"],
            "val_by_grade": val_res["by_grade"],
        }, f, indent=2)
    print(f"Saved audit JSON -> {audit_json}")


if __name__ == "__main__":
    main()

