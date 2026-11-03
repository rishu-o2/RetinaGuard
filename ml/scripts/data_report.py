"""Generate comprehensive data quality and distribution report for APTOS 2019.

Requirements:
- Class counts per split (from data/processed/splits_v1.csv)
- Image-size statistics (min, max, mean width/height)
- Brightness and blur statistics (using Laplacian variance and mean luminance)
- Quality gate assessment: count how many images per grade fail the quality gate
- Per-grade sample grids saved to docs/data_report/
- Markdown summary report saved to docs/data_report/report.md
"""

import csv
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Add paths for retinaguard and retinaguard_ml
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "backend" / "src"))
sys.path.insert(0, str(repo_root / "ml" / "src"))

from retinaguard.core.models import CameraType
from retinaguard.services.quality import QualityGateService


def create_sample_grid(image_paths: list[Path], output_path: Path, thumb_size: tuple[int, int] = (256, 256), cols: int = 4) -> None:
    """Create a composite image grid from a list of image paths."""
    if not image_paths:
        return
    thumbs = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                img_rgb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                # Pad to square thumb_size if necessary
                square = Image.new("RGB", thumb_size, (0, 0, 0))
                offset = ((thumb_size[0] - img_rgb.width) // 2, (thumb_size[1] - img_rgb.height) // 2)
                square.paste(img_rgb, offset)
                thumbs.append(square)
        except Exception as e:
            print(f"Warning: could not process {p} for grid: {e}")

    if not thumbs:
        return

    n_images = len(thumbs)
    cols = min(cols, n_images)
    rows = (n_images + cols - 1) // cols

    grid_w = cols * thumb_size[0]
    grid_h = rows * thumb_size[1]
    grid_img = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))

    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        grid_img.paste(thumb, (c * thumb_size[0], r * thumb_size[1]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_path, "PNG")


def generate_data_report(
    splits_csv: Path,
    raw_images_dir: Path,
    output_dir: Path,
    num_samples_per_grade: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_service = QualityGateService()

    # 1. Load splits
    split_records = []
    with open(splits_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_records.append({
                "id_code": row["id_code"],
                "diagnosis": int(row["diagnosis"]),
                "split": row["split"],
                "group_id": row["group_id"],
            })

    total_images = len(split_records)
    print(f"Generating data report for {total_images} records...")

    # 2. Count distributions per split & grade
    split_grade_counts: dict[str, dict[int, int]] = {
        "train": {g: 0 for g in range(5)},
        "val": {g: 0 for g in range(5)},
        "test": {g: 0 for g in range(5)},
    }
    grade_to_paths: dict[int, list[Path]] = {g: [] for g in range(5)}

    for rec in split_records:
        g = rec["diagnosis"]
        s = rec["split"]
        if s in split_grade_counts:
            split_grade_counts[s][g] += 1
        img_path = raw_images_dir / f"{rec['id_code']}.png"
        if not img_path.exists():
            img_path = raw_images_dir / f"{rec['id_code']}.jpg"
        if img_path.exists():
            grade_to_paths[g].append(img_path)

    # 3. Create sample grids per grade
    print("Generating sample grids per grade...")
    for grade, paths in grade_to_paths.items():
        # Pick first N samples deterministically
        sample_paths = paths[:num_samples_per_grade]
        grid_file = output_dir / f"grade_{grade}_samples.png"
        create_sample_grid(sample_paths, grid_file, thumb_size=(256, 256), cols=4)

    # 4. Image statistics & Quality Gate evaluation
    widths = []
    heights = []
    blurs = []
    luminances = []
    quality_failures_per_grade: dict[int, int] = {g: 0 for g in range(5)}
    failure_reasons_count: dict[str, int] = {}

    print("Analyzing resolutions, brightness, focus blur, and quality gate...")
    for idx, rec in enumerate(split_records):
        img_path = raw_images_dir / f"{rec['id_code']}.png"
        if not img_path.exists():
            img_path = raw_images_dir / f"{rec['id_code']}.jpg"
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                img_rgb = np.array(img.convert("RGB"))
        except Exception:
            continue

        # Evaluate quality gate
        q_result = quality_service.evaluate(img_rgb, camera_type=CameraType.TABLETOP)
        blurs.append(q_result.scores.blur_variance)
        luminances.append(q_result.scores.mean_luminance)

        if not q_result.passed:
            quality_failures_per_grade[rec["diagnosis"]] += 1
            for code in q_result.failure_codes:
                failure_reasons_count[code] = failure_reasons_count.get(code, 0) + 1

        if (idx + 1) % 500 == 0 or (idx + 1) == total_images:
            print(f"  Processed {idx + 1}/{total_images}...")

    # Statistics calculation
    widths = np.array(widths)
    heights = np.array(heights)
    blurs = np.array(blurs)
    luminances = np.array(luminances)

    total_failures = sum(quality_failures_per_grade.values())
    fail_rate = (total_failures / total_images * 100.0) if total_images > 0 else 0.0

    # 5. Write Markdown Report
    grade_names = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }

    report_md = f"""# RetinaGuard - APTOS 2019 Dataset Analysis Report

Generated by `ml/scripts/data_report.py`.
Strictly reflects empirical measurements from the ingested cohort.

---

## 1. Class Distribution Per Split

| ICDR Grade | Clinical Category | Train Count (Col %) | Val Count (Col %) | Test Count (Col %) | Total Images |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
    train_total = sum(split_grade_counts["train"].values())
    val_total = sum(split_grade_counts["val"].values())
    test_total = sum(split_grade_counts["test"].values())

    for g in range(5):
        tr = split_grade_counts["train"][g]
        va = split_grade_counts["val"][g]
        te = split_grade_counts["test"][g]
        tot = tr + va + te
        tr_pct = (tr / train_total * 100.0) if train_total > 0 else 0.0
        va_pct = (va / val_total * 100.0) if val_total > 0 else 0.0
        te_pct = (te / test_total * 100.0) if test_total > 0 else 0.0
        report_md += f"| **{g}** | {grade_names[g]} | {tr} ({tr_pct:.1f}%) | {va} ({va_pct:.1f}%) | {te} ({te_pct:.1f}%) | **{tot}** |\n"

    report_md += f"| **Total** | -- | **{train_total}** | **{val_total}** | **{test_total}** | **{total_images}** |\n\n"

    report_md += f"""---

## 2. Image Resolution & Morphology Statistics

| Metric | Width (px) | Height (px) | Aspect Ratio (W/H) |
| :--- | :---: | :---: | :---: |
| **Minimum** | {widths.min()} | {heights.min()} | {(widths / heights).min():.2f} |
| **Maximum** | {widths.max()} | {heights.max()} | {(widths / heights).max():.2f} |
| **Mean ± Std** | {widths.mean():.1f} ± {widths.std():.1f} | {heights.mean():.1f} ± {heights.std():.1f} | {(widths / heights).mean():.2f} ± {(widths / heights).std():.2f} |
| **Median (IQR)** | {int(np.median(widths))} ({int(np.percentile(widths, 25))}-{int(np.percentile(widths, 75))}) | {int(np.median(heights))} ({int(np.percentile(heights, 25))}-{int(np.percentile(heights, 75))}) | {np.median(widths / heights):.2f} |

---

## 3. Optical Illumination & Focus Clarity Metrics

Computed inside the detected circular retinal aperture:

* **Laplacian Focus Variance (Blur)**:
  * Mean ± Std: `{blurs.mean():.1f} ± {blurs.std():.1f}`
  * 5th – 95th Percentile: `{np.percentile(blurs, 5):.1f} – {np.percentile(blurs, 95):.1f}`
  * Range: `[{blurs.min():.1f}, {blurs.max():.1f}]`
* **Mean Grayscale Luminance (0–255)**:
  * Mean ± Std: `{luminances.mean():.1f} ± {luminances.std():.1f}`
  * 5th – 95th Percentile: `{np.percentile(luminances, 5):.1f} – {np.percentile(luminances, 95):.1f}`
  * Range: `[{luminances.min():.1f}, {luminances.max():.1f}]`

---

## 4. Classical CV Quality Gate Audit

*Notice: Images failing the quality gate are documented here for clinical auditability. They are **not removed** from training/testing sets automatically.*

* **Total Images Scanned**: `{total_images}`
* **Images Flagged by Quality Gate**: `{total_failures}` (`{fail_rate:.2f}%`)

### Quality Gate Failures by DR Grade

| ICDR Grade | Class Name | Total Class Images | Quality Gate Failures | Failure Rate (%) |
| :---: | :--- | :---: | :---: | :---: |
"""

    for g in range(5):
        tot_g = sum(split_grade_counts[s][g] for s in ("train", "val", "test"))
        fails = quality_failures_per_grade[g]
        rate = (fails / tot_g * 100.0) if tot_g > 0 else 0.0
        report_md += f"| **{g}** | {grade_names[g]} | {tot_g} | {fails} | {rate:.1f}% |\n"

    report_md += "\n### Triggered Quality Rejection Reasons\n\n"
    for reason, count in sorted(failure_reasons_count.items(), key=lambda x: x[1], reverse=True):
        report_md += f"* **`{reason}`**: {count} images ({count / total_images * 100.0:.1f}%)\n"

    report_md += f"""
---

## 5. Per-Grade Representative Sample Grids

Each grid displays 8 representative fundus images from that diagnostic category.

### Grade 0 (No Diabetic Retinopathy)
![Grade 0 Samples](grade_0_samples.png)

### Grade 1 (Mild Non-Proliferative DR)
![Grade 1 Samples](grade_1_samples.png)

### Grade 2 (Moderate Non-Proliferative DR)
![Grade 2 Samples](grade_2_samples.png)

### Grade 3 (Severe Non-Proliferative DR)
![Grade 3 Samples](grade_3_samples.png)

### Grade 4 (Proliferative Diabetic Retinopathy)
![Grade 4 Samples](grade_4_samples.png)
"""

    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[+] Data quality report successfully saved to: {report_path}")


def main() -> None:
    splits_csv = Path("data/processed/splits_v1.csv")
    raw_images_dir = Path("data/raw/aptos/train_images")
    output_dir = Path("docs/data_report")

    if not splits_csv.exists():
        print(f"[ERROR] Splits file not found: {splits_csv}")
        print("Please run `prepare_splits.py` first.")
        sys.exit(1)

    if not raw_images_dir.exists():
        print(f"[ERROR] Raw images dir not found: {raw_images_dir}")
        sys.exit(1)

    generate_data_report(splits_csv, raw_images_dir, output_dir)


if __name__ == "__main__":
    main()

