"""Ingestion script for IDRiD (Indian Diabetic Retinopathy Image Dataset) Part B.

Reads the Training and Testing groundtruth CSVs from data/raw/idrid_download,
resolves image paths robustly without copying or modifying original files,
and generates data/processed/idrid_manifest.csv.
"""

import argparse
import csv
import sys
from pathlib import Path


def clean_col(s: str) -> str:
    """Normalize whitespace and strip extra formatting."""
    return s.strip().lower().replace(" ", "_")


def find_image_file(images_dir: Path, base_name: str) -> Path | None:
    """Match image filename robustly across case and common extensions."""
    # Direct match first
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate

    # Case-insensitive scan
    base_lower = base_name.lower()
    for f in images_dir.iterdir():
        if f.is_file() and f.stem.lower() == base_lower:
            return f

    return None


def ingest_idrid(raw_dir: Path, output_csv: Path) -> list[dict]:
    # Root of Part B
    disease_grading_dir = raw_dir / "B. Disease Grading"
    if not disease_grading_dir.exists():
        # Fallback if unzipped without outer wrapper
        disease_grading_dir = raw_dir

    gt_dir = disease_grading_dir / "2. Groundtruths"
    imgs_dir = disease_grading_dir / "1. Original Images"

    if not gt_dir.exists() or not imgs_dir.exists():
        raise FileNotFoundError(
            f"Expected IDRiD subdirectories not found under {disease_grading_dir}.\n"
            f"Expected '1. Original Images' and '2. Groundtruths'."
        )

    split_configs = [
        {
            "split": "train",
            "csv_pattern": "*Training Labels*.csv",
            "images_dir": imgs_dir / "a. Training Set",
        },
        {
            "split": "test",
            "csv_pattern": "*Testing Labels*.csv",
            "images_dir": imgs_dir / "b. Testing Set",
        },
    ]

    manifest_rows: list[dict] = []
    unmatched_rows: list[dict] = []

    for cfg in split_configs:
        split_name = cfg["split"]
        matching_csvs = list(gt_dir.glob(cfg["csv_pattern"]))
        if not matching_csvs:
            raise FileNotFoundError(f"Could not find groundtruth CSV for pattern: {cfg['csv_pattern']}")

        csv_path = matching_csvs[0]
        images_dir = cfg["images_dir"]
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        print(f"[prepare_idrid] Ingesting {split_name} set from {csv_path.name}...")

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = [clean_col(c) for c in next(reader)]

            try:
                img_col_idx = header.index("image_name")
                dr_col_idx = header.index("retinopathy_grade")
                # Risk of macular edema column might have trailing space
                dme_col_idx = None
                for i, col in enumerate(header):
                    if "macular" in col or "edema" in col:
                        dme_col_idx = i
                        break
                if dme_col_idx is None:
                    dme_col_idx = 2
            except ValueError as e:
                raise ValueError(f"Required column not found in header {header} of {csv_path.name}") from e

            row_count = 0
            for row in reader:
                if not row or not any(row):
                    continue
                raw_name = row[img_col_idx].strip()
                if not raw_name:
                    continue

                dr_val = int(row[dr_col_idx].strip())
                dme_val = int(row[dme_col_idx].strip()) if len(row) > dme_col_idx and row[dme_col_idx].strip() else None

                # Locate corresponding image
                matched_file = find_image_file(images_dir, raw_name)
                if matched_file is None:
                    unmatched_rows.append({"split": split_name, "raw_name": raw_name, "csv": csv_path.name})
                    continue

                # Unique image_id formatting to avoid train/test collision (e.g. IDRiD_train_001)
                clean_num = raw_name.replace("IDRiD_", "").replace("IDRID_", "")
                unique_image_id = f"IDRiD_{split_name}_{clean_num}"

                manifest_rows.append({
                    "image_id": unique_image_id,
                    "file_path": str(matched_file.resolve()),
                    "dr_grade": dr_val,
                    "dme_risk": dme_val if dme_val is not None else -1,
                    "original_split": split_name,
                    "dataset": "idrid",
                    "device_note": "Kowa VX-10 alpha (per IDRiD paper; 50 degree FOV)",
                    "site": "Nanded, India",
                })
                row_count += 1

            print(f"  Processed {row_count} valid records for {split_name}.")

    if unmatched_rows:
        print(f"\n[WARNING] Found {len(unmatched_rows)} unmatched rows in groundtruth CSVs:")
        for u in unmatched_rows:
            print(f"  - Split: {u['split']}, Name: {u['raw_name']}")
    else:
        print("[prepare_idrid] 100% of CSV records matched image files on disk.")

    # Write output manifest
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "file_path",
        "dr_grade",
        "dme_risk",
        "original_split",
        "dataset",
        "device_note",
        "site",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[+] IDRiD external test manifest saved to: {output_csv.resolve()}")
    print(f"    Total records: {len(manifest_rows)}")
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest IDRiD dataset as external test set.")
    parser.add_argument("--raw-dir", default="data/raw/idrid_download", help="Path to unzipped IDRiD directory")
    parser.add_argument("--out", default="data/processed/idrid_manifest.csv", help="Output manifest path")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    out_path = Path(args.out)

    if not raw_path.exists():
        print(f"[ERROR] Raw directory not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    ingest_idrid(raw_path, out_path)


if __name__ == "__main__":
    main()

