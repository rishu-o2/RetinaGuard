#!/usr/bin/env python3
"""Dataset Verification Tool for APTOS 2019 Blindness Detection.

Verifies:
1. train.csv exists with valid schema (id_code, diagnosis).
2. 1:1 correspondence between CSV id_code entries and train_images/*.png files.
3. Diagnostic labels strictly within {0, 1, 2, 3, 4}.
4. Image integrity across a random sample (or all) via PIL.
5. Empirical class distribution statistics.
"""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Set
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VerificationResult:
    """Summary result of dataset verification."""
    is_valid: bool
    total_images_in_csv: int
    total_image_files_on_disk: int
    class_distribution: Dict[int, int]
    missing_image_files: List[str] = field(default_factory=list)
    extra_image_files: List[str] = field(default_factory=list)
    invalid_diagnosis_rows: List[dict] = field(default_factory=list)
    corrupted_images: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def verify_aptos_dataset(
    dataset_dir: Path | str, sample_size: int = 50, seed: int = 42
) -> VerificationResult:
    """Perform comprehensive structural and visual integrity verification on an APTOS directory."""
    dataset_dir = Path(dataset_dir)
    train_csv = dataset_dir / "train.csv"
    images_dir = dataset_dir / "train_images"

    errors: List[str] = []

    # 1. Check folder and CSV existence
    if not dataset_dir.exists():
        return VerificationResult(
            is_valid=False,
            total_images_in_csv=0,
            total_image_files_on_disk=0,
            class_distribution={},
            errors=[f"Dataset directory does not exist: {dataset_dir}"],
        )

    if not train_csv.exists():
        return VerificationResult(
            is_valid=False,
            total_images_in_csv=0,
            total_image_files_on_disk=0,
            class_distribution={},
            errors=[f"Required file 'train.csv' missing in {dataset_dir}"],
        )

    if not images_dir.exists():
        return VerificationResult(
            is_valid=False,
            total_images_in_csv=0,
            total_image_files_on_disk=0,
            class_distribution={},
            errors=[f"Required folder 'train_images' missing in {dataset_dir}"],
        )

    # 2. Parse train.csv and validate columns
    csv_ids: Set[str] = set()
    id_to_grade: Dict[str, int] = {}
    invalid_rows: List[dict] = []
    class_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "id_code" not in fieldnames or "diagnosis" not in fieldnames:
            return VerificationResult(
                is_valid=False,
                total_images_in_csv=0,
                total_image_files_on_disk=0,
                class_distribution={},
                errors=[f"train.csv schema invalid! Expected columns ['id_code', 'diagnosis'], got {fieldnames}"],
            )

        for row_idx, row in enumerate(reader, start=2):
            id_code = row.get("id_code", "").strip()
            raw_diag = row.get("diagnosis", "").strip()

            if not id_code:
                errors.append(f"Row {row_idx}: Empty id_code")
                continue

            try:
                diag = int(raw_diag)
                if diag not in {0, 1, 2, 3, 4}:
                    invalid_rows.append({"row": row_idx, "id_code": id_code, "diagnosis": raw_diag})
                else:
                    class_counts[diag] += 1
                    id_to_grade[id_code] = diag
            except ValueError:
                invalid_rows.append({"row": row_idx, "id_code": id_code, "diagnosis": raw_diag})

            csv_ids.add(id_code)

    if invalid_rows:
        errors.append(f"Found {len(invalid_rows)} row(s) with invalid diagnosis values (outside {{0,1,2,3,4}}).")

    # 3. Check image files on disk
    disk_files = {p.stem: p for p in images_dir.glob("*.png")}
    total_disk_images = len(disk_files)
    disk_ids = set(disk_files.keys())

    missing_images = sorted(list(csv_ids - disk_ids))
    extra_images = sorted(list(disk_ids - csv_ids))

    if missing_images:
        errors.append(f"Missing {len(missing_images)} image files on disk matching train.csv.")
    if extra_images:
        # Extra images are reported but not necessarily fatal if non-empty
        pass

    # 4. Image sample integrity check using PIL
    corrupted_images: List[str] = []
    available_valid_ids = sorted(list(csv_ids & disk_ids))

    if available_valid_ids:
        rng = random.Random(seed)
        num_to_sample = min(sample_size, len(available_valid_ids))
        sampled_ids = rng.sample(available_valid_ids, num_to_sample)

        for sid in sampled_ids:
            img_path = disk_files[sid]
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Fast structural verification
                # Reopen to test actual decompression
                with Image.open(img_path) as img:
                    _ = img.size
                    img.load()
            except Exception as e:
                corrupted_images.append(f"{img_path.name} (error: {e})")

    if corrupted_images:
        errors.append(f"Detected {len(corrupted_images)} corrupt image(s) in sampled subset.")

    is_valid = len(errors) == 0

    return VerificationResult(
        is_valid=is_valid,
        total_images_in_csv=len(csv_ids),
        total_image_files_on_disk=total_disk_images,
        class_distribution=class_counts,
        missing_image_files=missing_images,
        extra_image_files=extra_images,
        invalid_diagnosis_rows=invalid_rows,
        corrupted_images=corrupted_images,
        errors=errors,
    )


def print_verification_report(result: VerificationResult) -> None:
    """Format and print an audit summary report."""
    print("=================================================================")
    print("          RetinaGuard - APTOS 2019 Dataset Verification          ")
    print("=================================================================")
    print(f"Status:                      {'PASSED (VALID)' if result.is_valid else 'FAILED (ERRORS DETECTED)'}")
    print(f"Total Labeled Images in CSV: {result.total_images_in_csv}")
    print(f"Total Image Files on Disk:   {result.total_image_files_on_disk}")
    print(f"Missing Image Files:         {len(result.missing_image_files)}")
    print(f"Extra Image Files on Disk:   {len(result.extra_image_files)}")
    print("-----------------------------------------------------------------")
    print("ICDR Grade Distribution (Empirical Count):")

    grade_names = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }
    total = sum(result.class_distribution.values())
    for g in range(5):
        count = result.class_distribution.get(g, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  Grade {g} ({grade_names[g]:<16}): {count:>5} images ({pct:>5.1f}%)")
    print("-----------------------------------------------------------------")

    if result.errors:
        print("[!] Validation Errors:")
        for err in result.errors:
            print(f"  - {err}")
        if result.missing_image_files:
            print(f"  First missing files: {result.missing_image_files[:5]}")
        if result.invalid_diagnosis_rows:
            print(f"  Invalid diagnosis rows: {result.invalid_diagnosis_rows[:5]}")
        if result.corrupted_images:
            print(f"  Corrupted files: {result.corrupted_images[:5]}")
    else:
        print("[+] All structural and integrity checks passed successfully.")
    print("=================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify APTOS 2019 dataset integrity.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "aptos"),
        help="Path to APTOS dataset directory (default: data/raw/aptos)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of random images to test for corruption (default: 50)",
    )

    args = parser.parse_args()
    result = verify_aptos_dataset(dataset_dir=Path(args.data_dir), sample_size=args.sample_size)
    print_verification_report(result)

    if not result.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()

