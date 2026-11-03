"""prepare_splits.py — End-to-end leakage-safe split generation for APTOS 2019.

Workflow:
  1. Load duplicate_groups.csv  (run find_duplicates.py first)
  2. Load APTOS ImageRecords
  3. Build group-aware train/val/test splits (every dup group stays in one split)
  4. Save data/processed/splits_v1.csv

Usage:
    python ml/scripts/prepare_splits.py
    python ml/scripts/prepare_splits.py --dup-csv data/processed/duplicate_groups.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinaguard_ml.data.loaders.aptos import APTOS2019Loader
from retinaguard_ml.data.split import ContentGroupSplitter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build group-aware splits for APTOS 2019.")
    parser.add_argument("--root", default="data/raw/aptos")
    parser.add_argument("--dup-csv", default="data/processed/duplicate_groups.csv")
    parser.add_argument("--out", default="data/processed/splits_v1.csv")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    dup_csv = Path(args.dup_csv)
    if not dup_csv.exists():
        print(f"[ERROR] Duplicate groups CSV not found: {dup_csv}", file=sys.stderr)
        print("  Run: python ml/scripts/find_duplicates.py  first.", file=sys.stderr)
        sys.exit(1)

    # ── Load image-to-group mapping ────────────────────────────────────────
    id_to_group: dict[str, str] = {}
    with open(dup_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_group[row["id_code"]] = row["group_id"]

    print(f"[prepare_splits] Loaded {len(id_to_group)} image->group mappings from {dup_csv}")

    # ── Load APTOS records ─────────────────────────────────────────────────
    loader = APTOS2019Loader()
    records = loader.load_records(root_dir=args.root)
    print(f"[prepare_splits] Loaded {len(records)} APTOS records")

    # Override patient_id with content-hash group_id so the splitter groups correctly
    for rec in records:
        rec.patient_id = id_to_group.get(rec.image_id, rec.image_id)

    # ── Split ──────────────────────────────────────────────────────────────
    splitter = ContentGroupSplitter(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    splits = splitter.split(records)

    # ── Save splits_v1.csv ─────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id_code", "diagnosis", "group_id", "split"
        ])
        writer.writeheader()
        for split_name, recs in splits.items():
            for rec in recs:
                writer.writerow({
                    "id_code": rec.image_id,
                    "diagnosis": rec.grade,
                    "group_id": rec.patient_id,
                    "split": split_name,
                })

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Split Manifest: {out_path}")
    print(f"{'='*55}")
    for split_name in ("train", "val", "test"):
        recs = splits.get(split_name, [])
        grade_counts = {}
        for r in recs:
            grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1
        dist = "  ".join(f"G{g}={grade_counts.get(g,0)}" for g in range(5))
        print(f"  {split_name:5s}: {len(recs):5d} images  [{dist}]")
    print(f"{'='*55}\n")

    # Verify no leakage
    splitter.assert_no_leakage(splits)
    print("[OK] Leakage check passed — no group spans multiple splits.\n")


if __name__ == "__main__":
    main()

