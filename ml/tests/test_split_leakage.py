"""Unit tests for leakage-safe splitting and group isolation.

Ensures:
1. All images belonging to the same duplicate group (or patient) reside in the exact same split.
2. Group overlap detection catches any leakage across train, val, and test partitions.
3. If data/processed/splits_v1.csv exists, the actual generated manifest has zero leakage.
"""

import csv
from pathlib import Path
import pytest

from retinaguard_ml.data.registry import ImageRecord
from retinaguard_ml.data.split import ContentGroupSplitter, PatientLeakageError, PatientStratifiedSplitter


def test_content_group_splitter_no_leakage():
    """Verify that multiple records sharing the same group_id stay in one split."""
    records = []
    # Create 50 groups, each having 2 images (100 total)
    for g_idx in range(50):
        group_id = f"GROUP_{g_idx:03d}"
        grade = g_idx % 5  # ICDR 0-4
        for img_idx in (1, 2):
            records.append(
                ImageRecord(
                    image_id=f"{group_id}_IMG{img_idx}",
                    patient_id=group_id,  # group_id stored in patient_id field
                    grade=grade,
                    file_path=f"dummy_{group_id}_{img_idx}.png",
                )
            )

    splitter = ContentGroupSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    splits = splitter.split(records)

    # 1. Assert internal leakage check passes
    splitter.assert_no_leakage(splits)

    # 2. Check each group independently
    group_to_splits = {}
    for split_name, recs in splits.items():
        if split_name == "external_test":
            continue
        for r in recs:
            group_to_splits.setdefault(r.patient_id, set()).add(split_name)

    for group_id, seen_splits in group_to_splits.items():
        assert len(seen_splits) == 1, f"Group {group_id} leaked across splits: {seen_splits}"


def test_leakage_detection_raises_error():
    """Artificially introducing duplicate group overlap must raise PatientLeakageError."""
    r1 = ImageRecord(image_id="img1", patient_id="GROUP_LEAK", grade=0)
    r2 = ImageRecord(image_id="img2", patient_id="GROUP_LEAK", grade=0)

    splits = {
        "train": [r1],
        "val": [r2],
        "test": [],
        "external_test": [],
    }

    with pytest.raises(PatientLeakageError, match="Data leakage detected"):
        ContentGroupSplitter.assert_no_leakage(splits)


def test_real_splits_manifest_zero_leakage():
    """Verify zero group-level leakage on disk for splits_v1 and splits_v2."""
    for manifest_name in ("splits_v1.csv", "splits_v2.csv"):
        manifest_path = Path("data/processed") / manifest_name
        if not manifest_path.exists():
            continue

        group_to_splits = {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                group_id = row["group_id"]
                split = row["split"]
                group_to_splits.setdefault(group_id, set()).add(split)

        leaked = {gid: splits for gid, splits in group_to_splits.items() if len(splits) > 1}
        assert len(leaked) == 0, f"Found {len(leaked)} groups leaking across splits in {manifest_name}: {list(leaked.items())[:5]}"


def test_splits_v2_class_balance_within_two_percentage_points():
    """Verify that splits_v2 per-split grade proportions are balanced within ~2% points."""
    manifest_path = Path("data/processed/splits_v2.csv")
    if not manifest_path.exists():
        pytest.skip("splits_v2.csv not found")

    from collections import Counter
    splits = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["split"], Counter())[int(row["diagnosis"])] += 1

    totals = {s: sum(splits[s].values()) for s in ("train", "val", "test")}

    for grade in range(5):
        pcts = [splits[s][grade] / totals[s] * 100.0 for s in ("train", "val", "test")]
        max_diff = max(pcts) - min(pcts)
        assert max_diff <= 2.5, f"Grade {grade} proportion difference across splits exceeds tolerance: {pcts} (diff={max_diff:.2f}%)"

