"""Unit tests for IDRiD ingestion pipeline, verification, and external-test guard.

Tests:
1. Synthetic IDRiD dataset ingestion (valid case).
2. Missing image handling & detection.
3. Invalid DR grade rejection (out of {0, 1, 2, 3, 4}).
4. Name mismatch reporting between CSV and disk images.
5. External-test guard verification: raises error if external_test_only dataset
   is assigned to train or val split, or passed to DRTrainer.
"""

import csv
from pathlib import Path
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from retinaguard_ml.data.registry import ImageRecord
from retinaguard_ml.data.split import ContentGroupSplitter, ExternalHoldoutContaminationError
from retinaguard_ml.training.loss import JointCategoricalOrdinalLoss
from retinaguard_ml.training.trainer import DRTrainer


def _create_synthetic_idrid_dir(
    base_dir: Path,
    num_train: int = 5,
    num_test: int = 3,
    introduce_missing_image: bool = False,
    introduce_bad_grade: bool = False,
    introduce_name_mismatch: bool = False,
) -> Path:
    """Create a minimal synthetic IDRiD Part B directory structure for testing."""
    root = base_dir / "B. Disease Grading"
    gt_dir = root / "2. Groundtruths"
    train_img_dir = root / "1. Original Images" / "a. Training Set"
    test_img_dir = root / "1. Original Images" / "b. Testing Set"

    for d in (gt_dir, train_img_dir, test_img_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Training Set
    train_rows = [["Image name", "Retinopathy grade", "Risk of macular edema "]]
    for i in range(1, num_train + 1):
        name = f"IDRiD_{i:03d}"
        grade = 5 if (introduce_bad_grade and i == 1) else (i % 5)
        dme = 0
        train_rows.append([name, str(grade), str(dme)])

        # Create tiny synthetic image
        if not (introduce_missing_image and i == 2):
            img_file_name = f"IDRiD_MISMATCH_{i:03d}.jpg" if (introduce_name_mismatch and i == 3) else f"{name}.jpg"
            img = Image.new("RGB", (64, 64), color=(100, 30, 20))
            img.save(train_img_dir / img_file_name)

    train_csv = gt_dir / "a. IDRiD_Disease Grading_Training Labels.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    # 2. Testing Set
    test_rows = [["Image name", "Retinopathy grade", "Risk of macular edema "]]
    for i in range(1, num_test + 1):
        name = f"IDRiD_{i:03d}"
        test_rows.append([name, str(i % 5), "0"])
        img = Image.new("RGB", (64, 64), color=(120, 40, 20))
        img.save(test_img_dir / f"{name}.jpg")

    test_csv = gt_dir / "b. IDRiD_Disease Grading_Testing Labels.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(test_rows)

    return base_dir


def test_idrid_ingest_and_verify_valid(tmp_path: Path):
    """Test standard valid ingestion and verification."""
    from ml.scripts.prepare_idrid import ingest_idrid
    from ml.scripts.verify_idrid import verify_idrid

    raw_dir = _create_synthetic_idrid_dir(tmp_path / "raw")
    manifest_csv = tmp_path / "processed" / "idrid_manifest.csv"

    records = ingest_idrid(raw_dir, manifest_csv)
    assert len(records) == 8  # 5 train + 3 test
    assert manifest_csv.exists()

    # Run verification script function
    verify_idrid(manifest_csv, sample_size=5)


def test_idrid_ingest_name_mismatch_detected(tmp_path: Path, capsys):
    """Test that mismatched image names are flagged in prepare_idrid."""
    from ml.scripts.prepare_idrid import ingest_idrid

    raw_dir = _create_synthetic_idrid_dir(tmp_path / "mismatch", introduce_name_mismatch=True)
    manifest_csv = tmp_path / "processed" / "mismatch_manifest.csv"

    records = ingest_idrid(raw_dir, manifest_csv)
    # 1 mismatched image is excluded from manifest
    assert len(records) == 7
    captured = capsys.readouterr()
    assert "unmatched rows in groundtruth CSVs" in captured.out


def test_idrid_verify_fails_on_missing_image(tmp_path: Path):
    """Test that verify_idrid exits non-zero if an image is missing on disk."""
    from ml.scripts.prepare_idrid import ingest_idrid
    from ml.scripts.verify_idrid import verify_idrid

    raw_dir = _create_synthetic_idrid_dir(tmp_path / "missing", introduce_missing_image=True)
    manifest_csv = tmp_path / "processed" / "missing_manifest.csv"
    ingest_idrid(raw_dir, manifest_csv)

    # Manually delete one image pointed to in manifest to test disk verification failure
    with open(manifest_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    del_path = Path(rows[0]["file_path"])
    if del_path.exists():
        del_path.unlink()

    with pytest.raises(SystemExit):
        verify_idrid(manifest_csv)


def test_idrid_verify_fails_on_bad_grade(tmp_path: Path):
    """Test that verify_idrid exits non-zero on an out-of-range grade."""
    from ml.scripts.prepare_idrid import ingest_idrid
    from ml.scripts.verify_idrid import verify_idrid

    raw_dir = _create_synthetic_idrid_dir(tmp_path / "bad_grade", introduce_bad_grade=True)
    manifest_csv = tmp_path / "processed" / "bad_manifest.csv"
    ingest_idrid(raw_dir, manifest_csv)

    with pytest.raises(SystemExit):
        verify_idrid(manifest_csv)


def test_external_test_guard_in_split():
    """Verify that ContentGroupSplitter raises ExternalHoldoutContaminationError if external_test is put in train/val."""
    r_external = ImageRecord(
        image_id="IDRiD_001",
        patient_id="IDRiD_001",
        grade=2,
        dataset_name="idrid",
        split="train",  # illegal attempt to put external_test_only in train
    )

    splitter = ContentGroupSplitter()
    with pytest.raises(ExternalHoldoutContaminationError, match="external_test_only"):
        splitter.split([r_external])


def test_external_test_guard_in_trainer():
    """Verify that DRTrainer raises ExternalHoldoutContaminationError if dataloader has external dataset."""
    class DummyExternalDataset(Dataset):
        dataset_name = "idrid"

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return torch.zeros(3, 64, 64), 0

    dummy_ds = DummyExternalDataset()
    loader = DataLoader(dummy_ds, batch_size=2)

    model = torch.nn.Linear(10, 5)
    criterion = JointCategoricalOrdinalLoss(num_classes=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = DRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    with pytest.raises(ExternalHoldoutContaminationError, match="external_test_only"):
        trainer.train_epoch(loader)

