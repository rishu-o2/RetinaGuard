"""Tests for APTOS 2019 dataset verification logic using synthetic test fixtures."""

import csv
import tempfile
from pathlib import Path
from PIL import Image
import pytest

from ml.scripts.verify_aptos import verify_aptos_dataset


def create_mock_aptos_dataset(
    base_dir: Path,
    num_samples: int = 5,
    include_images: bool = True,
    invalid_diagnosis: bool = False,
    corrupt_image: bool = False,
    missing_one_image: bool = False,
) -> Path:
    """Helper to generate a miniature mock APTOS dataset for testing."""
    train_images_dir = base_dir / "train_images"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_csv = base_dir / "train.csv"

    rows = []
    for i in range(num_samples):
        id_code = f"mock_{i:04d}"
        diagnosis = i % 5

        if invalid_diagnosis and i == 0:
            diagnosis = 99  # Invalid grade outside {0, 1, 2, 3, 4}

        rows.append({"id_code": id_code, "diagnosis": diagnosis})

        if include_images:
            img_path = train_images_dir / f"{id_code}.png"
            if missing_one_image and i == (num_samples - 1):
                continue  # Intentionally skip creating this image

            if corrupt_image and i == 0:
                # Write corrupt non-image bytes
                img_path.write_bytes(b"CORRUPTED_NOT_AN_IMAGE_DATA_123456")
            else:
                # Create a valid minimal 32x32 PNG
                img = Image.new("RGB", (32, 32), color=(180, 80, 20))
                img.save(img_path, format="PNG")

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(rows)

    return base_dir


def test_verify_valid_dataset() -> None:
    """Verify that a correctly formatted APTOS dataset passes all checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_mock_aptos_dataset(Path(tmpdir), num_samples=5)
        res = verify_aptos_dataset(dataset_path, sample_size=5)

        assert res.is_valid is True
        assert res.total_images_in_csv == 5
        assert res.total_image_files_on_disk == 5
        assert len(res.missing_image_files) == 0
        assert len(res.invalid_diagnosis_rows) == 0
        assert len(res.corrupted_images) == 0
        assert len(res.errors) == 0

        # Class counts should have 1 sample each for grades 0-4
        for g in range(5):
            assert res.class_distribution[g] == 1


def test_verify_missing_image_file() -> None:
    """Verify that an image listed in train.csv but missing on disk fails validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_mock_aptos_dataset(Path(tmpdir), num_samples=5, missing_one_image=True)
        res = verify_aptos_dataset(dataset_path, sample_size=5)

        assert res.is_valid is False
        assert len(res.missing_image_files) == 1
        assert "mock_0004" in res.missing_image_files[0]
        assert any("Missing 1 image files" in err for err in res.errors)


def test_verify_invalid_diagnosis_label() -> None:
    """Verify that diagnosis values outside {0, 1, 2, 3, 4} fail validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_mock_aptos_dataset(Path(tmpdir), num_samples=5, invalid_diagnosis=True)
        res = verify_aptos_dataset(dataset_path, sample_size=5)

        assert res.is_valid is False
        assert len(res.invalid_diagnosis_rows) == 1
        assert res.invalid_diagnosis_rows[0]["diagnosis"] == "99"
        assert any("invalid diagnosis" in err.lower() for err in res.errors)


def test_verify_corrupted_image() -> None:
    """Verify that unopenable/corrupt image files fail validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_mock_aptos_dataset(Path(tmpdir), num_samples=5, corrupt_image=True)
        res = verify_aptos_dataset(dataset_path, sample_size=5)

        assert res.is_valid is False
        assert len(res.corrupted_images) >= 1
        assert "mock_0000.png" in res.corrupted_images[0]
        assert any("corrupt" in err.lower() for err in res.errors)


def test_verify_missing_train_csv() -> None:
    """Verify that a directory missing train.csv fails immediately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = verify_aptos_dataset(Path(tmpdir))
        assert res.is_valid is False
        assert any("missing in" in err.lower() for err in res.errors)

