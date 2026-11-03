"""Comprehensive test suite for ML Data Pipeline & India Domain-Shift Suite."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from retinaguard_ml.data.augmentations import (
    FundusClinicalAugmentor,
    SmartphoneDegradationPipeline,
)
from retinaguard_ml.data.imbalance import (
    FocalLoss,
    compute_class_weights,
    create_weighted_sampler,
)
from retinaguard_ml.data.ingestion import (
    ClinicalIntakeRecord,
    LocalClinicalDatasetIngestor,
    MultiGraderAnnotation,
)
from retinaguard_ml.data.preprocessing import (
    FundusPreprocessor,
    ben_graham_enhancement,
    crop_black_borders,
)
from retinaguard_ml.data.registry import (
    DATASET_REGISTRY,
    ImageRecord,
    get_dataset_metadata,
)
from retinaguard_ml.data.reporting import DataReporter
from retinaguard_ml.data.split import (
    PatientLeakageError,
    PatientStratifiedSplitter,
)
from retinaguard_ml.evaluate import evaluate_per_device_and_site


def test_crop_black_borders_trims_background() -> None:
    """Verify black border cropping tightens the bounding box around the fundus circle."""
    # Synthetic 600x600 image with 100px black padding around a 400x400 bright region
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    img[100:500, 100:500] = 180  # Non-zero retina area

    cropped = crop_black_borders(img, tol=10)
    assert cropped.shape[0] < 600
    assert cropped.shape[1] < 600
    assert cropped.shape[0] >= 390
    assert cropped.shape[1] >= 390


def test_preprocessor_shapes_and_ben_graham() -> None:
    """Verify preprocessor produces specified target size and valid normalized tensor."""
    preprocessor = FundusPreprocessor(target_size=(512, 512), use_ben_graham=True)

    raw_img = np.random.randint(20, 240, size=(700, 800, 3), dtype=np.uint8)
    processed = preprocessor.process(raw_img)
    assert processed.shape == (512, 512, 3)
    assert processed.dtype == np.uint8

    tensor = preprocessor.to_normalized_tensor(raw_img)
    assert tensor.shape == (1, 3, 512, 512)
    assert isinstance(tensor, torch.Tensor)


def test_patient_stratified_split_zero_leakage() -> None:
    """Verify strict patient-level separation across train/val/test splits."""
    # Create 30 patients, each having 2 images (OD and OS)
    records: list[ImageRecord] = []
    for p_idx in range(30):
        pid = f"PATIENT_{p_idx:03d}"
        grade = p_idx % 5  # ICDR 0 to 4
        records.append(
            ImageRecord(
                image_id=f"{pid}_OD",
                patient_id=pid,
                grade=grade,
                eye="OD",
                device="Tabletop_Zeiss",
                site="AIIMS_Delhi",
            )
        )
        records.append(
            ImageRecord(
                image_id=f"{pid}_OS",
                patient_id=pid,
                grade=grade,
                eye="OS",
                device="Tabletop_Zeiss",
                site="AIIMS_Delhi",
            )
        )

    splitter = PatientStratifiedSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    splits = splitter.split(records)

    train_patients = {r.patient_id for r in splits["train"]}
    val_patients = {r.patient_id for r in splits["val"]}
    test_patients = {r.patient_id for r in splits["test"]}

    # Assert zero patient intersection
    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0

    # Total patients preserved
    all_patients = train_patients | val_patients | test_patients
    assert len(all_patients) == 30


def test_leakage_assertion_catches_overlap() -> None:
    """Verify PatientLeakageError is raised if an overlapping patient is injected."""
    rec1 = ImageRecord(image_id="1", patient_id="LEAK_PATIENT", grade=0)
    rec2 = ImageRecord(image_id="2", patient_id="LEAK_PATIENT", grade=0)

    compromised_splits = {
        "train": [rec1],
        "val": [rec2],
        "test": [],
    }

    with pytest.raises(PatientLeakageError, match="Data leakage detected"):
        PatientStratifiedSplitter.assert_no_leakage(compromised_splits)


def test_dataset_registry_and_licensing() -> None:
    """Verify all 5 datasets exist in registry with valid licensing information."""
    expected_keys = ["aptos2019", "idrid", "eyepacs", "messidor2", "ddr"]
    for key in expected_keys:
        meta = get_dataset_metadata(key)
        assert meta.license != ""
        assert meta.citation != ""
        assert meta.default_device != ""
        assert meta.default_site != ""

    # Verify IDRiD is strictly locked as an external test set
    idrid_meta = get_dataset_metadata("idrid")
    assert idrid_meta.is_external_test_only is True


def test_smartphone_degradation_pipeline() -> None:
    """Verify smartphone simulation pipeline degrades images while preserving dimensions."""
    degrader = SmartphoneDegradationPipeline(p=1.0, seed=123)
    img = np.full((512, 512, 3), 150, dtype=np.uint8)

    degraded = degrader(img)
    assert degraded.shape == (512, 512, 3)
    assert degraded.dtype == np.uint8
    assert not np.array_equal(degraded, img), "Degradation should modify image pixels"


def test_class_imbalance_weights_and_focal_loss() -> None:
    """Verify class weights appropriately prioritize underrepresented severe DR classes."""
    # Highly skewed distribution: mostly grade 0, very few grade 4
    labels = [0] * 100 + [1] * 20 + [2] * 10 + [3] * 5 + [4] * 2
    weights = compute_class_weights(labels, num_classes=5, method="effective_samples")

    assert len(weights) == 5
    assert weights[4] > weights[0], "Rare Class 4 weight must be higher than common Class 0"

    sampler = create_weighted_sampler(labels, num_classes=5)
    assert len(sampler) == len(labels)

    # Test FocalLoss forward pass
    focal = FocalLoss(alpha=weights, gamma=2.0)
    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 2, 4, 1])
    loss = focal(logits, targets)
    assert loss.item() > 0.0


def test_multi_grader_ingestion_and_adjudication() -> None:
    """Verify adjudication logic handles majority vote and senior specialist overrides."""
    # Case 1: Majority vote (two graders say 2, one says 1)
    rec1 = ClinicalIntakeRecord(
        screening_id="SCR-001",
        patient_id="PAT-001",
        eye="OD",
        device="Remidio NM-FOP",
        site="PHC_Koppal",
        file_path="/fake/path1.jpg",
        grader_annotations=[
            MultiGraderAnnotation(grader_id="G1", experience_level="RESIDENT", grade=2),
            MultiGraderAnnotation(grader_id="G2", experience_level="FELLOW", grade=2),
            MultiGraderAnnotation(grader_id="G3", experience_level="RESIDENT", grade=1),
        ],
    )
    grade1 = LocalClinicalDatasetIngestor.adjudicate(rec1)
    assert grade1 == 2
    assert rec1.adjudication_method == "MAJORITY_VOTE"

    # Case 2: Senior specialist override
    rec2 = ClinicalIntakeRecord(
        screening_id="SCR-002",
        patient_id="PAT-002",
        eye="OS",
        device="Forus 3nethra",
        site="Camp_Madurai",
        file_path="/fake/path2.jpg",
        grader_annotations=[
            MultiGraderAnnotation(grader_id="G1", experience_level="RESIDENT", grade=1),
            MultiGraderAnnotation(grader_id="G2", experience_level="RESIDENT", grade=1),
        ],
        senior_adjudicator_grade=3,
    )
    grade2 = LocalClinicalDatasetIngestor.adjudicate(rec2)
    assert grade2 == 3
    assert rec2.adjudication_method == "SENIOR_SPECIALIST_OVERRIDE"

    # Batch standardization
    standardized = LocalClinicalDatasetIngestor.ingest_records([rec1, rec2])
    assert len(standardized) == 2
    assert standardized[0].grade == 2
    assert standardized[0].device == "Remidio NM-FOP"
    assert standardized[1].grade == 3
    assert standardized[1].device == "Forus 3nethra"


def test_evaluate_per_device_and_site() -> None:
    """Verify performance metrics are stratified per camera hardware and screening site."""
    records = [
        ImageRecord(image_id="1", patient_id="P1", grade=0, device="Tabletop_Zeiss", site="AIIMS_Delhi"),
        ImageRecord(image_id="2", patient_id="P2", grade=2, device="Tabletop_Zeiss", site="AIIMS_Delhi"),
        ImageRecord(image_id="3", patient_id="P3", grade=0, device="Remidio_FOP", site="PHC_Koppal"),
        ImageRecord(image_id="4", patient_id="P4", grade=3, device="Remidio_FOP", site="PHC_Koppal"),
    ]
    y_true = np.array([0, 2, 0, 3])
    y_pred = np.array([0, 2, 1, 3])

    res = evaluate_per_device_and_site(records, y_true, y_pred)

    assert "overall" in res
    assert "by_device" in res
    assert "by_site" in res

    assert "Tabletop_Zeiss" in res["by_device"]
    assert "Remidio_FOP" in res["by_device"]
    assert "AIIMS_Delhi" in res["by_site"]
    assert "PHC_Koppal" in res["by_site"]

    # Tabletop had 100% accuracy in this synthetic setup
    assert res["by_device"]["Tabletop_Zeiss"]["accuracy"] == 1.0


def test_data_reporter_generates_markdown() -> None:
    """Verify DataReporter generates a comprehensive markdown cohort summary."""
    records = [
        ImageRecord(image_id="1", patient_id="P1", grade=0, device="Zeiss", site="SiteA", split="train"),
        ImageRecord(image_id="2", patient_id="P2", grade=2, device="Remidio", site="SiteB", split="val"),
        ImageRecord(image_id="3", patient_id="P3", grade=4, device="Forus", site="SiteC", split="test"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = DataReporter.generate_report(records, output_dir=tmpdir)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Total Fundus Images" in content
        assert "ICDR Grade Distribution" in content
        assert "Imaging Device Breakdown" in content
        assert "Clinical Screening Sites" in content

