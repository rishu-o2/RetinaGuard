"""Unit tests for the DR training pipeline, dual-head model, and clinical evaluation."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from retinaguard_ml.dataset import SyntheticFundusDataset
from retinaguard_ml.evaluate import (
    compute_per_class_recall,
    compute_referable_dr_sensitivity_at_specificity,
    evaluate_cohort,
)
from retinaguard_ml.export import (
    export_to_onnx,
    is_onnx_available,
    verify_onnx_parity,
)
from retinaguard_ml.models.efficientnet import EfficientNetDRDualHead
from retinaguard_ml.training.loss import (
    JointCategoricalOrdinalLoss,
    encode_ordinal_targets,
)
from retinaguard_ml.training.trainer import DRTrainer


def test_efficientnet_dual_head_shapes() -> None:
    """Verify dual-head outputs: [B, 5] categorical logits and [B, 4] ordinal cumulative logits."""
    model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert "logits" in out
    assert "ordinal_logits" in out
    assert out["logits"].shape == (batch_size, 5)
    assert out["ordinal_logits"].shape == (batch_size, 4)

    # Probabilities & grades
    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (batch_size, 5)
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), 1.0, atol=1e-5)

    ord_probs = model.predict_ordinal_probabilities(dummy_input)
    assert ord_probs.shape == (batch_size, 4)
    assert torch.all(ord_probs >= 0.0) and torch.all(ord_probs <= 1.0)

    grades = model.predict_grades(dummy_input, method="ordinal_sum")
    assert grades.shape == (batch_size,)
    assert torch.all(grades >= 0) and torch.all(grades <= 4)


def test_ordinal_target_encoding() -> None:
    """Verify cumulative ordinal binary target vectors for grades 0 through 4."""
    labels = torch.tensor([0, 1, 2, 3, 4])
    encoded = encode_ordinal_targets(labels, num_classes=5)

    expected = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],  # Grade 0
        [1.0, 0.0, 0.0, 0.0],  # Grade 1
        [1.0, 1.0, 0.0, 0.0],  # Grade 2
        [1.0, 1.0, 1.0, 0.0],  # Grade 3
        [1.0, 1.0, 1.0, 1.0],  # Grade 4
    ])

    torch.testing.assert_close(encoded, expected)


def test_joint_loss_computation_and_backward() -> None:
    """Verify joint loss backward pass computes gradients for both categorical and ordinal heads."""
    model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
    criterion = JointCategoricalOrdinalLoss(ordinal_weight=0.50, num_classes=5)

    dummy_input = torch.randn(3, 3, 128, 128)
    labels = torch.tensor([0, 2, 4])

    outputs = model(dummy_input)
    loss_dict = criterion(outputs, labels)

    assert "loss" in loss_dict
    assert "categorical_loss" in loss_dict
    assert "ordinal_loss" in loss_dict
    assert loss_dict["loss"].item() > 0.0

    # Test backward pass
    loss_dict["loss"].backward()
    assert model.fc_categorical.weight.grad is not None
    assert model.fc_ordinal.weight.grad is not None


def test_clinical_metrics_suite() -> None:
    """Verify QWK, sensitivity at 90%/95% specificity, per-class recall, and ROC-AUC."""
    # Synthetic ground truths and predictions
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 3, 4])

    # Dummy probabilities where referable classes (2, 3, 4) have higher scores for positive cases
    y_probs = np.zeros((8, 5))
    for i, g in enumerate(y_pred):
        y_probs[i, g] = 0.8
        y_probs[i, (g + 1) % 5] = 0.2

    metrics = evaluate_cohort(y_true, y_pred, y_probs=y_probs)

    assert "qwk" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "referable_dr_auc" in metrics
    assert "sens_at_90_spec" in metrics
    assert "sens_at_95_spec" in metrics
    assert "per_class_recall" in metrics
    assert "confusion_matrix" in metrics

    # Verify per-class recall keys
    assert set(metrics["per_class_recall"].keys()) == {0, 1, 2, 3, 4}
    # Grade 0 was 100% accurate (both 0s predicted as 0)
    assert metrics["per_class_recall"][0] == 1.0


def test_trainer_lifecycle_and_checkpointing() -> None:
    """Verify trainer executes train and evaluation epochs, saving checkpoints properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "training": {
                "mixed_precision": False,
                "gradient_clip_val": 1.0,
                "early_stopping_patience": 2,
                "save_dir": str(Path(tmpdir) / "checkpoints"),
            },
            "logging": {
                "logger_type": "file",
                "log_dir": str(Path(tmpdir) / "logs"),
                "experiment_name": "test_run",
            },
        }

        model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
        criterion = JointCategoricalOrdinalLoss(ordinal_weight=0.50)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = DRTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        train_ds = SyntheticFundusDataset(num_samples=6, image_size=(64, 64))
        val_ds = SyntheticFundusDataset(num_samples=4, image_size=(64, 64))

        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)

        # Run 2 epochs
        results = trainer.fit(train_loader, val_loader, num_epochs=2)

        assert "best_qwk" in results
        assert len(results["history"]) == 2

        # Verify checkpoints saved
        chkpt_dir = Path(tmpdir) / "checkpoints"
        assert (chkpt_dir / "latest_checkpoint.pt").exists()
        assert (chkpt_dir / "best_model.pt").exists()


def test_onnx_export_and_parity_check() -> None:
    """Verify ONNX export and parity tolerance verification."""
    model = EfficientNetDRDualHead(backbone_name="efficientnet_b0", pretrained=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "dualhead_dr.onnx"
        if not is_onnx_available():
            with pytest.raises(RuntimeError):
                export_to_onnx(model, onnx_file, input_shape=(1, 3, 128, 128))
        else:
            export_to_onnx(model, onnx_file, input_shape=(1, 3, 128, 128))
            assert onnx_file.exists()

            # Parity check
            diffs = verify_onnx_parity(model, onnx_file, input_shape=(1, 3, 128, 128), atol=1e-3)
            assert "logits" in diffs
            assert "ordinal_logits" in diffs
            assert diffs["logits"] < 1e-3


def test_dataloader_uses_plain_shuffled_sampler_when_weighted_sampler_false() -> None:
    """Verify DataLoader uses a plain shuffled sampler (RandomSampler) when
    use_weighted_sampler is False, and WeightedRandomSampler when True.
    """
    from torch.utils.data import RandomSampler, WeightedRandomSampler
    from ml.scripts.train import build_train_loader

    class MockDataset(SyntheticFundusDataset):
        def get_weighted_sampler(self) -> WeightedRandomSampler:
            weights = torch.ones(len(self))
            return WeightedRandomSampler(weights, len(weights))

    ds = MockDataset(num_samples=10, image_size=(64, 64))

    # When False: must use plain RandomSampler (shuffled)
    loader_plain = build_train_loader(ds, batch_size=2, use_weighted_sampler=False)
    assert isinstance(loader_plain.sampler, RandomSampler), (
        f"Expected RandomSampler when use_weighted_sampler=False, got {type(loader_plain.sampler)}"
    )
    assert not isinstance(loader_plain.sampler, WeightedRandomSampler)

    # When True: must use WeightedRandomSampler
    loader_weighted = build_train_loader(ds, batch_size=2, use_weighted_sampler=True)
    assert isinstance(loader_weighted.sampler, WeightedRandomSampler), (
        f"Expected WeightedRandomSampler when use_weighted_sampler=True, got {type(loader_weighted.sampler)}"
    )
