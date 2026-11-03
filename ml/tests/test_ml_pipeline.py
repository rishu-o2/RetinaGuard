"""Tests for the ML package: architecture, metrics, and export."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from retinaguard_ml.evaluate import compute_quadratic_weighted_kappa, compute_referable_dr_metrics
from retinaguard_ml.export import (
    export_to_onnx,
    export_to_torchscript,
    is_onnx_available,
)
from retinaguard_ml.model import RetinaGuardDRNet


def test_model_forward_pass_shape() -> None:
    """Verify DRNet outputs logits of shape [batch_size, 5]."""
    model = RetinaGuardDRNet(num_classes=5)
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, 5)


def test_predict_probabilities_sum_to_one() -> None:
    """Verify calibrated softmax probabilities sum to 1.0 per sample."""
    model = RetinaGuardDRNet(num_classes=5)
    model.eval()

    dummy_input = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        probs = model.predict_probabilities(dummy_input, temperature=1.2)

    assert probs.shape == (3, 5)
    sums = probs.sum(dim=-1).numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_referable_dr_metrics_calculation() -> None:
    """Verify sensitivity and specificity for referable DR (grade >= 2)."""
    # 2 non-referable (0, 1), 2 referable (2, 4)
    y_true = np.array([0, 1, 2, 4])
    y_pred = np.array([0, 2, 3, 4])  # 1 false positive on sample idx 1

    metrics = compute_referable_dr_metrics(y_true, y_pred)
    assert metrics["sensitivity"] == 1.0  # Caught both referable cases
    assert metrics["specificity"] == 0.5  # 1 out of 2 non-referable correct
    assert metrics["accuracy"] == 0.75


def test_quadratic_weighted_kappa_perfect_score() -> None:
    """Verify QWK yields 1.0 when true and pred are identical."""
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    kappa = compute_quadratic_weighted_kappa(y, y)
    assert pytest.approx(kappa, 1e-4) == 1.0


def test_export_to_torchscript() -> None:
    """Verify PyTorch model exports to TorchScript."""
    model = RetinaGuardDRNet(num_classes=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_file = Path(tmpdir) / "test_dr_model.pt"
        exported = export_to_torchscript(model, ts_file, input_shape=(1, 3, 224, 224))
        assert exported.exists()
        assert exported.stat().st_size > 0


def test_export_to_onnx_and_verify_execution() -> None:
    """Verify ONNX export when onnx is present or proper exception when missing."""
    model = RetinaGuardDRNet(num_classes=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test_dr_model.onnx"
        if not is_onnx_available():
            with pytest.raises(RuntimeError, match="requires 'onnx' package"):
                export_to_onnx(model, onnx_file)
        else:
            exported_path = export_to_onnx(model, onnx_file, input_shape=(1, 3, 224, 224))
            assert exported_path.exists()
            assert exported_path.stat().st_size > 0
