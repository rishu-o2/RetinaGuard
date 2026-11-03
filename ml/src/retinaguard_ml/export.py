"""Export PyTorch models to ONNX format with strict numerical parity verification."""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

try:
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def is_onnx_available() -> bool:
    """Return True if ONNX export toolchain is available in current environment."""
    return HAS_ONNX


class ExportParityError(Exception):
    """Raised when ONNX runtime outputs deviate from PyTorch reference outputs."""
    pass


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path | str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    opset_version: int = 17,
) -> Path:
    """Export PyTorch DR model to ONNX and verify graph execution with ONNX Runtime."""
    if not HAS_ONNX:
        raise RuntimeError("ONNX exporter requires 'onnx' package installed.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape, requires_grad=False)

    # Determine output names based on module return type (dict vs single tensor)
    with torch.no_grad():
        sample_out = model(dummy_input)

    if isinstance(sample_out, dict):
        output_names = list(sample_out.keys())
        dynamic_axes = {"input": {0: "batch_size"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
    else:
        output_names = ["logits"]
        dynamic_axes = {"input": {0: "batch_size"}, "logits": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    return output_path


def verify_onnx_parity(
    model: torch.nn.Module,
    onnx_path: Path | str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, float]:
    """Verify that ONNX Runtime matches PyTorch outputs within specified numerical tolerance."""
    if not HAS_ONNX:
        raise RuntimeError("ONNX runtime verification requires 'onnxruntime' installed.")

    model.eval()
    dummy_input = torch.randn(*input_shape, requires_grad=False)

    # 1. PyTorch inference
    with torch.no_grad():
        pt_out = model(dummy_input)

    # Normalize to dict of numpy arrays
    pt_arrays: Dict[str, np.ndarray] = {}
    if isinstance(pt_out, dict):
        for k, v in pt_out.items():
            pt_arrays[k] = v.detach().cpu().numpy()
    else:
        pt_arrays["logits"] = pt_out.detach().cpu().numpy()

    # 2. ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)

    # Map ONNX outputs to output names
    ort_arrays: Dict[str, np.ndarray] = {}
    for meta, out_arr in zip(session.get_outputs(), ort_outs):
        ort_arrays[meta.name] = out_arr

    # 3. Numerical parity checks
    max_abs_diffs: Dict[str, float] = {}
    for name, pt_arr in pt_arrays.items():
        if name not in ort_arrays:
            raise ExportParityError(f"Output '{name}' missing from ONNX model outputs.")

        ort_arr = ort_arrays[name]
        diff = float(np.max(np.abs(pt_arr - ort_arr)))
        max_abs_diffs[name] = diff

        try:
            np.testing.assert_allclose(
                pt_arr, ort_arr, rtol=rtol, atol=atol, err_msg=f"Parity mismatch in head '{name}'"
            )
        except AssertionError as e:
            raise ExportParityError(f"Parity verification failed for output '{name}': {e}") from e

    return max_abs_diffs


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: Path | str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
) -> Path:
    """Export PyTorch model to TorchScript archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(output_path))
    return output_path
