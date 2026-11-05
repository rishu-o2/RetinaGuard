"""ONNX Inference Service for DR Grading.

Loads the exported ONNX model and runs inference on preprocessed fundus images.
Falls back gracefully when no model is available (returns UNCERTAIN).

ADR-003: ML model strictly produces grade + confidence; referral is the rules engine's job.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from retinaguard.core.config import _find_repo_root
from retinaguard.core.logging import get_logger
from retinaguard.core.models import ICDRGrade, InferenceOutput

logger = get_logger("retinaguard.services.inference")

# Standard fundus preprocessing constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEFAULT_SIZE = 512


def _is_onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def preprocess_image_for_inference(img_rgb: np.ndarray, size: int = _DEFAULT_SIZE) -> np.ndarray:
    """Resize and normalise a fundus image to the model's expected input format.

    Args:
        img_rgb: uint8 (H, W, 3) RGB array.
        size: Target spatial resolution for the model.

    Returns:
        float32 (1, 3, size, size) NCHW array ready for ONNX inference.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for inference preprocessing") from exc

    resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    # HWC -> NCHW
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis]
    return arr


@dataclass
class _NoModelAvailable:
    """Sentinel returned when the ONNX model file does not exist."""
    reason: str


class ONNXInferenceService:
    """Loads and runs the exported ONNX DR-grading model.

    The service intentionally fails *open* (returns UNCERTAIN rather than
    crashing) so the rules engine can still produce a safe outcome.
    """

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        if model_path is None:
            model_path = self._resolve_model_path()
        self.model_path = Path(model_path)
        self._session = None  # lazy-loaded
        self.temperature = 1.0
        self.image_size = _DEFAULT_SIZE
        self._load_registry_metadata()

        if not self.model_path.exists():
            logger.warning(
                "ONNX model not found at %s. Inference will return UNCERTAIN until the model is exported.",
                self.model_path,
            )

    def _load_registry_metadata(self) -> None:
        """Load temperature and image size from registry if available."""
        registry_path = _find_repo_root() / "models" / "registry.json"
        if registry_path.exists():
            try:
                import json
                with open(registry_path, encoding="utf-8") as f:
                    reg = json.load(f)
                entries = reg.get("models", [])
                if entries:
                    entry = entries[0]
                    self.temperature = float(entry.get("temperature", 1.0))
                    sz = entry.get("image_size", _DEFAULT_SIZE)
                    if isinstance(sz, (list, tuple)) and len(sz) > 0:
                        self.image_size = int(sz[0])
                    elif isinstance(sz, int):
                        self.image_size = sz
                    logger.info("Loaded registry metadata: T=%.4f, size=%d", self.temperature, self.image_size)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read registry metadata: %s", exc)

    @staticmethod
    def _resolve_model_path() -> Path:
        """Resolve ONNX model path: prefer models/registry.json entry, else legacy path."""
        repo_root = _find_repo_root()
        registry_path = repo_root / "models" / "registry.json"
        if registry_path.exists():
            try:
                import json
                with open(registry_path, encoding="utf-8") as f:
                    reg = json.load(f)
                entries = reg.get("models", [])
                if entries:
                    onnx_rel = entries[0].get("onnx_path", "")
                    if onnx_rel and onnx_rel != "SKIPPED":
                        candidate = repo_root / onnx_rel
                        if candidate.exists():
                            logger.info("Using ONNX model from registry: %s", candidate)
                            return candidate
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read models/registry.json: %s", exc)
        # Legacy fallback
        return repo_root / "ml" / "exports" / "retinaguard_dr.onnx"

    def _load_session(self):
        """Lazy-load the ONNX Runtime InferenceSession."""
        if self._session is not None:
            return self._session
        if not _is_onnxruntime_available():
            raise RuntimeError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            )
        import onnxruntime as ort  # type: ignore
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        try:
            shape = self._session.get_inputs()[0].shape
            if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
                self.image_size = shape[2]
        except Exception:
            pass
        logger.info("ONNX model loaded from %s (input size: %d)", self.model_path, self.image_size)
        return self._session

    @property
    def is_available(self) -> bool:
        """True if the model file exists and onnxruntime is installed."""
        return self.model_path.exists() and _is_onnxruntime_available()

    def predict(self, img_rgb: np.ndarray) -> Optional[InferenceOutput]:
        """Run inference on a preprocessed fundus image.

        Args:
            img_rgb: uint8 (H, W, 3) RGB array. Preprocessing is applied internally.

        Returns:
            InferenceOutput with predicted_grade and calibrated probabilities,
            or None if the model is unavailable.
        """
        if not self.model_path.exists():
            logger.warning("ONNX model unavailable — returning None (UNCERTAIN pathway).")
            return None

        try:
            session = self._load_session()
        except RuntimeError as exc:
            logger.error("Failed to load ONNX session: %s", exc)
            return None

        input_tensor = preprocess_image_for_inference(img_rgb, size=self.image_size)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        # Model outputs: [logits_categorical (1,5), logits_ordinal (1,4)]
        # Use temperature scaling on categorical logits
        cat_logits = outputs[0][0] / max(self.temperature, 1e-4)
        exp_l = np.exp(cat_logits - np.max(cat_logits))
        probs = exp_l / exp_l.sum()

        # Grade prediction: use ordinal sum if available, else argmax
        if len(outputs) > 1 and outputs[1].shape[-1] == 4:
            ord_logits = outputs[1][0]
            ord_probs = 1.0 / (1.0 + np.exp(-ord_logits))
            predicted_idx = int(np.clip(np.round(np.sum(ord_probs)), 0, 4))
        else:
            predicted_idx = int(np.argmax(probs))

        confidence = float(probs[predicted_idx])

        return InferenceOutput(
            predicted_grade=ICDRGrade(predicted_idx),
            confidence=confidence,
            raw_probabilities={i: float(probs[i]) for i in range(5)},
        )

