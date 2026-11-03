"""Vision Agent for RetinaGuard.

Pipeline:
  1. Quality Gate (Laplacian blur, brightness, resolution, circular FOV)
  2. OOD Check (Optical fundus spectrum R>G>B, deep MSP/energy)
  3. ONNX Model Inference (ICDR grade 0-4 + calibrated confidence)

If any check fails or confidence is low, flags early_exit=True and is_uncertain=True.
"""

from __future__ import annotations

from typing import Optional

from retinaguard.agents.state import ScreeningAgentState
from retinaguard.core.logging import get_logger
from retinaguard.services.inference import ONNXInferenceService
from retinaguard.services.ood import OODDetectorService
from retinaguard.services.quality import QualityGateService

logger = get_logger("retinaguard.agents.vision")


class VisionAgent:
    """Agent encapsulating image decoding, quality gating, OOD detection, and model inference."""

    def __init__(
        self,
        quality_service: Optional[QualityGateService] = None,
        ood_service: Optional[OODDetectorService] = None,
        inference_service: Optional[ONNXInferenceService] = None,
        min_confidence_threshold: float = 0.50,
    ) -> None:
        self.quality_service = quality_service or QualityGateService()
        self.ood_service = ood_service or OODDetectorService()
        self.inference_service = inference_service or ONNXInferenceService()
        self.min_confidence_threshold = min_confidence_threshold

    def __call__(self, state: ScreeningAgentState) -> dict:
        """Run the vision pipeline on the image in state."""
        img_rgb = state.get("image_rgb")
        metadata = state["image_metadata"]
        uncertainty_reasons: list[str] = list(state.get("uncertainty_reasons", []))

        if img_rgb is None:
            uncertainty_reasons.append("Image could not be decoded as valid RGB data.")
            return {
                "is_uncertain": True,
                "uncertainty_reasons": uncertainty_reasons,
                "early_exit": True,
            }

        # 1. Quality Gate Check
        quality_res = self.quality_service.evaluate(img_rgb, camera_type=metadata.camera_type)
        if not quality_res.passed:
            failure_str = quality_res.human_readable_reason or "Quality below clinical threshold"
            uncertainty_reasons.append(failure_str)
            logger.info("VisionAgent: Quality gate failed: %s", failure_str)
            return {
                "quality_result": quality_res,
                "is_uncertain": True,
                "uncertainty_reasons": uncertainty_reasons,
                "early_exit": True,
            }

        # 2. Out-of-Distribution (OOD) Check
        ood_res = self.ood_service.evaluate(img_rgb)
        if not ood_res.is_in_distribution:
            failure_str = ood_res.failure_reason_en or "Image not consistent with human retinal fundus"
            uncertainty_reasons.append(failure_str)
            logger.info("VisionAgent: OOD gate triggered: %s", failure_str)
            return {
                "quality_result": quality_res,
                "ood_result": ood_res,
                "is_uncertain": True,
                "uncertainty_reasons": uncertainty_reasons,
                "early_exit": True,
            }

        # 3. ONNX Model Inference
        inference_out = self.inference_service.predict(img_rgb)
        if inference_out is None:
            uncertainty_reasons.append("Inference model unavailable; falling back to UNCERTAIN.")
            return {
                "quality_result": quality_res,
                "ood_result": ood_res,
                "inference_output": None,
                "is_uncertain": True,
                "uncertainty_reasons": uncertainty_reasons,
                "early_exit": True,
            }

        # Low confidence check
        if inference_out.confidence < self.min_confidence_threshold:
            uncertainty_reasons.append(
                f"Model confidence ({inference_out.confidence:.2f}) below threshold ({self.min_confidence_threshold:.2f})."
            )
            return {
                "quality_result": quality_res,
                "ood_result": ood_res,
                "inference_output": inference_out,
                "is_uncertain": True,
                "uncertainty_reasons": uncertainty_reasons,
                "early_exit": True,
            }

        # Vision pipeline passed successfully
        return {
            "quality_result": quality_res,
            "ood_result": ood_res,
            "inference_output": inference_out,
            "is_uncertain": False,
            "early_exit": False,
        }
