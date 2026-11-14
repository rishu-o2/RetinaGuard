"""Config endpoints — exposes YAML-driven thresholds to the frontend.

GET /config/precheck?camera_type=tabletop
  Returns the browser pre-check thresholds so the frontend never duplicates values.
  camera_type must be one of the profile keys in configs/quality_thresholds.yaml.
"""
from __future__ import annotations

from typing import Literal

import yaml
from fastapi import APIRouter, HTTPException, Query, status

from retinaguard.core.config import _find_repo_root
from retinaguard.core.logging import get_logger

logger = get_logger("retinaguard.api.config")

router = APIRouter(prefix="/config", tags=["config"])

_THRESHOLDS_PATH = _find_repo_root() / "configs" / "quality_thresholds.yaml"

_PROFILE_ALIASES: dict[str, str] = {
    "tabletop": "tabletop",
    "TABLETOP": "tabletop",
    "smartphone_portable": "smartphone_portable",
    "SMARTPHONE_PORTABLE": "smartphone_portable",
}


def _load_profile(camera_type: str) -> dict:
    """Load a named quality profile from the YAML file."""
    if camera_type not in _PROFILE_ALIASES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown camera_type '{camera_type}'. Valid: tabletop, smartphone_portable",
        )
    key = _PROFILE_ALIASES[camera_type]
    try:
        with open(_THRESHOLDS_PATH, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quality thresholds configuration not found",
        ) from exc

    profiles = data.get("profiles", {})
    if key not in profiles:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile '{key}' missing from quality_thresholds.yaml",
        )
    return profiles[key]


@router.get("/precheck", summary="Browser pre-check thresholds for a camera type")
def get_precheck_thresholds(
    camera_type: str = Query(
        default="tabletop",
        description="Camera profile: tabletop or smartphone_portable",
    ),
) -> dict:
    """Return the quality thresholds the browser pre-check should use.

    All values come from configs/quality_thresholds.yaml — the frontend must
    never hard-code these numbers.

    Fields returned:
    - min_width / min_height: minimum pixel dimensions
    - blur_variance_threshold: minimum Laplacian variance (grayscale)
    - mean_luminance_min / max: acceptable brightness range (0-255)
    - non_black_pixel_min_ratio: minimum share of non-black pixels (derived
      from fov_min_area_ratio in the YAML)
    - red_dominance_required: bool — mean R > G > B expected (retinal fundus)
    """
    profile = _load_profile(camera_type)
    return {
        "camera_type": camera_type,
        "min_width": profile.get("min_resolution_width", 256),
        "min_height": profile.get("min_resolution_height", 256),
        "blur_variance_threshold": profile.get("blur_variance_threshold", 8.0),
        "mean_luminance_min": profile.get("mean_luminance_min", 30.0),
        "mean_luminance_max": profile.get("mean_luminance_max", 215.0),
        # Non-black pixel ratio derived from FOV area ratio
        "non_black_pixel_min_ratio": profile.get("fov_min_area_ratio", 0.28),
        # Retinal fundus images should have mean R > G > B
        "red_dominance_required": True,
    }
