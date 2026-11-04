"""Fundus image preprocessing pipeline.

Includes:
1. Automated black-border cropping (tight fundus bounding box).
2. Configurable resizing (default 512x512).
3. Optional Ben Graham-style local Gaussian contrast enhancement.
4. Channel-wise normalization for deep learning backbones.
"""

from typing import Tuple
import cv2  # type: ignore
import numpy as np
import torch

FUNDUS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
FUNDUS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_black_borders(img: np.ndarray, tol: int = 10) -> np.ndarray:
    """Crop away black background borders around the circular fundus region.

    Uses thresholding across color channels to locate non-zero bounding box.
    """
    if img.ndim == 2:
        mask = img > tol
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
    else:
        return img

    # Check if image is entirely black/dark
    if not np.any(mask):
        return img

    # Find row and column bounds where mask is true
    row_indices = np.where(mask.any(axis=1))[0]
    col_indices = np.where(mask.any(axis=0))[0]

    ymin, ymax = row_indices[0], row_indices[-1] + 1
    xmin, xmax = col_indices[0], col_indices[-1] + 1

    # Ensure valid bounding box
    if (ymax - ymin > 10) and (xmax - xmin > 10):
        return img[ymin:ymax, xmin:xmax]
    return img


def ben_graham_enhancement(
    img: np.ndarray, sigma_x: int = 30, radius_ratio: float = 0.95
) -> np.ndarray:
    """Ben Graham's method (Kaggle DR 1st place technique).

    Subtracts local average color to mitigate non-uniform illumination and enhance
    subtle microvascular lesions (microaneurysms, hard exudates).
    """
    # 1. Scale image and local Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_x)

    # 2. Add weighted difference to highlight high-frequency vascular structures
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)

    # 3. Create a circular mask to prevent boundary artifacts
    h, w = enhanced.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(center[0], center[1]) * radius_ratio)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)

    # Blend outside boundary to neutral gray (128)
    result = np.full_like(enhanced, 128)
    result[mask == 1] = enhanced[mask == 1]

    return result


class FundusPreprocessor:
    """Configurable preprocessor for retinal fundus images."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        crop_borders: bool = True,
        use_ben_graham: bool = False,
        ben_graham_sigma: int = 30,
    ):
        self.target_size = target_size
        self.crop_borders = crop_borders
        self.use_ben_graham = use_ben_graham
        self.ben_graham_sigma = ben_graham_sigma

    def process(self, img_rgb: np.ndarray) -> np.ndarray:
        """Run preprocessing pipeline and return uint8 RGB array [H, W, 3]."""
        # Step 1: Crop black borders
        if self.crop_borders:
            img_rgb = crop_black_borders(img_rgb)

        # Step 2: Resize to target dimension
        resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        # Step 3: Optional Ben Graham local contrast enhancement
        if self.use_ben_graham:
            resized = ben_graham_enhancement(resized, sigma_x=self.ben_graham_sigma)

        return resized

    def to_normalized_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess and convert to PyTorch tensor [1, 3, H, W] normalized with standard stats."""
        processed = self.process(img_rgb)
        norm = processed.astype(np.float32) / 255.0
        norm = (norm - FUNDUS_MEAN) / FUNDUS_STD

        # Transpose [H, W, C] to [C, H, W] and add batch dimension
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

