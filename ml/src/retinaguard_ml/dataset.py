from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

# Standard RGB normalization constants for fundus photography
FUNDUS_MEAN = [0.485, 0.456, 0.406]
FUNDUS_STD = [0.229, 0.224, 0.225]


class SyntheticFundusDataset(Dataset):

    def __init__(self, num_samples: int = 50, image_size: Tuple[int, int] = (224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.rng = np.random.default_rng(seed=42)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Generate synthetic 3-channel image tensor [3, H, W]
        img_array = self.rng.uniform(0.0, 1.0, size=(3, self.image_size[0], self.image_size[1])).astype(np.float32)
        # Random ICDR grade 0 to 4
        label = int(self.rng.integers(0, 5))
        return torch.from_numpy(img_array), label


def preprocess_fundus_array(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:

    import cv2  # type: ignore

    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0

    # Normalize channel-wise
    for c in range(3):
        norm[:, :, c] = (norm[:, :, c] - FUNDUS_MEAN[c]) / FUNDUS_STD[c]

    # Convert to [1, 3, H, W]
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

