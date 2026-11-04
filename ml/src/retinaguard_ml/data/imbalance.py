"""Class-imbalance mitigation for Diabetic Retinopathy ordinal grading.

Provides:
- Inverse frequency & effective number of samples class weighting.
- WeightedRandomSampler generation for PyTorch DataLoader.
- Multi-class Focal Loss and class-weighted Cross-Entropy Loss.
"""

from collections import Counter
from typing import List, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int = 5,
    method: str = "effective_samples",
    beta: float = 0.999,
) -> torch.Tensor:
    """Calculate balanced class weights to address the severe skew towards Grade 0.

    Methods:
    - 'effective_samples': Cui et al., Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019).
    - 'inverse_frequency': w_c = N / (num_classes * count_c).
    """
    counts = Counter(labels)
    class_counts = [counts.get(c, 1) for c in range(num_classes)]

    if method == "effective_samples":
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num, dtype=np.float32)
    elif method == "inverse_frequency":
        total_samples = len(labels)
        weights = np.array(
            [total_samples / (num_classes * max(c, 1)) for c in class_counts],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize weights so they sum to num_classes
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels: Sequence[int], num_classes: int = 5) -> WeightedRandomSampler:
    """Create a PyTorch WeightedRandomSampler that samples each ICDR grade with equal probability."""
    counts = Counter(labels)
    class_weights = {c: 1.0 / max(counts.get(c, 1), 1) for c in range(num_classes)}
    sample_weights = [class_weights[int(label)] for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for mitigating hard negative imbalance in DR grading."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

