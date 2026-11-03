"""Joint Categorical and Ordinal Loss Module.

Combines weighted multi-class loss with cumulative binary ordinal loss:
L_total = L_cat + lambda_ord * L_ord
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from retinaguard_ml.data.imbalance import FocalLoss


def encode_ordinal_targets(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Encode integer ICDR labels [B] into binary cumulative targets [B, 4].

    Example:
    Grade 0 -> [0, 0, 0, 0]
    Grade 1 -> [1, 0, 0, 0]
    Grade 2 -> [1, 1, 0, 0]
    Grade 3 -> [1, 1, 1, 0]
    Grade 4 -> [1, 1, 1, 1]
    """
    batch_size = labels.size(0)
    num_thresholds = num_classes - 1

    # Thresholds: 1, 2, 3, 4
    thresholds = torch.arange(1, num_classes, device=labels.device).unsqueeze(0).expand(batch_size, -1)
    expanded_labels = labels.unsqueeze(1).expand(-1, num_thresholds)

    ordinal_targets = (expanded_labels >= thresholds).float()
    return ordinal_targets


class JointCategoricalOrdinalLoss(nn.Module):
    """Joint Multi-Task Loss optimizing for both classification accuracy and ordinal ordering."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ordinal_weight: float = 0.50,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        num_classes: int = 5,
    ):
        super().__init__()
        self.ordinal_weight = ordinal_weight
        self.num_classes = num_classes

        if use_focal_loss:
            self.categorical_criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.categorical_criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.ordinal_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute joint loss and return components dict."""
        cat_logits = outputs["logits"]
        ord_logits = outputs["ordinal_logits"]

        # 1. Categorical Loss (Weighted CE or Focal)
        l_cat = self.categorical_criterion(cat_logits, targets)

        # 2. Ordinal Binary Cross-Entropy Loss
        ord_targets = encode_ordinal_targets(targets, num_classes=self.num_classes)
        l_ord = self.ordinal_criterion(ord_logits, ord_targets)

        # 3. Combined objective
        total_loss = l_cat + (self.ordinal_weight * l_ord)

        return {
            "loss": total_loss,
            "categorical_loss": l_cat,
            "ordinal_loss": l_ord,
        }

