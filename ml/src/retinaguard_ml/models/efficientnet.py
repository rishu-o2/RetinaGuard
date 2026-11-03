"""Dual-Head EfficientNet Architecture for Diabetic Retinopathy.

Implements ADR-010:
- Configurable EfficientNet backbone (via timm with torchvision fallback).
- Head 1: 5-class categorical classification logits.
- Head 2: 4-output ordinal cumulative threshold logits (P(grade >= k)).
- Joint representation learning for optimal Quadratic Weighted Kappa (QWK).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


def _build_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Instantiate backbone using timm if available, or torchvision fallback."""
    try:
        import timm  # type: ignore
        model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_features = model.num_features
        return model, in_features
    except ImportError:
        # Fallback to torchvision
        import torchvision.models as tv_models  # type: ignore

        clean_name = backbone_name.lower().replace("-", "_")
        if hasattr(tv_models, clean_name):
            builder = getattr(tv_models, clean_name)
            weights = "DEFAULT" if pretrained else None
            try:
                base = builder(weights=weights)
            except Exception:
                base = builder(pretrained=pretrained)

            # Extract feature extractor and classifier in_features
            in_features = base.classifier[1].in_features
            # Replace classifier with identity
            base.classifier = nn.Identity()
            return base, in_features
        else:
            # Default to torchvision efficientnet_b0
            base = tv_models.efficientnet_b0(weights=None)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            return base, in_features


class EfficientNetDRDualHead(nn.Module):
    """Dual-Head Diabetic Retinopathy Network.

    Outputs:
    1. 'logits': Raw categorical logits of shape [B, 5] for ICDR grades 0-4.
    2. 'ordinal_logits': Cumulative threshold logits of shape [B, 4] for P(grade >= 1..4).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout_rate: float = 0.30,
        num_classes: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_ordinal_thresholds = num_classes - 1  # 4 thresholds for 5 grades

        # Feature extractor
        self.backbone, self.in_features = _build_backbone(backbone_name, pretrained=pretrained)

        self.dropout = nn.Dropout(p=dropout_rate)

        # Head 1: Categorical 5-class logits
        self.fc_categorical = nn.Linear(self.in_features, self.num_classes)

        # Head 2: Ordinal 4-threshold logits
        self.fc_ordinal = nn.Linear(self.in_features, self.num_ordinal_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone and dual heads."""
        feats = self.backbone(x)
        dropped = self.dropout(feats)

        categorical_logits = self.fc_categorical(dropped)
        ordinal_logits = self.fc_ordinal(dropped)

        return {
            "logits": categorical_logits,
            "ordinal_logits": ordinal_logits,
        }

    @torch.no_grad()
    def predict_probabilities(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Return calibrated softmax probabilities across the 5 ICDR classes [B, 5]."""
        outputs = self.forward(x)
        logits = outputs["logits"] / max(temperature, 1e-4)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_ordinal_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return cumulative probabilities P(grade >= k) for k in {1, 2, 3, 4} [B, 4]."""
        outputs = self.forward(x)
        return torch.sigmoid(outputs["ordinal_logits"])

    @torch.no_grad()
    def predict_grades(self, x: torch.Tensor, method: str = "ordinal_sum") -> torch.Tensor:
        """Predict integer ICDR grades (0 to 4).

        Methods:
        - 'argmax': Argmax of categorical softmax.
        - 'ordinal_sum': Sum of cumulative probabilities P(grade >= k) rounded to nearest int.
          (Frank & Hall / ordinal thresholding, proven to maximize QWK).
        """
        if method == "argmax":
            probs = self.predict_probabilities(x)
            return torch.argmax(probs, dim=-1)
        elif method == "ordinal_sum":
            ord_probs = self.predict_ordinal_probabilities(x)
            # Grade is sum of binary indicator expectations: sum(P(grade >= k))
            expected_grade = torch.sum(ord_probs, dim=-1)
            return torch.round(expected_grade).long().clamp(0, self.num_classes - 1)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
