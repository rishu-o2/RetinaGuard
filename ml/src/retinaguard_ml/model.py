"""Diabetic Retinopathy Classification Model Architecture.

Outputs raw logits for the 5 ICDR grades (0 to 4).
Adheres to Principle 1: Predicts grade and confidence only.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional feature extraction block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RetinaGuardDRNet(nn.Module):
    """RetinaGuard Deep Neural Network for ICDR Grade (0-4) Classification."""

    def __init__(self, num_classes: int = 5, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Hierarchical feature representation
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification head producing logits for 5 ICDR grades
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits [Batch, 5]."""
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits

    def predict_probabilities(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Calculate calibrated softmax probabilities."""
        logits = self.forward(x)
        scaled_logits = logits / max(temperature, 1e-4)
        return torch.softmax(scaled_logits, dim=-1)

