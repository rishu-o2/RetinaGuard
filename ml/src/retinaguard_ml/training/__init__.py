"""Training package for RetinaGuard ML."""

from retinaguard_ml.training.loss import JointCategoricalOrdinalLoss, encode_ordinal_targets

__all__ = ["JointCategoricalOrdinalLoss", "encode_ordinal_targets"]

