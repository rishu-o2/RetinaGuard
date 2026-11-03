"""Compatibility forwarding to retinaguard.services.quality."""

from retinaguard.services.quality import (
    QualityGateResult,
    QualityGateService,
    QualityScores,
)

__all__ = ["QualityGateResult", "QualityGateService", "QualityScores"]

