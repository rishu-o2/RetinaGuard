"""Compatibility shim for app.services."""

from retinaguard.services.quality import QualityGateResult, QualityGateService, QualityScores
from retinaguard.services.ood import OODDetectorService, OODResult

__all__ = [
    "QualityGateResult",
    "QualityGateService",
    "QualityScores",
    "OODDetectorService",
    "OODResult",
]

