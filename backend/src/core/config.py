"""Configuration loader for clinical rules and quality gate parameters.

Ensures that all clinical numbers are validated and loaded directly from YAML
with documented audit trails and citations.
"""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field

from retinaguard.core.models import CameraType, ICDRGrade, Urgency


class ICDRGradeRule(BaseModel):
    name: str
    referable: bool
    base_urgency: Urgency
    base_interval_months: int
    description: str


class PregnancyModifier(BaseModel):
    applies_to_female_only: bool = True
    min_urgency: Urgency
    max_interval_months: int
    notes: str


class HighRiskGlycemicModifier(BaseModel):
    hba1c_threshold_percent: float
    max_interval_months: int
    upgrade_urgency_if_grade_ge: int


class LongStandingDiabetesModifier(BaseModel):
    duration_years_threshold: float
    max_interval_months: int
    upgrade_urgency_if_grade_ge: int


class ModifiersConfig(BaseModel):
    pregnancy: PregnancyModifier
    high_risk_glycemic_control: HighRiskGlycemicModifier
    long_standing_diabetes: LongStandingDiabetesModifier


class ConfidenceThresholdsConfig(BaseModel):
    min_acceptable_confidence: float = Field(..., ge=0.0, le=1.0)
    low_confidence_action: str


class ClinicalRulesConfig(BaseModel):
    metadata: dict[str, Any]
    icdr_grades: dict[int, ICDRGradeRule]
    modifiers: ModifiersConfig
    confidence_thresholds: ConfidenceThresholdsConfig


class CameraQualityProfile(BaseModel):
    name: str
    description: str
    min_resolution_width: int
    min_resolution_height: int
    blur_variance_threshold: float
    underexposure_threshold: float
    overexposure_threshold: float
    glare_area_max_ratio: float
    mean_luminance_min: float
    mean_luminance_max: float


class QualityThresholdsConfig(BaseModel):
    version: str
    profiles: dict[str, CameraQualityProfile]
    rejection_behavior: dict[str, Any]

    def get_profile_for_camera(self, camera_type: CameraType) -> CameraQualityProfile:
        """Fetch corresponding quality profile for tabletop vs smartphone/portable."""
        if camera_type == CameraType.SMARTPHONE_PORTABLE:
            return self.profiles["smartphone_portable"]
        # Default to tabletop standard
        return self.profiles["tabletop"]


def load_clinical_rules(config_path: Path | str | None = None) -> ClinicalRulesConfig:
    """Load and validate clinical_rules.yaml."""
    if config_path is None:
        # Default relative to repository root
        config_path = Path(__file__).resolve().parents[3] / "configs" / "clinical_rules.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Clinical rules config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ClinicalRulesConfig.model_validate(data)


def load_quality_thresholds(config_path: Path | str | None = None) -> QualityThresholdsConfig:
    """Load and validate quality_thresholds.yaml."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "quality_thresholds.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Quality thresholds config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return QualityThresholdsConfig.model_validate(data)

