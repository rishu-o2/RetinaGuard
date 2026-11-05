"""Configuration loader for clinical rules and quality gate parameters.

Ensures that all clinical numbers are validated and loaded directly from YAML
with documented audit trails and citations.
"""

from pathlib import Path
from typing import Any, Optional
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


class SightThreateningEscalationModifier(BaseModel):
    enabled: bool = True
    threshold_soon: float = Field(default=0.05, ge=0.0, le=1.0)
    target_urgency_soon: Urgency = Urgency.SOON
    max_interval_months_soon: int = 3
    threshold_urgent: float = Field(default=0.30, ge=0.0, le=1.0)
    target_urgency_urgent: Urgency = Urgency.URGENT
    max_interval_months_urgent: int = 1
    probability_threshold: Optional[float] = None
    target_urgency: Optional[Urgency] = None
    max_interval_months: Optional[int] = None
    notes: Optional[str] = None


class ModifiersConfig(BaseModel):
    pregnancy: PregnancyModifier
    high_risk_glycemic_control: HighRiskGlycemicModifier
    long_standing_diabetes: LongStandingDiabetesModifier
    sight_threatening_escalation: Optional[SightThreateningEscalationModifier] = None


class ReferableBandAbstentionConfig(BaseModel):
    enabled: bool = True
    referral_cutoff: float = Field(default=0.50, ge=0.0, le=1.0)
    lower_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    upper_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    notes: Optional[str] = None


class ConfidenceThresholdsConfig(BaseModel):
    min_acceptable_confidence: float = Field(..., ge=0.0, le=1.0)
    low_confidence_action: str
    referable_band: Optional[ReferableBandAbstentionConfig] = None


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
    fov_min_area_ratio: float = 0.30
    fov_min_circularity: float = 0.60
    mean_luminance_min: float
    mean_luminance_max: float
    p5_luminance_min: float = 10.0
    p95_luminance_max: float = 245.0
    underexposure_max_ratio: float = 0.15
    overexposure_max_ratio: float = 0.08
    # Backward compatibility aliases
    underexposure_threshold: Optional[float] = None
    overexposure_threshold: Optional[float] = None
    glare_area_max_ratio: Optional[float] = None


class QualityThresholdsConfig(BaseModel):
    version: str
    profiles: dict[str, CameraQualityProfile]
    failure_messages: Optional[dict[str, Any]] = None
    rejection_behavior: Optional[dict[str, Any]] = None

    def get_profile_for_camera(self, camera_type: CameraType) -> CameraQualityProfile:
        """Fetch corresponding quality profile for tabletop vs smartphone/portable."""
        if camera_type == CameraType.SMARTPHONE_PORTABLE:
            return self.profiles["smartphone_portable"]
        return self.profiles["tabletop"]


def _find_repo_root() -> Path:
    """Locate the root of the RetinaGuard repository containing 'configs'."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "clinical_rules.yaml").exists():
            return parent
    return Path.cwd()


def load_clinical_rules(config_path: Path | str | None = None) -> ClinicalRulesConfig:
    """Load and validate clinical_rules.yaml."""
    if config_path is None:
        config_path = _find_repo_root() / "configs" / "clinical_rules.yaml"
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
        config_path = _find_repo_root() / "configs" / "quality_thresholds.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Quality thresholds config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return QualityThresholdsConfig.model_validate(data)
