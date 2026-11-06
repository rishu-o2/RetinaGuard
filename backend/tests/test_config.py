"""Tests for configuration validation and loading."""

from pathlib import Path
import pytest
from retinaguard.core.config import (
    CameraType,
    load_clinical_rules,
    load_quality_thresholds,
)
from retinaguard.core.models import Urgency


def test_load_clinical_rules() -> None:
    """Verify that clinical_rules.yaml loads with all 5 ICDR grades and valid metadata."""
    config = load_clinical_rules()

    assert config.metadata["country_context"] == "IN"
    assert "AIOS" in config.metadata["governing_bodies"]
    assert "RSSDI" in config.metadata["governing_bodies"]

    # ICDR grades 0 through 4 must exist
    for grade_idx in range(5):
        assert grade_idx in config.icdr_grades
        rule = config.icdr_grades[grade_idx]
        assert isinstance(rule.referable, bool)
        assert isinstance(rule.base_urgency, Urgency)
        assert rule.base_interval_months >= 0

    # Grade 0 and 1 must NOT be referable under standard guidelines
    assert config.icdr_grades[0].referable is False
    assert config.icdr_grades[1].referable is False

    # Grade 2, 3, 4 MUST be referable
    assert config.icdr_grades[2].referable is True
    assert config.icdr_grades[3].referable is True
    assert config.icdr_grades[4].referable is True

    # PDR must have emergency urgency
    assert config.icdr_grades[4].base_urgency == Urgency.EMERGENCY


def test_clinical_rules_modifiers() -> None:
    """Verify that clinical modifiers (pregnancy, HbA1c, duration) are configured."""
    config = load_clinical_rules()
    modifiers = config.modifiers

    assert modifiers.pregnancy.min_urgency == Urgency.PRIORITY
    assert modifiers.pregnancy.max_interval_months <= 3
    assert modifiers.high_risk_glycemic_control.hba1c_threshold_percent > 7.0
    assert modifiers.long_standing_diabetes.duration_years_threshold >= 10.0


def test_load_quality_thresholds() -> None:
    """Verify quality profiles for tabletop and smartphone portable fundus cameras."""
    config = load_quality_thresholds()

    tabletop_profile = config.get_profile_for_camera(CameraType.TABLETOP)
    # Resolution lowered to 256 to match the ONNX model input size (NEEDS CLINICAL REVIEW)
    assert tabletop_profile.min_resolution_width >= 256
    assert tabletop_profile.blur_variance_threshold > 0

    portable_profile = config.get_profile_for_camera(CameraType.SMARTPHONE_PORTABLE)
    assert portable_profile.min_resolution_width >= 256
    # Portable profile should be tuned with a more lenient blur threshold for handheld operations
    assert portable_profile.blur_variance_threshold <= tabletop_profile.blur_variance_threshold


def test_missing_config_file_raises_error() -> None:
    """Verify FileNotFoundError is raised if an invalid path is specified."""
    with pytest.raises(FileNotFoundError):
        load_clinical_rules(Path("/non/existent/path/rules.yaml"))

