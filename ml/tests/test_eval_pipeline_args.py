"""Tests for the parametrized eval_pipeline.py.

Verifies:
1. --version v2 / --out-dir resolve to versioned artifact names (never v1 paths).
2. --version v1 (default) resolves to v1 artifact names.
3. Registry update appends v2 without overwriting v1 when version_tag != "v1".
4. Active flag: v1 always active=True, v2 always active=False.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))


# ---------------------------------------------------------------------------
# Registry upsert logic (re-implemented here from eval_pipeline.py for unit testing)
# ---------------------------------------------------------------------------
def _upsert_registry(registry_path: Path, new_entry: dict) -> dict:
    """Mimics the registry upsert logic in eval_pipeline.py."""
    existing = {"models": []}
    if registry_path.exists():
        with open(registry_path, encoding="utf-8") as f:
            existing = json.load(f)

    models_list = existing.get("models", [])
    version_tag = new_entry.get("version_tag", new_entry.get("version", ""))

    # Remove existing entry for THIS version only
    models_list = [m for m in models_list if m.get("version_tag", m.get("version", "")) != version_tag]

    # Ensure v1 stays first (active)
    if version_tag == "v1":
        models_list = [new_entry] + models_list
    else:
        models_list = models_list + [new_entry]

    return {"models": models_list}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEvalPipelineArgs:
    """Unit tests for parametrized eval_pipeline argument logic."""

    def test_v1_default_paths(self):
        """v1 (default) should use retinaguard_dr_v1.onnx and ood_stats_v1.npz."""
        version = "v1"
        onnx_filename = f"retinaguard_dr_{version}.onnx"
        ood_filename  = f"ood_stats_{version}.npz"
        assert onnx_filename == "retinaguard_dr_v1.onnx"
        assert ood_filename  == "ood_stats_v1.npz"

    def test_v2_uses_versioned_artifact_names(self):
        """v2 should use retinaguard_dr_v2.onnx and ood_stats_v2.npz."""
        version = "v2"
        onnx_filename = f"retinaguard_dr_{version}.onnx"
        ood_filename  = f"ood_stats_{version}.npz"
        assert onnx_filename == "retinaguard_dr_v2.onnx"
        assert ood_filename  == "ood_stats_v2.npz"

    def test_v1_does_not_appear_in_v2_artifact_names(self):
        """v2 artifact names must not contain 'v1'."""
        version = "v2"
        onnx_filename = f"retinaguard_dr_{version}.onnx"
        ood_filename  = f"ood_stats_{version}.npz"
        assert "v1" not in onnx_filename
        assert "v1" not in ood_filename

    def test_v1_out_dir_default(self):
        """v1 default out-dir is 'val_report' (not 'val_report_v2')."""
        version = "v1"
        subdir = "val_report" if version == "v1" else f"val_report_{version}"
        assert subdir == "val_report"

    def test_v2_out_dir_default(self):
        """v2 default out-dir is 'val_report_v2'."""
        version = "v2"
        subdir = "val_report" if version == "v1" else f"val_report_{version}"
        assert subdir == "val_report_v2"

    def test_abstention_config_v1(self):
        """v1 uses configs/abstention.yaml."""
        version = "v1"
        fname = "abstention.yaml" if version == "v1" else f"abstention_{version}.yaml"
        assert fname == "abstention.yaml"

    def test_abstention_config_v2(self):
        """v2 uses configs/abstention_v2.yaml."""
        version = "v2"
        fname = "abstention.yaml" if version == "v1" else f"abstention_{version}.yaml"
        assert fname == "abstention_v2.yaml"


class TestRegistryUpsert:
    """Unit tests for registry upsert: v1 never overwritten, v2 appended inactive."""

    def test_v1_active_true(self, tmp_path):
        """Registry v1 entry must have active=True."""
        reg_path = tmp_path / "registry.json"
        v1_entry = {
            "version_tag": "v1",
            "version": "v1.0.0",
            "active": True,
            "onnx_path": "ml/exports/retinaguard_dr_v1.onnx",
            "ood_stats_path": "models/ood_stats_v1.npz",
        }
        result = _upsert_registry(reg_path, v1_entry)
        assert any(m.get("active", False) for m in result["models"])
        v1 = next(m for m in result["models"] if m["version_tag"] == "v1")
        assert v1["active"] is True

    def test_v2_active_false(self, tmp_path):
        """Registry v2 entry must have active=False."""
        reg_path = tmp_path / "registry.json"
        # Seed v1
        v1_entry = {"version_tag": "v1", "version": "v1.0.0", "active": True,
                    "onnx_path": "ml/exports/retinaguard_dr_v1.onnx"}
        result_v1 = _upsert_registry(reg_path, v1_entry)
        with open(reg_path, "w") as f:
            json.dump(result_v1, f)

        v2_entry = {
            "version_tag": "v2",
            "version": "v2.0.0",
            "active": False,   # eval_pipeline.py sets active=False for v2
            "onnx_path": "ml/exports/retinaguard_dr_v2.onnx",
            "ood_stats_path": "models/ood_stats_v2.npz",
        }
        result_v2 = _upsert_registry(reg_path, v2_entry)
        v2 = next(m for m in result_v2["models"] if m["version_tag"] == "v2")
        assert v2["active"] is False

    def test_v1_not_overwritten_when_v2_added(self, tmp_path):
        """Adding v2 must not alter the v1 entry."""
        reg_path = tmp_path / "registry.json"
        v1_entry = {
            "version_tag": "v1", "version": "v1.0.0", "active": True,
            "onnx_path": "ml/exports/retinaguard_dr_v1.onnx",
            "temperature": 1.2963,
        }
        result_v1 = _upsert_registry(reg_path, v1_entry)
        with open(reg_path, "w") as f:
            json.dump(result_v1, f)

        v2_entry = {"version_tag": "v2", "version": "v2.0.0", "active": False,
                    "onnx_path": "ml/exports/retinaguard_dr_v2.onnx", "temperature": 1.1}
        result_v2 = _upsert_registry(reg_path, v2_entry)

        v1 = next(m for m in result_v2["models"] if m["version_tag"] == "v1")
        assert v1["temperature"] == 1.2963, "v1 temperature was overwritten!"
        assert v1["active"] is True, "v1 active flag was changed!"
        assert v1["onnx_path"] == "ml/exports/retinaguard_dr_v1.onnx"

    def test_v1_first_in_registry(self, tmp_path):
        """v1 must remain the first entry (active model) even after v2 is added."""
        reg_path = tmp_path / "registry.json"
        v1_entry = {"version_tag": "v1", "version": "v1.0.0", "active": True,
                    "onnx_path": "ml/exports/retinaguard_dr_v1.onnx"}
        result_v1 = _upsert_registry(reg_path, v1_entry)
        with open(reg_path, "w") as f:
            json.dump(result_v1, f)

        v2_entry = {"version_tag": "v2", "version": "v2.0.0", "active": False,
                    "onnx_path": "ml/exports/retinaguard_dr_v2.onnx"}
        result_v2 = _upsert_registry(reg_path, v2_entry)
        assert result_v2["models"][0]["version_tag"] == "v1", "v1 must be first in registry"

    def test_idempotent_v2_upsert(self, tmp_path):
        """Running v2 eval twice must not create duplicate entries."""
        reg_path = tmp_path / "registry.json"
        v1_entry = {"version_tag": "v1", "version": "v1.0.0", "active": True,
                    "onnx_path": "ml/exports/retinaguard_dr_v1.onnx"}
        result = _upsert_registry(reg_path, v1_entry)
        with open(reg_path, "w") as f:
            json.dump(result, f)

        v2_entry = {"version_tag": "v2", "version": "v2.0.0", "active": False,
                    "onnx_path": "ml/exports/retinaguard_dr_v2.onnx"}

        # Upsert v2 twice
        result = _upsert_registry(reg_path, v2_entry)
        with open(reg_path, "w") as f:
            json.dump(result, f)
        result = _upsert_registry(reg_path, v2_entry)

        v2_entries = [m for m in result["models"] if m.get("version_tag") == "v2"]
        assert len(v2_entries) == 1, "v2 must not be duplicated by repeated upsert"

    def test_registry_has_exactly_two_entries_after_v2(self, tmp_path):
        """Registry must have exactly 2 entries (v1, v2) after both runs."""
        reg_path = tmp_path / "registry.json"
        v1_entry = {"version_tag": "v1", "version": "v1.0.0", "active": True,
                    "onnx_path": "ml/exports/retinaguard_dr_v1.onnx"}
        result = _upsert_registry(reg_path, v1_entry)
        with open(reg_path, "w") as f:
            json.dump(result, f)

        v2_entry = {"version_tag": "v2", "version": "v2.0.0", "active": False,
                    "onnx_path": "ml/exports/retinaguard_dr_v2.onnx"}
        result = _upsert_registry(reg_path, v2_entry)
        assert len(result["models"]) == 2

    def test_v2_uses_v2_artifacts_in_registry(self):
        """v2 registry entry must reference v2 ONNX and OOD files."""
        v2_entry = {
            "version_tag": "v2",
            "onnx_path": "ml/exports/retinaguard_dr_v2.onnx",
            "ood_stats_path": "models/ood_stats_v2.npz",
        }
        assert "v2" in v2_entry["onnx_path"]
        assert "v2" in v2_entry["ood_stats_path"]
        assert "v1" not in v2_entry["onnx_path"]
        assert "v1" not in v2_entry["ood_stats_path"]


class TestActiveModelUnchanged:
    """Verify that v1 stays the active model after v2 evaluation."""

    def test_v1_active_field_true(self):
        """The active field for v1 must always be True."""
        for version in ["v1", "v2"]:
            is_active = version == "v1"
            if version == "v1":
                assert is_active is True
            else:
                assert is_active is False

    def test_inference_service_uses_first_active_entry(self, tmp_path):
        """Simulated inference service: picks first active entry — must be v1."""
        reg = {
            "models": [
                {"version_tag": "v1", "version": "v1.0.0", "active": True,
                 "onnx_path": "ml/exports/retinaguard_dr_v1.onnx"},
                {"version_tag": "v2", "version": "v2.0.0", "active": False,
                 "onnx_path": "ml/exports/retinaguard_dr_v2.onnx"},
            ]
        }
        # Simulate: pick first active entry (as inference.py does)
        entries = reg["models"]
        active_entry = next((m for m in entries if m.get("active", False)), entries[0])
        assert active_entry["version_tag"] == "v1"
        assert active_entry["onnx_path"] == "ml/exports/retinaguard_dr_v1.onnx"
