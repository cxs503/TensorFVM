"""Contract test for CASE-CYLINDER-RE100: unbounded cylinder vortex shedding.

Tests the verification module's ability to:
1. Build a valid SolverConfig for Re=100 unbounded cylinder
2. Run a short transient simulation and extract Cd/St
3. Compare against literature reference with tolerance bands
4. Publish results with figures

The full production run (2000+ steps) is delegated to the cluster;
this test uses a minimal run to verify the code path.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
REF_FILE = REPO / "artifacts" / "case-cylinder-re100" / "reference" / "reference.json"


def test_reference_file_exists_and_complete():
    ref = json.loads(REF_FILE.read_text())
    assert "values" in ref
    assert "cd_mean" in ref["values"]
    assert "st" in ref["values"]
    assert "tolerance_band" in ref
    assert "citation" in ref
    assert "provenance" in ref
    assert "braza_1986" in ref["provenance"]
    assert "schafer_turek_1996" in ref["provenance"]
    assert "doi" in ref["provenance"]["braza_1986"]
    assert "doi" in ref["provenance"]["schafer_turek_1996"]


def test_reference_values_reasonable():
    ref = json.loads(REF_FILE.read_text())
    vals = ref["values"]
    assert 1.0 < vals["cd_mean"] < 2.0, f"Cd_mean {vals['cd_mean']} out of range"
    assert 0.1 < vals["st"] < 0.3, f"St {vals['st']} out of range"
    bands = ref["tolerance_band"]
    assert bands["cd_mean"] > 0
    assert bands["st"] > 0


def test_config_builds():
    from tensorfvm.verification.cylinder_re100 import _build_config
    config = _build_config("64x32", 0.02, 100)
    assert config.reynolds == 100.0
    assert config.time_step == 0.02
    assert config.mesh_type == "body-fitted"
    assert config.outer_boundary == "far-field"
    assert config.cylinder_radius == 0.5


def test_short_transient_run(tmp_path):
    """Run a very short transient (10 steps) to verify code path."""
    from tensorfvm.verification.cylinder_re100 import (
        _build_config, run_transient, compute_errors, publish
    )
    config = _build_config("32x16", 0.05, 10, device="cpu")
    result, solver = run_transient(config, warmup_steps=2)
    assert result["cd_mean"] > 0
    # Ten steps cannot establish a periodic regime, so the honest gate must
    # withhold the Strouhal number instead of reporting a spectral peak of the
    # rounding noise (the defect this test now pins down).
    assert result["shedding_detected"] is False
    assert result["st"] is None
    assert result["gate_reasons"], "gate_reasons must explain the withheld St"
    assert len(result["force_history"]) == 10
    errors = compute_errors(result)
    assert "cd_mean" in errors
    assert "st" in errors
    assert "computed" in errors["cd_mean"]
    assert "relative_error" in errors["cd_mean"]
    assert errors["st"]["computed"] is None
    assert errors["gates"]["passed"] is False
    passed = publish(tmp_path, result, errors)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "errors.json").exists()
    assert (tmp_path / "force_history.csv").exists()
    assert (tmp_path / "force_time_series.png").exists()
    assert (tmp_path / "lift_spectrum.png").exists()
    assert (tmp_path / "error_bars.png").exists()
    assert (tmp_path / "phase_plot.png").exists()
    assert (tmp_path / "report.md").exists()


def test_module_imports():
    from tensorfvm.verification import cylinder_re100
    assert hasattr(cylinder_re100, "REFERENCE")
    assert hasattr(cylinder_re100, "REFERENCE_BAND")
    assert hasattr(cylinder_re100, "run_transient")
    assert hasattr(cylinder_re100, "compute_errors")
    assert hasattr(cylinder_re100, "publish")
    assert hasattr(cylinder_re100, "main")


def test_cli_help():
    proc = subprocess.run(
        [sys.executable, "-m", "tensorfvm.verification.cylinder_re100", "--help"],
        capture_output=True, text=True, timeout=10,
        cwd=str(REPO), env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}
    )
    assert proc.returncode == 0
    assert "--output" in proc.stdout
    assert "--grid" in proc.stdout
