"""GCI analysis gate for CASE-WOODWARD-COLELLA.

Verifies that the committed ladder data produces a well-formed GCI analysis
and that the key diagnostics are consistent with the shock-dominated nature
of the double blast wave problem.

No solver runs: pure numpy on committed artifacts (<1 s).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "docs" / "woodward-colella-double-blast" / "ladder"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gci = _load("wc_gci_analysis", LADDER / "wc_gci_analysis.py")


def test_gci_analysis_runs_and_returns_all_metrics():
    """The analysis must return results for all three committed metrics."""
    results = gci.analyse_gci()
    assert set(results.keys()) == {"peak_density", "l1_error", "window_mass"}


def test_peak_density_not_asymptotic():
    """The peak density is shock-dominated and must NOT be in the asymptotic range."""
    r = gci.analyse_gci()["peak_density"]
    assert r.asymptotic is False
    assert r.observed_order < 1.0  # far below theory order 2.0
    assert r.gci_percent > 10.0  # wide uncertainty band


def test_window_mass_small_gci():
    """The window mass integral is a smooth functional and must have small GCI."""
    r = gci.analyse_gci()["window_mass"]
    assert r.gci_percent < 1.0  # already converged
    assert r.ref_rel_error_percent is not None
    assert r.ref_rel_error_percent < 3.0  # within 3% of reference


def test_l1_error_monotone_convergence():
    """The L1 error must decrease monotonically with refinement."""
    r = gci.analyse_gci()["l1_error"]
    assert r.convergence == "monotone"
    assert r.fine_value < r.medium_value < r.coarse_value


def test_gci_json_artifact_exists():
    """The committed GCI JSON must exist and be well-formed."""
    import json
    json_path = LADDER / "data" / "gci_analysis.json"
    assert json_path.exists(), f"GCI JSON not found at {json_path}"
    data = json.loads(json_path.read_text())
    assert set(data.keys()) == {"peak_density", "l1_error", "window_mass"}
    for key in data:
        assert "gci_percent" in data[key]
        assert "observed_order" in data[key]
        assert "asymptotic" in data[key]


def test_summary_table_is_nonempty():
    """The summary table must contain all three metrics."""
    results = gci.analyse_gci()
    table = gci.summary_table(results)
    assert "peak_density" in table
    assert "l1_error" in table
    assert "window_mass" in table
    assert "GCI Analysis Summary" in table
