"""Regression tests for the CASE-WOODWARD-COLELLA low-density-region diagnosis.

These tests pin down three things and nothing else:

1. the metric machinery (window mean / window mass / observed order /
   Richardson) is *arithmetically* right — checked against synthetic profiles
   whose integrals are known in closed form;
2. the diagnosis of the shipped reference curve is reproducible from the
   committed data: the ``low_plateau`` statistic of
   ``wc_reference.metrics_from_profiles`` is an extremum *at the border* of a
   comparison window, i.e. a truncation of a monotone ramp, not a trough;
3. the stored ladder artifacts are self-consistent (each npz matches its
   metrics json), so the numbers quoted in the report can be regenerated.

No solver runs here: the whole file is numpy on committed artifacts (<2 s).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "docs" / "woodward-colella-double-blast" / "ladder"
DATA = LADDER / "data"
DOCS = ROOT / "docs" / "woodward-colella-double-blast"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not (DOCS / "reference" / "reference_density.csv").exists(),
    reason="digitised reference not present in this checkout",
)

wpc = _load("wc_plateau_convergence", LADDER / "wc_plateau_convergence.py")


# ----------------------------------------------------------------------
# 1. metric arithmetic on synthetic data
# ----------------------------------------------------------------------
def test_window_mean_and_mass_match_closed_form():
    x = np.linspace(0.0, 1.0, 20001)
    rho = 0.5 + 0.25 * x                      # linear ramp, exact integrals
    lo, hi = 0.2, 0.6
    exact_mean = 0.5 + 0.25 * (lo + hi) / 2
    exact_mass = (hi - lo) * exact_mean
    assert wpc.window_mean(x, rho, (lo, hi)) == pytest.approx(exact_mean, rel=1e-10)
    assert wpc.window_mass(x, rho, (lo, hi)) == pytest.approx(exact_mass, rel=1e-10)
    assert wpc.probe(x, rho, 0.5) == pytest.approx(0.625, rel=1e-10)


def test_window_min_detects_edge_truncation():
    x = np.linspace(0.0, 1.0, 1001)
    ramp = 0.1 + x                            # monotone increasing: no interior min
    info = wpc.edge_degeneracy(x, ramp, (0.25, 0.75))
    assert info["edge_degenerate"] is True
    assert info["nearest_edge"] == "low"
    assert info["x_min"] == pytest.approx(0.25, abs=1e-12)
    # a genuine interior dip must not be flagged
    dip = 0.5 + np.abs(x - 0.5)
    assert wpc.edge_degeneracy(x, dip, (0.25, 0.75))["edge_degenerate"] is False


def test_observed_order_and_richardson_recover_exact_values():
    sizes = [200, 400, 800, 1600]
    limit, coeff, p = 3.0, 2.5, 2.0
    vals = [limit + coeff * n ** (-p) for n in sizes]
    errs = [abs(v - limit) for v in vals]
    assert wpc.observed_order(errs, sizes) == pytest.approx(p, rel=1e-8)
    assert wpc.richardson(vals, sizes, p) == pytest.approx(limit, rel=1e-6)
    assert wpc.monotone_decreasing(vals) is True


def test_richardson_rejects_degenerate_order():
    assert wpc.richardson([1.0, 2.0], [100, 200], None) is None
    assert wpc.richardson([1.0, 2.0], [100, 200], 0.0) is None


# ----------------------------------------------------------------------
# 2. the diagnosis, from the committed reference curve
# ----------------------------------------------------------------------
def test_reference_window_min_is_a_truncation_not_a_trough():
    """The published ``low_plateau`` stat is the value at the window border."""
    ref = wpc.wcr.load_reference()
    x, rho = ref["x"], ref["rho"]
    info = wpc.edge_degeneracy(x, rho, wpc.ORIGINAL_WINDOW)
    assert info["edge_degenerate"] is True, info
    assert info["nearest_edge"] == "low"
    lo = wpc.ORIGINAL_WINDOW[0]
    assert info["x_min"] == pytest.approx(lo, abs=0.01)
    # ... and the density rises monotonically across the low-density region,
    # so there is no interior minimum to speak of
    sel = (x >= lo) & (x <= 0.50)
    assert np.all(np.diff(rho[sel]) > 0)
    # the reference's own domain starts left of the comparison window, i.e. the
    # real trough (if any) is outside what was digitised
    assert x.min() < lo


def test_plateau_windows_stay_inside_low_density_region():
    ref = wpc.wcr.load_reference()
    for lo, hi in wpc.PLATEAU_WINDOWS:
        assert wpc.ORIGINAL_WINDOW[0] - 1e-12 <= lo < hi <= 0.60
        vals = ref["rho"][(ref["x"] >= lo) & (ref["x"] <= hi)]
        assert vals.max() < 0.20      # still the shallow low-density ramp
        assert vals.min() > 0.10


# ----------------------------------------------------------------------
# 3. stored ladder artifacts are self-consistent
# ----------------------------------------------------------------------
def _stored():
    if not DATA.exists():
        return {}
    return {p.stem.replace("ladder_profile_", ""): p
            for p in sorted(DATA.glob("ladder_profile_*.npz"))}


def test_stored_profiles_are_positive_and_ordered():
    for tag, path in _stored().items():
        d = np.load(path)
        x, rho = d["x"], d["rho"]
        assert x.size == rho.size >= 100, tag
        assert np.all(np.diff(x) > 0), tag
        assert np.all(rho > 0) and np.all(np.isfinite(rho)), tag


def test_stored_metrics_match_recomputation():
    """Every committed metrics json must follow from its committed profile."""
    ref = wpc.wcr.load_reference()
    for tag, path in _stored().items():
        jpath = DATA / f"ladder_metrics_{tag}.json"
        if not jpath.exists():
            continue
        d = np.load(path)
        stored = json.loads(jpath.read_text())
        got = wpc.metrics(d["x"], d["rho"], ref)
        assert got["published"]["low_plateau_rel_percent"] == pytest.approx(
            stored["low_plateau"]["rel_error_percent"], rel=1e-9), tag
        assert got["published"]["peak_density_rel_percent"] == pytest.approx(
            stored["peak_density"]["rel_error_percent"], rel=1e-9), tag
        assert got["published"]["l1_percent"] == pytest.approx(
            stored["l1_relative"]["percent"], rel=1e-9), tag


def test_analysis_runs_and_reports_well_posed_metrics():
    res = wpc.analyse(DATA, DOCS / "artifacts", outdir=DATA)
    assert res["rows"], "no profiles analysed"
    for tag, row in res["rows"].items():
        assert 0.0 <= row["plateau_mean"]["rel_error_percent"] < 300.0, tag
        assert row["window_mass"]["rel_error_percent"] < 10.0, tag
    # the window-min diagnosis must hold for the reference and for every profile
    diag = res["diagnosis"]["window_min_is_edge_degenerate"]
    assert diag["ref"] is True
    assert diag["all_solutions"] is True
    tables = (DATA / "plateau_tables.md").read_text()
    assert "window_min" in tables and "plateau_mean" in tables