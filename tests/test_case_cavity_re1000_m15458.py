"""Anchor ``CASE-CAVITY-RE1000-FVM`` round 2: re-derive the 5 % verdict from evidence.

The point of this test is that the *claim* cannot drift from the *bundle*: every
number it checks is read back from ``artifacts/CASE-CAVITY-RE1000-FVM/summary.json``
(written by ``scripts/run_cavity_re1000_m15458.py``) and re-scored here against the
committed Ghia transcription in :mod:`tensorfvm.lid_cavity_reference`.  It does not
re-run the solver (that is hours of CPU); it re-runs the *judgement*.

Skips cleanly when the artefact bundle is absent, so a checkout without the
artefacts does not fail the suite.
"""

from __future__ import annotations

import json
import os

import pytest

from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "artifacts", "CASE-CAVITY-RE1000-FVM", "summary.json")
TOL = 0.05


def _summary() -> dict:
    if not os.path.exists(SUMMARY):
        pytest.skip(f"no artefact bundle at {SUMMARY}")
    with open(SUMMARY) as fh:
        return json.load(fh)


def test_every_reported_row_is_converged():
    summary = _summary()
    rows = summary["cases"]
    assert rows, "no cases in the bundle"
    for case in rows:
        assert case["solver"]["converged"] is True, f"n={case['n']} not converged"
        assert case["solver"]["residual_rate"] <= 1e-6
        assert case["solver"]["max_divergence"] < 1e-10


def test_ledger_verdict_matches_its_own_numbers():
    """``verdict_5pct`` must be reproducible from the metrics it reports."""
    summary = _summary()
    for case in summary["cases"]:
        verdict = case["verdict_5pct"]
        assert verdict["tolerance"] == TOL
        recomputed = all(value <= TOL for value in verdict["metrics"].values())
        assert verdict["pass"] is bool(recomputed), f"n={case['n']} verdict inconsistent"
        for key, ok in verdict["per_metric"].items():
            assert ok is bool(verdict["metrics"][key] <= TOL)


def test_key_quantities_inside_five_percent():
    """The anchor's acceptance: u_min, v_max and the vortex centre inside 5 %."""
    summary = _summary()
    ref_u = dict(table(1000.0, "u"))
    ref_v = dict(table(1000.0, "v"))
    production = [c for c in summary["cases"] if c["n"] >= 64]
    assert production, "no production-resolution row in the bundle"
    # the deliberately under-resolved control rows must stay reported as failing, not
    # quietly dropped, so that a 5 % pass cannot hide a coarse grid
    for case in summary["cases"]:
        if case["n"] < 64:
            assert case["verdict_5pct"]["pass"] is False
            continue
        u_min = case["u"]["extremum"]["min_computed"][1]
        v_max = case["v"]["extremum"]["max_computed"][1]
        y_ref = case["u"]["extremum"]["min_reference"][0]
        x_ref = case["v"]["extremum"]["max_reference"][0]
        assert abs(u_min - ref_u[y_ref]) / abs(ref_u[y_ref]) < TOL
        assert abs(v_max - ref_v[x_ref]) / abs(ref_v[x_ref]) < TOL
        vx, vy = case["vortex_centre"]["computed"]
        gx, gy = VORTEX_CENTRE[1000.0]
        assert max(abs(vx - gx), abs(vy - gy)) < TOL


def test_two_resolutions_present_and_finer_is_not_worse():
    summary = _summary()
    grids = sorted(c["n"] for c in summary["cases"])
    assert len(grids) >= 2, f"needs two resolutions, have {grids}"
    fine = max(grids)
    row = {c["n"]: c for c in summary["cases"]}[fine]
    coarse = max(g for g in grids if g < fine)
    crow = {c["n"]: c for c in summary["cases"]}[coarse]
    # pointwise centreline error must not grow under refinement by more than the band
    assert row["u"]["max_abs_error"] <= crow["u"]["max_abs_error"] + TOL / 2
    assert row["v"]["max_abs_error"] <= crow["v"]["max_abs_error"] + TOL / 2


def test_report_tables_come_from_the_bundle():
    """The rendered report must quote the bundle, not a hand-typed copy."""
    report = os.path.join(ROOT, "docs", "CASE-CAVITY-RE1000-FVM", "report.md")
    if not os.path.exists(SUMMARY) or not os.path.exists(report):
        pytest.skip("report or bundle absent")
    summary = _summary()
    text = open(report, encoding="utf-8").read()
    for case in summary["cases"]:
        token = f"{case['u']['extremum']['min_computed'][1]:.5f}"
        assert token in text, f"report does not quote u_min={token} for n={case['n']}"