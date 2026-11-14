"""Regression tests for the m20007 audit claims (CASE-FVM-CHANNEL-DEVELOP).

Invariants asserted (not merely "code executes"):

A1  for the delivered n_y = 8 the 99 % threshold 1.485 u_bar lies above the
    cell-centre ceiling of the exact parabola (1.4766) -> the criterion cannot
    be met by construction, independent of the solver;
A2  the ceiling crosses 1.485 only for n_y >= 10 (1.5 - 1.5/n_y^2), so n_y = 8
    is the special case the delivered PLAN locked in (ny == nx_per_H);
A3  every delivered L_e/H row equals the no-crossing fallback L/H - dx/2, i.e.
    no entrance length was measured;
A4  the delivered bundle is internally inconsistent (CSV truncated to one case);
A5  the delivered "dp/dx ref" column disagrees with the module's own formula
    -12 nu ubar / H^2 for Re_H = 50 and 100 (reproduced: -0.2328 / -0.24),
    i.e. those artifact rows cannot have been produced by the delivered source;
A6  the delivered source still contains the fallback path and the hard-coded
    0.06 coefficient.

Run:  python3 -m pytest docs/audit_m20007/test_audit_claims.py -q
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "audited_runs"
SRC = HERE / "audited_src/benchmark_entrance_length.py"
ART = HERE / "artifacts"
THRESHOLD = 0.99 * 1.5


def parabola_centre_ceiling(ny: int) -> float:
    y = (np.arange(ny) + 0.5) / ny
    return float((6.0 * y * (1.0 - y)).max())


def test_ny8_ceiling_below_threshold():
    assert parabola_centre_ceiling(8) == pytest.approx(1.4765625, abs=1e-12)
    assert parabola_centre_ceiling(8) < THRESHOLD


def test_ceiling_crosses_threshold_only_at_ny_ge_10():
    even = [ny for ny in range(2, 65) if ny % 2 == 0]
    crossing = [ny for ny in even if parabola_centre_ceiling(ny) >= THRESHOLD]
    assert min(crossing) == 10                     # even ny < 10 stay below 1.485
    assert parabola_centre_ceiling(8) < THRESHOLD
    for ny in even:
        assert parabola_centre_ceiling(ny) == pytest.approx(1.5 - 1.5 / ny ** 2, abs=1e-12)
    for ny in (9, 17, 33):                          # odd ny samples y=H/2 exactly
        assert parabola_centre_ceiling(ny) == pytest.approx(1.5, abs=1e-12)


def test_midplane_never_sampled_for_even_ny():
    for ny in (8, 16, 32):
        y = (np.arange(ny) + 0.5) / ny
        assert abs(y[ny // 2] - 0.5) == pytest.approx(0.5 / ny, abs=1e-12)


def test_all_delivered_rows_are_domain_end_sentinels():
    bench = json.load(open(RUNS / "benchmark.json"))
    assert bench["verdict"] == "FAIL"
    for case in bench["cases"]:
        dx = case["L/H"] / case["nx"]
        sentinel = case["L/H"] - 0.5 * dx
        assert case["L_e/H (computed)"] == pytest.approx(sentinel, abs=1e-9), case["name"]
        assert case["L_e/H error %"] > 15.0, case["name"]   # above the platform floor too
        assert case["u_max/u_bar"] < THRESHOLD, case["name"]


def test_delivered_bundle_is_inconsistent():
    bench = json.load(open(RUNS / "benchmark.json"))
    with open(RUNS / "metrics.csv") as f:
        metrics = list(csv.DictReader(f))
    with open(RUNS / "centerline_profiles.csv") as f:
        prof_cases = {r["case"] for r in csv.DictReader(f)}
    assert len(bench["cases"]) == 3
    assert len(metrics) < len(bench["cases"]), "CSV truncation defect is gone -- update the audit"
    assert len(prof_cases) < len(bench["cases"])


def test_delivered_dpdx_ref_disagrees_with_module_formula():
    """Auditor re-run of the delivered source gives -0.2328 vs ref -0.24 (Re_H=50)."""
    bench = json.load(open(RUNS / "benchmark.json"))
    repro50 = json.load(open(ART / "repro50/benchmark.json"))["cases"][0]
    formula = {50.0: -0.24, 100.0: -0.12, 200.0: -0.06}   # -12 nu ubar / H^2, ubar=H=1
    delivered = {c["re_h"]: c for c in bench["cases"]}
    for re_h, ref in formula.items():
        assert delivered[re_h]["dp/dx ref"] == pytest.approx(ref, rel=1e-9) \
            if re_h == 200.0 else delivered[re_h]["dp/dx ref"] != pytest.approx(ref, rel=1e-6)
    assert repro50["dp/dx"] == pytest.approx(-0.232838, abs=1e-6)
    assert delivered[50.0]["dp/dx"] == pytest.approx(-0.0296, abs=1e-6)
    # the delivered row understates |dp/dx| by ~7.9x -> not reproducible from the source
    assert abs(repro50["dp/dx"] / delivered[50.0]["dp/dx"]) > 7.0


def test_fallback_path_and_hardcoded_coefficient_present():
    src = SRC.read_text()
    assert "if le_over_h is None:" in src
    assert "le_over_h = centerline_x_vals[-1]" in src
    assert "WHITE_COEFF = 0.06" in src


def test_developed_region_physics_ok_when_measured_correctly():
    """Auditor re-run: |dp/dx| error vs the analytic Poiseuille value is < 3 %."""
    repro50 = json.load(open(ART / "repro50/benchmark.json"))["cases"][0]
    repro200 = json.load(open(ART / "repro/benchmark.json"))["cases"][0]
    for row in (repro50, repro200):
        assert row["dp/dx error %"] < 3.0, row["name"]