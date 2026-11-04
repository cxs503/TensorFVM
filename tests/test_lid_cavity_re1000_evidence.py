"""Regression gate for the converged Ghia ladder (verification debt repayment).

Reads the committed evidence bundle ``docs/verification-lid-cavity-ladder/summary.json``;
it does not re-run the solver (that costs minutes to an hour per row).  Pinned here:

* every row reached the steady-state gate instead of hitting the step cap,
* the projection stays divergence free at every resolution,
* both primary extrema and the u-centreline are inside 3 % on every grid that
  resolves the lid shear layer (n >= 64),
* the defect the ladder exposed -- the Re = 400 v-centreline extraction, whose error
  grows under refinement -- stays visible as a strict xfail instead of a green suite.

A live smoke run of the same code path lives in ``tests/test_lid_cavity.py``.
"""

from __future__ import annotations

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "docs", "verification-lid-cavity-ladder", "summary.json")


def _load() -> dict:
    if not os.path.exists(BUNDLE):  # pragma: no cover - evidence checkout policy
        pytest.skip("evidence bundle not present in this checkout")
    with open(BUNDLE) as handle:
        return json.load(handle)


def _cases() -> dict[tuple[float, int], dict]:
    return {(c["reynolds"], c["n"]): c for c in _load()["cases"]}


RESOLVED = [(rey, n) for rey in (100.0, 400.0, 1000.0) for n in (64, 128)]


def test_ladder_covers_the_debt_row_and_the_three_reynolds():
    cases = _cases()
    assert (1000.0, 64) in cases, "the Re=1000 row this bundle exists for is missing"
    for rey in (100.0, 400.0, 1000.0):
        assert (rey, 64) in cases, f"Re={rey:g} missing the n=64 row"


def test_no_row_hit_the_step_cap_instead_of_the_tolerance():
    meta = _load()["meta"]
    for (rey, n), case in _cases().items():
        solver = case["solver"]
        assert solver["converged"], f"Re={rey:g} n={n} did not converge"
        assert solver["residual_rate"] <= meta["tolerance"], (rey, n, solver["residual_rate"])
        assert solver["steps"] < meta["max_steps"], (
            f"Re={rey:g} n={n} consumed the whole cap {meta['max_steps']}")


def test_every_row_is_divergence_free():
    for (rey, n), case in _cases().items():
        assert case["max_divergence"] < 1e-11, (rey, n, case["max_divergence"])


@pytest.mark.parametrize("reynolds,n", [p for p in RESOLVED if p[1] >= 64])
def test_u_centreline_and_both_extrema_are_inside_three_percent(reynolds, n):
    key = (reynolds, n)
    if key not in _cases():
        pytest.skip(f"Re={reynolds:g} n={n} not in this bundle")
    case = _cases()[key]
    u_ext, v_ext = case["u"]["extremum"], case["v"]["extremum"]
    u_rel = abs(u_ext["min_computed"][1] - u_ext["min_reference"][1]) / abs(u_ext["min_reference"][1])
    v_rel = abs(v_ext["max_computed"][1] - v_ext["max_reference"][1]) / abs(v_ext["max_reference"][1])
    assert u_rel < 0.03, (reynolds, n, u_rel)
    assert v_rel < 0.03, (reynolds, n, v_rel)
    assert case["u"]["max_abs_error"] < 0.03, (reynolds, n, case["u"]["max_abs_error"])


def test_no_row_that_resolves_the_shear_layer_loses_the_extrema():
    for (rey, n), case in _cases().items():
        if n < 64:
            continue
        assert case["u"]["extremum"]["min_computed"][1] < 0.0, (rey, n)   # reverse flow exists
        assert case["v"]["extremum"]["max_computed"][1] > 0.0, (rey, n)


def test_extrema_are_grid_converged_between_n64_and_n128():
    cases = _cases()
    for rey in (100.0, 400.0, 1000.0):
        if (rey, 128) not in cases:
            continue
        coarse, fine = cases[(rey, 64)], cases[(rey, 128)]
        u_c = coarse["u"]["extremum"]["min_computed"][1]
        u_f = fine["u"]["extremum"]["min_computed"][1]
        v_c = coarse["v"]["extremum"]["max_computed"][1]
        v_f = fine["v"]["extremum"]["max_computed"][1]
        assert abs(u_f - u_c) / abs(u_f) < 0.05, (rey, u_c, u_f)
        assert abs(v_f - v_c) / abs(v_f) < 0.05, (rey, v_c, v_f)


@pytest.mark.xfail(strict=True, reason=(
    "Documented open question (docs/verification-lid-cavity-ladder/report.md 5.1): at "
    "Re=400 the v-centreline error grows under refinement (0.115/0.142/0.149 for "
    "n=32/64/128) while u and both extrema converge. Attribution updated by "
    "scripts/audit_lid_cavity_v_centreline.py: the extraction path is exonerated -- on "
    "an analytic staggered field it reproduces the profile to 1.5e-4 relative "
    "(observed order 4.5 in h, 1000x below the recorded defect, and a half-cell "
    "sampling error is detected at 100x), while a fresh n=32 run localises the "
    "largest node error at x/L=0.9688, i.e. beside the right wall. So the defect is "
    "in the stored solution field next to the wall, not in the extraction. Must be "
    "resolved, not tolerated."))
def test_re400_v_centreline_discrepancy_is_still_present_and_unexplained():
    row = _cases()[(400.0, 64)]
    assert row["acceptance"]["pass"], row["acceptance"]["criteria"]
