"""Guards for the Lamb-ellipsoid added-mass benchmark (`benchmarks/added_mass_lamb`).

What each assertion protects:

* **anchor exactness** -- the analytic anchor of the benchmark must return the
  classical sphere value ``Cm = 0.5`` and satisfy the Lamb sum rule
  ``alpha0+beta0+gamma0 = 2``; the benchmark's criterion 1 is only meaningful if
  the anchor itself is right;
* **two independent routes** -- closed form vs defining integral for prolate and
  oblate spheroids, including the degenerate aspect ratios where the integral
  route used to collapse (regression guard: the pre-fix route gave
  ``alpha0+beta0+gamma0 = 0.53`` instead of 2 for ``c/a = 1e-3``);
* **the solver still converges** -- one coarse and one medium sphere in the
  benchmark's own configuration (``gauss`` quadrature): the error must shrink
  and the medium mesh must already be inside the benchmark's 5 % band;
* **the driver end-to-end** -- a two-level smoke run writes the three artefacts
  and the report, with one CSV row per (case, level, axis), so the published
  artefacts really come from this code path.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "added_mass_lamb"


@pytest.fixture(scope="module")
def bench():
    """The benchmark driver, imported from the repo (not an installed copy)."""
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import run_benchmark

    return run_benchmark


def test_sphere_anchor_is_exact(bench):
    """Sphere: Lamb factors 2/3 and ``k = 0.5`` on every axis, to round-off."""
    entry = bench.anchor_case(1.0, 1.0, 1.0)
    assert entry["alpha0_beta0_gamma0"] == pytest.approx([2.0 / 3.0] * 3, rel=1e-15)
    for k in entry["k"]:
        assert k == pytest.approx(0.5, rel=1e-14)
    # M_11 must equal the independent closed-form sphere helper
    assert entry["M_diag"][0] == pytest.approx(0.5 * 4.0 / 3.0 * math.pi, rel=1e-14)


def test_anchor_limits_all_pass(bench):
    """Criterion 1 must be reproducible from the anchor functions alone."""
    limits = bench.limit_ladders()
    assert {r["verdict"] for r in limits} == {"PASS"}
    sphere = [r for r in limits if r["limit"] == "sphere"]
    assert max(r["worst_deviation_pct"] for r in sphere) < 1e-10   # far inside 0.1 %
    prolate = next(r for r in limits if r["limit"] == "slender prolate")
    axial = [e["k_axial"] for e in prolate["value"]]
    assert axial == sorted(axial, reverse=True) and axial[-1] < 0.01
    oblate = next(r for r in limits if r["limit"] == "flat oblate disc")
    disc_ratio = [e["M_normal_over_disc"] for e in oblate["value"]]
    assert disc_ratio == sorted(disc_ratio) and abs(disc_ratio[-1] - 1.0) < 0.01


@pytest.mark.parametrize("a,b,c", [
    (1.5, 1.0, 1.0),      # prolate, closed form (4)
    (1.0, 1.0, 0.25),     # oblate, closed form (5)
    (1.0, 1.0, 0.001),    # flat oblate, degenerate limit
    (2.0, 1.0, 0.5),      # general triaxial, integral route only
])
def test_anchor_routes_agree_and_sum_rule(a, b, c):
    """Closed form and defining integral are independent routes on the same value."""
    from tensorfvm.panel_added_mass import lamb_ellipsoid_factors, lamb_integral_factors

    factors = lamb_ellipsoid_factors(a, b, c)
    integral = lamb_integral_factors(a, b, c)
    assert sum(factors) == pytest.approx(2.0, abs=1e-9)
    assert sum(integral) == pytest.approx(2.0, abs=1e-6)
    # tolerance: the two routes must agree well inside the 5 % engineering band
    assert max(abs(x - y) for x, y in zip(factors, integral)) < 1e-4
    assert all(0.0 < f < 2.0 for f in factors)


def test_integral_route_survives_degenerate_axes():
    """Regression: the log-mapped integral must not lose the flat-body term.

    With the previous ``lambda = u/(1-u)`` mapping the flattened factor was lost
    as soon as the axes differed by two decades -- ``sum(lamb_integral_factors(1,
    1, 1e-3))`` returned 0.53, i.e. a 73 % error on a quantity that must equal 2.
    """
    from tensorfvm.panel_added_mass import lamb_integral_factors

    gammas = []
    for flat in (1e-2, 1e-3, 1e-4):
        factors = lamb_integral_factors(1.0, 1.0, flat)
        assert sum(factors) == pytest.approx(2.0, abs=1e-6), flat
        gammas.append(factors[2])
    # gamma0 rises monotonically towards 2 as the body flattens (2 - gamma0 ~ O(c/a))
    assert gammas == sorted(gammas) and gammas[0] > 1.9


def test_sphere_panel_error_shrinks_and_enters_the_band(bench):
    """One coarse and one medium sphere in the benchmark's own configuration."""
    coarse = bench.run_level("sphere_1_1_1", "sphere", 1.0, 1.0, 1.0, 12, 8)
    medium = bench.run_level("sphere_1_1_1", "sphere", 1.0, 1.0, 1.0, 24, 18)
    k_exact = bench.anchor_case(1.0, 1.0, 1.0)["k"]
    dev_coarse = max(abs(k / ke - 1.0) for k, ke in zip(coarse["k_panel"], k_exact))
    dev_medium = max(abs(k / ke - 1.0) for k, ke in zip(medium["k_panel"], k_exact))
    assert dev_medium < dev_coarse
    assert dev_medium * 100.0 < bench.TOL_PCT
    # geometry is validated first: the inscribed latitude/longitude polyhedron
    # under-predicts the volume (24x18 sphere: ~1.9 %), and that faceting deficit is
    # the same error family as the added-mass deficit -- so it is bounded, not zero.
    assert abs(medium["volume_rel_error"]) < 2.5e-2
    assert abs(medium["volume_rel_error"]) < abs(coarse["volume_rel_error"])
    assert all(abs(e) < 1e-9 for e in medium["axis_extent_rel_error"])
    # the discrete matrix is diagonal in the principal frame
    assert medium["offdiag_rel"] < 1e-9


def test_driver_writes_all_artefacts(bench, tmp_path):
    """Smoke run: two levels, all artefacts, no hand-written numbers."""
    rc = bench.main(["--out", str(tmp_path), "--levels", "8x6,12x8",
                     "--no-quad-study"], )
    assert rc in (0, 1)                      # 1 = a criterion failed, still no crash
    for name in ("anchors.json", "ellipsoid_scan.csv", "convergence.json", "README.md"):
        assert (tmp_path / name).stat().st_size > 0, name

    anchors = json.loads((tmp_path / "anchors.json").read_text())
    assert anchors["anchor_id"] == "AN-LAMB-ELLIPSOID"
    assert anchors["criterion_1_verdict"] == "PASS"

    with (tmp_path / "ellipsoid_scan.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(bench.CASES) * 2 * 3
    assert {r["pass_5pct"] for r in rows} <= {"PASS", "FAIL"}

    conv = json.loads((tmp_path / "convergence.json").read_text())
    assert [c["id"] for c in conv["criteria"]] == [1, 2, 3, 4, 5]
    crit5 = next(c for c in conv["criteria"] if c["id"] == 5)
    assert crit5["verdict"] == "PASS"
    assert conv["worst"] is not None
    readme = (tmp_path / "README.md").read_text()
    assert "criterion" in readme and "Singular integral" in readme