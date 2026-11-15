"""Invariants for the Woodward-Colella refinement closure (``verification.wc_refine``).

Two kinds of assertions live here on purpose:

* **synthetic invariants** -- conservative restriction must preserve mass and
  must be the identity at ratio 1; a known perturbation must come back as the
  same L1; the error model must invert.  These do not touch the stored ladder.
* **regression locks** -- the legacy interpolating metric must still reproduce
  the numbers published in ``CERTIFIED-REF-CLOSURE.md`` (1.776 % / 0.544 %), so
  the new module cannot silently disagree with the earlier verdict.  Missing
  ladder files make these skip rather than fail (a checkout without the
  artifacts is not a bug).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import wc_refine as wr

DATA = Path(__file__).resolve().parents[1] / "docs" / "woodward-colella-double-blast" / "ladder" / "data"

needs_3200 = pytest.mark.skipif(
    wr.try_load_profile(DATA, 3200, "hllc") is None
    or wr.try_load_profile(DATA, 12800, "weno5") is None,
    reason="stored Woodward-Colella ladder profiles not present in this checkout",
)


# --------------------------------------------------------------------------- #
# conservative restriction
# --------------------------------------------------------------------------- #
def test_restrict_preserves_mass_exactly():
    rng = np.random.default_rng(7)
    fine = rng.random(64) + 0.5
    for ratio in (2, 4, 8):
        coarse = wr.restrict(fine, ratio)
        assert coarse.size == fine.size // ratio
        # cell averages: mean(fine) == mean(coarse) to round-off
        assert math.isclose(coarse.mean(), fine.mean(), rel_tol=0, abs_tol=1e-14)


def test_restrict_of_constant_is_that_constant():
    assert np.allclose(wr.restrict(np.full(32, 3.25), 4), 3.25)


def test_restrict_ratio_one_is_identity():
    q = np.arange(10, dtype=float)
    assert np.array_equal(wr.restrict(q, 1), q)


def test_restrict_rejects_non_nested_grid():
    with pytest.raises(ValueError):
        wr.restrict(np.zeros(10), 3)
    with pytest.raises(ValueError):
        wr.restrict(np.zeros(8), 0)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_cell_average_l1_is_zero_for_a_nested_pair():
    rng = np.random.default_rng(11)
    fine = rng.random(400) + 1.0
    coarse = wr.restrict(fine, 4)
    out = wr.cell_average_l1(coarse, fine)
    assert out["l1_percent"] < 1e-13
    assert out["peak_percent"] < 1e-13
    assert out["mass_percent"] < 1e-13


def test_cell_average_l1_recovers_a_known_perturbation():
    """A uniform +delta offset must come back as exactly delta percent."""
    fine = np.ones(800) * 2.0
    coarse = wr.restrict(fine, 4) * 1.05          # +5 % everywhere inside the window
    out = wr.cell_average_l1(coarse, fine)
    assert out["l1_percent"] == pytest.approx(5.0, abs=1e-10)
    assert out["peak_percent"] == pytest.approx(5.0, abs=1e-10)


def test_cell_average_l1_rejects_non_nested_pair():
    with pytest.raises(ValueError):
        wr.cell_average_l1(np.zeros(100), np.zeros(400))


def test_window_limits_the_comparison():
    fine = np.ones(400)
    coarse = wr.restrict(fine, 4).copy()
    coarse[0:40] *= 3.0                           # x < 0.40 only, outside the window
    inside = wr.cell_average_l1(coarse, fine, window=(0.4, 1.0))
    whole = wr.cell_average_l1(coarse, fine, window=(0.0, 1.0))
    assert inside["l1_percent"] == pytest.approx(0.0, abs=1e-12)
    assert whole["l1_percent"] > 50.0


# --------------------------------------------------------------------------- #
# error model
# --------------------------------------------------------------------------- #
def test_observed_order_recovers_a_known_slope():
    e1, e2 = 4.0, 1.0
    assert wr.observed_order(e1, e2, 100, 200) == pytest.approx(2.0, abs=1e-12)


def test_forecast_and_inverse_are_consistent():
    e = wr.forecast_error(1.77, 3200, 0.88, 6400)
    assert e < 1.77
    nx = wr.nx_for_error(1.77, 3200, 0.88, e)
    assert nx == pytest.approx(6400.0, rel=1e-10)
    # already below the tolerance -> no refinement needed
    assert wr.nx_for_error(1.0, 3200, 0.9, 2.0) == 3200.0


def test_error_model_rejects_bad_input():
    with pytest.raises(ValueError):
        wr.observed_order(1.0, 2.0, 200, 100)      # not a refinement
    with pytest.raises(ValueError):
        wr.observed_order(0.0, 1.0, 100, 200)
    with pytest.raises(ValueError):
        wr.forecast_error(1.0, 100, -0.5, 200)


# --------------------------------------------------------------------------- #
# stored-ladder locks (compatibility with the published verdict)
# --------------------------------------------------------------------------- #
@needs_3200
def test_legacy_pointwise_metric_reproduces_published_l1():
    """CERTIFIED-REF-CLOSURE.md quotes 1.776 % (MUSCL@3200 vs WENO5@12800)."""
    prof = wr.load_profile(DATA, 3200, "hllc")
    ref = wr.load_profile(DATA, 12800, "weno5")
    out = wr.pointwise_l1(prof["x"], prof["rho"], ref["x"], ref["rho"])
    assert out["l1_percent"] == pytest.approx(1.776, abs=5e-3)
    assert out["peak_percent"] == pytest.approx(0.758, abs=5e-3)


@needs_3200
def test_metric_bias_is_below_the_predeclared_bound():
    """Prediction M1: interpolating vs restricting the reference differ <0.05 pp."""
    prof = wr.load_profile(DATA, 3200, "hllc")
    ref = wr.load_profile(DATA, 12800, "weno5")
    agg = wr.cell_average_l1(prof["rho"], ref["rho"])
    pw = wr.pointwise_l1(prof["x"], prof["rho"], ref["x"], ref["rho"])
    assert abs(pw["l1_percent"] - agg["l1_percent"]) < 0.05


@needs_3200
def test_reference_self_certification_reproduces_published_value():
    """WENO5@6400 vs @12800: 0.544 % L1, 0.023 % peak (gate 1 of the closure)."""
    if wr.try_load_profile(DATA, 6400, "weno5") is None:
        pytest.skip("WENO5@6400 rung not present")
    out = wr.reference_selfcert(DATA, 6400, 12800)
    assert out["l1_percent"] == pytest.approx(0.544, abs=5e-3)
    assert out["peak_percent"] == pytest.approx(0.023, abs=5e-3)
    assert out["self_certified"] is True


@needs_3200
def test_audit_reports_missing_rungs_instead_of_raising():
    out = wr.audit(DATA)
    assert out["ladder"], "graded ladder must not be empty"
    for row in out["ladder"]:
        if not row["measured"]:
            assert row["nx"] in out["not_measured"]
    # every measured rung must be graded against the same reference grid
    assert out["reference"]["nx"] == 12800
    assert out["certified_bar_percent"] == 1.0
    assert out["ledger_tolerance_percent"] == 15.0


@needs_3200
def test_metric_choice_does_not_flip_the_verdict():
    out = wr.audit(DATA)
    assert out["metric_decision_invariant"] is True


def test_missing_profile_returns_none():
    assert wr.try_load_profile(DATA, 10_000_000, "hllc") is None

# --------------------------------------------------------------------------- #
# residual attribution
# --------------------------------------------------------------------------- #
def test_residual_density_shares_integrate_to_the_l1():
    rng = np.random.default_rng(3)
    fine = rng.random(800) + 1.0
    coarse = wr.restrict(fine, 4).copy()
    coarse[60:] *= 1.02
    agg = wr.cell_average_l1(coarse, fine)
    dens = wr.residual_density(coarse, fine)
    assert 100.0 * float(np.sum(dens["share"])) == pytest.approx(agg["l1_percent"], rel=1e-12)


def test_region_shares_sum_to_one_hundred_percent():
    rng = np.random.default_rng(5)
    fine = rng.random(400) + 1.0
    coarse = wr.restrict(fine, 4) * 1.01
    dens = wr.residual_density(coarse, fine)
    regions = [("a", (0.40, 0.70)), ("b", (0.70, 1.00))]
    parts = wr.region_shares(dens, regions)
    assert sum(p["share_percent_of_l1"] for p in parts) == pytest.approx(100.0, rel=1e-12)


@needs_3200
def test_residual_lives_on_the_fronts_not_on_the_plateau():
    """The 3200 residual is front-localised: the smooth left plateau carries <1 %.

    This is the quantitative form of "the remaining L1 is the cost of resolving
    the two steep fronts", and it is what makes a refinement forecast meaningful.
    """
    data = wr.default_data_dir()
    prof = wr.load_profile(data, 3200, "hllc")
    ref = wr.load_profile(data, 12800, "weno5")
    dens = wr.residual_density(prof["rho"], ref["rho"])
    parts = wr.region_shares(dens, [("left_plateau", (0.40, 0.55)),
                                    ("rest", (0.55, 1.00))])
    by = {p["region"]: p["share_percent_of_l1"] for p in parts}
    assert by["left_plateau"] < 1.0
    assert by["rest"] > 99.0
