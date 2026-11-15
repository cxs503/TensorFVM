"""Tests for the periodic-hill grid/time-step convergence attributor (round 22).

The point of these tests is that the module's *verdicts* are pinned by synthetic
sequences whose answer is known analytically -- an attribution rule that only
reproduces the run it was written for would be worthless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.verification.periodic_hill_grid_ladder import (
    ASYMPTOTE_OUTSIDE_BAND,
    DISCRETISATION_ATTRIBUTABLE,
    INCONCLUSIVE,
    ORDER_UNDEFINED,
    attribute,
    combined_span,
    gci_fine,
    observed_order,
    refinement_ratio,
    richardson_extrapolation,
    temporal_sensitivity,
)


def synthetic(h, x_inf, coefficient, order):
    """``x_inf + coefficient * h**order`` -- the exact asymptotic sequence."""
    return x_inf + coefficient * h ** order


# --------------------------------------------------------------------------
# elementary estimators
# --------------------------------------------------------------------------
def test_refinement_ratio_rejects_non_refinement():
    assert refinement_ratio(0.094, 0.047) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        refinement_ratio(0.05, 0.10)
    with pytest.raises(ValueError):
        refinement_ratio(0.0, 0.1)


@pytest.mark.parametrize("order", [1.0, 1.5, 2.0, 3.0])
def test_observed_order_recovers_exact_order(order):
    r = 2.0
    # observe_order() takes fine -> medium -> coarse
    vals = [synthetic(1.0 / r ** i, 5.41, 3.0, order) for i in (2, 1, 0)]
    got = observed_order(vals, r)
    assert got["status"] == "order_estimated"
    assert got["p"] == pytest.approx(order, rel=1e-10)


def test_observed_order_undefined_for_non_monotone_and_flat():
    non_monotone = observed_order([5.0, 6.0, 5.5], 2.0)
    assert non_monotone["p"] is None
    assert non_monotone["status"] == ORDER_UNDEFINED
    assert "non-monotone" in non_monotone["reason"]

    flat = observed_order([5.0, 5.0, 5.0], 2.0)
    assert flat["p"] is None
    assert "zero difference" in flat["reason"]

    with pytest.raises(ValueError):
        observed_order([1.0, 2.0], 2.0)


def test_richardson_recovers_limit_and_rejects_bad_order():
    r = 2.0
    coarse = synthetic(1.0 / r, 5.41, 2.0, 2.0)
    fine = synthetic(1.0 / r ** 2, 5.41, 2.0, 2.0)
    assert richardson_extrapolation(fine, coarse, r, 2.0) == pytest.approx(5.41)
    with pytest.raises(ValueError):
        richardson_extrapolation(fine, coarse, r, 0.0)


def test_gci_shrinks_with_refinement():
    r = 2.0
    h = 1.0 / r
    g_1 = gci_fine(synthetic(h, 5.41, 2.0, 2.0),
                   synthetic(2 * h, 5.41, 2.0, 2.0), r, 2.0)
    g_2 = gci_fine(synthetic(h / 2, 5.41, 2.0, 2.0),
                   synthetic(h, 5.41, 2.0, 2.0), r, 2.0)
    assert 0.0 < g_2 < g_1
    # for a pure h**p sequence the fine-level GCI equals Fs * C h^p / |x_fine|
    expected = 1.25 * 2.0 * (h / 2) ** 2 / synthetic(h / 2, 5.41, 2.0, 2.0)
    assert g_2 == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# the pre-registered attribution rule
# --------------------------------------------------------------------------
def test_attribute_discretisation_attributable():
    """Pure first-order discretisation of a 5.41 limit must be attributed to it."""
    ladder = [(1.0, synthetic(1.0, 5.41, 8.0, 1.0)),
              (0.5, synthetic(0.5, 5.41, 8.0, 1.0)),
              (0.25, synthetic(0.25, 5.41, 8.0, 1.0))]
    out = attribute(ladder, 5.41, tolerance=0.10)
    assert out["order"] == pytest.approx(1.0)
    assert out["x_extrapolated"] == pytest.approx(5.41)
    assert out["verdict"] == DISCRETISATION_ATTRIBUTABLE
    assert out["error_extrapolated_rel"] == pytest.approx(0.0, abs=1e-12)


def test_attribute_asymptote_outside_band():
    """A converged ladder that lands far from the reference is NOT discretisation."""
    ladder = [(1.0, synthetic(1.0, 8.26, 0.6, 2.0)),
              (0.5, synthetic(0.5, 8.26, 0.6, 2.0)),
              (0.25, synthetic(0.25, 8.26, 0.6, 2.0))]
    out = attribute(ladder, 5.41, tolerance=0.10)
    assert out["verdict"] == ASYMPTOTE_OUTSIDE_BAND
    assert out["x_extrapolated"] == pytest.approx(8.26)
    assert out["movement_rel"] < 0.10
    assert out["error_extrapolated_rel"] > 0.10


def test_attribute_inconclusive_while_still_moving():
    ladder = [(1.0, synthetic(1.0, 9.0, 5.0, 1.0)),
              (0.5, synthetic(0.5, 9.0, 5.0, 1.0)),
              (0.25, synthetic(0.25, 9.0, 5.0, 1.0))]
    out = attribute(ladder, 5.41, tolerance=0.10)
    assert out["verdict"] == INCONCLUSIVE
    assert "extend the ladder" in out["rule"]


def test_attribute_order_undefined_blocks_a_verdict():
    ladder = [(1.0, 8.0), (0.5, 9.0), (0.25, 8.5)]
    out = attribute(ladder, 5.41, tolerance=0.10)
    assert out["verdict"] == ORDER_UNDEFINED
    assert out["x_extrapolated"] is None


def test_attribute_two_levels_cannot_claim_discretisation():
    ladder = [(1.0, synthetic(1.0, 5.41, 8.0, 1.0)),
              (0.5, synthetic(0.5, 5.41, 8.0, 1.0))]
    out = attribute(ladder, 5.41, tolerance=0.10)
    assert out["verdict"] != DISCRETISATION_ATTRIBUTABLE
    assert out["x_extrapolated"] is None
    assert out["n_levels"] == 2


def test_attribute_rejects_non_uniform_ladder():
    # ratios 2.0 then 2.5: a single order p has no meaning on this ladder
    out = attribute([(1.0, 9.0), (0.5, 8.0), (0.2, 7.5)], 5.41, tolerance=0.10)
    assert out["rule"] == "non-uniform refinement ratios"
    assert out["verdict"] == INCONCLUSIVE


def test_attribute_single_level_and_non_finite():
    assert attribute([(1.0, 8.0)], 5.41)["verdict"] == INCONCLUSIVE
    assert attribute([(1.0, math.nan), (0.5, 8.0)], 5.41)["rule"] == "non-finite level"


# --------------------------------------------------------------------------
# time-step axis
# --------------------------------------------------------------------------
def test_temporal_sensitivity_flags_divergence():
    ok = temporal_sensitivity(8.264, 8.31)
    assert ok["resolved"] is True
    assert ok["relative"] == pytest.approx(abs(8.31 - 8.264) / 8.264)

    diverged = temporal_sensitivity(float("nan"), 8.31)
    assert diverged["resolved"] is False
    assert diverged["relative"] == math.inf


def test_combined_span_ignores_nan():
    assert combined_span([8.1, math.nan, 8.4, None]) == pytest.approx(0.3, abs=1e-12)


def test_attribute_tolerance_must_be_positive():
    with pytest.raises(ValueError):
        attribute([(1.0, 9.0), (0.5, 8.0)], 5.41, tolerance=0.0)