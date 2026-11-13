"""Self-verification of the three-grid log-log-slope order readout for cell-centred FV.

Three groups of assertions:

``metric algebra``
    the exact cell average agrees with quadrature, the log-log slope recovers a
    pure power law to round-off (uniform *and* non-uniform ladders), and the
    log-log slope and the exact three-grid solve are the same number on such a
    series -- so the two estimators are proven to be reading the same thing
    before either of them is used on solver output.
``negative controls``
    a synthetic cell field of true order 3 must be read as 3 by the cell-average
    metric and as ~2 by the centroid-point metric (the sampling floor); the
    textbook uniform-ratio formula must be wrong on a non-uniform ladder while
    the exact solve is right.
``solver facts``
    the merged cell-centred solver (``EulerSolver``, MUSCL-Hancock + HLLC)
    recovers its declared order 2 (and 1) within the owner's 3 % band on the L1
    cell-average metric, and the centroid sampling bias is a measurable fraction
    of that error.  These tests pin what the code delivers; they do not ask for a
    different order.
"""
from functools import lru_cache

import numpy as np
import pytest

from tensorfvm.verification import cellcentre_loglog as cc
from tensorfvm.verification.richardson import power_law_series, richardson_gci

UNIFORM = (40, 80, 160)
NONUNIFORM = (50, 100, 175)


@lru_cache(maxsize=None)
def _ladder(grids, order):
    return cc.ladder(grids, order=order)


def _l1(rows, metric, norm="l1"):
    return [float(row[metric][norm]) for row in rows]


# ---------------------------------------------------------------- metric algebra
def test_exact_cell_average_matches_quadrature_and_conserves_mass():
    for nx in (8, 20, 40):
        analytic = cc.exact_cell_average(nx, 0.2)
        numeric = cc.cell_average_quadrature(nx, 0.2, subcell=256)
        assert np.max(np.abs(analytic - numeric)) < 1e-7
        # a full period of the density wave: the volume average is exactly 1
        assert abs(analytic.mean() - 1.0) < 1e-14


def test_sampling_difference_between_point_and_average_is_second_order():
    for nx in (20, 40, 80, 160):
        h = 1.0 / nx
        bias = np.abs(cc.exact_point(nx, 0.2) - cc.exact_cell_average(nx, 0.2)).mean()
        # closed-form leader of the restriction error: A k^2 h^2 / 24 * <|sin|>
        predicted = cc.AMPLITUDE * cc.WAVENUMBER ** 2 * h ** 2 / 24.0 * (2 / np.pi)
        assert bias == pytest.approx(predicted, rel=0.05)


@pytest.mark.parametrize("grids", [UNIFORM, NONUNIFORM])
@pytest.mark.parametrize("order", [1.0, 1.5, 2.0, 3.0, 5.0])
def test_loglog_slope_recovers_a_pure_power_law(grids, order):
    h = cc.spacings(grids)
    e = power_law_series(0.0, 3e-3, order, h)
    assert cc.loglog_slope(h, e) == pytest.approx(order, abs=1e-12)
    assert cc.adjacent_slopes(h, e) == pytest.approx([order] * 2, abs=1e-12)


@pytest.mark.parametrize("grids", [UNIFORM, NONUNIFORM])
def test_loglog_slope_and_three_grid_solve_agree_on_a_power_law(grids):
    h = cc.spacings(grids)
    e = power_law_series(0.0, 3e-3, 2.0, h)
    read = cc.three_grid_readout(grids, e, theory_order=2.0)
    assert read["p_loglog"] == pytest.approx(2.0, abs=1e-9)
    assert read["p_richardson"] == pytest.approx(2.0, abs=1e-9)
    assert read["loglog_pass_3"] is True


def test_readout_reports_a_deviation_instead_of_hiding_it():
    h = cc.spacings(UNIFORM)
    e = power_law_series(0.0, 3e-3, 1.4, h)
    read = cc.three_grid_readout(UNIFORM, e, theory_order=2.0)
    assert read["loglog_rel_dev"] == pytest.approx(0.30, abs=1e-9)
    assert read["loglog_pass_15"] is False and read["loglog_pass_3"] is False
    # ... and it is flagged against the order actually observed, too
    assert read["p_loglog"] == pytest.approx(1.4, abs=1e-9)


def test_error_norms_are_volume_weighted():
    widths = np.array([0.5, 0.25, 0.25])
    computed = np.array([1.0, 2.0, 3.0])
    norms = cc.error_norms(computed, np.zeros(3), widths)
    assert norms["l1"] == pytest.approx(0.5 * 1 + 0.25 * 2 + 0.25 * 3)
    assert norms["l2"] == pytest.approx(np.sqrt(0.5 * 1 + 0.25 * 4 + 0.25 * 9))
    assert norms["linf"] == pytest.approx(3.0)
    # an equal-weight mean would be 2.0 here, i.e. a different statistic
    assert norms["l1"] != pytest.approx(2.0)


# ------------------------------------------------------------- negative controls
@pytest.mark.parametrize("true_order", [2.0, 3.0, 4.0])
def test_centroid_metric_floors_the_observed_order_at_two(true_order):
    """A synthetic cell field of true order p: only the average metric sees p."""
    coefficient = 10.0
    rows = [_synthetic(nx, true_order, coefficient) for nx in UNIFORM]
    h = cc.spacings(UNIFORM)
    p_cellavg = cc.loglog_slope(h, [row["cell_average"]["l1"] for row in rows])
    p_point = cc.loglog_slope(h, [row["point"]["l1"] for row in rows])
    assert p_cellavg == pytest.approx(true_order, rel=2e-3)
    if true_order > 2.0:
        assert p_point < min(true_order - 0.05, 2.05)


def _synthetic(nx, order, coefficient):
    h = 1.0 / nx
    average = cc.exact_cell_average(nx, 0.2)
    point = cc.exact_point(nx, 0.2)
    x = (np.arange(nx) + 0.5) * h
    field = average + coefficient * h ** order * np.sin(cc.WAVENUMBER * (x - cc.VELOCITY * 0.2))
    widths = np.full(nx, h)
    return {"cell_average": cc.error_norms(field, average, widths),
            "point": cc.error_norms(field, point, widths)}


def test_uniform_ratio_formula_is_exact_on_a_uniform_ladder():
    h = cc.spacings(UNIFORM)
    e = power_law_series(0.0, 3e-3, 2.0, h)
    bias = cc.uniform_ratio_bias(UNIFORM, e, theory_order=2.0)
    assert bias["uniform_ladder"] is True
    assert bias["order_rel_dev"] < 1e-12
    # the metric is an error norm: its limit is zero, so the limit comparison is
    # made in absolute terms (a relative one would be withheld as nan)
    assert bias["limit_relative_meaningful"] is False
    assert bias["limit_abs_diff"] < 1e-12 * bias["limit_scale"]


def test_uniform_ratio_formula_is_wrong_on_a_nonuniform_ladder():
    """Even for a pure C h^p series the single-r formula is off by >15 %."""
    h = cc.spacings(NONUNIFORM)
    e = power_law_series(0.0, 3e-3, 2.0, h)
    bias = cc.uniform_ratio_bias(NONUNIFORM, e, theory_order=2.0)
    assert bias["uniform_ladder"] is False
    assert bias["p_true"] == pytest.approx(2.0, abs=1e-9)
    assert bias["order_rel_dev"] > 0.15
    assert bias["order_rel_dev_vs_theory"] > 0.15
    # the same mistake displaces the extrapolated (zero) limit by >1 % of the
    # error scale; on the uniform ladder it was 1e-13 of it
    assert bias["limit_abs_diff_over_scale"] > 0.01


def test_consistency_gate_rejects_a_non_monotone_triplet():
    good = cc.three_grid_readout(UNIFORM, [3e-3, 7.5e-4, 1.9e-4], theory_order=2.0)
    bad = cc.three_grid_readout(UNIFORM, [3e-3, 7.5e-4, 8.0e-4], theory_order=2.0)
    gate = cc.consistency(good, bad)
    assert gate["pass_15"] is False
    assert not gate["rel_diff"] < 1.0


# ------------------------------------------------------------------ solver facts
def test_solver_recovers_its_declared_order_on_the_cell_average_metric():
    for order in (1, 2):
        rows = _ladder(UNIFORM, order)
        read = cc.three_grid_readout(UNIFORM, _l1(rows, "cell_average"),
                                    theory_order=float(order))
        assert read["p_loglog"] == pytest.approx(float(order), rel=0.03), read
        assert read["p_richardson"] == pytest.approx(float(order), rel=0.10), read
        assert read["loglog_pass_3"] is True


def test_sampling_bias_is_a_measurable_fraction_of_the_finest_error():
    rows = _ladder(UNIFORM, 2)
    finest = rows[-1]
    ratio = finest["sampling_bias"]["l1"] / finest["cell_average"]["l1"]
    assert 0.2 < ratio < 0.6, ratio
    # the point metric is not merely a smaller number: it reads a higher order
    p_cell = cc.loglog_slope(cc.spacings(UNIFORM), _l1(rows, "cell_average"))
    p_point = cc.loglog_slope(cc.spacings(UNIFORM), _l1(rows, "point"))
    assert p_point - p_cell > 0.02


def test_solver_run_reports_the_time_it_actually_reached():
    rows = _ladder(UNIFORM, 2)
    assert all(row["steps"] > 0 and row["t"] > 0 for row in rows)
    assert rows[0]["steps"] < rows[-1]["steps"]


def test_gci_band_of_the_finest_uniform_ladder_covers_the_declared_order():
    rows = _ladder(UNIFORM, 2)
    e = _l1(rows, "cell_average")
    h = cc.spacings(UNIFORM)
    study = richardson_gci(e[-1], e[-2], e[-3], list(reversed(h)), theory_order=2.0)
    assert abs(study.order_used - 2.0) <= 0.2
    assert study.convergence == "monotone"
    assert study.asymptotic is True