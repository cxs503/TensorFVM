"""Tests for the three-grid log--log-slope order estimator (cell-centred FV).

The estimator is verified *before* it is trusted: every claim the module makes in
its docstring is exercised here, including the ones that exist because the
practice is easy to get wrong (point values vs cell averages, cell count as
abscissa, volume weighting, non-monotone ladders).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.verification.loglog_order import (
    ASYMPTOTIC_TOLERANCE,
    ORDER_TOLERANCE,
    POINT_VALUE_CONSTANT_FACTOR,
    cell_error_norms,
    loglog_order,
    mm_cell_average,
    mm_cell_average_second_derivative,
    mm_derivative,
    mm_value,
    solution_error_ladder,
    stretched_faces,
    stretched_weights,
    truncation_ladder,
    truncation_stretched,
    truncation_uniform,
    uniform_faces,
)


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 4.0])
def test_exact_order_recovered(p):
    h = np.array([1 / 8, 1 / 16, 1 / 32])
    study = loglog_order(h, 0.7 * h ** p, theory_order=p)
    assert study.verdict == "PASS"
    assert abs(study.order - p) / p < 1e-12
    assert study.order_stderr < 1e-12
    assert study.monotone and study.asymptotic
    assert study.r2 == pytest.approx(1.0, abs=1e-12)
    assert study.withheld_reason == ""


def test_stderr_is_defined_and_tiny_on_exact_data():
    h = np.array([1 / 4, 1 / 8, 1 / 16])
    study = loglog_order(h, 3.0 * h ** 2)
    assert study.order == pytest.approx(2.0, rel=1e-12)
    assert math.isfinite(study.order_stderr)
    assert study.order_stderr < 1e-12        # three points: 1 dof, still defined


def test_error_growing_under_refinement_is_withheld():
    """An inverted ladder (error grows as h shrinks) carries no order at all."""
    study = loglog_order([1 / 4, 1 / 8, 1 / 16], 3.0 * (1 / np.array([1 / 4, 1 / 8, 1 / 16])) ** 2)
    assert study.verdict == "WITHHELD"
    assert study.withheld_reason == "non-monotone"


def test_input_order_does_not_matter():
    h = np.array([1 / 8, 1 / 16, 1 / 32])
    e = 0.5 * h ** 2
    forward = loglog_order(h, e, theory_order=2.0)
    backward = loglog_order(h[::-1], e[::-1], theory_order=2.0)
    assert forward.order == pytest.approx(backward.order, rel=1e-14)
    assert forward.spacings == backward.spacings        # both coarse -> fine


def test_noise_falls_inside_stderr():
    rng = np.random.default_rng(7)
    h = np.array([1 / 8, 1 / 16, 1 / 32, 1 / 64])
    exact = 0.4 * h ** 2
    study = loglog_order(h, exact * (1.0 + 1e-3 * rng.standard_normal(h.size)))
    assert abs(study.order - 2.0) < 5.0 * study.order_stderr


def test_non_monotone_is_withheld():
    study = loglog_order([0.5, 0.25, 0.125], [1e-2, 3e-2, 2e-3])
    assert study.verdict == "WITHHELD"
    assert study.withheld_reason == "non-monotone"
    assert math.isnan(study.order)


def test_stagnant_is_withheld():
    study = loglog_order([0.5, 0.25, 0.125], [1e-2, 1e-3, 1e-3 * (1.0 - 1e-15)])
    assert study.verdict == "WITHHELD"
    assert study.withheld_reason == "stagnant"


def test_tiny_refinement_ratio_is_withheld():
    study = loglog_order([0.5, 0.4995, 0.125], [1e-2, 4e-3, 1e-3])
    assert study.verdict == "WITHHELD"
    assert study.withheld_reason == "refinement-ratio-too-small"


def test_pre_asymptotic_ladder_is_flagged_not_failed():
    # two truncation terms: the coarse pair sees a blended slope
    h = np.array([1 / 4, 1 / 8, 1 / 16])
    e = 1e-4 * h ** 2 + 1e-2 * h ** 4       # crossover between the two finest levels
    study = loglog_order(h, e, theory_order=2.0)
    assert study.spread > ASYMPTOTIC_TOLERANCE * 2.0
    assert study.asymptotic is False
    assert study.verdict in ("PASS", "FAIL")            # qualified, not withheld


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        loglog_order([0.5, 0.25], [1.0, 0.5])            # two levels
    with pytest.raises(ValueError):
        loglog_order([0.5, 0.25, 0.125], [1.0, -0.5, 0.0])   # non-positive
    with pytest.raises(ValueError):
        loglog_order([0.5, 0.25, 0.125], [1.0, 0.5])         # length mismatch
    with pytest.raises(ValueError):
        loglog_order([0.5, 0.5, 0.125], [1.0, 0.5, 0.1])     # duplicate spacing


# --------------------------------------------------------------------------- #
# norms
# --------------------------------------------------------------------------- #
def test_norms_agree_on_a_uniform_mesh():
    error = np.array([1.0, -2.0, 3.0, -4.0])
    norms = cell_error_norms(error, np.full(4, 0.25))
    expected_l2 = math.sqrt(float(np.mean(error ** 2)))
    assert norms["l2"] == pytest.approx(expected_l2, rel=1e-15)
    assert norms["l2_unweighted"] == pytest.approx(expected_l2, rel=1e-15)
    assert norms["linf"] == pytest.approx(4.0)
    assert norms["l1"] == pytest.approx(float(np.mean(np.abs(error))), rel=1e-15)
    assert norms["l2_integral"] == pytest.approx(expected_l2, rel=1e-15)   # V_total = 1


def test_unweighted_norm_differs_on_a_stretched_mesh():
    volume = np.array([0.1, 0.4, 0.4, 0.1])
    error = np.array([1.0, 0.0, 0.0, 1.0])
    norms = cell_error_norms(error, volume)
    assert norms["l2"] == pytest.approx(math.sqrt(0.2 / 1.0), rel=1e-15)
    assert norms["l2_unweighted"] == pytest.approx(math.sqrt(0.5), rel=1e-15)
    assert norms["l2"] < norms["l2_unweighted"]


# --------------------------------------------------------------------------- #
# manufactured solution
# --------------------------------------------------------------------------- #
def test_cell_average_matches_gauss_quadrature():
    faces = uniform_faces(16)
    left, right = faces[:-1], faces[1:]
    closed = mm_cell_average(left, right)
    # independent 6-point Gauss-Legendre quadrature of the same integrals
    xg, wg = np.polynomial.legendre.leggauss(6)
    mid, half = 0.5 * (left + right), 0.5 * (right - left)
    # Int u dx over a cell = half * sum(w u), and the average divides by 2*half
    numeric = np.array([0.5 * np.sum(wg * mm_value(m + hl * xg)) for m, hl in zip(mid, half)])
    assert np.allclose(closed, numeric, atol=1e-13)


def test_source_cell_average_equals_derivative_jump():
    faces = uniform_faces(8)
    left, right = faces[:-1], faces[1:]
    assert np.allclose(mm_cell_average_second_derivative(left, right),
                       (mm_derivative(right) - mm_derivative(left)) / (right - left),
                       atol=0.0, rtol=1e-15)


# --------------------------------------------------------------------------- #
# operators: the two cell-average traps and the stencil pairing
# --------------------------------------------------------------------------- #
def test_second_order_operator_is_second_order():
    block = truncation_ladder("second_order")
    assert block["study"]["verdict"] == "PASS"
    assert abs(block["study"]["order"] - 2.0) / 2.0 < ORDER_TOLERANCE


def test_cell_average_fourth_order_stencil_is_fourth_order():
    block = truncation_ladder("fourth_order_cellavg")
    assert block["study"]["verdict"] == "PASS"
    assert abs(block["study"]["order"] - 4.0) / 4.0 < ORDER_TOLERANCE


def test_point_value_stencil_degrades_on_cell_averages():
    """Discretisation trap: the textbook 4-point stencil is only 2nd order here."""
    block = truncation_ladder("fourth_order", cell_average=True)
    order = block["study"]["order"]
    assert abs(order - 2.0) / 2.0 < ORDER_TOLERANCE
    assert abs(order - 4.0) / 4.0 > ORDER_TOLERANCE


def test_second_order_truncation_constant_and_point_values():
    """C8: the cell-average constant is exactly twice the point-value one."""
    with_avg = truncation_ladder("second_order", cell_average=True)
    with_point = truncation_ladder("second_order", cell_average=False)
    ratio = with_avg["errors"][-1] / with_point["errors"][-1]
    assert ratio == pytest.approx(POINT_VALUE_CONSTANT_FACTOR, rel=0.02)


def test_solution_ladder_second_order():
    block = solution_error_ladder((16, 32, 64), "second_order")
    assert block["study"]["verdict"] == "PASS"
    assert abs(block["study"]["order"] - 2.0) / 2.0 < ORDER_TOLERANCE


def test_solution_ladder_fourth_order_needs_the_matching_stencil():
    correct = solution_error_ladder((16, 32, 64), "fourth_order_cellavg")
    assert correct["study"]["verdict"] == "PASS"
    assert abs(correct["study"]["order"] - 4.0) / 4.0 < ORDER_TOLERANCE
    native = solution_error_ladder((16, 32, 64), "fourth_order")
    assert abs(native["study"]["order"] - 4.0) / 4.0 > ORDER_TOLERANCE


def test_point_value_reference_masks_high_order():
    """Measurement trap: a 4th-order solution measured at cell centres reads 2."""
    block = solution_error_ladder((16, 32, 64), "fourth_order_cellavg",
                                  reference="point_value")
    order = block["study"]["order"]
    assert abs(order - 2.0) / 2.0 < ORDER_TOLERANCE
    assert abs(order - 4.0) / 4.0 > ORDER_TOLERANCE


def test_displaced_flux_is_first_order():
    block = truncation_ladder("shifted")
    assert block["study"]["verdict"] == "PASS"
    assert abs(block["study"]["order"] - 1.0) < ORDER_TOLERANCE


# --------------------------------------------------------------------------- #
# stretched mesh: the weighting claim
# --------------------------------------------------------------------------- #
def test_stretched_faces_are_monotone_and_periodic():
    for n in (16, 64):
        faces = stretched_faces(n)
        assert faces[0] == pytest.approx(0.0)
        assert faces[-1] == pytest.approx(1.0)
        assert np.all(np.diff(faces) > 0.0)


def test_stretched_truncation_is_second_order():
    faces = stretched_faces(64)
    left, right = faces[:-1], faces[1:]
    tau, volume = truncation_stretched(mm_cell_average(left, right), faces)
    source = mm_cell_average_second_derivative(left, right)
    norms = cell_error_norms(tau - source, volume)
    assert norms["l2"] > 0.0


def test_weighting_is_order_neutral_but_not_magnitude_neutral():
    block = stretched_weights()
    assert block["order_difference"] <= 0.05                  # declared claim C12
    assert abs(block["weighted"]["order"] - 2.0) / 2.0 < ORDER_TOLERANCE
    assert block["magnitude_ratio"] != pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# cross-validation against the Richardson/GCI family
# --------------------------------------------------------------------------- #
def test_loglog_matches_richardson_on_the_same_ladder():
    from tensorfvm.verification.richardson import richardson_gci

    block = truncation_ladder("second_order")
    values = block["errors"]
    rich = richardson_gci(values[-1], values[-2], values[-3],
                          tuple(reversed(block["spacings"])), theory_order=2.0)
    assert abs(block["study"]["order"] - rich.order) / 2.0 < ORDER_TOLERANCE
    assert rich.convergence == "monotone"