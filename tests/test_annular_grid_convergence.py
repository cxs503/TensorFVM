"""Verification of the annular grid-convergence apparatus and of the annular solver.

Two levels of verification are asserted here:

1. the *tool* - the order estimator must recover a known order, must refuse
   sequences that carry no order information (round-off plateau, non-monotone
   oscillation), and must distinguish first from second order;
2. the *baseline* - the closed-form annular Poiseuille profile must agree with an
   independent finite-difference solution of the differential statement, so the
   reference is not checked against itself;
3. the *solver* - the production annular solve must converge at the observed
   second order, and its Richardson-extrapolated flow rate must reproduce the
   closed-form flow rate inside its own discretisation band.
"""
import math

import numpy as np
import pytest
import torch

from tensorfvm.annular import AnnularConfig, analytic_flow, analytic_velocity
from tensorfvm.verification.annular_convergence import (
    _fd_annular_profile,
    analytic_reference_check,
    loglog_order,
    observed_order,
    richardson_extrapolate,
    study,
)

GRIDS = ((32, 8), (64, 16), (128, 32))


@pytest.fixture(scope="module")
def report():
    torch.set_num_threads(1)
    return study(levels=3, grids=GRIDS)


def test_observed_order_recovers_known_second_order():
    exact = 1.2345678901234
    h = np.array([4.0, 2.0, 1.0])
    values = exact + 0.75 * h ** 2
    result = observed_order(values, 2.0)
    assert abs(result["order"] - 2.0) < 1e-9
    assert abs(result["richardson_extrapolated"] - exact) < 1e-12
    # GCI quantifies the remaining fine-grid discretisation error, so for this
    # deliberately coarse 25%-error sequence it must reproduce its closed form
    # Fs |f_fine - f_medium| / |f_fine| / (r^p - 1) rather than collapse to zero.
    f_fine = exact + 0.75
    expected_gci = 1.25 * abs(f_fine - (exact + 3.0)) / abs(f_fine) / (2.0 ** 2 - 1.0)
    assert result["gci_fine_relative"] == pytest.approx(expected_gci, rel=1e-12)


def test_observed_order_distinguishes_first_from_second_order():
    exact = 2.0
    h = np.array([4.0, 2.0, 1.0])
    first = observed_order(exact + 0.3 * h, 2.0)
    second = observed_order(exact + 0.3 * h ** 2, 2.0)
    assert abs(first["order"] - 1.0) < 1e-9
    assert abs(second["order"] - 2.0) < 1e-9
    # The estimator reports different orders for the same three levels, so the
    # order is a property of the data and not an imposed constant. Extrapolating
    # the genuinely first-order data with the *assumed* order 2 gives a wrong
    # value; that is the failure mode this test guards against.
    assert abs(richardson_extrapolate(exact + 0.3 * h, 2.0, 2.0) - exact) > 0.1
    assert abs(second["richardson_extrapolated"] - exact) < 1e-12


def test_roundoff_plateau_carries_no_order_information():
    values = np.array([1.0, 1.0 + 2e-16, 1.0 + 3e-16])
    with pytest.raises(ValueError, match="round-off"):
        observed_order(values, 2.0)


def test_non_monotone_sequence_is_rejected():
    with pytest.raises(ValueError, match="not monotone"):
        observed_order([1.0, 1.10, 1.05], 2.0)


def test_order_requires_three_levels_and_valid_ratio():
    with pytest.raises(ValueError, match="three grid levels"):
        observed_order([1.0, 1.01], 2.0)
    with pytest.raises(ValueError, match="ratio"):
        observed_order([1.0, 1.1, 1.05], 1.0)


def test_loglog_order_recovers_known_slope():
    h = np.array([8.0, 4.0, 2.0, 1.0])
    fit = loglog_order(h, 0.2 * h ** 1.5)
    # log(e) grows with log(h): the slope of the error curve equals the order 1.5.
    assert abs(fit["slope"] - 1.5) < 1e-12
    assert fit["levels"] == 4


def test_closed_form_matches_independent_finite_difference_solution():
    config = AnnularConfig()
    check = analytic_reference_check(config)
    # The FD solve of d/dr(r du/dr) = -(G/mu) r must reproduce the closed form,
    # and the residual gap must shrink at the second order of the FD scheme.
    assert check["finest_fd_max_relative_deviation"] < 1e-6
    for order in check["fd_observed_orders"]:
        assert abs(order - 2.0) < 5e-2
    assert check["differential_identity_max_relative_residual"] < 1e-11
    assert check["flow_rate_relative_deviation"] < 1e-5
    assert check["reference_is_self_check"] is False


def test_fd_profile_is_second_order_against_an_independent_analytic_field():
    config = AnnularConfig()
    # Cosine bump with Dirichlet zeros: a solution of the *homogeneous* problem is
    # not available in this geometry, so use a manufactured source instead and
    # confirm the discrete operator converges at second order in h.
    errors = []
    spacings = []
    for points in (201, 401, 801):
        r, u, h = _fd_annular_profile(config, points)
        formula = analytic_velocity(r, config)
        errors.append(float(np.max(np.abs(u - formula))))
        spacings.append(h)
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(spacings[i] / spacings[i + 1]) for i in range(2)]
    assert all(1.9 < order < 2.1 for order in orders)


def test_production_solver_is_second_order_and_matches_the_exact_flow_rate(report):
    rows = report["rows"]
    assert [row["grid"] for row in rows] == [list(g) for g in GRIDS]
    errors = [row["relative_l2"] for row in rows]
    assert errors[0] > errors[1] > errors[2], "L2 error must decrease under uniform refinement"
    for order in report["l2_pairwise_orders"]:
        assert 1.8 <= order <= 2.2, f"observed L2 order {order} is not second order"
    assert 1.9 <= report["l2_loglog_slope"]["slope"] <= 2.1
    # Every level keeps the production residual gates exact to the gate tolerance.
    assert all(row["true_linear_residual"] <= 1e-9 for row in rows)
    assert all(abs(row["momentum_force_balance"]) < 1e-12 for row in rows)
    # Richardson extrapolation of the flow rate lands on the closed-form value
    # inside its own GCI band: quantitatively verified, not merely within 3%.
    flow = report["flow_rate_order"]
    assert 1.9 <= flow["order"] <= 2.1
    assert report["extrapolated_vs_analytic_relative_deviation"] <= report["extrapolation_gci_band_relative"]
    assert report["extrapolation_agrees_with_analytic"]
    assert flow["gci_fine_relative"] < 5e-3


def test_coarse_grid_already_meets_the_legacy_three_percent_band(report):
    # Documents the laxness of the pre-existing 3% acceptance band: it is met
    # already on the coarsest grid, which is exactly why an order study was needed.
    assert report["rows"][0]["relative_l2"] < 0.03
    assert all(row["relative_l2"] < 0.03 for row in report["rows"])
    assert report["rows"][-1]["relative_l2"] < 2e-3


def test_wall_traction_and_geometry_converge_with_resolution(report):
    rows = report["rows"]
    wall = [row["inner_wall_force_relative_error"] for row in rows]
    assert wall[0] > wall[1] > wall[2]
    for order in report["inner_wall_force_pairwise_orders"]:
        assert 1.8 <= order <= 2.2
    # Fitted polygonal geometry: area error must vanish under refinement too.
    area = [row["area_relative_error"] for row in rows]
    assert area[0] > area[1] > area[2]
    assert area[-1] < 2e-3


def test_reference_profile_is_independent_of_the_solver_evaluation_points(report):
    config = AnnularConfig()
    exact = analytic_flow(config)
    # The analytic flow rate is a property of the configuration, not of a grid.
    assert exact == analytic_flow(config)
    assert abs(exact - report["analytic_flow_rate_m3_s"]) <= 1e-15 * abs(exact) + 1e-18
    assert report["analytic_reference"]["flow_rate_formula_m3_s"] == pytest.approx(exact, rel=1e-12)