"""Tests for the annular design-order verification harness.

The harness measures the production annular backend against the closed-form
annular Poiseuille solution; these tests protect both halves: the analytic
yardstick must stay self-consistent, and the measured second-order convergence
must hold on a cheap pair of grids so that a regression in the solver (or in the
harness) fails loudly.
"""
import math

import numpy as np
import pytest

from tensorfvm.verification import annular_order as ao


@pytest.fixture(scope="module")
def self_check():
    return ao.reference_self_check()


@pytest.fixture(scope="module")
def two_level_study():
    return ao.convergence(((8, 32), (16, 64)))


def test_analytic_reference_is_self_consistent(self_check):
    assert self_check["passed"], self_check["checks"]
    checks = self_check["checks"]
    assert checks["momentum_equation_relative_residual"] < ao.SELF_CHECK_LIMIT
    assert checks["no_slip_relative_residual"] < ao.SELF_CHECK_LIMIT
    assert checks["closed_form_vs_quadrature_flow_rate_relative"] < 1e-10
    # Analytic identity: pressure force on the cross-section equals the wall drag.
    assert checks["global_equilibrium_identity_relative_residual"] < 1e-12


def test_analytic_max_velocity_matches_numerical_argmax():
    value, radius = ao.analytic_max_velocity()
    grid = np.linspace(ao.RI, ao.RO, 400001)
    profile = ao.analytic_velocity(grid)
    assert radius == pytest.approx(grid[int(np.argmax(profile))], abs=1e-5)
    assert value == pytest.approx(profile.max(), rel=1e-9)
    # du/dr = 0 must hold at that radius.
    h = 1e-9
    slope = (ao.analytic_velocity(radius + h) - ao.analytic_velocity(radius - h)) / (2 * h)
    assert abs(slope) / value * ao.RI < 1e-5


def test_geometry_audit_reports_second_order_area_defect():
    coarse, fine = ao.geometry_audit(8), ao.geometry_audit(16)
    assert coarse["positive_volumes"] and fine["positive_volumes"]
    assert fine["cells"] == 16 * 64
    assert abs(fine["relative_area_error"]) < abs(coarse["relative_area_error"])
    order = math.log(abs(coarse["relative_area_error"]) / abs(fine["relative_area_error"])) / math.log(2)
    assert order == pytest.approx(2.0, abs=0.15)
    # Straight-edge walls always undershoot the true annulus area.
    assert fine["relative_area_error"] < 0


def test_production_solver_matches_integrated_quantities():
    row = ao.run_case(16)
    values = row["values"]
    assert row["cells"] == 1024
    assert row["solver_relative_residual"] < 1e-10
    assert abs(values["flow_rate_error"]) < 0.02
    assert abs(values["l2"]) < ao.ERROR_LIMIT
    assert abs(values["inner_wall_shear_pa"] - abs(values["exact_inner_wall_shear_pa"])) / abs(
        values["exact_inner_wall_shear_pa"]) < ao.ERROR_LIMIT
    assert abs(values["outer_wall_shear_pa"] - abs(values["exact_outer_wall_shear_pa"])) / abs(
        values["exact_outer_wall_shear_pa"]) < ao.ERROR_LIMIT
    # The pressure source and the total wall traction balance on the discrete system.
    assert row["wall_traction_balance_relative"] < 1e-9


def test_observed_second_order_and_acceptance(two_level_study):
    orders = two_level_study["orders"]
    assert orders["velocity_l2"][0] == pytest.approx(2.0, abs=0.1)
    assert orders["velocity_linf"][0] > ao.ORDER_LIMIT
    assert orders["flow_rate"][0] > ao.ORDER_LIMIT
    verdict = ao.acceptance(two_level_study)
    assert verdict["passed"], verdict


def test_order_estimator_negative_control():
    """A first-order error sequence must not pass the second-order gate."""
    first_order = ao._order(1e-3, 1e-3 / 2.1, 1.0 / 8, 1.0 / 16)
    assert first_order < ao.ORDER_LIMIT
    assert ao._order(1e-3, 1e-3 / 4.0, 1.0 / 8, 1.0 / 16) == pytest.approx(2.0, abs=1e-12)
    # Negative radius is outside the closed form's domain: numpy yields nan, not silently
    # a plausible positive velocity.
    assert np.isnan(ao.analytic_velocity(-1.0))
    with pytest.raises(ValueError):
        ao._annular_config(0)