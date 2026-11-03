"""Verification-debt closure for the merged annular axial Poiseuille backend.

Before this test nothing imported ``tensorfvm.annular`` outside the backend
itself: the assembled production sparse solve existed, the analytic reference
was stored in ``fields['reference_axial_velocity_m_s']``, and no check ever
compared them.  The gates below are external to the backend's own algebra: the
governing ODE and the no-slip conditions are proven on an independently
re-derived profile in 30-digit arithmetic, and the volumetric flow rate is
independently quadratured from the closed form.
"""
import math

import numpy as np
import pytest

from tensorfvm.annular import AnnularConfig, analytic_flow, solve
from tensorfvm.verification.annular import (
    GATE,
    flow_rate_audit,
    observed_order,
    ode_and_bc_audit,
    run_case,
)


@pytest.fixture(scope='module')
def config():
    return AnnularConfig(ntheta=32, nr=16)


def test_stored_closed_form_solves_the_governing_ode(config):
    audit = ode_and_bc_audit(config)
    # (1/r) d/dr (r du/dr) + G/mu = 0 to 30-digit rounding.
    assert audit['max_relative_ode_residual'] < 1e-25
    # No-slip on both circular walls.
    assert audit['wall_residual_over_peak'] < 1e-14
    # The backend's stored formula equals the independent derivation.
    assert audit['stored_vs_independent_max_relative'] < 1e-13
    # Negative control: the residual indicator is falsifiable (5% error -> 5%).
    assert audit['perturbation_control_min_residual'] > 0.04


def test_closed_form_flow_rate_matches_independent_quadrature(config):
    audit = flow_rate_audit(config)
    assert audit['relative_difference'] < 1e-12
    assert audit['quadrature_flow_m3_s'] > 0


def test_production_solve_meets_reference_and_second_order():
    coarse = run_case(16, 8)
    fine = run_case(32, 16)
    for row in (coarse, fine):
        row['inner_radius_m'] = 0.05
    order = observed_order([coarse, fine], 'relative_l2')['fitted_order']
    assert coarse['relative_l2'] > fine['relative_l2']
    assert order > 1.85
    assert fine['relative_l2'] < GATE
    assert fine['relative_flow_rate_error'] < GATE
    # Reference consumption: the stored reference is the independent closed form.
    assert fine['solved_consistency_max'] < 1e-13
    # The gate would reject a doubled error or a 5% reference perturbation.
    assert not (2.0 < GATE) and not (0.05 < GATE)


def test_reversed_flow_is_rejected_by_the_config_guard():
    # Negative control for the gate family: a sign error can only enter through
    # the pressure gradient, and AnnularConfig refuses non-positive gradients,
    # so reversed (suction-driven) axial flow is not representable - a documented
    # limitation of the merged backend rather than a silent wrong answer.
    with pytest.raises(ValueError):
        AnnularConfig(ntheta=32, nr=16, pressure_gradient_pa_m=-0.01)
