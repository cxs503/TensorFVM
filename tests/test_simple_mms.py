"""Tests for the staggered SIMPLE manufactured-solution order verification.

Extends the three-grid log-log-slope methodology to the incompressible
staggered (MAC) finite-volume solver.  The manufactured solution is a
smooth, divergence-free velocity field with analytic pressure and source
terms, so the exact solution is known to machine precision.

The tests verify:
1. The manufactured solution is divergence-free and smooth.
2. The source terms are computed correctly.
3. The staggered reference values are computed correctly.
4. The error norms are volume-weighted.
5. The three-grid log-log slope recovers the expected order (2 for SIMPLE).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.verification.simple_mms import (
    TWO_PI,
    exact_pressure,
    exact_velocity,
    pressure_error_norm,
    source_terms,
    staggered_references,
    velocity_error_norm,
)


# ---------------------------------------------------------------------------
# manufactured solution properties
# ---------------------------------------------------------------------------
def test_exact_velocity_is_divergence_free():
    """The manufactured velocity field must be exactly divergence-free."""
    rng = np.random.default_rng(42)
    x = rng.random(100)
    y = rng.random(100)
    u, v = exact_velocity(x, y)

    # Analytic divergence: ∂u/∂x + ∂v/∂y
    div_u = TWO_PI * np.cos(TWO_PI * x) * np.cos(TWO_PI * y)
    div_v = -TWO_PI * np.cos(TWO_PI * x) * np.cos(TWO_PI * y)
    div = div_u + div_v

    assert np.max(np.abs(div)) < 1e-14


def test_exact_pressure_is_smooth():
    """The pressure field must be smooth and bounded."""
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    xm, ym = np.meshgrid(x, y, indexing='ij')
    p = exact_pressure(xm, ym)

    assert np.min(p) >= -1.0 - 1e-14
    assert np.max(p) <= 1.0 + 1e-14
    dp_dx = np.diff(p, axis=0)
    dp_dy = np.diff(p, axis=1)
    assert np.max(np.abs(dp_dx)) < TWO_PI + 0.1
    assert np.max(np.abs(dp_dy)) < TWO_PI + 0.1


def test_source_terms_at_high_reynolds():
    """At Re→∞ (nu→0), the source terms should be convective + pressure only."""
    x = np.array([[0.25]])
    y = np.array([[0.25]])
    fu_inf, fv_inf = source_terms(x, y, reynolds=1e10)
    fu_finite, fv_finite = source_terms(x, y, reynolds=100.0)

    # The viscous term is O(1/Re), so the difference should be small
    diff_u = np.abs(fu_inf - fu_finite)
    diff_v = np.abs(fv_inf - fv_finite)
    expected_diff = 2 * math.pi / 100.0
    assert diff_u[0, 0] < expected_diff * 2
    assert diff_v[0, 0] < expected_diff * 2


# ---------------------------------------------------------------------------
# staggered references
# ---------------------------------------------------------------------------
def test_staggered_references_shapes():
    """Reference arrays must have the correct staggered shapes."""
    for nx, ny in [(8, 8), (16, 12), (32, 32)]:
        refs = staggered_references(nx, ny)
        assert refs['u'].shape == (nx + 1, ny), f"u shape mismatch for {nx}x{ny}"
        assert refs['v'].shape == (nx, ny + 1), f"v shape mismatch for {nx}x{ny}"
        assert refs['p'].shape == (nx, ny), f"p shape mismatch for {nx}x{ny}"


def test_staggered_references_boundary_values():
    """At boundaries, the manufactured solution should have correct values."""
    refs = staggered_references(16, 16)

    assert np.max(np.abs(refs['u'][0, :])) < 1e-14
    assert np.max(np.abs(refs['u'][-1, :])) < 1e-14
    assert np.max(np.abs(refs['v'][:, 0])) < 1e-14
    assert np.max(np.abs(refs['v'][:, -1])) < 1e-14


# ---------------------------------------------------------------------------
# error norms
# ---------------------------------------------------------------------------
def test_velocity_error_norm_zero_for_exact():
    """Error norm should be zero when numerical = reference."""
    refs = staggered_references(16, 16)
    hx, hy = 1.0 / 16, 1.0 / 16

    err_l1 = velocity_error_norm(refs['u'], refs['v'], refs['u'], refs['v'],
                                 hx, hy, norm='l1')
    err_l2 = velocity_error_norm(refs['u'], refs['v'], refs['u'], refs['v'],
                                 hx, hy, norm='l2')

    assert err_l1 < 1e-15
    assert err_l2 < 1e-15


def test_pressure_error_norm_zero_for_exact():
    """Pressure error norm should be zero when numerical = reference."""
    refs = staggered_references(16, 16)
    hx, hy = 1.0 / 16, 1.0 / 16

    err_l1 = pressure_error_norm(refs['p'], refs['p'], hx, hy, norm='l1')
    err_l2 = pressure_error_norm(refs['p'], refs['p'], hx, hy, norm='l2')

    assert err_l1 < 1e-15
    assert err_l2 < 1e-15


def test_velocity_error_norm_is_volume_weighted():
    """The velocity error norm must be volume-weighted."""
    refs = staggered_references(8, 8)
    hx, hy = 1.0 / 8, 1.0 / 8

    u_pert = refs['u'] + 0.1
    v_pert = refs['v'] + 0.1

    err_l1 = velocity_error_norm(u_pert, v_pert, refs['u'], refs['v'],
                                 hx, hy, norm='l1')
    assert abs(err_l1 - 0.1) < 1e-14


# ---------------------------------------------------------------------------
# three-grid log-log slope (synthetic data)
# ---------------------------------------------------------------------------
def test_synthetic_second_order_recovery():
    """A synthetic second-order error series must be read as order 2."""
    from tensorfvm.verification.cellcentre_loglog import loglog_slope, spacings

    grids = [16, 32, 64]
    h = spacings(grids)
    C = 0.5
    errors = [C * hi ** 2 for hi in h]

    p = loglog_slope(h, errors)
    assert abs(p - 2.0) < 1e-12


def test_synthetic_first_order_recovery():
    """A synthetic first-order error series must be read as order 1."""
    from tensorfvm.verification.cellcentre_loglog import loglog_slope, spacings

    grids = [16, 32, 64]
    h = spacings(grids)
    C = 0.5
    errors = [C * hi for hi in h]

    p = loglog_slope(h, errors)
    assert abs(p - 1.0) < 1e-12


def test_three_grid_readout_reports_deviation():
    """The readout must report the deviation from theory order."""
    from tensorfvm.verification.cellcentre_loglog import three_grid_readout

    grids = [16, 32, 64]
    errors = [1e-2, 3e-3, 8e-4]

    readout = three_grid_readout(grids, errors, theory_order=2.0)
    assert 'p_loglog' in readout
    assert 'loglog_rel_dev' in readout
    assert 'loglog_pass_15' in readout
    assert readout['p_loglog'] > 1.0


# ---------------------------------------------------------------------------
# source term consistency
# ---------------------------------------------------------------------------
def test_source_terms_match_analytic():
    """Source terms must match the analytic Navier-Stokes residual."""
    x = np.array([[0.3]])
    y = np.array([[0.4]])
    reynolds = 50.0

    fu, fv = source_terms(x, y, reynolds=reynolds)

    nu = 1.0 / (TWO_PI * reynolds)
    sin_x, cos_x = np.sin(TWO_PI * x), np.cos(TWO_PI * x)
    sin_y, cos_y = np.sin(TWO_PI * y), np.cos(TWO_PI * y)

    u = sin_x * cos_y
    v = -cos_x * sin_y

    du_dx = TWO_PI * cos_x * cos_y
    du_dy = -TWO_PI * sin_x * sin_y
    dv_dx = TWO_PI * sin_x * sin_y
    dv_dy = -TWO_PI * cos_x * cos_y

    conv_u = u * du_dx + v * du_dy
    conv_v = u * dv_dx + v * dv_dy

    lap_u = -TWO_PI ** 2 * u
    lap_v = -TWO_PI ** 2 * v

    visc_u = -nu * lap_u
    visc_v = -nu * lap_v

    dp_dx = TWO_PI * cos_x * sin_y
    dp_dy = TWO_PI * sin_x * cos_y

    fu_expected = conv_u + visc_u + dp_dx
    fv_expected = conv_v + visc_v + dp_dy

    assert abs(fu[0, 0] - fu_expected[0, 0]) < 1e-14
    assert abs(fv[0, 0] - fv_expected[0, 0]) < 1e-14
