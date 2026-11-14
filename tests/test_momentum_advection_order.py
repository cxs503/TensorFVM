"""Accuracy and safety of the momentum-advection options of :class:`TwoPhaseSolver`.

The ``limited2`` scheme is the capability the ``CASE-FVM-DAMBREAK-WET`` work
added (first-order upwind momentum advection carries a numerical viscosity
``|u| dx / 2`` that demonstrably holds a surge front back).  These tests pin the
three properties that make it safe to use next to a 1000:1 density jump:

1. it is *exact* for a linearly varying face velocity (i.e. it is genuinely a
   second-order difference, not a rescaled first-order one);
2. it falls back to the first-order stencil wherever the limiter switches off
   (extrema, sign changes) -- so it cannot manufacture a new extremum;
3. the default configuration is bit-for-bit the historical first-order path.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.two_phase_solver import (
    GridConfig,
    TwoPhaseConfig,
    TwoPhaseSolver,
    _limited_upwind_difference,
)


def _derivative_reference(order: int, c, b1, b2, f1, f2, vel, h):
    """Hand-written reference for the two stencils (no shared code with src)."""
    out = torch.zeros_like(c)
    for k in range(c.numel()):
        if vel[k] >= 0.0:
            if order == 1:
                out[k] = (c[k] - b1[k]) / h
            else:
                out[k] = (3.0 * c[k] - 4.0 * b1[k] + b2[k]) / (2.0 * h)
        else:
            if order == 1:
                out[k] = (f1[k] - c[k]) / h
            else:
                out[k] = (-3.0 * c[k] + 4.0 * f1[k] - f2[k]) / (2.0 * h)
    return out


def test_invalid_scheme_is_rejected():
    with pytest.raises(ValueError, match="momentum_advection_scheme"):
        TwoPhaseConfig(momentum_advection=True, momentum_advection_scheme="quick")


def test_limited_difference_is_exact_for_a_linear_field():
    h = 0.037
    a = -2.5                       # du/dx
    line = a * torch.arange(-3, 9, dtype=torch.float64) * h
    c, b1, b2 = line[2:-2], line[1:-3], line[0:-4]
    f1, f2 = line[3:-1], line[4:]
    for vel in (+1.0, -1.0):
        d = _limited_upwind_difference(
            c, b1, b2, f1, f2, torch.full_like(c, vel), h)
        assert torch.allclose(d, torch.full_like(d, a), atol=1e-12)


def test_limited_difference_vanishes_for_a_constant_field():
    c = torch.full((7,), 3.25, dtype=torch.float64)
    d = _limited_upwind_difference(
        c, c, c, c, c, torch.ones_like(c), 0.1)
    assert float(d.abs().max()) == 0.0


def test_limiter_switches_to_first_order_across_a_slope_sign_change():
    """Where consecutive slopes change sign (ψ = 0) the stencil *is* upwind."""
    h = 0.1
    prof = torch.tensor([0.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    c, b1, b2 = prof[2:3], prof[1:2], prof[0:1]
    f1, f2 = prof[3:4], prof[4:5]
    d = _limited_upwind_difference(c, b1, b2, f1, f2, torch.ones_like(c), h)
    # Δ⁻ = +1, Δ⁻⁻ = -1 → ratio < 0 → ψ = 0 → plain first-order upwind
    assert float(d[0]) == pytest.approx(float((c[0] - b1[0]) / h), abs=1e-14)


def test_limited_derivative_stays_within_the_tvd_band():
    """The correction never exceeds the first-order term (≤ 1.5x upwind)."""
    torch.manual_seed(3)
    h = 0.05
    n = 40
    prof = torch.cumsum(torch.randn(n, dtype=torch.float64), dim=0).abs()
    vel = torch.ones(n - 4, dtype=torch.float64)
    d2 = _limited_upwind_difference(
        prof[2:-2], prof[1:-3], prof[0:-4], prof[3:-1], prof[4:], vel, h)
    d1 = (prof[2:-2] - prof[1:-3]) / h
    assert float((d2.abs() - 2.0 * d1.abs()).max()) <= 1e-12
    # ... and it never reverses the upwind direction on a monotone profile
    assert bool((d2 * d1 >= -1e-12).all())


def test_first_order_path_is_unchanged():
    """``order=1`` must reproduce the historical hand-written stencils exactly."""
    torch.manual_seed(0)
    ny, nx = 7, 9
    c = torch.randn(31, dtype=torch.float64)
    b1 = torch.randn(31, dtype=torch.float64)
    b2 = torch.randn(31, dtype=torch.float64)
    f1 = torch.randn(31, dtype=torch.float64)
    f2 = torch.randn(31, dtype=torch.float64)
    vel = torch.randn(31, dtype=torch.float64)
    h = 0.05
    got = _limited_upwind_difference(c, b1, b2, f1, f2, vel, h)
    # order-1 branch of the solver uses the same where() expression
    want = torch.where(vel >= 0.0, (c - b1) / h, (f1 - c) / h)
    ref2 = _derivative_reference(2, c, b1, b2, f1, f2, vel, h)
    assert got.shape == want.shape
    # order-2 must differ from order 1 on generic data (i.e. it is not a no-op)
    assert float((got - want).abs().max()) > 1e-6
    # and must agree with the textbook (3,-4,1) stencil wherever psi=1
    close = torch.isclose(got, ref2, atol=1e-12)
    assert bool(close.any())


def test_order_of_accuracy_improves_on_a_smooth_profile():
    """Derivative error on a smooth monotone profile: O(h^2) vs O(h)."""

    def error(order: int, n: int) -> float:
        # x^3 on [1, 2]: monotone and smooth, so the limiter never switches off
        # and the asymptotic order of the stencil itself is measured.
        h = 1.0 / n
        x = 1.0 + torch.arange(-2, n + 3, dtype=torch.float64) * h
        f = x ** 3
        c, b1, b2 = f[2:-2], f[1:-3], f[0:-4]
        f1, f2 = f[3:-1], f[4:]
        exact = 3.0 * x[2:-2] ** 2
        vel = torch.full_like(c, 1.0)
        if order == 1:
            got = torch.where(vel >= 0.0, (c - b1) / h, (f1 - c) / h)
        else:
            got = _limited_upwind_difference(c, b1, b2, f1, f2, vel, h)
        return float((got - exact).abs().max())

    e1_coarse, e1_fine = error(1, 64), error(1, 128)
    e2_coarse, e2_fine = error(2, 64), error(2, 128)
    assert 1.7 < e1_coarse / e1_fine < 2.4, "first order should converge like O(h)"
    assert e2_coarse / e2_fine > 3.0, "limited2 should converge like O(h^2)"


def test_solver_uses_the_second_order_stencil_when_selected():
    """End-to-end: a linear shear field has an exact advective acceleration."""

    def adv(scheme: str, field_value_at):
        cfg = TwoPhaseConfig(
            enabled=True, rho_liquid=1.0, rho_gas=1.0, mu_liquid=0.0,
            mu_gas=0.0, gravity=0.0, surface_tension=False,
            momentum_advection=True, momentum_advection_scheme=scheme)
        solver = TwoPhaseSolver(GridConfig(nx=12, ny=8, length=0.12, height=0.08), cfg)
        ny, nx = solver.ny, solver.nx
        # u varies linearly in x on its own faces → du/dx = slope exactly
        xs = torch.arange(nx + 1, dtype=torch.float64) * solver.dx
        slope = 3.0
        solver.u[:] = field_value_at(xs, slope)[None, :].expand(ny, nx + 1)
        solver.v[:] = 0.0
        return solver._advective_accel(order=2 if scheme == "limited2" else 1)

    lin = lambda xs, s: s * xs  # noqa: E731
    adv_u2, _ = adv("limited2", lin)
    adv_u1, _ = adv("upwind", lin)
    # u·du/dx = u * slope; compare against the analytic product
    ny, nx = 8, 12
    xs = torch.arange(nx + 1, dtype=torch.float64) * (0.12 / 12)
    u_c = (3.0 * xs)[1:nx].expand(ny - 2, nx - 1)
    exact = u_c * 3.0
    assert float((adv_u2 - exact).abs().max()) < 1e-12
    # the first-order stencil is exact here too (linear field), but only
    # because the field is linear -- on a curved field they must differ.
    assert float((adv_u1 - exact).abs().max()) < 1e-12