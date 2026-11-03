"""Verification anchors for the merged two-phase VOF solver.

``tensorfvm.two_phase_solver.TwoPhaseSolver`` (``tp.enabled=True``) is checked
against *analytic* references that need no external data:

1. hydrostatic pressure initialisation, continuous across the interface:
   ``p = rho_g g (H - h_i) + rho_l g (h_i - y)`` below and
   ``p = rho_g g (H - y)`` above it (exact, piecewise linear);
2. a stratified column at rest must stay at rest under gravity
   (exact steady state of the discrete equations);
3. the 2D Young-Laplace pressure jump of a static circular drop,
   ``dp = sigma / R`` (exact for a circular interface in 2D; the sphere value
   ``2 sigma / R`` does not apply);
4. a fully sealed domain (``top_atmosphere=False``), whose pressure Poisson
   problem is all-Neumann with a one-dimensional null space, must be rejected;
5. the Chorin projection must be an *exact* projection: one isolated
   correction step has to remove the discrete divergence of an arbitrary
   velocity field (regression guard for the operator-sign and cell-volume
   factors in ``_projection_pressure``).

All five were red before the repair (three as ``xfail(strict=True)`` executed
against the defects documented in
``scripts/verification-two-phase-solver/report.md``).  They are now hard
assertions; the evidence trail lives in
``scripts/verification-two-phase-projection/``.

Cases are small (n <= 64, <= 40 steps) and run in a few seconds.
"""

import math

import pytest
import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver


def _solver(n=32, gravity=9.81, sigma=0.0, surface_tension=False,
            top_atmosphere=True, mu=1e-3, drop_radius=None):
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=1000.0, rho_gas=1.0,
        mu_liquid=mu, mu_gas=mu, gravity=gravity,
        sigma=sigma, surface_tension=surface_tension,
        top_atmosphere=top_atmosphere,
    )
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dy
    if drop_radius is None:
        yy = y.reshape(-1, 1).expand(n, n)
        solver.set_alpha(torch.where(yy < 0.5, torch.ones_like(yy),
                                     torch.zeros_like(yy)))
        solver.initialize_hydrostatic_pressure(0.5)
        return solver, None
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.5) ** 2)
    alpha = torch.clamp((drop_radius - rr) / max(solver.dx, solver.dy) + 0.5, 0.0, 1.0)
    solver.set_alpha(alpha)
    return solver, rr


def test_hydrostatic_pressure_initialisation_is_exact():
    """``initialize_hydrostatic_pressure`` reproduces the analytic column.

    The reference is continuous at the interface (the liquid branch carries
    the weight of the gas column above it) and the top cell centre carries the
    discrete-consistent value ``rho_g g dy/2`` because the atmospheric datum
    ``p = 0`` is imposed on the top *face*, half a cell higher.
    """
    n = 32
    solver, _ = _solver(n=n)
    g, rho_l, rho_g, h_i = 9.81, 1000.0, 1.0, 0.5
    y = (torch.arange(n, dtype=torch.float64) + 0.5).reshape(-1, 1) * solver.dy
    p_iface = rho_g * g * (1.0 - h_i)
    exact = torch.where(y < h_i,
                        p_iface + rho_l * g * (h_i - y),
                        rho_g * g * (1.0 - y)).expand(n, n)
    error = (solver.p - exact).abs().max().item()
    assert error / exact.abs().max().item() < 1e-12
    assert abs(solver.p[-1, 0].item() - rho_g * g * solver.dy / 2) < 1e-12
    assert abs(solver.p[0, 0].item()
               - (p_iface + rho_l * g * (h_i - 0.5 * solver.dy))) < 1e-9


def test_alpha_field_volume_is_conserved_at_initialisation():
    """A drop discretised by a signed-distance ramp keeps its area."""
    solver, _ = _solver(n=64, gravity=0.0, drop_radius=0.2)
    area = float(solver.alpha.sum()) * solver.dx * solver.dy
    exact = math.pi * 0.2 ** 2
    assert abs(area - exact) / exact < 3e-3


def test_stratified_column_at_rest_stays_at_rest():
    """Gravity balanced by the analytic pressure gradient: u = v = 0 for all t."""
    solver, _ = _solver(n=16, gravity=9.81, mu=1e-3)
    for _ in range(10):
        solver.step(1e-5)
    assert float(solver.u.abs().max()) < 1e-9
    assert float(solver.v.abs().max()) < 1e-9


def test_stratified_column_remains_at_rest_beyond_ten_steps():
    """The rest state must be a genuine steady state, not a slow transient."""
    solver, _ = _solver(n=16, gravity=9.81, mu=1e-3)
    for _ in range(40):
        solver.step(1e-5)
    assert float(solver.u.abs().max()) < 1e-9
    assert float(solver.v.abs().max()) < 1e-9


def test_laplace_pressure_of_static_drop_matches_sigma_over_R():
    """2D Young-Laplace: the pressure jump across a circular drop is sigma/R."""
    radius, sigma = 0.2, 0.07
    solver, rr = _solver(n=32, gravity=0.0, sigma=sigma, surface_tension=True,
                         mu=1e-2, drop_radius=radius)
    for _ in range(10):
        solver.step(1e-5)
    inside = float(solver.p[rr < 0.5 * radius].mean())
    outside = float(solver.p[rr > 1.6 * radius].mean())
    exact = sigma / radius
    assert abs((inside - outside) - exact) / exact < 0.15


def test_fully_sealed_domain_is_rejected():
    """A sealed domain has no pressure datum; the solver must not run silently."""
    solver, _ = _solver(n=16, gravity=9.81, mu=1e-3, top_atmosphere=False)
    with pytest.raises(ValueError):
        solver.step(1e-5)


def test_projection_removes_discrete_divergence():
    """One isolated Chorin correction must be an exact projection.

    Regression guard for the two defects that made the projection amplify
    divergence: the off-diagonal signs of the five-point operator and the cell
    volume missing from the right-hand side.
    """
    n, dt = 16, 1e-5
    solver, _ = _solver(n=n, gravity=0.0, mu=0.0)
    torch.manual_seed(3)
    solver.u.normal_()
    solver.v.normal_()
    solver._boundaries(projection=True)
    u_star, v_star = solver.u.clone(), solver.v.clone()
    div_before = float(solver._divergence().abs().max())
    dp = solver._projection_pressure(u_star, v_star, dt)
    inv_rho_u = 2.0 / (solver.rho_field[:, :-1] + solver.rho_field[:, 1:])
    inv_rho_v = 2.0 / (solver.rho_field[:-1, :] + solver.rho_field[1:, :])
    solver.u[:, 1:-1] = u_star[:, 1:-1] - dt * inv_rho_u * (dp[:, 1:] - dp[:, :-1]) / solver.dx
    solver.v[1:-1, :] = v_star[1:-1, :] - dt * inv_rho_v * (dp[1:, :] - dp[:-1, :]) / solver.dy
    inv_rho_top = 1.0 / solver.rho_field[-1, :]
    solver.v[-1, :] = v_star[-1, :] + dt * inv_rho_top * dp[-1, :] / (0.5 * solver.dy)
    div_after = float(solver._divergence().abs().max())
    assert div_after < 1e-6 * div_before


def test_single_phase_mode_is_dormant():
    """``enabled=False`` keeps every two-phase field unallocated."""
    solver = TwoPhaseSolver(GridConfig(nx=8, ny=8))
    assert solver.alpha is None and solver.vof is None and solver.fs_bc is None
    assert solver.rho_field is not None and float(solver.rho_field.max()) == 1.0