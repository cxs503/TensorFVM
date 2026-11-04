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
import numpy as np


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



# --------------------------------------------------------------------------- #
# m2564 additions rebased on the merged solver (m2654 absorbed the operator
# repair; these anchors are the unique verification increment)
# --------------------------------------------------------------------------- #

def _stratified_alpha(solver: TwoPhaseSolver, interface_y: float) -> torch.Tensor:
    y = (torch.arange(solver.ny, dtype=torch.float64) + 0.5) * solver.dy
    col = (y < interface_y).to(torch.float64).reshape(-1, 1)
    return col.expand(solver.ny, solver.nx).clone()






def _circular_alpha(nx: int, ny: int, radius: float, dx: float,
                    cy: float = 0.5, cx: float = 0.5,
                    eps_factor: float = 0.75) -> torch.Tensor:
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
    r = torch.sqrt((x[None, :] - cx) ** 2 + (y[:, None] - cy) ** 2)
    return 0.5 * (1.0 - torch.tanh((r - radius) / (eps_factor * dx)))






def _column_solver(n: int = 40, dt: float = 1e-3) -> TwoPhaseSolver:
    cfg = GridConfig(nx=n, ny=n, length=1.0, height=1.0, time_step=dt)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1.0,
                        mu_liquid=1.0e-3, mu_gas=1.0e-5,
                        gravity=9.81, top_atmosphere=True)
    solver = TwoPhaseSolver(cfg, tp)
    solver.set_alpha(_stratified_alpha(solver, 0.5))
    solver.initialize_hydrostatic_pressure(0.5)
    return solver


# --------------------------------------------------------------------------- #
# anchor 1: static water column stays at rest (was xfail(strict=True))
# --------------------------------------------------------------------------- #




def test_static_column_stays_at_rest():
    """A hydrostatic two-phase column at rest must not generate motion.

    Well-balanced variable-density projection: gravity in the momentum
    predictor is exactly cancelled by the pressure gradient of the
    variable-density Poisson solve, so ``u`` and ``v`` stay at round-off and
    the discrete divergence is nulled.
    """
    solver = _column_solver(n=40, dt=1e-3)
    dt = 1e-3
    with torch.no_grad():
        for _ in range(10):
            solver.step(dt)
            assert torch.isfinite(solver.u).all() and torch.isfinite(solver.v).all()

    max_u = float(torch.abs(solver.u).max())
    max_v = float(torch.abs(solver.v).max())
    max_div = float(torch.abs(solver._divergence()).max())
    assert max_u < 1e-9, f"static column: max|u| = {max_u:.3e} (must be < 1e-9)"
    assert max_v < 1e-9, f"static column: max|v| = {max_v:.3e} (must be < 1e-9)"
    assert max_div < 1e-10, f"static column: max|div u| = {max_div:.3e}"

    # The pressure must be the hydrostatic field, i.e. O(rho*g*dy) in
    # magnitude -- not the O(rho*g*dy/dt) blow-up of the inconsistent operator.
    p_bottom = float(solver.p[0, 0])
    p_expected = (1000.0 * 9.81 * (0.5 - 0.5 * solver.dy)
                  + 1.0 * 9.81 * 0.5)
    assert abs(p_bottom - p_expected) / p_expected < 0.01, (
        f"static column: p_bottom = {p_bottom:.1f} Pa vs hydrostatic "
        f"{p_expected:.1f} Pa")






def test_static_column_divergence_does_not_grow():
    """The residual divergence must stay bounded (no exponential growth)."""
    solver = _column_solver(n=32, dt=1e-3)
    divs = []
    with torch.no_grad():
        for _ in range(10):
            solver.step(1e-3)
            divs.append(float(torch.abs(solver._divergence()).max()))
    assert max(divs) < 1e-10, f"divergence grew to {max(divs):.3e}"
    # monotone-ish, never amplified by orders of magnitude
    assert divs[-1] < 1e-6


# --------------------------------------------------------------------------- #
# anchor 2: Young-Laplace pressure jump of a static droplet (was xfail)
# --------------------------------------------------------------------------- #




@pytest.mark.parametrize("radius", [0.2, 0.25, 0.3])
def test_static_droplet_laplace_pressure(radius):
    """dp = sigma/R for a static droplet held by surface tension.

    Acceptance: relative error < 15%.  This test only passes if the CSF force
    actually reaches the momentum prediction (``surface_tension=True`` is
    otherwise inert and dp would be 0).
    """
    n, sigma, dt = 64, 0.0728, 1e-4
    cfg = GridConfig(nx=n, ny=n, length=1.0, height=1.0, time_step=dt)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1000.0,
                        mu_liquid=1.0e-3, mu_gas=1.0e-3, sigma=sigma,
                        surface_tension=True, gravity=0.0, top_atmosphere=True)
    solver = TwoPhaseSolver(cfg, tp)
    solver.set_alpha(_circular_alpha(n, n, radius, solver.dx))

    with torch.no_grad():
        for _ in range(5):
            solver.step(dt)

    c = n // 2
    p_center = float(solver.p[c - 1:c + 1, c - 1:c + 1].mean())
    p_far = float(solver.p[2:5, 2:5].mean())
    dp = p_center - p_far
    dp_ref = sigma / radius
    rel = abs(dp - dp_ref) / dp_ref
    assert rel < 0.15, (
        f"Laplace jump: dp = {dp:.5f} Pa vs sigma/R = {dp_ref:.5f} Pa "
        f"(rel.err {rel * 100:.2f}% > 15%)")






def test_surface_tension_switch_is_respected():
    """Negative control: with surface_tension=False the CSF force is off."""
    n, sigma, dt = 48, 0.0728, 1e-4
    cfg = GridConfig(nx=n, ny=n, length=1.0, height=1.0, time_step=dt)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=1000.0,
                        mu_liquid=1.0e-3, mu_gas=1.0e-3, sigma=sigma,
                        surface_tension=False, gravity=0.0, top_atmosphere=True)
    solver = TwoPhaseSolver(cfg, tp)
    solver.set_alpha(_circular_alpha(n, n, 0.25, solver.dx))
    with torch.no_grad():
        for _ in range(3):
            solver.step(dt)
    c = n // 2
    dp = float(solver.p[c, c] - solver.p[2, 2])
    assert abs(dp) < 1e-6, f"CSF must be gated off, got dp = {dp:.3e} Pa"
    assert float(torch.abs(solver.u).max()) < 1e-9


# --------------------------------------------------------------------------- #
# anchor 3: fully closed domain rejected at configuration time (was xfail)
# --------------------------------------------------------------------------- #




def test_pressure_scale_is_hydrostatic():
    """Reject the old operator: p must be O(rho*g*dy), not O(rho*g*dy/dt)."""
    solver = _column_solver(n=32, dt=1e-3)
    with torch.no_grad():
        solver.step(1e-3)
    rho_g_dy = 1000.0 * 9.81 * solver.dy
    p_max = float(solver.p.abs().max())
    # the column is 0.5 m high -> ~20 cells
    assert 5.0 * rho_g_dy < p_max < 100.0 * rho_g_dy, (
        f"p_max = {p_max:.1f} Pa is not of hydrostatic magnitude "
        f"(rho*g*dy = {rho_g_dy:.1f} Pa)")


# --------------------------------------------------------------------------- #
# single-phase path must remain untouched
# --------------------------------------------------------------------------- #




def test_single_phase_path_is_unchanged():
    """two_phase disabled -> no gravity/CSF coupling, still a quiet field."""
    cfg = GridConfig(nx=16, ny=16, length=1.0, height=1.0, time_step=0.0,
                     density=1.0)
    solver = TwoPhaseSolver(cfg, TwoPhaseConfig(enabled=False))
    with torch.no_grad():
        solver.step(0.0)
    assert float(torch.abs(solver.u).max()) == 0.0
    assert float(torch.abs(solver.v).max()) == 0.0
    assert solver.alpha is None


# --------------------------------------------------------------------------- #
# carried from the m2538 generation (see module docstring)
# --------------------------------------------------------------------------- #




def test_hydrostatic_anchor_is_continuous_at_the_interface():
    """Pressure must be continuous across a flat interface without surface tension."""
    n = 32
    solver, _ = _solver(n=n)
    jlo, jhi = n // 2 - 1, n // 2
    jump = float(solver.p[jhi, 0] - solver.p[jlo, 0])
    rho_face = 0.5 * (1000.0 + 1.0)
    np.testing.assert_allclose(jump, -rho_face * 9.81 * solver.dy, rtol=0,
                               atol=1e-9)
