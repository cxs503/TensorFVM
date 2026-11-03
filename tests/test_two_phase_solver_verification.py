"""Verification anchors for the merged two-phase VOF solver.

``tensorfvm.two_phase_solver.TwoPhaseSolver`` (846 lines, ``tp.enabled=True``)
previously had no test at all.  These tests check it against *analytic*
references that need no external data:

1. hydrostatic pressure initialisation, ``p = rho_l g (h_i - y)`` below the
   interface and ``p = rho_g g (H - y)`` above it (exact, piecewise linear);
2. a stratified column at rest must stay at rest under gravity
   (steady state of the discrete equations, exact);
3. the 2D Young-Laplace pressure jump of a static circular drop,
   ``dp = sigma / R`` (exact for a circular interface in 2D; the sphere value
   ``2 sigma / R`` does not apply);
4. a fully sealed domain (``top_atmosphere=False``), whose pressure Poisson
   problem is all-Neumann with a one-dimensional null space.

Tests 2-3 are ``xfail(strict=True)``: the current transient projection path
does not reproduce the analytic reference, so they are executable
reproductions of the defects documented in
``scripts/verification-two-phase-solver/report.md``.  ``strict=True`` turns
each of them red the moment the solver is fixed, which is the signal to
promote it to a plain assertion.  Test 4 (sealed-domain well-posedness
guard) is a hard assertion since m2751: the solver now rejects the
all-Neumann configuration at ``step()`` time.

Cases are small (n <= 64, <= 10 steps) and run in a few seconds.
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

    The top cell row is excluded: it is pinned to the atmospheric gauge
    ``p = 0``, whereas the piecewise formula still returns
    ``rho_g g (H - y_top) > 0`` there.
    """
    n = 32
    solver, _ = _solver(n=n)
    g, rho_l, rho_g, h_i = 9.81, 1000.0, 1.0, 0.5
    y = (torch.arange(n, dtype=torch.float64) + 0.5).reshape(-1, 1) * solver.dy
    exact = torch.where(y < h_i, rho_l * g * (h_i - y),
                        rho_g * g * (1.0 - y)).expand(n, n)
    error = (solver.p[:-1] - exact[:-1]).abs().max().item()
    assert error / exact.abs().max().item() < 1e-12
    assert abs(solver.p[-1, 0].item()) < 1e-12          # atmospheric gauge
    assert abs(solver.p[0, 0].item() - rho_l * g * (h_i - 0.5 * solver.dy)) < 1e-9


def test_alpha_field_volume_is_conserved_at_initialisation():
    """A drop discretised by a signed-distance ramp keeps its area."""
    solver, _ = _solver(n=64, gravity=0.0, drop_radius=0.2)
    area = float(solver.alpha.sum()) * solver.dx * solver.dy
    exact = math.pi * 0.2 ** 2
    assert abs(area - exact) / exact < 3e-3


@pytest.mark.xfail(strict=True, reason=(
    "the transient projection path amplifies a resting hydrostatic column "
    "(scripts/verification-two-phase-solver/report.md)"))
def test_stratified_column_at_rest_stays_at_rest():
    """Gravity balanced by the analytic pressure gradient: u = v = 0 for all t."""
    solver, _ = _solver(n=16, gravity=9.81, mu=1e-3)
    for _ in range(10):
        solver.step(1e-5)
    assert float(solver.u.abs().max()) < 1e-9
    assert float(solver.v.abs().max()) < 1e-9


@pytest.mark.xfail(strict=True, reason=(
    "surface_tension is never added to the transient projection momentum "
    "predictor, so a static drop carries dp = 0 instead of sigma/R"))
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
    """A sealed domain has no pressure datum; the solver must not run silently.

    All-Neumann pressure Poisson: the operator is singular (constant null
    space), so the configuration is ill-posed and must be rejected with a
    ``ValueError`` instead of iterating on garbage.
    """
    solver, _ = _solver(n=16, gravity=9.81, mu=1e-3, top_atmosphere=False)
    with pytest.raises(ValueError, match='sealed domain'):
        solver.step(1e-5)
    # A datum on any boundary restores well-posedness: the same grid with
    # the top row open must step without raising.
    open_solver, _ = _solver(n=16, gravity=9.81, mu=1e-3, top_atmosphere=True)
    open_solver.step(1e-5)


def test_single_phase_mode_is_dormant():
    """``enabled=False`` keeps every two-phase field unallocated."""
    solver = TwoPhaseSolver(GridConfig(nx=8, ny=8))
    assert solver.alpha is None and solver.vof is None and solver.fs_bc is None
    assert solver.rho_field is not None and float(solver.rho_field.max()) == 1.0