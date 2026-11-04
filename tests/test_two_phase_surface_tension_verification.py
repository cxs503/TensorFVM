"""Verification gates for the two-phase CSF (capillary) chain.

Evidence grading for this module (platform bar: >3% vs an analytic reference is a
FAIL):

* REAL -- ``csf_surface_tension`` is now coupled into the transient projection
  momentum predictor (it used to be computed and dropped; see
  ``test_csf_force_enters_momentum_predictor``).
* REAL -- the projection right-hand side is dimensionally consistent with the
  assembled operator (``-V*div(u*)/dt``) and the top-atmosphere Dirichlet row is
  eliminated symmetrically; both were missing and are now regression-gated by
  ``test_projection_pressure_is_laplace_scaled``.
* REAL -- the 2D Young--Laplace jump ``dp = sigma/R`` is reproduced inside the 3 %
  gate once the CSF force is sampled on the MAC faces with the same two-point
  ``alpha`` difference the pressure gradient uses (balanced force, Francois et
  al. 2006) and the curvature is taken from a Brackbill--Kothe--Zemach smoothed
  colour field.  Both are the solver defaults; the legacy cell-force average is
  kept as ``csf_balance="legacy"`` for the negative control below.  Grid ladder,
  curvature and spurious-current tables: ``docs/two-phase-csf-balance/``
  (``scripts/audit_two_phase_csf_balance.py``).  The former strict ``xfail`` here
  is retired by that change; the older unstable-chain evidence stays in
  ``docs/two-phase-surface-tension-audit/``.
"""
import math

import pytest
import torch

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver


def _solver(n=32, radius=0.2, sigma=0.07, mu=1e-2, surface_tension=True):
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=1000.0, rho_gas=1.0, mu_liquid=mu, mu_gas=mu,
        gravity=0.0, sigma=sigma, surface_tension=surface_tension,
        top_atmosphere=True,
    )
    solver = TwoPhaseSolver(GridConfig(nx=n, ny=n, length=1.0, height=1.0), tp)
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * solver.dx
    rr = torch.sqrt((x[None, :] - 0.5) ** 2 + (x[:, None] - 0.5) ** 2)
    solver.set_alpha(torch.clamp((radius - rr) / solver.dx + 0.5, 0.0, 1.0))
    return solver, rr


def test_csf_force_enters_momentum_predictor():
    """A capillary source must move the fluid; sigma=0 must stay exactly at rest."""
    sigma_solver, _ = _solver(sigma=0.07)
    sigma_solver.step(1e-5)
    moved = max(float(sigma_solver.u.abs().max()), float(sigma_solver.v.abs().max()))
    assert moved > 0.0, "capillary force never reached the momentum predictor"

    inert, _ = _solver(sigma=0.07, surface_tension=False)
    inert.step(1e-5)
    resting = max(float(inert.u.abs().max()), float(inert.v.abs().max()))
    assert resting == 0.0, "single-phase/disabled path must stay bit-identical"


def test_projection_pressure_is_laplace_scaled():
    """The first projection must be O(sigma/R), not O(sigma/R / (dx*dy)).

    Dropping the cell volume from the Poisson right-hand side (or leaving the
    Dirichlet row coupled) inflates the pressure by ~1/(dx*dy) -- three orders of
    magnitude on any usable grid -- so a loose order-unity band is enough to gate
    the defect while tolerating the coarse initial transient.
    """
    n, radius, sigma = 32, 0.2, 0.07
    solver, _ = _solver(n=n, radius=radius, sigma=sigma)
    solver.step(1e-5)
    exact = sigma / radius
    peak = float(solver.p.abs().max())
    assert peak < 10.0 * exact, (
        f"pressure {peak:.3g} Pa is far above the capillary scale {exact:.3g} Pa; "
        "the projection right-hand side lost its cell-volume factor")
    assert math.isfinite(peak)


def test_static_drop_young_laplace_jump():
    """2D Young-Laplace: a static circular drop must carry dp = sigma/R (<3%)."""
    radius, sigma = 0.2, 0.07
    solver, rr = _solver(radius=radius, sigma=sigma)
    for _ in range(6):
        solver.step(1e-5)
    assert torch.isfinite(solver.p).all()
    inside = float(solver.p[rr < 0.5 * radius].mean())
    outside = float(solver.p[rr > 1.6 * radius].mean())
    exact = sigma / radius
    assert abs((inside - outside) / exact - 1.0) < 0.03, (
        f"dp {(inside - outside):.5f} Pa vs sigma/R {exact:.5f} Pa")


def test_legacy_cell_force_is_the_negative_control():
    """The retired discretisation must still be reachable and still be worse.

    Averaging the *cell-centred* CSF force onto the faces leaves a residual curl
    the projection cannot cancel, so the same static drop develops spurious
    currents two orders of magnitude larger (measured 4.9e-5 vs 5.2e-7 m/s at
    n=32 in ``docs/two-phase-csf-balance/summary.json``).  Keeping the legacy
    path executable is what makes the balanced default falsifiable.
    """
    radius = 0.2
    balanced, _ = _solver(radius=radius)
    legacy, _ = _solver(radius=radius)
    legacy.tp.csf_balance = 'legacy'
    for _ in range(6):
        balanced.step(1e-5)
        legacy.step(1e-5)
    u_bal = max(float(balanced.u.abs().max()), float(balanced.v.abs().max()))
    u_leg = max(float(legacy.u.abs().max()), float(legacy.v.abs().max()))
    assert u_bal < u_leg / 50.0, (u_bal, u_leg)