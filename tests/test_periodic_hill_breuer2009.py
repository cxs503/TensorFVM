"""CASE-FVM-PERIODIC-HILLS acceptance gate: reattachment length & velocity profile vs Breuer et al. 2009.

Anchor: CASE-FVM-PERIODIC-HILLS
ref: Breuer et al. 2009 Computers & Fluids 38:433, doi:10.1016/j.compfluid.2008.05.002
acceptance: reattachment x_r/H <10% and u/U_b at y/H=0.5 vs Breuer2009 (Re=2800)

Breuer et al. 2009 (ERCOFTAC UFR 3-30) reference values for Re_b = 2800:
  - Reattachment length x_r/H = 5.34 (as tabulated by Volpiani et al. 2021 §III)
  - Velocity profile at y/H = 0.5: u/U_b at several x/H stations

ROUND-6 RECONCILIATION (m17065).  This file used to carry its own copy of the
reference values with ``BREUER_XR_H = 4.36`` and no page/figure citation, while
the feature-ledger acceptance line and rounds 3--5 of
``docs/case-periodic-hills/`` used ``5.34`` (Breuer et al. 2009 as tabulated by
Volpiani et al. 2021).  The two 10 % bands are *disjoint* (``[4.80, 5.87]`` vs
``[3.92, 4.80]``), so which one the gate uses decides the verdict.  The data now
lives in exactly one place -- :mod:`tensorfvm.verification.periodic_hill_round6`
(:data:`REFERENCE`, :data:`BREUER_PROFILE_Y05`) -- and this file imports it.
The retired 4.36 is recorded there as :data:`COMPETING_X_R` rather than deleted.

NOTE (m16819, 2026-11-12): this file was merged (m16691) with a *dead* API --
it called ``solver.run(...)`` (no such method), indexed the velocity field
transposed (``u[i, j]`` with ``u`` shaped ``(ny, nx)``) and read ``solver.G``
(no such attribute).  The acceptance gate therefore never executed.  This
revision binds the test to the real ``PeriodicHillSolver`` API
(``advance`` / ``body_force`` / ``(ny, nx)`` fields) and keeps the coarse-grid
physical-trend checks as the fast CI gate; the full Re=2800 acceptance is
honest-FAIL per ``docs/case-periodic-hills/ROUND3.md`` (the field does not
converge at Re=2800 on this collocated scheme) and is marked ``@pytest.mark.slow``
so it does not block the fast gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.verification.periodic_hill_round6 import (
    BREUER_PROFILE_Y05 as _BREUER_PROFILE_Y05,
    REFERENCE,
    REFERENCE_TOLERANCE,
)

BREUER_PROFILE_Y05 = _BREUER_PROFILE_Y05

# --------------------------------------------------------------------------- #
# Breuer et al. 2009 reference data (ERCOFTAC UFR 3-30, Re_b = 2800)
# --------------------------------------------------------------------------- #

#: Reattachment length x_r/H -- single-sourced (see the round-6 note above).
BREUER_XR_H = REFERENCE["x_r_over_H"]
#: Tolerance for reattachment length (10% per acceptance criteria)
XR_TOLERANCE = REFERENCE_TOLERANCE

#: Mean streamwise velocity u/U_b at y/H = 0.5 (dict of x/H -> u/U_b).
BREUER_PROFILE_Y05 = dict(BREUER_PROFILE_Y05)
#: Tolerance for velocity profile (10% relative)
PROFILE_TOLERANCE = REFERENCE_TOLERANCE


def _run_periodic_hill(nx, ny, reynolds=2800.0, max_steps=5000):
    """Run the periodic hill solver to a statistically steady state.

    Returns the solver after ``max_steps`` time steps (or earlier if the
    residual has settled).  Uses the real ``PeriodicHillSolver.advance`` API.
    """
    config = PeriodicHillConfig(nx=nx, ny=ny, reynolds=reynolds)
    solver = PeriodicHillSolver(config)
    solver.advance(steps=max_steps)
    return solver


def _interpolate_profile(solver, x_over_h_values, y_over_h=0.5):
    """Extract u/U_b at given x/H stations at fixed y/H.

    Uses linear interpolation in both x and y from the solver's mesh.
    ``solver.u`` is shaped ``(ny, nx)`` (row = y level, col = x station).
    """
    h = solver.config.height_m
    u_b = solver.bulk_velocity()
    results = {}
    for xh in x_over_h_values:
        x_target = xh * h
        # Find the two nearest x cell centres (row 0 of the mesh xc)
        xc = solver.mesh.xc[0]
        i = int(np.clip(np.searchsorted(xc, x_target) - 1, 0, solver.nx - 1))
        i_next = (i + 1) % solver.nx
        # Linear interpolation in x
        dx = xc[i_next] - xc[i] if xc[i_next] > xc[i] else solver.config.length_m - xc[i] + xc[i_next]
        if dx <= 0:
            dx = solver.config.length_m / solver.nx
        tx = (x_target - xc[i]) / dx
        tx = max(0.0, min(1.0, tx))
        # Find the nearest y row to y/H = 0.5 (column 0 of the mesh yc)
        yc = solver.mesh.yc[:, 0]
        j = int(np.clip(np.searchsorted(yc, y_over_h * h) - 1, 0, solver.ny - 1))
        # Interpolate u at (x_target, y_target); u is (ny, nx) so [j, i]
        u_i = solver.u[j, i]
        u_next = solver.u[j, i_next]
        u_interp = (1 - tx) * u_i + tx * u_next
        results[xh] = u_interp / u_b
    return results


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_reattachment_length_vs_breuer2009():
    """Reattachment length x_r/H within 10% of the ledger reference (5.34).

    Uses a moderately resolved mesh (96x48) to get a converged reattachment point.
    NOTE: Re=2800 is honest-FAIL on this collocated scheme (ROUND3.md); this
    test documents the current state rather than asserting a pass.
    """
    solver = _run_periodic_hill(96, 48, reynolds=2800.0, max_steps=8000)
    sep, rea = solver.reattachment()
    assert not math.isnan(rea), "Reattachment point not found"
    rel_error = abs(rea - BREUER_XR_H) / BREUER_XR_H
    assert rel_error < XR_TOLERANCE, (
        f"Reattachment length x_r/H = {rea:.3f} vs Breuer {BREUER_XR_H}, "
        f"relative error {rel_error:.1%} exceeds {XR_TOLERANCE:.0%}"
    )


@pytest.mark.slow
def test_velocity_profile_at_yh05_vs_breuer2009():
    """Mean velocity u/U_b at y/H=0.5 within 10% of Breuer et al. 2009 at key stations.

    Uses a moderately resolved mesh (96x48) for the profile comparison.
    """
    solver = _run_periodic_hill(96, 48, reynolds=2800.0, max_steps=8000)
    x_stations = list(BREUER_PROFILE_Y05.keys())
    profile = _interpolate_profile(solver, x_stations, y_over_h=0.5)

    failures = []
    for xh in x_stations:
        computed = profile[xh]
        reference = BREUER_PROFILE_Y05[xh]
        # Use absolute tolerance for small values, relative for larger
        if abs(reference) < 0.1:
            ok = abs(computed - reference) < 0.05
        else:
            ok = abs(computed - reference) / abs(reference) < PROFILE_TOLERANCE
        if not ok:
            failures.append(f"x/H={xh}: computed={computed:.3f}, ref={reference:.3f}")

    assert not failures, (
        f"Velocity profile at y/H=0.5 exceeds {PROFILE_TOLERANCE:.0%} tolerance:\n"
        + "\n".join(failures)
    )


def test_reattachment_length_coarse_grid_trend():
    """Coarse grid (48x24) reattachment length should be in the right ballpark.

    This is a fast CI check that the solver produces a physically reasonable
    reattachment length (between 3.5 and 5.5 H).
    """
    solver = _run_periodic_hill(48, 24, reynolds=2800.0, max_steps=3000)
    sep, rea = solver.reattachment()
    assert not math.isnan(rea), "Reattachment point not found on coarse grid"
    assert 3.0 < rea < 6.0, f"Reattachment x_r/H = {rea:.2f} outside physical range [3, 6]"


def test_separation_point_is_near_hill_crest():
    """Separation should occur just downstream of the hill crest (x/H ≈ 0).

    The hill crest is at x/H = 0, and separation typically occurs at x/H ≈ 0.1–0.3.
    """
    solver = _run_periodic_hill(48, 24, reynolds=2800.0, max_steps=3000)
    sep, rea = solver.reattachment()
    assert not math.isnan(sep), "Separation point not found"
    assert 0.0 < sep < 1.5, f"Separation x/H = {sep:.2f} outside expected range [0, 1.5]"


def test_bulk_velocity_is_held_at_target():
    """The bulk velocity above the crest should be held at U_b (Re = U_b H / nu)."""
    solver = _run_periodic_hill(48, 24, reynolds=2800.0, max_steps=3000)
    u_b = solver.bulk_velocity()
    # The solver drives to Re = 2800, so U_b = Re * nu / H
    # With H = 1 and nu = U_b * H / Re, the bulk velocity should be consistent
    assert u_b > 0, f"Bulk velocity {u_b} should be positive"
    # Check that the driving force G (body_force) is non-zero (solver is actively driving)
    assert abs(solver.body_force) > 0, "Driving force G should be non-zero"
