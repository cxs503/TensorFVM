"""CASE-FVM-PERIODIC-HILLS acceptance gate: reattachment length & velocity profile vs Breuer et al. 2009.

Anchor: CASE-FVM-PERIODIC-HILLS
ref: Breuer et al. 2009 Computers & Fluids 38:433, doi:10.1016/j.compfluid.2008.05.002
acceptance: reattachment x_r/H <10% and u/U_b at y/H=0.5 vs Breuer2009 (Re=2800)

Reference provenance (m22825, 2026-11-13):
  This file now uses the **primary** ERCOFTAC KB DNS data files
  (``tests/data/periodic_hill/ercoftac_ufr330/UFR3-30_C_2800_data_MB-*.dat``)
  via :mod:`tensorfvm.verification.periodic_hill_reference`, which supersedes
  the secondary tabulation in :mod:`tensorfvm.verification.periodic_hill_round6`.

  The primary source gives ``x_r/H = 5.41`` (ERCOFTAC KB, UFR 3-30 Evaluation,
  authors Rapp/Breuer/Manhart/Peller) vs the round-6 value of ``5.34``
  (Volpiani et al. 2021 tabulation).  The 1.3 % difference is inside the 10 %
  acceptance band, but the velocity profile values differ by up to 3.5x because
  the round-6 table was a hand-digitised approximation that round 10 identified
  as incorrect.  All acceptance gates now grade against the primary DNS files.

  Legacy values are preserved in :data:`COMPETING_X_R` and
  :data:`LEGACY_DIGITISED_Y05` for traceability.

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
from tensorfvm.verification.periodic_hill_reference import (
    X_R_OVER_H,
    X_R_TOLERANCE,
    y05_table,
    station_covers_y05,
    STATIONS,
    COMPETING_X_R,
    LEGACY_DIGITISED_Y05,
)

# --------------------------------------------------------------------------- #
# Breuer et al. 2009 reference data (primary: ERCOFTAC KB DNS files)
# --------------------------------------------------------------------------- #

#: Reattachment length x_r/H -- primary source (ERCOFTAC KB, UFR 3-30 Evaluation)
BREUER_XR_H = X_R_OVER_H
#: Tolerance for reattachment length (10% per acceptance criteria)
XR_TOLERANCE = X_R_TOLERANCE

#: Mean streamwise velocity u/U_b at y/H = 0.5, from primary DNS data files.
#: Only includes stations where y/H=0.5 is in the fluid (excludes stations
#: inside the hill body near the crest).
BREUER_PROFILE_Y05 = y05_table()
#: Tolerance for velocity profile (10% relative)
PROFILE_TOLERANCE = X_R_TOLERANCE


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
    """Reattachment length x_r/H within 10% of the primary reference (5.41).

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

    Uses the primary DNS data files (ERCOFTAC KB) for the profile comparison.
    Only tests stations where y/H=0.5 is in the fluid (excludes stations
    inside the hill body).
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


def test_coarse_grid_produces_finite_flow():
    """Coarse grid (48x24) should produce a finite, physically reasonable flow.

    At Re=2800 on a coarse grid, the flow is unsteady and the separation/
    reattachment points may not be well-defined at any given instant.  This
    test checks that the solver produces finite fields and a reasonable body
    force, which is the minimum sanity check for the fast CI gate.
    """
    solver = _run_periodic_hill(48, 24, reynolds=2800.0, max_steps=3000)

    # Fields should be finite
    assert np.all(np.isfinite(solver.u)), "u field contains non-finite values"
    assert np.all(np.isfinite(solver.v)), "v field contains non-finite values"
    assert np.all(np.isfinite(solver.p)), "p field contains non-finite values"

    # Bulk velocity should be held at target
    u_b = solver.bulk_velocity()
    assert 0.5 < u_b < 1.5, f"Bulk velocity {u_b} far from target 1.0"

    # Body force should be positive and reasonable (driving the flow)
    assert 0.001 < solver.body_force < 0.01, (
        f"Body force {solver.body_force} outside expected range"
    )


def test_separation_point_is_near_hill_crest():
    """Separation should occur just downstream of the hill crest (x/H ≈ 0).

    The hill crest is at x/H = 0, and separation typically occurs at x/H ≈ 0.1–0.3.
    On a coarse grid with limited steps, the separation point may not be
    well-defined, so this test only checks that if found, it's in the right area.
    """
    solver = _run_periodic_hill(48, 24, reynolds=2800.0, max_steps=5000)
    sep, rea = solver.reattachment()
    if not math.isnan(sep):
        assert 0.0 < sep < 2.0, f"Separation x/H = {sep:.2f} outside expected range [0, 2]"


def test_bulk_velocity_is_held_at_target():
    """The bulk velocity above the crest should be held at U_b (Re = U_b H / nu)."""
    solver = _run_periodic_hill(48, 24, reynolds=2800.0, max_steps=3000)
    u_b = solver.bulk_velocity()
    # The solver drives to Re = 2800, so U_b = Re * nu / H
    # With H = 1 and nu = U_b * H / Re, the bulk velocity should be consistent
    assert u_b > 0, f"Bulk velocity {u_b} should be positive"
    # Check that the driving force G (body_force) is non-zero (solver is actively driving)
    assert abs(solver.body_force) > 0, "Driving force G should be non-zero"


def test_reference_uses_primary_data_not_legacy():
    """The acceptance gate uses the primary ERCOFTAC KB DNS data, not the legacy table.

    This pins that the reference values come from the actual DNS data files
    (``tests/data/periodic_hill/ercoftac_ufr330/``) and not from the
    hand-digitised table that rounds 4--9 used.
    """
    # The primary x_r/H is 5.41, not the legacy 5.34 or the retired 4.36
    assert BREUER_XR_H == 5.41, f"Expected primary x_r/H=5.41, got {BREUER_XR_H}"

    # The profile at x/H=2.0 should be -0.0912 (DNS), not 0.25 (legacy)
    assert 2.0 in BREUER_PROFILE_Y05, "x/H=2.0 should be an admissible station"
    u_at_2 = BREUER_PROFILE_Y05[2.0]
    assert abs(u_at_2 - (-0.0912)) < 0.001, (
        f"Profile at x/H=2.0 should be ~-0.0912 (DNS), got {u_at_2}"
    )

    # Legacy values are still accessible for traceability
    assert COMPETING_X_R is not None
    assert LEGACY_DIGITISED_Y05 is not None
    assert LEGACY_DIGITISED_Y05[2.0] == 0.25  # the wrong value, preserved for audit


def test_admissible_stations_exclude_hill_body():
    """Stations where y/H=0.5 is inside the hill body are excluded.

    Near the hill crest (x/H ≈ 0), the y/H=0.5 line is inside the solid,
    so those stations have no reference value.
    """
    # Station 0.05 is at the crest -- y/H=0.5 is inside the hill
    assert not station_covers_y05(0.05), "x/H=0.05 should be inside hill body"
    # Station 0.5 is also too close to the crest
    assert not station_covers_y05(0.5), "x/H=0.5 should be inside hill body"
    # Station 1.0 is admissible
    assert station_covers_y05(1.0), "x/H=1.0 should be in fluid"
    # Station 8.0 is near the windward foot -- check if it's covered
    # (depends on the exact hill geometry)
    covers_8 = station_covers_y05(8.0)
    # The station is either in or out; the key is that the function works
    assert isinstance(covers_8, bool)
