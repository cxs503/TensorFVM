"""Coupled overset **Stokes** operator: what it verifies and what it does not.

Before this file the overset composite had been certified only as (a) an
interpolation device, (b) a scalar Laplacian (``test_overset_poisson.py``), with
``navier_stokes_verified: False`` recorded in its own metrics.  The velocity
field below is the first overset result in this repository measured against an
exact *coupled* solution, and the pressure row records an open defect instead
of a wish.

Three claims are pinned, in the order they matter:

1. the analytic reference really is an exact Stokes solution and exactly
   divergence free -- proved in SymPy at import, asserted here;
2. the assembled saddle-point system is solvable, algebraically converged, and
   its velocity error is below the 3 % acceptance band on two grids;
3. the gate can fail: flipping the pressure-gradient sign, or dropping the
   Rhie-Chow face-flux coupling, must destroy the solution.  A verification
   that cannot fail proves nothing.
"""
from __future__ import annotations


import numpy as np

from tensorfvm.overset import annular_grid, build_overset, cartesian_grid, circle_polygon
from tensorfvm.overset_stokes import (
    DIVERGENCE_IDENTICALLY_ZERO,
    STOKES_RESIDUAL_IDENTICALLY_ZERO,
    solve_overset_stokes,
)


def make_connectivity(n):
    return build_overset(
        cartesian_grid((-2, 2, -2, 2), n, n),
        annular_grid((0, 0), 0.3, 1.0, 2 * n, n // 4),
        circle_polygon((0, 0), 0.3),
        circle_polygon((0, 0), 0.65),
    )


def test_reference_field_is_an_exact_divergence_free_stokes_solution():
    assert DIVERGENCE_IDENTICALLY_ZERO
    assert STOKES_RESIDUAL_IDENTICALLY_ZERO


def test_coupled_stokes_solves_with_donor_coupled_pressure():
    coarse = solve_overset_stokes(make_connectivity(32)).metrics
    fine = solve_overset_stokes(make_connectivity(64)).metrics
    for m in (coarse, fine):
        assert m['stokes_linear_verified'] is True
        # No convection, no turbulence: this must not be sold as an NS solver.
        assert m['navier_stokes_verified'] is False
        assert m['strict_local_conservation'] is False
        assert m['pressure_gauge_rows'] == 1
        assert m['relative_algebraic_residual'] < 1e-10
        assert m['maximum_donor_constraint_residual'] < 1e-10
        assert m['fringe_cells'] > 0 and m['hole_cells'] > 0
        # Continuity is actually enforced, not merely assembled.
        assert m['maximum_discrete_divergence_defect_relative'] < 0.05
    for m in (coarse, fine):
        assert m['relative_l2_error_u'] < 0.03
        assert m['relative_l2_error_v'] < 0.03
        assert m['relative_l2_error_velocity'] < 0.03
    assert fine['relative_l2_error_velocity'] < coarse['relative_l2_error_velocity']


def test_pressure_error_stays_an_open_defect_not_a_pass():
    """Honest FAIL, pinned so it cannot be quietly upgraded to green.

    The velocity converges below 3 % while the pressure L2 error is 48 % -> 35 %
    with an observed order near 0.2: the Rhie-Chow coupling removes the exact
    rank deficiency of the collocated system but does not give a second-order
    pressure field on this composite.  These bounds are deliberately wide and
    wrap the *measured* values; they exist so a real fix (tighter bounds, order
    restored) has to be recorded instead of silently inherited.
    """
    coarse = solve_overset_stokes(make_connectivity(32)).metrics
    fine = solve_overset_stokes(make_connectivity(64)).metrics
    assert 0.3 < coarse['relative_l2_error_p'] < 0.7
    assert 0.2 < fine['relative_l2_error_p'] < 0.6


def test_gate_is_falsifiable():
    flipped = solve_overset_stokes(make_connectivity(32), flip_pressure_sign=True).metrics
    assert flipped['relative_l2_error_velocity'] > 1.0
    no_coupling = solve_overset_stokes(make_connectivity(32), rhie_chow=False).metrics
    assert no_coupling['relative_l2_error_p'] > 1.0


def test_donor_rows_cover_every_fringe_dof():
    """Each fringe cell must carry interpolation rows for u, v *and* p.

    A missing pressure row is exactly the defect that made an earlier version of
    this assembly silently singular; the solver raises on it, so a connectivity
    without fringe cells is rejected rather than solved.
    """
    connectivity = make_connectivity(32)
    fringe = sum(int(np.count_nonzero(np.asarray(s) == 2)) for s in connectivity.states)
    metrics = solve_overset_stokes(connectivity).metrics
    assert metrics['fringe_cells'] == fringe > 0
    assert metrics['active_cells'] > 0
    assert metrics['unknowns'] == 3 * sum(len(g.centers) for g in connectivity.grids)
