"""Regression guard for the spanwise (w) artefact of extruded 2D FVM cases.

Locks the m16524 attribution of ``CASE-FVM-KOVASZNAY-ORDER``:

* the discrete *cell* z-momentum row is consistent with the exact field
  (residual 0.0 for ``w == 0``) while the in-plane cell rows are O(h^2);
* the free tangential momentum unknowns on the symmetry (z-)planes are what
  generates the first-order spurious spanwise velocity: prescribing the same
  analytic Dirichlet data on those components reduces ``w`` to round-off;
* the delivered ``symmetry`` setup reproduces that artefact, and it lives on the
  *internal* faces, not on the symmetry planes.

If a future change to the hybrid face closure or the symmetry-plane handling
alters any of these, the test fails and the attribution must be revisited.
The cost is two n = 8 extruded solves plus one operator assembly.
"""

import numpy as np
import pytest

from tensorfvm.verification.kovasznay_extrusion import (
    error_budgets, exact_state_residual, extrusion_solver)

N = 8


@pytest.fixture(scope='module')
def symmetry_run():
    mesh, solver, exact = extrusion_solver(N, 40., tangential='symmetry')
    solver.solve(max_outer_iterations=80, tolerance=1e-8, relaxation=.7)
    return mesh, solver, exact


@pytest.fixture(scope='module')
def dirichlet_run():
    mesh, solver, exact = extrusion_solver(N, 40., tangential='dirichlet')
    solver.solve(max_outer_iterations=80, tolerance=1e-8, relaxation=.7)
    return mesh, solver, exact


def test_exact_state_z_momentum_row_is_consistent():
    """w == 0 satisfies the cell z-momentum equation for the analytic field."""
    mesh, solver, _ = extrusion_solver(N, 40., tangential='symmetry')
    residual = exact_state_residual(mesh, solver, 40.)
    assert residual['total']['cell-w'] == pytest.approx(0., abs=1e-14)
    # the in-plane cell rows are *not* exact: they carry the O(h^2) truncation
    h = 1. / N
    assert residual['total']['cell-u'] > 0.1 * h ** 2
    # the pressure path has no spanwise cross-leak into the cell rows
    assert residual['pressure']['cell-w'] == pytest.approx(0., abs=1e-14)


def test_free_tangential_planes_generate_first_order_spanwise_velocity(symmetry_run):
    mesh, solver, exact = symmetry_run
    budgets = error_budgets(mesh, solver, exact)
    assert budgets['max_abs_w'] > 1e-3                    # real artifact, not round-off
    # it lives on the internal (in-plane) faces, not on the symmetry planes
    assert budgets['face_w_abs_max_on_internal'] > 1e-3
    assert budgets['face_max_abs_w'] > 0.


def test_prescribed_tangential_planes_kill_spanwise_velocity(dirichlet_run):
    """Strong-BC control: the same case without the free tangential dofs."""
    mesh, solver, exact = dirichlet_run
    budgets = error_budgets(mesh, solver, exact)
    assert budgets['max_abs_w'] < 1e-12
    assert budgets['face_max_abs_w'] < 1e-12


def test_spanwise_artifact_dominates_the_mixed_norm(symmetry_run, dirichlet_run):
    """In-plane accuracy is unchanged by the closure; w is the whole story."""
    free = error_budgets(*symmetry_run)
    strong = error_budgets(*dirichlet_run)
    assert free['inplane_l2'] == pytest.approx(strong['inplane_l2'], rel=.15)
    assert free['spanwise_l2'] > 1e-3
    assert strong['spanwise_l2'] < 1e-12
    assert free['span_share_of_squared_error'] > .02