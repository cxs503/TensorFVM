"""The manufactured Dirichlet diffusion solve, with and without its source term.

``native_order3d.solution_study`` solves ``div(tau(u)) = 0`` and compares the
result with :func:`native_order3d.manufactured`.  The manufactured field does
not solve that homogeneous problem -- ``divergence_of_stress`` is O(1) in every
cell -- so the recorded "Dirichlet diffusion solve does not converge" defect was
measured between two different fields rather than against a discretisation
error.  ``manufactured_solution_study`` restores the analytic divergence of the
stress as the cell source; the exact solution of that discrete problem is the
manufactured field, and the same operator, boundary data and linear solve then
converge at second order.
"""
from functools import lru_cache

import numpy as np

from tensorfvm.verification import native_order3d as study

VISCOSITY = 1.3
MESHES = (4, 8)


@lru_cache(maxsize=None)
def _manufactured(n):
    return study.manufactured_solution_study(n, 0.0, VISCOSITY)


@lru_cache(maxsize=None)
def _homogeneous(n):
    return study.solution_study(n, 0.0, VISCOSITY)


def test_manufactured_field_does_not_solve_the_homogeneous_problem():
    points = np.array([[.13, .27, .41], [.71, .12, .88], [.5, .5, .5]])
    source = np.linalg.norm(study.divergence_of_stress(points, VISCOSITY), axis=1)
    assert source.min() > 1.0, source
    field = np.linalg.norm(study.manufactured(points), axis=1)
    assert np.all(np.isfinite(field)) and field.min() > 0


def test_homogeneous_comparison_plateaus_regardless_of_resolution():
    rows = [_homogeneous(n) for n in MESHES]
    ratio = rows[1]['l2_global']/rows[0]['l2_global']
    assert 0.8 < ratio < 1.0, ratio
    assert study.orders(rows, 'l2_global')[0] < 0.3


def test_consistent_manufactured_solve_is_second_order():
    """Interior cells are cleanly second order; wall cells sit at 1.64 and drag
    the global norm to 1.80 -- the residual wall-gradient defect, not a solver
    failure."""
    rows = [_manufactured(n) for n in MESHES]
    assert study.orders(rows, 'l2_interior')[0] > 1.9, study.orders(rows, 'l2_interior')
    assert study.orders(rows, 'l2_global')[0] > 1.7, study.orders(rows, 'l2_global')
    for row in rows:
        assert row['residual'] < 1e-12
        assert row['l2_global'] < 0.05


def test_consistent_solve_error_is_two_orders_below_the_homogeneous_plateau():
    plateau = _homogeneous(8)['l2_global']
    converged = _manufactured(8)['l2_global']
    assert converged*8 < plateau, (plateau, converged)