"""Order-of-accuracy facts for the native 3-D viscous operator.

These tests do not state what the order *should* be; they pin what the merged
operator actually delivers, measured against an analytic manufactured field, so
that the recorded limitation cannot silently change in either direction.
"""
from functools import lru_cache

import numpy as np
import pytest

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import native_order3d as study

VISCOSITY = 1.3


# Mesh assembly dominates the runtime; one study per (n, warp) is enough.
@lru_cache(maxsize=None)
def _truncation(n, warp):
    return study.truncation_study(n, warp, VISCOSITY)


@lru_cache(maxsize=None)
def _faces(n, warp):
    return study.face_traction_study(n, warp, VISCOSITY)


@lru_cache(maxsize=None)
def _solution(n, warp):
    return study.solution_study(n, warp, VISCOSITY)


def test_analytic_stress_divergence_matches_finite_differences():
    points = np.array([[.13, .27, .41], [.71, .12, .88], [.5, .5, .5]])
    step = 1e-6
    gradient = study.manufactured_gradient(points)
    np.testing.assert_allclose(
        gradient, np.stack([(study.manufactured(points + step*np.eye(3)[b])
                            - study.manufactured(points - step*np.eye(3)[b]))/(2*step)
                           for b in range(3)], axis=2), atol=1e-7)
    numeric = np.zeros((len(points), 3))
    for j in range(3):
        unit = step*np.eye(3)[j]
        numeric += (study.stress(points + unit, VISCOSITY)
                    - study.stress(points - unit, VISCOSITY))[:, :, j]/(2*step)
    np.testing.assert_allclose(numeric, study.divergence_of_stress(points, VISCOSITY),
                               rtol=1e-7, atol=1e-6)


@pytest.mark.parametrize('warp', [0.0, 2.0])
def test_affine_velocity_gives_machine_precision_traction(warp):
    mesh = study.hex_box(4, warp)
    op = NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY)
    gradient = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
    base = np.array([1.2, .8, -.4])
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = base + mesh.face_centers[op.dirichlet] @ gradient.T
    flux = op.momentum_flux(base + mesh.cell_centers @ gradient.T, dirichlet_values=data)
    symmetric = gradient + gradient.T - 2/3*np.trace(gradient)*np.eye(3)
    expected = VISCOSITY*mesh.area_vectors @ symmetric.T
    np.testing.assert_allclose(flux, expected, atol=1e-12)
    # Piecewise-linear fields carry no cell-integrated viscous divergence.
    assert np.abs(op.op.integrate(flux)/mesh.cell_volumes[:, None]).max() < 1e-12


def test_interior_cell_divergence_is_second_order():
    rows = [_truncation(n, 0.0) for n in (4, 8)]
    order = study.orders(rows, 'l2_interior')[0]
    assert order > 1.5, order


def test_wall_cell_divergence_does_not_refine():
    """Recorded defect: wall-adjacent cells stay at O(1) truncation error."""
    rows = [_truncation(n, 0.0) for n in (4, 8)]
    assert study.orders(rows, 'l2_wall')[0] < 0.3
    assert rows[1]['linf_wall'] > 0.8*rows[0]['linf_wall']


def test_wall_normal_derivative_is_first_order():
    rows = [_faces(n, 0.0) for n in (4, 8)]
    order = study.orders(rows, 'rel_wall_normal_derivative')[0]
    assert 0.6 < order < 1.6, order


def test_homogeneous_comparison_shows_no_measurable_convergence():
    """Recorded plateau: comparing a homogeneous solve with a non-solution field."""
    rows = [_solution(n, 0.0) for n in (4, 8)]
    ratio = rows[1]['l2_global']/rows[0]['l2_global']
    assert ratio > 0.85, (ratio, study.orders(rows, 'l2_global')[0])
    assert rows[1]['linf_global'] > 0.9*rows[0]['linf_global']
    for row in rows:  # the linear solve itself is exact
        assert row['residual'] < 1e-12


@pytest.mark.xfail(strict=True, reason='Open defect: wall-adjacent cell divergence is O(1), not second order')
def test_wall_cell_divergence_must_be_second_order():
    rows = [_truncation(n, 0.0) for n in (4, 8)]
    assert study.orders(rows, 'l2_wall')[0] > 1.5


def test_dirichlet_diffusion_solution_converges_once_the_source_is_restored():
    """The former strict xfail here measured a missing source term, not the operator.

    ``_solution`` solves ``div(tau(u)) = 0`` and compares the result with the
    manufactured field, which has O(1) ``div(tau)``; the plateau above is that
    mismatch.  Restoring the analytic stress divergence as a cell source makes
    the manufactured field the exact solution of the discrete problem, and the
    same assembly converges at second order (see
    ``tests/test_native_viscous_manufactured_solve.py``).
    """
    rows = [study.manufactured_solution_study(n, 0.0, VISCOSITY) for n in (4, 8)]
    assert study.orders(rows, 'l2_interior')[0] > 1.9
    assert study.orders(rows, 'l2_global')[0] > 1.7