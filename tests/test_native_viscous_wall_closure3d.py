"""Audit of the ``wall_closure='three_point'`` Dirichlet wall closure.

The closure is documented in :mod:`tensorfvm.native_viscous3d` as "exact for
affine fields, second order for smooth fields", and it exists to attack the
recorded O(1) wall-cell divergence plateau of the default two-point closure.
Before this module it had **no test and no exercised call site**, and it was in
fact inert: ``_assemble_three_point_wall`` measured its alignment against the
stored face normal, which points *out* of the owner cell, so every candidate
second cell failed the ``align > 0.5`` test and every face was silently left on
the two-point closure.  The repair is exercised here from three sides:

1. activation -- the upgraded-face counter must be non-zero, so a silently
   inert closure can never again look like a passing feature;
2. exactness -- affine traction to machine precision and an exact quadratic
   face-normal derivative, which is the property the two-point closure lacks
   and the reason the closure exists;
3. safety -- faces whose centroid is off the cell-centre line (triangulated
   faces) keep the two-point closure, because the one-dimensional weights are
   only affine-exact when the abscissae are collinear with the face normal.

The recorded plateau itself is *not* claimed to be closed by this repair; the
last test keeps it visible.
"""
from functools import lru_cache

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import native_order3d as study

VISCOSITY = 1.3


def _operator(n, closure, triangulate=False, warp=0.0):
    mesh = study.hex_box(n, warp, triangulate_faces=triangulate)
    return mesh, NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY, wall_closure=closure)


@lru_cache(maxsize=None)
def _truncation(n, closure, triangulate=False):
    """Wall/interior cell divergence error of the assembled traction."""
    mesh, op = _operator(n, closure, triangulate)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = study.manufactured(mesh.face_centers[op.dirichlet])
    flux = op.momentum_flux(study.manufactured(mesh.cell_centers), dirichlet_values=data)
    discrete = op.op.integrate(flux)/mesh.cell_volumes[:, None]
    exact = study.divergence_of_stress(mesh.cell_centers, VISCOSITY)
    error = np.linalg.norm(discrete - exact, axis=1)
    wall = study._wall_mask(mesh, n)
    return dict(upgraded=int(op.wall_second_order_faces),
                faces=int(op.dirichlet.sum()),
                l2_wall=float(np.sqrt(np.mean(error[wall]**2))),
                linf_wall=float(error[wall].max()),
                l2_interior=float(np.sqrt(np.mean(error[~wall]**2))))


@lru_cache(maxsize=None)
def _dirichlet_solve(n, closure, triangulate=False):
    """Dirichlet solve of the consistent manufactured problem (exact source)."""
    mesh, op = _operator(n, closure, triangulate)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = study.manufactured(mesh.face_centers[op.dirichlet])
    rhs = -(op.vector_divergence_matrix @ op.momentum_flux_source(dirichlet_values=data).ravel())
    rhs = rhs + (study.divergence_of_stress(mesh.cell_centers, VISCOSITY)
                 * mesh.cell_volumes[:, None]).ravel()
    solved = spsolve(op.momentum_diffusion_matrix.tocsc(), rhs).reshape(len(mesh.cells), 3)
    error = np.linalg.norm(solved - study.manufactured(mesh.cell_centers), axis=1)
    wall = study._wall_mask(mesh, n)
    return dict(l2=float(np.sqrt(np.mean(error**2))), linf=float(error.max()),
                l2_wall=float(np.sqrt(np.mean(error[wall]**2))))


def _face_gradient(op, values, wall_values):
    """Full face gradient = assembled matrices + boundary-data part."""
    gradient = np.empty((op.F, 3))
    source = op.face_gradient_source(0.0, wall_values)
    for k in range(3):
        gradient[:, k] = op.face_gradient_matrices[k] @ values + source[:, k]
    return gradient


def test_two_point_default_leaves_the_three_point_closure_inert():
    _, op = _operator(4, 'two_point')
    assert op.wall_second_order_faces == 0
    assert np.all(op.wall_second_cell < 0)


@pytest.mark.parametrize('n', [4, 6])
def test_three_point_upgrades_every_dirichlet_face_on_quad_face_mesh(n):
    """Guards the silent no-op: two stacked defects once kept this count at 0."""
    _, op = _operator(n, 'three_point')
    assert op.wall_second_order_faces == int(op.dirichlet.sum())
    assert np.all(op.wall_second_cell[op.dirichlet] >= 0)


@pytest.mark.parametrize('closure', ['two_point', 'three_point'])
def test_affine_traction_is_exact_with_either_closure(closure):
    mesh, op = _operator(4, closure)
    gradient = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
    base = np.array([1.2, .8, -.4])
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = base + mesh.face_centers[op.dirichlet] @ gradient.T
    flux = op.momentum_flux(base + mesh.cell_centers @ gradient.T, dirichlet_values=data)
    symmetric = gradient + gradient.T - 2/3*np.trace(gradient)*np.eye(3)
    np.testing.assert_allclose(flux, VISCOSITY*mesh.area_vectors @ symmetric.T,
                               atol=1e-12, rtol=0)


def test_quadratic_face_normal_derivative_is_exact_with_three_point():
    """The measured gain of the closure: the two-point estimate is O(h)."""
    matrix = np.array([[.3, .1, -.2], [.05, -.4, .2], [.1, .2, .6]])
    shift = np.array([.2, -.1, .3])

    def field(points):
        return np.einsum('ia,ab,ib->i', points, matrix, points) + points @ shift

    def gradient(points):
        return points @ (matrix + matrix.T) + shift

    errors = {}
    for closure in ('two_point', 'three_point'):
        mesh, op = _operator(4, closure)
        wall_values = np.zeros((op.F, 3))
        wall_values[op.dirichlet] = field(mesh.face_centers[op.dirichlet])[:, None]
        face_gradient = _face_gradient(op, field(mesh.cell_centers), wall_values[:, 0])
        computed = np.einsum('fb,fb->f', face_gradient, op.normal)
        exact = np.einsum('fb,fb->f', gradient(mesh.face_centers), op.normal)
        errors[closure] = np.abs(computed - exact)[op.dirichlet].max()
    assert errors['three_point'] < 1e-12, errors
    assert errors['two_point'] > 1e-2, errors


def test_triangulated_face_centroids_keep_the_two_point_closure():
    """Off-axis centroids break the one-dimensional exactness, so they fall back."""
    _, quad = _operator(4, 'three_point', triangulate=False)
    _, tri = _operator(4, 'three_point', triangulate=True)
    assert quad.wall_second_order_faces == int(quad.dirichlet.sum())
    assert tri.wall_second_order_faces == 0
    assert np.all(tri.wall_second_cell < 0)
    mesh, baseline = _operator(4, 'two_point', triangulate=True)
    values = study.manufactured(mesh.cell_centers)
    data = np.zeros((baseline.F, 3))
    data[baseline.dirichlet] = study.manufactured(mesh.face_centers[baseline.dirichlet])
    np.testing.assert_array_equal(tri.momentum_flux(values, dirichlet_values=data),
                                  baseline.momentum_flux(values, dirichlet_values=data))


def test_three_point_reduces_the_dirichlet_solve_error_at_fixed_resolution():
    two_point = _dirichlet_solve(8, 'two_point')['l2']
    three_point = _dirichlet_solve(8, 'three_point')['l2']
    assert three_point < 0.8*two_point, (three_point, two_point)


def test_wall_cell_divergence_plateau_is_not_closed_by_the_closure():
    """Honest record: the O(1) plateau survives; this repair is not its cure."""
    for closure in ('two_point', 'three_point'):
        rows = [_truncation(n, closure) for n in (4, 8)]
        order = study.orders(rows, 'l2_wall')[0]
        assert order < 0.3, (closure, order)
        assert rows[1]['linf_wall'] > 0.8*rows[0]['linf_wall'], (closure, rows)