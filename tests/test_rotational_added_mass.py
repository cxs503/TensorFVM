"""Tests for the rotational (M_44..M_66) added mass and its exact anchors.

The heavy convergence study lives in
:mod:`tensorfvm.verification.rotational_added_mass`; these tests pin the
structure on coarse meshes so they stay fast:

* sphere about a diameter -> exactly zero (``r x n = 0``);
* body of revolution roll -> exactly zero to machine precision;
* parallel-axis anchor ``I_oo = m_a d^2`` for a sphere off the reference point;
* the ``6 x 6`` translational block reproduces the existing translational API;
* symmetry of the assembled matrix and the per-panel normal-velocity entry point;
* triaxial geometry: ``ellipsoid_panels`` reduces to ``spheroid_panels`` for
  ``c = b`` (bit-identical), carries the analytic volume, and the three
  rotational entries of a genuinely triaxial body match the closed form of
  :mod:`tensorfvm.ellipsoid_added_inertia` with zero rotation/rotation coupling.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.ellipsoid_added_inertia import rotational_added_inertia as closed_form
from tensorfvm.panel_added_mass import (
    added_mass_matrix,
    added_mass_matrix_6dof,
    displaced_volume,
    ellipsoid_panels,
    rotational_added_inertia,
    sphere_added_mass,
    sphere_offset_axis_added_inertia,
    sphere_panels_offset,
    sphere_rotational_added_inertia,
    spheroid_panels,
    surface_generators,
)
from tensorfvm.panel_potential import solve_sources
from tensorfvm.twin_hull_geometry import sphere_panels


def test_sphere_rotational_block_is_zero():
    """A sphere has zero added moment of inertia about every diameter."""
    a = 1.0
    panels = sphere_panels(a, 32, 18)
    A = added_mass_matrix_6dof(panels, rho=1.0)
    m_a = sphere_added_mass(a, 1.0)
    assert np.abs(A[3:, 3:]).max() / (m_a * a**2) < 1.0e-2
    for i in range(3):  # faceting error of the polyhedral sphere, see the report
        assert A[i, i] == pytest.approx(m_a, rel=3.0e-2)
    assert sphere_rotational_added_inertia(a, 1.0) == 0.0


def test_spheroid_roll_is_exactly_zero():
    """Body of revolution: the roll generator ``(r x n)_x`` vanishes on the mesh."""
    panels = spheroid_panels(2.0, 1.0, 16, 10)
    psi = surface_generators(panels)
    # the roll generator vanishes identically up to coordinate round-off
    assert np.abs(psi[:, 3]).max() < 1.0e-14
    A = added_mass_matrix_6dof(panels, rho=1.0)
    scale = displaced_volume(panels) * 2.0**2
    assert abs(A[3, 3]) / scale < 1.0e-10


def test_translational_block_matches_translation_api():
    """``added_mass_matrix`` must be the 3x3 block of the general 6x6 matrix."""
    panels = spheroid_panels(2.0, 1.0, 16, 10)
    M = added_mass_matrix(panels, rho=1.0, order=3)
    A = added_mass_matrix_6dof(panels, rho=1.0, order=3, directions=(0, 1, 2))
    assert np.allclose(A, M, rtol=1.0e-12, atol=1.0e-12)


def test_parallel_axis_offset_sphere():
    """Rotation about an axis offset by ``d``: exact ``I_oo = m_a d^2``."""
    a, d = 1.0, 1.5
    panels = sphere_panels_offset(a, (0.0, 0.0, d), 24, 14)
    A = added_mass_matrix_6dof(panels, rho=1.0)
    exact = sphere_offset_axis_added_inertia(a, d, 1.0)
    assert A[3, 3] == pytest.approx(exact, rel=5.0e-2)
    assert A[4, 4] == pytest.approx(exact, rel=5.0e-2)
    assert abs(A[5, 5]) / exact < 5.0e-2      # axis through the centre -> zero
    assert abs(A[3, 4]) / exact < 5.0e-2      # no x-y product of inertia
    assert sphere_offset_axis_added_inertia(a, 0.0, 1.0) == 0.0


def test_matrix_is_symmetric_and_rotational_helper_matches():
    panels = spheroid_panels(2.0, 1.0, 16, 10)
    A = added_mass_matrix_6dof(panels, rho=1.0)
    assert np.array_equal(A, A.T)
    rot = rotational_added_inertia(panels, rho=1.0, order=3)
    assert np.allclose(rot, A[3:, 3:], rtol=1.0e-12, atol=1.0e-12)


def test_solve_sources_normal_velocity_equals_uniform_flow():
    """``normal_velocity=U.n`` must reproduce the uniform-onset-flow solve."""
    panels = spheroid_panels(2.0, 1.0, 16, 10)
    U = np.array([0.0, 0.0, 1.0])
    sigma_u = solve_sources(panels, U, order=2)
    sigma_n = solve_sources(panels, np.zeros(3), order=2,
                            normal_velocity=panels.normals @ U)
    assert np.allclose(sigma_u, sigma_n, rtol=1.0e-12, atol=1.0e-12)


def test_requested_directions_and_validation():
    panels = spheroid_panels(2.0, 1.0, 16, 10)
    two = added_mass_matrix_6dof(panels, directions=(4, 5))
    assert two.shape == (2, 2)
    assert two[0, 0] == pytest.approx(two[1, 1], rel=1.0e-10)
    with pytest.raises(ValueError):
        added_mass_matrix_6dof(panels, rho=0.0)
    with pytest.raises(ValueError):
        added_mass_matrix_6dof(panels, directions=(6,))


def test_ellipsoid_panels_reduces_to_spheroid_panels_bit_identically():
    """``c = b`` must reproduce the published spheroid panelisation exactly."""
    for a, b in ((2.0, 1.0), (1.0, 1.0), (4.0, 0.5)):
        spheroid = spheroid_panels(a, b, 16, 10)
        ellipsoid = ellipsoid_panels(a, b, b, 16, 10)
        assert np.array_equal(spheroid.centroids, ellipsoid.centroids)
        assert np.array_equal(spheroid.normals, ellipsoid.normals)
        assert np.array_equal(spheroid.areas, ellipsoid.areas)
        assert np.array_equal(spheroid.corners, ellipsoid.corners)


def test_triaxial_panels_volume_normals_and_validation():
    a, b, c = 2.0, 1.5, 1.0
    panels = ellipsoid_panels(a, b, c, 32, 18)
    v_exact = 4.0 / 3.0 * np.pi * a * b * c
    assert displaced_volume(panels) == pytest.approx(v_exact, rel=3.0e-2)
    # outward (into the fluid) normals: positive projection on grad(F)
    grad = np.stack([panels.centroids[:, 0] / a**2, panels.centroids[:, 1] / b**2,
                     panels.centroids[:, 2] / c**2], axis=1)
    assert np.all(np.einsum("ij,ij->i", panels.normals, grad) > 0.0)
    with pytest.raises(ValueError):
        ellipsoid_panels(1.0, 0.0, 1.0)


def test_closed_form_anchor_landmarks():
    """The reference used by criteria C7/C8: exact zeros, equality, positivity."""
    for axis in range(3):
        assert closed_form(1.0, 1.0, 1.0, axis=axis) == 0.0        # sphere
    assert closed_form(2.0, 1.0, 1.0, axis=0) == 0.0               # spheroid roll
    assert closed_form(2.0, 1.0, 1.0, axis=1) == pytest.approx(
        closed_form(2.0, 1.0, 1.0, axis=2), rel=1.0e-14)           # axisymmetry
    triax = [closed_form(2.0, 1.5, 1.0, axis=axis) for axis in range(3)]
    assert all(v > 0.0 for v in triax)
    assert len({round(v, 9) for v in triax}) == 3                   # all distinct


def test_triaxial_rotational_block_matches_closed_form():
    """Three distinct entries against the closed form + machine-precision decoupling."""
    a, b, c = 2.0, 1.5, 1.0
    panels = ellipsoid_panels(a, b, c, 24, 14)
    rot = rotational_added_inertia(panels, rho=1.0, order=3, quadrature="gauss")
    exact = [closed_form(a, b, c, axis=axis) for axis in range(3)]
    for axis in range(3):
        # coarse mesh: faceting error only, but a wrong moment arm or a wrong
        # normal orientation would show up as an O(1) discrepancy
        assert rot[axis, axis] == pytest.approx(exact[axis], rel=2.0e-1)
    scale = max(float(rot[axis, axis]) for axis in range(3))
    assert abs(rot[0, 1]) / scale < 1.0e-10
    assert abs(rot[0, 2]) / scale < 1.0e-10
    assert abs(rot[1, 2]) / scale < 1.0e-10
    # symmetric by construction
    assert np.array_equal(rot, rot.T)


def test_triaxial_rotational_block_narrows_with_mesh():
    """The panel value must move *towards* the closed form under refinement."""
    a, b, c = 2.0, 1.5, 1.0
    exact = [closed_form(a, b, c, axis=axis) for axis in range(3)]
    err = []
    for n_theta, n_phi in ((16, 10), (32, 18)):
        panels = ellipsoid_panels(a, b, c, n_theta, n_phi)
        rot = rotational_added_inertia(panels, rho=1.0, order=3, quadrature="gauss")
        err.append(max(abs(float(rot[k, k]) - exact[k]) / exact[k] for k in range(3)))
    assert err[1] < err[0]
    assert err[1] < 1.5e-1
