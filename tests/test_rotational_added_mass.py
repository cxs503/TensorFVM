"""Tests for the rotational (M_44..M_66) added mass and its exact anchors.

The heavy convergence study lives in
:mod:`tensorfvm.verification.rotational_added_mass`; these tests pin the
structure on coarse meshes so they stay fast:

* sphere about a diameter -> exactly zero (``r x n = 0``);
* body of revolution roll -> exactly zero to machine precision;
* parallel-axis anchor ``I_oo = m_a d^2`` for a sphere off the reference point;
* the ``6 x 6`` translational block reproduces the existing translational API;
* symmetry of the assembled matrix and the per-panel normal-velocity entry point.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.panel_added_mass import (
    added_mass_matrix,
    added_mass_matrix_6dof,
    displaced_volume,
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