"""Regression guards for the reference-point (parallel-axis) algebra of the 6x6 added mass.

These tests pin the *structure* that debt-ledger item D8 questioned: the
generator identity ``M(r0) = S(r0) M(0) S(r0)^T`` (see
:mod:`tensorfvm.verification.added_mass_parallel_axis`), the exact zero of a
sphere's yaw inertia when the reference point slides along the rotation axis, and
the exactness of :func:`tensorfvm.panel_added_mass.added_mass_matrix_about_point`.

They are deliberately small-mesh (the identity is a translation invariance of the
panel solve, not a convergence claim), so the whole file runs in seconds;
the quantitative error ladders live in the verification driver.
"""

import math

import numpy as np
import pytest

from tensorfvm.panel_added_mass import (
    added_mass_matrix_6dof,
    added_mass_matrix_about_point,
    displaced_volume,
    lamb_spheroid_factors,
    rigid_shift_matrix,
    sphere_added_mass,
    sphere_panels_offset,
    spheroid_panels,
)
from tensorfvm.ellipsoid_added_inertia import rotational_added_inertia as rotational_closed_form
from tensorfvm.twin_hull_geometry import PanelSet

RHO = 1.0
COARSE = (20, 12)


def _shift(panels: PanelSet, offset) -> PanelSet:
    off = np.asarray(offset, dtype=float)
    corners = None if panels.corners is None else panels.corners + off
    return PanelSet(panels.centroids + off, panels.normals.copy(),
                    panels.areas.copy(), corners, panels.label)


# ---------------------------------------------------------------------------
# rigid_shift_matrix
# ---------------------------------------------------------------------------
def test_rigid_shift_matrix_structure():
    off = np.array([0.3, -0.7, 1.1])
    S = rigid_shift_matrix(off)
    assert np.allclose(S[0:3, 0:3], np.eye(3))
    assert np.allclose(S[3:6, 3:6], np.eye(3))
    assert np.allclose(S[0:3, 3:6], 0.0)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        assert np.allclose(S[3 + i, 0:3], np.cross(e, off))


def test_rigid_shift_matrix_inverse_is_opposite_offset():
    off = np.array([1.0, 0.5, -2.0])
    assert np.allclose(rigid_shift_matrix(off) @ rigid_shift_matrix(-off), np.eye(6))


def test_rigid_shift_matrix_rejects_non_vector():
    with pytest.raises(ValueError):
        rigid_shift_matrix([1.0, 2.0])


# ---------------------------------------------------------------------------
# the identity itself (the D8 question)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("offset", [(0.0, 0.0, 1.5), (1.5, 0.0, 0.0), (0.6, -0.9, 0.4)])
def test_identity_sphere(offset):
    base = sphere_panels_offset(1.0, (0.0, 0.0, 0.0), *COARSE)
    M0 = added_mass_matrix_6dof(base, rho=RHO)
    Mo = added_mass_matrix_6dof(_shift(base, offset), rho=RHO)
    S = rigid_shift_matrix(np.array(offset))
    pred = S @ M0 @ S.T
    assert np.abs(Mo - pred).max() / np.abs(Mo).max() < 1e-12


@pytest.mark.parametrize("ratio", [2.0, 3.0])
def test_identity_spheroid(ratio):
    base = spheroid_panels(ratio, 1.0, *COARSE)
    off = np.array([0.5, -1.2, 0.3])
    M0 = added_mass_matrix_6dof(base, rho=RHO)
    Mo = added_mass_matrix_6dof(_shift(base, off), rho=RHO)
    S = rigid_shift_matrix(off)
    assert np.abs(Mo - S @ M0 @ S.T).max() / np.abs(Mo).max() < 1e-12


# ---------------------------------------------------------------------------
# added_mass_matrix_about_point
# ---------------------------------------------------------------------------
def test_about_point_equals_translated_body():
    base = sphere_panels_offset(1.0, (0.0, 0.0, 0.0), *COARSE)
    M0 = added_mass_matrix_6dof(base, rho=RHO)
    off = np.array([1.2, 0.0, -0.4])
    Mo = added_mass_matrix_6dof(_shift(base, off), rho=RHO)
    # matrix about a point displaced by -off  ==  matrix of the body displaced by +off
    assert np.abs(added_mass_matrix_about_point(M0, -off) - Mo).max() < 1e-12 * np.abs(Mo).max()


def test_about_point_roundtrip():
    M0 = added_mass_matrix_6dof(spheroid_panels(2.0, 1.0, *COARSE), rho=RHO)
    off = np.array([0.3, 1.4, -0.2])
    back = added_mass_matrix_about_point(added_mass_matrix_about_point(M0, off), -off)
    assert np.abs(back - M0).max() < 1e-12 * np.abs(M0).max()


def test_about_point_keeps_translational_block():
    M0 = added_mass_matrix_6dof(spheroid_panels(2.0, 1.0, *COARSE), rho=RHO)
    Mp = added_mass_matrix_about_point(M0, np.array([0.8, -0.5, 1.3]))
    assert np.allclose(Mp[0:3, 0:3], M0[0:3, 0:3], rtol=0.0, atol=1e-12)
    assert np.allclose(Mp, Mp.T)


def test_about_point_rejects_bad_shape():
    with pytest.raises(ValueError):
        added_mass_matrix_about_point(np.zeros((3, 3)), np.zeros(3))


# ---------------------------------------------------------------------------
# exact anchors
# ---------------------------------------------------------------------------
def test_sphere_offset_along_axis_has_zero_yaw_inertia():
    d = 1.5
    M = added_mass_matrix_6dof(sphere_panels_offset(1.0, (0.0, 0.0, d), *COARSE), rho=RHO)
    scale = sphere_added_mass(1.0, RHO)
    assert abs(M[5, 5]) / scale < 1e-12
    assert abs(M[3, 3] - M[4, 4]) / abs(M[3, 3]) < 1e-10


def test_sphere_parallel_axis_increment_is_exactly_d_squared_times_Myy():
    d = 1.5
    M0 = added_mass_matrix_6dof(sphere_panels_offset(1.0, (0.0, 0.0, 0.0), *COARSE), rho=RHO)
    Mo = added_mass_matrix_6dof(sphere_panels_offset(1.0, (0.0, 0.0, d), *COARSE), rho=RHO)
    # S[3, 0:3] = e_x x (0,0,d) = -d e_y  =>  I_xx(d) - I_xx(0) = d^2 M_yy(0).
    # The increment is the parallel-axis statement; I_xx(0) itself is the faceted
    # sphere's spurious rotational inertia (exact value 0, ~1e-5 here).
    assert abs((Mo[3, 3] - M0[3, 3]) / (d * d * M0[1, 1]) - 1.0) < 1e-12


def test_sphere_parallel_axis_value_within_3_percent():
    d = 1.5
    panels = sphere_panels_offset(1.0, (0.0, 0.0, d), 32, 18)
    M = added_mass_matrix_6dof(panels, rho=RHO)
    exact = sphere_added_mass(1.0, RHO) * d * d
    assert abs(M[3, 3] / exact - 1.0) < 0.03


def test_spheroid_offset_anchor_closed_form():
    """I_zz(d e_y) = I_zz(0)_closed + d^2 k_axial rho V on a prolate spheroid."""
    ratio, b, d = 2.0, 1.0, 1.5
    a = ratio * b
    vol = 4.0 / 3.0 * math.pi * a * b * b
    k_axial, _ = lamb_spheroid_factors(a, b)
    exact = rotational_closed_form(a, b, b, axis=2, rho=RHO) + d * d * k_axial * RHO * vol
    panels = spheroid_panels(a, b, 32, 18)
    M0 = added_mass_matrix_6dof(panels, rho=RHO)
    Mo = added_mass_matrix_6dof(_shift(panels, (0.0, d, 0.0)), rho=RHO)
    # the axial-translation / yaw coupling that the closed form drops is zero by
    # fore-aft symmetry -- asserted, not assumed
    assert abs(M0[0, 5]) < 1e-12
    assert displaced_volume(panels) > 0.0
    assert abs(Mo[5, 5] / exact - 1.0) < 0.03


# ---------------------------------------------------------------------------
# power of the identity: the classic mistakes must be detected
# ---------------------------------------------------------------------------
def test_mutation_dropping_parallel_axis_is_detected():
    base = sphere_panels_offset(1.0, (0.0, 0.0, 0.0), *COARSE)
    M0 = added_mass_matrix_6dof(base, rho=RHO)
    Mo = added_mass_matrix_6dof(_shift(base, (1.5, 0.0, 0.0)), rho=RHO)
    assert np.abs(M0 - Mo).max() / np.abs(Mo).max() > 0.5


def test_mutation_flipping_moment_arm_is_detected():
    base = spheroid_panels(3.0, 1.0, *COARSE)
    off = np.array([0.7, -1.1, 0.4])
    M0 = added_mass_matrix_6dof(base, rho=RHO)
    Mo = added_mass_matrix_6dof(_shift(base, off), rho=RHO)
    S = rigid_shift_matrix(off)
    flip = np.eye(6)
    flip[3:, 0:3] = -S[3:, 0:3]
    assert np.abs(flip @ M0 @ flip.T - Mo).max() / np.abs(Mo).max() > 1e-2