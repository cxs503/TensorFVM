"""Rotational added inertia (``M_44..M_66``) against exact and semi-analytic anchors.

Anchor hierarchy (weakest to strongest):

1. exact structural zeros: sphere (any axis) and any body of revolution about its
   symmetry axis must give *exactly* zero added inertia;
2. a semi-analytic spheroidal-harmonic reference whose Neumann residual is
   ~machine precision (uniqueness theorem -> it is the exact solution);
3. the classical Lamb closed form, cross-checked against (2);
4. convergence of the source-panel BEM to (2)/(3) under mesh refinement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tensorfvm.panel_added_inertia import (  # noqa: E402
    lamb_rotational_inertia,
    rigid_rotation_field,
    rotational_added_inertia,
)
from tensorfvm.panel_added_mass import spheroid_panels  # noqa: E402
from verify_rotational_added_inertia_reference import (  # noqa: E402
    spheroid_rotation_reference,
)


def test_rigid_rotation_field_is_omega_cross_r():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 0.5, 0.0]])
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        np.testing.assert_allclose(rigid_rotation_field(pts, axis), np.cross(e, pts))
    # points on the rotation axis do not move
    axis_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5], [0.0, 0.0, -1.0]])
    np.testing.assert_allclose(rigid_rotation_field(axis_pts, 2), 0.0, atol=0.0)


def test_sphere_rotational_added_inertia_is_zero():
    """A rotating sphere perturbs nothing in an ideal fluid: I = 0 exactly."""
    scaled = []
    for nt, nph in ((16, 12), (24, 18), (32, 24)):
        p = spheroid_panels(1.0, 1.0, n_theta=nt, n_phi=nph)
        I = rotational_added_inertia(p, order=3)
        scaled.append(np.abs(I).max())  # rho = a = 1
        assert np.all(np.diag(I) >= 0.0)
    assert scaled[0] < 5.0e-4
    assert scaled[-1] < 1.5e-4
    assert scaled[-1] < scaled[0]  # discretisation error, not a physical value


def test_symmetry_axis_rotational_added_inertia_is_zero():
    """Rotation about the axis of revolution leaves the fluid at rest."""
    p = spheroid_panels(2.0, 1.0, n_theta=40, n_phi=28)
    I = rotational_added_inertia(p, order=3)
    assert abs(I[0, 0]) < 1.0e-12
    off = np.abs(I - np.diag(np.diag(I))).max()
    assert off < 1.0e-12, f"off-diagonal coupling {off:.2e} should vanish by symmetry"
    # the two transverse axes are equivalent for a spheroid
    assert abs(I[1, 1] - I[2, 2]) < 1.0e-10
    assert I[1, 1] > 0.0


@pytest.mark.parametrize("aspect", [1.5, 2.0, 3.0])
def test_semi_analytic_reference_matches_lamb_closed_form(aspect):
    a, b = float(aspect), 1.0
    ref = spheroid_rotation_reference(a, b, n_terms=10, n_mu_fit=48, n_phi_fit=8)
    exact = lamb_rotational_inertia(a, b)[1, 1]
    assert ref["neumann_residual"] < 1.0e-10, "series does not solve the Neumann problem"
    assert exact > 0.0
    assert abs(ref["I_perp"] / exact - 1.0) < 1.0e-5


def test_closed_form_limits():
    # sphere limit: vanishes
    assert abs(lamb_rotational_inertia(1.0, 1.0)[1, 1]) < 1.0e-15
    # symmetry-axis entry vanishes for any spheroid
    for aspect in (1.2, 2.0, 5.0):
        assert abs(lamb_rotational_inertia(aspect, 1.0)[0, 0]) < 1.0e-15
    # transverse value grows with elongation
    vals = [lamb_rotational_inertia(a, 1.0)[1, 1] for a in (1.5, 2.0, 4.0)]
    assert vals[0] < vals[1] < vals[2]


def test_panel_converges_to_exact_transverse_value():
    """3% gate at the finest tested mesh; the error must decrease monotonically."""
    a, b = 2.0, 1.0
    exact = lamb_rotational_inertia(a, b)[1, 1]
    errs = []
    for nt, nph in ((24, 18), (40, 28), (60, 42)):
        p = spheroid_panels(a, b, n_theta=nt, n_phi=nph)
        I = rotational_added_inertia(p, order=3, axes=(1, 2))
        errs.append(abs(I[1, 1] / exact - 1.0))
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:])), errs
    assert errs[-1] < 0.03, f"finest-mesh error {errs[-1]:.3%} exceeds the 3% gate"