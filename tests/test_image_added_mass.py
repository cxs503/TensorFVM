"""Tests for the image (free-surface / double-body) added mass verification (debt D2).

The anchors are closed-form and independent of the panel solver:
tangential ``pi rho a^3/3`` (= ``m_full/2`` of the sphere), normal
``m_stretch/2`` with ``m_stretch = 4 pi rho a^3 sum c_n^2/((2n+1)(n+1))``,
quarter body ``k_axial rho V/4``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.panel_added_mass import lamb_spheroid_factors
from tensorfvm.panel_potential import IMAGE_SPECS, influence_matrix, panel_quadrature
from tensorfvm.verification.image_added_mass import (
    _legendre_coefficients,
    full_body_panels,
    half_body_anchors,
    half_body_panels,
    quarter_body_anchor,
    richardson,
    solve_added_mass,
    sphere_rigid_added_mass,
    sphere_stretch_added_mass,
    wall_probe,
)


# ---------------------------------------------------------------------------
# closed-form anchors
# ---------------------------------------------------------------------------
def test_legendre_coefficients_of_abs_x():
    """``|x|`` is even: odd coefficients vanish; the first even ones are exact."""
    c = _legendre_coefficients(8)
    assert c[0] == pytest.approx(0.5, abs=1e-12)
    assert c[2] == pytest.approx(5.0 / 8.0, abs=1e-12)
    assert c[4] == pytest.approx(-3.0 / 16.0, abs=1e-12)
    assert np.all(c[1::2] == 0.0)


def test_stretch_anchor_reduces_to_rigid_translation():
    """The generator ``cos theta`` (``c_1 = 1``) must return ``rho V / 2``.

    Independent check of the ``1/((2n+1)(n+1))`` weights: feeding the translation
    Legendre coefficients instead of ``|cos theta|`` must reproduce the textbook
    sphere added mass, otherwise the weights (not the solver) would be wrong.
    """
    c = np.zeros(8)
    c[1] = 1.0
    n = np.arange(8)
    m = 4.0 * np.pi * np.sum(c**2 / ((2 * n + 1) * (n + 1)))
    assert m == pytest.approx(sphere_rigid_added_mass(1.0, 1.0), rel=1e-14)


def test_stretch_anchor_value_and_scale():
    m = sphere_stretch_added_mass(1.0, 1.0)
    assert m == pytest.approx(3.480680593347541, rel=1e-12)
    # scaling laws
    assert sphere_stretch_added_mass(2.0, 1.0) == pytest.approx(8.0 * m, rel=1e-12)
    assert sphere_stretch_added_mass(1.0, 3.0) == pytest.approx(3.0 * m, rel=1e-12)
    # the normal direction of a half body is NOT half of the rigid added mass
    assert m / 2.0 > 1.5 * (np.pi / 3.0)


def test_half_body_anchors_sphere_and_spheroid():
    a = half_body_anchors(1.0, 1.0)
    assert a["M_xx"] == pytest.approx(np.pi / 3.0, rel=1e-14)
    assert a["M_yy"] == pytest.approx(np.pi / 3.0, rel=1e-14)
    assert a["M_zz"] == pytest.approx(0.5 * sphere_stretch_added_mass(1.0, 1.0), rel=1e-14)

    a2 = half_body_anchors(2.0, 1.0)
    k_axial, k_transv = lamb_spheroid_factors(2.0, 1.0)
    volume = 4.0 / 3.0 * np.pi * 2.0
    assert a2["M_xx"] == pytest.approx(0.5 * k_axial * volume, rel=1e-14)
    assert a2["M_yy"] == pytest.approx(0.5 * k_transv * volume, rel=1e-14)
    assert a2["M_zz"] is None  # no closed form for the normal direction of a spheroid
    assert quarter_body_anchor(2.0, 1.0) == pytest.approx(0.25 * k_axial * volume, rel=1e-14)


def test_richardson_helper():
    r = richardson([1.0, 0.5, 0.25])
    assert r["observed_order"] == pytest.approx(1.0, rel=1e-12)
    assert r["richardson"] == pytest.approx(0.0, abs=1e-12)
    r2 = richardson([0.0, 1.0, 1.25])  # quadratic: d1 = 1, d2 = .25 -> p = 2, est 1.3333
    assert r2["observed_order"] == pytest.approx(2.0, rel=1e-12)
    assert r2["richardson"] == pytest.approx(1.25 + 0.25 / 3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_half_and_quarter_geometry():
    half = half_body_panels(1.0, 1.0, 24, 18)
    quarter = half_body_panels(1.0, 1.0, 24, 18, quarter=True)
    assert half.n_panels == 24 * 18
    assert half.areas.sum() == pytest.approx(2.0 * np.pi, rel=0.02)     # half sphere area
    assert quarter.areas.sum() == pytest.approx(np.pi, rel=0.02)        # quarter area
    assert half.centroids[:, 2].min() >= -1e-12
    assert quarter.centroids[:, 1].min() >= -1e-12
    assert quarter.centroids[:, 2].min() >= -1e-12
    # outward normals point away from the body centre
    radial = np.einsum("ij,ij->i", half.normals, half.centroids)
    assert radial.min() > 0.0


def test_full_body_area_and_volume_scaling():
    full = full_body_panels(1.0, 1.0, 32, 12)
    assert full.areas.sum() == pytest.approx(4.0 * np.pi, rel=0.03)
    assert np.einsum("ij,ij->i", full.normals, full.centroids).min() > 0.0


# ---------------------------------------------------------------------------
# the defect fixed on the way: influence_matrix must not drop panels for
# field points that are not the panel centroids
# ---------------------------------------------------------------------------
def test_influence_matrix_keeps_every_panel_for_foreign_field_points():
    panels = half_body_panels(1.0, 1.0, 4, 3)
    points = np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, -2.0]])
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    images = IMAGE_SPECS["single_hull"]
    A = influence_matrix(points, normals, panels, order=2, images=images)

    # brute force with the same bilinear Gauss rule
    q, w = panel_quadrature(panels, order=2)
    ref = np.zeros_like(A)
    for spec in images:
        qq = q.copy()
        qq[..., 1] *= spec[0]
        qq[..., 2] *= spec[1]
        for i, (pt, nrm) in enumerate(zip(points, normals)):
            d = pt[None, None, :] - qq                      # (nq, npan, 3)
            r = np.linalg.norm(d, axis=-1)
            k = 1.0 / (4.0 * np.pi * r**3)
            ref[i] += np.sum(w * k * (d @ nrm), axis=0)          # n . d summed over all panels
    assert np.allclose(A, ref, rtol=0.0, atol=1e-12)


def test_influence_matrix_principal_value_kept_for_panel_field_points():
    """The flat-panel self term stays removed when the field points are the panels."""
    panels = half_body_panels(1.0, 1.0, 4, 3)
    A = influence_matrix(panels.centroids, panels.normals, panels, order=2,
                         images=((1.0, 1.0),))
    assert np.all(np.diag(A) == 0.0)


# ---------------------------------------------------------------------------
# panel solves vs the closed-form anchors
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def half_sphere_matrix():
    panels = half_body_panels(1.0, 1.0, 24, 18)
    return solve_added_mass(panels, IMAGE_SPECS["single_hull"])


def test_half_sphere_tangential_added_mass(half_sphere_matrix):
    anchor = np.pi / 3.0
    assert half_sphere_matrix[0, 0] == pytest.approx(anchor, rel=0.03)
    assert half_sphere_matrix[1, 1] == pytest.approx(anchor, rel=0.03)
    off = max(abs(half_sphere_matrix[0, 1]), abs(half_sphere_matrix[0, 2]),
              abs(half_sphere_matrix[1, 2]))
    assert off < 1e-6  # isotropic in the tangential plane


def test_half_sphere_normal_added_mass_is_the_straining_value(half_sphere_matrix):
    anchor = 0.5 * sphere_stretch_added_mass(1.0, 1.0)
    assert half_sphere_matrix[2, 2] == pytest.approx(anchor, rel=0.03)
    # and it is *not* the "half of the rigid added mass" shortcut
    assert half_sphere_matrix[2, 2] > 1.4 * (np.pi / 3.0)


def test_half_spheroid_anisotropy():
    a, b = 2.0, 1.0
    panels = half_body_panels(a, b, 24, 18)
    M = solve_added_mass(panels, IMAGE_SPECS["single_hull"])
    anchors = half_body_anchors(a, b)
    assert M[0, 0] == pytest.approx(anchors["M_xx"], rel=0.03)
    assert M[1, 1] == pytest.approx(anchors["M_yy"], rel=0.03)
    assert M[0, 0] < 0.5 * M[1, 1]  # prolate: axial added mass far below transverse


def test_quarter_body_axial_added_mass():
    panels = half_body_panels(1.0, 1.0, 24, 18, quarter=True)
    M = solve_added_mass(panels, IMAGE_SPECS["twin_hull"])
    assert M[0, 0] == pytest.approx(quarter_body_anchor(1.0, 1.0), rel=0.03)


def test_double_body_identity_is_exact():
    """``m(half + image) = m(full)/2``: continuum theorem, discrete identity."""
    n_theta, n_phi = 16, 12
    half = half_body_panels(1.0, 1.0, n_theta, n_phi)
    full = full_body_panels(1.0, 1.0, 2 * n_theta, n_phi)
    M_half = solve_added_mass(half, IMAGE_SPECS["single_hull"], directions=(0, 1))
    M_full = solve_added_mass(full, IMAGE_SPECS["body"], directions=(0, 1))
    assert M_half[0, 0] == pytest.approx(0.5 * M_full[0, 0], rel=1e-9)
    assert M_half[1, 1] == pytest.approx(0.5 * M_full[1, 1], rel=1e-9)


def test_wall_is_a_rigid_wall_not_a_phi_zero_free_surface():
    panels = half_body_panels(1.0, 1.0, 16, 12)
    probe = wall_probe(panels, IMAGE_SPECS["single_hull"], a=1.0, b=1.0)
    assert probe["max_wall_normal_velocity"] < 1e-8   # dphi/dn = 0 on the plane
    assert probe["max_wall_potential_over_ua"] > 1e-3  # phi != 0 -> not the omega->inf model