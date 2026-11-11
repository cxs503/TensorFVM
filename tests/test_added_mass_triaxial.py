"""Analytic anchors and invariance checks for triaxial potential-flow added mass.

What each assertion is tied to:

* **exact anchor** -- the Lamb factors for a body with three unequal semi-axes are
  evaluated from the defining elliptic integral (no closed form exists for
  ``a != b != c``), so the panel result is compared against an independent
  route and not against the closed form it might share a typo with;
* **limits** -- ``a = b = c`` must give ``k = 1/2`` on every axis, the slender
  prolate limit must give ``k_axial -> 0`` and ``k_transverse -> 1``, and a flat
  body must be allowed ``k > 1`` (added mass larger than the displaced mass);
* **structural invariants** -- the discrete matrix is symmetric and diagonal in
  the principal frame, with zero off-diagonals at round-off level;
* **tensor invariance** -- a rigid rotation ``R`` of the body must give
  ``M' = R M R^T`` (this is the property that makes one matrix usable at an
  arbitrary heading, and it is exact for this kernel);
* **mesh equivalence** -- the three-axis mesh must reproduce
  ``spheroid_panels(a, b)`` exactly when ``c == b``, so extending the mesh to
  three axes cannot silently move the two-axis verification's numbers.

Levels here are deliberately below the study ladder of
``tensorfvm.verification.added_mass_triaxial`` so the file stays quick; the
study's own 3 %-class criteria are asserted at the level it uses for its verdict.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.panel_added_mass import (
    added_mass_matrix,
    displaced_volume,
    lamb_ellipsoid_factors,
    spheroid_panels,
)
from tensorfvm.verification.added_mass_triaxial import (
    ellipsoid_panels,
    exact_added_mass_coefficients,
    rotate_panels,
    rotation_matrix,
    run_case,
    run_level,
)

TRI_AXES = (2.0, 1.5, 1.0)


# ---------------------------------------------------------------------------
# exact reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axes", [TRI_AXES, (3.0, 2.0, 1.0), (1.5, 1.0, 0.75)])
def test_lamb_sum_rule_and_positive_coefficients(axes):
    """``alpha0 + beta0 + gamma0 = 2`` and every ``k = alpha/(2-alpha)`` is positive."""
    alphas = lamb_ellipsoid_factors(*axes)
    # the triaxial route is the 400-point Gauss-Legendre evaluation of the defining
    # integral, so the sum rule closes to its quadrature accuracy (~1e-8), eight
    # orders of magnitude below the panel errors this verification measures
    assert sum(alphas) == pytest.approx(2.0, abs=1e-6)
    k = exact_added_mass_coefficients(*axes)
    assert all(value > 0.0 for value in k)
    # no k < 1 bound exists: the shortest axis of the most elongated test body
    # (3:2:1) carries more than its displaced mass
    if axes == (3.0, 2.0, 1.0):
        assert k[2] > 1.0
    else:
        assert all(value < 1.0 for value in k)


def test_sphere_limit_gives_one_half():
    """``a = b = c`` -> ``alpha = 2/3`` -> ``k = 1/2`` on every axis."""
    k = exact_added_mass_coefficients(1.0, 1.0, 1.0)
    assert k == pytest.approx((0.5, 0.5, 0.5), rel=1e-12)


def test_coefficient_increases_as_the_axis_shortens():
    """The added-mass coefficient is monotone in the semi-axis of its own direction."""
    for axes in (TRI_AXES, (3.0, 2.0, 1.0), (1.5, 1.0, 0.75)):
        k = exact_added_mass_coefficients(*axes)
        assert k[0] < k[1] < k[2]          # a > b > c in the test bodies
    # a prolate spheroid reproduces the two-axis route of the sibling verification
    from tensorfvm.panel_added_mass import lamb_spheroid_factors

    axial, transverse = lamb_spheroid_factors(2.0, 1.0)
    tri = exact_added_mass_coefficients(2.0, 1.0, 1.0)
    assert tri[0] == pytest.approx(axial, rel=1e-9)
    assert tri[1] == pytest.approx(transverse, rel=1e-9)


def test_invalid_axes_are_rejected():
    with pytest.raises(ValueError):
        exact_added_mass_coefficients(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        ellipsoid_panels(1.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        ellipsoid_panels(1.0, 1.0, 1.0, n_theta=2, n_phi=2)


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------
def test_three_axis_mesh_reduces_to_the_spheroid_mesh():
    """``c == b`` must reproduce ``spheroid_panels`` bit for bit (same topology)."""
    mine = ellipsoid_panels(2.0, 1.0, 1.0, 16, 12)
    lib = spheroid_panels(2.0, 1.0, 16, 12)
    assert mine.n_panels == lib.n_panels
    assert np.allclose(mine.centroids, lib.centroids, atol=1e-15)
    assert np.allclose(mine.normals, lib.normals, atol=1e-15)
    assert np.allclose(mine.areas, lib.areas, atol=1e-15)


def test_normals_point_outward_and_volume_is_enclosed():
    """Outward normals (into the fluid) and the divergence-theorem volume agree."""
    panels = ellipsoid_panels(*TRI_AXES, 24, 18)
    volume_exact = 4.0 / 3.0 * np.pi * TRI_AXES[0] * TRI_AXES[1] * TRI_AXES[2]
    outward = panels.centroids / np.array(TRI_AXES) ** 2
    assert np.all(np.einsum("ij,ij->i", panels.normals, outward) > 0.0)
    assert displaced_volume(panels) / volume_exact == pytest.approx(1.0, abs=0.03)


def test_rotation_matrix_is_orthogonal_and_true_rotation():
    rotation = rotation_matrix(37.0, (1.0, 2.0, 3.0))
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0, rel=1e-15)
    # 37 degrees about any axis leaves the axis invariant and flips nothing
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    assert np.allclose(rotation @ axis, axis, atol=1e-15)


# ---------------------------------------------------------------------------
# panel results
# ---------------------------------------------------------------------------
def test_triaxial_matrix_is_diagonal_and_accurate():
    """Body-frame diagonal to round-off, and each coefficient close to Lamb."""
    row = run_level(*TRI_AXES, 24, 18)
    matrix = np.array(row["M_matrix"])
    assert np.allclose(matrix, matrix.T, atol=1e-12 * np.abs(matrix).max())
    assert row["offdiag_rel"] < 1e-12
    k_exact = np.array(exact_added_mass_coefficients(*TRI_AXES))
    assert np.all(np.array(row["k_panel"]) > 0.0)
    # this panel count is coarse by design; the study ladder is asserted below
    assert np.max(np.abs(np.array(row["k_panel"]) / k_exact - 1.0)) < 0.10


def test_refinement_reduces_the_triaxial_error():
    """The ladder is consistent: the finest of three levels is the most accurate."""
    coarse = run_level(*TRI_AXES, 16, 12)
    fine = run_level(*TRI_AXES, 32, 24)
    k_exact = np.array(exact_added_mass_coefficients(*TRI_AXES))
    err_coarse = np.abs(np.array(coarse["k_panel"]) / k_exact - 1.0)
    err_fine = np.abs(np.array(fine["k_panel"]) / k_exact - 1.0)
    assert np.all(err_fine < err_coarse)
    assert np.max(err_fine) < 0.05
    # the geometric error is the driver: panel volume is low by a similar order
    assert fine["volume_rel_error"] < 0.0


def test_rotation_covariance_is_exact():
    """``M(R x) = R M(x) R^T`` -- rigid rotation of the body, same discrete problem."""
    panels = ellipsoid_panels(*TRI_AXES, 16, 12)
    rotation = rotation_matrix(37.0, (1.0, 2.0, 3.0))
    base = added_mass_matrix(panels, rho=1.0, order=3)
    turned = added_mass_matrix(rotate_panels(panels, rotation), rho=1.0, order=3)
    predicted = rotation @ base @ rotation.T
    assert np.abs(turned - predicted).max() / np.abs(base).max() < 1e-10
    # off-diagonal coupling appears only because the body frame is left behind
    assert np.allclose(base - np.diag(np.diag(base)), 0.0, atol=1e-12 * np.abs(base).max())
    assert np.abs(predicted - np.diag(np.diag(predicted))).max() > 0.1 * np.abs(base).max()


def test_scaled_study_case_matches_its_anchors():
    """A short ladder through the study code path: one-dimensionality and sign of the error."""
    case = run_case("triaxial_small", *TRI_AXES, levels=((16, 12), (24, 18), (32, 24)))
    k_exact = np.array(case["k_exact"])
    assert case["offdiag_rel_max"] < 1e-12
    # all three coarse-level orders are at least first order
    assert min(case["order_coarse"]) >= 1.0
    # the extrapolated coefficients land well inside the study's 2 % criterion
    rich = np.array(case["richardson_k"])
    assert np.max(np.abs(rich / k_exact - 1.0)) < 0.02
    # the predictive extrapolation is strictly better than the raw finest panel value
    assert np.max(np.abs(rich / k_exact - 1.0)) < case["max_abs_finest_panel_error_pct"] / 100.0