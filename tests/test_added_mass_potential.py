"""Analytic anchors and convergence checks for potential-flow added mass.

Every assertion is tied to a closed-form value:

* Lamb factors: closed form (prolate spheroid) vs the independently evaluated
  defining integral, plus the physical limits (sphere ``k = 1/2``, slender
  ``k_axial -> 0``, ``k_transverse -> 1``);
* panelised spheroid added mass vs ``M = k rho V`` with the Lamb factors;
* discrete invariants: symmetric matrix, zero off-diagonals and equal transverse
  pair for an axisymmetric body, small Neumann residual.

Levels in the tests are smaller than the study ladder in
``tensorfvm.verification.added_mass_potential`` so the file stays quick; the
3 % accuracy criterion is asserted at the level the study uses for its verdict.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.panel_added_mass import (
    added_mass_matrix,
    displaced_volume,
    lamb_ellipsoid_factors,
    lamb_integral_factors,
    lamb_spheroid_factors,
    source_potential,
    sphere_added_mass,
    spheroid_panels,
)
from tensorfvm.panel_potential import normal_velocity_residual, solve_sources


@pytest.mark.parametrize("a,b", [(1.5, 1.0), (2.0, 1.0), (4.0, 1.0)])
def test_lamb_closed_form_matches_defining_integral(a, b):
    """The closed form and the Lamb integral (independent routes) must agree."""
    closed = lamb_ellipsoid_factors(a, b, b)
    integral = lamb_integral_factors(a, b, b)
    assert closed == pytest.approx(integral, rel=1e-7, abs=1e-12)


def test_lamb_triaxial_matches_integral():
    closed = lamb_ellipsoid_factors(2.0, 1.5, 1.0)
    integral = lamb_integral_factors(2.0, 1.5, 1.0)
    assert closed == pytest.approx(integral, rel=1e-7)
    assert sum(closed) == pytest.approx(2.0, rel=1e-7)  # alpha0+beta0+gamma0 = 2


def test_lamb_limits():
    # sphere limit: every factor 2/3, every k = 1/2
    k_axial, k_transverse = lamb_spheroid_factors(1.0, 1.0)
    assert k_axial == pytest.approx(0.5, rel=1e-12)
    assert k_transverse == pytest.approx(0.5, rel=1e-12)
    # slender limit: axial -> 0, transverse -> 1, and monotone in aspect ratio
    ratios = (1.5, 2.0, 4.0, 8.0, 16.0)
    axial = [lamb_spheroid_factors(r, 1.0)[0] for r in ratios]
    transverse = [lamb_spheroid_factors(r, 1.0)[1] for r in ratios]
    assert all(x > y for x, y in zip(axial, axial[1:]))
    assert all(x < y for x, y in zip(transverse, transverse[1:]))
    assert axial[-1] < 0.02
    assert 0.9 < transverse[-1] < 1.0


def test_sphere_added_mass_value():
    assert sphere_added_mass(1.0, 1.0) == pytest.approx(0.5 * 4.0 / 3.0 * np.pi, rel=1e-15)


def test_panelled_volume_converges_first_order():
    """Inscribed polyhedron: volume error must fall monotonically towards zero."""
    a, b = 2.0, 1.0
    exact = 4.0 / 3.0 * np.pi * a * b * b
    errors = []
    for n_theta, n_phi in ((16, 12), (24, 18), (32, 24)):
        panels = spheroid_panels(a, b, n_theta, n_phi)
        assert panels.normals.shape == panels.centroids.shape
        errors.append(abs(displaced_volume(panels) / exact - 1.0))
    assert all(x > y for x, y in zip(errors, errors[1:])), errors
    assert errors[-1] < 1.5e-2


def test_source_potential_far_field_decays():
    """A closed body has zero net source strength: the far field is a dipole (1/r^2)."""
    panels = spheroid_panels(1.0, 1.0, 16, 12)
    sigma = solve_sources(panels, np.array([1.0, 0.0, 0.0]), order=3)
    r1 = np.array([[10.0, 0.0, 0.0]])
    r2 = np.array([[20.0, 0.0, 0.0]])
    phi1 = source_potential(r1, sigma, panels, order=3)[0]
    phi2 = source_potential(r2, sigma, panels, order=3)[0]
    assert phi1 == pytest.approx(4.0 * phi2, rel=5e-2)
    assert phi1 < 0.0 and phi2 < 0.0


def test_added_mass_matrix_symmetry_and_decoupling():
    panels = spheroid_panels(2.0, 1.0, 24, 18)
    M = added_mass_matrix(panels, rho=1.0, order=3)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T, atol=0.0)  # exactly symmetrised
    diag = np.diag(M)
    assert np.all(diag > 0.0)
    off = np.abs(M - np.diag(diag)).max()
    assert off < 1e-10 * diag.max()
    assert abs(M[1, 1] - M[2, 2]) < 1e-10 * M[1, 1]
    # axial added mass of a prolate spheroid is far below the transverse one
    assert diag[0] < 0.5 * diag[1]


def test_sphere_added_mass_within_three_percent():
    radius = 1.0
    exact = sphere_added_mass(radius, 1.0)
    errors = []
    for n_theta, n_phi in ((16, 12), (24, 18), (32, 24)):
        panels = spheroid_panels(radius, radius, n_theta, n_phi)
        M = added_mass_matrix(panels, order=3)
        errors.append(M[0, 0] / exact - 1.0)
    assert all(abs(b) < abs(a) for a, b in zip(errors, errors[1:])), errors
    assert abs(errors[-1]) < 0.03, errors


def test_prolate_spheroid_added_mass_within_three_percent():
    a, b = 2.0, 1.0
    volume = 4.0 / 3.0 * np.pi * a * b * b
    k_axial, k_transverse = lamb_spheroid_factors(a, b)
    panels = spheroid_panels(a, b, 40, 30)
    M = added_mass_matrix(panels, order=3)
    err_axial = M[0, 0] / (k_axial * volume) - 1.0
    err_transverse = 0.5 * (M[1, 1] + M[2, 2]) / (k_transverse * volume) - 1.0
    assert abs(err_axial) < 0.03, err_axial
    assert abs(err_transverse) < 0.03, err_transverse


def test_neumann_residual_is_machine_precision():
    panels = spheroid_panels(2.0, 1.0, 24, 18)
    U = np.array([1.0, 0.0, 0.0])
    sigma = solve_sources(panels, U, order=3)
    assert normal_velocity_residual(panels, sigma, U, order=3) < 1e-10


def test_slender_body_bound_is_an_upper_bound():
    """The empirical slender-body transverse added mass over-predicts, as documented."""
    a, b = 2.0, 1.0
    volume = 4.0 / 3.0 * np.pi * a * b * b
    _, k_transverse = lamb_spheroid_factors(a, b)
    k_slender = np.pi * b * b * 2.0 * a / volume
    assert k_slender > k_transverse
    assert k_slender / k_transverse == pytest.approx(2.130, abs=0.01)