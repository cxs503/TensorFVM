"""Tests for the half-submerged sphere added mass (free surface, Fn -> 0).

The reference side is the odd-harmonic series (validated against the closed-form
heave value), the panel side is the antisymmetric (odd) extension solved on a
closed sphere.  Both are cheap enough for the default test run: the heaviest case
here is 288 panels.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.halfsub_added_mass import (
    antisymmetric_added_mass,
    antisymmetric_normal_velocity,
    half_over_full_ratio,
    sphere_waterplane_panels,
)
from tensorfvm.halfsub_sphere_reference import (
    deep_submergence_added_mass,
    displaced_mass,
    heave_added_mass_exact,
    sway_added_mass_series,
    translation_added_mass,
)
from tensorfvm.panel_added_mass import source_potential
from tensorfvm.panel_potential import IMAGE_SPECS, solve_sources

RADIUS = 1.0
RHO = 1.0
SWAY_REFERENCE = 0.572280  # odd-harmonic series, 256 terms (see verification driver)


# ---------------------------------------------------------------------------
# reference side
# ---------------------------------------------------------------------------
def test_heave_exact_is_half_the_displaced_mass():
    assert heave_added_mass_exact(RADIUS, RHO) == pytest.approx(0.5 * displaced_mass(RADIUS, RHO))
    assert heave_added_mass_exact(RADIUS, RHO) == pytest.approx(np.pi / 3.0)


def test_wall_image_value_is_the_full_sphere_double():
    """The double-body (wall) answer is twice the free-surface heave added mass."""
    assert deep_submergence_added_mass(RADIUS, RHO) == pytest.approx(
        2.0 * heave_added_mass_exact(RADIUS, RHO), rel=1e-12)


@pytest.mark.parametrize("n_terms", [8, 24])
def test_spectral_machinery_reproduces_the_closed_form_heave(n_terms):
    """The series machinery is validated on the mode that has a closed form."""
    assert translation_added_mass(2, RADIUS, RHO, n_terms=n_terms, n_nodes=400) == pytest.approx(
        np.pi / 3.0, rel=1e-11)


def test_sway_series_converges():
    values = [sway_added_mass_series(RADIUS, RHO, n_terms=n, n_nodes=600)
              for n in (32, 64, 128, 256)]
    assert all(np.diff(values) < 0.0), "truncated odd-harmonic series decreases toward the limit"
    drift = abs(values[-2] - values[-1]) / values[-1]
    assert drift < 1e-3
    assert values[-1] == pytest.approx(SWAY_REFERENCE, abs=2e-4)


def test_sway_series_requires_valid_axis():
    with pytest.raises(ValueError):
        translation_added_mass(3, RADIUS, RHO)


# ---------------------------------------------------------------------------
# odd extension: data set
# ---------------------------------------------------------------------------
def test_heave_extension_is_rigid_heave():
    panels = sphere_waterplane_panels(RADIUS, 12, 6)
    psi = antisymmetric_normal_velocity(panels, 2)
    assert np.allclose(psi, panels.normals[:, 2], atol=1e-15)


def test_horizontal_extension_flips_sign_on_the_mirror_half():
    panels = sphere_waterplane_panels(RADIUS, 12, 6)
    lower = panels.centroids[:, 2] < 0.0
    psi = antisymmetric_normal_velocity(panels, 0)
    assert np.allclose(psi[lower], panels.normals[lower, 0], atol=1e-15)
    assert np.allclose(psi[~lower], -panels.normals[~lower, 0], atol=1e-15)


def test_odd_extension_potential_vanishes_on_the_waterplane():
    """``Phi(z=0) = 0`` is what makes the extension the free-surface solution."""
    panels = sphere_waterplane_panels(RADIUS, 12, 6)
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    pts = np.array([[r * np.cos(t), r * np.sin(t), 0.0]
                    for r in np.linspace(1.5, 3.0, 6) for t in theta])
    for axis in (0, 2):
        psi = antisymmetric_normal_velocity(panels, axis)
        sigma = solve_sources(panels, np.zeros(3), order=3, images=IMAGE_SPECS["body"],
                              normal_velocity=psi)
        phi = source_potential(pts, sigma, panels, order=3, images=IMAGE_SPECS["body"])
        assert np.abs(phi).max() < 1e-10


def test_sphere_waterplane_panels_needs_an_equator_ring():
    with pytest.raises(ValueError):
        sphere_waterplane_panels(RADIUS, 12, 7)


# ---------------------------------------------------------------------------
# odd extension: added mass
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coarse_matrix():
    panels = sphere_waterplane_panels(RADIUS, 24, 12)
    return antisymmetric_added_mass(panels, rho=RHO, order=3)


def test_half_body_heave_matches_the_closed_form(coarse_matrix):
    assert coarse_matrix[2, 2] == pytest.approx(np.pi / 3.0, rel=0.05)


def test_half_body_sway_matches_the_spectral_reference(coarse_matrix):
    assert coarse_matrix[0, 0] == pytest.approx(SWAY_REFERENCE, rel=0.12)


def test_translation_modes_decouple(coarse_matrix):
    off = coarse_matrix - np.diag(np.diag(coarse_matrix))
    assert np.abs(off).max() < 1e-12
    assert coarse_matrix[0, 0] == pytest.approx(coarse_matrix[1, 1], rel=1e-10)


def test_odd_extension_of_heave_is_exactly_half_the_full_sphere():
    panels = sphere_waterplane_panels(RADIUS, 24, 12)
    assert half_over_full_ratio(panels, rho=RHO, order=3) == pytest.approx(0.5, rel=1e-12)


def test_sway_needs_the_free_surface_not_the_full_sphere():
    """The free-surface sway value differs materially from the double-body one."""
    panels = sphere_waterplane_panels(RADIUS, 24, 12)
    M = antisymmetric_added_mass(panels, rho=RHO, order=3, axes=(0,))
    assert M[0, 0] < 0.75 * deep_submergence_added_mass(RADIUS, RHO)


def test_invalid_arguments():
    panels = sphere_waterplane_panels(RADIUS, 8, 4)
    with pytest.raises(ValueError):
        antisymmetric_added_mass(panels, rho=-1.0)
    with pytest.raises(ValueError):
        antisymmetric_normal_velocity(panels, 5)
    with pytest.raises(ValueError):
        heave_added_mass_exact(-1.0)