"""Tests for the closed-form ellipsoid rotational added inertia (verification debt D6).

The full study (panel convergence at ``a/b = 2, 4, 8``, Richardson/order fits, report and
figure) lives in :mod:`tensorfvm.verification.ellipsoid_rotational_reference` and takes
about a minute; the tests below pin the structure and the cheapest quantitative checks so
that they stay fast:

* exact zeros (sphere about a diameter, spheroid roll about its symmetry axis);
* the quadrature route and the independent prolate closed form of ``FF(0)`` agree;
* the closed-form surface moment ``(2 pi/15) a^3 b^3 c`` matches surface quadrature for a
  *triaxial* ellipsoid (an axisymmetric shape cannot expose a wrong exponent);
* the closed-form potential satisfies the rotation Neumann data (finite differences, so the
  derivation's own algebra is not used to check itself);
* the slender asymptote ``1/5`` and the near-sphere ``(a^2-b^2)^2`` law;
* axis-permutation consistency and input validation;
* one coarse panel comparison (a/b = 2, 24x14 mesh), the smallest executable form of the
  study's main quantity.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.ellipsoid_added_inertia import (
    displaced_volume,
    ellipsoidal_coordinate,
    harmonic_integral,
    harmonic_integral_spheroid,
    neumann_data,
    rotation_potential,
    rotational_added_inertia,
    slender_limit_ratio,
    surface_product_moment,
)
from tensorfvm.panel_added_mass import rotational_added_inertia as panel_rotational_added_inertia
from tensorfvm.panel_added_mass import spheroid_panels
from tensorfvm.verification.ellipsoid_rotational_reference import surface_moment_quadrature


def _surface_sample(a, b, c, n_theta=8, n_phi=6):
    """Exact (unfaceted) ellipsoid surface points and outward unit normals."""
    xg, _ = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.5 * np.pi * (xg + 1.0)
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    T, P = np.meshgrid(theta, phi, indexing="ij")
    pts = np.stack([a * np.sin(T) * np.cos(P), b * np.sin(T) * np.sin(P), c * np.cos(T)],
                   axis=-1).reshape(-1, 3)
    grad = np.stack([2 * pts[:, 0] / a**2, 2 * pts[:, 1] / b**2, 2 * pts[:, 2] / c**2],
                    axis=-1)
    nrm = grad / np.linalg.norm(grad, axis=-1)[:, None]
    return pts, nrm


# ------------------------------------------------------------------ exact zeros
def test_sphere_and_spheroid_roll_are_exactly_zero():
    """A sphere has zero added inertia about any diameter; a spheroid rolling about its
    symmetry axis has none either (no ``(Omega x r) . n`` on that generator)."""
    assert rotational_added_inertia(1.0, 1.0, 1.0, axis=2) == 0.0
    assert rotational_added_inertia(3.0, 3.0, 3.0, axis=0) == 0.0
    assert rotational_added_inertia(2.0, 1.0, 1.0, axis=0) == 0.0  # prolate, roll
    assert rotational_added_inertia(0.5, 1.0, 1.0, axis=0) == 0.0  # oblate, roll


def test_transverse_entry_is_positive_and_axisymmetric():
    a, b = 2.0, 1.0
    pitch = rotational_added_inertia(a, b, b, axis=1)
    yaw = rotational_added_inertia(a, b, b, axis=2)
    assert pitch > 0.0 and yaw > 0.0
    assert pitch == pytest.approx(yaw, rel=1e-14)


# ------------------------------------------------------- closed form vs quadrature
@pytest.mark.parametrize("a,b", [(1.5, 1.0), (2.0, 1.0), (4.0, 1.0)])
def test_harmonic_integral_matches_prolate_closed_form(a, b):
    """Two independent routes to ``FF(0)``: adaptive quadrature and the
    ``u = sqrt(a^2+m)`` closed form for a prolate spheroid."""
    assert harmonic_integral(a, b, b) == pytest.approx(harmonic_integral_spheroid(a, b),
                                                      rel=1e-11)


def test_surface_product_moment_matches_triaxial_quadrature():
    """``I = (2 pi/15) a^3 b^3 c`` on a triaxial (non-axisymmetric) shape."""
    a, b, c = 1.6, 1.2, 0.9
    assert surface_product_moment(a, b, c) == pytest.approx(
        surface_moment_quadrature(a, b, c, 120), rel=1e-10)


# ------------------------------------------------------- does it solve the problem?
@pytest.mark.parametrize("axes", [(2.0, 1.0, 1.0), (1.6, 1.2, 0.9)])
def test_closed_form_potential_reproduces_rotation_data(axes):
    """Second-order finite differences of the closed-form potential must reproduce the
    prescribed rotation data ``psi = (Omega x r) . n`` -- the algebra of the amplitude
    derivation is not used here."""
    a, b, c = axes
    pts, nrm = _surface_sample(a, b, c)
    h = 2.0e-3 * min(a, b, c)

    def fn(p):
        return rotation_potential(p, a, b, c, axis=2)

    d_plus = fn(pts + h * nrm)
    d_minus = fn(pts - h * nrm)
    d_mid = ((2.0**2) * ((fn(pts + 0.5 * h * nrm) - fn(pts - 0.5 * h * nrm)) / h)
             - (d_plus - d_minus) / (2.0 * h)) / (2.0**2 - 1.0)
    psi = neumann_data(pts, nrm, a, b, c, axis=2)
    assert np.max(np.abs(d_mid - psi)) / np.max(np.abs(psi)) < 1e-5


def test_potential_decays_like_a_quadrupole():
    """``phi`` is a degree-2 harmonic (``~ 1/r^3``), not a dipole."""
    a, b, c = 2.0, 1.0, 1.0
    d = np.array([0.37, -0.55, 0.75])
    d /= np.linalg.norm(d)
    v8 = rotation_potential(d * 8.0 * a, a, b, c, axis=2)[0]
    v16 = rotation_potential(d * 16.0 * a, a, b, c, axis=2)[0]
    assert v16 / v8 == pytest.approx(0.125, rel=2e-2)


# ------------------------------------------------------------------- physics limits
def test_slender_asymptote_approaches_strip_value():
    ratios = (4.0, 8.0, 20.0)
    norms = [rotational_added_inertia(r, 1.0, 1.0, axis=2)
             / (displaced_volume(r, 1.0, 1.0) * r**2) for r in ratios]
    assert norms == sorted(norms)
    assert norms[-1] == pytest.approx(slender_limit_ratio(), rel=5e-2)
    assert slender_limit_ratio() == 0.2


def test_near_sphere_scaling_law():
    """``I_55 ~ (a^2-b^2)^2`` as the shape approaches a sphere."""
    a = 1.0
    vals = []
    for b in (0.99, 0.999, 0.9999):
        vals.append(rotational_added_inertia(a, b, b, axis=2) / (a * a - b * b) ** 2)
    assert (max(vals) - min(vals)) / np.mean(vals) < 2e-2


# ------------------------------------------------------------------ structure / API
def test_axis_permutation_consistency():
    """Rotating about ``x`` of ``(a, b, c)`` is rotating about the third axis of ``(b, c, a)``."""
    a, b, c = 1.6, 1.2, 0.9
    for axis in (0, 1, 2):
        perm = {0: (b, c, a), 1: (c, a, b), 2: (a, b, c)}[axis]
        assert rotational_added_inertia(a, b, c, axis=axis) == pytest.approx(
            rotational_added_inertia(*perm, axis=2), rel=1e-13)


def test_ellipsoidal_coordinate_is_zero_on_the_surface():
    a, b, c = 1.6, 1.2, 0.9
    pts, _ = _surface_sample(a, b, c, 6, 4)
    lam = ellipsoidal_coordinate(a, b, c, pts)
    assert np.max(np.abs(lam)) < 1e-12


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        rotational_added_inertia(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        rotational_added_inertia(1.0, 1.0, 1.0, axis=3)
    with pytest.raises(ValueError):
        rotational_added_inertia(1.0, 1.0, 1.0, rho=0.0)
    with pytest.raises(ValueError):
        harmonic_integral_spheroid(1.0, 2.0)  # not prolate


# ------------------------------------------------------------- one panel comparison
def test_panel_solver_agrees_on_a_coarse_mesh():
    """Smallest executable form of the study's main result: a/b = 2 on the 24x14 mesh.

    The panel value converges *from below* (2.39 % low at 64x34), so a 10 % window on the
    coarsest mesh is a contract on the sign and the order of magnitude only.
    """
    a, b = 2.0, 1.0
    exact = rotational_added_inertia(a, b, b, axis=2)
    panels = spheroid_panels(a, b, 24, 14)
    block = panel_rotational_added_inertia(panels)
    panel_value = float(block[1, 1])
    assert panel_value > 0.0
    assert abs(panel_value - exact) / exact < 0.10
    assert abs(block[1, 1] - block[2, 2]) / block[2, 2] < 1e-8  # axisymmetry