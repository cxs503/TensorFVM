"""References for the added mass of a half-submerged sphere (free surface, Fn -> 0).

Why this module exists
----------------------
At infinite frequency (``Fn -> 0``, no waves, no gravity) the free surface is the
Dirichlet plane ``phi = 0``.  A sphere cut by that plane at its centre is the
simplest finite body with a closed-form answer, so it is the anchor for the
free-surface branch of :mod:`tensorfvm.panel_potential` (image entries
``"free_surface"`` / ``"twin_hull_free_surface"``, whose waterplane image carries
the **opposite** source strength).

Heave (vertical, ``axis = 2``)
    The full sphere's translation potential ``phi = -U a^3 cos(theta) / (2 r^2)``
    already vanishes on the equatorial plane ``z = 0``, so the lower half of that
    field *is* the half-submerged solution.  Half of the full sphere's added mass
    therefore remains::

        M_33 = (1/3) pi rho a^3 = 0.5 * (displaced mass of the hemisphere)

    which is why the wall (double-body) image, that reproduces the *full* sphere,
    ``M_33 = (2/3) pi rho a^3``, over-predicts by a factor of two.  This module
    returns the exact value; :func:`tensorfvm.verification.free_surface_added_mass`
    measures the factor explicitly.

Sway / surge (horizontal, ``axis = 0, 1``)
    No closed form: the free-surface condition forces an *antisymmetric* Neumann
    data set on the mirrored sphere, i.e. an odd-in-``z`` interior harmonic
    series over ``P_n^1(cos theta) cos(azimuth)`` with ``n`` even.  Here that
    series is solved directly (Galerkin least squares in the normal-derivative
    basis) and the added mass is integrated analytically in azimuth and by Gauss
    quadrature in ``cos theta``.  It is a *global* spectral approximation, so it
    is independent of the panel discretisation it is used to verify (the same
    role the Lamé closed form plays in
    :mod:`tensorfvm.ellipsoid_added_inertia`).

Sign/normalisation conventions follow the panel code: the added mass is
``M_ij = -rho * integral_S phi_i n_j dS`` with ``n`` pointing into the fluid, so
:func:`translation_added_mass` and
:func:`tensorfvm.panel_added_mass.added_mass_matrix` are directly comparable.
"""
from __future__ import annotations

import numpy as np
from scipy.special import lpmv, roots_legendre

__all__ = [
    "displaced_mass",
    "deep_submergence_added_mass",
    "heave_added_mass_exact",
    "translation_added_mass",
    "sway_added_mass_series",
]


def displaced_mass(radius: float = 1.0, rho: float = 1.0) -> float:
    """Displaced mass of the half-submerged sphere, ``rho * (2/3) pi a^3``."""
    if radius <= 0.0 or rho <= 0.0:
        raise ValueError("radius and rho must be positive")
    return rho * 2.0 / 3.0 * np.pi * radius**3


def deep_submergence_added_mass(radius: float = 1.0, rho: float = 1.0) -> float:
    """Added mass of the *full* sphere, ``(2/3) pi rho a^3`` (no free surface).

    This is the value a wall/plane-of-symmetry (same-sign) image returns for the
    hemisphere, i.e. the double-body answer, which is **not** the free-surface
    answer.
    """
    return 0.5 * rho * 4.0 / 3.0 * np.pi * radius**3


def heave_added_mass_exact(radius: float = 1.0, rho: float = 1.0) -> float:
    """Exact heave added mass of the half-submerged sphere, ``(1/3) pi rho a^3``.

    Equals ``0.5 * displaced_mass``; the potential is the lower half of the full
    sphere's ``cos(theta)`` dipole, whose added mass is halved because the kinetic
    energy is integrated over half the fluid.
    """
    if radius <= 0.0 or rho <= 0.0:
        raise ValueError("radius and rho must be positive")
    return rho * np.pi * radius**3 / 3.0


def _gauss_theta(n_nodes: int):
    """Gauss nodes/weights on ``x = cos theta in (-1, 0)`` (the submerged quarter)."""
    xg, wg = roots_legendre(int(n_nodes))
    x = 0.5 * (xg - 1.0)
    w = 0.5 * wg
    return x, w


def translation_added_mass(axis: int = 2, radius: float = 1.0, rho: float = 1.0,
                           U: float = 1.0, n_terms: int = 24, n_nodes: int = 400) -> float:
    """Spectral (odd-in-``z`` harmonic) added mass of the half-submerged sphere.

    ``axis = 2`` is heave (``m = 0``, odd Legendre orders) and must reproduce
    :func:`heave_added_mass_exact` to quadrature accuracy -- that identity is the
    self-check of the machinery; ``axis = 0, 1`` is sway/surge (``m = 1``, even
    orders) and has no closed form.
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if radius <= 0.0 or rho <= 0.0:
        raise ValueError("radius and rho must be positive")
    m = 0 if axis == 2 else 1
    # odd parity in z requires (-1)**(n+m) = -1
    orders = np.array([n for n in range(1, 2 * int(n_terms) + 2) if (n + m) % 2 == 1])
    x, w = _gauss_theta(n_nodes)
    sin_t = np.sqrt(np.clip(1.0 - x**2, 0.0, None))
    # n_axis (azimuth factor removed) and the matching Neumann data at r = a
    g = x if m == 0 else sin_t
    data = U * g
    P = np.array([lpmv(m, int(n), x) for n in orders])            # (k, n_nodes)
    basis = -(orders + 1)[:, None] * radius ** (-(orders + 2))[:, None] * P
    A, *_ = np.linalg.lstsq(basis.T, data, rcond=None)
    # M = -rho * (azimuth integral) * a^2 * sum A_n a^-(n+1) * integral P_n (x) g dx
    az = 2.0 * np.pi if m == 0 else np.pi      # int dphi (m=0) or int cos^2 phi dphi
    moments = (P * g[None, :] * w[None, :]).sum(axis=1)
    coef = A * radius ** (-(orders + 1)) * moments
    return float(-rho * az * radius**2 * coef.sum())


def sway_added_mass_series(radius: float = 1.0, rho: float = 1.0, U: float = 1.0,
                           n_terms: int = 24, n_nodes: int = 400) -> float:
    """Horizontal (sway/surge) added mass of the half-submerged sphere."""
    return translation_added_mass(0, radius=radius, rho=rho, U=U, n_terms=n_terms,
                                  n_nodes=n_nodes)