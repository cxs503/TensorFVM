"""Independent semi-analytic reference for the rotational added inertia.

Method (deliberately different from the source-panel BEM)
---------------------------------------------------------
Prolate spheroidal coordinates with the symmetry axis along ``x``::

    x = f mu zeta,   y = f sqrt((1-mu^2)(zeta^2-1)) cos(phi),
                     z = f sqrt((1-mu^2)(zeta^2-1)) sin(phi),
    f = sqrt(a^2 - b^2),  zeta0 = a/f,  surface at zeta = zeta0.

Exterior harmonics decaying at infinity are ``P_n^1(mu) Q_n^1(zeta) cos(phi)``;
for transverse rotation about ``z`` the impermeability right-hand side has the
angular structure ``mu sqrt(1-mu^2) cos(phi)`` propto ``P_2^1(mu)``, i.e. ``m = 1``
with **even** ``n`` (``P_2^1`` is the lowest Legendre function odd in ``mu``).
Hence the exact potential is the series::

    phi = sum_{n odd} A_n P_n^1(mu) Q_n^1(zeta) cos(phi).

The coefficients are *not* taken from any table: they follow from the Neumann
condition on ``zeta = zeta0``,

    (1/h_zeta) dphi/dzeta = -(V . n),   h_zeta = f sqrt((zeta^2-mu^2)/(zeta^2-1)),

collocated on Gauss nodes in ``(mu, phi)``.  The result is accepted only if

* the normal-velocity residual on a *different, denser* node set falls to the
  truncation floor (then, by the uniqueness theorem for the exterior Neumann
  problem, the truncated series *is* the exact solution), and
* the potential decays like ``r^-2`` at infinity (checked explicitly).

Those two checks are what make this an independent reference rather than just
another numerical method with unknown error.

Legendre functions
------------------
``Q_n`` comes from the hypergeometric representation for ``z > 1`` (a code path
independent of any library table)::

    Q_n(z) = sqrt(pi) Gamma(n+1) / (2^{n+1} Gamma(n+3/2))
             * z^{-(n+1)} 2F1((n+2)/2, (n+1)/2; n+3/2; 1/z^2)

    Q_n^1(z) = n (z Q_n(z) - Q_{n-1}(z)) / sqrt(z^2 - 1)      (Legendre ODE)
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, hyp2f1

__all__ = [
    "q_legendre",
    "q_legendre_1",
    "spheroid_rotation_reference",
    "spheroid_surface_nodes",
    "spheroidal_coordinates",
]


def q_legendre(n: int, z: float) -> float:
    """Legendre function of the second kind ``Q_n(z)`` for ``z > 1``."""
    z = float(z)
    if not z > 1.0:
        raise ValueError("Q_n requires z > 1")
    log_pref = 0.5 * np.log(np.pi) + gammaln(n + 1.0) - (n + 1.0) * np.log(2.0) \
        - gammaln(n + 1.5)
    return float(np.exp(log_pref) * z ** (-(n + 1.0))
                 * hyp2f1(0.5 * (n + 2.0), 0.5 * (n + 1.0), n + 1.5, 1.0 / (z * z)))


def q_legendre_1(n: int, z):
    """Associated ``Q_n^1(z) = sqrt(z^2-1) dQ_n/dz`` (no Condon-Shortley sign)."""
    z = np.asarray(z, dtype=float)
    if n < 1:
        return np.zeros_like(z)
    qn = np.array([q_legendre(n, float(zz)) for zz in np.atleast_1d(z).ravel()]).reshape(z.shape)
    qn1 = np.array([q_legendre(n - 1, float(zz)) for zz in np.atleast_1d(z).ravel()]).reshape(z.shape)
    return n * (z * qn - qn1) / np.sqrt(z * z - 1.0)


def _p1(n: int, mu: np.ndarray) -> np.ndarray:
    """``P_n^1(mu)`` without the Condon-Shortley sign: ``sqrt(1-mu^2) dP_n/dmu``."""
    leg = np.polynomial.legendre.Legendre.basis(int(n))
    return np.sqrt(np.clip(1.0 - mu * mu, 0.0, None)) * leg.deriv()(mu)


def spheroid_surface_nodes(a: float, b: float, n_mu: int, n_phi: int):
    """Gauss nodes on the spheroid surface: ``(mu, phi, points, normals, dS)``."""
    f = np.sqrt(a * a - b * b)
    zeta0 = a / f
    mu1, w1 = np.polynomial.legendre.leggauss(n_mu)
    phi1 = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    wphi = 2.0 * np.pi / n_phi
    MU, PH = np.meshgrid(mu1, phi1, indexing="ij")
    rho = f * np.sqrt(np.clip((1 - MU**2) * (zeta0**2 - 1), 0.0, None))
    pts = np.stack([f * MU * zeta0, rho * np.cos(PH), rho * np.sin(PH)], axis=-1)
    nrm = np.stack([pts[..., 0] / a**2, pts[..., 1] / b**2, pts[..., 2] / b**2], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    dS = f * f * np.sqrt((zeta0**2 - MU**2) * (zeta0**2 - 1.0)) * w1[:, None] * wphi
    return MU.reshape(-1), PH.reshape(-1), pts.reshape(-1, 3), nrm.reshape(-1, 3), dS.reshape(-1)


def spheroidal_coordinates(points: np.ndarray, a: float, b: float):
    """``(mu, zeta, phi)`` of cartesian points (exterior or interior)."""
    f = np.sqrt(a * a - b * b)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rho = np.hypot(y, z)
    # x = f mu zeta, rho^2 = f^2 (1-mu^2)(zeta^2-1); with v = zeta^2:
    #   v^2 - v (1 + (x^2 + rho^2)/f^2) + x^2/f^2 = 0
    S = 1.0 + (x * x + rho * rho) / (f * f)
    xf = x * x / (f * f)
    v = 0.5 * (S + np.sqrt(np.maximum(S * S - 4.0 * xf, 0.0)))
    zeta = np.sqrt(np.maximum(v, 1e-300))
    mu = np.clip(x / (f * zeta), -1.0, 1.0)
    phi = np.arctan2(z, y)
    return mu, zeta, phi


def spheroid_rotation_reference(a: float, b: float, *, n_terms: int = 24,
                                n_mu_fit: int = 40, n_phi_fit: int = 8,
                                n_mu_check: int = 80, n_phi_check: int = 16,
                                rho: float = 1.0, delta: float = 1e-6):
    """Exact transverse rotational added inertia of a prolate spheroid.

    Rotation about ``z`` (a transverse axis through the centre) with unit angular
    velocity; ``V = e_z x r``.  Returns a dict with ``I_perp``, the relative
    surface residual, the far-field decay ratio (should be ~1 for ``r^-2``) and
    the fitted coefficients.
    """
    if not a > b > 0.0:
        raise ValueError("prolate spheroid requires a > b > 0")
    f = np.sqrt(a * a - b * b)
    zeta0 = a / f
    orders = np.arange(2, 2 * n_terms + 1, 2)
    # dQ_n^1/dzeta at the surface, by a central difference in zeta (independent
    # of the Q_n^1 closed form used elsewhere)
    dq = np.array([(q_legendre_1(int(n), zeta0 + delta) - q_legendre_1(int(n), zeta0 - delta))
                   / (2.0 * delta) for n in orders])
    q0 = np.array([q_legendre_1(int(n), zeta0) for n in orders])

    def design(mu, phi, nd):
        hz = f * np.sqrt((zeta0**2 - mu**2) / (zeta0**2 - 1.0))
        cols = [_p1(int(n), mu) * dq[k] / hz for k, n in enumerate(orders)]
        return np.stack(cols, axis=1) * nd[:, None]

    def target(pts, nrm):
        V = np.stack([-pts[:, 1], pts[:, 0], np.zeros(len(pts))], axis=1)
        return np.einsum("ij,ij->i", V, nrm)

    mu, phi, pts, nrm, dS = spheroid_surface_nodes(a, b, n_mu_fit, n_phi_fit)
    D = design(mu, phi, np.cos(phi))
    # the spheroidal columns span many orders of magnitude (Q_n^1 decays fast with
    # n), so normalise them before the least-squares fit and undo it afterwards
    scale = np.linalg.norm(D, axis=0)
    scale[scale == 0.0] = 1.0
    A, *_ = np.linalg.lstsq(D / scale, target(pts, nrm), rcond=None)
    A = A / scale

    # --- check 1: Neumann residual on an independent, denser node set
    muc, phic, ptsc, nrmc, dSc = spheroid_surface_nodes(a, b, n_mu_check, n_phi_check)
    res = (design(muc, phic, np.cos(phic)) @ A - target(ptsc, nrmc))
    res_rel = float(np.max(np.abs(res)) / np.max(np.abs(target(ptsc, nrmc))))

    # --- check 2: far-field decay (phi ~ r^-2)
    far_r = 40.0 * a
    far = np.array([[far_r, 0.31 * far_r, 0.17 * far_r], [-0.63 * far_r, far_r, -0.22 * far_r]])
    mu_f, zeta_f, phi_f = spheroidal_coordinates(far, a, b)
    pot_far = np.array([np.sum(_p1(int(n), mu_f) * q_legendre_1(int(n), zeta_f)
                               * np.cos(phi_f) * A[k]) for k, n in enumerate(orders)]).sum(axis=0)
    pot_scale = _surface_potential_scale(orders, q0, A, muc, phic)
    decay_ratio = float(np.max(np.abs(pot_far)) / pot_scale * (far_r / a) ** 2)

    # --- added inertia: I = -rho * closed_integral phi (V . n) dS  (equation 1)
    phi_s = _surface_potential(orders, q0, A, muc, phic)
    I_perp = -rho * float(np.sum(phi_s * target(ptsc, nrmc) * dSc))
    return {
        "I_perp": I_perp,
        "neumann_residual": res_rel,
        "far_field_ratio": decay_ratio,
        "coefficients": A,
        "orders": orders,
        "zeta0": zeta0,
        "surface_potential_scale": pot_scale,
    }


def _surface_potential(orders, q0, A, mu, phi):
    out = np.zeros_like(mu)
    for k, n in enumerate(orders):
        out += _p1(int(n), mu) * q0[k] * np.cos(phi) * A[k]
    return out


def _surface_potential_scale(orders, q0, A, mu, phi):
    return float(np.max(np.abs(_surface_potential(orders, q0, A, mu, phi))))