"""Independent reference solution for the ``m = 1`` prolate-spheroid potential flow.

Why this module exists
----------------------
:func:`tensorfvm.panel_added_mass.rotational_added_inertia` gives the rotational
added inertia of a spheroid from a **surface source panel** method, but for the
transverse (pitch/yaw) block there is no exact anchor: no elementary closed form
exists in this tree (see ``docs/VERIFICATION_DEBT_LEDGER.md`` item D6).  The panel
numbers therefore had no independent reference, only self-consistency (symmetry
``I_55 = I_66``, reciprocity, mesh trend).

This module closes that gap with a **different discretisation of the same
mathematical problem**: a finite-difference solution of the Laplace equation in
prolate spheroidal coordinates instead of a boundary-integral equation.  The two
routes share nothing but the definition of the added mass -- different unknown
(potential field on a volume grid vs. source density on the surface),
different error structure (scheme truncation vs. panel quadrature), different
conditioning.  Agreement between them is a genuine cross-check.

Formulation
-----------
Prolate spheroidal coordinates (``x`` = symmetry axis, focal distance
``c = sqrt(a^2 - b^2)``, ``zeta0 = a / c``)::

    x = c mu zeta,  y = c sqrt((zeta^2-1)(1-mu^2)) cos(psi),
                    z = c sqrt((zeta^2-1)(1-mu^2)) sin(psi).

Both physical forcings that matter here -- unit translation along a transverse
axis and unit rotation about a transverse axis -- have ``cos(psi)`` azimuthal
dependence, so the disturbance potential is ``phi = f(mu, zeta) cos(psi)`` and
Laplace's equation reduces to the 2-D problem

    d/dmu[(1-mu^2) df/dmu] + d/dzeta[(zeta^2-1) df/dzeta]
        - f (zeta^2 - mu^2) / ((zeta^2-1)(1-mu^2)) = 0,          (1)

with the Neumann data on the body ``zeta = zeta0``

    df/dzeta = h_zeta * (V . n) / cos(psi),   h_zeta = c sqrt((zeta0^2-mu^2)/(zeta0^2-1)),

and ``f -> 0`` at infinity.  ``V . n`` is

* transverse unit translation ``U = e_y``:  ``n_y = sin(theta) / (b k)``,
* transverse unit rotation ``Omega = e_z``:  ``(Omega x r) . n = a b mu sin(theta)
  (1/b^2 - 1/a^2) / k``,   ``k = sqrt(mu^2/a^2 + sin(theta)^2/b^2)``,
  each divided by ``cos(psi)``, where ``mu = cos(theta)``.

The added mass / added inertia follows from the fluid kinetic energy
``T = -rho/2 * contour(phi, dphi/dn)``:

    M = -rho * pi * c * (zeta0^2 - 1) * integral_{-1}^{1} f df/dzeta dmu.   (2)

Validation built in
-------------------
The **translational** problem in (1)-(2) has an exact classical answer
(:func:`tensorfvm.panel_added_mass.lamb_spheroid_factors`), so the same solver
that produces the (anchorless) rotational number is first checked against a
known closed form at every aspect ratio -- that is what makes the rotational
number trustworthy rather than just self-consistent.

Discretisation notes
--------------------
* ``theta`` is the natural angular variable: on a uniform ``theta`` grid ``f``
  is smooth (``f ~ theta`` at the poles, no square-root behaviour), with
  Dirichlet ``f = 0`` at ``theta = 0, pi``.
* In ``zeta`` the solution has a boundary layer of thickness ``~ a b / c^2``
  next to the body, which a uniform grid in ``log(zeta)`` does *not* resolve
  (that mistake cost two orders of magnitude of accuracy for slender bodies).
  The grid is therefore built on ``zeta(tau) = zeta0 + delta * tau/(1-tau)``
  with ``delta = a b / c^2``, which both resolves the layer and maps infinity to
  ``tau = 1``; the operator is discretised in flux form so the non-uniform
  metric stays second-order accurate.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "transverse_added_mass_fd",
    "transverse_added_inertia_fd",
    "slender_limit_added_inertia",
]

_KINDS = ("translation", "rotation")


def _forcing(a: float, b: float, c: float, zeta0: float, mu: np.ndarray, kind: str) -> np.ndarray:
    """``df/dzeta`` on the body for unit translation / rotation (``cos psi`` removed)."""
    s2 = np.clip(1.0 - mu * mu, 0.0, None)
    s = np.sqrt(s2)
    k = np.sqrt(mu * mu / a ** 2 + s2 / b ** 2)
    h_zeta = c * np.sqrt((zeta0 ** 2 - mu ** 2) / (zeta0 ** 2 - 1.0))
    if kind == "translation":       # V = e_y  ->  n_y / cos(psi)
        return h_zeta * (s / (b * k))
    if kind == "rotation":          # Omega = e_z  ->  (Omega x r) . n / cos(psi)
        return h_zeta * (a * b * mu * s * (1.0 / b ** 2 - 1.0 / a ** 2) / k)
    raise ValueError(f"kind must be one of {_KINDS}")


def _solve(a: float, b: float, kind: str, n_theta: int, n_tau: int,
           tau_max: float, rho: float):
    if not (a >= b > 0.0):
        raise ValueError("prolate spheroid requires a >= b > 0")
    if a <= b * (1.0 + 1e-8):
        raise ValueError("a/b too close to 1: use the exact sphere result (zero)")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}")
    c = np.sqrt(a * a - b * b)
    zeta0 = a / c
    delta = a * b / (c * c)                     # boundary-layer thickness scale

    theta = np.linspace(0.0, np.pi, n_theta)
    dth = theta[1] - theta[0]
    mu = np.cos(theta)
    s = np.sin(theta)
    g = _forcing(a, b, c, zeta0, mu, kind)

    tau = np.linspace(0.0, tau_max, n_tau)
    dtau = tau[1] - tau[0]

    def zeta_of(t):
        return zeta0 + delta * t / (1.0 - t)

    def jac(t):
        return delta / (1.0 - t) ** 2

    zt = zeta_of(tau)
    jt = jac(tau)
    zf = zeta_of(tau + 0.5 * dtau)               # face-centred
    jf = jac(tau + 0.5 * dtau)
    flux = (zf ** 2 - 1.0) / jf                  # (zeta^2-1)/J at faces

    ni = n_theta - 2                             # interior theta nodes
    nj = n_tau - 1                               # tau nodes 0 .. n_tau-2 (last = far field)
    idx = lambda i, j: (i - 1) * nj + j          # noqa: E731

    rows, cols, vals = [], [], []
    rhs = np.zeros(ni * nj)
    for i in range(1, n_theta - 1):
        si = s[i]
        # theta operator: (1/s) d/dtheta (s df/dtheta), central differences
        ap = np.sin(theta[i] + 0.5 * dth) / (si * dth ** 2)
        am = np.sin(theta[i] - 0.5 * dth) / (si * dth ** 2)
        src = (zt ** 2 - mu[i] ** 2) / ((zt ** 2 - 1.0) * s[i] ** 2)
        for j in range(nj):
            r = idx(i, j)
            if j == 0:
                # Neumann df/dzeta = g  ->  df/dtau = J0 * g, second-order one sided
                rows += [r, r, r]
                cols += [idx(i, 0), idx(i, 1), idx(i, 2)]
                vals += [-3.0 / (2.0 * dtau), 4.0 / (2.0 * dtau), -1.0 / (2.0 * dtau)]
                rhs[r] = jt[0] * g[i]
                continue
            cp = flux[j] / (jt[j] * dtau ** 2)
            cm = flux[j - 1] / (jt[j] * dtau ** 2)
            c0 = -(cp + cm) - (ap + am) - src[j]
            if i - 1 >= 1:
                rows.append(r); cols.append(idx(i - 1, j)); vals.append(am)
            rows.append(r); cols.append(r); vals.append(c0)
            if i + 1 <= n_theta - 2:
                rows.append(r); cols.append(idx(i + 1, j)); vals.append(ap)
            rows.append(r); cols.append(idx(i, j - 1)); vals.append(cm)
            if j + 1 <= nj - 1:
                rows.append(r); cols.append(idx(i, j + 1)); vals.append(cp)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(ni * nj, ni * nj)).tocsr()
    f = spla.spsolve(A, rhs).reshape(ni, nj)

    f0 = f[:, 0]
    integral = float(np.sum(f0 * g[1:-1] * s[1:-1]) * dth)   # int f df/dzeta dmu
    value = -rho * np.pi * c * (zeta0 ** 2 - 1.0) * integral
    return value, zt, f0


def transverse_added_mass_fd(a: float, b: float, *, rho: float = 1.0,
                             n_theta: int = 129, n_tau: int = 161,
                             tau_max: float = 0.999) -> float:
    """Transverse added mass of a prolate spheroid, finite-difference reference."""
    value, _, _ = _solve(a, b, "translation", n_theta, n_tau, tau_max, rho)
    return value


def transverse_added_inertia_fd(a: float, b: float, *, rho: float = 1.0,
                                n_theta: int = 129, n_tau: int = 161,
                                tau_max: float = 0.999) -> float:
    """Transverse added moment of inertia of a prolate spheroid (about the origin).

    This is the anchor the panel method lacked: unit rotation about a transverse
    axis is a different forcing from unit translation, but the same solver, whose
    translational branch is validated against the exact Lamb closed form.
    """
    value, _, _ = _solve(a, b, "rotation", n_theta, n_tau, tau_max, rho)
    return value


def slender_limit_added_inertia(a: float, b: float, *, rho: float = 1.0) -> float:
    """Strip-theory (slender-body) limit ``(4/15) pi rho a^3 b^2``.

    With ``r(x) = b sqrt(1 - x^2/a^2)`` the strip sum
    ``integral_{-a}^{a} rho pi r(x)^2 x^2 dx`` evaluates exactly to
    ``(4/15) pi rho a^3 b^2 = (1/5) rho V a^2``; it is the ``a/b -> inf``
    asymptote of :func:`transverse_added_inertia_fd` (reached slowly, with
    logarithmic corrections, so it is a sanity limit, not a 3% anchor).
    """
    return rho * 4.0 / 15.0 * np.pi * a ** 3 * b ** 2