"""Closed-form rotational added inertia of a triaxial ellipsoid in ideal fluid.

Why this module exists
----------------------
``docs/VERIFICATION_DEBT_LEDGER.md`` entry **D6** is the honest-open half of D1: the
source-panel rotational block (``panel_added_mass.rotational_added_inertia``) was
delivered with exact anchors for the sphere (``= 0``), the spheroid roll (``= 0``) and
the sphere about an offset axis (``m_a d^2``), but the **non-zero** transverse block of a
spheroid had *no reference value*, so the acceptance sentence "analytic < 3 %" was
WITHHELD.  This module supplies that reference: the potential problem for pure rotation of
an ellipsoid has a classical closed-form solution in ellipsoidal coordinates, and the
added inertia follows with no discretisation error beyond a one-dimensional quadrature
(relative error ``~1e-13``, see :func:`harmonic_integral`).

Theory
------
Body: ``x^2/A^2 + y^2/B^2 + z^2/C^2 = 1`` (``A, B, C`` = semi-axes, ``F`` below is the
level function, *not* the integral).  A rotation with angular velocity ``Omega`` about a
principal axis drives the Neumann data

    psi = (Omega x r) . n,     n = grad(F)/|grad(F)|,   F = x^2/A^2 + y^2/B^2 + z^2/C^2,

and the perturbation potential ``phi`` solves

    laplace(phi) = 0,     dphi/dn = psi on S,     phi -> 0 at infinity.                (1)

For the rotation about the third axis (``z``) the data is

    psi = Omega * 2 x y (1/B^2 - 1/A^2) / |grad F|.                                   (2)

The degree-2 Lamé function that reproduces exactly this ``x y / |grad F|`` structure is

    phi = C_2 * x y * FF(lambda),   FF(lambda) = integral_lambda^inf dm / ((A^2+m)(B^2+m) Delta(m)),
    Delta(m) = sqrt((A^2+m)(B^2+m)(C^2+m)),                                           (3)

with ``lambda`` the outer ellipsoidal coordinate (``lambda = 0`` on ``S``).  (3) is
harmonic (checked numerically in the verification), decays like ``1/r`` and matches the
normal derivative (2) if and only if one uses the identity

    D := x^2/A^4 + y^2/B^4 + z^2/C^4 = |grad F|^2 / 4,   d(\(lambda\))/dn = 4/|grad F|, (4)

which turns ``dphi/dn`` into the single term

    dphi/dn = C_2 * x y / |grad F| * [ 2 (1/A^2 + 1/B^2) FF(0) - 4 / (A^2 B^2 A B C) ]. (5)

Matching (2) and (5) fixes the amplitude,

    C_2 = 2 (1/B^2 - 1/A^2) / coef,   coef = 2 (1/A^2 + 1/B^2) FF(0) - 4/(A^2 B^2 A B C), (6)

and the Kirchhoff added inertia ``A_zz = -rho * integral_S phi psi dS`` (the minus sign is
the orientation of the outward normal in Green's identity, ``T = -(1/2) rho integral_S
phi dphi/dn dS``) becomes

    A_zz = -4 rho (1/B^2 - 1/A^2)^2 FF(0) * I / coef,       I = integral_S x^2 y^2 / |grad F| dS
         = (2 pi / 15) A^3 B^3 C.                                                     (7)

``I`` is elementary: the shell identity ``integral_body h dV = const * integral_S
h/|grad F| dS`` for a degree-4 homogeneous ``h`` gives ``I = 4 (a^2 b^2 / 15) * (4/3 pi
A B C)`` in closed form; the value implemented here, ``(2 pi/15) A^3 B^3 C``, was
confirmed against high-order surface quadrature for triaxial (non-axisymmetric) shapes to
``7e-15`` relative (``verification/ellipsoid_rotational_reference.py``, criterion C2).

So (7) is a **closed form** whose only numerical ingredient is the one-dimensional
integral ``FF(0)`` (elliptic-type; quadrature ``1e-13``, plus an independent closed form
for prolate spheroids from the substitution ``u = sqrt(A^2+m)``, :func:`harmonic_integral_spheroid`).

Exact limits that this formula reproduces (all used as tests)
-------------------------------------------------------------
* sphere ``A = B = C``: ``1/B^2 - 1/A^2 = 0`` -> ``A_zz = 0`` exactly, the classical
  ``diag(m_a, m_a, m_a, 0, 0, 0)`` added-mass matrix of a sphere in ideal fluid;
* body of revolution about the rotation axis (``B = C``, rotation about ``x``):
  ``1/C^2 - 1/B^2 = 0`` -> roll added inertia ``= 0`` exactly;
* slender limit (``A/B -> inf``, rotation about a transverse axis): ``coef -> -2/A^3``,
  ``FF(0) -> 1/A^3``, giving ``A_zz -> (4/15) pi rho A^3 B^2`` i.e.
  ``A_55 / (rho V A^2) -> 1/5`` -- the strip-theory value, which is also the rigid-body
  ratio in that limit (transverse added inertia equals the body's own moment of inertia
  of a long spheroid);
* near-sphere vanishing ``~ (A^2 - B^2)^2``, the behaviour the panel study observed.

Sign / normal convention
------------------------
Equation (1) uses the **body's own** normal-velocity data (``dphi/dn = +psi``, ``n``
pointing out of the body into the fluid), which is the convention that makes the added
inertia positive definite.  ``panel_added_mass`` stores its matrix with the opposite
overall sign of the potential (it solves ``dphi/dn = -psi`` and integrates
``A_ab = -rho integral phi_a psi_b dS``); the two conventions therefore *agree in sign and
magnitude* for the physical added inertia, which is what the verification compares.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

__all__ = [
    "AXIS_NAMES",
    "displaced_volume",
    "ellipsoidal_coordinate",
    "harmonic_integral",
    "harmonic_integral_spheroid",
    "neumann_data",
    "normal_derivative_closed_form",
    "rotation_potential",
    "rotational_added_inertia",
    "slender_limit_ratio",
    "surface_product_moment",
]

AXIS_NAMES = ("roll_x", "pitch_y", "yaw_z")
_QUAD_REL = 1e-13


def displaced_volume(a: float, b: float, c: float) -> float:
    """Volume of the ellipsoid with semi-axes ``a, b, c``."""
    return 4.0 / 3.0 * np.pi * a * b * c


def _check_axes(a: float, b: float, c: float) -> None:
    if min(a, b, c) <= 0.0:
        raise ValueError("semi-axes must be positive")


def _axis_permutation(a: float, b: float, c: float, axis: int):
    """Return ``(A, B, C)`` such that the rotation axis is the third coordinate.

    Rotation about ``x`` mixes ``y, z`` -> ``(b, c, a)``; about ``y`` -> ``(c, a, b)``;
    about ``z`` -> ``(a, b, c)``.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    return {0: (b, c, a), 1: (c, a, b), 2: (a, b, c)}[axis]


def harmonic_integral(a: float, b: float, c: float, lam: float = 0.0) -> float:
    r"""``FF(lam) = int_lam^inf dm / ((a^2+m)(b^2+m) sqrt((a^2+m)(b^2+m)(c^2+m)))``.

    The substitution ``m = lam + t^2`` removes the integrable endpoint singularity at
    ``m = lam`` and maps the semi-infinite range to ``t in [0, inf)``; the integrand then
    decays like ``t^-6``, so :func:`scipy.integrate.quad` overshoots its target accuracy
    easily (verified against the independent closed form of
    :func:`harmonic_integral_spheroid`).
    """
    _check_axes(a, b, c)
    a2, b2, c2 = a * a, b * b, c * c

    def integrand(t: float) -> float:
        s1, s2, s3 = a2 + lam + t * t, b2 + lam + t * t, c2 + lam + t * t
        return 2.0 * t / (s1 * s2 * np.sqrt(s1 * s2 * s3))

    # The integrand is O(t^-7) at large t, so the semi-infinite range is safe for quad;
    # truncating it at a finite upper limit is *not* (the tail beyond the first panel is
    # not negligible and quad's local error estimate does not see it -- a truncation bug
    # that produced a 31 % error before this was fixed).
    val, _err = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=_QUAD_REL, limit=500)
    return float(val)


def harmonic_integral_spheroid(a: float, b: float) -> float:
    r"""Closed form of :func:`harmonic_integral` at ``lam = 0`` for a prolate spheroid.

    With ``c = b`` and eccentricity ``e = sqrt(1 - b^2/a^2)`` the substitution
    ``u = sqrt(a^2 + m)`` gives

        FF(0) = 2 int_a^inf du / (u^2 (u^2 - a^2 e^2)^2)
              = 2 [ 1/(a k^4) - L/(2 k^5) + a/(2 k^4 (a^2-k^2)) - L/(4 k^3 k^2) ],
        k = a e,   L = ln((1+e)/(1-e)),

    using ``1/(u^2 (u^2-k^2)^2) = 1/(k^4 u^2) - 1/(k^4 (u^2-k^2)) + 1/(k^2 (u^2-k^2)^2)``.
    This is a *different* route to the same number and is used as an independent check of
    the quadrature (it is not merely the same integral written twice).
    """
    _check_axes(a, b, 1.0)
    if not a > b:
        raise ValueError("expected a prolate spheroid, a > b = c")
    e = np.sqrt(1.0 - (b / a) ** 2)
    k = a * e
    L = np.log((1.0 + e) / (1.0 - e))
    s = (
        1.0 / (a * k**4)
        - L / (2.0 * k**5)
        + a / (2.0 * k**4 * (a * a - k * k))
        - L / (4.0 * k**5)
    )
    return float(2.0 * s)


def surface_product_moment(a: float, b: float, c: float) -> float:
    r"""``I = integral_S x^2 y^2 / |grad F| dS = (2 pi / 15) a^3 b^3 c`` (closed form).

    Derived from the shell identity for the level sets of ``F``: for a homogeneous ``h``
    of degree 4, ``integral_body h dV = (const) integral_{S} h/|grad F| dS`` with the
    constant determined by the sphere; equivalently by direct evaluation of the
    ``(theta, phi)`` quadrature (agreement ``7e-15``, verification criterion C2).
    """
    _check_axes(a, b, c)
    return float(2.0 * np.pi / 15.0 * a**3 * b**3 * c)


def _coefficient(a: float, b: float, c: float) -> tuple[float, float]:
    """Return ``(FF(0), coef)`` of equations (5)-(6), with ``c`` along the rotation axis."""
    ff0 = harmonic_integral(a, b, c)
    coef = 2.0 * (1.0 / a**2 + 1.0 / b**2) * ff0 - 4.0 / (a * a * b * b * a * b * c)
    return ff0, coef


def rotational_added_inertia(a: float, b: float, c: float, *, axis: int = 2,
                             rho: float = 1.0) -> float:
    r"""Added moment of inertia of an ellipsoid about a principal axis, closed form.

    ``axis = 0, 1, 2`` selects rotation about ``x`` (roll), ``y`` (pitch), ``z`` (yaw)
    of the ``a, b, c`` ellipsoid.  Returns equation (7),

        A_axis = -4 rho (1/B^2 - 1/A^2)^2 FF(0) (2 pi/15 A^3 B^3 C) / coef,

    which is positive definite (``coef < 0``) and exactly zero when the two semi-axes
    transverse to the rotation axis are equal (sphere: ``A = B = C``; spheroid rolling
    about its symmetry axis: ``B = C``).
    """
    _check_axes(a, b, c)
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    A, B, C = _axis_permutation(a, b, c, axis)
    ff0, coef = _coefficient(A, B, C)
    moment = surface_product_moment(A, B, C)
    return float(-4.0 * rho * (1.0 / B**2 - 1.0 / A**2) ** 2 * ff0 * moment / coef)


def slender_limit_ratio() -> float:
    """``A_55 / (rho V a^2) -> 1/5`` as ``a/b -> inf`` (strip theory).

    Route: ``coef -> -2/a^3``, ``FF(0) -> 1/a^3``, ``(1/b^2 - 1/a^2)^2 -> 1/b^4``,
    ``I -> (2 pi/15) a^3 b^4`` so that ``A -> -4 rho (1/b^4) a^3 (-1/2) (2 pi/15) a^3 b^4
    = (4/15) pi rho a^3 b^2``, and ``rho V a^2 = (4/3) pi rho a^3 b^2`` leaves ``1/5``.
    The same number is the rigid-body ratio in that limit: a long spheroid's transverse
    moment of inertia is ``(1/5) m (a^2 + b^2) -> (4/15) pi rho a^3 b^2``.
    """
    return 0.2


def ellipsoidal_coordinate(a: float, b: float, c: float, points) -> np.ndarray:
    r"""Outer ellipsoidal coordinate ``lambda`` of the points (``0`` on ``S``).

    Defined by ``x^2/(a^2+lam) + y^2/(b^2+lam) + z^2/(c^2+lam) = 1``.  ``g(lam)`` is
    strictly decreasing above the largest pole ``lam > -min(a^2, b^2, c^2)``, so a
    bracketed Brent root find is used.  Values are negative for interior points; the
    closed-form potential of :func:`rotation_potential` is only claimed outside the body.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    a2, b2, c2 = a * a, b * b, c * c
    pole = min(a2, b2, c2)
    out = np.empty(len(pts))
    for i, (x, y, z) in enumerate(pts):
        def g(lam: float) -> float:
            return x * x / (a2 + lam) + y * y / (b2 + lam) + z * z / (c2 + lam) - 1.0

        # g -> -1 as lam -> inf; start well above the pole and expand until g > 0
        hi = max(a2, b2, c2) + x * x + y * y + z * z + 1.0
        while g(hi) > 0.0:  # pragma: no cover - needs points far outside
            hi *= 2.0
        lo = -pole * (1.0 - 1e-13)
        # push lo away from the pole until g(lo) > 0 (guaranteed unless point == pole)
        for _ in range(200):
            if g(lo) > 0.0:
                break
            lo = 0.5 * (lo - pole)
        out[i] = brentq(g, lo, hi, xtol=1e-15 * max(1.0, hi), rtol=8.9e-16, maxiter=200)
    return out


def _permuted(points, axis: int) -> np.ndarray:
    """Reorder columns to the ``(A, B, axis)`` frame of :func:`_axis_permutation`."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    idx = {0: (1, 2, 0), 1: (2, 0, 1), 2: (0, 1, 2)}[axis]
    return pts[:, list(idx)]


def rotation_potential(points, a: float, b: float, c: float, *, axis: int = 2,
                       omega: float = 1.0) -> np.ndarray:
    r"""Closed-form potential of the unit-rotation problem (3), exterior points.

    ``phi(x) = C_2 * x_i x_j * FF(lambda(x))`` with ``C_2`` from (6) and ``(i, j)`` the
    two axes transverse to ``axis`` (``y z`` for ``axis=0``, ``z x`` for ``1``,
    ``x y`` for ``2``).  Exposed so that the verification can differentiate it
    numerically and compare the resulting normal derivative with the prescribed data
    (2) -- an algebra-independent check of the whole derivation.
    """
    _check_axes(a, b, c)
    A, B, C = _axis_permutation(a, b, c, axis)
    pts = _permuted(points, axis)
    ff0, coef = _coefficient(A, B, C)
    c2 = omega * 2.0 * (1.0 / B**2 - 1.0 / A**2) / coef
    lam = ellipsoidal_coordinate(A, B, C, pts)
    ff = np.array([harmonic_integral(A, B, C, float(v)) for v in lam])
    return c2 * pts[:, 0] * pts[:, 1] * ff


def neumann_data(points, normals, a: float, b: float, c: float, *, axis: int = 2,
                 omega: float = 1.0) -> np.ndarray:
    r"""Prescribed data ``psi = (Omega x r) . n`` of the rotation problem (2).

    ``normals`` must be the outward unit normals of the ellipsoid at ``points``.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    nrm = np.atleast_2d(np.asarray(normals, dtype=float))
    if pts.shape != nrm.shape:
        raise ValueError("points and normals must have the same shape")
    e = np.zeros(3)
    e[axis] = omega
    omega_cross_r = np.cross(np.broadcast_to(e, pts.shape), pts)
    return np.einsum("ij,ij->i", omega_cross_r, nrm)


def normal_derivative_closed_form(points, a: float, b: float, c: float, *, axis: int = 2,
                                  omega: float = 1.0) -> np.ndarray:
    r"""``dphi/dn`` of :func:`rotation_potential` on ``S`` from equation (5).

    Provided for completeness/speed; the verification deliberately differentiates
    :func:`rotation_potential` *numerically* instead, so that this algebra is not used to
    validate itself.
    """
    A, B, C = _axis_permutation(a, b, c, axis)
    pts = _permuted(points, axis)
    ff0, coef = _coefficient(A, B, C)
    c2 = omega * 2.0 * (1.0 / B**2 - 1.0 / A**2) / coef
    grad_norm = 2.0 * np.sqrt((pts[:, 0] / A**2) ** 2 + (pts[:, 1] / B**2) ** 2
                             + (pts[:, 2] / C**2) ** 2)
    return c2 * pts[:, 0] * pts[:, 1] / grad_norm * coef
