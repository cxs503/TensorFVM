"""Potential-flow added mass from the source-panel solver, with exact anchors.

Why this module exists
----------------------
:mod:`tensorfvm.maneuvering.dynamics` needs added masses ``m_x, m_y, J_z`` to
integrate the manoeuvring equations.  Those numbers currently come from
*slender-body* formulas and are documented there as an **upper bound**; nothing
in the tree measured how large that over-estimate is.  This module supplies the
first-principles route -- the added mass of a closed body in an unbounded ideal
fluid -- and the exact analytic references needed to verify it.

Theory
------
For a body translating with velocity ``U_j`` in an inviscid, irrotational,
incompressible fluid the fluid kinetic energy is ``T = 1/2 (M_ij) U_i U_j`` with

    M_ij = -rho * contour_integral_S phi_i n_j dS,                       (1)

where ``phi_i`` is the perturbation potential driven by unit translation in
direction ``i``,

    laplace(phi_i) = 0,   dphi_i/dn = -n_i on S,   phi_i -> 0 at infinity. (2)

``n`` in (2) is the normal pointing **into the fluid** (outward from the body),
which is exactly the convention of :class:`tensorfvm.twin_hull_geometry.PanelSet`;
the minus sign in (1) is the corresponding orientation factor (the equivalent
statement with the body normal ``n_body = -n`` carries no minus).  The result is
positive definite for a closed body, as it must be.  Equations (1)-(2) are the
standard added-mass definition (Lamb 1932, Hydrodynamics, §108-114; Landau &
Lifshitz, Fluid Mechanics, §11; Newman 1977, Marine Hydrodynamics, §4.4).

The Neumann problem (2) is precisely what :func:`tensorfvm.panel_potential.solve_sources`
solves: with ``U = e_i`` the returned source strengths ``sigma`` satisfy
``(U + grad(phi)) . n = 0``, i.e. ``dphi/dn = -n_i``.  The potential of a
constant-strength single layer is ``phi(x) = sum_j sigma_j / (4 pi) * integral_j dS/|x-r'|``,
which :func:`source_potential` evaluates with the same bilinear Gauss rule as the
velocity kernel, so the potential and its normal derivative are discretised
consistently.

Exact anchors (what makes the numbers verifiable)
-------------------------------------------------
An ellipsoid with semi-axes ``a, b, c`` has diagonal added mass
``M_ii = k_i * rho * V``, ``V = 4/3 pi a b c``, with the Lamb factors

    alpha0 = a b c * integral_0^inf dl / ((a^2+l) sqrt((a^2+l)(b^2+l)(c^2+l))),
    k_1 = alpha0 / (2 - alpha0),                                          (3)

and cyclic permutations for ``k_2, k_3``.  For an axisymmetric spheroid the
three factors satisfy ``alpha0 + beta0 + gamma0 = 2``, so ``beta0 = (2-alpha0)/2``.
For a prolate spheroid (``a > b = c``, eccentricity ``e = sqrt(1 - b^2/a^2)``)
(3) has the closed form

    alpha0 = ((1 - e^2) / e^3) * (ln((1+e)/(1-e)) - 2e).                  (4)

An oblate spheroid (``a = b > c``) has the companion closed form (5) in
:func:`lamb_ellipsoid_factors`.  Both closed forms and the defining integral (3)
are implemented here and the tests require them to agree, so a typo in any route
is caught.  The physical limits are also pinned:

* ``e -> 0`` (sphere): ``alpha0 = beta0 = 2/3`` so ``k = 1/2`` for every axis --
  the classical added-mass coefficient of a sphere, ``M = 0.5 rho V``;
* ``e -> 1`` (slender prolate): axial ``k -> 0``, transverse ``k -> 1``;
* ``e -> 1`` (flat oblate): ``gamma0 -> 2`` so ``k_3 -> inf`` at fixed displaced
  volume while the added mass itself keeps the finite disc value
  ``M_33 -> (8/3) rho a**3``;
* combination with the slender-body estimate ``m_perp ~ rho pi b^2 L`` shows the
  latter exceeds the exact value (see ``docs/verification-added-mass-potential``).

Sign convention note: :func:`added_mass_matrix` returns ``M_ij`` of (1), i.e.
positive definite for a closed body with ``M_ii > 0``.

References
----------
* Lamb, H. (1932) *Hydrodynamics*, 6th ed., §108-114 (ellipsoid added mass).
* Landau & Lifshitz (1987) *Fluid Mechanics*, §11 (definition of added mass).
* Newman, J.N. (1977) *Marine Hydrodynamics*, §4.4 (added-mass coefficients).
* Hess, J.L. & Smith, A.M.O. (1964) *Progress in Aeronautical Sciences* 8, 1-138
  (constant-strength source panel method used by :mod:`panel_potential`).
"""

from __future__ import annotations

import numpy as np

from .panel_potential import IMAGE_SPECS, image_points, panel_quadrature, solve_sources
from .twin_hull_geometry import PanelSet, panelise_nodes

__all__ = [
    "added_mass_matrix",
    "displaced_volume",
    "lamb_ellipsoid_factors",
    "lamb_integral_factors",
    "lamb_spheroid_factors",
    "source_potential",
    "sphere_added_mass",
    "spheroid_panels",
    "surface_quadrature",
]


# ---------------------------------------------------------------------------
# quadrature shared by the analytic anchor integrals
# ---------------------------------------------------------------------------
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached ``n``-point Gauss-Legendre nodes/weights on ``[-1, 1]``.

    ``numpy.polynomial.legendre.leggauss`` costs O(n**2) Newton iterations
    (seconds for n ~ 600); the anchor integrals below ask for the same rule on
    every call, so caching keeps an aspect-ratio sweep cheap.
    """
    hit = _GL_CACHE.get(n)
    if hit is None:
        hit = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = hit
    return hit


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def displaced_volume(panels: PanelSet) -> float:
    """Enclosed volume ``V = (1/3) contour_integral x . n dS`` (divergence theorem).

    Equal to the analytic volume of the body only up to the panelisation error;
    the difference is the geometric discretisation error that dominates the
    added-mass error below (see the verification report).
    """
    return float(np.sum(np.einsum("ij,ij->i", panels.centroids, panels.normals) * panels.areas) / 3.0)


def spheroid_panels(a: float, b: float, n_theta: int = 24, n_phi: int = 18) -> PanelSet:
    """Prolate spheroid (semi-axes ``a`` along x, ``b = c``) as quad panels.

    Latitude/longitude topology identical to
    :func:`tensorfvm.twin_hull_geometry.sphere_panels`, so ``a = b`` reproduces
    the sphere panelisation.  Normals are oriented outward (into the fluid) by
    comparing them with the outward gradient of ``(x/a)^2 + (y/b)^2 + (z/b)^2``.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("spheroid semi-axes must be positive")
    th = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    ph = np.linspace(0.0, np.pi, n_phi + 1)
    nodes = np.empty((n_theta + 1, n_phi + 1, 3))
    for i, t in enumerate(th):
        for j, p in enumerate(ph):
            nodes[i, j] = (a * np.cos(p), b * np.sin(p) * np.cos(t), b * np.sin(p) * np.sin(t))
    panels = panelise_nodes(nodes, label="spheroid")
    out_grad = np.stack(
        [panels.centroids[:, 0] / a**2, panels.centroids[:, 1] / b**2, panels.centroids[:, 2] / b**2],
        axis=1,
    )
    flip = np.einsum("ij,ij->i", panels.normals, out_grad) < 0.0
    panels.normals[flip] *= -1.0
    return panels


# ---------------------------------------------------------------------------
# single-layer potential
# ---------------------------------------------------------------------------
def source_potential(points: np.ndarray, sigma: np.ndarray, panels: PanelSet, *,
                     order: int = 3, images=IMAGE_SPECS["body"],
                     chunk: int = 256) -> np.ndarray:
    """Potential of the constant-strength source sheet at ``points``.

    ``phi(x) = sum_j sigma_j/(4 pi) * integral_{panel j} dS / |x - r'|`` evaluated
    with the bilinear Gauss rule of :func:`panel_potential.panel_quadrature`.
    The single-layer potential is continuous across the sheet, so panel
    centroids are legitimate field points; the ``1/r`` integrand is integrable
    over the flat self-panel, and quadrature nodes that would coincide with the
    field point are dropped (the same near-field cutoff used for the velocity
    kernel).

    ``sigma`` may be a single panel vector or a ``(k, n_panels)`` stack, in which
    case ``(k, len(points))`` is returned.  The stacked form exists because the
    :math:`1/|x - r'|` kernel does not depend on ``sigma``: the added-mass study
    needs the potential of three (six) Neumann solutions at the same nodes, and
    building the kernel once instead of per direction is what makes the
    consistent Gauss quadrature affordable.
    """
    points = np.ascontiguousarray(points, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    single = sig.ndim == 1
    sig = np.atleast_2d(sig)
    q, w = panel_quadrature(panels, order)
    nq, npan, _ = q.shape
    qf = q.reshape(-1, 3)
    sw = sig[:, None, :] * w.reshape(1, nq, npan)          # (k, nq, npan)
    out = np.zeros((sig.shape[0], points.shape[0]))
    for spec in images:
        qq = image_points(qf, spec)
        for start in range(0, points.shape[0], chunk):
            stop = min(start + chunk, points.shape[0])
            d = (points[start:stop, None, :] - qq[None, :, :]).reshape(stop - start, nq, npan, 3)
            r2 = np.einsum("iqjk,iqjk->iqj", d, d)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = 1.0 / (4.0 * np.pi * np.sqrt(r2))
            g[~np.isfinite(g) | (r2 < 1.0e-20)] = 0.0
            out[:, start:stop] += np.einsum("kqj,iqj->ki", sw, g)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# added mass
# ---------------------------------------------------------------------------
def surface_quadrature(panels: PanelSet, *, order: int = 3, quadrature: str = "centroid"):
    """Nodes, weights and six generators for the added-mass surface integral.

    The Neumann problem is discretised with the bilinear Gauss rule of
    :func:`panel_potential.panel_quadrature` (``order x order`` nodes per panel);
    the added-mass integral must use the *same* rule, otherwise the surface
    integral is only first-order accurate while the solver already represents the
    double layer to higher order.  Two rules are available:

    ``"centroid"``
        One node per panel: ``(centroid, flat area, generators at the centroid)``
        -- the historical midpoint rule, kept as the default so the numbers
        published in the existing reports stay bit-reproducible.
    ``"gauss"``
        The solver's own ``order x order`` Gauss nodes with their area weights and
        the generators evaluated at the nodes.  For a translation the generator
        ``n_a`` is constant over the flat panel, for a rotation
        ``psi_{3+a} = (r x n)_a`` varies linearly, so the Gauss rule integrates
        both exactly up to the panel-representation (geometry) error.  This is the
        rule used by the added-mass verification study; it removes the
        *quadrature* part of the error and typically halves to thirds the
        panel-count error at fixed panelisation.

    Returns ``(nodes, weights, psi)`` with ``nodes``/``psi`` shaped ``(n, 3)`` and
    ``(n, 6)``: ``psi[:, 0:3] = n`` (translation generators) and
    ``psi[:, 3:6] = r x n`` (rotation generators), i.e. the values of
    :func:`surface_generators` at the quadrature nodes.
    """
    if quadrature == "centroid":
        nodes = panels.centroids
        weights = panels.areas
        psi = surface_generators(panels)
        return nodes, weights, psi
    if quadrature != "gauss":
        raise ValueError("quadrature must be 'centroid' or 'gauss'")
    q, w = panel_quadrature(panels, order)
    nq, npan, _ = q.shape
    nodes = np.ascontiguousarray(q.reshape(-1, 3))
    weights = np.asarray(w, dtype=float).reshape(-1)
    normals = np.broadcast_to(panels.normals[None, :, :], (nq, npan, 3)).reshape(-1, 3)
    psi = np.concatenate([normals, np.cross(nodes, normals)], axis=1)
    return nodes, weights, psi


def added_mass_matrix(panels: PanelSet, *, rho: float = 1.0, order: int = 3,
                      directions: tuple[int, ...] = (0, 1, 2),
                      images=IMAGE_SPECS["body"], quadrature: str = "centroid") -> np.ndarray:
    """Translational added-mass matrix ``M_ij = -rho * closed_integral phi_i n_j dS``.

    Row/column ``i`` are set only for the requested ``directions``; the other
    entries stay zero.  ``phi_i`` solves (2) via
    :func:`panel_potential.solve_sources` with ``U = e_i``, and the surface
    integral is the panel-sum ``sum_n phi_i(x_n) n_j(x_n) w_n`` over the nodes of
    :func:`surface_quadrature` (midpoint rule by default, the solver's own Gauss
    rule with ``quadrature="gauss"``).
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    n = panels.n_panels
    out = np.zeros((3, 3))
    nodes, weights, psi = surface_quadrature(panels, order=order, quadrature=quadrature)
    sigmas = []
    for i in directions:
        U = np.zeros(3)
        U[i] = 1.0
        sigmas.append(solve_sources(panels, U, order=order, images=images))
    phi = source_potential(nodes, np.array(sigmas), panels, order=order, images=images)
    # n points into the fluid (outward from the body) -> minus sign of (1)
    for row, i in enumerate(directions):
        out[i, :] = -rho * np.sum((phi[row] * weights)[:, None] * psi[:, :3], axis=0)
    if len(directions) == 3:
        # the discrete problem is symmetric in the ideal-fluid sense; average out
        # the quadrature asymmetry so the returned matrix is exactly symmetric
        out = 0.5 * (out + out.T)
    if n == 0:  # pragma: no cover - defensive
        raise ValueError("empty panel set")
    return out


def sphere_added_mass(radius: float = 1.0, rho: float = 1.0) -> float:
    """Exact added mass of a sphere, ``M = 1/2 rho (4/3 pi a^3)`` (Cm = 0.5)."""
    if radius <= 0.0 or rho <= 0.0:
        raise ValueError("radius and rho must be positive")
    return 0.5 * rho * (4.0 / 3.0 * np.pi * radius**3)


# ---------------------------------------------------------------------------
# exact Lamb factors for spheroids / ellipsoids
# ---------------------------------------------------------------------------
def lamb_integral_factors(a: float, b: float, c: float, n: int = 400) -> tuple[float, float, float]:
    """Lamb ``alpha0, beta0, gamma0`` from the defining integral (equation 3).

    The integral over ``lambda in [0, inf)`` is mapped with a **logarithmic**
    substitution,

        lambda = lmin**2 * exp(v),   dlambda = lambda dv,   lmin = min(a, b, c),

    so the rule resolves every decade of ``lambda`` equally.  This matters when
    the axes are far apart: the integrand of the flattened/slender expect is
    concentrated at ``lambda ~ (minor axis)**2``, and the previous
    ``lambda = u/(1-u)`` mapping -- which clusters its nodes towards
    ``lambda -> inf`` -- returned ``alpha0+beta0+gamma0 = 0.53`` instead of ``2``
    for ``c/a = 1e-3``, silently corrupting the flat-body anchor.  With the log
    mapping the sum rule holds to round-off for ``c/a`` down to ``1e-6`` and the
    route agrees with the closed forms of :func:`lamb_ellipsoid_factors` to
    ``~1e-12`` for every spheroid (prolate and oblate).  The truncated tails
    (``lambda -> 0``, ``lambda -> inf``) contribute below ``1e-15``.

    Independent of :func:`lamb_ellipsoid_factors`, so a typo in either route is
    caught by the tests.
    """
    axes = np.array([a, b, c], dtype=float)
    if np.any(axes <= 0.0):
        raise ValueError("ellipsoid semi-axes must be positive")
    x, w = _gauss_legendre(n)
    v_min, v_max = -45.0, 25.0
    v = 0.5 * (x + 1.0) * (v_max - v_min) + v_min       # [-1, 1] -> [v_min, v_max]
    w = 0.5 * w * (v_max - v_min)
    lam = float(axes.min()) ** 2 * np.exp(v)            # dlambda = lambda dv
    delta = np.sqrt(np.prod([axes[k] ** 2 + lam for k in range(3)], axis=0))
    out = []
    for axis in range(3):
        out.append(float(np.sum(w * a * b * c / ((axes[axis] ** 2 + lam) * delta) * lam)))
    return (out[0], out[1], out[2])


def lamb_ellipsoid_factors(a: float, b: float, c: float) -> tuple[float, float, float]:
    """``alpha0, beta0, gamma0`` from the closed forms; integral route otherwise.

    * ``a == b == c``: sphere, ``(2/3, 2/3, 2/3)`` (hence ``k = 1/2`` on every axis);
    * ``a == b > c``: **oblate** spheroid with flattening
      ``e = sqrt(1 - c^2/a^2)``, closed form

          gamma0 = (2/e^2) * (1 - (sqrt(1 - e^2)/e) * arcsin(e)),
          alpha0 = beta0 = (2 - gamma0)/2,                                 (5)

      which gives ``gamma0 -> 2`` (``k_3 -> inf`` at fixed displaced volume) for
      a flat circular disc ``e -> 1`` -- the disc's finite added mass
      ``M_33 -> (8/3) rho a^3`` is the invariant form of that limit, see
      ``benchmarks/added_mass_lamb``;
    * ``b == c < a``: prolate spheroid, closed form (4);
    * otherwise (three unequal axes): :func:`lamb_integral_factors`.

    The oblate branch was added because without it the flattened family fell
    through to the integral route, i.e. the "closed form vs integral" cross-check
    of ``tests/test_added_mass_potential.py`` had no oblate instance at all and
    the flat-disc limit was only reachable numerically.
    """
    axes = np.array([a, b, c], dtype=float)
    if np.any(axes <= 0.0):
        raise ValueError("ellipsoid semi-axes must be positive")
    if np.isclose(a, b) and np.isclose(b, c):
        third = 2.0 / 3.0
        return (third, third, third)
    if np.isclose(a, b) and c < a:               # oblate spheroid, minor axis c
        e = np.sqrt(1.0 - (c / a) ** 2)
        if e == 0.0:                             # pragma: no cover - guarded above
            third = 2.0 / 3.0
            return (third, third, third)
        gamma0 = (2.0 / e**2) * (1.0 - (np.sqrt(1.0 - e**2) / e) * np.arcsin(e))
        alpha0 = 0.5 * (2.0 - gamma0)
        return (float(alpha0), float(alpha0), float(gamma0))
    if np.isclose(b, c) and a > b:               # prolate spheroid, major axis a
        e = np.sqrt(1.0 - (b / a) ** 2)
        if b / a > 1.0 - 1e-12:
            third = 2.0 / 3.0
            return (third, third, third)
        alpha0 = ((1.0 - e**2) / e**3) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
        beta0 = 0.5 * (2.0 - alpha0)
        return (float(alpha0), float(beta0), float(beta0))
    return lamb_integral_factors(a, b, c)


def lamb_spheroid_factors(a: float, b: float) -> tuple[float, float]:
    """Added-mass coefficients ``(k_axial, k_transverse)`` of a prolate spheroid.

    ``M_axial = k_axial rho V``, ``M_transverse = k_transverse rho V`` with
    ``V = 4/3 pi a b^2``.  Limits: sphere ``(0.5, 0.5)``, slender
    ``(0, 1)``.
    """
    alpha0, beta0, _ = lamb_ellipsoid_factors(a, b, b)
    k_axial = alpha0 / (2.0 - alpha0)
    k_transverse = beta0 / (2.0 - beta0)
    return (float(k_axial), float(k_transverse))

# ---------------------------------------------------------------------------
# six-degree-of-freedom (translational + rotational) added mass
# ---------------------------------------------------------------------------
def surface_generators(panels: PanelSet) -> np.ndarray:
    """The six surface velocity generators ``psi_a`` (``N x 6``).

    For a rigid body motion with translational velocity ``U`` and angular
    velocity ``Omega`` about the coordinate origin the surface velocity is
    ``U + Omega x r``, so the Neumann data of the perturbation potential is
    ``dphi/dn = (U + Omega x r) . n = U_i psi_i + Omega_i psi_{3+i}`` with

        psi_a = n_a                  (a = 0, 1, 2),
        psi_{3+a} = (r x n)_a        (a = 0, 1, 2),

    using ``(e_a x r) . n = e_a . (r x n)``.  The same six vectors are the
    integrands of the generalised added-mass matrix, so a single evaluation
    gives the whole ``6 x 6`` block (see :func:`added_mass_matrix_6dof`).

    Structural property (used as an exact verification anchor): for every body
    of revolution whose panel set is mirror-symmetric about each meridian
    plane, and in particular for the roll generator of a body of revolution
    about its symmetry axis, ``psi_3`` vanishes *identically* on the surface
    (``r`` and ``n`` are coplanar with the axis), so the roll added inertia is
    exactly zero in an unbounded ideal fluid.
    """
    normals = panels.normals
    rot = np.cross(panels.centroids, normals)
    return np.concatenate([normals, rot], axis=1)


def added_mass_matrix_6dof(panels: PanelSet, *, rho: float = 1.0, order: int = 3,
                           directions: tuple[int, ...] | None = None,
                           images=IMAGE_SPECS["body"],
                           symmetrize: bool = True,
                           quadrature: str = "centroid") -> np.ndarray:
    """Full ``6 x 6`` added-mass matrix about the coordinate origin.

    Generalises :func:`added_mass_matrix` from the translational block to the
    complete Kirchhoff matrix.  With the six generators ``psi_a`` of
    :func:`surface_generators` and ``phi_a`` the perturbation potential driven by

        laplace(phi_a) = 0,   dphi_a/dn = -psi_a on S,   phi_a -> 0 at infinity,

    the entries are

        A_ab = -rho * contour_integral_S phi_a psi_b dS,     a, b = 0..5,

    which is the standard added-mass definition (Lamb 1932, Hydrodynamics,
    §108-114; Landau & Lifshitz, Fluid Mechanics, §11; Newman 1977, Marine
    Hydrodynamics, §4.4) with the same orientation convention as (1) above.
    Rows/columns ``0..2`` are the translational block, ``3..5`` the rotational
    block (moments about the origin) and the off-diagonal blocks the
    translation-rotation coupling.

    All six Neumann problems share one influence matrix, so they are solved
    with a single LU factorisation.  The discrete matrix is symmetric in the
    ideal-fluid sense; the residual asymmetry is the quadrature difference
    between row ``a`` and column ``a`` and is averaged out unless
    ``symmetrize=False`` (used by the verification to *measure* that residual).
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if panels.n_panels == 0:  # pragma: no cover - defensive
        raise ValueError("empty panel set")
    dirs = tuple(range(6)) if directions is None else tuple(int(d) for d in directions)
    if any(d < 0 or d > 5 for d in dirs):
        raise ValueError("directions must lie in 0..5")
    nodes, weights, psi_all = surface_quadrature(panels, order=order, quadrature=quadrature)
    psi = psi_all[:, dirs]
    # The Neumann problem is solved on the flat panels, so its data is the
    # panel-centroid generator (constant strength per panel); only the *surface
    # integral* is refined by ``quadrature``.
    solve_psi = surface_generators(panels)
    sigmas = np.array([
        solve_sources(panels, np.zeros(3), order=order, images=images,
                      normal_velocity=solve_psi[:, a])
        for a in dirs
    ])
    phi = source_potential(nodes, sigmas, panels, order=order, images=images)
    out = -rho * ((phi * weights[None, :]) @ psi)
    if symmetrize and len(dirs) > 1:
        out = 0.5 * (out + out.T)
    return out


def rotational_added_inertia(panels: PanelSet, *, rho: float = 1.0, order: int = 3,
                             images=IMAGE_SPECS["body"],
                             quadrature: str = "centroid") -> np.ndarray:
    """Rotational block of :func:`added_mass_matrix_6dof` (moments about the origin).

    This is the ``J`` term :mod:`tensorfvm.maneuvering.dynamics` needs for the
    yaw/roll equations.  For a body of revolution the diagonal entry about its
    symmetry axis is zero in an unbounded ideal fluid (no wetted-surface
    ``(r x n)`` component along the axis), which is why the roll equation is
    dominated by viscous/vortex effects rather than by potential flow.
    """
    full = added_mass_matrix_6dof(panels, rho=rho, order=order, images=images,
                                  directions=(3, 4, 5), quadrature=quadrature)
    return full


def sphere_rotational_added_inertia(radius: float = 1.0, rho: float = 1.0) -> float:
    """Exact added moment of inertia of a **sphere about a diameter: zero**.

    On a sphere ``r x n = 0`` identically, so the rotational Neumann data
    ``(Omega x r) . n`` vanishes; with ``phi -> 0`` at infinity the exterior
    problem has only the trivial solution ``phi = 0``, the fluid stays at rest
    and the pressure perturbation (``p = -rho dphi/dt``) vanishes.  Hence
    ``I_44 = I_55 = I_66 = 0`` exactly for a sphere in an unbounded ideal fluid
    (the classical result: Fossen, Handbook of Marine Craft Hydrodynamics,
    the sphere's added-mass matrix is ``diag(m_a, m_a, m_a, 0, 0, 0)``).

    Note the corollary that matters for the debt ledger: any quoted non-zero
    "sphere added inertia" (e.g. ``(5/8) pi rho a^5``) is not the ideal-fluid
    added inertia; the only non-zero rotational anchor of a sphere is the
    *parallel-axis* value about an axis offset by ``d`` (see
    :func:`sphere_offset_axis_added_inertia`).
    """
    if radius <= 0.0 or rho <= 0.0:
        raise ValueError("radius and rho must be positive")
    return 0.0


def sphere_offset_axis_added_inertia(radius: float, distance: float,
                                     rho: float = 1.0) -> float:
    """Exact added inertia of a sphere about an axis a distance ``d`` from its centre.

    ``radius`` and ``distance`` are **mandatory** (debt ledger D8, m16122): the
    historical default ``distance = 1.0`` let a caller silently compare a panel
    value at ``d = 1.5`` against the ``d = 1.0`` anchor (``m_a`` vs ``m_a d^2``),
    which is the phantom inconsistency flagged in
    ``docs/maneuvering-added-inertia/REPORT.md`` (m14901).

    Rotation with angular velocity ``Omega`` about an axis through the point
    ``r0`` is ``Omega x (r - r0) = Omega x r - Omega x r0``: the rigid rotation
    of the sphere (zero added inertia, previous function) superposed on the
    uniform translation ``U = -Omega x r0`` with ``|U| = Omega d``.  The fluid
    kinetic energy is therefore ``1/2 m_a |U|^2`` and

        I_oo = m_a d^2,   m_a = 1/2 rho (4/3 pi a^3).

    This is the non-trivial rotational anchor used by the verification: it
    exercises the moment arm ``r`` and the sign of the pressure-moment
    integration with a value that is zero only in the degenerate ``d = 0`` case.
    """
    if radius <= 0.0 or rho <= 0.0:
        raise ValueError("radius and rho must be positive")
    if distance < 0.0:
        raise ValueError("distance must be non-negative")
    return sphere_added_mass(radius, rho) * distance**2


def sphere_panels_offset(radius: float = 1.0, centre=(0.0, 0.0, 0.0),
                         n_theta: int = 32, n_phi: int = 18) -> PanelSet:
    """Sphere panelisation with its centre at ``centre``.

    Needed for the parallel-axis anchor of
    :func:`sphere_offset_axis_added_inertia`: the added-mass matrix is taken
    about the coordinate origin, so placing the sphere off the origin turns the
    exact rotational anchor into the non-zero ``m_a d^2``.
    """
    from .twin_hull_geometry import sphere_panels as _sphere_panels

    panels = _sphere_panels(radius, n_theta, n_phi)
    shift = np.asarray(centre, dtype=float)
    if not np.any(shift):
        return panels
    corners = None if panels.corners is None else panels.corners + shift
    return PanelSet(panels.centroids + shift, panels.normals.copy(),
                    panels.areas.copy(), corners, panels.label)
