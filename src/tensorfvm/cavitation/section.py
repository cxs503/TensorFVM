"""Section geometry and a lifting panel solver for 2-D rudder sections.

Why this module exists
----------------------
Cavitation inception on a rudder section is decided by the *minimum* pressure on
the surface: the classical minimum-pressure criterion states that a vapour
cavity appears where ``p = p_v``, i.e. where

    Cp = (p - p_inf) / (1/2 rho V^2)

reaches ``Cp_min``.  So the inception cavitation number is

    sigma_i = -Cp_min(alpha, section).                                  (1)

Equation (1) is an identity of Bernoulli's equation (steady, inviscid, along a
streamline) and it is the criterion the ITTC procedures use for model-scale
inception tests; everything that remains is producing a trustworthy ``Cp_min``.
This module therefore solves the 2-D potential flow around the closed section
contour with a classical **constant-source / constant-vortex panel method**
(Katz & Plotkin, *Low-Speed Aerodynamics*, panel method II): ``N`` source
strengths plus one vortex strength, ``N`` flow-tangency equations and a single
Kutta condition ``V_t(panel_0) + V_t(panel_{N-1}) = 0`` at the trailing edge.

Verification anchors (see ``tests/test_cavitation_rudder.py``)
-------------------------------------------------------------
* ``alpha = 0`` symmetry: the upper/lower ``Cp`` are identical to round-off.
* Thin-section lift slope: ``Cl_alpha -> 2 pi (1 + 0.77 t/c)`` (Abbott &
  von Doenhoff thickness correction).
* Exact conformal-mapping solution (:mod:`.conformal`): ``Cp`` and ``Cl`` of a
  12 %-thick Joukowski section at incidence, to a few 1e-3.
* Refinement of ``Cp_min`` under panel doubling.
* Independent formulation: the repository's source-only double-body panel
  method (:mod:`tensorfvm.panel_potential`) reproduces ``Cp_min`` at ``alpha=0``
  where the flow is non-lifting and both methods are valid.

The solver is inviscid, so it says nothing about separation; the reported
inception numbers are therefore limited to attached-flow rudder angles
(``|alpha| <~ 12 deg``), which is stated as a limitation in the node report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "PanelSolution", "naca4_thickness", "naca4_camber", "naca4_surface",
    "naca4_contour", "panel_solve", "cp_min", "lift_coefficient",
]


# --------------------------------------------------------------------------- #
# NACA 4-digit geometry (Abbott & von Doenhoff, eq. 6-2 / 6-3)
# --------------------------------------------------------------------------- #
def naca4_thickness(x, t: float = 0.12, closed_te: bool = True) -> np.ndarray:
    """Half-thickness distribution ``y_t(x)`` of a NACA 4-digit section.

    ``closed_te=True`` uses the ``-0.1036`` last coefficient so that
    ``y_t(1) = 0`` exactly (the widely quoted ``-0.1015`` leaves a ``0.0021 t``
    gap at the trailing edge, which the panel method would have to close by
    hand).
    """
    x = np.asarray(x, dtype=float)
    a4 = -0.1036 if closed_te else -0.1015
    return 5.0 * t * (0.2969 * np.sqrt(np.clip(x, 0.0, None))
                      - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)


def naca4_camber(x, m: float = 0.0, p: float = 0.4):
    """Camber line ``y_c(x)`` and its slope ``dy_c/dx`` (zero for symmetric sections)."""
    x = np.asarray(x, dtype=float)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m == 0.0:
        return yc, dyc
    if not 0.0 < p < 1.0:
        raise ValueError("camber position p must be in (0, 1)")
    fore = x < p
    xf, xa = x[fore], x[~fore]
    yc[fore] = m / p ** 2 * (2.0 * p * xf - xf ** 2)
    dyc[fore] = 2.0 * m / p ** 2 * (p - xf)
    yc[~fore] = m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * xa - xa ** 2)
    dyc[~fore] = 2.0 * m / (1.0 - p) ** 2 * (p - xa)
    return yc, dyc


def _resample_closed(points: np.ndarray, n_panel: int, cluster: float = 0.7) -> np.ndarray:
    """Resample a dense ordered closed contour to ``n_panel`` equal-ish panels.

    The nodes are placed at uniform steps in a *stretched* arc-length coordinate

        s(xi) = xi + (cluster / 2 pi) sin(2 pi xi),     xi in [0, 1],

    so that the panels are finest at the middle of the contour (``xi = 1/2``,
    i.e. at the leading edge for the contours of this module) and coarsest at
    the two ends (the trailing edge).  This avoids the nearly-degenerate
    trailing-edge panels that a plain cosine spacing produces on a cusped
    section: those panels make the source-strength solution ill-conditioned
    while barely changing the resolved surface velocity.
    """
    pts = np.asarray(points, dtype=float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    xi = np.linspace(0.0, 1.0, n_panel + 1)
    target = (xi + cluster / (2.0 * math.pi) * np.sin(2.0 * math.pi * xi)) * total
    out = np.column_stack([np.interp(target, arc, pts[:, 0]),
                           np.interp(target, arc, pts[:, 1])])
    out[0], out[-1] = pts[0], pts[0]
    return out


def naca4_surface(t: float = 0.12, m: float = 0.0, p: float = 0.4,
                  n_dense: int = 4001) -> np.ndarray:
    """Dense NACA 4-digit contour, TE -> upper -> LE -> lower -> TE (closed)."""
    beta = np.linspace(0.0, math.pi, n_dense)
    x = 0.5 * (1.0 - np.cos(beta))          # 0 (LE) -> 1 (TE)
    yt = naca4_thickness(x, t)
    yc, dyc = naca4_camber(x, m, p)
    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    # clockwise: TE -> lower -> LE -> upper -> TE, so that panels 0 and N-1 sit
    # at the trailing edge where the Kutta condition is imposed
    return np.column_stack([np.concatenate([xl[::-1], xu[1:]]),
                            np.concatenate([yl[::-1], yu[1:]])])


def naca4_contour(t: float = 0.12, m: float = 0.0, p: float = 0.4,
                  n_half: int = 200) -> np.ndarray:
    """Panel nodes of a NACA 4-digit section: ``(2 n_half + 1, 2)``, closed.

    The points are distributed by arc length with the refinement concentrated at
    the leading edge (see :func:`_resample_closed`).
    """
    return _resample_closed(naca4_surface(t=t, m=m, p=p), 2 * n_half)


# --------------------------------------------------------------------------- #
# constant-source / constant-vortex panel method
# --------------------------------------------------------------------------- #
@dataclass
class PanelSolution:
    """Result of :func:`panel_solve` (all lengths normalised by the chord)."""

    nodes: np.ndarray                  # (N+1, 2) closed contour, clockwise
    centroids: np.ndarray              # (N, 2)
    tangents: np.ndarray               # (N, 2) unit, A -> B
    normals: np.ndarray                # (N, 2) unit, outward
    lengths: np.ndarray                # (N,)
    sigma: np.ndarray                  # (N,) source strengths
    gamma: float                       # single vortex strength
    vt: np.ndarray                     # (N,) surface tangential velocity / U
    cp: np.ndarray                     # (N,) pressure coefficient
    alpha_deg: float
    tangency_residual: float           # max |V.n| / U  on the panel centroids
    extra: dict = field(default_factory=dict)

    @property
    def cp_min(self) -> float:
        return float(np.min(self.cp))

    @property
    def cp_max(self) -> float:
        return float(np.max(self.cp))

    def cl(self, chord: float = 1.0) -> float:
        """Section lift coefficient from the pressure integral."""
        return lift_coefficient(self.cp, self.normals, self.lengths, chord=chord)

    def x_cp_min(self) -> float:
        return float(self.centroids[int(np.argmin(self.cp)), 0])

    def x_cp_max(self) -> float:
        return float(self.centroids[int(np.argmax(self.cp)), 0])


def _influence(centroids: np.ndarray, nodes: np.ndarray, tangents: np.ndarray,
               normals: np.ndarray, lengths: np.ndarray):
    """Influence matrices of unit-strength source and vortex panels.

    Returns ``(St, Sn, Vt, Vn)``, each ``(N_field, N_panel)``: the tangential
    (``t``) and normal (``n``) velocity component *in the field panel's frame*
    induced at the panel centroids by a unit source sheet (``S``) or a unit
    vortex sheet (``V``) on panel ``j``.

    The kernels are the exact integrals of a constant-strength 2-D sheet over a
    straight segment (Katz & Plotkin panel method II),

        St = ln(r1^2/r2^2) / (4 pi),      Sn = -beta / (2 pi),
        Vt = +beta / (2 pi),              Vn = ln(r1^2/r2^2) / (4 pi),

    (vortex positive counter-clockwise; the individual kernels are checked
    against numerical quadrature of the sheet integrals in the test suite).

    with ``beta = theta1 - theta2`` the signed subtended angle.  The self term
    uses the exterior limit ``beta -> -pi``, ``ln(r1^2/r2^2) -> 0``, i.e.
    ``Sn = +1/2`` (source jump) and ``Vt = -1/2`` (vortex sheet seen from
    outside), which is exactly the one-sided limit needed on the body surface.
    """
    n_field, n_panel = centroids.shape[0], nodes.shape[0] - 1
    d = centroids[:, None, :] - nodes[None, :-1, :]                       # (F, P, 2)
    x0 = np.einsum("fpk,pk->fp", d, tangents)
    y0 = np.einsum("fpk,pk->fp", d, normals)
    ell = lengths[None, :]
    r1 = np.hypot(x0, y0)
    r2 = np.hypot(x0 - ell, y0)
    beta = np.arctan2(y0, x0) - np.arctan2(y0, x0 - ell)
    log_ratio = np.log(np.clip(r1, 1e-300, None) ** 2 / np.clip(r2, 1e-300, None) ** 2)
    idx = np.arange(min(n_field, n_panel))
    beta[idx, idx] = -math.pi
    log_ratio[idx, idx] = 0.0
    # frame projections: ``tjt[i, j] = t_j . n_i`` and ``njt[i, j] = n_j . t_i``
    # are NOT the same matrix (the (t, n) frame is right-handed: n = t rotated by
    # +90 deg), so they must be built separately.
    t_dot_t = np.einsum("fk,pk->fp", tangents, tangents)
    tjt = np.einsum("pk,fk->fp", tangents, normals)
    njt = np.einsum("fk,pk->fp", tangents, normals)
    n_dot_n = np.einsum("fk,pk->fp", normals, normals)
    st = log_ratio / (4.0 * math.pi)
    sn = -beta / (2.0 * math.pi)
    vt_v = beta / (2.0 * math.pi)
    vn_v = log_ratio / (4.0 * math.pi)
    St = st * t_dot_t + sn * njt
    Sn = st * tjt + sn * n_dot_n
    Vt = vt_v * t_dot_t + vn_v * njt
    Vn = vt_v * tjt + vn_v * n_dot_n
    return St, Sn, Vt, Vn


def panel_solve(nodes: np.ndarray, alpha_deg: float, u_inf: float = 1.0,
                kutta: bool = True) -> PanelSolution:
    """Solve the potential flow about a closed clockwise contour.

    The unknowns are the ``N`` panel source strengths and the single vortex
    strength; the equations are the ``N`` flow-tangency conditions on the panel
    centroids and the trailing-edge Kutta condition
    ``V_t(panel 0) + V_t(panel N-1) = 0``.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must be (N+1, 2)")
    if not np.allclose(nodes[0], nodes[-1], atol=1e-12):
        raise ValueError("contour must be closed: nodes[-1] == nodes[0]")
    tangents_raw = nodes[1:] - nodes[:-1]
    lengths = np.linalg.norm(tangents_raw, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("degenerate (zero-length) panel in contour")
    tangents = tangents_raw / lengths[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])  # outward for clockwise
    centroids = 0.5 * (nodes[1:] + nodes[:-1])
    n = lengths.size
    St, Sn, Vt, Vn = _influence(centroids, nodes, tangents, normals, lengths)

    a = math.radians(float(alpha_deg))
    u_vec = u_inf * np.array([math.cos(a), math.sin(a)])
    # flow tangency: sum_j (sigma_j * Vn[j][i]) + gamma * VT[i] = -U . n
    mat = np.zeros((n + 1, n + 1))
    mat[:n, :n] = Sn
    mat[:n, n] = Vn.sum(axis=1)
    rhs = np.zeros(n + 1)
    rhs[:n] = -np.einsum("ik,ik->i", np.broadcast_to(u_vec, centroids.shape), normals)
    # Kutta condition: tangential velocity at the first and the last panel cancels
    mat[n, :n] = St[0, :] + St[n - 1, :]
    # the vortex self term (-gamma/2 on each of the two panels) is already inside
    # the row sums of ``Vt``, so it must not be added again here
    mat[n, n] = Vt.sum(axis=1)[0] + Vt.sum(axis=1)[n - 1]
    rhs[n] = -(np.dot(u_vec, tangents[0]) + np.dot(u_vec, tangents[-1]))

    if not kutta:
        # no trailing edge: pure source (double-body) solution, gamma = 0.  This is
        # the form used for verification against the analytic sphere/circle field.
        mat = mat[:n, :n]
        rhs = rhs[:n]
        sigma, gamma = np.linalg.solve(mat, rhs), 0.0
    else:
        solution = np.linalg.solve(mat, rhs)
        sigma, gamma = solution[:n], float(solution[n])

    # surface tangential velocity and pressure coefficient
    vt_surface = tangents @ u_vec + St @ sigma + gamma * Vt.sum(axis=1)
    tangency = normals @ u_vec + Sn @ sigma + gamma * Vn.sum(axis=1)
    cp = 1.0 - (vt_surface / u_inf) ** 2
    return PanelSolution(
        nodes=nodes, centroids=centroids, tangents=tangents, normals=normals,
        lengths=lengths, sigma=sigma, gamma=gamma, vt=vt_surface, cp=cp,
        alpha_deg=float(alpha_deg),
        tangency_residual=float(np.abs(tangency).max() / u_inf),
    )


def lift_coefficient(cp: Sequence[float], normals: np.ndarray, lengths: np.ndarray,
                     chord: float = 1.0) -> float:
    """``Cl = -(1/c) * closed_integral Cp n_y ds`` (outward normal convention)."""
    return float(-np.sum(np.asarray(cp) * normals[:, 1] * lengths) / chord)


def cp_min(alpha_deg: float, t: float = 0.12, m: float = 0.0, p: float = 0.4,
           n_half: int = 200) -> float:
    """Convenience wrapper: ``Cp_min`` of a NACA 4-digit section at incidence."""
    sol = panel_solve(naca4_contour(t=t, m=m, p=p, n_half=n_half), alpha_deg)
    return sol.cp_min