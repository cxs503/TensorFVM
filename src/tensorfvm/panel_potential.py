"""Constant-strength source panel method (double-body potential flow).

Used by the twin-hull resistance analysis to produce the inviscid outer flow
whose surface speed feeds the integral boundary layer, and whose (displacement
body) pressure integral gives the viscous pressure resistance.

Formulation
-----------
A single-layer source distribution of constant strength sigma_j on every quad
panel satisfies the Neumann condition ``V . n = 0`` on the body.  The velocity
kernel is the Rankine source ``(r_i - r') / (4 pi |r_i - r'|^3)`` and the panel
integrals are evaluated with bilinear Gauss quadrature (2x2 or 3x3 per panel),
so the discretisation can be verified by mesh and quadrature refinement.

Mirror images supply the planes of symmetry:

* ``body``        - closed body in an unbounded fluid (verification case);
* ``single_hull`` - hull + waterplane image (double body at Fn -> 0);
* ``twin_hull``   - hull + waterplane image + the partner demihull and its
  waterplane image (the catamaran flow is symmetric about the centre plane, so
  one demihull is the unknown and the partner enters through the kernel).

The self-panel principal value of the normal velocity is zero for a flat panel
(the quadrature points are coplanar with the centroid, so the normal projection
vanishes); the ``+1/2`` jump of a source sheet is added to the matrix diagonal.

Verification
------------
``tests/test_twin_hull_form_factor.py`` checks the sphere against the analytic
``Cp = 1 - 9/4 sin^2(theta)`` solution, including quadrature- and
mesh-refinement of the error.
"""
from __future__ import annotations

import numpy as np

from .twin_hull_geometry import PanelSet

IMAGE_SPECS = {
    "body": ((1.0, 1.0),),
    "single_hull": ((1.0, 1.0), (1.0, -1.0)),
    "twin_hull": ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)),
}

# squared-distance cutoff of the Rankine kernel (m^2): removes self-panel
# quadrature nodes that land on the field point after round-off
_NEAR_FIELD_CUTOFF = 1.0e-20

_GAUSS_X = {
    1: (0.0,),
    2: (-0.5773502691896257, 0.5773502691896257),
    3: (-0.7745966692414834, 0.0, 0.7745966692414834),
}
_GAUSS_W = {
    1: (2.0,),
    2: (1.0, 1.0),
    3: (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0),
}


def panel_quadrature(panels: PanelSet, order: int = 2):
    """Bilinear Gauss points and weights: ``(nq, N, 3)`` points, ``(nq, N)`` weights."""
    if panels.corners is None:
        raise ValueError("panel set needs corners for quadrature")
    if order < 2:
        raise ValueError("quadrature order must be >= 2: the single-point rule puts "
                         "its node on the panel centroid, where the self-panel and "
                         "waterline-image kernels are singular")
    gx, gw = _GAUSS_X[order], _GAUSS_W[order]
    p = panels.corners  # (N, 4, 3): p00, p10, p11, p01
    pts, wts = [], []
    for a, wa in zip(gx, gw):
        for b, wb in zip(gx, gw):
            w = 0.25 * (1.0 + a) * (1.0 + b)
            wp = 0.25 * (1.0 + a) * (1.0 - b)
            wm = 0.25 * (1.0 - a) * (1.0 - b)
            wq = 0.25 * (1.0 - a) * (1.0 + b)
            x = w * p[:, 1] + wp * p[:, 2] + wm * p[:, 3] + wq * p[:, 0]
            du = 0.25 * ((1 + b) * (p[:, 1] - p[:, 0]) + (1 - b) * (p[:, 2] - p[:, 3]))
            dv = 0.25 * ((1 + a) * (p[:, 2] - p[:, 1]) + (1 - a) * (p[:, 3] - p[:, 0]))
            pts.append(x)
            wts.append(wa * wb * np.linalg.norm(np.cross(du, dv), axis=1))
    return np.stack(pts, axis=0), np.stack(wts, axis=0)


def image_points(points: np.ndarray, spec) -> np.ndarray:
    """Reflect points about the centre plane (sy) and/or the waterline (sz)."""
    sy, sz = spec
    out = np.array(points, dtype=float, copy=True)
    out[..., 1] *= sy
    out[..., 2] *= sz
    return out


def influence_matrix(points: np.ndarray, normals: np.ndarray, panels: PanelSet, *,
                     order: int = 2, images=IMAGE_SPECS["body"],
                     chunk: int = 256) -> np.ndarray:
    """Normal-velocity influence matrix ``A[i, j]`` (field point i, source panel j)."""
    q, w = panel_quadrature(panels, order)
    nq, npan, _ = q.shape
    qf = q.reshape(-1, 3)
    n_pts = points.shape[0]
    A = np.zeros((n_pts, npan))
    for spec in images:
        qq = image_points(qf, spec)
        self_image = spec == (1.0, 1.0)
        for start in range(0, n_pts, chunk):
            stop = min(start + chunk, n_pts)
            m = stop - start
            d = (points[start:stop, None, :] - qq[None, :, :]).reshape(m, nq, npan, 3)
            r2 = np.einsum("iqjk,iqjk->iqj", d, d)
            r = np.sqrt(r2)
            with np.errstate(divide="ignore", invalid="ignore"):
                k = 1.0 / (4.0 * np.pi * r * r2)
            # a quadrature node can land on the field point (the centre node of a
            # 3-point rule).  Exact zeros are non-finite, but round-off makes the
            # distance ~1e-17 instead of 0, which would blow the kernel up; the
            # near-field cutoff removes both cases.
            k[~np.isfinite(k) | (r2 < _NEAR_FIELD_CUTOFF)] = 0.0
            dn = np.einsum("ik,iqjk->iqj", normals[start:stop], d)
            Amat = np.einsum("iqj,iqj->ij", k * w.reshape(1, nq, npan), dn)
            if self_image:
                idx = np.arange(stop - start)
                Amat[idx, start + idx] = 0.0  # flat-panel principal value
            A[start:stop] += Amat
    return A


def subsect(panels: PanelSet, mask: np.ndarray) -> PanelSet:
    """Panel subset (used to drop degenerate tip/pole panels from the unknowns)."""
    corners = None if panels.corners is None else panels.corners[mask]
    return PanelSet(panels.centroids[mask], panels.normals[mask], panels.areas[mask], corners,
                    panels.label)


def solve_sources(panels: PanelSet, U: np.ndarray, *, order: int = 2,
                  images=IMAGE_SPECS["body"], active: np.ndarray | None = None,
                  normal_velocity: np.ndarray | None = None) -> np.ndarray:
    """Source strengths satisfying ``V . n = 0`` on every (active) panel.

    ``U`` is the uniform onset flow (the classical use).  For a non-uniform
    onset normal velocity -- e.g. the rotational surface velocity
    ``(Omega x r) . n`` of :func:`tensorfvm.panel_added_mass.added_mass_matrix_6dof`
    -- pass the per-panel value ``psi`` through ``normal_velocity``; ``U`` is
    then ignored and the solved perturbation satisfies ``dphi/dn = -psi``.

    Degenerate panels (zero-area tips, mesh poles) carry no reliable normal, so
    they are excluded from the unknowns; their source strength is returned as
    zero and they contribute nothing to the field because their area vanishes.
    """
    n_all = panels.n_panels
    mask = panels.active_mask() if active is None else np.asarray(active, bool)
    sub = subsect(panels, mask)
    A = influence_matrix(sub.centroids, sub.normals, sub, order=order, images=images)
    A[np.diag_indices(sub.n_panels)] += 0.5
    if normal_velocity is None:
        U = np.broadcast_to(np.asarray(U, float), (n_all, 3))
        rhs = -np.einsum("ij,ij->i", U[mask], sub.normals)
    else:
        rhs = -np.asarray(normal_velocity, float).reshape(n_all)[mask]
    sigma = np.zeros(n_all)
    sigma[mask] = np.linalg.solve(A, rhs)
    return sigma


def normal_velocity_residual(panels: PanelSet, sigma: np.ndarray, U: np.ndarray, *,
                             order: int = 2, images=IMAGE_SPECS["body"],
                             active: np.ndarray | None = None) -> float:
    """max |V . n| on the active panels (nondimensionalised by |U|)."""
    mask = panels.active_mask() if active is None else np.asarray(active, bool)
    Uv = np.asarray(U, dtype=float)
    u0 = float(np.linalg.norm(Uv))
    U = np.broadcast_to(Uv, (panels.n_panels, 3))
    v = U[mask] + induced_velocity(panels.centroids[mask], sigma, panels, order=order,
                                   images=images)
    # the field point sits on the sheet: add the +sigma/2 jump of the exterior limit
    vn = np.einsum("ij,ij->i", v, panels.normals[mask]) + 0.5 * sigma[mask]
    return float(np.abs(vn).max() / u0)


def induced_velocity(points: np.ndarray, sigma: np.ndarray, panels: PanelSet, *,
                     order: int = 2, images=IMAGE_SPECS["body"], chunk: int = 512) -> np.ndarray:
    """Perturbation velocity at ``points`` due to the source sheet."""
    q, w = panel_quadrature(panels, order)
    nq, npan, _ = q.shape
    qf = q.reshape(-1, 3)
    sw = sigma[None, :] * w.reshape(nq, npan)
    out = np.zeros_like(points, dtype=float)
    for spec in images:
        qq = image_points(qf, spec)
        for start in range(0, points.shape[0], chunk):
            stop = min(start + chunk, points.shape[0])
            m = stop - start
            d = (points[start:stop, None, :] - qq[None, :, :]).reshape(m, nq, npan, 3)
            r2 = np.einsum("iqjk,iqjk->iqj", d, d)
            r = np.sqrt(r2)
            with np.errstate(divide="ignore", invalid="ignore"):
                k = sw.reshape(1, nq, npan) / (4.0 * np.pi * r * r2)
            k[~np.isfinite(k) | (r2 < _NEAR_FIELD_CUTOFF)] = 0.0
            out[start:stop] += np.einsum("iqj,iqjk->ik", k, d)
    return out


def surface_velocity(points: np.ndarray, normals: np.ndarray, sigma: np.ndarray,
                     panels: PanelSet, U: np.ndarray, *, order: int = 2,
                     images=IMAGE_SPECS["body"]):
    """Tangential surface speed and pressure coefficient at ``points``."""
    Uv = np.asarray(U, dtype=float)
    u0 = float(np.linalg.norm(Uv))
    v = np.broadcast_to(Uv, points.shape).astype(float) + induced_velocity(
        points, sigma, panels, order=order, images=images)
    un = np.einsum("ij,ij->i", v, normals)
    vt = v - un[:, None] * normals
    speed = np.linalg.norm(vt, axis=1)
    cp = 1.0 - (speed / u0) ** 2
    return speed, cp


def pressure_drag(panels: PanelSet, cp: np.ndarray, q: float, direction=None) -> float:
    """Resistance from the pressure field: ``-q sum Cp (n . u_hat) A``.

    ``direction`` is the free-stream direction (unit vector); the returned value
    is positive when the pressure force opposes it.
    """
    u = np.array([1.0, 0.0, 0.0]) if direction is None else np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    return float(-q * np.sum(cp * np.einsum("ij,j->i", panels.normals, u) * panels.areas))


def sphere_cp_analytic(panels: PanelSet, U) -> np.ndarray:
    """Analytic ``Cp = 1 - 9/4 sin^2 theta`` for a sphere in uniform flow."""
    c = panels.centroids
    r = np.linalg.norm(c, axis=1)
    u = np.asarray(U, dtype=float)
    if u.ndim == 0:
        u = np.array([0.0, 0.0, float(u)])
    u = u / np.linalg.norm(u)
    cos_t = np.einsum("ij,j->i", c, u) / r
    sin2 = np.clip(1.0 - cos_t ** 2, 0.0, 1.0)
    return 1.0 - 2.25 * sin2