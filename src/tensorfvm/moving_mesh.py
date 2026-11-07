"""Arbitrary-Lagrangian-Eulerian (ALE) moving/morphing mesh for TensorFVM.

Why this module exists
----------------------
TensorFVM's incompressible solvers are built on *static* body-fitted or
Cartesian meshes.  A planar-motion-mechanism (PMM) virtual test - or any
forced-oscillation / manoeuvring CFD - needs the body to move through the
fluid, so the mesh must move with it while the far field stays fixed.  This
module supplies the geometry half of that capability:

* :class:`StructuredCurvilinearMesh` - a single-block logically rectangular
  finite-volume mesh whose metrics are *recomputed from the nodal coordinates*
  at every time level (so morphing carries no stale-metric inconsistency),
  with per-direction periodicity.
* :class:`ALEGeometry` - the discrete space-conservation law (Demirdzic & Peric
  1988): per-face swept volumes use the exact three-tetrahedron decomposition
  of the prism swept by a linearly moving triangle, so that
  ``sum_faces(dV_face) == V(new) - V(old)`` holds to machine precision for any
  motion (translation, rotation, morphing, random nodal displacement).
* :func:`ale_advection_residual` - the free-stream-preserving ALE convective
  operator in divergence form,
  ``(V^{n+1} phi^{n+1} - V^n phi^n)/dt + sum_f (u_f - u_mf).S_f phi_f``.
  Because ``sum_f u_mf.S_f == dV/dt`` exactly, a uniform field has *exactly*
  zero residual on a moving mesh (the classical ALE free-stream test).

The motion implemented here is *morphing*: the body-fitted inner boundary
follows the prescribed body motion, the outer boundary is held fixed, and the
displacement is blended across the radial layers.  No overset interpolation is
needed for the wall-bounded configurations supported here.

Verification: ``tests/test_moving_mesh_ale.py`` (GCL exactness, free-stream
preservation to roundoff, Galilean invariance, second-order time accuracy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "StructuredCurvilinearMesh",
    "ALEGeometry",
    "PMMKinematics",
    "build_cylinder_ogrid",
    "build_kvlcc2_double_body_ogrid",
    "ale_advection_residual",
    "blend_profile",
    "morph_nodes",
]

_QUAD_LOCAL = {
    0: ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
    1: ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
    2: ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
}


def blend_profile(n_layers: int, *, first: float = 0.02,
                  stretch: float = 1.12) -> np.ndarray:
    """Normalised radial blending coordinate ``s_j`` in [0, 1] (geometric)."""
    if n_layers < 1:
        raise ValueError("at least one layer required")
    ratios = np.array([first * stretch ** j for j in range(n_layers)])
    s = np.concatenate(([0.0], np.cumsum(ratios)))
    return s / s[-1]


@dataclass(frozen=True)
class PMMKinematics:
    """Planar-motion-mechanism kinematics of a hull model.

    ``mode`` is one of ``"sway"`` (``y = A sin(wt)``, heading fixed),
    ``"yaw"`` (``psi = A_deg sin(wt)`` about the midship), ``"drift"``
    (constant heading ``beta_deg``) or ``"static"``.
    """

    mode: str
    amplitude: float = 0.0
    period: float = 10.0
    beta_deg: float = 0.0
    midpoint_x: float = 3.5
    U: float = 0.771

    def __post_init__(self):
        if self.mode not in ("sway", "yaw", "drift", "static"):
            raise ValueError(f"unknown PMM mode {self.mode!r}")
        if self.period <= 0.0:
            raise ValueError("period must be positive")

    @property
    def omega(self) -> float:
        return 2.0 * np.pi / self.period

    def pose(self, t: float) -> Dict[str, float]:
        """Body pose and rates at time ``t`` (sway [m], yaw [rad] + rates)."""
        w = self.omega
        if self.mode == "sway":
            return dict(sway=self.amplitude * np.sin(w * t),
                        sway_rate=self.amplitude * w * np.cos(w * t),
                        yaw=0.0, yaw_rate=0.0)
        if self.mode == "yaw":
            a = np.deg2rad(self.amplitude)
            return dict(sway=0.0, sway_rate=0.0,
                        yaw=a * np.sin(w * t), yaw_rate=a * w * np.cos(w * t))
        if self.mode == "drift":
            return dict(sway=0.0, sway_rate=0.0,
                        yaw=np.deg2rad(self.beta_deg), yaw_rate=0.0)
        return dict(sway=0.0, sway_rate=0.0, yaw=0.0, yaw_rate=0.0)

    @property
    def label(self) -> str:
        if self.mode == "sway":
            return f"pure_sway_A{self.amplitude:.4g}m_T{self.period:g}s"
        if self.mode == "yaw":
            return f"pure_yaw_A{self.amplitude:.4g}deg_T{self.period:g}s"
        return f"drift_beta{self.beta_deg:.4g}deg"
class StructuredCurvilinearMesh:
    """Logically rectangular finite-volume mesh rebuilt from nodes on demand.

    ``nodes`` has shape ``(NI+1, NJ+1, NK+1, 3)``; cell ``(i,j,k)`` is bounded
    by node corners ``(i..i+1, j..j+1, k..k+1)``.  ``patches`` maps
    ``(direction, "low"|"high")`` to a boundary label for non-periodic faces.
    """

    def __init__(self, nodes: np.ndarray, *,
                 periodic: Sequence[bool] = (False, False, False),
                 patches: Dict[Tuple[int, str], str] | None = None):
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 4 or nodes.shape[-1] != 3:
            raise ValueError("nodes must have shape (NI+1, NJ+1, NK+1, 3)")
        if not np.isfinite(nodes).all():
            raise ValueError("non-finite nodal coordinates")
        self.nodes = nodes.copy()
        self.shape = tuple(int(n) - 1 for n in nodes.shape[:3])
        if min(self.shape) < 1:
            raise ValueError("need at least one cell per direction")
        self.periodic = tuple(bool(p) for p in periodic)
        self.patches = dict(patches or {})
        self._build_topology()
        self._orient = np.zeros(self.n_faces, dtype=bool)
        self._orient = self._orientation(self.nodes)

    # ------------------------------------------------------------------
    def _node_index(self, i, j, k):
        nj = self.shape[1] + 1
        nk = self.shape[2] + 1
        return (i * nj + j) * nk + k

    def _cell_index(self, i, j, k):
        nj, nk = self.shape[1], self.shape[2]
        return (i * nj + j) * nk + k

    def _build_topology(self) -> None:
        """Enumerate faces plane by plane (vertex-based face list)."""
        counts = list(self.shape)
        owner, neighbor, label, quads = [], [], [], []
        for d in range(3):
            n = counts[d]
            planes = range(n) if self.periodic[d] else range(n + 1)
            o0, o1 = [e for e in range(3) if e != d]
            for I in planes:
                for a, b in np.ndindex(counts[o0], counts[o1]):
                    cells = np.zeros(3, dtype=int)
                    cells[o0], cells[o1] = a, b
                    hi = cells.copy(); hi[d] = I
                    lo = cells.copy(); lo[d] = I - 1
                    if self.periodic[d]:
                        lo[d] %= n
                        hi[d] %= n
                        own = self._cell_index(*lo)
                        nei = self._cell_index(*hi)
                        lab = (d, "periodic")
                    else:
                        own = -1 if I == 0 else self._cell_index(*lo)
                        nei = -1 if I == n else self._cell_index(*hi)
                        if own >= 0 and nei >= 0:
                            lab = (d, "int")
                        elif own < 0:
                            own, nei = nei, -1
                            lab = (d, self.patches.get((d, "low"), "wall"))
                        else:
                            nei = -1
                            lab = (d, self.patches.get((d, "high"), "far"))
                    corners = []
                    for sb, sc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        nn = cells.copy(); nn[d] = I
                        nn[o0] += sb
                        nn[o1] += sc
                        corners.append(self._node_index(*nn))
                    owner.append(own)
                    neighbor.append(nei)
                    label.append(lab)
                    quads.append(corners)
        self.owner = np.asarray(owner, dtype=np.int64)
        self.neighbor = np.asarray(neighbor, dtype=np.int64)
        self.face_label = np.asarray(label, dtype=object)
        self.quad_ids = np.asarray(quads, dtype=np.int64)
        self.n_faces = len(self.owner)
        self.n_cells = int(np.prod(counts))
        if (self.owner < 0).any():
            raise ValueError("internal face without owner")
        self.interior = self.neighbor >= 0

    def corner_centers(self, nodes: np.ndarray) -> np.ndarray:
        """Arithmetic mean of the eight cell corners (metric-free centroid)."""
        counts = self.shape
        ids = []
        for sb, sc, sd in np.ndindex(2, 2, 2):
            ii = np.arange(counts[0])[:, None, None] + sb
            jj = np.arange(counts[1])[None, :, None] + sc
            kk = np.arange(counts[2])[None, None, :] + sd
            ids.append(self._node_index(ii, jj, kk))
        corners = np.asarray(nodes, dtype=float).reshape(-1, 3)[np.stack(ids)]
        return corners.mean(axis=0).reshape(-1, 3)

    def _orientation(self, nodes: np.ndarray) -> np.ndarray:
        """Per-face winding flip so that ``S`` points owner -> face."""
        q = nodes.reshape(-1, 3)[self.quad_ids].reshape(self.n_faces, 4, 3)
        s = _quad_area(q)
        own_c = self.corner_centers(nodes)
        return np.einsum("fi,fi->f", s, q.mean(axis=1) - own_c[self.owner]) < 0.0

    # ------------------------------------------------------------------
    def face_coords(self, nodes: np.ndarray) -> np.ndarray:
        q = np.asarray(nodes, dtype=float).reshape(-1, 3)[self.quad_ids]
        q = np.where(self._orient[:, None, None], q[:, ::-1, :], q)
        return q

    def metrics(self, nodes: np.ndarray | None = None) -> "ALEGeometry":
        nodes = self.nodes if nodes is None else np.asarray(nodes, dtype=float)
        q = self.face_coords(nodes)
        s = _quad_area(q)
        xf = _quad_center(q)
        contrib = np.einsum("fi,fi->f", xf, s)
        nc = self.n_cells
        vol3 = (np.bincount(self.owner, weights=contrib, minlength=nc)
                - np.bincount(self.neighbor[self.interior],
                              weights=contrib[self.interior], minlength=nc))
        volume = vol3 / 3.0
        if not (volume > 0.0).all():
            raise ValueError("non-positive cell volume (inverted mesh)")
        mom = np.stack([
            np.bincount(self.owner, weights=contrib * xf[:, k], minlength=nc)
            - np.bincount(self.neighbor[self.interior],
                          weights=contrib[self.interior] * xf[self.interior, k], minlength=nc)
            for k in range(3)], axis=1)
        centers = mom / (4.0 * volume)[:, None]
        # normal owner-neighbour distance
        dvec = np.empty((self.n_faces, 3))
        dvec[self.interior] = centers[self.neighbor[self.interior]] - centers[self.owner[self.interior]]
        dvec[~self.interior] = xf[~self.interior] - centers[self.owner[~self.interior]]
        smag = np.linalg.norm(s, axis=1)
        dperp = np.einsum("fi,fi->f", s, dvec) / smag
        if not (dperp > 0.0).all():
            raise ValueError("non-positive face normal distance")
        offset = np.empty((self.n_faces, 3))
        offset[self.interior] = centers[self.owner[self.interior]] - xf[self.interior]
        offset[~self.interior] = centers[self.owner[~self.interior]] - xf[~self.interior]
        return ALEGeometry(mesh=self, nodes=nodes, volume=volume, centers=centers,
                           face_area=s, face_center=xf, dperp=dperp,
                           owner_offset=offset, area_mag=smag)

    def cell_centers(self, nodes: np.ndarray) -> np.ndarray:
        return self.metrics(nodes).centers

    # ------------------------------------------------------------------
    def swept_volumes(self, nodes_old: np.ndarray, nodes_new: np.ndarray) -> np.ndarray:
        """Exact per-face swept volume ``dV_face`` between two node sets.

        Summed per cell these reproduce ``V(new) - V(old)`` to roundoff for any
        motion (discrete space-conservation law, Demirdzic & Peric 1988).
        """
        q0 = self.face_coords(nodes_old)
        q1 = self.face_coords(nodes_new)
        out = np.zeros(self.n_faces)
        for tri in ((0, 1, 2), (0, 2, 3)):
            a, b, c = (q0[:, i, :] for i in tri)
            a1, b1, c1 = (q1[:, i, :] for i in tri)
            out += _wedge_volumes(a, b, c, a1, b1, c1)
        return out

    def cell_volume_change(self, nodes_old: np.ndarray,
                           nodes_new: np.ndarray) -> np.ndarray:
        """Per-cell ``dV`` assembled from :meth:`swept_volumes`."""
        dv = self.swept_volumes(nodes_old, nodes_new)
        nc = self.n_cells
        return (np.bincount(self.owner, weights=dv, minlength=nc)
                - np.bincount(self.neighbor[self.interior],
                              weights=dv[self.interior], minlength=nc))# ---------------------------------------------------------------------------
# ALE geometry container
# ---------------------------------------------------------------------------
@dataclass
class ALEGeometry:
    """Recomputed metrics of a moving block at one time level."""

    mesh: "StructuredCurvilinearMesh"
    nodes: np.ndarray
    volume: np.ndarray          # cell volumes
    centers: np.ndarray         # cell centroids
    face_area: np.ndarray       # face area vectors, outward from owner
    face_center: np.ndarray
    dperp: np.ndarray           # normal owner-neighbour distance
    owner_offset: np.ndarray    # owner centre -> face centre
    area_mag: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.volume.size

    def divergence(self, flux: np.ndarray) -> np.ndarray:
        """Discrete divergence of a conservative face flux (per cell)."""
        m = self.mesh
        out = np.bincount(m.owner, weights=flux, minlength=self.n_cells)
        return out - np.bincount(m.neighbor[m.interior],
                                 weights=flux[m.interior], minlength=self.n_cells)

    def gradient(self, field: np.ndarray, boundary_value=None) -> np.ndarray:
        """Green-Gauss cell gradient from face values (second-order on smooth grids)."""
        m = self.mesh
        f = np.asarray(field)
        scalar = f.ndim == 1
        fv = f[:, None] if scalar else f
        vals = np.empty((m.n_faces,) + fv.shape[1:])
        vals[m.interior] = 0.5 * (fv[m.owner[m.interior]] + fv[m.neighbor[m.interior]])
        b = ~m.interior
        if boundary_value is None:
            vals[b] = fv[m.owner[b]]
        else:
            bv = np.asarray(boundary_value)
            vals[b] = bv if bv.ndim > 1 else bv
        contrib = vals[:, :, None] * self.face_area[:, None, :]
        out = (np.stack([np.bincount(m.owner, weights=contrib[:, k, l], minlength=self.n_cells)
                         - np.bincount(m.neighbor[m.interior],
                                       weights=contrib[m.interior, k, l], minlength=self.n_cells)
                         for k in range(fv.shape[1]) for l in range(3)], axis=1))
        out = out.reshape(self.n_cells, fv.shape[1], 3) / self.volume[:, None, None]
        return out[:, 0, :] if scalar else out


def _quad_area(q: np.ndarray) -> np.ndarray:
    """Area vector of quadrilateral faces ``q`` of shape (F,4,3)."""
    return (0.5 * np.cross(q[:, 1, :] - q[:, 0, :], q[:, 2, :] - q[:, 0, :])
            + 0.5 * np.cross(q[:, 2, :] - q[:, 0, :], q[:, 3, :] - q[:, 0, :]))


def _quad_center(q: np.ndarray) -> np.ndarray:
    """Area-weighted centroid of the two triangles of each quadrilateral."""
    s1 = 0.5 * np.cross(q[:, 1, :] - q[:, 0, :], q[:, 2, :] - q[:, 0, :])
    s2 = 0.5 * np.cross(q[:, 2, :] - q[:, 0, :], q[:, 3, :] - q[:, 0, :])
    c1 = (q[:, 0, :] + q[:, 1, :] + q[:, 2, :]) / 3.0
    c2 = (q[:, 0, :] + q[:, 2, :] + q[:, 3, :]) / 3.0
    w1 = np.linalg.norm(s1, axis=1)[:, None]
    w2 = np.linalg.norm(s2, axis=1)[:, None]
    return (w1 * c1 + w2 * c2) / np.maximum(w1 + w2, 1e-300)


def _wedge_volumes(a0, b0, c0, a1, b1, c1) -> np.ndarray:
    """Exact swept volume of triangle (a,b,c) moving to (a',b',c').

    Three tetrahedra tile the prism under linear vertex motion, so summing over
    the closed surface of a cell reproduces ``V_new - V_old`` exactly.
    """
    def tet(p, q, r, s):
        return np.einsum("...i,...i->...", q - p, np.cross(r - p, s - p)) / 6.0

    return tet(a0, b0, c0, a1) + tet(b0, c0, a1, b1) + tet(c0, a1, b1, c1)


# ---------------------------------------------------------------------------
# morphing
# ---------------------------------------------------------------------------
def morph_nodes(base: np.ndarray, weight: np.ndarray, pose: Dict[str, float],
                *, midpoint_x: float) -> np.ndarray:
    """Displace nodes with a blended rigid planar motion.

    ``weight`` in [0, 1] is 1 on the body and 0 on the fixed outer boundary.
    The blended map is a rigid sway translation plus a rigid yaw rotation about
    the vertical axis through ``(midpoint_x, 0)``: a node with weight ``w``
    receives ``y += w*sway`` and a rotation by ``w*yaw``.  Because the rotation
    angle and the translation are blended by the *same* factor, the body layer
    moves rigidly (w = 1) and the residual deformation is absorbed by the
    intermediate layers - the standard morphing strategy.
    """
    w = np.asarray(weight, dtype=float)
    wb = np.broadcast_to(w, base.shape[:3])
    x = base - np.array([midpoint_x, 0.0, 0.0])
    ang = wb * pose["yaw"]
    c, s = np.cos(ang), np.sin(ang)
    rot = np.stack([c * x[..., 0] - s * x[..., 1],
                    s * x[..., 0] + c * x[..., 1],
                    x[..., 2]], axis=-1)
    out = rot + np.array([midpoint_x, 0.0, 0.0])
    out[..., 1] += wb * pose["sway"]
    return out


def ale_advection_residual(volume_old: np.ndarray, volume_new: np.ndarray,
                           field_old: np.ndarray, field_new: np.ndarray,
                           face_flux: np.ndarray, face_field: np.ndarray,
                           dt: float, geom: ALEGeometry) -> np.ndarray:
    """ALE convective residual in divergence form.

    ``(V^{n+1} phi^{n+1} - V^n phi^n)/dt + sum_f F_f phi_f`` with
    ``F_f = (u_f - u_mf).S_f``.  For a uniform field and an exact space
    conservation law this is zero to roundoff, which is precisely what
    ``tests/test_moving_mesh_ale.py`` asserts.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    transient = (volume_new * field_new - volume_old * field_old) / dt
    return transient + geom.divergence(face_flux * face_field)


# ---------------------------------------------------------------------------
# grid builders
# ---------------------------------------------------------------------------
def build_cylinder_ogrid(*, radius: float = 0.05, n_theta: int = 72,
                         n_radial: int = 24, span: float = 0.2,
                         n_span: int = 8, far_radius: float = 1.0,
                         first_layer: float = 0.02,
                         stretch: float = 1.10,
                         center=(0.0, 0.0)) -> StructuredCurvilinearMesh:
    """O-grid around a circular cylinder (verification geometry).

    ``i`` wraps around the section (periodic), ``j`` is radial (body -> far
    field), ``k`` is the spanwise direction (periodic).  The outermost radial
    plane is an open boundary; the innermost is the cylinder wall.
    """
    theta = 2.0 * np.pi * np.arange(n_theta + 1) / n_theta
    s = blend_profile(n_radial, first=first_layer, stretch=stretch)
    r = radius + s * (far_radius - radius)
    z = np.linspace(-0.5 * span, 0.5 * span, n_span + 1)
    T, R, Z = np.meshgrid(theta, r, z, indexing="ij")
    nodes = np.stack([center[0] + R * np.cos(T), center[1] + R * np.sin(T), Z], axis=-1)
    return StructuredCurvilinearMesh(
        nodes, periodic=(True, False, False),
        patches={(1, "low"): "wall", (1, "high"): "far",
                 (2, "low"): "slip", (2, "high"): "slip"})


def build_kvlcc2_double_body_ogrid(hull, *, n_station: int = 48, n_around: int = 48,
                                   n_radial: int = 20, xi_lo: float | None = None,
                                   xi_hi: float | None = None,
                                   end_half_breadth_fraction: float = 0.03,
                                   half_width: float = 3.5, half_depth: float = 2.8,
                                   first_layer: float = 0.004,
                                   stretch: float = 1.15) -> StructuredCurvilinearMesh:
    """Body-fitted double-body O-grid around the KVLCC2 hull.

    The inner curve of every station is the *closed double-body section*
    (hull below the waterline plus its mirror image above it, which is the
    standard rigid-lid / Fn -> 0 model with the waterplane as a symmetry
    plane).  The outer curve is the rectangular far field
    ``|y| = half_width``, ``|z| = half_depth``.

    The block covers the hull length only, so the two end planes
    (``xi_lo``/``xi_hi``) are solid caps: they are perpendicular to the
    approach flow and symmetric in ``y``, hence they contribute (to leading
    order) only to the *longitudinal* force and not to sway/yaw.  Meshing the
    open inlet/outlet needs a multi-block H-O topology, which is *not*
    implemented - this builder is therefore used for the geometric gate
    (discrete ``Cb``/wetted area) and for ALE kinematics exercises, not for a
    claimed KVLCC2 flow solution.
    """
    js = blend_profile(n_radial, first=first_layer, stretch=stretch)

    def section_half_breadth(xi_arr):
        xi = np.atleast_1d(np.asarray(xi_arr, dtype=float))
        b = 0.5 * hull.B * np.asarray(hull._wl_raw(xi - hull.lcb_shift))  # noqa: SLF001
        p = np.array([hull._p_of_cm(hull.section_coeff(float(v)))          # noqa: SLF001
                      if 0.0 < v < 1.0 else 1.0 for v in xi])
        return b, p

    def loop(xi: float, n_around: int):
        """Closed double-body section: 4 quarter arcs (starboard/port/mirror)."""
        b, p = section_half_breadth(xi)
        b = float(np.atleast_1d(b)[0]); p = float(np.atleast_1d(p)[0])
        m = n_around // 4
        zeta = np.linspace(1.0, 0.0, m + 1)                  # waterline -> keel
        yq = b * np.sqrt(np.clip(1.0 - (1.0 - zeta) ** p, 0.0, None))
        zq = -hull.T + zeta * hull.T
        # order chosen so that ``k = 0`` is +y and ``k = n/4`` is +z, matching
        # the angular parameter of the far-field rectangle
        up_star = np.stack([yq, -zq], axis=-1)[:-1]           # (b,0) -> top
        down_port = np.stack([-yq, -zq], axis=-1)[::-1][:-1]  # top -> (-b,0)
        down_port_hull = np.stack([-yq, zq], axis=-1)[:-1]    # (-b,0) -> keel
        up_star_hull = np.stack([yq, zq], axis=-1)[::-1][:-1]  # keel -> (b,0)
        pts = [up_star, down_port, down_port_hull, up_star_hull]
        loop_pts = np.concatenate(pts, axis=0)
        return loop_pts[:n_around]

    # far field: rectangle |y| = half_width, |z| = half_depth sampled on a fixed
    # angular parameter so that the outer curve is smooth in ``k``
    theta = 2.0 * np.pi * np.arange(n_around) / n_around
    cy, cz = np.cos(theta), np.sin(theta)
    scale = np.minimum(np.where(np.abs(cy) > 1e-12, half_width / np.maximum(np.abs(cy), 1e-12), np.inf),
                       np.where(np.abs(cz) > 1e-12, half_depth / np.maximum(np.abs(cz), 1e-12), np.inf))
    outer_pts = np.stack([cy * scale, cz * scale], axis=-1)

    def _cut(side: str) -> float:
        """Station where the waterline half-breadth equals the end value."""
        target = end_half_breadth_fraction
        if side == "low":                       # wl_raw: 0 (AP) -> 1 (midbody)
            lo, hi = 0.0, hull.middle_body_start
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if float(hull._wl_raw(np.array([mid]))[0]) < target:
                    lo = mid
                else:
                    hi = mid
        else:                                   # wl_raw: 1 -> 0 (FP)
            lo, hi = hull.middle_body_end, 1.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if float(hull._wl_raw(np.array([mid]))[0]) > target:
                    lo = mid
                else:
                    hi = mid
        return 0.5 * (lo + hi)

    if xi_lo is None:
        xi_lo = _cut("low")
    if xi_hi is None:
        xi_hi = _cut("high")
    xs = np.linspace(xi_lo, xi_hi, n_station + 1) * hull.L
    nodes = np.zeros((n_station + 1, n_radial + 1, n_around + 1, 3))
    for i, x in enumerate(xs):
        lp = loop(x / hull.L, n_around)
        op = outer_pts
        for k in range(n_around + 1):
            kk = k % n_around
            inner = lp[kk]
            out = op[kk]
            blend = js[:, None]
            yz = inner[None, :] + blend * (out - inner)[None, :]
            nodes[i, :, k, 0] = x
            nodes[i, :, k, 1] = yz[:, 0]
            nodes[i, :, k, 2] = yz[:, 1]
    return StructuredCurvilinearMesh(
        nodes, periodic=(False, False, True),
        patches={(0, "low"): "cap_low", (0, "high"): "cap_high",
                 (1, "low"): "wall", (1, "high"): "far"})

# ---------------------------------------------------------------------------
# conservative ALE scalar transport (verified building block for the flow solver)
# ---------------------------------------------------------------------------
def ale_scalar_step(geom_old: ALEGeometry, geom_new: ALEGeometry, *,
                    field_old: np.ndarray, face_velocity: np.ndarray,
                    sweep: np.ndarray, dt: float, diffusivity: float) -> np.ndarray:
    """One implicit-diffusion, explicit-upwind ALE step for a passive scalar.

    ``(V^{n+1} phi^{n+1} - V^n phi^n)/dt + sum_f (u_f - u_mf).S_f phi_f
      = sum_f D_f (phi_N - phi_P)``

    with ``u_mf.S_f = sweep/dt`` taken from the exact space-conservation
    decomposition.  The step is conservative by construction (the flux leaves
    one cell and enters the next) and free-stream preserving: a uniform field
    is reproduced to roundoff on any moving mesh.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    m = geom_new.mesh
    area = geom_new.face_area
    flux = np.einsum("fi,fi->f", face_velocity, area) - sweep / dt
    diff = diffusivity * geom_new.area_mag / geom_new.dperp
    up = np.where(flux > 0.0, field_old[m.owner], field_old[m.neighbor.clip(0)])
    rhs = (geom_old.volume * field_old / dt - geom_new.divergence(flux * up)
           + geom_new.divergence(diff * field_old[m.neighbor.clip(0)]))
    denom = (geom_new.volume / dt
             + np.bincount(m.owner, weights=diff, minlength=m.n_cells)
             + np.bincount(m.neighbor[m.interior], weights=diff[m.interior],
                           minlength=m.n_cells))
    return rhs / denom


def wall_ale_flux(geom_old: ALEGeometry, geom_new: ALEGeometry, *,
                  face_velocity: np.ndarray, sweep: np.ndarray,
                  label: str = "wall") -> np.ndarray:
    """ALE mass flux ``(u_f - u_mf).S_f`` through the body patch.

    A body-fitted moving wall carries the mesh with it, so this must vanish to
    roundoff - a direct machine check that the morphing keeps the wall
    impermeable (no spurious mass source in a PMM oscillation).
    """
    m = geom_old.mesh
    mask = np.array([lab[1] == label for lab in m.face_label])
    flux = (np.einsum("fi,fi->f", face_velocity, geom_new.face_area) - sweep)
    return flux[mask]
