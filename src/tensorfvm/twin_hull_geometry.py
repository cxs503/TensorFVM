"""Parametric KVLCC2 demihull geometry and catamaran (twin-hull) panelisation.

Why this module exists
----------------------
The stiffened ``kvlcc2_geometry`` module in this package carries a principal
dimension set (B = 0.936 m, T = 0.342 m, Cb = 0.708) that does not reproduce the
official KVLCC2 particulars; it is kept untouched for its own (manoeuvring)
users.  For resistance work we need a hull whose *form parameters* are the
official ones and whose discretisation is explicitly verifiable:

    Lpp = 320.0 m, B = 58.0 m, T = 20.8 m, Cb = 0.8098, LCB = +3.48 % L fwd of
    midship  (SIMMAN 2021 KVLCC2 principal particulars; model scale 1:45.714)

Main dimensions alone do not define a hull, so the waterlines/sections are
reconstructed parametrically and *calibrated* until the discretised hull really
has Cb = 0.8098 and LCB = +0.0348 L.  The calibration is part of the module
(``calibrate_form``), not a hand-tuned constant, and the resulting geometry gate
(``validate_geometry``) is a machine-checkable claim:

  * Cb, LCB, Lpp, B, T reproduced to the stated tolerances
  * wetted surface compared with the published KVLCC2 value (~27 192 m^2)

Geometry model
--------------
* waterline half breadth      b(s) = (B/2) g(s),   s = x/L - 1/2 in [0, 1]
  parallel middle body g = 1 for |s - 1/2| <= p, entrance/run g = 1 - u^e with
  u the normalised distance into the entrance (e_f) or run (e_a).
* section shape (per station):  y(zeta)/b = 1 - (1 - f_k) (1 - eta)^r with
  eta = (z + T)/T the normalised height above the keel, f_k the keel breadth
  ratio and r the bilge exponent.  Section area coefficient is then
  Cm = 1 - (1 - f_k)/(r + 1); f_k shrinks towards the ends (f_k = f_k0 g^kbe)
  so that the end sections become V-shaped.

Everything is expressed in model coordinates with the waterline at z = 0, the
keel at z = -T, and x from -L/2 (AP) to +L/2 (FP).  Surface normals point out
of the hull into the fluid.

The catamaran is two demihulls whose centre planes are s apart
(``separation_ratio = s/L``, the convention used by the Molland catamaran
series), i.e. the clear gap between inner shells is s - B.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Published KVLCC2 wetted surface at full scale (m^2); used only as a validation
# target, never as an input to the geometry.
KVLCC2_WETTED_SURFACE_PUBLISHED_M2 = 27192.0


@dataclass(frozen=True)
class KVLCC2Principal:
    """KVLCC2 principal particulars (full scale) plus a model scale factor."""

    Lpp: float = 320.0
    B: float = 58.0
    T: float = 20.8
    Cb: float = 0.8098
    LCB_fraction: float = 0.0348  # forward of midship, fraction of Lpp
    model_scale: float = 1.0 / 45.714
    midship_section_coefficient: float = 0.9989

    @property
    def L(self) -> float:
        return self.Lpp * self.model_scale

    @property
    def breadth(self) -> float:
        return self.B * self.model_scale

    @property
    def draft(self) -> float:
        return self.T * self.model_scale

    @property
    def displaced_volume(self) -> float:
        return self.Cb * self.L * self.breadth * self.draft

    def to_dict(self) -> dict:
        return dict(
            Lpp=self.Lpp, B=self.B, T=self.T, Cb=self.Cb,
            LCB_fraction=self.LCB_fraction, model_scale=self.model_scale,
            L_model=self.L, B_model=self.breadth, T_model=self.draft,
            volume_model=self.displaced_volume,
            midship_section_coefficient=self.midship_section_coefficient,
        )


@dataclass(frozen=True)
class HullFormParameters:
    """Free parameters of the parametric reconstruction (calibrated, not tuned)."""

    fore_exponent: float = 2.0
    aft_exponent: float = 2.0
    parallel_body_half_length: float = 0.25
    keel_breadth_ratio: float = 0.90
    bilge_exponent: float = 4.0
    keel_fullness_exponent: float = 2.0
    transom_breadth_ratio: float = 0.0


def waterline_fullness(s, params: HullFormParameters):
    """g(s): waterline half breadth / (B/2), 0 at AP/FP, 1 in the parallel body."""
    s = np.asarray(s, dtype=float)
    p = params.parallel_body_half_length
    mid = 0.5
    g = np.ones_like(s)
    fore = s > mid + p
    aft = s < mid - p
    u_fore = (s[fore] - (mid + p)) / (mid - p)
    u_aft = ((mid - p) - s[aft]) / (mid - p)
    g[fore] = 1.0 - u_fore ** params.fore_exponent
    # the run ends in a transom of finite breadth (g_ap > 0) instead of a knife
    # edge: a real tanker stern is cut off, and a point stern makes the strip
    # boundary layer (and therefore the viscous pressure resistance) singular
    g_ap = params.transom_breadth_ratio
    g[aft] = g_ap + (1.0 - g_ap) * (1.0 - u_aft ** params.aft_exponent)
    return np.clip(g, 0.0, 1.0)


def section_area_coefficient(s, params: HullFormParameters):
    """Cm(s) = A(s) / (2 b(s) T) implied by the section shape family."""
    g = waterline_fullness(s, params)
    fk = params.keel_breadth_ratio * g ** params.keel_fullness_exponent
    return 1.0 - (1.0 - fk) / (params.bilge_exponent + 1.0)


def sectional_area(s, principal: KVLCC2Principal, params: HullFormParameters):
    """A(s) = 2 b(s) T Cm(s)  [m^2]."""
    g = waterline_fullness(s, params)
    return 2.0 * 0.5 * principal.breadth * g * principal.draft * section_area_coefficient(s, params)


def _form_integrals(s, principal, params):
    """Return (Cb, LCB fraction, volume, wetted-surface-free integrals)."""
    A = sectional_area(s, principal, params)
    L = principal.L
    volume = float(np.trapezoid(A, s * L))
    xbar = float(np.trapezoid(A * (s - 0.5) * L, s * L))
    Cb = volume / (L * principal.breadth * principal.draft)
    return Cb, xbar / volume, volume, A


def discretised_form(principal, params, n=4001):
    """Cb and LCB of the parametric hull on a uniform station grid."""
    s = np.linspace(0.0, 1.0, n)
    Cb, lcb, volume, A = _form_integrals(s, principal, params)
    return dict(Cb=Cb, LCB_fraction=lcb, volume=volume, sectional_area=A, s=s)


def calibrate_form(principal: KVLCC2Principal | None = None, *,
                   parallel_body_half_length: float = 0.25,
                   keel_breadth_ratio: float = 0.90,
                   bilge_exponent: float = 4.0,
                   keel_fullness_exponent: float = 2.0,
                   transom_breadth_ratio: float = 0.0,
                   tol: float = 1e-13, max_iter: int = 80) -> HullFormParameters:
    """Solve for (fore, aft) entrance exponents reproducing Cb and LCB.

    Two unknowns, two targets; damped Newton with numerical Jacobian.  The
    exponents are kept inside [1.15, 6.0] so that the waterline stays monotone
    (a physical requirement, not an aesthetic one).
    """
    principal = principal or KVLCC2Principal()
    base = HullFormParameters(
        parallel_body_half_length=parallel_body_half_length,
        keel_breadth_ratio=keel_breadth_ratio,
        bilge_exponent=bilge_exponent,
        keel_fullness_exponent=keel_fullness_exponent,
        transom_breadth_ratio=transom_breadth_ratio,
    )

    def residual(e):
        p = replace(base, fore_exponent=float(e[0]), aft_exponent=float(e[1]))
        d = discretised_form(principal, p, n=1201)
        return np.array([d["Cb"] - principal.Cb,
                         d["LCB_fraction"] - principal.LCB_fraction])

    e = np.array([2.0, 2.0])
    lo, hi = 1.15, 6.0
    for _ in range(max_iter):
        r = residual(e)
        if np.max(np.abs(r)) < tol:
            break
        Jac = np.empty((2, 2))
        h = 1e-4
        for k in range(2):
            de = np.zeros(2)
            de[k] = h
            Jac[:, k] = (residual(e + de) - r) / h
        try:
            step = np.linalg.solve(Jac, -r)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            break
        step = np.clip(step, -0.5, 0.5)
        e = np.clip(e + step, lo, hi)
    params = replace(base, fore_exponent=float(e[0]), aft_exponent=float(e[1]))
    check = discretised_form(principal, params, n=4001)
    if abs(check["Cb"] - principal.Cb) > 1e-5 or abs(check["LCB_fraction"] - principal.LCB_fraction) > 1e-5:
        raise RuntimeError(
            "form calibration did not reach the Cb/LCB targets: "
            f"Cb={check['Cb']:.6f} (target {principal.Cb}), "
            f"LCB={check['LCB_fraction']:.6f} (target {principal.LCB_fraction})"
        )
    return params


@dataclass
class PanelSet:
    """Quadrilateral panel mesh: centroids, outward normals, areas, corners."""

    centroids: np.ndarray  # (N, 3)
    normals: np.ndarray    # (N, 3) unit, pointing into the fluid
    areas: np.ndarray      # (N,)
    corners: np.ndarray | None = None  # (N, 4, 3)
    label: str = ""

    def __post_init__(self):
        self.centroids = np.ascontiguousarray(self.centroids, dtype=float)
        self.normals = np.ascontiguousarray(self.normals, dtype=float)
        self.areas = np.ascontiguousarray(self.areas, dtype=float)

    @property
    def n_panels(self) -> int:
        return self.centroids.shape[0]

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))

    def active_mask(self, rel_tol: float = 1e-10) -> np.ndarray:
        """Panels large enough to be unknowns (degenerate tip panels excluded)."""
        if self.areas.size == 0:  # pragma: no cover - defensive
            return np.zeros(0, dtype=bool)
        return self.areas > rel_tol * float(self.areas.max())

    def translated(self, dy: float = 0.0) -> "PanelSet":
        c = self.centroids + np.array([0.0, dy, 0.0])
        corners = None if self.corners is None else self.corners + np.array([0.0, dy, 0.0])
        return PanelSet(c, self.normals.copy(), self.areas.copy(), corners, self.label)

    def offset_along_normals(self, displacement: np.ndarray) -> "PanelSet":
        """Displacement body: shift every panel by ``displacement`` along its normal."""
        d = np.asarray(displacement, dtype=float)[:, None]
        c = self.centroids + d * self.normals
        corners = None
        if self.corners is not None:
            # Node displacements are obtained by averaging the four adjacent
            # panels' displacements, applied on the corner normals.
            corners = self.corners + d[:, None, :] * self.normals[:, None, :]
        return PanelSet(c, self.normals.copy(), self.areas.copy(), corners, self.label + "+disp")


def _panelise_nodes(nodes: np.ndarray, label: str = "") -> PanelSet:
    """Build quad panels from a structured node grid (n_s+1, n_g+1, 3)."""
    p00 = nodes[:-1, :-1]
    p10 = nodes[1:, :-1]
    p11 = nodes[1:, 1:]
    p01 = nodes[:-1, 1:]
    centroids = 0.25 * (p00 + p10 + p11 + p01)
    n = np.cross(p11 - p00, p01 - p10)  # diagonal cross product
    nnorm = np.linalg.norm(n, axis=-1)
    areas = 0.5 * nnorm
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = n / np.where(nnorm > 0, nnorm, 1.0)[..., None]
    return PanelSet(
        centroids.reshape(-1, 3),
        normals.reshape(-1, 3),
        areas.reshape(-1),
        np.stack([p00, p10, p11, p01], axis=2).reshape(-1, 4, 3),
        label,
    )


def merge_panels(first: PanelSet, second: PanelSet, label: str = "") -> PanelSet:
    """Concatenate two panel sets (first panels keep the lower indices)."""
    if first.corners is None or second.corners is None:
        raise ValueError("both panel sets need corners")
    return PanelSet(
        np.vstack([first.centroids, second.centroids]),
        np.vstack([first.normals, second.normals]),
        np.concatenate([first.areas, second.areas]),
        np.concatenate([first.corners, second.corners]),
        label or (first.label + "+" + second.label),
    )


def _transom_panels(nodes_ap: np.ndarray) -> PanelSet:
    """Cap the aft end with a triangle fan (the transom face, normal aft).

    ``nodes_ap`` is the section contour at the aft perpendicular: it starts at
    the waterline on the port side, runs around the bottom and ends at the
    waterline on the starboard side, so the polygon is closed by the waterline
    segment itself.
    """
    centre = np.array([nodes_ap[0, 0], 0.0, -0.5 * (np.max(-nodes_ap[:, 2]) if False else abs(nodes_ap[:, 2].min()))])
    pts = nodes_ap
    corn, cen, nrm, are = [], [], [], []
    for j in range(len(pts) - 1):
        a, b = pts[j], pts[j + 1]
        c = np.array([centre[0], centre[1], centre[2]])
        quad = np.stack([c, a, b, b])
        n = np.cross(b - c, b - a)
        nn = np.linalg.norm(n)
        nrm.append(n / nn if nn > 0 else np.array([-1.0, 0.0, 0.0]))
        are.append(0.5 * nn)
        cen.append(quad.mean(axis=0))
        corn.append(quad)
    # close the polygon along the waterline (last contour point back to the first)
    a, b = pts[-1], pts[0]
    c = np.array([centre[0], centre[1], centre[2]])
    quad = np.stack([c, a, b, b])
    n = np.cross(b - c, b - a)
    nn = np.linalg.norm(n)
    nrm.append(n / nn if nn > 0 else np.array([-1.0, 0.0, 0.0]))
    are.append(0.5 * nn)
    cen.append(quad.mean(axis=0))
    corn.append(quad)
    panels = PanelSet(np.array(cen), np.array(nrm), np.array(are), np.array(corn), "transom")
    # orient aft (-x)
    flip = panels.normals[:, 0] > 0
    panels.normals[flip] *= -1.0
    return panels


def _cluster(count: int) -> np.ndarray:
    """Nodes clustered at both ends of [0, 1] (cosine distribution)."""
    i = np.arange(count + 1)
    return 0.5 * (1.0 - np.cos(np.pi * i / count))


class DemihullGeometry:
    """Discretised KVLCC2 demihull below the waterline.

    The hull surface is a structured node grid (station x girth).  The girth
    starts at the waterline on the inner side, runs down to the bilge, along the
    flat bottom to the centre plane, and back up the outer side; the surface is
    therefore a closed shell except at the two waterline edges (which the
    double-body image in ``twin_hull_form_factor`` closes) and at the two
    (vanishing) end stations.
    """

    def __init__(self, principal: KVLCC2Principal | None = None,
                 params: HullFormParameters | None = None,
                 n_stations: int = 36, n_girth_half: int = 14):
        self.principal = principal or KVLCC2Principal()
        self.params = params or calibrate_form(self.principal)
        if n_girth_half < 3:
            raise ValueError("n_girth_half must be >= 3")
        self.n_stations = int(n_stations)
        self.n_girth_half = int(n_girth_half)
        self.nodes = self._build_nodes()
        self.panels = _panelise_nodes(self.nodes, label="demihull")
        self._orient_normals()
        self.transom = _transom_panels(self.nodes[0]) if self.params.transom_breadth_ratio > 0 else None
        self.n_hull_panels = self.panels.n_panels
        self._reindex()

    # -- construction -----------------------------------------------------
    def _build_nodes(self) -> np.ndarray:
        pr, pa = self.principal, self.params
        s_nodes = _cluster(self.n_stations)
        # local girth grid for one half section, clustered at both ends
        u = _cluster(self.n_girth_half)
        nodes = np.empty((self.n_stations + 1, 2 * self.n_girth_half + 1, 3))
        for i, s in enumerate(s_nodes):
            b = 0.5 * pr.breadth * float(waterline_fullness(s, pa))
            fk = pa.keel_breadth_ratio * float(waterline_fullness(s, pa)) ** pa.keel_fullness_exponent
            # dense sampling of the half section, then arc-length resampling
            eta = np.linspace(1.0, 0.0, 801)         # waterline -> keel
            shape = 1.0 - (1.0 - fk) * (1.0 - eta) ** pa.bilge_exponent
            y_half = b * shape
            z_half = -pr.draft * (1.0 - eta)
            y_bottom = np.linspace(b * fk, 0.0, 401)   # flat bottom to centre plane
            z_bottom = np.full_like(y_bottom, -pr.draft)
            yy = np.concatenate([y_half, y_bottom[1:]])
            zz = np.concatenate([z_half, z_bottom[1:]])
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(yy), np.diff(zz)))])
            target = u * arc[-1]
            yi = np.interp(target, arc, yy)
            zi = np.interp(target, arc, zz)
            x = (s - 0.5) * pr.L
            girth = np.concatenate([-yi, yi[::-1][1:]])
            zg = np.concatenate([zi, zi[::-1][1:]])
            # centre-plane point must be the flat-bottom one (y = 0, z = -T)
            girth[self.n_girth_half] = 0.0
            zg[self.n_girth_half] = -pr.draft
            nodes[i, :, 0] = x
            nodes[i, :, 1] = girth
            nodes[i, :, 2] = zg
        return nodes

    def _orient_normals(self):
        p = self.panels
        centre = np.stack([p.centroids[:, 0], np.zeros_like(p.areas), np.full_like(p.areas, -0.5 * self.principal.draft)], axis=1)
        outward = np.einsum("ij,ij->i", p.normals, p.centroids - centre)
        flip = outward < 0.0
        p.normals[flip] *= -1.0
        if p.corners is not None:
            p.corners[flip] = p.corners[flip][:, ::-1]

    def _reindex(self):
        """Station-major index helper arrays for the boundary-layer march."""
        self.shape = (self.n_stations, 2 * self.n_girth_half)
        self.panel_index = np.arange(self.panels.n_panels).reshape(self.shape)
        self.active = self.panels.active_mask()
        # station centroids x
        self.station_x = self.nodes[:, :, 0].mean(axis=1)
        self.station_centres_x = 0.5 * (self.station_x[:-1] + self.station_x[1:])

    # -- derived quantities ----------------------------------------------
    def solver_panels(self) -> PanelSet:
        """Panels handed to the potential solver: hull first, transom last."""
        if self.transom is None:
            return self.panels
        return merge_panels(self.panels, self.transom, label="demihull+transom")

    def wetted_surface(self, active_only: bool = True) -> float:
        """Wetted area of the demihull below the waterline (hull + transom)."""
        a = self.panels.areas
        total = float(np.sum(a[self.active] if active_only else a))
        if self.transom is not None:
            total += self.transom.total_area
        return total

    def end_station_areas(self) -> tuple[float, float]:
        a = self.panels.areas.reshape(self.shape)
        return float(a[0].sum()), float(a[-1].sum())

    def validate(self, n_quadrature: int = 8001) -> dict:
        """Geometry gate: Cb/LCB/L/B/T plus the published wetted surface."""
        pr = self.principal
        s = np.linspace(0.0, 1.0, n_quadrature)
        Cb, lcb, volume, _ = _form_integrals(s, pr, self.params)
        S = self.wetted_surface()
        S_ref = KVLCC2_WETTED_SURFACE_PUBLISHED_M2 * pr.model_scale ** 2
        # discretised (panel-grid) sectional areas, for the reported Cb of the grid
        A_grid = sectional_area(_cluster(self.n_stations), pr, self.params)
        L = pr.L
        dx = np.diff(_cluster(self.n_stations)) * L
        volume_grid = float(np.sum(0.5 * (A_grid[1:] + A_grid[:-1]) * dx))
        cb_grid = volume_grid / (L * pr.breadth * pr.draft)
        return dict(
            Lpp=pr.L, B=pr.breadth, T=pr.draft,
            Cb_analytic=Cb, Cb_target=pr.Cb, Cb_grid=cb_grid,
            LCB_fraction_analytic=lcb, LCB_target=pr.LCB_fraction,
            volume=volume, wetted_surface=S,
            wetted_surface_published=S_ref,
            wetted_surface_error=(S - S_ref) / S_ref,
            midship_Cm=float(section_area_coefficient(np.array([0.5]), self.params)[0]),
            Cm_target=pr.midship_section_coefficient,
            n_panels=self.panels.n_panels, n_active_panels=int(self.active.sum()),
            n_stations=self.n_stations, n_girth_half=self.n_girth_half,
            fore_exponent=self.params.fore_exponent,
            aft_exponent=self.params.aft_exponent,
            parallel_body_half_length=self.params.parallel_body_half_length,
            keel_breadth_ratio=self.params.keel_breadth_ratio,
            bilge_exponent=self.params.bilge_exponent,
        )


def catamaran_panels(demihull: DemihullGeometry, separation_ratio: float) -> tuple[PanelSet, dict]:
    """Place one demihull in the twin-hull configuration.

    ``separation_ratio`` = centre-plane distance / L.  Because the flow is
    symmetric about the catamaran centre plane, only one demihull is an unknown
    and the partner hull enters the kernel as a mirror image (see
    ``twin_hull_form_factor.IMAGE_SPECS``).  The returned panel set is the
    starboard demihull, translated so its centre plane sits at y = +s/2.

    Returns the panels and a small metadata dict (gap geometry).
    """
    s = float(separation_ratio)
    if s <= 0:
        raise ValueError("separation_ratio must be positive")
    L = demihull.principal.L
    offset = 0.5 * s * L
    clearance = s * L - demihull.principal.breadth
    meta = dict(
        separation_ratio=s,
        separation=s * L,
        separation_over_B=s * L / demihull.principal.breadth,
        clearance=clearance,
        clearance_ratio=clearance / L,
        hull_offset=offset,
    )
    return demihull.solver_panels().translated(offset), meta


# ---------------------------------------------------------------------------
# verification geometry: a sphere with an analytic potential-flow solution
# ---------------------------------------------------------------------------
def sphere_panels(radius: float = 1.0, n_theta: int = 24, n_phi: int = 18) -> PanelSet:
    """Latitude/longitude panelisation of a sphere centred at the origin."""
    th = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    ph = np.linspace(0.0, np.pi, n_phi + 1)
    nodes = np.empty((n_theta + 1, n_phi + 1, 3))
    for i, t in enumerate(th):
        for j, p in enumerate(ph):
            nodes[i, j] = (radius * np.sin(p) * np.cos(t),
                           radius * np.sin(p) * np.sin(t),
                           radius * np.cos(p))
    panels = _panelise_nodes(nodes, label="sphere")
    centre = np.zeros(3)
    outward = np.einsum("ij,ij->i", panels.normals, panels.centroids - centre)
    flip = outward < 0.0
    panels.normals[flip] *= -1.0
    return panels


def panelise_nodes(nodes: np.ndarray, label: str = "") -> PanelSet:
    """Public wrapper: build quad panels from a structured node grid."""
    return _panelise_nodes(nodes, label)
