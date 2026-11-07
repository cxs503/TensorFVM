"""KVLCC2 double-body viscous-flow case: coefficient-validated hull, stair-step
Cartesian mesh, RANS driver and form-factor extraction.

What a "double body" is
-----------------------
The free surface is replaced by a plane of symmetry: the hull is mirrored about
the design waterline, so the wavy problem becomes a closed-body, single-phase
problem.  The viscous resistance of the double body then splits exactly into

    R_V = R_shear + R_pressure,   C_F = R_shear/(q S),  C_Pv = R_pressure/(q S)
    (1+k) = (C_F + C_Pv) / C_F

over the *ship's* wetted surface ``S`` (the mirror doubles both the area and the
force, so the coefficient is unchanged).  ``(1+k)`` obtained this way is the
Hughes form factor that the Holtrop regression is compared against.

Geometry discipline
-------------------
No trustworthy KVLCC2 offset table exists in this workspace: the platform
``KVLCC2_Hull.stl`` is a 304-triangle placeholder and ``tensorfvm.kvlcc2_geometry``
ships a synthetic table.  The body used here is therefore a parametric section
family whose integral coefficients (``Cb``, ``Cm``, ``Cwp``, ``LCB``) are
*fitted* to the principal particulars, so the dimensionless geometry that the
resistance correlations depend on is exact by construction and is verified
numerically by :meth:`HullShape.coefficients` -- geometry before mesh, as
required.

The mission benchmark ``L = 7 m, Cb = 0.708`` is the "KVLCC2 at model scale"
convention used throughout this repository.  The real KVLCC2 has
``Cb = 0.8098`` and ``L/B = 5.52`` at every scale, so those model particulars are
*not* geometrically similar to KVLCC2 (``L/B = 7.48`` here);
:data:`KVLCC2_FULL_SCALE` carries the real ship for the sensitivity this causes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .resistance_chain import (
    RHO_SEAWATER, HullPrincipal, form_factor_from_double_body, ittc57_cf,
)

__all__ = [
    "KVLCC2_MODEL", "KVLCC2_FULL_SCALE", "HullShape", "fit_hull_shape",
    "DoubleBodyMesh", "build_double_body_mesh", "wall_cell_height_for_y_plus",
    "FITTED_KVLCC2_MODEL_PARAMS", "default_hull_shape",
    "drag_components", "run_double_body_rans",
]

#: Mission benchmark particulars ("KVLCC2 at model scale" as used in this repo).
KVLCC2_MODEL = HullPrincipal(name="KVLCC2_model_L7", L=7.0, B=0.936, T=0.342,
                             Cb=0.708, Cm=0.985, Cwp=0.808, lcb_pct=-1.688)
#: The real KVLCC2 (MOERI/KRISO) at full scale, for the geometry audit.
KVLCC2_FULL_SCALE = HullPrincipal(name="KVLCC2_full_scale", L=320.0, B=58.0,
                                  T=20.8, Cb=0.8098, Cm=0.9982, Cwp=0.87,
                                  lcb_pct=-3.0)


# --------------------------------------------------------------------------- #
# Parametric hull with fitted integral coefficients
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HullShape:
    """Hull as a parametric section family defined by its integral coefficients.

    Station coordinate ``xi = x/L`` in ``[0, 1]`` (0 = AP, 1 = FP), depth
    ``s = zeta/T`` in ``[0, 1]`` below the waterline.

    * waterline half breadth ``w(xi)`` (fraction of ``B/2``): fore
      ``(1 - t^p_fore)^(1/q_fore)``, aft
      ``w_transom + (1 - w_transom)(1 - u^p_aft)^(1/q_aft)``, with
      ``w(0.5) = 1`` and ``w(1) = 0`` (stem);
    * section ``y = (B/2) w(xi) (1 - (1-g) s^m)^(1/m)``: a superellipse family
      from a fine V section (``m -> 1``) to the flat-bottomed midship section
      (``m >= 3``, ``g -> 1``).

    ``m``, ``g`` and the waterline exponents are station dependent through
    ``w(xi)``, which is what allows ``Cb/Cwp`` (average section fullness) to
    differ from ``Cm`` (midship fullness); ``Cb = Cm Cwp`` would otherwise force
    an impossible hull.
    """

    principal: HullPrincipal
    q_fore: float = 1.60
    q_aft: float = 2.00
    w_transom: float = 0.62
    p_fore: float = 2.0
    p_aft: float = 2.0
    m_mid: float = 3.20
    m_end: float = 1.00
    g_mid: float = 0.94
    g_end: float = 0.10
    section_power: float = 2.0

    # -- dimensionless station quantities ------------------------------- #
    def waterline(self, xi) -> np.ndarray:
        """Dimensionless waterline half breadth ``w(xi)`` with ``w(0.5) = 1``."""
        xi = np.asarray(xi, dtype=float)
        t = np.clip((xi - 0.5) / 0.5, 0.0, 1.0)
        u = np.clip((0.5 - xi) / 0.5, 0.0, 1.0)
        w_fore = (1.0 - t ** self.p_fore) ** (1.0 / self.q_fore)
        w_aft = self.w_transom + (1.0 - self.w_transom) * (
            1.0 - u ** self.p_aft) ** (1.0 / self.q_aft)
        return np.where(xi >= 0.5, w_fore, w_aft)

    def _fullness_weight(self, xi) -> np.ndarray:
        """Station fullness weight ``w(xi)^section_power`` (1 midships, 0 at stem)."""
        return self.waterline(xi) ** self.section_power

    def section_exponent(self, xi) -> np.ndarray:
        return self.m_end + (self.m_mid - self.m_end) * self._fullness_weight(xi)

    def keel_ratio(self, xi) -> np.ndarray:
        return self.g_end + (self.g_mid - self.g_end) * self._fullness_weight(xi)

    @staticmethod
    def _section_integral(m, g, n_samples: int = 401) -> np.ndarray:
        """``I(m,g) = int_0^1 (1-(1-g)s^m)^(1/m) ds``: section area factor."""
        s = np.linspace(0.0, 1.0, n_samples)
        m = np.atleast_1d(np.asarray(m, dtype=float))
        g = np.atleast_1d(np.asarray(g, dtype=float))
        integrand = (1.0 - (1.0 - g)[:, None] * s[None, :] ** m[:, None]) ** (1.0 / m[:, None])
        return np.trapezoid(integrand, s, axis=-1)

    # -- integral coefficients (verify geometry before you mesh) -------- #
    def coefficients(self, n_stations: int = 1201) -> Dict[str, float]:
        xi = np.linspace(0.0, 1.0, n_stations)
        w = self.waterline(xi)
        i_sec = self._section_integral(self.section_exponent(xi), self.keel_ratio(xi))
        p = self.principal
        cwp = float(np.trapezoid(w, xi))
        c_mid = float(self._section_integral(self.m_mid, self.g_mid)[0])
        c_b = float(np.trapezoid(w * i_sec, xi))
        lcb = float(np.trapezoid(xi * w * i_sec, xi) / c_b - 0.5)
        return {
            "Cb": c_b, "Cm": c_mid, "Cwp": cwp,
            "LCB_fraction": lcb, "LCB_pct": 100.0 * lcb,
            "nabla": c_b * p.L * p.B * p.T,
            "target_Cb": p.Cb, "target_Cm": p.Cm, "target_Cwp": p.Cwp,
            "target_LCB_pct": p.lcb_pct,
        }

    def coefficient_errors(self) -> Dict[str, float]:
        c = self.coefficients()
        return {
            "Cb": c["Cb"] / c["target_Cb"] - 1.0,
            "Cm": c["Cm"] / c["target_Cm"] - 1.0,
            "Cwp": c["Cwp"] / c["target_Cwp"] - 1.0,
            "LCB_fraction": c["LCB_fraction"] - 0.01 * c["target_LCB_pct"],
        }

    # -- geometry ------------------------------------------------------- #
    def half_breadth(self, x, zeta) -> np.ndarray:
        """Half breadth ``b(x, zeta)`` [m] at depth ``zeta`` below the waterline."""
        p = self.principal
        xi = np.clip(np.asarray(x, dtype=float) / p.L, 0.0, 1.0)
        s = np.clip(np.asarray(zeta, dtype=float) / p.T, 0.0, 1.0)
        w = self.waterline(xi)
        m = self.section_exponent(xi)
        g = self.keel_ratio(xi)
        return 0.5 * p.B * w * (g + (1.0 - g) * (1.0 - s ** m) ** (1.0 / m))

    def inside(self, x, y, z) -> np.ndarray:
        """Double-body occupancy test ``|y| <= b(x, |z|)`` for ``|z| <= T``.

        ``z`` is measured from the design waterline, so ``z < 0`` is the real
        hull and ``z > 0`` its mirror image.  Points outside the length range or
        below the keel are fluid.
        """
        p = self.principal
        x = np.asarray(x, dtype=float)
        y = np.abs(np.asarray(y, dtype=float))
        z = np.abs(np.asarray(z, dtype=float))
        in_length = (x >= 0.0) & (x <= p.L)
        in_draft = z <= p.T
        breadth = self.half_breadth(np.clip(x, 0.0, p.L), z)
        return in_length & in_draft & (y <= breadth)

    def wetted_area(self, n_stations: int = 801, n_depth: int = 4001) -> float:
        """Analytic wetted surface of the real hull (both sides) [m^2].

        Per station the wetted girth is the arc length of the section curve
        ``zeta -> b(x, zeta)`` plus the flat-bottom strip ``b(x, T)``; the
        section tangent becomes horizontal at the keel, so the arc length is
        sampled densely rather than integrated with a coarse quadrature.
        """
        p = self.principal
        x = np.linspace(0.0, p.L, n_stations)
        zeta = np.linspace(0.0, p.T, n_depth)
        total = 0.0
        girths = np.empty(n_stations)
        for i, xi_pos in enumerate(x):
            b = self.half_breadth(np.full_like(zeta, xi_pos), zeta)
            arc = np.trapezoid(np.sqrt(1.0 + np.gradient(b, zeta) ** 2), zeta)
            girths[i] = arc + b[-1]
        total = float(2.0 * np.trapezoid(girths, x))
        return total


#: Shape parameters produced by :func:`fit_hull_shape` for :data:`KVLCC2_MODEL`
#: (deterministic bounded least squares, residuals < 4e-5 on all four
#: coefficients).  Kept as a constant so meshing and tests are cheap and
#: reproducible; :func:`fit_hull_shape` re-derives them from scratch.
FITTED_KVLCC2_MODEL_PARAMS = dict(q_fore=1.9223, q_aft=1.9621, w_transom=0.2495,
                                  p_fore=2.0, p_aft=2.0, m_mid=2.3983, m_end=0.8106,
                                  g_mid=0.8803, g_end=0.10, section_power=3.7831)


def default_hull_shape(principal: HullPrincipal = KVLCC2_MODEL) -> HullShape:
    """The fitted KVLCC2-model body without re-running the optimiser."""
    params = dict(FITTED_KVLCC2_MODEL_PARAMS) if principal is KVLCC2_MODEL else None
    if params is None:
        return fit_hull_shape(principal)
    return HullShape(principal=principal, **params)


def fit_hull_shape(principal: HullPrincipal = KVLCC2_MODEL,
                   fixed: Optional[Dict[str, float]] = None,
                   max_error: float = 1e-3, max_nfev: int = 800) -> HullShape:
    """Fit the section family to ``Cb``, ``Cm``, ``Cwp`` and ``LCB``.

    Four shape parameters (``q_fore``, ``q_aft``, ``m_mid``, ``g_mid``) are
    adjusted by bounded least squares against the four integral targets; the
    remaining knobs stay at documented defaults.  The returned shape is accepted
    only if every relative coefficient error is below ``max_error``.
    """
    from scipy.optimize import least_squares

    base = dict(q_fore=1.6, q_aft=2.0, w_transom=0.62, p_fore=2.0, p_aft=2.0,
                m_mid=3.2, m_end=1.0, g_mid=0.94, g_end=0.10, section_power=2.0)
    if fixed:
        base.update(fixed)
    names = ("q_fore", "q_aft", "w_transom", "m_mid", "m_end", "g_mid", "section_power")
    lower = np.array([1.02, 1.02, 0.05, 1.02, 0.80, 0.30, 0.30])
    upper = np.array([8.0, 10.0, 0.95, 8.0, 2.00, 0.999, 8.0])
    start = np.array([base[name] for name in names])

    def residuals(vec: np.ndarray) -> np.ndarray:
        params = dict(base)
        params.update({name: float(v) for name, v in zip(names, vec)})
        c = HullShape(principal=principal, **params).coefficients(n_stations=601)
        return np.array([
            (c["Cb"] - principal.Cb) / principal.Cb,
            (c["Cm"] - principal.Cm) / principal.Cm,
            (c["Cwp"] - principal.Cwp) / principal.Cwp,
            (c["LCB_pct"] - principal.lcb_pct) / 10.0,
        ])

    solution = least_squares(residuals, start, bounds=(lower, upper), xtol=1e-14,
                             ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
    params = dict(base)
    params.update({name: float(v) for name, v in zip(names, solution.x)})
    shape = HullShape(principal=principal, **params)
    errors = shape.coefficient_errors()
    if max(abs(v) for v in errors.values()) > max_error:
        raise ValueError(f"hull coefficient fit did not converge: {errors}")
    return shape


# --------------------------------------------------------------------------- #
# Stair-step Cartesian mesh
# --------------------------------------------------------------------------- #
def wall_cell_height_for_y_plus(y_plus: float, speed: float,
                                viscosity: float, cf=None) -> float:
    """First-cell height whose *cell centre* sits at ``y_plus``.

    Uses ``u_tau = U sqrt(Cf/2)`` with the ITTC-57 value of ``Cf`` at the hull
    Reynolds number, and returns the full cell height (twice the centre
    distance), which is what a cell-centred Cartesian mesh needs.
    """
    if y_plus <= 0.0 or speed <= 0.0 or viscosity <= 0.0:
        raise ValueError("y_plus, speed and viscosity must be positive")
    return 2.0 * y_plus * viscosity / (speed * math.sqrt((cf if cf else 3.2e-3) / 2.0))


def _graded_axis(start: float, stop: float, spacing: float, growth: float,
                 max_spacing: Optional[float] = None) -> np.ndarray:
    """Coordinate array on ``[start, stop]`` with the given spacing/growth.

    ``spacing`` applies to the first half of the segment if ``spacing`` is
    smaller than the uniform value, then spacing grows geometrically up to
    ``max_spacing`` (default: the uniform spacing) until ``stop`` is reached.
    """
    if spacing <= 0.0 or growth < 1.0:
        raise ValueError("spacing must be positive and growth >= 1")
    uniform = abs(stop - start)
    coords = [start]
    step = spacing
    direction = 1.0 if stop >= start else -1.0
    limit = np.inf if max_spacing is None else max_spacing
    while abs(coords[-1] - stop) > 1e-12:
        step = min(step * growth if len(coords) > 1 else step, limit)
        nxt = coords[-1] + direction * step
        if direction > 0 and nxt >= stop - 1e-12:
            nxt = stop
        elif direction < 0 and nxt <= stop + 1e-12:
            nxt = stop
        coords.append(nxt)
        if len(coords) > 100000:
            raise RuntimeError("graded axis failed to terminate")
    arr = np.asarray(coords)
    return arr if uniform > 0 else np.array([start, stop])


def _axis_with_fine_band(low: float, high: float, fine_lo: float, fine_hi: float,
                         spacing: float, far_spacing: float, growth: float) -> np.ndarray:
    """Axis from ``low`` to ``high`` refined to ``spacing`` inside a band."""
    parts: List[np.ndarray] = []
    if low < fine_lo - 1e-12:
        parts.append(_graded_axis(low, fine_lo, far_spacing, growth, far_spacing)[:-1])
    parts.append(_graded_axis(fine_lo, fine_hi, spacing, growth, far_spacing))
    if high > fine_hi + 1e-12:
        rev = _graded_axis(high, fine_hi, far_spacing, growth, far_spacing)[:-1]
        parts.append(rev[::-1])
    return np.unique(np.concatenate(parts))


@dataclass
class DoubleBodyMesh:
    """A structured-connectivity, unstructured-storage hexahedral mesh."""

    points: np.ndarray
    cells: List[List[int]]
    patches: Dict[Tuple[int, ...], str]
    meta: Dict[str, float]
    shape: HullShape

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def patch_faces(self, label: str) -> List[Tuple[int, ...]]:
        return [face for face, name in self.patches.items() if name == label]


def build_double_body_mesh(shape: HullShape, spacing: float = 0.06,
                           far_spacing: Optional[float] = None,
                           growth: float = 1.15,
                           x_range: Optional[Tuple[float, float]] = None,
                           y_half: Optional[float] = None,
                           z_depth: Optional[float] = None) -> DoubleBodyMesh:
    """Stair-step hexahedral mesh of the closed double body in a rectangular box.

    The double body is ``hull + mirror about z = 0`` glued along the waterplane,
    i.e. a *closed* body, so the flow needs no symmetry-plane boundary
    condition (the native solver only offers Dirichlet velocity data).  Cells
    whose centre lies inside the body are removed and their exposed faces are
    labelled ``hull`` (no-slip); the box faces are ``inlet``/``outlet``/``far``.

    Stair-stepping is the documented limitation: hull surface alignment is only
    O(spacing), which is why the mesh resolution is reported with every result
    and why the form factor must be read together with its grid study.
    """
    p = shape.principal
    far_spacing = far_spacing or spacing
    x0, x1 = x_range if x_range else (-0.75 * p.L, 1.75 * p.L)
    y_half = y_half if y_half else 2.0 * p.B
    z_depth = z_depth if z_depth else 2.5 * p.T

    # x: fine across the hull, graded in the far field
    xs = _axis_with_fine_band(x0, x1, 0.0, p.L, spacing, max(far_spacing, spacing * 2.0), growth)
    # y: fine out to the widest section, graded towards the far field
    ys_post = _axis_with_fine_band(0.0, y_half, 0.0, 0.5 * p.B * 1.05, spacing,
                                   max(far_spacing, spacing * 2.0), growth)
    ys = np.concatenate([-ys_post[::-1][:-1], ys_post])
    # z: fine over the draft (real hull), graded below the keel (mirror side)
    zs_top = _axis_with_fine_band(0.0, z_depth, 0.0, p.T, spacing,
                                  max(far_spacing, spacing * 2.0), growth)
    zs = np.concatenate([-zs_top[::-1][:-1], zs_top])

    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    index = lambda i, j, k: i * (ny + 1) * (nz + 1) + j * (nz + 1) + k  # noqa: E731
    points = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)
    centres = np.array([[0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1]),
                         0.5 * (zs[k] + zs[k + 1])]
                        for i in range(nx) for j in range(ny) for k in range(nz)])
    body = shape.inside(centres[:, 0], centres[:, 1], centres[:, 2])
    body_idx = np.flatnonzero(body)
    body_set = set(int(i) for i in body_idx)
    dx = np.diff(xs)
    dy = np.diff(ys)
    dz = np.diff(zs)

    def faces_of(i, j, k):
        q0, q1 = index(i, j, k), index(i + 1, j, k)
        q2, q3 = index(i + 1, j + 1, k), index(i, j + 1, k)
        t0, t1 = index(i, j, k + 1), index(i + 1, j, k + 1)
        t2, t3 = index(i + 1, j + 1, k + 1), index(i, j + 1, k + 1)
        return [
            ((i - 1, j, k), (q0, q3, t3, t0)),   # x-min
            ((i + 1, j, k), (q1, q2, t2, t1)),   # x-max
            ((i, j - 1, k), (q0, q1, t1, t0)),   # y-min
            ((i, j + 1, k), (q2, q3, t3, t2)),   # y-max
            ((i, j, k - 1), (q0, q1, q2, q3)),   # z-min
            ((i, j, k + 1), (t0, t1, t2, t3)),   # z-max
        ]

    cells: List[List[int]] = []
    body_volume = 0.0
    patches: Dict[Tuple[int, ...], str] = {}
    faces_by_cell: Dict[Tuple[int, int, int], List] = {}
    counter = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                flat = i * ny * nz + j * nz + k
                if flat in body_set:
                    body_volume += dx[i] * dy[j] * dz[k]
                    continue
                cells.append([index(i, j, k), index(i + 1, j, k), index(i + 1, j + 1, k),
                              index(i, j + 1, k), index(i, j, k + 1), index(i + 1, j, k + 1),
                              index(i + 1, j + 1, k + 1), index(i, j + 1, k + 1)])
                faces_by_cell[(i, j, k)] = faces_of(i, j, k)
                counter += 1
    n_cells = counter

    cell_set = set(faces_by_cell)
    for (i, j, k), faces in faces_by_cell.items():
        for (ni, nj, nk), verts in faces:
            if (ni, nj, nk) in cell_set:
                continue                     # interior face, shared with a fluid cell
            # NativeMesh3D splits tagged faces into triangles starting at the
            # minimum vertex id, so the tag must be the cell-local cyclic order
            # rotated to start there (a *sorted* tuple would cut the wrong diagonal)
            pivot = verts.index(min(verts))
            key = tuple(verts[pivot:]) + tuple(verts[:pivot])
            if ni < 0 or ni >= nx:
                patches[key] = "inlet" if ni < 0 else "outlet"
            elif nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                patches[key] = "far"
            else:
                patches[key] = "hull"        # neighbour is a removed body cell

    meta = {
        "spacing_m": float(spacing),
        "far_spacing_m": float(far_spacing),
        "nx": float(nx), "ny": float(ny), "nz": float(nz),
        "n_cells": float(n_cells),
        "n_body_cells": float(len(body_idx)),
        "body_volume_m3": float(body_volume),
        "body_volume_ratio": float(body_volume / (2.0 * p.nabla)),
        "L_m": p.L, "B_m": p.B, "T_m": p.T,
    }
    return DoubleBodyMesh(points=points, cells=cells, patches=patches,
                          meta=meta, shape=shape)


# --------------------------------------------------------------------------- #
# Force bookkeeping
# --------------------------------------------------------------------------- #
def drag_components(forces: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Resistance split from the solver's ``forces`` dictionary.

    Convention (pinned by ``tests/test_kvlcc2_double_body.py``): the native 3D
    solver returns ``pressure_force = sum p n_fluid dA`` and
    ``viscous_force = -sum tau dA``, both of which point *upstream* for a body in
    a uniform stream.  The drag components are therefore the negatives of the
    streamwise components, and the module refuses to return a negative friction
    drag (that would mean the sign convention was mixed up).
    """
    for key in ("pressure_force", "viscous_force"):
        if key not in forces:
            raise KeyError(f"forces dictionary must contain {key!r}")
    pressure = -float(np.asarray(forces["pressure_force"], dtype=float)[0])
    viscous = -float(np.asarray(forces["viscous_force"], dtype=float)[0])
    if viscous <= 0.0:
        raise ValueError(
            f"friction drag came out non-positive ({viscous:.6g} N): the surface "
            "normal or the force sign convention is inconsistent"
        )
    return {"pressure_drag_n": pressure, "friction_drag_n": viscous,
            "total_drag_n": pressure + viscous}


def run_double_body_rans(shape: HullShape, speed: float, viscosity: float,
                         mesh: Optional[DoubleBodyMesh] = None,
                         spacing: float = 0.06,
                         density: float = RHO_SEAWATER,
                         wall_treatment: str = "spalding",
                         max_outer_iterations: int = 300,
                         pseudo_time_step: float = 0.2,
                         turbulence: bool = True,
                         rho_air: float = 1.225,
                         net_axe: Optional[object] = None) -> Dict[str, object]:
    """Solve the double-body RANS problem and extract the form factor.

    Returns the forces on the ``hull`` patch, the ``C_F``/``C_Pv``/(1+k) split,
    the reference values used and the convergence history.  ``converged`` is
    reported, never assumed: on a stair-step mesh at an affordable spacing the
    native SST path has been observed to diverge, and a diverged run must not be
    read as a form factor.  The ``net_axe``
    argument accepts an already-built ``(mesh, solver)`` pair so callers can run
    the same mesh with different settings without rebuilding it.
    """
    import numpy as np

    from .native_collocated3d import NativeSteadySettings
    from .native_mesh3d import NativeMesh3D
    from .native_simple3d import NativeSimpleSolver3D
    from .native_rans3d import NativeRansSettings

    if mesh is None:
        mesh = build_double_body_mesh(shape, spacing=spacing)

    # triangulate_faces: the SST wall-distance operator needs actual triangular
    # facets (native_mesh3d refuses warped quads otherwise)
    native = NativeMesh3D.from_cells(mesh.points, mesh.cells,
                                     ["hexahedron"] * len(mesh.cells),
                                     boundary_patches=mesh.patches,
                                     triangulate_faces=True)
    faces = native.face_centers
    n_faces = faces.shape[0]
    boundary = native.neighbor < 0
    labels = native.patch_labels
    # The SST wall model requires the no-slip patch to be *prescribed* Dirichlet
    # data (native_rans3d raises otherwise), so the hull faces are in ``fixed``
    # with zero velocity and the inlet/far-field faces carry the free stream.
    fixed = boundary & (labels != "outlet")
    velocity_bc = np.zeros((n_faces, 3))
    velocity_bc[fixed & (labels != "hull"), 0] = speed
    pressure_dirichlet = boundary & (labels == "outlet")

    settings = NativeSteadySettings(viscosity=viscosity, density=density,
                                    max_iterations=max(50, max_outer_iterations // 3),
                                    nonlinear_tolerance=1e-6, linear_tolerance=1e-8,
                                    pressure_velocity_algorithm="simple")
    if turbulence:
        rans = NativeRansSettings(max_outer_iterations=max_outer_iterations,
                                  pseudo_time_step=pseudo_time_step,
                                  wall_treatment=wall_treatment)
        solver = __import__("tensorfvm.native_rans3d", fromlist=["x"]).NativeSimpleSSTRansSolver3D(
            native, fixed, velocity_bc, pressure_dirichlet=pressure_dirichlet,
            settings=settings, rans_settings=rans, wall_patch="hull")
    else:
        solver = NativeSimpleSolver3D(native, fixed, velocity_bc,
                                      pressure_dirichlet=pressure_dirichlet,
                                      settings=settings)
    if turbulence:
        # SST needs an initialised turbulence state: k = 3/2 (Tu U)^2 with 2 %
        # free-stream intensity, omega from the mixing-length estimate.  The
        # default pseudo-time step is deliberately conservative; see the docs
        # for the divergence this path showed on the affordable coarse mesh.
        k0 = 1.5 * (0.02 * speed) ** 2
        omega0 = math.sqrt(k0) / (0.09 ** 0.25 * 0.05 * shape.principal.L)
        solution = solver.solve_rans(k0, omega0, inlet_k=k0, inlet_omega=omega0)
    else:
        solution = solver.solve()
    forces = solver.forces("hull")
    drag = drag_components(forces)
    wetted = shape.wetted_area()
    q = 0.5 * density * speed ** 2
    split = form_factor_from_double_body(drag["pressure_drag_n"], drag["friction_drag_n"],
                                         q, wetted)
    reynolds = speed * shape.principal.L / viscosity
    return {
        "reynolds": reynolds,
        "speed_ms": speed,
        "viscosity": viscosity,
        "density": density,
        "spacing_m": mesh.meta["spacing_m"],
        "n_cells": mesh.n_cells,
        "wetted_area_m2": wetted,
        "dynamic_pressure_pa": q,
        "drag": drag,
        "split": split,
        "cf_ittc57": ittc57_cf(reynolds),
        "converged": bool(solution.get("numerically_converged",
                                       solution.get("converged", False))),
        "history": solution.get("history", []),
    }