"""Fluid/solid (here ice/solid) conjugate heat transfer on layered walls.

Three solvers, deliberately independent of each other so that they can audit one
another:

``LayeredWall``
    single-grid cell-centred finite-volume steady conduction through a stack of
    layers, with a *contact resistance* on the layer faces.  The face
    conductance is the series combination of the two half cells and ``R_c``::

        C_f = 1 / ( dx_P/(2 k_P) + R_c + dx_N/(2 k_N) )

``LayeredWall2D``
    the same wall extended laterally (per-column ice thickness / heat transfer
    coefficient), used for the field figures.  Lateral conduction is resolved, so
    a laterally uniform case must reproduce the 1-D answer.

``TwoDomainCHT``
    the ice and the steel are solved as *two separate* domains, coupled only by
    the interface heat flux ``q`` and the contact condition
    ``T_ice(0) - T_steel(0) = q R_c`` (Dirichlet-Neumann iteration).  Interface
    continuity is therefore an *earned* property: it is a fixed point of the
    coupling, not something the discretisation grants for free.

``interface_audit`` re-derives both one-sided fluxes from the converged fields
with an independent one-sided difference (no reuse of the assembled face
conductance) and reports

* ``flux_imbalance_rel``          ``|q_ice - q_steel| / |q|``           (limit 1e-6)
* ``temperature_continuity_rel``  ``|jump - q R_c| / dT_drive``         (limit 1e-2)
* ``jump_rel``                    ``|jump - q R_c| / |q R_c|``          (limit 1e-2)

``TwoDomainCHT`` accepts a ``mutation`` argument that injects a realistic
regression (single coupling sweep, flipped interface flux, decoupled steel).  The
verification tests assert that the clean run passes and that every mutation turns
the flux-continuity criterion red.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "ICE",
    "STEEL",
    "LayeredWall",
    "LayeredWall2D",
    "Material",
    "SurfaceBC",
    "TwoDomainCHT",
    "analytic_series_field",
    "interface_audit",
]


@dataclass(frozen=True)
class Material:
    name: str
    k: float
    rho: float = 1.0
    c: float = 1.0

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.c)


ICE = Material("ice", 2.2, 917.0, 2100.0)
STEEL = Material("steel", 50.0, 7850.0, 500.0)


@dataclass(frozen=True)
class SurfaceBC:
    """Boundary flux ``q = q0 + (t_ext - T_face) / resistance`` (positive inward).

    ``resistance = 0`` imposes the wall value at the face, ``1/h`` is a
    convective (Robin) condition and ``inf`` a prescribed heat flux ``q0``.
    """

    t_ext: float = 0.0
    resistance: float = 0.0
    q0: float = 0.0

    def flux(self, t_face: float) -> float:
        if math.isinf(self.resistance):
            return self.q0
        return self.q0 + (self.t_ext - t_face) / self.resistance


@dataclass
class LayeredWall:
    """Steady 1-D conduction across ``layers`` = [(thickness, material, n_cells), ...]."""

    layers: list
    r_contacts: list = field(default_factory=list)   # one entry per internal face
    t_top: SurfaceBC = field(default_factory=SurfaceBC)
    t_bottom: SurfaceBC = field(default_factory=SurfaceBC)

    def __post_init__(self) -> None:
        self.dx = []
        self.k = []
        self.layer_of = []
        self.face_bc_thickness = []
        for i, (thickness, material, n) in enumerate(self.layers):
            if n < 1:
                raise ValueError("each layer needs at least one cell")
            self.dx.extend([float(thickness) / n] * n)
            self.k.extend([float(material.k)] * n)
            self.layer_of.extend([i] * n)
        self.dx = np.asarray(self.dx)
        self.k = np.asarray(self.k)
        self.layer_of = np.asarray(self.layer_of)
        self.r_contacts = list(self.r_contacts)
        if self.r_contacts and len(self.r_contacts) != len(self.layers) - 1:
            raise ValueError("need one contact resistance per internal face")
        x = np.concatenate(([0.0], np.cumsum(self.dx)))
        self.x_faces = x
        self.xc = 0.5 * (x[1:] + x[:-1])
        self._build()

    # ---------------------------------------------------------------- geometry
    def _build(self) -> None:
        n = self.dx.size
        cond = np.zeros(n - 1)
        for j in range(n - 1):
            left = 0.5 * self.dx[j] / self.k[j]
            right = 0.5 * self.dx[j + 1] / self.k[j + 1]
            r_c = 0.0
            if self.layer_of[j] != self.layer_of[j + 1]:
                r_c = self.r_contacts[self.layer_of[j]]
            cond[j] = 1.0 / (left + r_c + right)
        self.face_cond = cond
        self.r_half_top = 0.5 * self.dx[0] / self.k[0]
        self.r_half_bottom = 0.5 * self.dx[-1] / self.k[-1]
        self.c_top = 0.0 if math.isinf(self.t_top.resistance) else \
            1.0 / (self.r_half_top + self.t_top.resistance)
        self.c_bottom = 0.0 if math.isinf(self.t_bottom.resistance) else \
            1.0 / (self.r_half_bottom + self.t_bottom.resistance)

    def interface_face(self, layer_left: int) -> int:
        """Index of the first cell of ``layer_left + 1`` (the interface face)."""
        return int(np.nonzero(self.layer_of > layer_left)[0][0])

    def _refresh_boundaries(self) -> None:
        """Recompute the boundary conductances from the *current* BCs."""
        self.c_top = 0.0 if math.isinf(self.t_top.resistance) else \
            1.0 / (self.r_half_top + self.t_top.resistance)
        self.c_bottom = 0.0 if math.isinf(self.t_bottom.resistance) else \
            1.0 / (self.r_half_bottom + self.t_bottom.resistance)

    def solve(self) -> np.ndarray:
        self._refresh_boundaries()
        n = self.dx.size
        diag = np.zeros(n)
        diag[:-1] += self.face_cond
        diag[1:] += self.face_cond
        upper = -self.face_cond
        lower = -self.face_cond
        diag[0] += self.c_top
        diag[-1] += self.c_bottom
        rhs = np.zeros(n)
        rhs[0] = self.c_top * self.t_top.t_ext + self.t_top.q0
        rhs[-1] = self.c_bottom * self.t_bottom.t_ext + self.t_bottom.q0
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        t = sla.solve_banded((1, 1), ab, rhs)
        self.temperature = t
        return t


def interface_audit(t_ice: np.ndarray, dx_ice: float, k_ice: float,
                    t_steel: np.ndarray, dx_steel: float, k_steel: float,
                    r_contact: float, dT_drive: float) -> dict:
    """Independent audit of an ice/steel interface (one-sided reconstruction).

    The two one-sided fluxes come from a *backward/forward difference on the two
    outermost cell centres of each side*, i.e. from the converged temperature
    field only -- the assembled face conductance is deliberately not reused.
    """
    t_ice = np.asarray(t_ice, dtype=float)
    t_steel = np.asarray(t_steel, dtype=float)
    q_ice = k_ice * (t_ice[-2] - t_ice[-1]) / dx_ice
    q_steel = k_steel * (t_steel[0] - t_steel[1]) / dx_steel
    q = 0.5 * (q_ice + q_steel)
    t_if_ice = t_ice[-1] - q_ice * 0.5 * dx_ice / k_ice
    t_if_steel = t_steel[0] + q_steel * 0.5 * dx_steel / k_steel
    jump = float(t_if_ice - t_if_steel)
    expected = q * r_contact
    scale = max(abs(q_ice), abs(q_steel), 1e-30)
    dev = abs(jump - expected)
    return {
        "q_ice": float(q_ice),
        "q_steel": float(q_steel),
        "q_mean": float(q),
        "flux_imbalance_rel": float(abs(q_ice - q_steel) / scale),
        "t_iface_ice": float(t_if_ice),
        "t_iface_steel": float(t_if_steel),
        "t_iface_jump": jump,
        "t_iface_jump_expected": float(expected),
        "temperature_continuity_rel": float(dev / max(abs(dT_drive), 1e-30)),
        "jump_rel": float(dev / max(abs(expected), 1e-30)),
        "interface_area_resistance_error_rel": float(
            abs((t_if_ice - t_if_steel) / max(q, 1e-30) - r_contact)
            / max(abs(r_contact), 1e-30)),
    }


@dataclass
class TwoDomainCHT:
    """Ice and steel solved as two domains coupled by flux + contact condition."""

    ice: LayeredWall
    steel: LayeredWall
    r_contact: float = 1.0e-4
    q_init: float = 0.0
    mutation: str | None = None

    def _solve_ice(self, q: float) -> np.ndarray:
        wall = self.ice
        wall.t_bottom = SurfaceBC(q0=-q, resistance=math.inf)   # flux leaving the ice
        return wall.solve()

    def _solve_steel(self, q: float) -> np.ndarray:
        wall = self.steel
        if self.mutation == "steel_decoupled":
            q = 0.0
        if self.mutation == "sign_flipped":
            q = -q
        wall.t_top = SurfaceBC(q0=q, resistance=math.inf)       # flux entering the steel
        return wall.solve()

    def solve(self, relax: float = 0.5, tol: float = 1.0e-12,
              max_iter: int = 200) -> dict:
        max_iter = 1 if self.mutation == "single_sweep" else max_iter
        q = float(self.q_init)
        dx_i, k_i = self.ice.dx[0], self.ice.k[0]
        dx_s, k_s = self.steel.dx[0], self.steel.k[0]
        t_ice = t_steel = None
        history = []
        for it in range(max_iter):
            t_ice = self._solve_ice(q)
            t_steel = self._solve_steel(q)
            t_if_ice = t_ice[-1] - q * 0.5 * dx_i / k_i
            t_if_steel = t_steel[0] + q * 0.5 * dx_s / k_s
            r = (t_if_ice - t_if_steel) - q * self.r_contact
            history.append({"iteration": it, "q": q, "residual": float(r)})
            if abs(r) <= tol:
                break
            # secant slope from a perturbed solve (the operator is affine in q)
            dq = max(1.0, abs(q) * 1.0e-3)
            t_ice_p = self._solve_ice(q + dq)
            t_steel_p = self._solve_steel(q + dq)
            r_p = ((t_ice_p[-1] - (q + dq) * 0.5 * dx_i / k_i)
                   - (t_steel_p[0] + (q + dq) * 0.5 * dx_s / k_s)
                   - (q + dq) * self.r_contact)
            slope = (r_p - r) / dq
            q -= relax * r / slope
        self.q = q
        return {
            "q": float(q),
            "iterations": len(history),
            "residual": float(history[-1]["residual"]),
            "history": history,
            "t_ice": t_ice,
            "t_steel": t_steel,
        }


def analytic_series_field(thicknesses, conductivities, t_surface: float,
                          t_back: float, r_contact: float,
                          n_cells: int = 200, contact_faces=None) -> dict:
    """Exact 1-D series-resistance field (piecewise linear per layer).

    Returns the temperature at ``n_cells`` points through the stack, the flux and
    the temperatures on both sides of every contact face.
    """
    layers = list(zip(thicknesses, conductivities))
    resistances = [t / k for t, k in layers]
    n_faces = len(layers) - 1
    contact_faces = list(range(n_faces)) if contact_faces is None else list(contact_faces)
    r_total = sum(resistances) + len(contact_faces) * r_contact
    q = (t_surface - t_back) / r_total
    total = sum(thicknesses)
    x = np.linspace(0.0, total, n_cells)
    t = np.empty_like(x)
    x_face = 0.0
    t_now = t_surface
    faces = []
    for i, (thickness, k) in enumerate(layers):
        r = thickness / k
        inside = (x >= x_face - 1e-15) & (x <= x_face + thickness + 1e-15)
        t[inside] = t_now - q * (x[inside] - x_face) / k
        x_face += thickness
        t_now -= q * r
        if i < n_faces:
            if i in contact_faces:
                t_ice_side = t_now
                t_now -= q * r_contact
                faces.append({
                    "layer": i,
                    "x": x_face,
                    "t_left": float(t_ice_side),
                    "t_right": float(t_now),
                    "q": float(q),
                })
    return {
        "x": x,
        "temperature": t,
        "q": float(q),
        "r_total": float(r_total),
        "contacts": faces,
        "t_at_back": float(t_now),
    }


@dataclass
class LayeredWall2D:
    """Steady 2-D conduction through a layered wall (thickness x, lateral y)."""

    layers: list
    r_contacts: list = field(default_factory=list)
    t_bottom: SurfaceBC = field(default_factory=SurfaceBC)
    top_bc: list = field(default_factory=list)      # one SurfaceBC per column
    ny: int = 1

    def __post_init__(self) -> None:
        self.nx_cells = sum(n for _, _, n in self.layers)
        self.dx = np.concatenate([[th / n] * n for th, _, n in self.layers])
        self.k = np.concatenate([[m.k] * n for _, m, n in self.layers])
        self.layer_of = np.concatenate(
            [[i] * n for i, (_, _, n) in enumerate(self.layers)])
        self.xc = np.cumsum(self.dx) - 0.5 * self.dx
        self.ny = max(1, int(self.ny))
        self.yc = (np.arange(self.ny) + 0.5) / self.ny
        if not self.top_bc:
            self.top_bc = [SurfaceBC()] * self.ny
        if len(self.top_bc) != self.ny:
            raise ValueError("need one top boundary condition per column")
        # lateral spacing: unit width
        self.dy = 1.0 / self.ny

    def solve(self) -> np.ndarray:
        nx, ny = self.nx_cells, self.ny
        dx, k = self.dx, self.k
        n = nx * ny
        rows, cols, vals = [], [], []
        rhs = np.zeros(n)

        def idx(i, j):
            return j * nx + i

        for j in range(ny):
            for i in range(nx):
                p = idx(i, j)
                # through-thickness neighbours
                for nb in (i - 1, i + 1):
                    if 0 <= nb < nx:
                        left = 0.5 * dx[min(i, nb)] / k[min(i, nb)]
                        right = 0.5 * dx[max(i, nb)] / k[max(i, nb)]
                        r_c = 0.0
                        if self.layer_of[min(i, nb)] != self.layer_of[max(i, nb)]:
                            r_c = self.r_contacts[self.layer_of[min(i, nb)]]
                        c = self.dy / (left + r_c + right)
                        rows.append(p); cols.append(idx(nb, j)); vals.append(-c)
                        rows.append(p); cols.append(p); vals.append(c)
                # lateral neighbours (harmonic mean within a layer, no contact R)
                for nb in (j - 1, j + 1):
                    if 0 <= nb < ny:
                        # lateral face: area dx*1, distance dy
                        c = k[i] * dx[i] / self.dy
                        rows.append(p); cols.append(idx(i, nb)); vals.append(-c)
                        rows.append(p); cols.append(p); vals.append(c)
                if i == 0:
                    bc = self.top_bc[j]
                    if math.isinf(bc.resistance):
                        rhs[p] += bc.q0 * self.dy
                    else:
                        c = self.dy / (0.5 * dx[0] / k[0] + bc.resistance)
                        rows.append(p); cols.append(p); vals.append(c)
                        rhs[p] += c * bc.t_ext + bc.q0 * self.dy
        for j in range(ny):
            p = idx(nx - 1, j)
            bc = self.t_bottom
            if math.isinf(bc.resistance):
                rhs[p] += bc.q0
            else:
                c = self.dy / (0.5 * dx[-1] / k[-1] + bc.resistance)
                rows.append(p); cols.append(p); vals.append(c)
                rhs[p] += c * bc.t_ext + bc.q0 * self.dy
        a = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        t = spla.spsolve(a, rhs)
        self.temperature = t.reshape(ny, nx)
        return self.temperature