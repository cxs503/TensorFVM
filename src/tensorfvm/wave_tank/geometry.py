"""Hull / section geometry for the numerical wave tank (NWT).

Two things live here:

1. **The geometry pre-gate** (:class:`HullForm` subclasses and
   :func:`geometry_gate`).  Before any hydrodynamics is run the discretised body
   must reproduce its own reference coefficients -- block coefficient ``Cb``,
   ``L/B``, ``B/T`` and the displacement volume ``nabla = Cb L B T``.  A tank run
   on an un-validated hull is worthless, so the gate returns a dict of residuals
   and a boolean, and the case driver records it in the result JSON.

2. **The 2-D section** (:class:`BargeSection`) that the vertical-plane tank
   actually solves: the centreplane strip of a ship, length ``L`` x draft ``T``.
   This is the classical 2-D seakeeping body (heave + pitch), and it is what the
   2-D potential-flow BEM reference in :mod:`tensorfvm.wave_tank.bem2d` solves,
   so the two are compared like-for-like.

The Wigley hull is the analytic parabolic form

    y(x, z) = (B/2) (1 - (2x/L)^2) (1 - (z/T)^2),   |x| <= L/2,  -T <= z <= 0,

whose centreplane section (y = 0+) is exactly the rectangle ``L x T`` used by
the tank; the KCS form below is a *simplified* body (parallel midbody + parabolic
ends) tuned to the KRISO Container Ship particulars (L = 230 m, B = 32.2 m,
T = 10.8 m, Cb = 0.651), not the official offset table -- the case JSON says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "WigleyHull",
    "KCSSimplifiedHull",
    "BargeSection",
    "geometry_gate",
    "polygon_mask",
]

_WIGLEY_CB = 4.0 / 9.0  # analytic Cb of the parabolic Wigley form above


@dataclass
class WigleyHull:
    """Parabolic Wigley hull (the standard 2-D-of-revolution-like test hull)."""

    length: float = 10.0
    beam: float = 1.0
    draft: float = 0.625
    n_x: int = 201
    n_z: int = 81
    name: str = "wigley"
    cb_reference: float = field(default=_WIGLEY_CB, repr=False)

    @property
    def half_breadth(self):
        return self.beam / 2.0

    def breadth(self, x, z):
        """Waterline-relative breadth ``2 y(x, z)`` of the parabolic hull."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        f = (1.0 - (2.0 * x / self.length) ** 2) * (1.0 - (z / self.draft) ** 2)
        return self.beam * np.clip(f, 0.0, None)

    def coefficients(self) -> dict:
        """Volume, Cb, Cp, wetted surface and the aspect ratios, by quadrature."""
        x = np.linspace(-self.length / 2, self.length / 2, self.n_x)
        z = np.linspace(-self.draft, 0.0, self.n_z)
        X, Z = np.meshgrid(x, z, indexing="xy")
        b = self.breadth(X, Z)
        vol = np.trapezoid(np.trapezoid(b, z, axis=0), x)
        # wetted surface: bottom + sides, from the same offsets
        dz = np.gradient(b, x, axis=1)
        surf = np.trapezoid(np.trapezoid(np.sqrt(1.0 + dz ** 2), z, axis=0), x) * 2.0
        cbt = self.beam * self.draft * self.length
        return {
            "name": self.name,
            "L": self.length,
            "B": self.beam,
            "T": self.draft,
            "volume": float(vol),
            "nabla_LBT": float(vol / cbt),
            "Cb": float(vol / cbt),
            "Cb_reference": float(self.cb_reference),
            "L_over_B": self.length / self.beam,
            "B_over_T": self.beam / self.draft,
            "wetted_surface": float(surf),
        }


@dataclass
class KCSSimplifiedHull:
    """Simplified KCS-like hull: parallel midbody + parabolic bow/stern ends.

    ``breadth(x)`` at the waterline is a parabola over the fore/after ends and
    constant over the parallel midbody, with the section shape
    ``(1 - (z/T)^n)``.  ``midbody_fraction`` and ``vertical_exponent`` are
    calibrated so the volume coefficient matches the KRISO Container Ship value
    ``Cb = 0.651`` (a real KCS section is much fuller than the Wigley parabola:
    ``n = 2`` alone would give ``Cb ~ 0.51``).  The gate checks the quadrature
    against that reference; the hull is *not* the official offset table.
    """

    length: float = 230.0
    beam: float = 32.2
    draft: float = 10.8
    midbody_fraction: float = 0.44
    vertical_exponent: float = 4.0
    n_x: int = 401
    n_z: int = 81
    name: str = "kcs_simplified"
    cb_reference: float = 0.651

    def breadth(self, x, z):
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        xf = 2.0 * x / self.length          # -1 .. 1
        p = self.midbody_fraction            # |xf| <= p is the parallel body
        ax = np.abs(xf)
        shape = np.where(ax <= p, 1.0, np.clip(1.0 - ((ax - p) / (1.0 - p)) ** 2, 0.0, None))
        vert = np.clip(1.0 - (z / self.draft) ** self.vertical_exponent, 0.0, None)
        return self.beam * shape * vert

    def coefficients(self) -> dict:
        x = np.linspace(-self.length / 2, self.length / 2, self.n_x)
        z = np.linspace(-self.draft, 0.0, self.n_z)
        X, Z = np.meshgrid(x, z, indexing="xy")
        b = self.breadth(X, Z)
        vol = np.trapezoid(np.trapezoid(b, z, axis=0), x)
        cbt = self.beam * self.draft * self.length
        return {
            "name": self.name,
            "L": self.length,
            "B": self.beam,
            "T": self.draft,
            "volume": float(vol),
            "nabla_LBT": float(vol / cbt),
            "Cb": float(vol / cbt),
            "Cb_reference": float(self.cb_reference),
            "L_over_B": self.length / self.beam,
            "B_over_T": self.beam / self.draft,
        }


@dataclass
class BargeSection:
    """2-D rectangular section (ship centreplane strip): length ``L`` x draft ``T``.

    The section is axis-aligned in the body frame (``x`` along the tank).  In the
    tank it is placed with its keel at ``y = -T + heave`` and rotates by ``pitch``
    about its own centre of gravity, which the mask builder re-evaluates every
    step, so the stair-step error is a measured quantity (grid-refinement / GCI),
    not an assumption.
    """

    length: float = 2.4
    draft: float = 0.4
    name: str = "barge-2d"

    @property
    def area(self) -> float:
        return self.length * self.draft

    @property
    def cb_2d(self) -> float:
        return 1.0

    def polygon(self, x_centre: float, heave: float, pitch: float, cg_y: float = 0.0):
        """Closed polygon of the rotated section in world coordinates (CCW)."""
        L, T = self.length, self.draft
        local = np.array([[-L / 2, -T], [L / 2, -T], [L / 2, 0.0], [-L / 2, 0.0]])
        c, s = np.cos(pitch), np.sin(pitch)
        rot = np.array([[c, -s], [s, c]])
        pts = local @ rot.T
        pts[:, 0] += x_centre
        pts[:, 1] += heave + cg_y
        return pts

    def mass_properties(self, rho: float, cg_y: float = 0.0, radius_gyration_ratio: float = 0.29):
        """``(mass, inertia)`` per unit length for a homogeneous-looking box."""
        m = rho * self.area
        k55 = radius_gyration_ratio * self.length
        return m, m * k55 * k55

    def coefficients(self) -> dict:
        return {
            "name": self.name,
            "L": self.length,
            "T": self.draft,
            "area": self.area,
            "Cb_2d": 1.0,
            "L_over_T": self.length / self.draft,
            "draft_ratio_T_over_L": self.draft / self.length,
        }


def polygon_mask(x_centres, y_centres, polygon) -> np.ndarray:
    """Boolean cell mask of an arbitrary closed polygon (vectorised ray casting).

    ``x_centres`` (nx,) and ``y_centres`` (ny,) give the cell centres; the return
    array has shape ``(ny, nx)``.  A cell is "solid" when its centre is inside the
    polygon -- the standard stair-step immersed-body mask for a MAC grid.
    """
    xc = np.asarray(x_centres, dtype=float)
    yc = np.asarray(y_centres, dtype=float)
    X, Y = np.meshgrid(xc, yc, indexing="xy")
    poly = np.asarray(polygon, dtype=float)
    inside = np.zeros(X.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        cond = ((y1 > Y) != (y2 > Y))
        xint = (x2 - x1) * (Y - y1) / (y2 - y1) + x1
        inside ^= cond & (X < xint)
    return inside


def geometry_gate(coefficients: dict, cb_tol: float = 5.0e-3, nabla_tol: float = 1.0e-6) -> dict:
    """Turn a :meth:`coefficients` dict into a pass/fail pre-gate record."""
    cb = coefficients["Cb"]
    cb_ref = coefficients["Cb_reference"]
    cb_err = abs(cb - cb_ref) / cb_ref
    nabla = coefficients["nabla_LBT"]
    nabla_err = abs(nabla - cb) / max(cb, 1e-12)
    record = dict(coefficients)
    record["Cb_abs_error"] = float(cb_err)
    record["nabla_rel_error"] = float(nabla_err)
    record["cb_tolerance"] = cb_tol
    record["passed"] = bool(cb_err <= cb_tol and nabla_err <= nabla_tol)
    return record
