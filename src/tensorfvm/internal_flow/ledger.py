"""Control-volume momentum ledger for the internal-flow benchmark set.

The point of this module is *accounting*: for a converged staggered (MAC) field it
itemises, term by term, the momentum budget of a rectangular control volume::

    convective flux + pressure forces + viscous forces + volume sources
        = residual  ->  0 at a steady state

The control volume is the **fluid** part of the box.  Its surface therefore
consists of

* open cut faces (box boundary inside the fluid): convective flux, pressure and
  normal/shear viscous stress integrated there, with the face pressure taken as the
  two-sided average (second order);
* wetted solid faces, *including the solid faces inside the box*: a sudden
  expansion has an inlet channel whose walls are interior to a full-height box, and
  a ledger that only integrates the four outer box faces misses that wall shear
  entirely (measured defect: 30-50 % spurious residual, see the report);
* physical in/outflow boundaries, treated as open faces.

Every force uses the same discrete geometry as the flow solver -- half-cell wall
distance for a tangential wall shear (``tau = 2 mu u_adjacent / d``), a full cell
distance for the normal stress between two staggered face velocities -- so the
residual measures the solution's fidelity to the momentum balance rather than a
post-processing convention.

All functions are pure: a stored field can be re-graded without a solver run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

EPS = 1e-300


@dataclass(frozen=True)
class Ledger:
    """Itemised momentum budget of one control volume (per unit depth)."""

    component: str
    convective: float
    pressure: float
    viscous: float
    buoyancy: float
    residual: float
    scale: float
    terms: dict = field(default_factory=dict)

    @property
    def relative(self) -> float:
        return abs(self.residual) / max(self.scale, EPS)

    def as_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "convective": self.convective,
            "pressure": self.pressure,
            "viscous": self.viscous,
            "buoyancy": self.buoyancy,
            "residual": self.residual,
            "scale": self.scale,
            "relative_residual": self.relative,
            "terms": dict(self.terms),
        }


def _wall_forces_x(u: torch.Tensor, p: torch.Tensor, fluid: torch.Tensor,
                   dx: float, dy: float, rho: float, nu: float,
                   box: tuple[int, int, int, int],
                   open_faces: tuple[bool, bool] = (False, False)
                   ) -> tuple[float, float, float, dict]:
    """x-forces of every wetted face whose fluid side lies inside ``box``.

    Returns ``(shear, wall_pressure, normal_stress, counts)``: tangential wall
    shear on horizontal faces, pressure on vertical faces (sign set by which side
    is solid) and the viscous normal stress on vertical faces.
    """
    ny, nx = fluid.shape
    j0, j1, i0, i1 = box
    shear = 0.0
    n_shear = 0
    zero_rows = torch.zeros(i1 - i0, dtype=torch.bool)
    for j in range(j0, j1 + 1):
        above = fluid[j, i0:i1] if j < ny else zero_rows
        below = fluid[j - 1, i0:i1] if j > 0 else zero_rows
        wall = above ^ below
        if not bool(wall.any()):
            continue
        row = min(j, ny - 1) if j < ny else ny - 1
        adj = torch.where(above, u[row, i0:i1], u[max(j - 1, 0), i0:i1])
        shear += float((-2.0 * rho * nu * adj / dy * dx)[wall].sum())
        n_shear += int(wall.sum())
    wall_pressure = 0.0
    normal_stress = 0.0
    n_wall = 0
    zero_cols = torch.zeros(j1 - j0, dtype=torch.bool)
    for i in range(i0, i1 + 1):
        if (i == 0 and open_faces[0]) or (i == nx and open_faces[1]):
            continue   # physical inlet/outlet: an open cut face, handled above
        east = fluid[j0:j1, i] if i < nx else zero_cols
        west = fluid[j0:j1, i - 1] if i > 0 else zero_cols
        wall = east ^ west
        if not bool(wall.any()):
            continue
        p_adj = torch.where(east, p[j0:j1, min(i, nx - 1)],
                            p[j0:j1, max(i - 1, 0)])
        sign = torch.where(east, torch.ones_like(p_adj), -torch.ones_like(p_adj))
        wall_pressure += float((sign * p_adj * dy)[wall].sum())
        col_east = min(i + 1, nx)
        col_west = max(i - 1, 0)
        u_adj = torch.where(east, u[j0:j1, col_east], u[j0:j1, col_west])
        normal_stress += float((-2.0 * rho * nu * u_adj / dx * dy)[wall].sum())
        n_wall += int(wall.sum())
    return shear, wall_pressure, normal_stress, {"n_shear_faces": n_shear,
                                                 "n_vertical_wall_faces": n_wall}


def _ledger_x(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, fluid: torch.Tensor,
              dx: float, dy: float, rho: float, nu: float,
              box: tuple[int, int, int, int],
              source: torch.Tensor | None = None,
              open_faces: tuple[bool, bool] = (False, False)) -> Ledger:
    """x-momentum budget of the cell box ``(j0, j1, i0, i1)``."""
    ny, nx = fluid.shape
    j0, j1, i0, i1 = box
    if not (0 <= j0 < j1 <= ny and 0 <= i0 < i1 <= nx):
        raise ValueError(f"box {box} outside the {ny}x{nx} grid")
    f = fluid
    inside = f[j0:j1, i0:i1]

    # ------------------------------------------------ open vertical cut faces
    west_inside = f[j0:j1, i0]
    uf_w = u[j0:j1, i0]
    if i0 > 0:
        open_w = west_inside & f[j0:j1, i0 - 1]
        p_w = torch.where(open_w, 0.5 * (p[j0:j1, i0 - 1] + p[j0:j1, i0]),
                          p[j0:j1, i0])
    elif open_faces[0]:
        open_w = west_inside.clone()          # physical inlet boundary
        p_w = p[j0:j1, i0]
    else:
        open_w = torch.zeros_like(west_inside)  # closed west boundary (wall)
        p_w = p[j0:j1, i0]
    conv_in = float((rho * uf_w ** 2 * dy)[open_w].sum())
    p_in_open = float((p_w * dy)[open_w].sum())
    visc_w = 0.0
    if i0 > 0:
        visc_w = float((-2.0 * rho * nu * (uf_w - u[j0:j1, i0 - 1]) / dx * dy
                        )[open_w].sum())

    east_inside = f[j0:j1, i1 - 1]
    uf_e = u[j0:j1, i1]
    if i1 < nx:
        open_e = east_inside & f[j0:j1, i1]
        p_e = torch.where(open_e, 0.5 * (p[j0:j1, i1 - 1] + p[j0:j1, i1]),
                          p[j0:j1, i1 - 1])
    elif open_faces[1]:
        open_e = east_inside.clone()          # physical outflow boundary
        p_e = p[j0:j1, i1 - 1]
    else:
        open_e = torch.zeros_like(east_inside)  # closed east boundary (wall)
        p_e = p[j0:j1, i1 - 1]
    conv_out = float((rho * uf_e ** 2 * dy)[open_e].sum())
    p_out = float((p_e * dy)[open_e].sum())
    visc_e = float((2.0 * rho * nu * (uf_e - u[j0:j1, i1 - 1]) / dx * dy
                    )[open_e].sum())

    # ---------------------------------------------- open transverse cut faces
    conv_t = 0.0
    visc_s = 0.0
    visc_n = 0.0
    if j0 > 0:
        south_open = f[j0, i0:i1] & f[j0 - 1, i0:i1]
        uf = 0.5 * (u[j0, i0:i1] + u[j0, i0 + 1:i1 + 1])
        conv_t += float((-rho * uf * v[j0, i0:i1] * dx)[south_open].sum())
        visc_s = float((-rho * nu * (u[j0, i0:i1] - u[j0 - 1, i0:i1]) / dy * dx
                        )[south_open].sum())
    if j1 < ny:
        north_open = f[j1 - 1, i0:i1] & f[j1, i0:i1]
        uf = 0.5 * (u[j1, i0:i1] + u[j1, i0 + 1:i1 + 1])
        conv_t += float((rho * uf * v[j1, i0:i1] * dx)[north_open].sum())
        visc_n = float((rho * nu * (u[j1, i0:i1] - u[j1 - 1, i0:i1]) / dy * dx
                        )[north_open].sum())

    # ------------------------------------------------------ wetted solid faces
    shear, wall_p, wall_n, counts = _wall_forces_x(u, p, f, dx, dy, rho, nu, box,
                                                   open_faces)

    buoy = 0.0
    if source is not None:
        buoy = float((source[j0:j1, i0:i1] * dx * dy)[inside].sum())

    convective = conv_in - conv_out + conv_t
    pressure = p_in_open - p_out + wall_p
    viscous = visc_w + visc_e + visc_s + visc_n + shear + wall_n
    residual = convective + pressure + viscous + buoy
    terms = {
        "conv_in": conv_in, "conv_out": conv_out, "conv_transverse": conv_t,
        "p_in_open": p_in_open, "p_out": p_out, "p_wall": wall_p,
        "visc_cut_west": visc_w, "visc_cut_east": visc_e,
        "visc_cut_south": visc_s, "visc_cut_north": visc_n,
        "visc_wall_shear": shear, "visc_wall_normal": wall_n,
        "volume_source": buoy, "fluid_cells": int(inside.sum()),
        **counts,
    }
    scale = max(abs(convective), abs(pressure), abs(viscous), abs(buoy), EPS)
    return Ledger("x", convective, pressure, viscous, buoy, residual, scale, terms)


def momentum_ledger(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor,
                    fluid: torch.Tensor, dx: float, dy: float, nu: float, *,
                    component: str = "x", rho: float = 1.0,
                    box: tuple[int, int, int, int] | None = None,
                    source: torch.Tensor | None = None,
                    open_faces: tuple[bool, bool] = (False, False)) -> Ledger:
    """Momentum budget of ``box`` (default: the whole domain) along ``component``."""
    ny, nx = fluid.shape
    if box is None:
        box = (0, ny, 0, nx)
    if component == "x":
        return _ledger_x(u, v, p, fluid, dx, dy, rho, nu, box, source=source,
                         open_faces=open_faces)
    if component != "y":
        raise ValueError("component must be 'x' or 'y'")
    src = None if source is None else source.T.contiguous()
    led = _ledger_x(v.T.contiguous(), u.T.contiguous(), p.T.contiguous(),
                    fluid.T.contiguous(), dy, dx, rho, nu,
                    (box[2], box[3], box[0], box[1]), source=src,
                    open_faces=(open_faces[1], open_faces[0]))
    return Ledger("y", led.convective, led.pressure, led.viscous, led.buoyancy,
                  led.residual, led.scale, led.terms)


# ------------------------------------------------------------------ stations
def station_pressure(p: torch.Tensor, fluid: torch.Tensor, column: int) -> float:
    """Area-averaged static pressure of cell column ``column``."""
    mask = fluid[:, column]
    height = float(mask.sum())
    if height <= 0:
        raise ValueError(f"column {column} has no fluid")
    return float(p[:, column][mask].sum()) / height


def station_flux(u: torch.Tensor, fluid: torch.Tensor, face: int, dy: float) -> float:
    """Volume flux through vertical face ``face`` (per unit depth)."""
    nx = fluid.shape[1]
    rows = fluid[:, min(face, nx - 1)]
    return float(u[:, face][rows].sum()) * dy


def station_height(fluid: torch.Tensor, column: int, dy: float) -> float:
    """Wetted height of cell column ``column``."""
    return float(fluid[:, column].sum()) * dy