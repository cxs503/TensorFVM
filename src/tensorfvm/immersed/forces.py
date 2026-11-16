"""Integral force extraction for immersed-solid runs: a momentum budget.

The masked projection arm never applies an explicit force to the body (the
solid faces are *hard constraints*), so the hydrodynamic force has to be
recovered from the flow field.  This module integrates the momentum balance of
a fixed rectangular control volume ``V`` whose boundary ``S`` lies in the fluid
and which fully encloses the body:

    F_body = d/dt int_V rho*u dV + oint_S [ rho*u (u.n) + p n - tau.n ] dS

with ``n`` the outward normal of ``V``.  Every term is a sum over grid-aligned
faces of the staggered MAC grid (``u`` on x-faces, ``v`` on y-faces, ``p`` at
cell centres), so no interpolation of the primary unknowns to the body surface
is needed and the result is independent of how the mask staircases the body.

The unsteady term is a backward difference of the volume momentum, so the
caller drives :class:`ForceRecorder` once per time step (or once per sample).

Conventions follow :mod:`tensorfvm.immersed.geometry`: ``u[j, i]`` lives at
``x = x0 + i*dx``, ``v[j, i]`` at ``y = y0 + j*dy``, cells at half-offsets.
Signs are exercised by ``tests/test_immersed_forces.py``: a uniform pressure
field gives zero force, and a linear pressure field ``p = -G x`` closing a
body-free box returns exactly ``-G * V`` (the divergence theorem limit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .geometry import CartesianGrid
from ..registry import capability


def cell_velocity(flow) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell-centred velocity ``(u_c, v_c)`` of an :class:`ImmersedFlow`."""
    uc = 0.5 * (flow.u[:, :-1] + flow.u[:, 1:])
    vc = 0.5 * (flow.v[:-1, :] + flow.v[1:, :])
    return uc, vc


@dataclass(frozen=True)
@capability('fvm.immersed.control_box',
            kind='module',
            maturity='developing',
            test_path='tests/test_cylinder_re100_case.py',
            wired=True,
            tested=True,
            verified=False,
            evidence={'grade': 'MECHANICS', 'docs': 0})
class ControlBox:
    """Index bounds of a rectangular control volume, on grid faces.

    ``i0``/``i1`` are ``u``-face indices (west/east boundary of ``V``) and
    ``j0``/``j1`` are ``v``-face indices (south/north boundary of ``V``).
    """

    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def cell_slice(self) -> tuple[slice, slice]:
        return (slice(self.j0, self.j1), slice(self.i0, self.i1))

    def as_dict(self) -> dict[str, int]:
        return {"i0": self.i0, "i1": self.i1, "j0": self.j0, "j1": self.j1}


def control_box(grid: CartesianGrid, *, x_min: float, x_max: float,
                y_min: float, y_max: float, require_fluid: bool = True) -> ControlBox:
    """Snap a physical box to the nearest enclosing grid faces and audit it."""
    dx, dy = grid.dx, grid.dy
    i0 = int(round((x_min - grid.x0) / dx))
    i1 = int(round((x_max - grid.x0) / dx))
    j0 = int(round((y_min - grid.y0) / dy))
    j1 = int(round((y_max - grid.y0) / dy))
    if not (1 <= i0 < i1 <= grid.nx - 1):
        raise ValueError(f"x-extent {x_min}..{x_max} does not fit the grid")
    if not (1 <= j0 < j1 <= grid.ny - 1):
        raise ValueError(f"y-extent {y_min}..{y_max} does not fit the grid")
    # The discrete gradients need one interior cell on each side of a face.
    if i0 < 2 or i1 > grid.nx - 2 or j0 < 2 or j1 > grid.ny - 2:
        raise ValueError("control box must stay >= 2 cells away from the domain edges")
    if require_fluid:
        # The body is *inside* V on purpose (its surface terms cancel against the
        # body-reaction force), but every flux face must lie in the fluid.
        strips = (grid.solid[j0:j1, i0 - 1:i0 + 1], grid.solid[j0:j1, i1 - 1:i1 + 1],
                  grid.solid[j0 - 1:j0 + 1, i0:i1], grid.solid[j1 - 1:j1 + 1, i0:i1])
        if any(bool(strip.any()) for strip in strips):
            raise ValueError("control-volume flux faces intersect the solid mask; "
                             "grow the box away from the body")
    return ControlBox(i0=i0, i1=i1, j0=j0, j1=j1)


@dataclass
class ForceSample:
    """One momentum-budget evaluation (dimensional force, per unit depth)."""

    time: float
    fx: float
    fy: float
    fx_unsteady: float = 0.0
    fy_unsteady: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"time": self.time, "fx": self.fx, "fy": self.fy,
                "fx_unsteady": self.fx_unsteady, "fy_unsteady": self.fy_unsteady,
                "components": dict(self.components)}


def _box_momentum(flow, box: ControlBox, rho: float) -> tuple[float, float]:
    grid = flow.config.grid
    uc, vc = cell_velocity(flow)
    fluid = grid.fluid[box.cell_slice].to(uc.dtype)
    dV = grid.dx * grid.dy
    mx = float((rho * uc[box.cell_slice] * fluid).sum()) * dV
    my = float((rho * vc[box.cell_slice] * fluid).sum()) * dV
    return mx, my


def budget_terms(flow, box: ControlBox, *, rho: float | None = None
                 ) -> dict[str, float]:
    """Evaluate every momentum-budget term separately (diagnostics)."""
    grid = flow.config.grid
    cfg = flow.config
    rho = cfg.density if rho is None else rho
    mu = cfg.viscosity * rho
    dx, dy = grid.dx, grid.dy
    u, v, p = flow.u, flow.v, flow.p
    uc, vc = cell_velocity(flow)
    js = slice(box.j0, box.j1)          # fluid rows inside V
    is_ = slice(box.i0, box.i1)         # fluid columns inside V

    # ---------------------------------------------------------------- x-momentum
    # east face (i = i1, n = +x)
    u_e = u[js, box.i1]
    conv_xe = float((rho * u_e * u_e).sum()) * dy
    pres_xe = float(p[js, box.i1 - 1].sum()) * dy
    visc_xe = -float(((u[js, box.i1] - u[js, box.i1 - 1]) / dx).sum()) * mu * dy
    # west face (i = i0, n = -x)
    u_w = u[js, box.i0]
    conv_xw = -float((rho * u_w * u_w).sum()) * dy
    pres_xw = -float(p[js, box.i0].sum()) * dy
    visc_xw = float(((u[js, box.i0 + 1] - u[js, box.i0]) / dx).sum()) * mu * dy
    # north face (j = j1, n = +y)
    v_n = v[box.j1, is_]
    uc_n = 0.5 * (u[box.j1 - 1, box.i0:box.i1] + u[box.j1 - 1, box.i0 + 1:box.i1 + 1])
    conv_xn = float((rho * v_n * uc_n).sum()) * dx
    visc_xn = -float(((uc[box.j1 - 1, is_] - uc[box.j1 - 2, is_]) / dy).sum()) * mu * dx
    # south face (j = j0, n = -y)
    v_s = v[box.j0, is_]
    uc_s = 0.5 * (u[box.j0, box.i0:box.i1] + u[box.j0, box.i0 + 1:box.i1 + 1])
    conv_xs = -float((rho * v_s * uc_s).sum()) * dx
    visc_xs = float(((uc[box.j0, is_] - uc[box.j0 - 1, is_]) / dy).sum()) * mu * dx

    # ---------------------------------------------------------------- y-momentum
    vc_e = 0.5 * (v[js, box.i1 - 1] + v[js, box.i1])
    conv_ye = float((rho * u_e * vc_e).sum()) * dy
    visc_ye = -float(((vc[js, box.i1 - 1] - vc[js, box.i1 - 2]) / dx).sum()) * mu * dy
    vc_w = 0.5 * (v[js, box.i0 - 1] + v[js, box.i0])
    conv_yw = -float((rho * u_w * vc_w).sum()) * dy
    visc_yw = float(((vc[js, box.i0] - vc[js, box.i0 - 1]) / dx).sum()) * mu * dy
    conv_yn = float((rho * v_n * v_n).sum()) * dx
    pres_yn = float(p[box.j1 - 1, is_].sum()) * dx
    visc_yn = -float(((v[box.j1, is_] - v[box.j1 - 1, is_]) / dy).sum()) * mu * dx
    conv_ys = -float((rho * v_s * v_s).sum()) * dx
    pres_ys = -float(p[box.j0, is_].sum()) * dx
    visc_ys = float(((v[box.j0 + 1, is_] - v[box.j0, is_]) / dy).sum()) * mu * dx

    return {
        "conv_fx": conv_xe + conv_xw + conv_xn + conv_xs,
        "press_fx": pres_xe + pres_xw,
        "visc_fx": visc_xe + visc_xw + visc_xn + visc_xs,
        "conv_fy": conv_ye + conv_yw + conv_yn + conv_ys,
        "press_fy": pres_yn + pres_ys,
        "visc_fy": visc_ye + visc_yw + visc_yn + visc_ys,
    }


class ForceRecorder:
    """Step-by-step momentum-budget force with an unsteady term.

    ``sample(t)`` must be called after every solver step: the change of the
    control-volume momentum between two consecutive calls is divided by the
    elapsed time, matching the backward-difference discretisation of the solver.
    """

    def __init__(self, flow, box: ControlBox, *, rho: float | None = None):
        self.flow = flow
        self.box = box
        self.rho = flow.config.density if rho is None else rho
        self._t_prev: float | None = None
        self._m_prev: tuple[float, float] | None = None
        self.history: list[ForceSample] = []

    def sample(self, time: float) -> ForceSample:
        terms = budget_terms(self.flow, self.box, rho=self.rho)
        mx, my = _box_momentum(self.flow, self.box, self.rho)
        dmx = dmy = 0.0
        if self._t_prev is not None and time > self._t_prev:
            dt = time - self._t_prev
            dmx = (mx - self._m_prev[0]) / dt
            dmy = (my - self._m_prev[1]) / dt
        self._t_prev, self._m_prev = time, (mx, my)
        # F_body = -d/dt int_V rho u dV - oint_S [ rho u (u.n) + p n - tau.n ] dS
        sample = ForceSample(
            time=time,
            fx=-(dmx + terms["conv_fx"] + terms["press_fx"] + terms["visc_fx"]),
            fy=-(dmy + terms["conv_fy"] + terms["press_fy"] + terms["visc_fy"]),
            fx_unsteady=-dmx, fy_unsteady=-dmy, components=terms)
        self.history.append(sample)
        return sample


__all__ = ["ControlBox", "ForceSample", "ForceRecorder", "control_box",
           "budget_terms", "cell_velocity"]