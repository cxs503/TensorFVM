"""Momentum-coupled dam break against the Martin & Moyce (1952) experiment.

Case statement
--------------
A rectangular column of water (width ``W``, depth ``H0 = 2 W``, the aspect
ratio ``H0/W = 2`` of the Martin & Moyce apparatus) rests on a rigid bed behind
a gate in a long tank.  The gate is removed at ``t = 0`` and the surge front is
tracked.  The case is the classical validation of the *coupled* two-phase
problem: interface transport, variable-density projection and gravity all take
part, so a transport-only verification (an imposed analytic velocity field)
cannot substitute for it.

Reference data
--------------
The digitised Martin & Moyce surge-front curve lives in
:mod:`tensorfvm.vof_phase0.anchors` (``MARTIN_MOYCE_FRONT_T`` /
``MARTIN_MOYCE_FRONT_X``, ``h0/W = 2``, smooth bed) and is **imported**, not
re-typed, so this node cannot drift from the repository's registered anchor.
With ``a`` the column width, ``t* = t sqrt(g/a)`` and ``x* = (x_front - W)/W``.

What this module adds over :mod:`tensorfvm.vof_phase0`
-----------------------------------------------------
``vof_phase0`` solved the same case with an early projection solver and
reported an honest FAIL (``-60.5 % / -31.0 % / -11.7 %`` at
``t* = 1, 2, 3`` -- see ``docs/vof-phase0/report.md``).  Two defects were found
in that campaign and are fixed here:

1. **No well-balanced initial pressure.**  ``pressure_form="reduced"`` (the
   only admissible form when gravity and a free surface coexist -- a
   ``"total"`` pressure pool free-falls, see
   :class:`~tensorfvm.two_phase_solver.TwoPhaseConfig`) needs the *dynamic*
   pressure of the rest state as the initial condition.  Starting from
   ``p = 0`` leaves the whole gas phase accelerating at
   ``(rho_l - rho_g) g / rho_g ~ 1e4 m/s^2`` for one step; the first version of
   this driver blew up to CFL ~ 7e5 exactly that way.  The rest-state dynamic
   pressure is integrated face by face from the same face densities the
   momentum equation uses (:func:`reduced_hydrostatic_pressure`), so
   ``dp/dy = (rho_l - rho_face) g`` holds to round-off.
2. **No step-size control for the gravity wave.** Neither ``stable_time_step``
   (CFL on the current velocity, which is zero at ``t = 0``) nor the viscous
   limit restricts the *gravity-wave* step at ``t = 0``; the driver therefore
   takes ``dt = cfl * h / sqrt(g H0)`` as the fourth limit.

The front is located with sub-cell interpolation on the *bed* rows, i.e. the
same "leading edge at the bottom" definition the experiment used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from ..vof_phase0.anchors import martin_moyce_front, to_dimensionless_front, to_dimensionless_time

__all__ = [
    "DamBreakMMConfig",
    "front_position",
    "initial_alpha",
    "martin_moyce_error",
    "reduced_hydrostatic_pressure",
    "run_dam_break",
]


@dataclass
class DamBreakMMConfig:
    """Geometry, grid and time integration of the Martin & Moyce case."""

    #: Column width ``W`` and depth ``H0 = aspect * W`` (aspect = 2 for the
    #: digitised reference curve).
    column_width: float = 0.05715
    aspect: float = 2.0
    #: Tank: ``length`` x ``height`` [m].  The reference curve is measured in a
    #: long tank, so the outlet must not reflect inside the time window.
    length: float = 0.4
    height: float = 0.2
    #: Grid (``nx`` along the tank, ``ny`` across it).
    nx: int = 100
    ny: int = 50
    #: Fluid properties.
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.0e-5
    gravity: float = 9.81
    #: Transport / momentum discretisation.
    vof_scheme: str = "plic"
    momentum_advection: bool = True
    #: Time integration.
    cfl: float = 0.35
    t_end: float = 0.30
    #: Times [s] at which the front is compared with the experiment.
    t_samples: tuple = (0.10, 0.20, 0.30)
    track_every: int = 1
    max_steps: int = 20000

    def __post_init__(self) -> None:
        if self.column_width <= 0.0 or self.aspect <= 0.0:
            raise ValueError("column_width and aspect must be positive")
        if self.depth > self.height + 1e-15:
            raise ValueError("column depth exceeds the tank height")
        if self.column_width >= self.length:
            raise ValueError("the column must be narrower than the tank")
        if self.nx < 20 or self.ny < 10:
            raise ValueError("grid too coarse for a dam-break simulation")

    @property
    def depth(self) -> float:
        """Initial water depth ``H0``."""
        return self.aspect * self.column_width

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def time_scale(self) -> float:
        """Seconds per unit ``t*``: ``sqrt(W/g)`` (equals ``sqrt(H0/(2g))``)."""
        return math.sqrt(self.column_width / self.gravity)


def initial_alpha(solver: TwoPhaseSolver, cfg: DamBreakMMConfig) -> torch.Tensor:
    """Unit volume fraction inside ``[0, W] x [0, H0]``, zero elsewhere."""
    x = (torch.arange(cfg.nx, dtype=torch.float64) + 0.5) * solver.dx
    y = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * solver.dy
    inside_x = (x <= cfg.column_width).double()
    inside_y = (y <= cfg.depth).double()
    return inside_y[:, None] * inside_x[None, :]


def reduced_hydrostatic_pressure(solver: TwoPhaseSolver) -> torch.Tensor:
    """Dynamic pressure of the rest state, integrated on the v-faces.

    With ``pressure_form="reduced"`` the momentum balance of a fluid at rest is

    .. math::  \\frac{dp}{dy} = (\\rho_l - \\rho^{face}) g ,

    so the cell-centre dynamic pressure is the telescoping sum

    .. math::  p_{j-1} = p_j + (\\rho^{face}_j - \\rho_l) g \\, \\Delta y

    started from the atmospheric datum on the top *face*
    (``p = 0``) at half a cell.  The face densities are the same arithmetic
    means :meth:`~tensorfvm.two_phase_solver.TwoPhaseSolver._rho_face_v` gives
    the momentum equation, which is what makes the balance discrete rather
    than merely analytic: for a stratified column the residual is at round-off.

    Requires ``alpha`` to be set already (it defines the face densities).
    """
    rho_l = solver.tp.rho_liquid
    g = solver.tp.gravity
    if solver.rho_field is None:
        raise ValueError("set_alpha() must be called before the pressure initialisation")
    if g == 0.0:
        return torch.zeros((solver.ny, solver.nx), dtype=solver.p.dtype)
    rho_face = solver._rho_face_v()  # (ny+1, nx): face j sits between cells j-1, j
    p = torch.zeros((solver.ny, solver.nx), dtype=solver.p.dtype)
    dy = solver.dy
    # Top cell centre is half a cell below the atmospheric datum on the top face
    # (where the operator pins p = 0 and uses the one-sided density rho_face[-1]).
    p[-1, :] = 0.5 * (rho_face[-1, :] - rho_l) * g * dy
    for j in range(solver.ny - 1, 0, -1):
        # the segment between cells j-1 and j is the v-face with index j
        p[j - 1, :] = p[j, :] + (rho_face[j, :] - rho_l) * g * dy
    return p



def _column_height(alpha: torch.Tensor, dy: float, cfg: DamBreakMMConfig) -> float:
    """Residual depth against the upstream wall (columns of the original column)."""
    ncol = max(1, int(round(cfg.column_width / (cfg.length / cfg.nx))))
    strip = alpha[:, :ncol].mean(dim=1)
    return float((strip > 0.5).sum()) * dy

def front_position(alpha: torch.Tensor, dx: float, *, level: float = 0.5,
                   bed_rows: int = 3) -> dict:
    """Leading-edge position of the liquid on the bed, with sub-cell interpolation.

    The front is the last crossing of ``alpha = level`` along the bottom
    ``bed_rows`` cell rows of the tank (the experiment measured the foot of the
    surge).  The crossing is interpolated linearly between cell centres, which
    removes the half-cell staircase of a pure cell-count definition.

    Returns ``{"x_front", "column", "interpolated"}``; ``x_front`` is measured
    from the upstream wall.
    """
    rows = alpha[:bed_rows, :]  # row 0 is the bed (v[0] is the no-flux wall)
    col = rows.max(dim=0).values
    above = torch.nonzero(col > level).flatten()
    if above.numel() == 0:
        return {"x_front": 0.0, "column": -1, "interpolated": False}
    last = int(above[-1])
    if last + 1 < col.numel():
        a0 = float(col[last])
        a1 = float(col[last + 1])
        frac = (a0 - level) / (a0 - a1) if a0 > a1 else 0.0
        frac = min(max(frac, 0.0), 1.0)
        x = (last + 0.5 + frac) * dx
        return {"x_front": x, "column": last, "interpolated": True}
    return {"x_front": (last + 0.5) * dx, "column": last, "interpolated": False}


def martin_moyce_error(t: float, x_front: float, cfg: DamBreakMMConfig) -> dict:
    """Relative front error against the digitised experiment at physical time ``t``."""
    t_star = to_dimensionless_time(t, cfg.column_width, cfg.gravity)
    x_star_sim = to_dimensionless_front(x_front, cfg.column_width)
    x_star_ref = martin_moyce_front(t_star)
    rel = (x_star_sim - x_star_ref) / x_star_ref if x_star_ref > 0 else float("nan")
    return {
        "t": t,
        "t_star": t_star,
        "x_front": x_front,
        "x_star_sim": x_star_sim,
        "x_star_reference": x_star_ref,
        "relative_error": rel,
        "abs_relative_error": abs(rel),
    }


def _time_step(solver: TwoPhaseSolver, cfg: DamBreakMMConfig) -> float:
    """Combined gravity-wave / CFL / viscous / capillary step."""
    h = min(solver.dx, solver.dy)
    c_gravity = math.sqrt(cfg.gravity * cfg.depth)
    u_max = max(float(solver.u.abs().max()), float(solver.v.abs().max()))
    dt_wave = cfg.cfl * h / max(c_gravity, u_max, 1e-12)
    dt_visc = math.inf
    nu = max(cfg.mu_liquid / cfg.rho_liquid, cfg.mu_gas / cfg.rho_gas)
    if nu > 0.0:
        dt_visc = 0.25 * h * h / nu
    return float(min(dt_wave, dt_visc))


def run_dam_break(cfg: DamBreakMMConfig | None = None, **overrides) -> dict:
    """Run the coupled Martin & Moyce dam break and return the acceptance record.

    The returned dict carries the front-position history, the comparison at
    ``cfg.t_samples``, the global mass ledger and the solver fingerprint.  It is
    JSON-ready except for the ``_solver`` / ``_alpha_final`` keys.
    """
    cfg = cfg or DamBreakMMConfig()
    if overrides:
        cfg = DamBreakMMConfig(**{**cfg.__dict__, **overrides})
    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=cfg.rho_liquid,
        rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid,
        mu_gas=cfg.mu_gas,
        gravity=cfg.gravity,
        surface_tension=False,
        top_atmosphere=True,
        bottom_wall=True,
        pressure_form="reduced",
        warm_start_pressure=True,
        momentum_advection=cfg.momentum_advection,
        vof_scheme=cfg.vof_scheme,
    )
    solver = TwoPhaseSolver(
        GridConfig(nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height), tp
    )
    solver.set_alpha(initial_alpha(solver, cfg))
    solver.p = reduced_hydrostatic_pressure(solver)
    m0 = float(solver.alpha.sum())

    samples = sorted(cfg.t_samples)
    out_samples = []
    hist = {"t": [], "x_front": [], "mass_drift": [], "max_u": [], "column_height": []}
    pending = list(samples)
    step = 0
    while solver.time < cfg.t_end - 1e-12 and step < cfg.max_steps:
        dt = _time_step(solver, cfg)
        if pending and solver.time + dt > pending[0]:
            dt = max(pending[0] - solver.time, 1e-12)
        solver.step(dt)
        step += 1
        front = front_position(solver.alpha, solver.dx)
        drift = abs(float(solver.alpha.sum()) - m0) / m0
        if step % cfg.track_every == 0 or step == 1:
            hist["t"].append(solver.time)
            hist["x_front"].append(front["x_front"])
            hist["mass_drift"].append(drift)
            hist["max_u"].append(max(float(solver.u.abs().max()), float(solver.v.abs().max())))
            hist["column_height"].append(_column_height(solver.alpha, solver.dy, cfg))
        while pending and solver.time >= pending[0] - 1e-9:
            t = pending.pop(0)
            rec = martin_moyce_error(t, front["x_front"], cfg)
            rec["mass_drift"] = drift
            rec["max_u"] = hist["max_u"][-1]
            out_samples.append(rec)
        if not torch.isfinite(solver.u).all():
            raise RuntimeError("non-finite velocity")

    worst = max((s["abs_relative_error"] for s in out_samples), default=float("nan"))
    return {
        "case": "dam_break_martin_moyce",
        "nx": cfg.nx,
        "ny": cfg.ny,
        "dx": solver.dx,
        "dy": solver.dy,
        "column_width": cfg.column_width,
        "depth": cfg.depth,
        "aspect": cfg.aspect,
        "cfl": cfg.cfl,
        "vof_scheme": cfg.vof_scheme,
        "momentum_advection": cfg.momentum_advection,
        "t_end": solver.time,
        "steps": step,
        "samples": out_samples,
        "max_abs_relative_error": worst,
        "mass_drift_max": max(hist["mass_drift"]) if hist["mass_drift"] else 0.0,
        "history_t": hist["t"],
        "history_x_front": hist["x_front"],
        "history_x_star": [to_dimensionless_front(x, cfg.column_width) for x in hist["x_front"]],
        "history_t_star": [to_dimensionless_time(t, cfg.column_width, cfg.gravity) for t in hist["t"]],
        "history_mass_drift": hist["mass_drift"],
        "history_column_height": hist["column_height"],
        "_solver": solver,
        "_alpha_final": solver.alpha.clone(),
    }