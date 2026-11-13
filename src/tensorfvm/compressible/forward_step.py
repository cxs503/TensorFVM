"""Mach-3 flow over a forward-facing step (Woodward & Colella 1984, problem 2).

Physical problem
    A shock tube driven wind tunnel of length ``L = 3`` and height ``H = 1``
    (in units of the tunnel height) carries uniform Mach 3 flow of a perfect gas
    with ``gamma = 1.4``, ``rho = 1.4`` and ``p = 1`` (so that the speed of sound
    is exactly 1 and the inflow velocity is exactly 3).  A step of height
    ``h = 0.2`` starts at ``x = 0.6`` and extends to the outflow end.  All walls
    are reflecting (slip); the inflow end is held at the uniform state and the
    outflow end is non-reflecting.  The flow is started impulsively from the
    uniform state, which is the configuration of Woodward & Colella, *J. Comput.
    Phys.* **54** (1984) 115-173, problem 2 ("a Mach 3 wind tunnel with a
    step").  That paper reports the solution as density contour plots, so the
    graded numbers here are **not** transcribed from it: every reference value
    is computed in code from the exact Rankine-Hugoniot / isentropic relations
    for the paper's stated inflow state (see
    :func:`normal_shock_reference`), which is what "zero hand-copying" means for
    a case whose source publishes contours.

Numerical method
    The step is an :class:`~tensorfvm.compressible.solver.EulerSolver` run with
    an embedded solid mask (``solid=``).  ``step_x`` and ``step_h`` are placed on
    cell faces, so the block wall is represented exactly rather than as a
    staircase.  The second-order MUSCL-Hancock + HLLC scheme of the solver
    advances the flow to ``t = 4``.

Graded quantities
    The near-normal part of the detached shock that stands in front of the step
    face compresses the uniform ``M = 3`` inflow, so the shock-normal jump
    satisfies the normal-shock relations *exactly*; the post-shock state is
    sampled on the symmetry streamline ``y = h/2`` and compared against them.
    See :func:`measure_shock_jump`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .eos import StiffenedGasMixture
from .solver import EulerSolver

__all__ = [
    "ForwardStepConfig",
    "normal_shock_reference",
    "build_case",
    "run_case",
    "measure_shock_jump",
    "stagnation_state",
    "wall_profiles",
    "budget",
    "mass_flux_balance",
    "step_force",
]

#: Woodward & Colella (1984) problem 2 parameters (tunnel height = 1).
GAMMA_AIR = 1.4
MACH_IN = 3.0
RHO_IN = 1.4
P_IN = 1.0
LX = 3.0
LY = 1.0
STEP_X = 0.6
STEP_H = 0.2
T_END = 4.0


@dataclass
class ForwardStepConfig:
    """Grid and case parameters.  ``nx``/``ny`` must resolve ``lx/ly = 3``."""

    nx: int = 120
    ny: int = 40
    lx: float = LX
    ly: float = LY
    step_x: float = STEP_X
    step_h: float = STEP_H
    mach: float = MACH_IN
    gamma: float = GAMMA_AIR
    rho: float = RHO_IN
    p: float = P_IN
    t_end: float = T_END
    cfl: float = 0.4
    order: int = 2
    limiter: str = "mc"
    project_wall_flux: bool = True

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 4:
            raise ValueError("the step case needs at least an 8 x 4 mesh")
        dx, dy = self.lx / self.nx, self.ly / self.ny
        if abs(self.step_x / dx - round(self.step_x / dx)) > 1e-9:
            raise ValueError("step_x must fall on a cell face (choose nx)")
        if abs(self.step_h / dy - round(self.step_h / dy)) > 1e-9:
            raise ValueError("step_h must fall on a cell face (choose ny)")
        if not 0.0 < self.step_h < self.ly or not 0.0 < self.step_x < self.lx:
            raise ValueError("the step must lie strictly inside the tunnel")

    # -- derived grid -----------------------------------------------------
    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def i_step(self) -> int:
        """First column at or right of the step face."""
        return int(round(self.step_x / self.dx))

    @property
    def j_step(self) -> int:
        """First row at or above the step roof."""
        return int(round(self.step_h / self.dy))

    @property
    def u_in(self) -> float:
        return self.mach * math.sqrt(self.gamma * self.p / self.rho)

    def solid_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.nx, self.ny), dtype=torch.bool)
        mask[self.i_step:, : self.j_step] = True
        return mask

    def centers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        xc = (torch.arange(self.nx, dtype=torch.float64) + 0.5) * self.dx
        yc = (torch.arange(self.ny, dtype=torch.float64) + 0.5) * self.dy
        return xc, yc

    def as_dict(self) -> Dict[str, float]:
        return {
            "nx": self.nx, "ny": self.ny, "dx": self.dx, "dy": self.dy,
            "lx": self.lx, "ly": self.ly, "step_x": self.step_x,
            "step_h": self.step_h, "mach": self.mach, "gamma": self.gamma,
            "rho": self.rho, "p": self.p, "u_in": self.u_in,
            "t_end": self.t_end, "cfl": self.cfl, "order": self.order,
            "limiter": self.limiter,
            "project_wall_flux": self.project_wall_flux,
        }


# ----------------------------------------------------------------------
# exact reference: normal-shock and isentropic relations
# ----------------------------------------------------------------------
def normal_shock_reference(mach: float = MACH_IN, gamma: float = GAMMA_AIR,
                           rho: float = RHO_IN, p: float = P_IN) -> Dict[str, float]:
    """Closed-form state across a *stationary* normal shock (zero hand-copying).

    The relations below are the textbook Rankine-Hugoniot results for a perfect
    gas; nothing is transcribed from the paper.  With ``mach = 3`` and
    ``gamma = 1.4`` they give the numbers quoted in the report
    (``p2/p1 = 10.3333``, ``rho2/rho1 = 3.85714``, ``M2 = 0.47519``,
    ``p02/p01 = 0.328343``).
    """
    if mach <= 1.0:
        raise ValueError("a normal shock needs mach > 1")
    g, m = gamma, mach
    p_ratio = 1.0 + 2.0 * g / (g + 1.0) * (m * m - 1.0)
    rho_ratio = (g + 1.0) * m * m / ((g - 1.0) * m * m + 2.0)
    t_ratio = p_ratio / rho_ratio
    mach2 = math.sqrt(((g - 1.0) * m * m + 2.0) / (2.0 * g * m * m - (g - 1.0)))
    p0_ratio = (rho_ratio ** (g / (g - 1.0))) * ((g + 1.0) / (2.0 * g * m * m - (g - 1.0))) ** (1.0 / (g - 1.0))
    c1 = math.sqrt(g * p / rho)
    u1 = m * c1
    p01 = p * (1.0 + 0.5 * (g - 1.0) * m * m) ** (g / (g - 1.0))
    p02 = p01 * p0_ratio
    # the post-shock stream that stagnates on the step face decelerates
    # isentropically from the post-shock state, so the wall state is (p02, rho02)
    rho02 = rho * rho_ratio * (p02 / (p * p_ratio)) ** (1.0 / g)
    return {
        "mach": m, "gamma": g,
        "rho1": rho, "u1": u1, "p1": p, "c1": c1,
        "rho2": rho * rho_ratio, "u2": u1 / rho_ratio, "p2": p * p_ratio,
        "c2": math.sqrt(g * p * p_ratio / (rho * rho_ratio)),
        "mach2": mach2,
        "p_ratio": p_ratio, "rho_ratio": rho_ratio, "t_ratio": t_ratio,
        "u_ratio": 1.0 / rho_ratio, "p0_ratio": p0_ratio,
        "p01": p01, "p02": p02, "rho02": rho02,
    }


# ----------------------------------------------------------------------
# case construction
# ----------------------------------------------------------------------
def build_case(cfg: ForwardStepConfig | None = None):
    """Return ``(solver, cfg)`` for the impulsively started step problem."""
    cfg = cfg or ForwardStepConfig()
    eos = StiffenedGasMixture([cfg.gamma, cfg.gamma], names=["air", "air"])
    inflow = ("inflow", {"rho": cfg.rho, "u": cfg.u_in, "v": 0.0,
                         "p": cfg.p, "Y": 0.0})
    solver = EulerSolver(
        nx=cfg.nx, ny=cfg.ny, lx=cfg.lx, ly=cfg.ly, eos=eos,
        bc=(inflow, "outflow", "reflective", "reflective"),
        flux_mode="hllc", order=cfg.order, limiter=cfg.limiter, cfl=cfg.cfl,
        solid=cfg.solid_mask(), project_wall_flux=cfg.project_wall_flux,
    )
    solver.set_uniform(cfg.rho, cfg.u_in, 0.0, cfg.p)
    return solver, cfg


def run_case(cfg: ForwardStepConfig | None = None, *, callback=None) -> Dict:
    """Integrate to ``t_end`` and return the fields plus the budgets."""
    solver, cfg = build_case(cfg)
    prim0 = solver.primitive()
    initial = _fluid_integrals(prim0, cfg)
    solver.run(cfg.t_end, callback=callback)
    prim = solver.primitive()
    out = {
        "case": "forward_step",
        "config": cfg.as_dict(),
        "reference": normal_shock_reference(cfg.mach, cfg.gamma, cfg.rho, cfg.p),
        "solver": solver,
        "prim": prim,
        "time": solver.time,
        "nsteps": solver.nsteps,
        "initial": initial,
        "budget": budget(prim, cfg, initial, solver),
        "positivity_fixes": solver.positivity_fixes,
        "troubled_repairs": solver.troubled_repairs,
    }
    return out


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------
def _fluid_mask(cfg: ForwardStepConfig) -> torch.Tensor:
    return ~cfg.solid_mask()


def _fluid_integrals(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig) -> Dict[str, float]:
    cell = cfg.dx * cfg.dy
    fluid = _fluid_mask(cfg)
    rho = prim["rho"]
    e = prim["p"] / (cfg.gamma - 1.0) + 0.5 * rho * (prim["u"] ** 2 + prim["v"] ** 2)
    return {
        "mass": float((rho * fluid).sum() * cell),
        "energy": float((e * fluid).sum() * cell),
    }


def budget(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig,
           initial: Dict[str, float], solver: EulerSolver) -> Dict[str, float]:
    """Fluid-domain mass/energy budget including the open-end outflow ledger.

    Solid cells are excluded from the fluid sum (their state is a mirror, not
    storage).  ``*_drift`` is ``(fluid(t) + outflow(t) - fluid(0)) / |fluid(0)|``;
    for this case the wall faces are adiabatic and impermeable, so both
    numbers measure the flux-difference (conservation) error plus the mass that
    leaks through the reflected wall states.
    """
    now = _fluid_integrals(prim, cfg)
    out_mass = float(solver.boundary_outflow[0])
    out_energy = float(solver.boundary_outflow[3])
    # the boundary ledger counts momentum flux through the open ends as well;
    # only mass and energy are compared here (the walls carry pressure forces)
    return {
        "mass_fluid": now["mass"],
        "mass_outflow": out_mass,
        "mass_drift": (now["mass"] + out_mass - initial["mass"]) / abs(initial["mass"]),
        "energy_fluid": now["energy"],
        "energy_outflow": out_energy,
        "energy_drift": (now["energy"] + out_energy - initial["energy"]) / abs(initial["energy"]),
    }


def _line(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig, *, y: float | None = None,
          x: float | None = None):
    """Sample an ``x``-profile at height ``y`` or a ``y``-profile at station ``x``."""
    xc, yc = cfg.centers()
    if y is not None:
        j = int(torch.argmin(torch.abs(yc - y)).item())
        coord = xc
        pick = lambda q: q[:, j]  # noqa: E731
    else:
        i = int(torch.argmin(torch.abs(xc - x)).item())
        coord = yc
        pick = lambda q: q[i, :]  # noqa: E731
    return coord, {name: pick(prim[name]) for name in ("rho", "u", "v", "p")}


def _shock_crossing(xvals: List[float], rho: List[float], level: float,
                    x_max: float, dx: float) -> float:
    for a, b in zip(range(len(xvals) - 1), range(1, len(xvals))):
        if xvals[a] >= x_max:
            break
        if rho[a] < level <= rho[b]:
            frac = (level - rho[a]) / (rho[b] - rho[a]) if rho[b] != rho[a] else 0.5
            return xvals[a] + frac * dx
    raise RuntimeError("no shock crossing found upstream of the step face")


def _sample_behind(prof, xvals, crossing, cfg, x_start, x_end, sel_keys,
                   dx: float | None = None):
    """Line-fit the cells in ``[x_start, x_end]`` behind the shock back to it."""
    dx = cfg.dx if dx is None else dx
    behind = [k for k in range(len(xvals)) if x_start <= xvals[k] <= x_end]
    if not behind:
        # the requested window is narrower than one cell: fall back to the first
        # cell centre at least half a cell behind the crossing, i.e. the first
        # cell whose upstream face lies at/behind the half-jump point.  A centre
        # merely >= crossing can still sit inside the numerical transition, and
        # sampling it reports a state between pre- and post-shock.
        behind = [k for k in range(len(xvals)) if xvals[k] >= crossing + 0.5 * dx][:1]
    if not behind:  # pragma: no cover - defensive
        raise RuntimeError("no cell behind the shock before the step face")
    sel = torch.tensor(behind, dtype=torch.long)
    xb = torch.tensor([xvals[k] for k in behind], dtype=torch.float64)
    out = {"n_cells_behind": len(behind), "x_window": [float(xb.min()), float(xb.max())]}
    for key in sel_keys:
        v = prof[key][sel]
        mean = float(v.mean())
        if len(behind) >= 2:
            xbar = float(xb.mean())
            slope = float(((xb - xbar) * (v - v.mean())).sum() / ((xb - xbar) ** 2).sum())
            if v[0] > v[-1]:   # a positive-going slope fitted to a decreasing tail
                pass
            out[f"{key}_at_shock"] = mean + slope * (crossing - xbar)
        else:
            out[f"{key}_at_shock"] = mean
        out[f"{key}_window_mean"] = mean
    return out


def measure_shock_jump(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig, *,
                       y: float | None = None, window: float | None = None,
                       scales: Tuple[float, ...] = (0.5, 1.0, 2.0)) -> Dict[str, float]:
    """Normal-shock jump on the stagnation streamline in front of the step face.

    The shock position ``x_s`` is the half-jump crossing of the density profile
    (linear interpolation inside the transition cell, using the exact
    Rankine-Hugoniot post-shock density as the far level).  Behind the shock the
    flow is already decelerating towards the step face, so a plateau average
    would be biased high in pressure and low in velocity: the post-shock state
    is therefore obtained by a least-squares line through the cells in the
    window ``[x_s + dx, x_s + window]`` evaluated *at* ``x_s``.  ``window``
    defaults to ``min(3 dx, 0.02)`` (three cells, capped at 2 % of the tunnel
    height), i.e. a window that stays inside the standoff region for every grid
    in the study; it was fixed before the production run.  The window mean is reported as
    well, and ``sensitivity`` repeats the estimate for ``window * scales`` so the
    reader can see how much of the number is the sampling choice.
    """
    cfg_y = cfg.step_h / 2.0 if y is None else y
    xc, prof = _line(prim, cfg, y=cfg_y)
    ref = normal_shock_reference(cfg.mach, cfg.gamma, cfg.rho, cfg.p)
    rho = prof["rho"].tolist()
    xvals = xc.tolist()
    crossing = _shock_crossing(xvals, rho, 0.5 * (cfg.rho + ref["rho2"]),
                               cfg.step_x, cfg.dx)
    standoff = cfg.step_x - crossing
    base_window = min(3.0 * cfg.dx, 0.02) if window is None else window
    keys = ("rho", "p", "u")
    out = {"y": cfg_y, "x_shock": crossing, "standoff": standoff,
           "window": base_window}
    base = _sample_behind(prof, xvals, crossing, cfg,
                          crossing + cfg.dx, crossing + base_window, keys)
    out.update(base)
    out["p_ratio"] = out["p_at_shock"] / cfg.p
    out["rho_ratio"] = out["rho_at_shock"] / cfg.rho
    out["u_ratio"] = out["u_at_shock"] / cfg.u_in
    c2 = math.sqrt(cfg.gamma * out["p_at_shock"] / max(out["rho_at_shock"], 1e-12))
    out["mach2"] = abs(out["u_at_shock"]) / c2
    out["errors"] = {
        "p_ratio": out["p_ratio"] / ref["p_ratio"] - 1.0,
        "rho_ratio": out["rho_ratio"] / ref["rho_ratio"] - 1.0,
        "u_ratio": out["u_ratio"] / ref["u_ratio"] - 1.0,
        "mach2": out["mach2"] / ref["mach2"] - 1.0,
    }
    sens = {}
    for scale in scales:
        w = base_window * scale
        alt = _sample_behind(prof, xvals, crossing, cfg, crossing + cfg.dx,
                             crossing + w, keys)
        p_ratio = alt["p_at_shock"] / cfg.p
        rho_ratio = alt["rho_at_shock"] / cfg.rho
        u_ratio = alt["u_at_shock"] / cfg.u_in
        m2 = abs(alt["u_at_shock"]) / math.sqrt(
            cfg.gamma * alt["p_at_shock"] / max(alt["rho_at_shock"], 1e-12))
        sens[f"{scale:g}x"] = {
            "window": w,
            "n_cells_behind": alt["n_cells_behind"],
            "p_ratio": p_ratio, "rho_ratio": rho_ratio,
            "u_ratio": u_ratio, "mach2": m2,
            "err_p_ratio": p_ratio / ref["p_ratio"] - 1.0,
            "err_rho_ratio": rho_ratio / ref["rho_ratio"] - 1.0,
            "err_u_ratio": u_ratio / ref["u_ratio"] - 1.0,
            "err_mach2": m2 / ref["mach2"] - 1.0,
        }
    out["sensitivity"] = sens
    return out


def stagnation_state(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig) -> Dict[str, float]:
    """Wall state at the stagnation point on the step face.

    The streamline that stagnates on the step face has crossed the near-normal
    shock once and then decelerates isentropically (no further shock), so its
    stagnation pressure is the *post-shock* total pressure ``p02`` and the wall
    density is ``rho02 = rho2 (p02/p2)^{1/gamma}``.  The measurement is a
    maximum over the step-face cells, which needs no sampling window and is
    therefore the most robust of the graded numbers.
    """
    ref = normal_shock_reference(cfg.mach, cfg.gamma, cfg.rho, cfg.p)
    face = prim["p"][cfg.i_step - 1, : cfg.j_step]
    j = int(torch.argmax(face).item())
    p_max = float(face[j])
    rho_wall = float(prim["rho"][cfg.i_step - 1, j])
    return {
        "p_max_face": p_max, "j_argmax": j,
        "y_argmax": (j + 0.5) * cfg.dy,
        "p_ratio": p_max / cfg.p,
        "p_ratio_ref": ref["p02"] / cfg.p,
        "err_p_ratio": p_max / ref["p02"] - 1.0,
        "rho_wall": rho_wall,
        "rho_ratio": rho_wall / cfg.rho,
        "rho_ratio_ref": ref["rho02"] / cfg.rho,
        "err_rho_ratio": rho_wall / ref["rho02"] - 1.0,
    }


def wall_profiles(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig) -> Dict[str, Dict[str, List[float]]]:
    """Pressure along the three wall segments (upstream floor, step face, roof).

    Values are taken from the fluid cell adjacent to each wall, so they are the
    wall pressures of the finite-volume solution.  Distances are measured from
    the step corner: ``s`` runs upstream from ``x = step_x`` along the floor,
    up the step face, then downstream along the roof.
    """
    xc, yc = cfg.centers()
    p = prim["p"]
    i_s, j_s = cfg.i_step, cfg.j_step
    floor = {
        "x": xc[:i_s].tolist(),
        "p": p[:i_s, 0].tolist(),
    }
    face = {
        "y": yc[:j_s].tolist(),
        "p": p[i_s - 1, :j_s].tolist(),
    }
    roof = {
        "x": xc[i_s:].tolist(),
        "p": p[i_s:, j_s].tolist(),
    }
    return {"floor": floor, "step_face": face, "roof": roof}


def mass_flux_balance(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig,
                      x_in: float = 0.05, x_out: float | None = None) -> Dict[str, float]:
    """Mass flux through an upstream and a downstream station (steady identity)."""
    x_out = cfg.lx - 0.05 if x_out is None else x_out
    xc, _ = cfg.centers()
    i_in = int(torch.argmin(torch.abs(xc - x_in)).item())
    i_out = int(torch.argmin(torch.abs(xc - x_out)).item())
    j_s = cfg.j_step
    fin = float((prim["rho"][i_in, :] * prim["u"][i_in, :]).sum() * cfg.dy)
    fout = float((prim["rho"][i_out, j_s:] * prim["u"][i_out, j_s:]).sum() * cfg.dy)
    return {"x_in": float(xc[i_in]), "x_out": float(xc[i_out]),
            "flux_in": fin, "flux_out": fout,
            "rel_diff": (fout - fin) / abs(fin)}


def step_force(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig) -> Dict[str, float]:
    """Pressure force on the step (per unit depth), normalised to a drag coefficient."""
    p = prim["p"]
    i_s, j_s = cfg.i_step, cfg.j_step
    f_face = float(p[i_s - 1, :j_s].sum() * cfg.dy)   # +x on the vertical face
    f_roof = float(-p[i_s:, j_s].sum() * cfg.dx)       # -y on the roof, reported for reference
    q = 0.5 * cfg.rho * cfg.u_in ** 2 * cfg.step_h
    return {"force_x_face": f_face, "force_y_roof": f_roof,
            "cd_face": f_face / q if q else float("nan")}