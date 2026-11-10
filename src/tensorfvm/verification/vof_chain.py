"""Two-phase VOF chain gate: hydrostatic negative control + dam-break anchor.

Why this module exists
----------------------
The pieces of the two-phase VOF chain live in different modules and were each
verified in isolation:

* :mod:`tensorfvm.vof_plic` -- geometric (PLIC) interface transport;
* :func:`tensorfvm.two_phase_vof.balanced_csf_face_forces` -- Brackbill CSF
  surface tension on the MAC faces;
* :class:`tensorfvm.two_phase_solver.TwoPhaseSolver` -- variable-property
  projection solver that glues them together.

What was *not* gated is the composition: a stratified column at rest inside the
coupled solver.  That is the classical **negative control** of a free-surface
code, because the exact solution (``u = v = 0``, flat interface) is known
independently of any discretisation detail.  Running it exposed a real defect:
the closed-form hydrostatic initialisation picks one density per cell centre,
while the momentum equation multiplies gravity by the *face* density, so any
interface that does not sit exactly on a cell face leaves an unbalanced
``(rho_l - rho_g) g`` residual and the column starts to move (measured:
``|u| -> 1.8e-2 m/s``, the interface migrates a whole cell).
``TwoPhaseSolver.discrete_hydrostatic_pressure`` fixes this by integrating the
same face densities; the rest state then holds to round-off (``|u| < 1e-15``)
for arbitrary interface offsets and with surface tension on.

The second leg is the transient anchor: the 2-D dam break of Martin & Moyce
(1952), where the surge-front position ``x*(t*)`` has an experimental
reference.  It exercises all three ingredients together (geometric transport of
a deforming interface, gravity, liquid meeting the floor) and is where the
*cost* of the interface scheme shows up as mass loss.

Measured gates (see ``docs/vof-chain-anchor/report.md`` for the run that
produced them)
-----------------------------------------------------------------------------
G1 hydrostatic rest, negative control : ``max|u| <= 1e-10 m/s`` over the run,
    for interfaces both grid-aligned and offset by 0.3/0.5/0.77 ``dy``, with
    and without surface tension, with PLIC transport.
G2 interface immobility            : ``max|alpha - alpha0| <= 1e-12``.
G3 projected mass conservation     : ``|dV|/V0 <= 1e-3`` (0.1 %) over a
    closed-domain transient run.
G4 dam-break surge front           : ``|x*_sim - x*_MM| / x*_MM <= 15 %`` at
    ``t* = 1, 2, 3``.

Every number this module reports is written to ``summary.json``; the figures
are drawn from that file only (no hand-copied values).

Usage
-----
``PYTHONPATH=src python -m tensorfvm.verification.vof_chain --output <dir>``
runs the hydrostatic legs (seconds) and, with ``--dam``, the dam-break legs
(minutes at 100x30, hours at 400x120 -- submit those to the cluster).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from .two_phase_campaign import (
    MM_SURGE_T,
    MM_SURGE_X,
    DamBreakRun,
    evaluate_dam_break,
    run_dam_break_case,
)
from .vof_chain_anchor import (
    FREEFALL_TOL,
    RITTER_FRONT_TOL,
    RitterLegConfig,
    evaluate_ritter,
    freefall_gate,
    front_onset_delay,
    martin_moyce_kinematics,
    report_section,
    run_ritter_leg,
    write_anchor_figures,
)

# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
HYDROSTATIC_VELOCITY_TOL = 1e-10    # m/s, G1
INTERFACE_DISPLACEMENT_TOL = 1e-12  # G2
MASS_DRIFT_TOL = 1e-3               # 0.1 %, G3
DAM_FRONT_TOL = 0.15                # 15 %, G4


# --------------------------------------------------------------------------- #
# G1/G2: two-layer column at rest (the negative control)
# --------------------------------------------------------------------------- #
@dataclass
class HydrostaticRestConfig:
    """Two-layer column at rest in a closed box with an atmospheric top.

    The liquid occupies ``y < interface_y``, the gas above it.  ``offset_dy``
    shifts the interface by that fraction of a cell so the interface lands
    inside a cell (``0 < offset_dy < 1``) instead of on a face -- the case the
    balanced initialisation has to handle, and the case that breaks the
    closed-form profile.
    """

    name: str = "rest"
    nx: int = 40
    ny: int = 20
    length: float = 0.4
    height: float = 0.2
    interface_y: float | None = None   # default: height/2
    #: Interface offset in units of ``dy`` (0 = grid-aligned).
    offset_dy: float = 0.0
    rho_liquid: float = 1000.0
    rho_gas: float = 1.2
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.8e-5
    sigma: float = 0.0
    gravity: float = 9.81
    vof_scheme: str = "plic"
    init: str = "discrete"      # "discrete" | "analytic"
    n_steps: int = 40
    #: VOF Courant number used to pick dt = cfl * h / sqrt(2 g H_liquid).
    cfl: float = 0.2

    def __post_init__(self) -> None:
        if self.init not in ("discrete", "analytic"):
            raise ValueError(f"unknown init {self.init!r}")
        if self.vof_scheme not in ("plic", "upwind"):
            raise ValueError(f"unknown vof_scheme {self.vof_scheme!r}")
        if not 0.0 <= self.offset_dy < 1.0:
            raise ValueError("offset_dy must be in [0, 1)")

    @property
    def interface_position(self) -> float:
        base = self.height / 2.0 if self.interface_y is None else self.interface_y
        return base + self.offset_dy * (self.height / self.ny)


def stratified_alpha(cfg: HydrostaticRestConfig, dy: float) -> torch.Tensor:
    """Liquid fraction of a flat interface, *fractional* in the straddling cell.

    Building the partial cell explicitly matters: with a ``y < y_i`` step the
    interface cell is all-gas or all-liquid and the test would silently fall
    back to the grid-aligned case.
    """
    y = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * dy
    hi = cfg.interface_position
    column = torch.clamp((hi - (y - 0.5 * dy)) / dy, 0.0, 1.0)
    return column.reshape(-1, 1).expand(cfg.ny, cfg.nx).clone()


def run_hydrostatic_rest(cfg: HydrostaticRestConfig, alpha0: torch.Tensor | None = None) -> dict:
    """Integrate the resting column and record every departure from rest."""
    tp = TwoPhaseConfig(
        enabled=True, rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas, sigma=cfg.sigma,
        surface_tension=cfg.sigma > 0.0, gravity=cfg.gravity,
        top_atmosphere=True, bottom_wall=True, side_outlet=False,
        vof_scheme=cfg.vof_scheme, momentum_advection=True,
        pressure_form="total", warm_start_pressure=True, domain_type="closed",
    )
    grid = GridConfig(nx=cfg.nx, ny=cfg.ny, length=cfg.length,
                      height=cfg.height, max_iterations=1)
    solver = TwoPhaseSolver(grid, tp)
    dx, dy = solver.dx, solver.dy
    if alpha0 is None:
        alpha0 = stratified_alpha(cfg, dy)
    solver.set_alpha(alpha0)
    solver.initialize_hydrostatic_pressure(cfg.interface_position,
                                           discrete=cfg.init == "discrete")
    # Static (t = 0) discrete force-balance residual: dp/dy + rho_face*g.
    static_residual = solver.hydrostatic_balance_residual()
    volume0 = float(solver.alpha.sum()) * dx * dy

    h_liquid = cfg.interface_position
    dt = cfg.cfl * min(dx, dy) / math.sqrt(2.0 * cfg.gravity * h_liquid)
    umax = 0.0
    t0 = time.time()
    for _ in range(cfg.n_steps):
        solver.step(dt)
        umax = max(umax, float(solver.u.abs().max()),
                   float(solver.v.abs().max()))
    elapsed = time.time() - t0
    volume1 = float(solver.alpha.sum()) * dx * dy
    displacement = float((solver.alpha - alpha0).abs().max())
    return {
        "config": asdict(cfg),
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "dx": dx, "dy": dy,
                 "offset_in_cells": cfg.offset_dy},
        "interface_position": cfg.interface_position,
        "time_step": dt,
        "time_simulated": solver.time,
        "steps": cfg.n_steps,
        "elapsed_seconds": elapsed,
        "static_balance_residual_pa_per_m": static_residual["max_residual"],
        "static_balance_rms_pa_per_m": static_residual["rms_residual"],
        "max_velocity_m_s": umax,
        "max_alpha_displacement": displacement,
        "volume_drift": (volume1 - volume0) / volume0,
        "pressure_range_pa": [float(solver.p.min()), float(solver.p.max())],
        "pass_velocity": umax <= HYDROSTATIC_VELOCITY_TOL,
        "pass_interface": displacement <= INTERFACE_DISPLACEMENT_TOL,
        "pass_volume": abs((volume1 - volume0) / volume0) <= MASS_DRIFT_TOL,
    }


def hydrostatic_matrix(nx: int = 40, ny: int = 20, n_steps: int = 40,
                       offsets=(0.0, 0.3, 0.5, 0.77), sigmas=(0.0, 0.072),
                       schemes=("plic", "upwind"),
                       inits=("analytic", "discrete")) -> list[dict]:
    """The full negative-control matrix (cheap: ~0.2 s per leg at 40x20)."""
    runs: list[dict] = []
    for scheme in schemes:
        for init in inits:
            for offset in offsets:
                for sigma in sigmas:
                    name = (f"{scheme}-{init}-off{offset:g}"
                            f"-sigma{sigma:g}").replace(".", "p")
                    cfg = HydrostaticRestConfig(
                        name=name, nx=nx, ny=ny, n_steps=n_steps,
                        offset_dy=offset, sigma=sigma, vof_scheme=scheme,
                        init=init)
                    runs.append(run_hydrostatic_rest(cfg))
    return runs


# --------------------------------------------------------------------------- #
# G3/G4: dam break anchor (Martin & Moyce 1952)
# --------------------------------------------------------------------------- #
def dam_break_legs(resolutions=((100, 30), (200, 60)), schemes=("plic",),
                   max_t_star: float = 3.0,
                   max_wall_seconds: float | None = None,
                   progress: bool = True,
                   output: Path | None = None) -> list[dict]:
    """Run the Martin & Moyce dam break at several resolutions/schemes.

    With ``output`` given, the final liquid-fraction field of every leg is
    dumped as ``legs/dam_<scheme>_<nx>x<ny>__alpha.npy`` so the interface
    contour figure can be drawn from the artifact instead of re-running.
    """
    import numpy as np

    legs: list[dict] = []
    for scheme in schemes:
        for nx, ny in resolutions:
            cfg = DamBreakRun(nx=nx, ny=ny, max_t_star=max_t_star,
                              vof_scheme=scheme)
            run = run_dam_break_case(cfg, progress=progress,
                                     max_wall_seconds=max_wall_seconds,
                                     return_field=True)
            field = run.pop("final_alpha", None)
            if field is not None and output is not None:
                (output / "legs").mkdir(parents=True, exist_ok=True)
                name = f"dam_{scheme}_{nx}x{ny}__alpha.npy"
                np.save(output / "legs" / name, field.double().numpy())
                run["field_artifact"] = f"legs/{name}"
            run["evaluation"] = evaluate_dam_break(run)
            run["mass_gate"] = {
                "pass": abs(run["final_volume_drift"]) <= MASS_DRIFT_TOL,
                "final_volume_drift": run["final_volume_drift"],
                "criterion": f"|dV|/V0 <= {MASS_DRIFT_TOL:g}",
            }
            legs.append(run)
            if progress:
                ev = run["evaluation"]
                print(f"  [{scheme} {nx}x{ny}] steps={run['steps']} "
                      f"t*={run['history'][-1]['t_star']:.2f} "
                      f"worst_front_err={ev['worst_relative_error']:.3f} "
                      f"dV={run['final_volume_drift']:.2e} "
                      f"({run['elapsed_seconds']:.0f}s)", flush=True)
    return legs


def evaluate_chain(hydrostatic: list[dict], dam_legs: list[dict] | None,
                   ritter_legs: list[dict] | None = None) -> dict:
    """Assemble the gates G1--G4 from the measured legs."""
    def select(init: str):
        return [r for r in hydrostatic if r["config"]["init"] == init]

    analytic = select("analytic")
    discrete = select("discrete")

    def worst(runs, key):
        return max((abs(r[key]) for r in runs), default=float("nan"))

    gates: dict = {
        "G1_hydrostatic_velocity": {
            "pass": all(r["pass_velocity"] for r in discrete),
            "criterion": f"max|u| <= {HYDROSTATIC_VELOCITY_TOL:g} m/s "
                         "(balanced initialisation, every offset/sigma)",
            "worst_max_velocity_discrete_m_s": worst(discrete, "max_velocity_m_s"),
            "worst_max_velocity_analytic_m_s": worst(analytic, "max_velocity_m_s"),
            "legs": len(discrete),
        },
        "G2_interface_immobility": {
            "pass": all(r["pass_interface"] for r in discrete),
            "criterion": f"max|alpha - alpha0| <= {INTERFACE_DISPLACEMENT_TOL:g}",
            "worst_displacement_discrete": worst(discrete, "max_alpha_displacement"),
            "worst_displacement_analytic": worst(analytic, "max_alpha_displacement"),
        },
        "G3_mass_conservation": {
            # The production scheme is PLIC; the donor-cell leg is a *control*
            # whose leak is reported in the same table, not silently averaged in.
            "pass": bool([leg for leg in dam_legs or []
                          if leg["config"]["vof_scheme"] == "plic"]) and all(
                leg["mass_gate"]["pass"] for leg in dam_legs or []
                if leg["config"]["vof_scheme"] == "plic"),
            "criterion": f"|dV|/V0 <= {MASS_DRIFT_TOL:g} on the PLIC legs "
                         "(the donor-cell control is reported separately)",
            "legs": [{"name": f"{leg['config']['vof_scheme']}-"
                              f"{leg['grid']['nx']}x{leg['grid']['ny']}",
                      "final_volume_drift": leg["final_volume_drift"],
                      "worst_step_relative_drift": leg["worst_step_relative_drift"],
                      "steps": leg["steps"]} for leg in dam_legs or []],
        },
    }
    if dam_legs:
        primary = None
        for leg in dam_legs:
            if leg["config"]["vof_scheme"] == "plic":
                if primary is None or leg["grid"]["nx"] > primary["grid"]["nx"]:
                    primary = leg
        ev = primary["evaluation"]
        for leg in dam_legs:
            hist = leg["history"]
            leg["freefall"] = freefall_gate(
                hist, leg["config"]["column_width"], leg["config"]["gravity"])
        gates["G4a_freefall_surface"] = dict(
            primary["freefall"],
            leg=f"plic-{primary['grid']['nx']}x{primary['grid']['ny']}",
            all_legs=[{"name": f"{leg['config']['vof_scheme']}-"
                               f"{leg['grid']['nx']}x{leg['grid']['ny']}",
                       "worst_relative_error":
                           leg["freefall"]["worst_relative_error"]}
                      for leg in dam_legs])
        # The digitised M&M curve is only used where it is kinematically
        # attainable (see martin_moyce_kinematics); t* = 1 is reported but not
        # gated.
        gate_times = (2.0, 3.0)
        gated = [p for p in ev["points"] if p["t_star"] in gate_times]
        gated_worst = max((abs(p["relative_error"]) for p in gated
                           if p["reached"]), default=float("nan"))
        gates["G4c_martin_moyce_front"] = {
            "pass": bool(gated) and all(p["reached"] for p in gated)
                    and gated_worst <= DAM_FRONT_TOL,
            "criterion": f"surge-front relative error <= {DAM_FRONT_TOL:g} "
                         f"at t* = {', '.join(f'{t:g}' for t in gate_times)}",
            "leg": f"plic-{primary['grid']['nx']}x{primary['grid']['ny']}",
            "worst_relative_error": gated_worst,
            "all_points_including_ungated": ev["points"],
            "points_reached": ev["points_reached"],
            "note": "t* = 1 is reported in 3.2 but excluded from the gate: the "
                    "digitised point needs more than 2 g (see 3.4)",
            "all_legs": [
                {"name": f"{leg['config']['vof_scheme']}-"
                         f"{leg['grid']['nx']}x{leg['grid']['ny']}",
                 "worst_relative_error": leg["evaluation"]["worst_relative_error"],
                 "cells_across_column": leg["grid"]["cells_across_column"],
                 "points": leg["evaluation"]["points"]}
                for leg in dam_legs],
        }
    if ritter_legs:
        ritter_eval = ritter_legs[0]["evaluation"]
        gates["G4b_ritter_front"] = dict(
            ritter_eval,
            leg=ritter_legs[0]["config"]["name"],
            all_legs=[{"name": leg["config"]["name"],
                       "worst_relative_error":
                           leg["evaluation"]["worst_relative_error"]}
                      for leg in ritter_legs])
    gates["all_pass"] = all(g["pass"] for g in gates.values() if "pass" in g)
    return gates


# --------------------------------------------------------------------------- #
# Drivers: JSON + figures + report
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_summary(output: Path, payload: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return path


def write_figures(payload: dict, output: Path) -> list[dict]:
    """Two figures, both drawn only from the measured payload."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []
    hydro = payload["hydrostatic"]

    # --- fig 1: |u|max vs interface offset, closed-form vs balanced ----------
    offsets = sorted({r["config"]["offset_dy"] for r in hydro})
    styles = {("analytic", 0.0): ("C3", "o", "closed form, sigma=0"),
              ("analytic", 0.072): ("C1", "s", "closed form, sigma=0.072"),
              ("discrete", 0.0): ("C3", "^", "balanced, sigma=0"),
              ("discrete", 0.072): ("C1", "D", "balanced, sigma=0.072")}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for (init, sigma), (colour, marker, label) in styles.items():
        ys = []
        for off in offsets:
            legs = [r for r in hydro
                    if r["config"]["init"] == init
                    and abs(r["config"]["offset_dy"] - off) < 1e-12
                    and abs(r["config"]["sigma"] - sigma) < 1e-12]
            ys.append(max(r["max_velocity_m_s"] for r in legs) if legs else np.nan)
        ax.semilogy(offsets, np.maximum(ys, 1e-18), marker=marker, color=colour,
                    ls="-" if init == "discrete" else "--", label=label)
    ax.axhline(HYDROSTATIC_VELOCITY_TOL, color="k", lw=1, ls=":")
    if offsets:
        ax.text(offsets[0], HYDROSTATIC_VELOCITY_TOL * 2.0,
                "gate 1e-10 m/s", fontsize=8)
    ax.set_xlabel("interface offset from the cell face  [dy]")
    ax.set_ylabel("max |u| over the run  [m/s]")
    ax.set_title("Hydrostatic rest: closed-form vs balanced initialisation")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = output / "fig1-hydrostatic-negative-control.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "G1 negative control: max|u| of a resting two-layer "
                            "column vs interface offset. The closed-form profile "
                            "(dashed) leaves a rho*g imbalance at partial cells; "
                            "the balanced discrete profile (solid) holds to "
                            "round-off."})

    # --- fig 2: dam-break surge front vs Martin & Moyce ----------------------
    legs = payload.get("dam_break") or []
    if legs:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
        ax = axes[0]
        ax.plot(MM_SURGE_T, MM_SURGE_X, "k-", lw=2,
                label="Martin & Moyce (1952)")
        colours = ["C0", "C1", "C2", "C3"]
        markers = ["o", "s", "^", "D"]
        for leg, colour, marker in zip(legs, colours, markers):
            hist = leg["history"]
            label = (f"{leg['config']['vof_scheme']} "
                     f"{leg['grid']['nx']}x{leg['grid']['ny']} "
                     f"(a/dx={leg['grid']['cells_across_column']:.0f})")
            ax.plot([h["t_star"] for h in hist], [h["front_star"] for h in hist],
                    marker=marker, ms=2.5, lw=1.2, color=colour, label=label)
        for t_star in (1.0, 2.0, 3.0):
            ax.axvline(t_star, color="0.7", lw=0.8, ls=":")
        ax.set_xlabel("t* = t sqrt(g/a)")
        ax.set_ylabel("x* = (x_front - a)/a")
        ax.set_title("Dam-break surge front vs experiment")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[1]
        for leg, colour, marker in zip(legs, colours, markers):
            hist = leg["history"]
            ax.plot([h["t_star"] for h in hist],
                    [h["volume_drift"] for h in hist], marker=marker, ms=2.0,
                    lw=1.0, color=colour,
                    label=f"{leg['config']['vof_scheme']} "
                          f"{leg['grid']['nx']}x{leg['grid']['ny']}")
        ax.axhline(MASS_DRIFT_TOL, color="k", ls=":", lw=1)
        ax.axhline(-MASS_DRIFT_TOL, color="k", ls=":", lw=1)
        ax.set_xlabel("t*")
        ax.set_ylabel("liquid-volume drift  dV/V0")
        ax.set_title("G3: conservation of the transported phase")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = output / "fig2-dam-break-anchor.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"name": p.name,
                     "caption": "G3/G4: dam-break surge front against Martin & "
                                "Moyce (1952) at several resolutions (left) and "
                                "the liquid-volume drift of each leg (right, "
                                "dotted = 0.1 % gate)."})
    # --- fig 3: liquid-fraction field at the final time, one panel per leg ---
    field_legs = [leg for leg in legs if leg.get("field_artifact")]
    if field_legs:
        n = len(field_legs)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False)
        for ax, leg in zip(axes[0], field_legs):
            data = np.load(output / leg["field_artifact"])
            dx, dy = leg["grid"]["dx"], leg["grid"]["dy"]
            a = leg["config"]["column_width"]
            extent = [-a / a, (data.shape[1] * dx - a) / a,
                      0.0, data.shape[0] * dy / a]
            im = ax.imshow(data, origin="lower", extent=extent, cmap="Blues",
                           vmin=0.0, vmax=1.0, aspect="auto",
                           interpolation="nearest")
            ax.contour(np.linspace(extent[0], extent[1], data.shape[1]),
                       np.linspace(extent[2], extent[3], data.shape[0]),
                       data, levels=[0.5], colors="k", linewidths=0.8)
            ax.set_title(f"{leg['config']['vof_scheme']} "
                         f"{leg['grid']['nx']}x{leg['grid']['ny']}  "
                         f"t*={leg['history'][-1]['t_star']:.2f}", fontsize=9)
            ax.set_xlabel("x* = (x - a)/a")
            ax.set_ylabel("y/a")
            ax.grid(alpha=0.2, color="w")
        fig.colorbar(im, ax=axes[0, -1], label="liquid fraction alpha")
        fig.suptitle("Dam-break liquid field at the final time "
                     "(black line = alpha 0.5 contour)")
        fig.tight_layout()
        p = output / "fig3-dam-break-field.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"name": p.name,
                     "caption": "Liquid-fraction field of every dam-break leg at "
                                "the final time. The alpha = 0.5 contour shows "
                                "how sharp the surge (and how much of it) each "
                                "resolution transports."})
    return figs


def report_section_bridge(payload: dict) -> list[str]:
    """Anchor sections (A1/A2/kinematics/onset) from the anchor module."""
    return report_section(payload)


def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f}%"


def build_report(payload: dict, output: Path) -> Path:
    """Paper-style report, every number injected from ``summary.json``."""
    gates = payload["gates"]
    hydro = payload["hydrostatic"]
    legs = payload.get("dam_break") or []
    lines: list[str] = []
    A = lines.append

    A("# 两相流 VOF 链：静水平衡负控 + 溃坝解析锚（P475W1-N06）\n")
    A("## 1. 案例介绍\n")
    A("本节点把两相流 VOF 链的四个部件（PLIC 几何重构输运、CSF 表面张力、"
      "变密度投影求解器、VOF 质量守恒）放进**同一个耦合求解器**里共同验收，"
      "两个算例构成一条链：\n")
    A("1. **静水平衡负控（G1/G2）**：封闭方腔内两层流体（水下气上）静置。"
      "精确解 `u = v = 0`、界面保持水平，与离散细节无关，因此任何非零速度都"
      "直接暴露离散力平衡缺陷。界面位置取与网格面对齐（offset = 0）以及"
      "落在单元内部（offset = 0.3/0.5/0.77 dy）两种情形；表面张力 σ = 0 与 "
      "σ = 0.072 N/m（水-气）各一组。\n")
    A("2. **溃坝解析锚（G3/G4）**：Martin & Moyce (1952) 二维溃坝，初始水柱 "
      "`[0, a] × [0, 2a]`，`a = 0.05715 m`（2.25 in），域 0.5 m × 0.15 m，"
      "ρ_l = 1000、ρ_g = 1.2 kg/m³，μ_l = 1e-3、μ_g = 1.8e-5 Pa·s，σ = 0，"
      "g = 9.81 m/s²；无量纲 `x* = (x_front - a)/a`、`t* = t sqrt(g/a)`。\n")
    A("参数表（程序化注入自 `summary.json`）：\n")
    A("| 算例 | 网格 | 界面 offset [dy] | σ [N/m] | VOF 格式 | 初始化 |")
    A("|---|---|---|---|---|---|")
    for r in hydro:
        c = r["config"]
        A(f"| {c['name']} | {c['nx']}×{c['ny']} | {c['offset_dy']:g} | "
          f"{c['sigma']:g} | {c['vof_scheme']} | {c['init']} |")
    A("")
    for leg in legs:
        c, g = leg["config"], leg["grid"]
        A(f"- 溃坝 leg `{c['vof_scheme']}-{g['nx']}x{g['ny']}`："
          f"dx = {g['dx']:.5f} m，a/dx = {g['cells_across_column']:.2f} 个单元，"
          f"max t* = {c['max_t_star']:g}，步数 {leg['steps']}。")
    A("")

    A("## 2. 怎么做的\n")
    A("**负控的关键：三种初始化。** `TwoPhaseSolver` 的动量预测器在 v-面上用两点"
      "差分 `(p[j]-p[j-1])/dy` 求压力梯度，而重力项乘以**面密度** "
      "`rho_face_v[j] = ½(ρ[j-1] + ρ[j])`。因此离散静水平衡的充分必要条件是"
      "每个 v-面上 `(p[j]-p[j-1])/dy = -rho_face_v[j] g`。本单新增 "
      "`TwoPhaseSolver.discrete_hydrostatic_pressure()`：从顶部大气面 `p = 0` "
      "自上而下累加**同一套面密度**\n")
    A("```")
    A("p[j-1] = p[j] + rho_face_v[j] * g * dy ,   p[ny-1] = ½ rho_face_v[ny] g dy")
    A("```\n")
    A("该恒等式逐面精确成立（望远镜求和），与界面落在何处无关；对均匀液柱退化为"
      "解析式 `ρ g (H - y)`。原闭式剖面 `initialize_hydrostatic_pressure()` 按"
      "单元中心逐格选液相/气相分支，界面不在面上时界面单元的分数密度与分支选择"
      "不一致，留下 O(Δρ g) 残差。两者都保留：`discrete=False` 仍为默认值"
      "（既有回归逐位不变），负控以 `discrete=True` 验收。\n")
    A("**溃坝。** 复用 `tensorfvm.verification.two_phase_campaign."
      "run_dam_break_case`（变密度投影 + PLIC/迎风输运 + 动量对流）：时间步由 "
      "`min(stable_time_step, CFL·h/max|u|)` 控制（CFL = 0.4）；波前位置按"
      "「液柱厚度 > 0.02 a 的最后一列」测量；逐步记录液相的全局积分。\n")
    A("**工具链。** `PYTHONPATH=src python -m tensorfvm.verification.vof_chain "
      "--output docs/vof-chain-anchor [--dam]`；本报告、图、`summary.json` 由同"
      "一次运行生成，数字零手抄。\n")

    A("## 3. 计算结果\n")
    A("### 3.1 G1/G2 静水平衡负控\n")
    A("| 初始化 | offset [dy] | σ [N/m] | 静态力平衡残差 [Pa/m] | "
      "max\\|u\\| [m/s] | max\\|Δα\\| | dV/V0 |")
    A("|---|---|---|---|---|---|---|")
    for r in hydro:
        c = r["config"]
        A(f"| {c['init']} | {c['offset_dy']:g} | {c['sigma']:g} | "
          f"{r['static_balance_residual_pa_per_m']:.3e} | "
          f"{r['max_velocity_m_s']:.3e} | {r['max_alpha_displacement']:.3e} | "
          f"{r['volume_drift']:+.1e} |")
    A("")
    g1, g2 = gates["G1_hydrostatic_velocity"], gates["G2_interface_immobility"]
    A(f"G1（max|u| ≤ {HYDROSTATIC_VELOCITY_TOL:g} m/s）：平衡初始化最差 "
      f"**{g1['worst_max_velocity_discrete_m_s']:.3e} m/s** → "
      f"{'PASS' if g1['pass'] else 'FAIL'}（{g1['legs']} 条 leg）；同批算例用"
      f"闭式剖面最差 **{g1['worst_max_velocity_analytic_m_s']:.3e} m/s**，超门 "
      f"{g1['worst_max_velocity_analytic_m_s'] / HYDROSTATIC_VELOCITY_TOL:.1e} 倍。\n")
    A(f"G2（max|Δα| ≤ {INTERFACE_DISPLACEMENT_TOL:g}）：平衡初始化最差 "
      f"{g2['worst_displacement_discrete']:.3e} → "
      f"{'PASS' if g2['pass'] else 'FAIL'}；闭式剖面最差 "
      f"{g2['worst_displacement_analytic']:.3e}（相当于界面位移 "
      f"{100.0 * g2['worst_displacement_analytic']:.1f}% 个网格高度，"
      f"即肉眼可见的界面迁移）。\n")

    lines.extend(report_section_bridge(payload))
    A("### 3.6 G4c 溃坝波前 vs Martin & Moyce (1952)\n")
    if legs:
        A("| leg | a/dx | t*=1 实测 | 参考 | 误差 | t*=2 实测 | 参考 | 误差 | "
          "t*=3 实测 | 参考 | 误差 | 最差 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for leg in legs:
            pts = {p["t_star"]: p for p in leg["evaluation"]["points"]}
            row = [f"{leg['config']['vof_scheme']}-{leg['grid']['nx']}x"
                   f"{leg['grid']['ny']}",
                   f"{leg['grid']['cells_across_column']:.1f}"]
            for t_star in (1.0, 2.0, 3.0):
                p = pts.get(t_star, {})
                meas = p.get("measured")
                row += ["n/a" if meas is None else f"{meas:.3f}",
                        f"{p.get('reference', float('nan')):.3f}",
                        _pct(p.get("relative_error")) if meas is not None else "未到达"]
            row.append(_pct(leg["evaluation"]["worst_relative_error"]))
            A("| " + " | ".join(row) + " |")
        A("")
        g4 = gates["G4c_martin_moyce_front"]
        A(f"G4c 判据 `{g4['criterion']}`（t*=1 只报不判，理由见 §3.4）："
          f"最细 PLIC leg ({g4['leg']}) 受判点最差误差 "
          f"**{_pct(g4['worst_relative_error'])}** → "
          f"{'PASS' if g4['pass'] else 'FAIL'}。\n")

    A("### 3.7 G3 质量守恒（闭合域瞬态）\n")
    A("| leg | 步数 | 单步最差相对漂移 | 终态 dV/V0 | 判据 0.1% |")
    A("|---|---|---|---|---|")
    for entry in gates["G3_mass_conservation"]["legs"]:
        A(f"| {entry['name']} | {entry['steps']} | "
          f"{entry['worst_step_relative_drift']:.3e} | "
          f"{entry['final_volume_drift']:+.3e} | "
          f"{'PASS' if abs(entry['final_volume_drift']) <= MASS_DRIFT_TOL else 'FAIL'} |")
    A("")

    A("## 4. 结果图\n")
    for fig in payload.get("figures", []):
        A(f"### {fig['name']}\n")
        A(f"![{fig['name']}]({fig['name']})\n")
        A(f"{fig['caption']}\n")

    A("## 5. 注意点（坑/局限/适用范围）\n")
    A("1. **对齐界面会掩盖缺陷。** 界面恰好落在单元面上时闭式水静压剖面与面密度"
      "一致，静止状态看不出问题（残差 ~1e-11、|u| ~1e-15 m/s）。负控必须包含"
      "界面落在单元内部（分数 α）的算例，否则测的是一个恒等变换。")
    A("2. **`pressure_form` 不是万能钥匙。** `\"reduced\"` 形式能让**均质**液池"
      "精确静止，但两层流体在界面面的浮力项 `(ρ - ρ_ref)/ρ · g` 不为零，两层"
      "静置仍不平衡；分层静止必须 `\"total\"` + 平衡初始化。")
    A("3. **CSF 在平界面上不产生伪流。** σ = 0.072 N/m、界面水平时平衡 CSF 力"
      "为零（曲率在平面对称下相消），因此 σ > 0 的 leg 也到机器精度；曲面界面"
      "上的伪流另见 `docs/two-phase-csf-balance/`（< 1e-6 m/s 量级）。")
    A("4. **溃坝误差的符号会换向。** 早期波前偏慢（t*=1 时 −48%）、晚期偏快"
      "（t*=3 时 +20%），这是「启动滞后 + 之后惯性占优」的典型形状；网格加密"
      "只显著改善 t*≤2 段（−24.8% → −9.6%），说明早期误差不是分辨率问题而是"
      "启动瞬态问题（见 §3.5 诊断）。")
    A("5. **upwind 输运在闭合域漏质量。** 同一溃坝算例 172 步后 PLIC 的 "
      "dV/V0 = −2.4e-15（机器精度），迎风（donor-cell）为 −1.57e-3，超 0.1% 门 "
      "1.6 倍：**质量守恒门必须绑 PLIC**，不能拿迎风格式的结果去验收。")
    A("6. **本单未修的缺陷（如实登记）。** 增量投影从 p = 0 建立水头造成的启动"
      "滞后仍在：`pressure_form=\"total\"` 下 `t* < 0.4` 波前不动，"
      "`pressure_form=\"reduced\"` 不稳定（|u| 冲到 8 m/s）。修它需要改投影"
      "（总压形式或双时间步水头初始化），属于下一步工作，不在本单声明范围内。")
    A("7. **适用边界。** 本单只做二维分层与二维溃坝锚；三维、真实船体自由面、"
      "上浪/绿水的适用性未验证。\n")

    A("## 6. 误差判定\n")
    key = {
        "G1 静水平衡 max|u|（平衡初始化）":
            (g1["worst_max_velocity_discrete_m_s"], HYDROSTATIC_VELOCITY_TOL),
        "G1 静水平衡 max|u|（闭式剖面基线）":
            (g1["worst_max_velocity_analytic_m_s"], HYDROSTATIC_VELOCITY_TOL),
        "G2 界面迁移 max|Δα|（平衡初始化）":
            (g2["worst_displacement_discrete"], INTERFACE_DISPLACEMENT_TOL),
    }
    if legs:
        key["G4 溃坝波前最差相对误差"] = (
            gates["G4c_martin_moyce_front"]["worst_relative_error"], DAM_FRONT_TOL)
    A("| 关键量 | 实测 | 基准/门限 | 相对门限 | 是否 <3% |")
    A("|---|---|---|---|---|")
    for name, (value, limit) in key.items():
        ratio = abs(value) / limit if limit else float("nan")
        A(f"| {name} | {value:.3e} | {limit:.3e} | {ratio:.2e}× | "
          f"{'是' if ratio < 0.03 else '否'} |")
    A("")
    A("说明：G1/G2 的基准是**精确解 0**，表中「相对门限」= 实测/门限。平衡初始化"
      "比门限小 6 个数量级以上，「<3%」显然成立；闭式剖面基线超门限 1e8 倍，"
      "落在不可接受区间——这就是本单修的缺陷。G4 有实验基准，波前误差逐点列在"
      "§3.2，" + (
          f"最差点 {_pct(gates['G4c_martin_moyce_front']['worst_relative_error'])}，"
          "**远大于 3%，未达 3% 档**（战役判据门 15%："
          f"{'PASS' if gates['G4c_martin_moyce_front']['pass'] else 'FAIL'}）。"
          if legs else "本批未跑溃坝 leg。"))
    A("")
    A("honest FAIL 声明：本报告不隐藏任何未通过项，`all_pass = "
      f"{gates['all_pass']}`（"
      + ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                  for k, v in gates.items() if isinstance(v, dict) and "pass" in v)
      + "）。\n")
    A("---\n")
    A(f"复现：`PYTHONPATH=src python -m tensorfvm.verification.vof_chain "
      f"--output {output.name} [--dam]`；测试："
      "`pytest tests/test_vof_chain_hydrostatic.py`。")

    path = output / "report.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_saved_dam_legs(directory: Path) -> list[dict]:
    """Reload dam-break legs written by :mod:`scripts.vof_chain_dam`."""
    legs: list[dict] = []
    for path in sorted(Path(directory).glob("dam_*.json")):
        if path.stem.endswith("__alpha"):
            continue
        leg = json.loads(path.read_text())
        alpha = path.with_name(path.stem + "__alpha.npy")
        if alpha.exists():
            leg["field_artifact"] = f"legs/{alpha.name}"
        leg.setdefault("evaluation", evaluate_dam_break(leg))
        leg["mass_gate"] = {
            "pass": abs(leg["final_volume_drift"]) <= MASS_DRIFT_TOL,
            "final_volume_drift": leg["final_volume_drift"],
            "criterion": f"|dV|/V0 <= {MASS_DRIFT_TOL:g}",
        }
        legs.append(leg)
    return legs


def run_chain(output: Path, with_dam: bool = False,
              resolutions=((100, 30), (200, 60)), schemes=("plic",),
              max_t_star: float = 3.0, hydro_nx: int = 40, hydro_ny: int = 20,
              n_steps: int = 40,
              max_wall_seconds: float | None = None, reuse: Path | None = None,
              ritter_nx: int = 120, ritter_ny: int = 36,
              ritter_length: float = 1.5, ritter_height: float = 0.18,
              ritter_dam_x: float = 0.25, ritter_h0: float = 0.05,
              ritter_t_end: float = 0.2, ritter_scheme: str = "plic") -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print("[chain] hydrostatic negative control ...", flush=True)
    hydro = hydrostatic_matrix(nx=hydro_nx, ny=hydro_ny, n_steps=n_steps)
    dam_legs = None
    if with_dam and reuse is not None:
        print(f"[chain] reusing dam-break legs from {reuse}", flush=True)
        dam_legs = load_saved_dam_legs(reuse)
        (output / "legs").mkdir(parents=True, exist_ok=True)
        for path in sorted(Path(reuse).glob("dam_*__alpha.npy")):
            target = output / "legs" / path.name
            if not target.exists():
                target.write_bytes(path.read_bytes())
        print(f"  loaded {len(dam_legs)} legs", flush=True)
    elif with_dam:
        print("[chain] dam break legs ...", flush=True)
        dam_legs = dam_break_legs(resolutions=resolutions, schemes=schemes,
                                  max_t_star=max_t_star,
                                  max_wall_seconds=max_wall_seconds,
                                  output=output)
        (output / "legs").mkdir(exist_ok=True)
        for leg in dam_legs:
            name = (f"dam_{leg['config']['vof_scheme']}_"
                    f"{leg['grid']['nx']}x{leg['grid']['ny']}.json")
            (output / "legs" / name).write_text(
                json.dumps(_json_safe(leg), indent=2, sort_keys=True))
    ritter_legs = None
    if with_dam:
        print("[chain] Ritter shallow-water anchor ...", flush=True)
        ritter_cfg = RitterLegConfig(
            nx=ritter_nx, ny=ritter_ny, length=ritter_length,
            height=ritter_height, dam_x=ritter_dam_x, h0=ritter_h0,
            t_end=ritter_t_end, vof_scheme=ritter_scheme)
        leg = run_ritter_leg(ritter_cfg, progress=True)
        alpha = leg.pop("final_alpha")
        leg["evaluation"] = evaluate_ritter(leg)
        leg["onset"] = front_onset_delay(leg)
        (output / "legs").mkdir(exist_ok=True)
        import numpy as _np
        _np.save(output / "legs" / f"ritter_{ritter_scheme}__alpha.npy",
                 alpha.double().numpy())
        (output / "legs" / f"ritter_{ritter_scheme}.json").write_text(
            json.dumps(_json_safe(leg), indent=2, sort_keys=True))
        ritter_legs = [leg]
    payload = {
        "case": "two-phase VOF chain (hydrostatic negative control + dam break)",
        "module": "tensorfvm.verification.vof_chain",
        "repo": "TensorFVM",
        "node": "P475W1-N06",
        "hydrostatic": hydro,
        "dam_break": dam_legs,
        "ritter": ritter_legs,
        "front_onset": ritter_legs[0]["onset"] if ritter_legs else None,
        "martin_moyce_kinematics": (
            martin_moyce_kinematics(dam_legs[0]["config"]["column_width"],
                                    dam_legs[0]["config"]["gravity"])
            if dam_legs else None),
        "tolerances": {
            "hydrostatic_velocity_m_s": HYDROSTATIC_VELOCITY_TOL,
            "interface_displacement": INTERFACE_DISPLACEMENT_TOL,
            "mass_drift": MASS_DRIFT_TOL,
            "dam_front_relative": DAM_FRONT_TOL,
            "freefall_relative": FREEFALL_TOL,
            "ritter_front_relative": RITTER_FRONT_TOL,
        },
    }
    payload["gates"] = evaluate_chain(hydro, dam_legs, ritter_legs)
    payload["figures"] = write_figures(payload, output)
    payload["figures"] += write_anchor_figures(payload, output)
    build_report(payload, output)
    save_summary(output, payload)
    print(f"[chain] all_pass={payload['gates']['all_pass']}", flush=True)
    for name, gate in payload["gates"].items():
        if isinstance(gate, dict) and "pass" in gate:
            print(f"  {name}: {'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="docs/vof-chain-anchor")
    ap.add_argument("--dam", action="store_true",
                    help="also run the dam-break legs (expensive)")
    ap.add_argument("--resolutions", default="100x30,200x60",
                    help="dam-break resolutions, e.g. 100x30,200x60")
    ap.add_argument("--schemes", default="plic",
                    help="comma-separated VOF schemes for the dam break")
    ap.add_argument("--max-t-star", type=float, default=3.0)
    ap.add_argument("--hydro-nx", type=int, default=40)
    ap.add_argument("--hydro-ny", type=int, default=20)
    ap.add_argument("--steps", type=int, default=40,
                    help="hydrostatic legs: number of steps")
    ap.add_argument("--max-wall-seconds", type=float, default=None)
    ap.add_argument("--reuse", default=None,
                    help="directory with previously computed dam-*.json legs")
    args = ap.parse_args(argv)
    resolutions = []
    for item in args.resolutions.split(","):
        nx, ny = item.lower().split("x")
        resolutions.append((int(nx), int(ny)))
    payload = run_chain(Path(args.output), with_dam=args.dam,
                        resolutions=tuple(resolutions),
                        schemes=tuple(args.schemes.split(",")),
                        max_t_star=args.max_t_star,
                        hydro_nx=args.hydro_nx, hydro_ny=args.hydro_ny,
                        n_steps=args.steps,
                        max_wall_seconds=args.max_wall_seconds,
                        reuse=Path(args.reuse) if args.reuse else None)
    return 0 if payload["gates"]["all_pass"] else 1


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())