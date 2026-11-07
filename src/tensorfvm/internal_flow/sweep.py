"""Single-configuration runners for the internal-flow verification face.

:mod:`tensorfvm.internal_flow.study` runs a whole family (every mesh, every
Reynolds number) in one process, which is convenient on a laptop and useless on
a cluster: the fine meshes dominate the wall clock and nothing is parallel.
This module exposes the *atomic* unit of work -- one case, one parameter -- so
that a grid-convergence study can be split across machines:

    python -m tensorfvm.internal_flow.sweep --case bfs --param 40 --out DIR

Every run writes ``raw_<case>_<tag>.json`` (scalars, criteria, history) and
``fields_<case>_<tag>.npz`` (the field snapshot the report figures are made
from), plus a ``checkpoint_<case>_<tag>.pt`` so an interrupted run can be
resumed with ``--resume``.  Nothing is transcribed by hand: the JSON is
produced from the computed fields by the same code that decides whether the run
has reached a steady state.

Steady state is declared on *monitored physical quantities*, not on the
velocity residual alone: the reattachment length of a laminar recirculation
approaches its asymptote algebraically, so a residual of ``1e-6`` can still be
several percent away from the steady reattachment point.  A run stops when
every monitored quantity has moved by less than ``--tol`` (relative) over
``--blocks`` consecutive blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .backward_step import BackwardStepCase, BackwardStepConfig
from .natural_convection import NaturalConvectionCase, NaturalConvectionConfig
from .reference import (BACKWARD_STEP_REFERENCE, DE_VAHL_DAVIS_NU,
                        borda_carnot_loss, channel_wall_friction,
                        developed_channel_friction)
from .sudden_expansion import SuddenExpansionCase, SuddenExpansionConfig

SWEEP_VERSION = "internal-flow-sweep-1"


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _stationary(history: list[dict], keys: tuple[str, ...], tol: float,
                blocks: int) -> bool:
    """True once every ``key`` changed by less than ``tol`` (relative) per block."""
    if len(history) <= blocks:
        return False
    tail = history[-(blocks + 1):]
    for key in keys:
        values = [entry.get(key) for entry in tail]
        if any(value is None for value in values):
            return False
        for previous, current in zip(values, values[1:]):
            scale = max(abs(previous), abs(current), 1e-12)
            if abs(current - previous) / scale > tol:
                return False
    return True


def _march(solver, block: int, max_steps: int, keys: tuple[str, ...], monitor,
           tol: float, blocks: int, history: list[dict], checkpoint: Path) -> dict:
    """March in blocks until every monitored quantity is stationary."""
    started = time.time()
    while solver.steps < max_steps:
        solver.run(max_steps=block, tolerance=1e-30)
        entry = {"steps": solver.steps, "residual": _finite(solver.residual)}
        entry.update(monitor())
        history.append(entry)
        torch.save({"u": solver.u, "v": solver.v, "theta": solver.theta,
                    "steps": solver.steps}, checkpoint)
        if _stationary(history, keys, tol, blocks):
            break
        if any(entry.get(key) is None for key in keys) and len(history) > 8 * blocks:
            break
    return {"stationary": _stationary(history, keys, tol, blocks),
            "wall_seconds": time.time() - started}


def _resume(solver, checkpoint: Path) -> int:
    if not checkpoint.exists():
        return 0
    state = torch.load(checkpoint, map_location="cpu")
    solver.u = state["u"].clone()
    solver.v = state["v"].clone()
    if solver.config.energy and state.get("theta") is not None:
        solver.theta = state["theta"].clone()
    solver.steps = int(state["steps"])
    solver.apply_boundary_conditions()
    return solver.steps


def save_fields(case, out: Path, tag: str, kind: str) -> str:
    """Store the field snapshot that the report figures are drawn from."""
    import numpy

    solver = case.solver
    payload = {"u": solver.u.numpy(), "v": solver.v.numpy(),
               "p": solver.p.numpy(), "nx": solver.nx, "ny": solver.ny,
               "hx": float(solver.hx), "hy": float(solver.hy),
               "steps": solver.steps}
    if solver.config.energy:
        payload["theta"] = solver.theta.numpy()
    path = out / f"fields_{kind}_{tag}.npz"
    numpy.savez_compressed(path, **payload)
    return str(path)


def _prolong(case, kind: str, init_from: Path | None, init_param: int | None,
             reynolds: float | None, rayleigh: float | None, cfl: float,
             wall_gradient: str) -> int:
    """Initialise ``case`` from the converged state of the next coarser mesh.

    The interpolated coarse state removes the start-up transient of the fine
    mesh (the equations, and therefore the converged answer, are unchanged);
    on the expensive meshes it is the difference between a run that finishes
    and one that does not.
    """
    if init_from is None:
        return 0
    state = torch.load(init_from, map_location="cpu")
    if kind == "bfs":
        coarse = BackwardStepCase(BackwardStepConfig(
            cells_per_step_height=init_param, reynolds=reynolds or 800.0, cfl=cfl))
    elif kind == "expansion":
        coarse = SuddenExpansionCase(SuddenExpansionConfig(
            cells_per_downstream_height=init_param, reynolds=reynolds or 100.0, cfl=cfl))
    else:
        coarse = NaturalConvectionCase(NaturalConvectionConfig(
            n=init_param, rayleigh=rayleigh or 1e4, cfl=cfl,
            wall_gradient=wall_gradient))
    coarse.solver.u = state["u"].clone()
    coarse.solver.v = state["v"].clone()
    if coarse.solver.config.energy and state.get("theta") is not None:
        coarse.solver.theta = state["theta"].clone()
    coarse.solver.apply_boundary_conditions()
    case.solver.prolong_from(coarse.solver)
    return int(state["steps"])


# --------------------------------------------------------------------- cases
def run_bfs(n: int, out: Path, max_steps: int = 200_000, cfl: float = 0.5,
            reynolds: float = 800.0, block: int | None = None,
            tol: float = 2e-3, blocks: int = 3, resume: bool = False,
            init_from: Path | None = None, init_param: int | None = None) -> dict:
    """Backward-facing step; ``reynolds`` is based on the step height and ``U_in``."""
    case = BackwardStepCase(BackwardStepConfig(
        cells_per_step_height=n, reynolds=reynolds, cfl=cfl,
        max_steps=max_steps, tolerance=1e-30, check_every=200))
    solver = case.solver
    checkpoint = out / f"checkpoint_bfs_{n}.pt"
    prolonged_from = _prolong(case, "bfs", init_from, init_param, reynolds, None,
                              cfl, "second_order")
    resumed_from = _resume(solver, checkpoint) if resume else 0
    history: list[dict] = []

    def monitor() -> dict:
        xr = case.reattachment()
        return {"xr_step_heights": _finite(None if xr is None else xr)}

    march = _march(solver, block or max(400, 20 * n), max_steps,
                   ("xr_step_heights",), monitor, tol, blocks, history, checkpoint)

    x_cf, cf = case.cf_profile()
    shear = solver.bottom_wall_friction()
    reference = solver.config.reference_velocity
    xr = case.reattachment()
    developed = channel_wall_friction(2.0, 0.5 * reference,
                                      case.config.viscosity, reference)
    stations = {}
    for station in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
        index = int(round(station * n))
        if 0 <= index <= solver.nx:
            stations[f"{station:g}"] = float(shear[index])
    profiles = {}
    for station in (2.0, 4.0, 8.0, 16.0):
        index = int(round(station * n))
        if 0 < index <= solver.nx:
            profiles[f"{station:g}"] = solver.u[index, :].tolist()
    target = BACKWARD_STEP_REFERENCE["campaign_target"]
    metrics = {
        "case": "backward_step", "param": n, "sweep_version": SWEEP_VERSION,
        "cells_per_step_height": n,
        "mesh": {"nx": solver.nx, "ny": solver.ny, "hx": float(solver.hx),
                 "hy": float(solver.hy)},
        "reynolds_step_height": reynolds,
        "reynolds_downstream_height": 2.0 * reynolds,
        "viscosity": case.config.viscosity,
        "inlet_mean_velocity": reference,
        "steps": solver.steps, "dt": float(solver.dt),
        "residual": _finite(solver.residual),
        "max_divergence": _finite(float(solver.divergence().abs().max())),
        "reattachment_step_heights": _finite(xr),
        "reattachment_downstream_heights": _finite(None if xr is None else xr / 2.0),
        "cf_stations": stations,
        "cf_outlet": float(shear[-1]),
        "cf_developed_analytic": developed,
        "cf_developed_relative_error": abs(stations.get("16", float("nan")) - developed)
        / developed if "16" in stations else None,
        "u_profiles": profiles,
        "x_cf": x_cf.tolist(), "cf": cf.tolist(),
        "stationary": march["stationary"],
        "stationarity_rule": {"relative_change_per_block": tol, "consecutive_blocks": blocks,
                              "block_steps": block or None},
        "wall_seconds": march["wall_seconds"],
        "resumed_from_step": resumed_from,
        "prolonged_from_step": prolonged_from,
        "history": history,
        "campaign_target_downstream_heights": target,
        "campaign_band_downstream_heights": BACKWARD_STEP_REFERENCE["campaign_band"],
        "campaign_relative_error": None if xr is None
        else abs(xr / 2.0 - target) / target,
        "developed_cf_analytic_re_800": developed_channel_friction(reynolds),
        "fields": Path(save_fields(case, out, str(n), "bfs")).name,
    }
    return metrics


def run_expansion(n: int, out: Path, max_steps: int = 200_000, cfl: float = 0.5,
                  reynolds: float = 100.0, block: int | None = None,
                  tol: float = 2e-3, blocks: int = 3, resume: bool = False,
                  init_from: Path | None = None, init_param: int | None = None) -> dict:
    """Planar sudden expansion, area ratio 4, Reynolds number ``reynolds``."""
    case = SuddenExpansionCase(SuddenExpansionConfig(
        cells_per_downstream_height=n, reynolds=reynolds, cfl=cfl,
        max_steps=max_steps, tolerance=1e-30, check_every=200))
    solver = case.solver
    checkpoint = out / f"checkpoint_expansion_{n}.pt"
    prolonged_from = _prolong(case, "expansion", init_from, init_param, reynolds, None,
                              cfl, "second_order")
    resumed_from = _resume(solver, checkpoint) if resume else 0
    history: list[dict] = []

    def monitor() -> dict:
        diagnostics = case.diagnostics()
        return {"xr_bottom_over_D": _finite(diagnostics["reattachment_bottom"]),
                "xr_top_over_D": _finite(diagnostics["reattachment_top"]),
                "k_raw": _finite(diagnostics["k_raw"])}

    march = _march(solver, block or max(400, 20 * n), max_steps,
                   ("xr_bottom_over_D", "xr_top_over_D", "k_raw"),
                   monitor, tol, blocks, history, checkpoint)
    metrics = case.diagnostics()
    metrics.update({
        "case": "sudden_expansion", "param": n, "sweep_version": SWEEP_VERSION,
        "reynolds": reynolds,
        "area_ratio": case.config.area_ratio,
        "cells_per_downstream_height": n,
        "mesh": {"nx": solver.nx, "ny": solver.ny, "hx": float(solver.hx),
                 "hy": float(solver.hy)},
        "inlet_height_over_downstream_height": 1.0 / case.config.area_ratio,
        "borda_carnot": borda_carnot_loss(case.config.area_ratio),
        "stationary": march["stationary"],
        "stationarity_rule": {"relative_change_per_block": tol, "consecutive_blocks": blocks,
                              "block_steps": block or None},
        "wall_seconds": march["wall_seconds"],
        "resumed_from_step": resumed_from,
        "prolonged_from_step": prolonged_from,
        "history": history,
        "fields": Path(save_fields(case, out, str(n), "expansion")).name,
    })
    return metrics


def run_cavity(n: int, out: Path, rayleigh: float, prandtl: float = 0.71,
               max_steps: int = 400_000, cfl: float = 0.5,
               wall_gradient: str = "second_order", block: int | None = None,
               tol: float = 2e-3, blocks: int = 3, resume: bool = False,
               init_from: Path | None = None, init_param: int | None = None) -> dict:
    """Differentially heated square cavity at Rayleigh number ``rayleigh``."""
    case = NaturalConvectionCase(NaturalConvectionConfig(
        n=n, rayleigh=rayleigh, prandtl=prandtl, cfl=cfl,
        wall_gradient=wall_gradient, max_steps=max_steps, tolerance=1e-30,
        check_every=200))
    solver = case.solver
    tag = f"Ra{rayleigh:g}_N{n}"
    checkpoint = out / f"checkpoint_cavity_{tag}.pt"
    prolonged_from = _prolong(case, "cavity", init_from, init_param, None, rayleigh,
                              cfl, wall_gradient)
    resumed_from = _resume(solver, checkpoint) if resume else 0
    history: list[dict] = []

    def monitor() -> dict:
        diagnostics = case.diagnostics()
        return {"nu_mean": _finite(diagnostics["nu_mean"]),
                "psi_min": _finite(diagnostics["psi_min"])}

    march = _march(solver, block or max(400, 20 * n), max_steps,
                   ("nu_mean", "psi_min"), monitor, tol, blocks, history, checkpoint)
    metrics = case.diagnostics()
    wall = 0.5 * case.config.temperature_drop
    nu_first_local = (wall - solver.theta[0, :]) / (0.5 * solver.hx)
    metrics.update({
        "case": "natural_convection", "param": n, "sweep_version": SWEEP_VERSION,
        "wall_gradient": wall_gradient,
        "mesh": {"nx": solver.nx, "ny": solver.ny, "hx": float(solver.hx),
                 "hy": float(solver.hy)},
        "nu_mean_first_order": float(nu_first_local.mean()),
        "stationary": march["stationary"],
        "stationarity_rule": {"relative_change_per_block": tol, "consecutive_blocks": blocks,
                              "block_steps": block or None},
        "wall_seconds": march["wall_seconds"],
        "resumed_from_step": resumed_from,
        "prolonged_from_step": prolonged_from,
        "history": history,
        "reference": DE_VAHL_DAVIS_NU.get(rayleigh),
        "fields": Path(save_fields(case, out, tag, "cavity")).name,
    })
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["bfs", "expansion", "cavity"])
    parser.add_argument("--param", type=int, required=True,
                        help="cells per step height / downstream height / cavity side")
    parser.add_argument("--rayleigh", type=float, default=None)
    parser.add_argument("--reynolds", type=float, default=None)
    parser.add_argument("--prandtl", type=float, default=0.71)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument("--tol", type=float, default=2e-3)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wall-gradient", default="second_order")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint of the coarser mesh to prolongate from")
    parser.add_argument("--init-param", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common = {"max_steps": args.max_steps, "cfl": args.cfl, "block": args.block,
              "tol": args.tol, "blocks": args.blocks, "resume": args.resume,
              "init_from": Path(args.init_from) if args.init_from else None,
              "init_param": args.init_param}
    if args.case == "bfs":
        tag = str(args.param)
        metrics = run_bfs(args.param, out, reynolds=args.reynolds or 800.0, **common)
    elif args.case == "expansion":
        tag = str(args.param)
        metrics = run_expansion(args.param, out, reynolds=args.reynolds or 100.0,
                                **common)
    else:
        if args.rayleigh is None:
            parser.error("--rayleigh is required for the cavity")
        tag = f"Ra{args.rayleigh:g}_N{args.param}"
        metrics = run_cavity(args.param, out, args.rayleigh, prandtl=args.prandtl,
                             wall_gradient=args.wall_gradient, **common)
    path = out / f"raw_{args.case}_{tag}.json"
    path.write_text(json.dumps(metrics, indent=1))
    print(f"[{args.case} {tag}] steps={metrics['steps']} "
          f"stationary={metrics['stationary']} -> {path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())