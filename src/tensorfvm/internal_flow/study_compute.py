"""Computation-only driver for the internal-flow verification.

This module runs the three cases and writes the JSON ledger without
requiring matplotlib.  Plotting is done separately by :mod:`study`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .backward_step import BackwardStepCase, BackwardStepConfig
from .natural_convection import NaturalConvectionCase, NaturalConvectionConfig
from .reference import (BACKWARD_STEP_REFERENCE, DE_VAHL_DAVIS_NU,
                        borda_carnot_loss, developed_channel_friction)
from .sudden_expansion import SuddenExpansionCase, SuddenExpansionConfig

STUDY_VERSION = "internal-flow-verification-1"

ACCEPTANCE = {
    "bfs_reattachment_in_downstream_heights": 0.2 / 6.0,
    "expansion_loss_coefficient": 0.05,
    "cavity_nusselt": 0.03,
}


def _dump_partial(out: Path, key: str, ledger: dict) -> None:
    path = out / "summary.json"
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.setdefault("study_version", STUDY_VERSION)
    existing[key] = ledger
    path.write_text(json.dumps(existing, indent=1))


def run_backward_step(meshes: list[int], out: Path, max_steps: int = 60000,
                      tolerance: float = 2e-6, cfl: float = 0.5,
                      reynolds: float = 800.0) -> dict:
    started = time.time()
    cases: dict[int, BackwardStepCase] = {}
    ledger: dict[str, object] = {"meshes": {}, "config": {
        "cells_per_step_height": meshes, "max_steps": max_steps,
        "tolerance": tolerance, "cfl": cfl, "reynolds": reynolds,
    }}
    previous = None
    for n in meshes:
        case = BackwardStepCase(BackwardStepConfig(
            cells_per_step_height=n, reynolds=reynolds, cfl=cfl,
            max_steps=max_steps, tolerance=tolerance, check_every=200))
        if previous is not None:
            case.solver.prolong_from(previous.solver)
        wall = time.time()
        block = max(500, 100 * n)
        history: list[tuple[int, float | None]] = []
        stationary = 0
        previous_xr = None
        steps = 0
        while steps < max_steps:
            case.solver.run(max_steps=block, tolerance=1e-30)
            steps += block
            current = case.reattachment()
            history.append((steps, current))
            if current is not None and previous_xr is not None and abs(
                    current - previous_xr) <= 2e-3 * abs(current):
                stationary += 1
                if stationary >= 2:
                    break
            else:
                stationary = 0
            previous_xr = current
        diagnostics = case.diagnostics()
        diagnostics["wall_seconds"] = time.time() - wall
        diagnostics["steps"] = steps
        diagnostics["converged"] = stationary >= 2
        diagnostics["reattachment_history"] = history
        ledger["meshes"][str(n)] = diagnostics
        cases[n] = case
        previous = case
        _dump_partial(out, "backward_step", dict(ledger))
        print(f"[bfs] n={n} steps={steps} xr/H_out="
              f"{diagnostics['reattachment_in_downstream_heights']} "
              f"stationary={diagnostics['converged']} "
              f"({diagnostics['wall_seconds']:.0f}s)", flush=True)

    finest = meshes[-1]
    reference = BACKWARD_STEP_REFERENCE
    xr_values = [ledger["meshes"][str(n)]["reattachment_in_downstream_heights"]
                 for n in meshes]
    target = reference["campaign_target"]
    ledger["acceptance"] = {
        "target_reattachment_in_downstream_heights": target,
        "band": reference["campaign_band"],
        "finest_mesh_value": xr_values[-1],
        "finest_mesh_error": None if xr_values[-1] is None
        else (xr_values[-1] - target),
        "relative_error": None if xr_values[-1] is None
        else abs(xr_values[-1] - target) / target,
        "garlting_reference": reference["value"],
        "garlting_relative_error": None if xr_values[-1] is None
        else abs(xr_values[-1] - reference["value"]) / reference["value"],
        "within_band": None if xr_values[-1] is None
        else abs(xr_values[-1] - target) <= reference["campaign_band"],
        "developed_cf_analytic": developed_channel_friction(
            ledger["config"]["reynolds"]),
        "cf_outlet_finest": ledger["meshes"][str(finest)]["cf_at_outlet"],
    }
    analytic = developed_channel_friction(ledger["config"]["reynolds"])
    ledger["acceptance"]["cf_relative_error"] = abs(
        ledger["acceptance"]["cf_outlet_finest"] - analytic) / analytic

    ledger["wall_seconds_total"] = time.time() - started
    return ledger


def run_sudden_expansion(meshes: list[int], out: Path, max_steps: int = 120000,
                         tolerance: float = 2e-6, cfl: float = 0.5) -> dict:
    started = time.time()
    ledger: dict[str, object] = {"meshes": {}, "config": {
        "cells_per_downstream_height": meshes, "max_steps": max_steps,
        "tolerance": tolerance, "cfl": cfl}}
    cases: dict[int, SuddenExpansionCase] = {}
    previous = None
    for n in meshes:
        case = SuddenExpansionCase(SuddenExpansionConfig(
            cells_per_downstream_height=n, cfl=cfl, max_steps=max_steps,
            tolerance=tolerance, check_every=200))
        if previous is not None:
            case.solver.prolong_from(previous.solver)
        wall = time.time()
        block = max(1000, 20 * n)
        history: list[tuple[int, float | None]] = []
        stationary = 0
        previous_xr = None
        steps = 0
        while steps < max_steps:
            case.solver.run(max_steps=block, tolerance=1e-30)
            steps += block
            current = case.reattachment("bottom")
            history.append((steps, current))
            if current is not None and previous_xr is not None and abs(
                    current - previous_xr) <= 2e-3 * abs(current):
                stationary += 1
                if stationary >= 2:
                    break
            else:
                stationary = 0
            previous_xr = current
        diagnostics = case.diagnostics()
        diagnostics["wall_seconds"] = time.time() - wall
        diagnostics["steps"] = steps
        diagnostics["converged"] = stationary >= 2
        diagnostics["reattachment_history"] = history
        ledger["meshes"][str(n)] = diagnostics
        cases[n] = case
        previous = case
        _dump_partial(out, "sudden_expansion", dict(ledger))
        print(f"[expansion] n={n} steps={steps} K_corr={diagnostics['k_corrected']:.4f} "
              f"K_raw={diagnostics['k_raw']:.4f} Lr={diagnostics['reattachment_mean_over_D']} "
              f"stationary={diagnostics['converged']} "
              f"({diagnostics['wall_seconds']:.0f}s)", flush=True)

    finest = meshes[-1]
    last = ledger["meshes"][str(finest)]
    theory = borda_carnot_loss()
    ledger["acceptance"] = {
        "borda_carnot": theory,
        "k_corrected_finest": last["k_corrected"],
        "k_extrapolated_finest": last["k_extrapolated"],
        "k_raw_finest": last["k_raw"],
        "k_friction_correction_finest": last["k_friction_correction"],
        "k_relative_error": abs(last["k_corrected"] - theory) / theory,
        "k_within_5_percent": bool(abs(last["k_corrected"] - theory) / theory < 0.05),
        "cp_momentum_balance": last["cp_recovery_momentum_balance"],
        "cp_extrapolated_finest": last["cp_recovery_extrapolated"],
        "cp_raw_finest": last["cp_recovery_raw"],
        "cp_relative_error": abs(last["cp_recovery_raw"]
                                 - last["cp_recovery_momentum_balance"])
        / last["cp_recovery_momentum_balance"],
        "reattachment_over_D_finest": last["reattachment_mean_over_D"],
        "reattachment_reference": None,
        "reattachment_reference_note": (
            "no literature value for the planar AR=4 case at Re_D=100 was certified "
            "for this run; the number is reported with its mesh sequence only "
            "(status WITHHELD)"),
    }

    ledger["wall_seconds_total"] = time.time() - started
    return ledger


def run_natural_convection(meshes: dict[float, list[int]], out: Path,
                           max_steps: int = 120000, tolerance: float = 2e-6,
                           cfl: float = 0.5) -> dict:
    started = time.time()
    ledger: dict[str, object] = {"cases": {}, "config": {
        "meshes_per_rayleigh": {str(k): v for k, v in meshes.items()},
        "max_steps": max_steps, "tolerance": tolerance, "cfl": cfl}}
    fields: dict[float, tuple[NaturalConvectionCase, int]] = {}
    for rayleigh, mesh_seq in meshes.items():
        previous = None
        for n in mesh_seq:
            case = NaturalConvectionCase(NaturalConvectionConfig(
                n=n, rayleigh=rayleigh, cfl=cfl, max_steps=max_steps,
                tolerance=tolerance, check_every=200))
            if previous is not None:
                case.solver.prolong_from(previous.solver)
            wall = time.time()
            block = max(2000, 50 * n)
            history: list[tuple[int, float | None]] = []
            stationary = 0
            previous_nu = None
            steps = 0
            while steps < max_steps:
                case.solver.run(max_steps=block, tolerance=1e-30)
                steps += block
                current = case.diagnostics()["nu_mean"]
                history.append((steps, current))
                if previous_nu is not None and abs(
                        current - previous_nu) <= 1e-3 * abs(current):
                    stationary += 1
                    if stationary >= 2:
                        break
                else:
                    stationary = 0
                previous_nu = current
            diagnostics = case.diagnostics()
            diagnostics["wall_seconds"] = time.time() - wall
            diagnostics["steps"] = steps
            diagnostics["converged"] = stationary >= 2
            diagnostics["nu_history"] = history
            ledger["cases"].setdefault(str(int(rayleigh)), {})[str(n)] = diagnostics
            fields[rayleigh] = (case, n)
            previous = case
            _dump_partial(out, "natural_convection", dict(ledger))
            print(f"[cavity] Ra={rayleigh:g} n={n} steps={steps} "
                  f"Nu={diagnostics['nu_mean']:.4f} "
                  f"stationary={diagnostics['converged']} "
                  f"({diagnostics['wall_seconds']:.0f}s)", flush=True)

    acceptance: dict[str, object] = {}
    for rayleigh in meshes:
        finest = meshes[rayleigh][-1]
        data = ledger["cases"][str(int(rayleigh))][str(finest)]
        reference = DE_VAHL_DAVIS_NU.get(rayleigh)
        value = data["nu_mean"]
        acceptance[str(int(rayleigh))] = {
            "nu_reference": reference,
            "nu_computed": value,
            "nu_relative_error": None if reference is None
            else abs(value - reference) / reference,
            "nu_within_3_percent": bool(reference is not None
                                        and abs(value - reference) / reference < 0.03),
            "mesh": int(finest),
        }
    ledger["acceptance"] = acceptance
    ledger["wall_seconds_total"] = time.time() - started
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/internal-flow")
    parser.add_argument("--only", default="bfs,expansion,cavity")
    parser.add_argument("--bfs-meshes", default="10,20,40")
    parser.add_argument("--bfs-reynolds", type=float, default=800.0)
    parser.add_argument("--expansion-meshes", default="20,40,80")
    parser.add_argument("--cavity-meshes", default="32,64",
                        help="comma separated n, last is finest; separate per-Rayleigh "
                             "lists with ';' (one list is repeated for every Rayleigh)")
    parser.add_argument("--cavity-rayleigh", default="10000,100000,1000000")
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(max(1, args.threads))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    summary: dict[str, object] = {"study_version": STUDY_VERSION,
                                  "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                time.gmtime())}
    ledger_path = out / "summary.json"
    if ledger_path.exists():
        summary.update(json.loads(ledger_path.read_text()))

    if "bfs" in wanted:
        meshes = [int(value) for value in args.bfs_meshes.split(",") if value]
        summary["backward_step"] = run_backward_step(
            meshes, out, max_steps=args.max_steps, tolerance=args.tolerance,
            cfl=args.cfl, reynolds=args.bfs_reynolds)
        ledger_path.write_text(json.dumps(summary, indent=1))
    if "expansion" in wanted:
        meshes = [int(value) for value in args.expansion_meshes.split(",") if value]
        summary["sudden_expansion"] = run_sudden_expansion(
            meshes, out, max_steps=args.max_steps, tolerance=args.tolerance, cfl=args.cfl)
        ledger_path.write_text(json.dumps(summary, indent=1))
    if "cavity" in wanted:
        rayleighs = [float(value) for value in args.cavity_rayleigh.split(",") if value]
        groups = [group for group in args.cavity_meshes.split(";") if group.strip()]
        if len(groups) == 1:
            groups = groups * len(rayleighs)
        if len(groups) != len(rayleighs):
            parser.error("--cavity-meshes needs one list per Rayleigh number")
        cavity_meshes = {
            rayleigh: [int(value) for value in group.split(",") if value]
            for rayleigh, group in zip(rayleighs, groups)}
        summary["natural_convection"] = run_natural_convection(
            cavity_meshes, out,
            max_steps=args.max_steps, tolerance=args.tolerance, cfl=args.cfl)
        ledger_path.write_text(json.dumps(summary, indent=1))

    ledger_path.write_text(json.dumps(summary, indent=1))
    print(f"ledger written to {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
