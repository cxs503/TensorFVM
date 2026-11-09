"""P40W3-N04 internal-flow verification maturation study.

This module runs the three verification cases at the campaign parameters,
computes the control-volume momentum ledger and the grid-convergence order,
and performs the outlet-boundary sensitivity analysis.  Every number that ends
up in the JSON ledger and in the report is written here directly from the
computed fields -- nothing is copied by hand.

Cases
-----
* sudden expansion -- area ratios 2, 3, 4 (Cp recovery vs the Borda-Carnot
  momentum-balance value);
* backward-facing step -- Re = 100, 200, 400 (reattachment length vs Armaly
  1983);
* natural convection -- Ra = 1e3, 1e4, 1e5, 1e6 (mean Nusselt number vs de
  Vahl Davis 1983).

Usage
-----
``python -m tensorfvm.internal_flow.maturation --out artifacts/internal-flow-maturation``
runs the full study; ``--only expansion,step,cavity`` restricts it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .backward_step import BackwardStepCase, BackwardStepConfig
from .momentum_ledger import channel_momentum_ledger, momentum_ledger_percent
from .natural_convection import NaturalConvectionCase, NaturalConvectionConfig
from .reference import (ARMALY_1983_REATTACHMENT, ARMALY_1983_SOURCE,
                        DE_VAHL_DAVIS_NU, borda_carnot_loss)
from .sudden_expansion import SuddenExpansionCase, SuddenExpansionConfig

STUDY_VERSION = "internal-flow-maturation-1"

#: Acceptance bands of the campaign node.
ACCEPTANCE = {
    "momentum_residual_percent": 1.0,
    "expansion_cp_percent": 5.0,
    "expansion_cp_target_percent": 3.0,
    "step_xr_percent": 2.0,
    "convergence_order": 1.8,
    "outlet_drift_percent": 5.0,
}


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return str(path)


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


def _observed_order(values: list[float], h: list[float]) -> float | None:
    """Observed order of convergence from a mesh sequence.

    ``values`` and ``h`` are ordered coarse-to-fine.  The order is computed
    from the finest pair: ``p = log(Q2 - Q3) / log(h2 / h3)``.  Returns
    ``None`` when fewer than two levels are available or the differences are
    too small to be meaningful.
    """
    if len(values) < 2 or len(h) != len(values):
        return None
    for i in range(len(values) - 1, 0, -1):
        d = values[i] - values[i - 1]
        if abs(d) > 1e-12 * max(1.0, abs(values[i])):
            return math.log(abs(values[i] - values[i - 1])) / math.log(h[i - 1] / h[i])
    return None


# --------------------------------------------------------------------------- expansion
def run_sudden_expansion(area_ratios: list[float], meshes: list[int], out: Path,
                         max_steps: int = 120000, tolerance: float = 2e-6,
                         cfl: float = 0.5) -> dict:
    plt = _matplotlib()
    started = time.time()
    ledger: dict[str, object] = {"area_ratios": {}, "config": {
        "area_ratios": area_ratios, "meshes": meshes,
        "max_steps": max_steps, "tolerance": tolerance, "cfl": cfl}}
    for ar in area_ratios:
        ar_ledger: dict[str, object] = {"meshes": {}}
        previous = None
        for n in meshes:
            case = SuddenExpansionCase(SuddenExpansionConfig(
                area_ratio=ar, cells_per_downstream_height=n, cfl=cfl,
                max_steps=max_steps, tolerance=tolerance, check_every=200))
            if previous is not None:
                case.solver.prolong_from(previous.solver)
            wall = time.time()
            block = max(1000, 20 * n)
            stationary = 0
            previous_xr = None
            steps = 0
            while steps < max_steps:
                case.solver.run(max_steps=block, tolerance=1e-30)
                steps += block
                current = case.reattachment("bottom")
                if current is not None and previous_xr is not None and abs(
                        current - previous_xr) <= 2e-3 * abs(current):
                    stationary += 1
                    if stationary >= 2:
                        break
                else:
                    stationary = 0
                previous_xr = current
            diagnostics = case.diagnostics()
            ledger_cv = channel_momentum_ledger(case.solver)
            diagnostics["momentum_ledger"] = ledger_cv
            diagnostics["momentum_ledger_percent"] = momentum_ledger_percent(ledger_cv)
            diagnostics["wall_seconds"] = time.time() - wall
            diagnostics["steps"] = steps
            diagnostics["converged"] = stationary >= 2
            ar_ledger["meshes"][str(n)] = diagnostics
            previous = case
            _dump_partial(out, f"expansion_ar{ar:g}", dict(ar_ledger))
            print(f"[expansion AR={ar:g}] n={n} steps={steps} "
                  f"Cp_raw={diagnostics['cp_recovery_raw']:.4f} "
                  f"ledger={diagnostics['momentum_ledger_percent']:.3f}% "
                  f"({diagnostics['wall_seconds']:.0f}s)", flush=True)
        # convergence order from the mesh sequence
        finest = meshes[-1]
        last = ar_ledger["meshes"][str(finest)]
        cp_values = [ar_ledger["meshes"][str(n)]["cp_recovery_raw"] for n in meshes]
        h_values = [1.0 / n for n in meshes]
        order = _observed_order(cp_values, h_values)
        cp_ref = last["cp_recovery_momentum_balance"]
        cp_err = abs(last["cp_recovery_raw"] - cp_ref) / cp_ref
        ar_ledger["acceptance"] = {
            "cp_reference": cp_ref,
            "cp_computed": last["cp_recovery_raw"],
            "cp_relative_error": cp_err,
            "cp_within_5_percent": bool(cp_err < 0.05),
            "cp_within_3_percent": bool(cp_err < 0.03),
            "convergence_order": order,
            "order_pass": bool(order is not None and order >= ACCEPTANCE["convergence_order"]),
            "momentum_ledger_percent": last["momentum_ledger_percent"],
            "momentum_residual_max": last["momentum_residual"]["max"],
        }
        ledger["area_ratios"][f"AR={ar:g}"] = ar_ledger
        _dump_partial(out, f"expansion_ar{ar:g}", dict(ar_ledger))

    # figure: Cp recovery vs AR
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ars = [float(k.split("=")[1]) for k in ledger["area_ratios"]]
    cp_computed = [ledger["area_ratios"][f"AR={ar:g}"]["acceptance"]["cp_computed"]
                  for ar in ars]
    cp_ref = [2.0 * (1.0 / ar) * (1.0 - 1.0 / ar) for ar in ars]
    ax.plot(ars, cp_ref, "ks--", ms=5, label="momentum balance")
    ax.plot(ars, cp_computed, "o-", label="this solver")
    for ar, c in zip(ars, cp_computed):
        ax.annotate(f"{c:.3f}", (ar, c), textcoords="offset points",
                    xytext=(3, 8), fontsize=7)
    ax.set_xlabel("area ratio AR")
    ax.set_ylabel(r"$C_p = (p_{out} - p_{in}) / (\frac{1}{2}\rho U_{in}^2)$")
    ax.set_title("sudden expansion: static-pressure recovery")
    ax.legend(fontsize=7)
    _save(fig, out / "expansion_cp.png")
    ledger["figures"] = ["expansion_cp.png"]
    plt.close(fig)
    ledger["wall_seconds_total"] = time.time() - started
    return ledger


# --------------------------------------------------------------------------- BFS
def run_backward_step(reynolds_list: list[float], meshes: list[int], out: Path,
                      max_steps: int = 60000, tolerance: float = 2e-6,
                      cfl: float = 0.5) -> dict:
    plt = _matplotlib()
    started = time.time()
    ledger: dict[str, object] = {"reynolds": {}, "config": {
        "reynolds": reynolds_list, "meshes": meshes,
        "max_steps": max_steps, "tolerance": tolerance, "cfl": cfl}}
    for re in reynolds_list:
        re_ledger: dict[str, object] = {"meshes": {}}
        previous = None
        for n in meshes:
            case = BackwardStepCase(BackwardStepConfig(
                cells_per_step_height=n, reynolds=re, cfl=cfl,
                max_steps=max_steps, tolerance=tolerance, check_every=200))
            if previous is not None:
                case.solver.prolong_from(previous.solver)
            wall = time.time()
            block = max(500, 100 * n)
            stationary = 0
            previous_xr = None
            steps = 0
            while steps < max_steps:
                case.solver.run(max_steps=block, tolerance=1e-30)
                steps += block
                current = case.reattachment()
                if current is not None and previous_xr is not None and abs(
                        current - previous_xr) <= 2e-3 * abs(current):
                    stationary += 1
                    if stationary >= 2:
                        break
                else:
                    stationary = 0
                previous_xr = current
            diagnostics = case.diagnostics()
            ledger_cv = channel_momentum_ledger(case.solver)
            diagnostics["momentum_ledger"] = ledger_cv
            diagnostics["momentum_ledger_percent"] = momentum_ledger_percent(ledger_cv)
            diagnostics["wall_seconds"] = time.time() - wall
            diagnostics["steps"] = steps
            diagnostics["converged"] = stationary >= 2
            re_ledger["meshes"][str(n)] = diagnostics
            previous = case
            _dump_partial(out, f"step_re{re:g}", dict(re_ledger))
            print(f"[bfs Re={re:g}] n={n} steps={steps} xr/H_out="
                  f"{diagnostics['reattachment_in_downstream_heights']} "
                  f"({diagnostics['wall_seconds']:.0f}s)", flush=True)
        finest = meshes[-1]
        last = re_ledger["meshes"][str(finest)]
        xr = last["reattachment_in_downstream_heights"]
        # Armaly reference: Re_Dh = 2 * Re_H (package convention)
        re_dh = 2.0 * re
        ref_lr_s = ARMALY_1983_REATTACHMENT.get(re_dh)
        ref_xr = None if ref_lr_s is None else ref_lr_s / 2.0
        xr_err = None if ref_xr is None else abs(xr - ref_xr) / ref_xr
        xr_values = [re_ledger["meshes"][str(n)]["reattachment_in_downstream_heights"]
                     for n in meshes]
        h_values = [1.0 / n for n in meshes]
        order = _observed_order(xr_values, h_values)
        re_ledger["acceptance"] = {
            "xr_computed": xr,
            "xr_reference": ref_xr,
            "xr_reference_source": ARMALY_1983_SOURCE if ref_xr is not None else None,
            "xr_relative_error": xr_err,
            "xr_within_2_percent": bool(xr_err is not None and xr_err < 0.02),
            "convergence_order": order,
            "order_pass": bool(order is not None and order >= ACCEPTANCE["convergence_order"]),
            "momentum_ledger_percent": last["momentum_ledger_percent"],
            "momentum_residual_max": last["momentum_residual"]["max"],
        }
        ledger["reynolds"][f"Re={re:g}"] = re_ledger
        _dump_partial(out, f"step_re{re:g}", dict(re_ledger))

    # figure: X_R vs Re
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    res = [float(k.split("=")[1]) for k in ledger["reynolds"]]
    xr_comp = [ledger["reynolds"][f"Re={re:g}"]["acceptance"]["xr_computed"]
              for re in res]
    ax.semilogx(res, xr_comp, "o-", label="this solver")
    # Armaly reference (in H_out units)
    armaly_res = sorted(ARMALY_1983_REATTACHMENT)
    armaly_xr = [ARMALY_1983_REATTACHMENT[r] / 2.0 for r in armaly_res]
    ax.semilogx(armaly_res, armaly_xr, "ks--", ms=4, label="Armaly (1983)")
    ax.set_xlabel(r"$Re_H = UH/\nu$ (step height)")
    ax.set_ylabel(r"$X_R / H_{out}$")
    ax.set_title("backward-facing step: reattachment length")
    ax.legend(fontsize=7)
    _save(fig, out / "step_xr.png")
    ledger["figures"] = ["step_xr.png"]
    plt.close(fig)
    ledger["wall_seconds_total"] = time.time() - started
    return ledger


# --------------------------------------------------------------------------- cavity
def run_natural_convection(rayleighs: list[float], meshes: dict[float, list[int]],
                           out: Path, max_steps: int = 60000,
                         tolerance: float = 2e-6, cfl: float = 0.5) -> dict:
    plt = _matplotlib()
    started = time.time()
    ledger: dict[str, object] = {"cases": {}, "config": {
        "rayleighs": rayleighs, "meshes": meshes,
        "max_steps": max_steps, "tolerance": tolerance, "cfl": cfl}}
    for rayleigh in rayleighs:
        case_meshes = meshes[rayleigh]
        case_ledger: dict[str, object] = {"meshes": {}}
        previous = None
        for n in case_meshes:
            case = NaturalConvectionCase(NaturalConvectionConfig(
                n=n, rayleigh=rayleigh, cfl=cfl, max_steps=max_steps,
                tolerance=tolerance, check_every=200))
            if previous is not None:
                case.solver.prolong_from(previous.solver)
            wall = time.time()
            case.run(max_steps=max_steps, tolerance=tolerance)
            diagnostics = case.diagnostics()
            diagnostics["wall_seconds"] = time.time() - wall
            case_ledger["meshes"][str(n)] = diagnostics
            previous = case
            _dump_partial(out, f"cavity_ra{rayleigh:g}", dict(case_ledger))
            print(f"[cavity Ra={rayleigh:g}] n={n} Nu="
                  f"{diagnostics['nu_mean']:.4f} "
                  f"({diagnostics['wall_seconds']:.0f}s)", flush=True)
        finest = case_meshes[-1]
        last = case_ledger["meshes"][str(finest)]
        nu_ref = DE_VAHL_DAVIS_NU.get(rayleigh)
        nu_err = None if nu_ref is None else abs(last["nu_mean"] - nu_ref) / nu_ref
        nu_values = [case_ledger["meshes"][str(n)]["nu_mean"] for n in case_meshes]
        h_values = [1.0 / n for n in case_meshes]
        order = _observed_order(nu_values, h_values)
        case_ledger["acceptance"] = {
            "nu_computed": last["nu_mean"],
            "nu_reference": nu_ref,
            "nu_relative_error": nu_err,
            "nu_within_3_percent": bool(nu_err is not None and nu_err < 0.03),
            "convergence_order": order,
            "order_pass": bool(order is not None and order >= ACCEPTANCE["convergence_order"]),
            "momentum_residual_max": last["momentum_residual"]["max"],
        }
        ledger["cases"][f"Ra={rayleigh:g}"] = case_ledger
        _dump_partial(out, f"cavity_ra{rayleigh:g}", dict(case_ledger))

    # figure: Nu vs Ra
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.loglog(rayleighs, [DE_VAHL_DAVIS_NU[r] for r in rayleighs], "ks--",
              ms=4, label="de Vahl Davis (1983)")
    ax.loglog(rayleighs, [ledger["cases"][f"Ra={r:g}"]["acceptance"]["nu_computed"]
              for r in rayleighs], "o-", label="this solver")
    ax.set_xlabel("$Ra$")
    ax.set_ylabel(r"$\overline{Nu}$")
    ax.set_title("natural convection: mean Nusselt number")
    ax.legend(fontsize=7)
    _save(fig, out / "cavity_nu.png")
    ledger["figures"] = ["cavity_nu.png"]
    plt.close(fig)
    ledger["wall_seconds_total"] = time.time() - started
    return ledger


# --------------------------------------------------------------------------- outlet
def run_outlet_sensitivity(area_ratio: float = 4.0, reynolds: float = 100.0,
                           n: int = 40, out: Path | None = None,
                           max_steps: int = 120000, cfl: float = 0.5) -> dict:
    """Outlet-boundary sensitivity: Dirichlet / convective / domain extension."""
    started = time.time()
    ledger: dict[str, object] = {"closures": {}, "config": {
        "area_ratio": area_ratio, "reynolds": reynolds, "n": n,
        "max_steps": max_steps, "cfl": cfl}}
    closures = {
        "zero_gradient": "outlet",
        "dirichlet_developed": "developed",
        "convective_orlanski": "convective",
    }
    for name, right in closures.items():
        case = SuddenExpansionCase(SuddenExpansionConfig(
            area_ratio=area_ratio, reynolds=reynolds,
            cells_per_downstream_height=n, cfl=cfl, max_steps=max_steps,
            tolerance=2e-6, check_every=200, right=right))
        case.run(max_steps=max_steps, tolerance=2e-6)
        d = case.diagnostics()
        ledger["closures"][name] = {
            "cp_raw": d["cp_recovery_raw"],
            "cp_extrapolated": d["cp_recovery_extrapolated"],
            "k_raw": d["k_raw"],
            "steps": case.solver.steps,
        }
        print(f"[outlet {name}] Cp_raw={d['cp_recovery_raw']:.4f} "
              f"({case.solver.steps} steps)", flush=True)
    # domain extension: double the length
    case = SuddenExpansionCase(SuddenExpansionConfig(
        area_ratio=area_ratio, reynolds=reynolds,
        cells_per_downstream_height=n, cfl=cfl, max_steps=max_steps,
        tolerance=2e-6, check_every=200, right="outlet",
        length_in_downstream_heights=24.0))
    case.run(max_steps=max_steps, tolerance=2e-6)
    d = case.diagnostics()
    ledger["closures"]["domain_extension_24D"] = {
        "cp_raw": d["cp_recovery_raw"],
        "cp_extrapolated": d["cp_recovery_extrapolated"],
        "k_raw": d["k_raw"],
        "steps": case.solver.steps,
    }
    print(f"[outlet domain_extension] Cp_raw={d['cp_recovery_raw']:.4f} "
          f"({case.solver.steps} steps)", flush=True)

    # drift: max pairwise difference in Cp_raw
    cps = {k: v["cp_raw"] for k, v in ledger["closures"].items()}
    values = list(cps.values())
    drift = max(values) - min(values)
    ref = values[0]
    ledger["acceptance"] = {
        "cp_values": cps,
        "cp_drift": drift,
        "cp_drift_percent": 100.0 * drift / abs(ref) if ref else 0.0,
        "drift_within_5_percent": bool(
            100.0 * drift / abs(ref) < ACCEPTANCE["outlet_drift_percent"]),
    }
    ledger["wall_seconds_total"] = time.time() - started
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/internal-flow-maturation")
    parser.add_argument("--only", default="expansion,step,cavity,outlet")
    parser.add_argument("--expansion-ars", default="2,3,4")
    parser.add_argument("--expansion-meshes", default="20,40,80")
    parser.add_argument("--step-res", default="100,200,400")
    parser.add_argument("--step-meshes", default="10,20,40")
    parser.add_argument("--cavity-rayleighs", default="1000,10000,100000,1000000")
    parser.add_argument("--cavity-meshes", default="32,64;32,64;48,96;64,128")
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

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

    if "expansion" in wanted:
        ars = [float(v) for v in args.expansion_ars.split(",") if v]
        meshes = [int(v) for v in args.expansion_meshes.split(",") if v]
        summary["sudden_expansion"] = run_sudden_expansion(
            ars, meshes, out, max_steps=args.max_steps,
            tolerance=args.tolerance, cfl=args.cfl)
        ledger_path.write_text(json.dumps(summary, indent=1))
    if "step" in wanted:
        res = [float(v) for v in args.step_res.split(",") if v]
        meshes = [int(v) for v in args.step_meshes.split(",") if v]
        summary["backward_step"] = run_backward_step(
            res, meshes, out, max_steps=args.max_steps,
            tolerance=args.tolerance, cfl=args.cfl)
        ledger_path.write_text(json.dumps(summary, indent=1))
    if "cavity" in wanted:
        rayleighs = [float(v) for v in args.cavity_rayleighs.split(",") if v]
        groups = [g for g in args.cavity_meshes.split(";") if g.strip()]
        if len(groups) == 1:
            groups = groups * len(rayleighs)
        if len(groups) != len(rayleighs):
            parser.error("--cavity-meshes needs one list per Rayleigh number")
        cavity_meshes = {
            r: [int(v) for v in g.split(",") if v]
            for r, g in zip(rayleighs, groups)}
        summary["natural_convection"] = run_natural_convection(
            rayleighs, cavity_meshes, out, max_steps=args.max_steps,
            tolerance=args.tolerance, cfl=args.cfl)
        ledger_path.write_text(json.dumps(summary, indent=1))
    if "outlet" in wanted:
        summary["outlet_sensitivity"] = run_outlet_sensitivity(
            out=out, max_steps=args.max_steps, cfl=args.cfl)
        ledger_path.write_text(json.dumps(summary, indent=1))

    ledger_path.write_text(json.dumps(summary, indent=1))
    print(f"ledger written to {ledger_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
