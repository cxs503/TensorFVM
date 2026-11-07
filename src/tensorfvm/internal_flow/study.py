"""Driver that produces the published artifacts of the internal-flow verification.

Each case is run on a sequence of meshes with **grid sequencing**: the steady
state of the next coarser mesh initialises the finer one
(:meth:`StaggeredMAC.prolong_from`), which removes the long start-up transient
without touching the converged solution.  Every number that ends up in the JSON
ledger and in the report is written here directly from the computed fields --
nothing is copied by hand.

Usage
-----
``python -m tensorfvm.internal_flow.study --out artifacts/internal-flow``
runs the full study; ``--only bfs,expansion,cavity`` and ``--bfs-meshes 10,20,40``
restrict it (used to keep the fine meshes in a separate background job).
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

#: Acceptance bands of the campaign node (see the report; the literature band is
#: used where the campaign quotes a literature value).
ACCEPTANCE = {
    "bfs_reattachment_in_downstream_heights": 0.2 / 6.0,   # +-0.2 about 6.0
    "expansion_loss_coefficient": 0.05,
    "cavity_nusselt": 0.03,
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
    """Write the ledger after every mesh so a long study can be interrupted."""
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

# --------------------------------------------------------------------------- BFS
def run_backward_step(meshes: list[int], out: Path, max_steps: int = 60000,
                      tolerance: float = 2e-6, cfl: float = 0.5,
                      reynolds: float = 800.0) -> dict:
    plt = _matplotlib()
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
        # The reattachment length approaches its asymptote algebraically, so the
        # velocity residual is not a usable stopping rule at these Reynolds
        # numbers: march in blocks and stop once L_r stops moving.
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

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for n in meshes:
        data = ledger["meshes"][str(n)]
        ax.plot(data["x"], data["cf"], lw=1.0, label=f"H/{n}")
    ax.axhline(analytic, color="k", ls="--", lw=1.0, label=r"$12/Re_H$")
    ax.axhline(0.0, color="0.6", lw=0.6)
    for n in meshes:
        xr = ledger["meshes"][str(n)]["reattachment_in_step_heights"]
        if xr is not None:
            ax.plot([xr], [0.0], "o", ms=4)
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$C_f$ (bottom wall)")
    ax.set_title("Backward-facing step, $Re_H=800$: wall friction")
    ax.set_xlim(0, min(12.0, ledger["meshes"][str(finest)]["x"][-1]))
    ax.legend(fontsize=7)
    ledger["figures"] = ["bfs_skin_friction.png"]
    _save(fig, out / "bfs_skin_friction.png")
    plt.close(fig)

    case = cases[finest]
    solver = case.solver
    import torch
    u, v = solver.cell_velocity()
    x, y = solver.cell_coordinates()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    speed = torch.sqrt(u ** 2 + v ** 2).numpy()
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    im = axes[0].pcolormesh(grid_x.numpy(), grid_y.numpy(), speed, shading="auto",
                            cmap="viridis")
    axes[0].streamplot(x.numpy(), y.numpy(), u.numpy().T, v.numpy().T,
                       color="w", linewidth=0.4, density=1.0)
    axes[0].set_title(f"|u| and streamlines (H/{finest})", fontsize=8)
    axes[0].set_xlim(0, min(14.0, float(x[-1])))
    axes[0].set_xlabel("$x/H$")
    axes[0].set_ylabel("$y/H$")
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    h = [1.0 / n for n in meshes]
    axes[1].plot(h, xr_values, "o-")
    axes[1].axhline(target, color="C3", ls="--", lw=1.0, label="target 6.0")
    axes[1].axhline(BACKWARD_STEP_REFERENCE["value"], color="k", ls=":", lw=1.0,
                    label="Gartling 6.1")
    axes[1].set_xlabel("$h/H$")
    axes[1].set_ylabel("$L_r/H_{out}$")
    axes[1].set_title("mesh convergence", fontsize=8)
    axes[1].legend(fontsize=6)
    _save(fig, out / "bfs_field.png")
    ledger["figures"].append("bfs_field.png")
    plt.close(fig)

    ledger["wall_seconds_total"] = time.time() - started
    return ledger


# --------------------------------------------------------------------- expansion
def run_sudden_expansion(meshes: list[int], out: Path, max_steps: int = 120000,
                         tolerance: float = 2e-6, cfl: float = 0.5) -> dict:
    plt = _matplotlib()
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
        # the reattachment length and the loss coefficient are the slow quantities,
        # so march in blocks and stop once L_r stops moving
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for n in meshes:
        data = ledger["meshes"][str(n)]
        axes[0].plot(data["x"], data["total"], lw=1.0, label=f"D/{n}")
    window = last["extrapolation_window"]
    axes[0].axvspan(window[0], window[1], color="0.85", zorder=0)
    axes[0].axhline(last["total"][0], color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("$x/D$")
    axes[0].set_ylabel(r"$\langle p_t \rangle$")
    axes[0].set_title("total pressure (shaded: fit window)", fontsize=8)
    axes[0].legend(fontsize=6)
    labels = ["Borda-Carnot", "corrected", "fitted", "outlet"]
    values = [theory, last["k_corrected"], last["k_extrapolated"], last["k_raw"]]
    axes[1].bar(labels, values, color=["0.6", "C0", "C1"])
    axes[1].axhline(theory, color="0.3", ls="--", lw=0.8)
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.01, f"{value:.3f}", ha="center", fontsize=7)
    axes[1].set_ylabel("$K = \\Delta p_t / (\\frac{1}{2}\\rho U_{in}^2)$")
    axes[1].set_title(f"loss coefficient, Re$_D$={last['reynolds']:g}", fontsize=8)
    _save(fig, out / "expansion_loss.png")
    ledger["figures"] = ["expansion_loss.png"]
    plt.close(fig)

    case = cases[finest]
    solver = case.solver
    import torch
    u, v = solver.cell_velocity()
    x, y = solver.cell_coordinates()
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    im = ax.pcolormesh(grid_x.numpy(), grid_y.numpy(),
                       torch.sqrt(u ** 2 + v ** 2).numpy(), shading="auto", cmap="magma")
    ax.streamplot(x.numpy(), y.numpy(), u.numpy().T, v.numpy().T, color="w",
                  linewidth=0.4, density=1.2)
    ax.set_xlim(0, min(4.0, float(x[-1])))
    ax.set_xlabel("$x/D$")
    ax.set_ylabel("$y/D$")
    ax.set_title("sudden expansion $AR=4$: speed and streamlines", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035)
    _save(fig, out / "expansion_field.png")
    ledger["figures"].append("expansion_field.png")
    plt.close(fig)

    ledger["wall_seconds_total"] = time.time() - started
    return ledger


# --------------------------------------------------------------- natural convection
def run_natural_convection(meshes: dict[float, list[int]], out: Path,
                           max_steps: int = 200000, tolerance: float = 2e-6,
                           cfl: float = 0.5) -> dict:
    plt = _matplotlib()
    started = time.time()
    ledger: dict[str, object] = {"cases": {}, "config": {
        "meshes": {str(k): v for k, v in meshes.items()}, "max_steps": max_steps,
        "tolerance": tolerance, "cfl": cfl}}
    fields: dict[float, tuple] = {}
    for rayleigh, n_list in meshes.items():
        previous = None
        per_mesh = {}
        for n in n_list:
            case = NaturalConvectionCase(NaturalConvectionConfig(
                n=n, rayleigh=rayleigh, cfl=cfl, max_steps=max_steps,
                tolerance=tolerance, check_every=200))
            if previous is not None:
                case.solver.prolong_from(previous.solver)
            wall = time.time()
            convergence = case.run()
            diagnostics = case.diagnostics()
            diagnostics["wall_seconds"] = time.time() - wall
            diagnostics["converged"] = convergence["converged"]
            per_mesh[str(n)] = diagnostics
            previous = case
            _dump_partial(out, "natural_convection",
                          {"cases": dict(ledger["cases"]), "config": ledger["config"]})
            fields[rayleigh] = (case, n)
            print(f"[cavity] Ra={rayleigh:g} n={n} steps={convergence['steps']} "
                  f"res={convergence['residual']:.2e} Nu={diagnostics['nu_mean']:.4f} "
                  f"({diagnostics['wall_seconds']:.0f}s)", flush=True)
        ledger["cases"][str(rayleigh)] = per_mesh

    acceptance = {}
    for rayleigh in meshes:
        finest = str(meshes[rayleigh][-1])
        value = ledger["cases"][str(rayleigh)][finest]["nu_mean"]
        reference = DE_VAHL_DAVIS_NU[rayleigh]
        acceptance[str(rayleigh)] = {
            "nu_computed": value,
            "nu_reference": reference,
            "relative_error": abs(value - reference) / reference,
            "within_3_percent": bool(abs(value - reference) / reference < ACCEPTANCE["cavity_nusselt"]),
            "mesh": int(finest),
        }
    ledger["acceptance"] = acceptance

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colours = {1.0e4: "C0", 1.0e5: "C1", 1.0e6: "C2"}
    for rayleigh in meshes:
        case, n = fields[rayleigh]
        data = ledger["cases"][str(rayleigh)][str(n)]
        axes[0].plot(data["nu_local"], data["y"], lw=1.1, color=colours[rayleigh],
                     label=f"Ra={rayleigh:g} (n={n})")
        axes[0].plot([DE_VAHL_DAVIS_NU[rayleigh]] * 2, [0.0, 1.0], ls=":",
                     color=colours[rayleigh], lw=0.8)
    axes[0].set_xlabel(r"$Nu(y)$ on the hot wall")
    axes[0].set_ylabel("$y/H$")
    axes[0].set_title("local Nusselt number (dotted: mean)", fontsize=8)
    axes[0].legend(fontsize=6)
    rayleighs = sorted(meshes)
    computed = [acceptance[str(r)]["nu_computed"] for r in rayleighs]
    reference = [acceptance[str(r)]["nu_reference"] for r in rayleighs]
    axes[1].loglog(rayleighs, reference, "ks--", ms=4, label="de Vahl Davis (1983)")
    axes[1].loglog(rayleighs, computed, "o-", label="this solver")
    for r, value in zip(rayleighs, computed):
        axes[1].annotate(f"{value:.3f}", (r, value), textcoords="offset points",
                         xytext=(3, -9), fontsize=6)
    axes[1].set_xlabel("$Ra$")
    axes[1].set_ylabel(r"$\overline{Nu}$")
    axes[1].set_title("mean Nusselt number", fontsize=8)
    axes[1].legend(fontsize=6)
    _save(fig, out / "cavity_nusselt.png")
    ledger["figures"] = ["cavity_nusselt.png"]
    plt.close(fig)

    import torch
    rayleigh = max(meshes)
    case, n = fields[rayleigh]
    solver = case.solver
    x, y = solver.cell_coordinates()
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    cs = axes[0].contourf(grid_x.numpy(), grid_y.numpy(), solver.theta.numpy(), 20,
                          cmap="RdBu_r")
    axes[0].set_title(f"temperature, Ra={rayleigh:g}", fontsize=8)
    fig.colorbar(cs, ax=axes[0], fraction=0.04)
    psi = solver.streamfunction()
    # streamfunction lives on the nodes; average to the cell centres for the plot
    psi_cell = 0.25 * (psi[:-1, :-1] + psi[1:, :-1] + psi[:-1, 1:] + psi[1:, 1:])
    axes[1].contour(grid_x.numpy(), grid_y.numpy(), psi_cell.numpy(), 18,
                    colors="k", linewidths=0.5)
    axes[1].set_title(r"streamlines $\psi$", fontsize=8)
    for axis in axes:
        axis.set_xlabel("$x/H$")
        axis.set_ylabel("$y/H$")
    _save(fig, out / "cavity_field.png")
    ledger["figures"].append("cavity_field.png")
    plt.close(fig)

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())