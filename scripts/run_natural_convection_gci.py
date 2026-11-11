#!/usr/bin/env python3
"""Driver for the buoyant-cavity ``Nu`` grid-convergence study (Ra = 1e5, 1e6).

Usage::

    python scripts/run_natural_convection_gci.py run --rayleigh 1e5 --cells 64
    python scripts/run_natural_convection_gci.py run --rayleigh 1e6 --cells 128
    python scripts/run_natural_convection_gci.py summary
    python scripts/run_natural_convection_gci.py figures

Every artefact lands in ``docs/natural-convection-gci`` (``artifacts/`` for the
raw runs, ``summary.json`` for the aggregated verdicts and ``figures/`` for the
PNG set).  The report is rendered from ``summary.json`` only, so no number in it
is copied by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.verification.natural_convection_gci import (  # noqa: E402
    BENCHMARK_NU,
    RunRequest,
    run_case,
    summarise,
)

ROOT = Path(__file__).resolve().parents[1] / "docs" / "natural-convection-gci"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"


def _progress(flow, step: int, residual: float) -> None:
    if step % 2000 == 0:
        print(f"    step {step:7d}  residual {residual:.3e}", flush=True)


def command_run(args: argparse.Namespace) -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    request = RunRequest(rayleigh=args.rayleigh, cells=args.cells, cfl=args.cfl,
                         tolerance=args.tolerance, max_steps=args.max_steps)
    started = time.time()
    print(f"[run] {request.tag} cfl={request.cfl} tol={request.tolerance} "
          f"max_steps={request.max_steps}", flush=True)
    try:
        payload = run_case(request, on_check=_progress)
    except Exception:                                    # pragma: no cover
        (ARTIFACTS / f"failure_{request.tag}.log").write_text(traceback.format_exc())
        raise
    payload["wall_seconds"] = time.time() - started
    fields = payload.pop("fields")
    np.savez_compressed(ARTIFACTS / f"fields_{request.tag}.npz", **fields)
    path = ARTIFACTS / f"raw_{request.tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    metrics = payload["metrics"]
    print(f"[done] {request.tag} Nu(half)={metrics['nusselt_hot_half_cell']:.4f} "
          f"Nu(quad)={metrics['nusselt_hot_quadratic']:.4f} "
          f"converged={metrics['converged']} steps={metrics['steps']} "
          f"wall={payload['wall_seconds']:.0f}s -> {path}", flush=True)
    return 0


def _records() -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted(ARTIFACTS.glob("raw_*.json"))]


def command_summary(_args: argparse.Namespace) -> int:
    records = _records()
    if not records:
        raise SystemExit("no raw_*.json found")
    summary = summarise(records)
    summary["raw_files"] = sorted(p.name for p in ARTIFACTS.glob("raw_*.json"))
    summary["records"] = [{"tag": rec["tag"], **rec["metrics"]} for rec in records]
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({f: {"verdict": b.get("verdict"),
                          "order": b["nusselt_hot_half_cell"]["refinement"].get(
                              "observed_order"),
                          "conv": b["nusselt_hot_half_cell"]["refinement"].get(
                              "richardson", {}).get("convergence")}
                      for f, b in summary["families"].items()}, indent=2))
    return 0


def command_figures(_args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "summary.json").read_text())
    records = summary["records"]

    # --- Figure 1: Nu vs grid with both estimators and the benchmark line
    fig, axes = plt.subplots(1, len(summary["families"]), figsize=(11, 4.2),
                            squeeze=False)
    for ax, (family, block) in zip(axes[0], summary["families"].items()):
        ra = float(family[2:])
        grids = [int(k) for k in block["nusselt_hot_half_cell"]["values"]]
        nu_half = [block["nusselt_hot_half_cell"]["values"][str(g)] for g in grids]
        nu_quad = [block["nusselt_hot_quadratic"]["values"][str(g)] for g in grids]
        ax.plot(grids, nu_half, "o-", label="Nu, half-cell flux (1st order)")
        ax.plot(grids, nu_quad, "s--", label="Nu, quadratic flux (2nd order)")
        ax.axhline(BENCHMARK_NU[ra], color="k", lw=1.2,
                   label=f"de Vahl Davis {BENCHMARK_NU[ra]}")
        ax.axhspan(BENCHMARK_NU[ra] * 0.97, BENCHMARK_NU[ra] * 1.03,
                   color="k", alpha=0.08, label="±3 % gate")
        extrapolated = block.get("verdict", {}).get("extrapolated")
        if extrapolated is not None:
            ax.axhline(extrapolated, color="C3", ls=":", lw=1.2,
                       label=f"Richardson limit {extrapolated:.4f}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(grids)
        ax.set_xticklabels([str(g) for g in grids])
        ax.set_xlabel("cells per side N")
        ax.set_ylabel("mean Nu (hot wall)")
        ax.set_title(f"{family}  (Pr = 0.71)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_nusselt_vs_grid.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: relative error vs h on log-log (order slope visible)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    markers = {"Ra1e+05": "o", "Ra1e+06": "s"}
    for family, block in summary["families"].items():
        ra = float(family[2:])
        errors = block["nusselt_hot_half_cell"]["errors_vs_reference"]
        grids = sorted(int(k) for k in errors)
        h = [1.0 / g for g in grids]
        err = [errors[str(g)] for g in grids]
        ax.loglog(h, err, markers.get(family, "^") + "-", label=f"{family} half-cell")
        order = block["nusselt_hot_half_cell"]["refinement"].get("observed_order")
        if order:
            scale = err[-1] / h[-1] ** order
            ax.loglog(h, [scale * x ** order for x in h], ":", color="grey", lw=1,
                      label=f"{family} slope p={order:.2f}")
    ax.axhline(0.03, color="k", lw=1.2, label="3 % gate")
    ax.set_xlabel("h = 1/N")
    ax.set_ylabel("|Nu - Nu_ref| / Nu_ref")
    ax.set_title("Nusselt error against grid spacing")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_nusselt_error_order.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: steady-state residual histories (convergence evidence)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for rec in records:
        history = rec.get("residual_history") or []
        if not history:
            continue
        steps = [row[0] for row in history]
        residual = [row[2] for row in history]
        ax.semilogy(steps, residual, label=rec["tag"])
    ax.set_xlabel("time step")
    ax.set_ylabel("max |du|/dt")
    ax.set_title("Steady-state residual history")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_residual_history.png", dpi=150)
    plt.close(fig)

    print("wrote", *(str(p) for p in sorted(FIGURES.glob("*.png"))), sep="\n  ")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--rayleigh", type=float, required=True)
    run.add_argument("--cells", type=int, required=True)
    run.add_argument("--cfl", type=float, default=0.25)
    run.add_argument("--tolerance", type=float, default=1e-7)
    run.add_argument("--max-steps", type=int, default=400_000)
    run.set_defaults(func=command_run)
    sub.add_parser("summary").set_defaults(func=command_summary)
    sub.add_parser("figures").set_defaults(func=command_figures)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())