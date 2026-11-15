"""Generate the figures for the JOUR-ICE-STRUCT-GUARD repair report (m22342).

Two figures, both programmatic (no hand-typed numbers):

1. ``fig_guard_gate.png`` — benchmark rows: measured error vs the declared
   tolerance on a log axis, plus the containment rates.  The claim under test is
   "an out-of-domain or non-finite input cannot reach the load arithmetic", so
   the horizontal axis is the *graded* quantity, not a physical field.
2. ``fig_ice_channel_physics.png`` — the integration caller still solves real
   physics: FVM / 1-D velocity profiles against the analytic fully-developed
   solution for the ice-under channel (top lid no-slip and slip), with the
   velocity error in the second panel.  This is the "field vs analytic
   reference" figure the guard itself cannot provide (a parameter validator has
   no field solution).

Run:
    PYTHONPATH=src python docs/ICE-STRUCT-GUARD-REPAIR/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfvm.ice.benchmark import NOMINAL_CASES, write_artifacts
from tensorfvm.ice_channel import (
    IceChannelConfig,
    analytic_u,
    solve_slip_1d,
    run_all,
)

HERE = Path(__file__).resolve().parent
ART = Path("docs/ICE-STRUCT-GUARD-REPAIR/artifacts")


def figure_gate(bundle: dict) -> Path:
    rows = bundle["rows"]
    graded = [r for r in rows if r["passed"] is not None]
    labels = [r["metric"].split()[0] for r in graded]
    errors = [max(r["error"], 1e-18) for r in graded]
    tols = [max(r["tolerance"], 1e-18) for r in graded]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    idx = range(len(graded))
    ax.bar([i - 0.2 for i in idx], errors, width=0.4, color="#2f6690",
           label="measured error")
    ax.bar([i + 0.2 for i in idx], tols, width=0.4, color="#c9d6df",
           label="declared tolerance")
    ax.set_yscale("log")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels)
    ax.set_ylabel("error (log)")
    ax.set_title("Benchmark rows: measured error vs tolerance", fontsize=10)
    ax.legend(fontsize=8)
    for i, (e, t) in enumerate(zip(errors, tols, strict=True)):
        ax.text(i - 0.2, e * 1.6, f"{e:.1e}", ha="center", fontsize=7)

    ax = axes[1]
    names = list(NOMINAL_CASES)
    vals = [bundle["detail"]["thickness_sweep"][i]["guarded_kN"] or 0.0
            for i in range(len(names))]
    ax.bar(names, vals, color="#7fb069")
    ax.set_ylabel("guarded resistance [kN]")
    ax.set_title("Guarded in-domain case points (pass-through, G5)", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=8)

    fig.suptitle("Ice guardrail gate — 12-parameter domain, "
                 "repro: PYTHONPATH=src python -m tensorfvm.ice", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "fig_guard_gate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_channel(tmp_dir: Path) -> Path:
    torch.set_num_threads(1)
    summary = run_all(output_dir=tmp_dir)

    cfg_slip = IceChannelConfig(top_bc="slip")
    slip = solve_slip_1d(cfg_slip, ny=32)
    y = slip["y"].double()
    u_num = slip["u"].double()
    u_exact = analytic_u(y, cfg_slip).double()

    noslip = summary["results"]["no_slip"][0]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    ax.plot(u_exact.numpy(), y.numpy(), "-", color="#1f3f70", lw=2,
            label="analytic slip profile")
    ax.plot(u_num.numpy(), y.numpy(), "o", ms=4, color="#b03a2e",
            label="1-D FV (guarded entry)")
    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("y [m]")
    ax.set_title("Ice-under channel: velocity profile\n"
                 f"guard status: {summary['ice_lid_guard']['slip']}", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    rel = (u_num - u_exact).abs() / u_exact.max()
    ax.semilogy(y.numpy(), rel.numpy() + 1e-18, "o-", ms=3, color="#2f6690",
                label="rel. error vs analytic")
    ax.axhline(0.03, color="#b03a2e", ls="--", label="3 % band")
    metrics = {
        "slip_u_l2": summary["results"]["slip"][-1]["velocity_l2"],
        "noslip_u_l2": noslip["velocity_l2"],
        "noslip_dpdx_err": noslip["dpdx_error"],
        "all_passed": summary["all_passed"],
    }
    ax.set_xlabel("y [m]")
    ax.set_ylabel("|u - u_exact| / max(u_exact)")
    ax.set_title("Pointwise velocity error, slip case", fontsize=10)
    ax.legend(fontsize=8)
    ax.text(0.03, 3e-4, json.dumps(metrics, indent=1), fontsize=7,
            family="monospace", va="top")

    (HERE / "channel_metrics.json").write_text(
        json.dumps({"summary_metrics": metrics,
                    "grid": {"ny_slip": 32, "noslip_grid": noslip["grid"]}},
                   indent=2) + "\n")

    fig.suptitle("Integration caller (tensorfvm.ice_channel) inside the guard",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "fig_ice_channel_physics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    write_artifacts(ART)
    bundle = json.loads((ART / "benchmark.json").read_text())
    p1 = figure_gate(bundle)
    p2 = figure_channel(ART / "channel_run")
    print(json.dumps({"figures": [str(p1), str(p2)]}, indent=2))


if __name__ == "__main__":
    main()