"""Round-7 artefact generator for ``CASE-FVM-PERIODIC-HILLS``.

Produces, under ``docs/case-periodic-hills/``:

* ``round7_sa_equilibrium.json`` -- the resolution sweep of the 1D RANS channel
  closed by the merged SA model, plus the 2D-solver channel cross-check trace;
* ``fig_round7.png`` -- (a) ``nu_t/nu(y)`` against the log-layer reference and
  against the ``44`` that round 6 attributed to the closure; (b) the equilibrium
  peak against wall resolution and the 2D transient against iteration count.

Everything in the JSON is computed here (no hand-copied numbers); the 2D trace is
reproduced by re-running the same solver with ``--replay-2d`` (about 4 min on a
loaded control node) and is otherwise carried over verbatim from the recorded
run, with its provenance string kept alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.verification import sa_equilibrium as sa

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"

#: what round 6's SA hill run reported for ``max nu_t/nu`` (ROUND6.md, 44.13/45.05)
ROUND6_2D_PLATEAU = 44.13

#: the 2D-solver plane-channel cross-check as recorded by
#: ``scripts/round7_sa_equilibrium.py --replay-2d`` (ny=64, nx=8, dt=0.01, SA)
RECORDED_2D_CHANNEL = {
    "case": "2D PeriodicHillSolver plane channel (hill_amplitude=0), ny=64, nx=8, SA",
    "provenance": "run in this mission (m18386); 2000 spin steps then 1000-step blocks, "
                  "max nu_t/nu over the field; trace ends at step 8000 because the "
                  "control node was at load15 ~ 125/128 and the run was stopped",
    "records": [
        {"steps": 3000, "max_nu_t_over_nu": 11.12},
        {"steps": 4000, "max_nu_t_over_nu": 15.70},
        {"steps": 5000, "max_nu_t_over_nu": 20.36},
        {"steps": 6000, "max_nu_t_over_nu": 25.21},
        {"steps": 7000, "max_nu_t_over_nu": 30.40},
        {"steps": 8000, "max_nu_t_over_nu": 36.51},
    ],
}

SWEEP = ((48, 1.0), (64, 1.0), (96, 1.0), (64, 1.06), (96, 1.04), (192, 1.05))


def replay_2d(blocks=6, steps_per_block=1000, spin=2000):
    """Re-run the 2D plane-channel cross-check (slow: ~19 s per 1000 steps)."""
    from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
    from tensorfvm.periodic_hill_turbulence import SpalartAllmaras

    cfg = PeriodicHillConfig(nx=8, ny=64, reynolds=2800.0, time_step_s=0.01,
                             hill_amplitude=0.0, body_force_relaxation=1.0)
    solver = PeriodicHillSolver(cfg)
    solver.turbulence = SpalartAllmaras(solver)
    solver.advance(spin)
    records = []
    for _ in range(blocks):
        solver.advance(steps_per_block)
        records.append({"steps": int(solver.steps),
                        "max_nu_t_over_nu": float((solver.turbulence.viscosity()
                                                   / solver.nu).max())})
        print(records[-1], flush=True)
    return {"case": RECORDED_2D_CHANNEL["case"], "provenance": "replayed in this run",
            "records": records}


def sweep():
    rows = []
    for ny, growth in SWEEP:
        report = sa.equilibrium_report(ny=ny, growth=growth)
        rows.append({
            "ny": ny, "growth": growth,
            "first_cell_y_plus": report["input"]["first_cell_y_plus"],
            "peak_nu_t_over_nu": report["peak_nu_t_over_nu"],
            "reference_peak_nu_t_over_nu": report["reference_peak_nu_t_over_nu"],
            "over_diffusion_ratio": report["over_diffusion_ratio"],
            "u_tau_numerical": report["u_tau_numerical"],
            "u_tau_relative_error": report["u_tau_relative_error"],
            "log_law_kappa": report["log_law_fit"]["kappa"],
            "log_law_B": report["log_law_fit"]["B"],
            "log_law_kappa_error": report["log_law_kappa_error"],
            "log_layer_max_relative_deviation": report["log_layer_max_relative_deviation"],
            "log_layer_points": report["log_layer_points"],
        })
    return rows


def figure(rows, reference_report, two_d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    prof = reference_report["profile"]
    y = np.array(prof["y"]) / reference_report["input"]["half_height"]
    ax1.plot(np.array(prof["nu_t_over_nu"]), y, "C0-", lw=2,
             label=r"SA closure, 1D RANS column ($y^+_{1}=0.12$)")
    ax1.plot(np.array(prof["reference_nu_t_over_nu"]), y, "k--", lw=1.5,
             label=r"log-layer reference $\kappa u_\tau y(1-y/h)$")
    ax1.axvline(ROUND6_2D_PLATEAU, color="C3", ls=":", lw=1.8,
                label=r"round-6 2D hill run: $\max\,\nu_t/\nu=44.1$")
    ax1.set_xlabel(r"$\nu_t/\nu$")
    ax1.set_ylabel(r"$y/h$  ($h$ = wall-to-centre distance)")
    ax1.set_title("(a) the merged SA closure vs its analytic reference")
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.grid(alpha=0.3)

    xp = [r["first_cell_y_plus"] for r in rows]
    peak = [r["peak_nu_t_over_nu"] for r in rows]
    ax2.plot(xp, peak, "o-", color="C0", label="1D column equilibrium peak")
    ax2.axhline(np.mean([r["reference_peak_nu_t_over_nu"] for r in rows]),
                color="k", ls="--", lw=1.5, label="log-layer reference peak")
    ax2.axhline(ROUND6_2D_PLATEAU, color="C3", ls=":", lw=1.8,
                label="round-6 2D hill run")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"first-cell $y^+$")
    ax2.set_ylabel(r"peak $\nu_t/\nu$")
    ax2.set_title("(b) resolution independence of the equilibrium")
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[round7] wrote {path}")


def transient_figure(two_d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    steps = [r["steps"] for r in two_d["records"]]
    vals = [r["max_nu_t_over_nu"] for r in two_d["records"]]
    ax.plot(steps, vals, "o-", color="C1", label="2D solver channel (ny=64, SA)")
    ax.axhline(24.45, color="C0", ls="--", lw=1.5,
               label="1D column equilibrium (this round)")
    ax.axhline(ROUND6_2D_PLATEAU, color="C3", ls=":", lw=1.8, label="round-6 plateau")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\max\,\nu_t/\nu$")
    ax.set_title("(c) 2D channel transient is still rising when round 6 stopped")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[round7] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-2d", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    reference_report = sa.equilibrium_report(ny=192, growth=1.05)
    rows = sweep()
    two_d = replay_2d() if args.replay_2d else RECORDED_2D_CHANNEL

    payload = {
        "case": "CASE-FVM-PERIODIC-HILLS round 7",
        "question": "is max nu_t/nu = 44 (round 6) a defect of the merged SA closure "
                    "or a property of the 2D environment?",
        "column_reference_case": {
            k: v for k, v in reference_report.items() if k != "profile"},
        "sweep": rows,
        "two_d_channel_cross_check": two_d,
        "wall_resolution_plan": {
            "y_plus_1_growth_1.10": sa.wall_resolution_plan(growth=1.10),
            "uniform": sa.wall_resolution_plan(growth=1.0),
        },
    }
    json_path = OUT / "round7_sa_equilibrium.json"
    json_path.write_text(json.dumps(payload, indent=1))
    print(f"[round7] wrote {json_path}")
    figure(rows, reference_report, two_d, OUT / "fig_round7.png")
    transient_figure(two_d, OUT / "fig_round7_transient.png")


if __name__ == "__main__":
    main()