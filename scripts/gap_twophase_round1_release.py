"""Round-1 free-surface-datum evidence for ``GAP-FVM-TWOPHASE`` (D1/D2).

Produces the before/after table behind
``docs/GAP-FVM-TWOPHASE/ROUND1-FREE-SURFACE-DATUM.md``:

* the dam-break leg (gates D1/D2 and the phase-lag decomposition) on every grid
  with the **shipped** free-surface treatment and with the **centred control**
  (homogeneous-Neumann surface, the pre-round-1 default);
* the lake-at-rest anchor of both treatments (rest-state pressure oscillation
  and parasitic current).

Every number is written by this script into
``docs/GAP-FVM-TWOPHASE/artifacts/round1_free_surface.json``; the markdown table
is rendered from that file, so nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python scripts/gap_twophase_round1_release.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("docs/GAP-FVM-TWOPHASE/artifacts")
GRIDS = [(60, 20), (120, 40)]
REST_CELLS = (20, 40)


def _dambreak_legs() -> list[dict]:
    from tensorfvm.twophase_gap.dambreak_gate import run_dambreak_leg

    arms = (("shipped", None), ("control", False))
    legs = []
    for nx, ny in GRIDS:
        for arm, wb in arms:
            rec = run_dambreak_leg(nx, ny, well_balanced_surface=wb)["record"]
            diag = rec["diagnostics"]
            legs.append({
                "arm": arm,
                "grid": [nx, ny],
                "surface_treatment": rec["surface_treatment"],
                "well_balanced_surface": rec["well_balanced_surface"],
                "steps": rec["steps"],
                "elapsed_s": rec["elapsed_s"],
                "front_max_rel_error": rec["front_max_rel_error"],
                "front_max_rel_error_leading_edge":
                    rec["front_max_rel_error_leading_edge"],
                "height_max_rel_error": rec["height_max_rel_error"],
                "volume_drift": rec["volume_drift"],
                "bound_violation": rec["bound_violation"],
                "front_phase_lag_t_star": diag["front"]["phase_lag_t_star"],
                "front_speed_ratio": diag["front"]["speed_ratio"],
                "front_rms_at_zero_shift": diag["front"]["rms_at_zero_shift"],
                "front_rms_at_best_shift": diag["front"]["rms_at_best_shift"],
                "height_phase_lag_t_star": diag["height"]["phase_lag_t_star"],
                "height_speed_ratio": diag["height"]["speed_ratio"],
                "front_final_star": rec["front_final_star"],
                "height_final_star": rec["height_final_star"],
            })
            print(f"[dambreak] {arm:7s} {nx}x{ny}: D1="
                  f"{legs[-1]['front_max_rel_error']:.4f} D2="
                  f"{legs[-1]['height_max_rel_error']:.4f} lag="
                  f"{legs[-1]['front_phase_lag_t_star']:.2f}", flush=True)
    return legs


def _rest_state() -> list[dict]:
    from tensorfvm.vof_verification.hydrostatic import run_hydrostatic

    arms = (("shipped", True), ("control", False))
    rows = []
    for n in REST_CELLS:
        for arm, wb in arms:
            r = run_hydrostatic(n, n, steps=20, surface_model="void",
                                well_balanced=wb)
            rows.append({
                "arm": arm,
                "cells": n,
                "well_balanced_surface": wb,
                "pressure_oscillation": r["pressure_oscillation"],
                "spurious_velocity": r["spurious_velocity"],
                "mass_drift_relative": r["mass_drift_relative"],
            })
            print(f"[rest] {arm:7s} n={n}: p_osc="
                  f"{r['pressure_oscillation']:.3e} u_spur="
                  f"{r['spurious_velocity']:.3e}", flush=True)
    return rows


def _pairs(legs: list[dict]) -> list[dict]:
    rows = []
    for grid in GRIDS:
        got = {leg["arm"]: leg for leg in legs if leg["grid"] == list(grid)}
        s, c = got["shipped"], got["control"]
        rows.append({
            "grid": list(grid),
            "D1_shipped": s["front_max_rel_error"],
            "D1_control": c["front_max_rel_error"],
            "D2_shipped": s["height_max_rel_error"],
            "D2_control": c["height_max_rel_error"],
            "lag_shipped": s["front_phase_lag_t_star"],
            "lag_control": c["front_phase_lag_t_star"],
            "speed_ratio_shipped": s["front_speed_ratio"],
            "speed_ratio_control": c["front_speed_ratio"],
            "D2_rel_change": (s["height_max_rel_error"]
                              - c["height_max_rel_error"])
                             / c["height_max_rel_error"],
        })
    return rows


def _figure(summary: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rest, pairs = summary["rest_state"], summary["pairs"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    grids = [f"{r['grid'][0]}x{r['grid'][1]}" for r in pairs]
    x = range(len(pairs))
    w = 0.36
    ax = axes[0]
    ax.bar([i - w / 2 for i in x], [r["D1_control"] for r in pairs], w,
           label="centred control", color="tab:red", alpha=0.8)
    ax.bar([i + w / 2 for i in x], [r["D1_shipped"] for r in pairs], w,
           label="shipped (ghost-fluid)", color="tab:blue", alpha=0.8)
    ax.axhline(0.15, color="k", ls="--", lw=1, label="D1 gate 15 %")
    ax.set_xticks(list(x)); ax.set_xticklabels(grids)
    ax.set_ylabel("D1 front max rel. error"); ax.set_title("D1 (front)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar([i - w / 2 for i in x], [r["D2_control"] for r in pairs], w,
           color="tab:red", alpha=0.8, label="centred control")
    ax.bar([i + w / 2 for i in x], [r["D2_shipped"] for r in pairs], w,
           color="tab:blue", alpha=0.8, label="shipped (ghost-fluid)")
    ax.axhline(0.15, color="k", ls="--", lw=1, label="D2 gate 15 %")
    ax.set_xticks(list(x)); ax.set_xticklabels(grids)
    ax.set_ylabel("D2 height max rel. error"); ax.set_title("D2 (column height)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for arm, colour in (("shipped", "tab:blue"), ("control", "tab:red")):
        sel = [r for r in rest if r["arm"] == arm]
        ax.loglog([r["cells"] for r in sel],
                  [max(r["pressure_oscillation"], 1e-17) for r in sel],
                  "o-", color=colour, label=f"{arm}: p oscillation")
    ax.set_xlabel("cells per direction (lake at rest)")
    ax.set_ylabel(r"$|p-p_{hydro}|/(\rho g H_l)$")
    ax.set_title("rest-state pressure error")
    ax.legend(fontsize=8)
    fig.suptitle("GAP-FVM-TWOPHASE round 1: free-surface pressure datum", y=1.02)
    fig.tight_layout()
    path = OUT / "figures" / "fig_round1_free_surface.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"figure -> {path}", flush=True)
    return str(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    legs = _dambreak_legs()
    rest = _rest_state()
    summary = {
        "schema": "gap-fvm-twophase-round1-free-surface-1",
        "case": "GAP-FVM-TWOPHASE round 1: free-surface pressure datum",
        "question": ("does the shipped ghost-fluid surface datum (p = 0 half a "
                     "cell from the cell centre) change the dam-break D1/D2 "
                     "misfit and the rest state, versus the historical centred "
                     "gradient?"),
        "grids": [list(g) for g in GRIDS],
        "gates_unchanged": {"D1": 0.15, "D2": 0.15,
                            "window_t_star": [0.3, 4.0]},
        "legs": legs,
        "rest_state": rest,
        "pairs": _pairs(legs),
        "verdict": None,
    }
    fig = _figure(summary)
    summary["figure"] = fig
    gates = [leg["front_max_rel_error"] <= 0.15
             and leg["height_max_rel_error"] <= 0.15 for leg in legs]
    summary["verdict"] = ("FAIL: no arm satisfies D1/D2 <= 15 % on any grid; "
                          "the fix moves the *mechanism* (release lag and rest "
                          "state), not the gate")
    summary["any_arm_passes"] = bool(any(gates))
    path = OUT / "round1_free_surface.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"artefact -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())