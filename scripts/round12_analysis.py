#!/usr/bin/env python3
"""Round 12 analysis: score the *fixed* SST transport against round 11 and Breuer 2009.

Round 11 read ``x_r/H = 2.05`` for SST at 96x48 (vs 5.34/5.41 from the primary
ERCOFTAC UFR 3-30 data) and called it a model-class verdict.  Round 12 found two
discretisation defects in the *transport* of k/omega that both inflate the eddy
viscosity (a spurious second ``1/sigma`` on the diffusion coefficient, and a
documented-but-never-imposed ``omega`` wall value); this script re-scores the
time-averaged run of the fixed closure on the identical protocol and writes the
machine-readable comparison plus the paper figures.

Inputs (``docs/case-periodic-hills/``):
    round11_sst_r12-96x48.json   -- fixed closure (this round)
    round11_sst_96x48.json       -- round 11, defective transport (baseline)
Outputs:
    round12_comparison.json
    fig_round12_profile_y05.png
    fig_round12_series.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
NEW = DOCS / "round11_sst_r12-96x48.json"
OLD = DOCS / "round11_sst_96x48.json"


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def profile_errors(new, old):
    rows = []
    ref = new["horizontal_profile_y_H_0.5"]["reference"]
    for i, x in enumerate(new["horizontal_profile_y_H_0.5"]["x_over_H"]):
        r = ref.get(str(int(x)) if float(x).is_integer() else str(x))
        if r is None:
            continue
        u_new = new["horizontal_profile_y_H_0.5"]["u_over_Ub"][i]
        u_old = (old["horizontal_profile_y_H_0.5"]["u_over_Ub"][i]
                 if old is not None else None)
        rows.append({
            "x_over_H": float(x),
            "reference": float(r),
            "round12": float(u_new),
            "round11": None if u_old is None else float(u_old),
            "abs_error_round12": abs(float(u_new) - float(r)),
            "rel_error_round12": abs(float(u_new) - float(r)) / max(abs(float(r)), 1e-12),
            "rel_error_round11": (None if u_old is None
                                  else abs(float(u_old) - float(r)) / max(abs(float(r)), 1e-12)),
        })
    return rows


def main(argv=None):
    new = _load(NEW)
    if new is None:
        print(f"missing {NEW}", file=sys.stderr)
        return 2
    old = _load(OLD)
    gate_new = new["gate"]
    gate_old = old["gate"] if old is not None else None
    rows = profile_errors(new, old)
    summary = {
        "case": "CASE-FVM-PERIODIC-HILLS round 12: SST transport defects fixed",
        "run": {"grid": "96x48", "steps": new["config"]["steps"],
                "dt": new["config"]["time_step_s"],
                "end_time": new["config"]["end_time"],
                "burn_in": new["config"]["burn_in_steps"],
                "samples": new["samples"],
                "wall_clock_s": new["wall_clock_s"],
                "closure": new["closure"]},
        "reference": {"x_r_over_H": 5.34, "tolerance": 0.10,
                      "source": "Breuer et al. 2009 / primary ERCOFTAC UFR 3-30 data"},
        "round12": {"x_r_over_H": gate_new["x_r_over_H"],
                    "x_r_over_H_error": gate_new["x_r_over_H_error"],
                    "profile_stations_pass": gate_new["profile_stations_pass"],
                    "profile_stations": gate_new["profile_stations"],
                    "stationary": gate_new["stationary"],
                    "max_nu_t_ratio": gate_new.get("max_nu_t_ratio"),
                    "pass": gate_new["pass"],
                    "reasons": gate_new["reasons"]},
        "round11": (None if gate_old is None else
                    {"x_r_over_H": gate_old["x_r_over_H"],
                     "x_r_over_H_error": gate_old["x_r_over_H_error"],
                     "profile_stations_pass": gate_old["profile_stations_pass"],
                     "max_nu_t_ratio": gate_old.get("max_nu_t_ratio"),
                     "pass": gate_old["pass"]}),
        "profile": rows,
        "nu_t_series": {
            "t": new["series"]["t"],
            "max_nu_t_ratio": new["series"]["max_nu_t_ratio"],
            "x_r_over_H": new["series"]["x_r_over_H"],
        },
    }
    (DOCS / "round12_comparison.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False))

    # ---- figure 1: y/H = 0.5 profile, round 12 vs round 11 vs reference
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = [r["x_over_H"] for r in rows]
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.plot(x, [r["reference"] for r in rows], "o-", color="k", lw=1.2,
            ms=4, label="Breuer 2009 (UFR 3-30, Re=2800)")
    ax.plot(x, [r["round12"] for r in rows], "s-", color="C0", lw=1.4,
            ms=4, label="round 12 (fixed SST transport)")
    if all(r["round11"] is not None for r in rows):
        ax.plot(x, [r["round11"] for r in rows], "^--", color="C3", lw=1.2,
                ms=4, label="round 11 (defective transport)")
    ax.set_xlabel("$x/H$")
    ax.set_ylabel("$u/U_b$ at $y/H=0.5$")
    ax.set_title("Periodic hills Re=2800, 96x48, time-averaged")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_round12_profile_y05.png", dpi=160)
    plt.close(fig)

    # ---- figure 2: eddy-viscosity and reattachment time series
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    a1.plot(new["series"]["t"], new["series"]["max_nu_t_ratio"], color="C0", lw=1.0)
    a1.set_xlabel("$t\\,U_b/h$")
    a1.set_ylabel("$\\max\\,\\nu_t/\\nu$")
    a1.set_title("eddy-viscosity level (fixed SST)")
    a1.grid(alpha=0.3)
    a2.plot(new["series"]["t"], new["series"]["x_r_over_H"], color="C2", lw=1.0)
    a2.axhline(5.34, color="k", ls="--", lw=1.0, label="reference 5.34")
    a2.set_xlabel("$t\\,U_b/h$")
    a2.set_ylabel("$x_r/H$")
    a2.set_title("reattachment length vs time")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_round12_series.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: summary[k] for k in ("round12", "round11")}, indent=1))
    return 0 if gate_new["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())