#!/usr/bin/env python3
"""Round 13 analysis: score the *fixed-momentum* SST run against rounds 11/12 and Breuer 2009.

Round 12 read ``x_r/H = 3.14`` for SST at 96x48 (vs 5.34/5.41 from the primary
ERCOFTAC UFR 3-30 data) after fixing the k/omega *transport* defects.  Round 13
found a third discretisation defect -- the eta-face viscosity pairing of the
*momentum* operator (one face, two conductances) -- and fixed it in
``periodic_hill.py`` / ``periodic_hill_turbulence.py``.  This script re-scores
the time-averaged run of the fixed-momentum closure on the identical protocol
and writes the machine-readable comparison plus the paper figures.

Inputs (``docs/case-periodic-hills/``):
    round11_sst_r13-96x48.json   -- fixed momentum (this round, job cj3108)
    round11_sst_r12-96x48.json   -- round 12, fixed transport (baseline)
    round11_sst_96x48.json       -- round 11, defective transport (baseline)
Outputs:
    round13_comparison.json
    fig_round13_profile_y05.png
    fig_round13_series.png
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
NEW = DOCS / "round11_sst_r13-96x48.json"
R12 = DOCS / "round11_sst_r12-96x48.json"
R11 = DOCS / "round11_sst_96x48.json"


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def profile_errors(new, old):
    """Per-station table of the y/H=0.5 profile vs Breuer 2009.

    The stations are the gate's own scored stations (the solver's
    ``profile_comparison`` block): the 96x48 mesh has no cell centre on an
    integer x/H, so an exact string match of the column index against the
    reference keys yields an empty table (the latent defect this round
    corrects; the round-12 comparison JSON carries the same gap).
    """
    rows = []
    old_pc = (old or {}).get("profile_comparison") or {}
    old_by_x = {s["x_over_H"]: s for s in old_pc.get("stations", [])}
    for s in (new.get("profile_comparison") or {}).get("stations", []):
        x = s["x_over_H"]
        o = old_by_x.get(x)
        rows.append({
            "x_over_H": float(x),
            "reference": float(s["reference"]),
            "round13": float(s["computed"]),
            "round12": None if o is None else float(o["computed"]),
            "abs_error_round13": abs(float(s["computed"]) - float(s["reference"])),
            "rel_error_round13": float(s["error"]),
            "rel_error_round12": (None if o is None else float(o["error"])),
            "pass_round13": bool(s["pass"]),
            "pass_round12": (None if o is None else bool(o["pass"])),
        })
    return rows


def main(argv=None):
    new = _load(NEW)
    if new is None:
        print(f"missing {NEW}", file=sys.stderr)
        return 2
    r12 = _load(R12)
    r11 = _load(R11)
    gate_new = new["gate"]
    gate_r12 = r12["gate"] if r12 is not None else None
    gate_r11 = r11["gate"] if r11 is not None else None
    rows = profile_errors(new, r12)
    summary = {
        "case": "CASE-FVM-PERIODIC-HILLS round 13: momentum eta-face pairing fixed",
        "run": {"grid": "96x48", "steps": new["config"]["steps"],
                "dt": new["config"]["time_step_s"],
                "end_time": new["config"]["end_time"],
                "burn_in": new["config"]["burn_in_steps"],
                "samples": new["samples"],
                "wall_clock_s": new["wall_clock_s"],
                "closure": new["closure"]},
        "reference": {"x_r_over_H": 5.34, "tolerance": 0.10,
                      "source": "Breuer et al. 2009 / primary ERCOFTAC UFR 3-30 data"},
        "round13": {"x_r_over_H": gate_new["x_r_over_H"],
                    "x_r_over_H_error": gate_new["x_r_over_H_error"],
                    "profile_stations_pass": gate_new["profile_stations_pass"],
                    "profile_stations": gate_new["profile_stations"],
                    "stationary": gate_new["stationary"],
                    "max_nu_t_ratio": gate_new.get("max_nu_t_ratio"),
                    "pass": gate_new["pass"],
                    "reasons": gate_new["reasons"]},
        "round12": (None if gate_r12 is None else
                    {"x_r_over_H": gate_r12["x_r_over_H"],
                     "x_r_over_H_error": gate_r12["x_r_over_H_error"],
                     "profile_stations_pass": gate_r12["profile_stations_pass"],
                     "max_nu_t_ratio": gate_r12.get("max_nu_t_ratio"),
                     "pass": gate_r12["pass"]}),
        "round11": (None if gate_r11 is None else
                    {"x_r_over_H": gate_r11["x_r_over_H"],
                     "x_r_over_H_error": gate_r11["x_r_over_H_error"],
                     "profile_stations_pass": gate_r11["profile_stations_pass"],
                     "max_nu_t_ratio": gate_r11.get("max_nu_t_ratio"),
                     "pass": gate_r11["pass"]}),
        "profile": rows,
        "nu_t_series": {
            "t": new["series"]["t"],
            "max_nu_t_ratio": new["series"]["max_nu_t_ratio"],
            "x_r_over_H": new["series"]["x_r_over_H"],
        },
    }
    (DOCS / "round13_comparison.json").write_text(
        json.dumps(summary, indent=1, allow_nan=False))

    # ---- figure 1: y/H = 0.5 profile, round 13 vs round 12 vs reference
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = [r["x_over_H"] for r in rows]
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.plot(x, [r["reference"] for r in rows], "o-", color="k", lw=1.2,
            ms=4, label="Breuer 2009 (UFR 3-30, Re=2800)")
    ax.plot(x, [r["round13"] for r in rows], "s-", color="C0", lw=1.4,
            ms=4, label="round 13 (fixed momentum)")
    if all(r["round12"] is not None for r in rows):
        ax.plot(x, [r["round12"] for r in rows], "^--", color="C3", lw=1.2,
                ms=4, label="round 12 (fixed transport)")
    ax.set_xlabel("$x/H$")
    ax.set_ylabel("$u/U_b$ at $y/H=0.5$")
    ax.set_title("Periodic hills Re=2800, 96x48, time-averaged")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_round13_profile_y05.png", dpi=160)
    plt.close(fig)

    # ---- figure 2: eddy-viscosity and reattachment time series
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    a1.plot(new["series"]["t"], new["series"]["max_nu_t_ratio"], color="C0", lw=1.0)
    a1.set_xlabel("$t\\,U_b/h$")
    a1.set_ylabel("$\\max\\,\\nu_t/\\nu$")
    a1.set_title("eddy-viscosity level (fixed momentum)")
    a1.grid(alpha=0.3)
    a2.plot(new["series"]["t"], new["series"]["x_r_over_H"], color="C2", lw=1.0)
    a2.axhline(5.34, color="k", ls="--", lw=1.0, label="reference 5.34")
    a2.set_xlabel("$t\\,U_b/h$")
    a2.set_ylabel("$x_r/H$")
    a2.set_title("reattachment length vs time")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_round13_series.png", dpi=160)
    plt.close(fig)

    print(json.dumps({k: summary[k] for k in ("round13", "round12", "round11")}, indent=1))
    return 0 if gate_new["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
