#!/usr/bin/env python3
"""Round 23 analysis: the periodic-hill ladder failure is a *stability* result.

Reads the round-23 probes (``docs/case-periodic-hills/round23_probe/*.json``) and
the four cluster ladder artefacts of round 22, classifies every run with the
pre-registered rule of :mod:`tensorfvm.verification.periodic_hill_stability`, and
writes ``round23_table.json`` + ``fig_round23_stability.png``.

Usage
-----
    PYTHONPATH=src python scripts/round23_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_stability import (
    STABLE,
    STABLE_BLEND_CFL,
    TARGET_BLEND_CFL,
    UNSTABLE_BLEND_CFL,
    classify,
    dt_for_blend_cfl,
    sustained_blend_cfl,
    sustained_local_cfl,
)

CASEDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
PROBEDIR = CASEDIR / "round23_probe"

#: the four ladder levels queued by round 22 (cj3869-cj3872); the base mesh is
#: round 20's published run.
LADDER = [
    ("96x48/dt0.020 (round 20 published)", "round20_laminar_Re2800_96x48.json"),
    ("96x48/dt0.010", "round20_laminar_Re2800_96x48dt01.json"),
    ("144x72/dt0.0133", "round20_laminar_Re2800_144x72.json"),
    ("192x96/dt0.010", "round20_laminar_Re2800_192x96.json"),
    ("240x120/dt0.008", "round20_laminar_Re2800_240x120.json"),
]


def probe_rows():
    rows = []
    for path in sorted(PROBEDIR.glob("*.json")):
        d = json.loads(path.read_text())
        cfl = sustained_local_cfl(d["trace"], start_step=200)
        bcf = sustained_blend_cfl(d["trace"], d["blend"], start_step=200)
        rows.append({
            "probe": path.stem,
            "mesh": f"{d['nx']}x{d['ny']}",
            "nx": d["nx"], "ny": d["ny"], "dt": d["dt"],
            "blend": d["blend"], "sweeps": d["sweeps"],
            "sustained_local_cfl": cfl,
            "sustained_blend_cfl": bcf,
            "verdict": classify(bcf),
            "diverged_at_step": d["diverged_at_step"],
            "diverged_where": d["diverged_where"],
            "max_abs_u_final": d["max_abs_u_final"],
            "steps_done": d["steps_done"],
            "wall_clock_s": d["wall_clock_s"],
            "dt_for_target_blend_cfl": (dt_for_blend_cfl(bcf, d["dt"], TARGET_BLEND_CFL)
                                        if bcf > 0.0 else None),
        })
    return rows


def ladder_rows():
    rows = []
    for label, name in LADDER:
        path = CASEDIR / name
        if not path.is_file():
            continue
        d = json.loads(path.read_text())
        cfg, ta, st = d["config"], d["time_averaged"], d["stationarity"]
        rows.append({
            "level": label, "artefact": name,
            "nx": cfg["nx"], "ny": cfg["ny"], "dt": cfg["time_step_s"],
            "diverged_at_step": cfg.get("diverged_at_step"),
            "samples": d.get("samples"),
            "x_r_over_H_published": ta.get("x_r_over_H"),
            "reattachment_coverage": st.get("reattachment_coverage"),
            "wall_clock_s": d.get("wall_clock_s"),
        })
    return rows


def figure(probes, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    cmap = {STABLE: "#2b8a3e", "marginal": "#e8a33d", "unstable": "#c92a2a"}
    for r in probes:
        ax.scatter(r["sustained_blend_cfl"], r["dt"],
                   s=70 if r["diverged_at_step"] is None else 110,
                   marker="o" if r["diverged_at_step"] is None else "X",
                   color=cmap.get(r["verdict"], "grey"),
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(f"{r['mesh']}", (r["sustained_blend_cfl"], r["dt"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    for x, style, lbl in ((UNSTABLE_BLEND_CFL, "--", f"UNSTABLE {UNSTABLE_BLEND_CFL:g}"),
                          (STABLE_BLEND_CFL, ":", f"STABLE {STABLE_BLEND_CFL:g}")):
        ax.axvline(x, color="grey", linestyle=style, linewidth=0.9)
        ax.text(x, ax.get_ylim()[1], lbl, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_xlabel(r"blend $\times$ sustained local Courant number")
    ax.set_ylabel(r"$\Delta t$")
    ax.set_title("round 23: circle = 1200 steps survived, X = divergent")

    # time series of max|u| for the failing level and its time-step refinement
    for stem, colour in (("base_144x72", "#c92a2a"), ("dt010", "#e8a33d"),
                         ("dt008", "#2b8a3e")):
        p = PROBEDIR / f"{stem}.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        steps = [s["step"] for s in d["trace"]]
        umax = [s["max_abs_u"] for s in d["trace"]]
        ax2.semilogy(steps, umax, color=colour, linewidth=1.1,
                     label=f"144x72 dt={d['dt']:g} " +
                           ("(diverged)" if d["diverged_at_step"] else "(stable)"))
    ax2.axhline(50.0, color="grey", linestyle=":", linewidth=0.9)
    ax2.text(10, 55.0, r"divergence guard $50U_b$", fontsize=7, color="grey")
    ax2.set_xlabel("step")
    ax2.set_ylabel(r"$\max|u|$")
    ax2.set_title("the same mesh: only $\\Delta t$ separates stable from divergent")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    probes = probe_rows()
    ladder = ladder_rows()
    if probes:
        figure(probes, CASEDIR / "fig_round23_stability.png")
    # the certified stable step for each refined ladder level, from its own probe
    stable_dt = {}
    for r in probes:
        key = r["mesh"]
        if r["verdict"] == STABLE:
            stable_dt.setdefault(key, []).append(r["dt"])
    table = {
        "round": 23,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "question": "why did every refined level of the round-22 ladder diverge?",
        "rule": {
            "indicator": "blend * (|u| dt/dx + |v| dt/deta), sustained",
            "stable_below": STABLE_BLEND_CFL, "unstable_above": UNSTABLE_BLEND_CFL,
            "target": TARGET_BLEND_CFL,
            "module": "tensorfvm.verification.periodic_hill_stability",
        },
        "probes": probes,
        "ladder": ladder,
        "stable_dt_by_mesh": {k: sorted(v) for k, v in stable_dt.items()},
        "probe_jobs": "scripts/round23_stability_probe.py (one config per process)",
        "ladder_jobs": "cj3869 (96x48 dt0.01), cj3870 (144x72), cj3871 (192x96), cj3872 (240x120)",
    }
    out = CASEDIR / "round23_table.json"
    out.write_text(json.dumps(table, indent=1))
    print(json.dumps({"n_probes": len(probes), "n_ladder": len(ladder),
                      "wrote": str(out)}, indent=1))
    for r in probes:
        print(f"{r['probe']:<24} {r['mesh']:<9} dt={r['dt']:.4f} "
              f"cfl={r['sustained_local_cfl']:.3f} {r['verdict']:<9} "
              f"div={r['diverged_at_step']}")
    for r in ladder:
        print(f"LADDER {r['level']:<34} div={r['diverged_at_step']} "
              f"samples={r['samples']} x_r/H={r['x_r_over_H_published']}")


if __name__ == "__main__":
    main()