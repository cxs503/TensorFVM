"""Figures for the IN63-N02 manoeuvring report (all panels from computed data).

Every curve is produced from the campaign artefact
(``artifacts/maneuvering_summary.json``) or from a deterministic re-run of the
model; no digitized or hand-entered data is plotted.

Usage::

    PYTHONPATH=src python scripts/maneuvering_report_figures.py \
        --docs docs/maneuvering-selfprop-cfd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.maneuvering import dynamics as D, hull as H, propeller as P
from tensorfvm.maneuvering import rudder as RD, selfpropulsion as S


def _load(docs: Path) -> dict:
    return json.loads((docs / "artifacts" / "maneuvering_summary.json").read_text())


def figure_open_water(summary: dict, docs: Path) -> Path:
    water = summary["open_water"]
    js = [p["J"] for p in water]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(js, [p["KT"] for p in water], "o-", label="$K_T$")
    ax.plot(js, [10 * p["KQ"] for p in water], "s-", label="$10K_Q$")
    ax.set_xlabel("advance ratio $J$"); ax.set_ylabel("$K_T$, $10K_Q$")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title("BEMT open-water curve (KP505-type geometry)")
    ax = axes[1]
    eta = [p["eta"] for p in water]
    ideal = [P.ideal_efficiency(p["KT"], p["J"]) for p in water]
    ax.plot(js, ideal, "--", label="actuator-disk ideal $\\eta_i$")
    ax.plot(js, eta, "o-", label="BEMT $\\eta_0$")
    sp = summary["self_propulsion"]
    ax.axvline(sp["J"], color="k", ls=":", lw=1)
    ax.annotate(f"self-propulsion\n$J$={sp['J']:.3f}", (sp["J"], 0.5),
                xytext=(sp["J"] + 0.05, 0.45), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("advance ratio $J$"); ax.set_ylabel("efficiency")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    ax.set_title("Efficiency and the Betz bound")
    fig.tight_layout()
    path = docs / "figures" / "fig1_open_water_selfpropulsion.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def figure_resistance(summary: dict, docs: Path) -> Path:
    res = summary["resistance"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    labels = ["$R_F$", "$R_V=(1+k)R_F$", "$R_A$", "$R_W$", "$R_T$"]
    values = [res["r_friction"], res["r_viscous"], res["r_allowance"],
              res["r_wave"], res["r_total"]]
    colors = ["#4c72b0", "#4c72b0", "#dd8452", "#55a868", "#8172b3"]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("resistance [N] (model scale)")
    ax.set_title(f"ITTC-78 composition at $Fn$={res['froude']:.3f}")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    sweep = summary["resistance_sweep"]
    speeds = [p["speed"] for p in sweep]
    ax.plot(speeds, [p["r_total"] for p in sweep], "o-", label="$R_T$ (chain)")
    anchor = summary["reference_status"]["turning"]  # placeholder to keep flake quiet
    c_t = res["band"]["c_t_anchor"]
    q = res["r_total"] / res["c_total"]
    ax.axhline(c_t * q, color="r", ls="--",
               label=f"anchor $C_T$={c_t:.2e} (withheld)")
    ax.fill_between([min(speeds), max(speeds)],
                    res["band"]["c_t_anchor_low"] * q,
                    res["band"]["c_t_anchor_high"] * q, color="r", alpha=0.15)
    ax.set_xlabel("model speed [m/s]"); ax.set_ylabel("$R_T$ [N]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("chain vs the (withheld) resistance anchor")
    fig.tight_layout()
    path = docs / "figures" / "fig2_resistance_chain.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def figure_rudder(docs: Path) -> Path:
    geometry = H.rudder_geometry()
    analytic = RD.analytic_rudder_polar(geometry, angles_deg=(-35, -20, -10, -5, 0,
                                                              5, 10, 20, 35))
    cfd = json.loads((docs / "artifacts" / "rudder_section_cfd.json").read_text())
    runs = cfd["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(analytic.angle_deg, analytic.cl, "o-", label="analytic (Helmbold fin-AR)")
    for level, marker in (("coarse", "s"), ("medium", "^")):
        pts = [(r["angle_deg"], r["cl"]) for r in runs if r["level"] == level]
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "--",
                    label=f"2-D RANS attempt ({level})", alpha=0.8)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("rudder incidence [deg]"); ax.set_ylabel("$C_L$")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("rudder lift polar: analytic vs the failed RANS attempt")
    ax = axes[1]
    ax.plot(analytic.angle_deg, analytic.cd, "o-", label="analytic")
    for level, marker in (("coarse", "s"), ("medium", "^")):
        pts = [(r["angle_deg"], r["cd"]) for r in runs if r["level"] == level]
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "--",
                    label=f"2-D RANS ({level})", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("rudder incidence [deg]"); ax.set_ylabel("$C_D$")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("rudder drag: the RANS level is two orders too high")
    fig.tight_layout()
    path = docs / "figures" / "fig3_rudder_polar.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def figure_manoeuvres(summary: dict, docs: Path) -> Path:
    sp = S.solve_self_propulsion()
    model = D.ManoeuvreModel(self_propulsion=sp, mode="direct")
    turn = D.simulate_turning_circle(model, 35.0, dt=0.25, t_end=400.0)
    zig = D.simulate_zigzag(model, psi_a_deg=10.0, delta_deg=10.0, dt=0.25,
                            t_end=600.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    ax.plot(turn.x / turn.length, turn.y / turn.length, lw=1.6)
    metrics = summary["turning_circle"]["dt_0.25"]
    ax.plot([metrics["advance_over_L"]], [metrics["transfer_over_L"]], "o",
            color="crimson", label="90 deg point")
    ax.set_xlabel("$x/L$ (initial course)"); ax.set_ylabel("$y/L$")
    ax.set_title("turning circle, $\\delta=35^\\circ$ (direct model)\n"
                 f"$A_D/L$={metrics['advance_over_L']:.2f}, "
                 f"$A_T/L$={metrics['transfer_over_L']:.2f}, "
                 f"$D_T/L$={metrics['tactical_diameter_over_L']:.2f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_aspect("equal")
    ax = axes[1]
    ax.plot(zig.time, np.degrees(zig.psi), label="$\\psi$ [deg]")
    ax.plot(zig.time, np.degrees(zig.delta) + 20.0, label="$\\delta$ + 20 [deg]")
    ax.axhline(10.0, color="grey", ls=":", lw=1)
    ax.axhline(-10.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("time [s]"); ax.set_ylabel("angle [deg]")
    zz = summary["zigzag"]["dt_0.25"]
    ax.set_title("10/10 zig-zag\n"
                 f"$\\psi_1$={zz['psi_1_deg']:.1f} deg (outside the 4-15 deg band)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = docs / "figures" / "fig4_manoeuvres.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def figure_sensitivity(summary: dict, docs: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    wake = summary["wake_sensitivity"]
    keys = sorted(wake, key=lambda k: float(k.split("_")[1]))
    ax.plot([float(k.split("_")[1]) for k in keys],
            [wake[k]["rps"] for k in keys], "o-")
    ax.set_xlabel("wake fraction $w$ (empirical input)")
    ax.set_ylabel("self-propulsion shaft rate [rps]")
    ax.set_title("shaft-rate sensitivity to the empirical wake")
    ax.grid(alpha=0.3)
    ax = axes[1]
    sweep = summary["turning_sweep"]
    deltas = sorted(float(k.split("_")[1]) for k in sweep)
    ax.plot(deltas, [sweep[f"delta_{d:g}"]["steady_turning_radius_over_L"] for d in deltas],
            "o-", label="$R/L$ (steady)")
    ax.plot(deltas, [sweep[f"delta_{d:g}"]["tactical_diameter_over_L"] for d in deltas],
            "s--", label="$D_T/L$ (max transfer)")
    band = summary["reference_status"]["turning"]["tactical_diameter_over_L"]
    ax.axhspan(band["low"], band["high"], color="green", alpha=0.12,
               label="literature band (withheld)")
    ax.set_xlabel("rudder angle [deg]"); ax.set_ylabel("lengths")
    ax.set_title("turning response vs rudder angle")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = docs / "figures" / "fig5_sensitivity.png"
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path,
                        default=Path("docs/maneuvering-selfprop-cfd"))
    args = parser.parse_args(argv)
    summary = _load(args.docs)
    produced = [figure_open_water(summary, args.docs),
                figure_resistance(summary, args.docs),
                figure_rudder(args.docs),
                figure_manoeuvres(summary, args.docs),
                figure_sensitivity(summary, args.docs)]
    for path in produced:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())