"""Figures for the floating-section campaign node (numbers read from artifacts).

    python scripts/wave_tank_td/make_body_figures.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("docs/wave-tank-td-body/artifacts")
FIG = Path("docs/wave-tank-td-body/figures")


def load(name):
    return json.loads((ART / f"{name}.json").read_text())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    forced = [load(Path(f).stem) for f in sorted(glob.glob(str(ART / "forced_heave_*.json")))]
    decay = load("decay_heave_nx160")
    rao = [load(Path(f).stem) for f in sorted(glob.glob(str(ART / "rao_T*.json")))]
    diff = [load(Path(f).stem) for f in sorted(glob.glob(str(ART / "diffraction_T*.json")))]

    # ---- figure 1: free decay + added mass vs potential flow
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    s = decay
    # the decay artifact carries no series; rebuild the amplitude envelope from
    # the stored numbers is not possible, so plot the measured period against the
    # predicted one instead
    k33 = s["hydrostatic_stiffness"]["C33"]
    a_disc = s["discrete_added_mass"][0][0]
    a_meas = s["added_mass_from_frequency"]
    labels = ["tank decay\n(closed basin)", "tank discrete\n$A_{33}$ (implicit)",
              "potential flow\n$A_{33}$"]
    vals = [a_meas, a_disc, s["added_mass_potential"]]
    ax[0].bar(labels, vals, color=["#1f77b4", "#7fb3d5", "#ff7f0e"])
    ax[0].set_ylabel("$A_{33}$ [kg/m]")
    ax[0].set_title("heave added mass: three independent routes\n"
                    "(free decay period %.3f s)" % s["decay_period"])
    for i, v in enumerate(vals):
        ax[0].text(i, v, "%.0f" % v, ha="center", va="bottom", fontsize=9)
    periods = [f["period"] for f in forced if "fine" not in f["case"]]
    a_tank = [f["radiation"]["A_direct"] for f in forced if "fine" not in f["case"]]
    a_pot = [f["radiation"]["A_potential"] for f in forced if "fine" not in f["case"]]
    ax[1].plot(periods, a_tank, "o-", label="tank (forced heave)")
    ax[1].plot(periods, a_pot, "s--", label="potential flow (2-D FV Laplace)")
    ax[1].fill_between(periods, np.array(a_pot) * 0.85, np.array(a_pot) * 1.15,
                       color="0.85", label="±15 % band")
    ax[1].set_xlabel("wave period $T$ [s]")
    ax[1].set_ylabel("$A_{33}$ [kg/m]")
    ax[1].set_title("radiation added mass vs frequency")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "added_mass.png", dpi=140)
    plt.close(fig)

    # ---- figure 2: RAO
    if rao:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
        per = [r["period"] for r in rao]
        ax[0].plot(per, [r["motion"]["rao_heave"] for r in rao], "o-", label="tank")
        ax[0].plot(per, [r["potential"]["rao"]["rao_heave"] for r in rao], "s--",
                   label="potential flow")
        ax[0].axhline(1.0, color="0.6", lw=0.8)
        ax[0].set_xlabel("wave period $T$ [s]")
        ax[0].set_ylabel("heave RAO $\\xi_3/a$")
        ax[0].set_title("heave response, head waves")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].plot(per, [r["motion"]["rao_pitch"] for r in rao], "o-", label="tank")
        ax[1].plot(per, [r["potential"]["rao"]["rao_pitch"] for r in rao], "s--",
                   label="potential flow")
        ax[1].set_xlabel("wave period $T$ [s]")
        ax[1].set_ylabel("pitch RAO $\\xi_5/(k a)$")
        ax[1].set_title("pitch response, head waves")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "rao.png", dpi=140)
        plt.close(fig)

        # ---- figure 3: wave field around the section + force history
        r = rao[len(rao) // 2]
        ser = r["series"]
        tt = np.asarray(ser["t"])
        fig, ax = plt.subplots(2, 1, figsize=(11.0, 6.4), sharex=True)
        ax[0].plot(tt, ser["heave"], lw=1.0, label="heave [m]")
        ax[0].plot(tt, np.asarray(ser["pitch"]) * 10.0, lw=1.0,
                   label="pitch ×10 [rad]")
        ax[0].set_ylabel("motion")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[0].set_title("T = %.2f s: heave/pitch time history (free section)" % r["period"])
        ax[1].plot(tt, ser["fx"], lw=1.0, color="#d62728")
        ax[1].axhline(r["loads"]["fx_mean"], color="k", ls="--", lw=1.0,
                      label="mean = %.1f N/m (added resistance)" % r["loads"]["fx_mean"])
        ax[1].axhline(r["loads"]["drift_far_field"], color="#2ca02c", ls=":", lw=1.2,
                      label="far-field drift = %.1f N/m" % r["loads"]["drift_far_field"])
        ax[1].set_xlabel("t [s]")
        ax[1].set_ylabel("surge force [N/m]")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "added_resistance.png", dpi=140)
        plt.close(fig)

    # ---- figure 4: excitation force, tank vs potential flow
    if diff:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
        per = [d["period"] for d in diff]
        w = 0.03
        ax[0].bar(np.array(per) - w, [d["loads"]["fy_amp"] for d in diff], width=2 * w,
                  label="tank (time domain)")
        ax[0].bar(np.array(per) + w,
                  [abs(complex(*d["potential"]["F_exc"][0])) for d in diff],
                  width=2 * w, label="potential flow")
        ax[0].set_ylabel("$|F_3|$ [N/m]")
        ax[0].set_xlabel("wave period $T$ [s]")
        ax[0].set_title("heave excitation force on the fixed section")
        ax[0].legend(fontsize=8)
        ax[1].bar(np.array(per) - w, [d["loads"]["m5_amp"] for d in diff], width=2 * w,
                  label="tank (time domain)")
        ax[1].bar(np.array(per) + w,
                  [abs(complex(*d["potential"]["F_exc"][1])) for d in diff],
                  width=2 * w, label="potential flow")
        ax[1].set_ylabel("$|F_5|$ [Nm/m]")
        ax[1].set_xlabel("wave period $T$ [s]")
        ax[1].set_title("pitch excitation moment on the fixed section")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "excitation.png", dpi=140)
        plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()