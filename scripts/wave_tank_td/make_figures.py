"""Make the report figures for the sigma-grid NWT from the evidence JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path("docs/wave-tank-td/artifacts")
FIG = Path("docs/wave-tank-td/figures")
FIG.mkdir(parents=True, exist_ok=True)


def wave_figure():
    d = json.load(open(ART / "wave_main.json"))
    t = np.asarray(d["series"]["t"])
    xs = d["series_x"]
    lam = d["wave"]["wavelength"]
    amp = d["wave"]["amp"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    ax = axes[0]
    for i, x in enumerate(xs):
        ax.plot(t, d["series"][f"eta_{i}"], lw=0.9, label=f"x={x / lam:.2f}$\\lambda$")
    tt = np.linspace(t[0], t[-1], 4000)
    ax.plot(tt, amp * np.cos(d["wave"]["wavenumber"] * xs[0] - d["wave"]["omega"] * tt),
            "k--", lw=0.8, label="analytic target")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\eta$ [m]")
    ax.set_title("wave_main: surface elevation at the probes (relaxation zone + beach)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    ax = axes[1]
    xn = np.array(d["probes_x_over_lambda"])
    ax.plot(xn, d["amp_ratio"], "o-", label="first harmonic / a")
    ax.plot(xn, np.asarray(d["amp_second_harmonic"]) / amp, "s--",
            label="2nd harmonic / a (Stokes-II bound wave)")
    ax.axhline(1.0 + 0.03, color="k", ls=":", lw=0.8)
    ax.axhline(1.0 - 0.03, color="k", ls=":", lw=0.8, label=r"$\pm3\%$ band")
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel("amplitude / a")
    ax.set_title(f"wave-height error = {100 * d['amp_error_max']:.2f} %, "
                 f"k error = {100 * d['k_error']:.2f} %, "
                 f"$C_R$ = {100 * d['goda']['C_R']:.2f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "wave_generation.png", dpi=140)
    plt.close(fig)


def reflection_figure():
    out = {}
    for key in ("wave_main", "control_no_beach"):
        out[key] = json.load(open(ART / f"{key}.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    labels, vals = [], []
    for key, col in (("wave_main", "tab:blue"), ("control_no_beach", "tab:red")):
        d = out[key]
        t = np.asarray(d["series"]["t"])
        xs = d["series_x"]
        i = 3
        ax.plot(t, d["series"][f"eta_{i}"], lw=0.8, color=col,
                label=f"{key} (x={xs[i] / d['wave']['wavelength']:.2f}$\\lambda$)")
        labels.append(key)
        vals.append(100.0 * d["goda"]["C_R"])
    ax.axhline(5.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\eta$ [m]")
    ax.set_title("beach on vs off (same probe)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(["with beach", "no beach (control)"], vals, color=["tab:blue", "tab:red"])
    ax.axhline(5.0, color="k", ls=":", label="criterion 5 %")
    ax.axhline(20.0, color="m", ls="--", label="control must exceed 20 %")
    ax.set_ylabel(r"$C_R$ [%] (Goda two-point)")
    ax.set_title("reflection coefficient")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "reflection.png", dpi=140)
    plt.close(fig)


def rao_figure():
    d = json.load(open(ART / "rao_coarse.json"))
    s = d["series"]
    t = np.asarray(s["t"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, np.asarray(s["heave"]) * 1000.0, lw=0.8)
    axes[0].set_ylabel("heave [mm]")
    axes[0].set_title(
        f"rao_coarse (T={d['wave']['period']:.2f} s, a={d['wave']['amp']} m): "
        f"heave RAO={d['rao_heave']:.2f}, harmonic residual={d['heave_residual']:.2f}")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, np.asarray(s["pitch"]) * 1e3, lw=0.8, color="tab:orange")
    axes[1].set_ylabel("pitch [mrad]")
    axes[1].set_xlabel("t [s]")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "rao.png", dpi=140)
    plt.close(fig)
    print("heave: min %.4f max %.4f ; pitch min %.2e max %.2e"
          % (min(s["heave"]), max(s["heave"]), min(s["pitch"]), max(s["pitch"])))
    print("first 20 heave", np.round(np.asarray(s["heave"])[:20], 4))
    print("last 20 heave", np.round(np.asarray(s["heave"])[-20:], 4))


def gci_figure():
    a = json.load(open(ART / "prop_coarse.json"))
    b = json.load(open(ART / "prop_fine.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, lbl, style in ((a, f"coarse {a['config']['nx']}x{a['config']['ns']}", "o-"),
                          (b, f"fine {b['config']['nx']}x{b['config']['ns']}", "s--")):
        ax.plot(np.asarray(d["probes_x_over_lambda"]), d["amp_ratio"], style,
                label=f"{lbl}: max err {100 * d['amp_error_max']:.2f} %")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_ylabel("first harmonic / a")
    ax.set_title("propagation accuracy on two grids (analytic start, no wavemaker)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "propagation_gci.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    wave_figure()
    reflection_figure()
    gci_figure()
    rao_figure()
    print("figures ->", FIG)
    for p in sorted(FIG.glob("*.png")):
        print(" ", p.name, p.stat().st_size)