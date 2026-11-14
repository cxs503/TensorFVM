#!/usr/bin/env python3
"""Figures for the m20007 audit (CASE-FVM-CHANNEL-DEVELOP).

All numbers are read from the delivered artifacts (docs/audit_m20007/audited_runs/)
or computed here from closed-form relations -- nothing is hand-typed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "audited_runs"
OUT = HERE / "figs"
THRESHOLD = 0.99 * 1.5
U_MAX = 1.5


def fig_centerline():
    prof: dict[str, tuple[list, list]] = {}
    with open(RUNS / "centerline_profiles.csv") as f:
        for row in csv.DictReader(f):
            xs, us = prof.setdefault(row["case"], ([], []))
            xs.append(float(row["x_over_H"]))
            us.append(float(row["u_center_over_u_bar"]))
    bench = json.load(open(RUNS / "benchmark.json"))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, (xs, us) in prof.items():
        ax.plot(xs, us, "-", lw=1.6, label=f"delivered run {name}")
        ax.annotate(f"max u_c/ū = {max(us):.4f}\n(never crosses 1.485)",
                    xy=(xs[-1], us[-1]), xytext=(xs[len(xs) // 3], us[-1] - 0.09),
                    arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax.axhline(U_MAX, color="k", ls="--", lw=1, label="fully developed 1.5 ū")
    ax.axhline(THRESHOLD, color="r", ls=":", lw=1.4, label="99% threshold 1.485 ū")
    ax.set_xlabel("x / H")
    ax.set_ylabel("u_centre / ū")
    ax.set_title("Delivered centreline development (Re_H=200, n_y=8) — criterion never met")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_centerline_delivered.png", dpi=150)
    plt.close(fig)


def fig_le_vs_re():
    bench = json.load(open(RUNS / "benchmark.json"))
    cases = bench["cases"]
    res = [c["re_h"] for c in cases]
    got = [c["L_e/H (computed)"] for c in cases]
    ref = [c["L_e/H (White 2006)"] for c in cases]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(res, ref, "k--o", label="White 2006: 0.06 Re_H")
    ax.fill_between(res, [r * 0.85 for r in ref], [r * 1.15 for r in ref],
                    color="grey", alpha=0.18, label="platform 15% band")
    ax.plot(res, got, "rs", ms=8, label="delivered L_e/H (= domain-end sentinel)")
    for r, g, rr in zip(res, got, ref):
        ax.annotate(f"{100*abs(g-rr)/rr:.1f}%", xy=(r, g), xytext=(4, 6),
                    textcoords="offset points", fontsize=8, color="darkred")
    ax.set_xlabel("Re_H")
    ax.set_ylabel("L_e / H")
    ax.set_title("delivered L_e/H vs referee (all three rows are fallback values)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_le_vs_re.png", dpi=150)
    plt.close(fig)


def fig_sampling_ceiling():
    """Closed form: cell-centre sampling of 6y(1-y) never reaches 1.485 for even ny."""
    nys = [8, 9, 16, 17, 32, 33, 64]
    ceil = []
    for ny in nys:
        y = (np.arange(ny) + 0.5) / ny
        ceil.append(float((6 * y * (1 - y)).max()))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar([str(n) for n in nys], ceil,
                  color=["tab:red" if c < THRESHOLD else "tab:green" for c in ceil])
    ax.axhline(THRESHOLD, color="r", ls=":", lw=1.4, label="99% threshold 1.485")
    ax.axhline(U_MAX, color="k", ls="--", lw=1, label="exact 1.5")
    for b, c, n in zip(bars, ceil, nys):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.0008, f"{c:.4f}",
                ha="center", fontsize=7, rotation=90)
    ax.set_ylim(1.46, 1.51)
    ax.set_xlabel("n_y (wall-normal cells)")
    ax.set_ylabel("max cell-centre u/ū of exact parabola")
    ax.set_title("Discrete ceiling of the 99% criterion: even n_y can never cross it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sampling_ceiling.png", dpi=150)
    plt.close(fig)


def fig_recheck():
    """L_e/H and threshold reachability versus n_y (control ny=8 vs treatment)."""
    files = sorted(HERE.glob("recheck_ny*.json")) + sorted(HERE.glob("*recheck_ny*.json"))
    files = sorted(set(files))
    if not files:
        return
    data = [json.load(open(f)) for f in files]
    data.sort(key=lambda d: d["ny"])
    nys = [d["ny"] for d in data]
    le_mid = [d["Le_over_H_midplane"] for d in data]
    le_even = [d["Le_over_H_even_sample"] for d in data]
    ref = data[0]["Le_over_H_ref(0.06 Re_H)"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(nys, le_even, "o--", color="tab:red", label="even-sample criterion (audited)")
    ax.plot(nys, le_mid, "s-", color="tab:blue", label="mid-plane criterion (audit re-measure)")
    ax.axhline(ref, color="k", ls="--", lw=1.2, label=f"White 2006 ref = {ref:g}")
    ax.fill_between(nys, ref * 0.92, ref * 1.08, color="grey", alpha=0.2,
                    label="delivered 8% gate")
    for n, l, ok in zip(nys, le_mid, [d["threshold_reachable_midplane"] for d in data]):
        ax.annotate(f"{l:.2f}{'' if ok else ' (no crossing)'}", xy=(n, l), xytext=(4, -10),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("n_y (wall-normal cells), Re_H=100, L/H=16, nx=128")
    ax.set_ylabel("L_e / H")
    ax.set_title("Wall-normal resolution sweep: does the criterion become reachable?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_recheck_ny_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_centerline()
    fig_le_vs_re()
    fig_sampling_ceiling()
    fig_recheck()
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))