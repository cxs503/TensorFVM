"""Figures for the m16204 audit report (built only from machine artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path("/root/tp-agent/work/m16204/runs/m16204")
FIG = Path("/root/tp-agent/work/m16204/docs/board_audit/m16204/figures")
FIG.mkdir(parents=True, exist_ok=True)

cmp_ = json.loads((HERE / "repro_compare.json").read_text())
probe = json.loads((HERE / "audit_probe.json").read_text())
head = cmp_["headline_numbers"]

# --- figure 1: seed independence + gate ------------------------------------
seeds = probe["P2_seed_independence"]
label_map = {"default-blasius": "default\n(Blasius seed)", "linear": "linear",
             "power-law-1over7": "1/7 power law", "uniform": "uniform"}
labels = [label_map[r["seed"]] for r in seeds]
cf_err = [100 * r["cf_err"] for r in seeds]
H_err = [100 * r["H_err"] for r in seeds]
x = np.arange(len(seeds))
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.bar(x - 0.19, cf_err, 0.38, label="$c_f$ vs Blasius", color="#2b6cb0")
ax.bar(x + 0.19, H_err, 0.38, label="$H=\\delta^*/\\theta$ vs Blasius", color="#c05621")
ax.axhspan(-3, 3, color="green", alpha=0.08, label="3 % gate")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("relative error [%]")
ax.set_title("CASE-FVM-BLASIUS-PLATE: march rung, initial-profile independence\n"
             "(mission 16182, ny=256, 401 stations, Re_x=1e5)", fontsize=9)
ax.legend(fontsize=7)
ax.set_ylim(-3, 1)
fig.tight_layout()
fig.savefig(FIG / "fig1_seed_independence.png", dpi=160)

# --- figure 2: grid/streamwise convergence --------------------------------
sens = cmp_["sensitivity"]
ny_rows = [r for r in sens if r["tag"].startswith("ny-")]
st_rows = [r for r in sens if r["tag"].startswith("stations-")]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
ax = axes[0]
ny = [r["ny"] for r in ny_rows]
ax.plot(ny, [100 * r["cf_relative_error"] for r in ny_rows], "o-", label="$c_f$")
ax.plot(ny, [100 * r["H_relative_error"] for r in ny_rows], "s-", label="$H$")
ax.axhline(0, color="k", lw=0.8)
ax.axhspan(-3, 3, color="green", alpha=0.08)
ax.set_xscale("log")
ax.set_xlabel("wall-normal nodes $n_y$ (401 stations)")
ax.set_ylabel("relative error [%]")
ax.set_title("wall-normal refinement", fontsize=9)
ax.legend(fontsize=7)
ax = axes[1]
st = [r["stations"] for r in st_rows]
ax.plot(st, [100 * r["cf_relative_error"] for r in st_rows], "o-", label="$c_f$")
ax.plot(st, [100 * r["H_relative_error"] for r in st_rows], "s-", label="$H$")
ax.axhline(0, color="k", lw=0.8)
ax.axhspan(-3, 3, color="green", alpha=0.08)
ax.set_xscale("log")
ax.set_xlabel("streamwise stations ($n_y=256$)")
ax.set_title("streamwise refinement", fontsize=9)
ax.legend(fontsize=7)
fig.suptitle("Error vs resolution: monotone decay toward the Blasius reference "
             "(green band = 3 % gate)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_resolution_convergence.png", dpi=160)

# --- figure 3: FVM flagship progress vs target ----------------------------
traces = json.loads(Path("/root/tp-agent/work/m16182/docs/case-blasius-plate/artifacts/"
                         "fvm-partial-traces.json").read_text())["traces"]
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for name, rows in traces.items():
    it = [r["iteration"] for r in rows]
    cf = [r["cf_trailing"] for r in rows]
    ax.semilogy(it, cf, "o-", label=f"{name} ({len(rows)} it.)")
ax.axhline(head["ref_cf"], color="k", ls="--", label="Blasius target $c_f$=%.4e" % head["ref_cf"])
ax.axhspan(head["ref_cf"] * 0.97, head["ref_cf"] * 1.03, color="green", alpha=0.15,
           label="±3 % band")
ax.set_xlabel("SIMPLE outer iteration")
ax.set_ylabel("trailing-edge $c_f$")
ax.set_title("Anchor device (2-D FVM, flat-plate grid) after 75 iterations: "
             "$c_f$ = %.3e = %.1f× target, WITHHELD"
             % (traces["fvm-baseline"][-1]["cf_trailing"],
                1.0 / (head["ref_cf"] / traces["fvm-baseline"][-1]["cf_trailing"])), fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig3_fvm_withheld.png", dpi=160)

print("figures:", sorted(p.name for p in FIG.glob("*.png")))