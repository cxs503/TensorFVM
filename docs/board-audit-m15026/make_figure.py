#!/usr/bin/env python3
"""Figures for the m15026 board audit (independent recomputation)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/board-audit-m15026")
ART = Path("docs/GAP-FVM-TWOPHASE/artifacts")

recomp = json.load(open(OUT / "audit_recompute.json"))
dam = json.load(open(OUT / "audit_dambreak.json"))

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

# (a) Zalesak: geometric error vs resolution (audited values)
for scheme, mk in (("plic", "o"), ("cicsam", "s")):
    rows = sorted([r for r in recomp["zalesak"] if r["scheme"] == scheme],
                  key=lambda r: r["cells"])
    n = [r["cells"] for r in rows]
    e = [100 * r["geom_err_audited"] for r in rows]
    ax[0].plot(n, e, mk + "-", label=f"{scheme.upper()} (audited)")
    for r in rows:
        ax[0].annotate(f"{100*r['geom_err_audited']:.2f}%",
                       (r["cells"], 100 * r["geom_err_audited"]),
                       textcoords="offset points", xytext=(4, 5), fontsize=8)
ax[0].axhline(5.0, color="crimson", ls="--", label="self-declared limit 5% (N=128)")
ax[0].axvline(128, color="gray", ls=":", label="declared resolution")
ax[0].set_xscale("log", base=2)
ax[0].set_xticks([64, 128])
ax[0].set_xticklabels(["64", "128"])
ax[0].set_xlabel("cells per direction N")
ax[0].set_ylabel("thresholded geometric error [%]")
ax[0].set_title("(a) Zalesak one revolution, recomputed from NPZ fields")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

# (b) dam-break front vs Martin & Moyce
colors = {0: "tab:blue", 1: "tab:green"}
for i, r in enumerate(dam):
    c = np.array(r["sample"], dtype=float)
    ax[1].plot(c[:, 0], c[:, 2], "k-", lw=2, label="Martin & Moyce (1952)")
    ax[1].plot(c[:, 0], c[:, 1], "o-", ms=4, color=colors[i],
               label=f"PLIC {r['grid']} front (half-cell)")
ax[1].set_xlabel(r"$t^*=t\sqrt{g/a}$")
ax[1].set_ylabel(r"$x^*=(x_f-a)/a$")
ax[1].set_title("(b) Surge front: numerical lag (honest FAIL)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

# (c) offset analysis: got - ref
for i, r in enumerate(dam):
    c = np.array(r["sample"], dtype=float)
    ax[2].plot(c[:, 0], c[:, 1] - c[:, 2], "o-", ms=4, color=colors[i],
               label=f"PLIC {r['grid']}")
ax[2].axhline(0.0, color="k", ls=":")
ax[2].set_xlabel(r"$t^*$")
ax[2].set_ylabel(r"$x^*_{num} - x^*_{MM}$")
ax[2].set_title("(c) Near-constant offset => definition/initialisation, not diffusion")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "fig_audit_m15026.png", dpi=130)
print("wrote", OUT / "fig_audit_m15026.png")

# quantitative offset statistics for the report
for r in dam:
    c = np.array(r["sample"], dtype=float)
    off = c[:, 1] - c[:, 2]
    print(r["grid"], "offset mean=%.3f std=%.3f min=%.3f max=%.3f" %
          (off.mean(), off.std(), off.min(), off.max()))