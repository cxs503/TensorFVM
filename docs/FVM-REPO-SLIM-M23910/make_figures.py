#!/usr/bin/env python3
"""FVM-REPO-SLIM m23910: B3 实测回填 — 报告图（程序化注入，零手抄）。

fig1: 场景 pack 体积（keep-all 基线 vs 各改写场景，实测 MiB + 降幅）
fig2: 方案估算 vs 实测（B3 估算区间 vs 实测点；B2/R2c/R3 历次复核漂移）
"""
import base64
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "measured_pack_B2-B3-R2c-R3.json")))
BASE = D["baseline_keepall_pack_MiB"]

# 历次复核基线（keep-all 重打包口径，m23278 2026-11-16 / m23910 2026-11-18）
HIST = [
    ("11-16\nm23278", 1344.7, 381.8, 63.7),
    ("11-18\nm23910", BASE, 385.92, 66.18),
]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
sc = [s for s in D["scenarios"] if s["tag"] in ("B3", "B3p", "R2c", "B2", "R3")]
sc.sort(key=lambda s: s["measured_pack_MiB"])
names = [s["tag"] for s in sc]
vals = [s["measured_pack_MiB"] for s in sc]
colors = ["#2a7de1" if t == "B3" else "#9db8d9" for t in names]
bars = ax.bar(names, vals, color=colors)
ax.axhline(BASE, color="#c0392b", lw=1.6, ls="--", label="keep-all baseline")
ax.set_yscale("log")
ax.set_ylabel("measured pack (MiB, log)")
ax.set_title("FVM-REPO-SLIM scenario packs @ HEAD 8210a126 (2026-11-18)\n"
             f"baseline {BASE:.1f} MiB, B3 measured {sc[0]['measured_pack_MiB']:.1f} MiB "
             f"(-{sc[0]['reduction_pct']:.1f}%)")
for b, s in zip(bars, sc):
    ax.annotate(f"{s['measured_pack_MiB']:.1f}\n(-{s['reduction_pct']:.1f}%)",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.annotate(f"baseline {BASE:.1f}", (4.45, BASE), ha="right", va="bottom", fontsize=8, color="#c0392b")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
x = range(len(HIST))
ax.plot(x, [h[1] for h in HIST], "o-", color="#c0392b", label="baseline keep-all")
ax.plot(x, [h[2] for h in HIST], "s-", color="#2a7de1", label="R2c (docs bin+svg)")
ax.plot(x, [h[3] for h in HIST], "^-", color="#2e8b57", label="R3 (docs bin + pyc)")
for i, h in enumerate(HIST):
    ax.annotate(f"{h[1]:.1f}", (i, h[1]), textcoords="offset points", xytext=(0, 8),
             ha="center", fontsize=8)
    ax.annotate(f"{h[2]:.1f}", (i, h[2]), textcoords="offset points", xytext=(0, 8),
             ha="center", fontsize=8)
    ax.annotate(f"{h[3]:.1f}", (i, h[3]), textcoords="offset points", xytext=(0, 8),
             ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels([h[0] for h in HIST])
ax.set_yscale("log")
ax.set_ylabel("pack (MiB, log)")
ax.set_title("2-day drift (guard active): baseline +4.7 MiB, R2c +4.1, R3 +2.5")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_scenario_packs.png"), dpi=130)
b64 = base64.b64encode(open(os.path.join(HERE, "fig_scenario_packs.png"), "rb").read()).decode()
json.dump({"png_base64": b64}, open(os.path.join(HERE, "fig_scenario_packs.b64"), "w"))
print("wrote fig_scenario_packs.png", os.path.getsize(os.path.join(HERE, "fig_scenario_packs.png")))
