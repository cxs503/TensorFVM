#!/usr/bin/env python3
"""m15790 图件：上游差集 + 最近 15 主线提交的主题/风险分布（数据源 diffset.json）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
facts = json.loads((HERE / "diffset.json").read_text())
commits = list(reversed(facts["commits"]))          # 时间正序
diffset = facts["diffset"]["upstream_only_records"]
up = facts["upstream"]

CATS = ["src", "tests", "scripts", "docs", "other"]
COLORS = {"src": "#c0392b", "tests": "#e67e22", "scripts": "#2980b9",
          "docs": "#7f8c8d", "other": "#bdc3c7"}

fig, axes = plt.subplots(4, 1, figsize=(12.5, 15.5),
                         gridspec_kw={"height_ratios": [0.55, 1.15, 1.0, 0.7]})


def stacked(ax, rows, tick_labels, title, hatch=None):
    bottom = [0] * len(rows)
    for cat in CATS:
        vals = [c["files_by_cat"].get(cat, 0) for c in rows]
        ax.bar(range(len(rows)), vals, bottom=bottom, color=COLORS[cat],
               label=cat, edgecolor="white", linewidth=0.5, hatch=hatch)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("files changed")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


# ---- panel 0: 上游差集（远端有、本地 HEAD 无）----
ax = axes[0]
labels0 = [("m" + c["mission_in_subject"][0]) if c["mission_in_subject"] else c["sha"] for c in diffset]
stacked(ax, diffset, labels0,
        f"UPSTREAM DIFFSET: {len(diffset)} commits on origin/main NOT in local HEAD "
        f"({up['origin_main_sha'][:8]} vs {up['local_head'][:8]}; local-only=0)",
        hatch="//")
ax.legend(ncol=5, fontsize=8, loc="upper left")

# ---- panel 1: 15 个主线提交 ----
ax = axes[1]
labels1 = [("m" + c["mission_in_subject"][0]) if c["mission_in_subject"] else c["sha"] for c in commits]
stacked(ax, commits, labels1,
        "Local mainline last 15 first-parent commits: files by area (src = red risk surface)")
ax.legend(ncol=5, fontsize=8, loc="upper left")

# ---- panel 2: 主题聚合 ----
ax = axes[2]
th = sorted(facts["theme_summary"], key=lambda t: t["insertions"])
ax.barh([t["theme"] for t in th], [max(t["insertions"], 1) for t in th], color="#16a085")
ax.set_xscale("log")
for i, t in enumerate(th):
    ax.text(max(t["insertions"], 1) * 1.08, i, f"{t['n_commits']} commits / {t['files']} files",
            va="center", fontsize=8)
ax.set_xlabel("insertions (log scale)")
ax.set_title("Change themes in the 15-commit window (bars = insertions)", fontsize=10)
ax.grid(axis="x", alpha=0.3)

# ---- panel 3: 风险计数（窗口 vs 差集）----
ax = axes[3]
rk, dk = facts["risk_summary"], facts["diffset_risk_summary"]
names = {"R1": "R1 src w/o test", "R2": "R2 blob inflation", "R3": "R3 no tree change",
         "R4": "R4 docs-only", "R5": "R5 meta-audit chain",
         "R6": "R6 docs src snapshot", "R7": "R7 merge-msg no topology"}
keys = sorted(set(rk) | set(dk))
x = range(len(keys))
ax.bar([i - 0.2 for i in x], [rk.get(k, 0) for k in keys], width=0.4,
       color="#8e44ad", label="15-commit window")
ax.bar([i + 0.2 for i in x], [dk.get(k, 0) for k in keys], width=0.4,
       color="#e67e22", label="upstream diffset (4)")
ax.set_xticks(list(x))
ax.set_xticklabels([names.get(k, k) for k in keys], fontsize=8, rotation=20, ha="right")
ax.set_ylabel("commits flagged")
ax.set_title("Risk annotations: window (violet) vs upstream diffset (orange)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out = HERE / "diffset_fig.png"
fig.savefig(out, dpi=140)
print("saved", out, out.stat().st_size, "bytes")