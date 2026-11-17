#!/usr/bin/env python3
"""汇总瘦身推演数字 -> census_summary.json + 报告用 matplotlib 图。"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

def read_csv(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))

by_ext = read_csv("by_extension.csv")[:8]
summary = {
    "baseline_pack_MiB": 1341.4,      # git count-objects: 1.31 GiB mirror 实测
    "planA_pack_MiB": 733.14,         # strip *.pyc *.npz *.pt
    "planB_pack_MiB": 88.89,          # A + *.png *.svg *.pdf *.msh
    "planC_pack_MiB": 1310.0,         # 只清937 agent分支+gc，不改写
    "blob_logical_MiB": 2125.23,
    "docs_logical_MiB": 1966.42,
    "main_reachable_MiB": 1956.17,
    "other_refs_only_MiB": 108.56,
    "refs_total": 1172,
    "agent_branches": 937,
    "bleed_7d_blobs": 3607,
    "bleed_7d_MiB": 156.4,
    "filter_repo_runtime_s": {"A": 10.7, "B": 6.7},
    "postB_git_MiB": 92.0,
    "postB_checkout_MiB": 171.0,
    "src_integrity": "src/tests/scripts/pyproject blob sha 逐字节一致，差异仅为 pyc/npz 删除",
}
with open(os.path.join(OUT, "census_summary.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)

# 图1: 按扩展名逻辑体积
labels = [r["extension"] for r in by_ext][::-1]
vals = [float(r["logical_MiB"]) for r in by_ext][::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(labels, vals, color="#3674b5")
ax.set_xlabel("Logical size (MiB)")
ax.set_title("TensorFVM history: blob logical size by extension (Top-8, total 2125 MiB)")
for i, v in enumerate(vals):
    ax.text(v + 8, i, f"{v:.0f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_by_extension.png"), dpi=130)
plt.close(fig)

# 图2: 各方案 pack 收益
plans = ["baseline", "C: branches+gc", "A: npz/pyc/pt", "B: all artifacts out"]
packs = [summary["baseline_pack_MiB"], summary["planC_pack_MiB"],
         summary["planA_pack_MiB"], summary["planB_pack_MiB"]]
colors = ["#888888", "#b5364a", "#d8912c", "#2e9e5b"]
fig, ax = plt.subplots(figsize=(7, 4.4))
bars = ax.bar(plans, packs, color=colors)
ax.set_ylabel("pack size (MiB)")
ax.set_title("Pack-size benefit per plan (measured on disposable /tmp mirrors)")
for b, v in zip(bars, packs):
    ax.text(b.get_x() + b.get_width() / 2, v + 18, f"{v:.0f}", ha="center", fontsize=10)
ax.annotate("−93.4%", xy=(3, 88.89), xytext=(2.35, 700), fontsize=12, color="#2e9e5b",
            arrowprops=dict(arrowstyle="->", color="#2e9e5b"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_plan_benefit.png"), dpi=130)
plt.close(fig)

# 图3: Top-10 大 blob
top = read_csv("top_blobs.csv")[:10]
paths = [r["path"].split("/")[-1][:38] + f" [{r['path'].split('/')[1][:18]}]" if "/" in r["path"] else r["path"] for r in top][::-1]
sizes = [float(r["size_MiB"]) for r in top][::-1]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(paths, sizes, color="#7a4fb5")
ax.set_xlabel("Single blob size (MiB)")
ax.set_title("Top-10 largest blobs (all under docs/verification-*, added 2026-10-10~11)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_top_blobs.png"), dpi=130)
plt.close(fig)

print("figures + census_summary.json done")
