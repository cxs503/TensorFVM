#!/usr/bin/env python3
"""FVM-SLIM-GUARD (D6, R4): live-volume-gate figure for the report.

Bar/line chart of the LIVE artifact volume gate (baseline / WARN / RED /
current) and the pack gate (baseline / WARN / FAIL / current), with the
R2c/R3 slim targets for context.  Pure infrastructure task -> no physical
field; the "comparison curve" is the gate threshold vs. the measured
volume.  Output: docs/m24008/figures/fig_live_gate.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "docs" / "m24008" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MiB = 1024 * 1024

# Live gate (measured).
live = json.loads((REPO_ROOT / "scripts" / "guard" / "live_baseline.json").read_text())
live_base = live["live_bytes"] / MiB          # 188.0 MiB (re-anchored)
live_ref = 183.0                              # m23216 census reference
live_warn = live_base + 50                    # +50 MiB WARN
live_red = live_base + 100                    # +100 MiB RED

# Pack gate (measured).
pack = json.loads((REPO_ROOT / "scripts" / "guard" / "pack_baseline.json").read_text())
pack_base = pack["pack_bytes"] / MiB          # 1362 MiB
pack_warn = pack_base + 50
pack_fail = pack_base + 150

# Slim targets (from REPO-SLIM-PLAN / m23216).
r2c = 382.6
r3 = 63.9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Live volume gate ---
labels = ["live\nbaseline\n(re-anchored)", "live\nWARN\n(+50 MiB)",
          "live\nRED\n(+100 MiB)", "current\nlive"]
vals = [live_base, live_warn, live_red, live_base]
colors = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"]
bars = ax1.bar(labels, vals, color=colors, alpha=0.85)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f} MiB",
             ha="center", va="bottom", fontsize=9)
ax1.axhline(live_ref, color="gray", ls="--", lw=1,
            label=f"m23216 reference {live_ref:.0f} MiB")
ax1.set_ylabel("live artifact volume (MiB)")
ax1.set_title("LIVE volume gate (unique HEAD-referenced blobs)")
ax1.legend(fontsize=8)
ax1.set_ylim(0, live_red + 40)

# --- Pack gate (context) ---
labels2 = ["pack\nbaseline", "pack\nWARN\n(+50 MiB)", "pack\nFAIL\n(+150 MiB)",
           "current\npack", "R2c\ntarget", "R3\ntarget"]
vals2 = [pack_base, pack_warn, pack_fail, pack_base, r2c, r3]
colors2 = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4", "#9467bd", "#8c564b"]
bars2 = ax2.bar(labels2, vals2, color=colors2, alpha=0.85)
for b, v in zip(bars2, vals2):
    ax2.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f} MiB",
             ha="center", va="bottom", fontsize=8)
ax2.set_ylabel("reachable pack volume (MiB)")
ax2.set_title("PACK gate (context) + slim targets")
ax2.set_ylim(0, pack_fail + 80)

fig.suptitle("FVM-SLIM-GUARD (D6, R4): repo volume gates — live artifacts + pack",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_live_gate.png", dpi=150)
print(f"wrote {OUT / 'fig_live_gate.png'}")
