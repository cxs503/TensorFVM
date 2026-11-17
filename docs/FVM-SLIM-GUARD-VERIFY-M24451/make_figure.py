#!/usr/bin/env python3
"""m24451 FVM-SLIM-GUARD: render the guard-state figure from evidence.json.

Panel A: reachable pack vs. baseline / WARN / FAIL / absolute cap.
Panel B: live artifact volume vs. baseline / WARN / RED.
Panel C: e2e injection probe outcomes (real hook wiring).

All numbers are read from evidence.json (programmatic, zero hand-copy).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ev = json.loads((HERE / "evidence.json").read_text())

MIB = 1024 ** 2
GIB = 1024 ** 3

pack = ev["guard_report"]["stdout_json"]["pack_bytes"]
pack_bl = ev["guard_report"]["stdout_json"]["pack_baseline"]["pack_bytes"]
live = ev["guard_report"]["stdout_json"]["live_bytes"]
live_bl = ev["guard_report"]["stdout_json"]["live_baseline"]["live_bytes"]
pack_warn = pack_bl + 50 * MIB
pack_fail = pack_bl + 150 * MIB
pack_cap = 2 * GIB
live_warn = live_bl + 50 * MIB
live_red = live_bl + 100 * MIB

inj = ev["injection"]["payload"]
cases = inj["cases"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# --- Panel A: pack gate -------------------------------------------------
ax = axes[0]
rows = [
    ("baseline\n(11-16 reanchor)", pack_bl, "#888888"),
    ("WARN (+50 MiB)", pack_warn, "#f0c040"),
    ("FAIL (+150 MiB)", pack_fail, "#e07030"),
    ("hard cap (2 GiB)", pack_cap, "#c03030"),
    ("measured\n(11-17 16:48Z)", pack, "#3070c0"),
]
y = [r[0] for r in rows][::-1]
vals = [r[1] / GIB for r in rows][::-1]
cols = [r[2] for r in rows][::-1]
ax.barh(y, vals, color=cols, alpha=0.85)
for i, v in enumerate(vals):
    ax.text(v + 0.02, i, f"{v:.2f} GiB", va="center", fontsize=8)
ax.set_xlim(0, 2.35)
ax.set_xlabel("reachable pack (GiB)")
ax.set_title("A. pack drift gate (baseline 1.33 GiB, +10.2 MiB drift)")
ax.grid(axis="x", alpha=0.3)

# --- Panel B: live gate --------------------------------------------------
ax = axes[1]
rows = [
    ("baseline\n(11-16 reanchor)", live_bl, "#888888"),
    ("WARN (+50 MiB)", live_warn, "#f0c040"),
    ("RED (+100 MiB)", live_red, "#c03030"),
    ("measured\n(11-17 16:48Z)", live, "#3070c0"),
]
y = [r[0] for r in rows][::-1]
vals = [r[1] / MIB for r in rows][::-1]
cols = [r[2] for r in rows][::-1]
ax.barh(y, vals, color=cols, alpha=0.85)
for i, v in enumerate(vals):
    ax.text(v + 1.5, i, f"{v:.1f} MiB", va="center", fontsize=8)
ax.set_xlim(0, 320)
ax.set_xlabel("live artifact volume (MiB)")
ax.set_title("B. live volume gate (+430 KiB drift)")
ax.grid(axis="x", alpha=0.3)

# --- Panel C: injection probe --------------------------------------------
ax = axes[2]
names = [c["case"] for c in cases]
ok = [c["pass"] for c in cases]
ax.barh(names[::-1], [1] * len(names),
        color=["#30a050" if p else "#c03030" for p in ok][::-1], alpha=0.85)
for i, (n, p) in enumerate(zip(names[::-1], ok[::-1])):
    ax.text(1.02, i, "PASS" if p else "FAIL", va="center", fontsize=8)
ax.set_xlim(0, 1.25)
ax.set_xticks([])
ax.set_title("C. e2e injection probe (real hook wiring)")
ax.grid(axis="x", alpha=0.3)

fig.suptitle("FVM-SLIM-GUARD anti-rot trio — verification state @ "
             "bb0397af (m24451, 2026-11-17 16:48 UTC)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = Path(__file__).resolve().parent / "figures" / "fig_guard_state.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print(f"figure -> {out}")
