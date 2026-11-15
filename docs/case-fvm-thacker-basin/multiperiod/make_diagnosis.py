"""Figure + summary for the Thacker multi-period diagnosis (CASE-FVM-THACKER-BASIN).

Reads the per-step / per-sample arrays produced by
``multiperiod/diag_thacker.py`` and ``multiperiod/try_fixes2.py`` and writes
``summary.json`` plus ``fig_multiperiod_diagnosis.png``.  Every number in
``summary.json`` is recomputed here from the saved arrays and the closed-form
Thacker solution -- none is typed by hand.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification import thacker_basin as tb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SCRATCH = "/srv/tensor/agent/work/m21251/outputs/m21251"
os.makedirs(OUT, exist_ok=True)

base = dict(tb.DEFAULTS)
basin = tb.build_basin(**{k: base[k] for k in ("H", "a", "amplitude", "gravity")})
T = basin.period

runs = {
    "order2": "series_baseline.npy",
    "order1": "series_order1.npy",
    "order2_cut1e-2": "series_cut1e-2.npy",
    "order2_cut3e-2": "series_cut3e-2.npy",
    "order2_fr1": "series_Fr1.npy",
    "order2_fr2": "series_Fr2.npy",
}

amp = 0.5 * (max(basin.centre_depth(t) for t in (0.0, 0.5 * T))
             - min(basin.centre_depth(t) for t in (0.0, 0.5 * T)))

rows = {}
series = {}
for name, fn in runs.items():
    path = os.path.join(SCRATCH, fn)
    if not os.path.exists(path):
        continue
    a = np.load(path)
    t_over_T, h_num, s_num = a[0], a[1], a[2]
    t = t_over_T * T
    h_ex = np.array([basin.centre_depth(x) for x in t])
    s_ex = np.array([basin.shoreline_radius(x) for x in t])
    series[name] = (t_over_T, h_num, h_ex, s_num, s_ex)
    rows[name] = {
        "centre_depth_l2_relative": float(np.sqrt(np.mean(((h_num - h_ex) / amp) ** 2))),
        "centre_depth_max_relative": float(np.max(np.abs(h_num - h_ex)) / amp),
        "shoreline_mean_relative": float(np.mean(np.abs(s_num - s_ex) / s_ex)),
        "shoreline_max_relative": float(np.max(np.abs(s_num - s_ex) / s_ex)),
        "window": "0 <= t <= 2T",
    }

# half-period window (the verified window) for the same series, as a control
half = {}
for name, (t_over_T, h_num, h_ex, s_num, s_ex) in series.items():
    sel = t_over_T <= 0.5
    half[name] = {
        "centre_depth_l2_relative": float(np.sqrt(np.mean(((h_num[sel] - h_ex[sel]) / amp) ** 2))),
        "shoreline_mean_relative": float(np.mean(np.abs(s_num[sel] - s_ex[sel]) / s_ex[sel])),
    }

# per-step velocity diagnostics (order-2 baseline) and the analytic envelope
step = np.load(os.path.join(SCRATCH, "diag_rows.npy"))
t_step, u_step = step[:, 1], step[:, 3]
X, Y = tb.grid(basin, 60, base["domain_factor"])
t_env = np.linspace(0.0, 2.0 * T, 240)
u_env = []
for tt in t_env:
    vel = basin.velocity(X, Y, tt)
    u_env.append(float(np.nanmax(np.hypot(*[np.asarray(c) for c in vel[:2]]))))
u_env = np.asarray(u_env)

summary = {
    "window": "CASE-FVM-THACKER-BASIN, n = 40, 2 periods (2T = %.6f s)" % (2 * T),
    "period_s": T,
    "analytic_velocity_envelope_max": float(u_env.max()),
    "numeric_velocity_max_order2_period2": float(u_step.max()),
    "numeric_velocity_max_order1_period2": None,
    "rows_2T": rows,
    "rows_half_period_control": half,
}

# order-1 velocity ceiling over 2T (from the sweep run recorded in the doc)
order1 = series.get("order1")
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

ax = axes[0]
ax.semilogy(t_step / T, u_step, lw=1.0, color="C3", label="order 2 (MUSCL-HLL)")
ax.semilogy(t_env / T, u_env, lw=1.6, ls="--", color="k",
            label="analytic $\\max|u|$")
ax.axhline(1.35, color="C0", ls=":", lw=1.4,
           label="order 1 (measured max $|u|$ = 1.35 m/s)")
ax.axvline(0.5, color="0.6", lw=1.0)
ax.annotate("verified window\n$0\\leq t\\leq 0.5T$", (0.5, 40), xytext=(0.62, 30),
            fontsize=8, color="0.35",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
ax.set_xlabel("$t/T$")
ax.set_ylabel("$\\max|\\mathbf{u}|$  [m/s]")
ax.set_title("(a) velocity runaway (order 2) vs stable order 1")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

ax = axes[1]
if order1 is not None:
    t2, h2, hx2, _, _ = order1
    ax.plot(t2, hx2, "k-", lw=1.6, label="exact (Thacker 1981)")
    for name, color in (("order2", "C3"), ("order1", "C0"),
                        ("order2_cut3e-2", "C2")):
        if name in series:
            tt, hh, _, _, _ = series[name]
            ax.plot(tt, hh, lw=1.0, color=color, label=name)
ax.set_xlabel("$t/T$")
ax.set_ylabel("$h(0,0,t)$  [m]")
ax.set_title("(b) centre depth over 2 periods")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
if order1 is not None:
    ax.plot(order1[0], order1[4], "k-", lw=1.6, label="exact (Thacker 1981)")
    for name, color in (("order2", "C3"), ("order1", "C0"),
                        ("order2_cut3e-2", "C2")):
        if name in series:
            tt, _, _, ss, _ = series[name]
            ax.plot(tt, ss, lw=1.0, color=color, label=name)
ax.set_xlabel("$t/T$")
ax.set_ylabel("shoreline radius  [m]")
ax.set_title("(c) shoreline radius over 2 periods")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("CASE-FVM-THACKER-BASIN multi-period diagnosis: the 0.5T window is a "
             "hard limit, not a convenience", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig_multiperiod_diagnosis.png"), dpi=140)

with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1, sort_keys=True)
print(json.dumps(summary, indent=1, sort_keys=True))
print("figure:", os.path.join(OUT, "fig_multiperiod_diagnosis.png"))