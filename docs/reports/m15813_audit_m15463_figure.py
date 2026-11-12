"""Figure for the m15813 cross-audit of m15463 (CASE-CHANNEL-PERIODIC).

Panel A: audit-recomputed relative errors of the production case vs the 2% gate.
Panel B: grid convergence of the velocity L2 error with the -2 reference slope,
         showing the coarse ny=8 point that honestly fails the gate.

Inputs are the JSON produced by ``m15813_audit_m15463_verify.py`` (zero hand-typed
numbers).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
AUDIT = json.loads((ROOT / "m15813_audit_m15463_verify.json").read_text())
TOL = AUDIT["tolerance"]

prod = next(r for r in AUDIT["rows"] if r["case"] == AUDIT["production_case"])
got = prod["registered_in_json"]
ind = prod["independent_checks"]

labels = [
    "velocity L2",
    "velocity Linf",
    "u_max",
    "dp/dx (json)",
    "dp/dx (audit, momentum)",
    "tau_w (2nd order fit)",
    "Po = f Re_Dh (json)",
    "Po (audit, momentum)",
]
values = [
    got["velocity_l2"],
    got["velocity_linf"],
    got["u_max_relative_error"],
    got["pressure_gradient_relative_error"],
    ind["dpdx_rel"],
    got["tau_w_quadratic_relative_error"],
    got["poiseuille_relative_error"],
    ind["poiseuille_rel"],
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

colors = ["#1f77b4" if "audit" not in lab else "#ff7f0e" for lab in labels]
bars = ax1.barh(range(len(labels)), [100 * v for v in values], color=colors)
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel("relative error vs analytic solution [%]")
ax1.set_title(f"Production case {AUDIT['production_case']}: 8 verified metrics", fontsize=11)
ax1.axvline(100 * TOL, color="crimson", linestyle="--", linewidth=1.6,
            label=f"predeclared gate {100 * TOL:.0f}%")
ax1.set_xlim(0, max(100 * TOL * 1.35, max(100 * v for v in values) * 1.6))
for bar, value in zip(bars, values):
    ax1.text(bar.get_width() + 0.005 * 100 * TOL, bar.get_y() + bar.get_height() / 2,
             f"{100 * value:.3f}%", va="center", fontsize=8.5)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(axis="x", alpha=0.3)

mesh = sorted((r for r in AUDIT["rows"]
               if r["case"].startswith("mesh-ny") and not r["zero_filled_record"]),
              key=lambda r: int(r["case"].split("ny")[-1]))
ny = [int(r["case"].split("ny")[-1]) for r in mesh]
l2 = [r["registered_in_json"]["velocity_l2"] for r in mesh]
ax2.loglog(ny, l2, "o-", color="#1f77b4", label="velocity L2 error (artifact)")
for value_x, value_y in zip(ny, l2):
    ax2.annotate(f"{100 * value_y:.3f}%", (value_x, value_y),
                 textcoords="offset points", xytext=(6, -12), fontsize=8.5)
ax2.loglog(ny, [l2[0] * (ny[0] / n) ** 2 for n in ny], "k--", linewidth=1.2,
           label="2nd-order reference slope")
ax2.axhline(TOL, color="crimson", linestyle="--", linewidth=1.4,
            label=f"gate {100 * TOL:.0f}%")
ax2.set_xlabel("n_y (uniform, L/H = 4, Re_H = 100)")
ax2.set_ylabel("velocity L2 relative error [-]")
orders = AUDIT["observed_orders"]
ax2.set_title("Grid convergence: observed order "
              f"{orders[0]['velocity_l2_order']:.2f} -> {orders[1]['velocity_l2_order']:.2f}",
              fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)

for ax in (ax1, ax2):
    ax.tick_params(labelsize=9)
fig.suptitle("m15813 cross-audit of m15463 — CASE-CHANNEL-PERIODIC (TensorFVM): "
             "audit-recomputed evidence, zero hand-copied numbers", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = ROOT / "figures" / "m15813_audit_m15463.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# machine-readable summary used by the finished report / verdict evidence
summary = {
    "production_case": AUDIT["production_case"],
    "gate_percent": 100 * TOL,
    "metrics_percent": {lab: 100 * val for lab, val in zip(labels, values)},
    "all_verified_within_gate": all(v < TOL for v in values),
    "mesh_ny": ny,
    "velocity_l2_percent": [100 * v for v in l2],
    "missing_from_declared_plan": AUDIT["missing_from_plan"],
    "zero_filled_records": [r["case"] for r in AUDIT["rows"] if r["zero_filled_record"]],
}
(ROOT / "m15813_audit_m15463_figure_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))