"""Figure for the catalog-debt mission (m16836): two panels, numbers read from artifacts."""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
rot = json.loads((ROOT / "docs/verification-rotating-frame/summary.json").read_text())
gci = json.loads((ROOT / "docs/natural-convection-gci/summary.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))

# ---- panel A: rotating frame criteria margin (decades below their gate)
names, margins, modes, passes = [], [], [], []
for name in sorted(rot["criteria"]):
    c = rot["criteria"][name]
    v, g, m = float(c["value"]), float(c["gate"]), c["mode"]
    if m == "max":
        margin = np.inf if v <= 0 else g / v
    else:
        margin = 1e16 if g <= 0 else v / g
    margins.append(min(margin, 1e16))
    names.append(name.split("_", 1)[0])
    modes.append(m); passes.append(bool(c["pass"]))
logm = np.log10(np.clip(margins, 1e-16, 1e16))
colors = ["#2a7" if ok else "#c33" for ok in passes]
ax1.barh(range(len(names)), logm, color=colors)
ax1.axvline(0.0, color="k", lw=1.2, ls="--", label="gate (=1, pass side is right)")
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=7)
ax1.set_xlabel(r"decades of margin,  $\log_{10}$(gate/value) for 'max', $\log_{10}$(value/gate) for 'min'")
ax1.set_title(f"Rotating frame / MRF: {sum(passes)}/{len(passes)} criteria PASS\n"
              "(right of the dashed line = inside the gate)")
ax1.legend(fontsize=7); ax1.grid(axis="x", alpha=.3)

# ---- panel B: buoyant cavity Nu error vs grid, 3% band
gate = 100 * float(gci["gate"])
for ra, style in (("Ra1e+05", "o-"), ("Ra1e+06", "s-")):
    fam = gci["families"][ra]["nusselt_hot_half_cell"]
    ns = sorted(fam["values"], key=int)
    err = [100 * fam["errors_vs_reference"][n] for n in ns]
    ax2.plot([int(n) for n in ns], err, style, label=f"{ra.replace('Ra1e+', 'Ra=1e')} (half-cell Nu)")
    ax2.annotate(f"{err[-1]:+.2f} %", (int(ns[-1]), err[-1]), textcoords="offset points",
                 xytext=(6, 4), fontsize=8)
ax2.axhline(gate, color="#c33", ls="--", lw=1.2, label="3 % acceptance gate")
ax2.set_xscale("log", base=2); ax2.set_xticks([16, 32, 64]); ax2.set_xticklabels(["16", "32", "64"])
ax2.set_xlabel("grid N"); ax2.set_ylabel("|Nu - Nu_ref| / Nu_ref  [%]")
ax2.set_title("Buoyant cavity vs de Vahl Davis (1983)\nRa=1e5 PASS at N=64, Ra=1e6 honest FAIL")
ax2.legend(fontsize=8); ax2.grid(alpha=.3)

fig.tight_layout()
out = pathlib.Path(__file__).with_name("fig_catalog_debt.png")
fig.savefig(out, dpi=130)
print("wrote", out)
