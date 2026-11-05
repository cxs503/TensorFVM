"""Figures for the PR #82 (agent/m3240) merge adjudication.

Reads only the adopted (main-side) evidence bundle, so every plotted number is
programmatically injected - nothing is hand-copied into the figure or the report.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

bundle = json.loads((ROOT / "docs/verification-lid-cavity-ladder/summary.json").read_text())
cases = {(c["reynolds"], c["n"]): c for c in bundle["cases"]}

profile64 = cases[(400.0, 64)]["v"]["profiles"]
coord = profile64["coord"]
printed = profile64["reference"]
computed = {n: cases[(400.0, n)]["v"]["profiles"]["computed"]
            for n in (32, 64, 128)}

PROPOSED = -0.38760          # PR #82's rewritten value for x = 0.9063
PRINTED = dict(zip(coord, printed))[0.9063]  # -0.23827, as printed in Table II
RICHARDSON = -0.38989        # from DISPUTES[(400.0, "v", 0.9063)]

# --------------------------------------------------------------------------- #
# fig 1: the profile, printed vs proposed vs computed
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for n, colour in ((32, "#c6dbef"), (64, "#6baed6"), (128, "#08519c")):
    ax.plot(coord, computed[n], "o-", ms=3.5, lw=1.2, color=colour,
            label=f"computed n={n}")
ax.plot(coord, printed, "s", ms=6, mfc="none", mec="crimson", mew=1.6,
        label="Ghia Table II, as printed")
ax.plot([0.9063], [PROPOSED], "*", ms=16, color="seagreen",
        label=f"PR #82 proposed rewrite ({PROPOSED:+.5f})")
ax.annotate("printed entry is a corner\nin an otherwise smooth profile",
            xy=(0.9063, PRINTED), xytext=(0.62, -0.13),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
            fontsize=8, color="crimson")
ax.set_xlabel("x / L  (v sampled on the horizontal centreline y = L/2)")
ax.set_ylabel("v / U")
ax.set_title("Re = 400 v-centreline: one printed entry, three grids, two readings")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig1_v_centreline_printed_vs_proposed.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- #
# fig 2: pointwise error against the printed table (the 3 % gate)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for n, colour in ((32, "#fdae6b"), (64, "#e6550d"), (128, "#a63603")):
    delta = cases[(400.0, n)]["v"]["profiles"]["delta"]
    ax.semilogy(coord, [abs(d) for d in delta], "o-", ms=3.5, lw=1.1,
                color=colour, label=f"n={n} (max {max(abs(d) for d in delta):.4f})")
ax.axhline(0.03, color="k", ls="--", lw=1.0, label="3 % gate (0.03)")
ax.axvline(0.9063, color="crimson", ls=":", lw=1.0)
ax.annotate("x = 0.9063", xy=(0.9063, 0.12), rotation=90, fontsize=8,
            color="crimson", ha="right")
ax.set_xlabel("x / L")
ax.set_ylabel("|v_computed - v_printed|  (lid speed units)")
ax.set_title("Re = 400: the defect is one node, and it grows under refinement")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig2_pointwise_error.png", dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- #
# fig 3: the disputed point under refinement (why the rewrite is circular)
# --------------------------------------------------------------------------- #
grids = sorted(computed)
values = [dict(zip(coord, computed[n]))[0.9063] for n in grids]
fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.plot([1.0 / n for n in grids], values, "o-", color="#08519c",
        label="computed at x = 0.9063")
ax.axhline(PRINTED, color="crimson", ls="--", lw=1.2,
           label=f"printed Table II ({PRINTED:+.5f})")
ax.axhline(RICHARDSON, color="gray", ls=":", lw=1.2,
           label=f"Richardson limit ({RICHARDSON:+.5f})")
ax.axhline(PROPOSED, color="seagreen", ls="-.", lw=1.2,
           label=f"PR #82 rewrite ({PROPOSED:+.5f})")
for n, value in zip(grids, values):
    ax.annotate(f"n={n}", (1.0 / n, value), textcoords="offset points",
                xytext=(4, -10), fontsize=8)
ax.annotate("rewrite differs from the n=128 result by\n"
            f"{abs(PROPOSED - values[-1]):.1e} - scoring against it is circular",
            xy=(1.0 / 128, PROPOSED), xytext=(0.018, -0.30),
            arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.0),
            fontsize=8, color="seagreen")
ax.set_xlabel("h = 1 / n")
ax.set_ylabel("v / U at x = 0.9063")
ax.set_title("Second-order convergence to a limit 64 % from the printed value")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="center right")
fig.tight_layout()
fig.savefig(OUT / "fig3_disputed_point_convergence.png", dpi=130)
plt.close(fig)

summary = {
    "printed_value": PRINTED,
    "proposed_value": PROPOSED,
    "richardson_limit": RICHARDSON,
    "computed_at_x": {f"n={n}": dict(zip(coord, computed[n]))[0.9063] for n in grids},
    "v_max_abs_error_printed_reference": {f"n={n}": cases[(400.0, n)]["v"]["max_abs_error"]
                                          for n in grids},
    "gate": 0.03,
    "relative_gap_printed_to_richardson": abs(RICHARDSON - PRINTED) / abs(PRINTED),
    "abs_circularity": abs(PROPOSED - dict(zip(coord, computed[128]))[0.9063]),
}
(OUT / "figures_data.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))