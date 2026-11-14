"""Figure for the causal forecast audit: skill vs forecast lead, three arms."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CASES = ("cd-pe10-gmres", "poisson-n64-tol1e-12", "poisson-n64-tol1e-8")
COLOURS = {"v1": "#444444", "v2": "#d62728", "v2g": "#1f77b4"}
LABELS = {"v1": "v1 (constant tail rate)", "v2": "v2 as shipped (linear-rate extrapolation)",
          "v2g": "v2g (hold-out gated)"}


def main() -> Path:
    data = json.loads((OUT / "causal_forecast_audit.json").read_text(encoding="utf-8"))
    rows = {r["case"]: r for r in data["cases"]}
    fig, axes = plt.subplots(1, len(CASES), figsize=(4.2 * len(CASES), 3.6), sharey=True)
    for ax, name in zip(axes, CASES):
        row = rows[name]
        for arm in ("v1", "v2", "v2g"):
            lt = row["arms"][arm]["lead_time"]
            if not lt:
                continue
            xs = [e["lead_fraction"] * 100 for e in lt]
            ys = [100 * e["error"] for e in lt]
            ax.plot(xs, ys, color=COLOURS[arm], lw=1.3, label=LABELS[arm])
        ax.axhline(15.0, color="g", ls="--", lw=1.0, label="15 % acceptance")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("forecast lead (% of stopping iteration still to come)")
        ax.set_title(f"{name} (stop={row['stopping_iteration']})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|forecast − actual stop| / actual stop  [%]")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Causal forecast skill: the 0 % of ROUND2.md is an ex-post identity",
                 fontsize=10)
    fig.tight_layout()
    png = OUT / "fig_causal_forecast_skill.png"
    fig.savefig(png, dpi=150)
    print(png)
    return png


if __name__ == "__main__":
    main()
