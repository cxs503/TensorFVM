#!/usr/bin/env python3
"""Round 23 stability figure: dt-sweep divergence boundary.

Reads ``round23_probe.json`` and plots the stability boundary:
diverged cases (×, red) vs stable cases (●, green) for each mesh.
Demonstrates that halving Δt at fixed mesh removes blow-up.

Usage
-----
    PYTHONPATH=src python scripts/round23_stability_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASEDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def main() -> None:
    probe_path = CASEDIR / "round23_probe.json"
    data = json.loads(probe_path.read_text())
    cases = data["cases"]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for mesh, marker in (((144, 72), "o"), ((192, 96), "s"), ((96, 48), "^")):
        pts = [c for c in cases if (c["nx"], c["ny"]) == mesh]
        if not pts:
            continue
        pts.sort(key=lambda c: c["dt"])
        dts = [c["dt"] for c in pts]
        td = [c["t_diverged"] if c["diverged_at_step"] else
              c["max_steps"] * c["dt"] for c in pts]
        ax.semilogx(dts, td, marker + "-", label=f"{mesh[0]}x{mesh[1]}")
        for c, t in zip(pts, td):
            ok = c["diverged_at_step"] is None
            ax.annotate("stable" if ok else f"div@{c['diverged_at_step']}",
                        (c["dt"], t), textcoords="offset points",
                        xytext=(4, 5), fontsize=7,
                        color="C2" if ok else "C3")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$t$ reached (or $t$ of blow-up)")
    ax.set_title("CASE-FVM-PERIODIC-HILLS round 23: dt-sweep stability boundary")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(CASEDIR / "fig_round23_stability.png", dpi=140)
    plt.close(fig)
    print(f"Generated {CASEDIR / 'fig_round23_stability.png'}")


if __name__ == "__main__":
    main()
