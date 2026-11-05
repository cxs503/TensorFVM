#!/usr/bin/env python3
"""Render the cross-platform scatter record for the CSF capillary evidence.

Reads ``docs/two-phase-capillary-evidence-refresh/platform-scatter.json`` (the
machine record written from two machines) and draws the one figure the record
needs: how far the Young--Laplace verdict moves between architectures relative
to the 3 % evidence bar.  Numbers are read from the JSON, never retyped.

Usage::

    python scripts/render_two_phase_capillary_scatter.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORD = (
    REPO / "docs" / "two-phase-capillary-evidence-refresh" / "platform-scatter.json"
)
FIGURE = RECORD.with_suffix(".png")
BAR = 0.03


def main() -> None:
    record = json.loads(RECORD.read_text())
    runs = {entry["n"]: entry for entry in record["runs"]}
    grids = sorted(runs)
    exact = record["exact_jump_pa"]

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    ax = axes[0]
    loong = [100.0 * runs[n]["relative_error"]["loong64"] for n in grids]
    ax.axhspan(-100 * BAR, 100 * BAR, color="tab:green", alpha=0.15, label="±3 % bar")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(
        grids, loong, "o-", color="tab:blue", label="loong64 (702 node, 128 threads)"
    )
    for n in grids:
        ax.annotate(
            f"{loong[grids.index(n)]:+.2f}%",
            (n, loong[grids.index(n)]),
            textcoords="offset points",
            xytext=(0, 8 if loong[grids.index(n)] >= 0 else -14),
            ha="center",
            fontsize=8,
            color="tab:blue",
        )
    for n in grids:
        value = runs[n]["relative_error"]["x86_64-ci"]
        if value is None:
            continue
        ax.plot(n, 100.0 * value, "s", color="tab:red", ms=8, label="x86_64 CI")
        ax.annotate(
            f"{100.0 * value:+.2f}%",
            (n, 100.0 * value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="tab:red",
        )
    bound = [n for n in grids if runs[n]["x86_64_ci_basis"].startswith("the pytest")]
    for n in bound:
        ax.annotate(
            "< -10 % (gate bound only)",
            (n, -12.0),
            textcoords="offset points",
            xytext=(0, -18),
            ha="center",
            fontsize=8,
            color="tab:orange",
        )
        ax.plot(n, -12.0, "v", color="tab:orange", ms=7)
    ax.set_xlabel("grid n")
    ax.set_ylabel(r"$(\Delta p - \sigma/R)/(\sigma/R)$  [%]")
    ax.set_title(f"Young–Laplace error vs grid (exact $\\sigma/R$ = {exact:g} Pa)")
    ax.set_xticks(grids)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, values, colours = [], [], []
    for n in (48, 64):
        for platform, colour in (("loong64", "tab:blue"), ("x86_64-ci", "tab:red")):
            value = runs[n]["relative_error"][platform]
            if value is None:
                labels.append(f"n={n}\n{platform}\n(>10 % gate)")
                values.append(10.0)
                colours.append("tab:orange")
                continue
            labels.append(f"n={n}\n{platform}")
            values.append(abs(100.0 * value))
            colours.append(colour)
    bars = ax.bar(labels, values, color=colours, alpha=0.85)
    ax.axhline(100 * BAR, color="tab:green", ls="--", lw=1.2, label="3 % bar")
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f} %",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel(r"$|\mathrm{rel.\ error}|$  [%]")
    ax.set_title("Finer-grid verdict is not portable (bars = |>3 %| miss)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(FIGURE, dpi=170)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()
