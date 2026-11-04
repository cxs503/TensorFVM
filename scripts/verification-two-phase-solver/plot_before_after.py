#!/usr/bin/env python3
"""Before/after comparison figure for the two-phase projection fixes (m2564).

Reads ``results_before.json`` / ``results_after.json`` (produced by
``verify_two_phase_solver.py --tag before|after``) and draws the static-column
residual history plus the droplet Laplace-jump comparison.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    with open(os.path.join(HERE, "results_%s.json" % tag)) as fh:
        return json.load(fh)


def main() -> int:
    before = load("before")
    after = load("after")

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))

    for tag, res, style in (("before fix", before, "o--"), ("after fix", after, "s-")):
        hist = res["static_column"]["history"]
        ax[0].semilogy([h["step"] for h in hist],
                       [max(h["max_div"], 1e-18) for h in hist],
                       style, label="max|div u| (%s)" % tag)
    ax[0].axhline(1e-10, color="k", lw=0.7, ls=":")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("max|div u| [1/s]")
    ax[0].set_title("A: static column, dt=1e-3 s")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=7)

    labels = ["before", "after", "sigma/R"]
    values = [before["static_droplet"]["dp_num"],
              after["static_droplet"]["dp_num"],
              after["static_droplet"]["dp_ref"]]
    bars = ax[1].bar(labels, values, color=["tab:red", "tab:blue", "tab:gray"])
    for b, v in zip(bars, values):
        ax[1].text(b.get_x() + b.get_width() / 2, v, "%.4f" % v,
                   ha="center", va="bottom", fontsize=7)
    ax[1].set_ylabel("dp [Pa]")
    ax[1].set_title("B: droplet Laplace jump (R=0.25 m, sigma=0.0728 N/m)")
    ax[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("two-phase projection: defect B/A before vs after (m2564)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(HERE, "fig_compare.png")
    fig.savefig(out, dpi=110)
    print("figure:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())