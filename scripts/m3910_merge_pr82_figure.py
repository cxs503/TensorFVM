"""Figure for the PR #82 merge report (unit m3910).

Plots, from committed evidence only:

* left  -- the Re = 400 v-centreline at y = 0.5: published Ghia (1982) Table II
           entries, unit m3240's proposed correction of the x = 0.9063 entry, and
           the solver profile at n = 32/64/128 taken from the committed ladder
           bundle;
* right -- the pointwise |error| of that same solver profile against the
           published table and against the corrected table, on a log axis, with
           the 3 % acceptance limit drawn in.

Run from the repository root:  PYTHONPATH=src python scripts/m3910_merge_pr82_figure.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfvm.lid_cavity_reference import table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "docs", "verification-lid-cavity-ladder", "summary.json")
OUT = os.path.join(ROOT, "docs", "m3910", "outputs", "pr82-merge-re400-v-centreline.png")
LIMIT = 0.03


def main() -> None:
    bundle = json.load(open(BUNDLE))
    cases = {(c["reynolds"], c["n"]): c for c in bundle["cases"]}
    published = table(400.0, "v")
    corrected = table(400.0, "v", corrected=True)
    coord = [p[0] for p in published]
    pub = [p[1] for p in published]
    cor = [p[1] for p in corrected]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax.plot(pub, coord, "o", ms=5, mfc="white", mec="C3", label="Ghia (1982) Table II, as printed")
    ax.plot(cor, coord, "x", ms=7, color="C0", label="unit m3240 proposal (x=0.9063)")
    for i, (n, style) in enumerate(((32, ":"), (64, "--"), (128, "-"))):
        case = cases.get((400.0, n))
        if case is None:
            continue
        prof = case["v"]["profiles"]
        ax.plot(prof["computed"], prof["coord"], style, color=f"C{2 + i}", lw=1.4,
                label=f"solver n={n} (max|err| vs printed = "
                      f"{case['v']['max_abs_error']:.4f})")
    ax.annotate("disputed entry\n-0.23827 -> -0.38760",
                xy=(pub[coord.index(0.9063)], 0.9063), xytext=(-0.10, 0.72),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"), fontsize=8.5)
    ax.set_xlabel("v / U")
    ax.set_ylabel("y / L")
    ax.set_title("Re = 400: v-centreline, published vs proposed entry")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.invert_xaxis()

    for i, n in enumerate((32, 64, 128)):
        case = cases.get((400.0, n))
        if case is None:
            continue
        prof = case["v"]["profiles"]
        err_pub = [abs(c - r) for c, r in zip(prof["computed"], pub)]
        err_cor = [abs(c - r) for c, r in zip(prof["computed"], cor)]
        bx.semilogy(coord, err_pub, "o-", ms=3, lw=1.0, color=f"C{2 + i}",
                    label=f"n={n} vs printed (max {max(err_pub):.4f})")
        bx.semilogy(coord, err_cor, "s--", ms=3, lw=0.9, color=f"C{2 + i}", alpha=0.55,
                    label=f"n={n} vs corrected (max {max(err_cor):.4f})")
    bx.axhline(LIMIT, color="k", lw=1.0, ls="-.", label="3 % acceptance limit")
    bx.set_xlabel("x / L")
    bx.set_ylabel("|v_computed - v_reference|")
    bx.set_title("pointwise error: the gain is the single disputed entry")
    bx.grid(alpha=0.3, which="both")
    bx.legend(fontsize=6.8, ncol=2)
    bx.invert_xaxis()

    fig.suptitle("TensorFVM PR #82 (agent/m3240) merged into main: the reference is kept, "
                 "the proposal is registered", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(OUT)


if __name__ == "__main__":
    main()