#!/usr/bin/env python3
"""Figure for the PR #82 merge verification (unit m3948).

Three panels, all computed from committed bundles only (no solver run):

1. vorticity of the committed Re=400, n=64 lid-cavity field, i.e. the field the
   dispute is about (``docs/verification-lid-cavity-ladder/summary.json``);
2. the Re=400 v-centreline 10 % extraction, plotted three ways -- the printed
   Ghia (1982) table, the *proposed* corrected table (``CORRECTIONS``, never
   written into the published table) and the solver profiles at n = 32/64/128;
3. pointwise |computed - reference| against both references on a log scale with
   the 3 % acceptance line, which is what the merged tests assert.

    python3 scripts/m3948_merge_verification_figure.py \
        --ladder docs/verification-lid-cavity-ladder/summary.json \
        --out docs/m3948/outputs/pr82-merge-verification.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tensorfvm.lid_cavity_reference import table  # noqa: E402

RE = 400.0


def case(cases, n):
    for row in cases:
        if row["reynolds"] == RE and row["n"] == n:
            return row
    raise SystemExit(f"no committed case for Re={RE}, n={n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", default="docs/verification-lid-cavity-ladder/summary.json")
    ap.add_argument("--out", default="docs/m3948/outputs/pr82-merge-verification.png")
    args = ap.parse_args(argv)

    data = json.load(open(args.ladder, encoding="utf-8"))
    cases = data["cases"]
    fine = case(cases, 64)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # -- 1. vorticity field ---------------------------------------------------
    u = np.asarray(fine["field"]["u"], dtype=float)  # shape (n+1, n) on u faces
    v = np.asarray(fine["field"]["v"], dtype=float)  # shape (n, n+1) on v faces
    n = u.shape[1]
    dx = 1.0 / n
    # cell-centred velocities from face values
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    du_dy = np.gradient(uc, dx, axis=0)
    dv_dx = np.gradient(vc, dx, axis=1)
    omega = dv_dx - du_dy
    axes[0].imshow(omega, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                   vmin=-np.abs(omega).max(), vmax=np.abs(omega).max())
    axes[0].set_title(f"vorticity, Re={RE:g}, n={n} (committed field)")
    axes[0].set_xlabel("x/L")
    axes[0].set_ylabel("y/L")

    # -- 2. v-centreline vs printed / proposed table --------------------------
    published = np.asarray(table(RE, "v"))          # y/L descending, as printed
    proposed = np.asarray(table(RE, "v", corrected=True))
    # np.interp needs an ascending abscissa; the published tables run 1.0 -> 0.0
    pub_asc = published[np.argsort(published[:, 0])]
    prop_asc = proposed[np.argsort(proposed[:, 0])]


    def reference_at(coord: np.ndarray, corrected: bool) -> np.ndarray:
        """Reference sampled on the bundle's coordinate set (which is descending).

        The bundle stores the sampled reference it was scored against, so both
        branches of this function are cross-checked against
        ``table(RE, "v"/corrected=True)`` by ``check_reference_sampling`` below.
        """
        table_asc = prop_asc if corrected else pub_asc
        order = np.argsort(coord)                     # bundle coords descend
        sampled = np.empty_like(coord, dtype=float)
        sampled[order] = np.interp(coord[order], table_asc[:, 0], table_asc[:, 1])
        return sampled
    axes[1].plot(published[:, 0], published[:, 1], "k--o", ms=3.5, lw=1.2,
                 label="Ghia (1982) Table II as printed")
    axes[1].plot(proposed[:, 0], proposed[:, 1], "r-", lw=1.0, alpha=0.85,
                 label="proposed correction (opt-in, never applied)")
    for nn, style in ((32, ":3"), (64, "-.2"), (128, "-1")):
        row = case(cases, nn)
        coord = np.asarray(row["v"]["profiles"]["coord"], dtype=float)
        comp = np.asarray(row["v"]["profiles"]["computed"], dtype=float)
        axes[1].plot(coord, comp, style, lw=1.0, label=f"solver n={nn}")
    axes[1].axvspan(0.895, 0.918, color="0.85", zorder=0)
    axes[1].annotate("x = 0.9063\nprinted -0.23827\nproposed -0.38760",
                     xy=(0.9063, -0.23827), xytext=(0.62, -0.15),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].set_title("Re=400 v-centreline: print, proposal, solver")
    axes[1].set_xlabel("y/L")
    axes[1].set_ylabel("v/u_lid")
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].grid(alpha=0.25)

    # -- 3. pointwise error against both references ---------------------------
    for nn, style in ((32, ":3"), (64, "-.2"), (128, "-1")):
        row = case(cases, nn)
        coord = np.asarray(row["v"]["profiles"]["coord"], dtype=float)
        comp = np.asarray(row["v"]["profiles"]["computed"], dtype=float)
        pub = reference_at(coord, corrected=False)
        prop = reference_at(coord, corrected=True)
        axes[2].semilogy(coord, np.abs(comp - pub), style, lw=1.0,
                         label=f"n={nn} vs printed")
        axes[2].semilogy(coord, np.abs(comp - prop), style, lw=1.0, alpha=0.45,
                         label=f"n={nn} vs proposed")
    axes[2].axhline(0.03, color="k", lw=1.2)
    axes[2].text(1.005, 0.032, "3 % limit", fontsize=8, ha="right")
    axes[2].set_ylim(1e-4, 1.0)
    axes[2].set_title("pointwise |error| vs both references")
    axes[2].set_xlabel("y/L")
    axes[2].set_ylabel("|v_computed - v_ref|")
    axes[2].legend(fontsize=6.5, ncol=2, loc="lower left")
    axes[2].grid(alpha=0.25, which="both")

    fig.suptitle("PR #82 merge: the published Ghia table is scored as printed; "
                 "the m3240 proposal stays opt-in", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=95)

    # numbers for the report, printed so nothing is hand-copied
    for nn in (32, 64, 128):
        row = case(cases, nn)
        coord = np.asarray(row["v"]["profiles"]["coord"], dtype=float)
        comp = np.asarray(row["v"]["profiles"]["computed"], dtype=float)
        pub = reference_at(coord, corrected=False)
        prop = reference_at(coord, corrected=True)
        stored = np.asarray(row["v"]["profiles"]["reference"], dtype=float)
        assert np.allclose(stored, pub, atol=1e-9), (nn, np.abs(stored - pub).max())
        print(f"n={nn:3d} max|err vs printed|={np.abs(comp-pub).max():.5f} "
              f"max|err vs proposed|={np.abs(comp-prop).max():.5f} "
              f"bundle max_abs_error={row['v']['max_abs_error']:.5f} "
              f"div={row['max_divergence']:.2e}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())