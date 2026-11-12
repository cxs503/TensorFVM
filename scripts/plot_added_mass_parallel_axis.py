"""Figures for the reference-point (parallel-axis) verification of the 6x6 added mass.

Reads ``summary.json`` produced by
``python -m tensorfvm.verification.added_mass_parallel_axis`` and writes
``fig_parallel_axis.png`` next to it.  Every number comes from the summary (no
hand-copied values).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="docs/verification-added-mass-parallel-axis")
    args = ap.parse_args(argv)
    d = pathlib.Path(args.dir)
    s = json.loads((d / "summary.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    # (a) convergence of the sphere parallel-axis anchor --------------------
    ax = axes[0]
    lad = s["sphere_offset_ladder"]
    n = np.array([r["n_panels"] for r in lad], dtype=float)
    err = np.array([abs(r["rel_err_I_xx"]) for r in lad]) * 100.0
    ax.loglog(n, err, "o-", label="panel $I_{xx}$ vs $m_a d^2$")
    ax.axhline(3.0, color="crimson", ls="--", lw=1.0, label="3 % criterion")
    o = s["sphere_richardson"]
    ax.axhline(abs(o["rel_err"]) * 100.0, color="k", ls=":", lw=1.0,
               label=f"Richardson {o['rel_err']:+.2%}")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(a) sphere parallel-axis anchor $m_a d^2$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (b) identity residual + mutation margins ----------------------------
    ax = axes[1]
    ident = [r["rel_residual"] for r in s["identity"]]
    labels = [f"{r['body']}\n{tuple(round(v, 1) for v in r['offset'])}" for r in s["identity"]]
    ax.barh(np.arange(len(ident)), ident, color="tab:blue", label="identity residual")
    ax.set_yticks(np.arange(len(ident)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(1e-12, color="crimson", ls="--", lw=1.0, label="1e-12 criterion")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\max|M(r_0) - S M(0) S^{T}| / \max|M(r_0)|$")
    ax.set_title("(b) $M(r_0)=S\\,M(0)\\,S^{T}$ residual (rounding level)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", which="both", alpha=0.3)

    # (c) spheroid offset anchor vs closed form --------------------------
    ax = axes[2]
    for g in s["spheroid_offset"]:
        rows = g["ladder"]
        nn = np.array([r["n_panels"] for r in rows], dtype=float)
        ee = np.array([abs(r["rel_err"]) for r in rows]) * 100.0
        ax.loglog(nn, ee, "o-", label=f"$a/b={g['a_over_b']:g}$ (offset $d e_y$)")
    ax.axhline(3.0, color="crimson", ls="--", lw=1.0, label="3 % criterion")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(c) spheroid offset anchor vs $I_{zz}(0)+d^2 k_{ax}\\rho V$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Parallel-axis algebra of the 6x6 added-mass matrix (debt ledger D8)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = d / "fig_parallel_axis.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())