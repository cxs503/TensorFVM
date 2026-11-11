"""Figure for the rotational added-mass verification (numbers from summary.json).

    python scripts/plot_rotational_added_mass.py --dir docs/verification-rotational-added-mass
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/verification-rotational-added-mass")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    s = json.loads((d / "summary.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # (a) parallel-axis anchor: relative error of I_xx vs mesh size
    lad = s["sphere_offset_axis"]["ladder"]
    n = np.array([r["n_panels"] for r in lad], float)
    err = np.array([r["rel_err_xx"] for r in lad]) * 100.0
    ax1.loglog(n, err, "o-", label="panel $I_{xx}$ vs exact $m_a d^2$")
    ax1.axhline(s["sphere_offset_axis"]["richardson_rel_err"] * 100.0, ls="--",
                color="tab:red",
                label=f"Richardson {s['sphere_offset_axis']['richardson_rel_err']*100:.2f}%")
    ax1.axhline(3.0, ls=":", color="k", label="3% acceptance")
    ax1.set_xlabel("panels")
    ax1.set_ylabel("relative error [%]")
    ax1.set_title("(a) sphere about an axis offset by $d=1.5a$\n(exact $I_{oo}=m_a d^2$)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    # (b) pitch/yaw block: scaled value vs mesh, slender-limit trend
    lad2 = s["spheroid_transverse"]["ladder"]
    n2 = np.array([r["n_panels"] for r in lad2], float)
    v2 = np.array([r["I_55_scaled_exactV"] for r in lad2])
    ax2.semilogx(n2, v2, "s-", label=r"$I_{55}/(\rho V a^2)$, $a/b=2$")
    sl = s["slender_limit"]["rows"]
    ax2.semilogx([r["a_over_b"] for r in sl], [r["I_55_scaled"] for r in sl], "^-",
                 label=r"$I_{55}/(\rho V a^2)$ vs $a/b$")
    ax2.axhline(s["slender_limit"]["strip_theory_scaled"], ls=":", color="k",
                label="slender/strip limit 1/5")
    ax2.axhline(0.0, ls="-", color="tab:gray", lw=0.8)
    ax2.set_xlabel("panels  (first series)   /   $a/b$  (second series)")
    ax2.set_ylabel(r"$I_{55}/(\rho V a^2)$")
    ax2.set_title("(b) spheroid pitch/yaw block\n(no analytic anchor: convergence + limits)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = d / "fig_rotational_added_mass.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())