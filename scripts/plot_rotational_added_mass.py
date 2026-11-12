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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.2))

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
    if "spheroid_analytic" in s:
        a_sp = s["spheroid_transverse"]["a_over_b"]        # semi-axis a, b = 1
        scale_sp = s["rho"] * s["spheroid_transverse"]["V_exact"] * a_sp**2
        ax2.axhline(s["spheroid_analytic"]["I_55_exact"] / scale_sp, ls="-.",
                    color="tab:red", label="closed form (C7)")
    ax2.axhline(0.0, ls="-", color="tab:gray", lw=0.8)
    ax2.set_xlabel("panels  (first series)   /   $a/b$  (second series)")
    ax2.set_ylabel(r"$I_{55}/(\rho V a^2)$")
    ax2.set_title("(b) spheroid pitch/yaw block\n(C7: closed-form anchor + limits)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    # (c) triaxial body (C8): all three rotational entries against the closed form
    tri = s.get("triaxial")
    if tri is not None:
        n3 = np.array([r["n_panels"] for r in tri["ladder"]], float)
        names = ("$I_{44}$ (roll)", "$I_{55}$ (pitch)", "$I_{66}$ (yaw)")
        for k in range(3):
            ax3.loglog(n3, [100.0 * r["rel_err"][k] for r in tri["ladder"]], "o-",
                       label=f"{names[k]} vs closed form")
        ax3.axhline(3.0, ls=":", color="k", label="3% acceptance")
        ax3.axhline(1.0, ls="--", color="tab:red", label="1% Richardson target")
        ax3.set_xlabel("panels")
        ax3.set_ylabel("relative error [%]")
        ax3.set_title("(c) triaxial " + ":".join(f"{v:g}" for v in tri["semi_axes"]) +
                      "\nthree distinct rotational entries")
        ax3.grid(True, which="both", alpha=0.3)
        ax3.legend(fontsize=7)

    fig.tight_layout()
    out = d / "fig_rotational_added_mass.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())