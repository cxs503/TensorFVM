#!/usr/bin/env python3
"""Render the dam-break error-profile figure from committed artefacts only.

Inputs (read, never re-simulated):
    artifacts/leg_dambreak_plic_<grid>.json   comparison rows per leg
    artifacts/dambreak_profile.json           decomposition (dambreak_profile module)

Output:
    artifacts/figures/fig4_dambreak_error_profile.png

Panel (a): the gate's own relative error vs t*, both grids, with the unchanged
15 % gate line, the segment boundaries and the developed-phase window shaded.
Panel (b): surge front x*(t*) against the Martin & Moyce digitisation, plus the
same numeric curve shifted by the best-fit startup delay -- the shift makes the
point that the developed slope already matches while the curve is late.

Usage:
    python docs/GAP-FVM-TWOPHASE/profile_report.py [--artifacts DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MM_T = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                 6.0, 6.5, 7.0, 7.5, 8.0])
MM_X = np.array([0.0, 0.75, 1.40, 1.95, 2.45, 2.90, 3.30, 3.65, 3.98, 4.28,
                 4.55, 4.80, 5.03, 5.25, 5.45, 5.64, 5.82])


def load(art: Path) -> tuple[dict, dict]:
    prof = json.loads((art / "dambreak_profile.json").read_text())
    legs = {}
    for name in prof["legs"]:
        legs[name] = json.loads((art / f"leg_dambreak_{name}.json").read_text())
    return prof, legs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts",
                    default="docs/GAP-FVM-TWOPHASE/artifacts")
    args = ap.parse_args()
    art = Path(args.artifacts)

    prof, legs = load(art)
    developed_min = float(prof["developed_min_t"])
    gate = float(prof["legs"][next(iter(prof["legs"]))]["gated_front_limit"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    styles = {"plic_120x40": ("C0", "o", "120x40"),
              "plic_240x80": ("C3", "s", "240x80")}

    # ---------------- panel (a): relative error profile ---------------- #
    for name, leg in legs.items():
        comp = np.asarray(leg["comparison"], dtype=float)
        t, got, ref = comp[:, 0], comp[:, 1], comp[:, 3]
        err = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-6)
        color, marker, label = styles.get(name, ("C7", "o", name))
        ax0.semilogy(t, np.maximum(err, 1e-16), marker=marker, ms=2.6, lw=1.1,
                     color=color, label=f"{label} (max {err.max():.1%})")

    ax0.axhline(gate, color="k", ls="--", lw=1.2, label=f"D1 gate {gate:.0%}")
    for lo, hi in ((0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)):
        ax0.axvline(hi, color="0.75", lw=0.8, ls=":")
    ax0.axvspan(0.3, developed_min, color="C1", alpha=0.10,
                label=f"startup (t* < {developed_min:.0f})")
    ax0.set_xlim(0.25, 4.05)
    ax0.set_xlabel(r"dimensionless time  $t^*=t\sqrt{g/a}$")
    ax0.set_ylabel(r"front error $|x^*_{\rm num}-x^*_{\rm exp}|/x^*_{\rm exp}$")
    ax0.set_title("(a) error decays with time: the maximum sits in the startup")
    ax0.grid(alpha=0.3, which="both")
    ax0.legend(fontsize=7.5, loc="upper right")

    # ---------------- panel (b): front curve + shift ---------------- #
    tt = np.linspace(0.0, 4.0, 400)
    ax1.plot(tt, np.interp(tt, MM_T, MM_X), "k-", lw=2.2,
             label="Martin & Moyce (1952)")
    for name, leg in legs.items():
        comp = np.asarray(leg["comparison"], dtype=float)
        color, marker, label = styles.get(name, ("C7", "o", name))
        shift = prof["legs"][name]["startup"]["shift"]
        d = float(shift["best_t_star_delay"])
        ax1.plot(comp[:, 0], comp[:, 1], marker=marker, ms=2.4, lw=1.1,
                 color=color, label=f"{label} (numeric)")
        ax1.plot(np.minimum(comp[:, 0] + d, 4.0), comp[:, 1], ls="--", lw=1.0,
                 color=color, alpha=0.75,
                 label=f"{label} shifted by +{d:.2f} t*")
        ax1.annotate("", xy=(comp[-1, 0] + d, comp[-1, 1]),
                     xytext=(comp[-1, 0], comp[-1, 1]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
    ax1.set_xlim(0.0, 4.05)
    ax1.set_ylim(0.0, 3.8)
    ax1.set_xlabel(r"dimensionless time  $t^*$")
    ax1.set_ylabel(r"surge front  $x^*=(x_f-a)/a$")
    ax1.set_title("(b) shifted curve tracks the data: a startup lag, not a slow front")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(
        "GAP-FVM-TWOPHASE dam break: where the 98.8 % max error lives "
        f"(developed-phase max {prof['legs']['plic_120x40']['developed']['front_max_rel_error']:.1%} "
        f"at t*>={developed_min:.0f})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = art / "figures" / "fig4_dambreak_error_profile.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())