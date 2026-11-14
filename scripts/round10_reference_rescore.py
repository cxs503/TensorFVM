"""Round 10 of CASE-FVM-PERIODIC-HILLS: re-score the merged run against the
primary (distributed) ERCOFTAC UFR 3-30 reference instead of the hand-digitised
table, and re-derive the reattachment verdict with the corrected band.

Inputs (both already in the repository, no recomputation):
  * ``docs/case-periodic-hills/round9_96x48.npz`` -- time-mean field of the
    96x48 / 20 000-step runs of round 9 (job cj2427, shard 1);
  * ``tests/data/periodic_hill/ercoftac_ufr330/*.dat`` -- DNS profiles.

Outputs:
  * ``docs/case-periodic-hills/round10_reference_rescore.json``
  * ``docs/case-periodic-hills/fig_round10_reference.png``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfvm.periodic_hill import hill_profile
from tensorfvm.verification.periodic_hill_reference import (
    LEGACY_DIGITISED_Y05, STATIONS, X_R_OVER_H, X_R_TOLERANCE, load_profile,
    profile_at, reattachment_band)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "docs" / "case-periodic-hills"
NPZ = CASE / "round9_96x48.npz"
Y_OVER_H = 0.5


def station_field(xc, yc, mean_u, x_over_h):
    """Vertical profile of the time-mean u at one station (linear in x)."""
    x_target = x_over_h
    column = np.empty(yc.shape[0])
    for j in range(yc.shape[0]):
        column[j] = np.interp(x_target, xc[j], mean_u[j])
    return column


def admissible(x_over_h):
    """The y/H = 0.5 sampling line must be one cell above the hill surface."""
    surface, _, _ = hill_profile(np.array([x_over_h]), height=1.0, length=9.0)
    return Y_OVER_H - float(surface[0]) > 0.05


def main():
    data = np.load(NPZ)
    xc, yc, mean_u = data["xc"], data["yc"], data["mean_u"]
    tau = data["mean_tau_b"]
    x_over_h = data["x_over_h"]

    # ---- reattachment from the block/time-mean wall shear (round-9 convention)
    sign = np.sign(tau)
    recross = []
    for i in range(1, len(x_over_h)):
        if x_over_h[i] <= 0.0:
            continue
        if sign[i - 1] < 0.0 and sign[i] >= 0.0:
            recross.append(float(x_over_h[i]))
    x_r = recross[0] if recross else None
    low, high = reattachment_band()

    stations = []
    for station in STATIONS:
        if not admissible(station):
            stations.append({"x_over_h": station, "admissible": False,
                             "reason": "y/H=0.5 line inside the hill body"})
            continue
        column = station_field(xc, yc, mean_u, station)
        y_line = np.array([np.interp(station, xc[j], yc[j]) for j in range(yc.shape[0])])
        order = np.argsort(y_line)
        computed = float(np.interp(Y_OVER_H, y_line[order], column[order]))
        ref = load_profile(station)
        ref_val = profile_at(station, Y_OVER_H)
        legacy = LEGACY_DIGITISED_Y05.get(station)
        scale = abs(ref_val) if abs(ref_val) >= 0.1 else 1.0
        entry = {
            "x_over_h": station,
            "admissible": True,
            "computed": computed,
            "reference_dns": ref_val,
            "reference_legacy": legacy,
            "error_primary": abs(computed - ref_val) / scale,
            "band": "relative" if abs(ref_val) >= 0.1 else "absolute",
            "pass_primary": bool(abs(computed - ref_val) / scale <= X_R_TOLERANCE),
            "reference_file": ref["source"],
        }
        if legacy is not None:
            entry["error_legacy"] = abs(computed - legacy) / (
                abs(legacy) if abs(legacy) >= 0.1 else 1.0)
        stations.append(entry)

    usable = [s for s in stations if s["admissible"]]
    summary = {
        "case": "CASE-FVM-PERIODIC-HILLS round 10: primary-reference re-score",
        "run": {"grid": "96x48", "steps": 20000, "npz": str(NPZ.relative_to(ROOT))},
        "reference": {
            "x_r_over_H": X_R_OVER_H,
            "band": [low, high],
            "profiles": "ERCOFTAC UFR 3-30 case 4 DNS (Re=2800, LESOCC), Breuer/HSU",
        },
        "x_r_over_H": x_r,
        "x_r_pass": None if x_r is None else bool(low <= x_r <= high),
        "x_r_status": "no positive-going crossing of the mean wall shear downstream of the crest"
        if x_r is None else "crossing found",
        "stations": stations,
        "profile_pass_primary": sum(1 for s in usable if s["pass_primary"]),
        "profile_usable": len(usable),
        "worst_error_primary": max((s["error_primary"] for s in usable), default=None),
        "worst_error_legacy": max((s["error_legacy"] for s in usable
                                   if s.get("error_legacy") is not None), default=None),
    }
    summary["gate_pass"] = bool(summary["x_r_pass"] and
                                summary["profile_pass_primary"] == summary["profile_usable"])

    (CASE / "round10_reference_rescore.json").write_text(json.dumps(summary, indent=1))

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    for station, colour in ((2.0, "tab:red"), (4.0, "tab:orange"), (6.0, "tab:green")):
        ref = load_profile(station)
        ax.plot(ref["u_over_ub"], ref["y_over_h"], colour, lw=2.0,
                label=f"DNS x/H={station:g}")
        y_line = np.array([np.interp(station, xc[j], yc[j]) for j in range(yc.shape[0])])
        order = np.argsort(y_line)
        ax.plot(station_field(xc, yc, mean_u, station)[order], y_line[order],
                "o", ms=2.2, mfc="none", color=colour, label=f"k-$\\omega$? FVM x/H={station:g}")
    ax.axhline(Y_OVER_H, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel(r"$u/U_b$")
    ax.set_ylabel(r"$y/H$")
    ax.set_ylim(0.0, 3.034)
    ax.set_xlim(-0.8, 1.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(a) wall-normal profiles: FVM 96x48 vs DNS")

    ax = axes[1]
    xs = [s["x_over_h"] for s in usable]
    ax.plot(xs, [s["reference_dns"] for s in usable], "k-o", ms=4, lw=2,
            label="DNS (primary, Breuer/HSU)")
    legacy_x = [x for x in LEGACY_DIGITISED_Y05 if x in xs]
    ax.plot(legacy_x, [LEGACY_DIGITISED_Y05[x] for x in legacy_x], "s--", color="tab:red",
            ms=5, label="legacy digitised table (rounds 4-9)")
    ax.plot(xs, [s["computed"] for s in usable], "d", color="tab:blue", ms=6,
            label="FVM 96x48, round-9 time mean")
    ax.set_xlabel(r"$x/H$")
    ax.set_ylabel(r"$u/U_b$ at $y/H=0.5$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("(b) the merged reference table was the wrong one")
    fig.tight_layout()
    fig.savefig(CASE / "fig_round10_reference.png", dpi=130)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("stations",)}, indent=1))
    for s in usable:
        print(f"x/H={s['x_over_h']:4.1f}  computed={s['computed']:+.3f}  "
              f"DNS={s['reference_dns']:+.3f} (err {100*s['error_primary']:5.1f}%)  "
              f"legacy={s['reference_legacy']} "
              f"(legacy err {100*s['error_legacy']:5.1f}%)  pass={s['pass_primary']}"
              if s.get("error_legacy") is not None else
              f"x/H={s['x_over_h']:4.1f}  computed={s['computed']:+.3f}  "
              f"DNS={s['reference_dns']:+.3f} (err {100*s['error_primary']:5.1f}%)  "
              f"pass={s['pass_primary']}")


if __name__ == "__main__":
    main()
