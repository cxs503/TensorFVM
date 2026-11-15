#!/usr/bin/env python3
"""Truncation-sensitivity of the Chebyshev spectral limit, lid-driven cavity Re = 1000.

What this closes
----------------
``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL_HIORDER.md`` §3.6 left one item ``not_run``:
the Krylov ladder at ``N = 96 / 128`` had been submitted but never returned, so the
spectral limit was extrapolated from ``N = 32/48/64`` only -- three orders, one
usable decay ratio, and therefore no way to see whether the extrapolated number
moves when the truncation is refined once more.

This script consumes whatever orders are present under
``docs/CASE-FVM-CAVITY-RE1000/runs/spectral/`` and answers three questions with
numbers taken from the artefacts only:

1. is the successive-difference decay *geometric* over the orders present, and
   what is the spread of the measured ratios (the assumption the extrapolation
   rests on, now testable over several pairs)?
2. does the extrapolated limit **stabilise** as orders are added -- i.e. is the
   drift between successive extrapolations smaller than the extrapolation's own
   increment (``d_last / (r - 1)``)?
3. does the residual gap to Ghia et al. (1982) stay put while the limit stops
   moving (which is what a reference-side offset predicts), or does it shrink
   with truncation (which is what a solver-side error would predict)?

Artifacts written
-----------------
``runs/spectral/spectral_limit_analysis.json``
    every number used by the document, including the rows rejected as
    under-resolved and the reason for rejecting them;
``figures/spectral_limit_*.png``
    three figures generated here (no hand-drawn values).

Nothing is transcribed by hand; the document generated from this JSON lives in
``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL_LIMIT.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm import lid_cavity_reference as ref

CASE = Path("docs/CASE-FVM-CAVITY-RE1000")
SPEC = CASE / "runs" / "spectral"
FIGS = CASE / "figures"

#: tags in merge order; a later tag overrides an earlier one for the same order
TAGS = ("re1000", "re1000hi96", "re1000hi128")
FUNCTIONS = ("u_min", "v_max", "psi_min")
#: a row is rejected as under-resolved when the no-slip defect it still carries
#: is larger than this fraction of the lid speed.  The value is the FVM ladder's
#: own finest-grid no-slip level (n = 128 control row) rounded up by one decade,
#: so it rejects solutions that are visibly not solutions of the stated
#: boundary-value problem without judging any particular order.
SLIP_TOL = 5e-3


def load_ladder(reynolds: float = 1000.0, tags: tuple[str, ...] = TAGS) -> list[dict]:
    """Merge stored spectral records for ``reynolds``; one record per order."""
    by_order: dict[int, dict] = {}
    for tag in tags:
        for f in sorted(SPEC.glob(f"cavity_spectral_N*_{tag}.json")):
            rec = json.loads(f.read_text())
            if abs(rec.get("reynolds", -1.0) - reynolds) > 1e-9:
                continue
            rec["_file"] = f.name
            by_order[int(rec["order"])] = rec
    return [by_order[n] for n in sorted(by_order)]


def split_rows(recs: list[dict], slip_tol: float = SLIP_TOL) -> tuple[list[dict], list[dict]]:
    """(accepted, rejected) with the reason recorded for every rejection."""
    accepted, rejected = [], []
    for r in recs:
        m = r.get("metrics")
        if not m:
            rejected.append({"order": r["order"], "file": r["_file"],
                             "reason": r.get("status", "no metrics"),
                             "lid_slip_defect": None})
            continue
        slip = float(m["lid_slip_defect"])
        if slip > slip_tol:
            rejected.append({"order": r["order"], "file": r["_file"],
                             "reason": f"under-resolved (lid slip defect {slip:.2e}"
                                       f" > {slip_tol:.0e})",
                             "lid_slip_defect": slip})
            continue
        accepted.append(r)
    return accepted, rejected


def geometric_limit(orders: list[int], values: list[float]) -> dict:
    """Geometric extrapolation of a spectrally converging sequence.

    The remaining error of the finest level is ``d_last / (r - 1)`` with ``r`` the
    measured decay ratio of the successive differences, taken as the mean over
    the *tail* pairs (the last two ratios when at least two exist, otherwise the
    only one).  The spread of those ratios is returned as well: for a genuinely
    geometric sequence it is ``O(1)``, and that is the evidence the model holds.
    """
    d = [abs(float(b - a)) for a, b in zip(values, values[1:])]
    ratios = [d[i] / d[i + 1] for i in range(len(d) - 1) if d[i + 1] > 0.0]
    tail = ratios[-2:] if len(ratios) >= 2 else ratios
    r = float(np.mean(tail)) if tail else float("nan")
    increment = float(d[-1] / (r - 1.0)) if r and r > 1.0 else float("inf")
    return {
        "orders": [int(o) for o in orders],
        "values": [float(v) for v in values],
        "successive_differences": [float(x) for x in d],
        "decay_ratios": [float(x) for x in ratios],
        "ratios_used": [float(x) for x in tail],
        "ratio_spread": (float(max(tail) / min(tail)) if len(tail) > 1 else None),
        "decay_ratio": r,
        "extrapolation_increment": increment,
        "limit": float(values[-1] + increment) if math.isfinite(increment)
                 else float("nan"),
        "finest_value": float(values[-1]),
        "finest_delta": float(d[-1]),
    }


def subset_limits(orders: list[int], values: list[float]) -> list[dict]:
    """Extrapolate from every suffix with >= 2 orders; the drift is the evidence."""
    out = []
    for k in range(0, len(orders) - 1):
        lim = geometric_limit(orders[k:], values[k:])
        out.append({"orders": lim["orders"], "decay_ratio": lim["decay_ratio"],
                    "extrapolation_increment": lim["extrapolation_increment"],
                    "limit": lim["limit"],
                    "uncertainty": abs(lim["extrapolation_increment"])})
    final = out[-1]["limit"] if out else float("nan")
    for row in out:
        row["delta_vs_full_limit"] = float(abs(row["limit"] - final))
    return out


def fvm_richardson() -> dict:
    """Richardson limits of the production FVM ladder, recomputed from its JSONs."""
    out = {}
    for key, field, which in (("u_min", "u", "min_computed"),
                              ("v_max", "v", "max_computed")):
        vals = []
        for n in (32, 64, 128):
            d = json.loads((CASE / "runs" / f"lid_cavity_Re1000_n{n}.json").read_text())
            vals.append(float(d[field]["extremum"][which][1]))
        d1, d2 = vals[1] - vals[0], vals[2] - vals[1]
        p = float(np.log(abs(d1 / d2)) / np.log(2.0))
        out[key] = {"ladder": vals, "p": p,
                    "limit": float(vals[2] + d2 / (2.0 ** p - 1.0))}
    return out


def ghia_table(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(coordinate, value) arrays of a centre-line table in the reference module."""
    t = getattr(ref, name)[1000.0]
    a = np.array(t, dtype=float)
    return a[:, 0], a[:, 1]


def ghia_values() -> dict:
    """Reference values read from the certified transcription, not typed here."""
    _, u = ghia_table("U_CENTRELINE")
    _, v = ghia_table("V_CENTRELINE")
    return {
        "u_min": float(np.min(u)),
        "v_max": float(np.max(v)),
        "vortex": tuple(float(x) for x in ref.VORTEX_CENTRE[1000.0]),
    }


def analyse_function(key: str, accepted: list[dict], fvm: dict, ghia: dict) -> dict:
    orders = [int(r["order"]) for r in accepted]
    values = [float(r["metrics"][key]) for r in accepted]
    full = geometric_limit(orders, values)
    subsets = subset_limits(orders, values)
    finest = values[-1]
    limit = full["limit"]
    if len(subsets) >= 2:
        drift = float(abs(subsets[-1]["limit"] - subsets[-2]["limit"]))
        drift_unc = float(subsets[-2]["uncertainty"])
    else:
        drift, drift_unc = float("nan"), float("nan")
    out = {
        "key": key,
        "ladder": full,
        "subset_limits": subsets,
        "limit": limit,
        "finest_value": finest,
        "stability": {
            "drift_finest_vs_previous_subset": drift,
            "previous_subset_uncertainty": drift_unc,
            "drift_over_uncertainty": (drift / drift_unc if drift_unc else None),
            "max_delta_over_subsets": float(max(r["delta_vs_full_limit"]
                                                for r in subsets)) if subsets else None,
        },
    }
    if key in ghia:
        out["vs_ghia"] = {
            "reference": ghia[key],
            "limit_rel_err": float(abs(limit - ghia[key]) / abs(ghia[key])),
            "finest_rel_err": float(abs(finest - ghia[key]) / abs(ghia[key])),
            "limit_minus_reference": float(limit - ghia[key]),
        }
    else:
        # psi_min is not tabulated by Ghia et al. (1982); no reference is claimed
        out["vs_ghia"] = {"reference": None,
                          "note": "not tabulated in the reference; no comparison claimed"}
    if key in fvm:
        out["vs_fvm"] = {
            "fvm_richardson_limit": fvm[key]["limit"],
            "fvm_observed_order": fvm[key]["p"],
            "rel": float(abs(limit - fvm[key]["limit"]) / abs(fvm[key]["limit"])),
        }
    return out


def centreline_residual(accepted: list[dict]) -> dict:
    """max|computed - Ghia| on the two centre lines at the finest accepted order."""
    if not accepted:
        return {}
    finest = accepted[-1]
    y = np.array(finest["profiles"]["u_centreline"]["y"], dtype=float)
    u = np.array(finest["profiles"]["u_centreline"]["u"], dtype=float)
    x = np.array(finest["profiles"]["v_centreline"]["x"], dtype=float)
    v = np.array(finest["profiles"]["v_centreline"]["v"], dtype=float)
    yr, ur = ghia_table("U_CENTRELINE")
    xr, vr = ghia_table("V_CENTRELINE")
    return {
        "order": int(finest["order"]),
        "u_max_abs": float(np.max(np.abs(np.interp(yr, y, u) - ur))),
        "v_max_abs": float(np.max(np.abs(np.interp(xr, x, v) - vr))),
        "u_rms": float(np.sqrt(np.mean((np.interp(yr, y, u) - ur) ** 2))),
        "v_rms": float(np.sqrt(np.mean((np.interp(xr, x, v) - vr) ** 2))),
    }


def figures(analysis: dict, accepted: list[dict], ghia: dict, fvm: dict) -> list[str]:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    # -- F1: successive-difference decay (is it geometric?) -----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, style in (("u_min", "o-"), ("v_max", "s--"), ("psi_min", "^-.")):
        lad = analysis["functions"][key]["ladder"]
        ax.semilogy(lad["orders"][1:], lad["successive_differences"], style,
                    label=f"{key} (ratios {', '.join(f'{r:.1f}' for r in lad['decay_ratios'])})")
    ax.set(xlabel="polynomial order N", ylabel=r"$|f_N - f_{N-}$|",
           title="successive-difference decay, Re = 1000")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIGS / "spectral_limit_decay.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))

    # -- F2: limit stability vs truncation ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key in zip(axes, ("u_min", "v_max")):
        sub = analysis["functions"][key]["subset_limits"]
        x = [max(r["orders"]) for r in sub]
        yv = [r["limit"] for r in sub]
        err = [r["uncertainty"] for r in sub]
        ax.errorbar(x, yv, yerr=err, fmt="o-", lw=1.2, capsize=3,
                    label="spectral limit (increment as error bar)")
        ax.axhline(ghia[key], color="k", ls="--", lw=1.0, label="Ghia et al. (1982)")
        if key in fvm:
            ax.axhline(fvm[key]["limit"], color="C3", ls=":", lw=1.2,
                       label="FVM Richardson limit")
        ax.set(xlabel="finest order N used in the extrapolation", ylabel=f"{key}",
               title=f"{key}: limit vs truncation")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIGS / "spectral_limit_stability.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))

    # -- F3: centre lines at the finest order vs Ghia ----------------------
    if accepted:
        finest = accepted[-1]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        uy = np.array(finest["profiles"]["u_centreline"]["y"], dtype=float)
        uu = np.array(finest["profiles"]["u_centreline"]["u"], dtype=float)
        vx = np.array(finest["profiles"]["v_centreline"]["x"], dtype=float)
        vv = np.array(finest["profiles"]["v_centreline"]["v"], dtype=float)
        axes[0].plot(uu, uy, "C0-", lw=1.4, label=f"spectral N={finest['order']}")
        uy_r, uu_r = ghia_table("U_CENTRELINE")
        axes[0].plot(uu_r, uy_r, "ko", ms=4, mfc="none", label="Ghia et al. (1982)")
        axes[0].set(xlabel="u / U_lid", ylabel="y / L", title="u along x = L/2")
        axes[1].plot(vx, vv, "C0-", lw=1.4, label=f"spectral N={finest['order']}")
        vx_r, vv_r = ghia_table("V_CENTRELINE")
        axes[1].plot(vx_r, vv_r, "ko", ms=4, mfc="none", label="Ghia et al. (1982)")
        axes[1].set(xlabel="x / L", ylabel="v / U_lid", title="v along y = L/2")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = FIGS / "spectral_limit_centrelines.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(str(p))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(SPEC / "spectral_limit_analysis.json"))
    ap.add_argument("--slip-tol", type=float, default=SLIP_TOL,
                    help="rows with a larger lid no-slip defect are rejected")
    ap.add_argument("--jobs", default="",
                    help="comma separated cluster job ids that produced the ladder")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    recs = load_ladder()
    accepted, rejected = split_rows(recs, args.slip_tol)
    if len(accepted) < 3:
        raise SystemExit(f"need >= 3 accepted orders, have {[r['order'] for r in accepted]}")
    fvm = fvm_richardson()
    ghia = ghia_values()
    functions = {k: analyse_function(k, accepted, fvm, ghia) for k in FUNCTIONS}

    vortex_orders = [int(r["order"]) for r in accepted]
    vortex = [(float(r["metrics"]["vortex_centre"][0]),
               float(r["metrics"]["vortex_centre"][1])) for r in accepted]
    gv = ghia["vortex"]
    analysis = {
        "audit_unit": "spectral limit truncation sensitivity (Chebyshev collocation, Newton)",
        "case": "CASE-FVM-CAVITY-RE1000",
        "reynolds": 1000.0,
        "tags": list(TAGS),
        "jobs": [j for j in args.jobs.split(",") if j],
        "slip_tolerance": args.slip_tol,
        "accepted_rows": [{"order": int(r["order"]), "file": r["_file"],
                           "linear_solver": r["method"].get("linear_solver"),
                           "lid_slip_defect": float(r["metrics"]["lid_slip_defect"]),
                           "wall_slip_defect": float(r["metrics"]["wall_slip_defect"]),
                           "poisson_residual_max": float(r["metrics"]["poisson_residual_max"]),
                           "newton_iterations": r["newton"].get("iterations_at_target"),
                           "wall_seconds": float(r["wall_seconds"])}
                          for r in accepted],
        "rejected_rows": rejected,
        "functions": functions,
        "vortex_centre": {"orders": vortex_orders, "values": vortex,
                          "reference": list(gv),
                          "finest_rel": [abs(vortex[-1][0] - gv[0]) / abs(gv[0]),
                                         abs(vortex[-1][1] - gv[1]) / abs(gv[1])]},
        "centreline_residual": centreline_residual(accepted),
        "fvm": fvm,
        "ghia": ghia,
    }
    analysis["figures"] = [] if args.no_figures else figures(analysis, accepted, ghia, fvm)
    Path(args.out).write_text(json.dumps(analysis, indent=1, default=str))

    print(json.dumps({
        "orders": [r["order"] for r in accepted],
        "rejected": [(r["order"], r["reason"]) for r in rejected],
        "limit": {k: {"limit": v["limit"], "ratio": v["ladder"]["decay_ratio"],
                      "ratio_spread": v["ladder"]["ratio_spread"],
                      "vs_ghia": v["vs_ghia"].get("limit_rel_err"),
                      "drift": v["stability"]["drift_finest_vs_previous_subset"]}
                  for k, v in functions.items()},
        "out": args.out,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())