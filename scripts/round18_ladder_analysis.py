#!/usr/bin/env python3
"""Round 18: grid-refinement ladder for ``CASE-FVM-PERIODIC-HILLS`` (Re = 2800).

What this round answers
-----------------------
Round 17 closed with ``x_r/H = 7.682`` (+42.0 % vs the ERCOFTAC value 5.41) for
the model-consistent (Patankar ``omega``-sink) 2-D URANS SST branch at ``96x48``
and left the obvious question open: **is the +42 % a discretisation error or the
2-D / closure dimension?**  Round 14 listed the same ladder as open item 3 and it
was never run (round 11's ``144x72`` shard stayed in flight).

This module takes the certified round-17 quantities as input and adds nothing to
the scoring path -- it *only* re-scores the stored time means with the same
certified operator and the same reference (``periodic_hill_reference``), so a
level cannot be scored by a different convention than round 17 used:

* ``x_r/H``      -> ``wall_shear_bubbles.separation_reattachment`` (window
  ``(1.9, 7.1)`` as a *reporting* filter, measured value never replaced by NaN);
* ``u/U_b`` at ``y/H = 0.5`` -> ``periodic_hill_round15.score_hill`` against the
  distributed DNS table (``y05_table``), 10 % band, relative/absolute convention
  of rounds 9--17 unchanged.

Three kinds of output
---------------------
1. ``round18_ladder.json`` -- per level: mesh spacing, solver budget, measured
   ``S_P/(beta omega)`` ratio (so an artefact cannot claim a variant it did not
   run), certified ``x_s/H`` / ``x_r/H``, error vs 5.41 and vs the legacy 5.34,
   profile station pass count, stationarity block spread.
2. A two-parameter grid-law fit ``x_r(h) = x_0 + C h^p`` (``p`` scanned, not
   assumed) **only when the sequence is monotone**; otherwise the fit is
   reported as inadmissible.  A non-monotone sequence cannot support an
   extrapolation and saying so is the honest answer, not a failure of the fit.
3. Three figures: the ladder against the +/-10 % band, the mean wall shear per
   level, and the ``y/H = 0.5`` profile error per level.

    PYTHONPATH=src python3 scripts/round18_ladder_analysis.py

Nothing here is hard-coded to a PASS: the verdict is derived from the artefacts
and the reference module, and the test ``tests/test_periodic_hill_grid_ladder.py``
re-derives it from the same files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.resolve if hasattr(Path, "resolve") else None
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorfvm.verification.periodic_hill_reference import (  # noqa: E402
    X_R_OVER_H, X_R_TOLERANCE)
from tensorfvm.verification.periodic_hill_round9 import (  # noqa: E402
    horizontal_profile)
from tensorfvm.verification.periodic_hill_round15 import (  # noqa: E402
    score_hill, write_json)
from tensorfvm.verification.wall_shear_bubbles import (  # noqa: E402
    separation_reattachment)

OUTDIR = REPO / "docs" / "case-periodic-hills"
WINDOW = (1.9, 7.1)
HILL_LENGTH = 9.0          # streamwise domain length in units of H (the case geometry)
LEGACY_X_R = 5.34          # rounds 3--9/ledger value, kept as a documented near miss

#: ladder levels, coarsest first.  The ``r18`` tags are the artefacts of
#: ``scripts/round17_measure.py`` submitted from this round.
LEVELS = ((72, 36), (96, 48), (120, 60), (144, 72))

#: levels that were *attempted and rejected*, with the measured reason: a ladder
#: level that cannot be run at the protocol time step must be recorded, not
#: quietly dropped (192x96 blows up with explicit-convection CFL at dt = 0.02:
#: ``max|u| = 1.5e85`` after 90 steps locally; stable at dt = 0.01, which would
#: quadruple the cost to ~10 000 s > the 2 h job cap).
REJECTED_LEVELS = (
    {"mesh": [192, 96], "dt": 0.02, "reason": "unstable at protocol dt (CFL)",
     "measured": "max|u| = 1.5e85 after 90 steps"},
    {"mesh": [160, 80], "dt": 0.02, "reason": "stable but 6600 s > budget margin",
     "measured": "stable, 0.33 s/step local"},
)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _rel(path: Path) -> str:
    """Path relative to the repository when possible, absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def level_paths(nx: int, ny: int, outdir: Path = OUTDIR) -> tuple[Path, Path]:
    stem = outdir / f"round11_sst_r18_{nx}x{ny}"
    # the driver's own summary is preferred (it carries the measured sink ratio and
    # the wall clock); the round-11 payload written by the same run is the fallback,
    # so a level is never dropped because one of two files went missing
    measure = outdir / f"round17_measure_{nx}x{ny}.json"
    fallback = outdir / f"round11_sst_r18_{nx}x{ny}.json"
    if not measure.exists() and fallback.exists():
        measure = fallback
    return stem.with_suffix(".npz"), measure


def score_from_npz(npz_path, payload_meta: dict | None = None) -> dict:
    """Re-score a stored time mean; the artefact carries the measured data.

    Round 17's lesson was that an artefact which stores only the *derived*
    crossings cannot be re-scored when the operator is fixed.  Every level here
    therefore stores ``mean_tau_b`` and ``mean_u``; this function reads them and
    re-derives ``x_s/H``, ``x_r/H`` and the ``y/H = 0.5`` profile from those
    numbers only.
    """
    data = np.load(npz_path)
    x_over_h = np.asarray(data["x_over_h"], dtype=float)
    tau_b = np.asarray(data["mean_tau_b"], dtype=float)
    mean_u = np.asarray(data["mean_u"], dtype=float)
    yc = np.asarray(data["yc"], dtype=float)

    rea = separation_reattachment(tau_b, x_over_h, window=WINDOW)
    hp = horizontal_profile(mean_u, yc, x_over_h, 0.5, 1.0)
    payload = {
        "time_averaged": {
            "x_s_over_H": rea["x_s_over_H"],
            "x_r_over_H": rea["x_r_over_H"],
        },
        "horizontal_profile_y_H_0.5": {
            "x_over_H": [float(v) for v in hp["x_over_h"]],
            "u_over_Ub": [float(v) for v in hp["u"]],
        },
        "config": payload_meta or {},
    }
    scored = score_hill(payload)
    return {
        "x_s_over_H": rea["x_s_over_H"],
        "x_r_over_H": rea["x_r_over_H"],
        "status": rea["status"],
        "out_of_window": bool(rea["out_of_window"]),
        "n_bubbles": int(rea["n_bubbles"]),
        "bubbles": rea["bubbles"],
        "score": scored,
        "tau_b": tau_b,
        "x_over_h": x_over_h,
        "profile_x": np.asarray(hp["x_over_h"], dtype=float),
        "profile_u": np.asarray(hp["u"], dtype=float),
    }


def load_level(nx: int, ny: int, outdir: Path = OUTDIR) -> dict | None:
    npz, js = level_paths(nx, ny, outdir)
    if not npz.exists():
        return None
    meta = json.loads(js.read_text()) if js.exists() else {}
    scored = score_from_npz(npz, meta.get("config") or {})
    cfg = (meta.get("config") or {})
    ref = float(X_R_OVER_H)
    x_r = scored["x_r_over_H"]
    return {
        "mesh": [nx, ny],
        "cells": nx * ny,
        "h_over_H": float(HILL_LENGTH / nx),
        "npz": _rel(npz),
        "json": _rel(js) if js.exists() else None,
        "config": {k: cfg.get(k) for k in
                   ("steps", "burn_in_steps", "time_step_s", "end_time",
                    "averaging_window", "convection_blend",
                    "transport_convection_blend", "reynolds",
                    "diverged_at_step")},
        "sink_coefficient_ratio": (meta.get("sink_coefficient_ratio") or {}).get("ratio"),
        "wall_clock_s": meta.get("wall_clock_s"),
        "x_s_over_H": scored["x_s_over_H"],
        "x_r_over_H": x_r,
        "agent_status": scored["status"],
        "out_of_window": scored["out_of_window"],
        "n_bubbles": scored["n_bubbles"],
        "x_r_error_vs_reference": (abs(x_r - ref) / ref
                                   if x_r is not None and np.isfinite(x_r) else None),
        "x_r_error_vs_legacy": (abs(x_r - LEGACY_X_R) / LEGACY_X_R
                                if x_r is not None and np.isfinite(x_r) else None),
        "x_r_pass": bool(x_r is not None and np.isfinite(x_r)
                         and abs(x_r - ref) / ref <= X_R_TOLERANCE),
        "profile_stations": scored["score"]["profile_stations"],
        "profile_pass": scored["score"].get("profile_pass"),
        "profile_headline": scored["score"].get("profile_headline"),
        "block_spread_rel": (meta.get("stationarity") or {}).get(
            "x_r_over_H_block_spread_rel"),
        "max_nu_t_ratio": meta.get("max_nu_t_ratio"),
        "mean_nu_t_ratio": meta.get("mean_nu_t_ratio"),
        "_arrays": scored,
    }


# --------------------------------------------------------------------------- #
# grid law
# --------------------------------------------------------------------------- #
def fit_grid_law(levels) -> dict:
    """Fit ``x_r(h) = x0 + C h^p`` with ``p`` scanned; reject when non-monotone.

    ``levels`` is a sequence of ``(h, x)`` with ``h`` strictly positive.  The fit
    is two-parameter *linear* in ``(x0, C)`` for every trial ``p``, so the residual
    is a one-dimensional function of ``p`` that is minimised on a fine grid and
    then refined -- the same deterministic strategy round 6 used for its plateau
    fit, and it cannot land in a spurious local minimum.
    """
    pts = [(float(h), float(x)) for h, x in levels
           if h and h > 0 and x is not None and np.isfinite(x)]
    if len(pts) < 3:
        return {"admissible": False,
                "reason": f"{len(pts)} admissible levels, 3 required"}
    pts.sort(key=lambda t: -t[0])       # coarse -> fine
    xs = [x for _, x in pts]
    deltas = np.diff(xs)
    monotone = bool(np.all(deltas < 0) or np.all(deltas > 0))
    h = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])

    def solve(p):
        basis = np.column_stack([np.ones_like(h), h ** p])
        coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
        resid = x - basis @ coef
        return float(coef[0]), float(coef[1]), float(np.sqrt(np.mean(resid ** 2)))

    grid = np.linspace(0.25, 4.0, 376)
    rms = [solve(p)[2] for p in grid]
    k = int(np.argmin(rms))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 200)
    fine_rms = [solve(p)[2] for p in fine]
    j = int(np.argmin(fine_rms))
    p_obs = float(fine[j])
    x0, C, rms_best = solve(p_obs)
    return {
        "admissible": bool(monotone),
        "reason": (None if monotone else
                   "measured sequence is not monotone in h; a grid-law "
                   "extrapolation is not supported"),
        "observed_order_p": p_obs,
        "extrapolated_x_r_over_H": x0,
        "coefficient_C": C,
        "rms_residual": rms_best,
        "monotone_in_h": monotone,
        "levels_coarse_to_fine": [{"h_over_H": float(a), "x_r_over_H": float(b)}
                                  for a, b in pts],
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_ladder(levels, law, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ref = float(X_R_OVER_H)
    band = X_R_TOLERANCE * ref
    ax.axhspan(ref - band, ref + band, color="tab:green", alpha=0.15,
               label=f"Breuer/ERCOFTAC $x_R/H$ = {ref} $\\pm$ {X_R_TOLERANCE:.0%}")
    ax.axhline(ref, color="tab:green", lw=1.2)
    hs = [lv["h_over_H"] for lv in levels]
    xs = [lv["x_r_over_H"] for lv in levels]
    ax.plot(hs, xs, "o-", color="tab:blue", label="2-D URANS SST (Patankar sink)")
    for h, x, lv in zip(hs, xs, levels):
        ax.annotate(f"{x:.2f}\n({lv['mesh'][0]}x{lv['mesh'][1]})", (h, x),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    if law.get("admissible"):
        hh = np.linspace(0, max(hs) * 1.15, 50)
        ax.plot(hh, law["extrapolated_x_r_over_H"]
                + law["coefficient_C"] * hh ** law["observed_order_p"],
                "--", color="tab:gray", lw=1.0,
                label=f"grid law $x_0+C h^p$, $p$ = {law['observed_order_p']:.2f}, "
                      f"$x_0$ = {law['extrapolated_x_r_over_H']:.2f}")
        ax.plot([0], [law["extrapolated_x_r_over_H"]], "x", color="tab:gray")
    ax.set_xlabel("$h/H$ (streamwise spacing, $L_x/H = 9$)")
    ax.set_ylabel("time-mean reattachment $x_r/H$")
    ax.set_xlim(0, max(hs) * 1.15)
    ax.set_ylim(min(min(xs), ref - band) - 0.6, max(xs) + 0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("CASE-FVM-PERIODIC-HILLS grid ladder, Re = 2800")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tau(levels, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lv in levels:
        arr = lv["_arrays"]
        ax.plot(arr["x_over_h"], arr["tau_b"], lw=1.2,
                label=f"{lv['mesh'][0]}x{lv['mesh'][1]}, "
                      f"$x_r/H$ = {lv['x_r_over_H']:.2f}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("$x/H$")
    ax.set_ylabel(r"time-mean wall shear $\tau_w/\rho$")
    ax.set_title("mean bottom-wall shear per mesh (zero crossing = reattachment)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_profile(levels, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    from tensorfvm.verification.periodic_hill_reference import y05_table
    table = y05_table()
    stations = sorted(table)
    ax.plot(stations, [table[s] for s in stations], "ks", ms=5,
            label="DNS UFR 3-30 (Breuer, LESOCC)")
    for lv in levels:
        arr = lv["_arrays"]
        xs, us = arr["profile_x"], arr["profile_u"]
        if xs.size < 2:
            continue
        order = np.argsort(xs)
        xs, us = xs[order], us[order]
        ax.plot(stations, [float(np.interp(s, xs, us)) for s in stations], "o-",
                ms=3.5, lw=1.0, label=f"{lv['mesh'][0]}x{lv['mesh'][1]}")
    ax.set_xlabel("$x/H$")
    ax.set_ylabel("$u/U_b$ at $y/H = 0.5$")
    ax.set_title("$y/H = 0.5$ mean profile vs DNS")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def analyse(outdir: Path = OUTDIR, figures: bool = True) -> dict:
    loaded = []
    missing = []
    for nx, ny in LEVELS:
        lv = load_level(nx, ny, outdir)
        if lv is None:
            missing.append(f"{nx}x{ny}")
        else:
            loaded.append(lv)
    if not loaded:
        raise SystemExit("no ladder artefacts found; nothing to analyse")

    law = fit_grid_law([(lv["h_over_H"], lv["x_r_over_H"]) for lv in loaded])
    ref = float(X_R_OVER_H)
    ref_band = (ref * (1 - X_R_TOLERANCE), ref * (1 + X_R_TOLERANCE))
    any_pass = any(lv["x_r_pass"] for lv in loaded)
    extrap_pass = bool(
        law.get("admissible") and law.get("extrapolated_x_r_over_H") is not None
        and abs(law["extrapolated_x_r_over_H"] - ref) / ref <= X_R_TOLERANCE)
    profile_pass_all = all((lv["profile_pass"] or 0) == len(lv["profile_stations"])
                          for lv in loaded)

    verdict = {
        "reference_x_r_over_H": ref,
        "reference_band": list(ref_band),
        "n_levels": len(loaded),
        "missing_levels": missing,
        "any_level_inside_band": bool(any_pass),
        "extrapolated_inside_band": extrap_pass,
        "all_profile_stations_pass_at_every_level": bool(profile_pass_all),
        # the case acceptance is a physics statement, so it is judged on the
        # measured finest level and on the extrapolant, never on the coarsest
        "acceptance_pass": bool(
            any_pass and profile_pass_all and
            (loaded[-1]["x_r_pass"] or extrap_pass)),
        "honest_summary": None,
    }
    finest = loaded[-1]

    def _fmt(v, spec=".2f"):
        return "undefined" if v is None else format(float(v), spec)

    def _trend(lv):
        return (f"{lv['mesh'][0]}x{lv['mesh'][1]}: x_r/H = {_fmt(lv['x_r_over_H'])}, "
                f"x_s/H = {_fmt(lv['x_s_over_H'])}, err = "
                f"{_fmt(lv['x_r_error_vs_reference'], '.1%')}")

    trend = "; ".join(_trend(lv) for lv in loaded)
    if verdict["acceptance_pass"]:
        verdict["honest_summary"] = (
            f"ladder inside the 10 % band ({trend}); grid law "
            f"x0 = {_fmt(law.get('extrapolated_x_r_over_H'))}")
    else:
        cause = ("sequence monotone in h but off the band by far more than the "
                 "level-to-level change" if law.get("admissible")
                 else "sequence not monotone in h: no grid extrapolation supported")
        verdict["honest_summary"] = (
            f"FAIL vs {ref} ({X_R_TOLERANCE:.0%} band): {trend}; finest level "
            f"profile stations passing {finest['profile_pass']}/"
            f"{len(finest['profile_stations'])}; {cause}")
    if figures:
        fig_ladder(loaded, law, outdir / "fig_round18_xr_ladder.png")
        fig_tau(loaded, outdir / "fig_round18_tau_profiles.png")
        fig_profile(loaded, outdir / "fig_round18_profile_y05.png")

    out = {
        "round": 18,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "purpose": ("attribute the round-17 +42 % reattachment error between "
                    "discretisation and the 2-D/turbulence-model dimension with "
                    "a grid-refinement ladder scored by the certified operator"),
        "levels": [{k: v for k, v in lv.items() if k != "_arrays"} for lv in loaded],
        "grid_law": law,
        "verdict": verdict,
        "legacy_reference_x_r_over_H": LEGACY_X_R,
        "rejected_levels": list(REJECTED_LEVELS),
        "figures": ["fig_round18_xr_ladder.png", "fig_round18_tau_profiles.png",
                    "fig_round18_profile_y05.png"],
    }
    write_json(out, outdir / "round18_ladder.json")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = analyse(Path(args.outdir), figures=not args.no_figures)
    print(json.dumps({"verdict": out["verdict"], "grid_law": out["grid_law"]},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())