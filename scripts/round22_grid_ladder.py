#!/usr/bin/env python3
"""Round 22 driver: grid / time-step convergence attribution for the periodic hill.

Reads the closure-free (``laminar``) ``Re = 2800`` run artefacts of
``CASE-FVM-PERIODIC-HILLS`` and answers one question that rounds 17--21 left
open: **is the +53 % error of ``x_r/H`` a discretisation error?**

The ladder is uniformly refined in space with the CFL number held fixed, plus one
time-step refinement at fixed mesh, so that the two axes can be separated:

=====  ======  ======  =========  =============================
level  nx      ny      dt         artefact
=====  ======  ======  =========  =============================
A      96      48      0.02       round20_laminar_Re2800_96x48
A'     96      48      0.01       round22 grid ladder, tag 96x48dt01
B      144     72      0.0133     tag 144x72
C      192     96      0.01       tag 192x96
D      240     120     0.008      tag 240x120
=====  ======  ======  =========  =============================

Everything is recomputed from the stored **mean wall shear** with the certified
round-17 operator (``wall_shear_bubbles.separation_reattachment``); nothing is
copied from a payload's derived fields.  Verdicts come from
:func:`tensorfvm.verification.periodic_hill_grid_ladder.attribute`, whose rule is
fixed in the module (pre-registered before the numbers are read).

Usage
-----
    PYTHONPATH=src python scripts/round22_grid_ladder.py [--out-dir docs/case-periodic-hills]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_grid_ladder import (
    attribute,
    combined_span,
    temporal_sensitivity,
)
from tensorfvm.verification.periodic_hill_round11 import _REATTACHMENT_WINDOW
from tensorfvm.verification.wall_shear_bubbles import separation_reattachment

CASEDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
REFERENCE_X_R = 5.41

#: the ladder: tag -> artefact stem.  ``h`` is the streamwise spacing ``Lx/nx``
#: in units of the hill height (the finest spacing governs the metric, since the
#: wall-shear crossing is a streamwise interpolation).
LEVELS = [
    ("A", "round20_laminar_Re2800_96x48", 96, 48, 0.02, 20000),
    ("Ap", "round22_laminar_Re2800_96x48dt01", 96, 48, 0.01, 20000),
    ("B", "round22_laminar_Re2800_144x72", 144, 72, 0.0133, 20000),
    ("C", "round22_laminar_Re2800_192x96", 192, 96, 0.01, 20000),
    ("D", "round22_laminar_Re2800_240x120", 240, 120, 0.008, 20000),
]


def _load(stem: str):
    js = CASEDIR / f"{stem}.json"
    npz = CASEDIR / f"{stem}.npz"
    if not (js.exists() and npz.exists()):
        return None
    payload = json.loads(js.read_text())
    with np.load(npz) as z:
        tau = np.asarray(z["mean_tau_b"], dtype=float)
        x = np.asarray(z["x_over_h"], dtype=float)
        mean_u = np.asarray(z["mean_u"], dtype=float)
        yc = np.asarray(z["yc"], dtype=float)
        xc = np.asarray(z["xc"], dtype=float)
    rea = separation_reattachment(tau, x, window=_REATTACHMENT_WINDOW)
    first = separation_reattachment(tau, x, window=_REATTACHMENT_WINDOW,
                                    primary="first") if rea["bubbles"] else rea
    # independent extraction of u/U_b on the physical line y/H = 0.5 by 1-D
    # linear interpolation inside each column (the published metric uses the
    # library extractor; this is the cross-check that the extractor is not the
    # reason the profile fails).
    y_over_h = yc / 1.0
    u_y05 = np.full(x.shape, np.nan)
    for j in range(x.shape[0]):
        col_y = y_over_h[:, j]
        col_u = mean_u[:, j]
        order = np.argsort(col_y)
        cy, cu = col_y[order], col_u[order]
        if cy.min() <= 0.5 <= cy.max():
            u_y05[j] = float(np.interp(0.5, cy, cu))
    st = payload.get("stationarity") or {}
    return {
        "stem": stem,
        "payload": payload,
        "x_s_over_H": rea["x_s_over_H"],
        "x_r_over_H": rea["x_r_over_H"],
        "x_r_first_over_H": first["x_r_over_H"],
        "x_s_first_over_H": first["x_s_over_H"],
        "bubble_table": [[float(b["separation"]), float(b["reattachment"]),
                          float(b["length"]), float(b["reversed_area"])]
                         for b in rea["bubbles"]],
        "bubbles": len(rea["bubbles"]),
        "status": rea["status"],
        "wall_clock_s": payload.get("wall_clock_s"),
        "block_spread_rel": st.get("x_r_over_H_block_spread_rel"),
        "block_x_r": st.get("x_r_over_H_by_block"),
        "coverage": st.get("reattachment_coverage"),
        "snapshots_without_reattachment": st.get("snapshots_without_reattachment"),
        "profile_stations": (payload.get("gate") or {}).get("profile_stations"),
        "profile_pass": (payload.get("gate") or {}).get("profile_stations_pass"),
        "x_over_h": x,
        "mean_tau_b": tau,
        "u_y05_independent": u_y05,
        "u_y05_published": np.asarray(
            (payload.get("horizontal_profile_y_H_0.5") or {}).get("u_over_Ub", []),
            dtype=float),
        "hp_x": np.asarray(
            (payload.get("horizontal_profile_y_H_0.5") or {}).get("x_over_H", []),
            dtype=float),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(CASEDIR))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for level, stem, nx, ny, dt, steps in LEVELS:
        got = _load(stem)
        if got is None:
            print(f"[skip] {level}: {stem} not present")
            continue
        got.update(level=level, nx=nx, ny=ny, dt=dt, steps=steps,
                   h_over_H=9.0 / nx)
        runs[level] = got
        print(f"[{level}] {nx}x{ny} dt={dt:g} x_r/H={got['x_r_over_H']:.4f} "
              f"x_s/H={got['x_s_over_H']:.4f} wall={got['wall_clock_s']:.0f}s")

    # ---- spatial ladder (coarsest -> finest), constant CFL ------------------
    spatial_levels = [lv for lv in ("A", "B", "C", "D") if lv in runs]
    ladder = [(runs[lv]["h_over_H"], runs[lv]["x_r_over_H"]) for lv in spatial_levels]
    spatial = attribute(ladder, REFERENCE_X_R, tolerance=0.10)
    spatial["levels"] = spatial_levels

    # the *object-consistent* metric: the bubble that starts at the crest is the
    # one Breuer 2009's x_r/H refers to; "largest reversed area" can select a
    # different bubble, which is exactly what the closure-free run does.
    crest_levels = [lv for lv in spatial_levels
                    if runs[lv]["x_r_first_over_H"] is not None]
    crest_ladder = [(runs[lv]["h_over_H"], runs[lv]["x_r_first_over_H"])
                    for lv in crest_levels]
    spatial_crest = attribute(crest_ladder, REFERENCE_X_R, tolerance=0.10)
    spatial_crest["levels"] = crest_levels
    spatial_crest["definition"] = (
        "reattachment of the bubble that separates first downstream of the crest "
        "(primary='first'); this is the object the DNS reference x_r/H=5.41 names")

    # same for the separation point (round 17's "cleaner diagnostic")
    sep_ladder = [(runs[lv]["h_over_H"], runs[lv]["x_s_over_H"]) for lv in spatial_levels]
    spatial_sep = attribute(sep_ladder, 0.22, tolerance=0.10)
    spatial_sep["levels"] = spatial_levels

    # ---- time axis at fixed mesh -------------------------------------------
    temporal = {}
    if "A" in runs and "Ap" in runs:
        temporal["dt_half"] = temporal_sensitivity(runs["A"]["x_r_over_H"],
                                                   runs["Ap"]["x_r_over_H"])
        temporal["dt_half"]["mesh"] = [runs["A"]["nx"], runs["A"]["ny"]]
        temporal["dt_half"]["dt_coarse"] = runs["A"]["dt"]
        temporal["dt_half"]["dt_fine"] = runs["Ap"]["dt"]

    # ---- how much of the mismatch either axis can account for --------------
    span_spatial = combined_span([runs[lv]["x_r_over_H"] for lv in spatial_levels])
    span_temporal = combined_span(
        [runs[lv]["x_r_over_H"] for lv in ("A", "Ap") if lv in runs])
    need = abs(REFERENCE_X_R - runs[spatial_levels[-1]]["x_r_over_H"])
    accounting = {
        "reference": REFERENCE_X_R,
        "x_r_finest": runs[spatial_levels[-1]]["x_r_over_H"],
        "mismatch_to_close": need,
        "span_spatial": span_spatial,
        "span_temporal_dt": span_temporal,
        "fraction_explained_spatial": (span_spatial / need) if need else None,
        "fraction_explained_temporal": (span_temporal / need) if need else None,
        "grid_refinement_products": [lv for lv in spatial_levels
                                     if runs[lv]["nx"] > runs[spatial_levels[0]]["nx"]],
    }

    # ---- independent profile cross-check on the finest run -----------------
    fine = runs[spatial_levels[-1]]
    pub, ref_ind = fine["u_y05_published"], fine["u_y05_independent"]
    profile_crosscheck = {"n_published": int(pub.size), "n_independent": int(ref_ind.size)}
    if pub.size and fine["hp_x"].size == pub.size:
        # the published line is a subset of columns; compare on the same x by
        # nearest-column lookup of the independent extraction
        idx = [int(np.argmin(np.abs(fine["x_over_h"] - xv))) for xv in fine["hp_x"]]
        ind = np.array([ref_ind[i] for i in idx])
        d = np.abs(ind - pub)
        profile_crosscheck.update(
            max_abs_difference=float(np.nanmax(d)),
            mean_abs_difference=float(np.nanmean(d)),
            agrees_within_1e_3=bool(np.nanmax(d) < 1e-3))

    table = {
        "case": "CASE-FVM-PERIODIC-HILLS round 22 (discretisation attribution)",
        "reference_x_r_over_H": REFERENCE_X_R,
        "protocol": {
            "closure": "laminar (nu_t == 0)",
            "reynolds": 2800.0,
            "convection_blend": 0.75,
            "constant_cfl": True,
            "averaging_window_t": [160.0, 400.0],
            "operator": "certified round-17 separation_reattachment on stored mean wall shear",
        },
        "levels": {
            lv: {k: runs[lv][k] for k in
                 ("stem", "nx", "ny", "dt", "steps", "h_over_H", "x_s_over_H",
                  "x_r_over_H", "x_s_first_over_H", "x_r_first_over_H",
                  "bubble_table", "bubbles", "status", "wall_clock_s",
                  "block_spread_rel", "coverage", "profile_stations", "profile_pass")}
            for lv in runs
        },
        "block_x_r_over_H": {lv: runs[lv]["block_x_r"] for lv in runs},
        "attribute_x_r_over_H": spatial,
        "attribute_x_r_over_H_crest_bubble": spatial_crest,
        "attribute_x_s_over_H": spatial_sep,
        "temporal": temporal,
        "accounting": accounting,
        "profile_crosscheck": profile_crosscheck,
    }
    (out / "round22_table.json").write_text(json.dumps(table, indent=1, default=float))
    print(json.dumps({"x_r_largest_verdict": spatial["verdict"],
                      "x_r_largest_rule": spatial["rule"],
                      "x_r_crest_verdict": spatial_crest["verdict"],
                      "x_r_crest_rule": spatial_crest["rule"],
                      "x_r_crest_finest": spatial_crest["x_finest"],
                      "accounting": accounting}, indent=1, default=float))

    if not args.no_figures:
        _figures(out, runs, spatial_levels, spatial, table)
    return 0


def _figures(out, runs, spatial_levels, spatial, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 1: x_r/H (and x_s/H) vs streamwise spacing
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    hs = [runs[lv]["h_over_H"] for lv in spatial_levels]
    xr = [runs[lv]["x_r_over_H"] for lv in spatial_levels]
    xs = [runs[lv]["x_s_over_H"] for lv in spatial_levels]
    ax[0].axhline(REFERENCE_X_R, color="k", label="Breuer 2009 DNS 5.41")
    ax[0].axhspan(REFERENCE_X_R * 0.9, REFERENCE_X_R * 1.1, color="0.85",
                  label=r"$\pm10\%$ band")
    ax[0].plot(hs, xr, "o-", label="largest-area bubble $x_r/H$ (published)")
    crest = [runs[lv]["x_r_first_over_H"] for lv in spatial_levels]
    if any(v is not None for v in crest):
        ax[0].plot(hs, crest, "^--", label="crest bubble $x_r/H$ (same object as DNS)")
    if spatial.get("x_extrapolated") is not None:
        ax[0].axhline(spatial["x_extrapolated"], ls="--", color="C3",
                      label=f"Richardson $h\\to0$ = {spatial['x_extrapolated']:.3f}")
    ax[0].set_xlabel(r"$\Delta x/H = 9/n_x$")
    ax[0].set_ylabel(r"$x_r/H$")
    ax[0].set_title("reattachment vs grid spacing")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].plot(hs, xs, "s-", color="C2", label="closure-free $x_s/H$")
    ax[1].axhline(0.22, color="k", label="Breuer 2009 DNS 0.22")
    ax[1].axhspan(0.22 * 0.9, 0.22 * 1.1, color="0.85")
    ax[1].set_xlabel(r"$\Delta x/H$")
    ax[1].set_ylabel(r"$x_s/H$")
    ax[1].set_title("separation (crest) vs grid spacing")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.suptitle("CASE-FVM-PERIODIC-HILLS round 22: closure-free Re=2800, CFL fixed")
    fig.tight_layout()
    fig.savefig(out / "fig_round22_ladder.png", dpi=140)
    plt.close(fig)

    # fig 2: mean near-wall shear profiles collapsing onto x/H
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for lv in spatial_levels:
        r = runs[lv]
        ax.plot(r["x_over_h"], r["mean_tau_b"], lw=1.0,
                label=f"{r['nx']}x{r['ny']}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("x/H")
    ax.set_ylabel(r"$\langle\tau_w\rangle/\rho$ (normalised)")
    ax.set_title("time-mean bottom-wall shear, ladder overlay")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_round22_tau.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())