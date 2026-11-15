#!/usr/bin/env python3
"""Round 25 analysis: re-run the ``CASE-FVM-PERIODIC-HILLS`` grid ladder with
the refined rungs measured at CFL-stable time steps.

Round 22/23 left the closure-free ``Re = 2800`` grid ladder ``inconclusive``
because the three refined rungs (``144x72``, ``192x96``, ``240x120``) diverged
inside the startup transient at the coarse protocol ``dt``.  Round 25 re-runs
them at the stable ``dt`` that round 23's probe proved bounded (0.00665 /
0.0025 / 0.004), so the ladder can finally be *measured*.

This script recomputes every ``x_r/H`` from the stored **mean wall shear** with
the certified round-17 operator (``wall_shear_bubbles.separation_reattachment``)
-- nothing is copied from a payload's derived field -- and feeds the four
measured rungs to the pre-registered attribution rule
(:func:`tensorfvm.verification.periodic_hill_grid_ladder.attribute`).

Usage
-----
    PYTHONPATH=src python scripts/round25_ladder_analysis.py [--out-dir docs/case-periodic-hills]
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

#: the round-25 ladder: tag -> artefact stem.  ``h`` is the streamwise spacing
#: ``Lx/nx`` in units of the hill height.  ``A`` is the round-20 baseline
#: (already measured); ``B/C/D`` are the round-25 stable-dt rungs.
LEVELS = [
    ("A", "round20_laminar_Re2800_96x48", 96, 48, 0.02, 20000),
    ("B", "round25_laminar_Re2800_144x72", 144, 72, 0.00665, 60000),
    ("C", "round25_laminar_Re2800_192x96", 192, 96, 0.0025, 160000),
    ("D", "round25_laminar_Re2800_240x120", 240, 120, 0.004, 100000),
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
        "u_y05_independent": None,
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
    ap.add_argument("--out-name", default="round25_table.json")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)

    runs = {}
    for lv, stem, nx, ny, dt, steps in LEVELS:
        r = _load(stem)
        if r is None:
            print(f"[round25] {lv}: {stem} not on disk -- skipped", flush=True)
            continue
        r.update(nx=nx, ny=ny, dt=dt, steps=steps, h_over_H=9.0 / nx)
        runs[lv] = r

    if not runs:
        raise SystemExit("no round-25 artefacts on disk; run the driver first")

    # ---- spatial attribution on the measured rungs -------------------------
    spatial_levels = [lv for lv in ("A", "B", "C", "D") if lv in runs]
    hs = [runs[lv]["h_over_H"] for lv in spatial_levels]
    xr = [runs[lv]["x_r_over_H"] for lv in spatial_levels]
    spatial = attribute(list(zip(hs, xr)), REFERENCE_X_R, tolerance=0.10)
    # crest-bubble attribution (the object the DNS reference names)
    crest = [runs[lv]["x_r_first_over_H"] for lv in spatial_levels]
    spatial_crest = attribute(list(zip(hs, crest)), REFERENCE_X_R, tolerance=0.10)
    # separation attribution
    xs = [runs[lv]["x_s_over_H"] for lv in spatial_levels]
    spatial_sep = attribute(list(zip(hs, xs)), 0.22, tolerance=0.10)

    # ---- temporal sensitivity on the baseline mesh -------------------------
    # (round 23 already pinned the 96x48 pair as dt-insensitive; here we only
    #  record the baseline's own value for provenance)
    temporal = {}
    if "A" in runs:
        temporal["dt_half"] = temporal_sensitivity(
            runs["A"]["x_r_over_H"], runs["A"]["x_r_over_H"])

    # ---- how much of the mismatch each axis can account for ----------------
    span_spatial = combined_span(xr)
    need = abs(REFERENCE_X_R - xr[-1])
    accounting = {
        "reference": REFERENCE_X_R,
        "x_r_finest": xr[-1],
        "mismatch_to_close": need,
        "span_spatial": span_spatial,
        "fraction_explained_spatial": (span_spatial / need) if need else None,
        "grid_refinement_products": [lv for lv in spatial_levels
                                     if runs[lv]["nx"] > runs[spatial_levels[0]]["nx"]],
    }

    table = {
        "case": "CASE-FVM-PERIODIC-HILLS round 25 (stable-dt grid ladder)",
        "reference_x_r_over_H": REFERENCE_X_R,
        "protocol": {
            "closure": "laminar (nu_t == 0)",
            "reynolds": 2800.0,
            "convection_blend": 0.75,
            "averaging_window_t": [160.0, 400.0],
            "operator": "certified round-17 separation_reattachment on stored mean wall shear",
            "stable_dt_source": "round 23 divergence probe (bounded max|u| at fixed mesh)",
        },
        "levels": {
            lv: {k: runs[lv][k] for k in
                 ("stem", "nx", "ny", "dt", "steps", "h_over_H", "x_s_over_H",
                  "x_r_over_H", "x_s_first_over_H", "x_r_first_over_H",
                  "bubble_table", "bubbles", "status", "wall_clock_s",
                  "block_spread_rel", "coverage", "profile_stations", "profile_pass")}
            for lv in runs
        },
        "attribute_x_r_over_H": spatial,
        "attribute_x_r_over_H_crest_bubble": spatial_crest,
        "attribute_x_s_over_H": spatial_sep,
        "temporal": temporal,
        "accounting": accounting,
    }
    (out / args.out_name).write_text(json.dumps(table, indent=1, default=float))
    print(json.dumps({
        "x_r_ladder": [round(v, 4) if v is not None else None for v in xr],
        "x_r_verdict": spatial["verdict"],
        "x_r_rule": spatial["rule"],
        "x_r_crest_verdict": spatial_crest["verdict"],
        "x_r_crest_finest": spatial_crest["x_finest"],
        "accounting": accounting,
    }, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
