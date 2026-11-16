#!/usr/bin/env python3
"""Round 32: complete the Re = 700 ladder and grade both acceptance metrics.

Why this round exists
---------------------
Round 31 opened the Re = 700 (laminar, 2-D-admissible) grid-convergence ladder
with a single measured rung (96x48, ``x_r/H = 8.08``, +54 % against the DNS
reference 5.24) and two ``ABSENT`` rungs (144x72, 192x96).  This round closes
the ladder with the two missing rungs (cluster jobs cj4666 / cj4667, same
protocol as the round-20 driver: laminar closure, ``dt = 0.02``, 20000 steps,
burn-in 8000, averaging window ``t in [160, 400]``) and, for the first time,
grades the *second* acceptance metric -- the mean velocity profile at
``y/H = 0.5`` -- at Re = 700, against the primary DNS profile files fetched in
this round (``tests/data/periodic_hill/ercoftac_ufr330_re700``, case 1 of the
ERCOFTAC KB, LESOCC Re = 700).

The attribution verdict comes from the **pre-registered** decision rule
(:func:`tensorfvm.verification.periodic_hill_grid_ladder.attribute`, fixed in
round 22 before any rung was read): the ladder is passed coarsest-first with
``h = dx = L/nx`` (``L = 9H``), so the refinement ratio is exactly 1.5 at both
steps.

Zero-compute: every number is recomputed from the stored artefacts with the
certified round-17 wall-shear operator; nothing is re-simulated.

Usage
-----
    PYTHONPATH=src python3 scripts/round32_re700_ladder_completion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorfvm.verification.periodic_hill_grid_ladder import attribute  # noqa: E402
from tensorfvm.verification.periodic_hill_re700_reference import (  # noqa: E402
    X_R_OVER_H,
    X_R_TOLERANCE,
    profile_at,
    relative_band_ok,
    station_covers_y05,
    STATIONS,
)
from tensorfvm.verification.wall_shear_bubbles import (  # noqa: E402
    separation_reattachment,
)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
RUNGS = (("96x48", 96, 48), ("144x72", 144, 72), ("192x96", 192, 96))
WINDOW = (1.9, 7.1)  # same certified window as round 17/20/31
DOMAIN_L_OVER_H = 9.0  # periodic hill domain length (periodic_hill.py)


def _stem(tag: str) -> str:
    return f"round20_laminar_Re700_{tag}"


def _score(tag: str) -> dict:
    """Score one stored artefact with the certified operator (or report absent)."""
    npz_path = OUTDIR / f"{_stem(tag)}.npz"
    json_path = OUTDIR / f"{_stem(tag)}.json"
    row = {"tag": tag, "present": npz_path.is_file()}
    if not row["present"]:
        row.update(x_r_over_H=None, x_s_over_H=None, status="ABSENT",
                   n_bubbles=None, error=None, below_10=None, below_3=None)
        return row
    with np.load(npz_path) as npz:
        tau = np.asarray(npz["mean_tau_b"], dtype=float)
        x = np.asarray(npz["x_over_h"], dtype=float)
        u = np.asarray(npz["mean_u"], dtype=float)
        yc = np.asarray(npz["yc"], dtype=float)
        if yc.ndim == 2:
            yc = yc[:, 0]
    scored = separation_reattachment(tau, x, window=WINDOW)
    x_r = scored["x_r_over_H"]
    err = (abs(x_r - X_R_OVER_H) / abs(X_R_OVER_H)
           if np.isfinite(x_r) else None)
    row.update(
        x_r_over_H=None if not np.isfinite(x_r) else float(x_r),
        x_s_over_H=None if not np.isfinite(scored["x_s_over_H"])
        else float(scored["x_s_over_H"]),
        status=scored["status"],
        n_bubbles=int(scored["n_bubbles"]),
        all_crossings=[float(c["x_over_H"]) for c in scored["all_crossings"]],
        reference_x_r_over_H=X_R_OVER_H,
        error=None if err is None else float(err),
        below_10=None if err is None else bool(err <= 0.10),
        below_3=None if err is None else bool(err <= 0.03),
    )
    # y/H = 0.5 mean-velocity profile vs the primary Re = 700 DNS files
    prof = []
    if u.size and np.all(np.isfinite(u)):
        row_idx = int(np.argmin(np.abs(yc - 0.5)))
        u_row = u[row_idx]
        for xs in STATIONS:
            if not station_covers_y05(xs):
                continue
            idx = int(np.argmin(np.abs(x - xs)))
            u_c = float(u_row[idx])
            chk = relative_band_ok(u_c, xs, 0.5)
            prof.append({"x_over_H": float(xs), **chk})
    row["profile_stations"] = prof
    row["profile_pass_count"] = sum(1 for p in prof if p["pass"])
    # cross-checks against the payload (never preferred)
    if json_path.is_file():
        payload = json.loads(json_path.read_text())
        row["payload_x_r_over_H"] = ((payload.get("time_averaged") or {})
                                     .get("x_r_over_H"))
        row["stationarity_ok"] = (payload.get("stationarity", {})
                                  .get("x_r_over_H", {}).get("ok"))
        row["wall_clock_s"] = payload.get("wall_clock_s")
        row["samples"] = payload.get("samples")
    return row


def main() -> int:
    rows = [_score(tag) for tag, _, _ in RUNGS]
    present = [r for r in rows if r["present"] and r["x_r_over_H"] is not None]
    # pre-registered attribution on the measured rungs (coarsest first)
    ladder = [(DOMAIN_L_OVER_H / nx, r["x_r_over_H"])
              for (tag, nx, _), r in zip(RUNGS, rows)
              if r["present"] and r["x_r_over_H"] is not None]
    attribution = attribute(ladder, X_R_OVER_H, tolerance=X_R_TOLERANCE) \
        if len(ladder) >= 2 else None
    out = {
        "round": 32,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "purpose": ("Re=700 grid ladder completed (144x72, 192x96) + first "
                    "y/H=0.5 profile grading at Re=700 vs primary DNS; "
                    "pre-registered discretisation attribution"),
        "reynolds": 700.0,
        "reference": {
            "x_r_over_H": X_R_OVER_H,
            "tolerance": X_R_TOLERANCE,
            "band": [X_R_OVER_H * (1 - X_R_TOLERANCE),
                     X_R_OVER_H * (1 + X_R_TOLERANCE)],
            "source": "ERCOFTAC KB UFR 3-30 Evaluation (x_r) + case-1 DNS "
                      "profile files (u/U_b at y/H=0.5), Breuer et al. 2009",
        },
        "protocol": {
            "closure": "laminar (nu_t == 0)",
            "dt": 0.02,
            "steps": 20000,
            "burn_in_steps": 8000,
            "averaging_window_t": [160.0, 400.0],
            "operator": "certified wall_shear_bubbles.separation_reattachment "
                        "on stored mean wall shear",
            "jobs": {"144x72": "cj4666", "192x96": "cj4667"},
        },
        "rows": rows,
        "n_measured": len(present),
        "attribution": attribution,
    }
    (OUTDIR / "round32_re700_ladder_completion.json").write_text(
        json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps({k: r.get(k) for r in rows
                      for k in ("tag", "present", "x_r_over_H", "error",
                                "below_10", "status", "profile_pass_count")},
                     indent=1))
    if attribution:
        print(json.dumps({k: attribution[k] for k in
                          ("verdict", "rule", "order", "x_extrapolated",
                           "gci_fine", "movement_rel",
                           "error_finest_rel", "error_extrapolated_rel")},
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
