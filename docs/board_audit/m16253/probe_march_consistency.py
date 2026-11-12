#!/usr/bin/env python3
"""Audit probe 2 for CASE-FVM-BLASIUS-PLATE (m16219).

Two independent re-derivations of the delivered numbers, plus a robustness
ladder for the initial condition:

A. Momentum-integral consistency.  For zero pressure gradient the von Karman
   integral gives c_f/2 = d(theta)/dx.  The delivered c_f comes from the wall
   gradient (boundary_layer.py:370-372).  Comparing the two checks whether the
   reported c_f is consistent with the *profiles* rather than only with the
   wall node.

B. Initial-condition ladder.  Probe 1 (probe_march_ic.py) showed that a
   whole-domain linear ramp at x0 breaks the march (c_f -90%, continuity
   residual 0.97) while a uniform start converges (-1.02%).  This ladder varies
   the thickness of the startup layer (k * delta_99 at x0, k = 1, 3, 10) to
   separate "shape sensitivity" from "domain-height policy" (the march never
   shrinks its domain, boundary_layer.py:328-334).

Usage: PYTHONPATH=<repo>/src python probe_march_consistency.py --repo <r> --out <d>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "src"))
    from tensorfvm.case_blasius_plate import REFERENCE_RE_X, blasius_reference
    from tensorfvm.rans_ship_resistance.boundary_layer import (
        march_boundary_layer, _stretch)

    stations, length, nu, ny, stretch = 401, 1.0, 1e-5, 256, 8.0
    x = np.linspace(length / stations, length, stations)
    ue = np.ones_like(x)
    ref = blasius_reference(REFERENCE_RE_X)
    payload = {"case": "CASE-FVM-BLASIUS-PLATE", "kind": "momentum_integral_and_ic_ladder",
               "reference": ref}

    # --- A. delivered configuration: wall-shear c_f vs momentum integral -----
    res = march_boundary_layer(x, ue, nu, model="laminar", trip_x=None,
                               ny=ny, stretching=stretch)
    cf = np.asarray(res.cf)
    theta = np.asarray(res.theta)
    contact = {}
    for span in (2, 5, 10):
        dx = float(x[-1] - x[-1 - span])
        contact[f"d_theta_dx_{span}"] = float((theta[-1] - theta[-1 - span]) / dx)
    payload["momentum_integral"] = {
        "cf_wall_shear": float(cf[-1]),
        "two_dtheta_dx": {k: 2.0 * v for k, v in contact.items()},
        "relative_difference": {k: (2.0 * v - cf[-1]) / cf[-1]
                                for k, v in contact.items()},
        "cf_reference": ref["cf"],
        "cf_wall_shear_error": (cf[-1] - ref["cf"]) / ref["cf"],
    }

    # --- B. startup-layer thickness ladder ----------------------------------
    y0 = 5.0 * math.sqrt(nu * x[0] / ue[0])
    span = max(x[-1] - x[0], 1e-12)
    re_span = max(float(np.mean(ue)) * span / nu, 1.0)
    Y0 = max(2.5 * 0.37 * span * re_span ** -0.2, 2.5 * y0)
    ygrid = Y0 * _stretch(ny, stretch)
    rows = []
    for k in (1.0, 3.0, 10.0):
        thick = k * y0
        seed_u = np.clip(ue[0] * ygrid / thick, 0.0, 1.0)
        seed_u = ue[0] * seed_u ** (1.0 / 7.0)   # 1/7-power ramp, not similarity
        r = march_boundary_layer(x, ue, nu, model="laminar", trip_x=None, ny=ny,
                                 stretching=stretch, seed=(ygrid, seed_u))
        cf_k = float(np.asarray(r.cf)[-1])
        H_k = float(np.asarray(r.H)[-1])
        rows.append({"k_delta99": k, "thickness_at_x0": thick, "cf": cf_k, "H": H_k,
                     "cf_err": (cf_k - ref["cf"]) / ref["cf"],
                     "H_err": (H_k - ref["H"]) / ref["H"],
                     "continuity_residual": float(r.continuity_residual),
                     "passed_3pct": bool(abs((cf_k - ref["cf"]) / ref["cf"]) < 0.03
                                         and abs((H_k - ref["H"]) / ref["H"]) < 0.03)})
        print(f"k={k:5.1f} thick={thick:.5f} cf_err={rows[-1]['cf_err']:+.2%} "
              f"H_err={rows[-1]['H_err']:+.2%} cont={rows[-1]['continuity_residual']:.2e}",
              flush=True)
    payload["ic_ladder"] = {"delta99_at_x0": y0, "domain_height": Y0, "rows": rows}

    print("momentum integral:", json.dumps(payload["momentum_integral"]["relative_difference"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "momentum_integral_probe.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())