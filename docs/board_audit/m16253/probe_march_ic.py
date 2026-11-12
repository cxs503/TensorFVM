#!/usr/bin/env python3
"""Audit probe for CASE-FVM-BLASIUS-PLATE (m16219): is the 0.71% agreement
independent of the initial condition?

The delivered case initialises the laminar thin-shear-layer march with the
*exact Blasius profile* at the first station (``boundary_layer.py:234-240``).
For a zero-pressure-gradient plate the Blasius profile is the exact solution at
every station, so a march started from it can only drift by discretisation
error.  That makes the reported -0.71% a preservation (manufactured-solution
style) check rather than a prediction.

This probe re-runs the same device with deliberately non-Blasius initial
profiles and asks whether it still lands on the Blasius values at Re_x=1e5:

  control   : seed=None            (delivered configuration, Blasius IC)
  linear    : u = ue * y/Y         (crude ramp, no similarity shape)
  uniform   : u = ue everywhere    (no boundary layer at all at x0)

Usage: PYTHONPATH=<repo>/src python probe_march_ic.py --repo <repo> --out <dir>
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
    ap.add_argument("--repo", required=True, help="delivery worktree (m16219)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "src"))
    from tensorfvm.case_blasius_plate import REFERENCE_RE_X, blasius_reference
    from tensorfvm.rans_ship_resistance.boundary_layer import (
        march_boundary_layer, _stretch)

    stations, length, nu, ny, stretch = 401, 1.0, 1e-5, 256, 8.0
    x = np.linspace(length / stations, length, stations)
    ue = np.ones_like(x)
    y0 = 5.0 * math.sqrt(nu * x[0] / ue[0])
    span = max(x[-1] - x[0], 1e-12)
    re_span = max(float(np.mean(ue)) * span / nu, 1.0)
    y_estimate = 0.37 * span * re_span ** -0.2
    Y0 = max(2.5 * y_estimate, 2.5 * y0)
    ygrid = Y0 * _stretch(ny, stretch)

    seeds = {
        "control": None,
        "linear": (ygrid, np.clip(ue[0] * ygrid / Y0, 0.0, 1.0)),
        "uniform": (ygrid, np.full(ny, ue[0])),
    }
    ref = blasius_reference(REFERENCE_RE_X)
    rows = []
    for tag, seed in seeds.items():
        result = march_boundary_layer(x, ue, nu, model="laminar", trip_x=None,
                                      ny=ny, stretching=stretch, seed=seed)
        cf = float(np.asarray(result.cf)[-1])
        H = float(np.asarray(result.H)[-1])
        rows.append({
            "tag": tag,
            "cf": cf,
            "H": H,
            "cf_err": (cf - ref["cf"]) / ref["cf"],
            "H_err": (H - ref["H"]) / ref["H"],
            "delta_star": float(np.asarray(result.delta_star)[-1]),
            "theta": float(np.asarray(result.theta)[-1]),
            "continuity_residual": float(result.continuity_residual),
            "passed_3pct": bool(abs((cf - ref["cf"]) / ref["cf"]) < 0.03
                                and abs((H - ref["H"]) / ref["H"]) < 0.03),
        })
        print(f"{tag:8s} cf={cf:.8f} ({rows[-1]['cf_err']:+.2%}) "
              f"H={H:.5f} ({rows[-1]['H_err']:+.2%}) "
              f"cont={rows[-1]['continuity_residual']:.2e}", flush=True)

    payload = {"case": "CASE-FVM-BLASIUS-PLATE", "kind": "ic_independence_probe",
               "reference": ref, "config": {"stations": stations, "ny": ny,
                                            "length": length, "nu": nu},
               "rows": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "march_ic_probe.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())