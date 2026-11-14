#!/usr/bin/env python3
"""Independent re-measurement of the CASE-FVM-CHANNEL-DEVELOP entrance length.

Audit of mission m20007 (TensorFVM).  Does **not** import
``tensorfvm.benchmark_entrance_length``; it only reuses the repo kernel
``tensorfvm.inc_bench.solver.SteadyFlow`` that the audited module also reuses,
and re-implements the measurement so the criterion defect can be probed:

  * ``ny`` is free (the audited module ties ny to nx/H, so only even ny >= 8 is
    reachable there); with odd ny the wall-normal centre line y = H/2 is a cell
    centre, with even ny it never is.
  * two criteria are reported side by side
      - ``even_sample``  : audited criterion, u at the (upper) centre cell centre
      - ``midplane``     : linear interpolation of u to exactly y = H/2
  * ``u_max`` is reported both at cell centres and at y = H/2 so that the
    discrete ceiling of the 99 % threshold (1.485 u_bar) is visible.

Decision recorded by this experiment (see docs/audit_m20007/AUDIT-*.md):
  H_A "coarse-grid numerical diffusion is the cause of the 165 % error"
  H_B "even-ny cell-centre sampling makes the 99 % threshold unreachable, i.e.
       the reported 165 % is a measurement artefact"
Predeclared discriminator: at Re_H = 100 (ref L_e/H = 0.06*Re_H = 6.0) with
L/H = 16, if the mid-plane measurement at ny = 32 falls within 8 % of 6.0 the
error is measurement/resolution driven (H_B); if it stays > 8 % the solver's
effective viscosity is implicated too (H_A contributes).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorfvm.inc_bench.solver import SolverOptions, SteadyFlow

U_MAX_FACTOR = 1.5
U_CENTER_THRESHOLD = 0.99
THRESHOLD = U_CENTER_THRESHOLD * U_MAX_FACTOR


def measure(ny: int, nx: int, re_h: float, aspect: float, ubar: float = 1.0,
            height: float = 1.0, density: float = 1.0) -> dict:
    nu = ubar * height / re_h
    dx, dy = aspect * height / nx, height / ny
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    inlet = torch.full((ny,), ubar, dtype=torch.float64)
    opts = SolverOptions(max_iterations=60000, tolerance=1.0e-8,
                         alpha_u=0.6, alpha_p=0.25)
    solver = SteadyFlow(fluid, dx=dx, dy=dy, viscosity=nu, density=density,
                        inlet_profile=inlet, outlet=True, options=opts)
    t0 = time.perf_counter()
    record = solver.solve()
    elapsed = time.perf_counter() - t0

    # cell-centre u (same interpolation as the audited module)
    cell_u = 0.5 * (solver.u[:, :-1] + solver.u[:, 1:])
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    x_over_h = (xc / height).tolist()

    row_hi = ny // 2                      # upper of the two centre cells (even ny)
    u_even = (cell_u[row_hi, :] / ubar).tolist()
    # mid-plane: interpolate to y = H/2 between the two rows straddling it
    y_rows = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    lo = max(r for r in range(ny) if float(y_rows[r]) <= height / 2)
    hi = min(r for r in range(ny) if float(y_rows[r]) >= height / 2)
    w = 0.0 if hi == lo else float((height / 2 - y_rows[lo]) / (y_rows[hi] - y_rows[lo]))
    u_mid = (cell_u[lo, :] * (1 - w) + cell_u[hi, :] * w).div(ubar).tolist()

    def first_cross(profile):
        for i, val in enumerate(profile):
            if val >= THRESHOLD:
                if i == 0:
                    return x_over_h[0], True
                x0, u0 = x_over_h[i - 1], profile[i - 1]
                frac = (THRESHOLD - u0) / (val - u0) if val != u0 else 0.0
                return x0 + frac * (x_over_h[i] - x0), True
        return x_over_h[-1], False        # fallback = domain end (audited behaviour)

    le_even, ok_even = first_cross(u_even)
    le_mid, ok_mid = first_cross(u_mid)

    le_ref = 0.06 * re_h
    return {
        "ny": ny, "nx": nx, "re_h": re_h, "L_over_H": aspect, "nu": nu,
        "dx": dx, "dy": dy,
        "centre_cell_y_over_H": float(y_rows[row_hi]) / height,
        "u_center_even_max": max(u_even),
        "u_center_midplane_max": max(u_mid),
        "threshold": THRESHOLD,
        "threshold_reachable_even": max(u_even) >= THRESHOLD,
        "threshold_reachable_midplane": max(u_mid) >= THRESHOLD,
        "Le_over_H_even_sample": le_even, "Le_crossed_even": ok_even,
        "Le_over_H_midplane": le_mid, "Le_crossed_midplane": ok_mid,
        "Le_over_H_ref(0.06 Re_H)": le_ref,
        "err_pct_even": 100.0 * abs(le_even - le_ref) / le_ref,
        "err_pct_midplane": 100.0 * abs(le_mid - le_ref) / le_ref,
        "iterations": record.get("iterations", 0),
        "converged": record.get("converged", False),
        "elapsed_s": round(elapsed, 1),
        "re_h_eff_midplane": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--re-h", type=float, default=100.0)
    ap.add_argument("--aspect", type=float, default=16.0)
    ap.add_argument("--out", type=str, default="/tmp/m20007_audit")
    a = ap.parse_args()
    r = measure(a.ny, a.nx, a.re_h, a.aspect)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"recheck_ny{a.ny}_nx{a.nx}_Re{a.re_h:g}.json"
    f.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()