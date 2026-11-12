"""Reproduce CASE-FVM-NACA0012-INVISCID: inviscid lift slope of NACA 0012.

Run with::

    PYTHONPATH=src python3 scripts/naca0012_inviscid_case.py --out docs/CASE-FVM-NACA0012-INVISCID

Outputs (all programmatically generated, nothing hand-copied):

* ``results.json``  -- every number used in the report (angle scan, panel
  convergence, thickness sensitivity, method-verification sweeps)
* ``cp_alpha2.csv`` -- surface pressure distribution at alpha = 2 deg
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorfvm.panel_airfoil import (
    cl_thin_aerofoil,
    naca4_panels,
    solve_lifting_panels,
    _influence,
)

# Target lift slope of the task card: Cl = 2*pi*alpha (thin-aerofoil theory),
# alpha = 2 deg -> 0.2198 as quoted by the card; the exact 2*pi*alpha(rad) is
# 0.219325, which is the value the tables below compare against.
CARD_THIN = 0.2198
THIN_2DEG = cl_thin_aerofoil(2.0)
TOLERANCE = 0.03


def circulation(result) -> float:
    """Ring integral of the surface speed along the closed traversal (unit U, c)."""
    geo = naca4_panels(0.12, 200)
    return float(np.sum(result.surface_speed * geo["arc"] * np.sign(result.surface_speed) * 0.0))


def cylinder_cp_error(n_panels: int) -> float:
    """Method verification: max |Cp - (1 - 4 sin^2 theta)| on a unit cylinder."""
    th = np.linspace(0.0, 2.0 * math.pi, n_panels, endpoint=False)
    pts = np.column_stack([np.cos(th), np.sin(th)])
    a = pts
    b = np.roll(pts, -1, axis=0)
    t = b - a
    arc = np.linalg.norm(t, axis=1)
    t = t / arc[:, None]
    n = np.column_stack([t[:, 1], -t[:, 0]])
    geo = dict(a=a, b=b, t=t, normal=n, arc=arc, mid=0.5 * (a + b), length=float(arc.sum()))
    An, At = _influence(geo["mid"], n, t, geo)
    U = np.array([1.0, 0.0])
    sigma = np.linalg.solve(An[:, :n_panels], -(n @ U))
    vt = t @ U + At[:, :n_panels] @ sigma
    return float(np.abs(1.0 - vt ** 2 - (1.0 - 4.0 * np.sin(th) ** 2)).max())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/CASE-FVM-NACA0012-INVISCID"))
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    payload = {"case": "CASE-FVM-NACA0012-INVISCID", "method": "lifting source+vortex panel (Hess-Smith with equal-pressure Kutta)",
               "card_thin_aerofoil_cl_2deg": CARD_THIN,
               "thin_aerofoil_cl_2deg_exact": THIN_2DEG, "tolerance": TOLERANCE}

    # 1. angle scan at a converged panel count
    scan = []
    for alpha in (0.0, 1.0, 2.0, 3.0, 4.0):
        r = solve_lifting_panels(alpha, 400)
        ref = cl_thin_aerofoil(alpha)
        scan.append({"alpha_deg": alpha, "cl": r.cl_pressure, "cd": r.cd_pressure,
                     "cl_thin": ref,
                     "slope_per_deg": (r.cl_pressure / alpha) if alpha else None,
                     "error_percent_vs_thin": (100.0 * (r.cl_pressure - ref) / ref) if alpha else 0.0,
                     "kutta_residual": r.kutta_residual, "normal_residual": r.normal_residual,
                     "te_speed": float(r.surface_speed[0]), "cp_min": float(r.cp.min())})
    payload["angle_scan_panels400"] = scan

    # 2. panel convergence at alpha = 2 deg
    conv = []
    prev = None
    for npan in (25, 50, 100, 200, 400, 800, 1600):
        r = solve_lifting_panels(2.0, npan)
        conv.append({"panels_per_side": npan, "panels": r.panels, "cl": r.cl_pressure,
                     "delta_vs_previous": None if prev is None else r.cl_pressure - prev,
                     "error_percent_vs_thin": 100.0 * (r.cl_pressure - THIN_2DEG) / THIN_2DEG,
                     "te_speed": float(r.surface_speed[0])})
        prev = r.cl_pressure
    payload["panel_convergence_alpha2"] = conv

    # 3. thickness sensitivity (thickness is the only geometric knob that can
    #    move an inviscid lift slope away from 2*pi*alpha)
    thick = []
    for tc in (0.02, 0.06, 0.09, 0.12, 0.15):
        r = solve_lifting_panels(2.0, 400, thickness=tc)
        thick.append({"thickness_ratio": tc, "cl": r.cl_pressure,
                      "error_percent_vs_thin": 100.0 * (r.cl_pressure - THIN_2DEG) / THIN_2DEG})
    payload["thickness_sensitivity_alpha2"] = thick

    # 4. method verification: cylinder Cp against the analytic solution
    payload["cylinder_cp_verification"] = [
        {"panels": n, "max_abs_cp_error": cylinder_cp_error(n)} for n in (32, 64, 128, 256)]

    # 5. final case numbers
    final = solve_lifting_panels(2.0, 400)
    payload["final"] = {
        "panels": final.panels,
        "cl": final.cl_pressure,
        "cd": final.cd_pressure,
        "cl_thin_exact": THIN_2DEG,
        "cl_thin_card": CARD_THIN,
        "error_percent_vs_thin_exact": 100.0 * (final.cl_pressure - THIN_2DEG) / THIN_2DEG,
        "error_percent_vs_card": 100.0 * (final.cl_pressure - CARD_THIN) / CARD_THIN,
        "within_3_percent": bool(abs(final.cl_pressure - CARD_THIN) / CARD_THIN < TOLERANCE),
        "kutta_residual": final.kutta_residual,
        "normal_residual": final.normal_residual,
        "cp_min": float(final.cp.min()),
        "cp_min_x": float(final.x_over_c[int(np.argmin(final.cp))]),
    }

    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cp = np.column_stack([final.x_over_c, final.cp, final.surface_speed])
    np.savetxt(out / "cp_alpha2.csv", cp, delimiter=",", header="x_over_c,cp,speed_over_U",
               comments="")
    print(json.dumps(payload["final"], indent=2))
    return payload


if __name__ == "__main__":
    main()