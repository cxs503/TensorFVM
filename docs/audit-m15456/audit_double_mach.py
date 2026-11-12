#!/usr/bin/env python3
"""Independent audit recompute for m15331 CASE-DOUBLE-MACH (TensorFVM).

Three parts, all numbers written to JSON (no hand transcription):

  A. Case-definition check: rebuild the post-shock state from normal-shock
     relations for (i) M_n = 10 and (ii) M_n = 10/sin(60 deg) and compare with
     what ``tensorfvm.compressible.cases._post_shock_state`` returns.  Also
     derive the *lab-frame* post-shock velocity by an independent projection
     (front-normal / tangential decomposition), which is the textbook way the
     Woodward & Colella (1984) double-Mach initial datum is written.
  B. Reproduction of the delivered artefact: rerun the shipped case at the only
     resolution present in ``runs/double_mach_reflection/report.json``
     (120x30, t=0.2) and compare every reported number.
  C. Counterfactual: same grid, same solver wiring, but the post-shock state
     replaced by the textbook value (M_n = 10).  Isolates "wrong initial
     datum" from "scheme diffusion".

Usage:  PYTHONPATH=<m15331>/src python3 audit_double_mach.py --out <dir>
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------- part A ----
GAMMA = 1.4


def normal_shock(m1: float, rho1: float = 1.4, p1: float = 1.0) -> dict:
    """Exact normal-shock jump for gamma=1.4 (shock frame)."""
    g = GAMMA
    rho2 = rho1 * (g + 1) * m1 * m1 / ((g - 1) * m1 * m1 + 2)
    p2 = p1 * (2 * g * m1 * m1 - (g - 1)) / (g + 1)
    # velocity drop across the shock, in units of c1
    du = (2.0 / (g + 1)) * (m1 - 1.0 / m1)
    return {"rho": rho2, "p": p2, "du_over_c1": du}


def lab_velocity(m_n: float, theta_deg: float, rho1: float = 1.4, p1: float = 1.0) -> dict:
    """Lab-frame post-shock velocity for a front inclined theta to the x axis.

    The front is the line  y = tan(theta) * (x - x0 - V*t); the post-shock gas
    sits *below* it (side of the wall at y=0).  Decomposition:

      n  = (-sin th,  cos th)      unit normal pointing into the pre-shock gas
      V_n = V * sin th = M_n * c1  ->  V = M_n c1 / sin th   (front speed in x)
      upstream (lab rest) velocity seen from the front: (-V, 0)
        normal component  : M_n c1
        tangential        : (-V, 0) - (M_n c1) n = (-V + M_n c1 sin th, -M_n c1 cos th)
                            with V sin th = M_n c1  ->  tangential = (0, -M_n c1 cos th)
      downstream (front frame): normal comp = (rho1/rho2) M_n c1, tangential unchanged
      lab velocity = downstream(front) + (V, 0)
    """
    c1 = math.sqrt(GAMMA * p1 / rho1)
    th = math.radians(theta_deg)
    js = normal_shock(m_n)
    v_x = m_n * c1 / math.sin(th)
    n = (-math.sin(th), math.cos(th))
    up_n = m_n * c1
    tan_x = -v_x - up_n * n[0]          # = 0
    tan_y = 0.0 - up_n * n[1]
    dn_n = (rho1 / js["rho"]) * m_n * c1
    u = dn_n * n[0] + tan_x + v_x
    v = dn_n * n[1] + tan_y
    return {
        "front_speed_x": v_x,
        "rho": js["rho"],
        "p": js["p"],
        "u": u,
        "v": v,
        "M_n": m_n,
        "tangential_y": tan_y,
        "note": "textbook: u = 8.25 cos30 = 7.1436, v = -8.25 sin30 = -4.1250 for M_n=10",
    }


# ---------------------------------------------------------------- parts B/C --
def _wall_shock_x(rho_wall, xc):
    """First x (from the wall inflow side) where the wall row is shocked."""
    hit = (rho_wall > 4.0).nonzero()
    if hit.numel() == 0:
        return float("nan")
    return round(float(xc[int(hit[0].item())]), 4)


def run_once(case_mod, cfg, corrected: bool):
    """Build and run the case; optionally with the textbook post-shock datum."""
    if corrected:
        orig = case_mod._post_shock_state
        good = lab_velocity(10.0, 60.0)

        def patched(gamma, mach, angle_deg, rho1, p1):
            return {
                "rho": good["rho"],
                "u": good["u"],
                "v": good["v"],
                "p": good["p"],
                "shock_speed": good["front_speed_x"],
                "mn": 10.0,
            }

        case_mod._post_shock_state = patched
        try:
            solver, meta = case_mod.double_mach_reflection(cfg)
        finally:
            case_mod._post_shock_state = orig
    else:
        solver, meta = case_mod.double_mach_reflection(cfg)

    prim0 = solver.primitive()
    ic = {
        "rho_min": float(prim0["rho"].min()), "rho_max": float(prim0["rho"].max()),
        "p_min": float(prim0["p"].min()), "p_max": float(prim0["p"].max()),
        "u_min": float(prim0["u"].min()), "u_max": float(prim0["u"].max()),
        "v_min": float(prim0["v"].min()), "v_max": float(prim0["v"].max()),
    }
    integ0 = solver.integrals()
    t0 = time.time()
    solver.run(cfg.t_end)
    elapsed = time.time() - t0

    prim = solver.primitive()
    rho = prim["rho"]
    xc, yc = meta["xc"], meta["yc"]
    am = torch.argmax(rho)
    i, j = int(am // rho.shape[1]), int(am % rho.shape[1])
    rho_int = rho[:, :-1]                       # drop the prescribed top row
    am_i = torch.argmax(rho_int)
    ii, jj = int(am_i // rho_int.shape[1]), int(am_i % rho_int.shape[1])
    return {
        "nx": cfg.nx, "ny": cfg.ny, "t_end": cfg.t_end, "cfl": cfg.cfl,
        "nsteps": int(solver.nsteps), "elapsed_s": round(elapsed, 2),
        "ic": ic,
        "max_density": round(float(rho.max()), 4),
        "max_density_xy": [round(float(xc[i]), 4), round(float(yc[j]), 4)],
        "max_density_interior": round(float(rho_int.max()), 4),
        "max_density_interior_xy": [round(float(xc[ii]), 4), round(float(yc[jj]), 4)],
        "min_density": round(float(rho.min()), 6),
        "wall_row_max_density": round(float(rho[:, 0].max()), 4),
        "wall_row_mean_density": round(float(rho[:, 0].mean()), 4),
        "max_pressure": round(float(prim["p"].max()), 4),
        "positivity_fixes": int(solver.positivity_fixes),
        "drift": {k: float(v) for k, v in solver.budget_drift(integ0).items()},
        "wall_shock_x": _wall_shock_x(prim["rho"][:, 0], xc),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--ny", type=int, default=30)
    ap.add_argument("--t-end", type=float, default=0.2)
    ap.add_argument("--reported", default=None,
                    help="path to runs/double_mach_reflection/report.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    case_mod = importlib.import_module("tensorfvm.compressible.cases")

    # ---- A
    got_ic = case_mod._post_shock_state(GAMMA, 10.0, 60.0, 1.4, 1.0)
    A = {
        "code_post_shock": {k: float(v) for k, v in got_ic.items()},
        "normal_shock_M10": normal_shock(10.0),
        "normal_shock_M_10_over_sin60": normal_shock(10.0 / math.sin(math.radians(60.0))),
        "implied_normal_mach_from_code_pressure": math.sqrt(
            ((got_ic["p"] * (GAMMA + 1) / 1.0) + (GAMMA - 1)) / (2 * GAMMA)),
        "implied_normal_mach_from_code_density": math.sqrt(
            (got_ic["rho"] / 1.4 * 2) / ((GAMMA + 1) - (GAMMA - 1) * (got_ic["rho"] / 1.4))),
        "textbook_lab_velocity_M10": lab_velocity(10.0, 60.0),
        "code_velocity_error": {
            "u_rel_err_vs_textbook": abs(got_ic["u"] - lab_velocity(10.0, 60.0)["u"])
            / lab_velocity(10.0, 60.0)["u"],
            "v_sign_flipped": bool(got_ic["v"] > 0 > lab_velocity(10.0, 60.0)["v"]),
            "p_rel_err_vs_M10": abs(got_ic["p"] - normal_shock(10.0)["p"]) / normal_shock(10.0)["p"],
        },
        "top_bc_sweep_speed_used": float(got_ic["shock_speed"]),
        "top_bc_sweep_speed_note": "cases.py uses cfg.mach*c1 = 10.0 for the x-sweep of the "
                                   "top boundary, while the same file uses 11.547 as the "
                                   "front speed -> the prescribed top-BC front lags "
                                   "1.547*0.2 = 0.309 in x at t=0.2",
    }

    # ---- B / C
    cfg_shipped = case_mod.DoubleMachConfig(nx=args.nx, ny=args.ny, t_end=args.t_end)
    B = run_once(case_mod, cfg_shipped, corrected=False)
    C = run_once(case_mod, cfg_shipped, corrected=True)

    rep = {}
    if args.reported and Path(args.reported).exists():
        reported = json.loads(Path(args.reported).read_text())
        r0 = reported["results"][0]
        rep = {
            "reported": {k: r0.get(k) for k in
                         ("nx", "ny", "nsteps", "max_density", "mach_stem_height",
                          "triple_point_x", "min_rho")},
            "reproduced": {k: B.get(k) for k in ("nx", "ny", "nsteps", "max_density",
                                                 "min_density")},
            "delta_max_density": round(B["max_density"] - float(r0["max_density"]), 4),
            "delta_min_density": round(B["min_density"] - float(r0["min_rho"]), 6),
        }

    payload = {"A_case_definition": A, "B_shipped_ic": B, "C_textbook_ic": C,
               "B_vs_reported": rep}
    (out / "audit_double_mach.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())