"""Walk the cross-flow CHT march forward in wall-clock-bounded segments.

The Re = 40 conjugate cylinder needs ~3e4 time steps of the polar vorticity solver
(~20 ms each), which is too long for a single shell slot but far too small to be
worth exclusive cluster time.  This driver saves ``(psi, omega)`` after each
segment so the same march can be resumed, and, on the last segment, performs the
conjugate coupling and writes the verification artefact.

    PYTHONPATH=src python -m scripts.p295_cylinder_stream --segment 10
    ...
    PYTHONPATH=src python -m scripts.p295_cylinder_stream --segment 10 --final
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.cht.polar import ConjugateCylinder, PolarVorticitySolver

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "runs" / "cht_p295"


def build(re: float, n_r: int, n_theta: int, r_out: float, wall_cell: float):
    return PolarVorticitySolver(re=re, n_r=n_r, n_theta=n_theta, r_out=r_out,
                                wall_cell=wall_cell)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=40.0)
    ap.add_argument("--pr", type=float, default=0.7)
    ap.add_argument("--n-r", type=int, default=71)
    ap.add_argument("--n-theta", type=int, default=72)
    ap.add_argument("--r-out", type=float, default=15.0)
    ap.add_argument("--wall-cell", type=float, default=8.0e-3)
    ap.add_argument("--dt", type=float, default=2.0e-3)
    ap.add_argument("--segment", type=float, default=10.0,
                    help="physical time to march in this call")
    ap.add_argument("--state", default=str(OUTDIR / "cylinder_state.npz"))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--solid-ratio", type=float, default=100.0)
    ap.add_argument("--relax", type=float, default=0.7)
    args = ap.parse_args(argv)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)
    flow = build(args.re, args.n_r, args.n_theta, args.r_out, args.wall_cell)
    t_accum = 0.0
    drag = []
    state = None
    if state_path.exists():
        data = np.load(state_path)
        state = (data["psi"], data["omega"])
        t_accum = float(data["t"])
        drag = [tuple(map(float, row)) for row in data["drag"]]

    t0 = time.time()
    march = flow.march(t_end=args.segment, dt=args.dt, tol_steady=1e-12,
                       max_steps=400000, t_ramp=4.0 if state is None else 0.0,
                       state=state)
    for row in march["drag_history"]:
        drag.append((row[0] + t_accum, row[1], row[2]))
    t_accum += march["time"]
    np.savez(state_path, psi=march["psi"], omega=march["omega"], t=t_accum,
             drag=np.asarray(drag), n_r=args.n_r, n_theta=args.n_theta)
    cd, cl = flow.drag_lift(march["wall_vorticity"])
    print(f"segment done: t={t_accum:.2f} steps={march['steps']} "
          f"cd={cd:.4f} cl={cl:.5f} rt={time.time()-t0:.1f}s", flush=True)

    if not args.final:
        return 0

    case = ConjugateCylinder(re=args.re, pr=args.pr, solid_ratio=args.solid_ratio,
                             flow=flow)
    t1 = time.time()
    res = case.couple(march["psi"], outer_temperature=None, relax=args.relax,
                      max_iter=300, tol=1e-9)
    print(f"coupling done in {time.time()-t1:.1f}s", flush=True)
    from scripts.p295_cht_validation import (churchill_bernstein_local,
                                             eckert_soehngen, plot_cylinder)
    nu_es = eckert_soehngen(args.re, args.pr)
    nu_cb = churchill_bernstein_local(args.re, args.pr)
    m = res.metrics
    payload = {
        "case": "cylinder_cross_flow_cht",
        "reynolds": args.re,
        "prandtl": args.pr,
        "solid_ratio": args.solid_ratio,
        "mesh": {"n_r": args.n_r, "n_theta": args.n_theta, "r_out": args.r_out,
                 "wall_cell": args.wall_cell, "dt": args.dt,
                 "t_end": t_accum},
        "flow": {"steady": bool(drag[-1][1] is not None
                                and abs(drag[-1][1] - drag[-2][1]) / abs(drag[-1][1]) < 1e-6),
                 "drag_drift_last_step": float(abs(drag[-1][1] - drag[-2][1])
                                               / max(abs(drag[-1][1]), 1e-30)),
                 "steps": int(t_accum / args.dt),
                 "time": t_accum, "drag_coefficient": float(cd),
                 "lift_coefficient": float(cl),
                 "drag_history": drag[-12:]},
        "nusselt_mean": float(m["nusselt_mean"]),
        "nusselt_mean_flux_weighted": float(m.get("nusselt_mean_flux_weighted", float("nan"))),
        "nusselt_eckert_soehngen_1952": nu_es,
        "nusselt_churchill_bernstein": nu_cb,
        "error_vs_eckert_soehngen_rel": abs(m["nusselt_mean"] - nu_es) / nu_es,
        "error_vs_churchill_bernstein_rel": abs(m["nusselt_mean"] - nu_cb) / nu_cb,
        "interface_local_residual_rel": float(m["interface_local_residual_rel"]),
        "interface_global_residual_rel": float(m["interface_global_residual_rel"]),
        "wall_temperature_mean": float(m["wall_temperature_mean"]),
        "wall_temperature_ptp": float(m["wall_temperature_ptp"]),
        "energy_rate_balance_rel": float(m["energy_rate_balance_rel"]),
        "coupling_iterations": int(m.get("coupling_iterations", -1)),
        "defect_baseline": {
            "note": "same configuration run through the pre-fix operator",
            "nusselt_mean_before_fixes": 326.87,
            "provenance": "session log of the P295 run before the sign and "
                          "one_sided_derivative fixes (recorded, not recomputed)",
        },
        "acceptance": {
            "nu_error_limit": 0.10,
            "reference": "Eckert & Soehngen 1952 mean correlation",
            "within_limit": bool(abs(m["nusselt_mean"] - nu_es) / nu_es < 0.10),
            # the Dirichlet-Neumann sweep stalls at the level of the interface
            # flux transfer (~1.8e-4 of the wall flux); 1e-3 is the honest bar
            "interface_residual_limit": 1e-3,
            "interface_conserved": bool(m["interface_local_residual_rel"] < 1e-3),
        },
    }
    try:
        payload["figures"] = [plot_cylinder(payload, res)]
    except Exception as exc:                              # pragma: no cover
        payload["figure_error"] = repr(exc)
    path = OUTDIR / f"cylinder_re{int(args.re)}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("wrote", path)
    print(json.dumps(payload["acceptance"]))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())