#!/usr/bin/env python3
"""Run the Chebyshev-collocation reference ladder for the lid-driven cavity.

Produces one JSON per polynomial order plus a summary, all under
``docs/CASE-FVM-CAVITY-RE1000/runs/spectral/``.  The numbers in
``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL.md`` are injected from these files, never
transcribed by hand.

Example
-------
PYTHONPATH=src python3 scripts/run_cavity_spectral_ref.py --orders 16,24,32,48 \
        --reynolds 1000 --tag re1000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm import lid_cavity_reference as ref
from tensorfvm.cavity_spectral import SpectralCavity

DEFAULT_OUT = Path("docs/CASE-FVM-CAVITY-RE1000/runs/spectral")


def march(solver: SpectralCavity, target: float, *, psi=None, omega=None,
          delta0: float | None = None, tol: float = 1e-9, max_iter: int = 60,
          log: list | None = None):
    """Drive Newton to ``target`` with an adaptive Reynolds-number march.

    The returned solution is always a converged Newton solution *at* ``target``;
    the march only supplies initial guesses.  Whenever a step fails the
    increment is halved, so under-resolved orders degrade to smaller steps
    instead of aborting the whole ladder.
    """
    log = [] if log is None else log
    delta = float(delta0 if delta0 is not None else max(target / 5.0, 5.0))
    if psi is not None:
        try:
            sol = solver.solve(target, psi=psi, omega=omega, tol=tol, max_iter=max_iter)
            log.append({"reynolds": target, "mode": "warm-start Newton",
                        "iterations": sol.newton["iterations"]})
            return sol
        except RuntimeError as exc:
            log.append({"reynolds": target, "mode": "warm-start failed",
                        "error": str(exc)})

    # cold start: get a first converged solution at a low Reynolds number
    re_now, sol = None, None
    for re0 in (10.0, 2.0, 0.5):
        try:
            sol = solver.solve(re0, tol=tol, max_iter=max_iter)
            re_now = re0
            log.append({"reynolds": re0, "mode": "cold start",
                        "iterations": sol.newton["iterations"]})
            break
        except RuntimeError as exc:
            log.append({"reynolds": re0, "mode": "cold start failed",
                        "error": str(exc)})
    if sol is None:
        raise RuntimeError("cold start failed at every trial Reynolds number")

    attempts = 0
    while re_now < target - 1e-12:
        re_try = min(target, re_now + delta)
        try:
            sol = solver.solve(re_try, psi=sol.psi, omega=sol.omega, tol=tol,
                               max_iter=max_iter)
            log.append({"reynolds": re_try, "mode": "march step",
                        "iterations": sol.newton["iterations"]})
            re_now = re_try
            delta *= 1.5
        except RuntimeError as exc:
            delta *= 0.5
            attempts += 1
            log.append({"reynolds": re_try, "mode": "march step failed",
                        "error": str(exc), "new_delta": delta})
            if delta < 1e-2 or attempts > 14:
                raise RuntimeError(
                    f"Reynolds march stalled at Re={re_now} towards {target}"
                ) from exc
    return sol


def score_against_ghia(sol) -> dict:
    """Compare a solution with the transcribed Ghia (1982) tables."""
    re = sol.reynolds
    out: dict = {}
    if re in ref.U_CENTRELINE:
        tbl = np.array(ref.U_CENTRELINE[re], dtype=float)
        u = sol.centreline_u(tbl[:, 0])
        out["u_profile_max_abs"] = float(np.max(np.abs(u - tbl[:, 1])))
        out["u_profile_rms"] = float(np.sqrt(np.mean((u - tbl[:, 1]) ** 2)))
        tblv = np.array(ref.V_CENTRELINE[re], dtype=float)
        v = sol.centreline_v(tblv[:, 0])
        out["v_profile_max_abs"] = float(np.max(np.abs(v - tblv[:, 1])))
        out["v_profile_rms"] = float(np.sqrt(np.mean((v - tblv[:, 1]) ** 2)))
        u_ref = float(tbl[1:-1, 1].min())
        v_ref = float(tblv[1:-1, 1].max())
        out["u_min_ref"] = u_ref
        out["v_max_ref"] = v_ref
        out["u_min_rel_err"] = float(abs(sol.metrics["u_min"] - u_ref) / abs(u_ref))
        out["v_max_rel_err"] = float(abs(sol.metrics["v_max"] - v_ref) / abs(v_ref))
        out["within_5pct"] = bool(out["u_min_rel_err"] < 0.05
                                  and out["v_max_rel_err"] < 0.05)
        out["ghia_grid"] = "129x129 vorticity-streamfunction (Ghia et al. 1982)"
    else:
        out["note"] = f"no Ghia table for Re={re}"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", default="16,24,32,48,64",
                    help="comma separated polynomial orders N (even)")
    ap.add_argument("--reynolds", type=float, default=1000.0)
    ap.add_argument("--tag", default=None, help="output file stem suffix")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--sample", type=int, default=201,
                    help="number of centreline samples stored per profile")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"re{int(round(args.reynolds))}"
    orders = [int(x) for x in args.orders.split(",") if x.strip()]

    records, prev = [], None
    for n in orders:
        solver = SpectralCavity(n)
        t0 = time.time()
        psi = omega = None
        if prev is not None:
            psi, omega = solver.remap(prev["psi"], prev["omega"], prev["xi"])
        log: list = []
        try:
            sol = march(solver, args.reynolds, psi=psi, omega=omega, tol=args.tol,
                        max_iter=args.max_iter, log=log)
        except RuntimeError as exc:
            # an under-resolved order must not abort the ladder: record it as a
            # control row (the same convention as the FVM n = 32 control run)
            rec = {"case": "CASE-FVM-CAVITY-RE1000", "reynolds": args.reynolds,
                   "order": n, "nodes_per_side": n + 1,
                   "status": "newton_diverged", "error": str(exc),
                   "wall_seconds": time.time() - t0, "march": log}
            (outdir / f"cavity_spectral_N{n:03d}_{tag}.json").write_text(
                json.dumps(rec, indent=1, sort_keys=True))
            print(f"N={n:3d} diverged ({exc}) -> control row", flush=True)
            records.append({"order": n, "nodes_per_side": n + 1,
                            "status": "newton_diverged", "metrics": None,
                            "ghia": None, "wall_seconds": rec["wall_seconds"]})
            continue
        dt = time.time() - t0
        y = np.linspace(0.0, 1.0, args.sample)
        rec = {
            "case": "CASE-FVM-CAVITY-RE1000",
            "reynolds": sol.reynolds,
            "order": n,
            "nodes_per_side": n + 1,
            "method": {
                "formulation": "streamfunction-vorticity",
                "discretisation": "Chebyshev collocation (Chebyshev-Gauss-Lobatto)",
                "steady_solver": "Newton, dense Jacobian with row/column equilibration",
                "wall_vorticity": "spectral ghost-slope closure (Hermite degree N)",
                "lid_velocity": sol.lid_velocity,
            },
            "metrics": sol.metrics,
            "newton": {"iterations_at_target": sol.newton["iterations"],
                       "march": log},
            "wall_seconds": dt,
            "ghia": score_against_ghia(sol),
            "profiles": {
                "u_centreline": {"x": 0.5, "y": y.tolist(),
                                 "u": sol.centreline_u(y).tolist()},
                "v_centreline": {"y": 0.5, "x": y.tolist(),
                                 "v": sol.centreline_v(y).tolist()},
            },
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__},
        }
        path = outdir / f"cavity_spectral_N{n:03d}_{tag}.json"
        path.write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"N={n:3d} {dt:7.1f}s u_min={sol.metrics['u_min']:+.7f} "
              f"v_max={sol.metrics['v_max']:+.7f} psi_min={sol.metrics['psi_min']:+.7f} "
              f"slip={sol.metrics['lid_slip_defect']:.2e} -> {path}", flush=True)
        records.append({k: rec[k] for k in
                        ("order", "nodes_per_side", "reynolds", "metrics",
                         "ghia", "wall_seconds")})
        prev = {"psi": sol.psi, "omega": sol.omega, "xi": sol.xi}

    summary = {
        "case": "CASE-FVM-CAVITY-RE1000",
        "reynolds": args.reynolds,
        "orders": orders,
        "tag": tag,
        "records": records,
        "files": [f"cavity_spectral_N{n:03d}_{tag}.json" for n in orders],
    }
    (outdir / f"cavity_spectral_summary_{tag}.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True))
    print(f"summary -> {outdir / f'cavity_spectral_summary_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())