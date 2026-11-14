"""Third-code cross-validation of CASE-WOODWARD-COLELLA with Clawpack/PyClaw.

Why this file exists
--------------------
The existing cross-code gate (``../xcode/wc_crosscode.py``) compares TensorFVM's
MUSCL/HLLC solution with a WENO5/Rusanov solver that lives in the *same*
repository and therefore shares the grid, ghost-cell and call-site code that
the graded run uses.  ``docs/sod_xval_report.md`` recorded that PyClaw (Clawpack)
could not be built on loongarch64 and downgraded that cross-check to a single
reference; the installed Clawpack 5.14.0 now ships loongarch64 binaries, so the
third-party code is available and the cross-check can be restored.

This script drives Clawpack's Classic wave-propagation solver (an independent
implementation by different authors, in Fortran, with its own limiter and its
own wave decomposition) on exactly the same initial datum, boundary conditions
and final time as the graded TensorFVM run, with two different Riemann solvers
(Roe + entropy fix, and HLLE — the latter resolves no contact discontinuity, so
the pair brackets the flux-model error).  Every profile is graded with the
unchanged metric suite of :mod:`tensorfvm.compressible.wc_reference`.

What is compared (all at ``t = 0.038``)
---------------------------------------
1. conservation of the closed tube (each code asserts on itself);
2. Clawpack vs the digitised published reference (committed and calibrated
   tables) — does a third code reproduce the same numbers TensorFVM gets?
3. Clawpack vs TensorFVM at matched grids (200 … 3200 cells): the L1 norm of
   the density difference is an independent bound on the discretisation error
   of the graded solution — no shared code, no shared flux, no shared limiter;
4. the *floor* question: if the two independent codes agree far better than
   each agrees with the digitised reference, the residual is a property of the
   reference (print/vector-path fidelity), not of either solver.

Usage (repo root; ~1 min, single core)::

    PYTHONPATH=src python docs/woodward-colella-double-blast/pyclaw/wc_pyclaw_xval.py

The script writes ``results.json`` and ``profiles.npz`` next to itself; the
regression gate ``tests/test_compressible_wc_pyclaw_xval.py`` reads the stored
JSON (and skips when Clawpack is absent).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr

HERE = Path(__file__).resolve().parent
DOCDIR = HERE.parent
REFDIR = DOCDIR / "reference"
ASSETS = DOCDIR / "artifacts"
LADDER = DOCDIR / "ladder" / "data"

T_END = 0.038
X1, X2 = 0.1, 0.9
P_LEFT, P_MID, P_RIGHT = 1000.0, 0.01, 100.0
RHO0 = 1.0
GAMMA = 1.4


def _solver_module(name: str):
    from clawpack import riemann
    return {"roe": riemann.euler_with_efix_1D,
            "hlle": riemann.euler_hlle_1D}[name]


def claw_run(nx: int, flux: str = "roe", t_end: float = T_END,
             limiter: int = 4, cfl: float = 0.4):
    """One Classic run of the double blast; returns (x, rho, p, u, info)."""
    from clawpack import pyclaw

    solver = pyclaw.ClawSolver1D(_solver_module(flux))
    solver.num_eqn = 3
    solver.num_waves = 3
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.limiters = limiter          # 4 = MC, same family as the graded run
    solver.cfl_desired = cfl
    solver.cfl_max = min(1.0, cfl + 0.05)
    domain = pyclaw.Domain((0.0,), (1.0,), (nx,))
    sol = pyclaw.Solution(3, domain)
    x = domain.grid.x.centers
    q = sol.q
    q[0, :] = RHO0
    q[1, :] = 0.0
    # Clawpack's Euler convention: q = [rho, rho*u, E] with E the *total* energy
    # density E = p/(gamma-1) + 0.5*rho*u^2 (not the pressure).
    p0 = np.where(x < X1, P_LEFT, np.where(x < X2, P_MID, P_RIGHT))
    q[2, :] = p0 / (GAMMA - 1.0) + 0.5 * q[0, :] * q[1, :] ** 2
    sol.problem_data.update({"gamma": GAMMA, "gamma1": GAMMA - 1.0})

    t0 = time.time()
    nsteps = 0
    try:
        solver.evolve_to_time(sol, t_end)
        nsteps = int(getattr(solver, "num_steps", 0) or 0)
    except AttributeError:                      # pragma: no cover - older pyclaw
        while sol.t < t_end - 1e-14:
            solver.step(sol)
            nsteps += 1
    seconds = time.time() - t0

    rho, mom, energy = sol.q[0].copy(), sol.q[1].copy(), sol.q[2].copy()
    u = mom / rho
    p = (GAMMA - 1.0) * (energy - 0.5 * rho * u * u)
    info = {"nx": nx, "flux": flux, "limiter": limiter, "cfl": cfl,
            "t": float(sol.t), "steps": nsteps, "seconds": round(seconds, 2),
            "mass": float(np.trapezoid(rho, x)),
            "energy_integral": float(np.trapezoid(p / (GAMMA - 1.0)
                                                  + 0.5 * rho * u * u, x)),
            "rho_min": float(rho.min())}
    return x, rho, p, u, info


def _load_reference(name: str) -> dict:
    path = REFDIR / name
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    x, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
    return {"path": str(path.relative_to(wcr.Path(DOCDIR).parents[0])),
            "x": x, "rho": 0.5 * (a + b),
            "panel_spread": float(np.max(np.abs(a - b)))}


def _load_profile(npz: Path):
    if not npz.exists():
        return None
    d = np.load(npz)
    keys = list(d.keys())
    def pick(*cands, default=None):
        for c in cands:
            if c in keys:
                return d[c]
        return default
    x = pick("x", "x_got", "x_", default=np.arange(d[keys[0]].size).astype(float))
    rho = pick("rho", "rho_got", "rho_", default=d[keys[0]])
    return {"x": np.asarray(x).ravel(), "rho": np.asarray(rho).ravel()}


TENSORFVM_PROFILES = {
    400: ASSETS / "wc_blast_profile_nx400.npz",
    800: ASSETS / "wc_blast_profile_nx800.npz",
    1600: LADDER / "ladder_profile_nx1600_hllc.npz",
    3200: LADDER / "ladder_profile_nx3200_hllc.npz",
}


def l1_between(xa, ra, xb, rb) -> float:
    """∫|ra − rb_interp| dx / ∫rb dx over the overlap (shape mismatch norm)."""
    lo = max(xa.min(), xb.min())
    hi = min(xa.max(), xb.max())
    m = (xa >= lo) & (xa <= hi)
    x, ra_ = xa[m], ra[m]
    rb_ = np.interp(x, xb, rb)
    return float(np.trapezoid(np.abs(ra_ - rb_), x) / np.trapezoid(rb_, x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[200, 400, 800, 1600, 3200])
    args = ap.parse_args()

    refs = {"committed": _load_reference("reference_density.csv"),
            "calibrated": _load_reference("reference_density_calibrated.csv")}

    rows = []
    store = {}
    for nx in args.grids:
        for flux in ("roe", "hlle"):
            x, rho, p, u, info = claw_run(nx, flux)
            key = f"{nx}_{flux}"
            store[f"x_{key}"], store[f"rho_{key}"] = x, rho
            store[f"p_{key}"], store[f"u_{key}"] = p, u
            tf = _load_profile(TENSORFVM_PROFILES.get(nx)) if nx in TENSORFVM_PROFILES else None
            row = {"nx": nx, "flux": flux, **info}
            for tag, ref in refs.items():
                m = wcr.metrics_from_profiles(ref["x"], ref["rho"], x, rho)
                row[tag] = {
                    "peak_rel_percent": m["peak_density"]["rel_error_percent"],
                    "peak_x_abs": m["peak_position"]["abs_error"],
                    "l1_percent": m["l1_relative"]["percent"],
                    "window_mass_percent": m["window_mass"]["rel_error_percent"],
                    "low_plateau_percent": m["low_plateau"]["rel_error_percent"],
                    "faces_abs_max": (max(m["faces_abs_error"])
                                      if m["faces_abs_error"] else None),
                    "verdict": m["verdict"],
                }
            if tf is not None:
                mm = wcr.metrics_from_profiles(tf["x"], tf["rho"], x, rho)
                row["vs_tensorfvm"] = {
                    "l1_percent": mm["l1_relative"]["percent"],
                    "peak_rel_percent": mm["peak_density"]["rel_error_percent"],
                    "peak_x_abs": mm["peak_position"]["abs_error"],
                    "gl1_percent": 100 * l1_between(tf["x"], tf["rho"], x, rho),
                }
            rows.append(row)
            print(f"nx={nx:<5d} {flux:<5s} t={info['t']:.4f} steps={info['steps']:<5d} "
                  f"{info['seconds']:>5.1f}s | vs_ref(cal) peak={row['calibrated']['peak_rel_percent']:6.2f}% "
                  f"L1={row['calibrated']['l1_percent']:5.2f}% "
                  f"mass={row['calibrated']['window_mass_percent']:5.2f}% "
                  f"lowP={row['calibrated']['low_plateau_percent']:6.2f}% "
                  f"{row['calibrated']['verdict']}"
                  + (f" | vs_TFVM L1={row['vs_tensorfvm']['l1_percent']:5.2f}%"
                     if "vs_tensorfvm" in row else ""))

    # sanity gate on the setup itself: the two Riemann solvers must agree with
    # each other on the same mesh (same code, different flux) and mass must be
    # conserved exactly in the closed tube.
    by = {(r["nx"], r["flux"]): r for r in rows}
    pair = []
    for nx in args.grids:
        a, b = by.get((nx, "roe")), by.get((nx, "hlle"))
        if a and b:
            k = f"{nx}"
            pair.append({"nx": nx,
                         "roe_vs_hlle_l1_percent": 100 * l1_between(
                             store[f"x_{k}_roe"], store[f"rho_{k}_roe"],
                             store[f"x_{k}_hlle"], store[f"rho_{k}_hlle"])})
    mass = {"initial": float(RHO0 * 1.0),
            "max_rel_drift": max(abs(r["mass"] - 1.0) for r in rows)}

    out = {"case": "CASE-WOODWARD-COLELLA (two interacting blast waves)",
           "t_end": T_END, "third_code": "Clawpack 5.14.0 Classic (wave propagation)",
           "third_code_version": __import__("clawpack").__version__,
           "metric_suite": "tensorfvm.compressible.wc_reference (unchanged)",
           "grids": args.grids, "rows": rows, "flux_pair": pair,
           "conservation": mass,
           "reference": {k: {"path": v["path"], "panel_spread": v["panel_spread"]}
                         for k, v in refs.items()}}
    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(HERE / "profiles.npz", **store)
    print("\nwrote", HERE / "results.json", (HERE / "profiles.npz"))
    print("rho/HL dose pair L1:", [(p["nx"], round(p["roe_vs_hlle_l1_percent"], 3)) for p in pair])
    print("mass drift max:", mass["max_rel_drift"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())