"""Third-code *high-resolution* reference for CASE-WOODWARD-COLELLA.

Why this file exists
--------------------
``wc_pyclaw_xval.py`` (m20159) graded TensorFVM against Clawpack at matched
grids up to nx = 1600, and reported that the ``nx = 3200`` Clawpack rung
"stops early at t ~ 0.0285" and was therefore excluded from every verdict.  The
recorded guess for the cause was "CFL step rejection or Roe positivity failure
close to the near-vacuum state at the wave collision".

That attribution is **wrong**, and the error hides a usable high-resolution
reference.  Instrumenting the run shows the truncated rung returns *without any
exception*, at ``t = 0.028457``, having done exactly 10000 accepted steps:
``nx = 3200`` at CFL 0.4 needs **12799** steps to reach ``t = 0.038``, and
``pyclaw.ClawSolver.max_steps`` defaults to **10000**.  The library simply
exhausts its step budget and returns a partially evolved solution (see
``docs/CASE-WOODWARD-COLELLA-round2-clawpack-errata.md``).  ``nx = 1600``
needs 6379 steps, which is why the smaller rungs always looked healthy.

The fix is one line (``solver.max_steps``), and it unlocks what round 2 of the
cross-validation actually asked for: an *independent code* run far beyond the
3200-cell production grid of the digitised published curve, so the
"reference-side error" of that curve can be measured instead of inferred from
our own WENO5 (which lives in this same repository and shares its grid,
ghost-cell and call-site code with the graded run).

What is produced
----------------
1. a Clawpack (Classic wave propagation, Fortran, Roe + entropy fix, MC
   limiter, CFL 0.4) ladder at 3200 / 6400 / 12800 / 25600 cells, every rung
   *asserted* to have reached ``t = 0.038`` (a truncated rung is now a hard
   error, not a silent row);
2. the third code's own self-convergence, which is its error bar as a pseudo
   truth;
3. the graded TensorFVM ladder (800 / 1600 / 3200, stored profiles, solver not
   re-run) graded against that truth — the number the ledger tolerance
   (``< 10 %`` for this case, ``< 15 %`` for the owner's precision rule) should
   have been asked for all along;
4. the cross-code agreement at matched grids 3200 / 6400, which bounds the
   discretisation error of the graded solution without any reference table at
   all.

Honest limitation: Clawpack Classic is second order, so "truth" here is a
numerical solution with its own ~1 % error bar, not an exact solution.  The
error bar is measured (successive-rung L1) and quoted with every claim.

Usage (repo root; ``--grids`` ladder is ~13 min single core at 3200..25600)::

    PYTHONPATH=src python docs/woodward-colella-double-blast/pyclaw/wc_clawpack_highres.py
    # one rung per shard ({i} is 1-based) for the compute gate
    ... --shard 1 --shards 4
    # analysis only, from stored profiles
    ... --analyze-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.compressible import wc_reference as wcr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wc_pyclaw_xval import (  # noqa: E402  (path set above)
    GAMMA,
    P_LEFT,
    P_MID,
    P_RIGHT,
    RHO0,
    T_END,
    X1,
    X2,
    _load_profile,
    l1_between,
)

OUTDIR = HERE / "highres"
PROFILES = OUTDIR / "profiles.npz"
RESULTS = OUTDIR / "results.json"

#: ``pyclaw.ClawSolver.max_steps`` defaults to 10000, which silently truncates
#: this case for nx >= 3200 (12799 steps needed).  Any rung that has not
#: reached ``t_end`` is rejected outright below.
MAX_STEPS = 10 ** 8
TRUNCATION_TOL = 1e-12

FLUXES = {"roe": "euler_with_efix_1D", "hlle": "euler_hlle_1D"}


def _solver_module(name: str):
    from clawpack import riemann
    return getattr(riemann, FLUXES[name])


def claw_run(nx: int, flux: str = "roe", t_end: float = T_END,
             limiter: int = 4, cfl: float = 0.4) -> tuple:
    """One Classic run; **raises** if the solver stops short of ``t_end``."""
    from clawpack import pyclaw

    solver = pyclaw.ClawSolver1D(_solver_module(flux))
    solver.num_eqn = 3
    solver.num_waves = 3
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.limiters = limiter           # 4 = MC, same family as the graded run
    solver.cfl_desired = cfl
    solver.cfl_max = min(1.0, cfl + 0.05)
    solver.max_steps = MAX_STEPS        # <-- the defect fix (default is 10000)

    domain = pyclaw.Domain((0.0,), (1.0,), (nx,))
    sol = pyclaw.Solution(3, domain)
    x = domain.grid.x.centers
    q = sol.q
    q[0, :] = RHO0
    q[1, :] = 0.0
    p0 = np.where(x < X1, P_LEFT, np.where(x < X2, P_MID, P_RIGHT))
    q[2, :] = p0 / (GAMMA - 1.0) + 0.5 * q[0, :] * q[1, :] ** 2
    sol.problem_data.update({"gamma": GAMMA, "gamma1": GAMMA - 1.0})

    t0 = time.time()
    solver.evolve_to_time(sol, t_end)
    seconds = time.time() - t0

    t_reached = float(sol.t)
    if abs(t_reached - t_end) > TRUNCATION_TOL:
        raise RuntimeError(
            f"Clawpack nx={nx} {flux} stopped at t={t_reached:.6f} "
            f"(target {t_end}); steps={solver.status['numsteps']}, "
            f"max_steps={solver.max_steps}. Refusing to store a truncated rung."
        )

    rho, mom, energy = sol.q[0].copy(), sol.q[1].copy(), sol.q[2].copy()
    u = mom / rho
    p = (GAMMA - 1.0) * (energy - 0.5 * rho * u * u)
    info = {
        "nx": nx, "flux": flux, "limiter": limiter, "cfl": cfl,
        "t": t_reached,
        "steps": int(solver.status["numsteps"]),
        "seconds": round(seconds, 2),
        "mass": float(np.trapezoid(rho, x)),
        "energy_integral": float(np.trapezoid(p / (GAMMA - 1.0)
                                              + 0.5 * rho * u * u, x)),
        "rho_min": float(rho.min()),
        "p_min": float(p.min()),
        "finite": bool(np.isfinite(rho).all() and np.isfinite(p).all()),
        "truncated": False,
    }
    return x, rho, p, u, info


def decay_probe(nx: int, max_steps: int, flux: str = "roe") -> dict:
    """Replay the *old* call (library default step cap) and show the truncation.

    Returns the ``t`` actually reached and the number of accepted steps, so the
    errata document can cite a reproducible number instead of a guess.
    """
    from clawpack import pyclaw

    solver = pyclaw.ClawSolver1D(_solver_module(flux))
    solver.num_eqn = 3
    solver.num_waves = 3
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.limiters = 4
    solver.cfl_desired = 0.4
    solver.cfl_max = 0.45
    if max_steps is not None:
        solver.max_steps = max_steps
    domain = pyclaw.Domain((0.0,), (1.0,), (nx,))
    sol = pyclaw.Solution(3, domain)
    x = domain.grid.x.centers
    q = sol.q
    q[0, :] = RHO0
    q[1, :] = 0.0
    p0 = np.where(x < X1, P_LEFT, np.where(x < X2, P_MID, P_RIGHT))
    q[2, :] = p0 / (GAMMA - 1.0)
    sol.problem_data.update({"gamma": GAMMA, "gamma1": GAMMA - 1.0})
    solver.evolve_to_time(sol, T_END)
    rho = sol.q[0].copy()
    p = (GAMMA - 1.0) * (sol.q[2] - 0.5 * sol.q[1] ** 2 / rho)
    return {"nx": nx, "flux": flux,
            "requested_max_steps": int(solver.max_steps),
            "steps_taken": int(solver.status["numsteps"]),
            "t_reached": float(sol.t),
            "target": T_END,
            "rho_min": float(rho.min()),
            "p_min": float(p.min()),
            "finite": bool(np.isfinite(rho).all() and np.isfinite(p).all())}


# --------------------------------------------------------------------------
# TensorFVM rungs (stored profiles; the solver is never re-run here)
# --------------------------------------------------------------------------
DOCDIR = HERE.parent
TFVM_PROFILES = {
    400: DOCDIR / "artifacts" / "wc_blast_profile_nx400.npz",
    800: DOCDIR / "artifacts" / "wc_blast_profile_nx800.npz",
    1600: DOCDIR / "ladder" / "data" / "ladder_profile_nx1600_hllc.npz",
    3200: DOCDIR / "ladder" / "data" / "ladder_profile_nx3200_hllc.npz",
}


def load_tfvm(nx: int):
    prof = TFVM_PROFILES.get(nx)
    if prof is None or not prof.exists():
        return None
    d = _load_profile(prof)
    return None if d is None else (d["x"], d["rho"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+",
                    default=[3200, 6400, 12800, 25600])
    ap.add_argument("--fluxes", nargs="+", default=["roe"])
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--truth-nx", type=int, default=25600)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    results_path = args.outdir / "results.json"
    profiles_path = args.outdir / "profiles.npz"

    # Resume from whatever earlier shards already stored.
    rows: list[dict] = []
    store: dict[str, np.ndarray] = {}
    if results_path.exists():
        prev = json.loads(results_path.read_text())
        rows = prev.get("rows", [])
        if profiles_path.exists():
            with np.load(profiles_path) as d:
                store = {k: d[k] for k in d.keys()}
    done = {(r["nx"], r["flux"]) for r in rows}

    todo = [(nx, f) for f in args.fluxes for nx in args.grids]
    if args.shard:
        n = args.shards or 1
        if not 1 <= args.shard <= n:
            raise SystemExit(f"shard {args.shard} outside 1..{n}")
        todo = todo[args.shard - 1::n]

    if not args.analyze_only:
        for nx, flux in todo:
            if (nx, flux) in done:
                print(f"skip nx={nx} {flux} (already stored)")
                continue
            x, rho, p, u, info = claw_run(nx, flux)
            key = f"{nx}_{flux}"
            store[f"x_{key}"], store[f"rho_{key}"] = x, rho
            store[f"p_{key}"], store[f"u_{key}"] = p, u
            rows.append(info)
            print(f"nx={nx:6d} {flux:5s} t={info['t']:.6f} "
                  f"steps={info['steps']:6d} {info['seconds']:7.1f}s "
                  f"rho_max={rho.max():.5f} p_min={info['p_min']:.3e}")
            rows.sort(key=lambda r: (r["flux"], r["nx"]))
            with open(profiles_path, "wb") as fh:
                np.savez_compressed(fh, **store)

    # ------------------------------------------------------------------
    # Analysis: Clawpack self-convergence -> truth + error bar
    # ------------------------------------------------------------------
    by_flux: dict[str, dict[int, tuple]] = {}
    for r in rows:
        key = f"{r['nx']}_{r['flux']}"
        if f"x_{key}" not in store:
            continue
        by_flux.setdefault(r["flux"], {})[r["nx"]] = (store[f"x_{key}"],
                                                      store[f"rho_{key}"])

    analysis: dict = {"self_convergence": {}, "truth": {}, "tfvm_vs_truth": {},
                      "cross_code": {}, "vs_digitised": {}}
    for flux, rungs in by_flux.items():
        sizes = sorted(rungs)
        sc = []
        for a, b in zip(sizes, sizes[1:]):
            xa, ra = rungs[a]
            xb, rb = rungs[b]
            sc.append({
                "coarser": a, "finer": b,
                "coarse_vs_fine_l1_percent": 100 * l1_between(xa, ra, xb, rb),
                "fine_vs_coarse_l1_percent": 100 * l1_between(xb, rb, xa, ra),
                "peak_coarse": float(ra.max()), "peak_fine": float(rb.max()),
                "peak_rel_percent": abs(ra.max() - rb.max()) / rb.max() * 100,
            })
        analysis["self_convergence"][flux] = sc

        if not sizes:
            continue
        truth_nx = args.truth_nx if args.truth_nx in rungs else sizes[-1]
        xr, rr = rungs[truth_nx]
        # error bar of the pseudo truth = L1 to the next coarser stored rung
        coarser = [n for n in sizes if n < truth_nx]
        bar = None
        if coarser:
            xc, rc = rungs[coarser[-1]]
            bar = 100 * l1_between(xc, rc, xr, rr)
        analysis["truth"][flux] = {
            "code": "Clawpack 5.14.0 Classic (wave propagation, Fortran)",
            "flux": flux, "nx": truth_nx,
            "peak": float(rr.max()),
            "error_bar_l1_percent": bar,
            "error_bar_note": ("L1 percent between truth and the next coarser "
                               "stored rung; a numerical solution has a "
                               "non-zero floor, this is its measured size"),
        }

        # graded TensorFVM ladder against this truth
        rows_tf = []
        for nx in sorted(TFVM_PROFILES):
            got = load_tfvm(nx)
            if got is None:
                continue
            xm, rm = got
            m = wcr.metrics_from_profiles(xr, rr, xm, rm)
            rows_tf.append({
                "nx": nx,
                "peak_percent": m["peak_density"]["rel_error_percent"],
                "l1_percent": m["l1_relative"]["percent"],
                "window_mass_percent": m["window_mass"]["rel_error_percent"],
                "peak_position_abs": m["peak_position"]["abs_error"],
                "low_plateau_percent": m["low_plateau"]["rel_error_percent"],
            })
        analysis["tfvm_vs_truth"][flux] = rows_tf

        # matched-grid cross-code agreement (no reference involved)
        cross = []
        for nx in sorted(by_flux["roe"]) if "roe" in by_flux else []:
            got = load_tfvm(nx)
            if got is None:
                continue
            xc, rc = by_flux["roe"][nx]
            xm, rm = got
            cross.append({"nx": nx,
                          "l1_percent": 100 * l1_between(xm, rm, xc, rc),
                          "peak_rel_percent": abs(rm.max() - rc.max())
                          / rc.max() * 100})
        analysis["cross_code"] = cross

        # how far the digitised published curve is from this truth
        for tag, calibrated in (("committed", False), ("calibrated", True)):
            ref = wcr.load_reference(calibrated=calibrated)
            m = wcr.metrics_from_profiles(xr, rr, ref["x"], ref["rho"])
            analysis["vs_digitised"][tag] = {
                "reference": ref["path"],
                "peak_percent": m["peak_density"]["rel_error_percent"],
                "l1_percent": m["l1_relative"]["percent"],
                "reference_peak": m["peak_density"]["ref"],
                "truth_peak": m["peak_density"]["got"],
            }

    payload = {
        "case": "CASE-WOODWARD-COLELLA (two interacting blast waves)",
        "t_end": T_END,
        "third_code": "Clawpack 5.14.0 Classic (wave propagation, Fortran)",
        "defect": {
            "symptom": "nx>=3200 rung silently truncated at t~0.0285",
            "root_cause": "pyclaw.ClawSolver.max_steps default 10000 < 12799 "
                          "steps required at nx=3200, CFL 0.4",
            "fix": "solver.max_steps raised; truncated rung now raises",
            "previous_attribution": "CFL rejection / Roe positivity failure "
                                    "at the collision — REFUTED",
        },
        "rows": rows,
        "analysis": analysis,
    }
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {results_path} and {profiles_path}")
    if analysis["self_convergence"]:
        for flux, sc in analysis["self_convergence"].items():
            for row in sc:
                print(f"  self-conv {flux} {row['coarser']}->{row['finer']}: "
                      f"L1 {row['coarse_vs_fine_l1_percent']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())