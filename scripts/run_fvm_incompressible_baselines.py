#!/usr/bin/env python3
"""Run the incompressible FVM verification baselines.

Three cases (backward-facing step, planar sudden expansion, differentially
heated cavity), each on three grid levels, writing one JSON per (case, level)
into the output directory plus the field data of every level so figures can be
regenerated and finer grids can be started from the coarser solution.

Usage::

    python scripts/run_fvm_incompressible_baselines.py --out runs/baselines
    python scripts/run_fvm_incompressible_baselines.py --only bfs_n040 --out runs/baselines
    python scripts/run_fvm_incompressible_baselines.py --job 3 --nshards 15 --out runs/baselines
    python scripts/run_fvm_incompressible_baselines.py --only bfs_n080 --init-from bfs_n040 ...

The ``--job/--nshards`` pair selects a strided subset of the job table so the
runs can be sharded over the cluster nodes.  ``--init-from`` names the tag of the
next coarser level of the *same* case: its field data is injected into the finer
grid (grid sequencing), which removes most of the slow advective start-up
transient that dominates a cold start at Re_H = 800.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _load_kernel():
    """Import the kernel without executing ``tensorfvm/__init__.py``.

    The package initialiser imports torch (an otherwise unrelated legacy
    solver).  On the loong64 compute nodes the installed torch wheel cannot load
    its BLAS runtime, and importing the package would then hard-fail even though
    this kernel is pure NumPy.  Loading the two modules under a synthetic package
    keeps the relative imports working and never touches torch.
    """
    import importlib.util
    import types

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    pkg_name = "_tfvm_incompressible"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name + ".incompressible_baselines"]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(src, "tensorfvm")]
    sys.modules[pkg_name] = pkg
    for mod in ("staggered_simple", "incompressible_baselines"):
        name = f"{pkg_name}.{mod}"
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(src, "tensorfvm", mod + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[pkg_name + ".incompressible_baselines"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


#: Grid family for the step case.  The contract suggested H/20, H/40, H/80; the
#: measured cost of this steady SIMPLE at Re_H = 800 grows faster than the cell
#: count (sparse factorisation + 1/h iteration count from the CFL-bound
#: pseudo-time step), so H/80 alone needs ~10 h.  The delivered family is
#: H/10, H/20, H/40 with a uniform ratio of 2; the deviation is recorded in
#: docs/incompressible-baselines/VERIFICATION.md.
BFS_LEVELS = (10, 20, 40)
#: Same reasoning as BFS_LEVELS: D/40 (76,800 cells) needs ~1 s per SIMPLE
#: iteration, so the delivered family is D/10, D/20, D/30.
EXPANSION_LEVELS = (10, 20, 30)
CAVITY_LEVELS = (32, 64, 96)
CAVITY_RA = (1.0e4, 1.0e5, 1.0e6)
#: Reynolds sweep used to show that the measured expansion loss approaches the
#: inviscid Borda--Carnot value as the viscous excess vanishes.
EXPANSION_RE_SWEEP = (100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)

_k = _load_kernel()
_ss = sys.modules["_tfvm_incompressible.staggered_simple"]
BFS_REFERENCE = _k.BFS_REFERENCE
NATURAL_CONVECTION_REFERENCE = _k.NATURAL_CONVECTION_REFERENCE
SUDDEN_EXPANSION_K_THEORY = _k.SUDDEN_EXPANSION_K_THEORY
backward_facing_step_metrics = _k.backward_facing_step_metrics
build_backward_facing_step = _k.build_backward_facing_step
build_natural_convection_cavity = _k.build_natural_convection_cavity
build_sudden_expansion = _k.build_sudden_expansion
converge = _k.converge
inject_coarse_solution = _k.inject_coarse_solution
natural_convection_metrics = _k.natural_convection_metrics
sudden_expansion_metrics = _k.sudden_expansion_metrics


def jobs():
    """(case, level, extra) table with the iteration budget of each level."""
    out = []
    bfs_budget = {10: 24000, 20: 30000, 40: 30000}
    for n in BFS_LEVELS:
        out.append(dict(case="backward_facing_step", level=n, tag=f"bfs_n{n:03d}",
                        budget=dict(max_iter=bfs_budget[n]), init_from=(
                            None if n == BFS_LEVELS[0] else f"bfs_n{n // 2:03d}")))
    exp_budget = {10: 16000, 20: 20000, 30: 24000}
    for n in EXPANSION_LEVELS:
        out.append(dict(case="sudden_expansion", level=n, tag=f"expansion_n{n:03d}",
                        budget=dict(max_iter=exp_budget[n]), init_from=(
                            None if n == EXPANSION_LEVELS[0] else f"expansion_n{n // 2:03d}")))
    for ra in CAVITY_RA:
        for n in CAVITY_LEVELS:
            out.append(dict(
                case="natural_convection_cavity", level=n, rayleigh=ra,
                tag=f"cavity_n{n:03d}_ra{ra:.0e}".replace("+", ""),
                budget=dict(max_iter=int({32: 20000, 64: 40000, 96: 60000}[n]
                                         * (1.4 if ra > 1e5 else 1.0))),
                init_from=None if n == CAVITY_LEVELS[0]
                else f"cavity_n{n // 2:03d}_ra{ra:.0e}".replace("+", "")))
    for re_d in EXPANSION_RE_SWEEP:
        out.append(dict(case="sudden_expansion", level=10, re_d=re_d,
                        tag=f"expansion_re{re_d:07.0f}",
                        budget=dict(max_iter=20000), init_from=None))
    return out


def _resolve_init(table, tag, outdir):
    """Find the requested tag in the job table (earlier jobs may also be sharded)."""
    for job in table:
        if job["tag"] == tag:
            return job, os.path.join(outdir, tag + "_fields.npz")
    return None, os.path.join(outdir, tag + "_fields.npz")


def run_one(job: dict, outdir: str, *, poisson_solver: str = "sparse",
            init_dir: str | None = None, table=None, alpha_u: float = 0.8,
            alpha_p: float = 0.5, deferred: int = 1,
            dc_relax: float = 0.05) -> dict:
    t0 = time.time()
    case = job["case"]
    if case == "backward_facing_step":
        solver = build_backward_facing_step(job["level"], poisson_solver=poisson_solver)
    elif case == "sudden_expansion":
        solver = build_sudden_expansion(job["level"], re_D=job.get("re_d", 100.0),
                                        poisson_solver=poisson_solver)
    else:
        solver = build_natural_convection_cavity(job["level"], rayleigh=job["rayleigh"],
                                                 poisson_solver=poisson_solver)

    if case != "natural_convection_cavity":
        # The cavity needs the slower under-relaxation of its builder (0.5/0.2):
        # at Ra = 1e6 the cell Peclet number is ~800, so the frozen van Leer
        # defect plus the looser relaxation (0.8/0.5) diverges within ~2500
        # iterations.  The step/expansion cases only use first-order convection
        # and are stable at 0.8/0.5, which is markedly faster there.
        solver.alpha_u = float(alpha_u)
        solver.alpha_p = float(alpha_p)
    injected = False
    init_tag = job.get("init_from")
    if init_tag and init_dir:
        path = os.path.join(init_dir, init_tag + "_fields.npz")
        if os.path.exists(path):
            with np.load(path) as data:
                coarse = {key: data[key] for key in ("u", "v", "p", "T")}
            inject_coarse_solution(solver, coarse)
            injected = True

    if case == "natural_convection_cavity":
        conv = converge(solver, max_iter=job["budget"]["max_iter"], tol=1e-7,
                        check=10, momentum_sweeps=5, p_cycles=3,
                        deferred_cycles=deferred, first_order_iters=1.0,
                        dc_relax=dc_relax, verbose=True)
        metrics = natural_convection_metrics(solver)
        reference = dict(NATURAL_CONVECTION_REFERENCE[job["rayleigh"]])
    else:
        # First order only: the limited (van Leer) defect correction is not a
        # contraction at the cell Peclet numbers of these two cases
        # (Pe_cell ~ 10 for the expansion, ~ 40 for the step at Re_H = 800), see
        # docs/incompressible-baselines/VERIFICATION.md.  Recorded as such.
        conv = converge(solver, max_iter=job["budget"]["max_iter"], tol=1e-6,
                        check=10, momentum_sweeps=4, p_cycles=2,
                        deferred_cycles=0, first_order_iters=1.0, verbose=True)
        if case == "backward_facing_step":
            metrics = backward_facing_step_metrics(solver)
            reference = dict(BFS_REFERENCE)
        else:
            metrics = sudden_expansion_metrics(solver)
            reference = dict(k_theory=SUDDEN_EXPANSION_K_THEORY,
                             re_d=job.get("re_d", 100.0))
    wall = time.time() - t0
    record = dict(job=job, metrics=metrics, reference=reference,
                  convergence=conv, meta=solver.meta,
                  settings=dict(alpha_u=solver.alpha_u, alpha_p=solver.alpha_p,
                                alpha_T=solver.alpha_T, limited=solver.limited,
                                dc_frozen=solver.dc_frozen,
                                poisson_solver=solver.poisson_solver,
                                grid_sequenced_from=init_tag if injected else None),
                  wall_seconds=round(wall, 1))
    with open(os.path.join(outdir, job["tag"] + ".json"), "w") as fh:
        json.dump(record, fh, indent=2, default=float)
    np.savez_compressed(
        os.path.join(outdir, job["tag"] + "_fields.npz"),
        u=solver.u, v=solver.v, p=solver.p, T=solver.T,
        fluid=solver.fluid, hx=solver.hx, hy=solver.hy)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/incompressible-baselines")
    ap.add_argument("--job", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma separated tags")
    ap.add_argument("--init-from", default=None,
                    help="directory holding the coarser level's *_fields.npz")
    ap.add_argument("--poisson", default="sparse", choices=("sparse", "multigrid"))
    ap.add_argument("--dc-relax", type=float, default=0.05,
                    help="defect-correction relaxation; the archived cavity ladder "
                         "used the frozen path (1.0), the damped default diverges there")
    ap.add_argument("--deferred", type=int, default=1,
                    help="deferred-correction cycles for the cavity cases (0 = first order)")
    ap.add_argument("--max-iter", type=int, default=0,
                    help="override the per-job iteration budget (0 = table value)")
    ap.add_argument("--alpha-u", type=float, default=0.8)
    ap.add_argument("--alpha-p", type=float, default=0.5)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    table = jobs()
    selected = table
    if args.only:
        wanted = set(args.only.split(","))
        selected = [j for j in table if j["tag"] in wanted]
    for idx, job in enumerate(selected):
        if args.nshards > 1 and idx % args.nshards != args.job:
            continue
        if args.max_iter:
            job["budget"] = dict(job["budget"], max_iter=int(args.max_iter))
        print(f"[{idx}] {job['tag']} start", flush=True)
        rec = run_one(job, args.out, poisson_solver=args.poisson,
                      init_dir=args.init_from or args.out, table=table,
                      alpha_u=args.alpha_u, alpha_p=args.alpha_p,
                      deferred=args.deferred, dc_relax=args.dc_relax)
        print(f"[{idx}] {job['tag']} done in {rec['wall_seconds']}s "
              f"metrics={json.dumps(rec['metrics'], default=float)[:400]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())