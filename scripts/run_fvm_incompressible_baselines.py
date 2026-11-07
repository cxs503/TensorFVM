#!/usr/bin/env python3
"""Run the incompressible FVM verification baselines.

Three cases (backward-facing step, planar sudden expansion, differentially
heated cavity), each on three grid levels, writing one JSON per (case, level)
into the output directory plus the field data of the finest level so figures can
be regenerated without re-running.

Usage::

    python scripts/run_fvm_incompressible_baselines.py --out runs/baselines
    python scripts/run_fvm_incompressible_baselines.py --job 3 --nshards 15 --out runs/baselines

The ``--job/--nshards`` pair selects a strided subset of the job table so the
runs can be sharded over cluster nodes.
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


BFS_LEVELS = (20, 40, 80)
EXPANSION_LEVELS = (10, 20, 40)
CAVITY_LEVELS = (32, 64, 96)
CAVITY_RA = (1.0e4, 1.0e5, 1.0e6)

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
natural_convection_metrics = _k.natural_convection_metrics
sudden_expansion_metrics = _k.sudden_expansion_metrics



def jobs():
    out = []
    for n in BFS_LEVELS:
        out.append(dict(case="backward_facing_step", level=n,
                        tag=f"bfs_n{n:03d}", budget=dict(max_iter={20: 5000, 40: 6000, 80: 9000}[n])))
    for n in EXPANSION_LEVELS:
        out.append(dict(case="sudden_expansion", level=n,
                        tag=f"expansion_n{n:03d}", budget=dict(max_iter={10: 5000, 20: 6000, 40: 9000}[n])))
    for ra in CAVITY_RA:
        for n in CAVITY_LEVELS:
            out.append(dict(case="natural_convection_cavity", level=n, rayleigh=ra,
                            tag=f"cavity_n{n:03d}_ra{ra:.0e}".replace("+", ""),
                            budget=dict(max_iter=int({32: 4000, 64: 7000, 96: 12000}[n] * (1.4 if ra > 1e5 else 1.0)))))
    return out


def run_one(job: dict, outdir: str) -> dict:
    t0 = time.time()
    case = job["case"]
    if case == "backward_facing_step":
        solver = build_backward_facing_step(job["level"])
        conv = converge(solver, max_iter=job["budget"]["max_iter"], tol=1e-6,
                        check=10, momentum_sweeps=4, p_cycles=2,
                        deferred_cycles=1, first_order_iters=1.0, verbose=True)
        metrics = backward_facing_step_metrics(solver)
        reference = dict(BFS_REFERENCE)
    elif case == "sudden_expansion":
        solver = build_sudden_expansion(job["level"])
        conv = converge(solver, max_iter=job["budget"]["max_iter"], tol=1e-6,
                        check=10, momentum_sweeps=4, p_cycles=2,
                        deferred_cycles=1, first_order_iters=1.0, verbose=True)
        metrics = sudden_expansion_metrics(solver)
        reference = dict(k_theory=SUDDEN_EXPANSION_K_THEORY)
    else:
        solver = build_natural_convection_cavity(job["level"], rayleigh=job["rayleigh"])
        conv = converge(solver, max_iter=job["budget"]["max_iter"], tol=1e-7,
                        check=10, momentum_sweeps=5, p_cycles=3,
                        deferred_cycles=1, first_order_iters=1.0, verbose=True)
        metrics = natural_convection_metrics(solver)
        reference = dict(NATURAL_CONVECTION_REFERENCE[job["rayleigh"]])
    wall = time.time() - t0
    record = dict(job=job, metrics=metrics, reference=reference,
                  convergence=conv, meta=solver.meta,
                  settings=dict(alpha_u=solver.alpha_u, alpha_p=solver.alpha_p,
                                alpha_T=solver.alpha_T, limited=solver.limited,
                                dc_frozen=solver.dc_frozen),
                  wall_seconds=round(wall, 1))
    with open(os.path.join(outdir, job["tag"] + ".json"), "w") as fh:
        json.dump(record, fh, indent=2, default=float)
    if job["level"] == max(BFS_LEVELS if case == "backward_facing_step" else
                           EXPANSION_LEVELS if case == "sudden_expansion" else
                           CAVITY_LEVELS):
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
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    table = jobs()
    if args.only:
        wanted = set(args.only.split(","))
        table = [j for j in table if j["tag"] in wanted]
    for idx, job in enumerate(table):
        if args.nshards > 1 and idx % args.nshards != args.job:
            continue
        print(f"[{idx}] {job['tag']} start", flush=True)
        rec = run_one(job, args.out)
        print(f"[{idx}] {job['tag']} done in {rec['wall_seconds']}s "
              f"metrics={json.dumps(rec['metrics'], default=float)[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())