"""Run many INDEPENDENT cgrid configs in parallel across cores.

WHY parallelise across cases, not inside one case
------------------------------------------------
The cgrid solver is an explicit time-march on a ~15k-cell structured grid.  The
time loop is sequentially dependent (step n needs step n-1), so a single case
cannot be parallelised across steps.  Worse, the dominant cost per step is the
multigrid Poisson solve whose matvecs are tiny (~5 nnz/row); at this size
OpenBLAS threading is *slower* than single-threaded (thread-spawn overhead >
flops).  So the 32 cores are best spent running 32 independent cases at once,
each pinned to ONE BLAS thread.

Usage
-----
    python3 run_batch.py            # runs the built-in demo sweep
or import and call run_batch(configs, nproc=...).
"""
import os
import multiprocessing as mp

import tensorlbm.run_cgrid as RC


def _worker(cfg):
    # pin each worker to a single BLAS thread: intra-case threading is a net
    # loss at this grid size, and we want the cores free for other workers.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        RC.run(**cfg)
    except Exception as e:  # a failing case must not kill the whole batch
        print(f"[batch] WORKER FAILED for {cfg.get('out')}: {e!r}")
    return cfg.get("out")


def run_batch(configs, nproc=None):
    nproc = nproc or min(len(configs), mp.cpu_count())
    print(f"[batch] {len(configs)} configs on {nproc} workers "
          f"(single BLAS thread each)")
    with mp.Pool(nproc) as pool:
        return pool.map(_worker, configs)


def _demo_configs():
    """rhie vs adjoint, plus a mesh-refinement pair -- all independent."""
    base = dict(geometry="airfoil", nsteps=4000, nj=71, ni=221, Rf=40,
                beta=6, aoa=4, Re=100, steady=True, field_every=0,
                scheme="ppm", cb="pair", reg_r=0.0, nu_hyp=0.0, Lw=None,
                pert_amp=0.0, pert_k0=0, pert_k1=0, n_surf=200)
    return [
        {**base, "mode": "rhie",    "out": "batch_rhie_Rf40"},
        {**base, "mode": "adjoint", "out": "batch_adj_Rf40"},
        # grid-refinement leg of the GCI study (denser wall, finer j)
        {**base, "nj": 120, "beta": 8, "mode": "adjoint",
         "out": "batch_adj_Rf40_b8_nj120"},
    ]


if __name__ == "__main__":
    run_batch(_demo_configs())
