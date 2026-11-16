#!/usr/bin/env python3
"""XVAL-FVM-DAMBREAK-THIRD-CODE: independent-code cross-check of the
Ritter/Stoker dam-break case pool entry (CASE-FVM-DAMBREAK-STOKER).

The pool entry validates TensorFVM's SWE solver against the *closed-form*
Ritter (1892) / Stoker (1957) solutions.  Every check so far has used the
same code family (TensorFVM FVM) on both sides of the comparison.  This
script closes the one axis that has been ``not_run`` since D29: a
**different code family** (CLAWPACK / pyclaw, Godunov-type FV with a
Fortran core) is run on the *identical* domain and parameters, and the two
solutions are compared.

Pre-declared criteria (frozen before the first run):

* C1  pyclaw vs exact, depth L1 relative error < 5 % (both cases)
      -- the third code must itself be accurate enough to be a reference;
* C2  pyclaw vs TensorFVM (archived n=1600 profile), depth L1 relative
      error < 5 % (both cases) -- the acceptance statement of the ledger
      item ("异族代码同解 L1 <= 5 %");
* C3  leading-front position at t* = 2 agrees between the two codes to
      within 2 cells of the n=1600 grid (dx = 10/1600 m);
* C4  mass (volume) conservation of the pyclaw run: |dV/V0| < 1e-6;
* C5  negative control: comparing pyclaw against the *wrong* case
      (ritter profile vs stoker state) must exceed 5 %, i.e. the metric
      has power to reject a mismatched pair.

Usage::

    PYTHONPATH=src python scripts/xval_dambreak_third_code.py \
        --out docs/case-fvm-dambreak-stoker/artifacts

Outputs: ``xval_third_code.json`` (all numbers, machine-generated) and
``fig_xval_third_code.png`` (profiles at t* = 2 for both cases).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tensorfvm.shallow_water import RITTER_DRY, STOKER_WET  # noqa: E402

ARTIFACT_DIR = os.path.join(
    REPO_ROOT, "src", "tensorfvm", "verification", "artifacts",
    "shallow-water-dam-break")

TOLERANCE = 0.05          # acceptance: L1 <= 5 %
FRONT_TOL_CELLS = 2.0     # C3: front agreement within 2 cells of n=1600
MASS_TOL = 1e-6           # C4
N_TFVM = 1600             # archived TensorFVM headline grid


# --------------------------------------------------------------------------- #
# pyclaw driver (third code)
# --------------------------------------------------------------------------- #
def run_pyclaw(h0, h1, g, x_dam, x_min, x_max, t_end, n):
    """Run the 1-D SWE dam-break with CLAWPACK/pyclaw (HLL(e) Riemann).

    Returns cell-centred ``(x, h, u)`` at ``t_end``.  The initial state is
    the same step as TensorFVM's ``run_case``: h = h0 for x < x_dam, h1
    otherwise, u = 0.  Wall boundaries (the front stays interior at t*=2).
    """
    from clawpack import pyclaw, riemann

    solver = pyclaw.ClawSolver1D(riemann.shallow_hlle_1D)
    solver.kernel_language = 'Fortran'
    x = pyclaw.Dimension(x_min, x_max, n, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 2)
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    state.problem_data['g'] = g
    state.problem_data['grav'] = g
    xc = domain.grid.x.centers
    state.q[0, :] = np.where(xc < x_dam, h0, h1)
    state.q[1, :] = 0.0
    solution = pyclaw.Solution(state, domain)
    solver.setup(solution)
    wall = time.perf_counter()
    solver.evolve_to_time(t_end)
    wall = time.perf_counter() - wall

    st = solution.states[0]
    h = np.asarray(st.q[0, :], dtype=float).copy()
    hu = np.asarray(st.q[1, :], dtype=float).copy()
    u = np.where(h > 1e-12, hu / np.maximum(h, 1e-12), 0.0)
    dx = (x_max - x_min) / n
    volume = float(np.sum(h) * dx)
    volume0 = float((x_dam - x_min) * h0 + (x_max - x_dam) * h1)
    return {"x": np.asarray(xc, dtype=float), "h": h, "u": u, "dx": dx,
            "n": n, "t": t_end, "wall_time_s": wall, "volume": volume,
            "volume0": volume0}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def l1_rel(a, b, dx):
    """Relative L1 error of ``a`` against ``b`` (cell averages)."""
    num = float(np.sum(np.abs(a - b)) * dx)
    den = float(np.sum(np.abs(b)) * dx)
    return num / den if den > 0 else num


def front_position(x, h, level=1e-6):
    """Rightmost cell centre whose depth exceeds ``level``."""
    wet = np.nonzero(h > level)[0]
    return float(x[wet[-1]]) if wet.size else float("nan")


def load_tfvm_profile(case_key):
    path = os.path.join(ARTIFACT_DIR, f"profile_{case_key}_n{N_TFVM}.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)
    return (np.asarray(data["x"], float), np.asarray(data["h_numeric"], float),
            np.asarray(data["u_numeric"], float))


def compare_case(case, pyclaw_res, out_dir):
    """Run C1-C5 for one case; return the result dict."""
    x_tf, h_tf, u_tf = load_tfvm_profile(case.key)
    dx = float(x_tf[1] - x_tf[0])
    assert abs(dx - pyclaw_res["dx"]) < 1e-12, "grid mismatch"
    assert np.allclose(x_tf, pyclaw_res["x"], atol=1e-12), "grid mismatch"

    h_ex, u_ex = case.exact(x_tf)
    h_pc, u_pc = pyclaw_res["h"], pyclaw_res["u"]

    c1 = l1_rel(h_pc, h_ex, dx)
    c2 = l1_rel(h_pc, h_tf, dx)
    front_tf = front_position(x_tf, h_tf)
    front_pc = front_position(pyclaw_res["x"], h_pc)
    front_ex = case.exact_front()
    c3 = abs(front_pc - front_tf) / dx
    c4 = abs(pyclaw_res["volume"] - pyclaw_res["volume0"]) / pyclaw_res["volume0"]

    return {
        "case": case.key,
        "kind": case.kind,
        "h0": case.h0, "h1": case.h1, "g": case.gravity,
        "x_min": case.x_min, "x_max": case.x_max,
        "t_end": case.t_end, "n": N_TFVM, "dx": dx,
        "pyclaw_wall_time_s": pyclaw_res["wall_time_s"],
        "C1_pyclaw_vs_exact_l1": c1,
        "C2_pyclaw_vs_tensorfvm_l1": c2,
        "front_tensorfvm_m": front_tf,
        "front_pyclaw_m": front_pc,
        "front_exact_m": front_ex,
        "C3_front_diff_cells": c3,
        "C4_pyclaw_mass_drift": c4,
        "pass": {
            "C1": c1 < TOLERANCE,
            "C2": c2 < TOLERANCE,
            "C3": c3 <= FRONT_TOL_CELLS,
            "C4": c4 < MASS_TOL,
        },
        "profiles": {"x": x_tf.tolist(), "h_tfvm": h_tf.tolist(),
                     "h_pyclaw": h_pc.tolist(), "h_exact": h_ex.tolist()},
    }


def make_figure(results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, res in zip(axes, results):
        p = res["profiles"]
        x = np.asarray(p["x"])
        ax.plot(x, p["h_exact"], "k-", lw=1.6, label="exact (Ritter/Stoker)")
        ax.plot(x, p["h_tfvm"], "b--", lw=1.2, label="TensorFVM FVM n=1600")
        ax.plot(x, p["h_pyclaw"], "r-", lw=1.2, label="CLAWPACK/pyclaw n=1600")
        ax.axvline(res["front_exact_m"], color="gray", ls=":", lw=0.8)
        ax.set_title(f"{res['case']}  (C2 L1 = {100*res['C2_pyclaw_vs_tensorfvm_l1']:.2f} %)")
        ax.set_xlabel("x [m]")
        ax.set_xlim(-4.0, 6.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("water depth h [m]")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Dam-break t*=2: third-code cross-validation (XVAL-FVM-DAMBREAK-THIRD-CODE)",
                  fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "docs", "case-fvm-dambreak-stoker", "artifacts"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = []
    for case in (RITTER_DRY, STOKER_WET):
        pc = run_pyclaw(case.h0, case.h1, case.gravity, case.x_dam,
                       case.x_min, case.x_max, case.t_end, N_TFVM)
        results.append(compare_case(case, pc, args.out))

    # C5 negative control: ritter pyclaw profile vs stoker exact state
    x_tf, h_tf, _ = load_tfvm_profile("stoker-wet")
    dx = float(x_tf[1] - x_tf[0])
    h_ex_st, _ = STOKER_WET.exact(x_tf)
    h_pc_rit = results[0]["profiles"]["h_pyclaw"]
    c5 = l1_rel(h_pc_rit, h_ex_st, dx)

    payload = {
        "feature_slug": "XVAL-FVM-DAMBREAK-THIRD-CODE",
        "anchor": "CASE-FVM-DAMBREAK-STOKER",
        "third_code": "CLAWPACK 5.14.0 / pyclaw classic (Godunov-type FV, Fortran core)",
        "tolerance": TOLERANCE,
        "front_tol_cells": FRONT_TOL_CELLS,
        "mass_tol": MASS_TOL,
        "cases": results,
        "C5_negative_control_l1": c5,
        "pass": {
            "C1": all(r["pass"]["C1"] for r in results),
            "C2": all(r["pass"]["C2"] for r in results),
            "C3": all(r["pass"]["C3"] for r in results),
            "C4": all(r["pass"]["C4"] for r in results),
            "C5": c5 > TOLERANCE,
        },
        "all_pass": all([all(r["pass"].values()) for r in results],
                    c5 > TOLERANCE),
    }
    out_json = os.path.join(args.out, "xval_third_code.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    out_png = os.path.join(args.out, "fig_xval_third_code.png")
    make_figure(results, out_png)

    print(json.dumps({k: payload[k] for k in
                       ("C5_negative_control_l1", "pass", "all_pass")}, indent=1))
    for r in results:
        print(r["case"], {k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in r.items() if k.startswith(("C", "front"))})
    print("wrote", out_json, out_png)


if __name__ == "__main__":
    main()
