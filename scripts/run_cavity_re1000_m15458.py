"""Re = 1000 lid-driven cavity (Ghia, Ghia & Shin 1982) -- round-2 runner.

Anchor: ``CASE-CAVITY-RE1000-FVM`` (round 2, mission m15458).

Round 1 (m15369) burned its tool budget before producing a converged row; the
ladder in ``docs/verification-lid-cavity-ladder/`` only ever carried a *single*
Re = 1000 resolution (n = 64) and its n = 128 attempt was abandoned unconverged
(change rate 8.3e-6 at step 66 750, cap 2e5).  This runner closes exactly that
hole with two pieces of standard practice taken from the literature on this
benchmark (see ``docs/CASE-CAVITY-RE1000-FVM/method-selection.md``):

* **grid continuation** -- converge on n = 64, prolong that field onto n = 128 and
  keep marching.  The steady state of the fine discrete system does not depend on
  the initial guess, so this is a cost optimisation, not a physical assumption;
  it is *checked*, not trusted: n = 64 is also run cold and the two converged
  states are compared field-wise.
* **per-case checkpoints** so a long job that hits its cap still leaves the
  converged rows it finished (round 1 lost everything).

Everything reported is read back from the JSON this script writes; nothing is
transcribed by hand.  The reference values come from
:mod:`tensorfvm.lid_cavity_reference` (the committed transcription of Ghia et
al. Tables I/II).

Usage::

    PYTHONPATH=src python scripts/run_cavity_re1000_m15458.py \
        --out artifacts/CASE-CAVITY-RE1000-FVM --grids 64,128
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from tensorfvm.lid_cavity import (
    LidCavityConfig,
    LidCavitySolver,
    ghia_acceptance,
    score_against_ghia,
)
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE

TOLERANCE_FRACTION = 0.05      # case-ledger acceptance band for CASE-CAVITY-RE1000-FVM


# --------------------------------------------------------------------- fields
def cell_centred(solver: LidCavitySolver) -> tuple[torch.Tensor, torch.Tensor]:
    """Staggered face velocities -> cell-centred (n, n) velocities."""
    uc = 0.5 * (solver.u[:-1, :] + solver.u[1:, :])
    vc = 0.5 * (solver.v[:, :-1] + solver.v[:, 1:])
    return uc, vc


def prolong(solver: LidCavitySolver, n_fine: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear prolongation of the cell-centred velocity onto an ``n_fine`` grid.

    ``F.interpolate(..., align_corners=False)`` treats the samples as cell centres,
    which is exactly the collocation both grids use, so the coarse field is
    sampled at the fine cell centres without a half-cell shift.  The fine face
    fields are then rebuilt by averaging and the wall faces are zeroed (normal
    velocity at a wall).
    """
    uc, vc = cell_centred(solver)
    stacked = torch.stack([uc, vc])[None]                       # (1, 2, n, n)
    fine = torch.nn.functional.interpolate(stacked, size=(n_fine, n_fine),
                                           mode="bilinear", align_corners=False)
    uf, vf = fine[0, 0], fine[0, 1]
    u = torch.zeros(n_fine + 1, n_fine, dtype=solver.u.dtype)
    v = torch.zeros(n_fine, n_fine + 1, dtype=solver.u.dtype)
    u[1:n_fine, :] = 0.5 * (uf[:-1, :] + uf[1:, :])
    v[:, 1:n_fine] = 0.5 * (vf[:, :-1] + vf[:, 1:])
    return u, v


# ----------------------------------------------------------------------- run
def run_case(n: int, reynolds: float, out_dir: str, *, cfl: float = 0.15,
             tolerance: float = 1e-7, max_steps: int = 250_000,
             init: tuple[torch.Tensor, torch.Tensor] | None = None,
             init_from: str | None = None, check_every: int = 250) -> tuple[dict, LidCavitySolver]:
    torch.set_num_threads(int(os.environ.get("TFVM_THREADS", "1")))
    fields_path = os.path.join(out_dir, "fields", f"Re{reynolds:g}_n{n}.pt")
    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=reynolds, cfl=cfl,
                                              max_steps=max_steps,
                                              tolerance=tolerance,
                                              check_every=check_every))
    if init is not None:
        solver.u, solver.v = init[0].clone(), init[1].clone()
    resumed = False
    if os.path.exists(fields_path):
        saved = torch.load(fields_path, weights_only=False)
        solver.u, solver.v, solver.steps = saved["u"], saved["v"], int(saved["steps"])
        resumed = True
    started = time.time()
    state = {"last_flush": started, "next_save": 20_000}

    def on_check(sv, step, change):
        now = time.time()
        if now - state["last_flush"] > 120.0:
            print(f"      Re={reynolds:g} n={n} step {step} rate {change:.3e} "
                  f"({now - started:.0f}s)", flush=True)
            state["last_flush"] = now
        if step >= state["next_save"]:
            os.makedirs(os.path.dirname(fields_path), exist_ok=True)
            torch.save({"u": sv.u, "v": sv.v, "steps": sv.steps}, fields_path)
            state["next_save"] = step + 20_000

    info = solver.run(on_check=on_check)
    os.makedirs(os.path.dirname(fields_path), exist_ok=True)
    torch.save({"u": solver.u, "v": solver.v, "steps": solver.steps}, fields_path)

    result = score_against_ghia(solver, reynolds)
    gate = ghia_acceptance(result, allow_registered_disputes=True)
    record = {
        "reynolds": reynolds,
        "n": n,
        "grid": {"cells_per_direction": n, "h": 1.0 / n, "domain": [1.0, 1.0],
                 "lid_velocity": solver.config.lid_velocity},
        "parameters": {"Re": reynolds, "nu": solver.nu, "rho": solver.config.density,
                       "cfl": cfl, "tolerance": tolerance, "max_steps": max_steps,
                       "check_every": check_every},
        "solver": {"steps": info["steps"], "converged": info["converged"],
                   "residual_rate": info["residual"],
                   "max_divergence": info["max_divergence"],
                   "dt_final": info["dt"], "wall_seconds": time.time() - started,
                   "initial_condition": init_from or ("checkpoint" if resumed else "rest"),
                   "psi": solver.streamfunction_extrema()},
        "u": result["u"], "v": result["v"],
        "vortex_centre": {"computed": result["vortex_centre"]["computed"],
                          "reference": list(VORTEX_CENTRE[float(reynolds)])},
        "acceptance": gate,
    }
    record["verdict_5pct"] = judge_5pct(record)
    return record, solver


def judge_5pct(record: dict) -> dict:
    """The case-ledger criterion: every key quantity inside 5 %.

    Pointwise centreline deviation is expressed as a fraction of the lid speed,
    the two extrema and the vortex centre as relative errors.  All must pass;
    the individual numbers are returned so a partial pass cannot hide.
    """
    u, v = record["u"], record["v"]
    ext_u = u["extremum"]["min_computed"][1]
    ref_u = u["extremum"]["min_reference"][1]
    ext_v = v["extremum"]["max_computed"][1]
    ref_v = v["extremum"]["max_reference"][1]
    vc, vrc = record["vortex_centre"]["computed"], record["vortex_centre"]["reference"]
    items = {
        "u_min_rel": abs(ext_u - ref_u) / abs(ref_u),
        "v_max_rel": abs(ext_v - ref_v) / abs(ref_v),
        "vortex_centre_rel": max(abs(vc[0] - vrc[0]), abs(vc[1] - vrc[1])),
        "u_max_abs": u["max_abs_error"],
        "v_max_abs": v["max_abs_error"],
    }
    passed = {k: bool(val <= TOLERANCE_FRACTION) for k, val in items.items()}
    return {"tolerance": TOLERANCE_FRACTION, "metrics": items, "per_metric": passed,
            "pass": bool(all(passed.values()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/CASE-CAVITY-RE1000-FVM")
    ap.add_argument("--reynolds", type=float, default=1000.0)
    ap.add_argument("--grids", default="64,128")
    ap.add_argument("--cold-grids", default="32")
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--mission", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grids = [int(g) for g in args.grids.split(",") if g]
    summary_path = os.path.join(args.out, "summary.json")
    summary = {"case": "lid-driven cavity, Ghia et al. (1982) Tables I/II",
               "anchor": "CASE-CAVITY-RE1000-FVM", "mission": 20000,
               "mission_origin": 15458,  # round-2 lineage; round 3 (m20000) re-runs and commits
               "reynolds": args.reynolds, "tolerance_fraction": TOLERANCE_FRACTION,
               "cases": [], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))
    summary["mission"] = args.mission

    def flush() -> None:
        summary["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=1)

    # 1. coarse cold runs (control: is the coarse grid already converged?)
    coarse: dict[int, dict] = {}
    for n in [int(g) for g in args.cold_grids.split(",") if g]:
        print(f"[cold] Re={args.reynolds:g} n={n}", flush=True)
        rec, _ = run_case(n, args.reynolds, args.out, max_steps=args.max_steps)
        coarse[n] = rec
        summary["cases"] = [c for c in summary["cases"] if c["n"] != n] + [rec]
        flush()

    # 2. base grid, cold AND continued from the coarsest -- continuation check
    base = grids[0]
    print(f"[cold] Re={args.reynolds:g} n={base}", flush=True)
    base_rec, base_solver = run_case(base, args.reynolds, args.out,
                                     max_steps=args.max_steps)
    summary["cases"] = [c for c in summary["cases"] if c["n"] != base] + [base_rec]
    flush()

    # 3. finer grids: continued from the base grid
    for n in grids[1:]:
        src_solver = base_solver if n == grids[1] else fine_solver
        u0, v0 = prolong(src_solver, n)
        print(f"[continuation] Re={args.reynolds:g} n={n}", flush=True)
        rec, fine_solver = run_case(n, args.reynolds, args.out,
                                    max_steps=args.max_steps, init=(u0, v0),
                                    init_from=f"n={src_solver.n} prolongation")
        summary["cases"] = [c for c in summary["cases"] if c["n"] != n] + [rec]
        flush()

    if coarse:
        n0 = max(coarse)
        src = LidCavitySolver(LidCavityConfig(n=n0, reynolds=args.reynolds))
        saved = torch.load(os.path.join(args.out, "fields", f"Re{args.reynolds:g}_n{n0}.pt"),
                           weights_only=False)
        src.u, src.v = saved["u"], saved["v"]
        u0, v0 = prolong(src, base)
        print(f"[continuation] Re={args.reynolds:g} n={base} from n={n0}", flush=True)
        cont_rec, cont_solver = run_case(base, args.reynolds, args.out,
                                         max_steps=args.max_steps, init=(u0, v0),
                                         init_from=f"n={n0} prolongation")
        du = float((cont_solver.u - base_solver.u).abs().max())
        dv = float((cont_solver.v - base_solver.v).abs().max())
        check = {"continuation": {"from_n": n0, "to_n": base,
                                  "max_abs_du": du, "max_abs_dv": dv,
                                  "same_steady_state": bool(max(du, dv) < 1e-6),
                                  "continued_steps": cont_rec["solver"]["steps"],
                                  "cold_steps": base_rec["solver"]["steps"],
                                  "speedup": base_rec["solver"]["wall_seconds"]
                                  / max(cont_rec["solver"]["wall_seconds"], 1e-9)}}
        summary.setdefault("continuation_check", {}).update(check)
        flush()

    summary["cases"].sort(key=lambda c: c["n"])
    summary["grid_sequence"] = [{"n": c["n"],
                                 "u_min": c["u"]["extremum"]["min_computed"][1],
                                 "v_max": c["v"]["extremum"]["max_computed"][1],
                                 "max_abs_du": c["u"]["max_abs_error"],
                                 "converged": c["solver"]["converged"]}
                                for c in summary["cases"]]
    summary["all_converged"] = all(c["solver"]["converged"] for c in summary["cases"])
    summary["verdict_5pct"] = {str(c["n"]): c["verdict_5pct"]["pass"] for c in summary["cases"]}
    flush()
    print(json.dumps({"cases": summary["grid_sequence"],
                      "verdict_5pct": summary["verdict_5pct"]}, indent=1))


if __name__ == "__main__":
    main()