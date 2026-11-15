"""Temporal (Courant-number) refinement study for anchor ``CASE-FVM-CAVITY-RE1000``.

The grid sequence in ``runs/`` fixes one time-step law (``cfl = 0.15``), so it can
only speak about *spatial* discretisation uncertainty.  ``UNCERTAINTY.md`` §4
therefore recorded the Courant axis as ``not_run`` and claimed nothing about it.
This driver produces the missing evidence: the **same grid** (default ``n = 128``,
the graded grid) is marched from the **same initial condition** (the converged
``n = 64`` field prolonged, exactly the initial condition the baseline row used)
with different Courant numbers, everything else untouched:

* identical tolerance (``1e-7``) and check cadence, so "converged" means the same
  thing for every row;
* identical initial condition, so the comparison contains the transient and is not
  trivially satisfied by starting *at* the fixed point;
* one artefact per Courant number under ``runs/dt/cfl<value>/`` with the same
  schema as the baseline row, so ``uncertainty.py`` can consume it directly.

A run that becomes non-finite is *not* silently dropped: the row is written to
``runs/dt/cfl<value>/diverged.json`` with the step at which it broke and the
process exits non-zero, because a stability edge is a result that has to be
visible, not an exception that disappears.

Usage::

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/dt_study.py \
        --n 128 --init-n 64 --cfls 0.075,0.300 --tolerance 1e-07 --max-steps 400000
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tensorfvm.case_cavity_re1000 import (  # noqa: E402
    REYNOLDS, prolong, run_resolution, write_json,
)
from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver  # noqa: E402

DEFAULT_RUNS = HERE / "runs"


class Diverged(RuntimeError):
    """Raised from the in-loop guard so a blow-up is recorded, not truncated."""


def make_guard(limit: float = 10.0):
    """Stop the march when the velocity stops being a velocity."""

    def guard(solver: LidCavitySolver, step: int, change: float) -> None:
        if not math.isfinite(change):
            raise Diverged(f"non-finite change rate at step {step}")
        if not bool(torch.isfinite(solver.u).all()) or not bool(torch.isfinite(solver.v).all()):
            raise Diverged(f"non-finite velocity at step {step}")
        if change > limit:
            raise Diverged(f"change rate {change:.3e} > {limit:g} at step {step}")

    return guard


def load_pt(path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor]:
    blob = torch.load(path, weights_only=False)
    return blob["u"], blob["v"]


def initial_condition(n: int, init_n: int, runs: pathlib.Path):
    """Prolong the converged coarse field, i.e. the baseline row's own start."""
    if init_n >= n:
        return None, "rest"
    u, v = load_pt(runs / "fields" / f"Re1000_n{init_n}.pt")
    coarse = LidCavitySolver(LidCavityConfig(n=init_n, reynolds=REYNOLDS))
    coarse.u, coarse.v = u, v
    return prolong(coarse, n), f"n={init_n} bilinear prolongation (baseline initial condition)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--init-n", type=int, default=64)
    ap.add_argument("--cfls", default="0.075,0.300")
    ap.add_argument("--tolerance", type=float, default=1e-07)
    ap.add_argument("--max-steps", type=int, default=400_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    runs = pathlib.Path(args.runs_dir)
    n, init = args.n, args.init_n
    init_fields, init_from = initial_condition(n, init, runs)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failures: list[str] = []

    for cfl in [float(c) for c in args.cfls.split(",") if c]:
        tag = f"cfl{cfl:05.3f}".replace(".", "")
        out = runs / "dt" / tag
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"lid_cavity_Re1000_n{n}.json"
        if path.exists() and not args.force:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if payload and payload.get("converged"):
                print(f"[cached] cfl={cfl:g} steps={payload['steps']} "
                      f"residual={payload['residual']:.3e}", flush=True)
                continue

        print(f"[run] n={n} cfl={cfl:g} from {init_from}", flush=True)
        try:
            rec, solver = run_resolution(
                n, init=init_fields, init_from=init_from, cfl=cfl,
                tolerance=args.tolerance, max_steps=args.max_steps,
                on_check=make_guard(),
            )
        except Diverged as exc:
            failures.append(f"cfl={cfl:g}: {exc}")
            write_json(out / "diverged.json", {
                "case": "CASE-FVM-CAVITY-RE1000", "n": n, "cfl": cfl,
                "tolerance": args.tolerance, "max_steps": args.max_steps,
                "init_from": init_from, "reason": str(exc), "generated": stamp,
            })
            print(f"[diverged] cfl={cfl:g}: {exc}", flush=True)
            continue

        rec["generated"] = stamp
        rec["temporal_study"] = {
            "role": "same-grid, different-Courant row of the temporal band",
            "baseline_cfl": 0.15,
            "init_from": init_from,
            "grid_n": n,
            "command": ("PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/dt_study.py "
                        f"--n {n} --init-n {init} --cfls {args.cfls} "
                        f"--tolerance {args.tolerance:g} --max-steps {args.max_steps}"),
        }
        write_json(path, rec)
        torch.save({"u": solver.u, "v": solver.v}, out / f"Re1000_n{n}_fields.pt")
        print(f"[done] cfl={cfl:g} steps={rec['steps']} residual={rec['residual']:.3e} "
              f"wall={rec['wall_seconds']:.0f}s verdict={rec['verdict']}", flush=True)

    if failures:
        print("failures: " + "; ".join(failures), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()