"""Campaign driver for anchor ``CASE-FVM-CAVITY-RE1000`` (2-D cavity, Re = 1000).

Runs the production solver through :mod:`tensorfvm.case_cavity_re1000` on a
grid sequence and writes the evidence bundle into ``runs/``:

    runs/lid_cavity_Re1000_n<n>.json    per-resolution artefact (schema consumed by
                                        tensorfvm.verification.ladder.lid_cavity_verdict)
    runs/summary.json                   case-level verdict, criteria, grid convergence
    runs/benchmark.json                 ref vs got numeric table (Ghia 1982 vs computed)
    runs/fields.npz                     converged velocity fields (re-score offline)
    runs/control/...                    cold-start control used for the continuation check

Grid sequence (default): n = 32 (cold) -> 64 (prolonged from 32) -> 128
(prolonged from 64), with the cold counter-check re-run at n = 64.  Continuation
changes only the cost: the cold n = 64 run is kept as the counter-check that the
fixed point is not selected by the prolongation.

The counter-check is only a test when the grid it is run on was *also* reached by
prolongation in this campaign — on a grid started from rest the two runs are the
same computation and their agreement is a determinism check, not evidence about
continuation.  ``select_control_grid`` therefore defaults to the coarsest
warm-started grid, and ``build_continuation_check`` withholds the "same steady
state" verdict when that condition does not hold instead of reporting agreement.

Every step is resumable: an already converged artefact is reused instead of
re-run (``--force`` re-runs).

Usage::

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/run_case.py
    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/run_case.py --grids 32,64
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from tensorfvm.case_cavity_re1000 import (  # noqa: E402
    CASE_SLUG, REYNOLDS, evaluate, prolong, run_resolution, save_fields, summarise,
    write_json,
)
from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "runs"


def load(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not payload.get("converged"):
        return None
    # criteria are always derived from the stored profiles, so a cached artefact
    # is re-scored with the current scoring code instead of trusting its verdict
    payload["criteria"] = evaluate(payload)
    checks = [v for k, v in payload["criteria"].items() if not k.startswith("_")]
    payload["verdict"] = "PASS" if all(v["pass"] for v in checks) else "FAIL"
    write_json(path, payload)
    return payload


def fields_of(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    blob = torch.load(path, weights_only=False)
    return {"u": blob["u"], "v": blob["v"]}


def warm_started_grids(artifacts: dict[int, dict]) -> list[int]:
    """Grids of this campaign whose steady state was reached from a coarser field."""
    return sorted(int(n) for n, rec in artifacts.items()
                  if str(rec.get("initial_condition", "rest")) != "rest")


def select_control_grid(grids, artifacts: dict[int, dict],
                        requested: int | None = None) -> tuple[int | None, bool]:
    """Pick the grid the cold counter-check runs on -> ``(n, warm_started)``.

    Default: the coarsest grid that was warm-started in this campaign, because that
    is the only place where "cold" and "prolonged" can disagree.  An explicit
    ``requested`` grid is honoured but reported as vacuous when this campaign
    started it from rest.
    """
    warm = warm_started_grids(artifacts)
    if requested is not None:
        n = int(requested)
        return n, n in warm
    if warm:
        return warm[0], True
    return (int(min(grids)) if grids else None), False


def build_continuation_check(continued: dict, control: dict, *, grid: int,
                             warm_started: bool, tol_same: float = 1e-5) -> dict:
    """Compare the prolonged fixed point with the cold one at the same grid.

    ``warm_started=False`` means the main record at this grid was itself started
    from rest, i.e. the two runs are the same computation: the agreement is a
    determinism check and says nothing about whether prolongation preserves the
    fixed point.  The verdict is then withheld (``None``) rather than claimed.
    """
    du = max(abs(a - b) for a, b in zip(continued["u"]["profiles"]["computed"],
                                        control["u"]["profiles"]["computed"]))
    dv = max(abs(a - b) for a, b in zip(continued["v"]["profiles"]["computed"],
                                        control["v"]["profiles"]["computed"]))
    check = {
        "grid": int(grid),
        "kind": "continuation" if warm_started else "repeat_run",
        "warm_started": bool(warm_started),
        "same_steady_state": bool(max(du, dv) < tol_same) if warm_started else None,
        "max_abs_du_centreline": du,
        "max_abs_dv_centreline": dv,
        "steps_continued": continued["steps"],
        "steps_cold": control["steps"],
        "wall_seconds_continued": continued["wall_seconds"],
        "wall_seconds_cold": control["wall_seconds"],
        "speedup": control["wall_seconds"] / max(continued["wall_seconds"], 1e-9),
    }
    if not warm_started:
        check["note"] = ("the main record at this grid also started from rest, so this "
                         "cold repeat is a determinism check only; it is NOT evidence "
                         "that prolongation preserves the fixed point")
    return check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--grids", default="32,64,128")
    ap.add_argument("--cold-control", type=int, default=None,
                    help="grid re-run cold as the continuation control; default = the "
                         "coarsest grid this campaign actually warm-started (a control "
                         "on a from-rest grid cannot test the continuation)")
    ap.add_argument("--tolerance", type=float, default=1e-07)
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--cfl", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    (out / "control").mkdir(parents=True, exist_ok=True)
    grids = [int(g) for g in args.grids.split(",") if g]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    artifacts: dict[int, dict] = {}
    solvers: dict[int, LidCavitySolver] = {}

    for n in grids:
        path = out / f"lid_cavity_Re1000_n{n}.json"
        fpath = out / "fields" / f"Re1000_n{n}.pt"
        cached, cached_fields = load(path), fields_of(fpath)
        if cached and cached_fields and not args.force:
            print(f"[cached] n={n} verdict={cached['verdict']}", flush=True)
            artifacts[n] = cached
            s = LidCavitySolver(LidCavityConfig(n=n, reynolds=REYNOLDS))
            s.u, s.v = cached_fields["u"], cached_fields["v"]
            solvers[n] = s
            continue

        init, init_from = None, None
        coarse = max([m for m in solvers if m < n], default=None)
        if coarse is not None:
            init = prolong(solvers[coarse], n)
            init_from = f"n={coarse} bilinear prolongation"
        try:
            rec, solver = run_resolution(n, init=init, init_from=init_from,
                                         tolerance=args.tolerance, max_steps=args.max_steps,
                                         cfl=args.cfl)
        except Exception as exc:  # keep honest partial evidence instead of losing the run
            print(f"[failed] n={n}: {type(exc).__name__}: {exc}", flush=True)
            break
        rec["generated"] = stamp
        write_json(path, rec)
        torch.save({"u": solver.u, "v": solver.v}, fpath)
        artifacts[n], solvers[n] = rec, solver

    # continuation control: the same grid reached cold, so the fixed point cannot
    # be an artefact of the prolongation.  The grid must be one that this campaign
    # warm-started, otherwise cold-vs-prolonged collapses into cold-vs-itself.
    ctrl_n, ctrl_warm = select_control_grid(grids, artifacts, args.cold_control)
    if ctrl_n is not None and not ctrl_warm:
        print(f"[warn] control at n={ctrl_n} cannot test the continuation: this campaign "
              f"started n={ctrl_n} from rest (warm-started grids: "
              f"{warm_started_grids(artifacts) or 'none'}); it is recorded as a "
              f"determinism check with the verdict withheld", flush=True)

    continuation_check = None
    if ctrl_n is not None:
        ctrl_path = out / "control" / f"lid_cavity_Re1000_n{ctrl_n}.json"
        ctrl_cached = load(ctrl_path)
        if ctrl_cached and not args.force:
            print(f"[cached] control n={ctrl_n}", flush=True)
            control = ctrl_cached
        else:
            print(f"[control] cold restart at n={ctrl_n}", flush=True)
            control, _ = run_resolution(ctrl_n, tolerance=args.tolerance,
                                        max_steps=args.max_steps, cfl=args.cfl)
            control["generated"] = stamp
            write_json(ctrl_path, control)

        continued = artifacts.get(ctrl_n)
        if continued is not None:
            continuation_check = build_continuation_check(
                continued, control, grid=ctrl_n, warm_started=ctrl_warm)

    rows = [artifacts[n] for n in sorted(artifacts)]
    summary = summarise(rows)
    summary["continuation_check"] = continuation_check
    summary["generated"] = stamp
    summary["command"] = ("PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/run_case.py "
                          f"--grids {','.join(str(n) for n in grids)} "
                          f"--cold-control {ctrl_n} --tolerance {args.tolerance:g} "
                          f"--max-steps {args.max_steps}")
    write_json(out / "summary.json", summary)
    write_json(out / "benchmark.json", {
        "case": CASE_SLUG,
        "reference": summary["reference"],
        "tolerance": summary["pool_tolerance"],
        "rows": summary["benchmark"],
        "verdict": summary["verdict"],
        "finest_verdict": summary["finest_verdict"],
    })
    save_fields(out / "fields.npz", solvers)
    if len(rows) >= 3:
        # discretisation uncertainty of this very sequence (Roache GCI); a separate
        # artefact so the case gate above is untouched
        from tensorfvm.cavity_uncertainty import study as uq_study
        write_json(out / "uncertainty_spatial.json",
                   uq_study({n: (solvers[n].u.detach(), solvers[n].v.detach())
                             for n in sorted(solvers)}))
    print(json.dumps({"resolutions": summary["resolutions"],
                      "benchmark": summary["benchmark"],
                      "continuation_check": continuation_check}, indent=1))


if __name__ == "__main__":
    main()