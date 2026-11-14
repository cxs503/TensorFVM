"""CASE-CAVITY-RE1000-FVM: lid-driven square cavity at Re = 1000 vs Ghia et al. (1982).

What this module is
-------------------
The canonical, importable implementation of the case-pool entry
``CASE-CAVITY-RE1000-FVM`` -- one *case*, self-contained and machine-checkable,
in the same spirit as :mod:`tensorfvm.case_cavity_re100` (which does Re = 100).
It runs :class:`tensorfvm.lid_cavity.LidCavitySolver` at a requested resolution,
scores the two mid-plane profiles point by point against the Ghia, Ghia & Shin
(1982) tables, evaluates an explicit list of acceptance criteria, and writes a
JSON bundle.  Reference numbers are **never retyped here**: they come from
:func:`tensorfvm.lid_cavity_reference.table`, the single transcription that the
reference-audit tests pin (``tests/test_lid_cavity_reference_audit.py``).

Case definition (Ghia et al. 1982, Sec. 2)

    domain      [0, L] x [0, L], L = 1
    walls       no-slip on x = 0, x = L and y = 0
    lid         u = U = 1 on y = L (tangential, no penetration)
    Reynolds    Re = U L / nu = 1000  ->  nu = 1e-3 (rho = 1)
    outputs     u(y) on x = L/2          -> Table I
                v(x) on y = L/2          -> Table II
                primary vortex centre    -> (0.5313, 0.5625) L

Cost strategy (see ``docs/CASE-CAVITY-RE1000-FVM/method-selection.md``)
----------------------------------------------------------------------
The shipped discretisation is an explicit projection scheme, so the step count
to a steady state grows like ``n`` (information must cross the box at speed U).
Two standard, purely *cost* devices are applied -- neither changes the discrete
steady state that is scored:

* **grid continuation** -- converge coarse, prolongate the cell-centred velocity
  bilinearly onto the fine grid, keep marching.  It is *checked*, not trusted:
  the base grid is also run cold and the two converged fields are compared.
* **checkpoints** every ``--checkpoint-every`` steps, so a run that hits its cap
  still leaves the rows it finished.

Usage
-----
    PYTHONPATH=src python -m tensorfvm.case_cavity_re1000 \
        --out docs/CASE-CAVITY-RE1000-FVM/artifacts --grids 64,128 --cold-grids 32

The CLI is the *entry point*; ``scripts/run_cavity_re1000_m15458.py`` is a thin
wrapper kept for the round-2 record.  Integration side: the same record is
emitted in the schema read by
:func:`tensorfvm.verification.ladder.lid_cavity_verdict` (``--ladder-dir``), so
the repository's C2 cavity criterion consumes this run directly instead of a
separately produced one.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import torch

from .lid_cavity import (
    LidCavityConfig,
    LidCavitySolver,
    ghia_acceptance,
    score_against_ghia,
)
from .lid_cavity_reference import VORTEX_CENTRE, table

__all__ = [
    "CASE_SLUG",
    "REYNOLDS",
    "TOLERANCE_FRACTION",
    "REFERENCE",
    "CRITERIA",
    "CaseRunConfig",
    "cell_centred",
    "prolong",
    "run_case",
    "judge",
    "ladder_row",
    "build_bundle",
    "write_bundle",
    "main",
]

CASE_SLUG = "CASE-CAVITY-RE1000-FVM"
REYNOLDS = 1000.0
LID_VELOCITY = 1.0
REFERENCE = ("U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, "
             "Table I (u at x = 0.5) and Table II (v at y = 0.5)")
#: case-pool acceptance band: "< 5 %" on the key quantities.
TOLERANCE_FRACTION = 0.05

#: The case-pool tolerance turned into named, individually reported checks, so a
#: partial pass cannot hide behind an average.  ``source`` says where the
#: threshold comes from; ``metric`` names the key of :func:`judge`'s dict.
CRITERIA: tuple[dict[str, object], ...] = (
    {"id": "C1_u_extremum_rel", "metric": "u_min_rel", "threshold": 0.05,
     "comparator": "le",
     "statement": "relative error of the reverse-flow minimum u (Table I) < 5 %"},
    {"id": "C2_v_extremum_rel", "metric": "v_max_rel", "threshold": 0.05,
     "comparator": "le",
     "statement": "relative error of the maximum v (Table II) < 5 %"},
    {"id": "C3_vortex_centre_abs", "metric": "vortex_centre_abs", "threshold": 0.05,
     "comparator": "le",
     "statement": "primary-vortex centre within 0.05 L of (0.5313, 0.5625)"},
    {"id": "C4_u_profile_max_abs", "metric": "u_max_abs", "threshold": 0.05,
     "comparator": "le",
     "statement": "max |u - u_Ghia| on x = 0.5 below 5 % of the lid speed"},
    {"id": "C5_v_profile_max_abs", "metric": "v_max_abs", "threshold": 0.05,
     "comparator": "le",
     "statement": "max |v - v_Ghia| on y = 0.5 below 5 % of the lid speed"},
    {"id": "C6_converged", "metric": "converged", "threshold": True,
     "comparator": "is",
     "statement": "steady state reached: max|du|/dt <= tolerance"},
    {"id": "C7_divergence", "metric": "max_divergence", "threshold": 1e-10,
     "comparator": "le",
     "statement": "discrete divergence of the converged field below 1e-10"},
)


class CaseRunConfig:
    """Numerics of one resolution of this case (defaults = shipped ladder)."""

    def __init__(self, n: int, *, reynolds: float = REYNOLDS, cfl: float = 0.15,
                 tolerance: float = 1e-7, max_steps: int = 250_000,
                 check_every: int = 250) -> None:
        self.n = int(n)
        self.reynolds = float(reynolds)
        self.cfl = float(cfl)
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)
        self.check_every = int(check_every)

    def solver_config(self) -> LidCavityConfig:
        return LidCavityConfig(n=self.n, reynolds=self.reynolds, cfl=self.cfl,
                               max_steps=self.max_steps, tolerance=self.tolerance,
                               check_every=self.check_every)

    def to_dict(self) -> dict:
        return {"n": self.n, "reynolds": self.reynolds, "cfl": self.cfl,
                "tolerance": self.tolerance, "max_steps": self.max_steps,
                "check_every": self.check_every}


# --------------------------------------------------------------------- fields
def cell_centred(solver: LidCavitySolver) -> tuple[torch.Tensor, torch.Tensor]:
    """Staggered face velocities -> cell-centred (n, n) velocities."""
    return (0.5 * (solver.u[:-1, :] + solver.u[1:, :]),
            0.5 * (solver.v[:, :-1] + solver.v[:, 1:]))


def prolong(solver: LidCavitySolver, n_fine: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear prolongation of the cell-centred velocity onto an ``n_fine`` grid.

    ``F.interpolate(..., align_corners=False)`` treats the samples as cell
    centres, which is exactly the collocation both grids use, so the coarse
    field is sampled at the fine cell centres without a half-cell shift.  The
    fine face fields are rebuilt by averaging; wall faces stay zero.
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
def run_case(n: int, *, out_dir: str | os.PathLike, reynolds: float = REYNOLDS,
             cfl: float = 0.15, tolerance: float = 1e-7, max_steps: int = 250_000,
             check_every: int = 250, init: tuple[torch.Tensor, torch.Tensor] | None = None,
             init_from: str | None = None, threads: int | None = None,
             checkpoint_every: int = 20_000,
             resume: bool = True,
             fields_name: str | None = None) -> tuple[dict, LidCavitySolver]:
    """Run one resolution to a steady state and score it against the Ghia tables.

    Returns the record that goes into the bundle plus the live solver (the caller
    needs it for grid continuation).  ``init``/``init_from`` record how the run
    was started, because a continued run and a cold run must reach the same
    steady state -- that equality is what the continuation check verifies.

    ``fields_name`` overrides the checkpoint file stem (default
    ``Re<Re>_n<n>.pt``).  Two runs of the *same* grid that must stay independent
    -- a cold run and its continuation check -- have to use different stems and
    ``resume=False``; otherwise the second run would silently resume the first
    one's converged field and "confirm" the continuation without solving it.
    """
    torch.set_num_threads(int(threads if threads is not None
                              else os.environ.get("TFVM_THREADS", "1")))
    out_dir = pathlib.Path(out_dir)
    stem = fields_name or f"Re{reynolds:g}_n{n}.pt"
    fields_path = out_dir / "fields" / stem
    cfg = CaseRunConfig(n, reynolds=reynolds, cfl=cfl, tolerance=tolerance,
                        max_steps=max_steps, check_every=check_every)
    solver = LidCavitySolver(cfg.solver_config())
    if init is not None:
        solver.u, solver.v = init[0].clone(), init[1].clone()
    resumed = False
    if resume and fields_path.exists():
        saved = torch.load(fields_path, weights_only=False)
        solver.u, solver.v, solver.steps = saved["u"], saved["v"], int(saved["steps"])
        resumed = True
    started = time.time()
    state = {"last_flush": started, "next_save": checkpoint_every}

    def on_check(sv, step, change):
        now = time.time()
        if now - state["last_flush"] > 120.0:
            print(f"      Re={reynolds:g} n={n} step {step} rate {change:.3e} "
                  f"({now - started:.0f}s)", flush=True)
            state["last_flush"] = now
        if step >= state["next_save"]:
            fields_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"u": sv.u, "v": sv.v, "steps": sv.steps}, fields_path)
            state["next_save"] = step + checkpoint_every

    info = solver.run(on_check=on_check)
    wall = time.time() - started
    fields_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"u": solver.u, "v": solver.v, "steps": solver.steps}, fields_path)

    score = score_against_ghia(solver, reynolds)
    gate = ghia_acceptance(score, allow_registered_disputes=True)
    record = {
        "case": CASE_SLUG,
        "reynolds": float(reynolds),
        "n": int(n),
        "config": cfg.to_dict(),
        "solver": {
            "steps": int(solver.steps),
            "converged": bool(info.get("converged", False)),
            "residual_rate": float(info.get("residual", float("nan"))),
            "wall_seconds": wall,
            "resumed_from_checkpoint": resumed,
            "fields_file": stem,
            "init": "rest" if init is None else (init_from or "provided"),
            "threads": int(torch.get_num_threads()),
            "dtype": "float64",
        },
        "u": score["u"],
        "v": score["v"],
        "vortex_centre": {"computed": list(score["vortex_centre"]["computed"]),
                          "reference": list(VORTEX_CENTRE[float(reynolds)]),
                          "abs_error": max(abs(score["vortex_centre"]["computed"][0]
                                               - VORTEX_CENTRE[float(reynolds)][0]),
                                           abs(score["vortex_centre"]["computed"][1]
                                               - VORTEX_CENTRE[float(reynolds)][1]))},
        "max_divergence": float(score["max_divergence"]),
        "acceptance": gate,
    }
    record["verdict"] = judge(record)
    return record, solver


# -------------------------------------------------------------------- judging
def judge(record: dict) -> dict:
    """Evaluate :data:`CRITERIA` against the numbers in ``record``.

    Pointwise centreline deviations are in units of the lid speed, the two
    extrema as relative errors, the vortex centre as an absolute distance in L.
    Every criterion is reported individually, so a partial pass cannot hide.
    """
    u, v = record["u"], record["v"]
    metrics = {
        "u_min_rel": abs(u["extremum"]["min_computed"][1]
                         - u["extremum"]["min_reference"][1])
        / abs(u["extremum"]["min_reference"][1]),
        "v_max_rel": abs(v["extremum"]["max_computed"][1]
                         - v["extremum"]["max_reference"][1])
        / abs(v["extremum"]["max_reference"][1]),
        "vortex_centre_abs": float(record["vortex_centre"]["abs_error"]),
        "u_max_abs": float(u["max_abs_error"]),
        "v_max_abs": float(v["max_abs_error"]),
        "converged": bool(record["solver"]["converged"]),
        "max_divergence": float(record["max_divergence"]),
    }
    checks = []
    for crit in CRITERIA:
        value = metrics[crit["metric"]]
        if crit["comparator"] == "is":
            ok = value is bool(crit["threshold"])
        else:
            ok = float(value) <= float(crit["threshold"])
        checks.append({"id": crit["id"], "metric": crit["metric"],
                       "value": value, "threshold": crit["threshold"],
                       "comparator": crit["comparator"], "pass": bool(ok),
                       "statement": crit["statement"]})
    return {"case": CASE_SLUG, "tolerance": TOLERANCE_FRACTION,
            "checks": checks, "metrics": metrics,
            "pass": bool(all(c["pass"] for c in checks))}


def ladder_row(record: dict) -> dict:
    """The same run in the schema read by ``verification.ladder.lid_cavity_verdict``.

    This is the integration seam: the C2 criterion of the repository's
    verification ladder is stated at Re = 1000 on the finest available grid, and
    it reads ``lid_cavity_Re<Re>_n<n>.json`` with these keys.
    """
    return {
        "reynolds": record["reynolds"],
        "n": record["n"],
        "steps": record["solver"]["steps"],
        "converged": record["solver"]["converged"],
        "residual": record["solver"]["residual_rate"],
        "max_divergence": record["max_divergence"],
        "wall_seconds": record["solver"]["wall_seconds"],
        "u": record["u"],
        "v": record["v"],
        "vortex_centre": record["vortex_centre"],
    }


# -------------------------------------------------------------------- bundle
def build_bundle(*, out_dir: str | os.PathLike, grids: tuple[int, ...] = (64, 128),
                 control_grids: tuple[int, ...] = (32,), reynolds: float = REYNOLDS,
                 cfl: float = 0.15, tolerance: float = 1e-7,
                 max_steps: int = 250_000, check_every: int = 250,
                 threads: int | None = None, continuation_check: bool = True,
                 on_case=None) -> dict:
    """Run the whole grid sequence and assemble the bundle dictionary.

    ``control_grids`` are deliberately under-resolved rows kept as a documented
    control: they must stay reported (and failing), so a 5 % pass cannot hide a
    coarse grid.  ``continuation_check`` re-runs the base grid cold *and*
    continued from the coarsest control, then compares the two converged fields.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "case": CASE_SLUG,
        "anchor": CASE_SLUG,
        "description": "2-D lid-driven square cavity, Re = U L / nu = 1000",
        "reference": REFERENCE,
        "reynolds": float(reynolds),
        "lid_velocity": LID_VELOCITY,
        "tolerance_fraction": TOLERANCE_FRACTION,
        "criteria": [dict(c) for c in CRITERIA],
        "reference_table": {"u": [list(p) for p in table(float(reynolds), "u")],
                            "v": [list(p) for p in table(float(reynolds), "v")],
                            "vortex_centre": list(VORTEX_CENTRE[float(reynolds)])},
        "cases": [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def add(record: dict) -> None:
        summary["cases"] = [c for c in summary["cases"] if c["n"] != record["n"]] + [record]
        summary["cases"].sort(key=lambda c: c["n"])
        summary["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # the bundle is rewritten after every completed case: a run that is
        # interrupted still leaves a valid, self-describing artefact behind
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
        if on_case is not None:
            on_case(record)

    common = dict(out_dir=out_dir, reynolds=reynolds, cfl=cfl, tolerance=tolerance,
                  max_steps=max_steps, check_every=check_every, threads=threads)

    # 1. under-resolved control rows, cold from rest
    for n in control_grids:
        print(f"[control] Re={reynolds:g} n={n} (deliberately under-resolved)", flush=True)
        rec, _ = run_case(n, **common)
        rec["role"] = "control (under-resolved)"
        add(rec)

    # 2. base grid, cold from rest
    base = int(grids[0])
    print(f"[cold] Re={reynolds:g} n={base}", flush=True)
    base_rec, base_solver = run_case(base, **common)
    base_rec["role"] = "production (cold start)"
    add(base_rec)

    # 3. finer grids, continued from the previous production grid
    prev_solver = base_solver
    for n in grids[1:]:
        u0, v0 = prolong(prev_solver, n)
        print(f"[continuation] Re={reynolds:g} n={n} from n={prev_solver.n}", flush=True)
        rec, prev_solver = run_case(n, init=(u0, v0),
                                    init_from=f"n={prev_solver.n} prolongation", **common)
        rec["role"] = f"production (continued from n={rec['solver']['init'].split(' ')[0][2:]})"
        add(rec)

    # 4. continuation must not change the steady state: same grid, other start
    if continuation_check and control_grids:
        n0 = int(max(control_grids))
        saved = torch.load(out_dir / "fields" / f"Re{reynolds:g}_n{n0}.pt",
                           weights_only=False)
        src = LidCavitySolver(CaseRunConfig(n0, reynolds=reynolds).solver_config())
        src.u, src.v = saved["u"], saved["v"]
        u0, v0 = prolong(src, base)
        print(f"[continuation-check] Re={reynolds:g} n={base} from n={n0}", flush=True)
        rec, cont_solver = run_case(
            base, init=(u0, v0), init_from=f"n={n0} prolongation",
            resume=False,                                   # never resume the cold run
            fields_name=f"Re{reynolds:g}_n{base}_from{n0}.pt", **common)
        du = float((cont_solver.u - base_solver.u).abs().max())
        dv = float((cont_solver.v - base_solver.v).abs().max())
        summary["continuation_check"] = {
            "from_n": n0, "to_n": base,
            "max_abs_du": du, "max_abs_dv": dv,
            "same_steady_state": bool(max(du, dv) < 1e-6),
            "continued_steps": rec["solver"]["steps"],
            "cold_steps": base_rec["solver"]["steps"],
            "continued_wall_seconds": rec["solver"]["wall_seconds"],
            "cold_wall_seconds": base_rec["solver"]["wall_seconds"],
            "speedup": base_rec["solver"]["wall_seconds"]
            / max(rec["solver"]["wall_seconds"], 1e-9),
            "continued_verdict_pass": rec["verdict"]["pass"],
            "continued_fields_file": rec["solver"]["fields_file"],
            "cold_fields_file": base_rec["solver"]["fields_file"],
            "independent_of_cold_run": rec["solver"]["resumed_from_checkpoint"] is False,
        }
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)

    finisher(summary)
    return summary


def finisher(summary: dict) -> dict:
    """Fill the derived blocks (grid sequence, verdict per resolution)."""
    summary["cases"].sort(key=lambda c: c["n"])
    summary["grid_sequence"] = [
        {"n": c["n"], "role": c.get("role", ""),
         "u_min": c["u"]["extremum"]["min_computed"][1],
         "v_max": c["v"]["extremum"]["max_computed"][1],
         "u_max_abs": c["u"]["max_abs_error"],
         "v_max_abs": c["v"]["max_abs_error"],
         "steps": c["solver"]["steps"],
         "converged": c["solver"]["converged"]}
        for c in summary["cases"]]
    production = [c for c in summary["cases"]
                  if not c.get("role", "").startswith("control")]
    summary["verdict_5pct"] = {str(c["n"]): c["verdict"]["pass"] for c in summary["cases"]}
    summary["all_converged"] = all(c["solver"]["converged"] for c in summary["cases"])
    summary["production_rows"] = [c["n"] for c in production]
    summary["production_all_pass"] = bool(production) and all(c["verdict"]["pass"]
                                                             for c in production)
    summary["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return summary


def write_bundle(summary: dict, *, out_dir: str | os.PathLike,
                 ladder_dir: str | os.PathLike | None = None) -> dict:
    """Write ``summary.json`` (+ optional ladder rows) and return the file map."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    json.dump(finisher(summary), open(summary_path, "w"), indent=1)
    files = {"summary": str(summary_path)}
    if ladder_dir is not None:
        ladder_dir = pathlib.Path(ladder_dir)
        ladder_dir.mkdir(parents=True, exist_ok=True)
        files["ladder"] = []
        for case in summary["cases"]:
            if "control" in case.get("role", ""):
                continue        # the ladder consumes production rows only
            path = ladder_dir / f"lid_cavity_Re{int(case['reynolds'])}_n{int(case['n'])}.json"
            json.dump(ladder_row(case), open(path, "w"), indent=1)
            files["ladder"].append(str(path))
    return files


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="CASE-CAVITY-RE1000-FVM: lid-driven cavity at Re = 1000 "
                    "(Ghia et al. 1982) in TensorFVM.")
    ap.add_argument("--out", default="docs/CASE-CAVITY-RE1000-FVM/artifacts")
    ap.add_argument("--reynolds", type=float, default=REYNOLDS)
    ap.add_argument("--grids", default="64,128",
                    help="production grids, coarse -> fine (continued)")
    ap.add_argument("--cold-grids", default="32",
                    help="under-resolved control rows, cold from rest")
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--tolerance", type=float, default=1e-7)
    ap.add_argument("--cfl", type=float, default=0.15)
    ap.add_argument("--check-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch threads (default: $TFVM_THREADS or 1)")
    ap.add_argument("--ladder-dir", default=None,
                    help="also emit ladder-schema rows for "
                         "tensorfvm.verification.ladder.lid_cavity_verdict")
    ap.add_argument("--no-continuation-check", action="store_true")
    args = ap.parse_args(argv)

    summary = build_bundle(
        out_dir=args.out,
        grids=tuple(int(g) for g in args.grids.split(",") if g),
        control_grids=tuple(int(g) for g in args.cold_grids.split(",") if g),
        reynolds=args.reynolds, cfl=args.cfl, tolerance=args.tolerance,
        max_steps=args.max_steps, check_every=args.check_every, threads=args.threads,
        continuation_check=not args.no_continuation_check)
    files = write_bundle(summary, out_dir=args.out, ladder_dir=args.ladder_dir)
    print(json.dumps({"grid_sequence": summary["grid_sequence"],
                      "verdict_5pct": summary["verdict_5pct"],
                      "files": files}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())