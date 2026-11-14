#!/usr/bin/env python3
"""Grade the Ritter/Stoker dam-break anchors with the **2-D** shallow-water solver.

Every number quoted in ``docs/case-fvm-dambreak-stoker-2d/REPORT.md`` is
produced here::

    PYTHONPATH=src python scripts/run_shallow_water_dam_break_2d.py \
        --out src/tensorfvm/verification/artifacts/shallow-water-dam-break-2d

Outputs (in ``--out``): ``metrics_2d.json``, ``convergence_2d.json``,
``modulation_2d.json``, ``exactness_2d.json``, ``profile_2d_<case>_nx<n>.csv``,
``dam_break_2d.png``, ``SUMMARY_2D.md`` and ``manifest_2d.json``.

The 1-D benchmark (``scripts/run_shallow_water_dam_break.py``) remains the
primary evidence for the anchor; this driver adds the 2-D operator to the same
grading, plus the transverse-symmetry / transverse-modulation probes described
in :mod:`tensorfvm.shallow_water.dam_break2d`.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.shallow_water import CASES, TOLERANCE, exactness_checks
from tensorfvm.shallow_water.dam_break2d import (
    DEFAULT_NY,
    MODULATION_AMPLITUDE,
    TRANSVERSE_ROUNDOFF,
    compare_with_1d,
    convergence_2d,
    run_2d_case,
    transverse_modulation_study,
    write_artifacts_2d,
)


def _fmt(value: float) -> str:
    return f"{100.0 * value:.3f} %"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _figure(runs: dict, studies: dict, modulation: dict, path: Path,
            t_star: float) -> None:
    """Four-panel figure: profiles, convergence, transverse mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))

    for ax, key in zip(axes[0], ("ritter-dry", "stoker-wet")):
        run = runs[key]
        prof = run["profile"]
        x = np.asarray(prof["x"])
        case = run["case"]
        ax.plot(x, prof["h_exact"], "k-", lw=1.6, label="exact (closed form)")
        ax.plot(x, prof["h_numeric_y_average"], "r--", lw=1.6,
                label="TensorFVM 2-D (y-average)")
        m = run["metrics"]
        if case["kind"] == "dry":
            ax.axvline(m["front_exact"], color="k", ls=":", lw=1.0)
            ax.plot([m["front_exact"]], [0.0], "k>", ms=7,
                    label="exact front  x_f = %.4f m" % m["front_exact"])
            ax.plot([m["front_fan"]], [0.0], "r>", ms=7,
                    label="FVM front  %.4f m (%.3f %%)"
                          % (m["front_fan"], 100.0 * m["front_fan_error"]))
        else:
            ax.axvline(m["shock_exact"], color="k", ls=":", lw=1.0)
            ax.plot([m["shock_exact"]], [m["h1_exact"]], "ks", ms=6,
                    label="exact bore  %.4f m" % m["shock_exact"])
            ax.plot([m["bore_x"]], [m["h1"]], "r^", ms=7,
                    label="FVM bore  %.4f m (%.3f %%)"
                          % (m["bore_x"], 100.0 * m["bore_x_error"]))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("depth h [m]")
        ax.set_title("%s at t* = %g (nx = %d, ny = %d)"
                     % (key, t_star, run["discretisation"]["nx"],
                        run["discretisation"]["ny"]))
        ax.set_xlim(-4.0, 5.0)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3)

    ax = axes[1][0]
    for key, study in studies.items():
        dx = [row["dx"] for row in study["rows"]]
        ax.loglog(dx, [row["depth_l1"] for row in study["rows"]], "o-",
                  label="%s depth L1" % key)
        ax.loglog(dx, [max(row["headline_error"], 1e-8) for row in study["rows"]],
                  "s--", label="%s headline error" % key)
    dx_ref = np.array([studies[k]["rows"][0]["dx"] for k in studies])
    ref = 0.02 * dx_ref / dx_ref[0]
    ax.loglog(dx_ref, ref, "k:", lw=1.0, label="first-order reference")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("relative error")
    ax.set_title("2-D grid refinement (ny = %d fixed)" % DEFAULT_NY)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][1]
    for key, study in modulation.items():
        series = study["series"]
        ax.semilogy(series["t_star"], np.maximum(series["modulated_defect"], 1e-17),
                    "o-", ms=3,
                    label="%s modulated (a = %.0e)"
                          % (key, study["amplitude"]))
        ax.semilogy(series["t_star"], np.maximum(series["baseline_defect"], 1e-17),
                    "k:", lw=1.0, label="%s baseline (symmetry)" % key)
        ax.axhline(study["amplitude"], color="0.4", ls="--", lw=1.0)
    ax.set_xlabel("t*")
    ax.set_ylabel(r"transverse defect  $\max|h-\langle h\rangle_y|/h_0$")
    ax.set_title("Transverse mode: no amplification (dashed = initial amplitude)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=(
        "src/tensorfvm/verification/artifacts/shallow-water-dam-break-2d"))
    parser.add_argument("--nx", type=int, default=1600,
                        help="x resolution of the graded run")
    parser.add_argument("--ny", type=int, default=DEFAULT_NY,
                        help="y resolution (channel cells)")
    parser.add_argument("--resolutions", default="200,400,800",
                        help="x resolutions of the refinement study")
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--t-star", type=float, default=2.0)
    parser.add_argument("--modulation", type=float, default=MODULATION_AMPLITUDE)
    parser.add_argument("--mod-nx", type=int, default=400)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--skip-modulation", action="store_true")
    parser.add_argument("--consistency-nx", type=int, default=800,
                        help="x resolution of the 2-D vs 1-D operator comparison")
    parser.add_argument("--skip-consistency", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("== closed-form self-checks (reused from the 1-D driver) ==")
    exactness = exactness_checks()
    print("   Stoker momentum jump residual: %.3e"
          % exactness["stoker"]["momentum_jump_relative"])

    print(f"\n== 2-D graded runs (nx = {args.nx}, ny = {args.ny}, CFL = {args.cfl},"
          f" order = {args.order}, t* = {args.t_star:g}) ==")
    runs = {}
    for key, case in CASES.items():
        run = run_2d_case(case, nx=args.nx, ny=args.ny, cfl=args.cfl,
                          order=args.order, t_star=args.t_star)
        runs[key] = run
        m, disc = run["metrics"], run["discretisation"]
        print(f"\n   [{key}] nx={disc['nx']} ny={disc['ny']} dx={disc['dx']:.5f} m "
              f"dy={disc['dy']:.4f} m steps={disc['steps']} "
              f"({disc['wall_time_s']:.2f} s)")
        if case.kind == "dry":
            print(f"     front  exact {m['front_exact']:.6f} m  FVM {m['front_fan']:.6f} m"
                  f" (fan)  error {_fmt(m['front_fan_error'])}"
                  f"  [{'PASS' if run['verdict']['front_fan<5%']['pass'] else 'FAIL'}]")
            print(f"     front  wetted edge {m['front_wetted']:.6f} m  error"
                  f" {_fmt(m['front_wetted_error'])}  (diagnostic only)")
        else:
            print(f"     bore   exact {m['shock_exact']:.6f} m  FVM {m['bore_x']:.6f} m"
                  f"  error {_fmt(m['bore_x_error'])}"
                  f"  [{'PASS' if run['verdict']['bore_data_driven<5%']['pass'] else 'FAIL'}]")
            print(f"     h2/h1  exact {m['step_ratio_exact']:.6f}  FVM"
                  f" {m['step_ratio']:.6f}  error {_fmt(m['step_ratio_error'])}")
        print(f"     depth L1 {m['depth_l1']:.3e}  transverse defect"
              f" {m['transverse_defect']:.2e}  v_max {m['v_max']:.2e} m/s"
              f"  TV {m['tv_ratio']:.6f}  extrema {m['local_extrema']}")
        series = m["front_series"]
        print(f"     series x_f(t) max error"
              f" {'-' if series['max_error'] is None else _fmt(series['max_error'])}"
              f" over {series['samples_graded']}/{series['samples']} graded samples"
              f"  [{'PASS' if series['pass'] else 'FAIL'}]")
        print(f"     verdict: {'ALL PASS' if run['verdict']['all_pass'] else 'FAILED'}")

    studies = {}
    if not args.skip_convergence:
        resolutions = tuple(int(v) for v in args.resolutions.split(",") if v.strip())
        print("\n== 2-D grid refinement ==")
        for key, case in CASES.items():
            study = convergence_2d(case, resolutions=resolutions, ny=args.ny,
                                  cfl=args.cfl, order=args.order)
            studies[key] = study
            for row in study["rows"]:
                print(f"   {key:12s} nx={row['nx']:5d} dx={row['dx']:.6f}"
                      f" L1={row['depth_l1']:.3e}"
                      f" headline={_fmt(row['headline_error'])}"
                      f" y_defect={row['transverse_defect']:.1e}"
                      f" ({row['wall_time_s']:.1f} s)")
            print(f"   {key:12s} observed order L1 {study['observed_order_l1']:.2f}")

    modulation = {}
    if not args.skip_modulation:
        print(f"\n== transverse modulation probe (nx = {args.mod_nx}, a ="
              f" {args.modulation:g}) ==")
        for key, case in CASES.items():
            study = transverse_modulation_study(case, nx=args.mod_nx,
                                                ny=args.ny,
                                                amplitude=args.modulation)
            modulation[key] = study
            print(f"   {key:12s} baseline {_fmt(study['baseline_headline_error'])} ->"
                  f" modulated {_fmt(study['modulated_headline_error'])}"
                  f" (shift {100.0 * study['headline_error_shift']:+.3f} pp),"
                  f" growth {study['growth_factor']:.3f},"
                  f" v_max {study['v_max_modulated']:.2e} m/s,"
                  f" {'PASS' if all(c['pass'] for c in study['criteria'].values()) else 'FAIL'}")

    print("\n== 2-D versus 1-D operator (same problem, same nx) ==")
    consistency = {}
    if not args.skip_consistency:
        for key, case in CASES.items():
            comp = compare_with_1d(case, nx=args.consistency_nx, ny=args.ny,
                                   cfl=args.cfl, order=args.order,
                                   t_star=args.t_star)
            consistency[key] = comp
            for name, row in comp["rows"].items():
                print(f"   {key:12s} {name:16s} 2-D {100.0 * row['two_d']:.4f} %"
                      f"  1-D {100.0 * row['one_d']:.4f} %"
                      f"  |d| {100.0 * row['difference']:.4f} pp"
                      f"  [{'PASS' if row['difference'] <= comp['limit'] else 'FAIL'}]")

    figure_path = out / "dam_break_2d.png"
    if studies and modulation:
        _figure(runs, studies, modulation, figure_path, args.t_star)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tolerance": TOLERANCE,
            "t_star": args.t_star,
            "ny": args.ny,
            "graded_nx": args.nx,
            "transverse_roundoff_limit": TRANSVERSE_ROUNDOFF,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "command": "PYTHONPATH=src python scripts/run_shallow_water_dam_break_2d.py "
                       + " ".join(sys.argv[1:]),
        },
        "exactness": exactness,
        "runs": runs,
        "convergence": studies,
        "modulation": modulation,
        "consistency": consistency,
    }
    written = write_artifacts_2d(out, payload)
    payload["meta"]["wall_time_s"] = time.perf_counter() - started
    (out / "manifest_2d.json").write_text(
        json.dumps({"meta": payload["meta"], "artifacts": written},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n== artifacts written to {out} ==")
    for name in sorted(written):
        print(f"   {name}")
    print(f"   figure: {figure_path}")

    failures = [key for key, run in runs.items() if not run["verdict"]["all_pass"]]
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())