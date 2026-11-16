#!/usr/bin/env python
"""Run the third-code cross-validation of ``CASE-FVM-DAMBREAK-STOKER``.

The third code is Clawpack 5.14.0 Classic (Fortran kernels) with two Riemann
solvers; see :mod:`tensorfvm.verification.dambreak_third_code` for why the case
pool needs a witness that is not in this package.  Everything the report quotes
is written here -- the JSON, the profile archive and the figure -- so no number
in ``docs/case-fvm-dambreak-stoker-third-code/REPORT.md`` is hand copied.

Usage::

    PYTHONPATH=src python scripts/run_dambreak_third_code.py            # ~40 s
    PYTHONPATH=src python scripts/run_dambreak_third_code.py --quick    # 2 grids
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.verification.dambreak_third_code import (  # noqa: E402
    CASES,
    DEFAULT_GRIDS,
    _jsonable,
    run_clawpack,
    run_tensorfvm,
    third_code_study,
)

ARTIFACT_DIR = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorfvm"
                / "verification" / "artifacts" / "shallow-water-third-code")
REPORT_DIR = (pathlib.Path(__file__).resolve().parents[1] / "docs"
              / "case-fvm-dambreak-stoker-third-code")


def _figure(study, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wet = CASES["stoker-wet"]
    grids = [int(n) for n in study["wet"]]
    n = max(grids)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    # (a) profiles at the finest grid, third code vs graded solver vs exact
    ax = axes[0][0]
    profiles = [run_tensorfvm(wet, n), run_clawpack(wet, n, "roe"),
                run_clawpack(wet, n, "hlle")]
    from tensorfvm.shallow_water.exact import stoker_state

    xe = np.linspace(wet.x_min, wet.x_max, 4001)
    he, _ = stoker_state(xe, wet.t_end, wet.h0, wet.h1, wet.gravity, wet.x_dam)
    ax.plot(xe, he, "k-", lw=2.2, label="Stoker 1957 exact")
    style = {"tensorfvm-hll-muscl-ssprk3": ("C0", "o", "TensorFVM HLL+MUSCL"),
             "clawpack-classic-roe": ("C1", "s", "Clawpack Roe+efix"),
             "clawpack-classic-hlle": ("C3", "^", "Clawpack HLLE")}
    for prof in profiles:
        col, mk, lab = style[prof["code"]]
        ax.plot(prof["x"][::8], prof["h"][::8], mk, ms=3.0, color=col, alpha=0.85,
                mfc="none", label=f"{lab} (n={n})")
    ax.set_xlabel("x [m]"); ax.set_ylabel("h [m]")
    ax.set_title(f"(a) wet-bed Stoker, t*=2, n={n}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) cross-code distance vs grid
    ax = axes[0][1]
    for code, col in (("clawpack-classic-roe", "C1"), ("clawpack-classic-hlle", "C3")):
        vals = [100.0 * study["wet"][str(g)]["cross_code"][code]["depth_l1"] for g in grids]
        ax.semilogy(grids, vals, "o-", color=col, label=code.replace("clawpack-classic-", ""))
    ax.axhline(100.0 * study["tolerance"], color="r", ls="--", lw=1.2,
               label=f"pool tolerance {100 * study['tolerance']:.0f} %")
    ax.set_xlabel("n"); ax.set_ylabel("relative depth L1 [%]")
    ax.set_title("(b) TensorFVM vs independent code, same grid")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (c) the anchor's two headline readings on the foreign profile
    ax = axes[1][0]
    for code, col in (("tensorfvm-hll-muscl-ssprk3", "C0"),
                      ("clawpack-classic-roe", "C1"),
                      ("clawpack-classic-hlle", "C3")):
        bore = [100.0 * study["wet"][str(g)]["codes"][code]["bore_error"] for g in grids]
        step = [100.0 * study["wet"][str(g)]["codes"][code]["step_error"] for g in grids]
        ax.loglog(grids, bore, "o-", color=col, label=f"{code} bore x_s")
        ax.loglog(grids, step, "s--", color=col, alpha=0.7, label=f"{code} step h2/h1")
    ax.axhline(100.0 * study["tolerance"], color="r", ls="--", lw=1.2)
    ax.set_xlabel("n"); ax.set_ylabel("error vs exact [%]")
    ax.set_title("(c) the same instrument on every code")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3, which="both")

    # (d) dry-limit ladder: bore front vs downstream depth h1
    ax = axes[1][1]
    ladder = sorted(((float(k), v) for k, v in study["dry_limit"].items()), key=lambda t: -t[0])
    hs = [k for k, _ in ladder]
    grids_l = sorted({int(n) for _, rungs in ladder for n in rungs})
    n_l = grids_l[-1]
    for code, col, mk in (("tensorfvm-hll-muscl-ssprk3", "C0", "o"),
                          ("clawpack-classic-roe", "C1", "s"),
                          ("clawpack-classic-hlle", "C3", "^")):
        errs = []
        for h1, rungs in ladder:
            row = rungs[str(n_l)]["codes"].get(code, {})
            value = row.get("bore_error") if row.get("ok") else None
            errs.append(100.0 * value if value is not None else np.nan)
        ax.loglog(hs, errs, mk + "-", color=col, label=code.replace("clawpack-classic-", ""))
    ax.axhline(100.0 * study["tolerance"], color="r", ls="--", lw=1.2)
    ax.axvline(1e-3, color="grey", ls=":", lw=1.0)
    ax.text(1.2e-3, 0.02, "third code non-finite\nbelow ~1e-3", fontsize=7, color="grey")
    ax.set_xlabel("downstream depth h1 [m] (h1 -> 0 = Ritter dry bed)")
    ax.set_ylabel("bore-front error [%]")
    ax.set_title(f"(d) approach to the dry limit, n={n_l}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle("CASE-FVM-DAMBREAK-STOKER · third-code (Clawpack 5.14.0) cross-validation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two grids only")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    grids = (400, 1600) if args.quick else DEFAULT_GRIDS
    study = third_code_study(grids)
    if not study.get("available"):
        print("clawpack unavailable -- nothing to do", file=sys.stderr)
        return 2

    from tensorfvm.verification.dambreak_third_code import (
        calibration_probes,
        ritter_front_position,
        stoker_constant_state,
        wettability_floor,
    )

    wet = CASES["stoker-wet"]
    max_n = max(grids)
    study["calibration"] = calibration_probes(run_clawpack(wet, max_n, "roe"), wet)
    study["cases"] = {k: v.summary() for k, v in CASES.items()}
    study["wettability_floor"] = wettability_floor(n=max_n)
    study["clawpack_version"] = __import__("clawpack").__version__
    dry = CASES["ritter-dry"]
    study["exact"] = {
        "ritter_front_x": float(ritter_front_position(dry.t_end, dry.h0, dry.gravity, dry.x_dam)),
        "stoker": {k: float(v) for k, v in
                   stoker_constant_state(wet.h0, wet.h1, wet.gravity).items()
                   if isinstance(v, (int, float))},
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(study)
    (ARTIFACT_DIR / "xval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # profile archive (wet case, finest grid + all table grids)
    archive = {}
    for n in grids:
        for name, prof in (("tfvm", run_tensorfvm(wet, n)),
                           ("clawpack-roe", run_clawpack(wet, n, "roe")),
                           ("clawpack-hlle", run_clawpack(wet, n, "hlle"))):
            if prof.get("ok"):
                archive[f"{name}_x_n{n}"] = prof["x"]
                archive[f"{name}_h_n{n}"] = prof["h"]
                archive[f"{name}_u_n{n}"] = prof["u"]
    np.savez_compressed(ARTIFACT_DIR / "profiles.npz", **archive)

    fig_path = None
    if not args.no_figure:
        fig_path = _figure(study, REPORT_DIR / "fig_third_code.png")

    verdict = study["verdict"]
    print(f"status: {verdict['status']}")
    for key, crit in verdict["criteria"].items():
        print(f"  {key}: {crit}")
    print(f"artifacts -> {ARTIFACT_DIR}")
    if fig_path:
        print(f"figure    -> {fig_path}")
    return 0 if verdict["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())