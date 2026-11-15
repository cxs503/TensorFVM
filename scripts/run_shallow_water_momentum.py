#!/usr/bin/env python3
"""Driver: momentum identity + velocity-field budget for CASE-FVM-DAMBREAK-STOKER.

Produces the artifacts quoted by ``docs/CASE-FVM-DAMBREAK-STOKER-MOMENTUM/``:

* ``momentum.json``  -- global momentum closure (closed form vs quadrature vs solver)
* ``velocity.json``  -- velocity L1 (unweighted and mass weighted) + error budget
                        + grid-refinement orders, both anchor cases
* ``control.json``   -- defect-injection sensitivity (the criteria are not empty)
* ``manifest.json``  -- reproducibility metadata (command, commit, versions)
* ``SUMMARY.md``     -- machine-written summary (numbers injected, never typed)
* ``fig_velocity_budget.png`` / ``fig_momentum.png``

Run: ``PYTHONPATH=src python scripts/run_shallow_water_momentum.py``
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET  # noqa: E402
from tensorfvm.shallow_water.momentum import (  # noqa: E402
    TOLERANCE,
    TIP_FRACTION,
    defect_injection_control,
    exact_momentum,
    exact_momentum_rate,
    momentum_closure,
    momentum_quadrature,
    similarity_identity_residual,
    undisturbed_guard,
    velocity_resolution_study,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D  # noqa: E402

CASES = (RITTER_DRY, STOKER_WET)
DEFAULT_OUT = Path("src/tensorfvm/verification/artifacts/shallow-water-momentum")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


def _profile(case, n):
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=0.4, order=2)
    solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
    solver.advance(case.t_end)
    h, u = solver.primitive()
    return solver, h, u


def figure_velocity(studies: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # (a) where the dry-bed velocity error lives
    ax = axes[0]
    case = RITTER_DRY
    h_ex_full = None
    for n, style in ((1600, "-"), (6400, "--")):
        solver, h, u = _profile(case, n)
        h_ex, u_ex = case.exact(solver.x)
        err = np.abs(u - np.asarray(u_ex)) / max(float(np.abs(u_ex).sum()), 1e-300)
        ax.plot(solver.x, err, style, lw=1.1, label=f"n = {n}")
        h_ex_full = np.asarray(h_ex)
    ax.axvline(case.exact_front(), color="k", lw=0.8, ls=":", label="analytic front")
    ax.set_xlim(2.6, 4.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$|u-u_{ex}|\,/\,\sum|u_{ex}|$  [1/m]")
    ax.set_title("(a) Ritter dry bed: error density is a tip phenomenon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) convergence of the three norms
    ax = axes[1]
    for key, label, style in (("u_l1", r"$\int|u-u_{ex}|\,dx$ (unweighted)", "o-"),
                              ("u_l1_mass", r"$\int|u-u_{ex}|\,h\,dx$ (mass weighted)", "s-"),
                              ("depth_l1", r"$\int|h-h_{ex}|\,dx$", "^-")):
        for case, colour in ((RITTER_DRY, "C0"), (STOKER_WET, "C1")):
            study = studies[case.key]
            ns = [row["n"] for row in study["rows"]]
            vals = [max(row[key], 1e-16) for row in study["rows"]]
            filled = case.key == "ritter-dry"
            ax.loglog(ns, vals, style, color=colour, lw=1.2,
                      markerfacecolor=colour if filled else "none",
                      label=f"{label} · {case.key}")
    ax.axhline(TOLERANCE, color="r", ls="--", lw=1.0, label="anchor 5 % gate")
    ax.set_xlabel("n")
    ax.set_ylabel("relative error")
    ax.set_title("(b) same case, three norms")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_momentum(closure: dict, control: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    for case, colour in ((RITTER_DRY, "C0"), (STOKER_WET, "C1")):
        rows = closure[case.key]["rows"]
        ns = [row["n"] for row in rows]
        ax.loglog(ns, [max(r["closure_error"], 1e-17) for r in rows], "o-",
                  color=colour, label=f"{case.key} · solver vs closed form")
        ax.loglog(ns, [max(r["quadrature_vs_closed_form"], 1e-17) for r in rows],
                  "x--", color=colour, alpha=0.7,
                  label=f"{case.key} · quadrature vs closed form")
    ax.axhline(1e-3, color="r", ls=":", lw=1.0, label="1e-3 defect floor")
    ax.set_xlabel("n")
    ax.set_ylabel("relative error")
    ax.set_title("(a) global momentum identity closes")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    rows = control["rows"]
    eps = [r["epsilon"] for r in rows]
    ax.loglog(eps, [max(r["momentum_measured"], 1e-18) for r in rows], "o-",
              label="momentum closure error (measured)")
    ax.loglog(eps, [max(r["momentum_predicted"], 1e-18) for r in rows], "k--", lw=1.0,
              label=r"analytic prediction $\epsilon\,\int_W|hu|/|M|$")
    ax.loglog(eps, [max(r["velocity_max_abs_delta"], 1e-18) for r in rows], "s-",
              color="C2", label="velocity L1 increment (max over sign)")
    ax.set_xlabel(r"injected relative defect $\epsilon$")
    ax.set_ylabel("measured response")
    ax.set_title("(b) both detectors respond linearly (gain 1)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resolutions", default="800,1600,3200,6400")
    parser.add_argument("--control-n", type=int, default=800)
    args = parser.parse_args(argv)

    resolutions = [int(v) for v in args.resolutions.split(",")]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    payload: dict = {"meta": {
        "command": "PYTHONPATH=src python scripts/run_shallow_water_momentum.py",
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "tolerance": TOLERANCE,
        "tip_fraction": TIP_FRACTION,
        "cases": [c.key for c in CASES],
        "resolutions": resolutions,
    }}

    # --- exact side -------------------------------------------------------
    payload["exact"] = {}
    for case in CASES:
        payload["exact"][case.key] = {
            "momentum_rate": exact_momentum_rate(case),
            "momentum_closed_form": exact_momentum(case),
            "momentum_quadrature": momentum_quadrature(case),
            "quadrature_relative": abs(momentum_quadrature(case)
                                       - exact_momentum(case)) / abs(exact_momentum(case)),
            "guard": undisturbed_guard(case),
        }
    payload["exact"]["similarity_identity_residual"] = similarity_identity_residual()

    # --- momentum closure -------------------------------------------------
    closure: dict = {}
    for case in CASES:
        rows = []
        for n in [r for r in resolutions if r <= 3200]:
            rows.append(momentum_closure(case, n=n))
        closure[case.key] = {"rows": rows}
    payload["momentum"] = closure

    # --- velocity field ---------------------------------------------------
    studies = {case.key: velocity_resolution_study(case, resolutions=resolutions)
               for case in CASES}
    payload["velocity"] = studies

    # --- sensitivity control ---------------------------------------------
    control = defect_injection_control(RITTER_DRY, n=args.control_n)
    payload["control"] = control

    (out / "momentum.json").write_text(json.dumps(
        {"meta": payload["meta"], "exact": payload["exact"],
         "momentum": payload["momentum"], "control": payload["control"]},
        indent=2, sort_keys=True), encoding="utf-8")
    (out / "velocity.json").write_text(json.dumps(
        {"meta": payload["meta"], "velocity": payload["velocity"]},
        indent=2, sort_keys=True), encoding="utf-8")
    (out / "control.json").write_text(json.dumps(
        {"meta": payload["meta"], "control": payload["control"]},
        indent=2, sort_keys=True), encoding="utf-8")

    figs = [figure_velocity(studies, out / "fig_velocity_budget.png"),
            figure_momentum(closure, control, out / "fig_momentum.png")]

    # --- machine-written summary -----------------------------------------
    lines = ["# Momentum identity + velocity budget — CASE-FVM-DAMBREAK-STOKER", ""]
    lines.append("All numbers below are injected from `momentum.json` / `velocity.json`.")
    lines.append("")
    lines.append("## Momentum identity (exact: `M(t) = g/2 (h_L^2 - h_R^2) t`)")
    lines.append("")
    lines.append("| case | rate [m^3/s^2] | M(t*) closed form | quadrature | quad rel err | guard |")
    lines.append("|---|---:|---:|---:|---:|:--:|")
    for case in CASES:
        e = payload["exact"][case.key]
        lines.append(f"| `{case.key}` | {e['momentum_rate']:.6f} | "
                     f"{e['momentum_closed_form']:.9f} | {e['momentum_quadrature']:.9f} | "
                     f"{e['quadrature_relative']:.2e} | "
                     f"{'ok' if e['guard']['undisturbed'] else 'DISTURBED'} |")
    lines += ["", "| case | n | solver M | exact M | closure rel err |",
              "|---|---:|---:|---:|---:|"]
    for case in CASES:
        for row in closure[case.key]["rows"]:
            lines.append(f"| `{case.key}` | {row['n']} | {row['momentum_numeric']:.9f} | "
                         f"{row['momentum_exact_closed_form']:.9f} | {row['closure_error']:.2e} |")
    lines += ["", "## Velocity field (never graded by the anchor's G1-G6)", "",
              "| case | n | u L1 | u L1 (mass) | hu L1 | depth L1 | missing-tip share | 5 % |",
              "|---|---:|---:|---:|---:|---:|---:|:--:|"]
    for case in CASES:
        for row in studies[case.key]["rows"]:
            lines.append(
                f"| `{case.key}` | {row['n']} | {row['u_l1']:.4%} | {row['u_l1_mass']:.4%} | "
                f"{row['hu_l1']:.4%} | {row['depth_l1']:.4%} | "
                f"{row['share_missing_tip']:.1%} | "
                f"{'PASS' if row['pass_5pct'] else 'FAIL'} |")
    lines += ["", "Observed orders (last to first):"]
    for case in CASES:
        s = studies[case.key]
        lines.append(f"* `{case.key}` u L1: {['%.2f' % o for o in s['observed_order_u_l1']]}, "
                     f"mass weighted: {['%.2f' % o for o in s['observed_order_u_l1_mass']]}, "
                     f"depth: {['%.2f' % o for o in s['observed_order_depth_l1']]}")
    lines += ["", "## Sensitivity control (Ritter, n = %d, window x in [%.2f, %.2f])"
              % (control["n"], control["window"][0], control["window"][1]), "",
              "| eps | momentum predicted | momentum measured | gain | "
              "velocity L1 max delta | velocity predicted |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in control["rows"]:
        lines.append(f"| {row['epsilon']:.0e} | {row['momentum_predicted']:.3e} | "
                     f"{row['momentum_measured']:.3e} | {row['momentum_gain']:.3f} | "
                     f"{row['velocity_max_abs_delta']:.3e} | "
                     f"{row['velocity_predicted']:.3e} |")
    lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "artifacts": sorted(p.name for p in out.glob("*") if p.is_file()),
        "meta": payload["meta"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                       encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {len(manifest['artifacts'])} artifacts to {out}")
    for fig in figs:
        print(f"figure: {fig}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())