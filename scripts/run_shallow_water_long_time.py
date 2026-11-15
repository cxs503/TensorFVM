#!/usr/bin/env python
"""Long-time (self-similar) dam-break study -- runner and artifact writer.

Produces, under ``src/tensorfvm/verification/artifacts/shallow-water-long-time``:

``study.json``   the full graded payload (rows, speed fits, collapse, verdicts)
``SUMMARY.md``   the same tables rendered programmatically (zero hand copy)
``long_time_errors.png``   position/speed error vs ``t*`` with the 5 % band
``long_time_collapse.png`` self-similar profiles ``h(x/t)`` at every ``t*``

Usage::

    python scripts/run_shallow_water_long_time.py            # full study
    python scripts/run_shallow_water_long_time.py --quick    # coarse smoke
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.shallow_water import long_time as lt  # noqa: E402


def _fmt(value, unit="%", scale=100.0, digits=4):
    if value is None or value != value:
        return "n/a"
    return f"{value * scale:.{digits}f} {unit}"


def render_summary(payload: dict) -> str:
    crit = payload["criteria"]
    lines = [
        "# CASE-FVM-DAMBREAK-STOKER · long-time (self-similar) verification",
        "",
        "Programmatically generated from `study.json` -- do not edit by hand.",
        f"dx = {crit['dx']} m, CFL = {crit['cfl']}, order = {crit['order']}, "
        f"t* = {', '.join(str(t) for t in crit['t_stars'])}",
        f"tolerances: position {crit['tolerance']:.0%}, speed "
        f"{crit['speed_tolerance']:.0%}, collapse L1 "
        f"{crit['collapse_tolerance']:.1%} h0, step drift "
        f"{crit['step_drift_tolerance']:.2%}",
        "",
    ]
    for key, case in payload["cases"].items():
        lines += [f"## {key}", "",
                  "| t* | n | steps | x_f exact | x_f measured | position err "
                  "| x_f/t err | speed (fit) err | depth L1 | extrema | "
                  "mass drift |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in case["rows"]:
            lines.append(
                f"| {r['t_star']:g} | {r['n']} | {r['steps']} | "
                f"{r['front_exact']:.4f} | {r['front_measured']:.4f} | "
                f"{_fmt(r['front_series_max_error'])} | "
                f"{_fmt(r['x_over_t_error'])} | "
                f"{_fmt(r['speed'].get('speed_error'))} | "
                f"{_fmt(r['depth_l1'])} | {r['local_extrema']} | "
                f"{r['mass_drift']:.2e} |")
        lines.append("")
        wet = [r for r in case["rows"] if r.get("step_ratio_error") is not None
               and r.get("step_ratio_error") == r.get("step_ratio_error")]
        if wet:
            lines += ["| t* | h2 exact | h2 measured | h2 err | h1 err | "
                      "step h2/h1 err | bore position err | plateau cells |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for r in wet:
                lines.append(
                    f"| {r['t_star']:g} | {r['h2_exact']:.4f} | "
                    f"{r['h2_measured']:.4f} | {_fmt(r['h2_error'])} | "
                    f"{_fmt(r['h1_error'])} | "
                    f"{_fmt(r['step_ratio_error'])} | "
                    f"{_fmt(r['shock_error'])} | "
                    f"{r['plateau_window_cells']} |")
            lines.append("")
        lines += ["| t* | collapse L1/h0 vs t*=2 (numerical) | "
                  "collapse L1/h0 (exact, noise floor) |",
                  "|---:|---:|---:|"]
        for c, e in zip(case["similarity"]["collapse"],
                        case["similarity"]["exact_collapse"]):
            lines.append(f"| {c['t_star']:g} | {c['l1_over_h0']:.3e} | "
                         f"{e['l1_over_h0']:.3e} |")
        lines.append("")
        verdict = case["verdict"]
        lines.append(f"**verdict: {'PASS' if verdict['pass'] else 'FAIL'}**")
        lines.append("")
        lines += ["| criterion | value | limit | pass |",
                  "|---|---:|---:|:--:|"]
        for c in verdict["criteria"]:
            value = c["value"]
            shown = ("%.3e" % value) if abs(value) < 1e-3 else ("%.4f" % value)
            lines.append(f"| {c['name']} | {shown} | {c['limit']:g} | "
                         f"{'yes' if c['pass'] else 'no'} |")
        lines.append("")
    controls = payload.get("controls")
    if controls:
        lines += [f"## controls (t* = {controls['t_star']:g})", "",
                  "### CFL refinement at fixed dx (temporal vs spatial error)",
                  "", "| CFL | case | position err | x_f/t err | speed fit err "
                  "| depth L1 |",
                  "|---:|---|---:|---:|---:|---:|"]
        for cfl, rows in controls["cfl_refinement"].items():
            for key, r in rows.items():
                lines.append(
                    f"| {cfl} | {key} | {_fmt(r['front_error'])} | "
                    f"{_fmt(r['x_over_t_error'])} | "
                    f"{_fmt(r['speed'].get('speed_error'))} | "
                    f"{_fmt(r['depth_l1'])} |")
        lines += ["", "### scheme order (falsifiability: the metric must "
                  "detect a coarser scheme)", "",
                  "| order | case | position err | x_f/t err | speed fit err | "
                  "depth L1 | extrema |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for scheme, rows in controls["order_reference"].items():
            for key, r in rows.items():
                lines.append(
                    f"| {scheme} | {key} | {_fmt(r['front_error'])} | "
                    f"{_fmt(r['x_over_t_error'])} | "
                    f"{_fmt(r['speed'].get('speed_error'))} | "
                    f"{_fmt(r['depth_l1'])} | {r['local_extrema']} |")
        lines.append("")
    lines.append(f"wall time: {payload['wall_time_s']:.1f} s")
    return "\n".join(lines) + "\n"


def figure_errors(payload: dict, path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {"ritter-dry": "C0", "stoker-wet": "C3"}
    for key, case in payload["cases"].items():
        ts = [r["t_star"] for r in case["rows"]]
        axes[0].plot(ts, [100 * r["front_series_max_error"] for r in case["rows"]],
                     "o-", color=colors.get(key, "C2"),
                     label=f"{key}: |x_f - x_f,exact| / x_f,exact")
        axes[1].plot(ts, [100 * r["x_over_t_error"] for r in case["rows"]],
                     "o-", color=colors.get(key, "C2"),
                     label=f"{key}: |x_f/t - U| / U")
        axes[1].plot(ts, [100 * r["speed"].get("speed_error", np.nan)
                          for r in case["rows"]],
                     "s--", color=colors.get(key, "C2"), alpha=0.6,
                     label=f"{key}: LSQ fit speed")
    for ax, title in zip(axes, ["wave position error",
                                "wave speed error (time-invariant)"]):
        ax.axhline(5.0, color="k", ls=":", lw=1.0, label="5 % acceptance band")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t* = t sqrt(g h0)/h0")
        ax.set_ylabel("relative error [%]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Dam-break verification beyond t*=2 (dx = %.4g m, MUSCL+HLL, "
                 "SSP-RK)" % payload["criteria"]["dx"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pathlib.Path(path)


def figure_collapse(payload: dict, path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = payload["cases"]
    fig, axes = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 4.2),
                             squeeze=False)
    for ax, (key, case) in zip(axes[0], cases.items()):
        xi = np.asarray(case["similarity"]["xi"], dtype=float)
        sim = case["similarity"]
        for prof in sim["profiles"]:
            ax.plot(xi, np.asarray(prof["h"]), lw=1.0, alpha=0.9,
                    label=f"numerical t*={prof['t_star']:g}")
        ax.plot(xi, np.asarray(sim["exact_profile"]), "k--", lw=1.6,
                label="exact (self-similar)")
        ax.set_xlabel(r"$\xi = x/t$  [m/s]")
        ax.set_ylabel("depth h [m]")
        ax.set_title(f"{key}: self-similar collapse of h(x/t)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pathlib.Path(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=lt.DX)
    ap.add_argument("--t-stars", type=float, nargs="+", default=list(lt.T_STARS))
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--outdir", default="src/tensorfvm/verification/artifacts/"
                                      "shallow-water-long-time")
    ap.add_argument("--figure-dir", default="docs/case-fvm-dambreak-stoker-long-time")
    ap.add_argument("--quick", action="store_true",
                    help="coarse smoke run (not the graded artifacts)")
    args = ap.parse_args(argv)
    if args.quick:
        args.dx = max(args.dx, 2.0e-2)
        args.t_stars = [1.0, 2.0]
        args.no_controls = True
    payload = lt.run_long_time_study(t_stars=tuple(args.t_stars), dx=args.dx,
                                     controls=not args.no_controls)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "study.json").write_text(json.dumps(payload, indent=1))
    (outdir / "SUMMARY.md").write_text(render_summary(payload))
    figdir = pathlib.Path(args.figure_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    figure_errors(payload, figdir / "long_time_errors.png")
    figure_collapse(payload, figdir / "long_time_collapse.png")
    for key, case in payload["cases"].items():
        v = case["verdict"]
        print(f"{key}: {'PASS' if v['pass'] else 'FAIL'} "
              f"({sum(1 for c in v['criteria'] if not c['pass'])} failing)")
    print(f"wall {payload['wall_time_s']:.1f} s -> {outdir}")
    return 0 if all(c["verdict"]["pass"] for c in payload["cases"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())