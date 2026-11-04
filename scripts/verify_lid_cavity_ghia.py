"""Ghia et al. (1982) lid-driven-cavity verification ladder for TensorFVM.

Runs :class:`tensorfvm.lid_cavity.LidCavitySolver` at several Reynolds numbers
and grid resolutions, scores both centreline profiles pointwise against the
published tables, checks the primary-vortex centre, estimates the observed order
of accuracy from the finest grid triplet, and writes

* ``summary.json``      - every number used in the report, machine generated,
* ``profiles_Re*.png``  - centreline profiles vs the Ghia symbols,
* ``field_Re*.png``     - vorticity contours, streamlines, primary vortex,
* ``convergence.png``   - grid convergence of the primary extremum,
* ``tables.md``         - markdown tables injected from the JSON.

Usage::

    PYTHONPATH=src python3 scripts/verify_lid_cavity_ghia.py \
        --reynolds 100,400,1000 --grids 32,64,128 --out scripts/verification-lid-cavity

Physical background, geometry and parameters are recorded in the JSON header so
the same file is enough to reproduce the published figures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from tensorfvm.lid_cavity import (
    LidCavityConfig,
    LidCavitySolver,
    ghia_acceptance,
    score_against_ghia,
)
from tensorfvm.lid_cavity_reference import SUPPORTED_REYNOLDS, VORTEX_CENTRE, self_check, table


def run_case(n: int, reynolds: float, tolerance: float, max_steps: int,
             cfl: float) -> dict:
    torch.set_num_threads(1)
    started = time.time()
    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=reynolds, cfl=cfl,
                                              max_steps=max_steps,
                                              tolerance=tolerance,
                                              check_every=250))
    info = solver.run(on_check=lambda sv, st, ch: print(
        f"    Re={reynolds:g} n={n} step {st} rate {ch:.2e}", flush=True))
    result = score_against_ghia(solver, reynolds)
    result["acceptance"] = ghia_acceptance(result)
    result["solver"] = {
        "steps": info["steps"],
        "converged": info["converged"],
        "residual_rate": info["residual"],
        "max_divergence": info["max_divergence"],
        "cfl": cfl,
        "viscosity": solver.nu,
        "time_step_final": info["dt"],
        "wall_seconds": time.time() - started,
        "psi": solver.streamfunction_extrema(),
    }
    result["field"] = {
        "u": [[round(float(x), 6) for x in row] for row in solver.u],
        "v": [[round(float(x), 6) for x in row] for row in solver.v],
    }
    return result


def observed_order(values: list[tuple[int, float]]) -> dict:
    """Grid convergence of a scalar quantity on a bisection sequence.

    ``values`` is ``[(n, q), ...]`` in increasing ``n``.  Returns the observed
    order ``p = log2(|q1 - q2| / |q2 - q3|)`` plus a Richardson extrapolation.
    """
    if len(values) < 3:
        return {}
    (n1, q1), (n2, q2), (n3, q3) = values[-3:]
    if not (n2 == 2 * n1 and n3 == 2 * n2):
        return {}
    coarse = abs(q1 - q2)
    fine = abs(q2 - q3)
    if fine == 0.0 or coarse == 0.0:
        return {"degenerate": True}
    order = math.log2(coarse / fine)
    ratio = 2.0 ** order
    extrapolated = q3 + (q3 - q2) / (ratio - 1.0)
    return {"order": order, "richardson": extrapolated,
            "finest": q3, "relative_vs_richardson": abs(q3 - extrapolated) / abs(extrapolated)}


def make_tables(summary: dict, out_dir: str) -> str:
    lines = ["# Ghia et al. (1982) lid-driven cavity - machine-generated tables", ""]
    lines.append("## Error against the published centreline tables")
    lines.append("")
    lines.append("| Re | n | steps | converged | max|du| | rms|du| | max|dv| | rms|dv| "
                 "| u_min (ref) | v_max (ref) | vortex (ref) | div max |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in summary["cases"]:
        u, v = case["u"], case["v"]
        ext_u, ext_v = u["extremum"], v["extremum"]
        vs = case["vortex_centre"]["computed"]
        ref_vs = VORTEX_CENTRE[case["reynolds"]]
        lines.append(
            f"| {case['reynolds']:g} | {case['n']} | {case['solver']['steps']} | "
            f"{'yes' if case['solver']['converged'] else 'no'} | "
            f"{u['max_abs_error']:.4f} | {u['rms_error']:.4f} | "
            f"{v['max_abs_error']:.4f} | {v['rms_error']:.4f} | "
            f"{ext_u['min_computed'][1]:.5f} ({ext_u['min_reference'][1]:.5f}) | "
            f"{ext_v['max_computed'][1]:.5f} ({ext_v['max_reference'][1]:.5f}) | "
            f"({vs[0]:.3f},{vs[1]:.3f}) ({ref_vs[0]:.3f},{ref_vs[1]:.3f}) | "
            f"{case['max_divergence']:.1e} |")
    lines.append("")
    lines.append("## Acceptance (`ghia_acceptance`, 30e-3 pointwise / 15 % extremum)")
    lines.append("")
    for case in summary["cases"]:
        acc = case["acceptance"]
        verdict = "PASS" if acc["pass"] else "FAIL"
        lines.append(f"* Re = {case['reynolds']:g}, n = {case['n']}: **{verdict}** - " + ", ".join(
            f"{name} = {item['value']:.4g} (limit {item['limit']:g})"
            for name, item in acc["criteria"].items()))
    lines.append("")
    lines.append("## Grid convergence of the primary extrema")
    lines.append("")
    for rey, block in summary["convergence"].items():
        if not block.get("u_min") or "order" not in block["u_min"]:
            continue
        u_min, v_max = block["u_min"], block["v_max"]
        lines.append(f"* Re = {rey}: u_min order {u_min['order']:.2f} -> Richardson "
                     f"{u_min['richardson']:.5f} (finest {u_min['finest']:.5f}, "
                     f"{100 * u_min['relative_vs_richardson']:.2f} % away); "
                     f"v_max order {v_max['order']:.2f} -> {v_max['richardson']:.5f} "
                     f"(finest {v_max['finest']:.5f}, "
                     f"{100 * v_max['relative_vs_richardson']:.2f} % away)")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "tables.md"), "w") as handle:
        handle.write(text)
    return text


def make_figures(summary: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    made: list[str] = []
    reys = sorted({case["reynolds"] for case in summary["cases"]})
    by_key = {(case["reynolds"], case["n"]): case for case in summary["cases"]}

    # --- centreline profiles, one panel per Reynolds number
    for which, name, axis_label in (("u", "profiles", "u/U  at  x = L/2"),
                                    ("v", "profiles", "v/U  at  y = L/2")):
        fig, axes = plt.subplots(1, len(reys), figsize=(4.2 * len(reys), 4.0))
        axes = np.atleast_1d(axes)
        for ax, rey in zip(axes, reys):
            pts = table(rey, which)
            coord = np.array([p[0] for p in pts])
            ref = np.array([p[1] for p in pts])
            ax.plot(ref, coord, "o", ms=4.5, mfc="none", label="Ghia et al. (1982)")
            for n in sorted({c["n"] for c in summary["cases"]}):
                case = by_key.get((rey, n))
                if case is None:
                    continue
                prof = case[which]["profiles"]
                ax.plot(prof["computed"], prof["coord"], "-", lw=1.1,
                        label=f"TensorFVM n = {n} (max err {case[which]['max_abs_error']:.4f})")
            ax.set_xlabel(axis_label)
            ax.set_ylabel("y/L" if which == "u" else "x/L")
            ax.set_title(f"Re = {rey:g}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        fig.suptitle("Lid-driven cavity centreline profiles vs Ghia et al. (1982)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{which}_profiles.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    # --- fields: vorticity contours + streamlines on the finest grid
    for rey in reys:
        n_max = max(c["n"] for c in summary["cases"] if c["reynolds"] == rey)
        solver = LidCavitySolver(LidCavityConfig(n=n_max, reynolds=rey,
                                                 cfl=summary["meta"]["cfl"]))
        solver.u = torch.tensor(by_key[(rey, n_max)]["field"]["u"], dtype=torch.float64)
        solver.v = torch.tensor(by_key[(rey, n_max)]["field"]["v"], dtype=torch.float64)
        omega = solver.vorticity().numpy()
        psi = solver.streamfunction().numpy()
        h = 1.0 / n_max
        xc = (np.arange(n_max) + 0.5) * h
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        cs = axes[0].contourf(xc, xc, omega, levels=41, cmap="RdBu_r")
        fig.colorbar(cs, ax=axes[0], shrink=0.9, label="omega L/U")
        axes[0].set_title(f"vorticity, Re = {rey:g}, n = {n_max}")
        axes[1].contour(xc, xc, psi, levels=25, colors="k", linewidths=0.5)
        vx, vy = by_key[(rey, n_max)]["vortex_centre"]["computed"]
        axes[1].plot([vx], [vy], "r+", ms=12, label="primary vortex (psi min)")
        axes[1].set_title(f"streamlines, Re = {rey:g}")
        axes[1].legend(fontsize=7)
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlabel("x/L")
            ax.set_ylabel("y/L")
        fig.tight_layout()
        path = os.path.join(out_dir, f"field_Re{int(rey)}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    # --- grid convergence of the primary extremum
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for rey in reys:
        block = summary["convergence"].get(str(int(rey)), {})
        series = block.get("u_min_series")
        if series:
            ns = [p[0] for p in series]
            qs = [abs(p[1]) for p in series]
            ax.loglog(ns, qs, "o-", label=f"Re = {rey:g} |u_min|")
    ax.set_xlabel("cells per direction n")
    ax.set_ylabel("|u_min| / U")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Grid convergence of the primary extremum")
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reynolds", default="100,400,1000")
    parser.add_argument("--grids", default="32,64,128")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--max-steps", type=int, default=120_000)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="scripts/verification-lid-cavity")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    reys = [float(x) for x in args.reynolds.split(",") if x]
    grids = [int(x) for x in args.grids.split(",") if x]
    for rey in reys:
        if rey not in SUPPORTED_REYNOLDS:
            raise SystemExit(f"Re={rey} has no Ghia table; available {SUPPORTED_REYNOLDS}")

    meta = {
        "code": "tensorfvm.lid_cavity.LidCavitySolver",
        "geometry": "unit square cavity, no-slip on x=0, x=L, y=0; lid u=U on y=L",
        "reynolds": reys,
        "grids": grids,
        "cfl": args.cfl,
        "tolerance": args.tolerance,
        "max_steps": args.max_steps,
        "reference": ("U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, "
                      "Tables I and II"),
        "reference_self_check": self_check(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("reference table self-check:", meta["reference_self_check"]["ok"], flush=True)

    cases: list[dict] = []
    jobs = [(n, rey) for rey in reys for n in grids if not (rey == 1000.0 and n == 32)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, n, rey, args.tolerance, args.max_steps, args.cfl): (rey, n)
                   for n, rey in jobs}
        for future, (rey, n) in futures.items():
            result = future.result()
            cases.append(result)
            print(f"finished Re={rey:g} n={n}: max|du|={result['u']['max_abs_error']:.4f} "
                  f"steps={result['solver']['steps']} "
                  f"wall={result['solver']['wall_seconds']:.0f}s", flush=True)
            # Crash resilience: a multi-hour ladder must not lose finished cases.
            # Every completed case is flushed to ``summary.partial.json``; the
            # final ``summary.json`` is written once, at the end, as before.
            with open(os.path.join(args.out, "summary.partial.json"), "w") as handle:
                json.dump({"meta": meta, "cases": sorted(cases, key=lambda c: (c["reynolds"], c["n"])),
                           "partial": True, "wall_seconds_total": time.time() - started},
                          handle, indent=1)

    cases.sort(key=lambda c: (c["reynolds"], c["n"]))
    summary = {"meta": meta, "cases": cases}

    # convergence of the extrema on the bisection sequences
    convergence: dict[str, dict] = {}
    for rey in reys:
        seq = [c for c in cases if c["reynolds"] == rey]
        u_series = [(c["n"], c["u"]["extremum"]["min_computed"][1]) for c in seq]
        v_series = [(c["n"], c["v"]["extremum"]["max_computed"][1]) for c in seq]
        convergence[str(int(rey))] = {
            "u_min_series": u_series,
            "v_max_series": v_series,
            "u_min": observed_order(u_series),
            "v_max": observed_order(v_series),
        }
    summary["convergence"] = convergence
    summary["wall_seconds_total"] = time.time() - started

    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    tables = make_tables(summary, args.out)
    print(tables, flush=True)

    if not args.no_figures:
        for path in make_figures(summary, args.out):
            print("figure", path, flush=True)
    print("done in %.1f s -> %s" % (time.time() - started, args.out), flush=True)


if __name__ == "__main__":
    main()