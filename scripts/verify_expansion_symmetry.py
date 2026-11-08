#!/usr/bin/env python3
"""Mirror-symmetry verification of the planar sudden expansion (Re_D = 100).

Why this exists
---------------
``docs/benchmarks/incompressible-verification.md`` (section 2.2) reported
``L_r/D = 0.20`` on the lower wall and ``11.86`` on the upper wall of the
``AR = 4`` planar expansion, and read that as *"the geometry and the inlet are
exactly symmetric ... which identifies a defect in the step-corner / wake
treatment"*.  The geometry built by
:func:`tensorfvm.incompressible_baselines.build_sudden_expansion` is **not**
symmetric: the inlet is flush with the bottom wall (``y in [0, 1]`` of a
``y in [0, 4]`` duct), so the jet hugs the bottom wall and the whole
recirculation belongs to the top (step-side) wall.

This script decides the question by measurement instead of by reading:

1. ``bottom`` -- the historical geometry (inlet flush with the bottom wall);
2. ``top``    -- its exact y-mirror (inlet flush with the top wall);
3. ``centred``-- the genuinely mirror-symmetric geometry (inlet band
   ``y in [1.5, 2.5]`` of the ``H2 = 4`` duct).

Checks
------
* ``expansion_mirror_metrics(bottom, top)``: if the solver is mirror-covariant,
  the two solutions are exact y-mirrors of each other.  A small residual proves
  that the lower/upper asymmetry of the historical case is a property of the
  *geometry*, not a solver defect.
* ``expansion_symmetry_metrics(centred)``: on the symmetric geometry the
  discrete solution must be mirror-symmetric; the residual is tracked over the
  iteration and on a refined grid (D/10 -> D/20).

Usage::

    PYTHONPATH=src python3 scripts/verify_expansion_symmetry.py \
        --out docs/verification-expansion-symmetry
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tensorfvm.incompressible_baselines import (  # noqa: E402
    build_sudden_expansion,
    expansion_mirror_metrics,
    expansion_symmetry_metrics,
    sudden_expansion_metrics,
)

#: geometry label -> inlet band on the west boundary (``D = 1``, ``H2 = 4``)
GEOMETRIES = {
    "bottom": None,          # historical case: inlet flush with the bottom wall
    "top": (3.0, 4.0),       # exact y-mirror of "bottom"
    "centred": (1.5, 2.5),   # genuinely mirror-symmetric geometry
}


def _symmetry(solver) -> dict:
    return expansion_symmetry_metrics(solver)


def converge_tracking(solver, *, max_iter: int = 20000, tol: float = 1e-6,
                      check: int = 10, sample: int = 20) -> dict:
    """First-order SIMPLE to a steady state, recording the symmetry residual.

    The iteration settings mirror ``scripts/run_fvm_incompressible_baselines.py``
    for the expansion case (``alpha_u = 0.8``, ``alpha_p = 0.5``, first-order
    convection only, ``momentum_sweeps = 4``, ``p_cycles = 2``), so the numbers
    here are directly comparable with ``docs/incompressible-baselines/
    baseline-runs/expansion_n0*.json``.  The driver is chunked instead of using
    ``run()`` directly only because the mirror-symmetry residual has to be
    sampled along the iteration.
    """
    solver.limited = False
    solver.alpha_u = 0.8
    solver.alpha_p = 0.5
    prev_u, prev_v = solver.u.copy(), solver.v.copy()
    history: list[dict] = []
    iterations = 0
    while iterations < max_iter:
        state = solver.run(max_iter=check, tol=-1.0, check=check,
                           record_every=0, max_stall=10 ** 9,
                           momentum_sweeps=4, p_cycles=2)
        iterations += check
        change = float(max(np.abs(solver.u - prev_u).max(),
                           np.abs(solver.v - prev_v).max()))
        prev_u, prev_v = solver.u.copy(), solver.v.copy()
        if iterations % (check * sample) == 0 or change < tol:
            history.append(dict(iteration=iterations,
                                continuity=float(state["continuity"]),
                                change=change, **_symmetry(solver)))
        if state["continuity"] < tol and change < tol:
            break
    return dict(iterations=iterations,
                continuity=float(state["continuity"]),
                change=change, history=history)


def run_case(geometry: str, level: int, *, re_D: float = 100.0,
             max_iter: int = 20000) -> dict:
    tag = f"expansion_{geometry}_n{level:03d}"
    t0 = time.time()
    solver = build_sudden_expansion(level, re_D=re_D,
                                    inlet_window=GEOMETRIES[geometry],
                                    poisson_solver="sparse")
    conv = converge_tracking(solver, max_iter=max_iter)
    metrics = sudden_expansion_metrics(solver)
    sym = _symmetry(solver)
    print(f"[{tag}] {time.time() - t0:6.1f}s it={conv['iterations']} "
          f"cont={conv['continuity']:.2e} sym={sym['symmetry_rel']:.3e} "
          f"Lr/D upper={metrics['reattachment_over_D_upper']} "
          f"lower={metrics['reattachment_over_D_lower']}", flush=True)
    return dict(tag=tag, geometry=geometry, level=level, re_D=re_D,
                inlet_window=tuple(solver.meta["inlet_window"]),
                symmetric_geometry=bool(solver.meta["symmetric_about_midplane"]),
                wall_seconds=round(time.time() - t0, 1),
                convergence=conv, metrics=metrics, symmetry=sym,
                solver=solver)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _velocity_magnitude(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = case["solver"].u, case["solver"].v
    uc = 0.5 * (u[:-1] + u[1:])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    return uc, vc, np.hypot(uc, vc)


def figure_fields(cases: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [("bottom", "inlet flush with bottom wall (historical)"),
              ("centred", "inlet centred (mirror-symmetric geometry)")]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    for ax, (key, title) in zip(axes, labels):
        case = cases[key]
        uc, vc, speed = _velocity_magnitude(case)
        hx = 1.0 / case["level"]
        x = (np.arange(uc.shape[0]) + 0.5) * hx
        y = (np.arange(uc.shape[1]) + 0.5) * hx
        im = ax.pcolormesh(x, y, speed.T, cmap="viridis", shading="auto")
        ax.contour(x, y, uc.T, levels=[0.0], colors="w", linewidths=1.0)
        ax.axhline(2.0, color="w", lw=0.8, ls="--")
        m = case["metrics"]
        ax.set_title(f"{title}   L_r/D upper = "
                     f"{m['reattachment_over_D_upper']}  lower = "
                     f"{m['reattachment_over_D_lower']}   "
                     f"mirror residual = {case['symmetry']['symmetry_rel']:.2e}",
                     fontsize=9)
        ax.set_ylabel("y/D")
        fig.colorbar(im, ax=ax, label="|U|/U_in")
    axes[-1].set_xlabel("x/D")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_symmetry(cases: dict, mirror: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for key, style in (("centred", "-o"), ("bottom", "-s")):
        hist = cases[key]["convergence"]["history"]
        it = [h["iteration"] for h in hist]
        ax1.semilogy(it, [max(h["symmetry_rel"], 1e-16) for h in hist],
                     style, ms=3,
                     label=f"{key} inlet (geometry "
                           f"{'symmetric' if cases[key]['symmetric_geometry'] else 'asymmetric'})")
    ax1.set_xlabel("SIMPLE iteration")
    ax1.set_ylabel("mirror-symmetry residual (rel.)")
    ax1.set_title("symmetry residual along the iteration")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    names = ["u", "v", "p"]
    mirrored = [mirror["u_mirror_rel"], mirror["v_mirror_rel"],
                mirror["p_mirror_rel"]]
    centred = [cases["centred"]["symmetry"]["u_symmetry_rel"],
               cases["centred"]["symmetry"]["v_symmetry_rel"],
               cases["centred"]["symmetry"]["p_symmetry_rel"]]
    xs = np.arange(len(names))
    ax2.bar(xs - 0.2, np.maximum(mirrored, 1e-16), 0.4,
            label="bottom vs mirrored-top geometry")
    ax2.bar(xs + 0.2, np.maximum(centred, 1e-16), 0.4,
            label="centred geometry, self-symmetry")
    ax2.set_yscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("residual (rel.)")
    ax2.set_title("mirror covariance vs self-symmetry")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_re_sweep(sweep: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    re = [float(k) for k in sweep]
    order = np.argsort(re)
    keys = [list(sweep)[i] for i in order]
    sym = [sweep[k]["symmetry"]["symmetry_rel"] for k in keys]
    upper = [sweep[k]["metrics"]["reattachment_over_D_upper"] for k in keys]
    lower = [sweep[k]["metrics"]["reattachment_over_D_lower"] for k in keys]
    res = [float(k) for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax1.semilogy(res, np.maximum(sym, 1e-16), "o-", label="mirror-symmetry residual")
    ax1.axhline(1e-2, color="r", ls="--", lw=0.8, label="1 % threshold")
    ax1.set_xlabel("$Re_D$")
    ax1.set_ylabel("mirror-symmetry residual (rel.)")
    ax1.set_title("centred geometry: symmetry vs $Re$")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(res, upper, "o-", label="$L_r/D$ upper wall")
    ax2.plot(res, [np.nan if v is None else v for v in lower], "s-",
             label="$L_r/D$ lower wall")
    ax2.set_xlabel("$Re_D$")
    ax2.set_ylabel("$L_r/D$")
    ax2.set_title("reattachment lengths, centred geometry")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/verification-expansion-symmetry")
    ap.add_argument("--coarse", type=int, default=10)
    ap.add_argument("--fine", type=int, default=20)
    ap.add_argument("--re", type=float, default=100.0,
                    help="Reynolds number of the main geometry comparison")
    ap.add_argument("--re-sweep", default="10,25,50",
                    help="comma separated Re for the centred geometry; the "
                         "symmetry-breaking onset is bracketed by the last "
                         "symmetric and the first asymmetric run")
    ap.add_argument("--mirror-re", type=float, default=25.0,
                    help="Re of the bottom-vs-top mirror-covariance pair "
                         "(low Re keeps the steady state unique)")
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the matplotlib figures (e.g. bare compute nodes); "
                         "the raw fields are still saved so they can be plotted "
                         "elsewhere")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    cases = {}
    for name in ("bottom", "top"):
        cases[name] = run_case(name, args.coarse, re_D=args.re,
                               max_iter=args.max_iter)
    for level in (args.coarse, args.fine):
        key = "centred" if level == args.coarse else "centred_fine"
        cases[key] = run_case("centred", level, re_D=args.re,
                              max_iter=args.max_iter)

    mirror = expansion_mirror_metrics(cases["bottom"]["solver"],
                                      cases["top"]["solver"])
    print("mirror(bottom, top) @ Re=%.0f:" % args.re,
          {k: round(v, 9) for k, v in mirror.items()}, flush=True)

    # ---- Reynolds sweep on the symmetric geometry -------------------------
    # If the centred geometry is asymmetric at every Re the solver breaks the
    # mirror symmetry (defect); if the asymmetry switches on above a critical
    # Re it is the known steady symmetry-breaking bifurcation of plane sudden
    # expansions (physics), and the Re = 100 steady state is legitimately
    # asymmetric.
    sweep = {}
    for re_d in [float(v) for v in args.re_sweep.split(",") if v.strip()]:
        sweep[f"{re_d:g}"] = run_case("centred", args.coarse, re_D=re_d,
                                      max_iter=args.max_iter)
    mirror_low = {}
    if args.mirror_re and abs(args.mirror_re - args.re) > 1e-9:
        low = {}
        for name in ("bottom", "top"):
            low[name] = run_case(name, args.coarse, re_D=args.mirror_re,
                                 max_iter=args.max_iter)
        mirror_low = expansion_mirror_metrics(low["bottom"]["solver"],
                                              low["top"]["solver"])
        mirror_low["re_D"] = args.mirror_re
        print("mirror(bottom, top) @ Re=%.0f:" % args.mirror_re,
              {k: round(v, 9) for k, v in mirror_low.items()}, flush=True)

    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    except Exception:  # pragma: no cover - git is optional at run time
        rev = None

    def _c(case):
        return {kk: vv for kk, vv in case.items() if kk != "solver"}

    summary = dict(
        case="planar sudden expansion, AR = 4, Re_D = %.1f" % args.re,
        commit=rev,
        settings=dict(alpha_u=0.8, alpha_p=0.5, limited=False,
                      momentum_sweeps=4, p_cycles=2, poisson="sparse",
                      max_iter=args.max_iter, tol=1e-6),
        cases={k: _c(v) for k, v in cases.items()},
        re_sweep_centred={k: _c(v) for k, v in sweep.items()},
        mirror_bottom_vs_top=mirror,
        mirror_bottom_vs_top_low_re=mirror_low,
        verdict=dict(
            # the historical geometry is NOT symmetric (inlet flush with the
            # bottom wall), so its lower/upper difference is geometric
            historical_geometry_symmetric=False,
            # mirror covariance of the solver at a low Re (unique steady state)
            mirror_covariant_low_re=bool(
                mirror_low and mirror_low["mirror_rel"] < 1e-2),
            centred_symmetric_at_re100=bool(
                cases["centred"]["symmetry"]["symmetry_rel"] < 1e-2),
            symmetry_breaks_above_re=next(
                (float(k) for k, v in sorted(
                    ((float(k), v) for k, v in sweep.items()),
                    key=lambda kv: kv[0])
                 if v["symmetry"]["symmetry_rel"] >= 1e-2), None),
        ),
    )
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    for key, case in list(cases.items()) + [
            (f"centred_re{re_d:g}", c) for re_d, c in
            ((float(k), v) for k, v in sweep.items())]:
        solver = case["solver"]
        np.savez_compressed(os.path.join(args.out, f"fields_{key}.npz"),
                            u=solver.u, v=solver.v, p=solver.p,
                            inlet_values=solver.inlet_values)

    if not args.no_figures:
        figure_fields({"bottom": cases["bottom"], "centred": cases["centred"]},
                      os.path.join(args.out, "fig_asymmetric_vs_centred.png"))
        figure_symmetry(cases, mirror,
                        os.path.join(args.out, "fig_symmetry_residuals.png"))
        figure_re_sweep(sweep,
                        os.path.join(args.out, "fig_symmetry_vs_re.png"))
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())