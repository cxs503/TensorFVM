"""CASE-FVM-THACKER-BASIN: 2-D well-balanced shallow water vs Thacker (1981).

The harness behind ``docs/case-fvm-thacker-basin/``.  It exercises the 2-D
shallow-water operator (:class:`tensorfvm.shallow_water.swe2d.ShallowWater2D`)
on the nonlinear paraboloid-basin oscillation whose closed form is
:class:`tensorfvm.shallow_water.thacker.ThackerBasin`, and it produces one JSON
artefact per predeclared criterion:

=========================  ====================================================
criterion                  artefact / key
=========================  ====================================================
C1 lake at rest            ``lake_at_rest.json``
C2 closed-form residual    ``exactness.json``
C3 volume conservation     ``metrics.json -> volume``
C4 centre-depth history    ``metrics.json -> centre_depth``
C5 shoreline radius        ``metrics.json -> shoreline``
C6 azimuthal symmetry      ``metrics.json -> symmetry``
C7 grid convergence        ``convergence.json``
C8 negative control        ``negative_control.json``
C9 numerical health        ``metrics.json -> health``
=========================  ====================================================

The metrics are the ones frozen in ``criteria.md`` before the run; nothing here
chooses a metric after seeing a result.  Measurements are deliberately
threshold-free (cell values and a linear ``h -> 0`` interpolation), so no
analytic input is smuggled into the measurement of the quantity it measures --
the one exception is documented inline: the four-cell average that reports
``h(0,0,t)`` is a bilinear reconstruction of a cell-centred field and carries an
``O(d^2)`` bias, which the grid-convergence study (C7) quantifies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..shallow_water.swe2d import ShallowWater2D
from ..shallow_water.thacker import ThackerBasin, check_thacker_solution

__all__ = [
    "DEFAULTS",
    "build_basin",
    "grid",
    "lake_at_rest_check",
    "negative_control",
    "run_case",
    "convergence_study",
    "figure_thacker",
    "write_artifacts",
]

#: Frozen benchmark parameters (``criteria.md``).
DEFAULTS = {
    "H": 1.0,
    "a": 1.0,
    "amplitude": 0.5,
    "gravity": 9.81,
    "domain_factor": 1.5,   # domain [-1.5a, 1.5a]^2 > max shoreline 1.316a
    "periods": 2.0,
    "samples": 41,
    "cfl": 0.4,
    "order": 2,
    "dry_depth": 1e-8,
    "resolution": 128,
    "resolutions": (32, 64, 128),
    "tolerance_centre": 0.03,
    "tolerance_shore": 0.03,
    "tolerance_symmetry": 0.02,
    "tolerance_volume_rel": 1e-3,
    "tolerance_volume_drift": 1e-9,
    "tolerance_lake_at_rest": 1e-11,
    "tolerance_exactness": 1e-6,
    "max_steps": 400000,
}


def build_basin(**kwargs) -> ThackerBasin:
    """Basin from the frozen defaults, overridable by keyword."""
    params = dict(DEFAULTS)
    params.update(kwargs)
    return ThackerBasin(params["H"], params["a"], params["amplitude"],
                        params["gravity"])


def grid(basin: ThackerBasin, n: int, domain_factor: float = 1.5):
    """Cell-centre grid ``(X, Y)`` of ``n x n`` cells over ``[-L, L]^2``."""
    L = domain_factor * basin.a
    edge = np.linspace(-L, L, n + 1)
    centres = 0.5 * (edge[:-1] + edge[1:])
    return np.meshgrid(centres, centres, indexing="ij")


def _solver(basin: ThackerBasin, n: int, params: dict,
            well_balanced: bool = True) -> ShallowWater2D:
    X, Y = grid(basin, n, params["domain_factor"])
    L = params["domain_factor"] * basin.a
    solver = ShallowWater2D(n, n, -L, L, -L, L, bed=basin.bed(X, Y),
                            gravity=basin.gravity, cfl=params["cfl"],
                            order=params["order"],
                            dry_depth=params["dry_depth"],
                            well_balanced=well_balanced)
    return solver


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def centre_depth(solver: ShallowWater2D) -> float:
    """Depth at the origin from the four central cells (bilinear).

    A cell-centred field is a cell *average*; the four-cell average is the
    second-order-accurate bilinear value at the origin, biased low by
    ``(5/6) c d^2`` for the parabolic profile (``c`` = surface curvature).
    The bias is ``O(d^2)`` and is what the C7 convergence study measures.
    """
    i = solver.nx // 2
    j = solver.ny // 2
    return float(0.25 * (solver.h[i - 1, j - 1] + solver.h[i - 1, j]
                         + solver.h[i, j - 1] + solver.h[i, j]))


def shoreline_radius(solver: ShallowWater2D) -> float:
    """Shoreline radius from the ``+x`` axis, linear ``h -> dry`` crossing.

    The two central rows are averaged (removing the ``y`` offset of the row
    centres to ``O(d^2)``), the outermost wet cell is located, and ``h`` is
    interpolated linearly to zero between that cell and its dry neighbour.
    Near the shoreline ``h`` is linear in ``(s - r)``, so the interpolation is
    second order.  The measure uses only ``h`` -- no analytic radius enters.
    """
    j = solver.ny // 2
    profile = 0.5 * (solver.h[:, j - 1] + solver.h[:, j])
    centres = solver.x_min + (np.arange(solver.nx) + 0.5) * solver.dx
    wet = np.nonzero(profile > solver.dry_depth)[0]
    if wet.size == 0:
        return 0.0
    last = int(wet.max())
    if last + 1 >= solver.nx:
        return float(centres[-1])
    h_last = float(profile[last])
    h_next = float(profile[last + 1])
    frac = h_last / max(h_last - h_next, 1e-300)
    return float(centres[last] + frac * (centres[last + 1] - centres[last]))


def symmetry_defect(solver: ShallowWater2D) -> dict:
    """Max ``|h - h_mirrored|`` under ``x -> -x``, ``y -> -y`` and ``x <-> y``."""
    h = solver.h
    flip_x = np.max(np.abs(h - h[::-1, :]))
    flip_y = np.max(np.abs(h - h[:, ::-1]))
    transpose = np.max(np.abs(h - h.T)) if solver.nx == solver.ny else float("nan")
    return {"flip_x": float(flip_x), "flip_y": float(flip_y),
            "transpose": float(transpose)}


# --------------------------------------------------------------------------- #
# the benchmark run
# --------------------------------------------------------------------------- #
def run_case(n: int = None, **kwargs) -> dict:
    """Integrate ``2T`` of the exact oscillation and measure it."""
    params = dict(DEFAULTS)
    params.update(kwargs)
    n = int(params["resolution"] if n is None else n)
    basin = build_basin(**{k: params[k] for k in
                           ("H", "a", "amplitude", "gravity")})
    solver = _solver(basin, n, params)
    X, Y = grid(basin, n, params["domain_factor"])
    h0, u0, v0 = basin.initial_state(X, Y)
    solver.set_state(h0, u0, v0)

    v0_total = solver.total_volume()
    amplitude_centre = 0.5 * (max(basin.centre_depth(t) for t in
                                  (0.0, 0.5 * basin.period))
                              - min(basin.centre_depth(t) for t in
                                    (0.0, 0.5 * basin.period)))
    t_end = params["periods"] * basin.period
    samples = int(params["samples"])
    times = [k * t_end / samples for k in range(samples + 1)]

    series = {"t": [], "t_over_T": [], "h_centre": [], "h_centre_exact": [],
              "s": [], "s_exact": [], "volume": [], "symmetry_max": []}
    for t in times:
        if t > solver.time:
            solver.advance(t, max_steps=int(params.get("max_steps", 400000)))
        s = solver.summary()
        sym = symmetry_defect(solver)
        series["t"].append(float(solver.time))
        series["t_over_T"].append(float(solver.time / basin.period))
        series["h_centre"].append(centre_depth(solver))
        series["h_centre_exact"].append(float(basin.centre_depth(solver.time)))
        series["s"].append(shoreline_radius(solver))
        series["s_exact"].append(float(basin.shoreline_radius(solver.time)))
        series["volume"].append(float(s["volume"]))
        series["symmetry_max"].append(max(sym.values()))

    h_num = np.asarray(series["h_centre"])
    h_ex = np.asarray(series["h_centre_exact"])
    s_num = np.asarray(series["s"])
    s_ex = np.asarray(series["s_exact"])
    vol = np.asarray(series["volume"])
    err_centre = math.sqrt(float(np.mean(((h_num - h_ex) / amplitude_centre)
                                         ** 2)))
    err_centre_max = float(np.max(np.abs(h_num - h_ex)) / amplitude_centre)
    err_shore = float(np.mean(np.abs(s_num - s_ex) / s_ex))
    err_shore_max = float(np.max(np.abs(s_num - s_ex) / s_ex))
    sym_max = float(np.max(series["symmetry_max"]))
    volume_rel = float(np.max(np.abs(vol - basin.volume)) / basin.volume)
    volume_drift = float(np.max(np.abs(vol - v0_total)) / v0_total)

    # boundary must stay dry (domain chosen large enough)
    boundary = np.concatenate([solver.h[0, :], solver.h[-1, :],
                               solver.h[:, 0], solver.h[:, -1]])
    health = {
        "h_min": float(solver.h.min()),
        "steps": int(solver.steps),
        "clipped_cells": int(solver.clipped_cells),
        "dry_flips": int(solver.dry_flips),
        "boundary_max_depth": float(np.max(boundary)),
        "cfl_number": float(solver.cfl_number()),
    }

    criteria = {
        "C3_volume": volume_rel <= params["tolerance_volume_rel"]
                      and volume_drift <= params["tolerance_volume_drift"],
        "C4_centre_depth": err_centre <= params["tolerance_centre"],
        "C5_shoreline": err_shore <= params["tolerance_shore"],
        "C6_symmetry": sym_max <= params["tolerance_symmetry"],
        "C9_health": (health["h_min"] >= 0.0
                      and health["clipped_cells"] == 0
                      and health["boundary_max_depth"] <= params["dry_depth"]),
    }

    # x-axis profile at T/4 for the figure (re-run is not needed: reuse state)
    profile_t = 0.25 * basin.period
    if abs(solver.time - profile_t) > 0.5 * t_end / samples:
        pass  # the figure takes the profile from a dedicated short run

    return {
        "resolution": n,
        "dx": solver.dx,
        "basin": basin.summary(),
        "discretisation": {"nx": solver.nx, "ny": solver.ny,
                           "cfl": params["cfl"], "order": params["order"],
                           "dry_depth": params["dry_depth"],
                           "well_balanced": solver.well_balanced,
                           "domain_factor": params["domain_factor"]},
        "centre_depth": {"l2_relative": err_centre,
                         "max_relative": err_centre_max,
                         "tolerance": params["tolerance_centre"],
                         "scale": amplitude_centre},
        "shoreline": {"mean_relative": err_shore,
                      "max_relative": err_shore_max,
                      "tolerance": params["tolerance_shore"]},
        "volume": {"exact": basin.volume, "initial": v0_total,
                   "max_relative_to_exact": volume_rel,
                   "max_drift": volume_drift,
                   "tolerance_relative": params["tolerance_volume_rel"],
                   "tolerance_drift": params["tolerance_volume_drift"]},
        "symmetry": {"max_defect": sym_max,
                     "scale": amplitude_centre,
                     "normalised": sym_max / amplitude_centre,
                     "tolerance": params["tolerance_symmetry"]},
        "health": health,
        "criteria": criteria,
        "series": series,
    }


def profile_at(basin: ThackerBasin, n: int, phases=(0.25, 0.75), **kwargs):
    """``h(x, y=0)`` along ``+x`` at the given phase fractions of one period.

    Used only for the report figure; it integrates to each phase from ``t = 0``
    with the same solver, so the profile is a genuine solution state.
    """
    params = dict(DEFAULTS)
    params.update(kwargs)
    solver = _solver(basin, n, params)
    X, Y = grid(basin, n, params["domain_factor"])
    solver.set_state(*basin.initial_state(X, Y))
    out = []
    t_now = 0.0
    for phase in phases:
        target = phase * basin.period
        if target > t_now:
            solver.advance(target)
            t_now = target
        j = solver.ny // 2
        profile = 0.5 * (solver.h[:, j - 1] + solver.h[:, j])
        centres = solver.x_min + (np.arange(solver.nx) + 0.5) * solver.dx
        keep = centres >= 0.0
        out.append({
            "phase": float(phase),
            "t": float(target),
            "x": centres[keep].tolist(),
            "h": profile[keep].tolist(),
            "h_exact": basin.depth(centres[keep], 0.0, target).tolist(),
            "s_exact": float(basin.shoreline_radius(target)),
        })
    return out


# --------------------------------------------------------------------------- #
# the other criteria
# --------------------------------------------------------------------------- #
def lake_at_rest_check(n: int = 64, half_period: float = 0.5, **kwargs) -> dict:
    """C1: a still lake on the paraboloid must survive ``T/2`` untouched."""
    params = dict(DEFAULTS)
    params.update(kwargs)
    basin = build_basin(**{k: params[k] for k in
                           ("H", "a", "amplitude", "gravity")})
    solver = _solver(basin, n, params)
    solver.lake_at_rest_state(0.0)
    dh, dhu, dhv = solver.rhs(solver.h, solver.hu, solver.hv)
    initial_rhs = max(float(np.max(np.abs(dh))), float(np.max(np.abs(dhu))),
                      float(np.max(np.abs(dhv))))
    solver.advance(half_period * basin.period)
    max_u = solver.max_spurious_velocity()
    drift = solver.surface_drift()
    return {
        "resolution": n,
        "t_end": float(solver.time),
        "steps": int(solver.steps),
        "initial_rhs_max": initial_rhs,
        "max_velocity": max_u,
        "surface_drift": float(drift),
        "tolerance": params["tolerance_lake_at_rest"],
        "pass": bool(max(max_u, drift, initial_rhs)
                     <= params["tolerance_lake_at_rest"]),
    }


def negative_control(n: int = 64, fraction: float = 0.1,
                     max_steps: int = 1200, **kwargs) -> dict:
    """C8: the same lake at rest with the naive centred bed source must fail."""
    params = dict(DEFAULTS)
    params.update(kwargs)
    basin = build_basin(**{k: params[k] for k in
                           ("H", "a", "amplitude", "gravity")})
    solver = _solver(basin, n, params, well_balanced=False)
    solver.lake_at_rest_state(0.0)
    blew_up = None
    try:
        solver.advance(fraction * basin.period, max_steps=max_steps)
    except (RuntimeError, FloatingPointError) as exc:      # pragma: no cover
        blew_up = str(exc)
    max_u = solver.max_spurious_velocity()
    return {
        "resolution": n,
        "t_end": float(solver.time),
        "steps": int(solver.steps),
        "max_velocity": max_u,
        "surface_drift": float(solver.surface_drift()),
        "failed": bool(max_u > 1e-6 or blew_up is not None),
        "error": blew_up,
        "note": "centred bed source, no hydrostatic reconstruction",
    }


def convergence_study(resolutions=None, **kwargs) -> dict:
    """C7: refine the grid and require monotone decrease with order >= 1."""
    params = dict(DEFAULTS)
    params.update(kwargs)
    resolutions = list(resolutions or params["resolutions"])
    rows = []
    for n in resolutions:
        run = run_case(n=n, **{k: v for k, v in params.items()})
        print("  convergence n = %d: centre L2 = %.4f %%, shore %.4f %%"
              % (n, 100 * run["centre_depth"]["l2_relative"],
                 100 * run["shoreline"]["mean_relative"]), flush=True)
        rows.append({"n": n, "dx": run["dx"],
                     "centre_l2": run["centre_depth"]["l2_relative"],
                     "centre_max": run["centre_depth"]["max_relative"],
                     "shore_mean": run["shoreline"]["mean_relative"],
                     "symmetry": run["symmetry"]["normalised"]})
    orders = {}
    for key in ("centre_l2", "shore_mean"):
        e = [max(row[key], 1e-16) for row in rows]
        p = [math.log(e[i] / e[i + 1]) / math.log(2.0)
             for i in range(len(e) - 1)]
        orders[key] = p
    monotone = all(rows[i]["centre_l2"] >= rows[i + 1]["centre_l2"]
                   for i in range(len(rows) - 1))
    observed = orders["centre_l2"][-1] if orders["centre_l2"] else 0.0
    return {
        "rows": rows,
        "orders_centre": orders["centre_l2"],
        "orders_shore": orders["shore_mean"],
        "observed_order": observed,
        "monotone": bool(monotone),
        "pass": bool(monotone and observed >= 1.0),
    }


def exactness_check(**kwargs) -> dict:
    """C2: substitute the closed forms into the SWE and report the residual."""
    params = dict(DEFAULTS)
    params.update(kwargs)
    basin = build_basin(**{k: params[k] for k in
                           ("H", "a", "amplitude", "gravity")})
    out = check_thacker_solution(basin)
    out["tolerance"] = params["tolerance_exactness"]
    out["pass"] = bool(out["max_relative_residual"]
                       <= params["tolerance_exactness"])
    return out


# --------------------------------------------------------------------------- #
# figures and artefacts
# --------------------------------------------------------------------------- #
def figure_thacker(run: dict, profiles, convergence: dict, lake: dict,
                   path) -> str:
    """Four panels: depth field, profiles, histories, convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basin = ThackerBasin(**{k: run["basin"][k] for k in
                            ("H", "a", "amplitude", "gravity")})
    series = run["series"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2))

    ax = axes[0, 0]
    # rebuild the field of the last sampled instant from a dedicated run
    n = run["resolution"]
    solver = _solver(basin, n, dict(DEFAULTS, resolution=n))
    X, Y = grid(basin, n, DEFAULTS["domain_factor"])
    solver.set_state(*basin.initial_state(X, Y))
    t_show = 0.25 * basin.period
    solver.advance(t_show)
    extent = [solver.x_min, solver.x_max, solver.y_min, solver.y_max]
    im = ax.imshow(solver.h.T, origin="lower", extent=extent,
                   cmap="viridis", vmin=0.0)
    theta = np.linspace(0.0, 2.0 * math.pi, 400)
    s_ex = basin.shoreline_radius(t_show)
    ax.plot(s_ex * np.cos(theta), s_ex * np.sin(theta), "w--", lw=1.2,
            label=f"exact shoreline s = {s_ex:.4f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title(f"depth h at t* = t/T = 0.25 (n = {n})")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, shrink=0.85, label="h [m]")

    ax = axes[0, 1]
    colors = ["tab:blue", "tab:red"]
    for colour, prof in zip(colors, profiles):
        ax.plot(prof["x"], prof["h_exact"], "-", color=colour, alpha=0.5,
                lw=1.4, label=f"exact, t/T = {prof['phase']:g}")
        ax.plot(prof["x"], prof["h"], "--", color=colour, lw=1.2,
                label=f"FVM, t/T = {prof['phase']:g}")
        ax.axvline(prof["s_exact"], color=colour, ls=":", lw=0.9, alpha=0.7)
    ax.set_xlim(0.0, 1.45)
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel("x [m] (y = 0)")
    ax.set_ylabel("h [m]")
    ax.set_title("shore-normal depth profiles")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    ax.plot(series["t_over_T"], series["h_centre"], "o-", ms=3,
            color="tab:blue", label="FVM centre depth")
    ax.plot(series["t_over_T"], series["h_centre_exact"], "-", lw=1.0,
            color="tab:blue", alpha=0.5, label="exact centre depth")
    ax.set_xlabel("t / T")
    ax.set_ylabel("h(x=0,y=0) [m]", color="tab:blue")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(series["t_over_T"], series["s"], "s--", ms=3, color="tab:green",
             label="FVM shoreline radius")
    ax2.plot(series["t_over_T"], series["s_exact"], "-", lw=1.0,
             color="tab:green", alpha=0.5, label="exact shoreline radius")
    ax2.set_ylabel("s [m]", color="tab:green")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")
    ax.set_title("periods: centre depth and shoreline (2 T)")

    ax = axes[1, 1]
    rows = convergence["rows"]
    dx = [row["dx"] for row in rows]
    ax.loglog(dx, [row["centre_l2"] for row in rows], "o-",
              label="centre-depth L2 error")
    ax.loglog(dx, [row["shore_mean"] for row in rows], "s-",
              label="shoreline mean error")
    ax.axhline(DEFAULTS["tolerance_centre"], color="k", ls=":", lw=1.0)
    ax.text(dx[0], DEFAULTS["tolerance_centre"] * 1.1, " 3 % criterion",
            fontsize=8)
    ref = [rows[-1]["centre_l2"] * (rows[-1]["dx"] / d) for d in dx]
    ax.loglog(dx, ref, "k--", lw=0.9, alpha=0.6, label="slope 1 reference")
    ax.set_xlabel("d [m]")
    ax.set_ylabel("relative error [-]")
    ax.set_title("grid convergence (C7)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_artifacts(outdir, payload: dict) -> dict:
    """Write ``outdir`` artefacts (JSON + figures) and a generated SUMMARY.md."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("metrics", "lake_at_rest", "negative_control", "exactness",
                 "convergence", "profiles"):
        if name in payload:
            path = out / f"{name}.json"
            path.write_text(json.dumps(payload[name], indent=2) + "\n")
            written[name] = str(path)
    if "figure" in payload:
        written["figure"] = payload["figure"]

    run = payload["metrics"]
    lake = payload["lake_at_rest"]
    neg = payload["negative_control"]
    exact = payload["exactness"]
    conv = payload["convergence"]
    lines = [
        "# CASE-FVM-THACKER-BASIN — generated summary",
        "",
        "| criterion | measured | threshold | verdict |",
        "|---|---|---|---|",
        f"| C1 lake at rest max\\|u\\| | {lake['max_velocity']:.3g} m/s "
        f"(drift {lake['surface_drift']:.3g} m) | <= {lake['tolerance']:.1e} "
        f"| {'PASS' if lake['pass'] else 'FAIL'} |",
        f"| C2 closed-form residual | {exact['max_relative_residual']:.3g} "
        f"| <= {exact['tolerance']:.1e} | "
        f"{'PASS' if exact['pass'] else 'FAIL'} |",
        f"| C3 volume vs exact | "
        f"{100 * run['volume']['max_relative_to_exact']:.4f} % "
        f"(drift {run['volume']['max_drift']:.3g}) | "
        f"<= {100 * run['volume']['tolerance_relative']:.1f} % / "
        f"<= {run['volume']['tolerance_drift']:.0e} | "
        f"{'PASS' if run['criteria']['C3_volume'] else 'FAIL'} |",
        f"| C4 centre depth L2 | "
        f"{100 * run['centre_depth']['l2_relative']:.3f} % | "
        f"<= {100 * run['centre_depth']['tolerance']:.0f} % | "
        f"{'PASS' if run['criteria']['C4_centre_depth'] else 'FAIL'} |",
        f"| C5 shoreline mean | "
        f"{100 * run['shoreline']['mean_relative']:.3f} % | "
        f"<= {100 * run['shoreline']['tolerance']:.0f} % | "
        f"{'PASS' if run['criteria']['C5_shoreline'] else 'FAIL'} |",
        f"| C6 symmetry defect | "
        f"{100 * run['symmetry']['normalised']:.3f} % | "
        f"<= {100 * run['symmetry']['tolerance']:.0f} % | "
        f"{'PASS' if run['criteria']['C6_symmetry'] else 'FAIL'} |",
        f"| C7 convergence order | {conv['observed_order']:.2f} "
        f"(monotone: {conv['monotone']}) | >= 1.0 | "
        f"{'PASS' if conv['pass'] else 'FAIL'} |",
        f"| C8 naive source control | max\\|u\\| = {neg['max_velocity']:.3g} m/s "
        f"| must fail | {'PASS (fails as required)' if neg['failed'] else 'FAIL'} |",
        f"| C9 health | h_min = {run['health']['h_min']:.3g}, "
        f"clipped = {run['health']['clipped_cells']}, "
        f"dry flips = {run['health']['dry_flips']}, "
        f"boundary h = {run['health']['boundary_max_depth']:.3g} | "
        f"h_min >= 0, clipped = 0, boundary dry | "
        f"{'PASS' if run['criteria']['C9_health'] else 'FAIL'} |",
        "",
        f"Resolution n = {run['resolution']} ({run['discretisation']['nx']} x "
        f"{run['discretisation']['ny']} cells, d = {run['dx']:.5f} m), "
        f"{run['health']['steps']} steps, 2 periods.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    written["summary"] = str(out / "SUMMARY.md")
    return written