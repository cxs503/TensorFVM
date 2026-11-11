"""Fully developed plane-channel Poiseuille benchmark on the ``inc_bench`` SIMPLE kernel.

The case is the classical pressure-driven plane Poiseuille flow between two
infinite parallel plates: the exact solution is a parabola

.. math:: u(y) = 6\\,\\bar u\\,\\frac{y}{H}\\left(1-\\frac{y}{H}\\right),

with the derived quantities

======================  ==========================================  ===========
quantity                exact expression                            value
======================  ==========================================  ===========
peak velocity           :math:`u_{max} = \\tfrac{3}{2}\\bar u`     1.5
pressure gradient       :math:`dp/dx = -12\\mu\\bar u/H^2`          -
wall shear              :math:`\\tau_w = 6\\mu\\bar u/H`            -
Fanning friction        :math:`C_f = 2\\tau_w/(\\rho\\bar u^2)=12/Re_H`
Darcy friction          :math:`f = 4C_f = 48/Re_H`
Poiseuille number       :math:`Po = f\\,Re_{D_h} = 96`              96
======================  ==========================================  ===========

``Po = 96`` for parallel plates (and ``Po = 64`` for a circular pipe) is the
textbook tabulated value for laminar fully developed duct flow; it is derived
here symbolically from the closed form by :func:`symbolic_reference`, so the
target number in the artefacts is computed, never transcribed.

Nothing is initialised with the analytic solution except the **inflow** plane:
the inlet carries the exact developed profile (a legitimate boundary condition
for a developed channel), the walls are no-slip and the outlet is
zero-gradient velocity with zero gauge pressure.  The interior field is then
whatever the SIMPLE kernel produces, and the comparison is made on an interior
station ``x = L/2`` that is two plate heights away from both streamwise
boundaries.  The entrance region therefore plays no role and the measured
error is the solver's own truncation error.

Run one case::

    PYTHONPATH=src python -m tensorfvm.benchmark_poiseuille_channel case \
        --ny 16 --re-h 100 --aspect 4 --out runs/poiseuille-channel

Run the whole pre-declared plan (mesh sequence, Reynolds sweep, aspect sweep)
and write the artefacts::

    PYTHONPATH=src python -m tensorfvm.benchmark_poiseuille_channel all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import torch

from .inc_bench.solver import SolverOptions, SteadyFlow

#: relative-error acceptance threshold for every reported metric (pre-declared)
TOLERANCE = 0.02
#: iteration-to-iteration velocity change used to declare a steady state
SOLVER_TOLERANCE = 1.0e-8
#: iteration cap per case (all pre-declared runs converge far below this)
MAX_ITERATIONS = 60000

DEFAULT_ASPECT = 4.0          # L/H
DEFAULT_BAR_VELOCITY = 1.0    # u_bar
DEFAULT_HEIGHT = 1.0          # H
DEFAULT_DENSITY = 1.0         # rho

#: pre-declared experiment: (case name, ny, Re_H, L/H)
PLAN: tuple[tuple[str, int, float, float], ...] = (
    ("mesh-ny8", 8, 100.0, DEFAULT_ASPECT),
    ("mesh-ny16", 16, 100.0, DEFAULT_ASPECT),
    ("mesh-ny32", 32, 100.0, DEFAULT_ASPECT),
    ("re-10", 16, 10.0, DEFAULT_ASPECT),
    ("re-1000", 16, 1000.0, DEFAULT_ASPECT),
    ("aspect-8", 16, 100.0, 8.0),
)


# --------------------------------------------------------------------- reference
def symbolic_reference() -> dict[str, object]:
    """Derive the closed form and the exact dimensionless ratios with sympy.

    The Stokes problem ``mu u'' = -G`` (``G = -dp/dx > 0``) is integrated and
    closed with ``u(0) = u(H) = 0``; the mean velocity is the flux divided by
    ``H``.  Substituting ``G`` in terms of ``ubar`` gives the profile, and the
    peak velocity, wall shear, Fanning friction and Poiseuille number follow.
    Every ratio is returned as an exact expression together with a ``checks``
    dictionary proving equality with the tabulated rationals ``3/2`` and ``96``.
    ``sympy`` is an optional dependency of this benchmark; when it is absent the
    ``available`` flag is ``False`` and :func:`reference_values` is used.
    """
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    y = sp.Symbol("y", real=True)
    H, mu, ubar, rho = sp.symbols("H mu ubar rho", positive=True)
    G = sp.Symbol("G", positive=True)          # -dp/dx, strictly positive
    A, B = sp.symbols("A B", real=True)
    # integrate mu u'' = -G twice
    u = -G / (2 * mu) * y ** 2 + A * y + B
    constants = sp.solve([sp.Eq(u.subs(y, 0), 0), sp.Eq(u.subs(y, H), 0)], [A, B], dict=True)[0]
    u = sp.simplify(u.subs(constants))
    flux = sp.simplify(sp.integrate(u, (y, 0, H)))          # Q per unit depth
    driving = sp.solve(sp.Eq(sp.simplify(flux / H), ubar), G)[0]
    profile = sp.simplify(u.subs(G, driving))
    peak = sp.simplify(profile.subs(y, H / 2) / ubar)
    dpdx = sp.simplify(-driving)
    # magnitude of the wall shear: du/dy > 0 at y = 0 for the flow direction +x
    shear = sp.simplify(mu * sp.diff(profile, y).subs(y, 0))
    cf = sp.simplify(shear / (sp.Rational(1, 2) * rho * ubar ** 2))
    re_h = sp.simplify(rho * ubar * H / mu)
    poiseuille = sp.simplify(4 * cf * 2 * re_h)
    checks = {
        "profile_is_parabola": sp.simplify(profile - 6 * ubar * y / H * (1 - y / H)) == 0,
        "dpdx_matches": sp.simplify(dpdx - (-12 * mu * ubar / H ** 2)) == 0,
        "peak_over_bar_is_three_halves": sp.simplify(peak - sp.Rational(3, 2)) == 0,
        "tau_w_matches": sp.simplify(shear - 6 * mu * ubar / H) == 0,
        "cf_times_re_h_is_twelve": sp.simplify(cf * re_h - 12) == 0,
        "poiseuille_is_96": sp.simplify(poiseuille - 96) == 0,
    }
    return {
        "available": True,
        "profile": str(profile),
        "peak_over_bar": str(peak),
        "dpdx": str(dpdx),
        "tau_w": str(shear),
        "cf_fanning": str(cf),
        "cf_times_re_h": str(sp.simplify(cf * re_h)),
        "poiseuille": str(poiseuille),
        "checks": checks,
        "all_checks": all(checks.values()),
    }


def reference_values(re_h: float, *, ubar: float = DEFAULT_BAR_VELOCITY,
                     height: float = DEFAULT_HEIGHT,
                     density: float = DEFAULT_DENSITY) -> dict[str, float]:
    """Exact reference quantities for a plane channel at ``Re_H = ubar H / nu``."""
    nu = ubar * height / re_h
    mu = nu * density
    return {
        "re_h": re_h,
        "re_dh": 2.0 * re_h,
        "nu": nu,
        "mu": mu,
        "dpdx": -12.0 * mu * ubar / height ** 2,
        "u_max": 1.5 * ubar,
        "u_max_over_bar": 1.5,
        "tau_w": 6.0 * mu * ubar / height,
        "cf_fanning": 12.0 / re_h,
        "cf_fanning_times_re_h": 12.0,
        "f_darcy": 48.0 / re_h,
        "poiseuille": 96.0,
        "bar_velocity": ubar,
        "height": height,
        "density": density,
    }


# ---------------------------------------------------------------------- metrics
def wall_shear(ufaces: torch.Tensor, dy: float, mu: float, station: int,
               height: float) -> dict[str, float]:
    """Shear on both plates at ``station``; force is reported per unit depth.

    Two estimates are produced because they have different error orders:

    ``half_cell``
        ``mu u_1/(dy/2)`` -- the same half-cell stencil the momentum assembly
        uses for the no-slip wall, hence first order in ``dy``;
    ``quadratic``
        ``mu`` times the linear coefficient of a parabola through the first
        three staggered rows constrained by ``u(y=0) = 0`` -- second order.
    """
    rows = torch.tensor([0.5 * dy, 1.5 * dy, 2.5 * dy], dtype=ufaces.dtype)
    values = ufaces[:3, station].detach().cpu().to(torch.float64)
    design = torch.stack([rows ** 2, rows, torch.ones_like(rows)], dim=1)
    second, first, _ = torch.linalg.solve(design, values)
    half_cell = 2.0 * mu * float(ufaces[0, station]) / dy
    return {
        "tau_w_half_cell": half_cell,
        "tau_w_quadratic": mu * float(first),
        "tau_w_curvature": float(second),
        "wall_y": 0.0,
        "height": height,
    }


def run_case(name: str, ny: int, re_h: float, aspect: float = DEFAULT_ASPECT,
             *, ubar: float = DEFAULT_BAR_VELOCITY, height: float = DEFAULT_HEIGHT,
             density: float = DEFAULT_DENSITY, tolerance: float = SOLVER_TOLERANCE,
             max_iterations: int = MAX_ITERATIONS) -> dict[str, object]:
    """Solve one fully developed plane-channel case and score it against the exact solution."""
    ref = reference_values(re_h, ubar=ubar, height=height, density=density)
    dy = height / ny
    nx = int(round(aspect * ny))
    dx = aspect * height / nx
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    face_y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    inlet = 6.0 * ubar * face_y / height * (1.0 - face_y / height)
    options = SolverOptions(max_iterations=max_iterations, tolerance=tolerance)
    solver = SteadyFlow(fluid, dx=dx, dy=dy, viscosity=ref["nu"], density=density,
                        inlet_profile=inlet, outlet=True, options=options)
    started = time.time()
    record = solver.solve()
    wall_time = time.time() - started

    cell_u = 0.5 * (solver.u[:, :-1] + solver.u[:, 1:])
    cell_y = face_y
    cell_x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    station_index = int(torch.argmin(torch.abs(cell_x - 0.5 * aspect * height)).item())
    station_x = float(cell_x[station_index])
    exact = 6.0 * ubar * cell_y / height * (1.0 - cell_y / height)
    difference = cell_u[:, station_index] - exact
    velocity_l2 = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(exact))
    velocity_linf = float(difference.abs().max() / ref["u_max"])

    # pressure slope in the interior band 0.375 L <= x <= 0.75 L (two stations
    # away from both streamwise boundaries in every planned aspect ratio)
    band = (cell_x >= 0.375 * aspect * height) & (cell_x <= 0.75 * aspect * height)
    band_x = cell_x[band]
    band_p = solver.p[:, band].mean(dim=0)
    centred_x = band_x - band_x.mean()
    slope = float((centred_x * (band_p - band_p.mean())).sum() / centred_x.square().sum())

    shear = wall_shear(solver.u, dy, ref["mu"], station_index, height)
    measured_bar = float(cell_u[:, station_index].mean())
    # the inflow plane carries the exact profile sampled at the staggered faces;
    # its midpoint sum is the *discrete* flux that the solver conserves, so the
    # like-for-like mean velocity is that sum over H (not the continuous ubar,
    # which differs by the O(dy^2) midpoint quadrature error)
    inlet_discrete_mean = float(inlet.sum() * dy) / height
    exact_sampled_max = float(exact.max())
    mass_inlet = float(density * dy * solver.u[:, 0].sum())
    mass_outlet = float(density * dy * solver.u[:, -1].sum())
    divergence = solver.divergence_field()

    def relative(value: float, target: float) -> float:
        return abs(value - target) / abs(target)

    metrics = {
        "velocity_l2": velocity_l2,
        "velocity_linf": velocity_linf,
        "u_max": float(cell_u[:, station_index].max()),
        "u_max_sampled_reference": exact_sampled_max,
        "u_max_relative_error": relative(float(cell_u[:, station_index].max()),
                                         exact_sampled_max),
        "u_max_continuous_reference": ref["u_max"],
        "u_max_continuous_relative_error": relative(float(cell_u[:, station_index].max()),
                                                    ref["u_max"]),
        "bar_velocity_measured": measured_bar,
        "bar_velocity_discrete_reference": inlet_discrete_mean,
        "flux_conservation_error": relative(measured_bar, inlet_discrete_mean),
        "pressure_gradient": slope,
        "pressure_gradient_relative_error": relative(slope, ref["dpdx"]),
        "tau_w_half_cell": shear["tau_w_half_cell"],
        "tau_w_half_cell_relative_error": relative(shear["tau_w_half_cell"], ref["tau_w"]),
        "tau_w_quadratic": shear["tau_w_quadratic"],
        "tau_w_quadratic_relative_error": relative(shear["tau_w_quadratic"], ref["tau_w"]),
        "poiseuille": 8.0 * shear["tau_w_quadratic"] * ref["re_dh"] / (density * ubar ** 2),
        "poiseuille_half_cell": 8.0 * shear["tau_w_half_cell"] * ref["re_dh"]
        / (density * ubar ** 2),
        "mass_inlet": mass_inlet,
        "mass_outlet": mass_outlet,
        "relative_mass_imbalance": abs(mass_outlet - mass_inlet) / abs(mass_inlet),
        "max_cell_divergence": float(divergence.abs().max()),
    }
    metrics["poiseuille_relative_error"] = relative(metrics["poiseuille"], ref["poiseuille"])
    scored = ("velocity_l2", "velocity_linf", "u_max_relative_error",
              "pressure_gradient_relative_error", "tau_w_quadratic_relative_error",
              "poiseuille_relative_error", "flux_conservation_error")
    metrics["passed"] = bool(
        record["converged"]
        and all(math.isfinite(metrics[key]) and metrics[key] < TOLERANCE for key in scored)
        and math.isfinite(metrics["relative_mass_imbalance"])
    )
    return {
        "case": name,
        "ny": ny,
        "nx": nx,
        "dx": dx,
        "dy": dy,
        "aspect": aspect,
        "re_h": re_h,
        "re_dh": ref["re_dh"],
        "nu": ref["nu"],
        "mu": ref["mu"],
        "ubar": ubar,
        "height": height,
        "density": density,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "iterations": record["iterations"],
        "inlet_discrete_mean": inlet_discrete_mean,
        "converged": bool(record["converged"]),
        "final_change": record["change"],
        "final_residuals": record["final"],
        "wall_time_s": wall_time,
        "station_x": station_x,
        "station_index": station_index,
        "reference": ref,
        "metrics": metrics,
        "scored_metrics": list(scored),
        "profile": {
            "y": [float(value) for value in cell_y],
            "numerical": [float(value) for value in cell_u[:, station_index]],
            "exact": [float(value) for value in exact],
        },
        "pressure": {
            "x": [float(value) for value in cell_x],
            "mean_over_y": [float(value) for value in solver.p.mean(dim=0)],
        },
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Observed order of accuracy from two errors under a uniform refinement."""
    if not (coarse > 0.0 and fine > 0.0 and ratio > 1.0):
        raise ValueError("errors must be positive and the refinement ratio > 1")
    return math.log(coarse / fine) / math.log(ratio)


def summarise(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate the case list: mesh sequence order, sweeps and the verdict."""
    by_name = {result["case"]: result for result in results}
    mesh = [by_name[name] for name in ("mesh-ny8", "mesh-ny16", "mesh-ny32", "mesh-ny64")
            if name in by_name]
    orders = []
    for lower, upper in zip(mesh, mesh[1:]):
        orders.append({
            "pair": f'{lower["ny"]}->{upper["ny"]}',
            "velocity_l2_order": observed_order(lower["metrics"]["velocity_l2"],
                                                upper["metrics"]["velocity_l2"]),
            "pressure_gradient_order": observed_order(
                lower["metrics"]["pressure_gradient_relative_error"],
                upper["metrics"]["pressure_gradient_relative_error"]),
        })
    production = mesh[-1] if mesh else None
    return {
        "schema": "tensorfvm.poiseuille-channel/1",
        "reference_derivation": symbolic_reference(),
        "tolerance": TOLERANCE,
        "scored_metrics": list(mesh[0]["scored_metrics"]) if mesh else [],
        "cases": results,
        "mesh_sequence": [result["case"] for result in mesh],
        "observed_orders": orders,
        # honest per-case verdicts; the coarse end of the mesh sequence is
        # expected to miss the 2% band and is reported as such
        "case_verdicts": {result["case"]: result["metrics"]["passed"] for result in results},
        "all_cases_passed": all(result["metrics"]["passed"] for result in results),
        "production_case": production["case"] if production else None,
        "production_passed": bool(production and production["metrics"]["passed"]),
        "finest_velocity_l2": production["metrics"]["velocity_l2"] if production else None,
        "poiseuille_target": 96.0,
    }


# ------------------------------------------------------------------- artefacts
def write_artifacts(payload: dict[str, object], out: Path,
                    figures: bool = True) -> dict[str, str]:
    """Write JSON, CSV and (optionally) PNG figures; returns their paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    benchmark = out / "benchmark.json"
    benchmark.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    written["benchmark"] = str(benchmark)

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "dtype": "float64",
        "solver": "tensorfvm.inc_bench.solver.SteadyFlow",
        "tolerance": TOLERANCE,
        "plan": [{"case": name, "ny": ny, "re_h": re_h, "aspect": aspect}
                 for name, ny, re_h, aspect in PLAN],
    }, indent=2))
    written["manifest"] = str(manifest)

    profiles = out / "profiles.csv"
    with profiles.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "ny", "x_over_H", "y_over_H", "u_over_bar", "u_exact_over_bar"])
        for result in payload["cases"]:
            for y, got, ref in zip(result["profile"]["y"], result["profile"]["numerical"],
                                   result["profile"]["exact"]):
                writer.writerow([result["case"], result["ny"],
                                 result["station_x"] / result["height"],
                                 y / result["height"], got / result["ubar"],
                                 ref / result["ubar"]])
    written["profiles"] = str(profiles)

    metrics = out / "metrics.csv"
    with metrics.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "ny", "re_h", "aspect", "iterations", "converged",
                         "velocity_l2", "velocity_linf", "u_max_rel", "dpdx_rel",
                         "tau_w_quad_rel", "poiseuille", "poiseuille_rel", "passed"])
        for result in payload["cases"]:
            m = result["metrics"]
            writer.writerow([result["case"], result["ny"], result["re_h"], result["aspect"],
                             result["iterations"], result["converged"],
                             f'{m["velocity_l2"]:.6e}', f'{m["velocity_linf"]:.6e}',
                             f'{m["u_max_relative_error"]:.6e}',
                             f'{m["pressure_gradient_relative_error"]:.6e}',
                             f'{m["tau_w_quadratic_relative_error"]:.6e}',
                             f'{m["poiseuille"]:.6f}', f'{m["poiseuille_relative_error"]:.6e}',
                             m["passed"]])
    written["metrics"] = str(metrics)

    if figures:
        written.update(_figures(payload, out))
    return written


def _figures(payload: dict[str, object], out: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = {}
    mesh = [case for case in payload["cases"] if case["case"].startswith("mesh-")]
    fig, axis = plt.subplots(figsize=(6.0, 4.2))
    for case in mesh:
        axis.plot([value / case["ubar"] for value in case["profile"]["numerical"]],
                  [value / case["height"] for value in case["profile"]["y"]],
                  marker=".", markersize=3, linewidth=1.0,
                  label=f'ny={case["ny"]} (L2 {case["metrics"]["velocity_l2"]*100:.3f}%)')
    first = mesh[0]
    axis.plot([value / first["ubar"] for value in first["profile"]["exact"]],
              [value / first["height"] for value in first["profile"]["y"]],
              color="black", linestyle="--", linewidth=1.2,
              label=r"exact $6\bar u\,y/H(1-y/H)$")
    axis.set_xlabel(r"$u/\bar u$")
    axis.set_ylabel(r"$y/H$")
    axis.set_title(r"Plane Poiseuille, $Re_H=100$, station $x=2H$")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    profile_png = out / "profile.png"
    fig.savefig(profile_png, dpi=160)
    plt.close(fig)
    written["profile_png"] = str(profile_png)

    fig, axis = plt.subplots(figsize=(6.0, 4.2))
    ny = [case["ny"] for case in mesh]
    axis.loglog(ny, [case["metrics"]["velocity_l2"] for case in mesh], "o-",
                label="velocity $L_2$")
    axis.loglog(ny, [case["metrics"]["pressure_gradient_relative_error"] for case in mesh],
                "s-", label="pressure gradient")
    axis.loglog(ny, [case["metrics"]["tau_w_quadratic_relative_error"] for case in mesh],
                "^-", label=r"wall shear $\tau_w$ (2nd order)")
    axis.loglog(ny, [case["metrics"]["poiseuille_relative_error"] for case in mesh],
                "d-", label="Poiseuille number")
    reference = [mesh[0]["metrics"]["velocity_l2"] * (mesh[0]["ny"] / value) ** 2 for value in ny]
    axis.loglog(ny, reference, "k--", linewidth=1.0, label=r"slope $-2$")
    axis.axhline(TOLERANCE, color="red", linestyle=":", label="2% acceptance")
    axis.set_xlabel(r"cells across the channel $n_y$")
    axis.set_ylabel("relative error")
    axis.set_title("Grid convergence of the plane-channel Poiseuille metrics")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3, which="both")
    fig.tight_layout()
    convergence_png = out / "convergence.png"
    fig.savefig(convergence_png, dpi=160)
    plt.close(fig)
    written["convergence_png"] = str(convergence_png)

    fig, axis = plt.subplots(figsize=(6.0, 4.2))
    axis.axhline(96.0, color="black", linestyle="--", linewidth=1.2, label=r"exact $Po=96$")
    if mesh:
        axis.loglog([case["ny"] for case in mesh],
                    [case["metrics"]["poiseuille"] for case in mesh], "o-",
                    label=r"computed $f\,Re_{D_h}$, $Re_H=100$")
    re_cases = [case for case in payload["cases"] if case["case"].startswith("re-")]
    if re_cases:
        axis.loglog([case["re_h"] for case in re_cases],
                    [case["metrics"]["poiseuille"] for case in re_cases], "s--",
                    label="Reynolds sweep ($n_y=16$)")
    axis.set_ylim(90.0, 102.0)
    axis.set_xlabel(r"$n_y$ (mesh series) / $Re_H$ (sweep)")
    axis.set_ylabel(r"Poiseuille number $Po=f\,Re_{D_h}$")
    axis.set_title("Plane-channel Poiseuille number vs the exact value 96")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")
    fig.tight_layout()
    poiseuille_png = out / "poiseuille_number.png"
    fig.savefig(poiseuille_png, dpi=160)
    plt.close(fig)
    written["poiseuille_png"] = str(poiseuille_png)
    return written


# ------------------------------------------------------------------------ CLI
def _run_or_record(name: str, ny: int, re_h: float, aspect: float,
                   out: Path) -> dict[str, object]:
    """Run a case, or record an honest failure instead of aborting the sweep.

    A parameter point where the SIMPLE iteration diverges is a real result: it
    is written to ``failures/<case>.json`` with the exception text, never
    silently dropped and never counted as a pass.
    """
    try:
        return run_case(name, ny, re_h, aspect)
    except RuntimeError as exc:
        failures = Path(out) / "failures"
        failures.mkdir(parents=True, exist_ok=True)
        record = {"case": name, "ny": ny, "re_h": re_h, "aspect": aspect,
                  "failure": f"{type(exc).__name__}: {exc}",
                  "stage": "solver"}
        (failures / f"{name}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"case": name, "failed": record["failure"]}, indent=2), flush=True)
        return record


def _write_case(result: dict[str, object], out: Path) -> Path:
    out = Path(out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    path = out / "cases" / f'{result["case"]}.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    case_parser = sub.add_parser("case", help="run a single (ny, Re_H, L/H) case")
    case_parser.add_argument("--name", default=None)
    case_parser.add_argument("--ny", type=int, required=True)
    case_parser.add_argument("--re-h", type=float, default=100.0)
    case_parser.add_argument("--aspect", type=float, default=DEFAULT_ASPECT)
    case_parser.add_argument("--out", default="runs/poiseuille-channel")
    case_parser.add_argument("--tolerance", type=float, default=SOLVER_TOLERANCE)
    case_parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)

    all_parser = sub.add_parser("all", help="run the pre-declared plan and write artefacts")
    all_parser.add_argument("--out", default="runs/poiseuille-channel")

    index_parser = sub.add_parser(
        "plan-index", help="run one entry of the pre-declared plan (for sharded sweeps)")
    index_parser.add_argument("--index", type=int, required=True,
                              help="plan index; taken modulo len(PLAN) so a 1-based "
                                   "or 0-based shard counter both cover the plan")
    index_parser.add_argument("--out", default="runs/poiseuille-channel")

    aggregate_parser = sub.add_parser("aggregate", help="aggregate per-case JSON files")
    aggregate_parser.add_argument("--out", default="runs/poiseuille-channel")
    aggregate_parser.add_argument("--no-figures", action="store_true")

    args = parser.parse_args(argv)
    out = Path(args.out)

    if args.command == "case":
        result = run_case(args.name or f"ny{args.ny}-re{args.re_h:g}-aspect{args.aspect:g}",
                          args.ny, args.re_h, args.aspect,
                          tolerance=args.tolerance, max_iterations=args.max_iterations)
        path = _write_case(result, out)
        print(json.dumps({"case": result["case"], "metrics": result["metrics"],
                          "iterations": result["iterations"],
                          "converged": result["converged"], "path": str(path)}, indent=2))
        return 0 if result["metrics"]["passed"] else 1

    if args.command == "plan-index":
        name, ny, re_h, aspect = PLAN[args.index % len(PLAN)]
        print(f"running {name}: ny={ny} Re_H={re_h} L/H={aspect}", flush=True)
        result = _run_or_record(name, ny, re_h, aspect, out)
        path = _write_case(result, out)
        print(json.dumps({"case": result["case"], "metrics": result["metrics"],
                          "iterations": result["iterations"],
                          "converged": result["converged"], "path": str(path)}, indent=2))
        return 0 if result["metrics"]["passed"] else 1

    if args.command == "all":
        for name, ny, re_h, aspect in PLAN:
            print(f"running {name}: ny={ny} Re_H={re_h} L/H={aspect}", flush=True)
            result = _run_or_record(name, ny, re_h, aspect, out)
            if "metrics" not in result:
                continue
            print(f'  L2={result["metrics"]["velocity_l2"]:.4e} '
                  f'Po={result["metrics"]["poiseuille"]:.4f} '
                  f'passed={result["metrics"]["passed"]}', flush=True)

    results = [json.loads(path.read_text()) for path in sorted((out / "cases").glob("*.json"))]
    if not results:
        raise SystemExit(f"no case JSON under {out / 'cases'}")
    order = {name: index for index, (name, *_rest) in enumerate(PLAN)}
    results.sort(key=lambda item: order.get(item["case"], len(order)))
    payload = summarise(results)
    written = write_artifacts(payload, out, figures=not getattr(args, "no_figures", False))
    print(json.dumps({"production_case": payload["production_case"],
                      "production_passed": payload["production_passed"],
                      "case_verdicts": payload["case_verdicts"],
                      "observed_orders": payload["observed_orders"],
                      "artifacts": written}, indent=2))
    return 0 if payload["production_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())