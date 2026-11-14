"""Entrance-length benchmark: uniform-inlet channel flow developing to Poiseuille.

CASE-FVM-CHANNEL-DEVELOP  ·  TensorFVM  ·  anchor for the benchmark pool.

Physical problem
----------------
A 2-D plane channel of height *H* receives a **uniform** (plug) inlet velocity
:math:`\\bar u`.  Viscous diffusion from the no-slip walls builds boundary layers
that merge downstream until the profile becomes the fully developed parabola

.. math:: u(y) = 6\\,\\bar u\\,\\frac{y}{H}\\left(1-\\frac{y}{H}\\right).

The **entrance length** :math:`L_e` is the streamwise distance at which the
centreline velocity first reaches 99 % of its fully developed value
:math:`u_{max}=1.5\\,\\bar u`.

Reference (zero hand-transcription)
-----------------------------------
White, F. M., *Viscous Fluid Flow*, 3rd ed., McGraw-Hill, 2006, §3-3.1
"The Entrance Region", Eq. (3-39) and surrounding discussion:

    ``L_e / H ≈ 0.06 Re_H``   (parallel plates, 99 % centreline criterion)

where :math:`Re_H = \\bar u H / \\nu`.  The coefficient 0.06 is the textbook
tabulated value; this module computes it as ``0.06 * re_h`` at run time so the
target number is never hand-copied.

Tolerance gate
--------------
Pre-declared: computed :math:`L_e/H` must be within **8 %** of the White 2006
correlation.  This is deliberately wider than the 2 % used for fully developed
Poiseuille because entrance length is an *integral* quantity sensitive to grid
resolution, inlet discretisation, and the exact 99 % interpolation.

Run one case::

    PYTHONPATH=src python -m tensorfvm.benchmark_entrance_length case \\
        --re-h 100 --nx-per-h 16 --aspect 12 --out runs/entrance-length

Run the full pre-declared plan::

    PYTHONPATH=src python -m tensorfvm.benchmark_entrance_length all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .inc_bench.solver import SolverOptions, SteadyFlow

# --------------------------------------------------------------------------- #
# Pre-declared constants
# --------------------------------------------------------------------------- #
TOLERANCE = 0.08               # 8 % gate on L_e/H (pre-declared before any run)
SOLVER_TOL = 1.0e-8
MAX_ITER = 60000
DEFAULT_BAR_VELOCITY = 1.0
DEFAULT_HEIGHT = 1.0
DEFAULT_DENSITY = 1.0
U_MAX_FACTOR = 1.5             # u_max / u_bar for plane Poiseuille
U_CENTER_THRESHOLD = 0.99      # 99 % criterion for entrance length

#: White 2006 coefficient for parallel plates (Eq. 3-39 form)
WHITE_COEFF = 0.06


def white_entrance_length(re_h: float) -> float:
    """White 2006 correlation: L_e/H = 0.06 Re_H (parallel plates, 99 % criterion).

    Computed at run time — never hand-transcribed.
    """
    return WHITE_COEFF * re_h


# --------------------------------------------------------------------------- #
# PLAN — fixed before any number is produced
# --------------------------------------------------------------------------- #
PLAN: tuple[tuple[str, int, float, float], ...] = (
    # (name, nx_per_H, Re_H, L/H)
    ("Re50-nx8",   8,  50.0,  8.0),
    ("Re50-nx16", 16,  50.0,  8.0),
    ("Re100-nx8",  8, 100.0, 16.0),
    ("Re100-nx16", 16, 100.0, 16.0),   # main case
    ("Re100-nx32", 32, 100.0, 16.0),
    ("Re200-nx16", 16, 200.0, 24.0),
)


# --------------------------------------------------------------------------- #
# Single case
# --------------------------------------------------------------------------- #
@dataclass
class EntranceLengthResult:
    """Holds all outputs of one entrance-length run."""
    name: str
    re_h: float
    nx_per_h: int
    aspect: float            # L/H
    ny: int
    nx: int
    dx: float
    dy: float
    iterations: int
    converged: bool
    elapsed_s: float
    # centreline velocity profile (x/H, u_center/u_bar)
    centerline_x: list[float] = field(default_factory=list)
    centerline_u: list[float] = field(default_factory=list)
    # entrance length
    le_over_h: float = 0.0
    le_over_h_ref: float = 0.0
    le_error: float = 0.0
    passed: bool = False
    # additional metrics
    u_max_computed: float = 0.0
    u_max_ref: float = U_MAX_FACTOR
    u_max_error: float = 0.0
    dpdx_computed: float = 0.0
    dpdx_ref: float = 0.0
    dpdx_error: float = 0.0
    # history
    history: list[dict] = field(default_factory=list)


def run_case(name: str, nx_per_h: int, re_h: float, aspect: float,
             *, ubar: float = DEFAULT_BAR_VELOCITY, height: float = DEFAULT_HEIGHT,
             density: float = DEFAULT_DENSITY) -> EntranceLengthResult:
    """Solve one entrance-length case and extract L_e."""
    nu = ubar * height / re_h
    dy = height / nx_per_h
    ny = nx_per_h
    nx = int(round(aspect * nx_per_h))
    dx = aspect * height / nx

    fluid = torch.ones((ny, nx), dtype=torch.bool)
    # Uniform inlet (plug flow) — the key difference from Poiseuille benchmark
    inlet = torch.full((ny,), ubar, dtype=torch.float64)

    opts = SolverOptions(
        max_iterations=MAX_ITER, tolerance=SOLVER_TOL,
        alpha_u=0.6, alpha_p=0.25,
    )
    solver = SteadyFlow(
        fluid, dx=dx, dy=dy, viscosity=nu, density=density,
        inlet_profile=inlet, outlet=True, options=opts,
    )

    t0 = time.perf_counter()
    record = solver.solve()
    elapsed = time.perf_counter() - t0

    # --- centreline velocity profile ---
    cell_u = 0.5 * (solver.u[:, :-1] + solver.u[:, 1:])
    mid_row = ny // 2  # centreline row (for even ny, this is the upper of the two centre cells)
    centerline_u_vals = cell_u[mid_row, :].detach().cpu().tolist()
    cell_x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    centerline_x_vals = (cell_x.detach().cpu() / height).tolist()

    # Normalise by u_bar
    centerline_u_norm = [u / ubar for u in centerline_u_vals]

    # --- find entrance length: first x where u_center >= 0.99 * 1.5 * u_bar ---
    u_threshold = U_CENTER_THRESHOLD * U_MAX_FACTOR  # 0.99 * 1.5 = 1.485
    le_over_h = None
    for i, (xh, uc) in enumerate(zip(centerline_x_vals, centerline_u_norm)):
        if uc >= u_threshold:
            # Linear interpolation between previous and current point
            if i > 0:
                x_prev, u_prev = centerline_x_vals[i - 1], centerline_u_norm[i - 1]
                frac = (u_threshold - u_prev) / (uc - u_prev) if uc != u_prev else 0.0
                le_over_h = x_prev + frac * (xh - x_prev)
            else:
                le_over_h = xh
            break

    if le_over_h is None:
        # Flow didn't reach 99 % within the domain — use last point as lower bound
        le_over_h = centerline_x_vals[-1]

    # --- reference ---
    le_ref = white_entrance_length(re_h)
    le_error = abs(le_over_h - le_ref) / le_ref if le_ref > 0 else float('inf')
    passed = le_error < TOLERANCE

    # --- max velocity at outlet (should approach 1.5) ---
    outlet_u = cell_u[:, -1].detach().cpu()
    u_max_computed = float(outlet_u.max()) / ubar
    u_max_error = abs(u_max_computed - U_MAX_FACTOR) / U_MAX_FACTOR

    # --- pressure gradient in developed region (last 25 % of domain) ---
    band_start = int(0.75 * nx)
    band_x = cell_x[band_start:].detach().cpu()
    band_p = solver.p[:, band_start:].mean(dim=0).detach().cpu()
    centred_x = band_x - band_x.mean()
    slope = float((centred_x * (band_p - band_p.mean())).sum() / centred_x.square().sum())
    dpdx_ref = -12.0 * nu * ubar / height ** 2
    dpdx_error = abs(slope - dpdx_ref) / abs(dpdx_ref) if dpdx_ref != 0 else float('inf')

    return EntranceLengthResult(
        name=name, re_h=re_h, nx_per_h=nx_per_h, aspect=aspect,
        ny=ny, nx=nx, dx=dx, dy=dy,
        iterations=record.get("iterations", 0),
        converged=record.get("converged", False),
        elapsed_s=elapsed,
        centerline_x=centerline_x_vals,
        centerline_u=centerline_u_norm,
        le_over_h=le_over_h,
        le_over_h_ref=le_ref,
        le_error=le_error,
        passed=passed,
        u_max_computed=u_max_computed,
        u_max_ref=U_MAX_FACTOR,
        u_max_error=u_max_error,
        dpdx_computed=slope,
        dpdx_ref=dpdx_ref,
        dpdx_error=dpdx_error,
        history=record.get("history", [])[-10:],
    )


# --------------------------------------------------------------------------- #
# Aggregation & artifacts
# --------------------------------------------------------------------------- #
def summarise(results: list[EntranceLengthResult]) -> dict:
    """Build the machine-checkable verdict."""
    rows = []
    for r in results:
        rows.append({
            "name": r.name,
            "re_h": r.re_h,
            "nx_per_H": r.nx_per_h,
            "L/H": r.aspect,
            "ny": r.ny,
            "nx": r.nx,
            "L_e/H (computed)": round(r.le_over_h, 4),
            "L_e/H (White 2006)": round(r.le_over_h_ref, 4),
            "L_e/H error %": round(r.le_error * 100, 2),
            "passed_8pct": r.passed,
            "u_max/u_bar": round(r.u_max_computed, 4),
            "u_max error %": round(r.u_max_error * 100, 2),
            "dp/dx": round(r.dpdx_computed, 6),
            "dp/dx ref": round(r.dpdx_ref, 6),
            "dp/dx error %": round(r.dpdx_error * 100, 2),
            "iterations": r.iterations,
            "converged": r.converged,
            "elapsed_s": round(r.elapsed_s, 1),
        })
    all_passed = all(r.passed for r in results)
    return {
        "schema": "tensorfvm.entrance-length/1",
        "reference": "White 2006, Viscous Fluid Flow 3rd ed., §3-3.1, Eq. (3-39)",
        "reference_formula": "L_e/H = 0.06 Re_H (parallel plates, 99% centreline)",
        "tolerance": TOLERANCE,
        "verdict": "PASS" if all_passed else "FAIL",
        "cases": rows,
    }


def write_artifacts(results: list[EntranceLengthResult], out_dir: str | Path):
    """Write JSON, CSV, and PNG artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = summarise(results)
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2))

    # CSV
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary["cases"][0].keys())
        w.writeheader()
        w.writerows(summary["cases"])

    # Centreline profiles CSV
    with open(out / "centerline_profiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "x_over_H", "u_center_over_u_bar"])
        for r in results:
            for xh, uc in zip(r.centerline_x, r.centerline_u):
                w.writerow([r.name, round(xh, 6), round(uc, 6)])

    # Figures
    _make_figures(results, out)

    # Manifest
    import hashlib, platform, subprocess, os
    manifest = {
        "repro_cmd": f"PYTHONPATH=src python -m tensorfvm.benchmark_entrance_length all --out {out}",
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cwd": os.getcwd(),
        "artifact_hashes": {},
    }
    for p in out.iterdir():
        if p.is_file():
            manifest["artifact_hashes"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return summary


def _make_figures(results: list[EntranceLengthResult], out: Path):
    """Generate centreline velocity and entrance-length comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Figure 1: Centreline velocity profiles
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        if r.re_h == 100.0:
            ax.plot(r.centerline_x, r.centerline_u, "o-", label=r.name, markersize=3)
    ax.axhline(U_MAX_FACTOR, color="k", ls="--", lw=1, label="Fully developed (1.5)")
    ax.axhline(U_CENTER_THRESHOLD * U_MAX_FACTOR, color="r", ls=":", lw=1,
               label="99% threshold (1.485)")
    # Mark entrance lengths
    for r in results:
        if r.re_h == 100.0:
            ax.axvline(r.le_over_h, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("x / H")
    ax.set_ylabel("u_center / ū")
    ax.set_title("Centreline velocity development (Re_H = 100)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_centerline_profiles.png", dpi=150)
    plt.close(fig)

    # Figure 2: L_e/H vs Re_H
    fig, ax = plt.subplots(figsize=(6, 4))
    re_vals = sorted(set(r.re_h for r in results))
    for nx_ph in sorted(set(r.nx_per_h for r in results)):
        subset = [r for r in results if r.nx_per_h == nx_ph]
        xs = [r.re_h for r in subset]
        ys = [r.le_over_h for r in subset]
        ax.plot(xs, ys, "o-", label=f"nx/H={nx_ph}", markersize=6)
    # White 2006 line
    xs_ref = [min(re_vals), max(re_vals)]
    ys_ref = [white_entrance_length(x) for x in xs_ref]
    ax.plot(xs_ref, ys_ref, "k--", lw=1.5, label="White 2006: 0.06 Re_H")
    ax.set_xlabel("Re_H")
    ax.set_ylabel("L_e / H")
    ax.set_title("Entrance length vs Reynolds number")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_entrance_length_vs_re.png", dpi=150)
    plt.close(fig)

    # Figure 3: Error bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r.name for r in results]
    errors = [r.le_error * 100 for r in results]
    colors = ["green" if r.passed else "red" for r in results]
    bars = ax.bar(names, errors, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(TOLERANCE * 100, color="red", ls="--", lw=1.5, label="8% gate")
    ax.set_ylabel("L_e/H error (%)")
    ax.set_title("Entrance length error vs White 2006")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_error_bars.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cli():
    ap = argparse.ArgumentParser(description="Entrance-length benchmark (CASE-FVM-CHANNEL-DEVELOP)")
    sub = ap.add_subparsers(dest="command")

    # single case
    p_case = sub.add_parser("case", help="Run one case")
    p_case.add_argument("--re-h", type=float, default=100.0)
    p_case.add_argument("--nx-per-h", type=int, default=16)
    p_case.add_argument("--aspect", type=float, default=16.0)
    p_case.add_argument("--name", type=str, default="custom")
    p_case.add_argument("--out", type=str, default="runs/entrance-length")

    # plan index
    p_idx = sub.add_parser("plan-index", help="Run one plan entry by index")
    p_idx.add_argument("--index", type=int, required=True)
    p_idx.add_argument("--out", type=str, default="runs/entrance-length")

    # all
    p_all = sub.add_parser("all", help="Run full plan and write artifacts")
    p_all.add_argument("--out", type=str, default="runs/entrance-length")

    args = ap.parse_args()

    if args.command == "case":
        r = run_case(args.name, args.nx_per_h, args.re_h, args.aspect)
        print(json.dumps({
            "name": r.name, "Le/H": r.le_over_h, "Le/H_ref": r.le_over_h_ref,
            "error_pct": round(r.le_error * 100, 2), "passed": r.passed,
            "u_max/u_bar": r.u_max_computed, "iterations": r.iterations,
            "converged": r.converged, "elapsed_s": round(r.elapsed_s, 1),
        }, indent=2))
        write_artifacts([r], args.out)

    elif args.command == "plan-index":
        entry = PLAN[args.index]
        r = run_case(entry[0], entry[1], entry[2], entry[3])
        print(json.dumps({
            "name": r.name, "Le/H": r.le_over_h, "Le/H_ref": r.le_over_h_ref,
            "error_pct": round(r.le_error * 100, 2), "passed": r.passed,
        }, indent=2))
        write_artifacts([r], args.out)

    elif args.command == "all":
        results = []
        for entry in PLAN:
            print(f"  Running {entry[0]} ...", flush=True)
            r = run_case(entry[0], entry[1], entry[2], entry[3])
            results.append(r)
            status = "PASS" if r.passed else "FAIL"
            print(f"    Le/H={r.le_over_h:.3f}  ref={r.le_over_h_ref:.3f}  "
                  f"err={r.le_error*100:.1f}%  [{status}]")
        summary = write_artifacts(results, args.out)
        print(f"\nVerdict: {summary['verdict']}")
        print(f"Artifacts written to {args.out}/")
    else:
        ap.print_help()


if __name__ == "__main__":
    cli()
