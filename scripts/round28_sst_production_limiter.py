#!/usr/bin/env python
"""Round 28: SST k-omega with production limiter for CASE-FVM-PERIODIC-HILLS.

Mission m22825 (selfstart, TensorFVM) · Anchor: CASE-FVM-PERIODIC-HILLS

Round 11 (m20762) identified that the SST closure at Re=2800 on 96x48 produces
``nu_t/nu = 342`` (13x the expected ~26 for a turbulent channel) because the
production limiter is **off by default**.  The SST-2003 model includes a
production limiter ``P_k <= 10 beta* k omega`` (Menter et al. 2003) that
prevents the eddy viscosity from growing without bound in regions where the
strain rate is over-predicted.

This script runs the SST closure with the production limiter enabled and
compares the results to the unlimited case (round 11) and the DNS reference.

Protocol:
  - Mesh: 96x48 (same as round 11)
  - Re = 2800, dt = 0.02
  - SST-2003 with production_limit = 10.0
  - Time averaging: t = [160, 400], sample every 20 steps
  - Metrics: x_r/H, u/U_b at y/H=0.5, max nu_t/nu, stationarity

Reference: Breuer et al. 2009, Comput. Fluids 38:433, doi:10.1016/j.compfluid.2008.05.002
  x_r/H = 5.41 (ERCOFTAC KB, UFR 3-30 Evaluation)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_sst import SSTKOmega, SSTKOmegaConfig
from tensorfvm.verification.periodic_hill_reference import (
    X_R_OVER_H,
    X_R_TOLERANCE,
    y05_table,
    profile_at,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Round28Config:
    nx: int = 96
    ny: int = 48
    reynolds: float = 2800.0
    dt: float = 0.02
    total_steps: int = 20000
    burn_in_steps: int = 8000  # t = 160
    sample_every: int = 20
    production_limit: float = 10.0  # SST-2003 standard limiter
    freestream_intensity: float = 0.05
    convection_blend: float = 0.0  # first-order upwind for k/omega


# --------------------------------------------------------------------------- #
# Statistics accumulator
# --------------------------------------------------------------------------- #

class TimeAveragedStats:
    """Accumulate time-averaged statistics for the periodic hill case."""

    def __init__(self, solver):
        self.solver = solver
        self.n_samples = 0
        self.sum_wall_shear_bottom = None
        self.sum_u_profile = None
        self.sum_nu_t_max = 0.0
        self.sum_body_force = 0.0
        self.reattachment_samples = []
        self.separation_samples = []

    def sample(self):
        """Take one sample from the current solver state."""
        solver = self.solver

        # Wall shear on bottom wall (j=0)
        tau_w = solver.wall_shear()
        if self.sum_wall_shear_bottom is None:
            self.sum_wall_shear_bottom = tau_w[0].copy()
        else:
            self.sum_wall_shear_bottom += tau_w[0]

        # u/U_b profile at y/H = 0.5
        h = solver.config.height_m
        u_b = solver.bulk_velocity()
        yc = solver.mesh.yc[:, 0]
        j = int(np.clip(np.searchsorted(yc, 0.5 * h) - 1, 0, solver.ny - 1))
        u_profile = solver.u[j] / u_b
        if self.sum_u_profile is None:
            self.sum_u_profile = u_profile.copy()
        else:
            self.sum_u_profile += u_profile

        # Max nu_t/nu
        if hasattr(solver, 'turbulence') and solver.turbulence is not None:
            nu_t = solver.turbulence.viscosity()
            self.sum_nu_t_max += float(np.max(nu_t) / solver.nu)

        # Body force
        self.sum_body_force += solver.body_force

        # Reattachment/separation
        sep, rea = solver.reattachment()
        self.reattachment_samples.append(rea if not math.isnan(rea) else None)
        self.separation_samples.append(sep if not math.isnan(sep) else None)

        self.n_samples += 1

    def mean_reattachment(self):
        """Mean reattachment length from samples where it was defined."""
        valid = [x for x in self.reattachment_samples if x is not None]
        if not valid:
            return float('nan')
        return float(np.mean(valid))

    def mean_separation(self):
        """Mean separation length from samples where it was defined."""
        valid = [x for x in self.separation_samples if x is not None]
        if not valid:
            return float('nan')
        return float(np.mean(valid))

    def reattachment_coverage(self):
        """Fraction of samples where reattachment was found."""
        valid = sum(1 for x in self.reattachment_samples if x is not None)
        return valid / max(len(self.reattachment_samples), 1)

    def mean_wall_shear_bottom(self):
        if self.sum_wall_shear_bottom is None or self.n_samples == 0:
            return None
        return self.sum_wall_shear_bottom / self.n_samples

    def mean_u_profile(self):
        if self.sum_u_profile is None or self.n_samples == 0:
            return None
        return self.sum_u_profile / self.n_samples

    def mean_nu_t_max(self):
        if self.n_samples == 0:
            return float('nan')
        return self.sum_nu_t_max / self.n_samples

    def mean_body_force(self):
        if self.n_samples == 0:
            return float('nan')
        return self.sum_body_force / self.n_samples


# --------------------------------------------------------------------------- #
# Profile comparison
# --------------------------------------------------------------------------- #

def compare_profile_at_y05(solver, stations=None):
    """Compare u/U_b at y/H=0.5 against DNS reference."""
    if stations is None:
        stations = list(y05_table().keys())

    h = solver.config.height_m
    u_b = solver.bulk_velocity()
    yc = solver.mesh.yc[:, 0]
    j = int(np.clip(np.searchsorted(yc, 0.5 * h) - 1, 0, solver.ny - 1))

    results = {}
    for xh in stations:
        x_target = xh * h
        xc = solver.mesh.xc[0]
        i = int(np.clip(np.searchsorted(xc, x_target) - 1, 0, solver.nx - 1))
        i_next = (i + 1) % solver.nx
        dx = xc[i_next] - xc[i] if xc[i_next] > xc[i] else solver.config.length_m / solver.nx
        tx = max(0.0, min(1.0, (x_target - xc[i]) / dx))
        u_interp = (1 - tx) * solver.u[j, i] + tx * solver.u[j, i_next]
        computed = u_interp / u_b

        try:
            ref = profile_at(xh, 0.5)
            if abs(ref) >= 0.1:
                error = abs(computed - ref) / abs(ref)
            else:
                error = abs(computed - ref)
            results[xh] = {
                "computed": float(computed),
                "reference": float(ref),
                "error": float(error),
                "pass": bool(error <= X_R_TOLERANCE),
            }
        except (ValueError, KeyError):
            results[xh] = {
                "computed": float(computed),
                "reference": None,
                "error": None,
                "pass": False,
            }

    return results


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #

def run_round28(config=None, out_dir=None, verbose=True):
    """Run the SST production limiter case and return results."""
    cfg = config or Round28Config()
    out_path = Path(out_dir) if out_dir else Path("docs/case-periodic-hills")

    if verbose:
        print(f"Round 28: SST with production_limit={cfg.production_limit}")
        print(f"  Mesh: {cfg.nx}x{cfg.ny}, Re={cfg.reynolds}, dt={cfg.dt}")
        print(f"  Total steps: {cfg.total_steps}, burn-in: {cfg.burn_in_steps}")

    # Create solver with SST closure
    solver_config = PeriodicHillConfig(
        nx=cfg.nx, ny=cfg.ny, reynolds=cfg.reynolds, time_step_s=cfg.dt
    )
    solver = PeriodicHillSolver(solver_config)

    # Attach SST closure with production limiter
    sst_config = SSTKOmegaConfig(
        production_limit=cfg.production_limit,
        freestream_intensity=cfg.freestream_intensity,
        convection_blend=cfg.convection_blend,
    )
    solver.turbulence = SSTKOmega(solver, sst_config)

    if verbose:
        print(f"  Initial nu_t/nu: {np.max(solver.turbulence.viscosity()) / solver.nu:.2f}")

    # Statistics accumulators
    burn_in_stats = TimeAveragedStats(solver)
    avg_stats = TimeAveragedStats(solver)

    t0 = time.time()
    step = 0

    while step < cfg.total_steps:
        solver.advance(steps=1)
        step += 1

        # Sample during burn-in
        if step <= cfg.burn_in_steps and step % cfg.sample_every == 0:
            burn_in_stats.sample()

        # Sample during averaging window
        if step > cfg.burn_in_steps and step % cfg.sample_every == 0:
            avg_stats.sample()

        if verbose and step % 2000 == 0:
            elapsed = time.time() - t0
            sep, rea = solver.reattachment()
            nu_t_max = np.max(solver.turbulence.viscosity()) / solver.nu
            print(f"  Step {step}/{cfg.total_steps} ({elapsed:.0f}s): "
                  f"G={solver.body_force:.6f}, nu_t/nu={nu_t_max:.1f}, "
                  f"x_s/H={sep:.3f}, x_r/H={rea:.3f}")

    elapsed = time.time() - t0

    # Compute time-averaged metrics
    mean_xr = avg_stats.mean_reattachment()
    mean_xs = avg_stats.mean_separation()
    coverage = avg_stats.reattachment_coverage()
    mean_nu_t = avg_stats.mean_nu_t_max()
    mean_G = avg_stats.mean_body_force()

    # Profile comparison
    profile_results = compare_profile_at_y05(solver)
    n_pass = sum(1 for v in profile_results.values() if v.get("pass", False))
    n_total = len(profile_results)

    # Error on x_r/H
    xr_error = abs(mean_xr - X_R_OVER_H) / X_R_OVER_H if not math.isnan(mean_xr) else float('inf')

    # Verdict
    verdict = "PASS" if (
        not math.isnan(mean_xr)
        and xr_error <= X_R_TOLERANCE
        and n_pass == n_total
        and coverage > 0.9
    ) else "FAIL"

    results = {
        "case": "CASE-FVM-PERIODIC-HILLS round 28 (SST with production limiter)",
        "config": asdict(cfg),
        "reference": {
            "x_r_over_H": X_R_OVER_H,
            "source": "ERCOFTAC KB, UFR 3-30 Evaluation",
        },
        "runtime_seconds": round(elapsed, 1),
        "samples": {
            "burn_in": burn_in_stats.n_samples,
            "averaging": avg_stats.n_samples,
        },
        "time_averaged": {
            "x_r_over_H": mean_xr if not math.isnan(mean_xr) else None,
            "x_s_over_H": mean_xs if not math.isnan(mean_xs) else None,
            "reattachment_coverage": coverage,
            "max_nu_t_ratio": mean_nu_t,
            "body_force": mean_G,
        },
        "profile_comparison": {
            "n_stations": n_total,
            "n_pass": n_pass,
            "stations": profile_results,
        },
        "error_x_r_pct": round(xr_error * 100, 1) if not math.isinf(xr_error) else None,
        "verdict": verdict,
        "verdict_reason": _verdict_reason(verdict, mean_xr, xr_error, n_pass, n_total, coverage, mean_nu_t),
    }

    # Save results
    out_json = out_path / "round28_sst_prodlim.json"
    out_json.write_text(json.dumps(results, indent=2, allow_nan=False))

    if verbose:
        print(f"\nResults saved to {out_json}")
        print(f"Verdict: {verdict}")
        print(f"  x_r/H = {mean_xr:.3f} (ref: {X_R_OVER_H}, error: {xr_error*100:.1f}%)")
        print(f"  nu_t/nu = {mean_nu_t:.1f}")
        print(f"  Profile: {n_pass}/{n_total} stations pass")
        print(f"  Coverage: {coverage*100:.1f}%")

    return results


def _verdict_reason(verdict, xr, xr_err, n_pass, n_total, coverage, nu_t):
    reasons = []
    if verdict == "PASS":
        return "All acceptance criteria met"
    if math.isnan(xr):
        reasons.append("x_r/H undefined")
    elif xr_err > X_R_TOLERANCE:
        reasons.append(f"x_r/H error {xr_err*100:.1f}% > {X_R_TOLERANCE*100:.0f}%")
    if n_pass < n_total:
        reasons.append(f"{n_pass}/{n_total} profile stations pass")
    if coverage < 0.9:
        reasons.append(f"reattachment coverage {coverage*100:.0f}% < 90%")
    if nu_t > 50:
        reasons.append(f"nu_t/nu = {nu_t:.0f} (expected ~26)")
    return "; ".join(reasons) if reasons else "Unknown"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Round 28: SST with production limiter")
    parser.add_argument("--output", default="docs/case-periodic-hills", help="Output directory")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--production-limit", type=float, default=10.0)
    args = parser.parse_args()

    cfg = Round28Config(
        nx=args.nx, ny=args.ny, total_steps=args.steps,
        production_limit=args.production_limit,
    )
    run_round28(cfg, out_dir=args.output)
