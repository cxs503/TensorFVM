"""Shu-Osher shock-entropy wave interaction benchmark.

Reference
    Shu, C.-W. & Osher, S. (1988). "Efficient implementation of essentially
    non-oscillatory shock-capturing schemes, II." *J. Comput. Phys.* **77**,
    439--471.  Problem: a Mach-3 shock at x=4 in [0,5] interacts with a
    sinusoidal density perturbation ``rho = 1 + 0.2 sin(pi x)``.

What this benchmark does
    1. Runs the MUSCL-Hancock + HLLC solver at three resolutions (N=200, 400, 800)
       to t=1.8.
    2. Computes a reference solution at N=2000 (the standard reference resolution
       in the literature).
    3. Compares density profiles against the reference, reporting L1/L2/Linf errors.
    4. Checks conservation (mass/energy drift with boundary outflow ledger).
    5. Generates figures: density field comparison and convergence plot.

Acceptance criteria
    - L1 density error at N=800 vs N=2000 reference < 5% (relative to max density range).
    - Mass conservation drift < 0.5%.
    - Energy conservation drift < 1.0%.
    - Observed order of convergence >= 0.5 (first-order dominated by the shock).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from tensorfvm.compressible.cases import (
    ShuOsherConfig,
    shu_osher,
    GAMMA_AIR,
)
from tensorfvm.compressible.solver import EulerSolver

__all__ = ["run_benchmark", "ShuOsherBenchmarkConfig"]


@dataclass
class ShuOsherBenchmarkConfig:
    """Resolution sweep and output settings."""

    coarse: int = 200
    medium: int = 400
    fine: int = 800
    reference: int = 2000
    t_end: float = 1.8
    cfl: float = 0.4
    order: int = 2
    limiter: str = "mc"
    output_dir: str = "results/shu-osher"


def _rel_error(numeric: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    """L1, L2, Linf errors, absolute and relative to the reference range."""
    diff = numeric - reference
    l1 = float(torch.mean(torch.abs(diff)))
    l2 = float(torch.sqrt(torch.mean(diff * diff)))
    linf = float(torch.max(torch.abs(diff)))
    ref_range = float(torch.max(reference) - torch.min(reference))
    scale = ref_range if ref_range > 1e-12 else 1.0
    return {
        "L1_abs": l1,
        "L2_abs": l2,
        "Linf_abs": linf,
        "L1_rel": l1 / scale,
        "L2_rel": l2 / scale,
        "Linf_rel": linf / scale,
        "ref_min": float(torch.min(reference)),
        "ref_max": float(torch.max(reference)),
        "ref_range": scale,
    }


def _observed_order(err_coarse: float, err_fine: float, ratio: float = 2.0) -> float:
    if err_coarse <= 0 or err_fine <= 0:
        return float("nan")
    return math.log(err_coarse / err_fine) / math.log(ratio)


def run_single(cfg: ShuOsherConfig) -> Tuple[EulerSolver, Dict]:
    """Run one resolution and return (solver, meta)."""
    solver, meta = shu_osher(cfg)
    initial = solver.integrals()
    t0 = time.time()
    solver.run(cfg.t_end)
    elapsed = time.time() - t0
    prim = solver.primitive()
    drift = solver.budget_drift(initial)
    meta["elapsed"] = round(elapsed, 2)
    meta["nsteps"] = solver.nsteps
    meta["positivity_fixes"] = solver.positivity_fixes
    meta["species_clamps"] = solver.species_clamps
    meta["drift_mass"] = drift["mass"]
    meta["drift_energy"] = drift["energy"]
    meta["drift_momentum_x"] = drift["momentum_x"]
    return solver, meta


def run_benchmark(cfg: ShuOsherBenchmarkConfig | None = None) -> Dict:
    """Run the full Shu-Osher benchmark and return results dict."""
    cfg = cfg or ShuOsherBenchmarkConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- reference solution (N=2000) ---
    print(f"[shu-osher] Computing reference solution (N={cfg.reference})...")
    ref_cfg = ShuOsherConfig(
        nx=cfg.reference, t_end=cfg.t_end, cfl=cfg.cfl,
        order=cfg.order, limiter=cfg.limiter,
    )
    ref_solver, ref_meta = run_single(ref_cfg)
    ref_prim = ref_solver.primitive()
    ref_rho = ref_prim["rho"][:, 0]
    ref_u = ref_prim["u"][:, 0]
    ref_p = ref_prim["p"][:, 0]
    ref_xc = ref_meta["xc"]
    print(f"  -> {ref_solver.nsteps} steps, {ref_meta['elapsed']}s")

    # --- sweep: coarse, medium, fine ---
    results: List[Dict] = []
    for nx in (cfg.coarse, cfg.medium, cfg.fine):
        print(f"[shu-osher] Running N={nx}...")
        run_cfg = ShuOsherConfig(
            nx=nx, t_end=cfg.t_end, cfl=cfg.cfl,
            order=cfg.order, limiter=cfg.limiter,
        )
        solver, meta = run_single(run_cfg)
        prim = solver.primitive()
        rho = prim["rho"][:, 0]
        u = prim["u"][:, 0]
        p = prim["p"][:, 0]
        xc = meta["xc"]

        # Interpolate reference to this grid for error computation
        # Use simple nearest-neighbour for now (conservative for shock problems)
        ref_rho_interp = torch.zeros_like(rho)
        ref_u_interp = torch.zeros_like(u)
        ref_p_interp = torch.zeros_like(p)
        for i in range(nx):
            xi = xc[i].item()
            # Find nearest reference cell
            j = int((xi / cfg.reference * cfg.reference * 5.0 / 5.0))
            # More precise: find index in ref_xc closest to xc[i]
            j = int(torch.argmin(torch.abs(ref_xc - xi)).item())
            ref_rho_interp[i] = ref_rho[j]
            ref_u_interp[i] = ref_u[j]
            ref_p_interp[i] = ref_p[j]

        rho_err = _rel_error(rho, ref_rho_interp)
        u_err = _rel_error(u, ref_u_interp)
        p_err = _rel_error(p, ref_p_interp)

        row = {
            "nx": nx,
            "dx": 5.0 / nx,
            "nsteps": meta["nsteps"],
            "elapsed": meta["elapsed"],
            "positivity_fixes": meta["positivity_fixes"],
            "species_clamps": meta["species_clamps"],
            "drift_mass": meta["drift_mass"],
            "drift_energy": meta["drift_energy"],
            "drift_momentum_x": meta["drift_momentum_x"],
            "rho_error": rho_err,
            "u_error": u_err,
            "p_error": p_err,
            # Store profiles for figure generation
            "x": xc.tolist(),
            "rho": rho.tolist(),
            "u": u.tolist(),
            "p": p.tolist(),
        }
        results.append(row)
        print(f"  -> L1 rho error: {rho_err['L1_rel']:.4e} ({rho_err['L1_abs']:.4e} abs)")

    # --- convergence order ---
    orders = {}
    for var in ("rho", "u", "p"):
        key = f"{var}_error"
        errs = [r[key]["L1_abs"] for r in results]
        o = []
        for a, b in zip(errs[:-1], errs[1:]):
            o.append(_observed_order(a, b, 2.0))
        orders[f"L1_{var}"] = o

    # --- verdicts ---
    fine = results[-1]
    verdicts = {
        "rho_L1_rel_lt_5pct": fine["rho_error"]["L1_rel"] < 0.05,
        "mass_drift_lt_0_5pct": abs(fine["drift_mass"]) < 0.005,
        "energy_drift_lt_1pct": abs(fine["drift_energy"]) < 0.01,
        "observed_order_ge_0_5": (
            len(orders.get("L1_rho", [])) > 0
            and min(orders["L1_rho"]) >= 0.5
        ),
    }

    # --- reference profile for figures ---
    ref_profile = {
        "x": ref_xc.tolist(),
        "rho": ref_rho.tolist(),
        "u": ref_u.tolist(),
        "p": ref_p.tolist(),
    }

    result = {
        "case": "shu_osher",
        "reference": "Shu & Osher (1988) JCP 77, 439",
        "config": asdict(cfg),
        "post_shock": ref_meta["post_shock"],
        "results": results,
        "convergence_orders": orders,
        "verdicts": verdicts,
        "reference_profile": ref_profile,
    }

    # Save JSON
    json_path = out_dir / "shu_osher.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[shu-osher] Results saved to {json_path}")

    # Save verdict summary
    verdict_path = out_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdicts, f, indent=2)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shu-Osher benchmark")
    parser.add_argument("--output", default="results/shu-osher", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode: N=100,200,400, ref=800")
    args = parser.parse_args()

    if args.quick:
        cfg = ShuOsherBenchmarkConfig(
            coarse=100, medium=200, fine=400, reference=800,
            output_dir=args.output,
        )
    else:
        cfg = ShuOsherBenchmarkConfig(output_dir=args.output)

    result = run_benchmark(cfg)
    print("\n=== Verdicts ===")
    for k, v in result["verdicts"].items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")
