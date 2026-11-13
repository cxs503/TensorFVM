"""Shu-Osher shock--entropy-wave interaction benchmark.

Reference
    Shu, C.-W. & Osher, S. (1988). "Efficient implementation of essentially
    non-oscillatory shock-capturing schemes, II." *J. Comput. Phys.* **77**,
    439--471 (test 3); the same configuration is reused by Jiang & Shu (1996),
    *J. Comput. Phys.* **126**, 202.

Setup (see :class:`tensorfvm.compressible.cases.ShuOsherConfig`)
    Domain ``x in [-5, 5]``, jump at ``x = -4``.  Left of the jump the
    Mach-3 post-shock state ``(rho, u, p) = (3.857143, 2.629369, 10.333333)``
    from the Rankine-Hugoniot relations; right of it ``rho = 1 + 0.2 sin(5x)``,
    ``u = 0``, ``p = 1``.  Output time ``t = 1.8``, at which the shock sits at
    ``x ~ 2.39`` -- i.e. *inside* the domain, with the entropy-wave train
    behind it.

What the benchmark measures
    1. Grid convergence of the MUSCL-Hancock + HLLC solver on
       ``N = 200, 400, 800, 1600`` against a ``N = 3200`` run of the *same*
       scheme.  Finite-volume solutions are compared as **cell averages**
       (the reference is block-averaged onto the coarse cells), which is the
       only comparison the scheme actually predicts.
    2. The discretisation error *of the reference itself*, estimated from the
       1600-vs-3200 pair with the observed order, so an L1 number can never be
       mistaken for "the exact solution".
    3. Two analytic, grid-independent invariants that pin the physics without
       any digitised literature data:
         * the region ahead of the shock must equal ``1 + 0.2 sin(5x)`` to
           within the (zero) boundary treatment error, and
         * the shock front must follow the Rankine-Hugoniot trajectory
           ``x_s(t) = -4 + D t`` with ``D = 3.54965``.
    4. Mass/energy budget accounting (open boundaries) and positivity.

Acceptance criteria
    - L1 density error vs the N=3200 reference < 5% of the reference density
      range (the ``<5%`` tolerance of the case pool entry).
    - Undisturbed-region error < 1% of the perturbation amplitude (0.2).
    - Shock position within 2% of ``x_s(1.8)`` (one cell at N=200 is
      0.05 / 6.39 = 0.8%).
    - Mass/energy budget drift < 0.5% / 1%, no positivity fixes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from tensorfvm.compressible.cases import (
    GAMMA_AIR,
    ShuOsherConfig,
    shu_osher,
)

__all__ = [
    "ShuOsherBenchmarkConfig",
    "run_benchmark",
    "block_average",
    "block_average_centers",
    "compare_to_reference",
    "analytic_invariants",
    "shock_front_position",
]


@dataclass
class ShuOsherBenchmarkConfig:
    """Resolution sweep and output settings."""

    coarse: int = 200
    medium: int = 400
    fine: int = 800
    very_fine: int = 1600
    reference: int = 3200
    t_end: float = 1.8
    cfl: float = 0.4
    order: int = 2
    limiter: str = "mc"
    output_dir: str = "results/shu-osher"

    @property
    def sweep(self) -> Tuple[int, ...]:
        return (self.coarse, self.medium, self.fine, self.very_fine)


# ----------------------------------------------------------------------
# finite-volume-consistent comparison helpers
# ----------------------------------------------------------------------
def block_average(values: torch.Tensor, ratio: int) -> torch.Tensor:
    """Average ``values`` over blocks of ``ratio`` cells (cell -> cell)."""
    if ratio < 1 or values.numel() % ratio:
        raise ValueError("values must be divisible into blocks of `ratio`")
    return values.reshape(-1, ratio).mean(dim=1)


def block_average_centers(xc: torch.Tensor, ratio: int) -> torch.Tensor:
    """Cell centres of the block-averaged grid."""
    if xc.numel() % ratio:
        raise ValueError("xc must be divisible into blocks of `ratio`")
    return xc.reshape(-1, ratio).mean(dim=1)


def compare_to_reference(
    coarse: torch.Tensor, reference: torch.Tensor
) -> Dict[str, float]:
    """L1/L2/Linf of ``coarse`` against a block-averaged reference.

    ``coarse`` are coarse-grid cell averages, ``reference`` the fine-grid cell
    averages (the fine grid is an integer refinement of the coarse one).
    Relative errors are normalised by the reference density range, which is
    the convention used for this case in the shock-capturing literature.
    """
    diff = coarse - reference
    l1 = float(torch.mean(torch.abs(diff)))
    l2 = float(torch.sqrt(torch.mean(diff * diff)))
    linf = float(torch.max(torch.abs(diff)))
    ref_min = float(torch.min(reference))
    ref_max = float(torch.max(reference))
    scale = ref_max - ref_min
    scale = scale if scale > 1e-12 else 1.0
    return {
        "L1_abs": l1,
        "L2_abs": l2,
        "Linf_abs": linf,
        "L1_rel": l1 / scale,
        "L2_rel": l2 / scale,
        "Linf_rel": linf / scale,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "ref_range": scale,
    }


def _observed_order(err_coarse: float, err_fine: float, ratio: float = 2.0) -> float:
    if err_coarse <= 0.0 or err_fine <= 0.0:
        return float("nan")
    return math.log(err_coarse / err_fine) / math.log(ratio)


# ----------------------------------------------------------------------
# solver drivers
# ----------------------------------------------------------------------
def run_single(cfg: ShuOsherConfig) -> Tuple[object, Dict]:
    """Run one resolution to ``cfg.t_end``; return ``(solver, meta)``."""
    solver, meta = shu_osher(cfg)
    initial = solver.integrals()
    t0 = time.time()
    solver.run(cfg.t_end)
    meta["elapsed"] = round(time.time() - t0, 2)
    meta["nsteps"] = solver.nsteps
    meta["positivity_fixes"] = solver.positivity_fixes
    meta["species_clamps"] = solver.species_clamps
    drift = solver.budget_drift(initial)
    meta["drift_mass"] = drift["mass"]
    meta["drift_energy"] = drift["energy"]
    meta["drift_momentum_x"] = drift["momentum_x"]
    return solver, meta


def shock_front_position(xc: torch.Tensor, rho: torch.Tensor) -> Dict[str, float]:
    """Locate the shock front from the steepest density gradient.

    The front is the mid-point of the cell pair with the largest drop; the
    position is refined linearly inside that pair so the estimate is not
    quantised to the cell size.
    """
    grad = torch.diff(rho)  # rho[i+1] - rho[i]
    i = int(torch.argmax(torch.abs(grad)).item())
    dx = float(xc[1] - xc[0])
    g = float(grad[i])
    # Parabolic refinement: fit g_{i-1}, g_i, g_{i+1} and take the vertex, which
    # gives a sub-cell front location instead of one quantised to dx/2.
    offset = 0.0
    if 0 < i < grad.numel() - 1:
        gm, g0, gp = float(grad[i - 1]), g, float(grad[i + 1])
        denom = gm - 2.0 * g0 + gp
        if abs(denom) > 1e-14:
            offset = 0.5 * (gm - gp) / denom
            offset = max(-0.5, min(0.5, offset))
    return {
        "index": i,
        "x_front": float(xc[i]) + (0.5 + offset) * dx,
        "jump": g,
        "quantization": 0.5 * dx,
        "rho_left": float(rho[i]),
        "rho_right": float(rho[i + 1]),
    }


def analytic_invariants(
    meta: Dict, prim: Dict, cfg: ShuOsherConfig, t: float
) -> Dict[str, float]:
    """Grid-independent checks: undisturbed upstream profile + RH shock track."""
    xc = meta["xc"]
    rho = prim["rho"][:, 0]
    front = shock_front_position(xc, rho)

    # --- undisturbed region: everything at least 3 cells ahead of the front
    dx = cfg.dx
    ahead = xc > front["x_front"] + 3.0 * dx
    exact = cfg.rho_pre + cfg.entropy_amp * torch.sin(cfg.entropy_wavenumber * xc)
    if int(ahead.sum()) > 0:
        und_max_abs = float((rho[ahead] - exact[ahead]).abs().max())
    else:  # pragma: no cover - only if the front reached the right boundary
        und_max_abs = float("nan")
    und_rel = und_max_abs / cfg.entropy_amp

    # --- Rankine-Hugoniot trajectory
    x_pred = cfg.shock_position(t)
    x_span = cfg.lx
    return {
        "x_front": front["x_front"],
        "x_front_pred": x_pred,
        "front_abs_err": abs(front["x_front"] - x_pred),
        "front_rel_err": abs(front["x_front"] - x_pred) / x_span,
        "undisturbed_max_abs_err": und_max_abs,
        "undisturbed_rel_err": und_rel,
        "undisturbed_cells": int(ahead.sum()),
        "shock_speed": cfg.post_shock["shock_speed"],
    }


# ----------------------------------------------------------------------
# main driver
# ----------------------------------------------------------------------
def _make_figure(
    profiles: List[Dict],
    ref_profile: Dict,
    out_png: Path,
    x_front_pred: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ref_x = ref_profile["x"]
    ref_rho = ref_profile["rho"]
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]

    ax = axes[0]
    ax.plot(ref_x, ref_rho, "k-", lw=1.6, label="reference N=%d" % ref_profile["nx"])
    for prof, c in zip(profiles, colors):
        ax.plot(prof["x"], prof["rho"], c, lw=0.9, alpha=0.9,
                label="N=%d" % prof["nx"])
    ax.axvline(x_front_pred, color="crimson", ls=":", lw=1.2,
               label="RH shock track $x_s(1.8)$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("Shu-Osher t = 1.8: density (full domain)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    lo, hi = -1.0, x_front_pred + 1.5
    ax.plot(ref_x, ref_rho, "k-", lw=1.8, label="reference")
    for prof, c in zip(profiles, colors):
        ax.plot(prof["x"], prof["rho"], c, lw=0.9, alpha=0.9,
                label="N=%d" % prof["nx"])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("x")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("zoom: compressed entropy-wave train behind the shock")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _make_convergence_figure(rows: List[Dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dx = np.array([r["dx"] for r in rows], dtype=float)
    err = np.array([r["rho_error"]["L1_abs"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.loglog(dx, err, "o-", color="#08519c", label="MUSCL-Hancock + HLLC (L1)")
    ref_dx = dx[-1] * np.array([1.0, 0.5, 0.25])
    for slope, style in ((1.0, "--"), (2.0, ":")):
        ax.loglog(ref_dx, err[-1] * (ref_dx / dx[-1]) ** slope, style,
                  color="gray", lw=1.0, label="slope %g" % slope)
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$L_1(\rho)$ vs N=3200 reference")
    ax.set_title("grid convergence of the density field")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_benchmark(cfg: Optional[ShuOsherBenchmarkConfig] = None) -> Dict:
    """Run the resolution sweep, analytic checks and error quantification."""
    cfg = cfg or ShuOsherBenchmarkConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- reference solution ------------------------------------------------
    print(f"[shu-osher] reference run N={cfg.reference} ...", flush=True)
    ref_solver, ref_meta = run_single(
        ShuOsherConfig(nx=cfg.reference, t_end=cfg.t_end, cfl=cfg.cfl,
                       order=cfg.order, limiter=cfg.limiter)
    )
    ref_prim = ref_solver.primitive()
    ref_rho = ref_prim["rho"][:, 0]
    ref_xc = ref_meta["xc"]
    print(f"  -> {ref_solver.nsteps} steps, {ref_meta['elapsed']}s", flush=True)

    # --- sweep ------------------------------------------------------------
    rows: List[Dict] = []
    sweep = sorted({n for n in cfg.sweep if n < cfg.reference})
    for nx in sweep:
        print(f"[shu-osher] N={nx} ...", flush=True)
        run_cfg = ShuOsherConfig(nx=nx, t_end=cfg.t_end, cfl=cfg.cfl,
                                 order=cfg.order, limiter=cfg.limiter)
        solver, meta = run_single(run_cfg)
        prim = solver.primitive()
        rho = prim["rho"][:, 0]
        xc = meta["xc"]
        ratio = cfg.reference // nx
        if ratio * nx != cfg.reference:
            raise ValueError("reference must be an integer refinement")
        ref_on_coarse = block_average(ref_rho, ratio)
        err = compare_to_reference(rho, ref_on_coarse)
        inv = analytic_invariants(meta, prim, run_cfg, cfg.t_end)
        row = {
            "nx": nx,
            "dx": run_cfg.dx,
            "nsteps": meta["nsteps"],
            "elapsed": meta["elapsed"],
            "positivity_fixes": meta["positivity_fixes"],
            "species_clamps": meta["species_clamps"],
            "drift_mass": meta["drift_mass"],
            "drift_energy": meta["drift_energy"],
            "drift_momentum_x": meta["drift_momentum_x"],
            "rho_error": err,
            "invariants": inv,
            "x": xc.tolist(),
            "rho": rho.tolist(),
            "u": prim["u"][:, 0].tolist(),
            "p": prim["p"][:, 0].tolist(),
        }
        rows.append(row)
        print(
            f"  -> steps={meta['nsteps']:5d} L1_rel={err['L1_rel']:.3e} "
            f"Linf_rel={err['Linf_rel']:.3e} front_err={inv['front_abs_err']:.3e} "
            f"und_err={inv['undisturbed_max_abs_err']:.2e}",
            flush=True,
        )

    # --- convergence orders & reference discretisation error ---------------
    l1 = [r["rho_error"]["L1_abs"] for r in rows]
    orders = [_observed_order(a, b, 2.0) for a, b in zip(l1[:-1], l1[1:])]
    order_est = orders[-1] if orders else float("nan")
    # |e_ref| ~= L1(N_finest vs ref) / (2^p - 1) for two nested grids
    ref_err_est = (l1[-1] / (2.0 ** order_est - 1.0)) if order_est == order_est else float("nan")

    fine_row = rows[-1]
    verdicts = {
        "rho_L1_rel_lt_5pct": bool(fine_row["rho_error"]["L1_rel"] < 0.05),
        "mass_drift_lt_0_5pct": bool(abs(fine_row["drift_mass"]) < 0.005),
        "energy_drift_lt_1pct": bool(abs(fine_row["drift_energy"]) < 0.01),
        "no_positivity_fixes": bool(fine_row["positivity_fixes"] == 0),
        "undisturbed_err_lt_1pct": bool(
            fine_row["invariants"]["undisturbed_rel_err"] < 0.01
        ),
        "shock_front_lt_2pct": bool(fine_row["invariants"]["front_rel_err"] < 0.02),
        "errors_decrease_monotone": bool(all(a > b for a, b in zip(l1[:-1], l1[1:]))),
    }
    # resolution at which the 5% tolerance is met (smallest N in the sweep)
    meeting = [r["nx"] for r in rows if r["rho_error"]["L1_rel"] < 0.05]

    result = {
        "case": "shu_osher",
        "reference": "Shu & Osher (1988) JCP 77, 439 (test 3); Jiang & Shu (1996) JCP 126, 202",
        "config": asdict(cfg),
        "setup": {
            "domain": [ShuOsherConfig().x_min, ShuOsherConfig().x_max],
            "shock_pos": ShuOsherConfig().shock_pos,
            "entropy_wavenumber": ShuOsherConfig().entropy_wavenumber,
            "t_end": ShuOsherConfig().t_end,
            "post_shock": ref_meta["post_shock"],
            "x_front_pred": ShuOsherConfig().shock_position(cfg.t_end),
        },
        "reference_run": {
            "nx": cfg.reference,
            "nsteps": ref_meta["nsteps"],
            "elapsed": ref_meta["elapsed"],
            "drift_mass": ref_meta["drift_mass"],
            "drift_energy": ref_meta["drift_energy"],
            "positivity_fixes": ref_meta["positivity_fixes"],
            "rho_min": float(ref_rho.min()),
            "rho_max": float(ref_rho.max()),
            "invariants": analytic_invariants(
                ref_meta, ref_prim, ShuOsherConfig(nx=cfg.reference), cfg.t_end
            ),
        },
        "results": rows,
        "convergence": {
            "L1_rho_abs": l1,
            "observed_orders": orders,
            "order_estimate": order_est,
            "reference_L1_error_estimate": ref_err_est,
        },
        "verdicts": verdicts,
        "resolution_meeting_5pct": meeting,
        "reference_profile": {
            "nx": cfg.reference,
            "x": ref_xc.tolist(),
            "rho": ref_rho.tolist(),
        },
    }

    (out_dir / "shu_osher.json").write_text(json.dumps(result, indent=2))
    (out_dir / "verdict.json").write_text(json.dumps(verdicts, indent=2))
    print(f"[shu-osher] JSON written to {out_dir / 'shu_osher.json'}", flush=True)

    # --- figures ----------------------------------------------------------
    try:
        _make_figure(rows, result["reference_profile"], out_dir / "fig_density_profile.png",
                     result["setup"]["x_front_pred"])
        _make_convergence_figure(rows, out_dir / "fig_convergence.png")
        print("[shu-osher] figures written", flush=True)
    except Exception as exc:  # pragma: no cover - figures are optional artifacts
        print(f"[shu-osher] figure generation skipped: {exc}", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Shu-Osher benchmark")
    parser.add_argument("--output", default="results/shu-osher")
    parser.add_argument("--quick", action="store_true",
                        help="N=100,200 sweep against an N=400 reference")
    args = parser.parse_args()

    if args.quick:
        cfg = ShuOsherBenchmarkConfig(coarse=100, medium=200, fine=400,
                                      very_fine=400, reference=400,
                                      output_dir=args.output)
    else:
        cfg = ShuOsherBenchmarkConfig(output_dir=args.output)

    result = run_benchmark(cfg)
    print("\n=== verdicts ===")
    for key, value in result["verdicts"].items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
    print(f"  resolution meeting 5%: {result['resolution_meeting_5pct']}")


if __name__ == "__main__":
    main()