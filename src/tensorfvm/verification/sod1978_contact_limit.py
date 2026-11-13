"""Why the Sod (1978) whole-field L1 rate sits below the design order — round 3.

The anchor ``REPRO-FVM-SOD1978`` grades a third criterion (C3) literally:
"whole-field L1 convergence order of the Sod solution >= 1".  Rounds 1 and 2
measured 0.62/0.65/0.66 (first order) and 0.98/0.89/0.86 (second order) and
reported an honest FAIL, but the round-2 diagnosis could only *hypothesise* the
mechanism, because the finest grid it reached was ``nx = 200`` (the ``nx`` up to
6400 sweep stayed queued on the cluster) and because no measurement existed for
the mechanism itself — the captured **contact discontinuity**, i.e. the linearly
degenerate field of the Riemann fan.

This module supplies the missing measurement.  It contains no new solver: it
drives the *same* ``EulerSolver`` (HLLC, MUSCL-Hancock, same limiter/CFL) that
produces the Sod numbers, on two problems:

``pure contact`` (control)
    Uniform velocity ``u0`` and uniform pressure ``p0``, density jump
    ``rho_L -> rho_R``.  The exact solution is a single linearly degenerate wave
    travelling at ``u0``; nothing else happens.  Uniformity of ``p`` and ``u``
    is an exact invariant (it is checked, not assumed), so whatever error the
    scheme accumulates here is *purely* the contact-capture error.  Measuring
    its decay rate is the cleanest available test of the mechanism.

``Sod zone budget``
    The same Sod run, split into disjoint cell sets around the four waves
    (rarefaction, contact, shock, remaining plateaus).  Because the sets are
    disjoint the per-zone absolute errors add up to the whole-field L1 error
    exactly, so the budget says *how much* of the whole-field error the contact
    carries at each grid, and whether that share grows under refinement.

Both measurements are declared-before-run: the thresholds live in
:data:`THRESHOLDS` at module level and are copied into the artifact, and the
verdicts are computed from them, never adjusted afterwards.  Nothing here
re-grades C3: if the literal criterion is missed the verdict says so.

CLI: ``python scripts/sod1978_contact_limit_study.py --out <dir>``.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .sod1978 import (
    GAMMA,
    LEFT,
    RIGHT,
    T_TARGET,
    X0,
    exact_sample,
    exact_star_state,
    l1_errors,
    run_case,
)

# ----------------------------------------------------------------------
# declared thresholds (fixed before any run; copied into the artifact)
# ----------------------------------------------------------------------
THRESHOLDS: Dict[str, float] = {
    # whole-field L1 rate required by the anchor criterion C3, literally
    "c3_literal_rate": 1.0,
    # the pure contact is a uniform-state problem: p and u must stay uniform.
    # relative drift above this means pressure/velocity noise contaminated the
    # contact measurement (the control would no longer isolate the mechanism)
    "uniformity_drift": 1.0e-10,
    # measured contact position vs x0 + u0 t, relative to the travelled distance
    "contact_position_tol": 0.01,
    # zone-budget additivity: the disjoint per-zone errors must sum to the
    # whole-field L1 error to round-off
    "zone_budget_tol": 1.0e-10,
    # the mechanism hypothesis is "the contact carries the majority of the
    # whole-field L1 error on the finest grid"
    "contact_dominant_fraction": 0.5,
    # ... and "the whole-field rate tracks the pure-contact rate": the two
    # rates must agree to this absolute tolerance
    "rate_tracking_tol": 0.20,
    # declared band (order 2) for a scheme whose global rate is limited by a
    # captured linear wave; also the band the anchor would need to use if C3
    # were re-specified as "literature band" instead of ">= 1 literally"
    "reviewed_band_order2": (0.80, 1.20),
}

# ----------------------------------------------------------------------
# pure contact control problem
# ----------------------------------------------------------------------
U0 = 0.5
P0 = 1.0
RHO_L = 1.0
RHO_R = 0.125
T_CONTACT = T_TARGET          # same time as the Sod anchor (t = 0.14)


def _grid_1d(nx: int) -> np.ndarray:
    return (np.arange(nx, dtype=float) + 0.5) / nx


def contact_exact(x: np.ndarray, t: float) -> Dict[str, np.ndarray]:
    """Exact pure-contact solution: a jump translating at ``U0``, nothing else."""
    x_c = X0 + U0 * t
    rho = np.where(x < x_c, RHO_L, RHO_R)
    return {"rho": rho,
            "u": np.full_like(rho, U0),
            "p": np.full_like(rho, P0),
            "x_contact": x_c}


def run_pure_contact(nx: int, *, order: int = 1, limiter: str = "mc",
                     cfl: float = 0.4, t_end: float = T_CONTACT,
                     flux_mode: str = "hllc") -> Dict:
    """Integrate the pure-contact control with the repository Euler solver."""
    import torch

    from ..compressible.cases import GAMMA_AIR
    from ..compressible.eos import StiffenedGasMixture
    from ..compressible.solver import EulerSolver

    eos = StiffenedGasMixture([GAMMA_AIR, GAMMA_AIR], names=["air", "air"])
    solver = EulerSolver(
        nx=nx, ny=1, lx=1.0, ly=1.0, eos=eos,
        bc=("outflow", "outflow", "periodic", "periodic"),
        flux_mode=flux_mode, order=order, limiter=limiter, cfl=cfl,
    )
    xc_t = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx
    left = xc_t < X0
    solver.set_primitive({
        "rho": torch.where(left, torch.full_like(xc_t, RHO_L),
                           torch.full_like(xc_t, RHO_R)),
        "u": torch.full_like(xc_t, U0),
        "v": torch.zeros_like(xc_t),
        "p": torch.full_like(xc_t, P0),
        "Y": torch.zeros_like(xc_t),
    })
    initial = solver.integrals()
    t0 = time.time()
    solver.run(t_end)
    seconds = time.time() - t0
    prim = solver.primitive()
    x = xc_t.numpy()
    out = {
        "nx": int(nx), "order": int(order), "limiter": limiter, "cfl": float(cfl),
        "flux_mode": flux_mode, "t_end": float(solver.time),
        "steps": int(solver.nsteps), "seconds": float(seconds),
        "positivity_fixes": int(solver.positivity_fixes),
        "budget_drift_mass": float(solver.budget_drift(initial)["mass"]),
        "x": x,
        "rho": prim["rho"][:, 0].numpy(),
        "u": prim["u"][:, 0].numpy(),
        "p": prim["p"][:, 0].numpy(),
    }
    return out


def contact_metrics(run: Dict) -> Dict[str, float]:
    """L1 errors, effective smearing width and position error of a contact run."""
    x, t = run["x"], run["t_end"]
    ex = contact_exact(x, t)
    h = float(x[1] - x[0])
    jump = abs(RHO_R - RHO_L)
    err = np.abs(run["rho"] - ex["rho"])
    # effective smearing width: the length over which the scheme has spread the
    # jump, i.e. the L1 mass of the error divided by the jump height
    width = float(err.sum() * h / jump)
    level = 0.5 * (RHO_L + RHO_R)
    pos = _crossing(x, run["rho"] - level, ex["x_contact"])
    return {
        "l1_rho": float(err.mean()),
        "l1_u": float(np.abs(run["u"] - ex["u"]).mean()),
        "l1_p": float(np.abs(run["p"] - ex["p"]).mean()),
        "width": width,
        "cells_smeared": width / h,
        "position": float(pos),
        "position_error": float(abs(pos - ex["x_contact"])),
        "position_error_rel": float(abs(pos - ex["x_contact"]) / (U0 * t)),
        "u_drift": float(np.abs(run["u"] - U0).max() / U0),
        "p_drift": float(np.abs(run["p"] - P0).max() / P0),
        "rho_min": float(run["rho"].min()),
    }


def _crossing(x: np.ndarray, f: np.ndarray, x_guess: float) -> float:
    """First sign change of ``f`` scanning outward from ``x_guess`` (linear interp)."""
    i0 = int(np.argmin(np.abs(x - x_guess)))
    n = len(f)
    for k in range(n):
        for i in (i0 - k, i0 + k - 1):
            if i < 0 or i + 1 >= n:
                continue
            if f[i] * f[i + 1] < 0.0:
                t = f[i] / (f[i] - f[i + 1])
                return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


# ----------------------------------------------------------------------
# Sod zone budget
# ----------------------------------------------------------------------
ZONES = ("rarefaction", "contact", "shock", "plateaus")


def sod_zone_budget(run: Dict, star: Dict[str, float], t: float,
                    margin_cells: int = 24) -> Dict[str, object]:
    """Split the Sod L1 error into disjoint wave zones.

    The cell sets are disjoint and their union is the whole domain, so
    ``sum(zone error) == mean|e|`` exactly (checked to ``zone_budget_tol``).
    The ``contact`` zone contributes the captured linearly degenerate wave; its
    ``contact_width`` is the L1 mass of that zone divided by the star-state
    density jump, i.e. the length over which the scheme has smeared the contact.
    """
    x = run["x"]
    h = float(x[1] - x[0])
    m = margin_cells * h
    xi = (x - X0) / t
    ex = exact_sample(xi, star)
    err = np.abs(run["rho"] - ex["rho"])

    x_head = X0 + star["rarefaction_head_speed"] * t
    x_tail = X0 + star["rarefaction_tail_speed"] * t
    x_c = X0 + star["contact_speed"] * t
    x_s = X0 + star["shock_speed"] * t

    in_rare = (x >= x_head - m) & (x <= x_tail + m)
    in_contact = (np.abs(x - x_c) <= m) & ~in_rare
    in_shock = (np.abs(x - x_s) <= m) & ~in_rare & ~in_contact
    in_plateau = ~(in_rare | in_contact | in_shock)

    masks = {"rarefaction": in_rare, "contact": in_contact,
             "shock": in_shock, "plateaus": in_plateau}
    # domain length is 1, so ``sum * h`` is the zone's additive share of the
    # whole-field mean absolute error
    zone = {k: float(err[mk].sum() * h) for k, mk in masks.items()}
    total = float(err.mean())
    jump = abs(star["rho_star_left"] - star["rho_star_right"])
    contact_width = float(err[in_contact].sum() * h / jump) if jump > 0 else float("nan")
    return {
        "l1_rho_full": total,
        "zone_l1": zone,
        "zone_fraction": {k: (v / total if total > 0 else float("nan"))
                          for k, v in zone.items()},
        "zone_cells": {k: int(mk.sum()) for k, mk in masks.items()},
        "budget_residual": float(abs(sum(zone.values()) - total)),
        "contact_width": contact_width,
        "contact_cells_smeared": contact_width / h,
        "star_density_jump": float(jump),
    }


def measure_sod(nx: int, *, order: int, limiter: str = "mc", cfl: float = 0.4,
                t_end: float = T_TARGET, margin_cells: int = 24,
                star: Dict[str, float] | None = None) -> Dict:
    star = star or exact_star_state()
    run = run_case(nx, order=order, t_end=t_end, limiter=limiter, cfl=cfl)
    budget = sod_zone_budget(run, star, run["t_end"], margin_cells=margin_cells)
    # robustness of the sampling convention: the whole-field error again, but
    # against the exact *cell average* instead of the cell-centre value.  If the
    # two rates agree, the sub-linear rate is not a point-sampling artifact.
    l1 = l1_errors(run, star)
    return {"nx": int(nx), "order": int(order), "limiter": limiter,
            "cfl": float(cfl), "steps": int(run["steps"]),
            "seconds": float(run["seconds"]),
            "l1_rho_cellavg": float(l1["l1_rho_cellavg"]),
            **{k: v for k, v in budget.items() if k != "l1_rho_full"},
            "l1_rho_full": budget["l1_rho_full"]}


def measure_contact(nx: int, *, order: int, limiter: str = "mc",
                    cfl: float = 0.4, t_end: float = T_CONTACT) -> Dict:
    run = run_pure_contact(nx, order=order, limiter=limiter, cfl=cfl, t_end=t_end)
    m = contact_metrics(run)
    return {"nx": int(nx), "order": int(order), "limiter": limiter,
            "cfl": float(cfl), "steps": run["steps"], "seconds": run["seconds"],
            "budget_drift_mass": run["budget_drift_mass"],
            "positivity_fixes": run["positivity_fixes"], **m}


# ----------------------------------------------------------------------
# rates
# ----------------------------------------------------------------------
def _val(row: Dict, key: str) -> float:
    cur: object = row
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return float("nan")
        cur = cur[part]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return float("nan")


def successive_rates(rows: Sequence[Dict], key: str) -> List[Dict]:
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        ea, eb = _val(a, key), _val(b, key)
        if ea > 0 and eb > 0 and math.isfinite(ea) and math.isfinite(eb):
            out.append({"nx_pair": [a["nx"], b["nx"]],
                        "rate": math.log(ea / eb) / math.log(b["nx"] / a["nx"])})
    return out


def fitted_exponent(rows: Sequence[Dict], key: str, finest: int = 3) -> float:
    sel = [r for r in rows
           if math.isfinite(_val(r, key)) and _val(r, key) > 0]
    sel = sorted(sel, key=lambda r: r["nx"])[-finest:]
    if len(sel) < 2:
        return float("nan")
    lx = np.log([1.0 / r["nx"] for r in sel])
    ly = np.log([_val(r, key) for r in sel])
    return float(np.polyfit(lx, ly, 1)[0])


RATE_KEYS_SOD = ("l1_rho_full", "l1_rho_cellavg", "zone_l1.contact", "zone_l1.rarefaction",
                 "zone_l1.shock", "zone_l1.plateaus", "contact_width")
RATE_KEYS_CONTACT = ("l1_rho", "width", "position_error")


# ----------------------------------------------------------------------
# study driver
# ----------------------------------------------------------------------
@dataclass
class ContactStudyConfig:
    grids: Tuple[int, ...] = (200, 400, 800, 1600, 3200, 6400)
    orders: Tuple[int, ...] = (1, 2)
    limiter: str = "mc"
    cfl: float = 0.4
    margin_cells: int = 24
    t_end: float = T_TARGET
    figures: bool = True
    notes: Dict[str, str] = field(default_factory=dict)


def run_study(cfg: ContactStudyConfig, outdir: Path | str, log=print) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    star = exact_star_state()
    started = time.time()

    sod: Dict[str, List[Dict]] = {}
    contact: Dict[str, List[Dict]] = {}
    for order in cfg.orders:
        rows_s, rows_c = [], []
        for nx in cfg.grids:
            r = measure_sod(nx, order=order, limiter=cfg.limiter, cfl=cfg.cfl,
                            t_end=cfg.t_end, margin_cells=cfg.margin_cells,
                            star=star)
            log(f"[sod]     order={order} nx={nx:5d} L1={r['l1_rho_full']:.4e} "
                f"contact_frac={r['zone_fraction']['contact']:.3f} "
                f"({r['seconds']:.1f}s)")
            rows_s.append(r)
            c = measure_contact(nx, order=order, limiter=cfg.limiter, cfl=cfg.cfl,
                                t_end=cfg.t_end)
            log(f"[contact] order={order} nx={nx:5d} L1={c['l1_rho']:.4e} "
                f"width={c['cells_smeared']:.2f} cells "
                f"u_drift={c['u_drift']:.2e} ({c['seconds']:.1f}s)")
            rows_c.append(c)
        sod[f"order{order}"] = rows_s
        contact[f"order{order}"] = rows_c

    def _rate_block(table: Dict[str, List[Dict]], keys: Sequence[str]) -> Dict:
        return {order: {key: {"successive": successive_rates(rows, key),
                              "fitted_exponent_finest3": fitted_exponent(rows, key)}
                        for key in keys}
                for order, rows in table.items()}

    rates_sod = _rate_block(sod, RATE_KEYS_SOD)
    rates_contact = _rate_block(contact, RATE_KEYS_CONTACT)

    verdicts = _verdicts(sod, contact)
    study = {
        "case": "sod1978_contact_limit",
        "anchor": "REPRO-FVM-SOD1978",
        "round": 3,
        "external_anchor": "Sod, G. A., J. Comput. Phys. 27 (1978) 1-31",
        "method_anchor": ("Banks, Aslam & Rider, J. Comput. Phys. 227 (2008) "
                          "6985-7002 (sub-linear convergence of captured "
                          "linearly degenerate waves)"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__,
                        "machine": platform.machine(),
                        "platform": platform.platform()},
        "config": {
            "grids": list(cfg.grids), "orders": list(cfg.orders),
            "limiter": cfg.limiter, "cfl": cfg.cfl,
            "margin_cells": cfg.margin_cells, "t_end": cfg.t_end,
            "gamma": GAMMA, "x0": X0,
            "sod_left": LEFT, "sod_right": RIGHT,
            "contact": {"rho_left": RHO_L, "rho_right": RHO_R,
                        "u": U0, "p": P0, "t_end": T_CONTACT},
            "notes": cfg.notes,
        },
        "thresholds_declared_before_run": {k: (list(v) if isinstance(v, tuple) else v)
                                           for k, v in THRESHOLDS.items()},
        "sod": sod,
        "contact": contact,
        "rates": {"sod": rates_sod, "contact": rates_contact},
        "verdicts": verdicts,
        "wall_seconds": time.time() - started,
    }
    if cfg.figures:
        study["figures"] = render_figures(study, outdir)
    (outdir / "contact_study.json").write_text(
        json.dumps(study, indent=2, sort_keys=True, default=str) + "\n")
    log(f"[done] {outdir/'contact_study.json'} wall={study['wall_seconds']:.1f}s")
    return study


def _finest(rows: Sequence[Dict]) -> Dict:
    return sorted(rows, key=lambda r: r["nx"])[-1]


def _verdicts(sod: Dict[str, List[Dict]], contact: Dict[str, List[Dict]]) -> Dict:
    th = THRESHOLDS
    out: Dict[str, object] = {}
    for order in (1, 2):
        key = f"order{order}"
        if key not in sod:
            continue
        s_rows, c_rows = sod[key], contact[key]
        s_rate = _val(_last(successive_rates(s_rows, "l1_rho_full")), "rate")
        c_rate = _val(_last(successive_rates(c_rows, "l1_rho")), "rate")
        w_rate = _val(_last(successive_rates(c_rows, "width")), "rate")
        frac = _finest(s_rows)["zone_fraction"]["contact"]
        lo, hi = th["reviewed_band_order2"]
        out[key] = {
            "sod_full_l1_rate_finest_pair": s_rate,
            "pure_contact_l1_rate_finest_pair": c_rate,
            "pure_contact_width_rate_finest_pair": w_rate,
            "rate_difference": abs(s_rate - c_rate)
            if math.isfinite(s_rate) and math.isfinite(c_rate) else float("nan"),
            "contact_fraction_finest": frac,
            "c3_literal_met": bool(s_rate >= th["c3_literal_rate"]),
            "contact_dominant": bool(frac >= th["contact_dominant_fraction"]),
            "rate_tracks_pure_contact": bool(
                math.isfinite(s_rate) and math.isfinite(c_rate)
                and abs(s_rate - c_rate) <= th["rate_tracking_tol"]),
            "uniformity_preserved": bool(
                _finest(c_rows)["u_drift"] <= th["uniformity_drift"]
                and _finest(c_rows)["p_drift"] <= th["uniformity_drift"]),
            "contact_position_ok": bool(
                _finest(c_rows)["position_error_rel"] <= th["contact_position_tol"]),
            "budget_additive": bool(
                max(r["budget_residual"] for r in s_rows) <= th["zone_budget_tol"]),
            "reviewed_band_order2": [lo, hi],
            "in_reviewed_band_order2": bool(
                order == 2 and math.isfinite(s_rate) and lo <= s_rate <= hi),
        }
    return out


def _last(seq: Sequence[Dict]) -> Dict:
    return seq[-1] if seq else {}


def render_figures(study: Dict, outdir: Path | str) -> List[Dict]:
    """Two figures: (a) error-vs-grid for Sod zones and the pure contact,
    (b) smeared contact width in cells vs grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                  # pragma: no cover
        return [{"name": "unavailable", "reason": str(exc)}]
    outdir = Path(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for order, style in ((1, "o-"), (2, "s-")):
        key = f"order{order}"
        if key not in study["sod"]:
            continue
        rows = study["sod"][key]
        nx = [r["nx"] for r in rows]
        axes[0].loglog(nx, [r["l1_rho_full"] for r in rows], style,
                       label=f"Sod whole field, order {order}")
        axes[0].loglog(nx, [r["zone_l1"]["contact"] for r in rows], style,
                       alpha=0.45,
                       label=f"Sod contact zone, order {order}")
        crow = study["contact"][key]
        axes[0].loglog([r["nx"] for r in crow], [r["l1_rho"] for r in crow],
                       style, alpha=0.7, markerfacecolor="none",
                       label=f"pure contact, order {order}")
        axes[1].loglog(nx, [r["contact_cells_smeared"] for r in rows], style,
                       label=f"Sod contact width, order {order}")
        axes[1].loglog([r["nx"] for r in crow],
                       [r["cells_smeared"] for r in crow], style, alpha=0.6,
                       markerfacecolor="none",
                       label=f"pure contact width, order {order}")
    for ax, ylab, title in (
            (axes[0], "L1 error of density", "L1 error vs grid (contact-limited)"),
            (axes[1], "smeared width / h  [cells]",
             "captured contact width vs grid")):
        ax.set_xlabel("nx")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "contact_limit_scaling.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append({"name": p.name,
                 "caption": "Left: whole-field and contact-zone L1 error of the "
                            "Sod run against the pure-contact control on the same "
                            "solver. Right: captured contact width in cells."})
    return made


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--out", default=str(root / "docs" / "repro-fvm-sod1978-contact-limit"))
    ap.add_argument("--grids", default="200,400,800,1600,3200,6400")
    ap.add_argument("--orders", default="1,2")
    ap.add_argument("--limiter", default="mc")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--margin-cells", type=int, default=24)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    def ints(s: str) -> Tuple[int, ...]:
        return tuple(int(x) for x in s.split(",") if x.strip())

    cfg = ContactStudyConfig(
        grids=ints(args.grids), orders=ints(args.orders), limiter=args.limiter,
        cfl=args.cfl, margin_cells=args.margin_cells,
        figures=not args.no_figures,
        notes={"cli": " ".join(argv if argv is not None else [])})
    run_study(cfg, args.out)
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())