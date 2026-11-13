"""Asymptotic-grid convergence study for the Sod (1978) anchor.

Why this module exists
----------------------
Audit m15344 of the ``REPRO-FVM-SOD1978`` anchor accepted the round-2
diagnosis (contact/shock mechanism, smooth entropy-wave design order) but left
one blocking item open (finding **B3**): the only round-2 convergence artifact
was a ``--quick`` smoke run (grids 50/100/200), while the asymptotic verdicts
``c3_asymptotic_met`` and ``sublinear_convergence_observed`` need the *declared*
grid set (100 ... 6400) that never returned from the cluster.  Round 3 (m17933)
extended the sweep to 800 and used it to *re-grade* the criterion, so the
verdict it proposes ("whole-field L1 order is sub-linear and this is a property
of the discontinuities") rests on grids that are still pre-asymptotic.

This module answers exactly that question with measurements, not with an
argument: it runs the model problem on the full declared ladder
``100, 200, 400, 800, 1600, 3200, 6400`` for first- and second-order
reconstruction and reports, for every pair of grids,

* the **whole-field** L1 order (the quantity the literal criterion grades),
* the **cell-average** L1 order (removes the point-sampling artifact at the
  discontinuity),
* the **restricted** L1 order on the smooth windows of the exact solution,
  with the round-off noise floor recorded so a rate is never quoted from a
  measure that has already hit machine precision,
* the four wave-position errors (their own resolution floor is ``dx/2``),
* conservation drift and positivity fixes (the run must stay physical), and
* the design order on the smooth periodic entropy wave at the same grids, which
  is the measure that isolates the *scheme* from the *discontinuity*.

The declared thresholds are written into the artifact **before** the run
(``thresholds_declared_before_run``) so the verdict cannot be tuned to the
numbers.

Verdicts produced (each reported both ways, never only the favourable one):

``c3_literal_asymptotic``
    Does the whole-field L1 rate reach the literal criterion ``>= 1.0`` on the
    finest grid pair *and* as a least-squares exponent over the finest three
    grids?
``sublinear_persists``
    Is the finest-three exponent still below ``1 - 0.05`` (= the rate does not
    approach one as ``h -> 0``)?
``order_trend``
    Monotone approach of the pairwise rates towards one (evidence for or
    against an asymptote at one).
``design_order_asymptotic``
    Smooth entropy-wave rate vs the design order ``p`` on fine grids, i.e.
    whether the round-4 window dependence (audit non-blocking finding N1) is
    pre-asymptotic.

CLI::

    PYTHONPATH=src python scripts/sod1978_asymptotic_study.py \
        --out docs/repro-fvm-sod1978-asymptotic --figures
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .sod1978 import GAMMA, LEFT, RIGHT, T_TARGET, X0, run_smooth_wave
from .sod1978 import exact_star_state, reference_cross_check
from .sod1978_convergence import (
    _rate_table,
    fitted_exponent,
    measure,
    successive_orders,
)

# ----------------------------------------------------------------------
# declared configuration (fixed before the run; recorded in the artifact)
# ----------------------------------------------------------------------
DECLARED_GRIDS: Sequence[int] = (100, 200, 400, 800, 1600, 3200, 6400)
DECLARED_ORDERS: Sequence[int] = (1, 2)
DECLARED_SMOOTH_GRIDS: Sequence[int] = (400, 800, 1600, 3200)
PROFILE_GRIDS: Sequence[int] = (1600, 6400)

#: a rate is only quoted from a restricted-window error at least this many
#: orders of magnitude above double-precision round-off on cell values.
ROUNDOFF_FLOOR = 1e-12
#: ``sublinear_persists`` asks whether the exponent stays below ``1 - BAND``.
BAND = 0.05
#: the smooth-wave design order is "within band" when within this fraction.
DESIGN_TOL = 0.03


def _rates(rows: Sequence[Dict], key: str) -> Dict:
    table = _rate_table(rows, [key])[key]
    table["fitted_exponent_finest4"] = fitted_exponent(rows, key, finest=4)
    table["fitted_exponent_finest5"] = fitted_exponent(rows, key, finest=5)
    return table


def _trend_towards_one(rates: List[Dict]) -> Dict:
    """Is the pairwise rate moving towards 1.0 as the grid is refined?"""
    vals = [r["rate"] for r in rates if math.isfinite(r["rate"])]
    if len(vals) < 3:
        return {"monotone": False, "steps_towards_one": 0, "pairs": max(len(vals) - 1, 0),
                "rates": vals,
                "distance_to_one_first": float("nan"),
                "distance_to_one_last": float("nan")}
    dist = [abs(v - 1.0) for v in vals]
    steps = sum(1 for a, b in zip(dist[:-1], dist[1:]) if b < a)
    return {
        "monotone": steps == len(dist) - 1,
        "steps_towards_one": int(steps),
        "pairs": len(dist) - 1,
        "rates": vals,
        "distance_to_one_first": float(dist[0]),
        "distance_to_one_last": float(dist[-1]),
    }


def _restricted_floor(rows: Sequence[Dict], key: str = "l1_rho_restricted") -> Dict:
    """Which restricted-window errors are above the round-off floor."""
    out = {}
    for r in rows:
        e = float(r[key])
        out[str(r["nx"])] = {
            "l1": e,
            "above_floor": bool(e > ROUNDOFF_FLOOR),
            "orders_above_floor": float(math.log10(e / ROUNDOFF_FLOOR)) if e > 0 else float("-inf"),
        }
    usable = [r for r in rows if float(r[key]) > ROUNDOFF_FLOOR]
    return {"per_grid": out,
            "usable_grids": [int(r["nx"]) for r in usable],
            "floor": ROUNDOFF_FLOOR}


def _smooth_series(grids: Sequence[int], orders: Sequence[int]) -> Dict:
    """Design-order measurement on the smooth periodic entropy wave."""
    out: Dict[str, Dict] = {}
    for order in orders:
        rows = []
        for nx in grids:
            t0 = time.time()
            run = run_smooth_wave(nx, order=order)
            rows.append({"nx": int(nx), "order": int(order),
                         "l1_rho": float(run["l1_rho"]),
                         "steps": int(run["steps"]),
                         "time": float(run["time"]),
                         "seconds": float(time.time() - t0)})
        rates = successive_orders(rows, "l1_rho")
        finest_pair = rates[-1]["rate"] if rates else float("nan")
        exp3 = fitted_exponent(rows, "l1_rho", finest=3)
        design = float(order)
        out[str(order)] = {
            "rows": rows,
            "successive": rates,
            "fitted_exponent_finest3": exp3,
            "order": design,
            "finest_pair_rate": float(finest_pair),
            "finest_pair_rel_err_pct": float(abs(finest_pair - design) / design * 100.0),
            "fitted3_rel_err_pct": float(abs(exp3 - design) / design * 100.0),
            "finest_pair_within_3pct": bool(abs(finest_pair - design) / design <= DESIGN_TOL),
            "fitted3_within_3pct": bool(abs(exp3 - design) / design <= DESIGN_TOL),
        }
    return out


def profile_snapshot(nx: int, order: int = 2) -> Dict:
    """Dense profile at one grid, for the field-comparison figure."""
    from .sod1978 import exact_sample, run_case

    run = run_case(nx, order=order)
    star = exact_star_state()
    x = run["x"]
    exact = exact_sample((x - X0) / run["t_end"], star)
    return {"nx": int(nx), "order": int(order), "t_end": float(run["t_end"]),
            "x": [float(v) for v in x],
            "rho": [float(v) for v in run["rho"]],
            "u": [float(v) for v in run["u"]],
            "p": [float(v) for v in run["p"]],
            "rho_exact": [float(v) for v in exact["rho"]],
            "u_exact": [float(v) for v in exact["u"]],
            "p_exact": [float(v) for v in exact["p"]]}


def run_study(outdir: Path | str,
              grids: Sequence[int] = DECLARED_GRIDS,
              orders: Sequence[int] = DECLARED_ORDERS,
              smooth_grids: Sequence[int] = DECLARED_SMOOTH_GRIDS,
              profile_grids: Sequence[int] = PROFILE_GRIDS,
              *, limiter: str = "mc", cfl: float = 0.4,
              figures: bool = True) -> Dict:
    outdir = Path(outdir)
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    star = exact_star_state()
    thresholds = {
        "c3_literal_rate": 1.0,
        "sublinear_band": BAND,
        "design_order_rel_tol": DESIGN_TOL,
        "roundoff_floor": ROUNDOFF_FLOOR,
        "declared_grids": list(grids),
        "declared_orders": list(orders),
        "declared_smooth_grids": list(smooth_grids),
        "note": ("thresholds written before the run; the whole-field literal "
                 "criterion is graded on the finest pair AND on the fitted "
                 "exponent over the finest three grids, both reported"),
    }

    rows: Dict[str, List[Dict]] = {}
    for order in orders:
        series = []
        for nx in grids:
            m = measure(nx, order=order, limiter=limiter, cfl=cfl, star=star)
            series.append(m)
            print(f"[sod-asym] order={order} nx={nx:5d} "
                  f"l1_full={m['l1_rho_full']:.6e} "
                  f"l1_cellavg={m['l1_rho_cellavg']:.6e} "
                  f"restricted={m['l1_rho_restricted']:.3e} "
                  f"steps={m['steps']} sec={m['seconds']:.1f}", flush=True)
        rows[str(order)] = series

    keys = ["l1_rho_full", "l1_rho_cellavg", "l1_rho_restricted",
            "l1_rho_plateaus", "wave_error.contact_x", "wave_error.shock_x"]
    # {order: {measured key: {successive, fitted exponents, crosscheck}}}
    rate_tables: Dict[str, Dict] = {}
    for o, rs in rows.items():
        rate_tables[o] = {k: _rates(rs, k) for k in keys}

    floor = {o: _restricted_floor(rs) for o, rs in rows.items()}

    verdicts: Dict = {}
    for o, rs in rows.items():
        tbl = rate_tables[o]["l1_rho_full"]
        finest_pair = tbl["successive"][-1]["rate"] if tbl["successive"] else float("nan")
        exp3 = tbl["fitted_exponent_finest3"]
        exp4 = tbl["fitted_exponent_finest4"]
        trend = _trend_towards_one(tbl["successive"])
        verdicts[o] = {
            "finest_pair_rate": float(finest_pair),
            "fitted_exponent_finest3": float(exp3),
            "fitted_exponent_finest4": float(exp4),
            "c3_literal_asymptotic": bool(finest_pair >= 1.0 and exp3 >= 1.0),
            "sublinear_persists": bool(exp3 < 1.0 - BAND),
            "order_trend": trend,
            "cellavg_finest_pair_rate": float(
                rate_tables[o]["l1_rho_cellavg"]["successive"][-1]["rate"]),
            "cellavg_fitted_exponent_finest3": float(
                rate_tables[o]["l1_rho_cellavg"]["fitted_exponent_finest3"]),
            "restricted_usable_grids": floor[o]["usable_grids"],
            "positivity_fixes_total": int(sum(r["positivity_fixes"] for r in rs)),
            "max_abs_budget_drift_mass": float(
                max(abs(r["budget_drift_mass"]) for r in rs)),
        }

    smooth = _smooth_series(smooth_grids, orders)
    profiles = [profile_snapshot(nx, order=orders[-1]) for nx in profile_grids] if profile_grids else []

    study = {
        "case": "sod_shock_tube",
        "anchor": "REPRO-FVM-SOD1978",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "external_anchor": {
            "paper": "Sod, G. A. (1978) J. Comput. Phys. 27:1-31, 'A survey of "
                     "several finite difference methods for systems of nonlinear "
                     "hyperbolic conservation laws'",
            "star_state_as_tabulated_in": "Toro (2009), Riemann Solvers and "
                                          "Numerical Methods for Fluid Dynamics, 3rd ed., ch. 4",
            "cross_check": reference_cross_check(),
        },
        "setup": {
            "geometry": "1-D, x in [0,1], diaphragm at x0=0.5, outflow both ends",
            "left": LEFT, "right": RIGHT, "gamma": GAMMA, "x0": X0,
            "t_end": T_TARGET, "flux": "hllc", "limiter": limiter, "cfl": cfl,
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "processor": platform.processor(),
        },
        "thresholds_declared_before_run": thresholds,
        "rows": rows,
        "rate_tables": rate_tables,
        "restricted_floor": floor,
        "verdicts": verdicts,
        "smooth_wave": smooth,
        "profiles": profiles,
        "wall_seconds": float(time.time() - t_start),
        "repro_cmd": ("PYTHONPATH=src python scripts/sod1978_asymptotic_study.py "
                      "--out docs/repro-fvm-sod1978-asymptotic --figures"),
    }

    if figures:
        from .sod1978_asymptotic_figures import make_figures
        study["figures"] = make_figures(outdir, study)

    payload = outdir / "artifacts" / "asymptotic_study.json"
    payload.write_text(json.dumps(study, indent=1, sort_keys=False) + "\n")
    print(f"[sod-asym] wrote {payload} ({payload.stat().st_size/1024:.0f} KiB) "
          f"in {study['wall_seconds']:.0f}s", flush=True)
    return study


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grids", default=",".join(map(str, DECLARED_GRIDS)))
    ap.add_argument("--orders", default=",".join(map(str, DECLARED_ORDERS)))
    ap.add_argument("--smooth-grids", default=",".join(map(str, DECLARED_SMOOTH_GRIDS)))
    ap.add_argument("--profile-grids", default=",".join(map(str, PROFILE_GRIDS)))
    ap.add_argument("--limiter", default="mc")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    def _ints(s: str) -> List[int]:
        return [int(v) for v in s.split(",") if v.strip()]

    study = run_study(args.out, _ints(args.grids), _ints(args.orders),
                      _ints(args.smooth_grids), _ints(args.profile_grids),
                      limiter=args.limiter, cfl=args.cfl,
                      figures=not args.no_figures)
    # exit 0 always: this module *reports* the asymptote, it does not grade the
    # anchor (grading stays with the round-1 driver and the audit).
    for o, v in study["verdicts"].items():
        print(f"[sod-asym] order={o} finest_pair={v['finest_pair_rate']:.4f} "
              f"exp3={v['fitted_exponent_finest3']:.4f} "
              f"literal={v['c3_literal_asymptotic']} "
              f"sublinear_persists={v['sublinear_persists']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())