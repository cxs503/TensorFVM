#!/usr/bin/env python3
"""Run one third-party-anchored incompressible benchmark and dump its ledger.

Usage
-----
    python scripts/run_incompressible_benchmarks.py bfs --resolution 20
    python scripts/run_incompressible_benchmarks.py expansion --resolution 40
    python scripts/run_incompressible_benchmarks.py convection --rayleigh 1e6 --resolution 64

Each invocation writes ``<root>/raw_<case>_<tag>.json`` (metrics + convergence
ledger + the parameters that produced them) and ``<root>/fields_<case>_<tag>.npz``
(field snapshots used by the figure script).  ``--summary`` reads every raw file
in the directory, forms the grid-convergence triplets, applies the acceptance
criteria and writes ``summary.json``.  Every number in the summary is computed
from the raw files, never typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.incompressible_benchmarks import (
    ARMALY_REATTACHMENT,
    BORDA_CARNOT_AR4,
    DE_VAHL_DAVIS,
    GARTLING_REATTACHMENT,
    backward_facing_step,
    loss_coefficient,
    mean_nusselt,
    natural_convection,
    reattachment_length,
    sudden_expansion,
    total_pressure_flux,
)

ROOT = (Path(__file__).resolve().parent.parent / "src" / "tensorfvm" / "verification"
        / "artifacts" / "incompressible")


def _finite(value: float) -> float | None:
    """``None`` for an unresolved (``nan``) metric, so the JSON stays strict."""
    value = float(value)
    return value if math.isfinite(value) else None


def _step_report(step: int, residual: float, extra: str = "") -> None:
    print(f"  step {step:7d}  residual {residual:.3e}  {extra}", flush=True)


def run_bfs(resolution: int, root: Path, *, max_steps: int, tolerance: float,
            cfl: float, warm: str | None) -> dict:
    flow = backward_facing_step(800.0, resolution, cfl=cfl)
    if warm:
        flow.load_state(Path(warm))
    def _bfs_check(f, s, r):
        x_ = f.x_face_coordinates() - 4.0
        tau_ = f.wall_shear_lower()
        _step_report(s, r, "Lr/H=%.3f" % reattachment_length(x_, tau_, start=0.02))

    ledger = flow.run_steady(tolerance=tolerance, max_steps=max_steps,
                             check_every=500, on_check=_bfs_check)
    x = flow.x_face_coordinates() - 4.0
    tau = flow.wall_shear_lower()
    q = 0.5 * 1.0 * 1.0
    lower = x > 0.0
    cf = tau / q
    lr = _finite(reattachment_length(x, tau, start=0.02, min_attached=1.0))
    # inlet/outlet flux balance as an independent mass-conservation probe
    return {
        "case": "backward_facing_step",
        "tag": f"H{resolution}",
        "parameters": {
            "re_h": 800.0, "cells_per_step": resolution, "nx": flow.nx,
            "ny": flow.ny, "h": flow.h, "nu": flow.nu, "cfl": flow.cfl,
            "upstream_length": 4.0, "downstream_length": flow.nx * flow.h - 4.0,
            "step_height": 1.0, "inlet": "fully developed plane Poiseuille",
        },
        "metrics": {
            "reattachment_length_over_H": lr,
            "reattachment_reference_gartling": GARTLING_REATTACHMENT,
            "reattachment_reference_armaly": ARMALY_REATTACHMENT,
            "reattachment_relative_error_gartling": (
                None if lr is None
                else abs(lr - GARTLING_REATTACHMENT) / GARTLING_REATTACHMENT),
            "reattachment_relative_error_armaly": (
                None if lr is None
                else abs(lr - ARMALY_REATTACHMENT) / ARMALY_REATTACHMENT),
            "min_cf_lower_wall": float(np.min(cf[lower])),
            "x_at_min_cf": float(x[lower][np.argmin(cf[lower])]),
            "inflow_flux": float(np.sum(flow.u[0, :])),
            "outflow_flux": float(np.sum(flow.u[-1, :])),
            "flux_imbalance": float(abs(np.sum(flow.u[0, :]) - np.sum(flow.u[-1, :]))
                                    / np.sum(flow.u[0, :])),
            "omega_min": float(flow.vorticity().min()),
            "steps_to_steady": ledger.steps,
            "simulated_time": float(ledger.history[-1][1]) if ledger.history else 0.0,
            "time_step": ledger.dt,
        },
        "ledger": ledger.as_dict(),
        "profiles": {
            "x_over_H": (flow.x_face_coordinates() - 4.0).tolist(),
            "cf_lower": cf.tolist(),
        },
        "fields": {"u": flow.u, "v": flow.v, "p": flow.p},
    }


def run_expansion(resolution: int, root: Path, *, max_steps: int, tolerance: float,
                  cfl: float) -> dict:
    flow = sudden_expansion(4.0, 100.0, resolution, cfl=cfl)
    ledger = flow.run_steady(tolerance=tolerance, max_steps=max_steps,
                             check_every=500,
                             on_check=lambda f, s, r: _step_report(s, r))
    x = flow.x_face_coordinates() - 3.0
    q = 0.5 * 1.0 * 1.0
    tau_lo = flow.wall_shear_lower()
    tau_hi = flow.wall_shear_upper()
    lr_lo = _finite(reattachment_length(x, tau_lo, start=0.02, min_attached=1.0))
    lr_hi = _finite(reattachment_length(x, tau_hi, start=0.02, min_attached=1.0))
    i_in = int(round((3.0 - 0.4) / flow.h))     # 0.4 D upstream of the plane
    i_out = int(round((3.0 + 2.0) / flow.h))    # 2 D downstream of the plane
    k_raw = loss_coefficient(flow, i_in, i_out)
    hp_in = total_pressure_flux(flow, i_in)
    hp_out = total_pressure_flux(flow, i_out)
    k_tp = (hp_in - hp_out) / q
    return {
        "case": "sudden_expansion",
        "tag": f"D{resolution}",
        "parameters": {
            "area_ratio": 4.0, "re_d": 100.0, "cells_per_diameter": resolution,
            "nx": flow.nx, "ny": flow.ny, "h": flow.h, "nu": flow.nu,
            "cfl": flow.cfl, "upstream_length": 3.0,
            "downstream_length": flow.nx * flow.h - 3.0,
            "pressure_stations_x_over_D": [-0.4, 2.0],
        },
        "metrics": {
            "reattachment_length_over_D_lower": lr_lo,
            "reattachment_length_over_D_upper": lr_hi,
            "k_static_raw": k_raw,
            "k_total_pressure": k_tp,
            "k_borda_carnot": BORDA_CARNOT_AR4,
            "k_relative_error_total_pressure": abs(k_tp - BORDA_CARNOT_AR4) / BORDA_CARNOT_AR4,
            "k_relative_error_static": abs(k_raw - BORDA_CARNOT_AR4) / BORDA_CARNOT_AR4,
            "flux_imbalance": float(abs(np.sum(flow.u[0, :]) - np.sum(flow.u[-1, :]))
                                    / np.sum(flow.u[0, :])),
            "steps_to_steady": ledger.steps,
            "time_step": ledger.dt,
        },
        "ledger": ledger.as_dict(),
        "profiles": {
            "x_over_D": (flow.x_face_coordinates() - 3.0).tolist(),
            "cf_lower": (tau_lo / q).tolist(),
        },
        "fields": {"u": flow.u, "v": flow.v, "p": flow.p},
    }


def run_convection(rayleigh: float, resolution: int, *, max_steps: int,
                   tolerance: float, cfl: float) -> dict:
    flow = natural_convection(rayleigh, cells=resolution, cfl=cfl)
    ledger = flow.run_steady(tolerance=tolerance, max_steps=max_steps,
                             check_every=500,
                             on_check=lambda f, s, r: _step_report(s, r))
    psi = flow.streamfunction()
    psi_ext = float(psi.max()) if abs(psi.max()) > abs(psi.min()) else float(psi.min())
    u_c, v_c = flow.cell_velocity()
    # ``de Vahl Davis`` tabulates ``u_max`` on the *vertical mid-line* (x = 0.5)
    # and ``v_max`` on the *horizontal mid-line* (y = 0.5), not the field
    # maxima.  At Ra >= 1e5 the horizontal velocity extremum sits in the wall
    # boundary layer, so a whole-field ``max|u|`` is up to 2x the reference
    # value and reports a spurious O(100 %) error.  Both are stored; only the
    # mid-line pair is compared against the reference contract.
    u_mid = flow.u[resolution // 2, :]
    v_mid = flow.v[:, resolution // 2]
    ref = DE_VAHL_DAVIS[float(rayleigh)]
    nu_hot = mean_nusselt(flow)
    cold_flux = (flow.theta[-1, :] - (-flow.theta_hot)) / (0.5 * flow.h)
    nu_cold = float(np.mean(cold_flux))
    wall_nu = (flow.theta_hot - flow.theta[0, :]) / (0.5 * flow.h)
    return {
        "case": "natural_convection",
        "tag": f"Ra{rayleigh:.0e}_N{resolution}",
        "parameters": {
            "rayleigh": rayleigh, "prandtl": flow.prandtl, "cells": resolution,
            "h": flow.h, "nu": flow.nu, "cfl": flow.cfl,
            "aspect_ratio": 1.0, "theta_hot": flow.theta_hot,
        },
        "metrics": {
            "nusselt_hot": nu_hot,
            "nusselt_cold": nu_cold,
            "nusselt_energy_imbalance": abs(nu_hot - nu_cold) / nu_hot,
            "nusselt_reference": ref["nu"],
            "nusselt_relative_error": abs(nu_hot - ref["nu"]) / ref["nu"],
            "psi_extremum": psi_ext,
            "psi_extremum_abs": abs(psi_ext),
            "psi_min": float(psi.min()),
            "psi_max": float(psi.max()),
            "psi_max_reference": ref["psi_max"],
            "psi_max_relative_error": abs(abs(psi_ext) - ref["psi_max"]) / ref["psi_max"],
            "u_max": float(np.abs(u_c).max()),
            "v_max": float(np.abs(v_c).max()),
            "u_max_reference": ref["u_max"],
            "v_max_reference": ref["v_max"],
            "u_max_midheight": float(np.abs(u_mid).max()),
            "v_max_midwidth": float(np.abs(v_mid).max()),
            "u_max_midheight_relative_error":
                abs(float(np.abs(u_mid).max()) - ref["u_max"]) / ref["u_max"],
            "v_max_midwidth_relative_error":
                abs(float(np.abs(v_mid).max()) - ref["v_max"]) / ref["v_max"],
            "steps_to_steady": ledger.steps,
            "time_step": ledger.dt,
        },
        "ledger": ledger.as_dict(),
        "profiles": {
            "y": ((np.arange(resolution) + 0.5) * flow.h).tolist(),
            "nusselt_hot_wall": wall_nu.tolist(),
            "u_midheight": flow.u[resolution // 2, :].tolist(),
            "v_midwidth": flow.v[:, resolution // 2].tolist(),
            "theta_midheight": flow.theta[resolution // 2, :].tolist(),
        },
        "fields": {"u": flow.u, "v": flow.v, "p": flow.p, "theta": flow.theta},
    }


def observed_order(values: dict[int, float], level: str) -> dict:
    """Observed order from three grid levels with refinement ratio 2 (fine->coarse)."""
    keys = sorted(values, reverse=True)          # finest first
    if len(keys) < 3:
        return {"levels": keys, "order": None, "reason": "fewer than three levels"}
    f1, f2, f3 = (values[k] for k in keys[:3])
    r = 2.0
    denom = (f2 - f3)
    if abs(denom) < 1e-14 or abs(f1 - f2) < 1e-14:
        return {"levels": keys, "order": None,
                "reason": "degenerate differences (grid-converged to round-off)"}
    # ``p = log|(f2-f3)/(f1-f2)| / log r``: the *coarse* difference is the larger
    # one for a converging series.  The previous form used the inverted ratio and
    # therefore reported a *negative* order (and extrapolated away from the limit)
    # for every monotonically converging metric.
    order = math.log(abs(denom / (f1 - f2))) / math.log(r)
    extrap = f1 + (f1 - f2) / (r ** order - 1.0)
    return {"levels": keys, "order": order, "richardson_limit": extrap,
            "finest": f1, "middle": f2, "coarse": f3}


def summarise(root: Path) -> dict:
    pattern = "raw_*.json" if any(root.glob("raw_*.json")) else "../fvm_*/raw_*.json"
    raws = sorted(root.glob(pattern))
    records = {}
    for path in raws:
        payload = json.loads(path.read_text())
        records[path.name] = payload
    out = {"raw_files": [p.name for p in raws], "families": {}}

    bfs = {}
    exp = {}
    nc = {}
    for rec in records.values():
        if rec["case"] == "backward_facing_step":
            bfs[int(rec["parameters"]["cells_per_step"])] = rec["metrics"]["reattachment_length_over_H"]
        elif rec["case"] == "sudden_expansion":
            exp[int(rec["parameters"]["cells_per_diameter"])] = rec["metrics"]["k_total_pressure"]
        elif rec["case"] == "natural_convection":
            nc[(float(rec["parameters"]["rayleigh"]), int(rec["parameters"]["cells"]))] = \
                rec["metrics"]["nusselt_hot"]
    out["families"]["bfs"] = {"reattachment_length_over_H": bfs,
                              "convergence": observed_order(bfs, "H")}
    out["families"]["expansion"] = {"k_total_pressure": exp,
                                    "convergence": observed_order(exp, "D")}
    out["families"]["convection"] = {}
    for ra in sorted({key[0] for key in nc}):
        vals = {n: v for (rar, n), v in nc.items() if rar == ra}
        out["families"]["convection"][f"Ra{ra:.0e}"] = {
            "nusselt_hot": {str(k): v for k, v in vals.items()},
            "convergence": observed_order(vals, "N"),
            "reference": DE_VAHL_DAVIS[ra]["nu"],
        }

    criteria = {}
    bfs = {k: v for k, v in bfs.items() if v is not None}
    exp = {k: v for k, v in exp.items() if v is not None}
    if bfs:
        best = min(bfs, key=lambda k: abs(k - 80))
        lr = bfs[best]
        criteria["bfs_reattachment_within_10pct"] = {
            "value": lr, "grid": f"H/{best}",
            "reference": [ARMALY_REATTACHMENT, GARTLING_REATTACHMENT],
            "error_vs_armaly": abs(lr - ARMALY_REATTACHMENT) / ARMALY_REATTACHMENT,
            "error_vs_gartling": abs(lr - GARTLING_REATTACHMENT) / GARTLING_REATTACHMENT,
            "threshold": 0.10,
        }
    if exp:
        best = max(exp)
        criteria["expansion_k_within_5pct"] = {
            "value": exp[best], "grid": f"D/{best}", "reference": BORDA_CARNOT_AR4,
            "error": abs(exp[best] - BORDA_CARNOT_AR4) / BORDA_CARNOT_AR4,
            "threshold": 0.05,
        }
        rec = [r for r in records.values() if r["case"] == "sudden_expansion"]
        criteria["expansion_k_static_within_5pct"] = {
            "value": [r["metrics"]["k_static_raw"] for r in rec],
            "reference": BORDA_CARNOT_AR4,
            "error": [r["metrics"]["k_relative_error_static"] for r in rec],
            "threshold": 0.05,
        }
    for ra, block in out["families"]["convection"].items():
        vals = block["nusselt_hot"]
        if not vals:
            continue
        best = max(int(k) for k in vals)
        value = vals[str(best)]
        criteria[f"convection_nu_{ra}_within_3pct"] = {
            "value": value, "grid": f"N{best}", "reference": block["reference"],
            "error": abs(value - block["reference"]) / block["reference"],
            "threshold": 0.03,
        }
    out["criteria"] = criteria
    orders = [out["families"][k]["convergence"]["order"]
              for k in ("bfs", "expansion")]
    orders += [b["convergence"]["order"] for b in out["families"]["convection"].values()
               if b["convergence"]["order"] is not None]
    orders = [o for o in orders if o is not None]
    out["observed_order_max"] = max(orders) if orders else None
    out["criteria"]["grid_convergence_order_ge_1.5"] = {
        "value": out["observed_order_max"], "threshold": 1.5,
    }
    for key, item in out["criteria"].items():
        value = item.get("value")
        if isinstance(value, (int, float)):
            item["pass"] = bool(item["error"] <= item["threshold"]) if "error" in item \
                else bool(value >= item["threshold"])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["bfs", "expansion", "convection", "summary"])
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--rayleigh", type=float, default=1.0e4)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--warm", type=str, default=None)
    args = parser.parse_args(argv)

    if args.case == "summary":
        target = args.root or (ROOT / "fvm_incompressible")
        summary = summarise(target)
        (target / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["criteria"], indent=2))
        return 0

    if args.case == "bfs":
        target = ROOT / "fvm_backward_step"
    elif args.case == "expansion":
        target = ROOT / "fvm_sudden_expansion"
    else:
        target = ROOT / "fvm_natural_convection"
    if args.root is not None:
        # ``--root`` selects the artifact directory for the *case* runs too, so a
        # new ladder (or a re-run with a different time step) can be archived
        # beside -- instead of on top of -- the committed reference artifacts.
        target = args.root
    target.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if args.case == "bfs":
        payload = run_bfs(args.resolution, target, max_steps=args.max_steps,
                          tolerance=args.tolerance, cfl=args.cfl, warm=args.warm)
    elif args.case == "expansion":
        payload = run_expansion(args.resolution, target, max_steps=args.max_steps,
                                tolerance=args.tolerance, cfl=args.cfl)
    else:
        payload = run_convection(args.rayleigh, args.resolution,
                                 max_steps=args.max_steps,
                                 tolerance=args.tolerance, cfl=args.cfl)
    payload["wall_seconds"] = time.time() - started
    payload["git_commit"] = subprocess_rev()
    fields = payload.pop("fields")
    np.savez_compressed(target / f"fields_{payload['case']}_{payload['tag']}.npz",
                        **fields)
    out = target / f"raw_{payload['case']}_{payload['tag']}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print(json.dumps(payload["metrics"], indent=2))
    return 0


def subprocess_rev() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())