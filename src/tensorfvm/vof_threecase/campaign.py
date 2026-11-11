"""Acceptance campaign for the three-case VOF verification face.

The module only assembles: each leg lives in its own module
(:mod:`.statics`, :mod:`.zalesak`, :mod:`.dambreak`), the reference data lives
in the repository's registered anchors, and the verdicts are computed from the
records with the thresholds written down here -- once, in one place, so a
report cannot pick a friendlier threshold than the campaign did.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time

import torch

from .dambreak import DamBreakMMConfig, run_dam_break
from .statics import StaticBubbleConfig, run_static_bubble
from .zalesak import NotchConfig, run_long_rotation

__all__ = [
    "CRITERIA",
    "dambreak_sweep",
    "static_bubble_sweep",
    "verdicts",
    "write_summary",
    "zalesak_sweep",
]

#: The node's machine-checkable thresholds (the report quotes these verbatim).
CRITERIA = {
    "a_front_front_error_10pct": {
        "description": "dam-break front position within 10 % of Martin & Moyce at "
                       "t = 0.1, 0.2, 0.3 s (production grid)",
        "threshold": 0.10,
    },
    "b_spurious_current": {
        "description": "static bubble max|u| / sqrt(sigma/(rho_l R)) <= 0.01 "
                       "(every balanced-CSF configuration, saturated runs)",
        "threshold": 0.01,
    },
    "c_notch_shape": {
        "description": "Zalesak notch-area error <= 5 % after 360 deg and "
                       "<= 15 % after 1000 deg (production scheme)",
        "threshold_360": 0.05,
        "threshold_1000": 0.15,
    },
    "d_mass_conservation": {
        "description": "global volume drift |M(t)-M(0)|/M(0) <= 0.1 % over every leg",
        "threshold": 1.0e-3,
    },
    "e_grid_convergence": {
        "description": "dam-break front error decreases monotonically over the "
                       "50x100 / 100x200 / 200x400 grid sequence",
    },
}

#: Grid sequence named by the node's criterion (ny x nx, square cells in a 2:1 tank).
DAM_GRID_SEQUENCE = ((50, 100), (100, 200), (200, 400))
#: The grid the front-error criterion is judged on.
DAM_PRODUCTION_GRID = (100, 200)
#: The Zalesak configuration judged by criterion (c).
ZALESAK_PRODUCTION = {"scheme": "cicsam", "cells": 128}


def _strip(record: dict) -> dict:
    """Drop the non-JSON payload of a record (fields and solver handles)."""
    return {k: v for k, v in record.items() if not k.startswith("_")}


def static_bubble_sweep(configs) -> list:
    out = []
    for cfg in configs:
        t0 = time.time()
        rec = run_static_bubble(cfg)
        out.append({**_strip(rec), "wall_seconds": time.time() - t0})
        print(f"[static] cells={rec['cells']} mu={rec['mu_liquid']:.1e} "
              f"La={rec['laplace_number']:.3g} {rec['csf_balance']} "
              f"{rec['csf_curvature']} ratio={rec['spurious_ratio']:.3e} "
              f"saturated={rec['saturated']} ({time.time() - t0:.0f}s)", flush=True)
    return out


def zalesak_sweep(configs) -> list:
    out = []
    for cfg in configs:
        t0 = time.time()
        rec = run_long_rotation(cfg)
        out.append(_strip(rec))
        print(f"[zalesak] {rec['scheme']} N={rec['cells']} {rec['degrees']:.0f}deg "
              f"notch={rec['notch_error_relative'] * 100:.2f}% "
              f"area={rec['area_error_contour_vs_initial'] * 100:.2f}% "
              f"drift={rec['mass_drift_relative']:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def dambreak_sweep(configs) -> list:
    out = []
    for cfg in configs:
        t0 = time.time()
        rec = run_dam_break(cfg)
        items = []
        for s in rec["samples"]:
            items.append({k: s[k] for k in
                          ("t", "t_star", "x_front", "x_star_sim",
                           "x_star_reference", "relative_error", "abs_relative_error")})
        out.append({**_strip(rec), "samples": items, "wall_seconds": time.time() - t0})
        head = " ".join(f"t={s['t']:.2f}:{s['abs_relative_error'] * 100:.1f}%"
                        for s in items)
        print(f"[dam] ny={rec['ny']} nx={rec['nx']} {head} "
              f"drift={rec['mass_drift_max']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _grid_key(rec) -> tuple:
    return (rec["ny"], rec["nx"])


def verdicts(static, zalesak, dam) -> dict:
    """Machine verdicts for criteria (a)-(e); every sub-verdict is kept."""
    v = {}

    # (a) front error on the production grid
    prod = [r for r in dam if _grid_key(r) == DAM_PRODUCTION_GRID]
    detail = []
    for r in prod:
        for s in r["samples"]:
            detail.append({"grid": f"{r['ny']}x{r['nx']}", "t": s["t"],
                           "abs_relative_error": s["abs_relative_error"],
                           "pass": s["abs_relative_error"] <= CRITERIA["a_front_front_error_10pct"]["threshold"]})
    v["a_front_front_error_10pct"] = {
        "pass": bool(detail) and all(d["pass"] for d in detail),
        "detail": detail,
        "threshold": CRITERIA["a_front_front_error_10pct"]["threshold"],
        "production_grid": f"{DAM_PRODUCTION_GRID[0]}x{DAM_PRODUCTION_GRID[1]}",
        "per_grid": [
            {"grid": f"{r['ny']}x{r['nx']}",
             "max_abs_relative_error": max((s["abs_relative_error"] for s in r["samples"]),
                                           default=float("nan")),
             "pass": bool(r["samples"]) and all(
                 s["abs_relative_error"] <= CRITERIA["a_front_front_error_10pct"]["threshold"]
                 for s in r["samples"])}
            for r in dam
        ],
    }

    # (b) spurious current
    balanced = [r for r in static if r["csf_balance"] == "balanced"]
    legacy = [r for r in static if r["csf_balance"] == "legacy"]
    v["b_spurious_current"] = {
        "pass": bool(balanced) and all(
            r["spurious_ratio"] <= CRITERIA["b_spurious_current"]["threshold"] and r["saturated"]
            for r in balanced),
        "threshold": CRITERIA["b_spurious_current"]["threshold"],
        "worst_ratio": max((r["spurious_ratio"] for r in balanced), default=float("nan")),
        "all_saturated": all(r["saturated"] for r in balanced),
        "detail": [{"cells": r["cells"], "mu_liquid": r["mu_liquid"],
                    "laplace_number": r["laplace_number"],
                    "csf_curvature": r["csf_curvature"],
                    "spurious_ratio": r["spurious_ratio"],
                    "saturated": r["saturated"],
                    "laplace_jump_error": r["jump_error_relative"]} for r in static],
        "legacy_control_worst_ratio": max((r["spurious_ratio"] for r in legacy),
                                          default=float("nan")),
    }

    # (c) notch shape
    prod_z = [r for r in zalesak
              if r["scheme"] == ZALESAK_PRODUCTION["scheme"]
              and r["cells"] == ZALESAK_PRODUCTION["cells"]]
    d360 = [r for r in prod_z if abs(r["degrees"] - 360.0) < 1e-9]
    d1000 = [r for r in prod_z if abs(r["degrees"] - 1000.0) < 1e-9]
    ok360 = bool(d360) and d360[0]["notch_error_relative"] <= CRITERIA["c_notch_shape"]["threshold_360"]
    ok1000 = bool(d1000) and d1000[0]["notch_error_relative"] <= CRITERIA["c_notch_shape"]["threshold_1000"]
    v["c_notch_shape"] = {
        "pass": bool(ok360 and ok1000),
        "threshold_360": CRITERIA["c_notch_shape"]["threshold_360"],
        "threshold_1000": CRITERIA["c_notch_shape"]["threshold_1000"],
        "production": dict(ZALESAK_PRODUCTION),
        "error_360": d360[0]["notch_error_relative"] if d360 else None,
        "error_1000": d1000[0]["notch_error_relative"] if d1000 else None,
        "detail": [{"scheme": r["scheme"], "cells": r["cells"], "degrees": r["degrees"],
                    "notch_error_relative": r["notch_error_relative"],
                    "area_error_contour_vs_initial": r["area_error_contour_vs_initial"],
                    "perimeter_error_analytic": r["perimeter_error_analytic"],
                    "peak_retention": r["peak_retention"]} for r in zalesak],
    }

    # (d) mass conservation over every leg
    drifts = ([r["mass_drift_relative"] for r in zalesak]
              + [r["volume_drift"] for r in static]
              + [r["mass_drift_max"] for r in dam])
    v["d_mass_conservation"] = {
        "pass": bool(drifts) and max(drifts) <= CRITERIA["d_mass_conservation"]["threshold"],
        "threshold": CRITERIA["d_mass_conservation"]["threshold"],
        "worst_relative_drift": max(drifts) if drifts else float("nan"),
    }

    # (e) monotone grid convergence of the front error
    seq = []
    for grid in DAM_GRID_SEQUENCE:
        rr = [r for r in dam if _grid_key(r) == grid]
        if rr:
            seq.append({"grid": f"{grid[0]}x{grid[1]}",
                        "max_abs_relative_error": max(
                            (s["abs_relative_error"] for s in rr[0]["samples"]),
                            default=float("nan")),
                        "mean_abs_relative_error": sum(
                            s["abs_relative_error"] for s in rr[0]["samples"])
                        / max(len(rr[0]["samples"]), 1)})
    mono = len(seq) >= 2 and all(seq[i + 1]["mean_abs_relative_error"]
                                 <= seq[i]["mean_abs_relative_error"] + 1e-12
                                 for i in range(len(seq) - 1))
    v["e_grid_convergence"] = {
        "pass": bool(len(seq) == len(DAM_GRID_SEQUENCE) and mono),
        "sequence": seq,
        "monotone": mono,
        "complete": len(seq) == len(DAM_GRID_SEQUENCE),
    }
    return v


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "cwd": "TensorFVM",
    }


def write_summary(path: str, payload: dict) -> dict:
    """Attach the verdicts to the payload and dump ``path`` (JSON, sorted keys)."""
    static = payload.get("static_bubble", [])
    zalesak = payload.get("zalesak", [])
    dam = payload.get("dam_break", [])
    payload["criteria"] = verdicts(static, zalesak, dam)
    payload["criteria_definitions"] = CRITERIA
    payload["environment"] = environment()
    payload["all_pass"] = all(c.get("pass", False) for c in payload["criteria"].values())
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    return payload


def default_configs() -> dict:
    """The campaign's production matrix (kept here so the driver stays thin)."""
    static = []
    for cells in (32, 64, 96, 128):
        static.append(StaticBubbleConfig(cells=cells, mu_liquid=1.0e-2))
    # Laplace-number sweep at fixed resolution (viscous damping sets the
    # saturation time, so each point runs for its own 6 tau_mu)
    for mu in (1.0e-3, 3.0e-2, 1.0e-1, 1.0):
        static.append(StaticBubbleConfig(cells=64, mu_liquid=mu, mu_gas=mu))
    # negative control: cell-centred (legacy) CSF with the same metric
    static.append(StaticBubbleConfig(cells=64, mu_liquid=1.0e-2, csf_balance="legacy"))
    zalesak = []
    for cells in (64, 128, 256):
        for deg in (360.0, 1000.0):
            zalesak.append(NotchConfig(cells=cells, scheme="cicsam", degrees=deg))
    for cells in (64, 128):
        zalesak.append(NotchConfig(cells=cells, scheme="plic", degrees=1000.0))
        zalesak.append(NotchConfig(cells=cells, scheme="upwind", degrees=360.0))
    dam = []
    for ny, nx in DAM_GRID_SEQUENCE:
        dam.append(DamBreakMMConfig(ny=ny, nx=nx, aspect=2.0,
                                    column_width=0.05715, length=0.4, height=0.2))
    return {"static_bubble": static, "zalesak": zalesak, "dam_break": dam}