"""Campaign driver: the two-phase VOF verification surface of TensorFVM.

Runs three exact-advection anchors (Zalesak slotted disk, Rider-Kothe single
vortex, two-cell double vortex) on three systematically refined grids and the
hydrostatic rest-state anchor, then evaluates the acceptance criteria.

The campaign is *sharded*: :func:`run_spec` executes one (case, scheme, grid)
combination and returns a plain JSON-able dict, so a cluster job can run a
subset (``--index i --count n``) and the results can be merged afterwards
(:func:`merge_runs`).  This keeps the compute reproducible and cheap to re-run.

Schemes under test
------------------
``plic``
    :func:`tensorfvm.vof_plic.advect_plic` -- geometric PLIC reconstruction with
    conservative bound repair.  Primary scheme.
``fct``
    :class:`tensorfvm.vof_phase0.advection.AlgebraicVOF` -- flux-corrected
    algebraic transport actually used by the Phase-0 two-phase solver.
``donor``
    First-order donor cell -- the monotone lower bound that any high-resolution
    scheme has to beat.
"""

from __future__ import annotations

import json
import os


from .cases import (
    advance,
    double_vortex_case,
    single_vortex_case,
    zalesak_case,
    Transport,
)
from .hydrostatic import run_hydrostatic
from .metrics import triple_summary

__all__ = [
    "SPECS",
    "run_spec",
    "run_hydrostatic_sweep",
    "merge_runs",
    "evaluate_criteria",
]

#: grid ladders per case (cell counts; the double vortex box is 2:1)
GRIDS = {
    "zalesak": [(64, 64), (128, 128), (256, 256)],
    # the vortex anchors need many more steps per cell (T = 8, deformation
    # reversal), so their ladder stops at 128^2 -- documented in the report
    "single_vortex": [(32, 32), (64, 64), (128, 128)],
    "double_vortex": [(32, 16), (64, 32), (128, 64)],
}
SCHEMES = ("plic", "fct", "donor")
CFL = 0.75


def _case(name: str):
    if name == "zalesak":
        return zalesak_case()
    if name == "single_vortex":
        return single_vortex_case()
    if name == "double_vortex":
        return double_vortex_case()
    raise ValueError(f"unknown case {name!r}")


def build_specs() -> list[dict]:
    specs = []
    for name in ("zalesak", "single_vortex", "double_vortex"):
        for scheme in SCHEMES:
            for nx, ny in GRIDS[name]:
                specs.append({"case": name, "scheme": scheme,
                              "cells": [nx, ny], "cfl": CFL})
    return specs


SPECS = build_specs()


def run_spec(spec: dict, keep_field: bool = False, case_override=None) -> dict:
    """Run one (case, scheme, grid) combination."""
    case = case_override if case_override is not None else _case(spec["case"])
    nx, ny = spec["cells"]
    transport = Transport(scheme=spec["scheme"], cfl=spec.get("cfl", CFL))
    result = advance(case, nx, ny, transport)
    out = dict(spec)
    out.update(result["metrics"])
    out["case"] = spec["case"]
    out["scheme"] = spec["scheme"]
    out["note"] = case.note
    if keep_field:
        out["alpha"] = result["alpha"]
        out["exact"] = result["exact"]
    return out


def run_hydrostatic_sweep(grids=(32, 64, 128), steps: int = 40,
                          depths=(0.25, 0.5, 0.75), density_ratios=(1.0, 100.0,
                                                                  1000.0),
                          unaligned_depths=(0.30,)) -> dict:
    """Hydrostatic rest-state sweep: control (shipped) vs well-balanced."""
    out = {"grids": list(grids), "steps": steps, "depths": list(depths),
           "density_ratios": list(density_ratios),
           "cases": []}
    for n in grids:
        for depth in depths:
            for ratio in density_ratios:
                row = {"cells": [n, n], "depth": depth, "density_ratio": ratio}
                for label, wb in (("shipped", False), ("well_balanced", True)):
                    r = run_hydrostatic(n, n, steps=steps, surface_model="void",
                                        depth=depth, rho_gas=1000.0 / ratio,
                                        well_balanced=wb)
                    row[label] = {k: r[k] for k in (
                        "pressure_oscillation", "spurious_velocity",
                        "mass_drift_relative", "dt", "cg")}
                out["cases"].append(row)
    return out


def run_deformation_sweep(durations=(1.0, 2.0, 4.0), schemes=("plic",),
                          grids=((64, 64), (128, 128)),
                          dv_grids=((64, 32), (128, 64))) -> dict:
    """Interface error vs deformation window (attribution of the tearing error).

    The standard anchors use ``T = 8``, which winds the disk into a filament far
    thinner than a cell, so their error is dominated by lost sub-grid
    information rather than by the reconstruction quality.  Sweeping the
    deformation window ``T`` (same field, ``g(t) = cos(pi t/T)`` still reverses
    exactly) shows how the error grows with the accumulated winding; the
    criterion is attainable while the interface stays resolvable.
    """
    out = {"durations": list(durations), "rows": []}
    for T in durations:
        for name, case, gridset in (
                ("single_vortex", single_vortex_case(T=T), grids),
                ("double_vortex", double_vortex_case(T=T), dv_grids)):
            for scheme in schemes:
                for nx, ny in gridset:
                    row = run_spec({"case": name, "scheme": scheme,
                                    "cells": [nx, ny], "cfl": CFL, "T": T},
                                   case_override=case)
                    out["rows"].append(row)
                    print(f"T={T:4.1f} {name:14s} {scheme:6s} {nx}x{ny} "
                          f"sym={row['symmetric_difference']:.4f} "
                          f"relL1={row['relative_l1']:.4f}", flush=True)
    return out


def _levels(runs: list[dict], case: str, scheme: str) -> list[dict]:
    rows = [r for r in runs if r["case"] == case and r["scheme"] == scheme]
    return sorted(rows, key=lambda r: r["cells"][0])


def merge_runs(runs: list[dict], hydrostatic: dict | None = None) -> dict:
    """Assemble the per-case convergence tables and the criteria verdicts."""
    cases = {}
    for name in ("zalesak", "single_vortex", "double_vortex"):
        entry = {"schemes": {}}
        for scheme in SCHEMES:
            levels = _levels(runs, name, scheme)
            if not levels:
                continue
            gci = {}
            for key in ("relative_l1", "symmetric_difference"):
                gci[key] = (triple_summary(levels, key)
                            if len(levels) >= 3 else None)
            entry["schemes"][scheme] = {
                "levels": levels,
                "gci": gci,
                "finest": levels[-1] if levels else None,
            }
        cases[name] = entry
    report = {
        "campaign": "vof-two-phase-verification",
        "parameters": {"grids": GRIDS, "schemes": list(SCHEMES), "cfl": CFL},
        "cases": cases,
        "hydrostatic": hydrostatic,
        "criteria": evaluate_criteria(cases, hydrostatic),
    }
    return report


def evaluate_criteria(cases: dict, hydrostatic: dict | None) -> dict:
    """Machine-checkable acceptance criteria C1-C6 (C7 is a pytest gate)."""
    out = {}

    def finest(name: str, scheme: str = "plic"):
        entry = cases.get(name, {}).get("schemes", {}).get(scheme)
        return entry["finest"] if entry else None

    z = finest("zalesak")
    out["C1_zalesak_area_conservation"] = {
        "value": z["mass_drift_relative"] if z else None,
        "threshold": 0.01,
        "passed": bool(z and z["mass_drift_relative"] < 0.01),
        "detail": "relative liquid-area drift after one revolution",
    }
    for key, name in (("C2_single_vortex", "single_vortex"),
                      ("C3_double_vortex", "double_vortex")):
        f = finest(name)
        out[key] = {
            "value": f["symmetric_difference"] if f else None,
            "relative_l1": f["relative_l1"] if f else None,
            "threshold": 0.05,
            "passed": bool(f and f["symmetric_difference"] < 0.05),
            "detail": "misplaced-area fraction of the reconstructed interface "
                      "vs the analytic initial field at t = T",
        }
    if hydrostatic and hydrostatic.get("cases"):
        worst_wb = max(c["well_balanced"]["pressure_oscillation"]
                       for c in hydrostatic["cases"])
        worst_ship = max(c["shipped"]["pressure_oscillation"]
                         for c in hydrostatic["cases"])
        out["C4_hydrostatic"] = {
            "value": worst_wb,
            "shipped_control": worst_ship,
            "threshold": 0.005,
            "passed": bool(worst_wb < 0.005),
            "detail": "worst pressure oscillation over the grid/depth/density "
                      "sweep, well-balanced vs shipped control",
        }
    else:
        out["C4_hydrostatic"] = {"value": None, "threshold": 0.005,
                                 "passed": False, "detail": "not run"}
    worst_mass = 0.0
    for name in ("zalesak", "single_vortex", "double_vortex"):
        for scheme in SCHEMES:
            entry = cases.get(name, {}).get("schemes", {}).get(scheme)
            if not entry:
                continue
            worst_mass = max(worst_mass, max(
                lv["mass_drift_relative"] for lv in entry["levels"]))
    out["C5_mass_conservation"] = {
        "value": worst_mass,
        "threshold": 0.01,
        "passed": bool(worst_mass < 0.01),
        "detail": "worst global volume drift over every benchmark and scheme",
    }
    orders = {}
    l1_orders = {}
    for name in ("zalesak", "single_vortex", "double_vortex"):
        entry = cases.get(name, {}).get("schemes", {}).get("plic")
        if not entry or not entry.get("gci"):
            continue
        if entry["gci"].get("symmetric_difference"):
            orders[name] = entry["gci"]["symmetric_difference"]
        if entry["gci"].get("relative_l1"):
            l1_orders[name] = entry["gci"]["relative_l1"]
    p_values = {k: v.get("observed_order") for k, v in l1_orders.items()
                if v.get("observed_order") is not None}
    out["C6_grid_convergence"] = {
        "value": min(p_values.values()) if p_values else None,
        "observed_orders": p_values,
        "observed_orders_symmetric_difference": {
            k: v.get("observed_order") for k, v in orders.items()},
        "gci_fine": {k: v.get("gci_fine") for k, v in l1_orders.items()},
        "gci_fine_symmetric_difference": {
            k: v.get("gci_fine") for k, v in orders.items()},
        "threshold": 1.0,
        "passed": bool(p_values) and all(p >= 1.0 for p in p_values.values()),
        "detail": "GCI observed order of the PLIC interface error against the "
                  "exact solution on three grids; criterion judged on the "
                  "relative-L1 norm (Rider-Kothe), the symmetric-difference "
                  "order is reported alongside",
    }
    return out


# --------------------------------------------------------------------------- #
# sharded execution
# --------------------------------------------------------------------------- #
def run_indices(indices: list[int], out_dir: str, keep_field: bool = False) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for idx in indices:
        spec = SPECS[idx]
        row = run_spec(spec, keep_field=keep_field)
        path = os.path.join(out_dir, f"run_{idx:03d}.json")
        with open(path, "w") as fh:
            json.dump(row, fh, indent=1, sort_keys=True)
        written.append(path)
        print(f"[{idx:03d}] {spec['case']:14s} {spec['scheme']:6s} "
              f"{spec['cells'][0]}x{spec['cells'][1]} "
              f"sym={row['symmetric_difference']:.4f} "
              f"relL1={row['relative_l1']:.4f} "
              f"drift={row['mass_drift_relative']:.1e} steps={row['steps']}",
              flush=True)
    return written


def load_runs(out_dir: str) -> list[dict]:
    runs = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("run_") and name.endswith(".json"):
            with open(os.path.join(out_dir, name)) as fh:
                runs.append(json.load(fh))
    return runs