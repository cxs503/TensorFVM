"""CASE-CAVITY-RE100: lid-driven square cavity at Re = 100 against Ghia et al. (1982).

What this module is
-------------------
One *case*, self-contained and machine-checkable, instead of "a solver plus a
ladder somewhere": it runs :class:`tensorfvm.lid_cavity.LidCavitySolver` at a
requested resolution, scores the two mid-plane profiles point by point against
the transcribed Ghia, Ghia & Shin (1982) tables, evaluates an explicit list of
acceptance criteria, and returns a JSON-serialisable artefact.  The reference
numbers are never retyped here - they come from
:func:`tensorfvm.lid_cavity_reference.table`, the single transcription that the
reference audit tests pin.

Case definition (Ghia et al. 1982, Sec. 2; the benchmark the paper tabulates)

    domain      [0, L] x [0, L], L = 1
    walls       no-slip on x = 0, x = L and y = 0
    lid         u = U = 1 on y = L (tangential, no penetration)
    Reynolds    Re = U L / nu = 100  ->  nu = 0.01 (rho = 1)
    outputs     u(y) on x = L/2          -> Table I
                v(x) on y = L/2          -> Table II
                primary vortex centre    -> (0.617, 0.734) L

Acceptance (case-pool entry ``CAVITY-RE100``, tolerance < 5 %)

The pool states only "ref: Ghia (1982) tables, tolerance < 5 %"; this module
turns that into seven explicit, individually reported checks so a partial pass
cannot hide behind an average (see :data:`CRITERIA`).  Every check is computed
from the artefact, so the gate in ``tests/test_case_cavity_re100.py`` recomputes
the same numbers from the stored profiles instead of trusting a stored verdict.

Usage
-----
    PYTHONPATH=src python scripts/case_cavity_re100.py --resolutions 32 64 128
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import torch

from .lid_cavity import LidCavityConfig, LidCavitySolver, score_against_ghia
from .lid_cavity_reference import VORTEX_CENTRE, table

CASE_SLUG = "CASE-CAVITY-RE100"
REYNOLDS = 100.0
LID_VELOCITY = 1.0
REFERENCE = ("U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, "
             "Table I (u at x = 0.5) and Table II (v at y = 0.5)")
CASE_POOL_TOLERANCE = 0.05  # "tolerance: < 5 %" of the CAVITY-RE100 pool entry

#: The case-pool tolerance turned into named, individually reported checks.
#: ``value`` is filled in by :func:`evaluate`; ``comparator`` is one of
#: ``"le"`` (value <= threshold) / ``"ge"`` (value >= threshold).
CRITERIA: tuple[dict[str, object], ...] = (
    {"id": "C1_u_profile_max_abs",
     "statement": "max |u_computed - u_Ghia| over all Table I points, in units of U",
     "comparator": "le", "threshold": CASE_POOL_TOLERANCE, "unit": "U"},
    {"id": "C2_v_profile_max_abs",
     "statement": "max |v_computed - v_Ghia| over all Table II points, in units of U",
     "comparator": "le", "threshold": CASE_POOL_TOLERANCE, "unit": "U"},
    {"id": "C3_u_min_relative",
     "statement": "relative error of the deepest reverse-flow u (Ghia -0.21090 at y = 0.4531)",
     "comparator": "le", "threshold": CASE_POOL_TOLERANCE, "unit": "1"},
    {"id": "C4_v_max_relative",
     "statement": "relative error of the strongest v (Ghia 0.17527 at x = 0.2344)",
     "comparator": "le", "threshold": CASE_POOL_TOLERANCE, "unit": "1"},
    {"id": "C5_vortex_centre",
     "statement": "primary-vortex centre displacement from (0.617, 0.734) L",
     "comparator": "le", "threshold": 0.05, "unit": "L"},
    {"id": "C6_divergence_free",
     "statement": "max |div u| of the converged discrete field (closed-box compatibility)",
     "comparator": "le", "threshold": 1e-10, "unit": "U/L"},
    {"id": "C7_steady_state",
     "statement": "steady-state residual max|du/dt|, |dv/dt| of the marching scheme",
     "comparator": "le", "threshold": 1e-06, "unit": "U^2/L"},
)


def _as_list(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.reshape(-1)]


def block_average(field: torch.Tensor, target: int) -> torch.Tensor:
    """Average an ``n x n`` cell field down to ``target x target`` (``n % target == 0``)."""
    n = int(field.shape[0])
    if n == target:
        return field.clone()
    if n % target:
        raise ValueError(f"cannot block-average {n} onto {target}")
    k = n // target
    return field.reshape(target, k, target, k).mean(dim=(1, 3))


def run_resolution(n: int, *, cfl: float = 0.15, tolerance: float = 1e-07,
                   max_steps: int = 200000, field_grid: int = 32,
                   reynolds: float = REYNOLDS, lid_velocity: float = LID_VELOCITY) -> dict:
    """March one resolution to steady state and return its scored artefact."""
    field_grid = min(int(field_grid), int(n))  # never upsample a coarse run
    cfg = LidCavityConfig(n=n, reynolds=reynolds, lid_velocity=lid_velocity,
                          cfl=cfl, tolerance=tolerance, max_steps=max_steps)
    solver = LidCavitySolver(cfg)
    t0 = time.time()
    ledger = solver.run()
    wall = time.time() - t0
    scored = score_against_ghia(solver, reynolds)
    u_c, v_c = solver.cell_velocity()
    psi = solver.streamfunction()
    vx, vy = solver.vortex_centre()

    artifact: dict = {
        "case": CASE_SLUG,
        "reference": REFERENCE,
        "reynolds": float(reynolds),
        "lid_velocity": float(lid_velocity),
        "n": int(n),
        "h": float(solver.h),
        "viscosity": float(solver.nu),
        "solver": {
            "class": "tensorfvm.lid_cavity.LidCavitySolver",
            "discretisation": "staggered MAC FVM, conservative van Leer flux, "
                              "explicit projection, DCT-II Neumann Poisson",
            "cfl": float(cfl),
            "tolerance": float(tolerance),
            "max_steps": int(max_steps),
            "steps": int(ledger["steps"]),
            "converged": bool(ledger["converged"]),
            "residual": float(ledger["residual"]),
            "dt_final": float(ledger["dt"]),
            "max_divergence": float(ledger["max_divergence"]),
            "wall_seconds": round(wall, 2),
        },
        "profiles": {
            "u": {k: (v if not isinstance(v, torch.Tensor) else _as_list(v))
                  for k, v in scored["u"]["profiles"].items()},
            "v": {k: (v if not isinstance(v, torch.Tensor) else _as_list(v))
                  for k, v in scored["v"]["profiles"].items()},
        },
        "errors": {
            "u_max_abs": float(scored["u"]["max_abs_error"]),
            "u_rms": float(scored["u"]["rms_error"]),
            "v_max_abs": float(scored["v"]["max_abs_error"]),
            "v_rms": float(scored["v"]["rms_error"]),
            "u_max_abs_consistent": float(scored["u"].get("max_abs_error_consistent", float("nan"))),
            "v_max_abs_consistent": float(scored["v"].get("max_abs_error_consistent", float("nan"))),
        },
        "extrema": {"u": scored["u"]["extremum"], "v": scored["v"]["extremum"]},
        "vortex_centre": {"computed": [float(vx), float(vy)],
                          "reference": [float(c) for c in VORTEX_CENTRE[float(reynolds)]]},
        "psi_min": float(psi.min()),
        "field_grid": int(field_grid),
        "field": {
            "u": [[float(x) for x in row] for row in block_average(u_c, field_grid).tolist()],
            "v": [[float(x) for x in row] for row in block_average(v_c, field_grid).tolist()],
            "psi": [[float(x) for x in row] for row in block_average(psi, field_grid).tolist()],
        },
        "reference_points": {"u": len(table(reynolds, "u")), "v": len(table(reynolds, "v"))},
    }
    artifact["criteria"] = evaluate(artifact)
    checks = {k: v for k, v in artifact["criteria"].items() if not k.startswith("_")}
    artifact["verdict"] = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    return artifact


def _profile_error(block: dict) -> tuple[float, float, float, float]:
    """Return ``(max_abs, rel_of_extremum, extremum_reference, extremum_computed)``."""
    delta = [abs(float(d)) for d in block["delta"]]
    ref = [float(r) for r in block["reference"]]
    got = [float(g) for g in block["computed"]]
    jmin = min(range(len(ref)), key=lambda i: ref[i])
    jmax = max(range(len(ref)), key=lambda i: ref[i])
    j = jmin if abs(ref[jmin]) >= abs(ref[jmax]) else jmax
    rel = abs(got[j] - ref[j]) / abs(ref[j]) if ref[j] else float("nan")
    return max(delta), rel, ref[j], got[j]


def evaluate(artifact: dict) -> dict:
    """Evaluate :data:`CRITERIA` on an artefact (recomputed from stored profiles)."""
    u_max, u_rel, u_ref, u_got = _profile_error(artifact["profiles"]["u"])
    v_max, v_rel, v_ref, v_got = _profile_error(artifact["profiles"]["v"])
    vx, vy = artifact["vortex_centre"]["computed"]
    rx, ry = artifact["vortex_centre"]["reference"]

    values = {
        "C1_u_profile_max_abs": u_max,
        "C2_v_profile_max_abs": v_max,
        "C3_u_min_relative": u_rel,
        "C4_v_max_relative": v_rel,
        "C5_vortex_centre": max(abs(vx - rx), abs(vy - ry)),
        "C6_divergence_free": float(artifact["solver"]["max_divergence"]),
        "C7_steady_state": float(artifact["solver"]["residual"]),
    }
    out: dict = {}
    for spec in CRITERIA:
        cid = str(spec["id"])
        value = values[cid]
        if spec["comparator"] == "le":
            ok = bool(value <= float(spec["threshold"]))
        else:
            ok = bool(value >= float(spec["threshold"]))
        out[cid] = {
            "statement": spec["statement"],
            "value": float(value),
            "threshold": float(spec["threshold"]),
            "comparator": spec["comparator"],
            "unit": spec["unit"],
            "pass": ok,
            "margin": float(spec["threshold"]) - float(value),
        }
    out["_extremum_basis"] = {"u": {"reference": u_ref, "computed": u_got},
                              "v": {"reference": v_ref, "computed": v_got}}
    return out


def field_difference(a: dict, b: dict) -> dict:
    """``L2``/``Linf`` difference of two artefacts' fields on the common coarse grid."""
    ua = torch.tensor(a["field"]["u"], dtype=torch.float64)
    va = torch.tensor(a["field"]["v"], dtype=torch.float64)
    ub = torch.tensor(b["field"]["u"], dtype=torch.float64)
    vb = torch.tensor(b["field"]["v"], dtype=torch.float64)
    if ua.shape != ub.shape:
        raise ValueError(f"field grids differ: {tuple(ua.shape)} vs {tuple(ub.shape)}")
    du, dv = (ua - ub), (va - vb)
    return {
        "l2_u": float(torch.sqrt((du ** 2).mean())),
        "linf_u": float(du.abs().max()),
        "l2_v": float(torch.sqrt((dv ** 2).mean())),
        "linf_v": float(dv.abs().max()),
        "l2_speed": float(torch.sqrt((du ** 2 + dv ** 2).mean())),
        "linf_speed": float(torch.sqrt(du ** 2 + dv ** 2).max()),
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Observed order ``p`` from two successive differences (``ratio`` = refinement)."""
    if coarse <= 0 or fine <= 0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def convergence(artifacts: list[dict]) -> dict:
    """Self-convergence of the sequence of artefacts (all on the same coarse grid)."""
    rows = sorted(artifacts, key=lambda a: a["n"])
    diffs = []
    for lo, hi in zip(rows, rows[1:]):
        d = field_difference(lo, hi)
        d["coarse_n"], d["fine_n"] = lo["n"], hi["n"]
        d["ratio"] = hi["n"] / lo["n"]
        diffs.append(d)
    orders = []
    for d0, d1 in zip(diffs, diffs[1:]):
        orders.append({
            "between": [d0["coarse_n"], d0["fine_n"], d1["fine_n"]],
            "p_l2_u": observed_order(d0["l2_u"], d1["l2_u"]),
            "p_linf_u": observed_order(d0["linf_u"], d1["linf_u"]),
            "p_l2_v": observed_order(d0["l2_v"], d1["l2_v"]),
            "p_l2_speed": observed_order(d0["l2_speed"], d1["l2_speed"]),
        })
    return {"differences": diffs, "orders": orders}


def summarise(artifacts: list[dict]) -> dict:
    """Aggregate per-resolution artefacts into the case-level verdict document."""
    rows = sorted(artifacts, key=lambda a: a["n"])
    table_rows = []
    for a in rows:
        table_rows.append({
            "n": a["n"],
            "h": a["h"],
            "steps": a["solver"]["steps"],
            "residual": a["solver"]["residual"],
            "max_divergence": a["solver"]["max_divergence"],
            "wall_seconds": a["solver"]["wall_seconds"],
            "u_max_abs": a["errors"]["u_max_abs"],
            "u_rms": a["errors"]["u_rms"],
            "v_max_abs": a["errors"]["v_max_abs"],
            "v_rms": a["errors"]["v_rms"],
            "u_min": a["criteria"]["_extremum_basis"]["u"]["computed"],
            "v_max": a["criteria"]["_extremum_basis"]["v"]["computed"],
            "vortex_centre": a["vortex_centre"]["computed"],
            "psi_min": a["psi_min"],
            "verdict": a["verdict"],
            "criteria": {k: v for k, v in a["criteria"].items() if not k.startswith("_")},
        })
    reference = {
        "u_min": [0.4531, -0.21090],
        "v_max": [0.2344, 0.17527],
        "vortex_centre": [float(c) for c in VORTEX_CENTRE[REYNOLDS]],
    }
    best = rows[-1]
    return {
        "case": CASE_SLUG,
        "round": 2,
        "reference": REFERENCE,
        "reynolds": REYNOLDS,
        "lid_velocity": LID_VELOCITY,
        "pool_tolerance": CASE_POOL_TOLERANCE,
        "reference_summary": reference,
        "resolutions": [a["n"] for a in rows],
        "rows": table_rows,
        "convergence": convergence(rows),
        "finest": rows[-1]["n"],
        "verdict": "PASS" if all(r["verdict"] == "PASS" for r in table_rows) else "FAIL",
        "finest_verdict": best["verdict"],
        "gate_criteria": [{k: v for k, v in spec.items()} for spec in CRITERIA],
    }


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")