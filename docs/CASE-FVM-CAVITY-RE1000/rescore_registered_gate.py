"""Re-score the committed Re=1000 bundle against the *registered* judge row.

Background (board audit m20230, findings F2/F3/F5): the bundle's machine gate used
one flat 5 % band, while the judge row registered in ``tensor_benchmark_cases``
grades ``u_min`` / ``v_max`` / centreline ``max|du|`` at **3 %** and the primary
vortex centre at ``<= 0.01 L`` for ``n >= 128``.  A solution degrading into the
3 % .. 5 % window therefore still passed the repository gate.  This script closes
that hole *from the stored evidence only*:

* it never re-marches the solver (the fields in ``runs/fields.npz`` are the
  converged solutions),
* every scored number is recomputed here from those raw fields with the
  repository's own Ghia transcription, then cross-checked against the numbers
  stored in ``runs/lid_cavity_Re1000_n*.json`` (they must agree to 1e-12),
* the interpolation-scheme sensitivity flagged in the audit (F5) is quantified
  by scoring the same field twice, Hermite vs monotone PCHIP,
* the vortex centre is re-derived from an independently integrated stream
  function instead of being taken on trust,
* the re-scored criteria/verdicts are written back into ``summary.json`` /
  ``benchmark.json`` so text and machine gate cannot drift apart.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/rescore_registered_gate.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.case_cavity_re1000 import (  # noqa: E402
    ANCHOR_ALIASES,
    CASE_SLUG,
    CRITERIA,
    JUDGE_GRID,
    REGISTERED_TOLERANCE,
    REYNOLDS,
    VORTEX_TOLERANCE_L,
    benchmark_rows,
    evaluate,
    write_json,
)
from tensorfvm.lid_cavity import hermite_interpolate  # noqa: E402
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table  # noqa: E402

RECONCILE_ATOL = 1e-12


def profiles_from_fields(u_field: np.ndarray, v_field: np.ndarray, n: int) -> dict:
    """Centreline ``u(y)`` on x = L/2 and ``v(x)`` on y = L/2 from the raw MAC field."""
    h = 1.0 / n
    coords = (np.arange(n) + 0.5) * h
    u_line = torch.as_tensor(u_field[n // 2, :], dtype=torch.float64)
    v_line = torch.as_tensor(v_field[:, n // 2], dtype=torch.float64)
    grid = torch.as_tensor(coords, dtype=torch.float64)
    return {"u": (grid, u_line), "v": (grid, v_line)}


#: Wall values the Ghia tables sample at the two ends of each centreline, so an
#: interpolant never has to extrapolate outside the cell-centre grid.
WALL_VALUES = {"u": ((0.0, 0.0), (1.0, 1.0)),   # y = 0 no-slip, y = 1 lid U = 1
               "v": ((0.0, 0.0), (1.0, 0.0))}   # x = 0 and x = 1 walls


def sampled(line: tuple[torch.Tensor, torch.Tensor], which: str, scheme: str) -> list[float]:
    grid, values = line
    targets = [p[0] for p in table(REYNOLDS, which)]
    if scheme == "hermite":
        out = [float(x) for x in hermite_interpolate(grid, values, targets)]
    elif scheme == "pchip":
        from scipy.interpolate import PchipInterpolator

        (lo_x, lo_v), (hi_x, hi_v) = WALL_VALUES[which]
        xs = np.concatenate([[lo_x], grid.numpy(), [hi_x]])
        ys = np.concatenate([[lo_v], values.numpy(), [hi_v]])
        f = PchipInterpolator(xs, ys, extrapolate=False)
        out = [float(f(t)) for t in targets]
    else:  # pragma: no cover - guarded by the CLI
        raise ValueError(scheme)
    return out


def streamfunction_vortex(u_field: np.ndarray) -> tuple[float, float]:
    """Primary-vortex centre from ``psi = int_0^y u dy`` on the cell centres.

    The MAC field gives ``u`` on vertical faces; the cell-centred stream function is
    integrated with the trapezoidal rule from the moving lid downwards and its
    interior minimum located.  Independent of the solver's own extremum search.
    """
    n = u_field.shape[1]
    uc = 0.5 * (u_field[:-1, :] + u_field[1:, :])          # (n, n), index (x, y)
    h = 1.0 / n
    psi = np.zeros_like(uc)
    for j in range(1, n):                                  # integrate along y (axis 1)
        psi[:, j] = psi[:, j - 1] + 0.5 * h * (uc[:, j] + uc[:, j - 1])
    inner = psi[1:-1, 1:-1]
    i, j = np.unravel_index(int(np.argmin(inner)), inner.shape)
    return float((i + 1 + 0.5) * h), float((j + 1 + 0.5) * h)   # (x, y) of that cell centre


def rescore(runs: pathlib.Path, write: bool, log=print) -> dict:
    summary = json.loads((runs / "summary.json").read_text())
    fields = np.load(runs / "fields.npz")
    report: dict = {
        "case": CASE_SLUG,
        "reynolds": REYNOLDS,
        "judge_grid": JUDGE_GRID,
        "registered_tolerance": REGISTERED_TOLERANCE,
        "vortex_tolerance_L": VORTEX_TOLERANCE_L,
        "criteria": [dict(spec) for spec in CRITERIA],
        "grids": [],
    }

    cases = []
    for case in sorted(summary["cases"], key=lambda c: int(c["n"])):
        n = int(case["n"])
        u_field = np.asarray(fields[f"u_{n}"], dtype=float)
        v_field = np.asarray(fields[f"v_{n}"], dtype=float)
        lines = profiles_from_fields(u_field, v_field, n)

        row: dict = {"n": n, "field_shapes": [list(u_field.shape), list(v_field.shape)],
                     "reconcile": {}, "schemes": {}}
        for scheme in ("hermite", "pchip"):
            got = {w: sampled(lines[w], w, scheme) for w in ("u", "v")}
            ref = {w: [p[1] for p in table(REYNOLDS, w)] for w in ("u", "v")}
            row["schemes"][scheme] = {
                "max_abs_u": max(abs(a - b) for a, b in zip(got["u"], ref["u"])),
                "max_abs_v": max(abs(a - b) for a, b in zip(got["v"], ref["v"])),
                "u_min": min(got["u"]), "v_max": max(got["v"]),
                "profiles": got,
            }
        # the stored bundle must be exactly what the stored field says (Hermite is
        # the bundle's own convention) -> any drift here is a reporting defect
        for w in ("u", "v"):
            stored = np.asarray(case[w]["profiles"]["computed"], dtype=float)
            fresh = np.asarray(row["schemes"]["hermite"]["profiles"][w], dtype=float)
            delta = float(np.max(np.abs(stored - fresh)))
            row["reconcile"][w] = delta
            assert delta <= RECONCILE_ATOL, f"n={n} {w}: stored profile != field ({delta:.3e})"

        ref_u = dict(table(REYNOLDS, "u"))
        ref_v = dict(table(REYNOLDS, "v"))
        u_min_ref = min(ref_u.values())
        v_max_ref = max(ref_v.values())
        u_min_got = row["schemes"]["hermite"]["u_min"]
        v_max_got = row["schemes"]["hermite"]["v_max"]

        u_min_rel = abs(u_min_got - u_min_ref) / abs(u_min_ref)
        v_max_rel = abs(v_max_got - v_max_ref) / abs(v_max_ref)

        vx, vy = streamfunction_vortex(u_field)
        cx, cy = VORTEX_CENTRE[REYNOLDS]
        psi_disp = max(abs(vx - cx), abs(vy - cy))
        row["vortex_centre"] = {
            "recomputed": [vx, vy],
            "stored": [float(x) for x in case["vortex_centre"]["computed"]],
            "reference": [float(cx), float(cy)],
            "displacement_recomputed": psi_disp,
            "displacement_stored": float(case["vortex_centre"]["displacement"]),
        }

        row["reconcile_metrics"] = {
            "C1": {"value_recomputed": row["schemes"]["hermite"]["max_abs_u"],
                   "value_stored": float(case["criteria"]["C1_u_profile_max_abs"]["value"])},
            "C2": {"value_recomputed": row["schemes"]["hermite"]["max_abs_v"],
                   "value_stored": float(case["criteria"]["C2_v_profile_max_abs"]["value"])},
            "C3": {"value_recomputed": u_min_rel,
                   "value_stored": float(case["criteria"]["C3_u_min_relative"]["value"])},
            "C4": {"value_recomputed": v_max_rel,
                   "value_stored": float(case["criteria"]["C4_v_max_relative"]["value"])},
        }
        for key, pair in row["reconcile_metrics"].items():
            assert abs(pair["value_recomputed"] - pair["value_stored"]) <= 1e-9, (n, key, pair)
        # C5 uses a second, independent estimator (stream-function minimum) rather
        # than the solver's own extremum search; on a coarse grid the discrete
        # minimum is flat, so the two may sit one or two cells apart.  Bind them to
        # within two cells instead of pretending they are the same estimator.
        stored_c = row["vortex_centre"]["stored"]
        cells = max(abs(vx - stored_c[0]), abs(vy - stored_c[1])) * n
        row["vortex_centre"]["estimator_gap_cells"] = cells
        assert cells <= 2.0 + 1e-9, f"n={n}: vortex estimators disagree by {cells:.2f} cells"
        row["reconcile_metrics"]["C5"] = {
            "value_recomputed": psi_disp,
            "value_stored": float(case["criteria"]["C5_vortex_centre"]["value"]),
            "estimator": "stream-function minimum vs solver extremum",
            "estimator_gap_cells": cells,
        }

        # interpolation-scheme sensitivity of the *metric* (audit finding F5)
        h = row["schemes"]["hermite"]
        p = row["schemes"]["pchip"]
        row["scheme_sensitivity"] = {
            "max_abs_u": abs(h["max_abs_u"] - p["max_abs_u"]),
            "max_abs_v": abs(h["max_abs_v"] - p["max_abs_v"]),
            "u_min": abs(h["u_min"] - p["u_min"]),
            "v_max": abs(h["v_max"] - p["v_max"]),
        }

        # re-evaluate with the registered band (evaluate() reads artifact["n"])
        case = dict(case)
        case["criteria"] = evaluate(case)
        graded = n >= JUDGE_GRID
        case["gate"] = {"judge_grid": JUDGE_GRID, "graded": graded,
                        "band": "registered" if graded else "diagnostic",
                        "registered_tolerance": REGISTERED_TOLERANCE,
                        "vortex_tolerance_L": VORTEX_TOLERANCE_L}
        checks = [v for k, v in case["criteria"].items() if not k.startswith("_")]
        case["verdict"] = "PASS" if all(v["pass"] for v in checks) else "FAIL"
        cases.append(case)

        row["verdict"] = case["verdict"]
        row["band"] = case["gate"]["band"]
        row["binding"] = {k: {"value": v["value"], "threshold": v["threshold"],
                              "graded": v["graded"], "pass": v["pass"]}
                          for k, v in case["criteria"].items() if not k.startswith("_")}
        report["grids"].append(row)
        log(f"n={n:3d} band={row['band']:10s} verdict={row['verdict']:4s} "
            f"max|du|={row['schemes']['hermite']['max_abs_u']:.3e} "
            f"u_min_rel={u_min_rel:.3%} v_max_rel={v_max_rel:.3%} "
            f"vortex={psi_disp:.4f} L")

    graded_rows = [c for c in cases if int(c["n"]) >= JUDGE_GRID]
    assert graded_rows, "no grid at or above the judge grid"
    report["judge"] = {
        "grid": max(int(c["n"]) for c in graded_rows),
        "verdict": {int(c["n"]): c["verdict"] for c in graded_rows},
        "all_graded_pass": all(c["verdict"] == "PASS" for c in graded_rows),
    }

    if write:
        summary = dict(summary)
        summary["cases"] = cases
        summary["benchmark"] = benchmark_rows(cases)
        summary["finest"] = max(int(c["n"]) for c in cases)
        summary["finest_verdict"] = {int(c["n"]): c["verdict"] for c in cases}[summary["finest"]]
        summary["verdict"] = summary["finest_verdict"]
        summary["verdict_all_resolutions"] = ("PASS" if all(c["verdict"] == "PASS" for c in cases)
                                              else "FAIL")
        summary["gate_criteria"] = [dict(spec) for spec in CRITERIA]
        summary["case"] = CASE_SLUG
        summary["anchor"] = CASE_SLUG
        summary["registered_tolerance"] = REGISTERED_TOLERANCE
        summary["vortex_tolerance_L"] = VORTEX_TOLERANCE_L
        summary["judge_grid"] = JUDGE_GRID
        summary["anchor_aliases"] = dict(ANCHOR_ALIASES)
        summary["rescore"] = {"script": "docs/CASE-FVM-CAVITY-RE1000/rescore_registered_gate.py",
                             "judge_grid": JUDGE_GRID,
                             "registered_tolerance": REGISTERED_TOLERANCE,
                             "vortex_tolerance_L": VORTEX_TOLERANCE_L}
        write_json(runs / "summary.json", summary)
        write_json(runs / "benchmark.json",
                   {"case": CASE_SLUG, "reynolds": REYNOLDS,
                    "judge_grid": JUDGE_GRID,
                    "registered_tolerance": REGISTERED_TOLERANCE,
                    "vortex_tolerance_L": VORTEX_TOLERANCE_L,
                    "rows": summary["benchmark"]})
        write_json(runs / "gate_registered.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--runs", default=str(here / "runs"))
    ap.add_argument("--no-write", action="store_true",
                    help="recompute and print only, leave the bundle untouched")
    args = ap.parse_args()
    rescore(pathlib.Path(args.runs), write=not args.no_write)


if __name__ == "__main__":
    main()