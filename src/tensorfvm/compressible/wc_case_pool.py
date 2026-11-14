"""Case-pool entry for ``CASE-WOODWARD-COLELLA`` (two interacting blast waves).

Ledger contract (``feature_ledger``)::

    【算例池|WOODWARD-COLELLA｜ref: Woodward & Colella 1984 包络｜容差: <10%】

Until now that contract lived only in prose: the metric arithmetic was in
:mod:`tensorfvm.compressible.wc_reference`, the machine-readable numbers in
``docs/woodward-colella-double-blast/reference/metrics_*.json`` and the verdict
in a series of reports.  A pool entry has to be *checkable*, i.e. a caller must
be able to obtain a single JSON record that states, per criterion, the
quantity, the comparator, the threshold and the unit, and a pass/fail computed
from the stored profiles, with no number copied by hand and no threshold that
is only implied (`程序化注入零手抄`).

This module is that entry.  It is deliberately a thin layer: the metric
definitions are *reused* from :mod:`tensorfvm.compressible.wc_reference` (same
window, same face finder, same 10 % ledger tolerance), so the pool record can
never drift away from the committed comparison code.

Two references are carried side by side and never merged:

``calibrated`` (primary)
    ``reference_density_calibrated.csv`` -- the digitised Dumbser et al. (JCP
    2019) Fig. 21 "Reference" curve (pure RPJ, 3200 cells) re-read with the
    y-axis anchored on the *tick marks* instead of the tick *label glyph
    centres*; the label centres sit 0.337 pt below the ticks, a constant
    ``+0.022017`` density bias that was diagnosed in
    ``docs/CASE-WOODWARD-COLELLA-calibration-bias.md``.
``committed`` (traceability only)
    ``reference_density.csv`` -- the original digitisation, kept unchanged so
    that every number published earlier can still be reproduced.

The primary verdict is computed against ``calibrated``; the ``committed``
numbers are reported next to it so the effect of the calibration is visible
rather than hidden.

CLI::

    python -m tensorfvm.compressible.wc_case_pool                 # pool grid, calibrated
    python -m tensorfvm.compressible.wc_case_pool --ladder        # every stored grid
    python -m tensorfvm.compressible.wc_case_pool --audit         # stored profiles vs
                                                                  # committed metrics JSON
Exit status is 0 iff the primary criteria pass on the declared pool grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .wc_reference import (
    DEFAULT_REFERENCE,
    POSITION_TOLERANCE,
    TOLERANCE_PERCENT,
    load_reference,
    metrics_from_profiles,
)

__all__ = [
    "CRITERIA",
    "GRID_STEPS",
    "POOL_GRID",
    "REFERENCE_KINDS",
    "audit_committed_metrics",
    "calibration_effect",
    "case_pool_entry",
    "evaluate_grid",
    "ladder",
    "load_calibrated_reference",
    "load_profile",
    "minimum_passing_grid",
    "main",
]

#: the case-pool tolerance is the ledger tolerance; it is *not* a knob.
POOL_TOLERANCE = TOLERANCE_PERCENT

#: grid the pool entry is run on (the finest grid whose profile is a committed
#: artifact and which passes every primary criterion -- see ``minimum_passing_grid``).
POOL_GRID = 800

#: all grids for which a solved profile is committed in this repository.
GRID_STEPS = (200, 400, 800, 1600, 3200)

_DOCS = (Path(__file__).resolve().parents[3] / "docs"
         / "woodward-colella-double-blast")
_REFERENCE_DIR = _DOCS / "reference"
_LADDER_DIR = _DOCS / "ladder" / "data"
_ARTIFACT_DIR = _DOCS / "artifacts"

#: reference tables; ``kind`` is the key used in the pool record.
REFERENCE_KINDS = {
    "calibrated": {
        "path": _REFERENCE_DIR / "reference_density_calibrated.csv",
        "columns": "x,panel_a_calibrated,panel_b_calibrated",
        "provenance": "Dumbser et al. JCP 2019 (arXiv:1902.07773) Fig. 21 "
                      "'Reference' (pure RPJ, 3200 cells), digitised from the PDF "
                      "vector paths, y-axis anchored on tick marks",
        "role": "primary",
    },
    "committed": {
        "path": DEFAULT_REFERENCE,
        "columns": "x,panel_a,panel_b",
        "provenance": "same digitisation read with the original (label-centre) "
                      "y-axis calibration; +0.022017 density units lower",
        "role": "traceability",
    },
}

#: stored solved profiles, keyed by grid; each entry is (path, x-key, rho-key).
_PROFILE_FILES = {
    200: (_LADDER_DIR / "ladder_profile_nx200_hllc.npz", "x", "rho"),
    400: (_ARTIFACT_DIR / "wc_blast_profile_nx400.npz", "x_got", "rho_got"),
    800: (_ARTIFACT_DIR / "wc_blast_profile_nx800.npz", "x_got", "rho_got"),
    1600: (_LADDER_DIR / "ladder_profile_nx1600_hllc.npz", "x", "rho"),
    3200: (_LADDER_DIR / "ladder_profile_nx3200_hllc.npz", "x", "rho"),
}

#: the acceptance contract, one row per criterion.  ``key`` indexes the
#: ``metrics_from_profiles`` output that supplies the value.
CRITERIA: List[dict] = [
    {"name": "peak_density", "key": "peak_density",
     "quantity": "max rho inside the digitised window x in [0.4, 1]",
     "comparator": "lt", "threshold": POOL_TOLERANCE, "unit": "%",
     "field": "rel_error_percent", "primary": True},
    {"name": "peak_position", "key": "peak_position",
     "quantity": "x of that maximum",
     "comparator": "lt", "threshold": POSITION_TOLERANCE, "unit": "x",
     "field": "abs_error", "primary": True},
    {"name": "l1_profile", "key": "l1_relative",
     "quantity": "integral |rho - rho_ref| dx / integral rho_ref dx over [0.4, 1]",
     "comparator": "lt", "threshold": POOL_TOLERANCE, "unit": "%",
     "field": "percent", "primary": True},
    {"name": "window_mass", "key": "window_mass",
     "quantity": "mass integral integral rho dx over [0.4, 1]",
     "comparator": "lt", "threshold": POOL_TOLERANCE, "unit": "%",
     "field": "rel_error_percent", "primary": True},
    {"name": "faces", "key": "faces",
     "quantity": "max |x_face - x_face_ref| over the four steepest density faces",
     "comparator": "lt", "threshold": POSITION_TOLERANCE, "unit": "x",
     "field": "max_abs_error", "primary": True},
    {"name": "low_plateau", "key": "low_plateau",
     "quantity": "min rho on x in [0.42, 0.98] (near-vacuum shallow trough)",
     "comparator": "lt", "threshold": POOL_TOLERANCE, "unit": "%",
     "field": "rel_error_percent", "primary": False,
     "excluded_reason": "ill-conditioned quantity (small denominator): the error is an "
                        "absolute offset of ~0.35 % of the peak density divided by a "
                        "local value of ~0.16, so it amplifies any constant bias in the "
                        "reference.  It was 13-14 % against the original digitisation and "
                        "is 0.04-0.59 % against the calibrated one at *every* grid -- the "
                        "14 % was a benchmark bias, not a local solver error "
                        "(docs/CASE-WOODWARD-COLELLA-calibration-bias.md).  Kept as an "
                        "informational row: gating a quantity whose denominator can go to "
                        "zero invites spurious FAILs, and every other criterion already "
                        "covers the trough region through the L1 and mass integrals.",
     },
]


def load_ref(kind: str = "calibrated") -> dict:
    """Load one of the two digitised reference tables (mean of both panels)."""
    if kind not in REFERENCE_KINDS:
        raise ValueError(f"unknown reference kind {kind!r}; "
                         f"choose from {sorted(REFERENCE_KINDS)}")
    spec = REFERENCE_KINDS[kind]
    if kind == "committed":
        # reuse the committed loader (it already averages the two digitisation panels)
        ref = load_reference(spec["path"])
        return dict(ref, kind=kind, provenance=spec["provenance"],
                    role=spec["role"])
    raw = np.loadtxt(spec["path"], delimiter=",", skiprows=1)
    x, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
    return {"kind": kind, "path": str(spec["path"]), "x": x,
            "rho": 0.5 * (a + b), "panel_spread": float(np.max(np.abs(a - b))),
            "provenance": spec["provenance"], "role": spec["role"]}


def load_calibrated_reference(path: str | Path | None = None) -> dict:
    """The primary reference table (calibrated y-axis)."""
    if path is None:
        return load_ref("calibrated")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    x, a, b = raw[:, 0], raw[:, 1], raw[:, 2]
    return {"kind": "calibrated", "path": str(path), "x": x,
            "rho": 0.5 * (a + b), "panel_spread": float(np.max(np.abs(a - b)))}


def load_profile(nx: int) -> dict:
    """Load the committed solved profile of one grid (no solver call)."""
    if nx not in _PROFILE_FILES:
        raise KeyError(f"no committed profile for nx={nx}; "
                       f"available: {sorted(_PROFILE_FILES)}")
    path, xk, rk = _PROFILE_FILES[nx]
    data = np.load(path)
    x = np.asarray(data[xk]).ravel()
    rho = np.asarray(data[rk]).ravel()
    order = np.argsort(x)
    out = {"nx": nx, "path": str(path), "x": x[order], "rho": rho[order]}
    if "p" in data and "u" in data:                       # ladder profiles carry the full state
        out["p"] = np.asarray(data["p"]).ravel()[order]
        out["u"] = np.asarray(data["u"]).ravel()[order]
    return out


def _criterion_value(name: str, m: dict) -> float | None:
    row = next(c for c in CRITERIA if c["name"] == name)
    if name == "faces":
        err = m.get("faces_abs_error")
        return None if not err else float(max(err))
    return float(m[row["key"]][row["field"]])


def _evaluate_criteria(m: dict) -> List[dict]:
    rows = []
    for c in CRITERIA:
        value = _criterion_value(c["name"], m)
        passed = (value is not None and value < c["threshold"])
        row = {
            "criterion": c["name"], "quantity": c["quantity"],
            "value": value, "unit": c["unit"], "comparator": c["comparator"],
            "threshold": c["threshold"], "primary": c["primary"],
            "pass": passed,
        }
        if not c["primary"]:
            row["excluded_reason"] = c["excluded_reason"]
        rows.append(row)
    return rows


def evaluate_grid(nx: int, reference_kind: str = "calibrated",
                  reference: dict | None = None) -> dict:
    """Per-criterion pool verdict for one grid against one reference table."""
    ref = reference or (load_ref(reference_kind) if reference_kind in REFERENCE_KINDS
                        else load_calibrated_reference(reference_kind))
    mine = load_profile(nx)
    m = metrics_from_profiles(ref["x"], ref["rho"], mine["x"], mine["rho"])
    rows = _evaluate_criteria(m)
    primary = [r for r in rows if r["primary"]]
    return {
        "nx": nx, "dx": 1.0 / nx, "profile": mine["path"],
        "reference": ref["path"], "reference_kind": ref.get("kind", str(reference_kind)),
        "criteria": rows,
        "all_primary_pass": all(r["pass"] for r in primary),
        "failed_primary": [r["criterion"] for r in primary if not r["pass"]],
        "verdict": "PASS" if all(r["pass"] for r in primary) else "FAIL",
        "metrics": m,
    }


def ladder(reference_kind: str = "calibrated",
           grids: Sequence[int] = GRID_STEPS) -> List[dict]:
    """Pool verdict on every grid with a committed profile."""
    ref = load_ref(reference_kind)
    return [evaluate_grid(nx, reference=ref) for nx in grids]


def minimum_passing_grid(reference_kind: str = "calibrated",
                         grids: Sequence[int] = GRID_STEPS) -> int | None:
    """Coarsest committed grid on which every primary criterion passes."""
    for row in ladder(reference_kind, grids):
        if row["all_primary_pass"]:
            return row["nx"]
    return None


def calibration_effect(grids: Sequence[int] = GRID_STEPS) -> List[dict]:
    """How much of each metric was the +0.022017 reference bias.

    The calibration report claims that the y-axis label-centre offset is a
    *constant additive* error in the digitised reference.  If that is true, the
    bias-dominated metrics must move by a constant and the mass integral must
    collapse towards zero -- and it must be independent of the grid.  This
    function measures that instead of asserting it.
    """
    out = []
    cal = {r["nx"]: {c["criterion"]: c["value"] for c in r["criteria"]}
           for r in ladder("calibrated", grids)}
    com = {r["nx"]: {c["criterion"]: c["value"] for c in r["criteria"]}
           for r in ladder("committed", grids)}
    for nx in sorted(cal):
        out.append({
            "nx": nx,
            "peak_density_error_percent": {"committed": com[nx]["peak_density"],
                                           "calibrated": cal[nx]["peak_density"]},
            "window_mass_error_percent": {"committed": com[nx]["window_mass"],
                                          "calibrated": cal[nx]["window_mass"]},
            "low_plateau_error_percent": {"committed": com[nx]["low_plateau"],
                                          "calibrated": cal[nx]["low_plateau"]},
        })
    return out


def case_pool_entry(nx: int = POOL_GRID, reference_kind: str = "calibrated") -> dict:
    """The pool record: contract + verdict on ``nx`` + the whole committed ladder.

    Nothing here is hard-coded to pass: every ``value`` is recomputed from a
    committed profile and a committed reference table, and the threshold is the
    ledger tolerance.
    """
    primary = evaluate_grid(nx, reference_kind)
    secondary = evaluate_grid(nx, "committed")
    grids = ladder(reference_kind)
    min_grid = next((g["nx"] for g in grids if g["all_primary_pass"]), None)
    return {
        "slug": "CASE-WOODWARD-COLELLA",
        "title": "算例池·双爆轰波相互作用 (Woodward & Colella 1984, two interacting blast waves)",
        "ledger_contract": {"reference": "Woodward & Colella 1984 envelope "
                                        "(via Dumbser et al. JCP 2019 Fig. 21 'Reference')",
                            "tolerance_percent": POOL_TOLERANCE,
                            "comparison_window_x": [0.4, 1.0]},
        "solver": {"scheme": "MUSCL-Hancock (2nd order in space and time, van Leer predictor-corrector) + MC limiter + HLLC",
                   "cfl": 0.4, "case": "tensorfvm.compressible.cases.double_blast"},
        "pool_grid": nx,
        "reference": {"kind": primary["reference_kind"], "path": primary["reference"],
                      "provenance": REFERENCE_KINDS[reference_kind]["provenance"],
                      "panel_spread": load_ref(reference_kind)["panel_spread"]},
        "criteria": primary["criteria"],
        "verdict": primary["verdict"],
        "all_primary_pass": primary["all_primary_pass"],
        "failed_primary": primary["failed_primary"],
        "minimum_passing_grid": min_grid,
        "metrics": primary["metrics"],
        "cross_reference": {
            "kind": "committed",
            "note": "same grid, same profile, original (label-centre) digitisation; "
                    "reported so the +0.022017 calibration offset stays visible",
            "criteria": secondary["criteria"],
            "verdict": secondary["verdict"],
        },
        "calibration_effect": calibration_effect(),
        "ladder": [{"nx": g["nx"], "verdict": g["verdict"],
                    "failed_primary": g["failed_primary"],
                    "peak_density_error_percent": next(
                        c["value"] for c in g["criteria"] if c["criterion"] == "peak_density"),
                    "l1_percent": next(
                        c["value"] for c in g["criteria"] if c["criterion"] == "l1_profile")}
                   for g in grids],
    }


def audit_committed_metrics() -> dict:
    """Recompute every grid from the raw profiles and compare with the stored JSON.

    The committed ``metrics_calibrated_nx*.json`` files were produced by earlier
    orders with their own scripts.  This audit is the check that the numbers in
    those files are still reproducible from the committed profiles and the
    committed reference tables -- the "no hand-copied number" invariant.
    """
    rows = []
    for nx in GRID_STEPS:
        path = _REFERENCE_DIR / f"metrics_calibrated_nx{nx}_hllc.json"
        if not path.exists():
            continue
        stored = json.loads(path.read_text())
        for kind in ("calibrated", "committed"):
            key = f"{kind}_reference"
            if key not in stored:
                continue
            mine = evaluate_grid(nx, kind)["metrics"]
            diffs = {
                "peak_density_rel_error_percent": abs(
                    mine["peak_density"]["rel_error_percent"]
                    - stored[key]["peak_density"]["rel_error_percent"]),
                "l1_percent": abs(mine["l1_relative"]["percent"]
                                  - stored[key]["l1_relative"]["percent"]),
                "window_mass_rel_error_percent": abs(
                    mine["window_mass"]["rel_error_percent"]
                    - stored[key]["window_mass"]["rel_error_percent"]),
            }
            rows.append({"nx": nx, "reference_kind": kind, "stored": str(path.name),
                         "max_abs_diff": max(diffs.values()), "diffs": diffs,
                         "reproduced": max(diffs.values()) < 1e-6})
    return {"rows": rows,
            "all_reproduced": all(r["reproduced"] for r in rows),
            "worst": max((r["max_abs_diff"] for r in rows), default=None)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=POOL_GRID)
    ap.add_argument("--reference", default="calibrated", choices=sorted(REFERENCE_KINDS))
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.audit:
        res = audit_committed_metrics()
        print(json.dumps(res, indent=1))
        if args.json:
            args.json.write_text(json.dumps(res, indent=1) + "\n")
        return 0 if res["all_reproduced"] else 1

    if args.ladder:
        rows = ladder(args.reference)
        for r in rows:
            print(f"nx={r['nx']:<5d} {r['verdict']:<4s} failed={r['failed_primary']}")
        payload: Dict[str, object] = {"ladder": [
            {"nx": r["nx"], "verdict": r["verdict"],
             "failed_primary": r["failed_primary"]} for r in rows]}
        if args.json:
            args.json.write_text(json.dumps(payload, indent=1) + "\n")
        return 0 if rows[-1]["all_primary_pass"] else 1

    entry = case_pool_entry(args.nx, args.reference)
    print(json.dumps(entry, indent=1))
    if args.json:
        args.json.write_text(json.dumps(entry, indent=1) + "\n")
    return 0 if entry["all_primary_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())