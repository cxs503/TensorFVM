"""Driver of the ``GAP-FVM-TWOPHASE`` verification face (CLI back-end).

Usage (see :mod:`tensorfvm.twophase_gap.__main__`)::

    python -m tensorfvm.twophase_gap --case all --out docs/GAP-FVM-TWOPHASE/artifacts

The driver is **resumable**: every leg writes its own ``leg_<name>.json`` (plus
``.npz`` field data) into ``--out`` and the summary :func:`summarise` re-reads
whatever legs are present.  That keeps a 20-minute 256^2 leg on the cluster
independent from the fast legs, and it makes the gate evaluation a pure
function of committed artefacts -- the report cannot contain a number that is
not in the JSON.

Gates are evaluated by :func:`evaluate_gates` and stored *inside* the same JSON
(``gates``), so a later reader can re-run the comparison without re-solving.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .dambreak_gate import (
    CONTROL_MAX_ABS_CHANGE,
    FRONT_ERROR_MAX,
    HEIGHT_ERROR_MAX,
    VOLUME_DRIFT_MAX,
    run_dambreak_leg,
)
from .hysing_gate import (
    CIRCULARITY_MAX,
    CONVERGENCE_MAX as HYSING_CONVERGENCE_MAX,
    MASS_DRIFT_MAX as HYSING_MASS_DRIFT_MAX,
    TERMINAL_VELOCITY_MAX,
    convergence_order as hysing_convergence_order,
    run_hysing_leg,
)
from .zalesak_gate import (
    BOUND_VIOLATION_MAX,
    DECLARED_CELLS,
    GEOMETRIC_ERROR_MAX,
    MASS_DRIFT_MAX,
    PERIMETER_ERROR_MAX,
    convergence_order,
    run_zalesak_leg,
)

__all__ = ["DEFAULTS", "GATES", "run_campaign", "summarise", "evaluate_gates",
           "render_report", "write_figures"]

#: Default legs.  ``(64, 128)`` is the resolution pair that brackets the
#: declared ``N = 128``; ``256`` can be requested with ``--cells``.
DEFAULTS = {
    "zalesak_cells": (64, 128),
    "zalesak_schemes": ("plic", "cicsam"),
    "declared_cells": DECLARED_CELLS,
    "dambreak_grids": ((120, 40),),
    "dambreak_t_star": 4.0,
    "hysing_cells": (32, 64),
}

#: Gate table: ``(id, case, metric, comparator, limit, description)``.
GATES = (
    ("Z1", "zalesak", "geometric_error", "<=", GEOMETRIC_ERROR_MAX,
     "thresholded geometric error at the declared resolution"),
    ("Z2", "zalesak", "mass_drift_relative", "<=", MASS_DRIFT_MAX,
     "relative liquid-area drift over one revolution"),
    ("Z3", "zalesak", "perimeter_error", "<=", PERIMETER_ERROR_MAX,
     "relative captured-contour length change"),
    ("Z4", "zalesak", "bound_violation", "<=", BOUND_VIOLATION_MAX,
     "max excursion of alpha outside [0, 1]"),
    ("Z5", "zalesak", "convergence_order", ">=", 0.8,
     "observed order of the geometric error, coarse -> declared resolution"),
    ("D1", "dambreak", "front_max_rel_error", "<=", FRONT_ERROR_MAX,
     "surge front vs Martin & Moyce (1952)"),
    ("D2", "dambreak", "height_max_rel_error", "<=", HEIGHT_ERROR_MAX,
     "residual column height vs Martin & Moyce (1952)"),
    ("D3", "dambreak", "volume_drift", "<=", VOLUME_DRIFT_MAX,
     "max relative liquid-volume drift"),
    ("D4", "dambreak", "bound_violation", "<=", 1e-10,
     "max excursion of alpha outside [0, 1]"),
    ("D5", "dambreak", "control_max_abs_change", "<=", CONTROL_MAX_ABS_CHANGE,
     "zero-gravity negative control: frozen interface"),
    ("B1", "hysing", "terminal_velocity_rel_error", "<=", TERMINAL_VELOCITY_MAX,
     "terminal velocity vs Hysing et al. (2009) ref"),
    ("B2", "hysing", "circularity_rel_error", "<=", CIRCULARITY_MAX,
     "minimum circularity vs Hysing et al. (2009) ref"),
    ("B3", "hysing", "worst_step_drift", "<=", HYSING_MASS_DRIFT_MAX,
     "worst per-step relative volume drift"),
    ("B4", "hysing", "convergence_order", ">=", HYSING_CONVERGENCE_MAX,
     "terminal velocity convergence coarse -> fine"),
)


def _leg_path(out: Path, name: str) -> Path:
    return out / f"leg_{name}.json"


def _leg_name(record: dict) -> str:
    if record["case"] == "zalesak":
        return f"zalesak_{record['scheme']}_{record['cells']}"
    if record["case"] == "hysing":
        return f"hysing_{record['cells_per_diameter']}"
    grid = "x".join(str(v) for v in record["grid"])
    return f"dambreak_{record['scheme']}_{grid}"


def _write_leg(out: Path, name: str, record: dict, arrays: dict | None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = _leg_path(out, name)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    if arrays:
        np.savez_compressed(out / f"fields_{name}.npz", **arrays)
    return path


def run_campaign(
    out: str | Path = "docs/GAP-FVM-TWOPHASE/artifacts",
    *,
    case: str = "all",
    zalesak_cells=DEFAULTS["zalesak_cells"],
    zalesak_schemes=DEFAULTS["zalesak_schemes"],
    dambreak_grids=DEFAULTS["dambreak_grids"],
    dambreak_t_star: float = DEFAULTS["dambreak_t_star"],
    hysing_cells=DEFAULTS["hysing_cells"],
    cfl: float = 0.4,
    revolutions: float = 1.0,
) -> dict:
    """Run the requested legs and re-summarise the artefact directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if case not in ("all", "zalesak", "dambreak", "hysing"):
        raise ValueError(f"unknown case {case!r}")

    if case in ("all", "zalesak"):
        for scheme in zalesak_schemes:
            for cells in zalesak_cells:
                leg = run_zalesak_leg(cells, scheme, cfl=cfl,
                                      revolutions=revolutions)
                rec = leg["record"]
                name = _leg_name(rec)
                _write_leg(out, name, rec,
                           {"alpha": leg["field"].numpy(),
                            "exact": leg["exact"].numpy()})
                print(f"[zalesak] {name}: geometric_error="
                      f"{rec['geometric_error']:.4f}", flush=True)

    if case in ("all", "dambreak"):
        for nx, ny in dambreak_grids:
            leg = run_dambreak_leg(nx, ny, t_star_end=dambreak_t_star)
            rec = leg["record"]
            name = _leg_name(rec)
            _write_leg(out, name, rec, {"alpha_final": leg["alpha_final"],
                                        "history": np.array(json.dumps(
                                            leg["history"]))})
            print(f"[dambreak] {name}: front_err="
                  f"{rec['front_max_rel_error']:.4f} height_err="
                  f"{rec['height_max_rel_error']:.4f}", flush=True)

    if case in ("all", "hysing"):
        for cpd in hysing_cells:
            leg = run_hysing_leg(cpd)
            rec = leg["record"]
            name = _leg_name(rec)
            _write_leg(out, name, rec, None)
            print(f"[hysing] {name}: ut={rec['terminal_velocity']:.4f} "
                  f"(ref {rec['reference']['terminal_velocity']}) "
                  f"err={rec['terminal_velocity_rel_error']:.4f}", flush=True)

    summary = summarise(out)
    print(f"summary -> {out / 'gap_twophase.json'}  "
          f"passed={summary['all_passed']}", flush=True)
    return summary


def _load_legs(out: Path) -> dict:
    legs = {}
    for path in sorted(out.glob("leg_*.json")):
        record = json.loads(path.read_text())
        legs[path.stem[len("leg_"):]] = record
    return legs


def summarise(out: str | Path) -> dict:
    """Re-read the legs, evaluate the gates, write JSON + markdown."""
    out = Path(out)
    legs = _load_legs(out)
    gates, metrics = evaluate_gates(legs)
    summary = {
        "schema": "tensorfvm/gap-fvm-twophase/1",
        "out": str(out),
        "legs": legs,
        "metrics": metrics,
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates) and bool(gates),
        "n_legs": len(legs),
        "environment": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
        },
    }
    (out / "gap_twophase.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    render_report(summary, out / "GAP-FVM-TWOPHASE.md")
    return summary


def evaluate_gates(legs: dict) -> tuple[list[dict], dict]:
    """Turn legs into gate verdicts plus the flat metric table.

    Zalesak gates are judged on the **declared** PLIC leg (``cells == 128``);
    every other leg's numbers are reported but not graded (they are the
    comparison/negative material).  Dam-break gates are judged on every PLIC
    leg (one per grid).  Hysing gates are judged on the declared resolution
    leg (``cells_per_diameter == 64``).
    """
    zalesak_declared = next(
        (rec for key, rec in legs.items()
         if rec["case"] == "zalesak" and rec["scheme"] == "plic"
         and rec.get("declared")), None)
    zalesak_coarse = next(
        (rec for key, rec in legs.items()
         if rec["case"] == "zalesak" and rec["scheme"] == "plic"
         and rec["cells"] < DECLARED_CELLS), None)
    dambreak = {key: rec for key, rec in legs.items()
                if rec["case"] == "dambreak" and rec["scheme"] == "plic"}
    hysing_declared = next(
        (rec for key, rec in legs.items()
         if rec["case"] == "hysing" and rec.get("declared")), None)
    hysing_coarse = next(
        (rec for key, rec in legs.items()
         if rec["case"] == "hysing" and not rec.get("declared")), None)

    metrics = {
        "zalesak_declared_cells": DECLARED_CELLS if zalesak_declared else None,
        "zalesak_geometric_error": (zalesak_declared["geometric_error"]
                                     if zalesak_declared else None),
        "zalesak_mass_drift": (zalesak_declared["mass_drift_relative"]
                               if zalesak_declared else None),
        "zalesak_perimeter_error": (zalesak_declared["perimeter_error"]
                                    if zalesak_declared else None),
        "zalesak_bound_violation": (zalesak_declared["bound_violation"]
                                    if zalesak_declared else None),
        "zalesak_convergence_order": None,
        "dambreak_front_error": None,
        "dambreak_height_error": None,
        "dambreak_volume_drift": None,
        "dambreak_control": None,
        "hysing_terminal_velocity_error": None,
        "hysing_circularity_error": None,
        "hysing_worst_step_drift": None,
        "hysing_convergence_order": None,
    }
    if zalesak_declared and zalesak_coarse:
        ratio = zalesak_declared["cells"] / zalesak_coarse["cells"]
        metrics["zalesak_convergence_order"] = convergence_order(
            zalesak_coarse["geometric_error"],
            zalesak_declared["geometric_error"], ratio=ratio)
        metrics["zalesak_coarse_cells"] = zalesak_coarse["cells"]
        metrics["zalesak_coarse_geometric_error"] = zalesak_coarse[
            "geometric_error"]
    if dambreak:
        worst_front = max(dambreak.values(),
                          key=lambda r: r["front_max_rel_error"])
        worst_height = max(dambreak.values(),
                           key=lambda r: r["height_max_rel_error"])
        metrics["dambreak_front_error"] = worst_front["front_max_rel_error"]
        metrics["dambreak_height_error"] = worst_height["height_max_rel_error"]
        metrics["dambreak_volume_drift"] = max(
            r["volume_drift"] for r in dambreak.values())
        controls = [r["control_max_abs_change"] for r in dambreak.values()
                    if r["control_max_abs_change"] is not None]
        metrics["dambreak_control"] = max(controls) if controls else None
    if hysing_declared:
        metrics["hysing_terminal_velocity_error"] = hysing_declared[
            "terminal_velocity_rel_error"]
        metrics["hysing_circularity_error"] = hysing_declared[
            "circularity_rel_error"]
        metrics["hysing_worst_step_drift"] = hysing_declared[
            "worst_step_drift"]
    if hysing_declared and hysing_coarse:
        metrics["hysing_convergence_order"] = hysing_convergence_order(
            hysing_coarse["terminal_velocity"],
            hysing_declared["terminal_velocity"])

    gates = []
    for gid, case, metric, comparator, limit, description in GATES:
        value = metrics.get(_METRIC_KEY[metric])
        missing = value is None
        if case == "zalesak":
            present = zalesak_declared is not None
        elif case == "dambreak":
            present = bool(dambreak)
        else:
            present = hysing_declared is not None
        passed = (not missing and present
                  and _compare(value, comparator, limit))
        gates.append({
            "id": gid,
            "case": case,
            "metric": metric,
            "comparator": comparator,
            "limit": limit,
            "got": value,
            "passed": bool(passed),
            "status": ("measured" if not missing else
                       ("not_run" if not present else "not_measured")),
            "description": description,
        })
    return gates, metrics


#: Map gate metric name -> key of the flat ``metrics`` table.
_METRIC_KEY = {
    "geometric_error": "zalesak_geometric_error",
    "mass_drift_relative": "zalesak_mass_drift",
    "perimeter_error": "zalesak_perimeter_error",
    "bound_violation": "zalesak_bound_violation",
    "convergence_order": "zalesak_convergence_order",
    "front_max_rel_error": "dambreak_front_error",
    "height_max_rel_error": "dambreak_height_error",
    "volume_drift": "dambreak_volume_drift",
    "control_max_abs_change": "dambreak_control",
    "terminal_velocity_rel_error": "hysing_terminal_velocity_error",
    "circularity_rel_error": "hysing_circularity_error",
    "worst_step_drift": "hysing_worst_step_drift",
}


def _compare(value, comparator: str, limit: float) -> bool:
    if comparator == "<=":
        return bool(value <= limit)
    if comparator == ">=":
        return bool(value >= limit)
    raise ValueError(f"unknown comparator {comparator!r}")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "not run"
    if isinstance(value, float) and (math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}g}"


def render_report(summary: dict, path: str | Path) -> Path:
    """Write the human-readable acceptance sheet from the summary dict."""
    path = Path(path)
    legs = summary["legs"]
    lines = ["# GAP-FVM-TWOPHASE -- VOF interface capture (Zalesak + dam break + Hysing bubble)", ""]
    lines.append(f"Verdict: **{'PASS' if summary['all_passed'] else 'FAIL'}** "
                 f"({sum(1 for g in summary['gates'] if g['passed'])}/"
                 f"{len(summary['gates'])} gates)")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append("| id | case | metric | limit | got | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for g in summary["gates"]:
        lines.append(f"| {g['id']} | {g['case']} | {g['metric']} | "
                     f"{g['comparator']} {g['limit']} | {_fmt(g['got'])} | "
                     f"{'PASS' if g['passed'] else 'FAIL'} |")
    lines.append("")

    # Zalesak table
    zalesak_legs = {k: v for k, v in legs.items() if v["case"] == "zalesak"}
    if zalesak_legs:
        lines.append("## Zalesak slotted disk")
        lines.append("")
        lines.append("| scheme | N | geometric_error | mass_drift | perimeter | bound_violation |")
        lines.append("|---|---|---|---|---|---|")
        for key in sorted(zalesak_legs):
            r = zalesak_legs[key]
            lines.append(f"| {r['scheme']} | {r['cells']} | "
                         f"{r['geometric_error']:.4f} | "
                         f"{r['mass_drift_relative']:.4e} | "
                         f"{r['perimeter_error']:.4f} | "
                         f"{r['bound_violation']:.4e} |")
        lines.append("")

    # Dam break table
    dam_legs = {k: v for k, v in legs.items() if v["case"] == "dambreak"}
    if dam_legs:
        lines.append("## Dam break vs Martin & Moyce (1952)")
        lines.append("")
        lines.append("| grid | front_err | height_err | volume_drift |")
        lines.append("|---|---|---|---|")
        for key in sorted(dam_legs):
            r = dam_legs[key]
            grid = "x".join(str(v) for v in r["grid"])
            lines.append(f"| {grid} | {r['front_max_rel_error']:.4f} | "
                         f"{r['height_max_rel_error']:.4f} | "
                         f"{r['volume_drift']:.4e} |")
        lines.append("")

    # Hysing table
    hysing_legs = {k: v for k, v in legs.items() if v["case"] == "hysing"}
    if hysing_legs:
        lines.append("## Hysing rising bubble (Case 2: Eo=10, M=1e-3)")
        lines.append("")
        lines.append("| cpd | terminal_velocity | ut_ref | ut_err | "
                     "min_circularity | circ_ref | circ_err | volume_drift |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key in sorted(hysing_legs):
            r = hysing_legs[key]
            ref = r["reference"]
            lines.append(f"| {r['cells_per_diameter']} | "
                         f"{r['terminal_velocity']:.4f} | "
                         f"{ref['terminal_velocity']} | "
                         f"{r['terminal_velocity_rel_error']:.4f} | "
                         f"{r['min_circularity']:.4f} | "
                         f"{ref['min_circularity']} | "
                         f"{r['circularity_rel_error']:.4f} | "
                         f"{r['worst_step_drift']:.4e} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    return path


def write_figures(out: str | Path) -> list[Path]:
    """Render figures from the leg artefacts (requires matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    figs = []

    legs = _load_legs(out)
    hysing_legs = {k: v for k, v in legs.items() if v["case"] == "hysing"}
    if hysing_legs:
        fig, ax = plt.subplots()
        for key in sorted(hysing_legs):
            r = hysing_legs[key]
            ax.plot(r["cells_per_diameter"], r["terminal_velocity"],
                    "o-", label=f"cpd={r['cells_per_diameter']}")
        ax.axhline(0.2342, color="r", ls="--", label="Hysing ref")
        ax.set_xlabel("cells per diameter")
        ax.set_ylabel("terminal velocity")
        ax.legend()
        ax.set_title("Hysing bubble: terminal velocity vs resolution")
        p = out / "figures" / "fig_hysing_terminal_velocity.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p)

    return figs
