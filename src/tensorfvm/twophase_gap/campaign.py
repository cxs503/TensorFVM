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
)


def _leg_path(out: Path, name: str) -> Path:
    return out / f"leg_{name}.json"


def _leg_name(record: dict) -> str:
    if record["case"] == "zalesak":
        return f"zalesak_{record['scheme']}_{record['cells']}"
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
    cfl: float = 0.4,
    revolutions: float = 1.0,
) -> dict:
    """Run the requested legs and re-summarise the artefact directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if case not in ("all", "zalesak", "dambreak"):
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
    leg (one per grid).
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

    gates = []
    for gid, case, metric, comparator, limit, description in GATES:
        value = metrics.get(_METRIC_KEY[metric])
        missing = value is None
        present = (zalesak_declared is not None
                   if case == "zalesak" else bool(dambreak))
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
    lines = ["# GAP-FVM-TWOPHASE -- VOF interface capture (Zalesak + dam break)", ""]
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
                     f"{g['comparator']} {_fmt(g['limit'])} | "
                     f"{_fmt(g['got'])} | "
                     f"{'PASS' if g['passed'] else 'FAIL'} ({g['status']}) |")
    lines.append("")
    lines.append("## Zalesak slotted disk (one rigid revolution)")
    lines.append("")
    lines.append("| scheme | N | geometric err | mass drift | perimeter err "
                 "| peak retention | interface width [cells] |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(legs):
        rec = legs[key]
        if rec["case"] != "zalesak":
            continue
        lines.append(f"| {rec['scheme']} | {rec['cells']} | "
                     f"{rec['geometric_error']:.4f} | "
                     f"{rec['mass_drift_relative']:.2e} | "
                     f"{rec['perimeter_error']:.4f} | "
                     f"{rec['peak_retention']:.4f} | "
                     f"{rec['interface_width_cells']:.3f} |")
    lines.append("")
    lines.append("Exact solution of one revolution = the initial field; the "
                 "analytic slotted-disk area/perimeter are in each leg's "
                 "`reference` block.")
    lines.append("")
    lines.append("## Dam break vs Martin & Moyce (1952)")
    lines.append("")
    lines.append("| scheme | grid | dx/a | front err | height err | volume "
                 "drift | control |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(legs):
        rec = legs[key]
        if rec["case"] != "dambreak":
            continue
        grid = "x".join(str(v) for v in rec["grid"])
        lines.append(f"| {rec['scheme']} | {grid} | "
                     f"{rec['dx_m'] / 0.05715:.3f} | "
                     f"{rec['front_max_rel_error']:.4f} | "
                     f"{rec['height_max_rel_error']:.4f} | "
                     f"{rec['volume_drift']:.2e} | "
                     f"{_fmt(rec['control_max_abs_change'])} |")
    lines.append("")
    lines.append("Front error uses the half-cell depth convention over "
                 "`0.3 < t* <= t_end`; the leading-edge convention is reported "
                 "per leg in `front_max_rel_error_leading_edge`.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_figures(out: str | Path) -> list[Path]:
    """Render the three artefact figures from the committed JSON/NPZ data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    summary = json.loads((out / "gap_twophase.json").read_text())
    legs = summary["legs"]

    # fig1: Zalesak fields (exact vs captured) at the declared resolution.
    plic = [rec for rec in legs.values()
            if rec["case"] == "zalesak" and rec["scheme"] == "plic"]
    if plic:
        rec = max(plic, key=lambda r: r["cells"])
        name = f"zalesak_plic_{rec['cells']}"
        data = np.load(out / f"fields_{name}.npz")
        alpha, exact = data["alpha"], data["exact"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
        for ax, field, title in zip(
                axes, (exact, alpha, alpha - exact),
                ("exact (initial = exact after 1 revolution)",
                 f"PLIC N={rec['cells']}",
                 f"alpha(T) - alpha(0), geometric err "
                 f"{rec['geometric_error'] * 100:.2f} %")):
            im = ax.imshow(field, origin="lower",
                           cmap="RdBu_r" if field is alpha - exact else "viridis",
                           vmin=-0.5 if field is alpha - exact else 0.0,
                           vmax=0.5 if field is alpha - exact else 1.0)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.75)
        path = figdir / "fig1_zalesak_fields.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    # fig2: geometric error vs resolution with the 5 % criterion.
    curves = {}
    for rec in legs.values():
        if rec["case"] != "zalesak":
            continue
        curves.setdefault(rec["scheme"], []).append(
            (rec["cells"], rec["geometric_error"]))
    if curves:
        fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
        for scheme, pts in sorted(curves.items()):
            pts.sort()
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-",
                      label=scheme)
        ax.axhline(0.05, color="crimson", ls="--", lw=1,
                   label="criterion 5 %")
        ax.axvline(DECLARED_CELLS, color="grey", ls=":", lw=1,
                   label=f"declared N={DECLARED_CELLS}")
        ax.set_xlabel("cells per direction")
        ax.set_ylabel("thresholded geometric error")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title("Zalesak shape fidelity after one revolution", fontsize=9)
        path = figdir / "fig2_zalesak_geometric_error.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    # fig3: dam-break surge front / column height vs Martin & Moyce.
    dambreaks = [rec for rec in legs.values() if rec["case"] == "dambreak"]
    if dambreaks:
        rec = max(dambreaks, key=lambda r: r["grid"][0])
        name = f"dambreak_{rec['scheme']}_" + "x".join(str(v) for v in rec["grid"])
        hist = json.loads(str(np.load(out / f"fields_{name}.npz")["history"]))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
        t = np.asarray(hist["t_star"])
        if hist["front_star"] is not None:
            fs = np.asarray(hist["front_star"])
            axes[0].plot(t, fs[:, 1], "-", color="tab:blue",
                         label="PLIC, half-cell front")
            axes[0].plot(t, fs[:, 0], "--", color="tab:cyan", lw=1,
                         label="PLIC, leading edge")
        axes[0].plot(t, np.asarray(hist["front_ref"]), "o", ms=4,
                     color="black", label="Martin & Moyce (1952)")
        axes[0].set_xlabel("t* = t sqrt(g/a)")
        axes[0].set_ylabel("x* = (x_f - a)/a")
        axes[0].set_title(f"surge front, err {rec['front_max_rel_error'] * 100:.1f} %",
                          fontsize=9)
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)
        axes[1].plot(t, np.asarray(hist["height_star"]), "-", color="tab:green",
                     label="PLIC column height")
        axes[1].plot(t, np.asarray(hist["height_ref"]), "o", ms=4,
                     color="black", label="Martin & Moyce (1952)")
        axes[1].set_xlabel("t*")
        axes[1].set_ylabel("h/h0")
        axes[1].set_title(f"residual height, err "
                          f"{rec['height_max_rel_error'] * 100:.1f} %",
                          fontsize=9)
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        path = figdir / "fig3_dambreak_martin_moyce.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    return made