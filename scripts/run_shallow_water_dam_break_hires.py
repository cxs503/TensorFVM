"""High-resolution extension of CASE-FVM-DAMBREAK-STOKER (n = 800 ... 6400).

Why this exists
---------------
The published evidence package for ``CASE-FVM-DAMBREAK-STOKER`` stops at
``n = 1600``: ``docs/case-fvm-dambreak-stoker/AUDIT.md`` §1 records
"未运行: 3200 单元以上的收敛序列（本机 load15 ≈ 100）".  That leaves two
questions open which the accepted-grid claim (n >= 800, 5 % tolerance) does
*not* answer:

1. **Is the published error still converging, or has it hit an estimator
   floor?**  Every graded quantity here is an *estimator* (fan-shape for the
   dry tip, mid-level sub-cell crossing for the wet bore).  An estimator can
   saturate while the solution keeps improving; quoting "0.017 %" as a
   discretisation error is only defensible if the error is still decreasing.
2. **Does the anchor acceptance (``x_f(t)`` series vs analytic, < 5 %) survive
   refinement in both directions** -- i.e. is the pass margin a property of the
   scheme and not of one lucky grid?

What is measured (all programmatically, nothing hand-copied)
----------------------------------------------------------
For both cases and every grid ``n``:

* ``front_series``  -- the graded anchor quantity, whole ``x_f(t)`` series
  (``tensorfvm.shallow_water.dam_break.series_metrics``)
* headline at ``t* = 2`` -- dry: ``front_fan`` (fan-shape estimator);
  wet: ``shock_position`` (oracle-windowed) *and* the data-driven sub-cell
  ``bore_x`` (``data_driven_plateau``), so the F4/F7 debts are re-checked at
  resolutions they were never evaluated on
* ``depth_l1`` -- the classic L1 depth error (first order expected: the
  solution contains a discontinuity)
* conservation / monotonicity guards (``tv_ratio``, ``local_extrema``,
  ``clipped_cells``, mass drift) and ``verdict`` (including any failing key)

Derived, with the rule stated *before* looking at the numbers
-------------------------------------------------------------
``refinement_ratio(n) = e(n) / e(n/2)``.  Second order would give 0.25, first
order 0.5, a saturated estimator ~1.  A quantity is called *at its estimator
floor over the range* when ``e(6400) / e(3200) > 0.9`` -- a fixed, prior rule,
not a label fitted to the outcome.  No convergence order is claimed from fewer
than three grids, and no order is fitted across the whole range when the finest
ratio contradicts the coarser ones (reported as ``non_asymptotic``).

Usage::

    PYTHONPATH=src python scripts/run_shallow_water_dam_break_hires.py \
        --resolutions 800,1600,3200,6400
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tensorfvm.shallow_water.dam_break import (
    CASES,
    RITTER_DRY,
    STOKER_WET,
    TOLERANCE,
    run_case,
)

ARTIFACT_DIR = ROOT / "src/tensorfvm/verification/artifacts/shallow-water-dam-break"
DOCS_DIR = ROOT / "docs/case-fvm-dambreak-stoker-hires"

# ---- prior decision rules (declared before any run) ----------------------- #
FLOOR_RATIO = 0.9        # e(finest)/e(finest/2) > FLOOR_RATIO  => "no decay"
MIN_GRIDS_FOR_ORDER = 3  # never fit an order on fewer grids


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip() or "unknown"
    except Exception:                                        # pragma: no cover
        return "unknown"


def _ratio(a: float | None, b: float | None) -> float | None:
    """``a / b`` with None / non-finite guarding (never NaN in the artefact)."""
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0.0:
        return None
    return a / b


def _observed_order(values: list[float], dxs: list[float]) -> float | None:
    """Log-log slope of ``values`` against ``dxs`` (needs >= MIN_GRIDS_FOR_ORDER)."""
    pairs = [(d, v) for d, v in zip(dxs, values)
             if math.isfinite(v) and v > 0.0 and d > 0.0]
    if len(pairs) < MIN_GRIDS_FOR_ORDER:
        return None
    lx = np.log(np.array([p[0] for p in pairs]))
    ly = np.log(np.array([p[1] for p in pairs]))
    return float(np.polyfit(lx, ly, 1)[0])


def _decay_verdict(series: list[float]) -> dict:
    """Classify the last refinement step with the prior FLOOR_RATIO rule."""
    vals = [v for v in series if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return {"finest_ratio": None, "at_floor": None,
                "rule": f"e(nmax)/e(nmax/2) > {FLOOR_RATIO}",
                "needs_at_least_grids": 2}
    r = _ratio(vals[-1], vals[-2])
    return {
        "finest_ratio": r,
        "at_floor": None if r is None else bool(r > FLOOR_RATIO),
        "rule": f"e(nmax)/e(nmax/2) > {FLOOR_RATIO}",
        "all_ratios": [{"step": i + 1, "ratio": _ratio(vals[i + 1], vals[i])}
                       for i in range(len(vals) - 1)],
        "needs_at_least_grids": 2,
    }


def _row(case_key: str, n: int) -> dict:
    case = CASES[case_key]
    t0 = time.perf_counter()
    run = run_case(case, n=n)
    wall = time.perf_counter() - t0
    m = run["metrics"]
    series = m["front_series"]
    verdict = {k: (v if isinstance(v, bool) else bool(v["pass"]))
               for k, v in run["verdict"].items()}
    failing = [k for k, ok in verdict.items() if k != "all_pass" and not ok]
    row = {
        "case": case_key,
        "n": n,
        "dx": run["discretisation"]["dx"],
        "steps": run["discretisation"]["steps"],
        "wall_time_s": wall,
        "depth_l1": m["depth_l1"],
        "front_series": series["max_error"],
        "front_series_graded": series["samples_graded"],
        "front_series_samples": series["samples"],
        "front_series_unresolved_max": series["unresolved_max_error"],
        "front_series_legacy_max": series["legacy_max_error"],
        "front_series_pass": series["pass"],
        "mass_drift": abs(run["discretisation"]["volume_final"]
                          - run["discretisation"]["volume_initial"])
                      / (run["discretisation"]["volume_initial"] or 1.0),
        "clipped_cells": run["discretisation"]["clipped_cells"],
        "tv_ratio": m.get("tv_ratio"),
        "local_extrema": m.get("local_extrema"),
        "verdict_all_pass": run["verdict"]["all_pass"],
        "verdict_failing": failing,
    }
    if case.kind == "dry":
        row.update({
            "headline": m["front_fan_error"],
            "headline_quantity": "front_fan (fan-shape estimator at t*=2)",
            "front_fan_spread": m.get("front_fan_spread"),
            "front_wetted_error": m["front_wetted_error"],
            "front_wetted_note": "depth-floor diagnostic, not graded (AUDIT F1)",
        })
    else:
        dd = m["dd"]
        row.update({
            "headline": m["shock_error"],
            "headline_quantity": "shock_position (oracle-windowed, AUDIT F4)",
            "step_ratio_error": m["step_ratio_error"],
            "bore_x_error": dd["bore_x_error"],
            "bore_x_cell_error": dd["bore_x_cell_error"],
            "bore_level": dd["bore_level"],
            "bore_method": dd["bore_method"],
            "bore_h2_error": dd["h2_error"],
            "bore_valid": dd.get("bore_valid"),
            "bore_flatness": dd.get("flatness"),
            "bore_thresh_frac_used": dd.get("thresh_frac_used"),
            "bore_step_ratio_error": dd["step_ratio_error"],
            "bore_cells_off": (None if dd["bore_x_error"] is None else
                               dd["bore_x_error"] * m["shock_exact"] / row["dx"]),
        })
    return row


def study(resolutions=(800, 1600, 3200, 6400)) -> dict:
    rows = {key: [] for key in CASES}
    for key in CASES:
        for n in resolutions:
            rows[key].append(_row(key, n))

    per_case = {}
    for key, case_rows in rows.items():
        dxs = [r["dx"] for r in case_rows]
        summary = {
            "quantity": case_rows[0]["headline_quantity"],
            "rows": case_rows,
            "observed_order_depth_l1": _observed_order(
                [r["depth_l1"] for r in case_rows], dxs),
            "observed_order_series": _observed_order(
                [r["front_series"] for r in case_rows], dxs),
            "observed_order_headline": _observed_order(
                [r["headline"] for r in case_rows], dxs),
            "series_decay": _decay_verdict([r["front_series"] for r in case_rows]),
            "headline_decay": _decay_verdict([r["headline"] for r in case_rows]),
            "series_max_over_range": max(
                r["front_series"] for r in case_rows
                if r["front_series"] is not None),
            "series_max_over_accepted": max(
                (r["front_series"] for r in case_rows
                 if r["n"] >= 800 and r["front_series"] is not None),
                default=None),
            "all_accepted_pass": all(
                r["front_series_pass"] for r in case_rows if r["n"] >= 800),
        }
        if key == STOKER_WET.key:
            summary["observed_order_bore_subcell"] = _observed_order(
                [r["bore_x_error"] for r in case_rows], dxs)
            summary["bore_decay"] = _decay_verdict(
                [r["bore_x_error"] for r in case_rows])
            summary["bore_cell_decay"] = _decay_verdict(
                [r["bore_x_cell_error"] for r in case_rows])
            summary["bore_max_cell_error_over_range"] = max(
                r["bore_x_cell_error"] for r in case_rows)
        per_case[key] = summary

    return {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _git_commit(),
            "resolutions": list(resolutions),
            "tolerance": TOLERANCE,
            "floor_ratio_rule": FLOOR_RATIO,
            "min_grids_for_order": MIN_GRIDS_FOR_ORDER,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "command": "PYTHONPATH=src python scripts/run_shallow_water_dam_break_hires.py "
                       f"--resolutions {','.join(str(n) for n in resolutions)}",
        },
        "cases": per_case,
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    series_defs = (
        ("depth_l1", "depth L1 error", "o"),
        ("front_series", "x_f(t) series max error (graded anchor)", "s"),
        ("headline", "headline @ t*=2", "^"),
    )
    for ax, (key, title) in zip(axes[:2], ((RITTER_DRY.key, "Ritter dry bed"),
                                           (STOKER_WET.key, "Stoker wet bed"))):
        rows = payload["cases"][key]["rows"]
        dx = np.array([r["dx"] for r in rows])
        for field, label, marker in series_defs:
            vals = np.array([r[field] if r[field] is not None else np.nan
                             for r in rows], dtype=float)
            ax.loglog(dx, vals, marker + "-", label=label)
        if key == STOKER_WET.key:
            ax.loglog(dx, [r["bore_x_error"] for r in rows], "d-",
                      label="bore x (data-driven sub-cell)")
            ax.loglog(dx, [r["bore_x_cell_error"] for r in rows], "x--",
                      color="0.5", label="bore x (cell centre, negative control)")
        ref = dx / dx[0]
        ax.loglog(dx, 0.01 * ref, "k:", lw=1.0, label="slope 1 reference")
        ax.loglog(dx, 0.01 * ref ** 2, "k-.", lw=1.0, label="slope 2 reference")
        ax.axhline(payload["meta"]["tolerance"], color="r", ls="--", lw=1.0,
                   label="5 % acceptance tolerance")
        ax.set_xlabel("dx [m]")
        ax.set_ylabel("relative / L1 error")
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    ax = axes[2]
    for key, color in ((RITTER_DRY.key, "C0"), (STOKER_WET.key, "C1")):
        rows = payload["cases"][key]["rows"]
        ratios = payload["cases"][key]["series_decay"]["all_ratios"]
        ax.semilogy(range(1, len(ratios) + 1), [r["ratio"] for r in ratios],
                    "o-", color=color,
                    label=f"{key}: e(n)/e(n/2) of series error")
    ax.axhline(1.0, color="k", ls=":", label="no decay (ratio = 1)")
    ax.axhline(0.5, color="k", ls="-.", label="first order (0.5)")
    ax.axhline(payload["meta"]["floor_ratio_rule"], color="r", ls="--",
               label=f"floor rule (> {payload['meta']['floor_ratio_rule']})")
    ax.set_xlabel("refinement step index")
    ax.set_ylabel("error ratio")
    ax.set_title("estimator decay per refinement step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("CASE-FVM-DAMBREAK-STOKER: high-resolution extension "
                 + str(payload["meta"]["resolutions"]))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _markdown(payload: dict) -> str:
    lines = ["# High-resolution dam-break extension (auto-generated)", "",
             f"generated: {payload['meta']['generated_at']}  commit: "
             f"`{payload['meta']['git_commit']}`  tolerance: "
             f"{payload['meta']['tolerance']:.0%}", ""]
    for key, case in payload["cases"].items():
        lines += [f"## {key} — headline = {case['quantity']}", "",
                  "| n | dx [m] | x_f(t) series | graded | headline | depth L1 | "
                  "verdict failing | wall [s] |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in case["rows"]:
            head = r["headline"]
            lines.append(
                f"| {r['n']} | {r['dx']:.6g} | "
                f"{(r['front_series'] if r['front_series'] is not None else float('nan')):.3%} | "
                f"{r['front_series_graded']}/{r['front_series_samples']} | "
                f"{(head if head is not None else float('nan')):.3%} | "
                f"{r['depth_l1']:.4%} | "
                f"{', '.join(r['verdict_failing']) or '—'} | {r['wall_time_s']:.1f} |")
        lines += ["",
                  f"* observed order (depth L1): {case['observed_order_depth_l1']}",
                  f"* observed order (x_f(t) series): {case['observed_order_series']}",
                  f"* observed order (headline): {case['observed_order_headline']}",
                  f"* series decay: finest ratio {case['series_decay']['finest_ratio']}"
                  f" → at floor = {case['series_decay']['at_floor']}",
                  f"* series max over accepted grids (n>=800): "
                  f"{case['series_max_over_accepted']:.4%} (all pass: "
                  f"{case['all_accepted_pass']})", ""]
        if "bore_x_error" in case["rows"][0]:
            lines += ["| n | bore x sub-cell | bore x cell centre (neg. control) | "
                      "bore h2 | bore step ratio |", "|---|---|---|---|---|"]
            for r in case["rows"]:
                lines.append(
                    f"| {r['n']} | {r['bore_x_error']:.4%} | {r['bore_x_cell_error']:.4%} | "
                    f"{r['bore_h2_error']:.3e} | {r['bore_step_ratio_error']:.4%} |")
            lines += ["",
                      f"* observed order (bore sub-cell): "
                      f"{case['observed_order_bore_subcell']}",
                      f"* bore sub-cell decay: finest ratio "
                      f"{case['bore_decay']['finest_ratio']} → at floor = "
                      f"{case['bore_decay']['at_floor']}",
                      f"* worst cell-centre error over range: "
                      f"{case['bore_max_cell_error_over_range']:.4%}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", default="800,1600,3200,6400")
    ap.add_argument("--outdir", default=str(ARTIFACT_DIR))
    ap.add_argument("--figure", default=str(DOCS_DIR / "hires_convergence.png"))
    args = ap.parse_args()
    resolutions = tuple(int(x) for x in args.resolutions.split(",") if x.strip())

    payload = study(resolutions)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hires.json").write_text(json.dumps(payload, indent=1))
    figure(payload, Path(args.figure))
    docs = Path(args.figure).parent
    (docs / "SUMMARY.md").write_text(_markdown(payload))
    print(_markdown(payload))
    print(f"\nwrote {outdir / 'hires.json'} and {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())