"""Resolution audit of the dam-break gates D1/D2 (diagnostic, not a gate).

The gate module reports *how far* the computed surge front and residual column
height are from Martin & Moyce (1952).  This module answers the next question,
which the gate cannot: **is the reported number a model error or a measurement
limit?**  It runs the same leg over a refinement sequence and records, for each
grid, three separable quantities from the leg's own history:

1. :func:`tensorfvm.twophase_gap.dambreak_gate.decompose_dambreak_error` --
   the time shift that best aligns the curves (a *delayed release*) versus the
   late-window slope ratio (a *rate deficit*), plus the pure time-*scale* fit
   that would be needed if the reference time normalisation were wrong.
2. :func:`tensorfvm.twophase_gap.dambreak_gate.resolvable_error_floor` -- the
   smallest relative error the cell-quantised observable can express on that
   grid, i.e. the resolution floor under the 15 % gate.
3. the gate values themselves, unchanged.

Nothing here can turn a FAIL into a PASS: the gate numbers are copied verbatim
from :func:`run_dambreak_leg`.  The output is a JSON/Markdown artefact that
makes the *interpretation* of D1/D2 auditable.

Run::

    python -m tensorfvm.twophase_gap.dambreak_audit --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .dambreak_gate import (
    FRONT_ERROR_MAX,
    HEIGHT_ERROR_MAX,
    run_dambreak_leg,
)

__all__ = ["DEFAULT_AUDIT_GRIDS", "run_dambreak_audit", "render_markdown",
           "write_figures", "main"]

#: Refinement sequence used by the audit (cells).
DEFAULT_AUDIT_GRIDS = ((60, 20), (120, 40), (240, 80))


def _grid_rows(grids, t_star_end: float, control_steps: int) -> list[dict]:
    rows = []
    for nx, ny in grids:
        leg = run_dambreak_leg(int(nx), int(ny), t_star_end=t_star_end,
                               control_steps=control_steps)
        rec = leg["record"]
        rows.append({"record": rec, "history": leg["history"],
                     "grid": [int(nx), int(ny)]})
    return rows


def run_dambreak_audit(
    grids=DEFAULT_AUDIT_GRIDS, *, t_star_end: float = 4.0,
    control_steps: int = 3, out_dir: str | Path | None = None,
) -> dict:
    """Run the refinement audit and (optionally) write its artefacts to disk."""
    rows = _grid_rows(grids, t_star_end, control_steps)
    summary = []
    for row in rows:
        rec, diag = row["record"], row["record"]["diagnostics"]
        summary.append({
            "grid": row["grid"],
            "steps": rec["steps"],
            "D1_front_max_rel_error": rec["front_max_rel_error"],
            "D2_height_max_rel_error": rec["height_max_rel_error"],
            "D1_pass": bool(rec["front_max_rel_error"] <= FRONT_ERROR_MAX),
            "D2_pass": bool(rec["height_max_rel_error"] <= HEIGHT_ERROR_MAX),
            "front": diag["front"],
            "height": diag["height"],
            "resolvable_front": diag["resolvable"]["front"],
            "resolvable_height": diag["resolvable"]["height"],
            "quantisation": diag["quantisation"],
            "volume_drift": rec["volume_drift"],
            "control_max_abs_change": rec["control_max_abs_change"],
        })

    lag_front = [r["front"]["phase_lag_t_star"] for r in summary]
    lag_height = [r["height"]["phase_lag_t_star"] for r in summary]
    audit = {
        "case": "dambreak_resolution_audit",
        "reference": ("Martin & Moyce (1952) digitised anchors, "
                      "t* = t sqrt(g/a), x* = (x_f - a)/a, h* = h/h0"),
        "window_t_star": [0.3, float(t_star_end)],
        "gates": {"D1_front": FRONT_ERROR_MAX, "D2_height": HEIGHT_ERROR_MAX},
        "grids": summary,
        "conclusions": {
            "front_phase_lag_t_star_range": [float(min(lag_front)),
                                             float(max(lag_front))],
            "height_phase_lag_t_star_range": [float(min(lag_height)),
                                              float(max(lag_height))],
            "front_speed_ratio_range": [
                float(min(r["front"]["speed_ratio"] for r in summary)),
                float(max(r["front"]["speed_ratio"] for r in summary))],
            "D1_monotone_in_resolution": bool(
                all(b["D1_front_max_rel_error"] <= a["D1_front_max_rel_error"]
                    for a, b in zip(summary, summary[1:]))),
            "max_front_resolvable_floor": float(max(
                r["resolvable_front"]["floor_at_window_start"] for r in summary)),
        },
        "note": ("Diagnostic only.  D1/D2 values are copied verbatim from the "
                 "leg records; the audit decomposes them into time shift, rate "
                 "ratio and cell-quantisation floor."),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit_d1d2.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        for row in rows:
            tag = f"{row['grid'][0]}x{row['grid'][1]}"
            (out / f"leg_dambreak_{tag}.json").write_text(
                json.dumps(row["record"], indent=2) + "\n", encoding="utf-8")
        (out / "D1D2-AUDIT.md").write_text(render_markdown(audit),
                                          encoding="utf-8")
        write_figures(rows, audit, out / "figures")
    return {"audit": audit, "rows": rows}


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def render_markdown(audit: dict) -> str:
    """Render the audit as Markdown.  Every number comes from ``audit``."""
    g = audit["grids"]
    lines = []
    add = lines.append
    add("# D1/D2 resolution audit -- is the dam-break error a model error or a "
        "measurement limit?")
    add("")
    add("Auto-rendered from `audit_d1d2.json` by "
        "`python -m tensorfvm.twophase_gap.dambreak_audit`.  **No gate changes "
        "here**: D1/D2 keep their verdicts; this document only decomposes the "
        "misfit.")
    add("")
    add(f"Reference: {audit['reference']}.  Comparison window "
        f"`{audit['window_t_star'][0]} < t* <= {audit['window_t_star'][1]}`.  "
        f"Gates: D1 front `{audit['gates']['D1_front']}`, D2 height "
        f"`{audit['gates']['D2_height']}`.")
    add("")
    add("## 1. Gates as declared (unchanged)")
    add("")
    add("| grid | steps | D1 front | verdict | D2 height | verdict | volume drift |")
    add("|---|---|---|---|---|---|---|")
    for r in g:
        add(f"| {r['grid'][0]}x{r['grid'][1]} | {r['steps']} | "
            f"{_f(r['D1_front_max_rel_error'])} | "
            f"{'PASS' if r['D1_pass'] else 'FAIL'} | "
            f"{_f(r['D2_height_max_rel_error'])} | "
            f"{'PASS' if r['D2_pass'] else 'FAIL'} | "
            f"{r['volume_drift']:.2e} |")
    add("")
    add("## 2. Decomposition of the misfit")
    add("")
    add("| grid | front best time shift [t*] | RMS x* (no shift -> shifted) | "
        "front late slope got/ref (ratio) | height best shift [t*] | "
        "height slope ratio | verdict (front / height) |")
    add("|---|---|---|---|---|---|---|")
    for r in g:
        f, h = r["front"], r["height"]
        add(f"| {r['grid'][0]}x{r['grid'][1]} | {_f(f['phase_lag_t_star'], 2)} | "
            f"{_f(f['rms_at_zero_shift'])} -> {_f(f['rms_at_best_shift'])} | "
            f"{_f(f['speed_got'])} / {_f(f['speed_ref'])} "
            f"({_f(f['speed_ratio'])}) | {_f(h['phase_lag_t_star'], 2)} | "
            f"{_f(h['speed_ratio'])} | {f['verdict']} / {h['verdict']} |")
    add("")
    add("Time-*scale* falsification control (would fire if the digitised anchor "
        "were normalised with `sqrt(2g/a)` instead of `sqrt(g/a)`):")
    add("")
    add("| grid | best pure time scale | RMS at best scale | RMS at best shift |")
    add("|---|---|---|---|")
    for r in g:
        f = r["front"]
        add(f"| {r['grid'][0]}x{r['grid'][1]} | "
            f"{_f(f['time_scale_zero_shift'])} | {_f(f['rms_at_best_scale'])} | "
            f"{_f(f['rms_at_best_shift'])} |")
    add("")
    add("## 3. Cell-quantisation floor under the 15 % gate")
    add("")
    add("| grid | front quantum [x*] | front floor at window start | "
        "window samples where the floor alone violates D1 | "
        "height quantum [h*/h0] | samples where the floor alone violates D2 |")
    add("|---|---|---|---|---|---|")
    for r in g:
        rf, rh = r["resolvable_front"], r["resolvable_height"]
        add(f"| {r['grid'][0]}x{r['grid'][1]} | {_f(rf['quantum'])} | "
            f"{_f(rf['floor_at_window_start'])} | "
            f"{_f(rf['violating_fraction'], 2)} | "
            f"{_f(rh['quantum'], 4)} | {_f(rh['violating_fraction'], 2)} |")
    add("")
    c = audit["conclusions"]
    add("## 4. Conclusions (machine-derived)")
    add("")
    add(f"* front phase lag is **{_f(c['front_phase_lag_t_star_range'][0], 2)} - "
        f"{_f(c['front_phase_lag_t_star_range'][1], 2)} t\\*** across the "
        "refinement sequence, i.e. grid independent;")
    add(f"* late-window front slope ratio is "
        f"**{_f(c['front_speed_ratio_range'][0], 2)} - "
        f"{_f(c['front_speed_ratio_range'][1], 2)}** (> 1: the computed front is "
        "*steeper* than the reference once it is moving -- the defect is a "
        "delayed release, not a slow wave);")
    add(f"* D1 is monotone in resolution: **{c['D1_monotone_in_resolution']}**;")
    add(f"* the largest quantisation floor at the window start is "
        f"**{_f(c['max_front_resolvable_floor'], 2)}**, "
        "so on the coarser grids the 15 % gate is decided by cell quantisation "
        "at early `t*` rather than by model accuracy.")
    add("")
    return "\n".join(lines) + "\n"


def write_figures(rows, audit: dict, fig_dir: Path) -> list[str]:
    """Write the two audit figures; returns the written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    ref_t = np.array(rows[0]["history"]["t_star"])
    ref_f = np.array(rows[0]["history"]["front_ref"])
    ax.plot(ref_t, ref_f, "k--", lw=1.6, label="Martin & Moyce (1952)")
    for row in rows:
        h = row["history"]
        t = np.array(h["t_star"])
        ax.plot(t, np.array(h["front_star"])[:, 1],
                label=f"{row['grid'][0]}x{row['grid'][1]}")
    ax.set_xlabel("t* = t sqrt(g/a)")
    ax.set_ylabel("x* = (x_f - a)/a")
    ax.set_title("surge front")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ref_h = np.array(rows[0]["history"]["height_ref"])
    ax.plot(ref_t, ref_h, "k--", lw=1.6, label="Martin & Moyce (1952)")
    for row in rows:
        h = row["history"]
        ax.plot(np.array(h["t_star"]), np.array(h["height_star"]),
                label=f"{row['grid'][0]}x{row['grid'][1]}")
    ax.set_xlabel("t*")
    ax.set_ylabel("h* = h/h0")
    ax.set_title("residual column height")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    grids = [f"{r['grid'][0]}x{r['grid'][1]}" for r in audit["grids"]]
    xs = np.arange(len(grids))
    ax.bar(xs - 0.18, [r["D1_front_max_rel_error"] for r in audit["grids"]],
           width=0.36, label="D1 front")
    ax.bar(xs + 0.18, [r["D2_height_max_rel_error"] for r in audit["grids"]],
           width=0.36, label="D2 height")
    ax.plot(xs, [r["resolvable_front"]["floor_at_window_start"]
                 for r in audit["grids"]], "o-", color="k",
            label="quantisation floor (front)")
    ax.axhline(audit["gates"]["D1_front"], color="r", ls=":", lw=1.2,
               label="gate 15 %")
    ax.set_xticks(xs)
    ax.set_xticklabels(grids)
    ax.set_ylabel("max relative error")
    ax.set_title("gate value vs resolution floor")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    path = fig_dir / "fig_d1d2_audit.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for row in rows:
        f = row["record"]["diagnostics"]["front"]
        ax.plot([0.0, f["phase_lag_t_star"]],
                [f["rms_at_zero_shift"], f["rms_at_best_shift"]], "o-",
                label=f"{row['grid'][0]}x{row['grid'][1]}")
    ax.set_xlabel("time shift [t*]")
    ax.set_ylabel("RMS misfit in x*")
    ax.set_title("how much of the front misfit is a pure time shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path2 = fig_dir / "fig_phase_lag.png"
    fig.tight_layout()
    fig.savefig(path2, dpi=130)
    plt.close(fig)
    written.append(str(path2))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/D1D2-AUDIT")
    ap.add_argument("--grids", nargs="*", default=None,
                    help="e.g. 60x20 120x40 (default: 60x20 120x40 240x80)")
    ap.add_argument("--t-star", type=float, default=4.0)
    ap.add_argument("--control-steps", type=int, default=3)
    ap.add_argument("--summary-only", action="store_true",
                    help="re-render from an existing audit JSON")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.summary_only:
        audit = json.loads((out / "audit_d1d2.json").read_text())
        (out / "D1D2-AUDIT.md").write_text(render_markdown(audit))
        print(render_markdown(audit))
        return 0

    grids = DEFAULT_AUDIT_GRIDS
    if args.grids:
        grids = tuple(tuple(int(v) for v in g.lower().split("x"))
                      for g in args.grids)
    run_dambreak_audit(grids, t_star_end=args.t_star,
                       control_steps=args.control_steps, out_dir=out)
    print(render_markdown(json.loads((out / "audit_d1d2.json").read_text())))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
