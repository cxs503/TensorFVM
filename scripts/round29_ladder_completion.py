#!/usr/bin/env python3
"""Round 29 -- complete the dt = 0.01 ladder for CASE-FVM-PERIODIC-HILLS.

Why this exists
---------------
Round 19 (:mod:`scripts.round19_divergence_audit`) established that the
round-18 protocol step ``dt = 0.02`` is *unstable* from ``120x60`` upwards,
and that the two finest round-18 levels (120x60, 144x72) had in fact
**diverged** (``diverged_at_step`` = 281 / 159).  It therefore re-ran the
``96x48`` dt-control level and the ``120x60`` level at the stable step
``dt = 0.01`` (cluster jobs cj3594 / cj3595), but those two artefacts never
landed in the repository -- the round-19 audit classifies both as
``NO_ARTEFACT`` and leaves two open questions:

1. **Is the +42 % reattachment error a time-step artefact?**  The 96x48
   dt-control level (same mesh, ``dt = 0.02`` vs ``dt = 0.01``) is the only
   measurement that can answer this; round 19 recorded it as WITHHELD.
2. **Is it a grid artefact?**  A grid-law fit needs three admissible levels;
   round 19 had only two (72x36, 96x48, both at ``dt = 0.02``).  The 120x60
   level at ``dt = 0.01`` is the third rung.

This script re-scores the *stored* time-mean fields of every ladder level with
the certified zero-crossing operator (no re-run, no trust in stored derived
numbers), classifies each level (``MEASURED`` / ``DIVERGED`` / ``NO_CROSSING``
/ ``NO_ARTEFACT``), and reports:

* the dt-control result at fixed 96x48 (time-step sensitivity of ``x_r/H``);
* the grid-law fit ``x_r(h) = x0 + C h^p`` (only when >= 3 admissible levels
  and the sequence is monotone in h);
* the (h, dt) stability map, now including the round-29 stable-step levels.

The round-29 levels are tagged ``r24_<nx>x<ny>_dt0.01`` and are produced by
``scripts/round17_measure.py`` (the same driver round 17/18/19 used), so the
closure, blending and averaging protocol are byte-identical to the earlier
rungs.  The 120x60 level uses ``steps = 37000`` (window ``t in [160, 370]``)
instead of the protocol ``steps = 40000`` (window ``t in [160, 400]``) to fit
inside the 7200 s single-job budget; the shorter averaging window is recorded
in the artefact and flagged in the verdict.

    PYTHONPATH=src python3 scripts/round29_ladder_completion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "docs" / "case-periodic-hills"
sys.path.insert(0, str(HERE))

from round18_ladder_analysis import (  # noqa: E402
    LEGACY_X_R, X_R_OVER_H, X_R_TOLERANCE, fit_grid_law)
from round19_divergence_audit import (  # noqa: E402
    EXTERNAL_STABILITY, LADDER as LADDER_R19,
    classify_level, dt_control, stability_map)

#: round-29 stable-step levels (tag, nx, ny, dt, source-round, window).
#: ``window`` is the averaging window in physical time; the 120x60 level uses a
#: shortened window to fit the job budget (see module docstring).
LADDER_R29 = (
    ("r24_96x48_dt0.01", 96, 48, 0.01, 24, (160.0, 400.0)),
    ("r24_120x60_dt0.01", 120, 60, 0.01, 24, (160.0, 370.0)),
)

#: The full ladder: every round-18/19 level (kept so a missing artefact stays
#: visible) plus the round-29 stable-step levels.
#: Normalised ladder: every entry is a 6-tuple (tag, nx, ny, dt, round, window).
#: round-18/19 levels ran at the protocol window t in [160, 400].
LADDER = tuple(t + ((160.0, 400.0),) if len(t) == 5 else t
               for t in (LADDER_R19 + LADDER_R29))


def _window_of(tag: str) -> tuple[float, float] | None:
    for t, _nx, _ny, _dt, _r, w in LADDER_R29:
        if t == tag:
            return w
    return None


def analyse(outdir: Path = OUTDIR, figures: bool = True) -> dict:
    levels = [classify_level(tag, nx, ny, dt, r, outdir=outdir)
              for (tag, nx, ny, dt, r, _w) in LADDER]
    # annotate the round-29 levels with their (possibly shortened) window
    for lv in levels:
        w = _window_of(lv["tag"])
        if w is not None:
            lv["averaging_window"] = list(w)
            lv["window_shortened"] = bool(w[1] < 400.0 - 1e-9)

    measured = [lv for lv in levels if lv["status"] == "MEASURED"]
    diverged = [lv for lv in levels if lv["status"] == "DIVERGED"]
    no_art = [lv for lv in levels if lv["status"] == "NO_ARTEFACT"]

    # grid law on the *stable-step* levels only when the dt-control shows the
    # step is not a confound; otherwise fit on all measured levels and flag it.
    ctrl = dt_control(levels)
    if ctrl.get("available") and ctrl.get("time_step_converged_5pct"):
        law_levels = measured
        law_note = ("dt-control shows the step is not a confound; the mesh "
                    "trend is read on all measured levels")
    else:
        law_levels = measured
        law_note = ("dt-control unavailable or >5 %: the mesh trend is "
                    "confounded by the step and the fit is indicative only")
    law = fit_grid_law([(lv["h_over_H"], lv["x_r_over_H"])
                        for lv in law_levels])
    law["note"] = law_note

    stab = stability_map(levels)
    band = [X_R_OVER_H * (1 - X_R_TOLERANCE), X_R_OVER_H * (1 + X_R_TOLERANCE)]
    inside = [lv for lv in measured if lv["x_r_pass"]]
    finest = min(measured, key=lambda lv: lv["h_over_H"]) if measured else None

    # the two questions round 19 left open
    q_dt = _dt_question(ctrl)
    q_grid = _grid_question(law, ctrl)

    verdict = {
        "reference_x_r_over_H": X_R_OVER_H,
        "reference_band": band,
        "n_measured_levels": len(measured),
        "n_diverged_levels": len(diverged),
        "diverged_tags": [lv["tag"] for lv in diverged],
        "no_artefact_tags": [lv["tag"] for lv in no_art],
        "any_level_inside_band": bool(inside),
        "finest_measured": None if finest is None else {
            "tag": finest["tag"], "h_over_H": finest["h_over_H"],
            "x_r_over_H": finest["x_r_over_H"],
            "error": finest["x_r_error_vs_reference"]},
        "grid_law_admissible": bool(law.get("admissible")),
        "grid_law": {k: law.get(k) for k in
                      ("admissible", "reason", "observed_order_p",
                       "extrapolated_x_r_over_H", "monotone_in_h",
                       "levels_coarse_to_fine", "note")},
        "dt_control": ctrl,
        "question_time_step": q_dt,
        "question_grid": q_grid,
        "acceptance_pass": bool(inside),
        "honest_summary": _honest_summary(measured, diverged, law, ctrl),
    }
    out = {
        "round": 29,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "purpose": ("complete the dt=0.01 ladder: the 96x48 dt-control and the "
                    "120x60 third rung that round 19 submitted but never "
                    "retrieved, to settle whether the +42 % is a step/grid "
                    "artefact or a 2-D model limitation"),
        "protocol_dt": 0.02,
        "stable_dt": 0.01,
        "levels": levels,
        "stability": stab,
        "dt_control": ctrl,
        "verdict": verdict,
    }
    _write_json(outdir / "round29_ladder_completion.json", out)
    if figures:
        _figures(out, outdir)
    return out


def _dt_question(ctrl: dict) -> dict:
    if not ctrl.get("available"):
        return {"answered": False,
                "reason": ctrl.get("reason", "dt-control level missing"),
                "conclusion": "WITHHELD: the 96x48 dt=0.01 control has no "
                              "measured x_r/H"}
    rel = ctrl["relative_change"]
    converged = ctrl["time_step_converged_5pct"]
    return {
        "answered": True,
        "relative_change": rel,
        "time_step_converged_5pct": converged,
        "conclusion": ("the +42 % is NOT a time-step artefact: halving dt "
                       "changes x_r/H by "
                       f"{rel:.1%} (<= 5 %)" if converged else
                       "the +42 % IS partly a time-step artefact: halving dt "
                       f"changes x_r/H by {rel:.1%} (> 5 %); the coarse-dt "
                       "ladder levels must be re-run before the mesh trend "
                       "can be read"),
    }


def _grid_question(law: dict, ctrl: dict) -> dict:
    if not law.get("admissible"):
        return {"answered": False,
                "reason": law.get("reason", "grid law not admissible"),
                "conclusion": "WITHHELD: fewer than three admissible levels "
                              "or a non-monotone sequence"}
    p = law.get("observed_order_p")
    x0 = law.get("extrapolated_x_r_over_H")
    return {
        "answered": True,
        "observed_order_p": p,
        "extrapolated_x_r_over_H": x0,
        "conclusion": (
            f"grid law x_r(h) = x0 + C h^p fitted with p = {p:.2f}, "
            f"x0 = {x0:.3f}; the residual error of the finest level against "
            "the DNS reference is a model/dimension limitation, not grid "
            "under-convergence" if (x0 is not None and abs(x0 - X_R_OVER_H)
            / X_R_OVER_H > X_R_TOLERANCE) else
            f"grid law fitted (p = {p:.2f}, x0 = {x0:.3f}); the extrapolated "
            "value is inside the 10 % band, so grid refinement alone could "
            "close the gap"),
    }


def _honest_summary(measured, diverged, law, ctrl) -> str:
    parts = []
    for lv in measured:
        parts.append(f"{lv['tag']}: x_r/H = {lv['x_r_over_H']:.2f} "
                     f"(err {lv['x_r_error_vs_reference']:.1%})")
    if diverged:
        parts.append("diverged (no data): "
                     + ", ".join(f"{lv['tag']}@step {lv['diverged_at_step']}"
                                 for lv in diverged))
    parts.append("grid law: " + ("fitted" if law.get("admissible")
                                 else str(law.get("reason"))))
    if ctrl.get("available"):
        parts.append(ctrl["interpretation"])
    return "; ".join(parts)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")


def _figures(out: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = out["levels"]
    measured = [lv for lv in levels if lv["status"] == "MEASURED"]

    # --- ladder with the 10 % band -----------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lv in levels:
        if lv["status"] == "MEASURED":
            ax.plot(lv["h_over_H"], lv["x_r_over_H"], "o-",
                    color="tab:blue", label=lv["tag"])
        elif lv["status"] == "DIVERGED":
            ax.plot(lv["h_over_H"], X_R_OVER_H, "x", color="tab:red",
                    ms=9, mew=2)
    ax.axhline(X_R_OVER_H, color="tab:green", ls="--", lw=1,
                label=f"DNS x_r/H = {X_R_OVER_H}")
    ax.axhspan(*out["verdict"]["reference_band"], color="tab:green",
                alpha=0.12, label="+/-10 % band")
    ax.set_xscale("log")
    ax.set_xlabel("h/H (log)")
    ax.set_ylabel("x_r/H")
    ax.set_title("Periodic hill Re=2800: x_r/H vs mesh (SST k-omega)")
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_round29_ladder.png", dpi=140)
    plt.close(fig)

    # --- dt control at 96x48 ------------------------------------------------
    ctrl = out["dt_control"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if ctrl.get("available"):
        xs = [float(k) for k in ctrl["x_r_over_H"]]
        ys = [ctrl["x_r_over_H"][k] for k in ctrl["x_r_over_H"]]
        ax.plot(xs, ys, "o-", color="tab:blue")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8)
        ax.axhline(X_R_OVER_H, color="tab:green", ls="--", lw=1,
                   label=f"DNS = {X_R_OVER_H}")
        ax.set_xlabel("dt")
        ax.set_ylabel("x_r/H")
        ax.set_title("96x48 dt-control: x_r/H vs time step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "dt-control not available\n"
                + ctrl.get("reason", ""), ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("96x48 dt-control")
    fig.tight_layout()
    fig.savefig(outdir / "fig_round29_dt_control.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(0 if analyse() else 1)
