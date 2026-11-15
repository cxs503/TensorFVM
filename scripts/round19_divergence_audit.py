#!/usr/bin/env python3
"""Round 19 -- divergence audit of the round-18 grid ladder (+ dt-control ladder).

Why this exists
---------------
``docs/case-periodic-hills/ROUND18.md`` concludes that the two finest ladder
levels (120x60, 144x72) are ``UNDEFINED`` and states, in section 5, that the
unit *confirmed* they are **not** a numerical divergence
(``diverged_at_step = null``).  Their own stored artefacts say otherwise: both
records carry ``diverged_at_step`` = 281 / 159, ``samples = 0`` and an
all-zero accumulated mean, while the stored instantaneous field of the same
run has ``max|u|`` of order ``1e281`` / ``1e169``.  The round-18 prose is
therefore wrong on two counts, and the conclusion it supports -- "refining does
not change the bubble length, hence the +40 % is not grid under-convergence" --
rests on data at only the two *coarsest* levels.

This script does the audit from the artefacts alone (no compute):

1. :func:`classify_level` -- every ladder level is classified as ``MEASURED``,
   ``DIVERGED`` (a recorded divergence; not a physical result) or
   ``NO_CROSSING`` (converged run, but the certified zero-crossing operator
   finds no separation bubble).  A level is never allowed to be reported as a
   physical ``UNDEFINED`` when the run that produced it diverged.
2. :func:`stability_map` -- collects every (mesh, dt, outcome) triple measured
   in rounds 18-19 into a stability map, which is what makes the "+40 % is not
   grid convergence" question answerable at all: the round-18 protocol
   (``dt = 0.02``) is *unstable* from ``120x60`` upwards, so the ladder has to
   be rerun with a mesh-dependent step.
3. :func:`dt_control` -- at fixed ``96x48``, compare ``dt = 0.02`` (round 18)
   with ``dt = 0.01`` (round 19): the time-step sensitivity of ``x_r/H``.

Outputs ``docs/case-periodic-hills/round19_divergence_audit.json`` plus the
figures ``fig_round19_stability.png`` and ``fig_round19_ladder.png``.

    PYTHONPATH=src python3 scripts/round19_divergence_audit.py
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
    LEGACY_X_R, OUTDIR as R18_OUTDIR, X_R_OVER_H, X_R_TOLERANCE,
    fit_grid_law, score_from_npz)

#: Field magnitude above which a stored velocity field cannot be a solution.
DIVERGED_FIELD = 1.0e6

#: round-18 ladder levels (tag ``r18_<nx>x<ny>``, dt = 0.02) + round-19
#: dt-control level.  Kept explicit so a missing artefact is visible as
#: ``status = NO_ARTEFACT`` instead of silently shrinking the ladder.
LADDER = (
    # (tag, nx, ny, dt, source-round)
    ("r18_72x36", 72, 36, 0.02, 18),
    ("r18_96x48", 96, 48, 0.02, 18),
    ("r18_120x60", 120, 60, 0.02, 18),
    ("r18_144x72", 144, 72, 0.02, 18),
    ("r19_96x48_dt0.01", 96, 48, 0.01, 19),
    ("r19_120x60_dt0.01", 120, 60, 0.01, 19),
)

#: Measurements from earlier rounds that did not leave an npz (round 18's
#: ``REJECTED_LEVELS``) -- they only carry a (mesh, dt, outcome) triple and are
#: part of the stability map, not of the ladder.
EXTERNAL_STABILITY = (
    {"mesh": [160, 80], "dt": 0.02, "outcome": "STABLE_SLOW",
     "note": "stable, 0.33 s/step, 6600 s projected > job budget (round 18)"},
    {"mesh": [192, 96], "dt": 0.02, "outcome": "DIVERGED",
     "note": "max|u| = 1.5e85 after 90 steps (round 18)"},
    {"mesh": [120, 60], "dt": 0.02, "outcome": "DIVERGED",
     "note": "round-19 local probe: max|u| = 9.6e9 at step 276 of 400"},
)


def _paths(tag: str, outdir: Path) -> tuple[Path, Path]:
    """Artefact paths for a driver tag (``round11_sst_<tag>`` stem)."""
    stem = outdir / f"round11_sst_{tag}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def classify_level(tag: str, nx: int, ny: int, dt: float, source_round: int,
                   outdir: Path = R18_OUTDIR) -> dict:
    """Classify one ladder level from its stored artefacts.

    The classification order is the point of this function: a recorded
    divergence wins over everything else, so a blown-up run can never be
    reported as a physical "no zero crossing" result.
    """
    npz, js = _paths(tag, outdir)
    if not npz.exists():
        return {"tag": tag, "mesh": [nx, ny], "dt": dt,
                "source_round": source_round, "status": "NO_ARTEFACT"}
    meta = json.loads(js.read_text()) if js.exists() else {}
    cfg = meta.get("config") or {}
    data = np.load(npz, allow_pickle=True)
    field_max = float(max(np.max(np.abs(data["u"])), np.max(np.abs(data["v"]))))
    scored = score_from_npz(npz, cfg)
    diverged_at = (cfg.get("diverged_at_step")
                   or (meta.get("config") or {}).get("diverged_at_step"))
    samples = int(meta.get("samples") or 0)
    x_r, x_s = scored["x_r_over_H"], scored["x_s_over_H"]
    # a non-finite crossing is *no measurement*, not the float nan: downstream
    # code must not be able to average, compare or plot it.
    x_r = float(x_r) if x_r is not None and np.isfinite(x_r) else None
    x_s = float(x_s) if x_s is not None and np.isfinite(x_s) else None

    x_r_was_nan = scored["x_r_over_H"] is not None and not np.isfinite(
        scored["x_r_over_H"])
    if diverged_at is not None or field_max > DIVERGED_FIELD:
        status = "DIVERGED"
        reason = (f"divergence recorded at step {diverged_at}; stored field "
                  f"max|u| = {field_max:.3e}; {samples} averaging samples")
    elif x_r is None:
        status = "NO_CROSSING"
        reason = "converged run, certified operator finds no separation bubble"
    else:
        status = "MEASURED"
        reason = "converged run with a certified separation/reattachment pair"

    return {
        "tag": tag,
        "mesh": [nx, ny],
        "dt": dt,
        "h_over_H": 9.0 / nx,
        "source_round": source_round,
        "status": status,
        "reason": reason,
        "non_finite_crossing": bool(x_r_was_nan),
        "diverged_at_step": diverged_at,
        "samples": samples,
        "stored_field_max": field_max,
        "x_s_over_H": x_s,
        "x_r_over_H": x_r,
        "x_r_error_vs_reference": (abs(x_r - X_R_OVER_H) / X_R_OVER_H
                                   if status == "MEASURED" else None),
        "x_r_error_vs_legacy": (abs(x_r - LEGACY_X_R) / LEGACY_X_R
                                if status == "MEASURED" else None),
        "x_r_pass": bool(status == "MEASURED"
                         and abs(x_r - X_R_OVER_H) / X_R_OVER_H <= X_R_TOLERANCE),
        "profile_pass": scored["score"].get("profile_pass"),
        "profile_stations": scored["score"].get("profile_stations"),
        "wall_clock_s": meta.get("wall_clock_s"),
    }


def stability_map(levels) -> dict:
    """Stability of the SST protocol in the (h, dt) plane."""
    entries = []
    for lv in levels:
        if lv["status"] == "NO_ARTEFACT":
            continue
        entries.append({"mesh": lv["mesh"], "h_over_H": lv["h_over_H"],
                        "dt": lv["dt"], "outcome": lv["status"],
                        "source_round": lv["source_round"]})
    entries.extend(EXTERNAL_STABILITY)
    unstable = [e for e in entries if e["outcome"] == "DIVERGED"]
    stable = [e for e in entries if e["outcome"] in ("MEASURED", "NO_CROSSING")]
    return {
        "entries": entries,
        "unstable": unstable,
        "stable": stable,
        "protocol_dt_unstable_at": sorted(
            {e["mesh"][0] for e in unstable if e["dt"] == 0.02}),
        "max_stable_dt_per_mesh": {
            f"{e['mesh'][0]}x{e['mesh'][1]}": e["dt"] for e in stable},
    }


def dt_control(levels) -> dict:
    """Time-step sensitivity of ``x_r/H`` at fixed ``96x48``."""
    by_dt = {lv["dt"]: lv for lv in levels
             if lv["mesh"] == [96, 48] and lv["status"] == "MEASURED"}
    if len(by_dt) < 2:
        return {"available": False,
                "reason": "need the same mesh measured at two time steps",
                "levels_present": sorted(by_dt)}
    coarse_dt, fine_dt = max(by_dt), min(by_dt)
    a, b = by_dt[coarse_dt]["x_r_over_H"], by_dt[fine_dt]["x_r_over_H"]
    rel = abs(a - b) / abs(b)
    return {
        "available": True,
        "mesh": [96, 48],
        "x_r_over_H": {str(coarse_dt): a, str(fine_dt): b},
        "relative_change": rel,
        "time_step_converged_5pct": bool(rel <= 0.05),
        "interpretation": ("halving dt changes x_r/H by "
                           f"{rel:.1%}; the ladder's mesh trend can be read "
                           "independently of the step" if rel <= 0.05 else
                           "halving dt changes x_r/H by "
                           f"{rel:.1%}: the dt difference between ladder "
                           "levels confounds the mesh trend"),
    }


def analyse(outdir: Path = R18_OUTDIR, figures: bool = True) -> dict:
    levels = [classify_level(*spec, outdir=outdir) for spec in LADDER]
    measured = [lv for lv in levels if lv["status"] == "MEASURED"]
    diverged = [lv for lv in levels if lv["status"] == "DIVERGED"]
    law = fit_grid_law([(lv["h_over_H"], lv["x_r_over_H"]) for lv in measured])
    ctrl = dt_control(levels)
    stab = stability_map(levels)

    band = [X_R_OVER_H * (1 - X_R_TOLERANCE), X_R_OVER_H * (1 + X_R_TOLERANCE)]
    inside = [lv for lv in measured if lv["x_r_pass"]]
    finest = min(measured, key=lambda lv: lv["h_over_H"]) if measured else None
    verdict = {
        "reference_x_r_over_H": X_R_OVER_H,
        "reference_band": band,
        "n_measured_levels": len(measured),
        "n_diverged_levels": len(diverged),
        "diverged_tags": [lv["tag"] for lv in diverged],
        "any_level_inside_band": bool(inside),
        "finest_measured": None if finest is None else {
            "tag": finest["tag"], "h_over_H": finest["h_over_H"],
            "x_r_over_H": finest["x_r_over_H"],
            "error": finest["x_r_error_vs_reference"]},
        "grid_law_admissible": bool(law.get("admissible")),
        "grid_law": {k: law.get(k) for k in
                     ("admissible", "reason", "observed_order_p",
                      "extrapolated_x_r_over_H", "monotone_in_h",
                      "levels_coarse_to_fine")},
        "acceptance_pass": bool(inside),
        "honest_summary": _honest_summary(measured, diverged, law, ctrl),
    }
    out = {"round": 19, "case": "CASE-FVM-PERIODIC-HILLS",
           "purpose": ("audit the round-18 ladder for divergence and close the "
                       "grid-convergence question with a stable-step ladder"),
           "protocol_dt": 0.02, "levels": levels, "stability": stab,
           "dt_control": ctrl, "verdict": verdict}
    _write_json(outdir / "round19_divergence_audit.json", out)
    if figures:
        _figures(out, outdir)
    return out


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


def _figures(out, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- stability map -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style = {"MEASURED": ("o", "tab:green", "measured"),
             "NO_CROSSING": ("s", "tab:olive", "converged, no bubble"),
             "DIVERGED": ("x", "tab:red", "diverged"),
             "STABLE_SLOW": ("^", "tab:blue", "stable, over budget")}
    seen = set()
    for e in out["stability"]["entries"]:
        marker, colour, label = style[e["outcome"]]
        h = e.get("h_over_H", 9.0 / e["mesh"][0])
        ax.scatter(h, e["dt"], marker=marker, s=70, color=colour,
                   label=None if label in seen else label)
        seen.add(label)
    ax.set_xlabel("cell size  h/H = 9/nx")
    ax.set_ylabel("time step  dt")
    ax.invert_xaxis()
    ax.set_title("round 19: SST protocol stability, dt = 0.02 loses 120x60 up")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_round19_stability.png", dpi=130)
    plt.close(fig)

    # --- ladder with divergence flags --------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    band = out["verdict"]["reference_band"]
    ax.axhspan(band[0], band[1], color="tab:blue", alpha=0.15,
               label="reference band 5.41 +/-10 %")
    ax.axhline(out["verdict"]["reference_x_r_over_H"], color="tab:blue",
               lw=1, ls="--")
    for tag, colour, marker in (("r18_", "tab:red", "o"),
                                ("r19_", "tab:green", "D")):
        pts = sorted([(lv["h_over_H"], lv["x_r_over_H"])
                      for lv in out["levels"]
                      if lv["tag"].startswith(tag) and lv["status"] == "MEASURED"],
                     reverse=True)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                    color=colour, label=f"{tag.strip('_')} ladder")
    for lv in out["levels"]:
        if lv["status"] == "DIVERGED":
            ax.annotate("diverged", (lv["h_over_H"], band[1] * 0.55),
                        rotation=90, fontsize=7, color="tab:red",
                        ha="center")
    ax.set_xlabel("cell size  h/H = 9/nx")
    ax.set_ylabel("reattachment length  $x_r/H$")
    ax.invert_xaxis()
    ax.set_title("round 19: measured ladder vs Breuer 2009 (Re = 2800)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_round19_ladder.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    out = analyse()
    print(json.dumps({"verdict": out["verdict"],
                      "dt_control": out["dt_control"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())