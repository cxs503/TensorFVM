#!/usr/bin/env python3
"""Round-17 grid-convergence study of CASE-FVM-PERIODIC-HILLS (patankar HEAD).

Round 16 (m21316) fixed the ``omega`` sink linearisation to the model-consistent
Patankar coefficient ``beta*omega`` and ran the acceptance A/B at a single
96x48 / 20000-step grid, where the reattachment length came back **UNDEFINED**
(no reattachment crossing in the whole statistical window t=160..400, 0 %
coverage).  Round 16 explicitly left the grid-convergence question open
("本单只做了 96x48 一档网格，没有网格收敛阶梯；因此 patankar 的 UNDEFINED
是否随加密消失，本轮不作断言").

This driver closes that verification debt: it runs the *same* patankar
configuration (same closure, same sink coefficient, same dt, same statistical
window) on a grid ladder 48x24 -> 96x48 -> 128x64 and asks, with measurements
instead of assertions:

  G1  Does ``x_r/H`` become *defined* (a reattachment crossing inside the
      window) as the grid is refined, and does it move toward the ERCOFTAC
      reference ``5.41``?
  G2  Is the reattachment-coverage ``0 %`` (UNDEFINED) at 96x48 a *grid
      artifact* (numerical diffusion suppressing the bubble on coarse grids)
      or a *model-dimension* signature (2-D URANS+SST cannot hold a
      reattachment bubble at this Re)?
  G3  Does the ``y/H = 0.5`` mean profile improve with refinement (more
      admissible stations inside the 10 % band)?

The verdict is the round-15 scoring path (``score_hill``: 5.41 + the distributed
DNS ``y/H = 0.5`` table), never the payload's own legacy 5.34 gate.  The
coarse-grid 32x16 / 2000-step trend probe of round 16 (``round16_local_ab.json``)
is included as a documented non-stationary data point, not as a gate.

Artefacts read (written by ``scripts/round16_acceptance.py``):
    r16g_patankar_48x24.json   (this round, cj3437)
    r16_patankar_96x48.json    (round 16, m21316)
    r16g_patankar_128x64.json  (this round, cj3438)
    round16_local_ab.json      (round 16 trend probe, 32x16 / 2000 steps)

Outputs:
    docs/case-periodic-hills/round17_grid_convergence.json   (table, zero hand-copy)
    docs/case-periodic-hills/fig_round17_grid_convergence.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.periodic_hill_reference import (
    STATIONS, X_R_OVER_H, X_R_TOLERANCE, load_profile, y05_table)

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
SOURCES = [OUT, Path("/tmp/r16"), Path("/tmp/r17")]

#: the grid-ladder artefacts, coarse -> fine (all patankar / HEAD)
LADDER = [
    "round16_local_ab.json",      # 32x16 / 2000 steps (trend probe, non-stationary)
    "r16g_patankar_48x24.json",   # 48x24 / 20000 steps (this round)
    "r16_patankar_96x48.json",    # 96x48 / 20000 steps (round 16)
    "r16g_patankar_128x64.json",  # 128x64 / 20000 steps (this round)
]


def load(name):
    for root in SOURCES:
        path = root / name
        if path.exists():
            return json.loads(path.read_text())
    return None


def _mesh_of(payload):
    """Mesh string from either the round-11 schema (``config``) or the
    round-16 trend-probe schema (top-level ``mesh``)."""
    cfg = payload.get("config") or {}
    if cfg.get("nx") is not None and cfg.get("ny") is not None:
        return f"{cfg['nx']}x{cfg['ny']}"
    m = payload.get("mesh")
    if m:
        return str(m)
    return "?"  # unknown -- caller must not sort on it


def _steps_of(payload):
    cfg = payload.get("config") or {}
    return cfg.get("steps") or payload.get("steps")


def row(name):
    """One ladder rung, normalised to the round-15 scoring schema.

    Two artefact schemas are accepted:

    * the round-11/16 acceptance schema (``time_averaged`` / ``score_hill`` /
      ``config`` blocks) -- the 48x24 / 96x48 / 128x64 runs;
    * the round-16 trend-probe schema (``round16_local_ab.json``: a ``rows``
      list with ``x_r_over_H`` / ``profile_pass`` at the row level) -- the
      32x16 / 2000-step probe.  The patankar row is ``rows[0]``.
    """
    payload = load(name)
    if payload is None:
        return None
    # --- trend-probe schema (round16_local_ab) --------------------------- #
    if "rows" in payload:
        probe = next((r for r in payload["rows"] if r.get("variant") == "patankar"),
                     payload["rows"][0])
        return {
            "artefact": name,
            "variant": probe.get("variant", "patankar"),
            "sink_coefficient_ratio": None,   # not measured on the trend probe
            "mesh": str(probe.get("mesh") or payload.get("mesh")),
            "steps": probe.get("steps") or payload.get("steps"),
            "window": None,
            "x_r_over_H": probe.get("x_r_over_H"),
            "x_r_over_H_reference": X_R_OVER_H,
            "error_vs_5.41": (abs(probe["x_r_over_H"] - X_R_OVER_H) / X_R_OVER_H
                              if probe.get("x_r_over_H") is not None else None),
            "crossings": [],
            "reattachment_coverage": None,
            "stationarity_spread_rel": None,
            "profile_pass": probe.get("profile_pass"),
            "max_nu_t_ratio": probe.get("max_nu_t_ratio"),
            "body_force": None,
            "wall_clock_s": None,
            "profile": {},
            "note": "trend probe (2000 steps = 40 time units, non-stationary)",
        }
    # --- round-11/16 acceptance schema ----------------------------------- #
    score = payload.get("score_hill") or {}
    ta = payload.get("time_averaged") or {}
    xr = ta.get("x_r_over_H")
    xr = float(xr) if isinstance(xr, (int, float)) and np.isfinite(float(xr)) else None
    crossings = ta.get("crossings") or []
    return {
        "artefact": name,
        "variant": payload.get("variant", "patankar"),
        "sink_coefficient_ratio": (payload.get("sink_coefficient_ratio") or {}).get("ratio"),
        "mesh": _mesh_of(payload),
        "steps": _steps_of(payload),
        "window": (payload.get("config") or {}).get("averaging_window"),
        "x_r_over_H": xr,
        "x_r_over_H_reference": X_R_OVER_H,
        "error_vs_5.41": (abs(xr - X_R_OVER_H) / X_R_OVER_H) if xr is not None else None,
        "crossings": crossings,
        "reattachment_coverage": (payload.get("gate") or {}).get("reattachment_coverage"),
        "stationarity_spread_rel": (payload.get("stationarity") or {}).get(
            "x_r_over_H_block_spread_rel"),
        "profile_pass": (f"{score.get('profile_stations_pass')}"
                         f"/{score.get('profile_stations_admissible')}"),
        "max_nu_t_ratio": ta.get("max_nu_t_ratio"),
        "body_force": ta.get("body_force"),
        "wall_clock_s": payload.get("wall_clock_s"),
        "profile": payload.get("horizontal_profile_y_H_0.5") or {},
    }


def grid_resolution(mesh):
    """Cells per hill height (H) -- the natural refinement coordinate."""
    ny = int(mesh.split("x")[1])
    return ny


def figure(rows, path):
    """(a) x_r/H vs grid resolution with the 5.41 +/-10 % band;
    (b) y/H = 0.5 mean profile vs the distributed DNS."""
    band = (X_R_OVER_H * (1 - X_R_TOLERANCE), X_R_OVER_H * (1 + X_R_TOLERANCE))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) x_r/H ladder
    defined = [r for r in rows if r["x_r_over_H"] is not None]
    undefined = [r for r in rows if r["x_r_over_H"] is None]
    ax[0].axhspan(*band, color="C2", alpha=0.15, label="5.41 +/-10 % band")
    ax[0].axhline(X_R_OVER_H, color="C2", ls="--", lw=1)
    if defined:
        ax[0].plot([grid_resolution(r["mesh"]) for r in defined],
                   [r["x_r_over_H"] for r in defined], "C0o-", ms=7,
                   label="defined x_r/H (patankar)")
    if undefined:
        # UNDEFINED rungs: plot at the top of the axis with a marker
        ax[0].plot([grid_resolution(r["mesh"]) for r in undefined],
                   [band[1] * 1.15] * len(undefined), "C3x", ms=12,
                   label="UNDEFINED (0 % coverage)")
    ax[0].set_xlabel("grid resolution ny (cells / H)")
    ax[0].set_ylabel("x_r / H")
    ax[0].set_title("Reattachment length vs grid (patankar / HEAD)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # (b) y/H = 0.5 profile
    y05 = y05_table()
    ax[1].plot([s for s in y05], [y05[s] for s in y05], "ko-", ms=5,
               label="DNS Breuer/UFR 3-30 (y/H=0.5)")
    for r in rows:
        hp = r["profile"] or {}
        xs = np.asarray(hp.get("x_over_H") or [], dtype=float)
        us = np.asarray(hp.get("u_over_Ub") or [], dtype=float)
        if xs.size >= 2:
            order = np.argsort(xs)
            ax[1].plot(xs[order], us[order], "-", lw=1.2,
                       label=f"{r['mesh']} ({r['profile_pass']} pass)")
    ax[1].set_xlabel("x / H")
    ax[1].set_ylabel("u / U_b")
    ax[1].set_title("y/H = 0.5 mean profile vs DNS")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    rows = [r for r in (row(n) for n in LADDER) if r is not None]
    # sort by grid resolution (coarse -> fine); skip rungs with unknown mesh
    rows = [r for r in rows if r["mesh"] != "?"]
    rows.sort(key=lambda r: grid_resolution(r["mesh"]))

    # --- verdict --------------------------------------------------------- #
    # G1: does x_r/H become defined and move toward 5.41 with refinement?
    defined_rungs = [r for r in rows if r["x_r_over_H"] is not None]
    undefined_rungs = [r for r in rows if r["x_r_over_H"] is None]
    finest = rows[-1]
    g1 = {
        "defined_rungs": [r["mesh"] for r in defined_rungs],
        "undefined_rungs": [r["mesh"] for r in undefined_rungs],
        "finest_x_r_over_H": finest["x_r_over_H"],
        "finest_error_vs_5.41": finest["error_vs_5.41"],
        "finest_defined": finest["x_r_over_H"] is not None,
        "finest_reattachment_coverage": finest["reattachment_coverage"],
        "converges_to_reference": bool(
            finest["x_r_over_H"] is not None
            and finest["error_vs_5.41"] is not None
            and finest["error_vs_5.41"] <= X_R_TOLERANCE),
    }
    # G2: is the 96x48 UNDEFINED a grid artifact or a model-dimension signature?
    #   A grid artifact would *disappear* on the finest grid (defined + in-band).
    #   A model-dimension signature *persists* (finest still undefined or out of band).
    g2 = {
        "hypothesis": ("grid-artifact (UNDEFINED disappears on refinement)"
                        if g1["finest_defined"] and g1["converges_to_reference"]
                        else "model-dimension (2-D URANS+SST cannot hold the bubble)"),
        "evidence": {
            "96x48_undefined": any(r["mesh"] == "96x48" and r["x_r_over_H"] is None
                                   for r in rows),
            "finest_defined": g1["finest_defined"],
            "finest_in_band": g1["converges_to_reference"],
        },
    }
    # G3: profile improvement with refinement
    g3 = {
        "profile_pass_by_mesh": {r["mesh"]: r["profile_pass"] for r in rows},
        "finest_profile_pass": finest["profile_pass"],
    }

    verdict = {
        "round": 17,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "reference": {"x_r_over_H": X_R_OVER_H, "tolerance": X_R_TOLERANCE,
                      "source": "ERCOFTAC UFR 3-30 (Breuer et al. 2009, Re=2800)"},
        "ladder": rows,
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "acceptance_pass": bool(g1["converges_to_reference"]),
        "acceptance_verdict": ("PASS" if g1["converges_to_reference"]
                               else "FAIL (honest)"),
    }

    out = OUT / "round17_grid_convergence.json"
    out.write_text(json.dumps(verdict, indent=1))
    fig = figure(rows, OUT / "fig_round17_grid_convergence.png")

    # --- console table --------------------------------------------------- #
    print("=== ROUND17 grid-convergence ladder (patankar / HEAD) ===")
    print(f"{'mesh':<12}{'x_r/H':<10}{'err%':<8}{'coverage':<10}{'profile':<10}{'max_nu_t':<10}")
    for r in rows:
        xr = f"{r['x_r_over_H']:.3f}" if r["x_r_over_H"] is not None else "UNDEF"
        err = f"{r['error_vs_5.41']*100:.1f}" if r["error_vs_5.41"] is not None else "-"
        cov = (f"{r['reattachment_coverage']*100:.0f}%"
               if r["reattachment_coverage"] is not None else "-")
        pp = r["profile_pass"] if r["profile_pass"] is not None else "-"
        nut = f"{r['max_nu_t_ratio']:.1f}" if r["max_nu_t_ratio"] is not None else "-"
        print(f"{r['mesh']:<12}{xr:<10}{err:<8}{cov:<10}{pp:<10}{nut:<10}")
    print()
    print("G1:", json.dumps(g1, indent=1))
    print("G2:", json.dumps(g2, indent=1))
    print("G3:", json.dumps(g3, indent=1))
    print("verdict:", verdict["acceptance_verdict"])
    print("artefact:", out)
    print("figure:", fig)
    return 0 if verdict["acceptance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
