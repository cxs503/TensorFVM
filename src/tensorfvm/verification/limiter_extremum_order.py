"""Where a MUSCL limiter eats the second order: local order at a smooth crest.

Why this module exists
----------------------
``XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR`` established, with a
three-grid log--log slope on the cell-average metric, that the periodic entropy
wave of :mod:`tensorfvm.verification.cellcentre_loglog` is integrated at the
design order (``p = 2.06`` on 80/160/320).  That is a statement about the
**global** ``L1`` norm only.  Every TVD slope limiter clips the reconstruction in
the cells around a **smooth extremum** -- the forward and backward differences
change sign there, so the limited slope is (at most) ``2 min(|a|,|b|)`` and the
local truncation error drops from ``O(h^2)`` to ``O(h)`` -- and yet the global
``L1`` norm keeps reading second order, because the clipped region is only a few
cells wide: it contributes ``O(h)`` error over ``O(h)`` cells, i.e. ``O(h^2)`` in
the mean (Suresh & Huynh 1997).  A global order study therefore *cannot* detect
the loss; a study that fixes the number of cells in the window can.

This module supplies that window and reuses the anchor's machinery unchanged:
the solver (:func:`tensorfvm.verification.cellcentre_loglog.run_case`, which is
:class:`~tensorfvm.compressible.solver.EulerSolver`, MUSCL-Hancock + HLLC), the
exact closed-form cell average, the volume-weighted global norms, and
:func:`tensorfvm.verification.cellcentre_loglog.three_grid_readout` for the
three-grid log--log slope with its standard error.  No number is typed in.

Predeclared criteria (``CRITERIA`` below, fixed before the study was run)
------------------------------------------------------------------------
``C1 global order``
    The global ``L1`` cell-average slope of the MC-limited scheme is within the
    owner's 15 % band of the design order 2 on both triplets -- i.e. the anchor
    result is reproduced by this driver.
``C2 crest degradation``
    The crest-window ``Linf`` slope of the MC-limited scheme is **more than 15 %
    below** 2 (``p <= 1.70``).  This is the owner threshold applied to the local
    order: the limiter's loss has to be measured, not asserted.
``C3 unlimited control``
    With ``limiter='none'`` (centred, unlimited slope) the same crest-window
    slope is within 15 % of 2 (``1.70 <= p <= 2.30``): the degradation is caused
    by the limiter, not by the metric or the window.
``C4 away from the crest``
    A fixed 5-cell window around the **maximum-slope** cell (where the limiter is
    inactive, ``r = 1``) still reads the design order for MC (within 15 %).
    Together with C2 this localises the loss in space.
``C5 every TVD limiter``
    ``mc``, ``minmod``, ``vanleer`` and ``superbee`` all show ``p <= 1.70`` at
    the crest: this is a property of TVD limiting, not of one limiter.
``C6 magnitude``
    The crest ``Linf`` error ratio ``mc/none`` is ``>= 10`` at the finest grid and
    grows with refinement (``ratio(80/160/320) / ratio(40/80/160) >= 1.5``), i.e.
    the MC crest error behaves like ``O(h)`` against the control's ``O(h^2)``.
``C7 invisibility in the global norm``
    On the same ladder, ``p_global >= 1.70`` while ``p_crest <= 1.70``: the two
    statements coexist, which is why a global-only order claim is not a
    verification of the scheme near a smooth extremum.
``C8 crest placement``
    The result does not depend on the crest coinciding with a cell face: with a
    phase shift that puts the crest inside a cell (>= 0.12 cell from the nearest
    face on every grid) the MC crest slope is again ``<= 1.70``.  (On the
    unshifted case, ``x* = 0.25 + u t = 0.35`` lies exactly on a cell face for
    all four grids -- a degeneracy that is removed by this control instead of
    being left unexamined.)

What this module does **not** claim
-----------------------------------
It does not claim the local slope is exactly 1.  A clipped reconstruction is
first-order accurate only *in the cells where it clips*, and those cells move
under the crest as the solution is advected; the measured fixed-window slope is
reported as-is (deviations from both 1 and 2 are printed in the study) and any
"order restored" statement would need a limiter modification (e.g. the
Suresh--Huynh smooth-extremum detector), which is out of scope here.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import cellcentre_loglog as cc

__all__ = [
    "WINDOW_HALF",
    "CREST_OFFSET",
    "MIDCELL_PHASE",
    "GRIDS",
    "TRIPLETS",
    "LIMITERS",
    "SERIES",
    "CRITERIA",
    "crest_x",
    "maxslope_x",
    "cell_index",
    "window",
    "local_norms",
    "run_grid",
    "ladder",
    "study",
    "crest_magnitude_ratio",
    "criteria_gate",
]

#: Half-width (in cells) of the fixed-count window used for the local norms.
#: The count, not the physical width, must be held fixed: with ``2*WINDOW_HALF+1``
#: cells a local error that is ``O(h)`` per cell shows up as ``O(h)``, while an
#: ``h``-wide *physical* window would integrate it back to ``O(h^2)``.
WINDOW_HALF = 2

#: The crest of ``sin(k(x - u t - phase))`` sits at ``x = phase + u t + CREST_OFFSET``
#: (``CREST_OFFSET = 1/4`` period).
CREST_OFFSET = 0.25

#: Phase that moves the unshifted crest (``x* = 0.35``, a cell face for every
#: grid used here) inside a cell on **every** grid of the ladder.  A single phase
#: cannot put the crest at a cell centre for all four grids at once, so the
#: choice is made on the worst case: at ``0.009`` the crest is 0.36 / 0.72 / 0.44
#: / 0.88 of a cell away from the nearest face for n = 40 / 80 / 160 / 320, i.e.
#: never closer than 0.12 cell.
MIDCELL_PHASE = 0.009

#: Refinement ladder, coarse -> fine (cells across one wavelength).
GRIDS = (40, 80, 160, 320)

#: The two overlapping triplets the three-grid readout is applied to.
TRIPLETS = ((40, 80, 160), (80, 160, 320))

#: Limiters carrying a TVD clip (criterion ``C5``) and the unlimited control.
LIMITERS = ("mc", "minmod", "vanleer", "superbee", "none")

#: Series read off every grid: global (volume weighted, all cells) and the two
#: fixed-count local windows.
SERIES = ("global_l1", "crest_linf", "crest_l1", "maxslope_linf")

#: Declared a priori -- see the module docstring for the wording of each item.
CRITERIA = {
    "C1_global_order": "global L1 cell-average slope of mc within 15% of 2 (both triplets)",
    "C2_crest_degradation": "mc crest-window Linf slope <= 1.70 (>15% below 2)",
    "C3_unlimited_control": "limiter='none' crest-window Linf slope within 15% of 2",
    "C4_away_from_crest": "mc max-slope-window Linf slope within 15% of 2",
    "C5_every_tvd_limiter": "mc/minmod/vanleer/superbee all <= 1.70 at the crest",
    "C6_crest_magnitude": "crest Linf error ratio mc/none >= 10 and grows >= 1.5x",
    "C7_global_blindness": "p_global >= 1.70 while p_crest <= 1.70 on the same ladder",
    "C8_crest_placement": "mc crest slope <= 1.70 with the crest inside a cell (>=0.12 cell from a face)",
}

#: Order tolerance (owner 15 % standard applied to the order itself).
ORDER_TOLERANCE = cc.THRESHOLD_ACCEPTABLE

#: Local-order ceiling used by criteria C2/C5/C7/C8.
DEGRADED_MAX = 2.0 * (1.0 - ORDER_TOLERANCE)


# ----------------------------------------------------------------------
# geometry of the windows (deterministic, exact positions from the analytic wave)
# ----------------------------------------------------------------------
def crest_x(t: float, velocity: float = cc.VELOCITY, phase: float = 0.0) -> float:
    """Exact ``x`` of the density maximum of ``1 + A sin(k(x - u t - phase))``."""
    return (phase + velocity * t + CREST_OFFSET) % 1.0


def maxslope_x(t: float, velocity: float = cc.VELOCITY, phase: float = 0.0) -> float:
    """Exact ``x`` of the largest ``|d rho/dx|`` (zero crossing of the wave)."""
    return (phase + velocity * t) % 1.0


def cell_index(nx: int, x: float) -> int:
    """Index of the cell of ``[0,1)`` whose centre is nearest ``x``."""
    centres = (np.arange(nx) + 0.5) / nx
    return int(np.argmin(np.abs(centres - (x % 1.0))))


def window(nx: int, centre: int, half: int = WINDOW_HALF) -> List[int]:
    """``2*half+1`` periodic cell indices centred on ``centre``.

    The **count** is grid independent on purpose (see ``WINDOW_HALF``).
    """
    if half < 0:
        raise ValueError("half must be non-negative")
    return [(centre + k) % nx for k in range(-half, half + 1)]


def local_norms(field: np.ndarray, reference: np.ndarray,
                indices: Sequence[int]) -> Dict[str, float]:
    """``L1``/``L2``/``Linf`` of ``field - reference`` on a fixed cell set.

    Unweighted by construction: the set has a grid-independent number of cells,
    so the mean is the local norm and not a (shrinking) volume average.  The
    global volume-weighted norms stay in
    :func:`tensorfvm.verification.cellcentre_loglog.error_norms`.
    """
    f = np.asarray(field, dtype=float)
    r = np.asarray(reference, dtype=float)
    if f.shape != r.shape or f.ndim != 1:
        raise ValueError("local_norms requires two 1-D arrays of equal shape")
    idx = np.asarray(list(indices), dtype=int)
    if idx.size == 0:
        raise ValueError("empty window")
    err = np.abs(f[idx] - r[idx])
    return {"l1": float(err.mean()), "l2": float(np.sqrt((err ** 2).mean())),
            "linf": float(err.max()), "cells": int(idx.size)}


# ----------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------
def run_grid(nx: int, limiter: str, *, order: int = 2, cfl: float = 0.4,
             t_end: float = 0.2, phase: float = 0.0) -> Dict:
    """One grid of one limiter, reduced to the four series used by the study."""
    row = cc.run_case(nx, order=order, cfl=cfl, t_end=t_end, limiter=limiter,
                      phase=phase)
    average = row["rho_exact_average"]
    rho = row["rho"]
    widths = np.full(int(nx), row["h"])
    error = rho - average
    t = row["t"]
    crest = window(nx, cell_index(nx, crest_x(t, phase=phase)))
    slope = window(nx, cell_index(nx, maxslope_x(t, phase=phase)))
    crest_norms = local_norms(rho, average, crest)
    slope_norms = local_norms(rho, average, slope)
    return {
        "nx": int(nx), "h": row["h"], "t": t, "steps": row["steps"],
        "limiter": limiter, "phase": float(phase),
        "crest_cell": int(crest[WINDOW_HALF]), "maxslope_cell": int(slope[WINDOW_HALF]),
        "crest_x": crest_x(t, phase=phase),
        "global_l1": float(cc.error_norms(rho, average, widths)["l1"]),
        "crest_linf": crest_norms["linf"],
        "crest_l1": crest_norms["l1"],
        "maxslope_linf": slope_norms["linf"],
        # raw fields are kept only when explicitly asked for (figures)
        "error": error.tolist(),
    }


def ladder(limiters: Iterable[str] = LIMITERS, grids: Sequence[int] = GRIDS,
           *, phase: float = 0.0, order: int = 2, cfl: float = 0.4,
           t_end: float = 0.2) -> Dict[str, List[Dict]]:
    """Run every limiter on every grid (coarse -> fine), keyed by limiter."""
    out: Dict[str, List[Dict]] = {}
    for limiter in limiters:
        out[limiter] = [run_grid(int(n), limiter, order=order, cfl=cfl,
                                 t_end=t_end, phase=phase) for n in grids]
    return out


def study(rows: Sequence[Dict], triplets: Sequence[Sequence[int]] = TRIPLETS,
          *, theory_order: float = 2.0) -> Dict:
    """Three-grid readouts of every series of one ladder, per triplet."""
    grids = [int(r["nx"]) for r in rows]
    out: Dict = {"grids": grids, "theory_order": float(theory_order),
                 "rows": [{k: v for k, v in r.items() if k != "error"} for r in rows],
                 "readouts": []}
    for triplet in triplets:
        triplet = [int(n) for n in triplet]
        if not set(triplet) <= set(grids):
            raise ValueError(f"triplet {triplet} not in ladder {grids}")
        selected = [next(r for r in rows if r["nx"] == n) for n in triplet]
        for series in SERIES:
            values = [float(r[series]) for r in selected]
            read = cc.three_grid_readout(triplet, values,
                                        theory_order=theory_order)
            read.update({"series": series, "triplet": triplet,
                         "values": values})
            out["readouts"].append(read)
    return out


def crest_magnitude_ratio(studies: Dict[str, Dict]) -> List[Dict]:
    """Per grid: ``crest_linf(mc) / crest_linf(none)`` -- the O(h)/O(h^2) witness."""
    mc = studies["mc"]["rows"]
    ref = studies["none"]["rows"]
    out = []
    for a, b in zip(mc, ref):
        if a["nx"] != b["nx"]:
            raise ValueError("ladders are not aligned")
        out.append({"nx": a["nx"], "mc": a["crest_linf"], "none": b["crest_linf"],
                    "ratio": a["crest_linf"] / b["crest_linf"]})
    return out


# ----------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------
def _read(study_row: Dict, series: str, triplet: Sequence[int]) -> Dict:
    triplet = [int(n) for n in triplet]
    for read in study_row["readouts"]:
        if read["series"] == series and read["triplet"] == triplet:
            return read
    raise KeyError(f"no readout {series} on {triplet}")


def _within(value: float, target: float, tolerance: float = ORDER_TOLERANCE) -> bool:
    return bool(math.isfinite(value) and abs(value - target) <= tolerance * target)


def criteria_gate(studies: Dict[str, Dict], ratios: Sequence[Dict],
                  *, phase_study: Optional[Dict] = None) -> List[Dict]:
    """Evaluate ``CRITERIA`` against the study data; one record per criterion.

    Every record carries the measured value it was decided on, so a FAIL cannot
    be reported without the number that caused it.
    """
    fine, coarse = list(TRIPLETS[1]), list(TRIPLETS[0])
    out: List[Dict] = []

    def add(name, passed, measured, detail=""):
        out.append({"criterion": name, "statement": CRITERIA[name],
                    "passed": bool(passed), "measured": measured, "detail": detail})

    mc_global_f = _read(studies["mc"], "global_l1", fine)["p_loglog"]
    mc_global_c = _read(studies["mc"], "global_l1", coarse)["p_loglog"]
    add("C1_global_order", _within(mc_global_f, 2.0) and _within(mc_global_c, 2.0),
        {"fine": mc_global_f, "coarse": mc_global_c, "triplets": [fine, coarse]},
        "global volume-weighted L1 of the cell-average error")

    mc_crest_f = _read(studies["mc"], "crest_linf", fine)["p_loglog"]
    mc_crest_c = _read(studies["mc"], "crest_linf", coarse)["p_loglog"]
    add("C2_crest_degradation", mc_crest_f <= DEGRADED_MAX and mc_crest_c <= DEGRADED_MAX,
        {"fine": mc_crest_f, "coarse": mc_crest_c, "ceiling": DEGRADED_MAX,
         "dev_vs_2": abs(mc_crest_f - 2.0) / 2.0, "dev_vs_1": abs(mc_crest_f - 1.0)},
        "fixed 5-cell window centred on the crest cell")

    none_crest_f = _read(studies["none"], "crest_linf", fine)["p_loglog"]
    none_crest_c = _read(studies["none"], "crest_linf", coarse)["p_loglog"]
    add("C3_unlimited_control", _within(none_crest_f, 2.0) and _within(none_crest_c, 2.0),
        {"fine": none_crest_f, "coarse": none_crest_c},
        "same window, centred (unlimited) slope")

    mc_slope_f = _read(studies["mc"], "maxslope_linf", fine)["p_loglog"]
    mc_slope_c = _read(studies["mc"], "maxslope_linf", coarse)["p_loglog"]
    add("C4_away_from_crest", _within(mc_slope_f, 2.0) and _within(mc_slope_c, 2.0),
        {"fine": mc_slope_f, "coarse": mc_slope_c},
        "5-cell window at the maximum-slope cell (limiter inactive)")

    per_limiter = {}
    for limiter in ("mc", "minmod", "vanleer", "superbee"):
        if limiter in studies:
            per_limiter[limiter] = {
                "fine": _read(studies[limiter], "crest_linf", fine)["p_loglog"],
                "coarse": _read(studies[limiter], "crest_linf", coarse)["p_loglog"]}
    add("C5_every_tvd_limiter",
        bool(per_limiter) and all(v["fine"] <= DEGRADED_MAX for v in per_limiter.values()),
        per_limiter, "crest-window slope per TVD limiter")

    by_nx = {r["nx"]: r for r in ratios}
    ratio_fine = by_nx[fine[-1]]["ratio"]
    ratio_coarse = by_nx[coarse[-1]]["ratio"]
    growth = ratio_fine / ratio_coarse if ratio_coarse > 0 else float("nan")
    add("C6_crest_magnitude", ratio_fine >= 10.0 and growth >= 1.5,
        {"ratio_finest": ratio_fine, "ratio_at_coarse_finest": ratio_coarse,
         "growth": growth})

    add("C7_global_blindness",
        mc_global_f >= DEGRADED_MAX and mc_crest_f <= DEGRADED_MAX,
        {"p_global": mc_global_f, "p_crest": mc_crest_f,
         "contrast": mc_global_f - mc_crest_f},
        "same ladder, same metric, same window family")

    if phase_study is not None:
        p_mid = _read(phase_study, "crest_linf", fine)["p_loglog"]
        p_mid_c = _read(phase_study, "crest_linf", coarse)["p_loglog"]
        add("C8_crest_placement", p_mid <= DEGRADED_MAX and p_mid_c <= DEGRADED_MAX,
            {"fine": p_mid, "coarse": p_mid_c, "phase": MIDCELL_PHASE,
             "crest_x": crest_x(float(phase_study["rows"][0]["t"]), phase=MIDCELL_PHASE)},
            "crest inside a cell instead of on a face")
    else:
        add("C8_crest_placement", False, {"phase": None},
            "phase control not run -- reported as not_run, not as pass")
    return out