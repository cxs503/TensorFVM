"""Long-time validity of the Ritter/Stoker dam-break case beyond ``t* = 2``.

Why this module exists
----------------------
``CASE-FVM-DAMBREAK-STOKER`` is graded at ``t* = 2`` (see
:mod:`tensorfvm.shallow_water.dam_break`), and every extension of that anchor so
far -- the four-wave Riemann family, the Galilean lift, the wet-bed ladder --
also reads a *single* instant.  That is a structural blind spot of the
acceptance quantity ``x_f(t)``:

    a **constant front-speed bias** ``c`` produces a position error ``c t``.
    At ``t* = 2`` such a bias hides inside the 5 % band as long as
    ``c / U <= 2.5 %``, and it grows *without bound* in ``t``.

Both references are **self-similar**: ``h`` and ``u`` are functions of
``xi = (x - x_dam) / t`` alone (Ritter 1892; Stoker 1957 ch. 10).  Hence three
quantities are known exactly at *every* time and can be graded without a table:

``front speed``  ``x_f(t) / t``
    ``2 sqrt(g h0)`` (Ritter front) and ``U`` (Stoker bore).  Unlike the
    position, it does not shrink with ``t``: a bias is a bias.
``still-water step``  ``h2 / h1``
    time-*invariant* in the exact solution, so any drift in ``t`` is
    non-physical (boundary reflection, non-conservative update, depth-floor
    leakage) rather than discretisation error.
``self-similar collapse``  ``h(xi; t)`` vs ``h(xi; t_ref)``
    the numerical solution must reproduce *itself* in ``xi`` at different
    times; this check never looks at the analytic solution, so it cannot be
    satisfied by an oracle-consistent measurement window.

The study keeps ``dx`` fixed while the domain grows with ``t`` (the waves must
stay interior), so the number of cells spanning the wave grows with ``t`` and
the accumulated error is measured against a fixed *relative* spatial
resolution -- the same resolution as the anchor's accepted grid
(``dx = 8 m / 1600 = 5 mm``).

Discretisation is **untouched**: the production :class:`ShallowWater1D`
(HLL + MUSCL/minmod + SSP-RK, adaptive CFL) integrates; nothing here changes
the scheme.  This is "grade an existing capability on a new axis", not a new
solver.
"""

from __future__ import annotations

import math
import time as _time

import numpy as np

from .dam_break import (
    EXTREMUM_FRACTION,
    STOKER_WET,
    RITTER_DRY,
    DamBreakCase,
    front_sample,
    _local_extrema,
    _level_crossing,
    _relative,
)
from .exact import stoker_constant_state, t_star_to_time
from .swe1d import ShallowWater1D

__all__ = [
    "T_STARS",
    "TOLERANCE",
    "SPEED_TOLERANCE",
    "COLLAPSE_TOLERANCE",
    "STEP_DRIFT_TOLERANCE",
    "DX",
    "long_time_domain",
    "similarity_grid",
    "collapse_residual",
    "front_speed_fit",
    "simulate",
    "run_point",
    "run_long_time_study",
]

# --------------------------------------------------------------------------- #
# predeclared criteria (frozen before any run, see docs report)
# --------------------------------------------------------------------------- #
#: anchor tolerance for the wave position, reused verbatim (``dam_break.TOLERANCE``)
TOLERANCE = 0.05
#: the same band applied to the *speed* ``x_f(t)/t`` -- the time-invariant
#: content of the position criterion
SPEED_TOLERANCE = 0.05
#: time drift allowance for the still-water level step ``h2 / h1``: the exact
#: value is constant in ``t``, so only discretisation noise may remain
STEP_DRIFT_TOLERANCE = 5.0e-3
#: ``L1`` residual of the numerical self-similar collapse, normalised by ``h0``
COLLAPSE_TOLERANCE = 2.0e-2
#: cell size; ``5 mm`` == the anchor's accepted grid (``8 m / 1600``)
DX = 5.0e-3
CFL = 0.4
ORDER = 2
#: still-water margin kept beyond each wave, in units of ``c0 t``
PAD_C0 = 0.5
#: graded times, in units of ``t* = t sqrt(g h0) / h0``; ``2`` is the anchor
T_STARS = (1.0, 2.0, 5.0, 10.0)
#: the anchor time, used as the collapse reference
T_REF = 2.0
#: in-run front samples (the ``x_f(t)`` series feeding the speed fit)
SAMPLES_PER_RUN = 12


# --------------------------------------------------------------------------- #
# domain
# --------------------------------------------------------------------------- #
def long_time_domain(case: DamBreakCase, t_star: float,
                     pad_c0: float = PAD_C0) -> dict:
    """Physical time and an integration domain that keeps both waves interior.

    The upstream rarefaction head travels at exactly ``-c0`` and the front/bore
    at ``2 c0`` (dry) or ``U`` (wet), so the domain has to *grow* with ``t``
    instead of the usual fixed ``[-4, 6] m`` box of the ``t* = 2`` benchmark.
    ``pad_c0`` of still water is kept beyond each wave so the far-field flat
    regions stay available for the ``h1`` reading and for the transmissive
    boundary to be genuinely transparent.
    """
    t_star = float(t_star)
    if not t_star > 0.0:
        raise ValueError("t_star must be positive")
    t = t_star_to_time(t_star, case.h0, case.gravity)
    c0 = math.sqrt(case.gravity * case.h0)
    head = case.x_dam - c0 * t
    if case.kind == "dry":
        front = case.x_dam + 2.0 * c0 * t
    else:
        front = case.x_dam + stoker_constant_state(
            case.h0, case.h1, case.gravity)["shock_speed"] * t
    pad = pad_c0 * c0 * t
    return {
        "t_star": t_star,
        "t": t,
        "c0": c0,
        "head": head,
        "front": front,
        "pad": pad,
        "x_min": head - pad,
        "x_max": front + pad,
    }


# --------------------------------------------------------------------------- #
# self-similar collapse
# --------------------------------------------------------------------------- #
def similarity_grid(case: DamBreakCase, samples: int = 2001,
                    xi_lo: float | None = None,
                    xi_hi: float | None = None) -> np.ndarray:
    """Uniform ``xi = x / t`` grid covering the exact wave structure.

    Defaults to the exact support: ``[-c0, 2 c0]`` (dry bed) or
    ``[-c0, U]`` (wet bed, fan + plateau + bore).  The far-field flats are
    excluded on purpose -- they are constant in ``xi`` and would dilute any
    residual.
    """
    c0 = math.sqrt(case.gravity * case.h0)
    if xi_lo is None:
        xi_lo = -c0
    if xi_hi is None:
        if case.kind == "dry":
            xi_hi = 2.0 * c0
        else:
            xi_hi = stoker_constant_state(case.h0, case.h1,
                                          case.gravity)["shock_speed"]
    return np.linspace(float(xi_lo), float(xi_hi), int(samples))


def _to_similarity(state: dict, xi: np.ndarray) -> np.ndarray:
    """Resample a ``(t, x, h)`` snapshot onto the self-similar coordinate."""
    t = state["t"]
    x = np.asarray(state["x"], dtype=float)
    h = np.asarray(state["h"], dtype=float)
    order = np.argsort(x)
    return np.interp(xi * t, x[order], h[order])


T_COLLAPSE_TRIM = 0.15


def collapse_residual(reference: dict, sample: dict,
                      xi: np.ndarray | None = None,
                      case: DamBreakCase | None = None,
                      trim: float = T_COLLAPSE_TRIM) -> dict:
    """Distance between two snapshots in the self-similar coordinate ``xi``.

    ``L1`` is normalised by ``h0 * (xi_hi - xi_lo)``, i.e. it is the mean depth
    discrepancy over the wave, in units of ``h0``.  ``reference``/``sample``
    are ``{"t", "x", "h"}`` dicts; the grid defaults to the exact support of
    ``case`` (required if ``xi`` is ``None``).
    """
    if xi is None:
        if case is None:
            raise ValueError("need either xi or case")
        xi = similarity_grid(case)
    h_a = _to_similarity(reference, xi)
    h_b = _to_similarity(sample, xi)
    diff = h_a - h_b
    span = xi[-1] - xi[0]
    # The zero-depth front tip (and the rarefaction head) are *not* expected to
    # collapse at fixed dx: their smearing width in xi is dx/t, which shrinks as
    # t grows, so a fixed-dx profile differs from its own past self by O(h0) in
    # a band of width ~dx/t.  That band is a resolution effect, not a loss of
    # self-similarity, so it is trimmed by ``trim`` (fraction of the xi span at
    # each end) and reported separately as ``*_full``.
    count = xi.size
    lo = int(round(trim * count))
    hi = count - lo
    interior = diff[lo:hi]
    span_i = (hi - lo) / max(count - 1, 1) * span
    return {
        "t_reference": float(reference["t"]),
        "t": float(sample["t"]),
        "linf_over_h0": float(np.abs(interior).max()),
        "l1_over_h0": float(np.abs(interior).sum()
                            / max((hi - lo) * span_i, 1e-300)),
        "linf_full_over_h0": float(np.abs(diff).max()),
        "l1_full_over_h0": float(np.abs(diff).sum() / max(count * span, 1e-300)),
        "trim": float(trim),
        "xi_span": [float(xi[0]), float(xi[-1])],
    }


# --------------------------------------------------------------------------- #
# front-speed fit
# --------------------------------------------------------------------------- #
def front_speed_fit(samples, exact_speed: float, dx: float,
                    key: str = "headline") -> dict:
    """Least-squares ``x_f = a t + b`` over the graded in-run samples.

    ``a`` is the observed front speed; ``b`` must vanish (the wave starts at the
    dam).  Samples flagged unresolved by :func:`front_sample` are reported but
    not fitted -- on the dry bed the fan estimator needs several cells of fan
    before its reading is a discretisation error rather than a quantisation.
    """
    graded = [s for s in samples
              if s.get("resolved") and s.get(key) == s.get(key)]
    dropped = len(samples) - len(graded)
    if len(graded) < 3:
        return {"samples_graded": len(graded), "samples_dropped": dropped,
                "speed": float("nan"), "speed_error": float("nan"),
                "intercept": float("nan"), "intercept_over_dx": float("nan"),
                "defined": False}
    t = np.array([s["t"] for s in graded], dtype=float)
    p = np.array([s[key] for s in graded], dtype=float)
    a, b = np.polyfit(t, p, 1)
    return {
        "samples_graded": len(graded),
        "samples_dropped": dropped,
        "speed": float(a),
        "speed_exact": float(exact_speed),
        "speed_error": _relative(float(a), float(exact_speed)),
        "intercept": float(b),
        "intercept_over_dx": float(b / dx) if dx else float("nan"),
        "t_span": [float(t.min()), float(t.max())],
        "residual_max": float(np.abs(np.polyval([a, b], t) - p).max()),
        "defined": True,
    }


# --------------------------------------------------------------------------- #
# one (case, t_star) point
# --------------------------------------------------------------------------- #
def simulate(case: DamBreakCase, t_star: float, dx: float = DX,
             cfl: float = CFL, order: int = ORDER,
             samples: int = SAMPLES_PER_RUN) -> dict:
    """Integrate one long-time point and return state + front series."""
    dom = long_time_domain(case, t_star)
    n = max(8, int(round((dom["x_max"] - dom["x_min"]) / dx)))
    solver = ShallowWater1D(n, dom["x_min"], dom["x_max"],
                            gravity=case.gravity, cfl=cfl, order=order)
    solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
    volume0 = solver.total_volume()
    series: list[dict] = []
    gate = {"next": 0.0}

    def callback(sol: ShallowWater1D) -> None:
        if sol.time < gate["next"]:
            return
        series.append(front_sample(case, sol))
        gate["next"] = sol.time + dom["t"] / max(int(samples), 1)

    wall = _time.perf_counter()
    steps = solver.advance(dom["t"], callback=callback)
    wall = _time.perf_counter() - wall
    final = front_sample(case, solver)
    if not series or series[-1]["t"] < final["t"]:
        series.append(final)
    else:
        series[-1] = final
    return {
        "domain": dom,
        "n": n,
        "dx": solver.dx,
        "cfl": cfl,
        "order": order,
        "steps": steps,
        "wall_time_s": wall,
        "volume_initial": volume0,
        "volume_final": solver.total_volume(),
        "series": series,
        "state": {"t": solver.time, "x": solver.x.copy(),
                  "h": solver.h.copy()},
        "clipped_cells": solver.clipped_cells,
        "dry_flips": solver.dry_flips,
    }


def run_point(case: DamBreakCase, t_star: float, dx: float = DX,
              cfl: float = CFL, order: int = ORDER) -> dict:
    """One graded ``(case, t*)`` row: position, speed, step, monotonicity."""
    sim = simulate(case, t_star, dx=dx, cfl=cfl, order=order)
    solver_state = sim["state"]
    x = solver_state["x"]
    h = solver_state["h"]
    t = sim["domain"]["t"]
    h_ex, _ = case.exact(x, t)
    graded = [s for s in sim["series"]
              if s["resolved"] and s["headline_error"] == s["headline_error"]]
    final = sim["series"][-1]
    if case.kind == "dry":
        exact_speed = 2.0 * sim["domain"]["c0"]
    else:
        exact_speed = stoker_constant_state(case.h0, case.h1,
                                            case.gravity)["shock_speed"]
    speed = front_speed_fit(sim["series"], exact_speed, sim["dx"])
    row = {
        "case": case.key,
        "kind": case.kind,
        "t_star": float(t_star),
        "t": t,
        "n": sim["n"],
        "dx": sim["dx"],
        "steps": sim["steps"],
        "wall_time_s": sim["wall_time_s"],
        "domain": {k: sim["domain"][k]
                   for k in ("x_min", "x_max", "head", "front", "pad")},
        "front_exact": final["exact"],
        "front_measured": final["headline"],
        "front_error": final["headline_error"],
        "front_series_max_error": (max(s["headline_error"] for s in graded)
                                   if graded else float("nan")),
        "front_series_samples": len(sim["series"]),
        "front_series_graded": len(graded),
        "x_over_t": final["headline"] / t if t > 0 else float("nan"),
        "x_over_t_error": _relative(final["headline"] / t, exact_speed)
        if t > 0 else float("nan"),
        "exact_speed": exact_speed,
        "speed": speed,
        "speed_error": speed.get("speed_error", float("nan")),
        "series_points": [[s["t"], s["headline"]]
                          for s in sim["series"]
                          if s.get("headline") == s.get("headline")],
        "depth_l1": float(np.abs(h - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "depth_linf": float(np.abs(h - h_ex).max() / case.h0),
        "local_extrema": _local_extrema(h, EXTREMUM_FRACTION * case.h0),
        "tv_numeric": float(np.abs(np.diff(h)).sum()),
        "tv_exact": float(case.h0 - case.h1),
        "mass_drift": _relative(sim["volume_final"], sim["volume_initial"]),
        "h_min": float(h.min()),
        "h_max_over_h0": float(h.max() / case.h0),
        "clipped_cells": sim["clipped_cells"],
        "dry_flips": sim["dry_flips"],
    }
    # normalise the self-similar state for the collapse study
    # physical ``x`` on purpose: :func:`_to_similarity` builds ``xi * t`` itself
    sim["similarity"] = {"t": t, "x": x, "h": h}
    if case.kind == "wet":
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        x_shock = case.x_dam + sol["shock_speed"] * t
        x_fan_end = case.x_dam + (sol["u2"] - sol["c2"]) * t
        lo = x_fan_end + 0.25 * (x_shock - x_fan_end)
        hi = x_fan_end + 0.75 * (x_shock - x_fan_end)
        plateau = (x >= lo) & (x <= hi)
        still = x >= x_shock + 0.5 * (sim["domain"]["x_max"] - x_shock)
        h2 = float(h[plateau].mean()) if np.any(plateau) else float("nan")
        h1 = float(h[still].mean()) if np.any(still) else float("nan")
        row.update({
            "shock_exact": x_shock,
            "shock_measured": _level_crossing(
                x, h, 0.5 * (sol["h1"] + sol["h2"])),
            "shock_error": _relative(
                _level_crossing(x, h, 0.5 * (sol["h1"] + sol["h2"])), x_shock),
            "h2_exact": sol["h2"],
            "h2_measured": h2,
            "h2_error": _relative(h2, sol["h2"]),
            "h1_exact": case.h1,
            "h1_measured": h1,
            "h1_error": _relative(h1, case.h1),
            "step_ratio_exact": sol["step_ratio"],
            "step_ratio_measured": h2 / h1 if h1 else float("nan"),
            "step_ratio_error": _relative(h2 / h1, sol["step_ratio"])
            if h1 else float("nan"),
            "plateau_window_cells": int(np.count_nonzero(plateau)),
            "still_window_cells": int(np.count_nonzero(still)),
        })
    row["_similarity"] = sim["similarity"]
    return row


# --------------------------------------------------------------------------- #
# study
# --------------------------------------------------------------------------- #
def _exact_snapshot(case: DamBreakCase, t: float, xi: np.ndarray) -> dict:
    h_ex, _ = case.exact(xi * t, t)
    return {"t": t, "x": xi * t, "h": np.asarray(h_ex, dtype=float)}


def _verdict(case_key: str, rows: list[dict], collapse: list[dict],
             speed: dict) -> dict:
    """Predeclared pass criteria for the long-time study of one case."""
    worst_front = max(r["front_series_max_error"] for r in rows)
    worst_x_over_t = max(r["x_over_t_error"] for r in rows)
    worst_collapse = max(c["l1_over_h0"] for c in collapse) if collapse else 0.0
    step_errors = [r["step_ratio_error"] for r in rows
                   if r.get("step_ratio_error") is not None
                   and r["step_ratio_error"] == r["step_ratio_error"]]
    step_drift = (max(step_errors) - min(step_errors)) if step_errors else 0.0
    extrema = sum(r["local_extrema"] for r in rows)
    criteria = [
        {"name": "front_position_max",
         "value": worst_front, "limit": TOLERANCE,
         "pass": bool(worst_front <= TOLERANCE)},
        {"name": "front_speed_x_over_t_max",
         "value": worst_x_over_t, "limit": SPEED_TOLERANCE,
         "pass": bool(worst_x_over_t <= SPEED_TOLERANCE)},
        {"name": "front_speed_fit",
         "value": speed.get("speed_error", float("nan")),
         "limit": SPEED_TOLERANCE,
         "pass": bool(speed.get("defined")
                      and speed["speed_error"] <= SPEED_TOLERANCE)},
        {"name": "similarity_collapse_l1",
         "value": worst_collapse, "limit": COLLAPSE_TOLERANCE,
         "pass": bool(worst_collapse <= COLLAPSE_TOLERANCE)},
        {"name": "spurious_extrema_total",
         "value": float(extrema), "limit": 0.0,
         "pass": bool(extrema == 0)},
        {"name": "mass_drift_max",
         "value": max(r["mass_drift"] for r in rows), "limit": 1.0e-10,
         "pass": bool(max(r["mass_drift"] for r in rows) <= 1.0e-10)},
        {"name": "negative_depth",
         "value": min(r["h_min"] for r in rows), "limit": 0.0,
         "pass": bool(min(r["h_min"] for r in rows) >= 0.0)},
    ]
    if step_errors:
        criteria.append({
            "name": "still_water_step_drift",
            "value": step_drift, "limit": STEP_DRIFT_TOLERANCE,
            "pass": bool(step_drift <= STEP_DRIFT_TOLERANCE),
        })
    return {"case": case_key,
            "pass": all(c["pass"] for c in criteria),
            "criteria": criteria}


def run_long_time_study(cases=(RITTER_DRY, STOKER_WET), t_stars=T_STARS,
                        dx: float = DX, cfl: float = CFL, order: int = ORDER,
                        controls: bool = True) -> dict:
    """Sweep ``t*`` at fixed ``dx`` and grade the time-invariant quantities."""
    t0 = _time.perf_counter()
    out = {"criteria": {"tolerance": TOLERANCE,
                        "speed_tolerance": SPEED_TOLERANCE,
                        "collapse_tolerance": COLLAPSE_TOLERANCE,
                        "step_drift_tolerance": STEP_DRIFT_TOLERANCE,
                        "dx": dx, "cfl": cfl, "order": order,
                        "t_stars": list(t_stars), "t_ref": T_REF},
           "cases": {}}
    for case in cases:
        rows, snaps = [], []
        for ts in t_stars:
            row = run_point(case, ts, dx=dx, cfl=cfl, order=order)
            snaps.append(row.pop("_similarity"))
            rows.append(row)
        xi = similarity_grid(case)
        # metric self-validation: the exact solution is self-similar by
        # construction, so its collapse residual is the metric's noise floor
        ref_exact = _exact_snapshot(case, t_star_to_time(T_REF, case.h0,
                                                         case.gravity), xi)
        exact_collapse = []
        for ts in t_stars:
            t = t_star_to_time(ts, case.h0, case.gravity)
            exact_collapse.append(collapse_residual(
                ref_exact, _exact_snapshot(case, t, xi), xi=xi))
        reference = next((s for s in snaps
                          if abs(s["t"] - t_star_to_time(T_REF, case.h0,
                                                         case.gravity)) < 1e-12),
                         None)
        if reference is None:  # t_ref not in the sweep -> integrate it
            ref_sim = simulate(case, T_REF, dx=dx, cfl=cfl, order=order)
            reference = {"t": ref_sim["state"]["t"],
                         "x": ref_sim["state"]["x"] / ref_sim["state"]["t"],
                         "h": ref_sim["state"]["h"]}
        collapse = [dict(collapse_residual(reference, s, xi=xi),
                         t_star=row["t_star"])
                    for s, row in zip(snaps, rows)]
        # front-speed fit: every run carries its own in-run x_f(t) series, and
        # the longest run accumulates the most time steps, so grade all of them
        speed_rows = [dict(r["speed"], t_star=r["t_star"]) for r in rows]
        worst = max((r for r in speed_rows if r.get("defined")),
                    key=lambda r: r["speed_error"], default=None)
        speed = {
            "per_t_star": speed_rows,
            "defined": all(bool(r.get("defined")) for r in speed_rows),
            "worst": worst or {"speed_error": float("nan")},
        }
        out["cases"][case.key] = {
            "reference": case.reference,
            "rows": rows,
            "similarity": {"xi": xi.tolist(),
                           "exact_profile": np.asarray(case.exact(
                               xi * t_star_to_time(T_REF, case.h0, case.gravity),
                               t_star_to_time(T_REF, case.h0, case.gravity))[0],
                               dtype=float).tolist(),
                           "profiles": [
                               {"t_star": row["t_star"],
                                "t": s["t"],
                                "h": _to_similarity(s, xi).tolist()}
                               for s, row in zip(snaps, rows)],
                           "exact_collapse": exact_collapse,
                           "collapse": collapse},
            "speed": speed,
            "verdict": _verdict(case.key, rows, collapse,
                                worst or {}),
        }
    if controls:
        out["controls"] = _controls(cases, t_stars, dx=dx, order=order)
    out["wall_time_s"] = _time.perf_counter() - t0
    return out


def _controls(cases, t_stars, dx: float = DX, order: int = ORDER) -> dict:
    """Two falsification controls: time-step refinement and a lower-order scheme.

    ``cfl``
        At the longest ``t*``, halving ``dt`` (``cfl`` 0.4 -> 0.2 -> 0.1) at
        fixed ``dx`` must barely move the readings.  If it does, the long-time
        error is *temporal* and the anchor's SSP-RK step needs auditing.
    ``order``
        A first-order scheme has a much larger smeared-front speed bias, so it
        must read a **larger** long-time front error: the metric has teeth.
    """
    t_last = float(t_stars[-1])
    cfl_rows = {}
    for cfl in (0.4, 0.2, 0.1):
        rows = {}
        for case in cases:
            row = run_point(case, t_last, dx=dx, cfl=cfl, order=order)
            row.pop("_similarity", None)
            rows[case.key] = {k: row[k] for k in
                              ("front_error", "x_over_t_error", "speed",
                               "steps", "wall_time_s", "depth_l1")}
        cfl_rows[str(cfl)] = rows
    order_rows = {}
    for scheme in (2, 1):
        rows = {}
        for case in cases:
            row = run_point(case, t_last, dx=dx, cfl=0.4, order=scheme)
            row.pop("_similarity", None)
            rows[case.key] = {k: row[k] for k in
                              ("front_error", "x_over_t_error", "speed",
                               "depth_l1", "local_extrema")}
        order_rows[str(scheme)] = rows
    return {"t_star": t_last, "cfl_refinement": cfl_rows,
            "order_reference": order_rows}