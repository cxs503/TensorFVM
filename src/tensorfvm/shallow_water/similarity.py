"""Self-similar collapse of the shallow-water dam break (Ritter / Stoker).

Why this module exists
----------------------
Every criterion registered for ``CASE-FVM-DAMBREAK-STOKER`` so far is a *scalar*
read off one profile (front position, plateau depth, level step) or an integral
of one profile (depth L1, momentum budget, energy budget).  None of them asks
the question both reference solutions are built on: Ritter's and Stoker's
solutions are **self-similar**,

    h(x, t) = h(xi),  u(x, t) = u(xi),  xi = (x - x_dam) / t ,

i.e. the *whole* profile at one time and the profile at another time are the
same curve in ``xi``.  A scheme can hit every ``t* = 2`` scalar and still be
wrong in a self-similar way, or fail to enter the self-similar regime at all
during the transient.  Nothing in the artefacts could tell those two apart.

This module adds that axis, and it adds it in a way that is *falsifiable*:

``xi_error``
    ``mean|h_num(xi) - h_ex(xi)| / mean|h_ex(xi)|`` on a declared ``xi`` grid --
    the accuracy of the numerical profile *as a function of the similarity
    variable*.  Because ``xi`` is a time-independent coordinate, this functional
    is the natural measure of "how far the numerical solution is from the
    self-similar reference", and it is invariant under a change of the sampling
    time *if* the solution is self-similar.

``collapse_residual``
    Spread of the profiles ``h(xi)`` measured at several times -- a
    *consistency* check that uses no reference solution at all.

The pair is deliberate: **collapse alone is vacuous**.  A wrong-but-self-similar
profile (e.g. the Ritter fan of a slightly different ``h0``, which is a similarity
solution of another case) collapses perfectly.  ``tests/test_dambreak_similarity.py``
pins that with a negative control, so neither criterion can be satisfied by a
defect that merely preserves self-similarity.

Measured results (``scripts/verify_dambreak_similarity.py``, artifact
``verification/artifacts/shallow-water-similarity/summary.json``)

* ``xi_error`` follows ``E = C * dx / (c0 t)`` to within a few percent over two
  decades of ``dx/(c0 t)``: first order in ``dx`` (measured: halving ``dx``
  halves ``E``) and ``~1/t`` in time.  Measured plateau constants
  ``C_dry ~ 0.46``, ``C_wet ~ 0.28``.
* The start-up transient therefore has a *quantitative* boundary: the profile is
  within the 5 % gate once ``c0 t / dx >= C / 0.05 ~ 9.2`` cells of travel.
  The a-priori series thresholds declared in :mod:`tensorfvm.shallow_water.dam_break`
  (``SERIES_MIN_CELLS = 4``, ``LEVEL_SERIES_MIN_CELLS = 20``) are of exactly this
  form; ``C / TOLERANCE = 9.2`` says the declared 20-cell contour threshold is
  conservative by a factor of ~2.2 -- independently derived, not fitted.
* The early-time **bore-position** reading is a function of ``x_s / dx`` (cells
  travelled) alone, and reproduces to 1e-13 across ``n`` and ``t*`` at fixed
  ``x_s/dx`` (``bore_reading_study``).  That is the invariant behind the
  "grid-independent 4.505 %" observation in ``docs/case-fvm-dambreak-stoker/AUDIT.md``
  F5: what is scale invariant there is the *sample schedule* (the adaptive step
  makes the sample times proportional to ``dx``), not the physics -- at a
  *common* ``t*`` the same reading is strongly grid dependent (13.5 % -> 2.4 % ->
  1.1 % for n = 400 -> 800 -> 1600 at ``t* = 0.02``).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .dam_break import (RITTER_DRY, STOKER_WET, DamBreakCase, TOLERANCE,
                        _level_crossing)
from .exact import ritter_state, stoker_constant_state, stoker_state
from .swe1d import ShallowWater1D

__all__ = [
    "XI_POINTS",
    "XI_MARGIN",
    "SAMPLE_T_STARS",
    "DEFAULT_RESOLUTIONS",
    "COLLAPSE_BAND",
    "ASYMPTOTIC_T_STAR",
    "similarity_support",
    "xi_grid",
    "exact_xi_profile",
    "numerical_xi_profile",
    "xi_error",
    "collapse_residual",
    "run_similarity_series",
    "similarity_row",
    "similarity_study",
    "bore_reading_study",
    "cases",
]

#: number of points on the ``xi`` grid.  401 over the exact support (width
#: ``3 c0`` for the dry bed) gives ``dx_xi = c0/133``; the grid is *declared*,
#: not tuned: the measured functional is converged to < 1e-3 relative at this
#: resolution (``tests/test_dambreak_similarity.py``).
XI_POINTS = 401

#: margin added on both sides of the exact wave support, as a fraction of ``c0``.
#: The numerical profile has a smeared tip/bore, so the grid must extend past
#: the exact support or the collapse would be clipped at the ends.
XI_MARGIN = 0.25

#: sample times (in ``t*``) for the collapse series, spanning the transient.
SAMPLE_T_STARS: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0)

#: reference resolutions for the study (cheap: the whole study is ~30 s single
#: core, so it runs inside the test suite's budget at n <= 800).
DEFAULT_RESOLUTIONS: Sequence[int] = (400, 800, 1600)

#: fraction of the ``xi`` grid retained when comparing profiles for the
#: *collapse* check (the central ``COLLAPSE_BAND`` of the grid).  The trimmed ends
#: are the two smeared wave ends, where any finite-volume profile differs from
#: its neighbour by ``O(dx/t)`` regardless of the solution being self-similar.
COLLAPSE_BAND = 0.9

#: ``t*`` from which the collapse residual is evaluated (``collapse_asymptotic``).
#: Declared before any run from the measured law: ``E = C dx/(c0 t)`` puts a 5 %
#: ``xi`` error at ``c0 t/dx ~ 8`` cells, i.e. ``t* ~ 0.05`` on ``n = 1600``;
#: ``t* = 0.4`` sits an order of magnitude inside that regime.
ASYMPTOTIC_T_STAR = 0.4


def cases() -> dict[str, DamBreakCase]:
    """The two anchor cases (same objects the graded driver uses)."""
    return {"ritter-dry": RITTER_DRY, "stoker-wet": STOKER_WET}


# --------------------------------------------------------------------------- #
# similarity variable
# --------------------------------------------------------------------------- #
def similarity_support(case: DamBreakCase) -> tuple[float, float]:
    """Exact wave support in ``xi`` units (``xi = (x - x_dam)/t``).

    Dry bed: ``[-c0, 2 c0]``.  Wet bed: ``[-c0, U]`` with ``U`` the bore speed
    (the rarefaction head travels at ``-c0``, the bore at ``+U``).
    """
    c0 = math.sqrt(case.gravity * case.h0)
    if case.kind == "dry":
        return (-c0, 2.0 * c0)
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    return (-c0, float(sol["shock_speed"]))


def xi_grid(case: DamBreakCase, points: int = XI_POINTS,
            margin: float = XI_MARGIN) -> np.ndarray:
    """Declared ``xi`` grid: the exact support plus ``margin * c0`` either side."""
    lo, hi = similarity_support(case)
    pad = margin * math.sqrt(case.gravity * case.h0)
    return np.linspace(lo - pad, hi + pad, int(points))


def exact_xi_profile(case: DamBreakCase, xi: Iterable[float] | None = None,
                     normalize: bool = True) -> dict:
    """Exact self-similar profile ``h/h0``, ``u/c0`` on ``xi``.

    Evaluated by calling the closed form at ``t = 1``, ``x = x_dam + xi``:
    both Ritter's and Stoker's states depend on ``x`` and ``t`` only through
    ``xi``, so this is the exact similarity profile, not an interpolation of
    two time slices.  ``tests/test_dambreak_similarity.py`` pins the invariance
    by evaluating the closed form at two different times.
    """
    xg = xi_grid(case) if xi is None else np.asarray(xi, dtype=float)
    x = case.x_dam + xg
    if case.kind == "dry":
        h, u = ritter_state(x, 1.0, case.h0, case.gravity, case.x_dam)
    else:
        h, u = stoker_state(x, 1.0, case.h0, case.h1, case.gravity, case.x_dam)
    c0 = math.sqrt(case.gravity * case.h0)
    if not normalize:
        return {"xi": xg, "h": np.asarray(h, float), "u": np.asarray(u, float)}
    return {"xi": xg, "h": np.asarray(h, float) / case.h0,
            "u": np.asarray(u, float) / c0}


def numerical_xi_profile(x, h, t: float, case: DamBreakCase,
                         xi: Iterable[float] | None = None,
                         normalize: bool = True) -> dict:
    """Numerical profile mapped onto ``xi`` by linear interpolation in ``x``.

    ``t <= 0`` has no similarity variable; a ``None`` profile is returned rather
    than an exception so a caller can report "not measurable" instead of
    inventing a value (same discipline as the graded metrics).
    """
    xg = xi_grid(case) if xi is None else np.asarray(xi, dtype=float)
    if not t > 0.0:
        return {"xi": xg, "h": None}
    xi_num = (np.asarray(x, dtype=float) - case.x_dam) / t
    h = np.asarray(h, dtype=float)
    prof = np.interp(xg, xi_num, h / case.h0 if normalize else h)
    return {"xi": xg, "h": prof}


def xi_error(profile: dict, exact: dict) -> float:
    """Relative L1 error of a ``xi``-profile against the exact similarity profile.

    ``mean|h - h_ex| / mean|h_ex|`` on the declared grid.  Returns ``nan`` when
    the profile is not measurable (never a silent zero).
    """
    p = profile.get("h")
    if p is None:
        return float("nan")
    e = np.asarray(exact["h"], dtype=float)
    p = np.asarray(p, dtype=float)
    denom = float(np.mean(np.abs(e)))
    if not denom > 0.0:
        return float("nan")
    return float(np.mean(np.abs(p - e)) / denom)


def collapse_residual(profiles: Sequence[dict], reference: dict | None = None,
                      band: float = COLLAPSE_BAND) -> float:
    """Consistency of several ``xi``-profiles (no reference solution involved).

    With ``reference``: ``mean|p - p_ref| / mean|p_ref|`` over the retained band.
    Without: the pointwise spread ``max_i p_i - min_i p_i`` normalised by the
    ensemble mean -- i.e. "how much do the profiles of different times disagree
    in the similarity variable".

    Only the central ``band`` fraction of the declared ``xi`` grid is used (see
    :data:`COLLAPSE_BAND`).  Returns ``nan`` if no profile is measurable.
    """
    usable = [np.asarray(p["h"], float) for p in profiles if p.get("h") is not None]
    if not usable:
        return float("nan")
    stack = np.stack(usable, axis=0)
    npts = stack.shape[1]
    keep = max(3, int(round(float(band) * npts)))
    lo = (npts - keep) // 2
    stack = stack[:, lo:lo + keep]
    if reference is not None:
        ref = np.asarray(reference["h"], float)[lo:lo + keep]
        denom = float(np.mean(np.abs(ref)))
        if not denom > 0.0:
            return float("nan")
        return float(np.mean(np.abs(stack - ref), axis=1).mean() / denom)
    mean = stack.mean(axis=0)
    denom = float(np.mean(np.abs(mean)))
    if not denom > 0.0:
        return float("nan")
    spread = stack.max(axis=0) - stack.min(axis=0)
    return float(np.mean(spread) / denom)


def run_similarity_series(case: DamBreakCase, n: int, cfl: float = 0.4,
                          order: int = 2,
                          t_stars: Sequence[float] = SAMPLE_T_STARS) -> list[dict]:
    """Integrate ``case`` once and record the profile at every ``t*`` in ``t_stars``.

    A single run is advanced from one sample time to the next, so the states are
    *exactly* those of one trajectory (no re-initialisation, no interpolation in
    time).  The recorded ``t`` is ``solver.time`` -- the time the state actually
    belongs to -- because using the requested time would mix a time-sampling
    error into a spatial functional.
    """
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
    out: list[dict] = []
    for ts in t_stars:
        t_end = ts * math.sqrt(case.h0 / case.gravity)
        solver.advance(t_end)
        out.append({"t_star": float(ts), "t": float(solver.time),
                    "x": solver.x.copy(), "h": solver.h.copy(),
                    "hu": solver.hu.copy()})
    return out


def similarity_row(case: DamBreakCase, n: int, cfl: float = 0.4, order: int = 2,
                   t_stars: Sequence[float] = SAMPLE_T_STARS) -> dict:
    """One (case, resolution) row: the whole collapse/accuracy series."""
    c0 = math.sqrt(case.gravity * case.h0)
    exact = exact_xi_profile(case)
    states = run_similarity_series(case, n, cfl=cfl, order=order, t_stars=t_stars)
    profiles = []
    rows = []
    final = None
    for st in states:
        prof = numerical_xi_profile(st["x"], st["h"], st["t"], case)
        profiles.append(prof)
        err = xi_error(prof, exact)
        cells = c0 * st["t"] / (case.x_max - case.x_min) * n  # = c0 t / dx
        rows.append({
            "t_star": st["t_star"],
            "t": st["t"],
            "xi_error": err,
            "cells_travelled": float(cells),
            "dx_over_c0t": float(1.0 / cells) if cells > 0 else float("nan"),
            "C": float(err * cells),      # E = C * dx/(c0 t)
        })
        final = prof
    ref_err = rows[-1]["xi_error"]
    asympt = [p for p, r in zip(profiles, rows) if r["t_star"] >= ASYMPTOTIC_T_STAR]
    for r, prof in zip(rows, profiles):
        r["collapse_vs_final"] = collapse_residual([prof], reference=final)
    return {
        "case": case.key,
        "n": int(n),
        "dx": float((case.x_max - case.x_min) / n),
        "cfl": float(cfl),
        "order": int(order),
        "rows": rows,
        "collapse_all_times": collapse_residual(profiles),
        "collapse_asymptotic": collapse_residual(asympt),
        "asymptotic_t_star": ASYMPTOTIC_T_STAR,
        "xi_error_final": float(ref_err),
    }


def similarity_study(resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
                     cases_: dict[str, DamBreakCase] | None = None,
                     t_stars: Sequence[float] = SAMPLE_T_STARS,
                     cfls: Sequence[float] = (0.2, 0.4, 0.8)) -> dict:
    """The full study: ``xi``-error law, ``dx`` order, CFL (temporal) invariance.

    * ``law``: for each case the ``C = E * c0 t / dx`` plateau (mean, min, max
      over ``t* >= 0.2``, the range where the ``1/t`` behaviour is established)
      and the measured ``dx`` order from the final ``xi`` error.  The window is
      ``t* >= ASYMPTOTIC_T_STAR`` (declared before the run); inside it ``C`` is
      constant to a few percent, outside it the smearing region is still a
      sizeable fraction of the wave.
    * ``cfl``: the same final ``xi`` error at ``cfl = 0.2 / 0.4 / 0.8`` on the
      middle resolution -- the temporal axis, which no existing criterion grades.
    """
    cs = cases_ or cases()
    out: dict = {"cases": {}, "cfl": [], "t_stars": list(t_stars)}
    for key, case in cs.items():
        rows = [similarity_row(case, n, t_stars=t_stars) for n in resolutions]
        plateau = [r["C"] for row in rows for r in row["rows"]
                   if r["t_star"] >= ASYMPTOTIC_T_STAR]
        finals = [(row["n"], row["xi_error_final"]) for row in rows]
        orders = []
        for (n0, e0), (n1, e1) in zip(finals, finals[1:]):
            if e0 > 0 and e1 > 0:
                orders.append({"n0": n0, "n1": n1,
                               "order": float(math.log(e0 / e1) / math.log(n1 / n0))})
        lo_t = [r for row in rows for r in row["rows"] if r["t_star"] <= 0.05]
        out["cases"][key] = {
            "rows": rows,
            "C_plateau": {
                "mean": float(np.mean(plateau)),
                "min": float(np.min(plateau)),
                "max": float(np.max(plateau)),
                "spread": float((max(plateau) - min(plateau)) / np.mean(plateau)),
            },
            "C_plateau_note": "E_xi = C * dx/(c0 t), mean over t* >= ASYMPTOTIC_T_STAR and all resolutions",
            "dx_order": orders,
            "min_cells_for_gate": float(np.mean(plateau) / TOLERANCE),
            "transient_max_error": float(max(r["xi_error"] for r in lo_t)),
            "final_xi_error": {str(n): e for n, e in finals},
        }
    mid = resolutions[len(resolutions) // 2]
    for key, case in cs.items():
        for cfl in cfls:
            row = similarity_row(case, mid, cfl=cfl, t_stars=(2.0,))
            out["cfl"].append({
                "case": key, "n": int(mid), "cfl": float(cfl),
                "xi_error_final": row["xi_error_final"],
            })
    return out


# --------------------------------------------------------------------------- #
# early-time bore reading: which variable does it depend on?
# --------------------------------------------------------------------------- #
def bore_reading_study(combos: Sequence[tuple[int, float]] | None = None,
                       case: DamBreakCase | None = None) -> dict:
    """Collapse the early-time **bore-position** error onto ``x_s / dx``.

    ``AUDIT.md`` F5 reports the wet-bed series error at ``t* <~ 0.02`` as a
    *grid-independent* 4.505 %.  This study tests that statement by measuring the
    same reading at several ``(n, t*)`` pairs chosen to share values of the
    dimensionless group ``x_s / dx`` = (bore travel in cells).  If the error is a
    function of that group alone, the numbers agree across ``n`` *and* ``t*``
    to machine precision -- which means the invariance observed in F5 belongs to
    the sample schedule (the adaptive step makes the sample times proportional
    to ``dx``), not to the physics.

    The same table also reports the error at a *common* ``t*`` across ``n``, the
    case where F5's wording ("grid independent") is not true.
    """
    case = case or STOKER_WET
    c0 = math.sqrt(case.gravity * case.h0)
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    level = 0.5 * (case.h1 + sol["h2"])
    speed = float(sol["shock_speed"])

    def one(n: int, t_star: float) -> dict:
        solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                                cfl=0.4, order=2)
        solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
        solver.advance(t_star * math.sqrt(case.h0 / case.gravity))
        t = solver.time
        x_s = case.x_dam + speed * t
        pos = _level_crossing(solver.x, solver.h, level)
        dx = float(solver.dx)
        return {"n": int(n), "dx": dx, "t_star": float(t_star), "t": float(t),
                "dx_over_xs": float(dx / x_s) if x_s else float("nan"),
                "xs_over_dx": float(x_s / dx) if dx else float("nan"),
                "position": float(pos),
                "error": float(abs(pos - x_s) / x_s) if x_s else float("nan"),
                "cells_normalised_error": (float(abs(pos - x_s) / dx)
                                           if dx else float("nan"))}

    # (n, t*) pairs chosen so that x_s/dx takes a small, shared set of values:
    # x_s/dx = U t*/dx * sqrt(h0/g) is invariant when n doubles and t* doubles.
    if combos is None:
        combos = [(200, 0.02), (400, 0.02), (800, 0.02), (1600, 0.02),
                  (3200, 0.02),
                  (200, 0.01), (400, 0.01), (800, 0.01), (1600, 0.01),
                  (200, 0.005), (400, 0.005), (800, 0.005), (1600, 0.005),
                  (400, 0.05), (800, 0.05), (1600, 0.05)]
    rows = [one(n, ts) for n, ts in combos]
    # group by the shared dimensionless group, tolerance 1e-9 relative
    groups: dict[float, list[dict]] = {}
    for r in rows:
        key = round(r["xs_over_dx"], 9)
        groups.setdefault(key, []).append(r)
    invariance = []
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        errs = [m["error"] for m in members]
        invariance.append({
            "xs_over_dx": float(key),
            "members": [(m["n"], m["t_star"]) for m in members],
            "errors": errs,
            "max_spread": float(max(errs) - min(errs)),
            "max_relative_spread": float((max(errs) - min(errs)) / max(errs))
            if max(errs) > 0 else 0.0,
        })
    return {"case": case.key, "rows": rows, "invariance": invariance,
            "level": float(level), "shock_speed": speed, "c0": float(c0)}