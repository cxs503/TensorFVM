"""Round 9 of CASE-FVM-PERIODIC-HILLS: time-averaged gate.

Rounds 1--8 established (and recorded as honest FAIL) that a *single snapshot*
of the 2-D URANS solution cannot decide the case: the SA closure keeps growing
past ``t = 60`` (``max nu_t/nu`` 0.2 -> 36 and rising, ROUND8.md section 2.2) and
``x_r/H`` oscillates 2.6 -> 4.1 -> 6.1 -> nan -> 5.6 while it does.  The
acceptance metrics are however *mean* quantities -- ``x_r/H`` and ``u/U_b`` on
the line ``y/H = 0.5`` are means in Breuer et al. (2009) -- so they must be
compared against a **time average over a statistically stationary window**,
not against one snapshot.

This module supplies exactly that post-processing, and nothing else:

* :func:`zero_crossings_periodic` -- separation/reattachment from a wall-shear
  profile on a periodic grid (the same search as
  ``PeriodicHillSolver.reattachment`` but usable on a *time-averaged* shear,
  where the instantaneous field may carry several crossings);
* :func:`time_averaged_reattachment` -- ``(x_s/H, x_r/H)`` from the mean
  bottom-wall shear, with the wall shear of every sample already accumulated by
  :class:`tensorfvm.periodic_hill.Statistics`;
* :func:`horizontal_profile` -- ``u/U_b`` sampled on the line ``y/H = const``.
  The mesh is body-fitted, so on the crest the line ``y/H = 0.5`` lies *inside
  the hill*: those columns are reported as inadmissible rather than
  extrapolated (round 6's rule, reused);
* :func:`block_means` / :func:`stationarity_report` -- the averaging window is
  only admissible if the block means agree; the spread is reported, never
  hidden;
* :func:`compare_profile` -- per-station error against the single-sourced
  Breuer table imported from :mod:`periodic_hill_round6`, with a *relative* band
  where the reference magnitude allows one and an *absolute* band where it does
  not (a relative error against ``u/U_b = -0.02`` is meaningless).

Nothing in this module decides PASS/FAIL by itself; the gate lives in
``tests/test_periodic_hill_round9.py`` and the verdict in ``docs/case-periodic-hills/ROUND9.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .periodic_hill_round6 import (
    BREUER_PROFILE_Y05,
    REFERENCE,
    REFERENCE_TOLERANCE,
    admissible_stations,
)

__all__ = [
    "REFERENCE",
    "REFERENCE_TOLERANCE",
    "BREUER_PROFILE_Y05",
    "zero_crossings_periodic",
    "time_averaged_reattachment",
    "horizontal_profile",
    "block_means",
    "stationarity_report",
    "compare_profile",
    "averaging_window",
]


# --------------------------------------------------------------------------- #
# wall-shear based separation / reattachment
# --------------------------------------------------------------------------- #
def zero_crossings_periodic(tau, x_over_h, lower=1.9, upper=7.1):
    """Zero crossings of a wall-shear profile on a periodic x-grid.

    Parameters
    ----------
    tau : (n,) array
        Wall shear at the face centres of one wall, periodic in ``x``.
    x_over_h : (n,) array
        Streamwise coordinate of those centres in units of the hill height,
        counted from the crest (``x/H = 0``), strictly increasing modulo the
        period.
    lower, upper : float
        The crest-adjacent separation crossing must lie below ``lower`` and the
        reattachment crossing inside ``[lower, upper]`` (the window of round 5;
        anything outside is a spurious crossing of an unresolved shear layer).

    Returns
    -------
    dict with ``separation``, ``reattachment`` (nan when not found),
    ``crossings`` (all of them, in ``x/H``), ``sign_changes`` (count of
    changes ``tau < 0 -> tau >= 0``).

    Separation and reattachment are *both* ``tau < 0 -> tau >= 0`` crossings of
    the wall shear: downstream of the crest the shear is negative (reversed
    flow), so the first such crossing ends the recirculation bubble and the
    next one starts the *next* bubble.  The convention here follows
    ``PeriodicHillSolver.reattachment`` (first crossing below ``lower`` =
    separation, first inside ``[lower, upper]`` = reattachment) so that a
    time-averaged value is directly comparable with the instantaneous one.
    """
    tau = np.asarray(tau, dtype=float)
    x = np.asarray(x_over_h, dtype=float)
    if tau.shape != x.shape or tau.ndim != 1:
        raise ValueError("tau and x_over_h must be 1-D arrays of equal length")
    if tau.size < 3:
        raise ValueError("need at least 3 wall-shear samples")
    if not np.all(np.isfinite(tau)):
        return {"separation": float("nan"), "reattachment": float("nan"),
                "crossings": [], "sign_changes": 0, "finite": False}
    period = float(x[-1] - x[0]) + float(x[1] - x[0])
    crossings = []
    for i in range(tau.size):
        a, b = tau[i], tau[(i + 1) % tau.size]
        if a < 0.0 <= b:
            t = -a / (b - a)
            dx = (x[(i + 1) % tau.size] - x[i]) % period
            crossings.append(float(x[i] + t * dx))
    sep = next((c for c in crossings if c < lower), float("nan"))
    rea = next((c for c in crossings if lower <= c <= upper), float("nan"))
    return {"separation": sep, "reattachment": rea,
            "crossings": crossings, "sign_changes": len(crossings),
            "finite": True}


def time_averaged_reattachment(mean_tau_bottom, x_over_h, lower=1.9, upper=7.1):
    """``(x_s/H, x_r/H)`` from a time-averaged bottom-wall shear profile."""
    out = zero_crossings_periodic(mean_tau_bottom, x_over_h, lower, upper)
    out["x_s_over_H"] = out["separation"]
    out["x_r_over_H"] = out["reattachment"]
    return out


# --------------------------------------------------------------------------- #
# mean profile on a horizontal line
# --------------------------------------------------------------------------- #
def horizontal_profile(mean_u, y, x_over_h, y_over_h=0.5, height=1.0):
    """``u/U_b`` on the horizontal line ``y/H = y_over_h`` of a body-fitted mesh.

    Parameters
    ----------
    mean_u : (ny, nx) array
        **Averaged** streamwise velocity (time mean or instantaneous).
    y : (ny, nx) array
        Cell-centre ``y`` in metres (``solver.mesh.yc``).
    x_over_h : (nx,) array
        Column positions in ``x/H`` (``solver.mesh.xc[0] / height``).
    y_over_h : float
        Height of the sampling line in units of ``H``.
    height : float
        ``H`` in metres.

    Returns
    -------
    dict
        ``x_over_h`` (admissible columns), ``u`` (interpolated, still in the
        solver's velocity units), ``indices``, ``inadmissible`` (columns whose
        lowest cell centre already sits above the line -- inside the hill) and
        ``mask`` (boolean over all columns).

    The interpolation is linear in ``y`` *cell centre to cell centre*, i.e. no
    wall extrapolation: a column is used only when
    ``y[0, i] <= y_over_h * height <= y[-1, i]``.
    """
    mean_u = np.asarray(mean_u, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x_over_h, dtype=float)
    if mean_u.shape != y.shape:
        raise ValueError("mean_u and y must have the same (ny, nx) shape")
    if x.shape != (mean_u.shape[1],):
        raise ValueError("x_over_h must have one entry per column")
    target = y_over_h * height
    cols, vals = [], []
    inadmissible = []
    for i in range(mean_u.shape[1]):
        col = y[:, i]
        if not np.all(np.isfinite(col)) or target < col[0] or target > col[-1]:
            inadmissible.append(i)
            continue
        vals.append(float(np.interp(target, col, mean_u[:, i])))
        cols.append(i)
    mask = np.zeros(mean_u.shape[1], dtype=bool)
    mask[cols] = True
    return {
        "x_over_h": x[cols],
        "u": np.asarray(vals, dtype=float),
        "indices": np.asarray(cols, dtype=int),
        "inadmissible": inadmissible,
        "mask": mask,
    }


# --------------------------------------------------------------------------- #
# averaging-window diagnostics
# --------------------------------------------------------------------------- #
def block_means(values, n_blocks=4):
    """Split ``values`` into ``n_blocks`` equal contiguous blocks, return means.

    Contiguous (not interleaved) blocks are used on purpose: they expose a slow
    drift, which interleaved sampling would average away.
    """
    values = np.asarray(values, dtype=float).ravel()
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    if values.size < n_blocks:
        raise ValueError(f"need at least {n_blocks} samples, got {values.size}")
    return np.array([b.mean() for b in np.array_split(values, n_blocks)])


def stationarity_report(times, values, n_blocks=4, tolerance=0.05,
                        drift_tolerance=0.05):
    """Is the sampling window stationary at the level of ``values``?

    A window is reported stationary when

    1. the spread of the contiguous block means is within ``tolerance``
       (relative to the overall mean magnitude, floored at ``1e-12``), **and**
    2. the least-squares slope over the window, extrapolated across the whole
       window, is within ``drift_tolerance`` of the same magnitude.

    Both numbers are returned whether or not the test passes, so a failing
    window still reports *how* it fails.  A non-finite sample makes the window
    non-stationary by definition (rounds 5 and 8 both hit ``x_r = nan``
    snapshots).
    """
    t = np.asarray(times, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    if t.shape != v.shape:
        raise ValueError("times and values must have equal length")
    finite = np.isfinite(v)
    out = {"samples": int(v.size), "finite_samples": int(finite.sum()),
           "t_start": float(t[0]) if t.size else float("nan"),
           "t_end": float(t[-1]) if t.size else float("nan"),
           "mean": float("nan"), "blocks": [], "spread_rel": float("nan"),
           "drift_rel": float("nan"), "ok": False}
    if v.size < n_blocks or not np.all(finite):
        out["reason"] = ("non-finite samples in window" if not np.all(finite)
                         else "fewer samples than blocks")
        return out
    blocks = block_means(v, n_blocks)
    mean = float(v.mean())
    scale = max(abs(mean), 1e-12)
    spread = float(blocks.max() - blocks.min()) / scale
    span = float(t[-1] - t[0])
    if span > 0.0:
        slope = float(np.polyfit(t - t[0], v, 1)[0])
    else:
        slope = 0.0
    drift = abs(slope) * span / scale
    out.update({
        "mean": mean,
        "blocks": [float(b) for b in blocks],
        "spread_rel": spread,
        "drift_rel": drift,
        "slope": slope,
        "ok": bool(spread <= tolerance and drift <= drift_tolerance),
        "reason": ("within tolerance" if spread <= tolerance and drift <= drift_tolerance
                   else f"spread {spread:.3g} > {tolerance:g} or drift {drift:.3g} > {drift_tolerance:g}"),
    })
    return out


def averaging_window(times, values, n_blocks=4, tolerance=0.05,
                     drift_tolerance=0.05, min_blocks=2):
    """Pick the *latest* sampling window that is stationary, by discarding blocks.

    Starting from the last ``n_blocks`` blocks and dropping the earliest one at
    a time, return the first window (longest admissible) that passes
    :func:`stationarity_report`.  If nothing passes, the *earliest-drop*
    failure is returned with ``ok = False`` -- no silent truncation, and the
    returned ``blocks_used`` says how much of the record survived.
    """
    t = np.asarray(times, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    if t.shape != v.shape:
        raise ValueError("times and values must have equal length")
    if n_blocks < min_blocks:
        raise ValueError("n_blocks must be >= min_blocks")
    if v.size < n_blocks:
        # an all-nan / too-short record is a *failed* window, not an exception:
        # the gate must be able to report "no admissible averaging window"
        return {"samples": int(v.size), "ok": False, "blocks_used": 0,
                "blocks_dropped": n_blocks, "t_start": float("nan"),
                "t_end": float("nan"), "mean": float("nan"),
                "spread_rel": float("nan"), "drift_rel": float("nan"),
                "reason": f"only {v.size} samples for {n_blocks} blocks"}
    edges = np.linspace(0, v.size, n_blocks + 1).astype(int)
    attempts = []
    for drop in range(0, n_blocks - min_blocks + 1):
        used = list(range(drop, n_blocks))
        start, stop = edges[used[0]], edges[used[-1] + 1]
        rep = stationarity_report(t[start:stop], v[start:stop],
                                  n_blocks=len(used), tolerance=tolerance,
                                  drift_tolerance=drift_tolerance)
        rep["blocks_used"] = len(used)
        rep["blocks_dropped"] = drop
        rep["t_start"], rep["t_end"] = float(t[start]), float(t[stop - 1])
        attempts.append(rep)
        if rep["ok"]:
            return rep
    worst = attempts[0] if attempts else {}
    worst["attempts"] = attempts
    return worst


# --------------------------------------------------------------------------- #
# comparison against the single-sourced reference table
# --------------------------------------------------------------------------- #
def compare_profile(x_over_h, u_over_ub, bulk_velocity=1.0,
                    profile=None, rel_tol=REFERENCE_TOLERANCE,
                    abs_tol=REFERENCE_TOLERANCE, rel_floor=0.10,
                    max_station_offset=0.25):
    """Per-station error of ``u/U_b`` against the Breuer table at ``y/H = 0.5``.

    ``u_over_ub`` may be given already normalised (``bulk_velocity = 1.0``) or
    in solver units (pass ``U_b``).  Measured columns are attributed to the
    nearest reference station within ``max_station_offset`` in ``x/H`` and only
    admissible stations (:func:`periodic_hill_round6.admissible_stations`) are
    compared; everything else is listed under ``skipped`` / ``unmatched``.

    The per-station error itself is delegated to
    :func:`periodic_hill_round6.profile_error`, i.e. the tolerance convention is
    *identical* to the merged round-6 gate (relative where ``|ref| >= 0.1``,
    absolute otherwise).  Using one convention rather than two matters: with
    ``u/U_b = -0.02`` a relative criterion reports a 250 % error for a miss that
    is 5 % of ``U_b``, and a gate that changes convention per round is not a
    gate.  The bands are reported per station so the reader can see which
    criterion decided each row.
    """
    from .periodic_hill_round6 import profile_error

    x = np.asarray(x_over_h, dtype=float).ravel()
    u = np.asarray(u_over_ub, dtype=float).ravel() / bulk_velocity
    if x.shape != u.shape:
        raise ValueError("x_over_h and u_over_ub must have equal length")
    profile = BREUER_PROFILE_Y05 if profile is None else profile
    usable = {row["x_over_H"] for row in admissible_stations(profile=profile)["admissible"]}
    computed, unmatched = {}, []
    for xi, ui in zip(x, u):
        if not np.isfinite(ui):
            unmatched.append(float(xi))
            continue
        nearest = min(profile, key=lambda k: abs(k - xi))
        if abs(nearest - xi) > max_station_offset or nearest not in usable:
            unmatched.append(float(xi))
            continue
        # first column within the offset window wins; later ones would be the
        # same station sampled twice (they are not, on a uniform grid, but the
        # rule must not depend on that)
        computed.setdefault(float(nearest), float(ui))
    base = profile_error(computed, profile=profile, tolerance=rel_tol)
    stations = []
    for row in base["stations"]:
        ref = float(row["reference"])
        stations.append({
            "x_over_H": float(row["x_over_H"]),
            "reference": ref,
            "computed": float(row["computed"]),
            "delta": float(row["computed"]) - ref,
            "error": float(row["error"]),
            "band": "relative" if abs(ref) >= rel_floor else "absolute",
            "pass": bool(abs(row["computed"]) and row["pass"]),
        })
    worst = max(stations, key=lambda s: s["error"], default=None)
    return {
        "stations": stations,
        "n_stations": len(stations),
        "n_pass": sum(1 for s in stations if s["pass"]),
        "worst": worst,
        "rms_error": base["rms_error"],
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "rel_floor": rel_floor,
        "ok": bool(base["pass"]),
        "unmatched_columns": unmatched,
        "skipped": sorted(k for k in profile if k not in usable),
    }


@dataclass
class Snapshot:
    """One time-averaged state, as produced by the round-9 driver."""

    time: float
    x_s_over_H: float
    x_r_over_H: float
    body_force: float
    bulk_velocity: float
    max_nu_t_ratio: float
    samples: int

    def as_dict(self):
        d = self.__dict__.copy()
        for key, value in d.items():
            if isinstance(value, float) and not math.isfinite(value):
                d[key] = None
        return d