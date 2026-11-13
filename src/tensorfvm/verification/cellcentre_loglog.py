"""Three-grid log-log-slope order verification for a cell-centred FV scheme.

Why a separate module
---------------------
A cell-centred finite-volume solution is a vector of **cell averages**
``u_h = (1/|V_i|) ∫_{V_i} u``.  Two consequences are easy to get wrong when the
observed order is read off a log-log plot, and both are measured here instead of
being asserted:

``sampling``
    comparing ``u_h`` against the exact value **at the cell centroid** adds the
    geometric restriction error ``avg(u) - u(x_c) = -(h^2/24) u'' + O(h^4)``.
    That term is ``O(h^2)``: it inflates the error constant of a second-order
    scheme by up to a factor 2 and it **floors** the log-log slope of any
    higher-order scheme at 2.  The exact cell average has a closed form for the
    periodic entropy wave used here, so both metrics can be measured.
``ratio``
    with a non-uniform ladder (``r21 != r32``) the textbook
    ``p = ln(e21/e32)/ln r`` and the uniform-``r`` Richardson extrapolation are
    both wrong; the three-grid equation of :mod:`tensorfvm.verification.richardson`
    is not.  The size of the mistake is reported.

The estimators themselves are *not* re-implemented: the three-grid readout uses
:func:`tensorfvm.verification.richardson.observed_order` (exact interpolation for
non-uniform ratios) and the log-log least-squares slope is fitted on the same
three points, so the classic "slope of the log-log plot" and the three-grid
Richardson order can be compared on identical data.

Everything is deterministic (fixed grids, fixed CFL, no randomness), pure NumPy
plus the compressible cell-centred solver, and every metric below is computed
from the analytic solution -- no number is typed in by hand.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .richardson import observed_order as richardson_order
from .richardson import richardson_gci

#: Amplitude of the periodic density wave ``rho = 1 + A sin(k x)``.
AMPLITUDE = 0.2
#: Uniform advection velocity of the exact Euler solution.
VELOCITY = 0.5
#: Wavenumber of the wave; one full period per unit length.
WAVENUMBER = 2.0 * math.pi
#: Owner precision standard (relative deviation that is *not* acceptable).
THRESHOLD_ACCEPTABLE = 0.15
#: Aspirational band used in this study's verdicts.
THRESHOLD_TIGHT = 0.03


# ----------------------------------------------------------------------
# exact solution: point values and cell averages
# ----------------------------------------------------------------------
def exact_point(nx: int, t: float, *, amplitude: float = AMPLITUDE,
                velocity: float = VELOCITY,
                wavenumber: float = WAVENUMBER) -> np.ndarray:
    """Exact ``rho`` at the centroids of ``nx`` uniform cells of ``[0, 1)``."""
    h = 1.0 / nx
    x = (np.arange(nx) + 0.5) * h
    return 1.0 + amplitude * np.sin(wavenumber * (x - velocity * t))


def exact_cell_average(nx: int, t: float, *, amplitude: float = AMPLITUDE,
                       velocity: float = VELOCITY,
                       wavenumber: float = WAVENUMBER) -> np.ndarray:
    """Exact volume average of ``rho`` over ``nx`` uniform cells of ``[0, 1)``.

    Closed form: ``(1/h)∫ 1 + A sin(k(x-ut)) dx
    = 1 + A[cos(k(a-ut)) - cos(k(b-ut))]/(k h)``.
    """
    h = 1.0 / nx
    a = np.arange(nx) * h
    b = a + h
    return 1.0 + amplitude * (np.cos(wavenumber * (a - velocity * t))
                             - np.cos(wavenumber * (b - velocity * t))) / (wavenumber * h)


def cell_average_quadrature(nx: int, t: float, subcell: int = 16, **kw) -> np.ndarray:
    """Same average by midpoint quadrature on ``subcell`` sub-cells (cross-check)."""
    h = 1.0 / nx
    off = h * (np.arange(subcell) + 0.5) / subcell
    sub = (np.arange(nx) * h)[:, None] + off[None, :]
    amplitude = kw.get("amplitude", AMPLITUDE)
    velocity = kw.get("velocity", VELOCITY)
    wavenumber = kw.get("wavenumber", WAVENUMBER)
    return (1.0 + amplitude * np.sin(wavenumber * (sub - velocity * t))).mean(axis=1)


# ----------------------------------------------------------------------
# error norms (volume weighted, so a non-uniform metric cannot slip in)
# ----------------------------------------------------------------------
def error_norms(computed: np.ndarray, reference: np.ndarray,
                widths: np.ndarray | None = None) -> Dict[str, float]:
    """Volume-weighted ``L1``, ``L2`` (RMS form) and ``Linf`` of a cell field."""
    e = np.asarray(computed, dtype=float) - np.asarray(reference, dtype=float)
    if widths is None:
        widths = np.full(e.shape, 1.0 / e.size)
    w = np.asarray(widths, dtype=float)
    w = w / w.sum()
    return {
        "l1": float(np.sum(w * np.abs(e))),
        "l2": float(math.sqrt(np.sum(w * e * e))),
        "linf": float(np.max(np.abs(e))),
        "l2_unnormalised": float(math.sqrt(np.mean(e * e))),
    }


# ----------------------------------------------------------------------
# the three-grid log-log slope
# ----------------------------------------------------------------------
def spacings(nx: Sequence[int]) -> List[float]:
    """``h = 1/nx``, ordered coarse -> fine exactly as ``nx`` is given."""
    return [1.0 / int(n) for n in nx]


def adjacent_slopes(h: Sequence[float], e: Sequence[float]) -> List[float]:
    """Pairwise log-log slopes ``ln(e_i/e_{i+1}) / ln(h_i/h_{i+1})``."""
    out = []
    for i in range(len(e) - 1):
        out.append(math.log(e[i] / e[i + 1]) / math.log(h[i] / h[i + 1]))
    return out


def loglog_slope(h: Sequence[float], e: Sequence[float]) -> float:
    """Least-squares slope of ``ln e`` against ``ln h`` (the log-log plot slope).

    With exactly three points this is an interpolation: it is *defined* for any
    positive series (it does not notice a non-monotone one), which is precisely
    why the plausibility gate of :mod:`tensorfvm.verification.richardson` is
    applied on top of it instead of replacing it.
    """
    x = np.log(np.asarray(h, dtype=float))
    y = np.log(np.asarray(e, dtype=float))
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den == 0.0:
        raise ValueError("refinement series has no spacing spread: slope undefined")
    return float(np.sum((x - xm) * (y - ym)) / den)


def three_grid_readout(nx: Sequence[int], e: Sequence[float], *,
                       theory_order: float) -> Dict:
    """Read the observed order off one triplet with three estimators.

    ``p_loglog`` is the slope of the log-log plot, ``p_adjacent`` the pairwise
    slopes and ``p_richardson`` the exact three-grid solution (identical to
    ``p_loglog`` only when the three points lie on an exact power law).
    """
    h = spacings(nx)
    e = [float(v) for v in e]
    if any(v <= 0.0 for v in e):
        raise ValueError("log-log slope requires strictly positive errors")
    adjacent = adjacent_slopes(h, e)
    p_loglog = loglog_slope(h, e)
    fine_to_coarse_h = list(reversed(h))
    fine_to_coarse_e = list(reversed(e))
    p_rich = float(richardson_order(fine_to_coarse_h, fine_to_coarse_e))
    p_rich_rel = (abs(p_rich - theory_order) / theory_order
                  if math.isfinite(p_rich) else float("inf"))
    return {
        "nx": [int(n) for n in nx],
        "h": h,
        "errors": e,
        "refinement_ratios": [h[i] / h[i + 1] for i in range(len(h) - 1)],
        "p_adjacent": adjacent,
        "p_loglog": p_loglog,
        "p_richardson": p_rich,
        "p_theory": float(theory_order),
        "loglog_rel_dev": abs(p_loglog - theory_order) / theory_order,
        "richardson_rel_dev": p_rich_rel,
        "loglog_pass_15": bool(abs(p_loglog - theory_order) / theory_order
                               <= THRESHOLD_ACCEPTABLE),
        "loglog_pass_3": bool(abs(p_loglog - theory_order) / theory_order
                              <= THRESHOLD_TIGHT),
        "richardson_pass_15": bool(p_rich_rel <= THRESHOLD_ACCEPTABLE),
    }


def consistency(p_first: Dict, p_second: Dict) -> Dict:
    """Overlapping-triplet consistency: two triplets must read the same order.

    Celik et al. (2008) eq. (17) is an algebraic identity when ``p`` comes from
    the triplet being tested, so this two-triplet comparison is used instead as
    the non-vacuous asymptotic-range gate.
    """
    a, b = float(p_first["p_richardson"]), float(p_second["p_richardson"])
    if not (math.isfinite(a) and math.isfinite(b)):
        return {"p_first": a, "p_second": b, "rel_diff": float("inf"),
                "pass_15": False, "note": "one triplet has no order signal"}
    rel = abs(a - b) / max(abs(a), abs(b))
    return {"p_first": a, "p_second": b, "rel_diff": rel,
            "pass_15": bool(rel <= THRESHOLD_ACCEPTABLE),
            "note": "non-vacuous asymptotic-range gate (replaces eq. 17)"}


# ----------------------------------------------------------------------
# the non-uniform-ratio trap
# ----------------------------------------------------------------------
def richardson_uniform_assumed(fine: float, medium: float, r: float,
                               p: float) -> float:
    """Textbook extrapolation that assumes one ratio ``r`` for both pairs."""
    return (r ** p * fine - medium) / (r ** p - 1.0)


def uniform_ratio_bias(nx: Sequence[int], e: Sequence[float], *,
                       theory_order: float) -> Dict:
    """Quantify the error of assuming a *uniform* ratio on a non-uniform triplet.

    With ``e = (coarse, medium, fine)`` the textbook three-grid formula is
    ``p = ln(|(e3-e2)/(e2-e1)|)/ln r`` where ``r`` is the (single) refinement
    ratio.  When ``r32 = h_coarse/h_medium != r21 = h_medium/h_fine`` that
    formula has no solution-consistency: it silently divides the coarse
    difference by the wrong ratio.  Both the order and the extrapolated limit
    inherit the bias; the exact three-grid solve
    (:func:`tensorfvm.verification.richardson.observed_order`, used by
    :func:`three_grid_readout`) does not.
    """
    h = spacings(nx)
    r21, r32 = h[1] / h[2], h[0] / h[1]
    study = richardson_gci(e[2], e[1], e[0], [h[2], h[1], h[0]],
                           theory_order=theory_order)
    p_true = float(study.order)
    limit_true = float(study.extrapolated)
    try:
        p_uniform = math.log(abs((e[0] - e[1]) / (e[1] - e[2]))) / math.log(r21)
    except (ValueError, ZeroDivisionError):
        p_uniform = float("nan")
    limit_uniform = (richardson_uniform_assumed(e[2], e[1], r21, p_uniform)
                     if math.isfinite(p_uniform) else float("nan"))
    # An error norm has a zero limit, so a *relative* limit comparison is
    # meaningless there; it is withheld (nan) and the absolute difference is
    # reported together with the scale of the series.
    scale = max(abs(v) for v in e)
    limit_meaningful = bool(limit_true and abs(limit_true) > 1e-6 * scale)
    abs_diff = (abs(limit_uniform - limit_true)
                if math.isfinite(limit_uniform) else float("nan"))
    return {
        "nx": [int(n) for n in nx],
        "r21": r21,
        "r32": r32,
        "ratio_mismatch": abs(r21 / r32 - 1.0),
        "p_true": p_true,
        "p_uniform_form": p_uniform,
        "order_rel_dev": abs(p_uniform - p_true) / abs(p_true),
        "order_rel_dev_vs_theory": abs(p_uniform - theory_order) / theory_order,
        "limit_true": limit_true,
        "limit_uniform_form": limit_uniform,
        "limit_scale": scale,
        "limit_relative_meaningful": limit_meaningful,
        "limit_abs_diff": abs_diff,
        "limit_abs_diff_over_scale": (abs_diff / scale
                                     if math.isfinite(abs_diff) else float("nan")),
        "limit_rel_dev": (abs_diff / abs(limit_true) if limit_meaningful
                          else float("nan")),
        "uniform_ladder": bool(abs(r21 / r32 - 1.0) <= 1e-12),
    }


# ----------------------------------------------------------------------
# driving the cell-centred solver
# ----------------------------------------------------------------------
def run_case(nx: int, *, order: int, cfl: float = 0.4, t_end: float = 0.2,
             amplitude: float = AMPLITUDE, velocity: float = VELOCITY) -> Dict:
    """One cell-centred FV run of the periodic entropy wave.

    The solver is :class:`tensorfvm.compressible.solver.EulerSolver` (collocated
    cell-centred FV, MUSCL-Hancock + HLLC), reached through the already-merged
    :func:`tensorfvm.verification.sod1978.run_smooth_wave` so that this module
    adds measurements, not a second solver path.
    """
    from .sod1978 import run_smooth_wave

    run = run_smooth_wave(nx, order=order, cfl=cfl, t_end=t_end,
                          amplitude=amplitude, velocity=velocity)
    t = float(run["time"])
    h = 1.0 / nx
    widths = np.full(nx, h)
    point = exact_point(nx, t, amplitude=amplitude, velocity=velocity)
    average = exact_cell_average(nx, t, amplitude=amplitude, velocity=velocity)
    rho = np.asarray(run["rho"], dtype=float)
    return {
        "nx": int(nx),
        "order": int(order),
        "cfl": float(cfl),
        "t": t,
        "h": h,
        "steps": int(run["steps"]),
        "cell_average": error_norms(rho, average, widths),
        "point": error_norms(rho, point, widths),
        "sampling_bias": error_norms(point, average, widths),
        "rho": rho,
        "rho_exact_average": average,
        "rho_exact_point": point,
        "x": np.asarray(run["x"], dtype=float),
    }


def ladder(nx: Iterable[int], *, order: int, cfl: float = 0.4) -> List[Dict]:
    """Run the ladder (one row per grid), coarse -> fine as given."""
    return [run_case(int(n), order=order, cfl=cfl) for n in nx]


def metric_series(rows: Sequence[Dict], metric: str, norm: str) -> List[float]:
    """Extract ``[row[metric][norm] for row in rows]`` in ladder order."""
    return [float(row[metric][norm]) for row in rows]


def study(grids: Sequence[int], *, order: int, cfl: float = 0.4,
          theory_order: float | None = None) -> Dict:
    """Full readout: solver ladder + three-grid slopes for both metrics.

    ``theory_order`` defaults to the scheme's declared order, which is what the
    observed slope is compared against; the comparison is reported for the
    cell-average metric (correct for a FV solution) and the centroid-point
    metric (the sampling-biased alternative).
    """
    theory = float(order if theory_order is None else theory_order)
    rows = ladder(grids, order=order, cfl=cfl)
    out: Dict = {"grids": [int(n) for n in grids], "order": int(order),
                 "cfl": float(cfl), "theory_order": theory, "rows": []}
    for row in rows:
        out["rows"].append({k: v for k, v in row.items()
                            if k not in ("rho", "rho_exact_average",
                                         "rho_exact_point", "x")})
    for metric in ("cell_average", "point", "sampling_bias"):
        for norm in ("l1", "l2", "linf"):
            series = metric_series(rows, metric, norm)
            read = three_grid_readout(grids, series, theory_order=theory)
            read["metric"] = metric
            read["norm"] = norm
            out.setdefault("readouts", []).append(read)
    return out


__all__ = [
    "AMPLITUDE", "VELOCITY", "WAVENUMBER", "THRESHOLD_ACCEPTABLE",
    "THRESHOLD_TIGHT", "exact_point", "exact_cell_average",
    "cell_average_quadrature", "error_norms", "spacings", "adjacent_slopes",
    "loglog_slope", "three_grid_readout", "consistency",
    "richardson_uniform_assumed", "uniform_ratio_bias", "run_case", "ladder",
    "metric_series", "study",
]