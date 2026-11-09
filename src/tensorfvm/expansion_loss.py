"""Loss-coefficient post-processing for the planar sudden expansion.

The Borda--Carnot value ``K_BC = (1 - 1/AR)**2`` is an *inviscid momentum*
balance for a sudden expansion: it is the total-pressure loss between the inlet
plane and the (idealised, well-mixed) downstream state, with no wall friction.

Three independent problems make a naive comparison to ``K_BC`` meaningless, and
all three are handled here explicitly:

1. **Station dependence.**  A ``K`` evaluated at an outlet plane accumulates the
   wall friction of the whole development length.  On the 12 D long
   ``Re_D = 100`` channel of :mod:`tensorfvm.incompressible_baselines` that
   friction is of the same order as the expansion loss itself, so the same mesh
   gives ``K = 1.026`` on a 12 D domain and ``K = 0.889`` on a 20 D domain
   (``docs/incompressible-baselines/baseline-runs/expansion_{,long_}n010.json``).
   A quantity that changes with the length of the pipe is not a loss
   coefficient; the fix is to fit the *linear* total-pressure decay of the
   developed region and extrapolate it back to the step plane
   (:func:`step_plane_value`), which removes the friction.

2. **Profile bias of the kinetic-energy term.**  ``p_t = p + 0.5 rho u**2``
   uses the local ``u``; a mass-flux weighted column average of ``p_t``
   therefore carries ``<u**3>/<u>`` instead of ``<u>**2``, i.e. the kinetic
   energy factor ``alpha = <u**3>/<u>**3`` of the (parabolic) downstream
   profile.  For a parabolic profile ``alpha = 1.8`` (2-D, ``54/35``); with a
   mean velocity ``U_out = U_in/AR`` this inflates ``K`` by
   ``(alpha - 1)/AR**2`` -- ``0.05`` here, but it is not a loss at all.

3. **Static-pressure route.**  Momentum conservation on a control volume with
   no shear gives the exact algebraic identity

       ``K = 1 - 1/AR**2 - Cp``,   ``Cp = (p_out - p_in)/q_in``,

   which uses the *primitive* static pressure (no kinetic-energy bias) and the
   exact mass-conservation velocity ratio.  This is the quantity that should be
   compared with ``K_BC``; it is the definition the companion MAC
   implementation reports as ``k_from_static_recovery``.

Nothing here is a solver: every function is a pure post-processing helper over
stored fields, so the archived runs can be re-graded without re-solving.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# reference values
# --------------------------------------------------------------------------- #
def borda_carnot(area_ratio: float = 4.0) -> float:
    """Inviscid sudden-expansion loss ``(1 - 1/AR)**2`` (uniform inlet)."""
    return (1.0 - 1.0 / area_ratio) ** 2


def dynamic_head(u: float) -> float:
    """Half the squared velocity, in the dimensionless units of the case."""
    return 0.5 * np.asarray(u, dtype=float) ** 2


# --------------------------------------------------------------------------- #
# section averages
# --------------------------------------------------------------------------- #
def _as_flow_field(field: np.ndarray, nx: int, ny: int, name: str) -> np.ndarray:
    """Return a cell-centred ``(nx, ny)`` view of ``field``.

    The stored arrays may be cell-centred or one wider in either direction
    (face-staggered).  Extra rows/columns are dropped so that the first ``nx *
    ny`` block lines up with the cell indexing used by the solver; a shape that
    matches neither is an error rather than a silent reinterpretation.
    """
    arr = np.asarray(field, dtype=float)
    if arr.shape[:2] == (nx, ny):
        return arr[:nx, :ny]
    if arr.shape[0] >= nx and arr.shape[1] >= ny:
        return arr[:nx, :ny]
    raise ValueError(f"{name} has shape {arr.shape}, expected at least ({nx}, {ny})")


def section_average(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted mean of ``values``; negative/zero weights are discarded.

    With ``weights=None`` this is the plain area average (the convention of the
    archived baseline runs).  With the streamwise velocity as weight it is the
    mass-flux average.
    """
    v = np.asarray(values, dtype=float)
    if weights is None:
        return float(v.mean())
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("mass-flux weights sum to zero: no inflow in this section")
    return float((w * v).sum() / total)


def kinetic_energy_factor(u: np.ndarray) -> float:
    """``alpha = <u**3>/<u>**3`` for the section (``1`` uniform, ``1.8`` parabolic)."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, None)
    mean = float(u.mean())
    if mean <= 0.0:
        return float("nan")
    return float((u ** 3).mean() / mean ** 3)


# --------------------------------------------------------------------------- #
# station metrics
# --------------------------------------------------------------------------- #
def station_metrics(
    u: np.ndarray,
    p: np.ndarray,
    inlet_rows: Sequence[int],
    outlet_column: int,
    area_ratio: float = 4.0,
) -> dict:
    """All loss routes evaluated *at a station*, without any extrapolation.

    ``u`` and ``p`` are cell-centred ``(nx, ny)`` arrays.  ``inlet_rows`` selects
    the inlet-window rows of column ``0``; ``outlet_column`` is the downstream
    station.  The returned ``k_*`` values are dimensionless, referenced to the
    inlet dynamic head ``0.5 U_in**2``.
    """
    rows = np.asarray(list(inlet_rows), dtype=int)
    if rows.size == 0:
        raise ValueError("inlet_rows must not be empty")
    j0, j1 = int(rows.min()), int(rows.max()) + 1
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)

    u_in = section_average(u[0, j0:j1])
    p_in = section_average(p[0, j0:j1])
    q_in = dynamic_head(u_in)
    if q_in <= 0.0:
        raise ValueError("no inflow at the inlet station")

    u_col = u[outlet_column, :]
    p_col = p[outlet_column, :]
    u_out = section_average(u_col)
    p_out = section_average(p_col)

    # mass-flux weighted total pressure at the two stations
    pt_in = section_average(p[0, j0:j1] + dynamic_head(u[0, j0:j1]), u[0, j0:j1])
    pt_out = section_average(p_col + 0.5 * u_col ** 2, u_col)
    # plain area average of the same total pressure (archived convention)
    pt_in_bulk = section_average(p[0, j0:j1]) + dynamic_head(u_in)
    pt_out_bulk = section_average(p_col) + dynamic_head(u_out)

    cp = (p_out - p_in) / q_in
    k_static = (1.0 - 1.0 / area_ratio ** 2) - cp
    return dict(
        u_in=u_in,
        u_out=u_out,
        p_in=p_in,
        p_out=p_out,
        q_in=q_in,
        velocity_ratio=u_out / u_in,
        cp=cp,
        k_station_bulk=(pt_in_bulk - pt_out_bulk) / q_in,
        k_station_flux=(pt_in - pt_out) / q_in,
        k_static_recovery=k_static,
        alpha_out=kinetic_energy_factor(u_col),
        kinetic_energy_bias=float(((kinetic_energy_factor(u_col) - 1.0)
                                   * (u_out / u_in) ** 2)),
        outlet_column=int(outlet_column),
    )


# --------------------------------------------------------------------------- #
# developed-region extrapolation
# --------------------------------------------------------------------------- #
def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope ``dy/dx`` (raises on a degenerate abscissa)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two samples for a slope")
    xm = x - x.mean()
    denom = float((xm ** 2).sum())
    if denom <= 0.0:
        raise ValueError("degenerate abscissa: cannot fit a slope")
    return float((xm * (y - y.mean())).sum() / denom)


def step_plane_value(station_value: float, slope_per_length: float,
                     distance: float) -> float:
    """Extrapolate a station value back to the step plane (``x = 0``).

    ``slope_per_length`` is ``d(value)/dx`` in the developed region and
    ``distance`` is the streamwise distance from the step plane to the station
    where ``station_value`` was taken.  A zero slope returns the station value
    unchanged, which is the regression check that the bare station metric is a
    special case.
    """
    return float(station_value - slope_per_length * distance)


def developed_slope(
    x: np.ndarray,
    column_values: Iterable[float],
) -> float:
    """Slope of a per-column profile over the supplied (developed) window."""
    y = np.asarray([float(v) for v in column_values], dtype=float)
    return linear_slope(np.asarray(x, dtype=float), y)


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def relative_error(value: float, reference: float) -> float:
    """``|value - reference| / |reference|`` (``nan`` if non-finite)."""
    if not (math.isfinite(value) and math.isfinite(reference)) or reference == 0.0:
        return float("nan")
    return abs(value - reference) / abs(reference)


def verdict(value: float, reference: float, tolerance: float = 0.03) -> str:
    """``PASS`` / ``FAIL`` / ``WITHHELD`` on the 3 % platform bar."""
    err = relative_error(value, reference)
    if not math.isfinite(err):
        return "WITHHELD"
    return "PASS" if err <= tolerance else "FAIL"