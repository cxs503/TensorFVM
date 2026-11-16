"""Dam-break benchmark: Ritter (dry bed) and Stoker (wet bed) at ``t* = 2``.

The benchmark is the shallow-water analogue of the Sod shock tube: a dam at
``x = x_dam`` separates still water of depth ``h0`` (left) from depth ``h1``
(right).  Two exact references are used, both evaluated in closed form:

* ``ritter-dry``  -- ``h1 = 0``; front ``x_f = 2 t sqrt(g h0)`` (Ritter 1892).
* ``stoker-wet``  -- ``h0 > h1 > 0``; rarefaction + constant state + bore of
  speed ``U`` (Stoker 1957, chapter 10).

Metrics (predeclared, see ``docs/shallow-water-dam-break/REPORT.md``):

``front_fan`` (dry bed headline) / ``shock_position`` (wet bed headline)
    Wave position.  Dry bed: mean over ``FAN_WINDOW`` of the per-cell exact
    identity ``x_f = x_i + 3 t sqrt(g h_i)`` of the Ritter fan
    (:func:`fan_front_estimate`) -- the estimator inherits the *shape* of the
    closed form instead of a depth threshold.  Wet bed: the sub-cell position
    where the depth crosses ``(h1 + h2) / 2``, linearly interpolated between
    cell centres.  Relative error against the closed form.

``front_wetted`` (dry bed, legacy diagnostic)
    Outermost cell with ``h > 0``.  It looks threshold-free but is in fact a
    depth-floor detector: the tip is a zero-depth cusp, so the reading is fixed
    by the numerical depth floor and does **not** converge under refinement
    (1.56 % -> 2.66 % -> 2.89 % -> 2.46 % for n = 400 ... 3200, while the fan
    estimator on the same states goes 0.106 % -> 0.039 % -> 0.017 % -> 0.008 %).
    Reported in every run and in the verdict for continuity, not as a headline.

``step_ratio``
    The still-water level step across the bore, ``h2 / h1`` measured as the
    mean depth over the middle half of the exact plateau divided by the mean
    depth in the undisturbed downstream region.  Relative error.

``depth_l1`` / ``depth_linf``
    Relative ``L1`` error of the depth profile over the whole domain,
    ``sum|h - h_exact| dx / sum h_exact dx``, and ``max|h - h_exact| / h0``.

``monotonicity``
    Total variation of the numerical depth against the exact TV (``h0 - h1``
    for the wet bed, ``h0`` for the dry bed) plus the number of interior local
    extrema of amplitude ``> 1e-4 h0``: the "waveform must stay monotone, no
    spurious oscillation" part of the acceptance criterion.

``scaled_front_error`` (dry bed only, secondary)
    Position where the depth crosses ``h0/100`` plus the exact fan offset
    ``3 t sqrt(g h_level)`` back to the tip.  Reported because the tip is a
    zero-depth cusp that any finite-volume scheme smears; the wetted-edge
    measure above is the primary one.

All relative errors are non-negative ``|measured - exact| / exact``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from .exact import (
    check_stoker_solution,
    ritter_front_position,
    ritter_state,
    stoker_constant_state,
    stoker_state,
    t_star_to_time,
)
from .swe1d import ShallowWater1D

__all__ = [
    "DamBreakCase",
    "RITTER_DRY",
    "STOKER_WET",
    "CASES",
    "run_case",
    "series_metrics",
    "convergence_study",
    "sensitivity_study",
    "front_definition_study",
    "exactness_checks",
    "write_artifacts",
    "figure_profiles_and_convergence",
    "TOLERANCE",
    "BORE_SUBCELL_MAX_CELLS",
    "plateau_from_depth_modes",
    "depth_mode_study",
    "PLATEAU_MODE_BINS",
    "PLATEAU_MIN_JUMP_FRACTION",
    "SERIES_MIN_GRADED_FRACTION",
]

#: acceptance threshold on every headline metric (task criterion: < 5 %)
TOLERANCE = 0.05

#: coarsest grid whose errors are admitted as evidence (declared, not fitted).
#: A single constant so that every "at accepted grid" statement in the module
#: and in the reports refers to the same resolution floor.
ACCEPTED_MIN_N = 800

#: depth level (fraction of h0) of the secondary dry-bed front metric
FRONT_LEVEL_FRACTION = 1e-2

#: amplitude (fraction of h0) above which a local extremum counts as oscillation
EXTREMUM_FRACTION = 1e-4

#: depth-fraction window ``(h / h0)`` used by :func:`fan_front_estimate`.
#:
#: The window must sit **inside the resolved rarefaction** -- deep enough that
#: the numerical depth is not dominated by the smeared zero-depth cusp, shallow
#: enough that it is not buried in the still-water plateau.  The bounds are
#: declared here (not inferred from a run) and the sensitivity to them is
#: measured in ``docs/verification-dam-break-front-metric/REPORT.md``.
FAN_WINDOW = (1e-2, 2e-1)

#: depth contours (fractions of ``h0``) at which the **dry-bed position** of the
#: Ritter profile is graded against :func:`~tensorfvm.shallow_water.exact.
#: ritter_level_position`, with no analytic quantity inside the measurement.
#:
#: ``level = 1`` is deliberately absent: the undisturbed tail carries ``h = h0``
#: over an interval, so a strict-crossing detector has no crossing there (it
#: returns ``nan``); levels approaching 0 approach the zero-depth tip, which any
#: finite-volume scheme smears over ``O(dx)``.  The five levels below span the
#: resolved fan.  Declared before any run, not chosen from the results.
LEVEL_POSITION_LEVELS = (0.5, 0.25, 0.1, 0.04, 0.01)

#: contour used by the oracle-free ``x(t)`` series (well inside the fan).
LEVEL_POSITION_SERIES_LEVEL = 0.04

#: cell count the exact front must have travelled before a sample of the
#: ``x_f(t)`` series is *graded*.
#:
#: This is an a-priori resolution criterion, declared before any run: while the
#: front/bore still sits inside the smearing region of the initial discontinuity
#: (a few cells), its measured position is a statement about the smearing, not
#: about the wave.  Samples below the threshold are still recorded and reported
#: (as ``unresolved_max_error``); they are simply not admitted to the pass/fail
#: series metric.
SERIES_MIN_CELLS = 4.0

#: contour travel (in cells) a level-position series sample must reach before it
#: is graded.
#:
#: Declared from the error budget, not from the data: a contour located to
#: within one cell has relative error ``~dx / x_contour``, so a sample can only
#: be *informative* at the 5 % gate when the contour sits at least
#: ``1 / TOLERANCE = 20`` cells downstream of the dam.  Near ``t = 0`` the
#: contour is a fraction of a cell from the dam and the relative error is large
#: at every resolution (measured: 10.9 % at n = 800 and 5.75 % at n = 1600 at
#: ``t* ~ 0.085`` -- i.e. one cell, halving with ``dx``, exactly the predicted
#: first-order behaviour).  Such samples are recorded and excluded, not hidden.
LEVEL_SERIES_MIN_CELLS = 1.0 / TOLERANCE

#: fan-window cell count below which a **dry-bed** series sample is not graded.
#:
#: ``SERIES_MIN_CELLS`` bounds how far the front has travelled; it does *not*
#: bound how many cells the estimator actually sees.  The fan window
#: (``h/h0 in FAN_WINDOW``) can hold as few as three cells while the front has
#: already travelled 7.46 cells, and there the estimator degenerates to a
#: cell-centre locator: measured on the committed artefacts the front error is
#: ``0.493 dx`` (half a cell) -- *identically* at ``n = 200`` and ``n = 400``
#: (6.613 % both, because both samples carry 3 fan cells / 7.46 travelled
#: cells), i.e. a resolution-**invariant** relative error rather than a
#: discretisation error.  That is what made the graded series error
#: non-monotone (4.70 % -> 6.61 % -> 6.61 % -> 1.40 % -> 0.43 % for
#: n = 100 ... 1600) and produced a spurious ``front_series<5%`` failure on the
#: coarse grids.  The identity itself is exact at *any* cell count when applied
#: to the exact profile (fixed-point check, error < 1e-15), so the floor is a
#: statement about the *numerical tip* being resolved, not about estimator bias.
#:
#: Evidence and the derivation of the threshold:
#: ``docs/case-fvm-dambreak-stoker/ANALYSIS-series-fan-cell-floor.md``.
SERIES_MIN_FAN_CELLS = 8

#: fraction of the ``x_f(t)`` series samples that must be *graded* before the
#: series verdict is admitted as evidence at all.
#:
#: The series gate is ``max graded error <= TOLERANCE``; without a floor on the
#: number of graded samples that gate could be satisfied by a run in which the
#: front was only resolvable for one or two samples ("passing by measuring
#: nothing").  The floor is declared before any run and both cases clear it by
#: a wide margin (24 of 25 samples graded, i.e. 0.96, on n = 800 and 1600).
SERIES_MIN_GRADED_FRACTION = 0.5

#: Gradient cut of the data-driven plateau window, as a fraction of the measured
#: ``max|dh/dx|`` (the jump).  Kept at the historical 1e-2 so the window is
#: unchanged on the range where it was validated (n <= 3200).
PLATEAU_THRESH_FRACTION = 1e-2
#: Declared validity gate for the plateau window: ``max|h - mean(h)| / jump``
#: must stay below this, otherwise the profile has no *flat* plateau
#: (``bore_valid = False``; the amplitude is still reported but must not be
#: graded).  Measured: 11 % of the jump at n = 800, 1.4 % at 1600, 0.2 % at
#: 6400 -- so the gate admits n >= 1600 and rejects n <= 800 honestly.
PLATEAU_FLAT_FRACTION = 5e-2
#: How many decades the plateau cut may be tightened before the detector gives
#: up (``plateau_unbounded``).  Three decades cover n <= 12800 for the Stoker
#: case (measured); beyond the declared count the honest answer is "no fan
#: found", not a wider window.
PLATEAU_THRESH_MAX_DECADES = 3

#: Minimum fraction of the exact level step ``h2 - h1`` that must be *present*
#: inside the oracle plateau window, otherwise the window reading is not a
#: measurement of ``h2`` and must not be graded.
#:
#: The window is placed from the closed form (the middle half of the exact
#: constant state), so it can be narrower than the numerically smeared bore once
#: the depth ratio ``h2/h1 = sqrt(8 h0/h1)`` grows: measured for ``h1/h0 =
#: 1e-6`` at ``n = 800`` the window contains only the undisturbed depth, so the
#: reported ``h2_error`` was ``99.96 %`` *without the bore being read at all*
#: (``docs/case-fvm-dambreak-stoker-degenerate/REPORT.md``).  Declared before
#: any run; the reading sees the full step (fraction 1.00) for every depth ratio
#: on ``n >= 800`` once ``h1/h0 >= 1e-3``.
PLATEAU_MIN_JUMP_FRACTION = 0.5

#: predeclared resolution criterion for the data-driven wet-bed bore position
#: (:func:`data_driven_plateau`), in units of ``dx``.
#:
#: Half a cell: a reading that cannot resolve below this is a cell-centre
#: detector quantised to ``dx`` (the same defect class as the dry-bed wetted
#: edge) and must not be used as a *position* metric.  Declared before any run;
#: the measured sub-cell error is 0.063 ... 0.12 ``dx`` on n = 200 ... 1600
#: (``scripts/verify_dambreak_bore_subcell.py``).
BORE_SUBCELL_MAX_CELLS = 0.5


@dataclass(frozen=True)
class DamBreakCase:
    """Geometry and reference solution of one dam-break case."""

    key: str
    kind: str            # "dry" | "wet"
    h0: float
    h1: float
    gravity: float
    x_dam: float
    x_min: float
    x_max: float
    t_star_end: float
    reference: str

    @property
    def t_end(self) -> float:
        return t_star_to_time(self.t_star_end, self.h0, self.gravity)

    def exact(self, x, t=None):
        """Exact ``(h, u)`` at ``x`` and time ``t`` (defaults to ``t_end``)."""
        t = self.t_end if t is None else t
        if self.kind == "dry":
            return ritter_state(x, t, self.h0, self.gravity, self.x_dam)
        return stoker_state(x, t, self.h0, self.h1, self.gravity, self.x_dam)

    def exact_front(self, t=None) -> float:
        """Position of the leading wave (front or bore) at ``t``."""
        t = self.t_end if t is None else t
        if self.kind == "dry":
            return ritter_front_position(t, self.h0, self.gravity, self.x_dam)
        sol = stoker_constant_state(self.h0, self.h1, self.gravity)
        return self.x_dam + sol["shock_speed"] * t

    def initial_mass(self) -> float:
        """Volume per unit width of the initial condition.

        The left inflow boundary is placed at ``x_min`` for an analytic
        estimate only when it lies upstream of the dam; the solver's own
        volume is the reference for the drift metric.
        """
        return self.h0 * (self.x_dam - self.x_min) + self.h1 * (self.x_max - self.x_dam)

    def summary(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "h0": self.h0,
            "h1": self.h1,
            "gravity": self.gravity,
            "x_dam": self.x_dam,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "t_star_end": self.t_star_end,
            "t_end": self.t_end,
            "reference": self.reference,
        }


RITTER_DRY = DamBreakCase(
    key="ritter-dry", kind="dry", h0=1.0, h1=0.0, gravity=9.81, x_dam=0.0,
    x_min=-4.0, x_max=6.0, t_star_end=2.0,
    reference="Ritter (1892) Z. Ver. Deutsch. Ing. 36:947-954",
)

STOKER_WET = DamBreakCase(
    key="stoker-wet", kind="wet", h0=1.0, h1=0.5, gravity=9.81, x_dam=0.0,
    x_min=-4.0, x_max=6.0, t_star_end=2.0,
    reference="Stoker (1957) Water Waves, ch. 10 (bore on a wet bed)",
)

CASES = {case.key: case for case in (RITTER_DRY, STOKER_WET)}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _relative(measured: float, exact: float) -> float:
    return abs(measured - exact) / abs(exact) if exact != 0.0 else abs(measured)


def _wetted_edge(x: np.ndarray, h: np.ndarray, dry_depth: float = 0.0) -> float:
    """Outermost cell centre whose depth exceeds ``dry_depth``."""
    wet = np.nonzero(h > dry_depth)[0]
    if wet.size == 0:
        return float("nan")
    return float(x[wet[-1]])


def _level_crossing(x: np.ndarray, h: np.ndarray, level: float) -> float:
    """Sub-cell abscissa where ``h`` falls through ``level`` (linear, rightmost)."""
    above = np.nonzero(h > level)[0]
    if above.size == 0:
        return float("nan")
    j = int(above[-1])
    if j + 1 >= h.size:
        return float(x[-1])
    h0, h1 = h[j], h[j + 1]
    if h0 == h1:
        return float(x[j])
    return float(x[j] + (h0 - level) / (h0 - h1) * (x[j + 1] - x[j]))


def _extrema_mask(h: np.ndarray, amplitude: float) -> np.ndarray:
    """Boolean mask of the interior cells counted by :func:`_local_extrema`.

    One predicate, one place: the count (:func:`_local_extrema`) and the
    per-extremum report (:func:`local_extrema_details`) both read this mask, so
    the two can never disagree about *which* cell failed a monotonicity gate.
    """
    h = np.asarray(h, dtype=float)
    mask = np.zeros(h.shape, dtype=bool)
    if h.size < 3:
        return mask
    left = h[1:-1] - h[:-2]
    right = h[2:] - h[1:-1]
    peak = (left > 0) & (right < 0) & (np.abs(left) > amplitude) & (np.abs(right) > amplitude)
    dip = (left < 0) & (right > 0) & (np.abs(left) > amplitude) & (np.abs(right) > amplitude)
    mask[1:-1] = peak | dip
    return mask


def _local_extrema(h: np.ndarray, amplitude: float) -> int:
    """Interior local maxima/minima whose prominence exceeds ``amplitude``."""
    return int(np.count_nonzero(_extrema_mask(h, amplitude)))


def local_extrema_details(h, amplitude, x=None) -> list:
    """Locate and size every extrema counted by :func:`_local_extrema`.

    ``_local_extrema`` answers *how many* cells break monotonicity; a gate that
    fails needs *which* cell, *how large* and *where relative to the analytic
    structure*.  This returns one dict per offending cell, so a report can say
    "one dip of ``1.23e-4 h0`` at ``s = -0.725``, inside Stoker's plateau"
    instead of "local_extrema = 1".

    Parameters
    ----------
    h : array_like
        Depth profile (cell centred).
    amplitude : float
        Same prominence gate as :func:`_local_extrema`; a cell is reported iff
        **both** adjacent slopes exceed it, so ``amplitude`` is a lower bound on
        the reported ``amplitude`` field.
    x : array_like, optional
        Cell abscissae of the same length; when given, each record carries the
        coordinate ``x`` of the offending cell.

    Returns
    -------
    list of dict
        Ordered by cell index, each with ``index``, ``kind`` (``"peak"`` /
        ``"dip"``), ``amplitude`` (the *smaller* of the two adjacent depth
        jumps -- the conservative prominence), ``value`` (the cell depth),
        ``left_slope`` / ``right_slope`` (``h[i] - h[i-1]``, ``h[i+1] - h[i]``)
        and, if ``x`` is given, ``x``.  ``amplitude / h_mean`` style
        normalisation is left to the caller: the routine does not know ``h0``.
    """
    h = np.asarray(h, dtype=float)
    xs = None if x is None else np.asarray(x, dtype=float)
    if xs is not None and xs.shape != h.shape:
        raise ValueError(f"x and h must have the same shape, got {xs.shape} vs {h.shape}")
    out = []
    for j in np.nonzero(_extrema_mask(h, amplitude))[0]:
        left = float(h[j] - h[j - 1])
        right = float(h[j + 1] - h[j])
        rec = {
            "index": int(j),
            "kind": "peak" if left > 0 else "dip",
            "amplitude": float(min(abs(left), abs(right))),
            "value": float(h[j]),
            "left_slope": left,
            "right_slope": right,
        }
        if xs is not None:
            rec["x"] = float(xs[j])
        out.append(rec)
    return out


def data_driven_plateau(x, h, thresh_frac: float = PLATEAU_THRESH_FRACTION) -> dict:
    """Locate the wet-bed bore and plateau from the numerical profile alone.

    This is the F4 fix: the oracle-guided ``h2``/``step_ratio`` in
    :func:`_wet_metrics` measure the plateau *amplitude* at the analytic window
    ``[x_fan_end, x_shock]``, so they are insensitive to a misplaced bore (a
    solver that puts the plateau in the wrong place but at the right amplitude
    still reads a machine-precision ``h2``).  Here **no analytic position or
    level is used**:

    * the bore is the cell of maximal ``|dh/dx|``,
    * the plateau is the contiguous run of cells upstream of it whose
      ``|dh/dx|`` stays below ``thresh_frac * max|dh/dx|``.

    Known limit of that rule (F9, ``docs/case-fvm-dambreak-stoker-hires/``)
    -----------------------------------------------------------------------
    The earlier docstring claimed this cut was grid-independent "because both
    ``|dh/dx|`` scales with 1/dx".  That is only true for the *jump*: the fan is
    a smooth feature whose physical slope is a constant (measured 0.298 ->
    0.308 m/m for n = 800 ... 12800) while ``max|dh/dx|`` doubles per
    refinement (5.8 -> 100).  The ratio fan/jump therefore falls like ``dx``
    (5.1e-2 -> 3.1e-3) and **crosses** the 1e-2 cut between n = 3200 and
    n = 6400; there the walk ran straight through the whole rarefaction fan and
    reported ``h2`` 15.5 % high, ``bore_x`` 0.85 cell off -- with
    ``bore_method == "mid_level_crossing"``, i.e. silently.

    Two declared guards now make that impossible:

    ``bore_method = "plateau_unbounded"`` (all outputs NaN)
        The walk did not find a cell steep enough to stop it before the domain
        edge, i.e. the profile does not contain a *detectable* fan -- so no
        plateau was delimited and no amplitude may be quoted.  This is the
        exact n >= 6400 failure mode above.
    ``bore_valid`` / ``flatness`` (value still reported, flagged)
        ``flatness = max|h - mean(h)| / jump`` over the window and
        ``bore_valid = flatness <= PLATEAU_FLAT_FRACTION``.  On coarse grids the
        numerical plateau is a *ramp* (measured 11 % of the jump at n = 800,
        1.4 % at n = 1600, 0.2 % at n = 6400), so the window mean stays a good
        amplitude estimate while the "flat plateau" premise does not hold;
        callers that grade must consult ``bore_valid``.

    The bore **position** is reported twice:

    ``bore_x``
        Sub-cell crossing of the level ``(h2 + h1) / 2`` (both measured from the
        profile), linearly interpolated between cell centres.
    ``bore_x_cell``
        Legacy cell centre of the max-gradient cell.  Kept as the documented
        negative control: quantised to ``dx``, non-monotone under refinement
        (0.73 % -> 0.068 % -> 0.399 % -> 0.233 % for n = 200 ... 1600).

    Returns ``bore_x`` / ``bore_x_cell``, ``bore_level``, ``bore_method``,
    ``h2``, ``h1``, ``jump``, ``flatness``, ``bore_valid``, ``n_plateau``,
    ``plateau_width`` (metres -- the grid-independent way to read the width),
    ``thresh`` and ``max_grad``.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    nan = float("nan")
    out = {"bore_x": nan, "bore_x_cell": nan, "bore_level": nan,
           "bore_method": "undefined", "h2": nan, "h1": nan,
           "n_plateau": 0, "plateau_width": nan, "jump": nan,
           "flatness": nan, "bore_valid": False,
           "thresh": nan, "max_grad": nan}
    if h.size < 4:
        return out
    dx = x[1] - x[0]
    grad = np.abs(np.diff(h)) / dx
    j = int(np.argmax(grad))
    max_grad = float(grad[j])
    if not np.isfinite(max_grad) or max_grad <= 0.0:
        return out
    # The cut is tried at ``thresh_frac`` and, if that cut cannot find a fan
    # (see the F9 note above: the fan slope is a physical constant while
    # ``max|dh/dx|`` grows like 1/dx), tightened by one decade at a time.  The
    # retry rule is declared a priori -- it is a resolution loop, not a fit to
    # the answer -- and every attempt is bounded by ``PLATEAU_THRESH_MAX_DECADES``
    # so a profile without any fan still returns "plateau_unbounded".
    attempts = []
    r = l = -1
    thresh = float("nan")
    for decade in range(PLATEAU_THRESH_MAX_DECADES + 1):
        frac = thresh_frac * (0.1 ** decade)
        thresh = frac * max_grad
        # walk left out of the bore smearing -> plateau right edge
        r = j - 1
        while r >= 0 and grad[r] > thresh:
            r -= 1
        # walk left while flat -> plateau left edge (where the fan steepens)
        l = r
        while l >= 0 and grad[l] < thresh:
            l -= 1
        l += 1
        bounded = bool(r > 0 and l > 0 and r >= l)
        attempts.append({"decade": decade, "thresh_frac": frac,
                         "thresh": float(thresh), "bounded": bounded,
                         "n_plateau": int(r - l + 1)})
        if bounded:
            break
    out["thresh"] = float(thresh)
    out["thresh_frac_used"] = attempts[-1]["thresh_frac"]
    out["attempts"] = attempts
    out["max_grad"] = max_grad
    out["bore_x_cell"] = float(x[j])
    if not bounded:
        # no fan detected upstream at any declared cut: the plateau is not
        # delimited, so nothing may be quoted (F9).
        out["bore_method"] = "plateau_unbounded"
        return out
    h2 = float(h[l:r + 1].mean())
    # downstream: walk right past the bore smearing, then the flat h1 region
    k2 = j + 1
    while k2 < len(grad) and grad[k2] > thresh:
        k2 += 1
    k3 = k2
    while k3 < len(grad) and grad[k3] < thresh:
        k3 += 1
    h1 = float(h[k2:k3 + 1].mean()) if k3 > k2 else float(h[k2])
    jump = abs(h2 - h1)
    if not np.isfinite(jump) or jump <= 0.0:
        out["bore_method"] = "plateau_unbounded"
        return out
    flatness = float(np.abs(h[l:r + 1] - h2).max()) / jump
    # sub-cell bore position: interpolate the crossing of the mid level between
    # the two *measured* states.  Anchored to the max-gradient cell (so a
    # smeared, slightly non-monotone jump still lands on the right crossing) but
    # not quantised to a cell centre.
    level = 0.5 * (h2 + h1)
    x_bore, method = float(x[j]), "cell_centre"
    i = r
    stop = min(k2 + 1, h.size - 1)
    if h2 > h1:
        while i < stop and not (h[i] >= level > h[i + 1]):
            i += 1
        if i < stop:
            dz = h[i] - h[i + 1]
            x_bore = float(x[i] + dx * (h[i] - level) / dz)
            method = "mid_level_crossing"
    out.update({"bore_x": x_bore, "bore_method": method,
                "bore_level": float(level), "h2": h2, "h1": h1,
                "jump": float(jump), "flatness": flatness,
                "bore_valid": bool(flatness <= PLATEAU_FLAT_FRACTION),
                "n_plateau": int(r - l + 1),
                "plateau_width": float((r - l + 1) * dx)})
    return out


#: histogram resolution of the oracle-free depth-mode detector
#: (:func:`plateau_from_depth_modes`): number of bins over ``[min h, max h]``.
#: 80 bins put one bin at ``0.0063 m`` for the wet-bed profile (``h`` in
#: ``[0.5, 1.0]`` m), i.e. ``2.8 %`` of the level step ``h2 - h1 = 0.227 m``;
#: the refinement below removes that quantisation from the quoted value.
PLATEAU_MODE_BINS = 80

#: A depth mode is quoted only if its bin holds at least this fraction of the
#: cells -- a *non-vacuity* floor, declared before any run.  Measured on the
#: Stoker profile at ``t* = 2`` for n = 200 ... 3200: the plateau spike holds
#: 26 % ... 30 % of the cells and the two outer states 18 % ... 41 %, while the
#: bins inside the smooth rarefaction fan hold 0.1 % ... 0.4 % -- the floor
#: separates a constant state from a ramp with roughly two orders of margin.
PLATEAU_MODE_MIN_FRACTION = 0.02

#: Bins merged on each side of a modal bin when refining it to the median of
#: its cells.  One bin each side: the plateau spike can straddle two bins on a
#: coarse grid (n = 200), and the median over the union is then insensitive to
#: which of the two bins happens to be the larger one.
PLATEAU_MODE_BAND = 1

#: A cell is counted as plateau if ``|h - h2| <= PLATEAU_MODE_TOL_FRACTION *
#: (h2 - h1)``.  The numerical plateau is a *ramp* on coarse grids (flatness
#: 11 % of the jump at n = 800, 1.4 % at n = 1600), so the quoted
#: ``plateau_width`` shrinks with resolution; it is a diagnostic of the run, not
#: a graded quantity.
PLATEAU_MODE_TOL_FRACTION = 5e-2


def _depth_modes(h, bins: int = PLATEAU_MODE_BINS,
                 min_fraction: float = PLATEAU_MODE_MIN_FRACTION,
                 band: int = PLATEAU_MODE_BAND) -> list:
    """Modal depths of a depth profile -- **no analytic input**.

    A depth profile of a dam break holds one constant state per region, so its
    *distribution* is multi-modal: one mode per level.  Each mode is detected
    from the histogram alone (bin holding at least ``min_fraction`` of the cells
    and strictly larger than its right neighbour, so a spike straddling two bins
    yields **one** mode) and refined to the median of the cells in the modal bin
    ``+/- band`` bins, which removes the bin quantisation from the value.

    Returns a list of ``{value, centre, count, bin}`` sorted by ``value``.
    """
    h = np.asarray(h, dtype=float)
    h = h[np.isfinite(h)]
    if h.size < 4:
        return []
    lo, hi = float(h.min()), float(h.max())
    if not hi > lo:
        return []
    counts, edges = np.histogram(h, bins=bins, range=(lo, hi))
    floor = max(float(min_fraction) * h.size, 1.0)
    modes = []
    for i in range(bins):
        count = int(counts[i])
        if count < floor:
            continue
        left = int(counts[i - 1]) if i > 0 else -1
        right = int(counts[i + 1]) if i < bins - 1 else -1
        # a mode must beat its left neighbour (ties go to the lower depth, so
        # only one of two equally tall adjacent bins is reported) and be
        # strictly larger than its right neighbour
        if count < left or count <= right:
            continue
        k0 = max(i - band, 0)
        k1 = min(i + band + 1, bins)
        cell = (h >= edges[k0]) & (h <= edges[k1])
        if not np.any(cell):
            continue
        modes.append({"value": float(np.median(h[cell])),
                      "centre": float(0.5 * (edges[i] + edges[i + 1])),
                      "count": count,
                      "bin": int(i)})
    modes.sort(key=lambda mode: mode["value"])
    return modes


def plateau_from_depth_modes(x, h) -> dict:
    """Wet-bed plateau depth, still-water depth and bore position from the
    **depth distribution only** (no analytic position, level or amplitude).

    This closes the residual oracle dependency recorded in
    ``docs/case-fvm-dambreak-stoker/AUDIT.md`` §6 (F4 / §7.4): the published
    ``h2`` / ``step_ratio`` are read inside a window placed from the closed form
    ``x_fan_end``/``x_shock`` (near-tautological), while
    :func:`data_driven_plateau` locates the plateau with a *gradient* walk whose
    cut has to be retried per grid (F9).  Here neither is used:

    * every level of the profile is a **mode of its depth distribution**
      (:func:`_depth_modes`): upstream state, plateau, still water;
    * the plateau depth ``h2`` is the *interior* mode (second largest, the
      largest being the undisturbed upstream state), the still-water depth
      ``h1`` the smallest;
    * the bore is the sub-cell crossing of the *measured* mid level
      ``(h2 + h1) / 2``, searched rightwards from the right edge of the cells
      that sit on ``h2`` within ``PLATEAU_MODE_TOL_FRACTION * (h2 - h1)``.

    Structural assumption (declared, not fitted): the profile contains three
    constant states -- upstream, plateau, downstream -- which is the topology of
    the wet-bed dam break for ``t* > 0``.  If fewer than three modes are found
    the reading is *withheld* (``method = "no_plateau"``, values NaN) instead of
    quoted: a profile without a constant intermediate state has no plateau, and
    a two-level profile has no interior mode.

    Returns ``h2``, ``h1``, ``h0``, ``jump``, ``step_ratio``, ``bore_x``,
    ``bore_level``, ``method``, ``plateau_cells``, ``plateau_width``,
    ``plateau_left``, ``plateau_right``, ``modes``, ``n_modes`` and
    ``bins``/``bin_width`` (so a reader can see the quantisation scale).
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    nan = float("nan")
    out = {"h2": nan, "h1": nan, "h0": nan, "jump": nan, "step_ratio": nan,
           "bore_x": nan, "bore_level": nan, "method": "undefined",
           "plateau_cells": 0, "plateau_width": nan,
           "plateau_left": nan, "plateau_right": nan,
           "modes": [], "n_modes": 0,
           "bins": int(PLATEAU_MODE_BINS), "bin_width": nan,
           "tolerance": PLATEAU_MODE_TOL_FRACTION, "valid": False}
    if x.size != h.size or h.size < 4:
        return out
    if not np.all(np.isfinite(h)):
        # a partially non-finite profile is not a state: withhold, never read
        # a subset of the cells silently
        return out
    dx = float(x[1] - x[0])
    hi, lo = float(h.max()), float(h.min())
    if hi > lo:
        out["bin_width"] = (hi - lo) / float(PLATEAU_MODE_BINS)
    modes = _depth_modes(h)
    out["modes"] = modes
    out["n_modes"] = len(modes)
    if len(modes) < 3:
        # no interior constant state -> no plateau to read (never quoted)
        out["method"] = "no_plateau"
        return out
    h1 = modes[0]["value"]
    h0 = modes[-1]["value"]
    h2 = modes[-2]["value"]
    out.update({"h1": h1, "h0": h0, "h2": h2})
    jump = h2 - h1
    if not (jump > 0.0 and h0 > h2):
        out["method"] = "no_plateau"
        return out
    out["jump"] = jump
    out["step_ratio"] = h2 / h1 if h1 > 0.0 else nan
    level = 0.5 * (h2 + h1)
    out["bore_level"] = level
    cells = np.where(np.abs(h - h2) <= PLATEAU_MODE_TOL_FRACTION * jump)[0]
    method = "depth_mode"
    if cells.size:
        anchor = int(cells[-1])
        out["plateau_cells"] = int(cells.size)
        out["plateau_left"] = float(x[int(cells[0])])
        out["plateau_right"] = float(x[anchor])
        out["plateau_width"] = float(cells.size * dx)
    else:
        # the plateau structure is below the tolerance: keep the reading but
        # record that the search started from the max-gradient cell instead
        anchor = int(np.argmax(np.abs(np.diff(h))))
        method = "depth_mode_gradient_anchor"
    stop = h.size - 1
    i = anchor
    while i < stop and not (h[i] >= level > h[i + 1]):
        i += 1
    if i < stop and h[i] > h[i + 1]:
        out["bore_x"] = float(x[i] + dx * (h[i] - level) / (h[i] - h[i + 1]))
        out["method"] = method
        out["valid"] = True
    else:
        # the profile never crosses the measured mid level to the right of the
        # plateau -> the position is not measured, and nothing is quoted
        out["method"] = method + "_unresolved"
    return out


def depth_mode_study(resolutions=(200, 400, 800, 1600, 3200), cfl: float = 0.4,
                     order: int = 2, t_star: float | None = None) -> dict:
    """Grid study of the oracle-free (:func:`plateau_from_depth_modes`) reading.

    For every resolution the Stoker wet-bed case is integrated to ``t*``
    (default: the case's own ``t* = 2``) and the *depth-distribution* reading of
    ``h2`` / ``h1`` / ``step_ratio`` / bore position is compared with the closed
    form.  The same profile is also read by the two existing mechanisms, so the
    three readings can be compared row by row:

    ``oracle_window``
        the :func:`_wet_metrics` window, placed from the analytic
        ``x_fan_end``/``x_shock`` -- near-tautological for ``h2`` (F4).
    ``gradient_walk``
        :func:`data_driven_plateau`, whose threshold cut is retried per grid
        (F9) and which withholds the reading for ``n >= 6400``.
    ``depth_mode``
        this module: modes of the depth distribution, no analytic input.

    Returns ``{"case", "t_star", "tolerance", "rows",
    "max_error_by_mechanism"}``.
    """
    t_star_end = STOKER_WET.t_star_end if t_star is None else float(t_star)
    t_end = t_star_to_time(t_star_end, STOKER_WET.h0, STOKER_WET.gravity)
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    rows = []
    for n in resolutions:
        solver = ShallowWater1D(n, STOKER_WET.x_min, STOKER_WET.x_max,
                                gravity=STOKER_WET.gravity, cfl=cfl, order=order)
        solver.set_state(np.where(solver.x < STOKER_WET.x_dam,
                                  STOKER_WET.h0, STOKER_WET.h1))
        solver.advance(t_end)
        t = solver.time
        x_shock = STOKER_WET.x_dam + sol["shock_speed"] * t
        x_fan_end = STOKER_WET.x_dam + (sol["u2"] - sol["c2"]) * t
        mode = plateau_from_depth_modes(solver.x, solver.h)
        walk = data_driven_plateau(solver.x, solver.h)
        # oracle window (F4): middle half of the exact constant state
        lo_w = x_fan_end + 0.25 * (x_shock - x_fan_end)
        hi_w = x_fan_end + 0.75 * (x_shock - x_fan_end)
        win = (solver.x >= lo_w) & (solver.x <= hi_w)
        h2_window = float(solver.h[win].mean()) if np.any(win) else float("nan")
        rows.append({
            "n": int(n),
            "dx": float(solver.dx),
            "depth_mode": {
                "h2": mode["h2"], "h1": mode["h1"], "jump": mode["jump"],
                "step_ratio": mode["step_ratio"],
                "h2_error": _relative(mode["h2"], sol["h2"]),
                "h1_error": _relative(mode["h1"], sol["h1"]),
                "step_ratio_error": _relative(mode["step_ratio"],
                                              sol["step_ratio"]),
                "bore_x": mode["bore_x"],
                "bore_x_error": _relative(mode["bore_x"], x_shock),
                "bore_x_cells": (None if not np.isfinite(mode["bore_x"])
                                 else abs(mode["bore_x"] - x_shock) / solver.dx),
                "plateau_cells": mode["plateau_cells"],
                "plateau_width": mode["plateau_width"],
                "method": mode["method"],
                "valid": mode["valid"],
                "n_modes": mode["n_modes"],
                "modes": mode["modes"],
            },
            "oracle_window": {
                "h2": h2_window,
                "h2_error": _relative(h2_window, sol["h2"]),
                "cells": int(np.count_nonzero(win)),
            },
            "gradient_walk": {
                "h2": walk["h2"], "h1": walk["h1"],
                "h2_error": _relative(walk["h2"], sol["h2"]),
                "bore_x": walk["bore_x"],
                "bore_x_error": _relative(walk["bore_x"], x_shock),
                "method": walk["bore_method"],
                "flatness": walk["flatness"],
                "bore_valid": walk["bore_valid"],
            },
            "exact": {"h2": sol["h2"], "h1": sol["h1"],
                      "step_ratio": sol["step_ratio"], "x_shock": x_shock},
        })
    keys = ("h2_error", "h1_error", "step_ratio_error", "bore_x_error")
    worst = {}
    for mechanism in ("depth_mode", "oracle_window", "gradient_walk"):
        worst[mechanism] = {}
        for key in keys:
            values = [row[mechanism].get(key) for row in rows
                      if row[mechanism].get(key) is not None]
            values = [v for v in values if v is not None and np.isfinite(v)]
            worst[mechanism][key] = max(values) if values else None
    graded = [row for row in rows if row["n"] >= ACCEPTED_MIN_N]
    worst_graded = {}
    for key in keys:
        values = [row["depth_mode"].get(key) for row in graded]
        values = [v for v in values if v is not None and np.isfinite(v)]
        worst_graded[key] = max(values) if values else None
    return {"case": STOKER_WET.summary(), "t_star": t_star_end,
            "tolerance": TOLERANCE, "cfl": cfl, "order": order,
            "bins": int(PLATEAU_MODE_BINS),
            "mode_min_fraction": PLATEAU_MODE_MIN_FRACTION,
            "mode_tolerance": PLATEAU_MODE_TOL_FRACTION,
            "accepted_min_n": ACCEPTED_MIN_N,
            "max_error_by_mechanism": worst,
            "max_error_depth_mode_accepted": worst_graded,
            "rows": rows}


def fan_front_estimate(x, h, t: float, gravity: float, h0: float,
                       window=FAN_WINDOW) -> dict:
    """Dry-bed front position from the **exact rarefaction shape**.

    Behind the tip the Ritter (1892) solution is a straight line in
    ``c = sqrt(g h)``::

        c(x, t) = (2 c0 - x / t) / 3,        c0 = sqrt(g h0),

    so every cell inside the fan individually carries the front,

    .. math:: x_f = 2 c0 t = x_i + 3 t \sqrt{g h_i},

    *exactly* -- no threshold, no extrapolation length.  Averaging that identity
    over a window of well-resolved fan cells is therefore a front estimator whose
    error measures the **shape** error of the numerical fan, not the position of
    a user-chosen depth contour.  This is the property that a ``h > 0`` wetted
    edge does *not* have: the tip is a zero-depth cusp with ``h ~ (x_f - x)^2``,
    so "outermost wetted cell" is a depth-floor detector whose reading depends on
    the numerical floor rather than on ``dx`` (measured non-convergence: 1.56 % ->
    2.66 % -> 2.89 % -> 2.46 % for n = 400 ... 3200).

    Parameters
    ----------
    x, h
        Cell centres and depths of the numerical solution.
    t, gravity, h0
        Time of the state, gravity, undisturbed upstream depth.
    window
        ``(lo, hi)`` in units of ``h / h0``; cells outside are ignored.

    Returns
    -------
    dict with ``front`` (mean estimate, m), ``cells`` (window size),
    ``spread`` (std of the per-cell estimates, m -- a resolution diagnostic),
    ``slope`` / ``slope_error`` (fitted ``dc/dx`` vs the exact ``-1/(3t)``).
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    nan = float("nan")
    out = {"front": nan, "cells": 0, "spread": nan, "slope": nan,
           "slope_error": nan, "window": [float(window[0]), float(window[1])]}
    if t <= 0.0 or h.size < 3:
        return out
    c = np.sqrt(np.maximum(gravity * h, 0.0))
    frac = h / h0
    sel = (frac >= window[0]) & (frac <= window[1])
    if int(np.count_nonzero(sel)) < 3:
        return out
    xs = x[sel]
    per_cell = xs + 3.0 * t * c[sel]
    out["front"] = float(np.mean(per_cell))
    out["cells"] = int(per_cell.size)
    out["spread"] = float(np.std(per_cell))
    slope = float(np.polyfit(xs, c[sel], 1)[0])
    out["slope"] = slope
    out["slope_error"] = abs(slope + 1.0 / (3.0 * t)) / (1.0 / (3.0 * t))
    return out


def front_sample(case: DamBreakCase, solver: ShallowWater1D) -> dict:
    """Wave position of the current state vs the exact position at that time.

    The sample carries **two** definitions on purpose:

    ``position`` / ``error``
        Legacy definition (dry: outermost wet cell, wet: mid-level crossing).
        Kept because downstream artefacts quote it; for the dry bed it is a
        depth-floor detector that does not converge (see
        :func:`fan_front_estimate`), so it must not be graded.
    ``headline`` / ``headline_error``
        The graded definition: the shape-consistent fan estimator for the dry
        bed (consistent with the headline metric of :func:`_dry_metrics`) and
        the mid-level crossing for the wet bed.  Grading a *series* with the
        legacy estimator was a defect: at ``t* = 0.005`` it reads a 212 %
        error on the dry bed because the numerical wetting front runs several
        cells ahead of the exact zero-depth tip (see
        ``docs/case-fvm-dambreak-stoker/AUDIT.md``).

    ``resolved`` is the a-priori resolution flag: the front must have travelled
    ``SERIES_MIN_CELLS`` cells *and*, on the dry bed, the fan window must hold
    ``SERIES_MIN_FAN_CELLS`` cells.  The second condition is what makes the
    series error a discretisation error; without it a 3-cell fan reads a
    half-cell quantisation that does not converge with ``dx``
    (``docs/case-fvm-dambreak-stoker/ANALYSIS-series-fan-cell-floor.md``).
    """
    t = solver.time
    exact = case.exact_front(t)
    dx = float(solver.dx)
    travelled = (exact - case.x_dam) / dx
    resolved = bool(travelled >= SERIES_MIN_CELLS)
    fan_cells_resolved = True
    if case.kind == "dry":
        pos = _wetted_edge(solver.x, solver.h)
        fan = fan_front_estimate(solver.x, solver.h, t, case.gravity, case.h0)
        headline = fan["front"]
        cells = fan["cells"]
        spread = fan["spread"]
        # the estimator needs a resolved fan, not just a front that moved:
        # below SERIES_MIN_FAN_CELLS its reading is a half-cell quantisation
        # (resolution invariant), so such a sample is not admissible.
        fan_cells_resolved = bool(cells is not None
                                  and cells == cells
                                  and int(cells) >= SERIES_MIN_FAN_CELLS)
    else:
        sol = stoker_constant_state(case.h0, case.h1, case.gravity)
        pos = _level_crossing(solver.x, solver.h, 0.5 * (case.h1 + sol["h2"]))
        headline = pos
        cells = None
        spread = None
    return {
        "t": t,
        "t_star": t * math.sqrt(case.gravity / case.h0),
        "position": pos,
        "exact": exact,
        "error": _relative(pos, exact),
        "headline": headline,
        "headline_error": (_relative(headline, exact)
                           if headline == headline else float("nan")),
        "headline_cells": cells,
        "headline_spread": spread,
        "resolved": bool(resolved and fan_cells_resolved),
        "travelled_cells": travelled,
        "travelled_resolved": resolved,
        "fan_cells_resolved": fan_cells_resolved,
        "unresolved_reason": (None if (resolved and fan_cells_resolved) else
                              ("front_travel" if not resolved
                               else "fan_cells")),
    }


def series_metrics(case: DamBreakCase, samples,) -> dict:
    """Grade the *whole* ``x_f(t)`` series, not just its final state.

    The acceptance criterion is a statement about the wave position *as a
    function of time* (``x_f(t)`` vs analytic); grading only ``t* = 2`` would
    leave the early transient unverified.  Graded samples are those with
    ``resolved`` true (see :data:`SERIES_MIN_CELLS`); the rest are reported
    under ``unresolved_*`` so a regression in the first few steps is still
    visible instead of silently dropped.
    """
    graded = [s for s in samples
              if s["resolved"] and s["headline_error"] == s["headline_error"]]
    reasons = {}
    for s in samples:
        if s not in graded:
            key = s.get("unresolved_reason") or "undefined_estimator"
            reasons[key] = reasons.get(key, 0) + 1
    # every sample that is not graded stays visible -- including the ones whose
    # headline estimator is undefined (too few fan cells): dropping them
    # silently would hide the first steps of the transient.
    unresolved = [s for s in samples if s not in graded]
    named = [s for s in unresolved if s["headline_error"] == s["headline_error"]]
    best = max(graded, key=lambda s: s["headline_error"]) if graded else None
    worst_unres = max(named, key=lambda s: s["headline_error"]) if named else None
    legacy = [s for s in samples if s["error"] == s["error"]]
    worst_legacy = max(legacy, key=lambda s: s["error"]) if legacy else None
    return {
        "min_cells": SERIES_MIN_CELLS,
        "min_fan_cells": SERIES_MIN_FAN_CELLS,
        "samples": len(samples),
        "samples_graded": len(graded),
        "max_error": None if best is None else float(best["headline_error"]),
        "max_error_t_star": None if best is None else float(best["t_star"]),
        "unresolved_samples": len(unresolved),
        "unresolved_reasons": reasons,
        "unresolved_undefined": len(unresolved) - len(named),
        "unresolved_max_error": (None if worst_unres is None
                                 else float(worst_unres["headline_error"])),
        "unresolved_max_error_t_star": (None if worst_unres is None
                                        else float(worst_unres["t_star"])),
        "legacy_max_error": (None if worst_legacy is None
                             else float(worst_legacy["error"])),
        "legacy_max_error_t_star": (None if worst_legacy is None
                                    else float(worst_legacy["t_star"])),
        "legacy_definition": ("wetted_edge (dry) / mid_level_crossing (wet)"),
        "headline_definition": ("fan_front_estimate (dry) / mid_level_crossing (wet)"),
        "tolerance": TOLERANCE,
        "pass": (best is not None and float(best["headline_error"]) <= TOLERANCE),
    }


def _dry_metrics(case: DamBreakCase, solver: ShallowWater1D, volume0: float) -> dict:
    t = solver.time
    h_ex, _ = case.exact(solver.x, t)
    x_f = case.exact_front(t)
    front_wetted = _wetted_edge(solver.x, solver.h)
    level = FRONT_LEVEL_FRACTION * case.h0
    x_level = _level_crossing(solver.x, solver.h, level)
    # exact fan: h = (x_f - x)^2 / (9 g t^2)  =>  x_f = x + 3 t sqrt(g h)
    x_level_scaled = x_level + 3.0 * t * math.sqrt(case.gravity * level)
    fan = fan_front_estimate(solver.x, solver.h, t, case.gravity, case.h0)
    tv_num = float(np.abs(np.diff(solver.h)).sum())
    ahead = solver.x > x_f
    return {
        "front_exact": x_f,
        # headline metric: shape-consistent fan estimator (see fan_front_estimate)
        "front_fan": fan["front"],
        "front_fan_error": _relative(fan["front"], x_f),
        "front_fan_cells": fan["cells"],
        "front_fan_spread": fan["spread"],
        "front_fan_slope": fan["slope"],
        "front_fan_slope_error": fan["slope_error"],
        "front_fan_window": fan["window"],
        # legacy/diagnostic definitions -- kept for continuity, NOT headline
        "front_wetted": front_wetted,
        "front_wetted_error": _relative(front_wetted, x_f),
        # signed shortfall of the wetted edge (negative => spurious wetting ahead)
        "wetted_shortfall": x_f - front_wetted,
        "precursor_length": float(max(0.0, front_wetted - x_f)),
        "precursor_depth": float(solver.h[ahead].max() / case.h0) if np.any(ahead) else 0.0,
        "front_level": x_level,
        "front_level_scaled": x_level_scaled,
        "front_level_scaled_error": _relative(x_level_scaled, x_f),
        "front_level_fraction": FRONT_LEVEL_FRACTION,
        "depth_l1": float(np.abs(solver.h - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "depth_linf": float(np.abs(solver.h - h_ex).max() / case.h0),
        "tv_numeric": tv_num,
        "tv_exact": float(case.h0),
        "tv_ratio": tv_num / case.h0,
        "local_extrema": _local_extrema(solver.h, EXTREMUM_FRACTION * case.h0),
        "mass_drift": _relative(solver.total_volume(), volume0),
        "h_max_over_h0": float(solver.h.max() / case.h0),
        "h_min": float(solver.h.min()),
    }


def _data_driven_crosscheck(case: DamBreakCase, solver: ShallowWater1D) -> dict:
    """Independent (data-driven) wet-bed plateau/bore verification (F4).

    The oracle-guided ``h2``/``step_ratio``/``shock_position`` in
    :func:`_wet_metrics` use the analytic ``x_fan_end``/``x_shock``/``h2`` to
    place the measurement window, so they are near-tautologies (insensitive to
    a misplaced bore).  This cross-check runs :func:`data_driven_plateau` on the
    numerical profile alone and reports how far the *data-driven* bore, plateau
    amplitude and level step sit from the closed form -- a genuine
    position-and-amplitude check.  All relative errors are ``< 5 %`` on the
    accepted grid (n >= 800).

    ``bore_x`` is the **sub-cell** crossing of the measured mid level
    (:func:`data_driven_plateau`); ``bore_x_cell`` keeps the legacy
    max-gradient **cell centre**, whose error is quantised to ``dx`` and
    non-monotone under refinement (0.051 -> 0.602 -> 0.705 cells for
    n = 400 ... 1600).  It is reported as the negative control, not graded.
    """
    dd = data_driven_plateau(solver.x, solver.h)
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    t = solver.time
    x_shock = case.x_dam + sol["shock_speed"] * t
    return {
        "bore_x": dd["bore_x"],
        "bore_x_error": _relative(dd["bore_x"], x_shock),
        "bore_x_cell": dd["bore_x_cell"],
        "bore_x_cell_error": _relative(dd["bore_x_cell"], x_shock),
        "bore_level": dd["bore_level"],
        "bore_method": dd["bore_method"],
        "h2": dd["h2"],
        "h2_error": _relative(dd["h2"], sol["h2"]),
        "h1": dd["h1"],
        "h1_error": _relative(dd["h1"], sol["h1"]),
        "step_ratio": dd["h2"] / dd["h1"] if dd["h1"] else float("nan"),
        "step_ratio_error": _relative(
            dd["h2"] / dd["h1"], sol["step_ratio"]) if dd["h1"] else float("nan"),
        "n_plateau": dd["n_plateau"],
        "plateau_width": dd["plateau_width"],
        "flatness": dd["flatness"],
        "bore_valid": dd["bore_valid"],
        "thresh_frac_used": dd.get("thresh_frac_used"),
        "bore_attempts": dd.get("attempts"),
        "thresh_frac": PLATEAU_THRESH_FRACTION,
    }


def _oracle_free_wet_readings(case: DamBreakCase, solver: ShallowWater1D) -> dict:
    """Wet-bed readings whose *measurement* uses no analytic input.

    The graded wet-bed quantities (:func:`_wet_metrics`) are read inside a
    window placed from the closed-form ``x_fan_end``/``x_shock`` and at the
    analytic mid level ``(h1 + h2) / 2``; ``AUDIT.md`` §6 recorded the residual
    oracle dependency (F4) and §9.5 left "promote the distribution-only reading
    to a graded criterion?" to the next independent review.  This function is
    that reading, in the form the verdict consumes:

    * ``h1`` / ``h2`` / ``step_ratio`` / ``bore_x`` come from
      :func:`plateau_from_depth_modes` (histogram modes + sub-cell crossing of the
      *measured* mid level) -- no analytic position, level or amplitude;
    * the analytic solution enters only as the **reference** the reading is
      compared with (``shock_exact`` / ``h2_exact`` / ``step_ratio_exact``),
      never as a threshold or a window;
    * ``no_plateau`` / ``unresolved`` readings are carried as ``valid = False``
      and are never silently quoted as a number.

    Returns the detector dict plus ``bore_position_error`` /
    ``step_ratio_error`` (relative, vs the Stoker solution) and
    ``agreement_bore_vs_oracle_window`` (relative distance between this
    oracle-free position and the oracle-window level-crossing reading of
    :func:`_wet_metrics`, reported by :func:`_wet_metrics` as ``shock_position``).
    """
    nan = float("nan")
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    x_shock = case.x_dam + sol["shock_speed"] * solver.time
    out = dict(plateau_from_depth_modes(solver.x, solver.h))
    out["shock_exact"] = x_shock
    out["h2_exact"] = sol["h2"]
    out["step_ratio_exact"] = sol["step_ratio"]
    bore_x = out.get("bore_x", nan)
    step = out.get("step_ratio", nan)
    out["bore_position_error"] = (_relative(bore_x, x_shock)
                                  if np.isfinite(bore_x) else nan)
    out["step_ratio_error"] = (_relative(step, sol["step_ratio"])
                               if np.isfinite(step) else nan)
    oracle = _level_crossing(solver.x, solver.h, 0.5 * (case.h1 + sol["h2"]))
    out["agreement_bore_vs_oracle_window"] = (
        _relative(bore_x, oracle) if np.isfinite(bore_x) else nan)
    return out


def _wet_metrics(case: DamBreakCase, solver: ShallowWater1D, volume0: float) -> dict:
    t = solver.time
    h_ex, _ = case.exact(solver.x, t)
    sol = stoker_constant_state(case.h0, case.h1, case.gravity)
    x_shock = case.x_dam + sol["shock_speed"] * t
    x_fan_end = case.x_dam + (sol["u2"] - sol["c2"]) * t
    level = 0.5 * (sol["h1"] + sol["h2"])
    x_meas = _level_crossing(solver.x, solver.h, level)
    # plateau: middle half of the exact constant state, shock smearing excluded
    lo = x_fan_end + 0.25 * (x_shock - x_fan_end)
    hi = x_fan_end + 0.75 * (x_shock - x_fan_end)
    plateau = (solver.x >= lo) & (solver.x <= hi)
    h2_meas = float(solver.h[plateau].mean()) if np.any(plateau) else float("nan")
    still = solver.x >= x_shock + 0.5 * (case.x_max - x_shock)
    h1_meas = float(solver.h[still].mean()) if np.any(still) else float("nan")
    tv_num = float(np.abs(np.diff(solver.h)).sum())
    # window coverage: how much of the exact level step the oracle window
    # actually sees.  Reported beside the reading so an under-resolved strong
    # bore cannot masquerade as a 100 % amplitude error (see
    # PLATEAU_MIN_JUMP_FRACTION).
    step_exact = sol["h2"] - sol["h1"]
    if np.any(plateau):
        jump = float((solver.h[plateau] - sol["h1"]).max())
        coverage = jump / step_exact
    else:
        jump = float("nan")
        coverage = float("nan")
    return {
        "shock_exact": x_shock,
        "shock_position": x_meas,
        "shock_level": level,
        "shock_error": _relative(x_meas, x_shock),
        "fan_end_exact": x_fan_end,
        "h2_exact": sol["h2"],
        "h2_measured": h2_meas,
        "h2_error": _relative(h2_meas, sol["h2"]),
        "h1_measured": h1_meas,
        "plateau_window_cells": int(np.count_nonzero(plateau)),
        "plateau_jump_fraction": coverage,
        "step_insufficient_evidence": bool(not (coverage >= PLATEAU_MIN_JUMP_FRACTION)),
        "step_ratio_exact": sol["step_ratio"],
        "step_ratio_measured": h2_meas / h1_meas,
        "step_ratio_error": _relative(h2_meas / h1_meas, sol["step_ratio"]),
        "shock_speed_exact": sol["shock_speed"],
        "u2_exact": sol["u2"],
        "c2_exact": sol["c2"],
        # --- data-driven (F4) cross-check: no analytic position/level used ---
        "dd": _data_driven_crosscheck(case, solver),
        "modes": _oracle_free_wet_readings(case, solver),
        "depth_l1": float(np.abs(solver.h - h_ex).sum() / max(h_ex.sum(), 1e-300)),
        "depth_linf": float(np.abs(solver.h - h_ex).max() / case.h0),
        "tv_numeric": tv_num,
        "tv_exact": float(case.h0 - case.h1),
        "tv_ratio": tv_num / (case.h0 - case.h1),
        "local_extrema": _local_extrema(solver.h, EXTREMUM_FRACTION * case.h0),
        "mass_drift": _relative(solver.total_volume(), volume0),
        "h_max_over_h0": float(solver.h.max() / case.h0),
        "h_min": float(solver.h.min()),
    }


def _series_verdict_entry(metrics: dict) -> dict:
    """Verdict entry for the anchor's own ``x_f(t)`` acceptance quantity.

    ``pass`` requires *both*

    * the worst graded sample error to be ``<= TOLERANCE``, and
    * at least ``SERIES_MIN_GRADED_FRACTION`` of the scheduled samples to be
      gradable at all -- an unresolvable series is *insufficient evidence* and
      fails rather than passes.

    ``value`` is ``None`` when the run produced no gradable sample; the
    supporting counts are kept in the entry so a reader can audit the gate.
    """
    series = metrics.get("front_series") or {}
    total = int(series.get("samples") or 0)
    graded = int(series.get("samples_graded") or 0)
    required = int(math.ceil(SERIES_MIN_GRADED_FRACTION * total)) if total else 0
    error = series.get("max_error")
    error = None if error is None else float(error)
    sufficient = total > 0 and graded >= required
    passed = bool(error is not None and sufficient and error <= TOLERANCE)
    return {
        "value": error,
        "limit": TOLERANCE,
        "pass": passed,
        "samples": total,
        "samples_graded": graded,
        "graded_required": required,
        "unresolved_max_error": series.get("unresolved_max_error"),
        "note": ("anchor acceptance x_f(t) series (graded max error); fails when "
                 "undefined or when fewer than SERIES_MIN_GRADED_FRACTION of the "
                 "samples are resolvable"),
    }


def _oracle_free_verdict_entries(metrics: dict) -> dict:
    """Verdict entries grading the **oracle-free** wet-bed reading.

    ``pass`` requires the reading to be *measured* (``valid``) **and** inside the
    predeclared ``TOLERANCE``.  A ``no_plateau`` / ``unresolved`` reading is
    *insufficient evidence* and fails rather than passes -- a profile whose depth
    distribution has lost its interior mode has lost the constant state the
    ledger quantity is about, and an amplitude read off a window mean must not be
    able to hide that (same discipline as :func:`_series_verdict_entry`).
    """
    modes = metrics.get("modes") or {}
    valid = bool(modes.get("valid"))
    notes = ("measured from the depth distribution only: histogram modes + the "
             "sub-cell crossing of the *measured* mid level; the analytic "
             "solution is the reference, never a threshold or a window")

    def entry(value):
        measured = value is not None and np.isfinite(float(value)) if value is not None else False
        err = float(value) if measured else None
        return {
            "value": err,
            "limit": TOLERANCE,
            "pass": bool(measured and valid and err <= TOLERANCE),
            "measured": bool(measured),
            "method": modes.get("method"),
            "n_modes": modes.get("n_modes"),
            "note": notes,
        }

    return {
        "bore_position_modes<5%": entry(modes.get("bore_position_error")),
        "step_ratio_modes<5%": entry(modes.get("step_ratio_error")),
    }


def _verdict(case: DamBreakCase, metrics: dict) -> dict:
    """Headline checks against the predeclared 5 % tolerance."""
    if case.kind == "dry":
        # ``front_wetted<5%`` is retained as a legacy diagnostic: the wetted edge
        # is a depth-floor detector (it does not converge under refinement, see
        # docs/verification-dam-break-front-metric/REPORT.md), so the headline is
        # the shape-consistent fan estimator.
        checks = {
            "front_fan<5%": metrics["front_fan_error"],
            "front_wetted<5%": metrics["front_wetted_error"],
            "depth_l1<5%": metrics["depth_l1"],
            "monotone(no_extrema)": 0.0 if metrics["local_extrema"] == 0 else 1.0,
            "tv_no_growth<0.5%": max(0.0, metrics["tv_ratio"] - 1.0),
        }
    else:
        checks = {
            "shock_position<5%": metrics["shock_error"],
            "step_ratio<5%": metrics["step_ratio_error"],
            "plateau_h2<5%": metrics["h2_error"],
            "depth_l1<5%": metrics["depth_l1"],
            "monotone(no_extrema)": 0.0 if metrics["local_extrema"] == 0 else 1.0,
            "tv_no_growth<0.5%": max(0.0, metrics["tv_ratio"] - 1.0),
        }
    limits = {key: TOLERANCE for key in checks}
    limits["tv_no_growth<0.5%"] = 5e-3
    verdict = {
        key: {
            "value": value,
            "limit": limits[key],
            "pass": bool(value <= limits[key]),
        }
        for key, value in checks.items()
    }
    # The anchor acceptance is the *whole* ``x_f(t)`` series, not only the
    # end-of-run snapshot already covered above.  Before this entry the dry-bed
    # run reported ``all_pass`` at n = 400 while the same run's graded series
    # error was 6.61 % > 5 %: the acceptance quantity was measured but never
    # graded.  It is graded here (see :func:`_series_verdict_entry`).
    if case.kind == "wet":
        # Oracle-free re-grade of the same two ledger quantities (AUDIT.md §6 F4
        # residual / §9.5 open decision).  Added as *new* keys so every existing
        # key keeps its meaning and history.
        verdict.update(_oracle_free_verdict_entries(metrics))
    verdict["front_series<5%"] = _series_verdict_entry(metrics)
    verdict["all_pass"] = bool(all(item["pass"] for item in verdict.values()))
    return verdict


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def run_case(case: DamBreakCase, n: int = 1600, cfl: float = 0.4, order: int = 2,
             t_star: float | None = None) -> dict:
    """Integrate one dam-break case and compare it with the exact solution.

    ``t_star`` overrides the case's own end time (used for the time series in
    the report).  The returned dict is JSON-serialisable except for the
    ``profile`` arrays, which are returned as plain lists.
    """
    t_star_end = case.t_star_end if t_star is None else float(t_star)
    t_end = t_star_to_time(t_star_end, case.h0, case.gravity)
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    initial = np.where(solver.x < case.x_dam, case.h0, case.h1)
    solver.set_state(initial)
    volume0 = solver.total_volume()
    samples: list[dict] = []
    gate = {"next": 0.0}

    def sample(sol: ShallowWater1D) -> None:
        if sol.time < gate["next"]:
            return
        samples.append(front_sample(case, sol))
        gate["next"] = sol.time + t_end / 25.0

    wall = time.perf_counter()
    steps = solver.advance(t_end, callback=sample)
    wall = time.perf_counter() - wall

    h_ex, u_ex = case.exact(solver.x, solver.time)
    metrics = (_dry_metrics if case.kind == "dry" else _wet_metrics)(
        case, solver, volume0)
    metrics["time_to_front_series"] = samples
    metrics["front_series"] = series_metrics(case, samples)
    metrics["front_series_max_error"] = metrics["front_series"]["max_error"]
    run = {
        "case": case.summary(),
        "discretisation": {
            "n": n,
            "dx": solver.dx,
            "cfl": cfl,
            "order": order,
            "steps": steps,
            "t_end": solver.time,
            "t_star_end": t_star_end,
            "wall_time_s": wall,
            "volume_initial": volume0,
            "volume_final": solver.total_volume(),
            "clipped_cells": solver.clipped_cells,
            "dry_momentum_flips": solver.dry_flips,
        },
        "metrics": metrics,
        "verdict": _verdict(case, metrics),
        "profile": {
            "x": solver.x.tolist(),
            "h_numeric": solver.h.tolist(),
            "u_numeric": solver.primitive()[1].tolist(),
            "h_exact": np.asarray(h_ex, dtype=float).tolist(),
            "u_exact": np.asarray(u_ex, dtype=float).tolist(),
        },
    }
    return run


def convergence_study(case: DamBreakCase, resolutions=(100, 200, 400, 800, 1600, 3200),
                      cfl: float = 0.4, order: int = 2) -> dict:
    """Grid-refinement study; returns the errors and the observed order."""
    rows = []
    for n in resolutions:
        run = run_case(case, n=n, cfl=cfl, order=order)
        m = run["metrics"]
        headline = ({"front_fan": m["front_fan_error"],
                     "front_wetted_diagnostic": m["front_wetted_error"]}
                    if case.kind == "dry"
                    else {"shock": m["shock_error"], "step": m["step_ratio_error"]})
        series = m["front_series"]
        failing = [key for key, item in run["verdict"].items()
                   if key != "all_pass" and not item["pass"]]
        rows.append({
            "n": n,
            "dx": run["discretisation"]["dx"],
            "depth_l1": m["depth_l1"],
            "headline": headline,
            # the anchor acceptance itself (x_f(t) series) per grid, so the
            # refinement table cannot hide a grid on which it fails
            "front_series": series["max_error"],
            "front_series_graded": series["samples_graded"],
            "front_series_samples": series["samples"],
            "verdict_failing": failing,
            "verdict_all_pass": run["verdict"]["all_pass"],
            "tv_ratio": m["tv_ratio"],
            "local_extrema": m["local_extrema"],
        })
    dx = np.array([row["dx"] for row in rows])
    l1 = np.array([row["depth_l1"] for row in rows])
    fit = np.polyfit(np.log(dx), np.log(l1), 1) if len(rows) > 2 else [float("nan"), 0.0]
    headline_errors = np.array([
        next(iter(row["headline"].values())) for row in rows
    ])
    positive = headline_errors > 0.0
    fit_head = (np.polyfit(np.log(dx[positive]), np.log(headline_errors[positive]), 1)[0]
                if int(np.count_nonzero(positive)) > 2 else float("nan"))
    return {
        "case": case.key,
        "order": order,
        "cfl": cfl,
        "rows": rows,
        "observed_order_l1": float(fit[0]),
        "observed_order_headline": float(fit_head),
        "headline_error_at_finest": float(headline_errors[-1]),
        "headline_error_at_coarsest": float(headline_errors[0]),
    }


def _max_over_accepted(rows, key: str, accepted_n: int = ACCEPTED_MIN_N) -> float:
    """Max of ``row[key]`` over the grids the benchmark accepts (``n >= 800``).

    A coarse-only study has no accepted grid; that returns NaN instead of raising
    ``max() arg is an empty sequence`` (found by the metric-repair tests).
    """
    values = [row[key] for row in rows if row["n"] >= accepted_n]
    return float(max(values)) if values else float("nan")


def front_definition_study(resolutions=(400, 800, 1600, 3200)) -> dict:
    """How much does "the front" depend on how it is measured? (dry bed)

    The Ritter tip is a zero-depth cusp, so a finite-volume scheme spreads it
    over a few cells and the reported "front error" depends on the definition.
    Three definitions are evaluated on every grid:

    ``wetted_edge``
        outermost cell with ``h > 0`` -- threshold-free in appearance only: it is
        a *depth-floor* detector (the reading is set by the numerical depth floor,
        not by ``dx``), so it does **not** converge.  Kept as a diagnostic.
    ``level_raw``
        abscissa where ``h`` crosses ``level * h0`` (a level inside the smeared
        tail is *not* the front, so this is reported to keep that visible).
    ``level_scaled``
        the same crossing plus the exact fan offset ``3 t sqrt(g * level * h0)``
        back to the tip.
    ``fan_shape``
        mean of the per-cell exact identity ``x_f = x_i + 3 t sqrt(g h_i)`` over
        ``FAN_WINDOW`` -- the headline metric (:func:`fan_front_estimate`).
    """
    levels = (FRONT_LEVEL_FRACTION, 1e-3)
    rows = []
    for n in resolutions:
        run = run_case(RITTER_DRY, n=n)
        m = run["metrics"]
        row = {
            "n": n,
            "dx": run["discretisation"]["dx"],
            "front_exact": m["front_exact"],
            "wetted_edge": m["front_wetted"],
            "wetted_edge_error": m["front_wetted_error"],
            "fan_shape": m["front_fan"],
            "fan_shape_error": m["front_fan_error"],
            "fan_shape_spread": m["front_fan_spread"],
            "level_raw_error": {},
            "level_scaled_error": {},
        }
        solver = ShallowWater1D(n, RITTER_DRY.x_min, RITTER_DRY.x_max,
                                gravity=RITTER_DRY.gravity, cfl=0.4, order=2)
        solver.set_state(np.where(solver.x < RITTER_DRY.x_dam, RITTER_DRY.h0,
                                  RITTER_DRY.h1))
        solver.advance(run["discretisation"]["t_end"])
        t = solver.time
        x_f = m["front_exact"]
        for level in levels:
            x_raw = _level_crossing(solver.x, solver.h, level * RITTER_DRY.h0)
            x_scaled = x_raw + 3.0 * t * math.sqrt(RITTER_DRY.gravity
                                                   * level * RITTER_DRY.h0)
            key = f"{level:g}"
            row["level_raw_error"][key] = _relative(x_raw, x_f)
            row["level_scaled_error"][key] = _relative(x_scaled, x_f)
        rows.append(row)
    return {
        "case": RITTER_DRY.key,
        "max_wetted_edge_error_at_accepted_grid": _max_over_accepted(rows, "wetted_edge_error"),
        "max_fan_shape_error_at_accepted_grid": _max_over_accepted(rows, "fan_shape_error"),
        "rows": rows,
    }


def level_position_metrics(solver, case: DamBreakCase,
                           levels=LEVEL_POSITION_LEVELS) -> dict:
    """Dry-bed depth-contour positions: measured from the profile, graded by algebra.

    The dry-bed headline (:func:`fan_front_estimate`) locates the front through
    the **exact similarity identity** ``x_f = x_i + 3 t sqrt(g h_i)``, so its
    residual measures the *shape* error of the numerical fan rather than the
    absolute position of the profile.  This function supplies the missing
    position-only evidence:

    * the *measurement* is the sub-cell abscissa where the numerical depth falls
      through ``level * h0`` -- it uses the profile and ``level`` only, never the
      exact solution, the tip position or the fan identity;
    * the *reference* is :func:`~tensorfvm.shallow_water.exact.
      ritter_level_position`, an algebraic statement about where that same
      contour sits.  The reference is a different closed-form point than the
      tip, so agreeing with it is not a restatement of ``x_f``.

    Returns one entry per level plus the fan/tip diagnostics for context.
    """
    from .exact import ritter_level_position

    t = float(solver.time)
    x = np.asarray(solver.x, dtype=float)
    h = np.asarray(solver.h, dtype=float)
    rows = {}
    for level in levels:
        measured = _level_crossing(x, h, level * case.h0)
        exact = ritter_level_position(level, t, case.h0, case.gravity,
                                      case.x_dam)
        rows[f"{level:g}"] = {
            "level": float(level),
            "measured": measured,
            "exact": exact,
            "error": (_relative(measured, exact)
                      if measured == measured else float("nan")),
            "measured_cells_from_dam": (measured - case.x_dam) / float(solver.dx),
        }
    fan = fan_front_estimate(x, h, t, case.gravity, case.h0)
    return {
        "t": t,
        "t_star": t * math.sqrt(case.gravity / case.h0),
        "levels": rows,
        "max_error": max((r["error"] for r in rows.values()
                          if r["error"] == r["error"]), default=float("nan")),
        # context: the headline (shape) metric and the non-convergent legacy one
        "front_fan": fan["front"],
        "front_fan_error": (_relative(fan["front"], case.exact_front(t))
                            if fan["front"] == fan["front"] else float("nan")),
        "front_wetted_error": _relative(_wetted_edge(x, h), case.exact_front(t)),
    }


def _level_series(case: DamBreakCase, n: int, cfl: float, order: int,
                  level: float, samples_per_unit: int = 25) -> tuple[object, list]:
    """Integrate the dry case recording one depth-contour position over time."""
    from .exact import ritter_level_position

    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
    t_end = case.t_end
    gate = {"next": 0.0}
    samples: list[dict] = []

    def callback(sol) -> None:
        if sol.time < gate["next"]:
            return
        measured = _level_crossing(sol.x, sol.h, level * case.h0)
        exact = ritter_level_position(level, sol.time, case.h0, case.gravity,
                                      case.x_dam)
        travelled = (exact - case.x_dam) / float(sol.dx)
        samples.append({
            "t": float(sol.time),
            "t_star": sol.time * math.sqrt(case.gravity / case.h0),
            "measured": measured,
            "exact": exact,
            "error": (_relative(measured, exact)
                      if measured == measured else float("nan")),
            "travelled_cells": travelled,
            "resolved": bool(travelled >= LEVEL_SERIES_MIN_CELLS),
        })
        gate["next"] = sol.time + t_end / float(samples_per_unit)

    solver.advance(t_end, callback=callback)
    return solver, samples


def level_position_study(resolutions=(100, 200, 400, 800, 1600, 3200),
                         cfl: float = 0.4, order: int = 2,
                         level: float = LEVEL_POSITION_SERIES_LEVEL) -> dict:
    """Oracle-free dry-bed **position** study: contour locations and their ``x(t)``.

    Motivation (verification debt): the accepted dry-bed number is
    ``front_fan``, a *shape* metric built on the exact fan identity, while the
    ledger acceptance is the wave **position** ``x_f(t)``.  The only
    position-only definitions in the module were the wetted edge (a depth-floor
    detector that does not converge) and ``level_scaled`` (which adds the exact
    fan offset, i.e. imports the answer).  This study closes that gap by grading
    contour positions against their own closed-form locations
    (:func:`~tensorfvm.shallow_water.exact.ritter_level_position`).

    Reported per resolution:

    ``levels[level].error``
        ``|x_measured - x_exact| / x_exact`` for every declared contour.
    ``levels[level].observed_order``
        ``log2(e(n) / e(n/2))``; a contour of a profile smeared over ``O(dx)``
        should converge at **first** order, and that is what is measured -- the
        honest statement is that this metric is a *weaker* (first-order) test of
        position than the shape metric is of shape, not that either is exact.
    ``series_max_error``
        max over the graded samples of the same construction at
        ``level = 0.04``: the oracle-free analogue of the graded ``x_f(t)``
        series.
    """
    rows = []
    for n in resolutions:
        solver, samples = _level_series(case=RITTER_DRY, n=n, cfl=cfl,
                                        order=order, level=level)
        metrics = level_position_metrics(solver, RITTER_DRY)
        graded = [s for s in samples if s["resolved"]]
        rows.append({
            "n": n,
            "dx": float(solver.dx),
            "levels": metrics["levels"],
            "max_error": metrics["max_error"],
            "front_fan_error": metrics["front_fan_error"],
            "front_wetted_error": metrics["front_wetted_error"],
            "series_level": float(level),
            "series_samples": len(samples),
            "series_graded": len(graded),
            "series_max_error": (max((s["error"] for s in graded
                                      if s["error"] == s["error"]),
                                     default=float("nan")) if graded
                                 else float("nan")),
            "series": samples,
        })
    # observed order per level between consecutive resolutions
    for coarse, fine in zip(rows, rows[1:]):
        for key, entry in fine["levels"].items():
            e_c = coarse["levels"][key]["error"]
            e_f = entry["error"]
            entry["observed_order"] = (
                math.log2(e_c / e_f)
                if (e_c == e_c and e_f == e_f and e_c > 0.0 and e_f > 0.0)
                else float("nan"))
    accepted = [r for r in rows if r["n"] >= ACCEPTED_MIN_N]
    return {
        "case": RITTER_DRY.key,
        "reference": ("Ritter (1892) rarefaction; contour abscissa "
                      "x = x_dam + c0 t (2 - 3 sqrt(level))"),
        "levels": list(LEVEL_POSITION_LEVELS),
        "cfl": cfl,
        "order": order,
        "accepted_min_n": ACCEPTED_MIN_N,
        "tolerance": TOLERANCE,
        "rows": rows,
        "max_error_at_accepted_grid": (
            max((r["max_error"] for r in accepted
                 if r["max_error"] == r["max_error"]), default=float("nan"))),
        "max_series_error_at_accepted_grid": (
            max((r["series_max_error"] for r in accepted
                 if r["series_max_error"] == r["series_max_error"]),
                default=float("nan"))),
        "min_observed_order_at_accepted_grid": (
            min((e.get("observed_order", float("nan"))
                 for r in accepted for e in r["levels"].values()
                 if e.get("observed_order", float("nan")) == e.get("observed_order", float("nan"))),
                default=float("nan"))),
    }


def sensitivity_study(n: int = 800) -> dict:
    """Parameter sensitivity of the headline metrics (task section 5).

    Three knobs are swept one at a time around the benchmark point
    (``n = 800``, ``CFL = 0.4``, ``order = 2``, ``h0/h1 = 2``):

    * time-step constraint: ``CFL in {0.2, 0.4, 0.8}``;
    * spatial order: MUSCL minmod (2) vs piecewise constant (1);
    * bore strength: ``h1 in {0.8, 0.5, 0.25}`` (``h0/h1 = 1.25, 2, 4``).

    Each row reports the headline metrics of both cases, so a knob that only
    flatters one case is visible.
    """
    rows = []

    def row(label: str, **kwargs) -> dict:
        h1 = kwargs.pop("h1", None)
        case_wet = STOKER_WET
        if h1 is not None:
            case_wet = DamBreakCase(
                key=f"stoker-wet-h1-{h1}", kind="wet", h0=STOKER_WET.h0, h1=h1,
                gravity=STOKER_WET.gravity, x_dam=STOKER_WET.x_dam,
                x_min=STOKER_WET.x_min, x_max=STOKER_WET.x_max,
                t_star_end=STOKER_WET.t_star_end, reference=STOKER_WET.reference)
        dry = run_case(RITTER_DRY, n=n, **kwargs)["metrics"]
        wet = run_case(case_wet, n=n, **kwargs)["metrics"]
        entry = {
            "variant": label,
            "n": n,
            "cfl": kwargs.get("cfl", 0.4),
            "order": kwargs.get("order", 2),
            "h1": case_wet.h1,
            "depth_ratio": case_wet.h0 / case_wet.h1,
            # headline (fan) first; the wetted edge stays as a diagnostic because
            # it is a depth-floor detector, not a solution error
            "dry_front_fan_error": dry["front_fan_error"],
            "dry_front_error": dry["front_wetted_error"],
            "dry_series_max_error": dry.get("front_series_max_error"),
            "dry_depth_l1": dry["depth_l1"],
            "wet_shock_error": wet["shock_error"],
            "wet_step_ratio_error": wet["step_ratio_error"],
            "wet_depth_l1": wet["depth_l1"],
            "wet_tv_ratio": wet["tv_ratio"],
            "wet_local_extrema": wet["local_extrema"],
        }
        rows.append(entry)
        return entry

    for cfl in (0.2, 0.4, 0.8):
        row(f"cfl={cfl}", cfl=cfl)
    row("order=1", order=1)
    for h1 in (0.8, 0.5, 0.25):
        row(f"h1={h1}", h1=h1)
    worst = max(
        max(entry["dry_front_fan_error"], entry["wet_shock_error"],
            entry["wet_step_ratio_error"])
        for entry in rows)
    return {"reference_point": {"n": n, "cfl": 0.4, "order": 2, "h0/h1": 2.0},
            "rows": rows, "worst_headline_error": worst,
            "tolerance": TOLERANCE}


def exactness_checks(gravity: float = 9.81) -> dict:
    """Certify the closed forms without transcribing any tabulated number.

    * Stoker: Riemann invariant and Rankine-Hugoniot residuals at the bracketed
      root (relative, machine precision expected).
    * Ritter: identity checks on the fan (depth/velocity at the dam, front
      speed) and on the mass in the fan, ``int h dx = h0 c0 t`` over
      ``[x_head, x_f]``, verified by quadrature of the evaluated profile.
    """
    h0, h1 = 1.0, 0.5
    g = gravity
    stoker = check_stoker_solution(h0, h1, g)
    sol = stoker_constant_state(h0, h1, g)
    t = t_star_to_time(2.0, h0, g)
    c0 = math.sqrt(g * h0)
    xd = 0.0
    # Ritter fan identities at t = t_end
    h_dam, u_dam = ritter_state(np.array([xd]), t, h0, g, xd)
    x_f = ritter_front_position(t, h0, g, xd)
    x_head = xd - c0 * t
    xs = np.linspace(x_head, x_f, 2_000_001)
    h_fan, _ = ritter_state(xs, t, h0, g, xd)
    mass_fan = float(np.trapezoid(h_fan, xs)) if hasattr(np, "trapezoid") else float(
        np.trapz(h_fan, xs))
    return {
        "stoker_h0_h1": [h0, h1],
        "stoker_gravity": g,
        "stoker": stoker,
        "stoker_state": {key: float(val) for key, val in sol.items()
                         if isinstance(val, (int, float))},
        "ritter": {
            "h_at_dam_over_h0": float(h_dam[0] / h0),
            "h_at_dam_expected": 4.0 / 9.0,
            "h_at_dam_residual": float(h_dam[0] / h0 - 4.0 / 9.0),
            "u_at_dam_over_c0": float(u_dam[0] / c0),
            "u_at_dam_expected": 2.0 / 3.0,
            "u_at_dam_residual": float(u_dam[0] / c0 - 2.0 / 3.0),
            "front_over_c0t": float(x_f / (c0 * t)),
            "front_expected": 2.0,
            "fan_mass": mass_fan,
            "fan_mass_expected": h0 * c0 * t,
            "fan_mass_relative": abs(mass_fan - h0 * c0 * t) / (h0 * c0 * t),
        },
    }


# --------------------------------------------------------------------------- #
# artefacts
# --------------------------------------------------------------------------- #
def figure_profiles_and_convergence(runs: dict, studies: dict, path,
                                    t_star: float = 2.0) -> str:
    """Two-panel figure: profiles at ``t*`` and grid convergence of the errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colors = {"ritter-dry": "tab:blue", "stoker-wet": "tab:red"}
    for key, run in runs.items():
        ax = axes[0]
        x = np.asarray(run["profile"]["x"])
        h = np.asarray(run["profile"]["h_numeric"])
        he = np.asarray(run["profile"]["h_exact"])
        ax.plot(x, he, "-", lw=1.2, color=colors[key], alpha=0.55,
                label=f"{key}: exact ({run['case']['reference'].split('(')[0].strip()})")
        ax.plot(x, h, "--", lw=1.2, color=colors[key], label=f"{key}: FVM n={run['discretisation']['n']}")
    ax = axes[0]
    ax.set_xlim(-3.0, 4.6)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth h [m]")
    ax.set_title(f"Dam break at t* = {t_star:g} (h0 = 1 m, g = 9.81 m/s²)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for key, study in studies.items():
        rows = study["rows"]
        dx = [row["dx"] for row in rows]
        ax.loglog(dx, [row["depth_l1"] for row in rows], "o-",
                  color=colors[key], label=f"{key}: depth L1 (order {study['observed_order_l1']:.2f})")
        headline = [next(iter(row["headline"].values())) for row in rows]
        ax.loglog(dx, headline, "s--", color=colors[key], alpha=0.7,
                  label=f"{key}: wave-position error")
    ax.axhline(TOLERANCE, color="k", ls=":", lw=1.0)
    ax.text(ax.get_xlim()[0], TOLERANCE * 1.1, " 5 % criterion", fontsize=8)
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("relative error [-]")
    ax.set_title("Grid convergence (SSP-RK3 + MUSCL minmod + HLL)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_artifacts(outdir, payload: dict) -> dict:
    """Write the JSON/CSV artefacts of a benchmark campaign into ``outdir``."""
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    meta = {key: val for key, val in payload.items()
            if key not in ("runs", "studies", "profiles")}
    metrics = {}
    for key, run in payload["runs"].items():
        row = {
            "case": run["case"],
            "discretisation": run["discretisation"],
            "metrics": {k: v for k, v in run["metrics"].items()
                        if k != "time_to_front_series"},
            "verdict": run["verdict"],
            # raw x_f(t) samples plus the *graded* summary of the series; the
            # two are separate keys because "front_series" used to be the raw
            # list while the pass/fail statement lived nowhere
            "front_series_samples": run["metrics"]["time_to_front_series"],
            "front_series_summary": run["metrics"]["front_series"],
        }
        metrics[key] = row
        # profiles as csv (x, h, exact h, u, exact u)
        p = run["profile"]
        csv_path = out / f"profile_{key}_n{run['discretisation']['n']}.csv"
        with csv_path.open("w") as fh:
            fh.write("x,h_numeric,h_exact,u_numeric,u_exact\n")
            for row_i in zip(p["x"], p["h_numeric"], p["h_exact"],
                             p["u_numeric"], p["u_exact"]):
                fh.write(",".join(f"{value!r}" for value in row_i) + "\n")
        written[csv_path.name] = str(csv_path)
    path = out / "metrics.json"
    path.write_text(json.dumps({"meta": meta, "cases": metrics}, indent=2) + "\n")
    written["metrics.json"] = str(path)
    path = out / "convergence.json"
    path.write_text(json.dumps(payload["studies"], indent=2) + "\n")
    written["convergence.json"] = str(path)
    path = out / "exactness.json"
    path.write_text(json.dumps(payload["exactness"], indent=2) + "\n")
    written["exactness.json"] = str(path)
    return written