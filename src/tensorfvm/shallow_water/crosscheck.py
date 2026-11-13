"""Independent analytic cross-check of the 1-D shallow-water dam-break solver.

Node: ``CASE-FVM-DAMBREAK-STOKER``.  This module is the *second, independent*
path to the same verdict as :mod:`tensorfvm.shallow_water.dam_break`.  The
existing benchmark measures front/bore with a data-driven plateau detector and a
wetted-edge diagnostic that depends on the grid's depth floor; the quantities
below depend on neither:

1. **The Stoker bore state is solved again, by Newton on the depth ratio.**
   ``exact.stoker_constant_state`` obtains ``(h2, u2, U)`` from a bisection on
   ``c2 = sqrt(g h2)``; here the unknown is ``r = h2/h1`` and the residual is
   the same physical matching condition in dimensionless form.  The two roots
   must agree to machine precision, which is what :func:`reference_selfcheck`
   asserts.  Nothing is transcribed from a table.
2. **Wave kinematics are read off fixed *depth levels*.**  Inside the
   rarefaction fan the exact solution inverts in closed form,
   ``h = (2 c0 - xi)^2 / (9 g)`` with ``xi = (x - x_dam)/t``, so a level
   crossing at depth ``h_p`` sits at ``xi_p = 2 c0 - 3 sqrt(g h_p)``.  The probe
   depths are fixed *fractions of the physical depth drop* ``h0 - h2`` (or of
   the bore step ``h2 - h1``), never multiples of ``dx`` and never a floor
   value, so the same probe is applied on every grid and a coarse grid cannot
   pass by accident.
3. **The depth/velocity fields are compared in similarity coordinates**, region
   by region (undisturbed upstream, fan, post-bore plateau, undisturbed
   downstream), which is where a scheme's error actually lives: one global L1
   number can hide a plateau that is 8 % low behind an accurate fan.

What is deliberately *not* gated: the L-infinity depth error.  The exact
solution jumps at the bore and has a kink at the fan head, so a captured
discontinuity always carries ``|e| ~ (h2 - h1)/2`` there, shrinking only at
first order in ``dx``; gating it at 5 % would either fail forever or hide the
number behind a fudge factor.  ``depth_linf`` is reported, the smooth-region L1
norms and the probe positions carry the gate.

Tolerances
----------
``CROSSCHECK_TOL = 0.05`` (5 %, the task tolerance) is applied to every probe
position, to the profile L1 norm and to the plateau state.  The verdict lists
the individual checks so a report can quote the gate that was actually applied
instead of an aggregate.

References
----------
Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**, 947; Stoker (1957) *Water Waves*,
ch. 10; Toro (2009) chapters 6-7 for the HLL/MUSCL/RK discretisation used by the
solver under test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .exact import (
    ritter_state,
    stoker_region_bounds,
    stoker_state,
    t_star_to_time,
)
from .swe1d import ShallowWater1D

__all__ = [
    "AnalyticCase",
    "CASES",
    "CROSSCHECK_TOL",
    "PROBE_FRACTIONS",
    "bore_state_newton",
    "crosscheck_case",
    "convergence_series",
    "exact_mass_over_domain",
    "fan_xi_at_depth",
    "level_crossing",
    "profile_metrics",
    "reference_selfcheck",
    "similarity_probes",
]

#: Task tolerance (5 %) applied to every probe position, to the profile L1 norm
#: and to the plateau state.  A module constant so a report can quote the gate.
CROSSCHECK_TOL = 0.05

#: Probe definition.  ``fan_head_drop`` / ``fan_mid_drop`` are fractions of the
#: *depth drop* ``h0 - h2`` measured down from ``h0``; ``front`` is a fraction of
#: ``h0`` above the dry bed (Ritter case only); ``bore_mid`` is a fraction of the
#: bore step ``h2 - h1`` measured up from ``h1``.  All are physical fractions, so
#: the probes are resolution independent by construction.
PROBE_FRACTIONS = {"fan_head_drop": 0.02, "fan_mid_drop": 0.5,
                   "front": 0.02, "bore_mid": 0.5}


# --------------------------------------------------------------------------- #
# case definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalyticCase:
    """A dam-break Riemann problem together with its closed-form reference.

    ``kind="dry"`` selects the Ritter fan (``h1 = 0``), ``kind="wet"`` the
    Stoker fan + bore (``h0 > h1 > 0``).  The default domain is wider than the
    wave reach at ``t*_end = 2`` (rarefaction head at ``c0 t = 2.00`` m to the
    left, bore at ``U t = 1.89`` m to the right), so the transmissive ends never
    see the solution and a genuine conservation statement becomes possible.
    """

    key: str
    kind: str
    h0: float
    h1: float = 0.0
    gravity: float = 9.81
    x_min: float = -4.0
    x_max: float = 6.0
    x_dam: float = 0.0
    t_star_end: float = 2.0

    def __post_init__(self) -> None:
        if self.kind not in ("dry", "wet"):
            raise ValueError(f"kind must be 'dry' or 'wet', got {self.kind!r}")
        if self.kind == "wet" and not (self.h0 > self.h1 > 0.0):
            raise ValueError(
                f"wet case needs h0 > h1 > 0, got h0={self.h0}, h1={self.h1}")
        if self.kind == "dry" and self.h1 != 0.0:
            raise ValueError("dry case needs h1 = 0")
        if not self.x_max > self.x_min:
            raise ValueError("x_max must exceed x_min")
        if not 0.0 < self.t_star_end:
            raise ValueError("t_star_end must be positive")

    @property
    def t_end(self) -> float:
        """Physical end time for the non-dimensional ``t*_end``."""
        return t_star_to_time(self.t_star_end, self.h0, self.gravity)

    @property
    def c0(self) -> float:
        return math.sqrt(self.gravity * self.h0)

    def exact_state(self, x, t: float):
        """Exact ``(h, u)`` at ``(x, t)`` from :mod:`tensorfvm.shallow_water.exact`."""
        if self.kind == "dry":
            return ritter_state(x, t, self.h0, self.gravity, self.x_dam)
        return stoker_state(x, t, self.h0, self.h1, self.gravity, self.x_dam)

    def bounds(self) -> dict:
        """Similarity-coordinate boundaries of the exact wave pattern."""
        if self.kind == "dry":
            front = 2.0 * self.c0
            return {"head": -self.c0, "tail": front, "shock": front,
                    "h2": 0.0, "u2": 0.0, "c0": self.c0, "c2": 0.0}
        return stoker_region_bounds(self.h0, self.h1, self.gravity)

    def initial_depth(self, x) -> np.ndarray:
        """Dam-break initial condition on the given cell centres."""
        return np.where(np.asarray(x, dtype=float) < self.x_dam, self.h0, self.h1)

    def summary(self) -> dict:
        return {"key": self.key, "kind": self.kind, "h0": self.h0,
                "h1": self.h1, "gravity": self.gravity, "x_dam": self.x_dam,
                "x_min": self.x_min, "x_max": self.x_max,
                "t_star_end": self.t_star_end, "t_end": self.t_end}


RITTER_DRY = AnalyticCase(key="ritter-dry", kind="dry", h0=1.0, h1=0.0)
STOKER_WET = AnalyticCase(key="stoker-wet", kind="wet", h0=1.0, h1=0.5)
CASES = {case.key: case for case in (RITTER_DRY, STOKER_WET)}


# --------------------------------------------------------------------------- #
# independent solution of the Stoker bore state
# --------------------------------------------------------------------------- #
def bore_state_newton(h0: float, h1: float, gravity: float = 9.81,
                      tol: float = 1e-15, max_iter: int = 200) -> dict:
    """Solve the wet-bed dam break by Newton iteration on ``r = h2/h1``.

    A *different* root finder on a different unknown than
    ``exact.stoker_constant_state`` (bisection on ``c2``), so agreement between
    the two certifies the closed form instead of restating it.  With
    ``r = h2/h1`` the three matching conditions

    * rarefaction invariant ``u2 + 2 c2 = 2 c0``,
    * bore mass jump ``U (h2 - h1) = h2 u2``,
    * bore momentum jump ``U h2 u2 = g (h2^2 - h1^2)/2 + h2 u2^2``,

    collapse into the single dimensionless residual

    ``F(r) = 2 (sqrt(h0/h1) - sqrt(r)) - (r - 1) sqrt((r + 1) / (2 r))``,

    with ``F(1) = 2 (sqrt(h0/h1) - 1) > 0`` and ``F(h0/h1) < 0``: a bracket that
    needs no tuning.  A bracketed Newton step (bisection fallback when a step
    would leave the bracket) keeps the iterate physical.
    """
    if not (h0 > h1 > 0.0):
        raise ValueError(f"Stoker needs h0 > h1 > 0, got h0={h0}, h1={h1}")
    ratio = float(h0) / float(h1)
    root = math.sqrt(ratio)

    def residual(r: float) -> float:
        return 2.0 * (root - math.sqrt(r)) - (r - 1.0) * math.sqrt(
            (r + 1.0) / (2.0 * r))

    def derivative(r: float) -> float:
        # d/dr [2 root - 2 sqrt(r) - (r - 1) sqrt(q)], q = (r + 1)/(2 r),
        # q' = -1/(2 r^2).
        q = (r + 1.0) / (2.0 * r)
        d_q = -1.0 / (2.0 * r * r)
        return (-1.0 / math.sqrt(r) - math.sqrt(q)
                - (r - 1.0) * d_q / (2.0 * math.sqrt(q)))

    lo, hi = 1.0 + 1e-12, ratio
    r = 0.5 * (lo + hi)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        f = residual(r)
        if f == 0.0:
            break
        if f > 0.0:
            lo = r
        else:
            hi = r
        slope = derivative(r)
        candidate = r - f / slope if slope != 0.0 else 0.5 * (lo + hi)
        r_next = candidate if lo < candidate < hi else 0.5 * (lo + hi)
        if abs(r_next - r) <= tol * max(1.0, abs(r)):
            r = r_next
            break
        r = r_next
    h2 = r * float(h1)
    c2 = math.sqrt(gravity * h2)
    u2 = 2.0 * (math.sqrt(gravity * float(h0)) - c2)
    shock_mass = h2 * u2 / (h2 - float(h1))
    # Independent second formula for the shock speed: eliminate ``u2`` from the
    # two jumps instead of from the invariant, giving U^2 = g h2 (h2+h1)/(2 h1).
    shock_momentum = math.sqrt(gravity * h2 * (h2 + float(h1)) / (2.0 * float(h1)))
    return {"r": r, "h2": h2, "u2": u2, "c2": c2,
            "shock_speed": shock_mass,
            "shock_speed_momentum": shock_momentum,
            "residual": residual(r), "iterations": iterations}


# --------------------------------------------------------------------------- #
# wave kinematics from depth-level crossings
# --------------------------------------------------------------------------- #
def fan_xi_at_depth(depth: float, h0: float, gravity: float = 9.81) -> float:
    """Similarity coordinate where the rarefaction fan carries ``depth``.

    Inverts ``h = (2 c0 - xi)^2 / (9 g)`` (Ritter 1892; valid for both the dry
    and the wet bed inside the fan): ``xi = 2 c0 - 3 sqrt(g depth)``.  A probe at
    ``depth = h0 - f (h0 - h2)`` therefore has the closed-form position
    ``2 c0 - 3 sqrt(g (h0 - f (h0 - h2)))``, independent of the grid.
    """
    if not 0.0 <= depth <= h0:
        raise ValueError(f"fan depth must be in [0, {h0}], got {depth}")
    return 2.0 * math.sqrt(gravity * h0) - 3.0 * math.sqrt(gravity * depth)


def level_crossing(x, values, level: float, direction: str = "decreasing") -> float:
    """First crossing of ``values`` through ``level``, linearly interpolated.

    ``x`` holds cell centres, ``values`` the field.  ``direction`` selects a
    crossing from above (``"decreasing"``) or from below (``"increasing"``);
    scanning left to right means the *first* such crossing is reported, which is
    what the probe definitions rely on.  Returns ``NaN`` when the level is not
    crossed -- an honest "not measured" instead of a fabricated position.
    """
    xs = np.asarray(x, dtype=float)
    d = np.asarray(values, dtype=float) - float(level)
    if not np.all(np.isfinite(d)):
        fill = -np.inf if direction == "decreasing" else np.inf
        d = np.where(np.isfinite(d), d, fill)
    if direction == "decreasing":
        hits = np.where((d[:-1] > 0.0) & (d[1:] <= 0.0))[0]
    elif direction == "increasing":
        hits = np.where((d[:-1] <= 0.0) & (d[1:] > 0.0))[0]
    else:
        raise ValueError(
            f"direction must be 'decreasing' or 'increasing', got {direction!r}")
    if hits.size == 0:
        return float("nan")
    i = int(hits[0])
    d0, d1 = float(d[i]), float(d[i + 1])
    weight = 0.5 if d0 == d1 else d0 / (d0 - d1)
    return float(xs[i] + weight * (xs[i + 1] - xs[i]))


def _probe_levels(case: AnalyticCase) -> dict:
    """Depth levels (and crossing directions) of the kinematic probes.

    Fan probes sit strictly between ``h2`` and ``h0`` -- a level below ``h2`` is
    never crossed inside the fan, which is exactly the mistake this helper makes
    impossible by construction -- and the bore probe sits strictly between
    ``h1`` and ``h2``, where the depth *decreases* to the right across the bore
    (post-bore ``h2`` upstream of the shock, ``h1`` downstream of it).
    """
    b = case.bounds()
    drop = case.h0 - b["h2"]
    levels = {
        "fan_head": (case.h0 - PROBE_FRACTIONS["fan_head_drop"] * drop,
                     "decreasing"),
        "fan_mid": (case.h0 - PROBE_FRACTIONS["fan_mid_drop"] * drop,
                    "decreasing"),
    }
    if case.kind == "dry":
        levels["front"] = (PROBE_FRACTIONS["front"] * case.h0, "decreasing")
    else:
        step = b["h2"] - case.h1
        levels["bore"] = (case.h1 + PROBE_FRACTIONS["bore_mid"] * step,
                          "decreasing")
    return levels


def similarity_probes(case: AnalyticCase, x, h, t: float) -> dict:
    """Locate the wave pattern in ``h(x)`` and compare it with the closed form.

    Every probe is a depth level with an analytic position, so the comparison is
    ``measured xi`` vs ``exact xi`` -- no reference to ``dx``, to the dry floor
    or to a plateau detector.  Fan probes invert :func:`fan_xi_at_depth`; front
    probes use the same fan inversion at a fixed fraction of ``h0`` (the dry-bed
    tip is the ``h -> 0`` limit of the fan); the bore probe uses the mid-jump
    level and the exact shock speed ``U``.

    Accuracy floor: a probe is a linear interpolation between cell centres, so
    the smooth fan probes are limited by curvature (``O(dx^2)``) while the bore
    probe, whose level straddles a discontinuity, is limited by the sampling
    (``O(dx)``: about half a cell, i.e. ``dx / (2 U t)`` in relative terms).
    That floor belongs to the measurement, not to the scheme, and it is stated
    here so the reported error is not mistaken for solver error.
    """
    xs = np.asarray(x, dtype=float)
    hs = np.asarray(h, dtype=float)
    if t <= 0.0:
        raise ValueError("probes need t > 0")
    b = case.bounds()
    probes: dict[str, dict] = {}
    for name, (level, direction) in _probe_levels(case).items():
        exact_xi = b["shock"] if name == "bore" else fan_xi_at_depth(
            level, case.h0, case.gravity)
        measured = level_crossing(xs, hs, level, direction)
        xi = measured / t if np.isfinite(measured) else float("nan")
        probes[name] = {
            "level_depth": float(level),
            "x_measured": measured,
            "xi_measured": xi,
            "xi_exact": float(exact_xi),
            "rel_error": (abs(xi - exact_xi) / abs(exact_xi)
                          if np.isfinite(xi) and exact_xi != 0.0 else float("nan")),
        }
    return probes


# --------------------------------------------------------------------------- #
# field errors in similarity coordinates
# --------------------------------------------------------------------------- #
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral (avoids the numpy 1.x/2.x ``trapz``/``trapezoid`` split)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _region_masks(case: AnalyticCase, xi: np.ndarray) -> dict:
    """Boolean masks of the exact solution's similarity regions."""
    b = case.bounds()
    if case.kind == "dry":
        return {"upstream": xi <= b["head"],
                "fan": (xi > b["head"]) & (xi < b["tail"]),
                "dry_bed": xi >= b["tail"]}
    return {"upstream": xi <= b["head"],
            "fan": (xi > b["head"]) & (xi < b["tail"]),
            "plateau": (xi >= b["tail"]) & (xi <= b["shock"]),
            "downstream": xi > b["shock"]}


def exact_mass_over_domain(case: AnalyticCase, t: float,
                           x_min: float | None = None,
                           x_max: float | None = None) -> float:
    """Closed-form water volume of the exact solution over ``[x_min, x_max]``.

    The exact solution is piecewise polynomial, so its mass integral has a closed
    form -- no quadrature, hence no quadrature error in the check that the
    reference conserves mass.  With ``A(xi) = 2 c0 - xi`` the fan integral is

    ``int A(xi)^2/(9 g) dx = t/(27 g) [A(xi_a)^3 - A(xi_b)^3]``,

    which is the ``h0 c0 t`` Ritter identity on the dry bed and its wet-bed
    counterpart on the wet bed.
    """
    x_min = case.x_min if x_min is None else float(x_min)
    x_max = case.x_max if x_max is None else float(x_max)
    if not x_max > x_min:
        raise ValueError("x_max must exceed x_min")
    if t <= 0.0:
        left = min(max(case.x_dam - x_min, 0.0), x_max - x_min)
        return case.h0 * left + case.h1 * (x_max - x_min - left)
    b = case.bounds()
    c0 = case.c0
    g = case.gravity
    x_head = case.x_dam + b["head"] * t
    x_tail = case.x_dam + b["tail"] * t
    x_shock = case.x_dam + b["shock"] * t
    mass = 0.0
    if x_head > x_min:
        mass += case.h0 * (min(x_head, x_max) - x_min)
    fan_lo, fan_hi = max(x_head, x_min), min(x_tail, x_max)
    if fan_hi > fan_lo:
        a_lo = 2.0 * c0 - (fan_lo - case.x_dam) / t
        a_hi = 2.0 * c0 - (fan_hi - case.x_dam) / t
        mass += t * (a_lo ** 3 - a_hi ** 3) / (27.0 * g)
    plateau_lo, plateau_hi = max(x_tail, x_min), min(x_shock, x_max)
    if plateau_hi > plateau_lo:
        mass += b["h2"] * (plateau_hi - plateau_lo)
    if x_shock < x_max:
        mass += case.h1 * (x_max - max(x_shock, x_min))
    return mass

def profile_metrics(case: AnalyticCase, x, h, u, t: float) -> dict:
    """Depth/velocity error norms of a numerical profile against the closed form.

    Norms are *relative* (divided by the corresponding exact norm), hence
    dimensionless, and are reported globally plus per similarity region
    (upstream / fan / plateau / downstream): a scheme that smears the bore while
    getting the fan right and one that gets the plateau depth wrong both look
    like "2 % L1" globally.

    ``mass_profile_error`` compares the trapezoidal integrals of the numeric and
    exact *cell-centre samples*: it therefore also contains an ``O(dx^2)`` (and,
    at the kinks, ``O(dx)``) quadrature difference and is a profile metric, not a
    conservation statement.  Conservation is measured separately, by the solver's
    own volume drift, in :func:`crosscheck_case`.
    """
    xs = np.asarray(x, dtype=float)
    hs = np.asarray(h, dtype=float)
    us = np.asarray(u, dtype=float)
    h_ex, u_ex = case.exact_state(xs, t)
    h_ex = np.asarray(h_ex, dtype=float)
    u_ex = np.asarray(u_ex, dtype=float)
    e = hs - h_ex
    exact_mass = _trapz(h_ex, xs)
    exact_mass_sq = _trapz(h_ex * h_ex, xs)
    u_norm = _trapz(np.abs(u_ex), xs)
    u_scale = max(float(np.max(np.abs(u_ex))), 1e-300)
    metrics = {
        "depth_l1": (_trapz(np.abs(e), xs) / exact_mass
                     if exact_mass > 0 else float("nan")),
        "depth_l2": (math.sqrt(_trapz(e * e, xs) / exact_mass_sq)
                     if exact_mass_sq > 0 else float("nan")),
        "depth_linf": float(np.max(np.abs(e))) / case.h0,
        "velocity_l1": (_trapz(np.abs(us - u_ex), xs) / u_norm
                        if u_norm > 0 else float("nan")),
        "velocity_linf": float(np.max(np.abs(us - u_ex))) / u_scale,
        "exact_mass_sampled": exact_mass,
        "numeric_mass_sampled": _trapz(hs, xs),
        "regions": {},
    }
    metrics["mass_profile_error"] = (
        abs(metrics["numeric_mass_sampled"] - exact_mass) / exact_mass
        if exact_mass > 0 else float("nan"))
    xi = (xs - case.x_dam) / t
    for name, mask in _region_masks(case, xi).items():
        if not np.any(mask):
            continue
        xr, er, hr = xs[mask], e[mask], h_ex[mask]
        mass = _trapz(hr, xr) if xr.size > 1 else float(np.sum(hr))
        # A region with zero exact mass (the dry bed) has no meaningful relative
        # norm: ``depth_l1`` is left undefined (NaN, serialised as null) and the
        # absolute ``depth_linf`` carries the statement instead.
        l1 = _trapz(np.abs(er), xr) / mass if mass > 0 else float("nan")
        metrics["regions"][name] = {"cells": int(np.count_nonzero(mask)),
                                    "exact_mass": mass,
                                    "depth_l1": l1,
                                    "depth_linf": float(np.max(np.abs(er))) / case.h0}
    metrics["probes"] = similarity_probes(case, xs, hs, t)
    if case.kind == "wet":
        b = case.bounds()
        dx = float(xs[1] - xs[0])
        pad = max(4.0 * dx / t, 0.05 * (b["shock"] - b["tail"]))
        inside = (xi >= b["tail"] + pad) & (xi <= b["shock"] - pad)
        if np.count_nonzero(inside) >= 2:
            h_plateau = float(np.median(hs[inside]))
            u_plateau = float(np.median(us[inside]))
            metrics["plateau"] = {
                "h_numeric": h_plateau, "h_exact": b["h2"],
                "u_numeric": u_plateau, "u_exact": b["u2"],
                "depth_rel_error": abs(h_plateau - b["h2"]) / b["h2"],
                "velocity_rel_error": (abs(u_plateau - b["u2"]) / b["u2"]
                                       if b["u2"] != 0.0 else float("nan")),
                "step_ratio_numeric": h_plateau / case.h1,
                "step_ratio_exact": b["h2"] / case.h1,
                "cells": int(np.count_nonzero(inside)),
            }
        else:
            metrics["plateau"] = {"cells": int(np.count_nonzero(inside)),
                                  "h_exact": b["h2"], "u_exact": b["u2"],
                                  "depth_rel_error": float("nan"),
                                  "velocity_rel_error": float("nan")}
    return metrics


def _verdict(metrics: dict, tol: float) -> dict:
    """Apply the acceptance gate; every check is named so a report can cite it."""
    probes = metrics["probes"]
    checks = {"depth_l1": metrics["depth_l1"],
              "fan_head": probes["fan_head"]["rel_error"],
              "fan_mid": probes["fan_mid"]["rel_error"]}
    for key in ("front", "bore"):
        if key in probes:
            checks[key] = probes[key]["rel_error"]
    plateau = metrics.get("plateau", {})
    if plateau.get("h_exact"):
        checks["plateau_depth"] = plateau["depth_rel_error"]
        checks["plateau_velocity"] = plateau["velocity_rel_error"]
    finite = {name: (float(value) if value is not None and np.isfinite(value)
                     else None) for name, value in checks.items()}
    failed = [name for name, value in finite.items()
              if value is None or value > tol]
    return {"tolerance": tol, "checks": finite, "failed": failed,
            "pass": not failed}


# --------------------------------------------------------------------------- #
# end-to-end runs
# --------------------------------------------------------------------------- #
def crosscheck_case(case: AnalyticCase, n: int = 800, cfl: float = 0.4,
                    order: int = 2, tol: float = CROSSCHECK_TOL) -> dict:
    """Integrate one case and return its cross-check record (JSON-serialisable)."""
    t_end = case.t_end
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    solver.set_state(case.initial_depth(solver.x))
    volume0 = solver.total_volume()
    steps = solver.advance(t_end)
    h, u = solver.primitive()
    metrics = profile_metrics(case, solver.x, h, u, solver.time)
    metrics["volume_initial"] = volume0
    metrics["volume_drift"] = (abs(solver.total_volume() - volume0) / volume0
                               if volume0 > 0 else float("nan"))
    metrics["exact_mass_analytic"] = exact_mass_over_domain(
        case, solver.time, solver.x_min, solver.x_max)
    return {
        "case": case.summary(),
        "bounds": case.bounds(),
        "discretisation": {"n": n, "dx": solver.dx, "cfl": cfl, "order": order,
                           "steps": steps, "t_end": solver.time,
                           "t_star_end": case.t_star_end,
                           "clipped_cells": solver.clipped_cells,
                           "dry_momentum_flips": solver.dry_flips},
        "metrics": metrics,
        "verdict": _verdict(metrics, tol),
    }


def convergence_series(case: AnalyticCase, resolutions=(200, 400, 800, 1600),
                       cfl: float = 0.4, order: int = 2,
                       tol: float = CROSSCHECK_TOL,
                       accepted_n: int = 800) -> dict:
    """Grid-refinement series of the cross-check metrics and its observed orders.

    The observed order is fitted on ``depth_l1`` and on the *fan-mid* probe, the
    two metrics that are smooth in ``dx``.  The dry-bed front probe is excluded
    from the order fit on purpose: at the tip the exact solution is not smooth
    (``h ~ (x_f - x)^2``), so a first-order-but-noisy behaviour there says
    nothing about the scheme in the smooth regions.

    Two verdicts are returned: ``all_pass`` (every resolution) and
    ``accepted_pass`` (only ``n >= accepted_n``).  They are *different* on
    purpose -- the fan-head probe is first-order limited and misses 5 % on coarse
    grids, so a series that reported only ``all_pass`` would hide where the
    acceptance boundary is, and one that reported only ``accepted_pass`` would
    hide that the coarse grids fail.
    """
    rows = []
    for n in resolutions:
        run = crosscheck_case(case, n=n, cfl=cfl, order=order, tol=tol)
        m = run["metrics"]
        probes = m["probes"]
        headline_key = "bore" if "bore" in probes else "front"
        rows.append({
            "n": n, "dx": run["discretisation"]["dx"],
            "depth_l1": m["depth_l1"], "depth_l2": m["depth_l2"],
            "depth_linf": m["depth_linf"],
            "fan_mid": probes["fan_mid"]["rel_error"],
            "headline": {headline_key: probes[headline_key]["rel_error"]},
            "volume_drift": m["volume_drift"],
            "pass": run["verdict"]["pass"],
        })
    dx = np.array([row["dx"] for row in rows], dtype=float)
    l1 = np.array([row["depth_l1"] for row in rows], dtype=float)
    fan = np.array([row["fan_mid"] for row in rows], dtype=float)

    def _order(values: np.ndarray) -> float:
        good = np.isfinite(values) & (values > 0.0)
        if int(np.count_nonzero(good)) < 3:
            return float("nan")
        return float(np.polyfit(np.log(dx[good]), np.log(values[good]), 1)[0])

    accepted = [row for row in rows if row["n"] >= accepted_n]
    return {"case": case.key, "cfl": cfl, "order": order,
            "accepted_n": accepted_n, "rows": rows,
            "observed_order_l1": _order(l1),
            "observed_order_fan_mid": _order(fan),
            "all_pass": all(row["pass"] for row in rows),
            "accepted_pass": bool(accepted) and all(row["pass"] for row in accepted)}


# --------------------------------------------------------------------------- #
# certification of the closed-form reference itself
# --------------------------------------------------------------------------- #
def reference_selfcheck(case: AnalyticCase = STOKER_WET,
                        tol: float = 1e-12) -> dict:
    """Certify the reference solution without reference to any table.

    Five independent statements, each of which would fail if the closed form (or
    its implementation) were wrong:

    1. the Newton root of :func:`bore_state_newton` coincides with the
       ``exact.stoker_constant_state`` bisection root (different unknown,
       different algorithm);
    2. the mass and momentum jump residuals of ``(h2, u2, U)`` vanish, the
       Riemann invariant holds, and the two independent formulas for ``U``
       agree;
    3. the fan formula evaluated *at* both similarity boundaries reproduces the
       neighbouring constant states (``h(head) = h0``, ``h(tail) = h2``) --
       checked analytically, not by differencing, so the statement is exact
       rather than finite-difference limited;
    4. the closed-form mass integral :func:`exact_mass_over_domain` at ``t_end``
       equals the initial mass over the same domain (the wave pattern stays
       inside it), and the Ritter identities hold on the dry bed;
    5. the quadrature of ``h(x, t)`` over a 240 m window agrees with the
       closed-form mass to the quadrature's own ``O(dx)`` accuracy at the kinks.

    ``tol`` is a certification threshold of the *reference*, not of the solver,
    hence far tighter than ``CROSSCHECK_TOL``.  Check 5 is reported separately as
    ``quadrature_relative`` because it measures the quadrature, not the solution.
    """
    def rel(a: float, b: float) -> float:
        return abs(a - b) / abs(b) if b != 0.0 else abs(a)

    out = {"case": case.summary(), "tolerance": float(tol), "checks": {}}
    b = case.bounds()
    t = case.t_end
    g = case.gravity
    c0 = case.c0
    if case.kind == "wet":
        newton = bore_state_newton(case.h0, case.h1, g)
        out["bore_state"] = {key: float(newton[key]) for key in
                            ("h2", "u2", "c2", "shock_speed", "r", "residual")}
        out["checks"].update({
            "newton_vs_bisection_h2": rel(newton["h2"], b["h2"]),
            "newton_vs_bisection_u2": rel(newton["u2"], b["u2"]),
            "newton_vs_bisection_shock": rel(newton["shock_speed"], b["shock"]),
            "shock_two_formulas": rel(newton["shock_speed"],
                                      newton["shock_speed_momentum"]),
            "riemann_invariant": rel(newton["u2"] + 2.0 * newton["c2"], 2.0 * c0),
            "mass_jump": rel(newton["shock_speed"] * (newton["h2"] - case.h1),
                             newton["h2"] * newton["u2"]),
            "momentum_jump": rel(
                newton["shock_speed"] * newton["h2"] * newton["u2"],
                0.5 * g * (newton["h2"] ** 2 - case.h1 ** 2)
                + newton["h2"] * newton["u2"] ** 2),
        })
    else:
        # Ritter identities: h(x_dam) = 4 h0/9 and u(x_dam) = 2 c0/3.
        h_dam, u_dam = case.exact_state(np.array([case.x_dam]), t)
        out["checks"]["h_at_dam"] = rel(float(h_dam[0]), 4.0 * case.h0 / 9.0)
        out["checks"]["u_at_dam"] = rel(float(u_dam[0]), 2.0 * c0 / 3.0)
        # Closed-form fan mass vs the Ritter identity: int over the fan of
        # (2 c0 - xi)^2/(9 g) dx = h0 c0 t, evaluated analytically.
        fan_mass = exact_mass_over_domain(
            case, t, case.x_dam - c0 * t, case.x_dam + 2.0 * c0 * t)
        out["checks"]["fan_mass_identity"] = rel(fan_mass, case.h0 * c0 * t)
    # 3. The fan formula must reproduce the neighbouring constant states at the
    #    boundaries.  Evaluated analytically (xi exactly at the boundary).
    a_head = 2.0 * c0 - b["head"]
    a_tail = 2.0 * c0 - b["tail"]
    out["checks"]["fan_head_matches_upstream"] = rel(a_head ** 2 / (9.0 * g), case.h0)
    out["checks"]["fan_tail_matches_plateau"] = rel(a_tail ** 2 / (9.0 * g), b["h2"])
    # 4. Closed-form mass conservation over the computational domain.
    mass_t = exact_mass_over_domain(case, t, case.x_min, case.x_max)
    mass_0 = exact_mass_over_domain(case, 0.0, case.x_min, case.x_max)
    out["checks"]["exact_mass_conservation"] = rel(mass_t, mass_0)
    out["exact_mass"] = mass_0
    out["exact_mass_at_end"] = mass_t
    # 5. Quadrature of the piecewise-polynomial solution over a wide window.
    half = 120.0
    x = np.linspace(-half, half, 1200001)
    h_t, _ = case.exact_state(x, t)
    mass_q = _trapz(h_t, x)
    mass_ref = exact_mass_over_domain(case, t, -half, half)
    out["quadrature_relative"] = rel(mass_q, mass_ref)
    out["checks_relative"] = {name: float(value)
                              for name, value in out["checks"].items()}
    out["worst_relative"] = float(max(out["checks_relative"].values()))
    out["pass"] = bool(out["worst_relative"] <= tol)
    return out
