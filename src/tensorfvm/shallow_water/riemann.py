"""General exact Riemann solver for the 1-D shallow-water equations.

Why this module exists
----------------------
``exact.py`` supplies the *dam-break* family of closed forms: still water on
both sides (``uL = uR = 0``), so the wave pattern is always
**left rarefaction + right bore** (wet bed, Stoker) or **left rarefaction + dry
front** (dry bed, Ritter).  That family was graded in ``dam_break.py``,
``dam_break2d*.py``, ``galilean.py`` and ``crosscheck.py``.  Two of the four
possible shallow-water wave patterns were therefore never exercised by any
analytic reference:

================================  ==========================================
pattern                           where it appears
================================  ==========================================
1-rarefaction + 2-shock           dam break (Stoker / Ritter) -- covered
1-shock + 2-shock                 two streams colliding (``uL > 0 > uR``)
1-shock + 2-rarefaction           stream running into slower/deeper water
1-rarefaction + 2-rarefaction     two streams diverging
================================  ==========================================

The HLL flux's *wave-speed estimates* and the MUSCL reconstruction behave very
differently on a compressive (shock) 1-family than on the expansive
(rarefaction) one, and a defect there is invisible in a dam-break run because
the 1-family is always expansive there.  This module closes that gap with the
general Riemann problem of the shallow-water system, solved in closed form
(self-similar, two constant states joined by two waves), and grades the same
1-D finite-volume solver (``swe1d.ShallowWater1D``) against it with the anchor's
5 % tolerance.

What is *not* transcribed from a table
--------------------------------------
Nothing.  The star state ``(h*, u*)`` is the root of the scalar matching
condition

    f(h*) = f_L(h*) + f_R(h*) + (uR - uL) = 0,

with the branch functions (Toro 2009, ch. 6, in shallow-water variables)

    f_K(h) = 2 (sqrt(g h) - sqrt(g h_K))                  h <= h_K
    f_K(h) = (h - h_K) sqrt(g (h + h_K) / (2 h h_K))       h >  h_K

i.e. the rarefaction Riemann invariant and the shock Rankine-Hugoniot relation
respectively; it is bracketed in ``[0, max(hL, hR)]`` and bisected to machine
precision -- the same technique ``exact._stoker_constants`` uses for the
dam-break sub-case, so nothing about the reference is fitted or copied from a
table.

The reference is then *certified numerically* on four independent paths
(:func:`riemann_residuals`, :func:`weak_form_residuals`,
:func:`smooth_region_ode_residuals`, :func:`degenerate_limits`):

1. **jump / invariant residuals** -- Rankine-Hugoniot conditions re-evaluated
   with shock speeds obtained from the *mass* jump, and rarefaction invariants
   re-evaluated at both fan edges, so a typo in a branch cannot survive;
2. **weak-form conservation identities** evaluated on the sampled solution
   *across* the discontinuities (no jump splitting): a statement about the
   solution as a whole, not about the algebra that built it;
3. **smooth-region similarity-ODE residuals** from finite differences of the
   sampled solution, which catch region-ordering / wrong-state-in-region
   defects -- a class the jump conditions cannot see;
4. **degenerate limits**: ``uL = uR = 0`` must reproduce Stoker's
   ``(h2, u2, U)`` and Ritter's front/dam values from
   :mod:`tensorfvm.shallow_water.exact` to machine precision, so this solver is
   a strict *generalisation* of the published reference, not a replacement.

Known limitation (stated, not hidden)
-------------------------------------
A Riemann problem whose solution contains a **dry middle region** (two
rarefactions pulling the water apart until ``h`` reaches zero: diverging streams
with ``-2(cL + cR) + (uR - uL) >= 0``) is *not* solved here; the constructor
raises ``ValueError`` naming that limit.  Use :func:`exact.dry_riemann_state` to sample a dry middle region
or an initially dry bed, including moving wet states.

References
----------
Toro, E. F. (2009).  *Riemann Solvers and Numerical Methods for Fluid
Dynamics*, 3rd ed., ch. 6 (shallow-water Riemann problem) and ch. 13.
Stoker, J. J. (1957).  *Water Waves*, ch. 10 (the dam-break sub-case).
Ritter, A. (1892).  *Z. Ver. Deutsch. Ing.* **36**, 947 (dry-bed limit).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .energy import _hydraulic_jump_loss, cell_energy
from .exact import (_riemann_cell_means, _similarity_flux_integral, ritter_front_position,
                    ritter_state, stoker_constant_state)

__all__ = [
    "DEFAULT_CASES",
    "EXTREMUM_FRACTION",
    "TOLERANCE",
    "RiemannCase",
    "RiemannSolution",
    "convergence_series",
    "degenerate_limits",
    "profile_metrics",
    "riemann_residuals",
    "run_case",
    "sample_exact",
    "riemann_cell_averages",
    "riemann_integrated_flux",
    "riemann_energy",
    "riemann_energy_budget",
    "riemann_shock_losses",
    "smooth_region_ode_residuals",
    "solve_riemann",
    "verdict",
    "wave_pattern",
    "weak_form_residuals",
    "write_artifacts",
]

#: acceptance tolerance of the anchor (``CASE-FVM-DAMBREAK-STOKER``): 5 %.
TOLERANCE = 0.05

#: relative amplitude (of the reference depth) above which an interior local
#: extremum counts as an oscillation -- the same floor the dam-break benchmark
#: uses (``dam_break.EXTREMUM_FRACTION``), so "no spurious oscillation" means
#: the same thing in both modules.
EXTREMUM_FRACTION = 1.0e-4


# --------------------------------------------------------------------------- #
# self-similar exact solution
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class RiemannSolution:
    """Exact self-similar solution of one shallow-water Riemann problem.

    ``xi = (x - x0) / t`` is the similarity coordinate.  Region ordering is
    ``left state | 1-wave | star state (h*, u*) | 2-wave | right state``; each
    wave is a rarefaction (``*_kind == "rarefaction"``, two edges
    ``*_head`` / ``*_tail``) or a shock (``*_kind == "shock"``, one speed stored
    in ``*_head``; ``*_tail`` equals ``*_head``).
    """

    hL: float
    uL: float
    hR: float
    uR: float
    gravity: float
    x0: float
    h_star: float
    u_star: float
    c_star: float
    left_kind: str
    right_kind: str
    left_head: float
    left_tail: float
    right_head: float
    right_tail: float
    left_shock_speed: float | None
    right_shock_speed: float | None

    @property
    def pattern(self) -> str:
        return f"1-{self.left_kind}+2-{self.right_kind}"

    @property
    def wave_speeds(self) -> tuple[float, ...]:
        """Every distinct wave speed of the solution (1 or 2 per side)."""
        speeds = [self.left_head]
        if self.left_kind == "rarefaction":
            speeds.append(self.left_tail)
        if self.right_kind == "rarefaction":
            speeds.append(self.right_head)
        speeds.append(self.right_tail if self.right_kind == "rarefaction"
                      else self.right_head)
        return tuple(dict.fromkeys(float(s) for s in speeds))

    @property
    def max_wave_speed(self) -> float:
        """Largest ``|wave speed|`` of the problem (CFL / domain bound)."""
        return max(abs(speed) for speed in self.wave_speeds)

    @property
    def span(self) -> tuple[float, float]:
        """Smallest and largest wave speed (the extent of the disturbed region)."""
        speeds = self.wave_speeds
        return min(speeds), max(speeds)

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["pattern"] = self.pattern
        out["wave_speeds"] = list(self.wave_speeds)
        return out


def _branch(h: float, hK: float, gravity: float) -> float:
    """Matching function of one side: rarefaction for ``h <= hK``, shock above."""
    if h <= hK:
        return 2.0 * (math.sqrt(gravity * h) - math.sqrt(gravity * hK))
    return (h - hK) * math.sqrt(gravity * (h + hK) / (2.0 * h * hK))


def solve_riemann(hL: float, uL: float, hR: float, uR: float,
                  gravity: float = 9.81, x0: float = 0.0) -> RiemannSolution:
    """Solve the shallow-water Riemann problem exactly (bisection on ``h*``).

    Raises ``ValueError`` for a non-positive initial depth and for a Riemann
    problem whose solution contains a dry middle region (see the module
    docstring: both are outside what this module claims).
    """
    if not (hL > 0.0 and hR > 0.0):
        raise ValueError(
            "solve_riemann needs hL > 0 and hR > 0 (dry states belong to the "
            f"Ritter dam-break reference in exact.py), got hL={hL}, hR={hR}")
    gravity = float(gravity)
    hL, hR, uL, uR = float(hL), float(hR), float(uL), float(uR)
    cL, cR = math.sqrt(gravity * hL), math.sqrt(gravity * hR)

    def residual(h: float) -> float:
        return _branch(h, hL, gravity) + _branch(h, hR, gravity) + (uR - uL)

    # f is *increasing* in h: f(0) = -2(cL + cR) + (uR - uL) and f -> +inf.
    # A non-negative f(0) therefore means the root is at h <= 0, i.e. the exact
    # solution runs dry in the middle.
    f_zero = -2.0 * (cL + cR) + (uR - uL)
    if f_zero >= 0.0:
        raise ValueError(
            "the solution contains a dry middle region (diverging streams with "
            f"f(0)={f_zero:.6e} >= 0); this solver covers wet Riemann problems "
            "only -- use exact.dry_riemann_state for generated vacuum")

    lo = 0.0
    hi = max(hL, hR)
    for _ in range(200):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover - the matching function grows without bound
        raise RuntimeError("failed to bracket the star depth")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 4.0 * math.ulp(mid):
            break
    h_star = 0.5 * (lo + hi)
    fL = _branch(h_star, hL, gravity)
    fR = _branch(h_star, hR, gravity)
    u_star = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    c_star = math.sqrt(gravity * h_star)

    if h_star > hL:
        left_kind = "shock"
        left_shock: float | None = (h_star * u_star - hL * uL) / (h_star - hL)
        left_head = left_tail = left_shock
    else:
        left_kind = "rarefaction"
        left_shock = None
        left_head, left_tail = uL - cL, u_star - c_star
    if h_star > hR:
        right_kind = "shock"
        right_shock: float | None = (h_star * u_star - hR * uR) / (h_star - hR)
        right_head = right_tail = right_shock
    else:
        right_kind = "rarefaction"
        right_shock = None
        right_head, right_tail = uR + cR, u_star + c_star
    return RiemannSolution(
        hL=hL, uL=uL, hR=hR, uR=uR, gravity=gravity, x0=float(x0),
        h_star=h_star, u_star=u_star, c_star=c_star,
        left_kind=left_kind, right_kind=right_kind,
        left_head=left_head, left_tail=left_tail,
        right_head=right_head, right_tail=right_tail,
        left_shock_speed=left_shock, right_shock_speed=right_shock)


def wave_pattern(hL: float, uL: float, hR: float, uR: float,
                 gravity: float = 9.81) -> str:
    """Pattern label of the Riemann problem, e.g. ``"1-shock+2-shock"``."""
    return solve_riemann(hL, uL, hR, uR, gravity).pattern


def sample_exact(sol: RiemannSolution, x, t: float):
    """Sample the exact solution at positions ``x`` and time ``t`` -> ``(h, u)``.

    For ``t <= 0`` the piecewise-constant initial data is returned, which keeps
    the routine usable for a zero-time plot.
    """
    xs = np.asarray(x, dtype=float)
    g = sol.gravity
    if t <= 0.0:
        left = xs - sol.x0 <= 0.0
        return (np.where(left, sol.hL, sol.hR), np.where(left, sol.uL, sol.uR))
    xi = (xs - sol.x0) / float(t)
    cL, cR = math.sqrt(g * sol.hL), math.sqrt(g * sol.hR)

    # ---- left half-plane: left state, 1-wave, star state -------------------
    if sol.left_kind == "rarefaction":
        c_fan = (sol.uL + 2.0 * cL - xi) / 3.0
        u_fan = xi + c_fan
        h_fan = c_fan * c_fan / g
        in_fan_l = (xi >= sol.left_head) & (xi < sol.left_tail)
    else:
        h_fan = np.zeros_like(xi)
        u_fan = np.zeros_like(xi)
        in_fan_l = np.zeros_like(xi, dtype=bool)
    h_l = np.where(xi < sol.left_head, sol.hL,
                   np.where(in_fan_l, h_fan, sol.h_star))
    u_l = np.where(xi < sol.left_head, sol.uL,
                   np.where(in_fan_l, u_fan, sol.u_star))

    # ---- right half-plane: star state, 2-wave, right state -----------------
    if sol.right_kind == "rarefaction":
        c_fan_r = (xi - sol.uR + 2.0 * cR) / 3.0
        u_fan_r = xi - c_fan_r
        h_fan_r = c_fan_r * c_fan_r / g
        in_fan_r = (xi > sol.right_tail) & (xi <= sol.right_head)
    else:
        h_fan_r = np.zeros_like(xi)
        u_fan_r = np.zeros_like(xi)
        in_fan_r = np.zeros_like(xi, dtype=bool)
    h_r = np.where(xi > sol.right_head, sol.hR,
                   np.where(in_fan_r, h_fan_r, sol.h_star))
    u_r = np.where(xi > sol.right_head, sol.uR,
                   np.where(in_fan_r, u_fan_r, sol.u_star))

    # The star-state velocity lies between the two waves in every inertial frame.
    return np.where(xi <= sol.u_star, h_l, h_r), np.where(xi <= sol.u_star, u_l, u_r)


def riemann_energy(sol: RiemannSolution, x_min: float, x_max: float, t: float):
    """Integrate exact wet-solution mechanical energy per width on a finite interval."""
    if not all(map(math.isfinite, (x_min, x_max, t))) or x_min >= x_max or t < 0:
        raise ValueError("require a finite increasing interval and nonnegative finite time")
    fronts = {sol.x0+t*speed for speed in sol.wave_speeds}
    edges = np.array([x_min, *sorted(x for x in fronts if x_min < x < x_max), x_max])
    half = np.diff(edges)/2
    # Mechanical energy density is quartic inside each rarefaction.
    nodes = np.array([-math.sqrt(.6), 0., math.sqrt(.6)])
    weights = np.array([5/9, 8/9, 5/9])
    h, u = sample_exact(sol, edges[:-1,None]+half[:,None]*(1+nodes), t)
    return float(half @ (cell_energy(h,h*u,sol.gravity)["total"] @ weights))


def riemann_shock_losses(sol: RiemannSolution):
    """Return mechanical-energy loss per time and width on each wave.

    A rarefaction contributes zero. Each shock contributes its rate while
    inside the chosen control volume, regardless of its propagation direction.
    The returned rates describe the waves on the full line.
    """
    losses = {}
    for side, h, u, speed in (("left", sol.hL, sol.uL, sol.left_shock_speed),
                              ("right", sol.hR, sol.uR, sol.right_shock_speed)):
        losses[side] = (0. if speed is None else
                        _hydraulic_jump_loss(sol.gravity, h*(u-speed), h, sol.h_star))
    return losses


def _face_energy_transport(sol, x, t):
    """Rightward energy through a face, taking its left trace at a stationary shock."""
    offset = x-sol.x0
    times = {0., t}
    for speed in sol.wave_speeds:
        if speed and min(0., t*speed) < offset < max(0., t*speed):
            times.add(offset/speed)
    ends = sorted(times)
    total = 0.
    for start, stop in zip(ends, ends[1:]):
        middle = .5*(start+stop)
        xi = offset/middle
        in_fan = (sol.left_head < xi < sol.left_tail or sol.right_tail < xi < sol.right_head)
        if in_fan and offset != 0:
            # In a smooth fan, d/dt [t*F - (x-x0)*E] = F. A fixed face
            # away from the initial dam enters the fan at a positive time.
            times_pair = np.array([start, stop])
            xi_pair = offset/times_pair
            if sol.left_head < xi < sol.left_tail:
                c = (sol.uL+2*math.sqrt(sol.gravity*sol.hL)-xi_pair)/3
                u = xi_pair+c
            else:
                c = (xi_pair-sol.uR+2*math.sqrt(sol.gravity*sol.hR))/3
                u = xi_pair-c
            h = c*c/sol.gravity
            e = cell_energy(h,h*u,sol.gravity)["total"]
            primitive = times_pair*u*(e+.5*sol.gravity*h*h)-offset*e
            total += primitive[1]-primitive[0]
        else:
            h, u = sample_exact(sol, x, middle)
            if offset == 0 and sol.left_shock_speed == 0:
                h, u = sol.hL, sol.uL
            elif offset == 0 and sol.right_shock_speed == 0:
                h, u = sol.h_star, sol.u_star
            total += (stop-start)*u*h*(.5*u*u+sol.gravity*h)
    return float(total)


def riemann_energy_budget(sol: RiemannSolution, x_min: float, x_max: float, t: float):
    """Exact cumulative energy balance per unit width over [x_min, x_max).

    Fluxes are positive rightwards. Shock losses accrue only during residence
    in the interval. A stationary shock on a boundary belongs to the interval
    on its right, and both face fluxes use the left trace.
    """
    final = riemann_energy(sol, x_min, x_max, t)
    initial = riemann_energy(sol, x_min, x_max, 0.)
    west = _face_energy_transport(sol, x_min, t)
    east = _face_energy_transport(sol, x_max, t)
    rates = riemann_shock_losses(sol)
    losses = {}
    for side, speed in (("left", sol.left_shock_speed), ("right", sol.right_shock_speed)):
        duration = 0.
        if speed == 0:
            duration = t if x_min <= sol.x0 < x_max else 0.
        elif speed is not None:
            entry, exit_time = sorted(((x_min-sol.x0)/speed, (x_max-sol.x0)/speed))
            duration = max(0., min(t, exit_time)-max(0., entry))
        losses[side] = duration*rates[side]
    return {"initial": initial, "final": final, "left_transport": west,
            "right_transport": east, "shock_losses": losses,
            "residual": final-initial-west+east+sum(losses.values())}


def riemann_integrated_flux(sol: RiemannSolution, x, t: float):
    """Time integrals of rightward mass and momentum flux at fixed positions.

    For a self-similar conservation law, the time derivative of
    t*F(U) - (x-x0)*U equals F(U). This primitive is continuous at a
    moving shock by Rankine-Hugoniot. Subtract its initial value to obtain
    the flux since time zero, including any wave crossings at the face.
    """
    if not math.isfinite(t) or t < 0:
        raise ValueError("time must be finite and nonnegative")
    x = np.asarray(x, dtype=float)
    h0, u0 = sample_exact(sol, x, 0.)
    h, u = sample_exact(sol, x, t)
    return _similarity_flux_integral(x-sol.x0, t, sol.gravity, h0, u0, h, u)


def riemann_cell_averages(sol: RiemannSolution, edges, t: float):
    """Return control-volume means of h and hu for a wet Riemann solution.

    Split nonuniform cells at every shock and rarefaction edge before exact
    polynomial quadrature. At zero time split the initial jump at sol.x0.
    """
    if not math.isfinite(t) or t < 0:
        raise ValueError("time must be finite and nonnegative")
    relative = dataclasses.replace(sol, x0=0.)
    return _riemann_cell_means(edges, t, sol.x0, sol.wave_speeds,
                               lambda offset: sample_exact(relative, offset, t),
                               sol.hL, sol.uL, sol.hR, sol.uR)


# --------------------------------------------------------------------------- #
# certification of the exact reference (paths 1-3)
# --------------------------------------------------------------------------- #
def riemann_residuals(sol: RiemannSolution) -> dict:
    """Jump / invariant residuals -- certification path 1.

    Every number is *relative* (scaled by the local mass or momentum flux), so
    "machine precision" is a meaningful statement for every state.
    """
    g = sol.gravity
    cL, cR = math.sqrt(g * sol.hL), math.sqrt(g * sol.hR)
    fL = _branch(sol.h_star, sol.hL, g)
    fR = _branch(sol.h_star, sol.hR, g)
    match = fL + fR + (sol.uR - sol.uL)
    out: dict = {
        "matching_residual": match,
        "matching_relative": abs(match) / (2.0 * (cL + cR)),
    }

    def _side(name: str, hK: float, uK: float, kind: str,
              speed: float | None) -> None:
        h_s, u_s = sol.h_star, sol.u_star
        if kind == "shock":
            mass = speed * (h_s - hK) - (h_s * u_s - hK * uK)
            mom = (speed * (h_s * u_s - hK * uK)
                   - ((h_s * u_s * u_s + 0.5 * g * h_s * h_s)
                      - (hK * uK * uK + 0.5 * g * hK * hK)))
            scale_m = max(abs(speed * (h_s - hK)), 1e-300)
            scale_p = max(abs(speed * h_s * u_s), 1e-300)
            out[f"{name}_kind"] = "shock"
            out[f"{name}_mass_jump_residual"] = mass
            out[f"{name}_mass_jump_relative"] = abs(mass) / scale_m
            out[f"{name}_momentum_jump_residual"] = mom
            out[f"{name}_momentum_jump_relative"] = abs(mom) / scale_p
        else:
            sign = +1.0 if name == "left" else -1.0
            invariant = ((u_s + sign * 2.0 * sol.c_star)
                         - (uK + sign * 2.0 * math.sqrt(g * hK)))
            wave_speed = ((u_s - sol.c_star) if name == "left"
                          else (u_s + sol.c_star))
            edge = (sol.left_head if name == "left" else sol.right_head)
            out[f"{name}_kind"] = "rarefaction"
            out[f"{name}_riemann_invariant_residual"] = invariant
            out[f"{name}_riemann_invariant_relative"] = (
                abs(invariant) / (2.0 * math.sqrt(g * hK)))
            # the fan must start at the acoustic speed of the outer state
            out[f"{name}_fan_edge_residual"] = abs(wave_speed - edge)

    _side("left", sol.hL, sol.uL, sol.left_kind, sol.left_shock_speed)
    _side("right", sol.hR, sol.uR, sol.right_kind, sol.right_shock_speed)
    # entropy: between the two waves the flow may be transcritical; the star
    # state of a rarefaction must be *sub*-acoustic relative to its own wave
    out["star_depth_positive"] = float(sol.h_star > 0.0)
    return out


def weak_form_residuals(sol: RiemannSolution, n: int = 200_001,
                        pad: float = 0.05) -> dict:
    """Weak-form conservation identities of the sampled solution (path 2).

    For a self-similar solution the shallow-water equations integrate to

        [h u - xi h] + int h dxi = 0,
        [h u^2 + g h^2 / 2 - xi h u] + int h u dxi = 0,

    over any interval that encloses the whole disturbed region.  Both identities
    hold *across* the discontinuities (they are the global statement whose local
    form is the Rankine-Hugoniot condition), so unlike the jump residuals they
    test the assembled solution rather than the algebra that built it.
    """
    lo, hi = sol.span
    width = max(hi - lo, 1e-12)
    x1, x2 = lo - pad * width, hi + pad * width
    xi = np.linspace(x1, x2, int(n))
    h, u = sample_exact(sol, sol.x0 + xi, 1.0)   # t = 1 -> x = x0 + xi
    hu = h * u
    mass = np.trapezoid(h, xi)
    momentum = np.trapezoid(hu, xi)
    flux_m = hu - xi * h
    flux_p = hu * u + 0.5 * sol.gravity * h * h - xi * hu
    scale_m = max(abs(flux_m[-1]), abs(flux_m[0]), abs(mass), 1e-300)
    scale_p = max(abs(flux_p[-1]), abs(flux_p[0]), abs(momentum), 1e-300)
    res_m = (flux_m[-1] - flux_m[0]) + mass
    res_p = (flux_p[-1] - flux_p[0]) + momentum
    return {
        "mass_residual": float(res_m),
        "mass_relative": float(abs(res_m) / scale_m),
        "momentum_residual": float(res_p),
        "momentum_relative": float(abs(res_p) / scale_p),
        "interval": [float(x1), float(x2)],
        "samples": int(n),
    }


def smooth_region_ode_residuals(sol: RiemannSolution, n: int = 20_001,
                                skip: float = 0.02) -> dict:
    """Similarity-ODE residuals on the smooth regions of the sampled solution.

    In smooth regions ``-xi h' + (h u)' = 0`` and
    ``-xi (h u)' + (h u^2 + g h^2/2)' = 0``, with ``'`` = ``d/dxi``.  The
    residuals are evaluated by central differences of the *sampled* solution
    (so a wrong state in a region, or a misplaced region boundary, shows up
    here even though every jump condition is satisfied), skipping a band
    ``skip * span`` around each wave speed where the sample is discontinuous.
    """
    lo, hi = sol.span
    width = max(hi - lo, 1e-12)
    pad = 0.05 * width
    xi = np.linspace(lo - pad, hi + pad, int(n))
    h, u = sample_exact(sol, sol.x0 + xi, 1.0)
    dxi = xi[1] - xi[0]
    mask = np.ones_like(xi, dtype=bool)
    for speed in sol.wave_speeds:
        mask &= np.abs(xi - speed) > skip * width
    mask[:2] = mask[-2:] = False
    hx = np.gradient(h, dxi)
    ux = np.gradient(u, dxi)
    m = h * u
    mx = np.gradient(m, dxi)
    fx = np.gradient(m * u + 0.5 * sol.gravity * h * h, dxi)
    r_mass = -xi[mask] * hx[mask] + mx[mask]
    r_mom = -xi[mask] * mx[mask] + fx[mask]
    scale_m = max(sol.gravity * sol.h_star ** 1.5, 1e-300)
    scale_p = max(sol.gravity * sol.h_star ** 2, 1e-300) / max(width, 1e-300)
    return {
        "mass_ode_max_abs": float(np.max(np.abs(r_mass))) if r_mass.size else 0.0,
        "mass_ode_relative": float(np.max(np.abs(r_mass)) / scale_m) if r_mass.size else 0.0,
        "momentum_ode_max_abs": float(np.max(np.abs(r_mom))) if r_mom.size else 0.0,
        "momentum_ode_relative": float(np.max(np.abs(r_mom)) / scale_p) if r_mom.size else 0.0,
        "masked_samples": int(mask.sum()),
        "samples": int(n),
    }


def degenerate_limits(gravity: float = 9.81) -> dict:
    """Path 4: the dam-break sub-cases must be reproduced to machine precision."""
    out: dict = {}
    for h1 in (0.5, 0.25, 0.8):
        sol = solve_riemann(1.0, 0.0, h1, 0.0, gravity)
        ref = stoker_constant_state(1.0, h1, gravity)
        out[f"stoker_h1_{h1}"] = {
            "pattern": sol.pattern,
            "h_star_relative_error": abs(sol.h_star - ref["h2"]) / ref["h2"],
            "u_star_absolute_error": abs(sol.u_star - ref["u2"]),
            "shock_speed_absolute_error": abs(
                sol.right_shock_speed - ref["shock_speed"]),
            "left_fan_head_absolute_error": abs(sol.left_head - ref["head_speed"]),
        }
    # dry-bed limit: hR -> 0 with uL = uR = 0 is Ritter; the bore speed and the
    # dam depth must approach 2 c0 and 4 h0 / 9 (the Ritter values) as h1 -> 0.
    small = 1e-10
    sol = solve_riemann(1.0, 0.0, small, 0.0, gravity)
    dam = float(np.asarray(sample_exact(sol, np.array([0.0]), 1.0)[0])[0])
    out["ritter_limit"] = {
        "h1": small,
        "front_speed_relative_error": abs(
            sol.right_shock_speed - 2.0 * math.sqrt(gravity)) / (
                2.0 * math.sqrt(gravity)),
        "dam_depth_relative_error": abs(dam - 4.0 / 9.0) / (4.0 / 9.0),
        "ritter_state_dam_depth": float(
            ritter_state(np.array([0.0]), 1.0, 1.0, gravity)[0][0]),
        "ritter_front_position": float(
            ritter_front_position(1.0, 1.0, gravity)),
    }
    return out


# --------------------------------------------------------------------------- #
# benchmark cases and their evaluation
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class RiemannCase:
    """One graded Riemann problem: initial states, domain, end time."""

    name: str
    hL: float
    uL: float
    hR: float
    uR: float
    x_min: float = -12.0
    x_max: float = 12.0
    t_end: float = 1.0
    gravity: float = 9.81
    note: str = ""
    x0: float = 0.0

    @property
    def h_ref(self) -> float:
        return max(self.hL, self.hR)

    @property
    def c_ref(self) -> float:
        return math.sqrt(self.gravity * self.h_ref)

    def solution(self) -> RiemannSolution:
        return solve_riemann(self.hL, self.uL, self.hR, self.uR,
                             self.gravity, x0=self.x0)

    def initial_state(self, x) -> tuple[np.ndarray, np.ndarray]:
        xs = np.asarray(x, dtype=float)
        left = xs <= self.x0
        return (np.where(left, self.hL, self.hR), np.where(left, self.uL, self.uR))

    def initial_cell_state(self, edges):
        """Primitive variables from cell-averaged initial mass and momentum."""
        h, hu = riemann_cell_averages(self.solution(), edges, 0.)
        return h, hu/h

    def exact(self, x, t: float):
        return sample_exact(self.solution(), x, t)

    def summary(self) -> dict:
        sol = self.solution()
        return {
            "name": self.name,
            "hL": self.hL, "uL": self.uL, "hR": self.hR, "uR": self.uR,
            "gravity": self.gravity, "x_min": self.x_min, "x_max": self.x_max,
            "x0": self.x0, "t_end": self.t_end, "note": self.note,
            "pattern": sol.pattern,
            "h_star": sol.h_star, "u_star": sol.u_star,
            "wave_speeds": list(sol.wave_speeds),
            "max_wave_speed": sol.max_wave_speed,
        }


#: the four wave patterns, one graded case each.  Velocities are chosen so that
#: every wave stays inside the domain for ``t <= t_end`` (``|S| t_end < 10 m``).
DEFAULT_CASES = (
    RiemannCase(
        "rarefaction-shock", hL=1.0, uL=0.0, hR=0.5, uR=0.0, t_end=1.0,
        note="dam-break control (Stoker sub-case): the only pattern the "
             "existing dam-break evidence graded"),
    RiemannCase(
        "rarefaction-rarefaction", hL=1.0, uL=-1.0, hR=0.5, uR=1.0, t_end=1.0,
        note="diverging streams: both families expansive, the star depth is a "
             "smooth minimum (no wave to hide behind)"),
    RiemannCase(
        "shock-shock", hL=1.0, uL=1.5, hR=0.6, uR=-1.0, t_end=1.0,
        note="colliding streams: both families compressive -- the pattern the "
             "HLL wave-speed estimates are least tested on"),
    RiemannCase(
        "shock-rarefaction", hL=0.5, uL=3.0, hR=1.0, uR=2.5, t_end=1.0,
        note="fast shallow stream overtaking deeper faster water: compressive "
             "1-family, expansive 2-family"),
)


def _local_extrema(h: np.ndarray, floor: float) -> tuple[int, float]:
    """Count interior extrema of amplitude above ``floor``.

    The predicate is *identical* to ``dam_break._extrema_mask`` (a cell is
    counted iff both adjacent slopes exceed ``floor`` in magnitude and change
    sign), so "no spurious oscillation" is measured the same way here and in the
    dam-break benchmark; ``tests/test_shallow_water_riemann.py`` asserts the two
    implementations agree cell by cell.
    """
    h = np.asarray(h, dtype=float)
    if h.size < 3:
        return 0, 0.0
    left = h[1:-1] - h[:-2]
    right = h[2:] - h[1:-1]
    peak = (left > 0.0) & (right < 0.0) & (np.abs(left) > floor) & (np.abs(right) > floor)
    dip = (left < 0.0) & (right > 0.0) & (np.abs(left) > floor) & (np.abs(right) > floor)
    mask = peak | dip
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0, 0.0
    amplitude = float(np.max(np.minimum(np.abs(left), np.abs(right))[mask]))
    return count, amplitude


def _crossings(x: np.ndarray, h: np.ndarray, level: float) -> np.ndarray:
    """All x where ``h`` crosses ``level`` (linear interpolation, sorted)."""
    above = h >= level
    edges = np.nonzero(np.diff(above.astype(np.int8)))[0]
    out = []
    for i in edges:
        h0, h1 = h[i], h[i + 1]
        if h0 == h1:
            out.append(float(x[i]))
        else:
            out.append(float(x[i] + (level - h0) / (h1 - h0) * (x[i + 1] - x[i])))
    return np.asarray(out, dtype=float)


def _wave_crossing(x: np.ndarray, h: np.ndarray, level: float,
                   side: str) -> tuple[float, int]:
    """Position of one wave read from a level crossing; ``(x, n_crossings)``.

    *Convention, declared before any run*: the 1-wave is the **leftmost**
    crossing of its level and the 2-wave the **rightmost** one.  A level chosen
    inside one wave's jump can legitimately be crossed by the *other* wave too
    (two-rarefaction and two-shock solutions both do this), so "the unique
    crossing" is not available in general; picking the outer one is the
    geometry-free way to isolate the wave that owns the side.  The crossing
    count is returned with the position so a report can show that the
    measurement is what it claims to be, and ``nan`` is returned (failing the
    gate) when the level is never crossed at all.
    """
    found = _crossings(x, h, level)
    if found.size == 0:
        return float("nan"), 0
    return float(found[0] if side == "left" else found[-1]), int(found.size)


def profile_metrics(case: RiemannCase, x, h, u, t: float) -> dict:
    """Grade a numerical profile against the exact solution of ``case``.

    Metrics (declared before any run):

    ``depth_l1`` / ``velocity_l1``  mean absolute error normalised by ``h_ref``
                                    and ``c_ref`` respectively;
    ``wave_*_error``                relative error of every wave position read
                                    from a *data-driven* level crossing at half
                                    the exact state jump across that wave (for
                                    a rarefaction: at the fan head, a fixed
                                    fraction of the fan depth drop);
    ``plateau_*_error``             relative error of the star depth/velocity
                                    sampled at the centre of the star region;
    ``extremum_count``              interior local extrema above
                                    ``EXTREMUM_FRACTION * h_ref`` (must be 0:
                                    the acceptance's "no spurious oscillation").
    """
    sol = case.solution()
    xs = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    u = np.asarray(u, dtype=float)
    h_ex, u_ex = sample_exact(sol, xs, t)
    out: dict = {
        "depth_l1": float(np.mean(np.abs(h - h_ex)) / case.h_ref),
        "velocity_l1": float(np.mean(np.abs(u - u_ex)) / case.c_ref),
        "depth_linf": float(np.max(np.abs(h - h_ex)) / case.h_ref),
        "h_max_over_href": float(np.max(h) / case.h_ref),
        "h_min": float(np.min(h)),
    }
    count, amp = _local_extrema(h, EXTREMUM_FRACTION * case.h_ref)
    out["extremum_count"] = count
    out["extremum_amplitude"] = amp

    g = case.gravity
    for side in ("left", "right"):
        kind = sol.left_kind if side == "left" else sol.right_kind
        h_out = sol.hL if side == "left" else sol.hR
        if kind == "shock":
            level = 0.5 * (h_out + sol.h_star)
            speed = sol.left_head if side == "left" else sol.right_head
        else:
            # Track the level one quarter of the way from the outer depth to
            # the star depth. Its similarity speed follows from the invariant.
            level = h_out - 0.25 * (h_out - sol.h_star)
            c_out = math.sqrt(g * h_out)
            c_p = math.sqrt(g * level)
            speed = (sol.uL + 2*c_out - 3*c_p if side == "left" else
                     sol.uR - 2*c_out + 3*c_p)
        x_meas, n_cross = _wave_crossing(xs, h, level, side)
        displacement = speed * t
        entry = {"kind": kind, "level": level, "x_exact": sol.x0 + displacement,
                 "x_measured": x_meas, "crossings": n_cross}
        scale = abs(displacement)
        # A stationary wave has no travel distance. Use the acoustic travel
        # distance for that limit and record the denominator with the error.
        if scale <= 16*np.finfo(float).eps*case.c_ref*abs(t):
            scale = case.c_ref*abs(t)
        entry["position_error_scale"] = scale
        entry["rel_error"] = (abs(entry["x_measured"]-entry["x_exact"])/scale
                              if scale > 0 else float("nan"))
        out[f"wave_{side}"] = entry

    # star plateau: sample the exact star region between the two waves
    left_edge = sol.x0 + sol.left_tail*t
    right_edge = sol.x0 + sol.right_tail*t
    x_star = .5*(left_edge + right_edge)
    idx = int(np.argmin(np.abs(xs - x_star)))
    resolved = left_edge < xs[idx] < right_edge
    out["plateau"] = {
        "x": float(xs[idx]), "resolved": bool(resolved),
        "depth_measured": float(h[idx]),
        "depth_exact": float(sol.h_star),
        "depth_rel_error": float(abs(h[idx] - sol.h_star) / sol.h_star) if resolved else float("nan"),
        "velocity_measured": float(u[idx]),
        "velocity_exact": float(sol.u_star),
        "velocity_rel_error": (float(abs(u[idx]-sol.u_star) / (abs(sol.u_star) or 1.))
                               if resolved else float("nan")),
    }
    return out


def verdict(metrics: dict, tol: float = TOLERANCE) -> dict:
    """Apply the anchor tolerance; every check is named so a report can cite it."""
    checks = {
        "depth_l1": metrics["depth_l1"],
        "wave_left": metrics["wave_left"]["rel_error"],
        "wave_right": metrics["wave_right"]["rel_error"],
        "plateau_depth": metrics["plateau"]["depth_rel_error"],
        "plateau_velocity": metrics["plateau"]["velocity_rel_error"],
    }
    finite = {name: (float(value) if value is not None and np.isfinite(value)
                     else None) for name, value in checks.items()}
    failed = [name for name, value in finite.items()
              if value is None or value > tol]
    monotone = metrics["extremum_count"] == 0
    return {"tolerance": tol, "checks": finite, "failed": failed,
            "monotone": monotone,
            "pass": (not failed) and monotone}


def run_case(case: RiemannCase, n: int = 1600, cfl: float = 0.4,
             order: int = 2, tol: float = TOLERANCE) -> dict:
    """Integrate one Riemann case and return its graded record (JSON-safe)."""
    from .swe1d import ShallowWater1D

    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=cfl, order=order)
    edges = np.linspace(case.x_min, case.x_max, n+1)
    h0, u0 = case.initial_cell_state(edges)
    solver.set_state(h0, u0)
    volume0 = solver.total_volume()
    momentum0 = solver.momentum()
    steps = solver.advance(case.t_end)
    h, u = solver.primitive()
    metrics = profile_metrics(case, solver.x, h, u, solver.time)
    sol = case.solution()
    h_mean, hu_mean = riemann_cell_averages(sol, edges, solver.time)
    metrics["cell_mean_depth_l1"] = float(np.mean(np.abs(h-h_mean)) / case.h_ref)
    metrics["cell_mean_discharge_l1"] = float(np.mean(np.abs(h*u-hu_mean)) / (case.h_ref*case.c_ref))
    metrics["energy_budget"] = riemann_energy_budget(sol, case.x_min, case.x_max, solver.time)
    energy_exact = metrics["energy_budget"]["final"]
    metrics["energy_exact"] = energy_exact
    metrics["energy_numerical"] = solver.kinetic_energy()
    metrics["energy_relative_error"] = abs(metrics["energy_numerical"]-energy_exact)/energy_exact
    metrics["shock_energy_loss_rates"] = riemann_shock_losses(sol)
    metrics["volume_initial"] = float(volume0)
    metrics["volume_drift"] = (float(abs(solver.total_volume() - volume0)
                                     / volume0) if volume0 > 0 else float("nan"))
    # The analytic boundary flux includes waves passing either domain end.
    mass_flux, impulse = riemann_integrated_flux(sol, edges[[0, -1]], solver.time)
    expected_mass = mass_flux[0]-mass_flux[1]
    expected_momentum = impulse[0]-impulse[1]
    volume_change = solver.total_volume()-volume0
    momentum_change = solver.momentum()-momentum0
    metrics["volume_change"] = float(volume_change)
    metrics["volume_change_expected"] = float(expected_mass)
    metrics["volume_budget_relative"] = float(abs(volume_change-expected_mass)/volume0)
    metrics["momentum_change"] = float(momentum_change)
    metrics["momentum_change_expected"] = float(expected_momentum)
    # Initial momentum may be zero for a dam break at rest.
    metrics["momentum_budget_relative"] = float(abs(momentum_change-expected_momentum)/(volume0*case.c_ref))
    metrics["momentum_initial"] = float(momentum0)
    metrics["momentum_drift"] = (float(abs(solver.momentum() - momentum0)
                                       / abs(momentum0))
                                 if momentum0 != 0.0 else float("nan"))
    return {
        "case": case.summary(),
        "discretisation": {"n": n, "dx": solver.dx, "cfl": cfl, "order": order,
                           "steps": int(steps), "t_end": float(solver.time),
                           "clipped_cells": int(solver.clipped_cells),
                           "dry_momentum_flips": int(solver.dry_flips)},
        "metrics": metrics,
        "verdict": verdict(metrics, tol),
    }


def convergence_series(case: RiemannCase, resolutions=(200, 400, 800, 1600),
                       cfl: float = 0.4, order: int = 2,
                       tol: float = TOLERANCE) -> dict:
    """Grid-refinement series of the graded record plus observed orders (if any)."""
    rows = []
    for n in resolutions:
        record = run_case(case, n=n, cfl=cfl, order=order, tol=tol)
        metrics = record["metrics"]
        rows.append({
            "n": n,
            "dx": record["discretisation"]["dx"],
            "steps": record["discretisation"]["steps"],
            "depth_l1": metrics["depth_l1"],
            "cell_mean_depth_l1": metrics["cell_mean_depth_l1"],
            "cell_mean_discharge_l1": metrics["cell_mean_discharge_l1"],
            "velocity_l1": metrics["velocity_l1"],
            "wave_left": metrics["wave_left"]["rel_error"],
            "wave_right": metrics["wave_right"]["rel_error"],
            "plateau_depth": metrics["plateau"]["depth_rel_error"],
            "plateau_velocity": metrics["plateau"]["velocity_rel_error"],
            "extremum_count": metrics["extremum_count"],
            "volume_drift": metrics["volume_drift"],
            "energy_relative_error": metrics["energy_relative_error"],
            "volume_budget_relative": metrics["volume_budget_relative"],
            "momentum_budget_relative": metrics["momentum_budget_relative"],
            "pass": record["verdict"]["pass"],
        })
    l1 = [row["depth_l1"] for row in rows]
    dx = [row["dx"] for row in rows]
    orders = [math.log(l1[i] / l1[i + 1]) / math.log(dx[i] / dx[i + 1])
              for i in range(len(l1) - 1)]
    return {
        "case": case.summary(),
        "rows": rows,
        "observed_order_depth_l1": orders,
        "all_pass": all(row["pass"] for row in rows),
    }

# --------------------------------------------------------------------------- #
# artefacts (JSON + figure), so no number in the report is typed by hand
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    """Replace non-finite floats by ``None`` (strict JSON has no NaN/Infinity)."""
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def certification(resolutions: tuple[int, ...] = (10_001, 20_001, 40_001,
                                                  80_001)) -> dict:
    """Reference-side certification record (paths 1-4) for every default case.

    The weak-form integral is refined as well: its residual is first-order in the
    step across a jump, so a single number would say nothing about whether the
    identity is satisfied or merely un-converged.
    """
    out: dict = {"cases": {}, "degenerate_limits": degenerate_limits()}
    for case in DEFAULT_CASES:
        sol = case.solution()
        entry = {
            "pattern": sol.pattern,
            "h_star": sol.h_star,
            "u_star": sol.u_star,
            "wave_speeds": list(sol.wave_speeds),
            "jump_residuals": riemann_residuals(sol),
            "smooth_ode": smooth_region_ode_residuals(sol),
            "weak_form_refinement": [
                dict(n=n, **weak_form_residuals(sol, n=n)) for n in resolutions],
        }
        out["cases"][case.name] = entry
    return out


def write_artifacts(out_dir, resolutions=(400, 800, 1600, 3200),
                    accepted_n: int = 3200, cfl: float = 0.4,
                    order: int = 2, tol: float = TOLERANCE) -> list:
    """Write ``metrics.json``, ``convergence.json``, ``certification.json`` and
    the figure into ``out_dir``; return the list of written paths."""
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list = []

    metrics = {"tolerance": tol, "accepted_n": accepted_n, "cfl": cfl,
               "order": order, "cases": {}}
    series = {"cases": {}}
    for case in DEFAULT_CASES:
        record = run_case(case, n=accepted_n, cfl=cfl, order=order, tol=tol)
        metrics["cases"][case.name] = record
        series["cases"][case.name] = convergence_series(
            case, resolutions=resolutions, cfl=cfl, order=order, tol=tol)

    cert = certification()

    # ---- figure: profiles at the accepted grid + convergence -----------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for ax, case in zip(axes.ravel(), DEFAULT_CASES):
        record = metrics["cases"][case.name]
        n = record["discretisation"]["n"]
        from .swe1d import ShallowWater1D

        solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                                cfl=cfl, order=order)
        h0, u0 = case.initial_cell_state(np.linspace(case.x_min, case.x_max, n+1))
        solver.set_state(h0, u0)
        solver.advance(case.t_end)
        h, _ = solver.primitive()
        x_ex = np.linspace(case.x_min, case.x_max, 4001)
        h_ex, _ = case.exact(x_ex, solver.time)
        ax.plot(x_ex, h_ex, "-", color="k", lw=1.6, label="exact (self-similar)")
        ax.plot(solver.x, h, "--", color="tab:blue", lw=1.0,
                label=f"FVM n={n}")
        for speed in case.solution().wave_speeds:
            ax.axvline(case.x0 + speed * solver.time, color="tab:red", lw=0.6, alpha=0.5)
        ax.set_title(f"{case.name}: {case.solution().pattern}", fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("h [m]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Shallow-water Riemann problem: exact self-similar solution vs "
                 "HLL+MUSCL FV solver (red: exact wave speeds)", fontsize=10)
    fig.tight_layout()
    fig_path = out / "riemann_profiles.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    written.append(fig_path)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for case in DEFAULT_CASES:
        rows = series["cases"][case.name]["rows"]
        dx = [row["dx"] for row in rows]
        axes[0].loglog(dx, [row["depth_l1"] for row in rows], "o-",
                       label=case.name, ms=3.5)
        axes[1].loglog(dx, [row["wave_right"] for row in rows], "o-",
                       label=case.name, ms=3.5)
        axes[1].loglog(dx, [row["wave_left"] for row in rows], "s--",
                       label=f"{case.name} (1-wave)", ms=3.0, alpha=0.6)
    axes[0].set_xlabel(r"$\Delta x$ [m]")
    axes[0].set_ylabel("relative depth L1 error")
    axes[0].set_title("depth L1 error vs grid", fontsize=9)
    axes[1].set_xlabel(r"$\Delta x$ [m]")
    axes[1].set_ylabel("relative wave-position error")
    axes[1].set_title("wave positions vs grid (5 % acceptance)", fontsize=9)
    axes[1].axhline(tol, color="k", ls=":", lw=1.0, label="tolerance 5 %")
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig_path = out / "riemann_convergence.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    written.append(fig_path)

    for name, payload in (("metrics.json", metrics),
                          ("convergence.json", series),
                          ("certification.json", cert)):
        path = out / name
        path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
        written.append(path)
    return written
