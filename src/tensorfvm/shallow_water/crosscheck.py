"""Independent exact Riemann solver for the 1-D shallow-water equations.

The dam-break benchmark (``docs/case-fvm-dambreak-stoker``) compares a finite
volume solution against the closed-form Ritter (dry bed) and Stoker (wet bed)
solutions in :mod:`tensorfvm.shallow_water.exact`.  Those closed forms are
*derived* from an assumed wave pattern (left rarefaction + constant state +
right bore) and are only self-checked through their own Rankine-Hugoniot
residuals, so a shared algebraic mistake would be invisible to that check.

This module provides a second, deliberately independent reference: a generic
exact Riemann solver for

    h_t + (h u)_x = 0,     (h u)_t + (h u^2 + g h^2 / 2)_x = 0

built from the textbook function-iteration approach (Toro, *Riemann Solvers and
Numerical Methods for Fluid Dynamics*, 2nd ed., ch. 13.3 "The Riemann problem
for the shallow water equations"):

* the star depth ``h*`` is the root of ``Phi(h) = f_L(h) + f_R(h) + u_R - u_L``,
  where ``f_K`` is the exact pressure-function integral for a shock
  (``h > h_K``) or a rarefaction (``h <= h_K``);
* the star velocity follows from ``u* = (u_L + u_R)/2 + (f_R(h*) - f_L(h*))/2``;
* discontinuity speeds are obtained from the *mass* Rankine-Hugoniot relation
  ``S = (h* u* - h_K u_K) / (h* - h_K)``, i.e. from the states rather than from
  a family-specific closed formula.

Nothing here imports :mod:`tensorfvm.shallow_water.exact`, so the two code paths
can only agree if both are right.  The solver is generic (``u_L`` and ``u_R``
are free), handles wet and dry beds, and is used by
``tests/test_shallow_water_analytic_crosscheck.py``.

References
----------
Toro, E. F. (2009).  *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Springer, ch. 13.  (shallow-water Riemann solver, pressure function).
Stoker, J. J. (1957).  *Water Waves*, Interscience, ch. 10.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "pressure_function",
    "riemann_star_state",
    "riemann_sample",
    "riemann_solution",
    "jump_trajectory_speed",
    "level_trajectory_speed",
]


# --------------------------------------------------------------------------- #
# pressure function
# --------------------------------------------------------------------------- #
def pressure_function(h, h_side: float, gravity: float = 9.81):
    """Exact pressure function ``f(h; h_side)`` of the shallow-water system.

    ``f = (h - h_side) sqrt(g (h + h_side) / (2 h h_side))`` for ``h > h_side``
    (the side wave is a shock) and ``f = 2 (sqrt(g h) - sqrt(g h_side))``
    otherwise (rarefaction).  It is strictly increasing in ``h`` and vanishes at
    ``h = h_side``, which is what makes the star-state iteration well posed.
    """
    h = np.asarray(h, dtype=float)
    g = float(gravity)
    hs = float(h_side)
    if hs <= 0.0:
        # a dry side cannot support a shock: the wave is a rarefaction into the
        # dry bed and the pressure function is the rarefaction branch alone
        return 2.0 * np.sqrt(np.maximum(g * h, 0.0))
    shock = h > hs
    out = 2.0 * (np.sqrt(np.maximum(g * h, 0.0)) - math.sqrt(g * hs))
    with np.errstate(divide="ignore", invalid="ignore"):
        jump = (h - hs) * np.sqrt(g * (h + hs) / (2.0 * h * hs))
    return np.where(shock, jump, out)


# --------------------------------------------------------------------------- #
# star state
# --------------------------------------------------------------------------- #
def riemann_star_state(h_l: float, u_l: float, h_r: float, u_r: float,
                       gravity: float = 9.81, tol: float = 1e-15) -> dict:
    """Solve one shallow-water Riemann problem for the star state.

    Returns ``h_star``, ``u_star``, ``c_star``, the outer wave speeds
    ``s_head_left`` / ``s_shock_right`` (the leftmost and rightmost
    characteristic of the Riemann fan), the wave families and the convergence
    diagnostics of the root find (``iterations``, ``residual``).

    A dry side (``h = 0``) degenerates to a single rarefaction whose tail state
    is ``h* = 0``, ``u* = u_side +/- 2 c_side``.
    """
    g = float(gravity)
    h_l, u_l, h_r, u_r = float(h_l), float(u_l), float(h_r), float(u_r)
    if h_l < 0.0 or h_r < 0.0:
        raise ValueError("negative depth in Riemann data")
    if h_l == 0.0 and h_r == 0.0:
        return {"h_star": 0.0, "u_star": 0.0, "c_star": 0.0,
                "s_head_left": float("nan"), "s_shock_right": float("nan"),
                "left_family": "dry", "right_family": "dry",
                "iterations": 0, "residual": 0.0, "phi_at_root": 0.0,
                "s_from_mass_left": None, "s_from_mass_right": None}
    if h_r == 0.0:                                   # dam break onto a dry bed
        c_l = math.sqrt(g * h_l)
        h_star, u_star = 0.0, u_l + 2.0 * c_l
        return {"h_star": 0.0, "u_star": u_star, "c_star": 0.0,
                "s_head_left": u_l - c_l, "s_shock_right": u_star,
                "left_family": "rarefaction", "right_family": "rarefaction",
                "iterations": 0, "residual": 0.0, "phi_at_root": 0.0,
                "s_from_mass_left": None, "s_from_mass_right": None}
    if h_l == 0.0:
        c_r = math.sqrt(g * h_r)
        h_star, u_star = 0.0, u_r - 2.0 * c_r
        return {"h_star": 0.0, "u_star": u_star, "c_star": 0.0,
                "s_head_left": u_star, "s_shock_right": u_r + c_r,
                "left_family": "rarefaction", "right_family": "rarefaction",
                "iterations": 0, "residual": 0.0, "phi_at_root": 0.0,
                "s_from_mass_left": None, "s_from_mass_right": None}

    def phi(h):
        return (float(pressure_function(h, h_l, g))
                + float(pressure_function(h, h_r, g)) + (u_r - u_l))

    lo = 0.0
    hi = max(h_l, h_r)
    for _ in range(200):                              # expand until bracketed
        if phi(hi) >= 0.0:
            break
        hi *= 2.0
    else:                                             # pragma: no cover
        raise RuntimeError("Riemann star state not bracketed")
    iterations = 0
    for iterations in range(1, 401):                  # bisection to machine eps
        mid = 0.5 * (lo + hi)
        if phi(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, mid):
            break
    h_star = 0.5 * (lo + hi)
    f_l = float(pressure_function(h_star, h_l, g))
    f_r = float(pressure_function(h_star, h_r, g))
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)
    c_star = math.sqrt(g * h_star)

    # discontinuity / characteristic speeds from the states themselves.
    # ``s_from_mass`` is the raw Rankine-Hugoniot mass jump; the speed actually
    # returned uses the algebraically identical but cancellation-free form
    # ``S = u_K + sqrt(g h* (h* + h_K) / (2 h_K))``, which keeps the weak-bore
    # limit accurate.  Both are reported so the conditioning can be audited.
    s_from_mass_left = s_from_mass_right = None
    if h_star > h_l:                                  # left shock
        left_family = "shock"
        s_from_mass_left = (h_star * u_star - h_l * u_l) / (h_star - h_l)
        s_head_left = u_l - math.sqrt(g * h_star * (h_star + h_l) / (2.0 * h_l))
    else:                                             # left rarefaction
        left_family = "rarefaction"
        s_head_left = u_l - math.sqrt(g * h_l)
    if h_star > h_r:                                  # right shock (bore)
        right_family = "shock"
        s_from_mass_right = (h_star * u_star - h_r * u_r) / (h_star - h_r)
        s_shock_right = u_r + math.sqrt(g * h_star * (h_star + h_r) / (2.0 * h_r))
    else:                                             # right rarefaction
        right_family = "rarefaction"
        s_shock_right = u_r + math.sqrt(g * h_r)
    return {"h_star": h_star, "u_star": u_star, "c_star": c_star,
            "s_head_left": s_head_left, "s_shock_right": s_shock_right,
            "s_from_mass_left": s_from_mass_left,
            "s_from_mass_right": s_from_mass_right,
            "left_family": left_family, "right_family": right_family,
            "iterations": iterations, "residual": abs(phi(h_star)),
            "phi_at_root": float(phi(h_star))}


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def riemann_sample(xi, h_l: float, u_l: float, h_r: float, u_r: float,
                   gravity: float = 9.81):
    """Sample the exact solution in the self-similar coordinate ``xi = x / t``.

    Returns ``(h, u)`` with the shape of ``xi``.  The wave pattern is read off
    the star state (``left_family`` / ``right_family``), so a rarefaction fan,
    a constant star region, and a bore are all handled by the same branch
    structure.
    """
    g = float(gravity)
    star = riemann_star_state(h_l, u_l, h_r, u_r, g)
    xs = np.asarray(xi, dtype=float)
    h_star, u_star = star["h_star"], star["u_star"]
    c_star = star["c_star"]
    depth = np.full_like(xs, float(h_star))
    vel = np.full_like(xs, float(u_star))

    # left wave: everything at or upstream of its trailing edge is star/plume
    if h_l == 0.0:
        depth = np.where(xs <= star["s_head_left"], 0.0, depth)
        vel = np.where(xs <= star["s_head_left"], 0.0, vel)
    elif star["left_family"] == "rarefaction":
        c_l = math.sqrt(g * h_l)
        s_head, s_tail = u_l - c_l, u_star - c_star
        depth = np.where(xs <= s_head, h_l, depth)
        vel = np.where(xs <= s_head, u_l, vel)
        fan = (xs > s_head) & (xs < s_tail)
        inv = u_l + 2.0 * c_l                          # u + 2c invariant
        c_fan = (inv - xs) / 3.0
        depth = np.where(fan, c_fan * c_fan / g, depth)
        vel = np.where(fan, (inv + 2.0 * xs) / 3.0, vel)
    else:
        depth = np.where(xs <= star["s_head_left"], h_l, depth)
        vel = np.where(xs <= star["s_head_left"], u_l, vel)

    # right wave
    if h_r == 0.0:
        # a dry right side: the degenerate right rarefaction collapses onto the
        # dry front xi = u*, downstream of which there is no water at all
        depth = np.where(xs >= star["s_shock_right"], 0.0, depth)
        vel = np.where(xs >= star["s_shock_right"], 0.0, vel)
    elif star["right_family"] == "rarefaction":
        c_r = math.sqrt(g * h_r)
        s_tail, s_head = u_star + c_star, u_r + c_r
        depth = np.where(xs >= s_head, h_r, depth)
        vel = np.where(xs >= s_head, u_r, vel)
        fan = (xs > s_tail) & (xs < s_head)
        inv = u_r - 2.0 * c_r                          # u - 2c invariant
        c_fan = (xs - inv) / 3.0
        depth = np.where(fan, c_fan * c_fan / g, depth)
        vel = np.where(fan, (inv + 2.0 * xs) / 3.0, vel)
    else:
        depth = np.where(xs >= star["s_shock_right"], h_r, depth)
        vel = np.where(xs >= star["s_shock_right"], u_r, vel)
    return np.maximum(depth, 0.0), vel


def riemann_solution(x, t: float, h_l: float, u_l: float, h_r: float,
                     u_r: float, gravity: float = 9.81, x0: float = 0.0):
    """Exact ``(h, u)`` of the Riemann problem at ``(x, t)``.

    For ``t <= 0`` the piecewise-constant initial data are returned.
    """
    xs = np.asarray(x, dtype=float)
    if t <= 0.0:
        up = xs <= x0
        return np.where(up, float(h_l), float(h_r)), np.where(
            up, float(u_l), float(u_r))
    return riemann_sample((xs - x0) / float(t), h_l, u_l, h_r, u_r, gravity)


# --------------------------------------------------------------------------- #
# independent measurement of a discontinuity's speed
# --------------------------------------------------------------------------- #
def jump_trajectory_speed(sample, xi_lo: float, xi_hi: float, t0: float,
                          t1: float, x0: float = 0.0, n: int = 400_001) -> dict:
    """Measure a discontinuity speed from a sampled profile, not from a formula.

    The caller passes ``sample(xi_array)`` -- the analytical profile expressed
    in the self-similar coordinate -- plus a bracket ``[xi_lo, xi_hi]`` that
    contains exactly one jump.  The jump position is located at two times by
    finding the steepest gradient on a fine grid, and the speed is obtained as
    ``(x_1 - x_0) / (t_1 - t_0)``.  This certifies the reference solution's
    wave speed without reusing the constant that the reference itself computed
    (a wrong bore speed in a closed form cannot survive this test, because the
    sampled jump would move at the wrong rate).
    """
    grid = np.linspace(xi_lo, xi_hi, n)
    out = {}
    for tag, t in (("t0", float(t0)), ("t1", float(t1))):
        h, _ = sample(grid)
        grad = np.abs(np.diff(h))
        j = int(np.argmax(grad))
        # parabolic refinement of the maximum of the gradient magnitude
        if 0 < j < grad.size - 1:
            y0, y1, y2 = grad[j - 1], grad[j], grad[j + 1]
            denom = (y0 - 2.0 * y1 + y2)
            shift = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
        else:
            shift = 0.0
        dxi = grid[1] - grid[0]
        xi_jump = grid[j] + shift * dxi
        out["xi_" + tag] = float(xi_jump)
        out["x_" + tag] = float(x0) + xi_jump * t
    out["speed"] = (out["x_t1"] - out["x_t0"]) / (t1 - t0)
    xi_j = out["xi_t0"]
    eps = 8.0 * (grid[1] - grid[0])          # wider than one cell, by design
    h_minus, _ = sample(np.array([xi_j - eps]))
    h_plus, _ = sample(np.array([xi_j + eps]))
    out["jump_height"] = float(abs(h_minus[0] - h_plus[0]))
    out["resolved_fraction"] = float(out["jump_height"]
                                    / max(abs(h_minus[0] + h_plus[0]), 1e-30))
    return out


def level_trajectory_speed(sample, level: float, xi_lo: float, xi_hi: float,
                           t0: float, t1: float, x0: float = 0.0,
                           n: int = 200_001) -> dict:
    """Measure the speed of a *depth level* in a sampled self-similar profile.

    Unlike :func:`jump_trajectory_speed` this locates the rightmost crossing of
    ``h = level`` by linear interpolation, so it also works for smooth or cusped
    features (e.g. the zero-depth tip of the Ritter fan, where the gradient
    vanishes and a steepest-gradient locator would lock onto the fan's inner
    edge instead of the front).
    """
    grid = np.linspace(xi_lo, xi_hi, n)
    h, _ = sample(grid)
    above = np.nonzero(h > level)[0]
    if above.size == 0:
        raise ValueError(f"level {level} not present in the sampled bracket")
    j = int(above[-1])
    if j + 1 < h.size:
        h0_, h1_ = h[j], h[j + 1]
        frac = (h0_ - level) / (h0_ - h1_) if h0_ != h1_ else 0.0
        xi_level = grid[j] + frac * (grid[1] - grid[0])
    else:
        xi_level = grid[j]
    return {"xi_t0": float(xi_level),
            "x_t0": float(x0) + float(xi_level) * float(t0),
            "x_t1": float(x0) + float(xi_level) * float(t1),
            "speed": float(xi_level),
            "level": float(level)}