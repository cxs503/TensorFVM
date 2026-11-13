"""Closed-form shallow-water dam-break solutions (Ritter dry bed, Stoker wet bed).

Both solutions are the inviscid, hydrostatic, flat-bed shallow-water Riemann
problem with a dam at ``x = x_dam`` and still water on both sides at ``t = 0``:

``ritter_state``
    Dry downstream bed (``h1 = 0``).  A single rarefaction fan connects the
    undisturbed upstream depth ``h0`` to the dry bed; the front travels at
    ``2 c0`` (``c0 = sqrt(g h0)``) and the depth at the dam is ``4 h0 / 9``.

``stoker_state``
    Wet downstream bed (``h0 > h1 > 0``).  A left rarefaction fan, a constant
    state ``(h2, u2)`` and a right-moving bore (shock) of speed ``U``.  The
    unknown ``c2 = sqrt(g h2)`` follows from matching the rarefaction Riemann
    invariant ``u + 2c = 2c0`` against the Rankine-Hugoniot momentum jump of the
    bore; the scalar root is bracketed in ``(c1, c0)`` and bisected to machine
    precision, so no tabulated value is transcribed.

References
----------
Ritter, A. (1892).  "Die Fortpflanzung der Wasserwellen."  *Zeitschrift des
Vereins Deutscher Ingenieure* **36**, 947-954 (dry-bed rarefaction solution;
front speed ``2 sqrt(g h0)``).
Stoker, J. J. (1957).  *Water Waves: The Mathematical Theory with
Applications*.  Interscience, New York, chapter 10: the dam-break problem with
a bore on a wet bed, i.e. the ``h1 > 0`` case solved here.

The two closed forms are *not* transcribed from tables: every number consumed
by the benchmark is evaluated from the formulae below, and
:func:`check_stoker_solution` re-derives the bore jump conditions so that the
"exact" reference is certified numerically (see the verification section of
``docs/shallow-water-dam-break/REPORT.md``).

Non-dimensional time
--------------------
``t* = t sqrt(g/h0)`` (time scaled by the gravity-wave time of the upstream
column).  This is the scaling used throughout the dam-break benchmark, so
``t* = 2`` means ``t = 2 sqrt(h0/g)`` and the dry front sits at ``x_f = 4 h0``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

__all__ = [
    "gravity_wave_speed",
    "ritter_front_position",
    "ritter_head_position",
    "ritter_state",
    "stoker_constant_state",
    "stoker_region_bounds",
    "stoker_state",
    "t_star_to_time",
    "time_to_t_star",
    "check_stoker_solution",
]


def gravity_wave_speed(depth, gravity: float = 9.81):
    """``c = sqrt(g h)``, clipped at ``h = 0`` so dry cells stay real."""
    return np.sqrt(np.maximum(gravity * np.asarray(depth, dtype=float), 0.0))


# --------------------------------------------------------------------------- #
# non-dimensional time
# --------------------------------------------------------------------------- #
def t_star_to_time(t_star: float, h0: float, gravity: float = 9.81) -> float:
    """Physical time for the dimensionless ``t* = t sqrt(g/h0)``."""
    return float(t_star) * math.sqrt(h0 / gravity)


def time_to_t_star(t: float, h0: float, gravity: float = 9.81) -> float:
    """Inverse of :func:`t_star_to_time`."""
    return float(t) / math.sqrt(h0 / gravity)


# --------------------------------------------------------------------------- #
# Ritter (1892): dam break on a dry bed
# --------------------------------------------------------------------------- #
def ritter_head_position(t: float, h0: float, gravity: float = 9.81,
                         x_dam: float = 0.0) -> float:
    """Upstream edge of the rarefaction fan, ``x_dam - c0 t``."""
    return x_dam - t * math.sqrt(gravity * h0)


def ritter_front_position(t: float, h0: float, gravity: float = 9.81,
                          x_dam: float = 0.0) -> float:
    """Front (dry-bed tip) position, ``x_dam + 2 t sqrt(g h0)`` (Ritter 1892)."""
    return x_dam + 2.0 * t * math.sqrt(gravity * h0)


def ritter_state(x, t: float, h0: float, gravity: float = 9.81,
                 x_dam: float = 0.0):
    """Exact depth/velocity of the Ritter solution at ``(x, t)``.

    Returns ``(h, u)`` as ``float64`` arrays of the shape of ``x``.

    ``h = (2 c0 - xi)^2 / (9 g)`` and ``u = 2 (c0 + xi) / 3`` inside the fan
    ``-c0 <= xi <= 2 c0`` with ``xi = (x - x_dam) / t``; ``h = h0, u = 0``
    upstream of the fan and ``h = u = 0`` downstream of the front.
    """
    xs = np.asarray(x, dtype=float)
    h0 = float(h0)
    c0 = math.sqrt(gravity * h0)
    depth = np.zeros_like(xs)
    vel = np.zeros_like(xs)
    if t <= 0.0:
        depth = np.where(xs <= x_dam, h0, 0.0)
        return depth, vel
    xi = (xs - x_dam) / t
    fan = (xi > -c0) & (xi < 2.0 * c0)
    c = np.where(fan, (2.0 * c0 - xi) / 3.0, 0.0)
    depth = np.where(fan, c * c / gravity, 0.0)
    depth = np.where(xi <= -c0, h0, depth)
    vel = np.where(fan, (2.0 / 3.0) * (c0 + xi), 0.0)
    return np.clip(depth, 0.0, h0), vel


# --------------------------------------------------------------------------- #
# Stoker (1957): dam break on a wet bed (rarefaction + bore)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _stoker_constants(h0: float, h1: float, gravity: float) -> tuple[float, ...]:
    """Solve for the bore state ``(c2, h2, u2, U)``; cached on the arguments."""
    c0 = math.sqrt(gravity * h0)
    c1 = math.sqrt(gravity * h1)

    def residual(c2: float) -> float:
        h2 = c2 * c2 / gravity
        return 2.0 * (c0 - c2) - (h2 - h1) * math.sqrt(
            gravity * (h2 + h1) / (2.0 * h2 * h1))

    lo, hi = c1, c0
    f_lo, f_hi = residual(lo), residual(hi)
    if not (f_lo > 0.0 > f_hi):
        raise ValueError(
            f"bore state not bracketed for h0={h0}, h1={h1}: "
            f"f(c1)={f_lo:.3e}, f(c0)={f_hi:.3e}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 4.0 * math.ulp(mid):
            break
    c2 = 0.5 * (lo + hi)
    h2 = c2 * c2 / gravity
    u2 = 2.0 * (c0 - c2)
    shock_speed = h2 * u2 / (h2 - h1)
    return c0, c1, c2, h2, u2, shock_speed


def stoker_constant_state(h0: float, h1: float, gravity: float = 9.81) -> dict:
    """Closed-form states of the Stoker wet-bed dam break.

    Returns a dict with the upstream depth ``h0``, the undisturbed downstream
    depth ``h1``, the bore speed ``U`` (``shock_speed``), the constant
    post-bore state ``(h2, u2)`` and the wave speeds ``c0, c1, c2``.

    Matching conditions used (both re-checked by :func:`check_stoker_solution`):

    * rarefaction invariant ``u2 + 2 c2 = 2 c0``  (left wave family),
    * bore mass jump ``U (h2 - h1) = h2 u2``,
    * bore momentum jump ``U h2 u2 = g (h2^2 - h1^2) / 2 + h2 u2^2``,

    the last two combine to ``u2 = (h2 - h1) sqrt(g (h2 + h1) / (2 h2 h1))``.
    """
    if not (h0 > h1 > 0.0):
        raise ValueError(f"Stoker needs h0 > h1 > 0, got h0={h0}, h1={h1}")
    gravity = float(gravity)
    c0, c1, c2, h2, u2, shock_speed = _stoker_constants(
        float(h0), float(h1), gravity)
    return {
        "h0": float(h0),
        "h1": float(h1),
        "g": gravity,
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "h2": h2,
        "u2": u2,
        "shock_speed": shock_speed,
        "tail_speed": u2 - c2,       # xi of the fan/constant-state junction
        "head_speed": -c0,           # xi of the upstream edge of the fan
        "step_ratio": h2 / float(h1),
    }


def stoker_state(x, t: float, h0: float, h1: float, gravity: float = 9.81,
                 x_dam: float = 0.0):
    """Exact depth/velocity of the Stoker wet-bed solution at ``(x, t)``.

    Returns ``(h, u)``.  Regions in the self-similar coordinate
    ``xi = (x - x_dam) / t``: ``xi <= -c0`` undisturbed upstream;
    ``-c0 < xi < u2 - c2`` rarefaction fan; ``u2 - c2 <= xi <= U`` the constant
    ``(h2, u2)`` state; ``xi > U`` undisturbed ``h1`` (the "still-water" region
    whose level step across the bore is the benchmark's second metric).
    """
    xs = np.asarray(x, dtype=float)
    sol = stoker_constant_state(h0, h1, gravity)
    depth = np.full_like(xs, sol["h1"])
    vel = np.zeros_like(xs)
    if t <= 0.0:
        up = xs <= x_dam
        return np.where(up, sol["h0"], sol["h1"]), vel
    xi = (xs - x_dam) / t
    c0, c2, h2, u2 = sol["c0"], sol["c2"], sol["h2"], sol["u2"]
    fan = (xi > -c0) & (xi < u2 - c2)
    c = np.where(fan, (2.0 * c0 - xi) / 3.0, 0.0)
    depth = np.where(fan, c * c / gravity, depth)
    vel = np.where(fan, (2.0 / 3.0) * (c0 + xi), vel)
    depth = np.where(xi <= -c0, sol["h0"], depth)
    plateau = (xi >= u2 - c2) & (xi <= sol["shock_speed"])
    depth = np.where(plateau, h2, depth)
    vel = np.where(plateau, u2, vel)
    return depth, vel


def check_stoker_solution(h0: float, h1: float, gravity: float = 9.81) -> dict:
    """Verify the closed form against its own jump/invariant conditions.

    Nothing here compares against a table: the residuals are evaluated at the
    bracketed root, so a *relative* residual near machine precision certifies
    that the returned ``(h2, u2, U)`` really solve the Stoker problem.
    """
    sol = stoker_constant_state(h0, h1, gravity)
    c0, c2, h2, u2 = sol["c0"], sol["c2"], sol["h2"], sol["u2"]
    U, h1 = sol["shock_speed"], sol["h1"]
    g = sol["g"]
    invariant = (u2 + 2.0 * c2) - 2.0 * c0
    mass_jump = U * (h2 - h1) - h2 * u2
    mom_jump = U * (h2 * u2 - 0.0) - (
        (h2 * u2 * u2 + 0.5 * g * h2 * h2) - 0.5 * g * h1 * h1)
    scale_mass = max(abs(U * (h2 - h1)), 1e-300)
    scale_mom = max(abs(U * h2 * u2), 1e-300)
    return {
        "riemann_invariant_residual": invariant,
        "riemann_invariant_relative": abs(invariant) / (2.0 * c0),
        "mass_jump_residual": mass_jump,
        "mass_jump_relative": abs(mass_jump) / scale_mass,
        "momentum_jump_residual": mom_jump,
        "momentum_jump_relative": abs(mom_jump) / scale_mom,
    }


def stoker_region_bounds(h0: float, h1: float, gravity: float = 9.81) -> dict:
    """Similarity-coordinate boundaries of the Stoker regions.

    ``xi = (x - x_dam) / t`` splits the wet-bed solution into four regions:
    undisturbed upstream (``xi <= head``), the left rarefaction fan
    (``head < xi < tail``), the constant post-bore state
    (``tail <= xi <= shock``) and the undisturbed downstream depth ``h1``
    (``xi > shock``).  Returned together with the states that meet at those
    boundaries, so a profile classifier can be written without re-deriving
    them (used by :mod:`tensorfvm.shallow_water.crosscheck`).
    """
    sol = stoker_constant_state(h0, h1, gravity)
    return {"head": sol["head_speed"], "tail": sol["tail_speed"],
            "shock": sol["shock_speed"], "h2": sol["h2"], "u2": sol["u2"],
            "c0": sol["c0"], "c2": sol["c2"]}

