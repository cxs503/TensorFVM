"""Reference anchors for the Phase-0 dam-break gate.

Two independent references are provided so that the acceptance test never
depends on a single digitisation:

1. **Martin & Moyce (1952)** experimental surge-front trajectory (digitised).
2. **Ritter (1892)** analytic shallow-water solution for the idealised
   inviscid dam break on a dry bed (exact, closed form).

Normalisation
-------------
With ``a`` the initial column half-width, ``h0`` the initial column height and
``g`` the gravitational acceleration, Martin & Moyce use

.. math::

    t^* = t\\sqrt{g/a}, \\qquad x^* = \\frac{x_f(t) - a}{a},

i.e. ``x*`` is the *displacement* of the surge front measured from its initial
position ``x = a``.  Using the absolute front position instead is a 1/a offset
that alone produces a 25-35 % error at early times; see
``docs/vof-phase0/report.md`` for the audit that found this defect in the
historical :mod:`tensorfvm.verification.dam_break` comparison.

Ritter normalisation: ``x*_R = x_f/(2 a)`` is not used; the analytic front is
reported in physical units, ``x_f(t) = a + 2 t \\sqrt{g h0}``.

Data caveat
-----------
The Martin & Moyce values are read off the published curve (Fig. 3 of the
1952 paper) and are therefore approximate to a few percent; they are *not* a
precision dataset.  The Phase-0 gate is set to 10 %, which is wider than the
digitisation uncertainty and narrower than the physical spread between the
2D simulations of different authors.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MARTIN_MOYCE_FRONT_T",
    "MARTIN_MOYCE_FRONT_X",
    "MARTIN_MOYCE_HEIGHT_T",
    "MARTIN_MOYCE_HEIGHT_H",
    "martin_moyce_front",
    "martin_moyce_height",
    "ritter_front",
    "ritter_depth",
    "ritter_velocity",
    "to_dimensionless_time",
    "to_dimensionless_front",
]

#: Dimensionless time ``t* = t sqrt(g/a)`` of the digitised surge-front curve.
MARTIN_MOYCE_FRONT_T = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
     7.0, 7.5, 8.0]
)

#: Surge-front displacement ``x* = (x_f - a)/a`` digitised from Martin & Moyce
#: (1952), Fig. 3 (aspect ratio ``h0/a = 2``, smooth bed).
MARTIN_MOYCE_FRONT_X = np.array(
    [0.0, 0.75, 1.40, 1.95, 2.45, 2.90, 3.30, 3.65, 3.98, 4.28, 4.55, 4.80,
     5.03, 5.25, 5.45, 5.64, 5.82]
)

#: Dimensionless time of the digitised residual-column-height curve.
MARTIN_MOYCE_HEIGHT_T = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
)

#: Residual column height ``h/h0`` (liquid depth against the upstream wall).
MARTIN_MOYCE_HEIGHT_H = np.array(
    [1.0, 0.72, 0.52, 0.40, 0.32, 0.27, 0.23, 0.20, 0.18, 0.16, 0.15]
)


def martin_moyce_front(t_star: float) -> float:
    """Linear interpolation of the digitised surge-front curve."""
    t = float(t_star)
    if t <= 0.0:
        return float(MARTIN_MOYCE_FRONT_X[0])
    if t >= MARTIN_MOYCE_FRONT_T[-1]:
        return float(MARTIN_MOYCE_FRONT_X[-1])
    return float(np.interp(t, MARTIN_MOYCE_FRONT_T, MARTIN_MOYCE_FRONT_X))


def martin_moyce_height(t_star: float) -> float:
    """Linear interpolation of the digitised residual-height curve."""
    t = float(t_star)
    if t <= 0.0:
        return float(MARTIN_MOYCE_HEIGHT_H[0])
    if t >= MARTIN_MOYCE_HEIGHT_T[-1]:
        return float(MARTIN_MOYCE_HEIGHT_H[-1])
    return float(np.interp(t, MARTIN_MOYCE_HEIGHT_T, MARTIN_MOYCE_HEIGHT_H))


# --------------------------------------------------------------------------- #
# Ritter (1892) analytic dam break
# --------------------------------------------------------------------------- #
def ritter_front(t: float, h0: float, gravity: float = 9.81, a: float = 0.0) -> float:
    """Front position of the Ritter solution, measured in the same frame.

    The dam sits at ``x = a`` (upstream wall at ``x = 0``), the initial depth is
    ``h0``.  The solution is the inviscid shallow-water rarefaction:

    .. math:: x_f(t) = a + 2 t \\sqrt{g h_0}.

    This is an upper bound for the physical (viscous, finite-depth) surge: it
    carries no bed friction and assumes hydrostatic pressure.
    """
    return a + 2.0 * t * math.sqrt(gravity * h0)


def ritter_depth(x: np.ndarray | float, t: float, h0: float,
                 gravity: float = 9.81, a: float = 0.0) -> np.ndarray:
    """Ritter depth profile ``h(x, t)`` (zero outside the rarefaction fan)."""
    xs = np.asarray(x, dtype=float)
    c0 = math.sqrt(gravity * h0)
    if t <= 0.0:
        return np.where(xs <= a, h0, 0.0)
    xi = (xs - a) / t
    depth = (2.0 * c0 - xi) ** 2 / (9.0 * gravity)
    depth = np.where(xi < -c0, h0, np.where(xi > 2.0 * c0, 0.0, depth))
    # upstream region never exceeds the initial depth
    return np.clip(depth, 0.0, h0)


def ritter_velocity(x: np.ndarray | float, t: float, h0: float,
                    gravity: float = 9.81, a: float = 0.0) -> np.ndarray:
    """Ritter velocity profile ``u(x, t)``."""
    xs = np.asarray(x, dtype=float)
    c0 = math.sqrt(gravity * h0)
    if t <= 0.0:
        return np.zeros_like(xs)
    xi = (xs - a) / t
    vel = (2.0 / 3.0) * (c0 + xi)
    vel = np.where(xi < -c0, 0.0, np.where(xi > 2.0 * c0, 0.0, vel))
    return vel


# --------------------------------------------------------------------------- #
# scaling helpers
# --------------------------------------------------------------------------- #
def to_dimensionless_time(t: float, a: float, gravity: float = 9.81) -> float:
    """``t* = t sqrt(g/a)``."""
    return float(t) * math.sqrt(gravity / a)


def to_dimensionless_front(x_front: float, a: float) -> float:
    """``x* = (x_f - a)/a`` -- displacement from the initial front."""
    return (float(x_front) - a) / a