"""Wall-shear (lid drag) estimators and corner-strip decomposition.

Companion to :mod:`tensorfvm.cavity_power_budget`; used by
``docs/CASE-FVM-CAVITY-RE1000/wall_shear_budget.py`` and its tests.

The production solver never forms the wall shear explicitly: the lid momentum
balance uses the mirror ghost ``u_ghost = 2U - u_adjacent``, i.e. the
two-point one-sided slope

    order = 1 :  (du/dy)|_lid = 2 (U - u_adjacent) / h                (O(h))

which is first order for a smooth profile.  A three-point quadratic through
``(0, U)``, ``(h/2, u1)``, ``(3h/2, u2)`` removes the leading term

    order = 2 :  (du/dy)|_lid = (8 U - 9 u1 + u2) / (3 h)            (O(h^2))

Both are exposed because the difference between them *is* the truncation error
of the quantity the solver would report as "drag".
"""

from __future__ import annotations

import numpy as np

__all__ = ["lid_wall_shear", "lid_drag", "drag_decomposition"]


def lid_wall_shear(u: np.ndarray, h: float, lid_velocity: float = 1.0,
                   order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(x, du/dy)`` samples along the lid from the lid-adjacent ``u`` row.

    ``u`` is the staggered x-velocity with shape ``(n + 1, n)``; the samples live
    at ``x = i h``, ``i = 0 .. n``.
    """
    u = np.asarray(u, dtype=float)
    n = u.shape[1]
    if u.shape != (n + 1, n):
        raise ValueError(f"expected staggered u of shape (n+1, n), got {u.shape}")
    if n < 3 and order == 2:
        raise ValueError("order 2 needs n >= 3")
    x = np.arange(n + 1) * h
    u1 = u[:, -1]                       # y = L - h/2
    if order == 1:
        shear = 2.0 * (lid_velocity - u1) / h
    elif order == 2:
        u2 = u[:, -2]                   # y = L - 3h/2
        shear = (8.0 * lid_velocity - 9.0 * u1 + u2) / (3.0 * h)
    else:
        raise ValueError("order must be 1 or 2")
    return x, shear


def _integrate(x: np.ndarray, f: np.ndarray,
               x_range: tuple[float, float] | None) -> float:
    """Trapezoidal integral; ``x_range = (lo, hi)`` with the lower node included
    and the upper node excluded, so that adjacent ranges partition the samples
    without double counting."""
    mask = np.ones_like(x, dtype=bool)
    if x_range is not None:
        lo, hi = x_range
        mask = (x >= lo - 1e-12) & (x <= hi + 1e-12)
    return float(np.trapezoid(f[mask], x[mask]))


def lid_drag(u: np.ndarray, h: float, nu: float, lid_velocity: float = 1.0,
             order: int = 1, x_range: tuple[float, float] | None = None,
             rho: float = 1.0) -> float:
    """Lid drag per unit depth, ``mu * int (du/dy)|_lid dx`` over ``x_range``."""
    x, shear = lid_wall_shear(u, h, lid_velocity=lid_velocity, order=order)
    return rho * nu * _integrate(x, shear, x_range)


def drag_decomposition(u: np.ndarray, h: float, nu: float, lid_velocity: float = 1.0,
                       order: int = 1, margin: float = 0.1, rho: float = 1.0) -> dict:
    """Split the lid drag into ``[0, margin] + [margin, 1-margin] + [1-margin, 1]``.

    The two outer strips contain the lid-corner singularities; on the FVM ladder
    they carry roughly half of the total drag while being the least resolved
    part of the field, which is why the total is reported next to the split.
    """
    n = np.asarray(u).shape[1]
    margin = round(margin * n) / n          # snap onto a grid node so the three
    #                                         trapezoids share endpoints exactly
    total = lid_drag(u, h, nu, lid_velocity, order, None, rho)
    left = lid_drag(u, h, nu, lid_velocity, order, (0.0, margin), rho)
    centre = lid_drag(u, h, nu, lid_velocity, order, (margin, 1.0 - margin), rho)
    right = lid_drag(u, h, nu, lid_velocity, order, (1.0 - margin, 1.0), rho)
    return {
        "total": total,
        "left_strip": left,
        "centre": centre,
        "right_strip": right,
        "corner_share": (left + right) / total if total else float("nan"),
        "margin": float(margin),
        "order": int(order),
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Grid-convergence order from two successive differences (sign-aware)."""
    d1, d2 = coarse, fine
    if d1 == 0.0 or d2 == 0.0 or d1 * d2 <= 0.0:
        return float("nan")
    import math
    return float(math.log(abs(d1 / d2)) / math.log(ratio))