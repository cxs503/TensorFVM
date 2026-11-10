"""Exact cell-volume geometry for VOF verification anchors.

Every benchmark in :mod:`tensorfvm.vof_verification` starts from a *known*
indicator field, so the reference volume fraction of each cell must be the exact
area of ``cell ∩ shape`` -- not a subsampled approximation.  A subsampled initial
field carries its own O(noise) error which would otherwise be indistinguishable
from the transport error once the grids are refined.

The exact quantity is the intersection area of a disk with an axis-aligned
rectangle.  Writing the disk as the vertical section
``|y - cy| <= f(x)``, ``f(x) = sqrt(r^2 - (x - cx)^2)``, the area is

.. math::

    A = \\int_{x_0}^{x_1} \\Big[\\min(y_1, c_y + f) - \\max(y_0, c_y - f)\\Big]_+ dx,

whose integrand is smooth *between* the "kink" abscissae where ``c_y ± f``
crosses a rectangle edge.  Splitting the interval at those kinks (at most four
of them) and applying 24-point Gauss-Legendre quadrature per sub-interval makes
the integral exact to round-off, which is verified against the closed-form whole
disk / half disk / quarter disk areas in the unit tests.

Only the ``O(n)`` cells that straddle the interface need the quadrature; cells
entirely inside or outside are filled analytically.  ``O(n^2)`` Python work is
therefore avoided.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "disk_rect_area",
    "circle_alpha",
    "zalesak_alpha",
    "analytic_area",
]

# 24-point Gauss-Legendre nodes/weights on [-1, 1] (Abramowitz & Stegun 25.4).
_GL_NODES = (
    -0.9951872199970213, -0.9747285559713095, -0.9382745520027328,
    -0.8864155270044011, -0.8200019859739029, -0.7401241915785544,
    -0.6480936519369755, -0.5454214713888396, -0.4337935076260451,
    -0.3150426796961634, -0.1911188674736163, -0.0640568928626056,
    0.0640568928626056, 0.1911188674736163, 0.3150426796961634,
    0.4337935076260451, 0.5454214713888396, 0.6480936519369755,
    0.7401241915785544, 0.8200019859739029, 0.8864155270044011,
    0.9382745520027328, 0.9747285559713095, 0.9951872199970213,
)
_GL_WEIGHTS = (
    0.0123412297999872, 0.0285313886289337, 0.0442774388174198,
    0.0592985849154368, 0.0733464814085803, 0.0861901615319533,
    0.0976186521041139, 0.1074442701159656, 0.1155056680537256,
    0.1216704729278034, 0.1258374563468283, 0.1279381953467522,
    0.1279381953467522, 0.1258374563468283, 0.1216704729278034,
    0.1155056680537256, 0.1074442701159656, 0.0976186521041139,
    0.0861901615319533, 0.0733464814085803, 0.0592985849154368,
    0.0442774388174198, 0.0285313886289337, 0.0123412297999872,
)


def _disk_quadrant_area(cx: float, cy: float, r: float, x0: float, x1: float,
                       y0: float, y1: float, sign: int) -> float:
    """Area contribution of one monotone half of the disk (``sign = -1``/``+1``).

    The ``x``-integrand ``min(y1, c_y+f) - max(y0, c_y-f)`` has a square-root
    derivative singularity at the tangency points ``x = c_x ± r``, where a fixed
    Gauss rule loses accuracy (~1e-5 relative, measured).  Substituting
    ``x = c_x + r sin(theta)`` makes ``f = r cos(theta)`` and turns the integrand
    into ``h(theta) * r cos(theta)``, which is smooth between the kink angles, so
    the quadrature becomes exact to round-off.
    """
    half_pi = 0.5 * math.pi
    lo, hi = (-half_pi, 0.0) if sign < 0 else (0.0, half_pi)
    xa = cx + r * math.sin(lo)
    xb = cx + r * math.sin(hi)
    lo_x, hi_x = max(x0, min(xa, xb)), min(x1, max(xa, xb))
    if hi_x <= lo_x:
        return 0.0
    # the x-interval clipped to [x0, x1] -> the corresponding theta interval
    def theta_of(x: float) -> float:
        return math.asin(max(-1.0, min(1.0, (x - cx) / r)))

    tha, thb = theta_of(lo_x), theta_of(hi_x)
    breaks = {lo, hi, tha, thb}
    for y in (y0, y1):
        d = abs(y - cy)
        if d < r:
            root = math.sqrt(r * r - d * d)
            for x in (cx - root, cx + root):
                if lo_x < x < hi_x:
                    breaks.add(theta_of(x))
    knots = sorted(t for t in breaks if lo - 1e-15 <= t <= hi + 1e-15)
    knots = [t for t in knots if tha - 1e-15 <= t <= thb + 1e-15]
    total = 0.0
    for a, b in zip(knots[:-1], knots[1:]):
        if b <= a:
            continue
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        acc = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
            th = mid + half * node
            x = cx + r * math.sin(th)
            f = r * math.cos(th)
            height = min(y1, cy + f) - max(y0, cy - f)
            if height > 0.0:
                acc += weight * height * f
        total += half * acc
    return total


def disk_rect_area(cx: float, cy: float, r: float,
                   x0: float, x1: float, y0: float, y1: float) -> float:
    """Exact area of ``disk((cx, cy), r) ∩ [x0, x1] × [y0, y1]``.

    Verified against the closed-form full / half / quarter disk areas to 1e-13
    relative in the unit tests.
    """
    if r <= 0.0 or x1 <= x0 or y1 <= y0:
        return 0.0
    return (_disk_quadrant_area(cx, cy, r, x0, x1, y0, y1, -1)
            + _disk_quadrant_area(cx, cy, r, x0, x1, y0, y1, +1))


def circle_alpha(
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
    centre: tuple[float, float] = (0.5, 0.75),
    radius: float = 0.15,
) -> torch.Tensor:
    """Exact volume fraction of a disk on a uniform ``nx × ny`` Cartesian grid."""
    dx, dy = lx / nx, ly / ny
    cx, cy = centre
    x0 = torch.arange(nx, dtype=torch.float64) * dx
    y0 = torch.arange(ny, dtype=torch.float64) * dy
    xc = x0 + 0.5 * dx
    yc = y0 + 0.5 * dy
    # signed distance of each cell centre to the circle: |d| > diag/2 => pure cell
    dist = torch.sqrt((xc[None, :] - cx) ** 2 + (yc[:, None] - cy) ** 2)
    half_diag = 0.5 * math.hypot(dx, dy)
    alpha = torch.zeros((ny, nx), dtype=torch.float64)
    alpha[dist <= radius - half_diag] = 1.0
    alpha[dist >= radius + half_diag] = 0.0
    band = (dist > radius - half_diag) & (dist < radius + half_diag)
    rows, cols = torch.nonzero(band, as_tuple=True) if band.any() else ((), ())
    cell = dx * dy
    for j, i in zip(rows.tolist(), cols.tolist()):
        area = disk_rect_area(cx, cy, radius, float(x0[i]), float(x0[i]) + dx,
                              float(y0[j]), float(y0[j]) + dy)
        alpha[j, i] = min(1.0, max(0.0, area / cell))
    return alpha


def zalesak_alpha(
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
    centre: tuple[float, float] = (0.5, 0.75),
    radius: float = 0.15,
    slot_half_width: float = 0.025,
) -> torch.Tensor:
    """Exact volume fraction of the Zalesak slotted disk.

    The classical Zalesak (1979) shape: a disk of radius ``r`` centred at
    ``(0.5, 0.75)`` with a rectangular slot of width ``2 * slot_half_width`` cut
    from its centre line up to the top of the disk.  The slot rectangle is fully
    contained in the disk for the classical parameters, so the cell volume
    fraction is ``A(cell ∩ disk) - A(cell ∩ disk ∩ slot)``.
    """
    dx, dy = lx / nx, ly / ny
    cx, cy = centre
    slot_y0 = cy
    slot_x0, slot_x1 = cx - slot_half_width, cx + slot_half_width
    x0 = torch.arange(nx, dtype=torch.float64) * dx
    y0 = torch.arange(ny, dtype=torch.float64) * dy
    xc = x0 + 0.5 * dx
    yc = y0 + 0.5 * dy
    dist = torch.sqrt((xc[None, :] - cx) ** 2 + (yc[:, None] - cy) ** 2)
    half_diag = 0.5 * math.hypot(dx, dy)
    alpha = torch.zeros((ny, nx), dtype=torch.float64)
    inside = dist <= radius - half_diag
    alpha[inside] = 1.0
    # cells fully covered by the slot but inside the disk -> zero
    slot_lo_x, slot_hi_x = slot_x0, slot_x1
    fully_in_slot_x = (x0 >= slot_lo_x - 1e-15) & (x0 + dx <= slot_hi_x + 1e-15)
    fully_in_slot_y = (y0 >= slot_y0 - 1e-15)
    slot_full = fully_in_slot_y[:, None] & fully_in_slot_x[None, :] & inside
    alpha[slot_full] = 0.0
    straddle = ~(inside | slot_full) & (dist < radius + half_diag)
    # also the cells whose disk part is removed by the slot but that are not
    # fully inside the disk
    touch_slot = (
        ((x0 + dx > slot_lo_x) & (x0 < slot_hi_x))[None, :]
        & (y0 + dy > slot_y0 - dy)[:, None]
    )
    band = straddle | (touch_slot & (dist < radius + half_diag))
    cell = dx * dy
    for j, i in zip(*[t.tolist() for t in torch.nonzero(band, as_tuple=True)]):
        xs, xe = float(x0[i]), float(x0[i]) + dx
        ys, ye = float(y0[j]), float(y0[j]) + dy
        area = disk_rect_area(cx, cy, radius, xs, xe, ys, ye)
        if area > 0.0:
            sx0, sx1 = max(xs, slot_x0), min(xe, slot_x1)
            sy0, sy1 = max(ys, slot_y0), min(ye, cy + radius)
            if sx1 > sx0 and sy1 > sy0:
                area -= disk_rect_area(cx, cy, radius, sx0, sx1, sy0, sy1)
        alpha[j, i] = min(1.0, max(0.0, area / cell))
    return alpha


def analytic_area(
    lx: float = 1.0,
    ly: float = 1.0,
    radius: float = 0.15,
    slot_half_width: float = 0.0,
) -> float:
    """Closed-form area of the disk with a slot cut from its centre to its apex.

    The slot rectangle ``[c_x - w, c_x + w] × [c_y, c_y + r]`` is *not* fully
    contained in the disk: near the apex the disk half-width drops below ``w``.
    With ``t* = sqrt(r^2 - w^2)`` the removed area is ``w t* + pi r^2 / 2 -
    r^2 asin(t*/r)`` -- the closed form of
    ``int_0^r min(2w, 2 sqrt(r^2 - t^2)) dt``, which the unit tests check against
    the quadrature used by :func:`zalesak_alpha`.
    """
    area = math.pi * radius ** 2
    w = slot_half_width
    if w > 0.0:
        t_star = math.sqrt(max(radius ** 2 - w ** 2, 0.0))
        removed = w * t_star + 0.5 * math.pi * radius ** 2 \
            - radius ** 2 * math.asin(min(1.0, t_star / radius))
        area -= removed
    return area