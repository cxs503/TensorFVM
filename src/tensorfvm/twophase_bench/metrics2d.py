"""Shape-fidelity metrics for interface-capturing (VOF) fields in 2-D.

The node's acceptance criteria are stated on *shape* (area, perimeter), not only
on the volume, because volume error is blind to a scheme that keeps the mass but
smears the interface.  Three quantities are defined here, each with a stated
discretisation so that a number can be reproduced from the artefacts:

``symmetric_area_error``
    ``|{alpha > 1/2} XOR {exact > 1/2}| / |{exact > 1/2}|`` — the misplaced
    *area* of the captured shape.  Threshold-based, so it is insensitive to a
    scheme that shifts a little liquid across the 1/2 level in a thin band
    (which the L1 norm would count).  This is the quantity criterion C2 uses.

``contour_perimeter``
    Length of the ``alpha = 1/2`` iso-line, obtained with the standard marching
    squares of :mod:`matplotlib` on the **cell-centred** lattice (linear
    interpolation along the quad edges, contour closed by construction for a
    closed region).  No smoothing is applied: a stair-cased field therefore
    reports a longer contour, which is the physically meaningful degradation of
    a captured interface.

``perimeter_error``
    ``|P(T) - P(0)| / P(0)`` for a rigidly rotating field whose exact solution
    *is* the initial field — a resolution-consistent measure that does not
    inherit the pixelation bias of the initial condition.  The analytic
    perimeter of the slotted disk is reported alongside so the bias of the
    initial condition itself is visible (metric noise floor).

``front_position``
    Outermost abscissa where a column-integrated liquid depth first rises above
    a physical threshold, linearly interpolated between cell centres — the
    resolution-independent front locator used by the dam-break leg.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "liquid_area",
    "contour_area",
    "mask_area",
    "symmetric_area_error",
    "contour_perimeter",
    "perimeter_error",
    "centroid",
    "column_depth",
    "front_position",
    "slotted_disk_reference",
]


def liquid_area(alpha: torch.Tensor, dx: float, dy: float) -> float:
    """``sum(alpha) dx dy`` — the phase volume the transport actually carries."""
    return float(alpha.sum().item()) * dx * dy


def mask_area(alpha: torch.Tensor, level: float, dx: float, dy: float) -> float:
    """Area of the cell mask ``alpha > level`` (a pixelated area)."""
    return float((alpha > level).sum().item()) * dx * dy


def symmetric_area_error(
    alpha: torch.Tensor, exact: torch.Tensor, level: float = 0.5
) -> dict:
    """Thresholded misplaced area between two fields, normalised by ``exact``."""
    if alpha.shape != exact.shape:
        raise ValueError("fields must have the same shape")
    num = (alpha > level)
    ref = (exact > level)
    ref_count = int(ref.sum().item())
    if ref_count == 0:
        raise ValueError("reference field has no liquid above the level")
    xor = int(torch.logical_xor(num, ref).sum().item())
    return {
        "level": level,
        "xor_cells": xor,
        "ref_cells": ref_count,
        "symmetric_area_error": xor / ref_count,
    }


def contour_perimeter(
    alpha: torch.Tensor, dx: float, dy: float, level: float = 0.5,
    x_offset: float = 0.0, y_offset: float = 0.0,
) -> float:
    """Length of the ``alpha = level`` iso-line (matplotlib marching squares).

    Cell centres sit at ``(x_offset + (i+1/2) dx, y_offset + (j+1/2) dy)``.
    A field whose contour never crosses ``level`` (all liquid or all gas) has
    length ``0``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = alpha.detach().cpu().numpy().astype(np.float64)
    ny, nx = a.shape
    x = x_offset + (np.arange(nx) + 0.5) * dx
    y = y_offset + (np.arange(ny) + 0.5) * dy
    lo, hi = float(a.min()), float(a.max())
    if not (lo < level < hi):
        return 0.0
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(x, y, a, levels=[level])
        total = 0.0
        for seg in cs.allsegs[0]:
            pts = np.asarray(seg, dtype=np.float64)
            if pts.shape[0] > 1:
                total += float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
        return total
    finally:
        plt.close(fig)


def contour_area(alpha: torch.Tensor, dx: float, dy: float, level: float = 0.5,
                 x_offset: float = 0.0, y_offset: float = 0.0) -> float:
    """Area enclosed by the ``alpha = level`` iso-line (shoelace on the contours).

    Sub-cell measure of the *shape's* area, complementary to two others:
    ``sum(alpha) dx dy`` (the transported volume, exact for a flux-form scheme)
    and the ``alpha > level`` cell count (pixelated, so it quantises to whole
    cells).  Sum of ``|signed area|`` over the closed contours, so a shape split
    into lobes (the slotted disk) is not cancelled.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = alpha.detach().cpu().numpy().astype(np.float64)
    ny, nx = a.shape
    x = x_offset + (np.arange(nx) + 0.5) * dx
    y = y_offset + (np.arange(ny) + 0.5) * dy
    if not (float(a.min()) < level < float(a.max())):
        return 0.0
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(x, y, a, levels=[level])
        total = 0.0
        for seg in cs.allsegs[0]:
            pts = np.asarray(seg, dtype=np.float64)
            if pts.shape[0] > 2:
                xs, ys = pts[:, 0], pts[:, 1]
                total += abs(0.5 * float(np.sum(xs * np.roll(ys, -1)
                                                - np.roll(xs, -1) * ys)))
        return total
    finally:
        plt.close(fig)


def perimeter_error(alpha: torch.Tensor, reference: torch.Tensor, dx: float,
                    dy: float, level: float = 0.5) -> dict:
    """Relative change of the captured contour length against a reference field."""
    p_ref = contour_perimeter(reference, dx, dy, level)
    p = contour_perimeter(alpha, dx, dy, level)
    return {
        "perimeter": p,
        "perimeter_reference": p_ref,
        "perimeter_error": abs(p - p_ref) / p_ref if p_ref else float("nan"),
    }


def centroid(alpha: torch.Tensor, dx: float, dy: float, x_offset: float = 0.0,
             y_offset: float = 0.0) -> tuple[float, float]:
    """Centre of liquid mass ``(x, y)`` of the field."""
    a = alpha.detach().cpu().numpy().astype(np.float64)
    ny, nx = a.shape
    x = x_offset + (np.arange(nx) + 0.5) * dx
    y = y_offset + (np.arange(ny) + 0.5) * dy
    m = float(a.sum())
    if m <= 0.0:
        return float("nan"), float("nan")
    return float((a.sum(axis=0) * x).sum() / m), float((a.sum(axis=1) * y).sum() / m)


def column_depth(alpha: torch.Tensor, dy: float) -> np.ndarray:
    """Depth per column ``h_j = sum_i alpha_ij dy`` (the shallow-water variable)."""
    return alpha.detach().cpu().numpy().astype(np.float64).sum(axis=0) * dy


def front_position(depth: np.ndarray, dx: float, x_offset: float,
                   threshold: float) -> float:
    """Outermost ``x`` where ``depth`` falls through ``threshold`` (linear interp).

    Scans columns from the far end inward, so it returns the leading edge of the
    wet region.  ``nan`` when the whole profile is above or below the threshold
    (no crossing to interpolate).
    """
    h = np.asarray(depth, dtype=np.float64)
    n = h.size
    centres = x_offset + (np.arange(n) + 0.5) * dx
    if h[-1] >= threshold:
        # front beyond the last centre: extrapolate on the last interval
        if h[-1] == h[-2]:
            return float(centres[-1] + 0.5 * dx)
        slope = (h[-1] - h[-2]) / dx
        return float(centres[-1] + (h[-1] - threshold) / slope)
    if h[0] < threshold:
        return float("nan")
    for j in range(n - 1, 0, -1):
        if h[j - 1] >= threshold > h[j]:
            # crossing between centre j-1 (wet) and j (dry)
            slope = (h[j] - h[j - 1]) / dx
            return float(centres[j - 1] + (threshold - h[j - 1]) / slope)
    return float("nan")


def slotted_disk_reference(radius: float = 0.15, slot_half_width: float = 0.05) -> dict:
    """Closed-form area and perimeter of the Zalesak slotted disk.

    Geometry used by :func:`tensorfvm.verification.vof_rotation.zalesak_disk`:
    a disk of radius ``R`` with the vertical strip ``|x - x_c| <= w`` removed
    (``w`` = ``slot_half_width``), which splits it into two circular lobes.

    * area: ``pi R^2 - 2 [ w sqrt(R^2-w^2) + R^2 asin(w/R) ]``
      (strip area = integral of ``2 sqrt(R^2-x^2)`` over ``[-w, w]``);
    * perimeter: ``2 pi R - 4 R asin(w/R) + 4 sqrt(R^2-w^2)``
      (circle minus four removed arcs of angle ``asin(w/R)``, plus one straight
      chord of length ``2 sqrt(R^2-w^2)`` per lobe).
    """
    w, r = slot_half_width, radius
    chord = math.sqrt(r * r - w * w)
    half_angle = math.asin(w / r)
    area = math.pi * r * r - 2.0 * (w * chord + r * r * half_angle)
    perimeter = 2.0 * math.pi * r - 4.0 * r * half_angle + 4.0 * chord
    return {
        "radius": r,
        "slot_half_width": w,
        "area": area,
        "perimeter": perimeter,
        "chord_half_length": chord,
        "slot_half_angle": half_angle,
    }