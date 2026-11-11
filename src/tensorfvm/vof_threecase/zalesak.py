"""Zalesak slotted disk over 360 deg and 1000 deg: notch shape preservation.

Case statement
--------------
A slotted disk (Zalesak 1979) is carried by the divergence-free rigid rotation
``u = -omega (y - y_c)``, ``v = omega (x - x_c)``.  The exact solution after an
integer number of revolutions is the initial field, so the *shape* error is
resolution-consistent: no pixelation bias is charged to the scheme.

Why the notch area is the criterion quantity
--------------------------------------------
A global area error is blind to the slot: it measures "how much liquid", not
"is the notch still there".  This module therefore measures the **notch**: the
missing liquid inside ``|x - x_c| <= slot_half_width`` and ``r <= R``, i.e.

.. math::  A_{notch}(\\alpha) = \\sum_{cells \\in slot \\cap disk} (1-\\alpha) \\, \\Delta x^2 ,

with the cell mask fixed by geometry (so the numerator and the reference use
the same cells).  ``A_notch`` starts at its exact discrete value, and after a
rigid rotation it must return to it: the relative error is the shape metric the
node's criterion is written in (<= 5 % at 360 deg, <= 15 % at 1000 deg).

The long rotation is the point.  One revolution can be passed by a scheme whose
interface is still diffusing (the slot is smeared but the *enclosed* area
recovers); over 1000 deg (2.78 revolutions) the interface is crossed by the
rotation three times and every crossing adds to the notch's smearing, so the
notch metric separates a shape-preserving scheme from a merely conservative one.

Transport is the repository's own algebraic CICSAM/HRIC and geometric PLIC
(:func:`tensorfvm.twophase_bench.zalesak.run_rotation`), imported rather than
re-implemented so this node cannot drift from the registered benchmark face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..twophase_bench.zalesak import run_rotation

__all__ = ["NotchConfig", "notch_area", "notch_fill", "notch_mask", "run_long_rotation"]

DEG = 360.0


@dataclass
class NotchConfig:
    """One (scheme, resolution, angle) sample."""

    cells: int = 128
    scheme: str = "cicsam"
    #: Total rotation in degrees.
    degrees: float = 360.0
    cfl: float = 0.4
    #: Report the notch at every intermediate full revolution (history).
    track_every: int = 25


def notch_mask(cells: int, domain: float = 1.0, radius: float = 0.15,
               slot_half_width: float = 0.05, degrees: float = 0.0) -> torch.Tensor:
    """Cells of the notch at rotation angle ``degrees``.

    The notch is ``slot ∩ disk`` *in the co-rotating frame*: for a cell centre
    ``p`` at time ``T`` the mask tests ``R(-omega T) (p - c) + c``, because the
    rigid rotation ``u = -omega (y-y_c)``, ``v = +omega (x-x_c)`` carries a
    material point counterclockwise, ``p(T) = R(+omega T) p(0)``.  At a
    multiple of 360 deg the mask is the initial one, so the 360-deg criterion is
    unaffected; at 1000 deg (2 turns + 280 deg) the notch really is somewhere
    else, and comparing against the *unrotated* notch would be comparing
    different geometry (the first version of this metric did exactly that and
    reported a meaningless 0.00 %).
    """
    dx = domain / cells
    c = (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    xx, yy = torch.meshgrid(c, c, indexing="xy")
    cx, cy = domain / 2, 0.75 * domain
    if abs(degrees) > 0.0:
        th = math.radians(-degrees)
        ct, st = math.cos(th), math.sin(th)
        dxr = xx - domain / 2
        dyr = yy - domain / 2
        xr = ct * dxr - st * dyr + domain / 2
        yr = st * dxr + ct * dyr + domain / 2
    else:
        xr, yr = xx, yy
    r = torch.sqrt((xr - cx) ** 2 + (yr - cy) ** 2)
    return ((xr - cx).abs() <= slot_half_width) & (r <= radius)


def notch_fill(alpha: torch.Tensor, cells: int, domain: float = 1.0,
               degrees: float = 0.0) -> float:
    """Vacant fraction of the notch (1 = a clean notch, 0 = filled in)."""
    mask = notch_mask(cells, domain, degrees=degrees)
    return float((1.0 - alpha)[mask].mean())


def notch_area(alpha: torch.Tensor, cells: int, domain: float = 1.0,
               degrees: float = 0.0) -> float:
    """Missing liquid inside the notch, in area units (diagnostic)."""
    mask = notch_mask(cells, domain, degrees=degrees)
    dx = domain / cells
    return float((1.0 - alpha)[mask].sum()) * dx * dx


def run_long_rotation(cfg: NotchConfig | None = None, **overrides) -> dict:
    """Rotate the slotted disk by ``cfg.degrees`` and measure the notch error.

    Returns a JSON-ready record.  ``notch_error_relative`` is the criterion
    quantity; ``area_error_contour_vs_initial`` / ``perimeter_error_analytic``
    and the peak-retention / L1 diagnostics are carried alongside so the
    global and local views of the same run stay side by side.
    """
    cfg = cfg or NotchConfig()
    if overrides:
        cfg = NotchConfig(**{**cfg.__dict__, **overrides})
    revolutions = cfg.degrees / DEG
    res = run_rotation(
        cells=cfg.cells,
        scheme=cfg.scheme,
        cfl=cfg.cfl,
        revolutions=revolutions,
        track_every=cfg.track_every,
    )
    alpha = res.pop("_field")
    exact = res.pop("_exact")
    # reference: the notch of the *co-rotating* mask, measured on the initial
    # field (which is exactly empty there, so the reference fill is 1)
    f0 = notch_fill(exact, cfg.cells, degrees=cfg.degrees)
    f1 = notch_fill(alpha, cfg.cells, degrees=cfg.degrees)
    a0 = notch_area(exact, cfg.cells, degrees=cfg.degrees)
    a1 = notch_area(alpha, cfg.cells, degrees=cfg.degrees)
    return {
        "case": "zalesak_notch",
        "scheme": cfg.scheme,
        "cells": cfg.cells,
        "degrees": cfg.degrees,
        "revolutions": revolutions,
        "cfl": cfg.cfl,
        "steps": res["steps"],
        "notch_area_initial": a0,
        "notch_area_final": a1,
        "notch_area_absolute_error": abs(a1 - a0),
        "notch_fill_initial": f0,
        "notch_fill_final": f1,
        # criterion quantity: the notch's vacant fraction against its initial
        # value (exactly 1), i.e. the relative area error of the notch
        "notch_error_relative": abs(f1 - f0) / f0 if f0 > 0 else float("nan"),
        **res,
        "_field": alpha,
        "_exact": exact,
    }