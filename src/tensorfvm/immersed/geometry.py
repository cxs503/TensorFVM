"""Mask-only Cartesian geometries for the immersed-solid benchmarks.

A geometry is a uniform Cartesian grid plus a boolean ``solid`` array on cell
centres.  Everything else (face closures, wall distances, immersed surface) is
derived from that mask, so the *same* geometry object feeds both projection arms
of :mod:`tensorfvm.immersed.solver` and the A/B comparison is single-grid.

Conventions
-----------
``u`` faces live at ``x = x0 + i*dx`` for ``i = 0..nx`` (so ``u[:, 0]`` is the
inlet plane and ``u[:, nx]`` the outlet plane); ``v`` faces live at
``y = y0 + j*dy`` for ``j = 0..ny``.  Cell centres are at ``x0 + (i+0.5)*dx``.
Cells outside the mask are solid; domain walls (``j = 0``, ``j = ny``) are solid
*faces* handled by the operator, not by the mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform grid plus a cell-centred solid mask."""

    nx: int
    ny: int
    dx: float
    dy: float
    x0: float
    y0: float
    solid: torch.Tensor
    name: str = ""

    def __post_init__(self) -> None:
        if self.solid.shape != (self.ny, self.nx):
            raise ValueError("solid mask must have shape (ny, nx)")

    @property
    def fluid(self) -> torch.Tensor:
        return ~self.solid

    @property
    def x_faces(self) -> torch.Tensor:
        return self.x0 + torch.arange(self.nx + 1, dtype=torch.float64) * self.dx

    @property
    def y_faces(self) -> torch.Tensor:
        return self.y0 + torch.arange(self.ny + 1, dtype=torch.float64) * self.dy

    @property
    def x_centres(self) -> torch.Tensor:
        return self.x0 + (torch.arange(self.nx, dtype=torch.float64) + 0.5) * self.dx

    @property
    def y_centres(self) -> torch.Tensor:
        return self.y0 + (torch.arange(self.ny, dtype=torch.float64) + 0.5) * self.dy

    @property
    def length(self) -> float:
        return self.nx * self.dx

    @property
    def height(self) -> float:
        return self.ny * self.dy

    def summary(self) -> dict[str, object]:
        fluid = self.fluid
        return {
            "name": self.name,
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "cells": int(self.nx * self.ny),
            "fluid_cells": int(fluid.sum()),
            "solid_fraction": float(self.solid.float().mean()),
            "domain_length": self.length,
            "domain_height": self.height,
            "x0": self.x0,
            "y0": self.y0,
        }


def _grid(extent_x: float, extent_y: float, x0: float, y0: float,
          solid_of, name: str, h: float) -> CartesianGrid:
    nx = int(round(extent_x / h))
    ny = int(round(extent_y / h))
    if nx < 8 or ny < 4:
        raise ValueError("grid too small; increase cells_per_h or the domain extent")
    dx = extent_x / nx
    dy = extent_y / ny
    xc = x0 + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    yc = y0 + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    solid = solid_of(xc[None, :], yc[:, None])
    if not bool((~solid).any()):
        raise ValueError("geometry produced no fluid cells")
    return CartesianGrid(nx=nx, ny=ny, dx=dx, dy=dy, x0=x0, y0=y0,
                         solid=solid.bool(), name=name)


def backward_facing_step(cells_per_h: int = 40, *, step_height: float = 1.0,
                         inlet_height: float = 1.0,
                         upstream_lengths: float = 3.0,
                         downstream_lengths: float = 17.0) -> CartesianGrid:
    """Armaly-type backward-facing step, expansion ratio ``(S + H_in)/H_in``.

    With the defaults ``S = H_in = 1`` the expansion ratio is exactly 2 and the
    step height ``H = 1`` is the reference length for ``cells_per_h``.  The inlet
    channel occupies ``S <= y <= S + H_in`` upstream of the step, the step block
    ``x < 0, y < S`` is solid.
    """
    if cells_per_h < 4:
        raise ValueError("cells_per_h must be at least 4")
    s = step_height
    h = s / cells_per_h
    extent_x = upstream_lengths * inlet_height + downstream_lengths * s
    extent_y = s + inlet_height
    x0 = -upstream_lengths * inlet_height
    y0 = 0.0

    def solid_of(xc, yc):
        return (xc < 0.0) & (yc < s)

    grid = _grid(extent_x, extent_y, x0, y0, solid_of,
                 "backward_facing_step", h)
    # Snap the step plane onto a grid line: the mask must not straddle it.
    xs = grid.x_faces
    if not bool(torch.isclose(xs, torch.zeros_like(xs)).any()):
        raise ValueError("step plane did not land on a grid line")
    return grid


def symmetric_expansion(cells_per_h: int = 40, *, inlet_height: float = 1.0,
                        area_ratio: float = 2.0,
                        upstream_lengths: float = 4.0,
                        downstream_lengths: float = 16.0) -> CartesianGrid:
    """Planar *symmetric* sudden expansion of area ratio ``area_ratio``.

    ``cells_per_h`` is cells per inlet-channel height.  The upstream channel is
    centred, so the two side walls of thickness ``(H_out - H_in)/2`` upstream of
    the expansion plane are solid.
    """
    if cells_per_h < 4:
        raise ValueError("cells_per_h must be at least 4")
    if area_ratio <= 1.0:
        raise ValueError("area_ratio must exceed 1")
    h = inlet_height / cells_per_h
    h_out = inlet_height * area_ratio
    extent_x = upstream_lengths * inlet_height + downstream_lengths * inlet_height
    x0 = -upstream_lengths * inlet_height
    y0 = 0.0
    half_in = 0.5 * inlet_height
    mid = 0.5 * h_out

    def solid_of(xc, yc):
        return (xc < 0.0) & ((yc - mid).abs() > half_in)

    grid = _grid(extent_x, h_out, x0, y0, solid_of,
                 "symmetric_expansion", h)
    xs = grid.x_faces
    if not bool(torch.isclose(xs, torch.zeros_like(xs)).any()):
        raise ValueError("expansion plane did not land on a grid line")
    return grid


def parabolic_inlet(grid: CartesianGrid, mean_velocity: float = 1.0,
                    lower: float = 0.0, upper: float = 1.0) -> torch.Tensor:
    """Poiseuille profile on the ``u`` faces of the inlet plane.

    ``lower``/``upper`` bound the open part of the inlet plane (same length unit
    as the grid coordinates).  Entries outside the open part are zero, i.e. the
    inlet plane closes against the solid/wall there.
    """
    y = grid.y_faces[:-1] + 0.5 * grid.dy
    height = upper - lower
    if height <= 0.0:
        raise ValueError("upper must exceed lower")
    eta = (y - lower) / height
    shape = 6.0 * eta * (1.0 - eta)
    open_rows = (y > lower) & (y < upper)
    profile = torch.where(open_rows, shape, torch.zeros_like(y))
    # normalise over the *open* part of the plane, so ``mean_velocity`` is the
    # mean velocity through the inlet and the inlet flux is
    # ``mean_velocity * open_height`` (that is what fixes Re_Dh).
    flux = float(profile[open_rows].mean())
    if flux <= 0.0:
        raise ValueError("inlet profile has no open faces")
    return mean_velocity * profile / flux


def step_position(grid: CartesianGrid) -> float:
    """x coordinate of the step/expansion plane (the grid line nearest ``x = 0``)."""
    xs = grid.x_faces
    return float(xs[torch.argmin(xs.abs())])


def expansion_ratio_realised(grid: CartesianGrid, upstream_probe: float = -0.5,
                             downstream_probe: float = 1.0) -> float:
    """Measure ``H_out / H_in`` from the mask, never from the setup recipe."""
    xc = grid.x_centres
    up = grid.fluid[:, (xc > upstream_probe - 1.0) & (xc < upstream_probe)]
    down = grid.fluid[:, (xc > downstream_probe) & (xc < downstream_probe + 1.0)]
    if up.numel() == 0 or down.numel() == 0:
        raise ValueError("probe windows contain no columns")
    h_in = float(up.float().sum(dim=0).mean()) * grid.dy
    h_out = float(down.float().sum(dim=0).mean()) * grid.dy
    if h_in <= 0.0:
        raise ValueError("no fluid at the upstream probe")
    return h_out / h_in


def best_resolution(extent: float, cells: int) -> float:
    """Grid spacing that divides ``extent`` into exactly ``cells`` cells."""
    if cells < 1:
        raise ValueError("cells must be positive")
    return extent / cells


def required_cells(cells_per_h: int, length_in_h: float) -> int:
    """Cells along a direction of ``length_in_h`` step heights (rounded)."""
    return int(math.ceil(cells_per_h * length_in_h - 1e-9))