"""Slope limiting and the MUSCL-Hancock predictor (van Leer 1979; Toro ch. 14).

The compressible surface uses a single reconstruction path: primitive-variable
slopes with a TVD limiter, a half-time Hancock predictor that advances each cell
by ``dt/2`` using the flux difference of its own two reconstructed face values,
and then a second pass of the limited slopes on the predicted cell average.  The
result is second order in space and time for smooth flow and TVD across shocks.
"""

from __future__ import annotations

import torch

__all__ = ["minmod", "vanleer", "mc", "superbee", "LIMITERS", "limited_slope"]


def minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minmod limiter: most diffusive of the TVD family, most robust."""
    return torch.where(
        a * b > 0.0,
        torch.where(a.abs() < b.abs(), a, b),
        torch.zeros_like(a),
    )


def vanleer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """van Leer harmonic limiter."""
    return torch.where(
        a * b > 0.0,
        2.0 * a * b / (a + b),
        torch.zeros_like(a),
    )


def mc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Monotonized-central limiter: sharpest of the three, still TVD."""
    centered = 0.5 * (a + b)
    lim = torch.minimum(2.0 * a.abs(), 2.0 * b.abs())
    lim = torch.minimum(lim, centered.abs())
    return torch.sign(centered) * lim


def superbee(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Superbee limiter (Roe 1986): the least diffusive member of the TVD family.

    ``phi(r) = max(0, min(2r, 1), min(r, 2))`` written on the two slopes
    directly: ``maxmod(minmod(2a, b), minmod(a, 2b))``.
    """
    s1 = minmod(2.0 * a, b)
    s2 = minmod(a, 2.0 * b)
    return torch.where(s1.abs() >= s2.abs(), s1, s2)


LIMITERS = {"minmod": minmod, "vanleer": vanleer, "mc": mc, "superbee": superbee}


def limited_slope(q: torch.Tensor, axis: int, limiter: str = "mc") -> torch.Tensor:
    """Limited slope along ``axis`` of a padded cell-centred field.

    ``q`` must carry at least two ghost cells on each side along ``axis``; the
    returned tensor has the same shape and is zero at the two outermost layers
    (which never enter an interior face stencil).  Slicing is used instead of
    ``roll`` so no wraparound copy is made per call.
    """
    if limiter not in LIMITERS:
        raise ValueError(f"unknown limiter {limiter!r}; choose from {sorted(LIMITERS)}")
    lim = LIMITERS[limiter]
    slope = torch.zeros_like(q)
    if axis == 0:
        fwd = q[2:, :] - q[1:-1, :]
        bwd = q[1:-1, :] - q[:-2, :]
        slope[1:-1, :] = lim(fwd, bwd)
    elif axis == 1:
        fwd = q[:, 2:] - q[:, 1:-1]
        bwd = q[:, 1:-1] - q[:, :-2]
        slope[:, 1:-1] = lim(fwd, bwd)
    else:
        raise ValueError("axis must be 0 or 1")
    return slope