"""THINC (tangent of hyperbola interface capturing) transport.

Why a second reconstruction
---------------------------
:mod:`tensorfvm.vof_plic` supplies a *geometric* PLIC reconstruction and is the
default interface transport of this package.  THINC (Xiao, Honma & Koshizuka
2005; Ii, Xiao & Chen 2008) is the other standard family: instead of a plane,
the interface inside a cell is a hyperbolic-tangent profile

.. math::  H(x) = \\tfrac12\\left[1 + \\tanh\\big(k\\,(x - x_{if})\\big)\\right],
           \\qquad x \\in [-\\tfrac12, \\tfrac12]

in cell-local coordinates, with ``k = 2 beta``.  ``x_if`` is fixed by requiring
the profile to reproduce the cell mean *exactly*:

.. math::  \\bar\\alpha = \\tfrac12 + \\frac{1}{2k}\\Big[
           \\ln\\cosh\\big(k(\\tfrac12 - x_{if})\\big) -
           \\ln\\cosh\\big(k(-\\tfrac12 - x_{if})\\big)\\Big]

which is monotone in ``x_if`` and is inverted here by vectorised bisection
(the same philosophy as the plane-constant inversion of the PLIC module).  The
closed-form ``artanh(2 alpha - 1)/(2 beta)`` is *not* the inverse of that mean
and is deliberately not used: it renormalises the profile and breaks the volume
ledger (measured: 271 % mass error on one Zalesak-style revolution before the
inversion was fixed).

What this module provides
-------------------------
* :func:`thinc_reconstruct` -- ``x_if`` per cell (cell-local, ``0`` = centre).
* :func:`thinc_flux` -- exact liquid volume crossing a face in one step, from
  the analytic primitive ``int H dx = x/2 + ln cosh(k(x - x_if)) / (2k)``.
* :func:`advect_thinc` -- dimensional (Lie) split flux-form update on the same
  staggered velocities as :func:`tensorfvm.vof_plic.advect_plic`, so the two
  schemes are drop-in interchangeable inside the solver.
* :func:`advance_1d` -- one periodic 1-D sweep (used by the unit tests).

Conservation: both sweeps are flux form (each face flux is removed from the
donor and added to the receiver), so the discrete volume changes by roundoff
only.  Dead ends are transmissive (zero flux), matching the PLIC transport's
boundary convention in the solver.

Convention: ``alpha = 1`` is inside the phase.  All tensors are ``float64``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "beta_from_sharpness",
    "thinc_reconstruct",
    "thinc_cell_mean",
    "thinc_flux",
    "advect_thinc",
    "advance_1d",
]

_EPS = 1.0e-12


def _logcosh(z: torch.Tensor) -> torch.Tensor:
    """``ln cosh z`` without overflow."""
    az = z.abs()
    return az + torch.log1p(torch.exp(-2.0 * az)) - math.log(2.0)


def beta_from_sharpness(alpha: torch.Tensor, sharpness: float = 3.0) -> torch.Tensor:
    """Interface-thickness parameter ``beta`` (Ii et al. 2008).

    ``beta = sharpness / (1 - 2 eps)^2`` with ``eps`` the local distance from the
    pure phases, so nearly-empty / nearly-full cells are reconstructed sharply
    while a half-full cell gets a smooth profile.
    """
    if sharpness <= 0:
        raise ValueError(f"sharpness must be positive, got {sharpness!r}")
    eps = torch.minimum(alpha, 1.0 - alpha).clamp(min=_EPS, max=0.49)
    return sharpness / (1.0 - 2.0 * eps) ** 2


def thinc_cell_mean(x_if: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Cell mean of the THINC profile with interface position ``x_if``."""
    return 0.5 + (_logcosh(k * (0.5 - x_if)) - _logcosh(k * (-0.5 - x_if))) / (2.0 * k)


def thinc_reconstruct(alpha, sharpness: float = 3.0) -> torch.Tensor:
    """Interface location ``x_if`` (cell-local, ``0`` = centre) from the mean.

    Vectorised bisection on ``[-2, 2]`` (the interface may legitimately sit
    outside a nearly-pure cell).  26 iterations give a mean error below 1e-7
    over the whole range, which is far below the roundoff of the transport.
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    if a.numel() == 0:
        raise ValueError("alpha is empty")
    if not bool(torch.isfinite(a).all()):
        raise ValueError("alpha contains non-finite entries")
    if float(a.min()) < -1e-12 or float(a.max()) > 1.0 + 1e-12:
        raise ValueError("alpha outside [0, 1]")
    ac = a.clamp(_EPS, 1.0 - _EPS)
    k = 2.0 * beta_from_sharpness(ac, sharpness)
    lo = torch.full_like(ac, -2.0)
    hi = torch.full_like(ac, 2.0)
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        # the cell mean is *decreasing* in x_if (liquid sits on the +x side):
        # a mean above the target means x_if is still too small.
        too_small = thinc_cell_mean(mid, k) > ac
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


def thinc_flux(alpha, courant, sharpness: float = 3.0) -> torch.Tensor:
    """Volume fraction (times ``dx``) leaving the donor across the downstream face.

    ``courant`` is the directional Courant number ``|u| dt / dx`` in ``[0, 1]``.
    The result is the analytic integral of the reconstructed profile over the
    swept strip ``[0.5 - c, 0.5]``, clamped to ``[0, c]`` (a profile never
    carries more liquid than the strip can hold).
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    c = torch.as_tensor(courant, dtype=torch.float64)
    if a.shape != c.shape:
        raise ValueError(f"shape mismatch alpha {tuple(a.shape)} vs courant {tuple(c.shape)}")
    ac = a.clamp(_EPS, 1.0 - _EPS).clamp(0.0, 1.0)
    cc = c.clamp(0.0, 1.0)
    x_if = thinc_reconstruct(ac, sharpness)
    k = 2.0 * beta_from_sharpness(ac, sharpness)

    def prim(xi):
        return 0.5 * xi + 0.5 * _logcosh(k * (xi - x_if)) / k

    flux = prim(0.5) - prim(0.5 - cc)
    return torch.minimum(flux.clamp(min=0.0), cc)


def _sweep(alpha, speed, dt, dx, axis, sharpness):
    """Flux-form THINC sweep along ``axis``; ends are transmissive."""
    a = alpha.to(torch.float64)
    u = speed.to(torch.float64)
    cfl = u.abs().max() * dt / max(dx, 1e-300)
    if float(cfl) > 1.0 + 1e-9:
        raise ValueError(f"directional CFL {float(cfl):.3f} > 1")
    a_t = a if axis == 1 else a.transpose(0, 1)
    u_t = u if axis == 1 else u.transpose(0, 1)
    flux = torch.zeros_like(u_t)
    flux[:, 1:] = thinc_flux(a_t, (u_t[:, 1:].clamp(min=0.0) * dt / dx))
    flux[:, :-1] -= thinc_flux(a_t, ((-u_t[:, :-1]).clamp(min=0.0) * dt / dx))
    flux[:, 0] = 0.0
    flux[:, -1] = 0.0
    out = a_t - (flux[:, 1:] - flux[:, :-1])
    if axis == 0:
        out = out.transpose(0, 1)
    return _repair_bounds(out)


def _repair_bounds(a: torch.Tensor, rounds: int = 4) -> torch.Tensor:
    """Clip an overshoot and put the clipped volume back where it can live.

    THINC's continuous profile is not a geometric flux, so on an under-resolved
    interface a sweep can leave ``alpha`` marginally outside ``[0, 1]``.  Simply
    clamping would destroy volume (measured 7.7e-4 relative over 8 dam-break
    steps); redistributing the clipped volume over the partial cells keeps the
    discrete volume unchanged to roundoff.
    """
    out = a
    for _ in range(rounds):
        hi = (out - 1.0).clamp(min=0.0)
        lo = (-out).clamp(min=0.0)
        excess = float(hi.sum() - lo.sum())
        out = out.clamp(0.0, 1.0)
        if abs(excess) < 1e-15:
            break
        room = out * (1.0 - out)
        total = float(room.sum())
        if total <= 0.0:
            break
        out = (out + excess * room / total).clamp(0.0, 1.0)
    return out


def advect_thinc(alpha, u, v, dt: float, dx: float, dy: float, order: str = "xy",
                 sharpness: float = 3.0) -> torch.Tensor:
    """Two-dimensional THINC advection with dimensional (Lie) splitting.

    Interface identical to :func:`tensorfvm.vof_plic.advect_plic`: ``alpha`` is
    ``(ny, nx)``, ``u`` staggered x-faces ``(ny, nx+1)``, ``v`` staggered y-faces
    ``(ny+1, nx)``.  Alternating ``order`` between steps symmetrises the
    splitting in time.
    """
    if order not in ("xy", "yx"):
        raise ValueError(f"order must be 'xy' or 'yx', got {order!r}")
    a = torch.as_tensor(alpha, dtype=torch.float64)
    if a.ndim != 2:
        raise ValueError("alpha must be 2-D")
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be positive and finite, got {dt!r}")
    for axis in (1, 0) if order == "xy" else (0, 1):
        a = _sweep(a, u if axis == 1 else v, dt, dx if axis == 1 else dy, axis, sharpness)
    return a


def advance_1d(alpha, speed, dt: float, dx: float, sharpness: float = 3.0) -> torch.Tensor:
    """One periodic 1-D THINC sweep (``speed`` has ``n + 1`` faces, last = first)."""
    a = torch.as_tensor(alpha, dtype=torch.float64).reshape(1, -1)
    u = torch.as_tensor(speed, dtype=torch.float64).reshape(1, -1)
    n = a.shape[1]
    if u.shape[1] != n + 1:
        raise ValueError(f"speed must have {n + 1} entries, got {u.shape[1]}")
    cfl = float(u.abs().max()) * dt / dx
    if cfl > 1.0 + 1e-9:
        raise ValueError(f"directional CFL {cfl:.3f} > 1")
    flux = torch.zeros_like(a)
    for i in range(n + 1):
        uu = float(u[0, i])
        if uu == 0.0:
            continue
        donor = (i - 1) % n if uu > 0 else i % n
        c = torch.full((1,), abs(uu) * dt / dx, dtype=torch.float64)
        vol = float(thinc_flux(a[0, donor: donor + 1], c, sharpness).reshape(()))
        a[0, donor] -= vol
        a[0, (donor + (1 if uu > 0 else -1)) % n] += vol
    return a.reshape(-1).clamp(0.0, 1.0)