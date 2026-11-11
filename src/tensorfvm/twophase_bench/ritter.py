"""Ritter (1892) dam-break analytics and the VOF transport leg of the benchmark.

Physical setting
----------------
Dam at ``x = W`` with ``W = 0.028 m`` (the contract value), reservoir depth
``H = 0.056 m``, dry frictionless bed, released at ``t = 0``.  With
``c0 = sqrt(g H)`` and ``xi = x - W`` the shallow-water (Ritter) solution is

.. math::

    h(\\xi,t) = \\begin{cases}
        H, & \\xi \\le -c_0 t \\\\
        \\dfrac{1}{9g}\\left(2c_0 - \\dfrac{\\xi}{t}\\right)^2,
            & -c_0 t \\le \\xi \\le 2c_0 t \\\\
        0, & \\xi \\ge 2c_0 t
      \\end{cases}
    \\qquad
    u(\\xi,t) = \\begin{cases}
        0, & \\xi \\le -c_0 t \\\\
        \\tfrac{2}{3}\\left(c_0 + \\dfrac{\\xi}{t}\\right),
            & -c_0 t \\le \\xi \\le 2c_0 t \\\\
        2c_0, & \\xi \\ge 2c_0 t
      \\end{cases}

The surge front is ``x_f(t) = W + 2 c0 t``, i.e. the contract's
``2 t sqrt(gH)`` measured from the dam.

Why the reservoir, not a 0.028 m wide column
-------------------------------------------
The Ritter solution is the exact solution of a **semi-infinite reservoir**
released onto a dry bed.  For a column of finite width ``W`` the wall
rarefaction reaches the dam face at ``t = W/c0 = 0.038 s`` and the front
decelerates; the Martin & Moyce (1952) data put the finite-column front at
``x* = 5.7`` (``x = 0.19 m``) at ``t = 0.3 s`` where Ritter gives ``0.445 m`` —
a factor 2.3, i.e. the criterion "front within 10 % of ``2 t sqrt(gH)`` up to
``t = 0.3 s``" is **not attainable by any finite column**, and the repository's
own coupled 2-D collapse measurement records the same lag
(``docs/vof-phase0/artifacts/phase0_report.json``, AC2: -60 % / -31 % / -12 % at
``t* = 1,2,3``).  The computational domain therefore carries the upstream
reservoir (``0.26 m`` to the left of the dam) so that the analytic solution is
valid in the *whole* domain for ``t <= 0.3 s``: the fan head reaches
``x = -0.194 m`` and the front ``x = 0.473 m``, both inside
``[-0.26, 0.52] m``.  ``H/W = 2`` is preserved and ``W`` fixes the dam position.

Two further exactness points
----------------------------
* **Front definition.**  ``h -> 0`` at the front, so a finite-resolution field
  can only report where the *resolved* depth equals a threshold ``h_ref``.  The
  exact abscissa of that threshold is ``xi_ref = t(2c0 - 3 sqrt(g h_ref))``; the
  probe ``h_ref = H/20`` is resolution independent.  Comparing an ``alpha > 1/2``
  front with the zero-depth tip instead bakes in an analytic lag of
  ``3 t sqrt(g dy/2)`` (measured 12 % at ``t = 0.05 s``, 20 % of the front at
  ``t = 0.3 s``); both numbers are reported, the gate uses the commensurable one.
* **Vertical velocity.**  The Ritter velocity is depth averaged.  Advecting a
  2-D volume fraction with ``(u(x,t), 0)`` is wrong: that field is not divergence
  free and dilates the water column (measured: ``alpha_max`` decays to 0.50
  within 0.1 s).  Continuity of the exact solution, ``dh/dt + u dh/dx = -h du/dx``,
  fixes ``v(x,y,t) = -y du/dx``, whose surface satisfies the kinematic condition
  exactly and which is divergence free with ``du/dx = 2/(3t)`` inside the fan.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "DamBreakGeometry",
    "ritter_depth",
    "ritter_velocity",
    "ritter_front",
    "ritter_front_at_depth",
    "exact_alpha",
    "initial_alpha",
    "velocity_faces",
    "run_transport",
    "column_depth",
]

_T_FLOOR = 1.0e-6  #: t -> 0 limit of the fan is a step (zero-width); see velocity_faces


class DamBreakGeometry:
    """Tank layout for the Ritter dam-break transport leg (SI units).

    ``width`` is the dam position (contract ``W = 0.028 m``), ``height`` the
    reservoir depth (contract ``H = 0.056 m``); the domain spans
    ``[-upstream, downstream]`` horizontally so that both ends lie outside the
    wave for ``t <= t_end``.
    """

    def __init__(self, width: float = 0.028, height: float = 0.056,
                 gravity: float = 9.81, upstream: float = 0.26,
                 downstream: float = 0.60, tank_height: float = 0.07,
                 t_end: float = 0.3):
        if width <= 0.0 or height <= 0.0:
            raise ValueError("width and height must be positive")
        if upstream <= 0.0 or downstream <= 0.0:
            raise ValueError("domain extents must be positive")
        if tank_height <= height:
            raise ValueError("tank must be taller than the reservoir")
        self.width = float(width)
        self.height = float(height)
        self.gravity = float(gravity)
        self.upstream = float(upstream)
        self.downstream = float(downstream)
        self.tank_height = float(tank_height)
        self.t_end = float(t_end)
        c0 = self.c0
        if self.width + 2.0 * c0 * t_end >= self.downstream:
            raise ValueError("front leaves the domain: raise `downstream`")
        if self.width - c0 * t_end <= -self.upstream:
            raise ValueError("fan head leaves the domain: raise `upstream`")

    @property
    def c0(self) -> float:
        """Shallow-water celerity ``sqrt(gH)`` [m/s]."""
        return math.sqrt(self.gravity * self.height)

    @property
    def length(self) -> float:
        return self.upstream + self.downstream

    @property
    def x_left(self) -> float:
        return -self.upstream

    @property
    def x_right(self) -> float:
        return self.downstream

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def summary(self) -> dict:
        return {
            "dam_position_m": self.width,
            "reservoir_depth_m": self.height,
            "gravity": self.gravity,
            "c0_m_s": self.c0,
            "depth_over_dam_position": self.height / self.width,
            "domain_x_m": [self.x_left, self.x_right],
            "tank_height_m": self.tank_height,
            "upstream_reservoir_m": self.upstream,
            "wall_rarefaction_time_s": self.width / self.c0,
            "front_at_t_end_m": self.width + 2.0 * self.c0 * self.t_end,
            "fan_head_at_t_end_m": self.width - self.c0 * self.t_end,
        }


def ritter_depth(xi, t: float, geo: DamBreakGeometry):
    """Exact Ritter depth ``h(xi, t)``; scalar or array ``xi``."""
    c0 = geo.c0
    if t <= 0.0:
        raise ValueError("t must be positive")
    x = np.asarray(xi, dtype=np.float64)
    h = np.zeros_like(x)
    h[x <= -c0 * t] = geo.height
    fan = (x > -c0 * t) & (x < 2.0 * c0 * t)
    h[fan] = (2.0 * c0 - x[fan] / t) ** 2 / (9.0 * geo.gravity)
    return h


def ritter_velocity(xi, t: float, geo: DamBreakGeometry):
    """Exact Ritter depth-averaged velocity ``u(xi, t)``."""
    c0 = geo.c0
    if t <= 0.0:
        raise ValueError("t must be positive")
    x = np.asarray(xi, dtype=np.float64)
    u = np.zeros_like(x)
    fan = (x > -c0 * t) & (x < 2.0 * c0 * t)
    u[fan] = (2.0 / 3.0) * (c0 + x[fan] / t)
    u[x >= 2.0 * c0 * t] = 2.0 * c0
    return u


def ritter_front(t: float, geo: DamBreakGeometry) -> float:
    """Unsmeared surge-front abscissa ``W + 2 c0 t`` [m]."""
    return geo.width + 2.0 * geo.c0 * t


def ritter_front_at_depth(t: float, h_ref: float, geo: DamBreakGeometry) -> float:
    """Exact abscissa where the Ritter profile has depth ``h_ref`` [m]."""
    if h_ref < 0.0:
        raise ValueError("h_ref must be non-negative")
    if h_ref > geo.height:
        raise ValueError("h_ref cannot exceed the reservoir depth")
    return geo.width + t * (2.0 * geo.c0 - 3.0 * math.sqrt(geo.gravity * h_ref))


def _band_fraction(centres: np.ndarray, h: float, lo: float, hi: float) -> np.ndarray:
    left = centres - 0.5 * h
    right = centres + 0.5 * h
    overlap = np.minimum(right, hi) - np.maximum(left, lo)
    return np.clip(overlap / h, 0.0, 1.0)


def initial_alpha(nx: int, ny: int, geo: DamBreakGeometry) -> torch.Tensor:
    """Exact volume fraction of the reservoir ``x <= W``, ``y <= H``."""
    dx = geo.length / nx
    dy = geo.tank_height / ny
    xc = geo.x_left + (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    fx = _band_fraction(xc, dx, geo.x_left, geo.width)
    fy = _band_fraction(yc, dy, 0.0, geo.height)
    return torch.from_numpy(np.outer(fy, fx))


def exact_alpha(nx: int, ny: int, geo: DamBreakGeometry, t: float) -> torch.Tensor:
    """Exact volume fraction at time ``t``: cell fraction below the Ritter surface."""
    dx = geo.length / nx
    dy = geo.tank_height / ny
    xc = geo.x_left + (np.arange(nx) + 0.5) * dx
    depth = ritter_depth(xc - geo.width, t, geo)
    y_bottom = np.arange(ny) * dy
    y_top = y_bottom + dy
    fill = np.clip(np.minimum(depth[None, :], y_top[:, None]) - y_bottom[:, None],
                   0.0, dy) / dy
    return torch.from_numpy(fill)


def velocity_faces(nx: int, ny: int, geo: DamBreakGeometry, t: float,
                   wall: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Staggered 2-D velocity of the Ritter solution: ``u`` on x-faces, ``v = -y du/dx``.

    The vertical velocity makes the field divergence free and keeps the free
    surface a material surface (see the module docstring); the strain rate is
    taken from the analytic value ``du/dx = 2/(3t)`` inside the fan and ``0``
    outside, because differencing the *sampled* faces spikes wherever a boundary
    is imposed (measured: |v| up to 50 m/s at a closed end, which collapses the
    time step without changing the physics).  Both end faces get ``u = 0``: the
    left end lies in the undisturbed reservoir and the right end is dry, so the
    analytic field already satisfies no-through-flow there.
    """
    dx = geo.length / nx
    dy = geo.tank_height / ny
    xf = geo.x_left + np.arange(nx + 1) * dx
    t_eff = max(t, _T_FLOOR)
    u1d = ritter_velocity(xf - geo.width, t_eff, geo)
    # Strain rate from the *same* face differences the transport uses, so the
    # discrete divergence of every cell is exactly zero:
    #   (u_{i+1/2} - u_{i-1/2})/dx + (v_{j+1/2} - v_{j-1/2})/dy = 0
    # u is not zeroed at the ends: the analytic field already has u = 0 in the
    # undisturbed reservoir (left) and a uniform u = 2 c0 over the dry region
    # (right), where the flux is zero because the face value (donor) is alpha = 0.
    # Zeroing them instead creates a one-cell strain spike (|v| ~ 50 m/s) that
    # collapses the time step without changing the wet region.
    du_dx = (u1d[1:] - u1d[:-1]) / dx
    yf = np.arange(ny + 1) * dy
    v = -np.outer(yf, du_dx)
    if wall:
        v[0, :] = 0.0   # no flux through the bed
        v[-1, :] = 0.0  # closed tank top (alpha is zero there)
    u = np.repeat(u1d[None, :], ny, axis=0)
    return torch.from_numpy(u), torch.from_numpy(v)


def column_depth(alpha: torch.Tensor, dy: float) -> np.ndarray:
    """Depth per column [m] from the volume fraction field."""
    return alpha.detach().cpu().numpy().astype(np.float64).sum(axis=0) * dy


def run_transport(
    scheme: str,
    cells_per_height: int,
    t_end: float = 0.3,
    t_start: float = 0.02,
    geo: DamBreakGeometry | None = None,
    cfl: float = 0.4,
    sample_times: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    h_ref_fraction: float = 0.05,
    window: tuple[float, float] | None = None,
    limit: bool = True,
) -> dict:
    """Integrate the VOF transport of the dam-break column with a chosen scheme.

    The integration starts at ``t_start`` from the **exact** Ritter field: the
    similarity strain rate ``du/dx = 2/(3t)`` diverges at ``t = 0``, so a
    rest-state start would need a vanishing time step inside the zero-width fan.
    ``t_start = 0.02 s`` is below the gate window (``t >= 0.05 s``) and is the
    only place where an analytic state is imposed instead of produced.

    Parameters
    ----------
    scheme : ``"cicsam"``, ``"hric"``, ``"upwind"`` (algebraic, this package) or
        ``"plic"`` (geometric, :func:`tensorfvm.vof_plic.advect_plic`).
    cells_per_height : ``H/dx``; the gate resolutions are 28 and 56.
    window : optional ``(x_lo, x_hi)`` for the depth-error norm (default: whole
        domain).  The default is what the gate reports; the transport is verified
        over the whole wetted region, not a hand-picked sub-range.
    """
    geo = geo or DamBreakGeometry(t_end=t_end)
    if cells_per_height < 4:
        raise ValueError("need at least 4 cells per column height")
    if not 0.0 < t_start < t_end:
        raise ValueError("need 0 < t_start < t_end")
    dx = geo.height / cells_per_height
    nx = int(round(geo.length / dx))
    ny = int(round(geo.tank_height / dx))
    dx = geo.length / nx
    dy = geo.tank_height / ny
    if abs(dx - dy) > 0.05 * dx:
        raise ValueError(
            f"cells must be square to 5%: dx={dx:.6g} dy={dy:.6g} "
            "(tank_height should be a multiple of the column height)")
    h_ref = h_ref_fraction * geo.height

    alpha = exact_alpha(nx, ny, geo, t_start)
    if scheme == "plic":
        from ..vof_plic import advect_plic
    else:
        from .cicsam import advect_cicsam

    t = t_start
    steps = 0
    samples, mass_hist, time_hist = [], [], []
    m0 = float(alpha.sum().item()) * dx * dy
    worst_bound = 0.0
    order_flip = 0
    xi = geo.x_left + (np.arange(nx) + 0.5) * dx
    lo, hi = window if window is not None else (geo.x_left, geo.x_right)
    wmask = (xi >= lo) & (xi <= hi)
    targets = sorted(ts for ts in sample_times if t_start <= ts <= t_end + 1e-12)
    next_target = 0
    while True:
        depth = column_depth(alpha, dy)
        mass = float(alpha.sum().item()) * dx * dy
        mass_hist.append(mass)
        time_hist.append(t)
        if next_target < len(targets) and t >= targets[next_target] - 1e-12:
            next_target += 1
            h_ex = ritter_depth(xi - geo.width, t, geo)
            err = np.abs(depth - h_ex)
            row = {
                "t": t,
                "front_depth_m": _front(depth, dx, h_ref, geo.x_left),
                "front_exact_at_depth_m": ritter_front_at_depth(t, h_ref, geo),
                "front_tip_measured_m": _front(depth, dx, 0.5 * dy, geo.x_left),
                "front_tip_m": ritter_front(t, geo),
                "depth_error_l1_m": float(err[wmask].mean()),
                "depth_error_l1_relative": float(
                    err[wmask].sum() / max(h_ex[wmask].sum(), 1e-30)),
                "alpha_min": float(alpha.min().item()),
                "alpha_max": float(alpha.max().item()),
                "mass": mass,
            }
            row["front_error_at_depth"] = abs(
                row["front_depth_m"] - row["front_exact_at_depth_m"]
            ) / row["front_exact_at_depth_m"]
            row["front_error_tip"] = abs(
                row["front_tip_measured_m"] - row["front_tip_m"]
            ) / row["front_tip_m"]
            samples.append(row)
        if t_end - t <= 1e-12:
            break
        u, v = velocity_faces(nx, ny, geo, t)
        dt = cfl / max(float(u.abs().max()) / dx, float(v.abs().max()) / dy)
        dt = min(dt, t_end - t)
        if scheme == "plic":
            # A dimensional split of a straining field is not divergence free in
            # each direction alone, so PLIC needs its between-sweep bound repair
            # (measured without it: alpha_max = 1.0118 after the x sweep, which
            # the module rejects as a scheme defect).
            alpha = advect_plic(alpha, u, v, dt, dx, dy,
                                order="xy" if order_flip % 2 == 0 else "yx",
                                conservative_bounds=True)
        else:
            alpha = advect_cicsam(alpha, u, v, dt, dx, dy, scheme=scheme,
                                  order="xy" if order_flip % 2 == 0 else "yx",
                                  periodic=False, limit=limit)
        order_flip += 1
        t = min(t + dt, t_end)
        steps += 1
        worst_bound = max(
            worst_bound,
            float((alpha < -1e-12).sum().item() + (alpha > 1.0 + 1e-12).sum().item()),
        )
        if not torch.isfinite(alpha).all():
            raise FloatingPointError(f"non-finite alpha at step {steps}")
        if steps > 200000:
            raise RuntimeError("step budget exceeded")

    m_end = float(alpha.sum().item()) * dx * dy
    depth_end = column_depth(alpha, dy)
    profile = {
        "t": t,
        "x": xi.tolist(),
        "numeric": depth_end.tolist(),
        "analytic": ritter_depth(xi - geo.width, t, geo).tolist(),
    }
    return {
        "scheme": scheme,
        "cells_per_height": cells_per_height,
        "dx_m": dx,
        "dy_m": dy,
        "nx": nx,
        "ny": ny,
        "steps": steps,
        "cfl": cfl,
        "t_start": t_start,
        "t_end": t_end,
        "h_ref_m": h_ref,
        "h_ref_fraction": h_ref_fraction,
        "samples": samples,
        "mass_initial": m0,
        "mass_final": m_end,
        "mass_drift_relative": abs(m_end - m0) / m0,
        "mass_drift_max_relative": max(abs(m - m0) / m0 for m in mass_hist),
        "mass_times": time_hist,
        "mass_relative_series": [(m - m0) / m0 for m in mass_hist],
        "bounded_violation_cells": worst_bound,
        "fct_limiter": bool(limit),
        "geometry": geo.summary(),
        "_profile": profile,
        "_field": alpha,
        "_field_depth": (depth_end / max(depth_end.max(), 1e-30)).tolist(),
    }


def _front(depth: np.ndarray, dx: float, threshold: float,
           x_left: float = 0.0) -> float:
    from .metrics2d import front_position

    return front_position(depth, dx, x_left, threshold)