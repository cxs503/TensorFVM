"""Three-benchmark verification surface for geometric VOF interface tracking.

Why this module exists
----------------------
:mod:`tensorfvm.vof_plic` implements the geometric (PLIC) transport and
:mod:`tensorfvm.verification.vof_plic_interface` measures its 1-D step response
and one slotted-disk revolution.  What was still missing is the part that makes
a two-phase interface tracker *believable* on a ship-hull free surface:

* a **shape-based** fidelity metric — the volume of a flux-form scheme is
  conserved to roundoff by construction, so ``|Σα − Σα₀|`` says nothing about
  whether the interface kept its shape.  The quantity that does is the area
  enclosed by the ``α = 0.5`` iso-contour, which is *not* conserved by a
  flux-form update and therefore discriminates schemes;
* a **stretch–reverse** (single vortex) benchmark — the only test in this
  family that measures irreversible interface damage: a stirring field is run
  forward and then exactly backward, so any error is pure numerical diffusion of
  the scheme, not a property of the flow;
* a **multi-dimensional convergence order** — a diagonally advected step, where
  the L1 error of a dimensional-split scheme is dominated by splitting rather
  than by the 1-D reconstruction, and where a first-order baseline must be
  visibly worse.

The three cases below are the classic benchmarks of the field: Zalesak (1979)
slotted disk, Rider & Kothe (1995) / LeVeque (1996) single-vortex
stretch–reverse, and Rider & Kothe's diagonal step-advection convergence study.
Each is compared with the first-order donor-cell stencil on the *same* velocity
field and the *same* boundary conditions, so the comparison isolates the
interface reconstruction.

Everything is deterministic: fixed CFL, integer step counts chosen so the exact
solution is the initial field (a whole period), and no random input.

Criteria (declared once, machine-checked in :func:`verdicts`)
-------------------------------------------------------------
===========================  ==============  =================================
quantity                     threshold       meaning
===========================  ==============  =================================
Zalesak contour-area drift   < 2 %           shape fidelity after 2π
vortex mass drift            < 0.1 %         conservative transport in a
                                             non-solenoidal split field
per-step mass residual       < 1e-12         flux form is algebraically
                                             conservative
diagonal L1 order (64→128)   >= 0.95         multi-D accuracy (target 1.0)
===========================  ==============  =================================

Measurement definitions
-----------------------
``contour_area``
    Area of ``{α ≥ level}`` computed by bilinear sub-cell sampling (``subcells``
    quadrature points per cell per direction, periodic wrap).  It is a
    scheme-agnostic estimate of the enclosed liquid area: it does not see the
    volume integral, only the level set.
``peak_retention``
    ``max α`` after the run normalised by the initial maximum (1.0 for a
    non-diffusive scheme on a binary field).
``mixed_fraction`` / ``max_transition_width``
    Fraction of cells with ``0.05 < α < 0.95`` and the longest contiguous run of
    such cells in a row or a column (in cells) — direct measures of interface
    smearing.

Usage
-----
    python -m tensorfvm.verification.vof_interface_tracking --output <dir>
    python -m tensorfvm.verification.vof_interface_tracking --quick --output <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from ..vof_plic import advect_plic, interface_width_cells
from .vof_plic_interface import donor_advect_periodic
from .vof_rotation import rotation_velocity

DOMAIN = 1.0
CFL = 0.4
SUBCELLS = 16

#: Declared acceptance thresholds (see the module docstring table).
CRITERIA = {
    "zalesak_area_drift_max": 0.02,        # |A(2π) - A(0)| / A(0)
    "vortex_mass_drift_max": 1e-3,         # |Σα(2T) - Σα(0)| / Σα(0)
    "step_mass_residual_max": 1e-12,       # per-step |ΔΣα| / Σα
    "diagonal_order_min": 0.95,            # measured L1 order, finest pair
    "diagonal_order_target": 1.0,          # the plan's target
}

__all__ = [
    "CRITERIA",
    "contour_area",
    "slotted_disk_area",
    "classic_slotted_disk",
    "full_strip_disk_area",
    "max_transition_width",
    "field_metrics",
    "zalesak_rotation",
    "vortex_velocity",
    "discrete_divergence",
    "vortex_reversal",
    "diagonal_step_field",
    "diagonal_advection",
    "diagonal_order_study",
    "vortex_time_symmetry_study",
    "verdicts",
    "run",
    "main",
]


# ---------------------------------------------------------------------------
# analytic reference geometry
# ---------------------------------------------------------------------------
def _circle_segment(x: float, radius: float) -> float:
    """Antiderivative of ``sqrt(r² − x²)`` (used for the slotted-disk area)."""
    x = max(-radius, min(radius, x))
    return 0.5 * (x * math.sqrt(radius * radius - x * x)
                  + radius * radius * math.asin(x / radius))


def slotted_disk_area(radius: float = 0.15, slot_half_width: float = 0.05) -> float:
    """Exact area of Zalesak's slotted disk.

    The disk of radius ``radius`` is cut by a slot ``|x − x_c| ≤ h`` that opens
    from the centre ``y_c`` upwards (see :func:`classic_slotted_disk`, centred at
    ``(0.5, 0.75)``).  The removed part is the area under ``y = sqrt(r² − x²)``
    between ``x = −h`` and ``h``::

        A = π r² − ∫_{−h}^{h} sqrt(r² − x²) dx
    """
    r = float(radius)
    h = float(slot_half_width)
    if not 0.0 < h < r:
        raise ValueError("slot_half_width must be in (0, radius)")
    return math.pi * r * r - (_circle_segment(h, r) - _circle_segment(-h, r))


def full_strip_disk_area(radius: float = 0.15, slot_half_width: float = 0.05) -> float:
    """Exact area of a disk cut by a **full-height** strip ``|x − x_c| ≤ h``.

    This is the shape of :func:`tensorfvm.verification.vof_rotation.zalesak_disk`
    (the historical helper, which clears the strip on both sides of the centre),
    quoted so the two geometries can be compared analytically instead of by eye::

        A = π r² − 2 ∫_{−h}^{h} sqrt(r² − x²) dx

    """
    r = float(radius)
    h = float(slot_half_width)
    if not 0.0 < h < r:
        raise ValueError("slot_half_width must be in (0, radius)")
    return math.pi * r * r - 2.0 * (_circle_segment(h, r) - _circle_segment(-h, r))


def classic_slotted_disk(
    n: int,
    domain: float = DOMAIN,
    centre: Tuple[float, float] = (0.5, 0.75),
    radius: float = 0.15,
    slot_half_width: float = 0.05,
) -> torch.Tensor:
    """Zalesak (1979) slotted disk as a **binary** cell-centred indicator.

    The disk is cut by the notch ``|x − x_c| ≤ h`` **for y ≥ y_c only**, i.e. the
    slot opens from the centre of the disk towards the top boundary — the
    geometry the benchmark is defined with, whose exact area is
    :func:`slotted_disk_area`.

    Note
    ----
    :func:`tensorfvm.verification.vof_rotation.zalesak_disk` (used by the earlier
    rotation verification) instead zeroes the **whole** vertical strip, which
    removes the lower lobe too: measured area ``0.0413`` against ``0.0560`` for
    the classic geometry, i.e. 26 % less liquid.  That field is still a perfectly
    valid transport test (indeed a harder one: two disconnected lobes), but it is
    not the published Zalesak shape, so this module uses its own geometry and pins
    the difference in the tests (``test_zalesak_geometry_matches_analytic_area``).
    """
    h = float(domain) / n
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * h
    xx, yy = torch.meshgrid(c, c, indexing="xy")
    cx, cy = float(centre[0]), float(centre[1])
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(radius) ** 2
    slot = ((xx - cx).abs() <= float(slot_half_width)) & (yy >= cy)
    return (inside & ~slot).to(torch.float64)


# ---------------------------------------------------------------------------
# sub-cell contour area (the shape-fidelity metric)
# ---------------------------------------------------------------------------
def _neighbour(a: torch.Tensor, dj: int, di: int, periodic: bool) -> torch.Tensor:
    """Value at cell ``(j + dj, i + di)``; periodic wrap or zero-gradient edge."""
    if periodic:
        return torch.roll(a, (-dj, -di), dims=(0, 1))
    out = torch.roll(a, (-dj, -di), dims=(0, 1)).clone()
    if dj:
        out[-1 if dj > 0 else 0] = a[-1 if dj > 0 else 0]
    if di:
        out[:, -1 if di > 0 else 0] = a[:, -1 if di > 0 else 0]
    return out


def contour_area(
    alpha: torch.Tensor,
    level: float = 0.5,
    dx: float = 1.0,
    dy: float | None = None,
    subcells: int = SUBCELLS,
    periodic: bool = True,
) -> float:
    """Area of ``{α ≥ level}`` by bilinear sub-cell sampling.

    Each cell is split into ``subcells × subcells`` quadrature points of the
    **bilinear interpolant** of ``α`` and the fraction of points above ``level``
    is accumulated.  This is a level-set measure: unlike ``Σα·ΔV`` it is not
    conserved by a flux-form update, so it is the quantity that reveals whether
    an advected shape kept its area.

    Parameters
    ----------
    alpha : (ny, nx) tensor
        Volume fraction field.
    level : float
        Contour level (``0.5`` for the standard VOF interface).
    dx, dy : floats
        Cell size (``dy`` defaults to ``dx``).
    subcells : int
        Quadrature points per cell per direction.  The default ``16`` keeps the
        quadrature error of this estimate at ~0.1 % on the slotted-disk geometry
        (``8`` gives ~0.2 %), i.e. an order of magnitude below the 2 % criterion
        it is used to judge; the sensitivity is pinned in the tests.
    periodic : bool
        Wrap the interpolation stencil at the boundary (``True``, consistent with
        the periodic transport used by the benchmarks) or replicate the edge
        value (``False``).
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    if a.dim() != 2:
        raise ValueError(f"alpha must be 2-D, got shape {tuple(a.shape)}")
    k = int(subcells)
    if k < 1:
        raise ValueError(f"subcells must be >= 1, got {k}")
    dy = float(dx) if dy is None else float(dy)
    total = torch.zeros_like(a)
    offs = (torch.arange(k, dtype=torch.float64) + 0.5) / k - 0.5
    for sy in offs:
        for sx in offs:
            ex = abs(float(sx))
            ey = abs(float(sy))
            east = _neighbour(a, 0, 1 if sx > 0 else -1, periodic)
            north = _neighbour(a, 1 if sy > 0 else -1, 0, periodic)
            diag = _neighbour(a, 1 if sy > 0 else -1, 1 if sx > 0 else -1, periodic)
            value = (
                a * (1.0 - ex) * (1.0 - ey)
                + east * ex * (1.0 - ey)
                + north * (1.0 - ex) * ey
                + diag * ex * ey
            )
            total = total + (value >= level).to(torch.float64)
    return float(total.sum()) * float(dx) * dy / float(k * k)


def _longest_true_run(flags: Sequence[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def max_transition_width(alpha: torch.Tensor, lo: float = 0.05, hi: float = 0.95) -> int:
    """Longest contiguous run of partially filled cells in a row or a column.

    A scheme that keeps the interface inside one cell returns ``1``; a diffusive
    scheme returns the width of the smeared band.  Computed over every row and
    every column (integer, so exact).
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    inside = ((a > lo) & (a < hi)).cpu().tolist()
    ny, nx = a.shape
    best = 0
    for j in range(ny):
        best = max(best, _longest_true_run(inside[j]))
    for i in range(nx):
        best = max(best, _longest_true_run([inside[j][i] for j in range(ny)]))
    return best


# ---------------------------------------------------------------------------
# common metrics
# ---------------------------------------------------------------------------
def field_metrics(
    alpha: torch.Tensor,
    exact: torch.Tensor | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    level: float = 0.5,
    subcells: int = SUBCELLS,
) -> Dict[str, float]:
    """Error, sharpness and area metrics of a volume-fraction field.

    ``l1``/``relative_l1``/``linf`` are only produced when ``exact`` is given;
    the relative L1 is normalised by the exact field's volume (the transported
    liquid), so it is dimensionless and comparable across resolutions.
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    dy = float(dx) if dy is None else float(dy)
    out: Dict[str, float] = {
        "minimum": float(a.min()),
        "maximum": float(a.max()),
        "mass": float(a.sum()) * float(dx) * dy,
        "contour_area": contour_area(a, level=level, dx=dx, dy=dy, subcells=subcells),
        "mixed_fraction": float(((a > 0.05) & (a < 0.95)).to(torch.float64).mean()),
        "max_transition_width": float(max_transition_width(a)),
        "row_transition_width": float(interface_width_cells(a, axis=1)),
    }
    if exact is not None:
        e = torch.as_tensor(exact, dtype=torch.float64)
        diff = (a - e).abs()
        volume = float(e.sum()) * float(dx) * float(dy)
        out["l1"] = float(diff.sum()) * float(dx) * float(dy)
        out["relative_l1"] = out["l1"] / volume if volume else float("nan")
        out["linf"] = float(diff.max())
        out["peak_retention"] = float(a.max()) / max(float(e.max()), 1e-300)
    return out


# ---------------------------------------------------------------------------
# case 1 — Zalesak slotted disk in solid-body rotation (2π)
# ---------------------------------------------------------------------------
def zalesak_rotation(
    n: int,
    cfl: float = CFL,
    revolutions: int = 1,
    scheme: str = "plic",
    subcells: int = SUBCELLS,
) -> Dict[str, object]:
    """``revolutions`` full turns of Zalesak's slotted disk (2π per revolution).

    The rotation is an exact advection with **zero discrete divergence** (``u``
    depends on ``y`` only, ``v`` on ``x`` only, both linear), so after an integer
    number of revolutions the initial field *is* the exact solution and the error
    is pure scheme error.

    Reported criteria:

    * ``area_drift`` = ``|A(T) − A(0)| / A(0)`` on the ``α = 0.5`` contour
      (criterion ``< 2 %``);
    * ``mass_drift`` = ``|Σα(T) − Σα(0)| / Σα(0)`` (must be roundoff);
    * ``max_step_residual`` = worst per-step ``|ΔΣα| / Σα``.

    Measurement口径 choices, stated because the two "area errors" differ:

    * ``area_drift`` measures **transport fidelity** — it compares the computed
      contour area with the contour area of the *discretised* initial field, so
      the O(Δx) error of rasterising the disk into binary cells is excluded;
    * ``area_vs_analytic`` compares with :func:`slotted_disk_area`, i.e. it also
      carries the initial rasterisation error, and is reported for information
      (it does not shrink to zero under refinement of the scheme alone).
    """
    if int(revolutions) != revolutions or revolutions < 1:
        raise ValueError("revolutions must be a positive integer")
    scheme = str(scheme)
    if scheme not in ("plic", "donor_cell"):
        raise ValueError(f"scheme must be 'plic' or 'donor_cell', got {scheme!r}")
    dx = DOMAIN / n
    alpha0 = classic_slotted_disk(n)
    u, v, dx_u = rotation_velocity(n)
    if abs(dx_u - dx) > 1e-14:
        raise ValueError(f"grid mismatch: velocity dx {dx_u} != {dx}")
    speed = float(max(u.abs().max(), v.abs().max()))
    nsteps = max(1, int(round(revolutions / (cfl * dx / speed))))
    dt = revolutions / nsteps

    a = alpha0.clone()
    worst_step = 0.0
    worst_clip = 0.0
    for k in range(nsteps):
        volume_old = float(a.sum())
        order = "xy" if k % 2 == 0 else "yx"
        if scheme == "plic":
            diag: dict = {}
            a = advect_plic(a, u, v, dt, dx, dx, order=order,
                            conservative_bounds=True, diagnostics=diag)
            worst_clip = max(worst_clip, float(diag.get("final_imbalance", 0.0)))
        else:
            a = donor_advect_periodic(a, u, v, dt, dx, dx, order=order)
        worst_step = max(worst_step, abs(float(a.sum()) - volume_old) / volume_old)

    m0 = field_metrics(alpha0, alpha0, dx, subcells=subcells)
    m1 = field_metrics(a, alpha0, dx, subcells=subcells)
    area0, area1 = m0["contour_area"], m1["contour_area"]
    analytic = slotted_disk_area()
    return {
        "case": "zalesak_rotation",
        "scheme": scheme,
        "cells": n,
        "steps": nsteps,
        "dt": dt,
        "courant_max": speed * dt / dx,
        "revolutions": revolutions,
        "radians": 2.0 * math.pi * revolutions,
        "geometry": {
            "source": "tensorfvm.verification.vof_interface_tracking.classic_slotted_disk",
            "domain": DOMAIN,
            "centre": [0.5, 0.75],
            "radius": 0.15,
            "slot_half_width": 0.05,
            "analytic_area": analytic,
        },
        "area_initial": area0,
        "area_final": area1,
        "area_drift": abs(area1 - area0) / area0,
        "area_vs_analytic": abs(area1 - analytic) / analytic,
        "mass_drift": abs(float(a.sum()) - float(alpha0.sum())) / float(alpha0.sum()),
        "max_step_residual": worst_step,
        "max_clip_imbalance": worst_clip,
        "metrics": m1,
        "field": a.clone(),
        "initial_field": alpha0.clone(),
    }


# ---------------------------------------------------------------------------
# case 2 — single vortex stretch/reverse
# ---------------------------------------------------------------------------
def vortex_velocity(n: int, phase: float = 1.0,
                    domain: float = DOMAIN) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Face-centred single-vortex velocity of LeVeque (1996).

    ``u = −sin²(πx) sin(2πy)``, ``v = sin(2πx) sin²(πy)`` on the unit square,
    staggered to the cell faces (``u`` on x-faces, ``v`` on y-faces) and scaled by
    ``phase`` — the stirring is replayed backwards with ``phase = −cos(πt/T)``,
    which passes through zero at the turning point.  ``|u|, |v| ≤ 1``.

    The staggered field is **not** exactly divergence-free in the discrete sense
    (measured by :func:`discrete_divergence` and reported with the results),
    which is exactly why this case tests conservation: a flux-form update must
    keep ``Σα`` at roundoff even when the discrete ``∇·u_h ≠ 0``.
    """
    dx = domain / n
    faces = torch.arange(n + 1, dtype=torch.float64) * dx
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    sxf = torch.sin(math.pi * faces) ** 2          # sin²(πx) on x-faces
    syc = torch.sin(2.0 * math.pi * centres)       # sin(2πy) on y-centres
    sxc = torch.sin(2.0 * math.pi * centres)       # sin(2πx) on x-centres
    syf = torch.sin(math.pi * faces) ** 2          # sin²(πy) on y-faces
    u = (-sxf[None, :] * syc[:, None]) * float(phase)
    v = (sxc[None, :] * syf[:, None]) * float(phase)
    return u, v, dx


def discrete_divergence(u: torch.Tensor, v: torch.Tensor, dx: float) -> float:
    """Max ``|∇·u_h|`` of a staggered field (finite-volume form)."""
    div = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dx
    return float(div.abs().max())


def vortex_reversal(
    n: int,
    cfl: float = CFL,
    half_period: float = 1.0,
    scheme: str = "plic",
    subcells: int = SUBCELLS,
    reversal: str = "smooth",
) -> Dict[str, object]:
    """Stir a slotted disk for ``half_period``, then un-stir it exactly.

    The velocity is modulated by ``cos(π t / T)`` (``T = half_period``), so the
    flow is smooth in time, vanishes at the turning point and is an exact time
    reversal: after ``2T`` the *exact* field equals the initial field.  Every
    deviation is therefore irreversible interface damage (numerical diffusion),
    which no conservation check can see.

    Criteria: ``mass_drift < 0.1 %`` and ``max_step_residual < 1e-12``.

    ``reversal`` selects the time modulation: ``"smooth"`` uses the LeVeque
    ``cos(πt/T)`` ramp (continuous, zero speed at the turning point),
    ``"hard"`` jumps from ``+1`` to ``−1`` at ``t = T`` (Rider & Kothe).  The
    hard switch is measurably worse at the same CFL (see
    :func:`vortex_time_symmetry_study`), because the jump is an O(1) temporal
    discontinuity that the first-order-in-time stepping cannot follow.
    """
    scheme = str(scheme)
    if scheme not in ("plic", "donor_cell"):
        raise ValueError(f"scheme must be 'plic' or 'donor_cell', got {scheme!r}")
    if reversal not in ("smooth", "hard"):
        raise ValueError(f"reversal must be 'smooth' or 'hard', got {reversal!r}")
    dx = DOMAIN / n
    alpha0 = classic_slotted_disk(n)
    total_time = 2.0 * half_period
    nsteps = max(2, int(round(total_time / (cfl * dx))))
    dt = total_time / nsteps
    a = alpha0.clone()
    worst_step = 0.0
    peak_at_turn = float("nan")
    minimum_at_turn = float("nan")
    for k in range(nsteps):
        t = (k + 0.5) * dt
        if reversal == "smooth":
            phase = math.cos(math.pi * min(t, total_time) / half_period)
        elif reversal == "hard":
            phase = 1.0 if k < nsteps // 2 else -1.0
        else:
            raise ValueError(f"reversal must be 'smooth' or 'hard', got {reversal!r}")
        u, v, _ = vortex_velocity(n, phase=phase)
        volume_old = float(a.sum())
        order = "xy" if k % 2 == 0 else "yx"
        if scheme == "plic":
            a = advect_plic(a, u, v, dt, dx, dx, order=order, conservative_bounds=True)
        else:
            a = donor_advect_periodic(a, u, v, dt, dx, dx, order=order)
        worst_step = max(worst_step, abs(float(a.sum()) - volume_old) / volume_old)
        if abs(t - half_period) < 0.5 * dt:
            peak_at_turn = float(a.max())
            minimum_at_turn = float(a.min())
    m0 = field_metrics(alpha0, alpha0, dx, subcells=subcells)
    m1 = field_metrics(a, alpha0, dx, subcells=subcells)
    u0, v0, _ = vortex_velocity(n, phase=1.0)
    return {
        "case": "vortex_reversal",
        "scheme": scheme,
        "cells": n,
        "steps": nsteps,
        "dt": dt,
        "courant_max": dt / dx,
        "half_period": half_period,
        "total_time": total_time,
        "velocity": {
            "u": "-sin^2(pi x) sin(2 pi y) cos(pi t / T)",
            "v": "sin(2 pi x) sin^2(pi y) cos(pi t / T)",
            "source": "LeVeque 1996 (time reversal of Rider & Kothe 1995)",
            "max_discrete_divergence": discrete_divergence(u0, v0, dx),
        },
        "mass_drift": abs(float(a.sum()) - float(alpha0.sum())) / float(alpha0.sum()),
        "max_step_residual": worst_step,
        "peak_at_turn": peak_at_turn,
        "minimum_at_turn": minimum_at_turn,
        "metrics": m1,
        "initial_metrics": m0,
        "field": a.clone(),
        "initial_field": alpha0.clone(),
    }


# ---------------------------------------------------------------------------
# case 3 — diagonal step advection: L1 convergence order
# ---------------------------------------------------------------------------
def diagonal_step_field(n: int, offset: float = 0.75) -> torch.Tensor:
    """Step function ``α = 1`` for ``x + y ≤ offset`` on the periodic unit square."""
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    return ((xx + yy) <= offset).to(torch.float64)


def diagonal_advection(
    n: int,
    cfl: float = CFL,
    scheme: str = "plic",
    periods: float = 1.0,
    subcells: int = SUBCELLS,
) -> Dict[str, object]:
    """Advect a step along the diagonal by ``periods`` full periods.

    ``u = v = 1`` (signed face velocities), so after a whole period the exact
    solution is the initial field and the L1 error needs no resampling.  The
    sweep order alternates every step, which makes the dimensional splitting
    symmetric in time (Strang-like); without it the splitting alone contributes
    an O(Δx) bias in the sweep direction.
    """
    scheme = str(scheme)
    if scheme not in ("plic", "donor_cell"):
        raise ValueError(f"scheme must be 'plic' or 'donor_cell', got {scheme!r}")
    dx = DOMAIN / n
    alpha0 = diagonal_step_field(n)
    nsteps = max(1, int(round(periods / (cfl * dx))))
    dt = periods / nsteps
    u = torch.full((n, n + 1), 1.0, dtype=torch.float64)
    v = torch.full((n + 1, n), 1.0, dtype=torch.float64)
    a = alpha0.clone()
    worst_step = 0.0
    for k in range(nsteps):
        volume_old = float(a.sum())
        order = "xy" if k % 2 == 0 else "yx"
        if scheme == "plic":
            a = advect_plic(a, u, v, dt, dx, dx, order=order, conservative_bounds=True)
        else:
            a = donor_advect_periodic(a, u, v, dt, dx, dx, order=order)
        worst_step = max(worst_step, abs(float(a.sum()) - volume_old) / volume_old)
    m = field_metrics(a, alpha0, dx, subcells=subcells)
    return {
        "case": "diagonal_advection",
        "scheme": scheme,
        "cells": n,
        "steps": nsteps,
        "dt": dt,
        "courant": dt / dx,
        "max_step_residual": worst_step,
        "mass_drift": abs(float(a.sum()) - float(alpha0.sum())) / float(alpha0.sum()),
        "l1": m["l1"],
        "relative_l1": m["relative_l1"],
        "linf": m["linf"],
        "metrics": m,
        "field": a.clone(),
        "initial_field": alpha0.clone(),
    }


def _orders(rows: List[Dict[str, object]]) -> List[float | None]:
    out: List[float | None] = [None]
    for prev, cur in zip(rows, rows[1:]):
        h1, h2 = 1.0 / float(prev["cells"]), 1.0 / float(cur["cells"])
        e1, e2 = float(prev["l1"]), float(cur["l1"])
        out.append(math.log(e1 / e2) / math.log(h1 / h2)
                   if e1 > 0.0 and e2 > 0.0 else None)
    return out


def _fit_order(rows: List[Dict[str, object]]) -> float | None:
    """Least-squares slope of ``log L1`` against ``log Δx`` over the whole ladder."""
    xs = [math.log(1.0 / float(r["cells"])) for r in rows]
    ys = [math.log(float(r["l1"])) for r in rows]
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else None


def diagonal_order_study(
    resolutions: Sequence[int] = (32, 64, 128),
    cfl: float = CFL,
    schemes: Sequence[str] = ("plic", "donor_cell"),
    subcells: int = SUBCELLS,
) -> Dict[str, object]:
    """Resolution ladder for the diagonal step: L1 error and observed order."""
    study: Dict[str, object] = {"resolutions": list(resolutions), "cfl": cfl}
    for scheme in schemes:
        rows = [diagonal_advection(int(n), cfl=cfl, scheme=scheme, subcells=subcells)
                for n in resolutions]
        orders = _orders(rows)
        for row, order in zip(rows, orders):
            row["observed_order"] = order
        study[scheme] = [{k: v for k, v in row.items()
                          if k not in ("field", "initial_field")} for row in rows]
        finite = [float(o) for o in orders if o is not None]
        study[scheme + "_order_last"] = orders[-1]
        study[scheme + "_order_fit"] = _fit_order(rows)
        study[scheme + "_order_min"] = min(finite) if finite else None
    return study


def vortex_time_symmetry_study(
    n: int = 32,
    cfls: Sequence[float] = (CFL, 0.1),
    reversals: Sequence[str] = ("smooth", "hard"),
    scheme: str = "plic",
) -> List[Dict[str, object]]:
    """How much of the stretch–reverse error is *time-integration* asymmetry?

    Runs the same case at several Courant numbers and for both reversal modes.
    A purely spatial error would be independent of ``cfl`` (at fixed grid): the
    measured dependence quantifies the temporal part, which is reported instead
    of being silently attributed to the interface reconstruction.
    """
    rows: List[Dict[str, object]] = []
    for reversal in reversals:
        for cfl in cfls:
            row = vortex_reversal(n, cfl=float(cfl), scheme=scheme, reversal=reversal)
            rows.append({
                "cells": n,
                "cfl": float(cfl),
                "reversal": reversal,
                "steps": row["steps"],
                "relative_l1": row["metrics"]["relative_l1"],
                "mass_drift": row["mass_drift"],
                "peak_retention": row["metrics"]["peak_retention"],
                "max_transition_width": row["metrics"]["max_transition_width"],
            })
    return rows


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
def verdicts(
    zalesak: List[Dict[str, object]],
    vortex: List[Dict[str, object]],
    diagonal: Dict[str, object],
) -> Dict[str, object]:
    """Machine-check every declared criterion on the produced numbers."""
    checks: Dict[str, Dict[str, object]] = {}
    plic_z = [r for r in zalesak if r["scheme"] == "plic"]
    plic_v = [r for r in vortex if r["scheme"] == "plic"]
    fine_z = max(plic_z, key=lambda r: float(r["cells"]))
    fine_v = max(plic_v, key=lambda r: float(r["cells"]))
    checks["zalesak_area_drift"] = {
        "value": fine_z["area_drift"],
        "threshold": CRITERIA["zalesak_area_drift_max"],
        "cells": fine_z["cells"],
        "rule": "value < threshold on the finest PLIC grid",
        "pass": bool(float(fine_z["area_drift"]) < CRITERIA["zalesak_area_drift_max"]),
    }
    checks["vortex_mass_drift"] = {
        "value": fine_v["mass_drift"],
        "threshold": CRITERIA["vortex_mass_drift_max"],
        "cells": fine_v["cells"],
        "rule": "value < threshold on the finest PLIC grid",
        "pass": bool(float(fine_v["mass_drift"]) < CRITERIA["vortex_mass_drift_max"]),
    }
    worst_residual = max([float(r["max_step_residual"]) for r in plic_z + plic_v])
    checks["step_mass_residual"] = {
        "value": worst_residual,
        "threshold": CRITERIA["step_mass_residual_max"],
        "rule": "value < threshold (worst per-step relative drift, all PLIC runs)",
        "pass": bool(worst_residual < CRITERIA["step_mass_residual_max"]),
    }
    order = diagonal["plic_order_last"]
    checks["diagonal_l1_order"] = {
        "value": order,
        "threshold": CRITERIA["diagonal_order_min"],
        "target": CRITERIA["diagonal_order_target"],
        "fit": diagonal["plic_order_fit"],
        "rule": "value >= threshold (finest resolution pair, PLIC)",
        "pass": bool(order is not None
                     and float(order) >= CRITERIA["diagonal_order_min"]),
    }
    return {
        "criteria": CRITERIA,
        "checks": checks,
        "all_pass": all(bool(c["pass"]) for c in checks.values()),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(
    output_dir: str | Path | None = None,
    resolutions: Sequence[int] = (32, 64, 128),
    subcells: int = SUBCELLS,
    make_figures: bool = True,
) -> Dict[str, object]:
    """Run all three benchmarks and (optionally) write JSON + figures."""
    zalesak = [zalesak_rotation(int(n), subcells=subcells, scheme=s)
               for n in resolutions for s in ("plic", "donor_cell")]
    vortex = [vortex_reversal(int(n), subcells=subcells, scheme=s)
              for n in resolutions for s in ("plic", "donor_cell")]
    diagonal = diagonal_order_study(resolutions, subcells=subcells)
    sensitivity = vortex_time_symmetry_study()
    verdict = verdicts(zalesak, vortex, diagonal)

    report: Dict[str, object] = {
        "module": "tensorfvm.verification.vof_interface_tracking",
        "scheme_under_test": ("tensorfvm.vof_plic.advect_plic: PLIC with Youngs "
                              "normal, exact cut-area inverse (bisection), "
                              "dimensional splitting, alternating sweep order"),
        "reference": "tensorfvm.verification.vof_plic_interface.donor_advect_periodic",
        "settings": {"domain": DOMAIN, "cfl": CFL, "subcells": subcells,
                     "resolutions": list(resolutions)},
        "analytic_area_slotted_disk": slotted_disk_area(),
        "zalesak": [{k: v for k, v in r.items()
                     if k not in ("field", "initial_field")} for r in zalesak],
        "vortex": [{k: v for k, v in r.items()
                    if k not in ("field", "initial_field")} for r in vortex],
        "diagonal": diagonal,
        "vortex_time_symmetry": sensitivity,
        "verdict": verdict,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if make_figures:
            report["figures"] = figures(zalesak, vortex, diagonal, out)
        (out / "vof_interface_tracking.json").write_text(
            json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return report


def figures(
    zalesak: List[Dict[str, object]],
    vortex: List[Dict[str, object]],
    diagonal: Dict[str, object],
    out_dir: Path,
) -> List[str]:
    """Diagnostic PNGs: field snapshots, error maps and the convergence curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[str] = []
    fine_z = max((r for r in zalesak if r["scheme"] == "plic"),
                 key=lambda r: float(r["cells"]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, field, title in (
        (axes[0], fine_z["initial_field"], "initial α field"),
        (axes[1], fine_z["field"], "after 2π (PLIC)"),
    ):
        im = ax.imshow(field.cpu().numpy(), origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{title}\nn={fine_z['cells']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    err = (fine_z["field"] - fine_z["initial_field"]).cpu().numpy()
    im = axes[2].imshow(err, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_title("α(T) − α(0), PLIC", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    path = out_dir / "zalesak_rotation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)

    fine_v = max((r for r in vortex if r["scheme"] == "plic"),
                 key=lambda r: float(r["cells"]))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    im = axes[0].imshow(fine_v["field"].cpu().numpy(), origin="lower", cmap="Blues",
                        vmin=0, vmax=1)
    axes[0].set_title(f"after stir + un-stir (PLIC)\nn={fine_v['cells']}", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    err = (fine_v["field"] - fine_v["initial_field"]).abs().cpu().numpy()
    im = axes[1].imshow(err, origin="lower", cmap="magma")
    axes[1].set_title("|α(2T) − α(0)|, PLIC", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    path = out_dir / "vortex_reversal.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for scheme, style in (("plic", "o-"), ("donor_cell", "s--")):
        rows = diagonal[scheme]
        dxs = [1.0 / r["cells"] for r in rows]
        l1s = [r["l1"] for r in rows]
        fit = diagonal[scheme + "_order_fit"]
        ax.loglog(dxs, l1s, style,
                  label=f"{scheme} (fit order {fit:.2f})" if fit else scheme)
    ref = diagonal["plic"][0]
    ax.loglog([1.0 / 128, 1.0 / 32],
              [ref["l1"] / 4.0, ref["l1"]], "k:", label="slope 1 reference")
    ax.set_xlabel("Δx")
    ax.set_ylabel("L1 error")
    ax.legend(fontsize=8)
    ax.set_title("diagonal step advection: L1 vs Δx", fontsize=9)
    path = out_dir / "diagonal_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for scheme, style in (("plic", "o-"), ("donor_cell", "s--")):
        rows = sorted((r for r in zalesak if r["scheme"] == scheme),
                      key=lambda r: float(r["cells"]))
        ax.plot([r["cells"] for r in rows],
                [100.0 * r["area_drift"] for r in rows], style, label=scheme)
    ax.axhline(2.0, color="r", ls=":", label="2 % criterion")
    ax.set_xlabel("cells per direction")
    ax.set_ylabel("contour-area drift [%]")
    ax.set_title("Zalesak: α=0.5 area drift after 2π", fontsize=9)
    ax.legend(fontsize=8)
    path = out_dir / "zalesak_area_drift.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path.name)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="directory for the JSON report and figures")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--subcells", type=int, default=SUBCELLS)
    parser.add_argument("--quick", action="store_true",
                        help="coarse ladder (32 64) for a fast smoke run")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    resolutions = [32, 64] if args.quick else list(args.resolutions)
    report = run(args.output, resolutions=resolutions, subcells=args.subcells,
                 make_figures=not (args.no_figures or args.quick))
    verdict = report["verdict"]
    for name, check in verdict["checks"].items():
        state = "PASS" if check["pass"] else "FAIL"
        print(f"{state:4s} {name:22s} value={check['value']!r} "
              f"threshold={check['threshold']!r}")
    print("all_pass:", verdict["all_pass"])
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())