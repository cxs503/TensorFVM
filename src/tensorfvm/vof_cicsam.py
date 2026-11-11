"""CICSAM interface-capturing advection of the volume fraction (Ubbink & Issa 1999).

Why this module exists
----------------------
:mod:`tensorfvm.vof_phase0.advection` transports ``alpha`` with a two-step
Zalesak (1979) flux-corrected-transport (FCT) scheme, and
:mod:`tensorfvm.vof_plic` with geometric PLIC reconstruction.  Both are
bounded and conservative, but they sit at opposite ends of the
sharpness/accuracy trade-off: FCT is bounded and *diffusive* (a moving
interface spreads over 3-6 cells), PLIC is sharp but costs a per-cell
polygon clipping pass and is only first-order accurate in the interface
topology.  CICSAM is the algebraic scheme that closes that gap: it is a
normalised-variable (NVD) scheme whose face value is a blend of the most
compressive interpolation allowed by the convection-boundedness criterion
(CBC) and a third-order ULTIMATE-QUICKEST interpolation, weighted by the
orientation of the interface relative to the sweep direction.

Because it is written in flux form on the MAC grid, global conservation is
structural: the cell update telescopes, so the total liquid volume changes
only through the (zero) boundary fluxes and through the bound repair that
removes the sub-percent overshoot a directional split produces.

Normalised variables
--------------------
With donor ``D``, acceptor ``A`` and the cell ``U`` upwind of the donor,
the normalised volume fraction is

.. math:: \\tilde\\alpha = \\frac{\\alpha - \\alpha_U}{\\alpha_A - \\alpha_U},

so upwinding is ``0`` and downwinding is ``1``; the face value is written as
``alpha_f = alpha_U + tilde_alpha_f (alpha_A - alpha_U)``.

Face value (three ingredients)
------------------------------
1. **Hyper-C** — the compressive limit.  The CBC (Gaskell & Lau 1988,
   Waterson & Deconinck 1995) bounds an explicit-scheme face value by

   .. math:: \\tilde\\alpha_f \\le \\min\\left(1, \\tilde\\alpha_D / C\\right),
       \\qquad C = |u_f|\\,\\Delta t / h .

   Hyper-C *is* that bound (as compressive as boundedness permits): it keeps
   the interface a step whenever the Courant number leaves room for it.

2. **ULTIMATE-QUICKEST (UQ)** — the accurate, low-compression member.  Third
   order in space, with the characteristic (QUICKEST) correction that makes
   the face value the value seen at ``t + dt/2``.  Expanding the advected
   state to second order about the face,

   .. math:: \\tilde\\alpha_{UQ} = \\frac38 + \\frac34\\tilde\\alpha_D
             - \\frac{C}{2}\\left(1 - \\tilde\\alpha_D\\right)
             + \\frac{C^2}{8}\\left(1 - 2\\tilde\\alpha_D\\right),

   which reduces to QUICK (``3/8 + 3/4 tilde_alpha_D``) at ``C = 0`` and
   collapses onto the exact extremes (``0`` for ``tilde_alpha_D = 0``,
   ``1`` for ``tilde_alpha_D = 1``) at ``C = 1``.  The result is clipped into
   the CBC band; a UQ value that would violate the CBC is replaced by the
   donor value (the standard CICSAM correction), which is what keeps the
   blend bounded.

3. **Blend factor** — compression where it is needed and accuracy where it is
   not.  Writing ``theta`` for the angle between the interface normal and the
   sweep direction, ``(1 + cos 2 theta)/2 = cos^2 theta``, so

   .. math:: \\gamma_x = k_\\gamma \\cos^2\\theta = k_\\gamma
             \\frac{n_x^2}{n_x^2 + n_y^2}

   gives ``gamma = 1`` (pure Hyper-C) when the interface is advected *normal*
   to itself — the configuration where a smeared profile is pure error — and
   ``gamma -> 0`` (pure UQ) when the flow is tangential to the interface,
   where compression would only create interface wrinkling.  ``k_gamma = 1``
   is the published setting.

Scope and limits
----------------
* Uniform Cartesian grid, staggered (MAC) face velocities — the same layout
  as :class:`tensorfvm.vof_phase0.AlgebraicVOF`.
* Dimensional (Strang-like) splitting; alternate the sweep order between
  steps.  Directional CFL must be ``<= 1`` (the scheme's stencil is 3 cells).
* ``advect`` is a drop-in replacement for ``AlgebraicVOF.advect``: the
  phase-0 two-phase solver accepts it by assigning ``solver.vof``.
* No interface reconstruction and no explicit anti-diffusion flux: the
  "interface compression" of CICSAM *is* the Hyper-C member of the blend.
* The split sweep can overfill a cell slightly (the CBC bounds the outflow,
  not the acceptor's capacity).  A *local* mass-exact repair
  (:func:`_repair_bounds_local`) bounds the field into ``[0, 1]`` and hands the
  volume back to the 3x3 neighbourhood of the offending cells — the global
  redistribution used by PLIC would smear a compressive scheme's interface.
  The volume it moves is reported per step in ``last_diagnostics``, so the mass
  audit is not blind to it.
"""

from __future__ import annotations

import torch

__all__ = [
    "CICSAMVOF",
    "interface_thickness_cells",
    "cbc_upper",
    "hyper_c",
    "ultimate_quickest",
    "blend_gamma",
    "cicsam_face_value",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# normalised-variable building blocks (vectorised over faces)
# --------------------------------------------------------------------------- #
def cbc_upper(tilde_donor: torch.Tensor, cour: torch.Tensor) -> torch.Tensor:
    """CBC upper bound of the normalised face value: ``min(1, tilde_D / C)``.

    ``cour`` is the directional Courant number ``|u_f| dt / h`` of the face.
    A face at rest (``C -> 0``) carries no flux, so the bound degenerates to
    ``1`` there and the value is irrelevant.
    """
    safe = cour.clamp_min(1e-9)
    return torch.clamp(tilde_donor / safe, max=1.0)


def hyper_c(tilde_donor: torch.Tensor, cour: torch.Tensor) -> torch.Tensor:
    """Hyper-C compressive face value (the CBC upper bound itself)."""
    return cbc_upper(tilde_donor, cour)


def ultimate_quickest(tilde_donor: torch.Tensor,
                      cour: torch.Tensor) -> torch.Tensor:
    """ULTIMATE-QUICKEST face value in normalised variables (see module doc)."""
    c = cour
    return (0.375 + 0.75 * tilde_donor
            - 0.5 * c * (1.0 - tilde_donor)
            + 0.125 * c * c * (1.0 - 2.0 * tilde_donor))


def blend_gamma(normal_component: torch.Tensor,
                normal_other: torch.Tensor,
                k_gamma: float = 1.0) -> torch.Tensor:
    """Weight of the compressive member: ``k_gamma * n_axis^2 / |n|^2``.

    ``normal_component`` is the interface-normal component along the sweep
    direction, ``normal_other`` the orthogonal one.  Both are evaluated at the
    faces.  Where the field is locally uniform (``|n| ~ 0``) the weight is
    set to ``0`` (pure UQ): there is no interface to compress.
    """
    sq = normal_component * normal_component
    tot = sq + normal_other * normal_other
    safe = tot > _EPS
    return torch.where(safe, torch.clamp(k_gamma * sq / tot.clamp_min(_EPS), 0.0, 1.0),
                       torch.zeros_like(sq))


def cicsam_face_value(alpha_d: torch.Tensor, alpha_a: torch.Tensor,
                      alpha_u: torch.Tensor, cour: torch.Tensor,
                      gamma: torch.Tensor) -> torch.Tensor:
    """Blended CICSAM face value in physical units.

    ``alpha_d``/``alpha_a``/``alpha_u`` are the donor, acceptor and
    donor-upwind cell values seen by each face (same shape), ``cour`` the
    directional Courant number and ``gamma`` the compressive weight.
    Where ``alpha_A == alpha_U`` the normalisation is undefined and the donor
    value is used (a locally uniform colour function carries no information
    about the interface position inside the stencil).
    """
    denom = alpha_a - alpha_u
    # The normalised coordinate is defined whenever ``alpha_A != alpha_U``;
    # the *sign* of the denominator is not restricted (a falling profile
    # advected downstream is the mirror of a rising one and must be handled by
    # the same limiter).  A stencil that is not monotone -- ``alpha_D`` at a
    # local extremum, so that the normalised donor value leaves ``[0, 1]`` --
    # has no normalised coordinate at all; the classic remedy is to fall back
    # on the donor (upwind) value, which the ``movable`` mask does.
    defined = denom.abs() > _EPS
    d = torch.where(defined, denom, torch.ones_like(denom))
    tilde = torch.where(defined, (alpha_d - alpha_u) / d, torch.zeros_like(alpha_d))
    movable = defined & (tilde >= 0.0) & (tilde <= 1.0)
    tilde_d = torch.clamp(tilde, 0.0, 1.0)

    tilde_hc = hyper_c(tilde_d, cour)
    tilde_uq = ultimate_quickest(tilde_d, cour)
    # CICSAM's UQ correction: never let the accurate member exceed what the
    # CBC allows; fall back on the donor value when it does.
    tilde_uq = torch.where(tilde_uq > tilde_hc, tilde_d,
                           torch.clamp(tilde_uq, 0.0, 1.0))
    tilde_f = torch.clamp(gamma * tilde_hc + (1.0 - gamma) * tilde_uq, 0.0, 1.0)
    face = alpha_u + tilde_f * denom
    # Interpolation may not create a new extremum inside its own stencil; this
    # is the second (local) safeguard on top of the CBC clamp.
    lo = torch.minimum(torch.minimum(alpha_u, alpha_d), alpha_a)
    hi = torch.maximum(torch.maximum(alpha_u, alpha_d), alpha_a)
    face = torch.clamp(face, min=lo, max=hi)
    return torch.where(movable, face, alpha_d)


# --------------------------------------------------------------------------- #
# grid helpers
# --------------------------------------------------------------------------- #
def _pad2(alpha: torch.Tensor, axis: int, periodic: bool = False) -> torch.Tensor:
    """Two ghost cells on both ends of ``axis`` (wrap-around or replicated)."""
    n = alpha.shape[axis]
    if periodic:
        if axis == 1:
            return torch.cat([alpha[:, n - 2:], alpha, alpha[:, :2]], dim=1)
        return torch.cat([alpha[n - 2:, :], alpha, alpha[:2, :]], dim=0)
    if axis == 1:
        return torch.cat([alpha[:, :1], alpha[:, :1], alpha,
                          alpha[:, -1:], alpha[:, -1:]], dim=1)
    return torch.cat([alpha[:1, :], alpha[:1, :], alpha,
                      alpha[-1:, :], alpha[-1:, :]], dim=0)


def _centre_normal(alpha: torch.Tensor, hx: float,
                   hy: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell-centred interface-normal components (edge-replicated differences)."""
    gx_pad = torch.cat([alpha[:, :1], alpha, alpha[:, -1:]], dim=1)
    gy_pad = torch.cat([alpha[:1, :], alpha, alpha[-1:, :]], dim=0)
    nx = (gx_pad[:, 2:] - gx_pad[:, :-2]) / (2.0 * hx)
    ny = (gy_pad[2:, :] - gy_pad[:-2, :]) / (2.0 * hy)
    return nx, ny


def _face_average(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Average cell-centred data onto the faces of ``axis`` (1 = x, 0 = y)."""
    if axis == 1:
        pad = torch.cat([field[:, :1], field, field[:, -1:]], dim=1)
        return 0.5 * (pad[:, 1:] + pad[:, :-1])
    pad = torch.cat([field[:1, :], field, field[-1:, :]], dim=0)
    return 0.5 * (pad[1:, :] + pad[:-1, :])


def _dilate(mask: torch.Tensor) -> torch.Tensor:
    """3x3 neighbourhood of ``mask`` (logical dilation)."""
    m = mask.to(torch.float64)
    pad = torch.cat([m[:, :1], m, m[:, -1:]], dim=1)
    hx = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) > 0.0
    hx = hx.to(torch.float64)
    pad = torch.cat([hx[:1, :], hx, hx[-1:, :]], dim=0)
    return ((pad[:-2, :] + pad[1:-1, :] + pad[2:, :]) > 0.0).to(torch.float64)


def _repair_bounds_local(alpha: torch.Tensor, target: float,
                         rounds: int = 8) -> torch.Tensor:
    """Bound ``alpha`` into ``[0, 1]`` and hand the volume back *locally*.

    :func:`tensorfvm.vof_plic._repair_bounds` distributes the clipped volume
    over **every** two-phase cell of the domain, which is harmless for PLIC
    (whose split overshoot is ~1e-3) but destroys a compressive scheme: the
    overshoot of CICSAM appears exactly at the interface, and spreading it
    over the whole interface of a benchmark disk turns a sharp front into a
    diffuse halo.  Here the correction stays inside the 3x3 neighbourhood of
    the cells that actually violated the bounds, weighted by ``alpha(1-alpha)``
    so that it lands on interface cells only.  Both properties are exact:
    ``0 <= alpha <= 1`` and the global sum is preserved to roundoff.
    """
    out = alpha
    tol = 1e-13 * max(1.0, abs(float(target)))
    for _ in range(rounds):
        bad = (out < -1e-15) | (out > 1.0 + 1e-15)
        clipped = out.clamp(0.0, 1.0)
        deficit = float(target) - float(clipped.sum())
        out = clipped
        if not bool(bad.any()) and abs(deficit) <= tol:
            break
        if abs(deficit) <= tol:
            break
        weight = _dilate(bad) * (out * (1.0 - out))
        if float(weight.sum()) <= 0.0:
            weight = _dilate(bad)
        if float(weight.sum()) <= 0.0:
            weight = out * (1.0 - out)
        if float(weight.sum()) <= 0.0:
            break
        out = out + deficit * weight / float(weight.sum())
    return out.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# the transport operator
# --------------------------------------------------------------------------- #
class CICSAMVOF:
    """Conservative CICSAM advection of ``alpha`` on a uniform MAC grid.

    Parameters
    ----------
    nx, ny : int
        Cell counts; ``alpha`` is ``(ny, nx)``.
    dx, dy : float
        Cell sizes (uniform).
    boundary : {"wall", "periodic"}
        ``"wall"`` zeroes the normal velocity on the outermost faces (no flux
        leaves the domain), which makes the case exactly conservative.
    max_cfl : float
        Hard limit on the directional Courant number (the 3-cell stencil is
        only valid up to ``C = 1``).
    k_gamma : float
        Scale of the compressive weight (``1`` = published CICSAM).
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, *,
                 boundary: str = "wall", max_cfl: float = 1.0,
                 k_gamma: float = 1.0) -> None:
        if boundary not in ("wall", "periodic"):
            raise ValueError(f"unknown boundary {boundary!r}")
        if nx < 3 or ny < 3:
            raise ValueError("CICSAM needs at least a 3x3 grid")
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("cell sizes must be positive")
        self.nx, self.ny = int(nx), int(ny)
        self.dx, self.dy = float(dx), float(dy)
        self.boundary = boundary
        self.max_cfl = float(max_cfl)
        self.k_gamma = float(k_gamma)
        self.sweeps = 0
        self.last_cfl = 0.0
        self.last_diagnostics: dict = {}

    # -- public ---------------------------------------------------------- #
    def advect(self, alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
               dt: float, order: str | None = None) -> torch.Tensor:
        """Advance ``alpha`` by ``dt`` (drop-in for ``AlgebraicVOF.advect``).

        ``u`` is ``(ny, nx+1)`` and ``v`` is ``(ny+1, nx)`` (staggered faces).
        """
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        alpha = alpha.to(torch.float64)
        u = u.to(torch.float64).clone()
        v = v.to(torch.float64).clone()
        if self.boundary == "wall":
            u[:, 0] = 0.0
            u[:, -1] = 0.0
            v[0, :] = 0.0
            v[-1, :] = 0.0
        cfl = max(float(u.abs().max().item()) * dt / self.dx,
                  float(v.abs().max().item()) * dt / self.dy)
        self.last_cfl = cfl
        if cfl > self.max_cfl + 1e-12:
            raise ValueError(
                f"directional CFL {cfl:.3f} exceeds max_cfl {self.max_cfl:.3f}: "
                "the 3-cell CICSAM stencil is unstable beyond Courant 1")
        if dt == 0.0:
            return alpha
        if order is None:
            order = "xy" if self.sweeps % 2 == 0 else "yx"
        elif order not in ("xy", "yx"):
            raise ValueError(f"unknown sweep order {order!r}")
        self.sweeps += 1

        target = float(alpha.sum().item())
        clipped = 0.0
        swept = 0
        steps = ("x", "y") if order == "xy" else ("y", "x")
        out = alpha
        for direction in steps:
            if direction == "x":
                out, vol = self._sweep(out, u, dt, self.dx, 1)
            elif self.ny > 1:
                out, vol = self._sweep(out, v, dt, self.dy, 0)
            else:
                vol = 0.0
            clipped += vol
            swept += 1
        # mass-exact bound repair: clip into [0, 1] and hand the removed
        # volume back to the two-phase cells (see vof_plic._repair_bounds)
        raw = out
        out = _repair_bounds_local(out, target)
        self.last_diagnostics = {
            "sweeps": swept,
            "cfl": cfl,
            "pre_repair_min": float(raw.min().item()),
            "pre_repair_max": float(raw.max().item()),
            "clip_volume_per_sweep": clipped,
            "repaired_volume": float(out.sum().item()) - float(raw.sum().item()),
            "mass_before": target,
            "mass_after": float(out.sum().item()),
        }
        return out

    # -- internals ------------------------------------------------------- #
    def _sweep(self, alpha: torch.Tensor, vel: torch.Tensor, dt: float,
               h: float, axis: int) -> tuple[torch.Tensor, float]:
        """One conservative CICSAM sweep along ``axis`` (1 = x, 0 = y)."""
        work = alpha if axis == 1 else alpha.t().contiguous()
        vface = vel if axis == 1 else vel.t().contiguous()
        n_along = self.nx if axis == 1 else self.ny
        n_cross = self.ny if axis == 1 else self.nx
        hx, hy = (self.dx, self.dy) if axis == 1 else (self.dy, self.dx)

        periodic = self.boundary == "periodic"
        ext = _pad2(work, 1, periodic)             # (n_cross, n_along + 4)
        a_l = ext[:, 1:n_along + 2]                # cell f-1
        a_r = ext[:, 2:n_along + 3]                # cell f
        a_ll = ext[:, 0:n_along + 1]               # cell f-2
        a_rr = ext[:, 3:n_along + 4]               # cell f+1
        positive = vface > 0.0
        a_d = torch.where(positive, a_l, a_r)
        a_a = torch.where(positive, a_r, a_l)
        a_u = torch.where(positive, a_ll, a_rr)

        cour = (vface.abs() * dt / h).clamp(max=1.0)
        nx_c, ny_c = _centre_normal(work, hx, hy)
        n_along = _face_average(nx_c, 1)
        n_cross = _face_average(ny_c, 1)
        gamma = blend_gamma(n_along, n_cross, self.k_gamma)

        a_f = cicsam_face_value(a_d, a_a, a_u, cour, gamma)
        a_f = a_f.clone()
        if periodic:
            # faces 0 and n coincide: sharing the value keeps the telescoping
            # sum closed (a per-face zeroing would leak the wrap-around flux)
            a_f[:, -1] = a_f[:, 0]
        else:
            a_f[:, 0] = 0.0                        # wall: no boundary flux
            a_f[:, -1] = 0.0
        flux = vface * a_f
        updated = work - (dt / h) * (flux[:, 1:] - flux[:, :-1])

        target = float(work.sum().item())
        clipped = float(updated.clamp(0.0, 1.0).sum().item() - target)
        if axis == 1:
            return updated, clipped
        return updated.t().contiguous(), clipped

# --------------------------------------------------------------------------- #
# interface-sharpness metric
# --------------------------------------------------------------------------- #
def interface_thickness_cells(alpha: torch.Tensor, lo: float = 0.1,
                              hi: float = 0.9) -> dict:
    """Thickness of the two-phase band in cells, from ``1/max|d alpha/d n|``.

    The gradient is taken as the **largest one-sided cell-to-cell difference**
    along the four axis directions at each interface cell.  For a profile that
    rises from 0 to 1 over ``L`` cells the largest step is ``1/L``, so
    ``1/max|Δα|`` returns ``L`` directly, in cell units:

    * a one-cell (sharp) interface measures ``1.0``;
    * a two-cell band (e.g. ``0, 0.5, 1``) measures ``2.0``;
    * a smeared field measures ``> 3``.

    The *central* difference used by a naive curvature stencil would report
    ``2.0`` for a perfectly sharp step (it sees ``(1-0)/2``), which is why the
    one-sided maximum is the right estimator here: it is a lower bound on
    ``|grad alpha|`` and therefore an upper bound on the thickness -- the
    conservative direction for an acceptance criterion.

    Statistics are reported over the interface cells only (``lo < alpha < hi``);
    the 90th percentile is the acceptance number, since a few grazing cells of
    a curved interface carry no information about the scheme.

    Returns ``{"median", "mean", "p90", "max", "cells", "interface_cells"}``.
    """
    a = alpha.to(torch.float64)
    if a.dim() != 2:
        raise ValueError("expected a 2-D volume-fraction field")
    # one-sided differences with replicated edges, in cell units
    dxp = torch.zeros_like(a)
    dxm = torch.zeros_like(a)
    dxp[:, :-1] = (a[:, 1:] - a[:, :-1]).abs()
    dxm[:, 1:] = (a[:, 1:] - a[:, :-1]).abs()
    dyp = torch.zeros_like(a)
    dym = torch.zeros_like(a)
    dyp[:-1, :] = (a[1:, :] - a[:-1, :]).abs()
    dym[1:, :] = (a[1:, :] - a[:-1, :]).abs()
    grad = torch.maximum(torch.maximum(dxp, dxm), torch.maximum(dyp, dym))
    band = (a > lo) & (a < hi) & (grad > 1e-9)
    n = int(band.sum().item())
    empty = {"median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0,
             "cells": [int(a.shape[1]), int(a.shape[0])], "interface_cells": 0}
    if n == 0:
        return empty
    thick = (1.0 / grad[band]).flatten()
    q = torch.quantile(thick, torch.tensor([0.5, 0.9], dtype=torch.float64))
    return {
        "median": float(q[0].item()),
        "mean": float(thick.mean().item()),
        "p90": float(q[1].item()),
        "max": float(thick.max().item()),
        "cells": [int(a.shape[1]), int(a.shape[0])],
        "interface_cells": n,
    }
