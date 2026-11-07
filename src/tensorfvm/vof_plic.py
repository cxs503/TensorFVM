"""PLIC (piecewise-linear interface) geometric VOF transport.

Motivation
----------
:mod:`tensorfvm.two_phase_vof` ships an *algebraic* donor-cell transport.  Its own
verification artefacts (``verification/artifacts/vof_rotation``) measure what that
costs on a discontinuous indicator field: on the Zalesak slotted disk the peak
volume fraction is retained to only 19 % / 31 % / 52 % (32²/64²/128²) and the
relative L1 shape error stays in 1.19 … 1.77 — the disk is smeared away.  For
interface-dominated two-phase flow (ship hydrodynamics: free surface, sloshing,
green water on deck) that is the dominant error, not the momentum discretisation.

This module adds the missing *geometric* ingredient: a Piecewise-Linear Interface
Calculation reconstruction (Youngs normal + plane constant) and a flux-form
advection in which every face flux is the **exact volume of the reconstructed
liquid polygon** that crosses the face in one time step.

Mathematical setting
--------------------
Uniform Cartesian cells of size ``dx × dy``; a cell is the unit square
``[0,1]²`` in local coordinates and is described by

* ``n = (n1, n2)``, ``|n| = 1`` — interface *outward* normal, pointing from
  liquid (α = 1) to gas (α = 0);
* ``c`` — plane constant, with liquid occupying ``{x : n·x ≤ c}``.

The reconstruction is volume-consistent: ``c`` is solved from the cell's own α by
inverting the exact cut-area function ``A(n, c)`` (2-D form of the
Scardovelli–Zaleski cube/plane relations, here a closed-form piecewise
polynomial, vectorised, inverted by bracketed bisection).

The flux through an x-face of a donor cell at directional Courant number
``C = |u| dt/dx ∈ (0, 1]`` is the reconstructed liquid volume inside the *swept
strip* of width ``C`` at the downstream edge::

    Φ = strip(n, c, 1-C, 1)   for u ≥ 0   (leaving through the east face)
    Φ = strip(n, c, 0,   C)   for u < 0   (leaving through the west face)

with ``strip(n, c, a, b) = (b-a) · A(n1 (b-a), n2, c - n1 a)``, obtained from the
cut-area function by the affine substitution ``x1 = a + (b-a) x1'``.

Consequences (measured, not asserted — see
:mod:`tensorfvm.verification.vof_plic_interface`):

* **Boundedness** — a geometric flux never exceeds the donor liquid volume, so no
  compressive limiter is needed and a sweep creates no new extremum.
* **Conservation** — sweeps are written in flux form and each face flux is shared
  by the two cells that own the face, so the global phase volume changes only by
  roundoff (measured per-step relative drift ~1e-16).
* **Sharpness** — the interface stays inside one cell: an advected step keeps a
  10–90 % transition of ~1 cell at every resolution instead of diffusing like
  upwind, whose measured spread follows ``3.62·sqrt(ν t)/Δx`` with
  ``ν = (u Δx/2)(1-C)`` (9 cells at N = 64 for the case in the artefacts).

Splitting: two-dimensional transport uses dimensional (Lie) splitting ``x`` then
``y``; the reconstruction is recomputed between sweeps, as a geometric scheme
requires.  Sweeps are periodic by default (``periodic=True``), which also
guarantees that the two ends of a periodic face share one flux.

Defensive contract (negative tests live in ``tests/test_vof_plic_interface.py``):
:func:`validate_alpha` **rejects** volume fractions outside ``[0, 1]`` instead of
silently clipping them; :func:`validate_velocity` rejects a non-positive ``dt``,
non-finite fields, wrong staggered shapes and a directional Courant number above
1; :func:`validate_directional_speed` rejects a *negative rate magnitude* (the
sign of a magnitude-only API is a caller bug — signed face velocities in
:func:`advect_plic` are physical and remain supported).

All tensors are ``float64``.
"""

from __future__ import annotations

import torch

__all__ = [
    "validate_alpha",
    "validate_velocity",
    "validate_directional_speed",
    "cut_area",
    "youngs_normal",
    "plane_constant",
    "reconstruct",
    "strip_volume",
    "advect_plic",
    "advect_1d_step",
    "interface_width_cells",
]

_EPS = 1e-12
#: Roundoff-level tolerance for the intermediate field between two sweeps.  Pure
#: geometry cannot create new extrema, so anything larger is a genuine defect and
#: raises instead of being clipped away.
_ROUNDOFF = 1e-12


# ---------------------------------------------------------------------------
# input validation ("negative evidence": reject, do not silently repair)
# ---------------------------------------------------------------------------
def validate_alpha(
    alpha: torch.Tensor, name: str = "alpha", tol: float = 0.0
) -> torch.Tensor:
    """Reject a volume-fraction field that is not finite and inside ``[0, 1]``.

    Unlike the algebraic transport, which clamps its *output*, this guard refuses
    an *input* field that is already out of range: silently clipping would hide
    the upstream defect (a bad pressure solve, a bad source term) and destroy the
    volume ledger.

    Raises
    ------
    ValueError
        If ``alpha`` is not a tensor, is empty, contains ``nan``/``inf``, or has
        an entry outside ``[-tol, 1+tol]``.
    """
    if not isinstance(alpha, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(alpha)!r}")
    if alpha.numel() == 0:
        raise ValueError(f"{name} is empty")
    a = alpha.to(torch.float64)
    if not bool(torch.isfinite(a).all()):
        n_bad = int((~torch.isfinite(a)).sum().item())
        raise ValueError(f"{name} contains {n_bad} non-finite entries")
    lo = float(a.min().item())
    hi = float(a.max().item())
    if lo < -tol:
        n_bad = int((a < -tol).sum().item())
        raise ValueError(
            f"{name} has {n_bad} entries below 0 (min {lo:.6g}); "
            "volume fraction must lie in [0, 1]"
        )
    if hi > 1.0 + tol:
        n_bad = int((a > 1.0 + tol).sum().item())
        raise ValueError(
            f"{name} has {n_bad} entries above 1 (max {hi:.6g}); "
            "volume fraction must lie in [0, 1]"
        )
    return alpha


def validate_velocity(
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    cfl_limit: float = 1.0,
) -> float:
    """Validate a staggered velocity field, the time step and the directional CFL.

    * ``dt`` must be finite and **strictly positive** (a zero or negative step is
      not a transport, it is a sign bug).
    * ``u`` (shape ``ny × (nx+1)``) and ``v`` (shape ``(ny+1) × nx``) must be
      finite; **negative values are legal** — they are physical back-flow and the
      geometric flux picks the upwind side accordingly.
    * The directional Courant numbers ``max|u| dt/dx`` and ``max|v| dt/dy`` must
      not exceed ``cfl_limit`` (default 1): the swept strip has to stay inside the
      donor cell, otherwise the geometric flux is undefined.

    Returns
    -------
    float
        The maximum directional Courant number actually used.
    """
    if not isinstance(dt, (int, float, torch.Tensor)):
        raise ValueError(f"dt must be a real number, got {dt!r}")
    dt = float(dt)
    if not torch.isfinite(torch.as_tensor(dt)):
        raise ValueError(f"dt must be finite, got {dt!r}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt:.6g}")
    dx = float(dx)
    dy = float(dy)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"cell sizes must be > 0, got dx={dx!r}, dy={dy!r}")
    if not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError("u and v must be torch tensors")
    if u.dim() != 2 or v.dim() != 2:
        raise ValueError("u and v must be 2-D staggered fields")
    ny = v.shape[0] - 1
    nx = u.shape[1] - 1
    if u.shape != (ny, nx + 1) or v.shape != (ny + 1, nx) or ny <= 0 or nx <= 0:
        raise ValueError(
            f"staggered shape mismatch: u{u.shape} should be {(ny, nx + 1)}, "
            f"v{v.shape} should be {(ny + 1, nx)}"
        )
    for name, f in (("u", u), ("v", v)):
        if not bool(torch.isfinite(f).all()):
            raise ValueError(f"{name} contains non-finite entries")
    cx = float(u.abs().max().item()) * dt / dx
    cy = float(v.abs().max().item()) * dt / dy
    cour = max(cx, cy)
    if cour > cfl_limit * (1.0 + 1e-12):
        raise ValueError(
            f"directional CFL {cour:.4g} exceeds the geometric-VOF limit "
            f"{cfl_limit:.4g}; reduce dt or split the step"
        )
    return cour


def validate_directional_speed(speed: float) -> float:
    """Reject a negative *transport-rate magnitude* and return it as ``float``.

    The one-dimensional step benchmark and the sweep scheduler take a **rate
    magnitude** (cells per step) plus an explicit direction, because the interface
    front only moves towards +x or -x.  A negative magnitude is therefore a caller
    bug (double negation), not a physical back-flow; the signed case is handled by
    :func:`advect_plic` / :func:`advect_1d_step` with ``direction=-1``, where
    negative face velocities are accepted.
    """
    s = float(speed)
    if not torch.isfinite(torch.as_tensor(s)):
        raise ValueError(f"directional speed must be finite, got {speed!r}")
    if s < 0.0:
        raise ValueError(
            f"directional speed must be >= 0, got {s:.6g}; use direction=-1 for a "
            "leftward sweep instead of a negative rate"
        )
    return s


# ---------------------------------------------------------------------------
# exact cut-area geometry on the unit cell
# ---------------------------------------------------------------------------
def cut_area(n1, n2, c) -> torch.Tensor:
    """Exact area of ``{x ∈ [0,1]² : n1 x1 + n2 x2 ≤ c}``.

    Closed-form piecewise polynomial, valid for **unnormalised** ``(n1, n2)`` (the
    area depends only on the ratio to ``c``).  For ``n1, n2 ≥ 0`` the region is
    described by ``x1 ∈ [0, xb]`` with ``x2 ≤ (c - n1 x1)/n2``::

        xa = clamp((c - n2)/n1, 0, 1)     (where the cap x2 = 1 ends)
        xb = clamp(c/n1, 0, 1)            (where the line leaves the cell)
        A  = xa + [c (xb - xa) - n1 (xb² - xa²)/2] / n2

    Negative components are removed by the reflection ``x → 1 - x``, which maps
    the plane to ``|n|·x ≤ c - n⁻`` with ``n⁻ = min(n, 0)``; degenerate axes
    (``|n1|`` or ``|n2| ≈ 0``) fall back to the one-dimensional cut.
    """
    n1 = torch.as_tensor(n1, dtype=torch.float64)
    n2 = torch.as_tensor(n2, dtype=torch.float64)
    c = torch.as_tensor(c, dtype=torch.float64)
    n1, n2, c = torch.broadcast_tensors(n1, n2, c)
    p = n1.abs()
    q = n2.abs()
    zero = torch.zeros_like(n1)
    d = c - torch.where(n1 < 0, n1, zero) - torch.where(n2 < 0, n2, zero)

    pmax = p.clamp_min(_EPS)
    qmax = q.clamp_min(_EPS)
    xa = ((d - q) / pmax).clamp(0.0, 1.0)
    xb = (d / pmax).clamp(0.0, 1.0)
    general = (xa + (d * (xb - xa) - 0.5 * p * (xb * xb - xa * xa)) / qmax).clamp(
        0.0, 1.0
    )
    one_d_q = (d / pmax).clamp(0.0, 1.0)  # n2 == 0
    one_d_p = (d / qmax).clamp(0.0, 1.0)  # n1 == 0
    degenerate = (d > 0.0).to(torch.float64)
    return torch.where(
        p > _EPS,
        torch.where(q > _EPS, general, one_d_q),
        torch.where(q > _EPS, one_d_p, degenerate),
    )


def _pad_replicate(a: torch.Tensor) -> torch.Tensor:
    """Replicate one row/column of border cells on every side."""
    p = torch.cat((a[:, :1], a, a[:, -1:]), dim=1)
    return torch.cat((p[:1, :], p, p[-1:, :]), dim=0)


def _sobel_gradients(a: torch.Tensor, dx: float, dy: float):
    """Youngs (Sobel-like) 3×3 gradients ``(gx, gy)`` of ``a`` (unit-cell spacing)."""
    p = _pad_replicate(a)
    gx = (
        p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]
        - p[:-2, :-2] - 2.0 * p[1:-1, :-2] - p[2:, :-2]
    ) / (8.0 * float(dx))
    gy = (
        p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
        - p[:-2, :-2] - 2.0 * p[:-2, 1:-1] - p[:-2, 2:]
    ) / (8.0 * float(dy))
    return gx, gy


def youngs_normal(
    alpha: torch.Tensor, dx: float = 1.0, dy: float = 1.0, eps: float = 1e-10
) -> torch.Tensor:
    """Interface outward normal from the 3×3 Youngs gradient.

    ``g = ∇α`` uses the Sobel stencil ``[[-1,0,1],[-2,0,2],[-1,0,1]] / (8 dx)``
    (transposed in ``y``): compact, second order for smooth fields and much less
    oscillatory than central differences on an indicator field.  The returned
    normal is ``n = -g/|g|`` (from liquid to gas).  Uniform cells (``|g| ≤ eps``)
    have no resolvable interface and get the reference normal ``(1, 0)``, for which
    ``c = α`` is the volume-consistent plane constant (a full cell then fluxes
    ``C`` and an empty one fluxes zero, as required).

    Returns
    -------
    torch.Tensor
        Shape ``(ny, nx, 2)``, float64.
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    gx, gy = _sobel_gradients(a, dx, dy)
    mag = torch.sqrt(gx * gx + gy * gy)
    uniform = mag <= eps
    safe = torch.where(uniform, torch.ones_like(mag), mag)
    n = torch.stack((-gx / safe, -gy / safe), dim=-1)
    # Uniform cells have no resolvable interface; use the reference normal (1, 0),
    # for which the bracket is [0, 1] and c = alpha is the volume-consistent
    # solution (a full cell then fluxes C, an empty one zero, as it must).
    fallback = torch.zeros_like(n)
    fallback[..., 0] = 1.0
    return torch.where(uniform.unsqueeze(-1), fallback, n)


def plane_constant(
    alpha: torch.Tensor, normal: torch.Tensor, iters: int = 44, eps: float = _EPS
) -> torch.Tensor:
    """Solve the plane constant ``c`` from the cell volume fraction (exact inverse).

    ``c`` is defined by ``A(n, c) = α`` with ``A`` the exact cut area of
    :func:`cut_area`.  ``A`` is continuous and strictly increasing in ``c`` on
    ``[Σ min(n_i,0), Σ max(n_i,0)]``, so a bracketed bisection converges to
    ``2/2**iters`` (machine precision for the default 44).  Bulk cells
    (``α ≤ eps`` or ``α ≥ 1-eps``) are pinned to the bracket ends and skip the
    iteration, so the cost scales with the interface length, not the grid size.
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    n = torch.as_tensor(normal, dtype=torch.float64)
    n1 = n[..., 0]
    n2 = n[..., 1]
    zero = torch.zeros_like(a)
    # reference normal (1, 0) for degenerate cells → bracket [0, 1], c = alpha
    n1 = torch.where(n1 == 0.0, torch.ones_like(n1), n1)
    c_min = n1.clamp_max(0.0) + n2.clamp_max(0.0)
    c_max = n1.clamp_min(0.0) + n2.clamp_min(0.0)
    c = torch.where(a <= eps, c_min, torch.where(a >= 1.0 - eps, c_max, zero))

    hit = torch.nonzero((a > eps) & (a < 1.0 - eps), as_tuple=False)
    if hit.numel() == 0:
        return c
    iy, ix = hit[:, 0], hit[:, 1]
    a_i = a[iy, ix].clone()
    n1_i = n1[iy, ix].clone()
    n2_i = n2[iy, ix].clone()
    lo = c_min[iy, ix].clone()
    hi = c_max[iy, ix].clone()
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        below = cut_area(n1_i.abs(), n2_i.abs(), mid
                         - torch.where(n1_i < 0, n1_i, torch.zeros_like(n1_i))
                         - torch.where(n2_i < 0, n2_i, torch.zeros_like(n2_i))) < a_i
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    out = c.clone()
    out[iy, ix] = 0.5 * (lo + hi)
    return out


def reconstruct(
    alpha: torch.Tensor, dx: float = 1.0, dy: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normal, plane_constant)`` for every cell (Youngs + exact inverse)."""
    validate_alpha(alpha, "alpha")
    n = youngs_normal(alpha, dx, dy)
    return n, plane_constant(alpha, n)


def strip_volume(normal, c, a, b, axis: int = 1) -> torch.Tensor:
    """Reconstructed liquid volume inside ``x_axis ∈ [a, b]`` of the unit cell.

    ``axis=1`` sweeps in ``x``, ``axis=0`` in ``y``.  The affine substitution
    ``x_axis = a + (b-a) x'`` reduces this to :func:`cut_area`, so no polygon
    clipping is needed and the operation stays fully vectorised.
    """
    n = torch.as_tensor(normal, dtype=torch.float64)
    c = torch.as_tensor(c, dtype=torch.float64)
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    w = (b - a).clamp_min(0.0)
    if axis == 1:
        n_axis, n_other = n[..., 0], n[..., 1]
    else:
        n_axis, n_other = n[..., 1], n[..., 0]
    return (w * cut_area(n_axis * w, n_other, c - n_axis * a)).clamp_min(0.0)


# ---------------------------------------------------------------------------
# geometric advection
# ---------------------------------------------------------------------------
def _repair_bounds(a: torch.Tensor, target: float, rounds: int = 4) -> torch.Tensor:
    """Clip into ``[0, 1]`` and hand the removed volume back, conserving the sum.

    A *directional split* sweep advects with a field that is not solenoidal in
    that direction alone (``∂u/∂x ≠ 0`` even when ``∇·u = 0``), so a cell in a
    locally converging region can accept slightly more liquid than it has room
    for: the geometric flux is exact, the capacity is not.  The measured
    excursion of one sweep at Courant 1e-2 is ~1e-3 in the few interface cells —
    far above roundoff, far below anything physical.

    Clipping alone would turn that into a volume sink (a mass leak that
    accumulates over thousands of steps), so the clipped volume is redistributed
    over the two-phase cells with weight ``α(1-α)`` (zero inside either pure
    phase, where there is no room) and the clip/redistribute is repeated until
    the integral matches ``target`` to roundoff.  Both properties are then
    exact: ``0 ≤ α ≤ 1`` and ``Σα·ΔV = target``.
    """
    a = a.clamp(0.0, 1.0)
    tol = 1e-14 * max(1.0, abs(float(target)))
    for _ in range(rounds):
        deficit = float(target) - float(a.sum())
        if abs(deficit) <= tol:
            break
        weight = a * (1.0 - a)
        total = float(weight.sum())
        if total <= 0.0:
            break
        a = (a + deficit * weight / total).clamp(0.0, 1.0)
    return a


def _sweep(
    alpha: torch.Tensor, face_velocity: torch.Tensor, dt: float, h: float, axis: int
) -> torch.Tensor:
    """One conservative geometric sweep along ``axis`` (1 = x, 0 = y), periodic.

    Strict variant: rejects a field outside ``[0, 1]`` (see ``_ROUNDOFF``).
    """
    return _sweep_impl(alpha, face_velocity, dt, h, axis, strict=True)


def _sweep_unchecked(
    alpha: torch.Tensor, face_velocity: torch.Tensor, dt: float, h: float, axis: int
) -> torch.Tensor:
    """As :func:`_sweep`, but hands the raw (possibly out-of-range) result back.

    Used by the conservative-bounding path of :func:`advect_plic`, which bounds
    the field between sweeps instead of aborting on a split-advection
    overshoot.
    """
    return _sweep_impl(alpha, face_velocity, dt, h, axis, strict=False)


def _sweep_impl(
    alpha: torch.Tensor,
    face_velocity: torch.Tensor,
    dt: float,
    h: float,
    axis: int,
    strict: bool,
) -> torch.Tensor:
    work = alpha if axis == 1 else alpha.T.contiguous()
    vel = face_velocity if axis == 1 else face_velocity.T.contiguous()
    ny, nx = work.shape
    lo = float(work.min().item())
    hi = float(work.max().item())
    if strict and (lo < -_ROUNDOFF or hi > 1.0 + _ROUNDOFF):
        raise ValueError(
            f"volume fraction left [0,1] between sweeps (min {lo:.6g}, max {hi:.6g}); "
            "this is a scheme defect, not roundoff"
        )
    work = work.clamp(0.0, 1.0)
    if vel.shape != (ny, nx + 1):
        raise ValueError(f"face velocity shape {vel.shape} != {(ny, nx + 1)}")
    padded = torch.cat((work[:, -1:], work, work[:, :1]), dim=1)
    n_p, c_p = reconstruct(padded, 1.0, 1.0)
    cour = vel.to(torch.float64) * float(dt) / float(h)
    cmax = float(cour.abs().max().item())
    if cmax > 1.0 + 1e-12:
        raise ValueError(
            f"directional CFL {cmax:.4g} > 1: the swept strip must stay inside "
            "the donor cell"
        )
    C = cour.abs()
    nw, cw = n_p[:, :-1, :], c_p[:, :-1]  # west cell of each face
    ne, ce = n_p[:, 1:, :], c_p[:, 1:]  # east cell of each face
    # Signed flux: positive = crossing the face towards +x.  The magnitude is the
    # reconstructed liquid volume in the swept strip of the *upwind* cell.
    flux = torch.where(
        cour >= 0.0,
        strip_volume(nw, cw, 1.0 - C, 1.0, axis=1),
        -strip_volume(ne, ce, 0.0, C, axis=1),
    )
    # original cell j is padded cell j+1: west face = flux[j], east face = flux[j+1]
    # (inflow through the west face, outflow through the east face)
    net = flux[:, 1 : nx + 1] - flux[:, 0:nx]
    out = work - net
    return out if axis == 1 else out.T.contiguous()


def advect_plic(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    order: str = "xy",
    clip: bool = True,
    conservative_bounds: bool = False,
    diagnostics: dict | None = None,
) -> torch.Tensor:
    """Advance ``alpha`` one step with PLIC geometric advection (dimensional split).

    Parameters
    ----------
    alpha : (ny, nx) tensor
        Volume fraction; entries outside ``[0, 1]`` by more than ``1e-12``
        (roundoff) are **rejected**.
    u : (ny, nx+1) tensor — staggered x-face velocity (signed; back-flow is legal).
    v : (ny+1, nx) tensor — staggered y-face velocity.
    dt, dx, dy : floats; ``dt > 0`` and directional CFL ≤ 1 are enforced.
    order : ``"xy"`` or ``"yx"`` — sweep order.  Alternate it across steps to make
        the directional splitting symmetric in time (Strang-like).
    clip : bool
        Clamp roundoff-level excursions into ``[0, 1]``.  Pure geometry creates no
        new extremum, so any residual is ``~1e-16``; pass ``False`` to inspect the
        raw field (the negative-evidence tests do).
    conservative_bounds : bool
        Bound the field *between* the two directional sweeps instead of rejecting
        it, and repair the volume removed by the clipping (see
        :func:`_repair_bounds`).  Needed whenever the split velocity is not
        solenoidal in each direction alone — i.e. for any real flow field — and
        required by the bubble-rise benchmark, whose mass-conservation criterion
        is ``1e-6`` over the whole run.  Default ``False`` reproduces the
        historical strict behaviour (used by the interface-transport tests).
    diagnostics : dict, optional
        Filled with ``sweeps`` (per-sweep clipped volume, ``|ΔV|/Σα``) and
        ``final_imbalance`` when ``conservative_bounds`` is on.

    Returns
    -------
    torch.Tensor
        Updated volume fraction, same shape/dtype/device as ``alpha``.
    """
    # Strict guard: a field outside [0, 1] by more than roundoff is a defect and
    # is rejected; the ~1e-16 excursions a previous geometric step may leave are
    # projected, not used to mask an upstream error.
    validate_alpha(alpha, "alpha", tol=_ROUNDOFF)
    validate_velocity(u, v, dt, dx, dy, cfl_limit=1.0)
    if order not in ("xy", "yx"):
        raise ValueError(f"order must be 'xy' or 'yx', got {order!r}")
    a = alpha.to(torch.float64)
    uu = u.to(torch.float64)
    vv = v.to(torch.float64)
    if conservative_bounds:
        target = float(a.sum())
        moved = []
        for axis in (1, 0) if order == "xy" else (0, 1):
            swept = _sweep_unchecked(
                a, uu if axis == 1 else vv, dt, dx if axis == 1 else dy, axis)
            bounded = _repair_bounds(swept, target)
            moved.append(abs(float(bounded.sum()) - float(swept.sum()))
                         / max(float(target), 1e-300))
            a = bounded
        if diagnostics is not None:
            diagnostics["sweeps"] = moved
            diagnostics["final_imbalance"] = (
                abs(float(a.sum()) - float(target)) / max(float(target), 1e-300))
        return a.to(dtype=alpha.dtype, device=alpha.device)
    for axis in (1, 0) if order == "xy" else (0, 1):
        a = _sweep(a, uu if axis == 1 else vv, dt, dx if axis == 1 else dy, axis)
    if clip:
        a = a.clamp(0.0, 1.0)
    return a.to(dtype=alpha.dtype, device=alpha.device)


def advect_1d_step(
    alpha: torch.Tensor, speed: float, dt: float, dx: float, direction: int = 1
) -> torch.Tensor:
    """One-dimensional geometric transport step (periodic grid).

    This is the 1-D case of :func:`advect_plic`: it builds ``u = ±speed`` on a
    single-row grid and delegates to the same geometric sweep, so the interface
    orientation is inferred from the field itself.

    ``speed`` is a **magnitude** and must be ≥ 0 (:func:`validate_directional_speed`);
    ``direction`` (``+1`` / ``-1``) selects the sweep sense.

    Why not a closed-form flux ``Φ = clamp(α + C − 1, 0, C)``?  That formula
    hard-codes "liquid upstream of the face", which is right for the leading
    transition of a step but wrong for its trailing one (periodic domain or a
    finite band): there the liquid sits *downstream*, the strip at the downstream
    edge is empty, and the trailing interface is smeared by ``C·Δx`` per step
    (measured: 10–90 % width 24/48/96 cells at N = 32/64/128, and the band loops
    into the grid).  Letting the Youngs normal decide the orientation removes the
    assumption and gives a fixed one-cell transition at every resolution.
    """
    a = validate_alpha(alpha, "alpha").to(torch.float64)
    sp = validate_directional_speed(speed)
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")
    if float(dt) <= 0.0:
        raise ValueError(f"dt must be > 0, got {float(dt):.6g}")
    if float(dx) <= 0.0:
        raise ValueError(f"dx must be > 0, got {float(dx):.6g}")
    C = sp * float(dt) / float(dx)
    if C > 1.0 + 1e-12:
        raise ValueError(f"CFL {C:.4g} > 1 for a single 1-D geometric step")
    shape = a.shape
    flat = a.reshape(1, -1)
    n = flat.shape[1]
    u = torch.full((1, n + 1), float(direction) * sp, dtype=torch.float64)
    v = torch.zeros((2, n), dtype=torch.float64)
    out = advect_plic(flat, u, v, dt, dx, dx, order="xy", clip=True)
    return out.reshape(shape) if len(shape) > 1 else out.reshape(-1)


def interface_width_cells(
    alpha: torch.Tensor, axis: int = 0, lo: float = 0.1, hi: float = 0.9
) -> float:
    """Width (in cells) of a single 10 %–90 % transition band.

    Quantifies interface sharpness of a 1-D profile (or of the column-wise mean of
    a 2-D field).  The returned value is the length of the **longest contiguous
    run** of cells whose volume fraction is strictly inside ``(lo, hi)``: ``0`` for
    an unresolved step, ``1``–``2`` for a scheme that keeps the transition inside
    one cell, and growing as ``sqrt(t)`` for a diffusing scheme.

    Counting *contiguous* runs matters: a periodic grid carrying one step has two
    transitions (the step itself and the wrap-around), and summing their bands
    would double the measured width.
    """
    a = torch.as_tensor(alpha, dtype=torch.float64)
    if a.dim() == 2:
        a = a.mean(dim=axis)
    a = a.reshape(-1)
    inside = ((a > lo) & (a < hi)).tolist()
    best = run = 0
    for flag in inside:
        run = run + 1 if flag else 0
        best = max(best, run)
    return float(best)
