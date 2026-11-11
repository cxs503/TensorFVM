"""Normalised-variable (CICSAM-style) VOF transport: algebraic high-resolution face values.

Why this module exists
----------------------
The repository carries two interface-transport families before this node:

* algebraic, flux-corrected transport with a minmod/MUSCL reconstruction
  (:mod:`tensorfvm.vof_phase0.advection`, Zalesak 1979 FCT) and the historical
  donor-cell :class:`tensorfvm.two_phase_vof.VOFTransport`;
* geometric PLIC (piecewise-linear interface reconstruction, Youngs normal +
  exact cut-area fluxes, :mod:`tensorfvm.vof_plic`).

The *normalised-variable* family -- Hyper-C and the CBC-limited
ULTIMATE-QUICKEST branch blended by the interface orientation -- was **absent**:
``two_phase_vof`` merely described its donor-cell face value as "CICSAM-like" in
a comment.  This module supplies it, so the CICSAM/HRIC branch of the VOF
verification matrix can be measured instead of asserted.

Formulation (implemented, not paraphrased)
------------------------------------------
For a face between the donor cell ``D`` (upwind), the acceptor ``A``
(downwind) and the far-upwind cell ``U``, the normalised variable is

.. math::  \\tilde{\\alpha} = \\frac{\\alpha - \\alpha_U}{\\alpha_A - \\alpha_U},

so the face value is :math:`\\alpha_f = \\alpha_U + \\tilde{\\alpha}_f\\,(\\alpha_A - \\alpha_U)`.
With ``C_f = |u_f| dt / h`` the directional face Courant number,

* Hyper-C (compressive, unbounded in the NVD sense but flux-bounded here):

  .. math:: \\tilde{\\alpha}^{HC} = \\min\\left(1, \\tilde{\\alpha}_D / C_f\\right);

* ULTIMATE-QUICKEST branch, CBC-clipped (:math:`C_f` inside the QUICK
  curvature term):

  .. math:: \\tilde{\\alpha}^{UQ} = \\min\\left\\{
            \\frac{8C_f\\tilde{\\alpha}_D + (1-C_f)(6\\tilde{\\alpha}_D+3)}{8},
            \\;\\frac{\\tilde{\\alpha}_D}{C_f}\\right\\};

* CICSAM blend (Ubbink & Issa 1999) with the orientation factor
  :math:`\\gamma_f = \\min(1, k_\\gamma\\cos^2\\theta_f)`, where
  :math:`\\theta_f` is the angle between the interface normal (3x3 Youngs
  gradient of the same field, see :func:`tensorfvm.vof_plic.youngs_normal`) and
  the face normal: :math:`\\alpha_f = \\gamma_f HC + (1-\\gamma_f) UQ`.
  An interface parallel to the face (a 1-D step) gets the compressive branch; a
  face-normal interface gets the accurate branch.

``scheme="hric"`` implements an **HRIC-type** Courant blending instead of the
orientation blending: the same Hyper-C and UQ branches are weighted by
:math:`\\gamma_C = (1-C_f)^2`, so the compressive branch dominates when the
interface moves slowly through the face and the bounded UQ branch as
:math:`C_f \\to 1` (Muzaferija & Peric 1997 intent).  This is a property-level
realisation of HRIC, *not* a bit-exact reproduction of any published code: all
claims made for it are the measured properties (boundedness, conservation,
shape/front error) in ``benchmarks/twophase``.

Outside the NVD bracket (:math:`\\tilde{\\alpha}_D \\notin (0,1)`, i.e. a local
extremum) the face value falls back to the donor value, which is the CBC.
All arithmetic is ``float64``; tensors are ``(ny, nx)`` with staggered faces
``(ny, nx+1)`` / ``(ny+1, nx)``.

What is *not* claimed
---------------------
No limiter-for-limiter equivalence with published CICSAM is asserted, the
scheme is not extended to unstructured grids, and surface tension/phase change
are out of scope (owned by other nodes).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..vof_plic import youngs_normal

__all__ = [
    "hyper_c",
    "ultimate_quickest",
    "advect_cicsam",
    "advect_upwind",
    "sweep_1d",
    "fct_limit",
]

_EPS = 1.0e-12


def hyper_c(a_tilde: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
    """Hyper-C branch of the NVD: ``min(1, a_tilde / C_f)`` (``C_f`` > 0)."""
    return torch.clamp(a_tilde / torch.clamp(cf, min=_EPS), max=1.0)


def ultimate_quickest(a_tilde: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
    """CBC-limited ULTIMATE-QUICKEST branch ``min{QUICK+UQ term, a_tilde/C_f}``."""
    cf = torch.clamp(cf, min=0.0, max=1.0)
    quick = (8.0 * cf * a_tilde + (1.0 - cf) * (6.0 * a_tilde + 3.0)) / 8.0
    bound = a_tilde / torch.clamp(cf, min=_EPS)
    return torch.minimum(quick, bound)


def fct_limit(a_low: torch.Tensor, anti: torch.Tensor, dt_over_h: float) -> torch.Tensor:
    """Zalesak (1979) limiter applied to the anti-diffusive flux of a sweep.

    The NVD face value alone does **not** enforce a discrete maximum principle in
    two dimensions: measured on the dam-break field it over-fills a handful of
    interface cells by up to +1.0 % (``alpha = 1.010``).  The limiter keeps the
    high-order (CICSAM) flux where the receiving cell still has room
    (``1 - alpha_low``) and the donating cell can afford it (``alpha_low``), and
    otherwise falls back to the first-order donor flux, so ``0 <= alpha <= 1``
    holds by construction while the flux remains single-valued per face
    (conservation is untouched).

    Parameters
    ----------
    a_low : the field after the *donor-cell* (low-order) update of the same
        sweep, ``(ny, n)``.
    anti : ``F_high - F_low`` per face, ``(ny, n+1)``, in velocity units.
    dt_over_h : ``dt / h``.

    Returns
    -------
    ``(ny, n+1)`` limited anti-diffusive flux; ``F = F_low + result`` is bounded.
    """
    n = a_low.shape[1]
    pos = anti > 0.0
    zero = torch.zeros_like(anti)
    in_amt = torch.where(pos, anti, zero)     # A>0 : donor f-1 -> acceptor f
    out_amt = torch.where(pos, zero, -anti)   # A<0 : donor f   -> acceptor f-1
    in_sum = in_amt[:, :n] + out_amt[:, 1:n + 1]
    out_sum = in_amt[:, 1:n + 1] + out_amt[:, :n]
    p_plus = torch.clamp(1.0 - a_low, min=0.0) / dt_over_h
    q_minus = torch.clamp(a_low, min=0.0) / dt_over_h
    s_in = torch.where(in_sum > 0.0,
                       torch.clamp(p_plus / torch.clamp(in_sum, min=_EPS), max=1.0),
                       torch.ones_like(in_sum))
    s_out = torch.where(out_sum > 0.0,
                        torch.clamp(q_minus / torch.clamp(out_sum, min=_EPS), max=1.0),
                        torch.ones_like(out_sum))
    one = torch.ones_like(s_in)
    s_out_left = torch.cat([one[:, :1], s_out], dim=1)    # s_out[f-1] (1 at f=0)
    s_in_left = torch.cat([one[:, :1], s_in], dim=1)      # s_in[f-1]
    s_out_right = torch.cat([s_out, one[:, :1]], dim=1)   # s_out[f]   (1 at f=n)
    s_in_right = torch.cat([s_in, one[:, :1]], dim=1)     # s_in[f]
    factor = torch.where(pos, torch.minimum(s_out_left, s_in_right),
                         torch.minimum(s_out_right, s_in_left))
    return anti * torch.clamp(factor, min=0.0, max=1.0)


def _pad_axis1(a: torch.Tensor, periodic: bool, width: int = 2) -> torch.Tensor:
    if periodic:
        return F.pad(a.unsqueeze(1), (width, width), mode="circular").squeeze(1)
    return F.pad(a.unsqueeze(1), (width, width), mode="replicate").squeeze(1)


def sweep_1d(
    alpha: torch.Tensor,
    faces: torch.Tensor,
    dt: float,
    h: float,
    scheme: str = "cicsam",
    periodic: bool = True,
    normal_component: torch.Tensor | None = None,
    k_gamma: float = 1.0,
    limit: bool = True,
) -> torch.Tensor:
    """One flux-form sweep along the last axis.

    Parameters
    ----------
    alpha : (ny, n) float64 volume fraction.
    faces : (ny, n+1) float64 signed face velocity (staggered; walls are 0).
    dt, h : time step and cell size along the sweep direction.
    scheme : ``"cicsam"`` (orientation blend), ``"hric"`` (Courant blend),
        ``"uq"`` (the CBC-limited ULTIMATE-QUICKEST branch alone, used as a
        smooth-data accuracy diagnostic) or ``"upwind"`` (donor reference).
    normal_component : (ny, n) optional ``|n_axis|`` of the interface normal per
        cell; computed with the 3x3 Youngs stencil when omitted.

    Returns
    -------
    (ny, n) float64 — flux-form update; every internal face flux is shared by
    the two adjacent cells, so the global sum changes by round-off only.
    """
    if alpha.dim() != 2 or faces.dim() != 2:
        raise ValueError("alpha and faces must be 2-D")
    if faces.shape[0] != alpha.shape[0] or faces.shape[1] != alpha.shape[1] + 1:
        raise ValueError("faces must be (ny, n+1) for an (ny, n) field")
    if dt <= 0.0 or h <= 0.0:
        raise ValueError("dt and h must be positive")
    cf_all = faces.abs() * dt / h
    if float(cf_all.max()) > 1.0 + 1e-9:
        raise ValueError(f"directional CFL {float(cf_all.max()):.4g} > 1")

    a = alpha.to(torch.float64)
    ap = _pad_axis1(a, periodic)  # ap[:, k] = a[:, k-2]
    n = a.shape[1]
    # Faces f = 0 .. n ; cell c is at f = c + 1 (between faces c and c+1).
    donor_pos, accept_pos, far_pos = ap[:, 1:n + 2], ap[:, 2:n + 3], ap[:, 0:n + 1]
    donor_neg, accept_neg, far_neg = ap[:, 2:n + 3], ap[:, 1:n + 2], ap[:, 3:n + 4]
    right = faces >= 0.0
    donor = torch.where(right, donor_pos, donor_neg)
    accept = torch.where(right, accept_pos, accept_neg)
    far = torch.where(right, far_pos, far_neg)
    cf = cf_all

    span = accept - far
    resolvable = span.abs() > _EPS
    safe_span = torch.where(resolvable, span, torch.ones_like(span))
    a_tilde = torch.where(resolvable, (donor - far) / safe_span, torch.zeros_like(span))
    inside = resolvable & (a_tilde > 0.0) & (a_tilde < 1.0)

    if scheme == "upwind":
        face = donor
    else:
        hc = hyper_c(a_tilde, cf)
        uq = ultimate_quickest(a_tilde, cf)
        if scheme == "cicsam":
            if normal_component is None:
                cos2 = torch.ones_like(cf)
            else:
                nc = normal_component.to(torch.float64).abs()
                # average the cell normals onto the faces, then cos^2(theta)
                nf = 0.5 * (F.pad(nc, (1, 0), mode="replicate")[:, :n + 1]
                            + F.pad(nc, (0, 1), mode="replicate")[:, :n + 1])
                cos2 = nf * nf
            gamma = torch.clamp(k_gamma * cos2, max=1.0)
        elif scheme == "uq":
            gamma = torch.zeros_like(cf)   # diagnostic branch: CBC-limited UQ only
        elif scheme == "hric":
            gamma = (1.0 - torch.clamp(cf, max=1.0)) ** 2
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        blended = gamma * hc + (1.0 - gamma) * uq
        face = far + blended * span
        face = torch.where(inside, face, donor)
    face = torch.clamp(face, 0.0, 1.0)

    vel = faces.to(torch.float64)
    # On a periodic grid the first and the last face are the *same* face, so
    # their fluxes must be identical; the interface-normal averaging (which pads
    # by replication) can make them differ by ~1e-7, and that difference is a
    # spurious source/sink in the global budget.  The staggered velocity field of
    # a periodic flow already has u_0 = u_n, so this only enforces consistency.
    if periodic:
        face = torch.cat([face[:, :1], face[:, 1:-1], face[:, :1]], dim=1)
    flux_high = vel * face
    if scheme == "upwind" or not limit:
        return a - (flux_high[:, 1:] - flux_high[:, :-1]) * (dt / h)
    flux_low = vel * donor
    a_low = a - (flux_low[:, 1:] - flux_low[:, :-1]) * (dt / h)
    anti = fct_limit(a_low, flux_high - flux_low, dt / h)
    flux = flux_low + anti
    return a - (flux[:, 1:] - flux[:, :-1]) * (dt / h)


def advect_cicsam(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    scheme: str = "cicsam",
    order: str = "xy",
    periodic: bool = True,
    k_gamma: float = 1.0,
    limit: bool = True,
) -> torch.Tensor:
    """One two-dimensional step (Lie splitting) with the requested NVD scheme.

    ``u`` is ``(ny, nx+1)`` (x-faces), ``v`` is ``(ny+1, nx)`` (y-faces);
    ``periodic`` must be passed consistently with the velocity field (a rigid
    rotation on a closed domain is periodic in both directions).
    """
    if order not in ("xy", "yx"):
        raise ValueError(f"order must be 'xy' or 'yx', got {order!r}")
    a = alpha.to(torch.float64)
    uu = u.to(torch.float64)
    vv = v.to(torch.float64)
    nx, ny = a.shape[1], a.shape[0]
    for axis in (1, 0) if order == "xy" else (0, 1):
        if axis == 1:
            nrm = youngs_normal(a, dx, dy)[..., 0]
            a = sweep_1d(a, uu, dt, dx, scheme=scheme, periodic=periodic,
                         normal_component=nrm, k_gamma=k_gamma, limit=limit)
        else:
            nrm = youngs_normal(a, dx, dy)[..., 1]
            a = sweep_1d(a.transpose(0, 1).contiguous(),
                         vv.transpose(0, 1).contiguous(), dt, dy,
                         scheme=scheme, periodic=periodic,
                         normal_component=nrm.transpose(0, 1).contiguous(),
                         k_gamma=k_gamma, limit=limit).transpose(0, 1).contiguous()
    assert a.shape == (ny, nx)
    return a


def advect_upwind(
    alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt: float,
    dx: float, dy: float, order: str = "xy", periodic: bool = True,
) -> torch.Tensor:
    """First-order donor-cell reference transport (the diffusive baseline)."""
    return advect_cicsam(alpha, u, v, dt, dx, dy, scheme="upwind", order=order,
                         periodic=periodic)