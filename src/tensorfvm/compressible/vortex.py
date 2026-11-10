"""Vortex diagnostics: Lamb-Oseen pair initial condition, vorticity, merging.

The planar analogue of vortex-ring merging used by this surface is a
co-rotating pair of identical compressible vortices.  Three measurements are
used, each with a definite analytic reference:

``circulation``
    ``Gamma = \\int \\omega dA`` must stay at twice the single-vortex
    circulation; Kelvin's theorem makes this a conservation check, not an
    approximation.
``principal_axis_angle``
    The second-moment principal axis of ``|omega|`` is a line through the
    centroid; for two equal point vortices separated by ``b`` the pair rotates
    rigidly at ``Omega = Gamma/(pi b^2)`` (Lamb 1932, *Hydrodynamics*, art. 155),
    so the unwrapped axis angle has that slope.
``core_separation`` / ``merged``
    Peak detection on the smoothed enstrophy field: two cores while the pair is
    separate, one core after merging.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

__all__ = [
    "lamb_oseen_pair",
    "vorticity",
    "principal_axis_angle",
    "enstrophy",
    "core_separation",
    "core_positions",
    "total_circulation",
    "swirl_mach",
]


def lamb_oseen_pair(
    *,
    n: int,
    size: float,
    circulation: float,
    core: float,
    separation: float,
    mach_vortex: float | None = None,
    gamma: float = 1.4,
    p_inf: float = 1.0,
    rho_inf: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Superposition of two equal co-rotating Lamb-Oseen vortices.

    Vortex 1 sits at ``(-b/2, 0)`` and vortex 2 at ``(+b/2, 0)``; both carry
    positive circulation, so the pair rotates counter-clockwise as a rigid
    dipole-free pair.  The pressure follows ``dp/dr = rho u_theta^2 / r`` applied
    to each swirl independently and the density is recovered isentropically, so
    the initial field carries no acoustic transient beyond the (small) error of
    superposing two swirl-induced pressure fields.
    """
    d = size / n
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * d
    X = x.reshape(-1, 1).expand(n, n)
    Y = x.reshape(1, -1).expand(n, n)
    c = math.sqrt(gamma * p_inf / rho_inf)

    # the pair is centred in the periodic box: placing a core outside the domain
    # (e.g. at x = -b/2 in [0, 2*pi]) silently clips half of it
    xc, yc = 0.5 * size, 0.5 * size

    def swirl(cx: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = X - (xc + cx)
        dy = Y - yc
        r2 = dx * dx + dy * dy
        a2 = core * core
        # Lamb-Oseen azimuthal velocity, vectorised
        g = 1.0 - torch.exp(-r2 / a2)
        fac = circulation / (2.0 * math.pi) * g / torch.clamp(r2, min=1e-14)
        u = -fac * dy
        v = fac * dx
        # |u_theta|^2 for the radial pressure balance
        mag2 = (fac * fac) * r2
        return u, v, mag2

    u1, v1, m1 = swirl(-0.5 * separation)
    u2, v2, m2 = swirl(+0.5 * separation)
    u = u1 + u2
    v = v1 + v2
    scale = 1.0
    peak = float(torch.sqrt((u * u + v * v).max()))
    if mach_vortex is not None and peak > 0.0:
        allowed = mach_vortex * c
        if peak > allowed:
            scale = allowed / peak
            u, v = u * scale, v * scale
            m1, m2 = m1 * scale * scale, m2 * scale * scale
    p = p_inf - 0.5 * rho_inf * (m1 + m2)
    p = torch.clamp(p, min=0.2 * p_inf)
    rho = rho_inf * torch.pow(p / p_inf, 1.0 / gamma)
    ones = torch.ones_like(X)
    return {"rho": rho, "u": u, "v": v, "p": p, "Y": torch.zeros_like(ones)}


def vorticity(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float,
              periodic: bool = True) -> torch.Tensor:
    """``omega_z = dv/dx - du/dy`` on a uniform grid."""
    if periodic:
        dvdx = (torch.roll(v, -1, dims=0) - torch.roll(v, 1, dims=0)) / (2.0 * dx)
        dudy = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2.0 * dy)
    else:
        dvdx = torch.gradient(v, spacing=(dx, dy))[0]
        dudy = torch.gradient(u, spacing=(dx, dy))[1]
    return dvdx - dudy


def enstrophy(omega: torch.Tensor) -> float:
    return float((omega * omega).sum())


def total_circulation(omega: torch.Tensor, dx: float, dy: float) -> float:
    return float(omega.sum() * dx * dy)


def _smooth(field: torch.Tensor, width: int = 1) -> torch.Tensor:
    out = field
    for _ in range(width):
        acc = torch.zeros_like(out)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                acc = acc + torch.roll(torch.roll(out, sx, dims=0), sy, dims=1)
        out = acc / 9.0
    return out


def principal_axis_angle(omega: torch.Tensor, dx: float, dy: float) -> float:
    """Angle of the principal axis of the enstrophy distribution (radians).

    The two-blob field has a dominant second moment along the line joining the
    cores; that line rotates with the pair.  The angle is returned modulo
    ``pi`` (a line, not a vector), which the caller must unwrap.
    """
    w = torch.abs(omega)
    tot = float(w.sum())
    if tot <= 0.0:
        return float("nan")
    n = omega.shape[0]
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    X = x.reshape(-1, 1).expand_as(w)
    Y = x.reshape(1, -1).expand_as(w)
    xc = float((w * X).sum() / tot)
    yc = float((w * Y).sum() / tot)
    ixx = float((w * (X - xc) ** 2).sum() / tot)
    iyy = float((w * (Y - yc) ** 2).sum() / tot)
    ixy = float((w * (X - xc) * (Y - yc)).sum() / tot)
    return 0.5 * math.atan2(2.0 * ixy, ixx - iyy)


def core_positions(omega: torch.Tensor, dx: float, dy: float,
                   smooth: int = 2) -> list:
    """Positions ``(x, y)`` of the strongest enstrophy cores (up to two)."""
    kept = _peaks(omega, dx, dy, smooth)
    return [((i + 0.5) * dx, (j + 0.5) * dy) for (i, j) in kept]


def _peaks(omega: torch.Tensor, dx: float, dy: float, smooth: int = 2) -> list:
    w = _smooth(torch.abs(omega), smooth)
    padded = torch.nn.functional.pad(w.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1),
                                     mode="circular")
    local_max = torch.nn.functional.max_pool2d(padded, 3, stride=1).squeeze()
    peaks = (w >= local_max) & (w > 0.15 * float(w.max()))
    idx = peaks.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return []
    vals = w[idx[:, 0], idx[:, 1]]
    order = torch.argsort(vals, descending=True)
    n = omega.shape[0]
    kept: list = []
    for k in order.tolist():
        i, j = int(idx[k, 0]), int(idx[k, 1])
        ok = True
        for (i2, j2) in kept:
            di = min(abs(i - i2), n - abs(i - i2))
            dj = min(abs(j - j2), n - abs(j - j2))
            if math.hypot(di * dx, dj * dy) < 2.5 * dx:
                ok = False
                break
        if ok:
            kept.append((i, j))
        if len(kept) == 2:
            break
    return kept


def core_separation(omega: torch.Tensor, dx: float, dy: float,
                    smooth: int = 2) -> Tuple[float, int]:
    """Distance between the two strongest enstrophy cores, and their count.

    Peaks are found on the smoothed ``|omega|`` by a local-maximum test against
    a 3x3 neighbourhood, then suppressed within ``2.5 dx`` of a stronger peak.
    A return of ``(0.0, 1)`` means the pair has merged into a single core.
    """
    kept = _peaks(omega, dx, dy, smooth)
    if len(kept) < 2:
        return 0.0, len(kept)
    (i1, j1), (i2, j2) = kept
    return math.hypot((i1 - i2) * dx, (j1 - j2) * dy), len(kept)


def swirl_mach(u: torch.Tensor, v: torch.Tensor, rho: torch.Tensor, p: torch.Tensor,
               gamma: float = 1.4) -> float:
    """Peak ``|u|/c`` of the field: the compressibility knob of the vortex case."""
    c = torch.sqrt(gamma * p / rho)
    return float((torch.sqrt(u * u + v * v) / c).max())