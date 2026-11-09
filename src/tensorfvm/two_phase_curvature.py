"""Height-function interface curvature for VOF fields (2-D).

Why this module exists
----------------------
``tensorfvm.two_phase_vof.interface_curvature`` differentiates the *cell
centred* normals with central differences: ``kappa = div(n)``, ``n = -grad(alpha)/|grad(alpha)|``.
That estimator is the historical default and is exact for a linear profile, but
it is not a convergent curvature estimator for a VOF colour function: the
magnitude is a grid-noise amplifier, and the band-averaged error of a static
circle does not fall with refinement.  Measured on the platform's own
verification case (R = 0.2 m, sigma/R = 0.35 Pa, ``docs/two-phase-surface-tension-audit``)
the band-mean error is

    n =  24   32   48   64
        +0.3% +10.0% +30.6% +2.4%

i.e. non-monotone and > 30% at n = 48.  That is a *verification* gap, not a
solver instability (the Young-Laplace pressure jump is inside 3% on all four
grids), but it blocks the claim "discrete curvature converges".

The fix implemented here is the height-function (HF) method, the standard VOF
curvature technique of Cummins, Francois & Kothe (2005), Francois et al. (2006)
and Popinet (2009):

* for each interface cell pick the scan direction in which the interface is
  most nearly perpendicular to the scan lines (``|n_y| >= |n_x|`` -> vertical
  scan); the height function is then single-valued,
* sample the interface height on three adjacent scan lines as the column sum
  of the volume fraction over a ``2m+1`` window,
* differentiate the height function twice with the central stencil and use the
  graph formula ``kappa = ± h'' / (1 + h'^2)^{3/2}``.

The sign of the normal is taken from a *smoothed* colour field: for pure-phase
cells (``|grad(alpha)| = 0``) the raw normal is undefined and a naive
per-cell sign test flips the branch, which destroys the second difference of
the height function (this was the first bug found while porting the method --
see ``docs/height-function-curvature/report.md`` §5).  Smoothing only selects
the *branch*; the heights themselves are summed from the raw volume fraction,
so the reconstruction order of the interface position is preserved.

Conventions match :func:`tensorfvm.two_phase_vof.interface_curvature`: the
normal points from liquid (``alpha ~ 1``) to gas (``alpha ~ 0``) and
``kappa = div(n)``, so for a circular drop of radius ``R`` the sign is
``kappa = +1/R`` everywhere on the interface.

References
----------
* Cummins, Francois & Kothe, Comput. Struct. 83 (2005) 425-434.
* Francois et al., J. Comput. Phys. 213 (2006) 141-173 (balanced CSF forces).
* Popinet, J. Comput. Phys. 228 (2009) 5838-5867 (height-function curvature).

All tensors are ``float64``, consistent with :mod:`tensorfvm.two_phase_vof`.
"""

from __future__ import annotations

import torch

from .two_phase_vof import smooth_alpha

__all__ = [
    "HEIGHT_FUNCTION_STENCIL",
    "height_function",
    "height_function_curvature",
]

#: half width (in cells) of the column-sum window used to sample the interface
#: height.  ``2*3+1 = 7`` cells is the value recommended by Popinet (2009) for
#: a second-order height function.
HEIGHT_FUNCTION_STENCIL = 3


def _central(a: torch.Tensor, axis: int, spacing: float) -> torch.Tensor:
    """First derivative along *axis* (1 = x, 0 = y), central interior."""
    d = torch.zeros_like(a)
    if axis == 1:
        d[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * spacing)
        d[:, 0] = (a[:, 1] - a[:, 0]) / spacing
        d[:, -1] = (a[:, -1] - a[:, -2]) / spacing
    else:
        d[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * spacing)
        d[0, :] = (a[1, :] - a[0, :]) / spacing
        d[-1, :] = (a[-1, :] - a[-2, :]) / spacing
    return d


def _window_sum(alpha: torch.Tensor, half: int, axis: int) -> torch.Tensor:
    """Sum of *alpha* over a ``2*half+1`` window along *axis* (0 = y, 1 = x).

    The domain edges are padded by replication, so the sum saturates (instead
    of collapsing) where the window leaves the box; the corresponding height is
    then a clipped value, which is the standard height-function treatment of
    scan lines that miss the interface.
    """
    a = alpha if axis == 0 else alpha.T
    n = a.shape[0]
    padded = torch.cat([a[:1].repeat(half, 1), a, a[-1:].repeat(half, 1)], dim=0)
    total = torch.zeros_like(a)
    for k in range(2 * half + 1):
        total += padded[k : k + n]
    return total if axis == 0 else total.T


def height_function(
    alpha: torch.Tensor,
    spacing: float,
    axis: int,
    *,
    half: int = HEIGHT_FUNCTION_STENCIL,
    normal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interface height sampled on every scan line of one direction.

    Parameters
    ----------
    alpha : (ny, nx) volume fraction (``1`` liquid, ``0`` gas).
    spacing : cell size along the scan direction (``dy`` for ``axis=0``).
    axis : ``0`` -> height measured in ``y`` on column scans (interface mostly
        horizontal); ``1`` -> height measured in ``x`` on row scans.
    half : half width of the summation window.
    normal : optional ``(ny, nx, 2)`` from a smoothed field, used to choose
        between the "liquid below/left" and "liquid above/right" branch.  When
        omitted it is built from *alpha* itself (fine for well resolved
        interfaces, ambiguous in pure-phase cells).

    Returns
    -------
    (ny, nx) tensor with the interface coordinate (in metres) for each scan
    line, clipped where the interface is outside the window.  The returned
    field is constant along the scan direction, so its derivatives along the
    scan lines are the derivatives of the interface shape.
    """
    ny, nx = alpha.shape
    if normal is None:
        n = _normal(alpha, 1.0, 1.0)
    else:
        n = normal
    comp = n[..., 1] if axis == 0 else n[..., 0]

    sums = _window_sum(alpha, half, axis)
    width = float(2 * half + 1)
    idx = (torch.arange(ny).view(-1, 1) if axis == 0
           else torch.arange(nx).view(1, -1))
    base = (idx - half) * spacing
    # ``liquid on the low side of the interface`` (n_component > 0) -> the
    # interface sits at the top of the liquid column; otherwise the gas column
    # is below it and the interface sits at the top of the gas.
    lower = base + sums * spacing
    upper = base + (width - sums) * spacing
    return torch.where(comp > 0, lower, upper)


def _normal(alpha: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    gx = _central(alpha, 1, dx)
    gy = _central(alpha, 0, dy)
    mag = torch.sqrt(gx * gx + gy * gy).clamp_min(1e-300)
    return torch.stack([-gx / mag, -gy / mag], dim=-1)


def height_function_curvature(
    alpha: torch.Tensor,
    dx: float,
    dy: float,
    *,
    half: int = HEIGHT_FUNCTION_STENCIL,
    smoothing: int = 2,
) -> torch.Tensor:
    """Mean curvature ``kappa = div(n)`` of the interface in a VOF field.

    Second-order height-function estimator (Cummins et al. 2005; Popinet 2009).
    Returns a ``(ny, nx)`` tensor; pure-phase cells (``|grad(alpha)| ~ 0``) are
    set to zero, matching :func:`tensorfvm.two_phase_vof.interface_curvature`.

    ``smoothing`` is the number of ``[1/4, 1/2, 1/4]`` passes applied to a copy
    of the colour field *only* to obtain reliable normal signs in pure-phase
    cells (see module docstring).  ``smoothing=0`` reproduces the naive
    per-cell sign test and is kept only for the falsification test.
    """
    ny, nx = alpha.shape
    half = int(max(1, min(half, (min(ny, nx) - 1) // 2)))

    guide = alpha if smoothing <= 0 else smooth_alpha(alpha, int(smoothing))
    n = _normal(guide, dx, dy)
    nx_c = n[..., 0]
    ny_c = n[..., 1]

    hv = height_function(alpha, dy, 0, half=half, normal=n)
    hv1 = _central(hv, 1, dx)
    hv2 = _central(hv1, 1, dx)
    k_v = -torch.sign(ny_c) * hv2 / (1.0 + hv1 * hv1) ** 1.5

    hh = height_function(alpha, dx, 1, half=half, normal=n)
    hh1 = _central(hh, 0, dy)
    hh2 = _central(hh1, 0, dy)
    k_h = -torch.sign(nx_c) * hh2 / (1.0 + hh1 * hh1) ** 1.5

    kappa = torch.where(ny_c.abs() >= nx_c.abs(), k_v, k_h)

    raw = _normal(alpha, dx, dy)
    mag = torch.sqrt(raw[..., 0] ** 2 + raw[..., 1] ** 2)
    scale = float(mag.max())
    if scale > 0.0:
        kappa = torch.where(mag > 1e-8 * scale, kappa, torch.zeros_like(kappa))
    return kappa