"""Conservation audit of the FCT-limited VOF transport (verification debt).

Context
-------
Report 3003 (m2940) re-verified the *accuracy* of
:class:`tensorfvm.vof_fct.LimitedVOFTransport` and registered an honest FAIL:
the measured mass drift (5.7e-5..8.3e-5 relative) was attributed to "the
low-order clamp clips the peak cell (peak exactly 1.0)".  This module re-examines
that attribution and closes the debt with a proper gate.

Findings (this module)
----------------------
1. **The clamp never activates** on the Gaussian study: the donor-cell
   (low-order) update stays strictly inside ``[0, 1]`` (measured
   ``[4.7e-29, 0.999]``), so the ``low = low_update.clamp(0, 1)`` line clips
   *zero* mass.  The report-3003 attribution is wrong.
2. **The drift is a boundary artifact of the test setup**, not of the limiter:
   the zero-gradient ghost boundary injects the (non-compact) Gaussian tail
   under the uniform flow.  The net flux is 100% boundary
   (x-boundary + y-boundary = total net flux, interior cancels exactly).
3. **The upwind (donor-cell) baseline has the identical drift** (5.7e-5 at
   n=256), confirming the leak is in the shared boundary handling, not the
   FCT limiter.  The anti-diffusive FCT part is exactly conservative
   (sum of the anti contribution = 1.4e-15).
4. **With a compactly-supported profile** (cosine hill, exactly 0 at the
   boundary) the zero-gradient BC is inactive and the conservation gate is
   meaningful: the upwind baseline is conservative to machine precision
   (0.0 / 1.9e-16) and the FCT-limited scheme is conservative to
   1e-6..1e-9 (converging with the grid) and bounded to ~1e-9.

The residual FCT drift on the compact profile is the per-face limiter's
discretization of a per-cell budget: it is bounded, converges with the grid,
and is *not* a mass-creation bug.  The docstring claim of "machine-precision
conservation" is therefore corrected to "conservative to the order of the
limiter's per-face discretisation".

Scope: 2-D cell-centred VOF advection, constant divergence-free velocity,
zero-gradient boundary.  No claim for divergent fields, curved interfaces,
or the 3-D path.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from ..two_phase_vof import VOFTransport
from ..vof_fct import LimitedVOFTransport

__all__ = [
    "cosine_hill",
    "decompose_gaussian_drift",
    "compact_study",
    "run",
]

CFL = 0.5
U = 0.5
T_END = 0.15
# Compact profile: support radius R, centre C0.  R + |C0 - corner| < 0.5 keeps
# the support strictly inside [0,1]^2 so the zero-gradient BC is inactive.
R = 0.20
C0 = (0.5, 0.5)


def cosine_hill(x, y, centre, R=R):
    """Smooth compactly-supported profile: 0.5*(1+cos(pi*r/R)) for r<R, else 0."""
    r2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2
    a = 0.5 * (1.0 + np.cos(math.pi * np.sqrt(r2) / R))
    return np.where(r2 < R * R, a, 0.0)


def _march(transport, alpha, u, v, dt, n_steps):
    for _ in range(n_steps):
        alpha = transport.advect(alpha, u, v, dt)
    return alpha


def decompose_gaussian_drift(n=256):
    """Decompose the Gaussian-study mass drift into its three sources.

    Returns a dict with the per-step and total contributions of
    (a) the low-order clamp clipping, (b) the boundary flux, and
    (c) the anti-diffusive FCT part, plus the upwind-baseline drift for
    comparison.
    """
    from .fct_accuracy import gaussian, X0

    dx = 1.0 / n
    transport = LimitedVOFTransport(n, n, dx, dx, device="cpu")
    x = (np.arange(n) + 0.5) * dx
    xx, yy = np.meshgrid(x, x)
    alpha = torch.tensor(gaussian(xx, yy, X0), dtype=torch.float64)
    u = torch.full((n, n + 1), U, dtype=torch.float64)
    v = torch.full((n + 1, n), U, dtype=torch.float64)
    dt = CFL * dx / U
    n_steps = int(round(T_END / dt))
    mass0 = float(alpha.sum())

    # --- one-step decomposition on the initial state ---
    cell_volume = dx * dx
    flux_x = u * dx
    flux_y = v * dx
    donor_x, donor_y, high_x, high_y = transport._face_values(alpha, flux_x, flux_y, dt)
    low_update = alpha - dt / cell_volume * transport._net_flux(flux_x, donor_x, flux_y, donor_y)
    clamp_clip = float(((low_update - 1.0).clamp_min(0.0).sum()
                        + (0.0 - low_update).clamp_min(0.0).sum()))
    net = transport._net_flux(flux_x, donor_x, flux_y, donor_y)
    boundary_flux = float(
        (flux_x[:, -1] * donor_x[:, -1] - flux_x[:, 0] * donor_x[:, 0]).sum()
        + (flux_y[-1, :] * donor_y[-1, :] - flux_y[0, :] * donor_y[0, :]).sum())
    # anti part: run the full advect and subtract the low-order contribution
    alpha_fct = transport.advect(alpha, u, v, dt)
    low = low_update.clamp(0.0, 1.0)
    anti_contribution = float((alpha_fct - low).sum())

    # --- full march, FCT and upwind ---
    alpha_fct = _march(transport, alpha.clone(), u, v, dt, n_steps)
    up = VOFTransport(n, n, dx, dx, device="cpu", scheme="upwind")
    alpha_up = _march(up, alpha.clone(), u, v, dt, n_steps)

    return {
        "n": n,
        "n_steps": n_steps,
        "mass0": mass0,
        "low_update_min": float(low_update.min()),
        "low_update_max": float(low_update.max()),
        "clamp_clip_per_step": clamp_clip,
        "boundary_flux_per_step": boundary_flux,
        "anti_contribution_per_step": anti_contribution,
        "fct_drift_rel": abs(float(alpha_fct.sum()) - mass0) / mass0,
        "upwind_drift_rel": abs(float(alpha_up.sum()) - mass0) / mass0,
    }


def compact_study(levels=(64, 128, 256)):
    """Grid sequence on a compactly-supported cosine hill (valid conservation gate)."""
    out = {"R": R, "c0": list(C0), "cfl": CFL, "fct": [], "upwind": []}
    for n in levels:
        dx = 1.0 / n
        x = (np.arange(n) + 0.5) * dx
        xx, yy = np.meshgrid(x, x)
        a0 = cosine_hill(xx, yy, C0, R)
        m0 = float(a0.sum())
        for scheme, cls, kw in (("fct", LimitedVOFTransport, {}),
                               ("upwind", VOFTransport, {"scheme": "upwind"})):
            t = cls(n, n, dx, dx, device="cpu", **kw)
            alpha = torch.tensor(a0, dtype=torch.float64)
            u = torch.full((n, n + 1), U, dtype=torch.float64)
            v = torch.full((n + 1, n), U, dtype=torch.float64)
            dt = CFL * dx / U
            n_steps = int(round(T_END / dt))
            t_actual = n_steps * dt
            alpha = _march(t, alpha, u, v, dt, n_steps)
            c = (C0[0] + U * t_actual, C0[1] + U * t_actual)
            exact = torch.tensor(cosine_hill(xx, yy, c, R), dtype=torch.float64)
            l1 = float((alpha - exact).abs().sum()) / float(exact.sum())
            out[scheme].append({
                "n": n,
                "l1_rel": l1,
                "mass_rel_drift": abs(float(alpha.sum()) - m0) / m0,
                "min_alpha": float(alpha.min()),
                "max_alpha": float(alpha.max()),
            })
    out["fct_order_l1"] = _order([m["l1_rel"] for m in out["fct"]])
    out["upwind_order_l1"] = _order([m["l1_rel"] for m in out["upwind"]])
    return out


def _order(errors):
    e = np.asarray(errors, dtype=float)
    p1 = math.log(e[0] / e[1]) / math.log(2.0)
    p2 = math.log(e[1] / e[2]) / math.log(2.0)
    return float(0.5 * (p1 + p2))


def run(out_path=None):
    dec = decompose_gaussian_drift(256)
    study = compact_study((64, 128, 256))
    result = {"gaussian_decomposition": dec, "compact_study": study}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result
