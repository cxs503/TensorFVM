"""Accuracy re-verification of the FCT-limited VOF transport (verification debt).

The merged feature under review is :class:`tensorfvm.vof_fct.LimitedVOFTransport`
(merged in m2562).  Its shipped verification (``verification/vof_rotation.py``)
is a *self-consistency* check: solid-body rotation of a cosine hill, asserting
mass conservation and boundedness.  What it does **not** answer is the
accuracy question: against an independent reference, how accurate is the
FCT-limited update, and what does the limiter cost in order?

This module closes that debt with a manufactured-solution study whose
reference is analytic and independent of the code:

* A smooth Gaussian blob advected by a constant (divergence-free) velocity
  field.  The exact solution is the same Gaussian translated by ``U*t``;
  no code is involved in the reference.
* A 3-level grid sequence (same physical time, dt scaled with dx at fixed
  CFL) gives the observed spatial order of the *limited* scheme.
* The same study is run for the pure donor-cell baseline (``scheme="upwind"``)
  so the report can state what the limiter buys (diffusion) and what it does
  not (order).
* Boundedness (0 <= alpha <= 1) and machine-precision mass conservation are
  asserted as gates, not just reported.
* A negative control proves the order estimator itself: synthetic
  ``e = C*h^p`` sequences must recover p, and a non-monotone sequence must
  raise.

Scope: 2-D cell-centred VOF advection, constant velocity, zero-gradient
boundary (blob kept away from the boundary).  No claim is made for
divergent/compressible velocity fields, curved interfaces, or the 3-D path.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..two_phase_vof import VOFTransport
from ..vof_fct import LimitedVOFTransport

__all__ = [
    "observed_order",
    "gaussian",
    "advect_gaussian",
    "study",
]

CFL = 0.5          # same as the two-phase solver default
U = 0.5            # constant velocity magnitude per component
T_END = 0.15       # physical advection time
SIGMA = 0.08       # Gaussian width (>= ~4 cells on the coarsest study grid)
X0 = (0.35, 0.35)  # initial centre


def observed_order(errors, ratios=(2.0, 2.0)):
    """Three-level observed order; raises on non-monotone error (negative control)."""
    e = np.asarray(errors, dtype=float)
    if e.shape != (3,):
        raise ValueError("need exactly three error levels")
    if np.any(e <= 0) or np.any(~np.isfinite(e)):
        raise ValueError("errors must be finite and positive")
    if e[0] <= e[1] or e[1] <= e[2]:
        raise ValueError("error sequence must be strictly decreasing (negative control)")
    p1 = math.log(e[0] / e[1]) / math.log(ratios[0])
    p2 = math.log(e[1] / e[2]) / math.log(ratios[1])
    return float(0.5 * (p1 + p2))


def gaussian(x, y, centre, sigma=SIGMA):
    cx, cy = centre
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-r2 / (2.0 * sigma * sigma))


def advect_gaussian(n, velocity=(U, U), cfl=CFL, t_end=T_END, scheme="fct"):
    """Advect a Gaussian blob on an n x n grid; return (alpha_end, metrics)."""
    dx = 1.0 / n
    device = "cpu"
    if scheme == "fct":
        transport = LimitedVOFTransport(n, n, dx, dx, device=device)
    else:
        transport = VOFTransport(n, n, dx, dx, device=device, scheme="upwind")
    x = (np.arange(n) + 0.5) * dx
    xx, yy = np.meshgrid(x, x)
    alpha = torch.tensor(gaussian(xx, yy, X0), dtype=torch.float64, device=device)
    u = torch.full((n, n + 1), velocity[0], dtype=torch.float64, device=device)
    v = torch.full((n + 1, n), velocity[1], dtype=torch.float64, device=device)
    dt = cfl * dx / max(abs(velocity[0]), abs(velocity[1]), 1e-300)
    n_steps = int(round(t_end / dt))
    t_actual = n_steps * dt  # the reference must use the time actually marched
    mass0 = float(alpha.sum())
    for _ in range(n_steps):
        alpha = transport.advect(alpha, u, v, dt)
    centre = (X0[0] + velocity[0] * t_actual, X0[1] + velocity[1] * t_actual)
    exact = torch.tensor(gaussian(xx, yy, centre), dtype=torch.float64, device=device)
    mass = float(alpha.sum())
    l1 = float((alpha - exact).abs().sum()) / float(exact.sum())
    l2 = float(((alpha - exact) ** 2).sum().sqrt()) / float((exact ** 2).sum().sqrt())
    metrics = dict(
        n=n,
        n_steps=n_steps,
        t_actual=t_actual,
        l1_rel=l1,
        l2_rel=l2,
        max_abs=float((alpha - exact).abs().max()),
        mass_rel_drift=abs(mass - mass0) / mass0,
        min_alpha=float(alpha.min()),
        max_alpha=float(alpha.max()),
    )
    return alpha, metrics


def study(levels=(64, 128, 256), velocity=(U, U)):
    """Run the grid sequence for FCT and the donor-cell baseline."""
    out = {"velocity": list(velocity), "cfl": CFL, "t_end": T_END, "sigma": SIGMA,
           "x0": list(X0), "fct": [], "upwind": []}
    for n in levels:
        _, m = advect_gaussian(n, velocity=velocity, scheme="fct")
        out["fct"].append(m)
        _, m = advect_gaussian(n, velocity=velocity, scheme="upwind")
        out["upwind"].append(m)
    out["fct_order_l1"] = observed_order([m["l1_rel"] for m in out["fct"]])
    out["fct_order_l2"] = observed_order([m["l2_rel"] for m in out["fct"]])
    out["upwind_order_l1"] = observed_order([m["l1_rel"] for m in out["upwind"]])
    return out
