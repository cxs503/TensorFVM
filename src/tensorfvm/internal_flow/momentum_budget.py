"""Control-volume momentum budget closure for the staggered MAC solver.

At steady state the discrete momentum equation for each interior face reads:

    tendency(u) - dp/dx = 0   (x-momentum)
    tendency(v) - dp/dy + buoyancy = 0   (y-momentum)

where tendency = conv + nu*lap (from the solver's tendencies() method).

This module evaluates each term independently and reports the relative
imbalance as a domain-integrated closure metric.
"""

from __future__ import annotations

import torch

from .mac import StaggeredMAC


def momentum_budget_x(solver: StaggeredMAC) -> dict[str, torch.Tensor]:
    """Evaluate x-momentum balance at interior x-faces.

    Returns dict with tensors of shape (nx+1, ny):
      - tendency: conv + diff (from solver)
      - pres: dp/dx
      - resid: tendency - pres (should be ~0 at steady state)
      - imbalance: |resid| / max(|tendency|, |pres|)
    """
    u = solver.u
    p = solver.p
    hx = solver.hx
    nx, ny = solver.nx, solver.ny
    dtype = solver.dtype
    device = solver.device

    # Pressure gradient at interior x-faces
    pres = torch.zeros(nx + 1, ny, dtype=dtype, device=device)
    pres[1:nx, :] = (p[1:, :] - p[:-1, :]) / hx

    # Tendency from solver (conv + diff)
    tu, _, _ = solver.tendencies()

    # Residual: at steady state, tendency - dp/dx = 0
    resid = torch.zeros(nx + 1, ny, dtype=dtype, device=device)
    resid[1:nx, :] = tu[1:nx, :] - pres[1:nx, :]

    # Imbalance metric
    scale = torch.maximum(tu.abs(), pres.abs())
    scale = torch.clamp(scale, min=1e-30)
    imbalance = resid.abs() / scale

    return {
        "tendency": tu, "pres": pres, "resid": resid, "imbalance": imbalance,
    }


def momentum_budget_y(solver: StaggeredMAC) -> dict[str, torch.Tensor]:
    """Evaluate y-momentum balance at interior y-faces."""
    v = solver.v
    p = solver.p
    hy = solver.hy
    nx, ny = solver.nx, solver.ny
    dtype = solver.dtype
    device = solver.device

    # Pressure gradient at interior y-faces
    pres = torch.zeros(nx, ny + 1, dtype=dtype, device=device)
    pres[:, 1:ny] = (p[:, 1:] - p[:, :-1]) / hy

    # Buoyancy source
    buoy = torch.zeros(nx, ny + 1, dtype=dtype, device=device)
    if solver.config.energy and solver.theta is not None:
        g_beta = solver.config.buoyancy
        buoy[:, 1:ny] = g_beta * 0.5 * (solver.theta[:, :-1] + solver.theta[:, 1:])

    # Tendency from solver
    _, tv, _ = solver.tendencies()

    # Residual: tendency - dp/dy + buoyancy = 0
    resid = torch.zeros(nx, ny + 1, dtype=dtype, device=device)
    resid[:, 1:ny] = tv[:, 1:ny] - pres[:, 1:ny] + buoy[:, 1:ny]

    scale = torch.maximum(tv.abs(), pres.abs())
    scale = torch.clamp(scale, min=1e-30)
    imbalance = resid.abs() / scale

    return {
        "tendency": tv, "pres": pres, "buoy": buoy,
        "resid": resid, "imbalance": imbalance,
    }


def global_momentum_closure(solver: StaggeredMAC) -> dict[str, float]:
    """Domain-integrated momentum budget closure metrics.

    Returns:
      - x_closure_pct: percentage x-momentum imbalance (lower is better)
      - y_closure_pct: percentage y-momentum imbalance
      - mass_divergence_max: max |div(u)| (continuity residual)
    """
    bx = momentum_budget_x(solver)
    by = momentum_budget_y(solver)

    # Interior faces only
    x_imb = bx["imbalance"][1:solver.nx, :]
    y_imb = by["imbalance"][:, 1:solver.ny]

    # L1 mean imbalance as percentage
    x_pct = float(x_imb.mean()) * 100.0
    y_pct = float(y_imb.mean()) * 100.0

    div = solver.divergence()
    mass_imb = float(div.abs().max())

    # Also compute term magnitudes for the report
    x_tend_l1 = float(bx["tendency"][1:solver.nx, :].abs().mean())
    x_pres_l1 = float(bx["pres"][1:solver.nx, :].abs().mean())
    x_resid_l1 = float(bx["resid"][1:solver.nx, :].abs().mean())

    y_tend_l1 = float(by["tendency"][:, 1:solver.ny].abs().mean())
    y_pres_l1 = float(by["pres"][:, 1:solver.ny].abs().mean())
    y_buoy_l1 = float(by["buoy"][:, 1:solver.ny].abs().mean())
    y_resid_l1 = float(by["resid"][:, 1:solver.ny].abs().mean())

    return {
        "x_closure_pct": x_pct,
        "y_closure_pct": y_pct,
        "mass_divergence_max": mass_imb,
        "x_tendency_l1": x_tend_l1,
        "x_pressure_l1": x_pres_l1,
        "x_residual_l1": x_resid_l1,
        "y_tendency_l1": y_tend_l1,
        "y_pressure_l1": y_pres_l1,
        "y_buoyancy_l1": y_buoy_l1,
        "y_residual_l1": y_resid_l1,
    }
