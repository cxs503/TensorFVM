"""Hydrostatic ("lake at rest") equilibrium anchor for the two-phase VOF solver.

Physical setting
----------------
A closed tank of height ``H`` holds liquid of depth ``H_l`` (flat interface
aligned exactly with a grid face) under gravity ``g``; the gas above is treated
by the solver's ``void`` free-surface model.  The exact solution is the rest
state

.. math:: \\mathbf u = 0, \\qquad p(y) = \\rho_l g (H_l - y) \\;\\text{for}\\; y \\le H_l.

This is the standing analytic anchor against which a two-phase/free-surface
solver is judged for *parasitic currents*: any imbalance of the discrete
gravity and pressure-gradient terms at or near the interface accelerates the
fluid from rest, and the resulting spurious pressure oscillation is the
quantity the acceptance criterion C4 bounds (``< 0.5 %`` of ``rho_l g H_l``).

Metrics
-------
``pressure_oscillation``
    ``max_{n, cells} |p^n - p(y)| / (rho_l g H_l)`` -- how far the computed
    pressure ever moves from the analytic hydrostatics (dimensionless).
``spurious_velocity``
    ``max_n max|u| / sqrt(g H_l)`` -- the parasitic current, scaled by the
    gravity-wave speed.
``mass_drift_relative``
    ``|V(T) - V(0)| / V(0)`` for the liquid volume.

A control experiment is run with the fully coupled ``two_phase`` surface model
on the same grid: it exposes the classic failure of a single cell-centred
pressure field with a harmonic face density across the density jump, and gives
the before/after baseline the report needs.
"""

from __future__ import annotations

import math

import torch

from ..vof_phase0.solver import Phase0TwoPhaseSolver

__all__ = ["hydrostatic_state", "run_hydrostatic"]


def hydrostatic_state(nx: int, ny: int, *, length: float = 1.0,
                      height: float = 1.0, depth: float = 0.5,
                      rho_liquid: float = 1000.0, gravity: float = 9.81):
    """Exact rest state: flat interface, zero velocity, analytic pressure."""
    dx, dy = length / nx, height / ny
    # The anchor requires the interface to lie *on* a cell face: only then is the
    # discrete rest state the analytic profile (a staircase surface is a
    # different, geometry-limited problem -- see ``aligned`` in run_hydrostatic).
    rows = int(round(depth / dy))
    depth = rows * dy
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    alpha = (ys[:, None] < depth - 1e-12).to(torch.float64).expand(ny, nx).contiguous()
    p = (rho_liquid * gravity * torch.clamp(depth - ys, min=0.0))[:, None] \
        .expand(ny, nx).contiguous()
    return alpha, p, dx, dy


def run_hydrostatic(nx: int, ny: int, *, steps: int = 200,
                    surface_model: str = "void", depth: float = 0.5,
                    length: float = 1.0, height: float = 1.0,
                    rho_liquid: float = 1000.0, rho_gas: float = 1.0,
                    gravity: float = 9.81, alpha_cfl: float = 0.4,
                    well_balanced: bool = False,
                    dt: float | None = None) -> dict:
    """March the rest state ``steps`` times and report the equilibrium error."""
    alpha0, p0, dx, dy = hydrostatic_state(nx, ny, length=length, height=height,
                                          depth=depth, rho_liquid=rho_liquid,
                                          gravity=gravity)
    solver = Phase0TwoPhaseSolver(nx, ny, length, height, rho_liquid=rho_liquid,
                                  rho_gas=rho_gas, gravity=gravity,
                                  alpha_cfl=alpha_cfl, surface_model=surface_model,
                                  well_balanced_surface=well_balanced)
    solver.set_alpha(alpha0)
    solver.p = p0.clone()
    depth_effective = int(round(depth / dy)) * dy
    aligned = abs(depth - depth_effective) < 1e-9      # requested depth on a face?
    requested = depth
    depth = depth_effective
    scale_p = rho_liquid * gravity * depth
    scale_u = math.sqrt(gravity * depth)
    volume0 = solver.liquid_volume()
    if dt is None:
        dt = solver.stable_dt()
    worst_p = 0.0
    worst_u = 0.0
    worst_mass = 0.0
    history = []
    for k in range(steps):
        solver.step(dt)
        dev_p = float((solver.p - p0).abs().max().item()) / scale_p
        spd = max(float(solver.u.abs().max().item()),
                  float(solver.v.abs().max().item())) / scale_u
        drift = abs(solver.liquid_volume() - volume0) / volume0
        worst_p = max(worst_p, dev_p)
        worst_u = max(worst_u, spd)
        worst_mass = max(worst_mass, drift)
        if k in (0, steps // 2, steps - 1):
            history.append({"step": k + 1, "time": solver.time,
                            "pressure_oscillation": dev_p,
                            "spurious_velocity": spd,
                            "mass_drift_relative": drift})
    return {
        "cells": [nx, ny],
        "dx": dx,
        "dy": dy,
        "steps": steps,
        "dt": dt,
        "surface_model": surface_model,
        "well_balanced_surface": well_balanced,
        "density_ratio": rho_liquid / rho_gas,
        "depth": depth,
        "depth_requested": requested,
        "aligned": bool(aligned),
        "pressure_scale": scale_p,
        "velocity_scale": scale_u,
        "pressure_oscillation": worst_p,
        "spurious_velocity": worst_u,
        "mass_drift_relative": worst_mass,
        "cg": dict(solver.cg_last),
        "history": history,
    }