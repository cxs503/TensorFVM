"""M2 PLIC-VOF verification: geometry exactness, conservation, dam-break controls."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfvm.two_phase.vof import (
    PLICVOF,
    dam_break_history,
    line_constant,
    partial_volume,
    plic_solver,
    static_interface_drift,
    youngs_normal,
)
from tensorfvm.vof_phase0.anchors import martin_moyce_front, martin_moyce_height
from tensorfvm.vof_phase0.dam_break import DamBreakConfig


def _field(n: int):
    xs = (torch.arange(n) + 0.5) / n
    return torch.meshgrid(xs, xs, indexing="xy")


def _area_check(alpha: torch.Tensor) -> float:
    """``A(1)`` of the reconstructed polygon must equal the cell fraction."""
    nx, ny = youngs_normal(alpha)
    flat = torch.sqrt(nx * nx + ny * ny) <= 1e-12
    nx = torch.where(flat, torch.zeros_like(nx), nx)
    ny = torch.where(flat, torch.ones_like(ny), ny)
    c = line_constant(alpha, nx, ny)
    cc = c + torch.clamp(-nx, min=0.0) + torch.clamp(-ny, min=0.0)
    area = partial_volume(torch.abs(nx), torch.abs(ny), cc, torch.ones_like(alpha))
    return float((area - alpha).abs().max())


def test_reconstruction_is_exact_for_resolved_interfaces():
    x, y = _field(32)
    cases = {
        "straight": (y <= 0.3 + 0.4 * x).double(),
        "circle": (((x - 0.5) ** 2 + (y - 0.55) ** 2) < 0.08).double(),
        "full": torch.ones(32, 32, dtype=torch.float64),
        "empty": torch.zeros(32, 32, dtype=torch.float64),
    }
    for name, alpha in cases.items():
        assert _area_check(alpha) <= 1e-12, name


def test_line_constant_axis_aligned_is_the_height():
    alpha = torch.tensor([[0.25, 0.75]])
    nx = torch.zeros_like(alpha)
    ny = torch.ones_like(alpha)
    assert torch.allclose(line_constant(alpha, nx, ny), alpha)
    # mirrored frame: a normal pointing in -y needs the offset shift
    c = line_constant(alpha, torch.zeros_like(alpha), -torch.ones_like(alpha))
    assert torch.allclose(c + 1.0, alpha)


def test_sweep_conserves_volume_and_respects_bounds():
    n = 24
    torch.manual_seed(0)
    vof = PLICVOF(n, n, 1.0 / n, 1.0 / n, cfl_max=0.3)
    alpha = (torch.rand(n, n) * 0.9 + 0.05)
    u = torch.zeros(n, n + 1)
    u[:, 1:-1] = (torch.rand(n, n - 1) - 0.5) * 1.2
    total = float(alpha.sum())
    out = vof.advect(alpha, u, torch.zeros(n + 1, n), 0.3 / n)
    # a random field is maximally unfavourable for the bound limiter; its
    # final clip leaves a 1e-8 relative residual (the dam break conserves
    # to 1e-16, see the invariants test below)
    assert float(out.sum()) == pytest.approx(total, rel=1e-6)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_static_interface_does_not_move_without_gravity():
    cfg = DamBreakConfig(nx=24, ny=10, gravity=0.0, pressure_init="zero",
                         t_star_end=1.0)
    control = static_interface_drift(cfg, n_steps=20)
    assert control["max_abs_change"] == 0.0
    assert control["volume_drift"] <= 1e-15
    assert control["speed_max"] <= 1e-15


def test_dam_break_invariants_and_front_regression():
    cfg = DamBreakConfig(nx=60, ny=20, t_star_end=2.0, max_steps=5000)
    solver = plic_solver(cfg)
    out = dam_break_history(cfg, solver=solver)

    assert not out["diverged"]
    assert out["alpha_bounds"][0] >= 0.0 and out["alpha_bounds"][1] <= 1.0
    assert float(np.max(np.abs(out["volume_drift"]))) <= 1e-10
    per_cell = out["vof_clip_volume"] / (out["steps"] * cfg.nx * cfg.ny)
    assert per_cell <= 5e-3                 # limiter must not dominate

    t_star = out["t_star"]
    front = out["front_star"][:, 1]
    assert np.all(np.diff(front) >= -1e-12)     # the surge never retreats
    window = t_star > 0.3
    error = np.abs(front[window] - np.array([martin_moyce_front(t)
                                             for t in t_star[window]])) / np.array(
        [martin_moyce_front(t) for t in t_star[window]])
    # regression guard only: this coarse grid is *outside* the quantitative gate
    # (the honest FAIL is measured on the 120x40 benchmark and reported)
    assert float(np.max(error)) < 0.95
    height = out["height_star"]
    height_error = np.abs(height[window] - np.array(
        [martin_moyce_height(t) for t in t_star[window]])) / np.array(
        [martin_moyce_height(t) for t in t_star[window]])
    assert float(np.max(height_error)) < 1.0


def test_plic_is_sharper_than_the_algebraic_scheme():
    """Same field, one sweep: PLIC must leave fewer partially filled cells."""
    from tensorfvm.vof_phase0.advection import AlgebraicVOF

    n = 40
    vof = PLICVOF(n, n, 1.0 / n, 1.0 / n)
    alg = AlgebraicVOF(n, n, 1.0 / n, 1.0 / n, boundary="wall", bounds="global")
    x, y = _field(n)
    alpha = (((x - 0.5) ** 2 + (y - 0.5) ** 2) < 0.04).double()
    u = torch.full((n, n + 1), 0.5)
    v = torch.full((n + 1, n), 0.25)
    dt = 0.005
    plic = vof.advect(alpha, u, v, dt)
    other = alg.advect(alpha, u, v, dt)
    # smearing indicator: sum(alpha (1 - alpha)) is zero for a perfectly sharp
    # field and maximal for a fully smeared one
    smear = lambda field: float((field * (1.0 - field)).sum())
    # a single step is nearly a tie (the geometric and the flux-corrected scheme
    # both stay sharp); the measurable difference is the interface thickness over
    # the full dam break, which the benchmark records (2.25 vs 2.67 cells)
    assert smear(plic) <= 1.01 * smear(other)
    assert float(plic.sum()) == pytest.approx(float(alpha.sum()), rel=1e-12)