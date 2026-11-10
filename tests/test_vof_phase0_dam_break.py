"""Dam-break anchor tests for the Phase-0 two-phase gate (node p233w1-N03).

These tests pin the *anchors* and the *invariants* (AC3/AC4): conservation,
boundedness, the 1000:1 density-ratio stability envelope and the exact rest
state of a stratified pool.  The quantitative front comparison (AC2) and the
grid-convergence study (AC5) are run by ``scripts/run_vof_phase0.py`` because
they need the finest grids (see ``docs/vof-phase0/artifacts/phase0_report.json``).
"""

import math

import pytest
import torch

from tensorfvm.vof_phase0.anchors import (
    MARTIN_MOYCE_FRONT_T,
    MARTIN_MOYCE_FRONT_X,
    martin_moyce_height,
    ritter_depth,
    ritter_front,
    to_dimensionless_front,
    to_dimensionless_time,
)
from tensorfvm.vof_phase0.dam_break import (
    DamBreakConfig,
    column_height,
    initial_alpha,
    run_dam_break,
    surge_front,
)
from tensorfvm.vof_phase0.solver import Phase0TwoPhaseSolver


def test_martin_moyce_curves_are_monotone():
    assert all(MARTIN_MOYCE_FRONT_X[i] <= MARTIN_MOYCE_FRONT_X[i + 1]
               for i in range(len(MARTIN_MOYCE_FRONT_X) - 1))
    assert all(MARTIN_MOYCE_FRONT_T[i] < MARTIN_MOYCE_FRONT_T[i + 1]
               for i in range(len(MARTIN_MOYCE_FRONT_T) - 1))
    t = [0.2 * i for i in range(1, 20)]
    heights = [martin_moyce_height(x) for x in t]
    assert all(heights[i] >= heights[i + 1] for i in range(len(heights) - 1))


def test_dimensionless_scaling_roundtrip():
    a, g = 0.05715, 9.81
    t = 0.1
    t_star = to_dimensionless_time(t, a, g)
    assert t_star == pytest.approx(t * math.sqrt(g / a), rel=1e-12)
    assert to_dimensionless_front(a * (1.0 + 2.0), a) == pytest.approx(2.0)


def test_ritter_solution_limits():
    a, h0, g = 0.05715, 0.1143, 9.81
    assert ritter_front(0.0, h0, g, a) == pytest.approx(a)
    # front speed is 2 sqrt(g h0)
    t = 0.05
    speed = (ritter_front(t, h0, g, a) - ritter_front(0.0, h0, g, a)) / t
    assert speed == pytest.approx(2.0 * math.sqrt(g * h0), rel=1e-12)
    # depth is h0 far upstream, zero beyond the front
    assert ritter_depth(0.0, t, h0, g, a) == pytest.approx(h0, rel=0.05)
    assert ritter_depth(ritter_front(t, h0, g, a) + 1e-3, t, h0, g, a) == 0.0


def test_config_rejects_column_taller_than_domain():
    with pytest.raises(ValueError):
        DamBreakConfig(a=0.1, aspect=4.0, height=0.2)


@pytest.fixture(scope="module")
def short_run():
    cfg = DamBreakConfig(nx=60, ny=20, t_star_end=1.0)
    solver = Phase0TwoPhaseSolver(
        nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height,
        surface_model="void")
    return cfg, run_dam_break(cfg, solver=solver)


def test_dam_break_conserves_liquid_volume(short_run):
    _, result = short_run
    assert result["max_volume_residual"] <= 1.0e-6
    assert len(result["t"]) > 5


def test_dam_break_alpha_stays_bounded(short_run):
    _, result = short_run
    lo, hi = result["alpha_bounds"]
    assert lo >= -1e-6
    assert hi <= 1.0 + 1e-6
    assert not result["diverged"]


def test_surge_front_advances_monotonically(short_run):
    _, result = short_run
    fronts = result["front_star"]
    assert fronts[-1] > fronts[0]
    # the surge never runs backwards by more than a cell
    dx_star = result["config"]["length"] / result["config"]["nx"] / \
        result["config"]["a"]
    for i in range(len(fronts) - 1):
        assert fronts[i + 1] >= fronts[i] - 2.0 * dx_star


def test_column_height_is_below_initial_value(short_run):
    cfg, result = short_run
    assert result["height_star"][-1] < 1.0
    assert column_height(torch.from_numpy(result["alpha_final"]),
                         cfg.dy) == pytest.approx(result["height"][-1])


def test_surge_front_zero_for_empty_field():
    alpha = torch.zeros((10, 10), dtype=torch.float64)
    assert surge_front(alpha, 0.1, 0.1) == 0.0


def test_hydrostatic_pool_is_an_exact_rest_state():
    """The void surface model must not manufacture spurious currents (AC3)."""
    nx, ny, length, height, depth = 40, 40, 0.4, 0.4, 0.2
    solver = Phase0TwoPhaseSolver(nx=nx, ny=ny, length=length, height=height,
                                 surface_model="void")
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * solver.dy
    alpha = (ys[:, None] < depth).to(torch.float64).expand(ny, nx).clone()
    solver.set_alpha(alpha)
    head = torch.clamp(depth - ys, min=0.0)[:, None].expand(ny, nx)
    solver.p = torch.where(alpha > 0.5, 1000.0 * 9.81 * head,
                           torch.zeros_like(head)).clone()
    volume = solver.liquid_volume()
    for _ in range(120):
        solver.step(solver.stable_dt())
    assert float(solver.u.abs().max()) < 1e-10
    assert float(solver.v.abs().max()) < 1e-10
    assert abs(solver.liquid_volume() - volume) <= 1e-12 * volume + 1e-18
    assert float(solver.alpha.max()) <= 1.0 + 1e-9


def test_density_ratio_1000_is_stable_for_100_steps():
    """AC3 smoke: the fully coupled 1000:1 projection must stay finite."""
    cfg = DamBreakConfig(nx=40, ny=16, t_star_end=1e9, max_steps=100)
    solver = Phase0TwoPhaseSolver(
        nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height,
        rho_liquid=1000.0, rho_gas=1.0, surface_model="two_phase")
    solver.set_alpha(initial_alpha(cfg))
    solver.set_hydrostatic_pressure(cfg.h0, cfg.a)
    v0 = solver.liquid_volume()
    for _ in range(100):
        info = solver.step(solver.stable_dt())
        assert math.isfinite(info["speed_max"])
    assert solver.steps == 100
    assert torch.isfinite(solver.alpha).all()
    assert float(solver.alpha.min()) >= -1e-3
    assert float(solver.alpha.max()) <= 1.0 + 1e-3
    assert abs(solver.liquid_volume() - v0) / v0 <= 1.0e-6


def test_solver_rejects_unknown_surface_model():
    with pytest.raises(ValueError):
        Phase0TwoPhaseSolver(nx=8, ny=8, length=1.0, height=1.0,
                             surface_model="nonsense")