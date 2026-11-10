"""Acceptance tests for the immersed-solid masked projection.

The tests are the machine-checkable part of the node criteria: mass conservation
of the masked arm (C1), rigidity/no-slip of the solid (C2), the *negative*
controls (``kappa -> inf`` degenerates to a rigid mask, ``kappa -> 0`` to the
obstacle-free baseline, C5) and -- most importantly -- that the legacy arm really
does reproduce the mass-source defect, so the fix is aimed at the right cause.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.immersed.geometry import (CartesianGrid, backward_facing_step,
                                         expansion_ratio_realised, parabolic_inlet,
                                         symmetric_expansion)
from tensorfvm.immersed.projection import MaskedPoisson, masked_pcg
from tensorfvm.immersed.solver import (ImmersedConfig, ImmersedFlow, field_l2,
                                       solid_boundary_flux)
from tensorfvm.immersed.study import reattachment, richardson

torch.set_num_threads(1)


def _flow(cells_per_h: int = 8, projection: str = "masked", **kwargs) -> ImmersedFlow:
    grid = backward_facing_step(cells_per_h)
    cfg = ImmersedConfig(grid=grid, viscosity=0.01, projection=projection,
                         inlet_bounds=(1.0, 2.0), reference_length=1.0,
                         t_end=0.0, **kwargs)
    return ImmersedFlow(cfg)


def _run(flow: ImmersedFlow, steps: int) -> ImmersedFlow:
    for _ in range(steps):
        flow.step()
    return flow


# --------------------------------------------------------------------- geometry
def test_expansion_ratio_is_exactly_two():
    for grid in (backward_facing_step(16), symmetric_expansion(16),
                 backward_facing_step(40), symmetric_expansion(40)):
        assert expansion_ratio_realised(grid) == pytest.approx(2.0, abs=1e-12)
        assert grid.solid.any() and grid.fluid.any()


def test_inlet_profile_closes_outside_the_open_plane():
    grid = backward_facing_step(10)
    profile = parabolic_inlet(grid, mean_velocity=1.0, lower=1.0, upper=2.0)
    assert profile.shape == (grid.ny,)
    # below the inlet channel (inside the step block) the plane is closed
    assert float(profile[grid.y_faces[:-1] + 0.5 * grid.dy <= 1.0].abs().max()) == 0.0
    # the profile integrates to the requested mean over the open part
    open_rows = (grid.y_faces[:-1] + 0.5 * grid.dy > 1.0)
    assert float(profile[open_rows].mean()) == pytest.approx(1.0, rel=1e-12)


# ----------------------------------------------------------------------- Poisson
def test_masked_poisson_solves_to_machine_precision():
    grid = backward_facing_step(16)
    solver = MaskedPoisson(grid.fluid)
    torch.manual_seed(0)
    b = torch.where(grid.fluid,
                    torch.randn(grid.ny, grid.nx, dtype=torch.float64),
                    torch.zeros(grid.ny, grid.nx, dtype=torch.float64))
    phi = solver.solve(b, tol=1e-12)
    residual = solver.residual(phi, b)
    assert solver.norm(residual) / solver.norm(b) < 1e-11
    # the correction is clamped to zero inside the body
    assert float(phi[grid.solid].abs().max()) == 0.0


def test_iterative_path_agrees_with_direct_path():
    grid = backward_facing_step(12)
    torch.manual_seed(1)
    b = torch.where(grid.fluid,
                    torch.randn(grid.ny, grid.nx, dtype=torch.float64),
                    torch.zeros(grid.ny, grid.nx, dtype=torch.float64))
    direct = MaskedPoisson(grid.fluid, direct=True).solve(b, tol=1e-12)
    iterative = MaskedPoisson(grid.fluid, direct=False).solve(b, tol=1e-11,
                                                              maxiter=4000)
    diff = (direct - iterative).abs().max()
    scale = direct.abs().max()
    assert float(diff / scale) < 1e-9


def test_masked_pcg_wrapper_reports_convergence():
    grid = backward_facing_step(10)
    b = torch.where(grid.fluid, torch.ones(grid.ny, grid.nx, dtype=torch.float64),
                    torch.zeros(grid.ny, grid.nx, dtype=torch.float64))
    _, report = masked_pcg(grid.fluid, b, tol=1e-10)
    assert report.relative_residual < 1e-9


# ------------------------------------------------- C1: masked arm keeps mass
def test_masked_arm_has_no_mass_source():
    flow = _run(_flow(8, "masked"), 30)
    led = flow.ledger
    assert led.div_max_fluid < 1e-10                     # C1 single-step divergence
    assert led.div_max_solid == 0.0                       # nothing inside the body
    assert led.drift == 0.0                               # C1 cumulative drift
    assert led.solid_flux_abs_max == 0.0                  # no flux through the body
    assert led.mass_imbalance_max < 1e-8                  # outlet balances inlet


# --------------------------------------- C1 negative: legacy must show the defect
def test_legacy_arm_reproduces_the_mass_source():
    flow = _run(_flow(8, "legacy", kappa=1.0e4), 30)
    led = flow.ledger
    # The post-projection relaxation breaks the continuity that was just enforced.
    # The divergence shows up in the *first fluid layer at the body surface*: the
    # body absorbs mass, which is why the ledger is normalised per cell.
    assert led.div_max_fluid > 1e-3
    assert led.solid_flux_abs_max > 0.0
    assert led.drift != 0.0
    # and the solid is not rigid (no-slip violated by the penalty)
    assert led.solid_speed_max > 1e-6


def test_legacy_drift_accumulates_linearly_in_time():
    flow = _flow(8, "legacy", kappa=1.0e4)
    _run(flow, 20)
    drift_1, t_1 = flow.ledger.drift, flow.ledger.time
    _run(flow, 20)
    drift_2, t_2 = flow.ledger.drift, flow.ledger.time
    rate_1 = abs(drift_1) / t_1
    rate_2 = abs(drift_2) / t_2
    assert rate_2 == pytest.approx(rate_1, rel=0.5)


# ------------------------------------------------------- C2: solid is rigid
def test_masked_arm_holds_the_body_at_rest():
    flow = _run(_flow(8, "masked"), 30)
    assert flow.ledger.solid_speed_max == 0.0
    # every face of a solid cell carries exactly zero normal velocity
    assert float(flow.u[flow.u_closed].abs().max()) == 0.0
    assert float(flow.v[flow.v_closed].abs().max()) == 0.0
    net, gross = solid_boundary_flux(flow.config.grid, flow.u, flow.v)
    assert net == 0.0 and gross == 0.0


# --------------------------------------------------- C5: negative controls
def test_kappa_to_infinity_degenerates_to_the_rigid_mask():
    loose = _run(_flow(8, "masked", kappa=1.0e2), 25)
    stiff = _run(_flow(8, "masked", kappa=1.0e12), 25)
    assert field_l2(loose, stiff) < 1e-12


def test_kappa_to_zero_matches_the_unobstructed_baseline():
    obstructed = _run(_flow(8, "legacy", kappa=0.0), 25)
    grid = backward_facing_step(8)
    free = CartesianGrid(nx=grid.nx, ny=grid.ny, dx=grid.dx, dy=grid.dy,
                         x0=grid.x0, y0=grid.y0,
                         solid=torch.zeros_like(grid.solid), name="no_obstacle")
    cfg = ImmersedConfig(grid=free, viscosity=0.01, projection="legacy", kappa=0.0,
                         inlet_bounds=(1.0, 2.0), reference_length=1.0, t_end=0.0)
    baseline = _run(ImmersedFlow(cfg), 25)
    assert field_l2(obstructed, baseline) < 1e-12


def test_arm_choice_changes_the_answer():
    masked = _run(_flow(8, "masked"), 40)
    legacy = _run(_flow(8, "legacy", kappa=1.0e4), 40)
    assert field_l2(masked, legacy) > 1e-3


# ------------------------------------------------------------ post-processing
def test_reattachment_finds_the_interpolated_crossing():
    grid = backward_facing_step(8)
    x = grid.x_faces
    tau = torch.where(x < 3.0, -torch.ones_like(x), torch.ones_like(x))
    out = reattachment(grid, tau, x_start=0.0)
    # sign change between the two faces straddling x = 3
    left = float(x[x < 3.0].max())
    right = float(x[x >= 3.0].min())
    assert out["primary_reattachment"] == pytest.approx(3.0, abs=right - left)
    assert len(out["crossings"]) == 1


def test_richardson_recovers_a_known_second_order_trend():
    fine = 6.0 + 0.1
    medium = 6.0 + 0.4
    coarse = 6.0 + 1.6      # errors 0.1, 0.4, 1.6 -> eps32/eps21 = 4 -> p = 2
    out = richardson(fine, medium, coarse, ratio=2.0)
    assert out["observed_order"] == pytest.approx(2.0, rel=1e-6)
    assert out["extrapolated"] == pytest.approx(6.0, abs=1e-6)
    assert math.isfinite(out["gci_fine_pct"])