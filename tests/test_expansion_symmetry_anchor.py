"""Symmetry and analytic anchors of the planar sudden expansion (``AR = 4``).

Three invariants that the campaign ledger never asserted on the *solved* field
(it only asserted them on the inlet profile):

1. the discrete solution of the centred sudden expansion is **mirror
   symmetric** in ``y`` -- the operator, the fractionally weighted inlet and the
   DCT pressure solve all commute with ``y -> H - y``, so any asymmetry is a
   numerical defect and must show up at round-off level, not at the 0.1 % level;
2. the inlet plane carries **exactly** ``U_in * H_in = U_in * D / AR`` -- the
   fractional weighting of the opening exists for that reason, and a whole-cell
   opening silently changes the area ratio (and hence the Reynolds number) with
   the mesh;
3. the wall shear of a parabolic profile on the MAC grid reproduces the analytic
   plane-Poiseuille value ``tau_w = 6 mu U / H``, which is the reference that the
   ``C_f`` based reattachment criterion and the Borda-Carnot friction correction
   are both built on.

Half of these tests are deliberately *negative controls*: a metric that cannot
report asymmetry cannot verify symmetry either.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.internal_flow.mac import MACConfig, StaggeredMAC
from tensorfvm.internal_flow.reference import developed_channel_friction
from tensorfvm.internal_flow.sudden_expansion import (
    SuddenExpansionCase,
    SuddenExpansionConfig,
    inlet_profile,
)

STEPS = 400


def mirror_residuals(case: SuddenExpansionCase) -> tuple[float, float]:
    u, v = case.solver.cell_velocity()
    return (float((u - u.flip(1)).abs().max()),
            float((v + v.flip(1)).abs().max()))


def test_inlet_profile_is_fractionally_weighted_on_every_mesh() -> None:
    """The discrete opening must carry ``U * D / AR`` for every cell count."""
    for n in (8, 10, 16, 20, 30, 40):
        config = SuddenExpansionConfig(cells_per_downstream_height=n)
        profile = inlet_profile(config)
        flux = sum(profile) * config.downstream_height / n
        assert math.isclose(flux, config.inlet_velocity * config.inlet_height,
                            rel_tol=1e-12), n
        # a whole-cell opening is the failure mode this guards against: it is
        # only ever exact by accident
        for lower, upper in zip(profile, reversed(profile)):
            assert lower == pytest.approx(upper)


def test_solved_inlet_flux_is_exact() -> None:
    """The velocity field, not just the profile, must carry the nominal flux."""
    config = SuddenExpansionConfig(cells_per_downstream_height=10)
    case = SuddenExpansionCase(config)
    case.solver.run(max_steps=STEPS)
    solver = case.solver
    flux = float(solver.u[0, :].sum() * solver.hy)
    nominal = config.inlet_velocity * config.inlet_height
    assert math.isclose(flux, nominal, rel_tol=1e-10), (flux, nominal)
    # the Reynolds number of the case is defined through that flux
    assert config.viscosity == pytest.approx(
        config.outlet_velocity * config.downstream_height / config.reynolds)


def test_solution_stays_mirror_symmetric() -> None:
    """``u`` must equal its y-mirror and ``v`` must be antisymmetric."""
    config = SuddenExpansionConfig(cells_per_downstream_height=10)
    case = SuddenExpansionCase(config)
    case.solver.run(max_steps=STEPS)
    u_asymmetry, v_antisymmetry = mirror_residuals(case)
    scale = float(case.solver.cell_velocity()[0].abs().max())
    assert u_asymmetry < 1e-11 * scale, u_asymmetry
    assert v_antisymmetry < 1e-11 * scale, v_antisymmetry
    _, cf_bottom = case.wall_friction("bottom")
    _, cf_top = case.wall_friction("top")
    assert float((cf_bottom - cf_top).abs().max()) < 1e-11


def test_asymmetric_inlet_is_detected() -> None:
    """Negative control: an off-centre opening must break the mirror symmetry."""
    config = SuddenExpansionConfig(cells_per_downstream_height=10)
    symmetric = list(inlet_profile(config))
    broken = list(symmetric)
    # move a quarter cell of opening from the upper edge cell to the lower one
    broken[3] = broken[3] + 0.25
    broken[6] = broken[6] - 0.25
    assert broken != symmetric
    case = SuddenExpansionCase(config)
    reference = SuddenExpansionCase(config)
    reference.solver.run(max_steps=STEPS)
    symmetric_residual, _ = mirror_residuals(reference)
    # the inlet profile lives in the frozen solver config, so drive the solver
    # directly with the broken profile instead of the case default
    broken_solver = StaggeredMAC(MACConfig(
        nx=case.solver.nx, ny=case.solver.ny,
        length=float(case.solver.nx * case.solver.hx),
        height=float(case.solver.ny * case.solver.hy),
        viscosity=config.viscosity, reference_velocity=config.inlet_velocity,
        cfl=config.cfl, max_steps=config.max_steps,
        tolerance=config.tolerance, check_every=config.check_every,
        left="inlet", right="outlet", inlet_profile=tuple(broken)))
    broken_solver.run(max_steps=STEPS)
    u = broken_solver.cell_velocity()[0]
    residual = float((u - u.flip(1)).abs().max())
    assert symmetric_residual < 1e-11, symmetric_residual
    assert residual > 1e-3, residual


def test_parabolic_wall_shear_matches_plane_poiseuille() -> None:
    """``tau_w`` on the MAC grid must reproduce ``6 mu U / H`` (analytic)."""
    ny = 20
    height = 1.0
    viscosity = 0.0025
    mean = 0.25
    solver = StaggeredMAC(MACConfig(nx=8, ny=ny, length=1.0, height=height,
                                    viscosity=viscosity, reference_velocity=1.0))
    y = (torch.arange(ny, dtype=solver.dtype) + 0.5) * solver.hy
    parabola = 6.0 * mean * (y / height) * (1.0 - y / height)
    solver.u = parabola.repeat(solver.nx + 1, 1)
    tau_bottom = float(solver.rho * solver.nu * 2.0 * solver.u[0, 0] / solver.hy)
    tau_top = float(solver.rho * solver.nu * 2.0 * solver.u[0, -1] / solver.hy)
    analytic = 6.0 * viscosity * mean / height
    # sampling a *continuum* parabola at the cell centres leaves the mirror-ghost
    # wall shear first-order biased by hy/(2H) = -2.5 % at ny = 20; the steady
    # discrete solution of the same scheme compensates that bias (the audit
    # measures 0.9 % at D/20), so the tolerance below is the sampling bias, not a
    # solver error
    sampling_bias = 0.5 * solver.hy / height
    assert tau_bottom == pytest.approx(analytic, rel=1.5 * sampling_bias), tau_bottom
    assert tau_top == pytest.approx(analytic, rel=1.5 * sampling_bias), tau_top
    # the normalised coefficient keeps its analytic form too
    cf = tau_bottom / (0.5 * solver.rho * 1.0 ** 2)
    assert cf == pytest.approx(
        developed_channel_friction(100.0) * (mean / 1.0) ** 2, rel=1.5 * sampling_bias)