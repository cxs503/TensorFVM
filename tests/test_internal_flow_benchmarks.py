"""Regression tests of the three internal-flow verification cases.

The expensive full-resolution reference runs live in
``tensorfvm.internal_flow.study`` (artifacts under ``artifacts/internal-flow``);
these tests keep the coarse-mesh versions so that a physics or sign regression
fails CI in a minute or two.
"""

import math

import pytest
import torch

from tensorfvm.internal_flow.backward_step import BackwardStepCase, BackwardStepConfig
from tensorfvm.internal_flow.natural_convection import (NaturalConvectionCase,
                                                        NaturalConvectionConfig)
from tensorfvm.internal_flow.reference import (DE_VAHL_DAVIS_NU, borda_carnot_loss,
                                              developed_channel_friction)
from tensorfvm.internal_flow.sudden_expansion import (SuddenExpansionCase,
                                                     SuddenExpansionConfig)


def test_analytic_reference_forms():
    assert math.isclose(borda_carnot_loss(4.0), 0.5625, rel_tol=1e-12)
    assert math.isclose(borda_carnot_loss(2.0), 0.25, rel_tol=1e-12)
    assert math.isclose(developed_channel_friction(800.0), 0.015, rel_tol=1e-12)
    assert DE_VAHL_DAVIS_NU[1.0e6] == 8.8
    with pytest.raises(ValueError):
        borda_carnot_loss(1.0)
    with pytest.raises(ValueError):
        developed_channel_friction(0.0)


def test_natural_convection_matches_de_vahl_davis_at_ra_1e4():
    """Coarse buoyant cavity: Nu(1e4) must land inside the 3% acceptance band."""
    case = NaturalConvectionCase(NaturalConvectionConfig(
        n=24, rayleigh=1e4, max_steps=40000, tolerance=1e-6, check_every=200))
    case.run()
    data = case.diagnostics()
    assert math.isclose(data["rayleigh_discrete"], 1e4, rel_tol=1e-10)
    assert data["nu_error_percent"] is not None
    assert abs(data["nu_error_percent"]) < 3.0
    # the extremum of the streamfunction sits in the upper half of the cavity
    assert data["psi_max_location"][1] > 0.5
    # the hot wall is hotter at the bottom (cold fluid arrives there)
    assert data["nu_local"][0] > data["nu_local"][-1]


def test_sudden_expansion_reproduces_borda_carnot():
    """Coarse sudden expansion: the fitted loss coefficient tracks (1-1/AR)^2."""
    case = SuddenExpansionCase(SuddenExpansionConfig(
        cells_per_downstream_height=24, max_steps=60000, tolerance=1e-6,
        check_every=200))
    case.run()
    data = case.diagnostics()
    theory = borda_carnot_loss(4.0)
    assert abs(data["k_corrected"] - theory) / theory < 0.10
    # the loss coefficient and the static-pressure recovery must stay consistent
    # with each other: K + Cp = (1 - 1/AR)^2 + 2(1/AR)(1 - 1/AR) = 1 - 1/AR^2
    recovered = data["cp_recovery_raw"] + data["k_raw"]
    assert abs(recovered - (1.0 - 1.0 / data["area_ratio"] ** 2)) < 0.05
    # the expansion is symmetric, so both walls detach at the same station
    assert data["reattachment_bottom"] is not None
    assert data["reattachment_top"] is not None
    assert abs(data["reattachment_bottom"] - data["reattachment_top"]) < 0.25
    assert 0.5 < data["reattachment_mean_over_D"] < 8.0


def test_backward_step_bubble_and_reattachment_extraction():
    """A short coarse run must produce a bounded field and a growing bubble."""
    case = BackwardStepCase(BackwardStepConfig(
        cells_per_step_height=8, cfl=0.4, max_steps=4000, tolerance=1e-30,
        check_every=1000))
    case.run()
    data = case.diagnostics()
    assert float(case.solver.u.abs().max()) < 2.0      # bounded, no blow-up
    assert data["min_cf"] < 0.0                        # a recirculation exists
    assert data["reattachment_in_step_heights"] is not None
    assert 0.0 < data["reattachment_in_step_heights"] < 22.0
    # downstream of the bubble the wall shear is positive again and the outlet
    # face carries (almost) no friction: 12/Re_H = 0.015
    x, cf = case.cf_profile()
    assert float(cf[int(0.5 * len(cf)):].max()) > 0.0    # attached again downstream
    assert abs(float(cf[-1])) < 0.02


def test_reattachment_interpolation_is_exact_on_a_linear_profile():
    """The zero-crossing interpolation must be exact for a linear skin friction."""
    case = BackwardStepCase(BackwardStepConfig(
        cells_per_step_height=8, max_steps=8, tolerance=1e-30, check_every=4))
    solver = case.solver
    hx = solver.hx
    target = 3.25 * hx
    x = torch.arange(solver.nx + 1, dtype=solver.dtype) * hx
    values = (x - target) * 1e-3
    solver.u[:, 0] = values * solver.hy / (2.0 * solver.nu)
    found = solver.reattachment_length()
    assert found is not None
    assert abs(found - float(target)) < 1e-9