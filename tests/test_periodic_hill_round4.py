"""Round-4 gates for the periodic-hill solver (defects viii/ix/x, mission 16765)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.verification.periodic_hill_round4 import (
    bulk_control_checks,
    channel_gate,
    constant_field_error,
    legacy_constant_field_error,
    upwind_stencil_mismatch,
)


def _solver(**kwargs):
    base = dict(nx=48, ny=24, reynolds=2800.0, time_step_s=0.01,
                body_force_relaxation=1.0)
    base.update(kwargs)
    return PeriodicHillSolver(PeriodicHillConfig(**base))


def test_convection_annihilates_a_constant_on_a_divergence_free_flux():
    # outflow is flux/V_donor *and* inflow is flux/V_receiver; on this mesh the
    # volume varies by 1.49, so the round-3 donor-volume form gave O(1) here
    value, div = constant_field_error(_solver())
    assert div < 1e-6
    assert value < 1e-6


def test_round3_assembly_is_the_documented_negative_control():
    # non-vacuousness of the test above: the old assembly is O(1) wrong
    assert legacy_constant_field_error(_solver()) > 0.1


def test_eta_face_sign_matches_an_independent_upwind_stencil():
    # the eta family is a net *outflow*: the cell the flux leaves gains, the cell
    # it enters loses.  Swapped signs = anti-upwind (negative diffusion).
    assert upwind_stencil_mismatch(_solver()) < 1e-10


def test_convection_of_a_linear_field_has_the_right_sign():
    # a positive upward mass flux must transport a positive lower value upward
    solver = _solver(nx=8, ny=16)
    m = solver.mesh
    flux_n = np.zeros((solver.ny + 1, solver.nx))
    flux_n[1:-1] = 0.5 * m.area_n[1:-1]              # uniform upward flux
    flux_e = np.zeros((solver.ny, solver.nx))
    phi = m.yc.copy()                                # increasing in y
    out = solver._convection(phi, flux_e, flux_n, blend=0.0)
    # net outflow = f (phi_P - phi_{P-1}) > 0 for dphi/dy > 0 and f > 0.  With the
    # round-3 sign inversion this is negative (anti-upwind), so the test bites.
    assert np.all(out[1:-1] > 0.0)


def test_bulk_velocity_is_imposed_to_round_off():
    result = bulk_control_checks()
    assert result["max_abs_bulk_error_exact"] < 1e-8
    # the legacy law had |lambda|^2 = 1 + relaxation > 1 (unstable for any value)
    assert result["legacy_growth_factor_sqrt_1_plus_relaxation"]["0.2"] > 1.0


def test_relaxation_is_a_stable_first_order_filter():
    # target step 1.0 -> 1.1: the bulk error must decay by (1 - relaxation) = 1/2
    ratios = bulk_control_checks()["observed_ratio_relaxation_half"]
    assert 0.45 < ratios[0] < 0.55


def test_plane_channel_gate_survives():
    gate = channel_gate()
    assert gate["rel_error"] < 0.03
    assert gate["max_cell_divergence"] < 1e-10


def test_re2800_hill_run_no_longer_blows_up():
    # rounds 2/3 went NaN inside 1000 steps here; 2000 steps is the cheap gate
    solver = _solver()
    solver.advance(2000)
    assert np.all(np.isfinite(solver.u)) and np.all(np.isfinite(solver.v))
    assert float(np.abs(solver.u).max()) < 5.0
    assert abs(solver.bulk_velocity() - 1.0) < 1e-8
    assert float(np.abs(solver.flux_divergence()).max()) < 1e-6


def test_convection_blend_is_respected():
    # guards the round-4 signature change: blend=0 (upwind) != blend=1 (central)
    solver = _solver(nx=16, ny=8)
    flux_e, flux_n = solver._face_flux(solver.u + 1.0, solver.v)
    phi = np.sin(np.linspace(0, 2 * np.pi, 16))[None, :] * np.ones((8, 1))
    upwind = solver._convection(phi, flux_e, flux_n, blend=0.0)
    central = solver._convection(phi, flux_e, flux_n, blend=1.0)
    assert np.abs(upwind - central).max() > 1e-6
    assert np.allclose(solver._convection(phi, flux_e, flux_n),
                       solver._convection(phi, flux_e, flux_n,
                                          blend=solver.config.convection_blend))