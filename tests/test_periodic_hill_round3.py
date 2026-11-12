"""Round-3 gates for the periodic-hill solver (defects vi/vii, mission 16672)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.verification.periodic_hill_round3 import (
    gradient_error,
    poiseuille_band_ratio,
    wave_fields,
)


def test_compact_gradient_first_order_on_skewed_mesh():
    e32 = gradient_error(32, 16, 1)
    e64 = gradient_error(64, 32, 1)
    e128 = gradient_error(128, 64, 1)
    assert e32 > e64 > e128
    assert 1.5 <= e32 / e64 <= 5.0
    assert 1.5 <= e64 / e128 <= 5.0
    assert e128 < 1.0e-2


def test_uncorrected_operator_is_inconsistent():
    # no convergence between 32x16 and 128x64 -> order-zero error
    e32 = gradient_error(32, 16, 0)
    e128 = gradient_error(128, 64, 0)
    assert abs(e32 / e128 - 1.0) < 0.10


def test_nonorthogonal_correction_is_worth_an_order_of_magnitude():
    assert gradient_error(128, 64, 0) / gradient_error(128, 64, 1) > 10.0


def test_gradient_reproduces_periodic_wave_on_flat_mesh():
    # with the hill removed the mesh is orthogonal: both the corrected and the
    # uncorrected operator must be second-order accurate there
    field, grad = wave_fields()
    solver = PeriodicHillSolver(PeriodicHillConfig(
        nx=32, ny=16, reynolds=100.0, hill_amplitude=0.0, channel_top_m=3.036))
    m = solver.mesh
    phi = field(m.xc, m.yc)
    ex, ey = grad(m.xc, m.yc)
    inner = slice(2, -2)          # the two wall-adjacent rows are one-sided
    for sweeps in (0, 1):
        gx, gy = solver.compact_gradient(phi, nonorthogonal_sweeps=sweeps)
        err = np.sqrt(np.mean((gx - ex)[inner] ** 2 + (gy - ey)[inner] ** 2)) \
            / max(abs(ex).max(), abs(ey).max())
        assert err < 2.0e-2


def test_south_face_index_shift_is_gone():
    # a field that is linear in y is reproduced exactly on an orthogonal mesh;
    # the round-2 index shift made row 0 see an interior jump through its wall
    # face and row ny-1 lose its south face, which is O(1) wrong
    solver = PeriodicHillSolver(PeriodicHillConfig(
        nx=8, ny=8, reynolds=100.0, hill_amplitude=0.0, channel_top_m=3.0))
    m = solver.mesh
    phi = 0.3 * m.yc
    gx, gy = solver.compact_gradient(phi, nonorthogonal_sweeps=0)
    assert np.abs(gx).max() < 1e-10
    assert np.abs(gy - 0.3).max() < 1e-10


def test_plane_channel_gate_preserved():
    solver = PeriodicHillSolver(PeriodicHillConfig(
        nx=8, ny=32, reynolds=100.0, hill_amplitude=0.0,
        channel_top_m=3.036, time_step_s=0.05))
    solver.advance(2000)
    analytic = poiseuille_band_ratio(solver)
    measured = solver.u.max() / solver.bulk_velocity()
    assert abs(measured - analytic) / analytic < 0.03
    assert np.abs(solver.divergence()).max() < 1e-10


def test_nonorthogonality_magnitudes_are_recorded():
    m = PeriodicHillSolver(PeriodicHillConfig(nx=16, ny=64, reynolds=100.0)).mesh
    assert np.max(np.abs(m.beta_n[1:-1]) / m.dn_n[1:-1]) > 0.5
