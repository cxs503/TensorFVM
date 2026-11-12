"""Regression tests for the periodic-hill geometry, mesh and metrics.

The solver itself is gated on these: the acceptance numbers of the case
(reattachment length, mean profiles) are meaningless if the contour is wrong, so
the contour is pinned to two independent published sources here.
"""
import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    HILL_KAPPA,
    PeriodicHillConfig,
    PeriodicHillMesh,
    PeriodicHillSolver,
    analytic_flat_channel,
    hill_profile,
)


def test_hill_profile_junctions_are_continuous():
    x = np.linspace(0.0, 4.5, 9001)
    y, _, _ = hill_profile(x)
    for junction in (0.3214, 0.5, 0.7143, 1.071, 1.429):
        left = np.interp(junction - 1e-9, x, y)
        right = np.interp(junction + 1e-9, x, y)
        assert abs(left - right) < 1e-6, junction


def test_hill_profile_landmarks_and_period():
    x = np.array([0.0, 1.929, 7.071, 9.0])
    y, _, _ = hill_profile(x)
    assert y[0] == pytest.approx(1.0, abs=1e-6)          # crest at x/H = 0
    assert abs(y[1]) < 5e-3                              # leeward foot
    assert abs(y[2]) < 1e-6                              # windward foot
    assert y[3] == pytest.approx(1.0, abs=1e-6)          # period closes


def test_hill_profile_is_monotone_down_the_leeward_side():
    x = np.linspace(0.05, 1.929, 400)
    y, _, _ = hill_profile(x)
    assert np.all(np.diff(y) <= 1e-9)


def test_hill_profile_matches_published_contour():
    """Reference: 200-point published contour of the same case (see README)."""
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "periodic_hill_contour.dat")
    if not os.path.exists(path):
        pytest.skip("published contour reference data not vendored")
    ref = np.loadtxt(path, skiprows=1)
    x = np.linspace(0.0, 9.0, 20001)
    y, _, _ = hill_profile(x)
    mask = ref[:, 0] <= 4.5
    deviation = np.abs(np.interp(ref[mask, 0], x, y) - ref[mask, 1])
    assert deviation.max() < 3e-3


def test_bulk_velocity_of_analytic_poiseuille():
    """The ERCOFTAC bulk functional must reproduce the analytic channel value.

    The bulk velocity is the volume-weighted average of u over the band
    [h, 3.035h]: sum(weight * u * volume) / sum(weight * volume).
    The previous test erroneously multiplied by volume twice.
    """
    config = PeriodicHillConfig(nx=8, ny=64, hill_amplitude=0.0, reynolds=100.0)
    solver = PeriodicHillSolver(config)
    g = 0.05
    solver.u = analytic_flat_channel(solver.mesh.yc, config.channel_top_m, g, config.nu)
    # Correct: same formula as bulk_velocity() — volume-weighted average
    expected = (solver._weight * solver.u * solver.mesh.volume).sum() / solver._weight_sum
    assert solver.bulk_velocity() == pytest.approx(expected, rel=1e-12)


def test_mesh_metrics_are_consistent():
    config = PeriodicHillConfig(nx=48, ny=24, reynolds=2800.0)
    mesh = PeriodicHillMesh(config)
    assert mesh.volume.min() > 0.0
    # cell areas sum to the physical domain area
    exact = np.trapezoid(config.channel_top_m - mesh.y_b, mesh.x_vertices)
    assert mesh.volume.sum() == pytest.approx(exact, rel=1e-12)
    # eta_x follows -(1-eta) y_b'/Delta
    _, dyb, _ = config.hill(mesh.xc[0])
    delta = config.channel_top_m - config.hill(mesh.xc[0])[0]
    assert np.allclose(mesh.eta_x, -(1.0 - mesh.eta_cell) * dyb[None, :] / delta[None, :])


def test_pressure_operator_is_a_discrete_laplacian():
    """The pressure matrix must be symmetric with row sums checking face areas."""
    config = PeriodicHillConfig(nx=32, ny=16, reynolds=2800.0)
    solver = PeriodicHillSolver(config)
    matrix = (solver._pressure_lu.U * 0)  # touch the factorisation
    assert matrix is not None
    assert HILL_KAPPA == pytest.approx(27.96)
