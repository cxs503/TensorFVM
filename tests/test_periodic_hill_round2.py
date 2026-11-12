"""Round-2 regression tests for the periodic-hill solver (mission 16503).

Every test here fails on the round-1 code; the failure mode is quoted next to
each assertion.  The analytic gate is the plane channel (``hill_amplitude = 0``,
for which :func:`tensorfvm.periodic_hill.analytic_flat_channel` is exact).
"""
import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillSolver,
    analytic_flat_channel,
)
from tensorfvm.verification.periodic_hill_round2 import poiseuille_band_ratio


def _interior(mask_ny, mask_nx):
    interior = np.ones((mask_ny, mask_nx), dtype=bool)
    interior[0, :] = False
    interior[-1, :] = False
    return interior


def test_momentum_operator_rows_are_consistent():
    """|diag| == sum|off-diag| per interior row; walls add the Dirichlet share.

    Round 1 accumulated ``diag_mom += coef`` with ``coef`` of shape ``(nx,)``
    inside a loop over an ``(ny, nx)`` array, so every row got every row's
    coefficients: ``|diag|`` was ~``ny`` times the off-diagonal row sum (56208
    vs 892 on a 16x64 mesh).
    """
    config = PeriodicHillConfig(nx=24, ny=12, reynolds=2800.0)
    solver = PeriodicHillSolver(config)
    matrix = solver.laplacian.tocsr()
    diag = -matrix.diagonal()
    off = np.zeros_like(diag)
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        off[i] = np.abs(row.data[row.indices != i]).sum()
    interior = _interior(config.ny, config.nx).ravel()
    assert np.all(diag > 0.0)
    assert np.abs(diag[interior] - off[interior]).max() < 1e-10 * diag.max()


def test_wall_faces_carry_no_slip_coefficient():
    """The two wall eta levels must carry |A|/d_n; round 1 left them at zero.

    With ``c_n[0] = c_n[-1] = 0`` the momentum operator had no wall friction at
    all and the channel free-slipped (plug flow).
    """
    config = PeriodicHillConfig(nx=16, ny=8, reynolds=100.0)
    solver = PeriodicHillSolver(config)
    mesh = solver.mesh
    expected = mesh.area_n[0] / mesh.dn_n[0]
    assert np.all(solver.c_n[0] > 0.0)
    assert np.all(solver.c_n[-1] > 0.0)
    assert np.allclose(solver.c_n[0], expected)
    # half a cell height -> the wall coefficient doubles; exact check on the
    # orthogonal (flat) mesh, where dn_n[1] is exactly one cell height
    flat = PeriodicHillSolver(PeriodicHillConfig(nx=16, ny=8, hill_amplitude=0.0,
                                                reynolds=100.0)).mesh
    assert np.allclose(flat.dn_n[0], 0.5 * flat.dn_n[1])
    assert np.allclose(flat.dn_n[-1], 0.5 * flat.dn_n[-2])


def test_streamwise_coupling_has_no_seam():
    """Face distances must use the minimum image across the periodic seam.

    Round 1 used ``xc[0] - xc[nx-1] = -(L - dx)`` (8.4375 instead of 0.5625 on a
    16-cell mesh), so the diffusion coefficient of the wrap face was 15x too
    small and the pressure-gradient stencil was one-sided there.
    """
    config = PeriodicHillConfig(nx=16, ny=8, hill_amplitude=0.0, reynolds=100.0)
    solver = PeriodicHillSolver(config)
    assert np.allclose(solver.mesh.dn_e, solver.mesh.dx)


def test_pressure_operator_has_no_volume_scaling_and_a_constant_null_space():
    """The Poisson operator is ``sum_f |A|/d_n`` (a flux), symmetric, zero rows.

    Round 1 divided the off-diagonals by the cell volume but not the diagonal,
    which is a factor ``Vol`` between the two halves of the same row.
    """
    config = PeriodicHillConfig(nx=16, ny=8, reynolds=2800.0)
    solver = PeriodicHillSolver(config)
    pressure = solver.pressure.tocsr()
    assert np.abs(pressure - pressure.T).max() < 1e-12
    assert pressure.diagonal().min() > 0.0
    row_sums = np.asarray(pressure.sum(axis=1)).ravel()
    # the constant vector is in the null space up to the ~1e-10 regularisation
    scale = np.abs(pressure.diagonal()).max()
    assert np.abs(row_sums).max() < 1e-6 * scale
    # off-diagonal magnitude is a face coefficient |A|/d_n, not |A|/(d_n Vol)
    off_diag = np.abs(pressure - np.diag(pressure.diagonal())).max()
    face_coefficient = (solver.mesh.area_n / solver.mesh.dn_n).max()
    assert off_diag <= face_coefficient * (1.0 + 1e-9)


def test_plane_channel_matches_analytic_poiseuille():
    """The round-1 plan's gate: plane channel on the analytic solution to <1e-3.

    Round 1 produced ``u_max / u_b = 1.003`` (plug flow) instead of the analytic
    band value 1.3475 -- a 26 % error, and no wall shear at all.
    """
    config = PeriodicHillConfig(nx=16, ny=64, hill_amplitude=0.0, reynolds=100.0,
                                time_step_s=0.05)
    solver = PeriodicHillSolver(config)
    solver.advance(6000)
    exact = analytic_flat_channel(solver.mesh.yc, config.channel_top_m,
                                  solver.body_force, config.nu)
    u_max = float(solver.u.max())
    assert np.abs(solver.u - exact).max() / u_max < 1e-3
    reference = poiseuille_band_ratio(solver)
    assert abs(u_max / solver.bulk_velocity() / reference - 1.0) < 1e-3
    # the band bulk velocity is not the full-channel mean: 1.5 is the wrong anchor
    assert 1.33 < reference < 1.36
    # transverse velocity must stay at round-off for a fully developed channel
    assert np.abs(solver.v).max() < 1e-12


def test_projection_conserves_mass_at_flux_level():
    """Mass conservation of the projected face fluxes (round 1: 2.33 in the hill case)."""
    config = PeriodicHillConfig(nx=16, ny=32, hill_amplitude=0.0, reynolds=100.0,
                                time_step_s=0.05)
    solver = PeriodicHillSolver(config)
    solver.advance(200)
    assert np.abs(solver.flux_divergence()).max() < 1e-10
    # the cell-average divergence is a *diagnostic*, not a constraint of the scheme
    assert np.abs(solver.divergence()).max() < 1e-8


def test_odd_even_mode_is_in_the_collocated_null_space():
    """Half of the discrete spectrum is invisible: documented, not a bug to hide.

    On the orthogonal mesh the face-averaged divergence and both cell gradients
    return exactly zero for the checkerboard mode, so a ``div(grad)`` Poisson
    operator would be singular there and Rhie-Chow (or a staggered layout) is
    mandatory.  On the slanted hill mesh the mode leaks only partially.
    """
    flat = PeriodicHillSolver(PeriodicHillConfig(nx=32, ny=16, hill_amplitude=0.0))
    checker = (-1.0) ** np.arange(32)[None, :] * np.ones((16, 1))
    zeros = np.zeros_like(checker)
    for gradient in (flat.gradient(checker), flat.compact_gradient(checker)):
        assert max(np.abs(gradient[0]).max(), np.abs(gradient[1]).max()) < 1e-12
    assert np.abs(flat.divergence(checker, zeros)).max() < 1e-12
    hill = PeriodicHillSolver(PeriodicHillConfig(nx=32, ny=16))
    checker_hill = (-1.0) ** np.arange(32)[None, :] * np.ones((16, 1))
    assert np.abs(hill.divergence(checker_hill, np.zeros_like(checker_hill))).max() > 0.0


def test_hill_case_is_reported_not_claimed():
    """Guard: the hill field solution is NOT verified -- only its reach is reported."""
    from tensorfvm.verification.periodic_hill_round2 import _hill_probe
    probe = _hill_probe(32, 16, 2800.0, 0.01, 40, 0.2)
    assert probe["steps_requested"] == 40
    assert set(probe) >= {"blew_up_at_step", "separation", "reattachment",
                          "flux_divergence", "cell_divergence"}