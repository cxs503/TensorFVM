"""Tests for the 3-D wall-resolved channel LES (:mod:`tensorfvm.channel_les3d`).

The tests are constructed so that a wrong sign, a missing wall term or a
non-conservative stencil fails them; none of them merely executes the code.
"""
import math

import numpy as np
import pytest

from tensorfvm.channel_les3d import (ChannelLES3D, ChannelLESConfig,
                                     reichardt_profile, smagorinsky_viscosity,
                                     van_driest_damping)


def small(**kwargs):
    base = dict(re_tau=10.0, nx=12, ny=16, nz=10, lx=2.0 * math.pi, lz=math.pi,
                smagorinsky=0.0, dt=2.0e-3)
    base.update(kwargs)
    return ChannelLESConfig(**base)


def test_config_rejects_bad_input():
    with pytest.raises(ValueError):
        ChannelLESConfig(nx=4)
    with pytest.raises(ValueError):
        ChannelLESConfig(dt=0.5)
    with pytest.raises(ValueError):
        ChannelLESConfig(smagorinsky=-1.0)


def test_wall_normal_grid_resolves_both_walls():
    solver = ChannelLES3D(small(ny=64, re_tau=180.0))
    assert solver.yf[0] == 0.0 and abs(solver.yf[-1] - 2.0) < 1e-12
    assert np.all(np.diff(solver.yf) > 0)
    # first cell centre in wall units must be of order one, not 10
    assert 0.2 < solver.y_plus[0] < 2.0
    # clustering is symmetric
    assert abs(solver.dy[0] - solver.dy[-1]) < 1e-12


def test_divergence_matches_an_independent_loop_implementation():
    """The vectorised divergence must equal the documented stencil, index by index.

    The reference is a plain triple loop written from the docstring of
    :meth:`ChannelLES3D.divergence` (``u`` at ``x_j``, ``v`` at ``yf``, ``w`` at
    ``z_k``), so an index-convention slip in the vectorised version -- including
    the wall rows of ``v`` -- cannot pass.
    """
    solver = ChannelLES3D(small())
    rng = np.random.default_rng(11)
    solver.u = rng.standard_normal(solver.u.shape)
    solver.v = rng.standard_normal(solver.v.shape)
    solver.w = rng.standard_normal(solver.w.shape)
    solver.v[0] = 0.0
    solver.v[-1] = 0.0
    ny, nx, nz = solver.config.ny, solver.config.nx, solver.config.nz
    reference = np.zeros((ny, nx, nz))
    for i in range(ny):
        for j in range(nx):
            for k in range(nz):
                reference[i, j, k] = (
                    (solver.u[i, (j + 1) % nx, k] - solver.u[i, j, k]) / solver.dx
                    + (solver.v[i + 1, j, k] - solver.v[i, j, k]) / solver.dy[i]
                    + (solver.w[i, j, (k + 1) % nz] - solver.w[i, j, k]) / solver.dz)
    assert np.allclose(solver.divergence(), reference, atol=1e-13)


def test_laplacian_matches_div_grad_on_a_known_field():
    # (a) on a uniform grid the operator is the textbook three-point Laplacian
    solver = ChannelLES3D(small(stretch=0.0))
    y = solver.yc[:, None, None]
    phi = (y ** 3) * np.ones_like(solver.u)
    lap = solver.laplacian(phi)
    assert np.allclose(lap[1:-1], (6.0 * y)[1:-1], rtol=1e-10, atol=1e-10)

    # (b) on the stretched grid it must equal the documented div(grad) stencil
    #     exactly, and stay second-order accurate in the middle of the channel
    solver = ChannelLES3D(small())
    y = solver.yc[:, None, None]
    phi = (y ** 3) * np.ones_like(solver.u)
    lap = solver.laplacian(phi)
    ny = solver.config.ny
    reference = np.zeros_like(lap)
    for i in range(1, ny - 1):
        reference[i] = (((phi[i + 1] - phi[i]) / solver.dy_cc[i]
                         - (phi[i] - phi[i - 1]) / solver.dy_cc[i - 1]) / solver.dy[i])
    assert np.allclose(lap[1:-1], reference[1:-1], atol=1e-13)
    middle = slice(1 + ny // 4, ny - 1 - ny // 4)
    assert np.allclose(lap[middle], (6.0 * y)[middle], rtol=0.15)


def test_projection_is_exactly_divergence_free():
    solver = ChannelLES3D(small(re_tau=50.0))
    rng = np.random.default_rng(3)
    solver.u = rng.standard_normal(solver.u.shape)
    solver.v[1:-1] = rng.standard_normal(solver.v[1:-1].shape)
    solver.w = rng.standard_normal(solver.w.shape)
    solver.v[0] = 0.0
    solver.v[-1] = 0.0
    before = np.abs(solver.divergence()).max()
    solver.project(solver.config.dt)
    after = np.abs(solver.divergence()).max()
    assert before > 1e-2
    assert after < 1e-10
    assert after < 1e-8 * before
    # wall no-penetration survives the projection
    assert np.all(solver.v[0] == 0.0) and np.all(solver.v[-1] == 0.0)


def test_uniform_flow_has_zero_convective_acceleration():
    """A uniform stream plus a uniform body force must stay exactly uniform."""
    solver = ChannelLES3D(small(re_tau=1.0, dt=1.0e-2))
    solver.u = 0.7 * np.ones_like(solver.u)
    conv_u, conv_v, conv_w = solver.convection()
    assert np.allclose(conv_u, 0.0, atol=1e-12)
    assert np.allclose(conv_v, 0.0, atol=1e-12)
    assert np.allclose(conv_w, 0.0, atol=1e-12)


def test_smagorinsky_recovers_analytic_strain():
    """Simple shear ``u = a y``: ``|S| = a/sqrt(2)``, so ``nu_t = (Cs d)^2 a/sqrt2 * f``."""
    zeros = np.zeros((6, 5, 4))
    a = 2.5
    du_dy = a * np.ones_like(zeros)
    delta, cs = 0.3, 0.1
    y_plus = np.full((6, 1, 1), 1.0e6)               # damping ~ 1
    nut = smagorinsky_viscosity(du_dy, zeros, zeros, zeros, zeros, zeros,
                                zeros, zeros, zeros, delta, y_plus, cs=cs)
    expected = (cs * delta) ** 2 * a
    assert np.allclose(nut, expected, rtol=1e-6)
    assert np.allclose(van_driest_damping(np.array([0.0, 25.0])),
                       [0.0, 1.0 - math.exp(-1.0)])


def test_laminar_channel_reaches_the_analytic_parabola():
    """At ``Re_tau = 2`` the flow is laminar: the steady state is the parabola.

    This is the end-to-end test of the momentum assembly (wall half-cell flux,
    body force, projection): the analytic solution is
    ``u(y) = G y (2 - y) / (2 nu)`` on ``[0, 2]``.

    History (2026-11-15, m21024): the test used ``dt = 5e-3``, which exceeds
    the explicit wall-diffusion limit of this grid (``3 nu dt / dy0^2 = 4.7``);
    the run diverged to ``1e129`` and the test was parked as ``xfail`` with the
    reason "not converged in 4000 steps".  The time step below is inside the
    limit, the diffusion flux spacing bug is fixed (``dy`` -> ``dy_cc``), and
    the tolerance is the truncation of the half-cell wall flux: the wall
    gradient is approximated by ``u(yc0)/yc0``, which for a parabola is 1 %
    off at this ``ny`` (measured 1.0e-2).  ``test_wall_flux_error_is_first_
    order_in_the_grid`` pins the order down.
    """
    cfg = small(re_tau=2.0, nx=8, ny=24, nz=8, dt=1.5e-3, stretch=1.2)
    solver = ChannelLES3D(cfg)
    y = solver.yc[:, None, None]
    analytic = solver.body_force * y * (2.0 - y) / (2.0 * solver.nu)
    solver.u = 0.25 * analytic * np.ones_like(solver.u)     # start 4x too slow
    solver.v[:] = 0.0
    solver.w[:] = 0.0
    assert solver.diffusion_number() < 2.0                  # inside the limit
    solver.run(steps=4000, sample_every=0, check_every=0)
    profile = solver.u.mean(axis=(1, 2))
    reference = analytic[:, 0, 0]
    deviation = np.max(np.abs(profile - reference) / reference)
    # the start was 25 % of the target and the run is monotone, not a fixed
    # point of the initial condition
    assert deviation < 2e-2, deviation
    tau_wall = solver.nu * profile[0] / (0.5 * solver.dy[0])
    assert abs(tau_wall - solver.body_force * 1.0) < 5e-3


def test_wall_flux_error_is_first_order_in_the_grid():
    """The half-cell wall flux is first order and is the only error source.

    For the analytic parabola the interior fluxes are exact (the midpoint
    gradient of a quadratic is exact), so the residual of the discrete
    momentum balance at the wall cell is ``O(h)`` while the interior residual
    tends to zero faster.  Asserting the *order* instead of a value stops both
    a silent regression (a first-order interior bug flattens the ratios to 1)
    and an over-claimed second order.
    """
    def residuals(ny):
        cfg = ChannelLESConfig(re_tau=2.0, nx=8, ny=ny, nz=8, smagorinsky=0.0,
                              dt=1e-3, stretch=1.0)
        solver = ChannelLES3D(cfg)
        y = solver.yc[:, None, None]
        solver.u = solver.body_force * y * (2.0 - y) / (2.0 * solver.nu) \
            * np.ones_like(solver.u)
        lap, _, _ = solver.diffusion(nut=np.zeros_like(solver.u))
        residual = np.abs(lap + solver.body_force)[:, 0, 0]
        return float(residual[0]), float(residual[1:-1].max())

    r24, inner24 = residuals(24)
    r48, inner48 = residuals(48)
    r96, inner96 = residuals(96)
    assert r24 > 10.0 * inner24                 # the wall cell owns the error
    # MEASURED (not ideal): the wall-cell residual is *h-independent*, because
    # the half-cell conductance approximates u'(0) by u(yc0)/yc0 whose error is
    # O(h) and is then divided by O(h).  The steady-state error is O(h) (1.0e-2
    # at ny = 24, see the laminar test); fixing it needs a second-order
    # one-sided wall gradient, which is an open item in the report.
    assert 0.85 < r24 / r48 < 1.02 and 0.85 < r48 / r96 < 1.02
    # the interior residual (away from the wall cell) is second order
    assert inner48 / inner24 < 0.4 and inner96 / inner48 < 0.4


def test_turbulent_short_run_stays_finite_and_divergence_free():
    """60 explicit steps of the SGS channel stay finite and solenoidal.

    The grid is deliberately small (``ny = 24``, ``y1+ ~ 1.6``) and the run
    short: on the *production* grid (``ny = 64``, ``y1+ ~ 0.5``) the explicit
    wall-normal diffusion of ``nu + nu_t`` sits at the stability limit and the
    run diverges -- see ``docs/case-fvm-channel-kmm180-les3d/REPORT.md``
    (open item: implicit wall-normal diffusion).
    """
    cfg = small(re_tau=180.0, nx=16, ny=24, nz=16, dt=1.0e-3, smagorinsky=0.1)
    solver = ChannelLES3D(cfg, seed=5).initialise(noise=0.1, profile="reichardt")
    solver.run(steps=60, sample_every=20, check_every=0)
    stats = solver.statistics()
    assert np.isfinite(stats["u_plus"]).all()
    assert np.abs(solver.divergence()).max() < 1e-9
    # the SGS model must add viscosity, i.e. the run is not pure DNS of a laminar state
    assert float(solver.eddy_viscosity().max()) > 0.0


def test_gradients_are_cell_centred_and_exact_for_a_linear_field():
    """Every gradient entry has ``(ny, nx, nz)`` rows and ``u = a y`` gives ``a``.

    Before 2026-11-15 the ``u``/``w`` gradients had ``ny-2`` rows against
    ``ny-1`` spacings (a hard ``ValueError``), and the ``v`` gradients carried
    ``ny-1`` rows because they were differenced on the y-faces; both made
    :func:`smagorinsky_viscosity` unusable on any grid.
    """
    solver = ChannelLES3D(small(nx=8, ny=10, nz=8))
    a = 3.7
    solver.u = a * solver.yc[:, None, None] * np.ones_like(solver.u)
    solver.v[:] = 0.0
    solver.w = 0.5 * a * solver.yc[:, None, None] * np.ones_like(solver.w)
    g = solver.velocity_gradients()
    for key in ("du_dx", "du_dy", "du_dz", "dv_dx", "dv_dy", "dv_dz",
                "dw_dx", "dw_dy", "dw_dz"):
        assert g[key].shape == solver.u.shape, (key, g[key].shape)
    assert np.allclose(g["du_dy"], a, rtol=1e-12, atol=1e-12)
    assert np.allclose(g["dw_dy"], 0.5 * a, rtol=1e-12, atol=1e-12)
    assert np.allclose(g["du_dx"], 0.0, atol=1e-12)
    assert np.allclose(g["dv_dy"], 0.0, atol=1e-12)


def test_sgs_viscosity_broadcasts_the_wall_normal_columns():
    """``delta`` and ``y+`` are y-columns; the viscosity must still be 3-D.

    ``delta = (dx dy dz)^(1/3)`` is an array on the stretched grid and
    ``y_plus`` is a column, so :func:`smagorinsky_viscosity` must broadcast
    them against the strain magnitude (a missing reshape raised
    ``ValueError: shapes (16,) (16,12,10)``).
    """
    ny, nx, nz = 12, 5, 4
    a = 2.0
    zeros = np.zeros((ny, nx, nz))
    du_dy = a * np.ones_like(zeros)
    delta = np.linspace(0.2, 0.6, ny)
    y_plus = np.full(ny, 1.0e6)
    nut = smagorinsky_viscosity(du_dy, zeros, zeros, zeros, zeros, zeros,
                                zeros, zeros, zeros, delta, y_plus, cs=0.1)
    assert nut.shape == (ny, nx, nz)
    assert np.allclose(nut[:, 0, 0], (0.1 * delta) ** 2 * a, rtol=1e-6)


def test_reichardt_profile_matches_the_imported_dns_mean_profile():
    """The turbulent initial condition must sit on the DNS profile.

    Reference: ``docs/rans-turbulence-validation-matrix/dns/0180`` (Lee &
    Moser 2015, Re_tau = 182).  Only the initial condition is graded here;
    no solver output is involved.
    """
    from tensorfvm.verification.kmm1987 import load_kmm1987_reference
    dns = load_kmm1987_reference()
    mask = (dns["y_plus"] >= 3.0) & (dns["y_plus"] <= 150.0)
    law = reichardt_profile(dns["y_plus"][mask])
    relative = np.abs(law - dns["u_plus"][mask]) / dns["u_plus"][mask]
    assert relative.max() < 0.06, relative.max()
    assert abs(float(reichardt_profile(30.0)) - 13.5) < 0.2


def test_initial_condition_noise_is_friction_velocity_scaled():
    """The IC perturbation must not scale with the laminar peak (Re_tau/2).

    ``noise * profile.max()`` injected O(10 u_tau) cell-to-cell noise into
    y+ ~ 0.5 cells at Re_tau = 180: the y-CFL was 4.2 and the explicit scheme
    diverged within ten steps.  The guard is the CFL itself.
    """
    cfg = small(re_tau=180.0, nx=32, ny=64, nz=16, dt=1e-3, smagorinsky=0.0)
    solver = ChannelLES3D(cfg, seed=7).initialise(noise=0.1, profile="reichardt")
    mean = reichardt_profile(solver.y_plus)[:, None, None] * np.ones(solver.u.shape)
    assert np.abs(solver.u - mean).max() < 1.0      # u_tau units, not 10 u_tau
    assert solver.convective_cfl() < 0.5
    quiet = ChannelLES3D(cfg, seed=7).initialise(noise=0.0, profile="reichardt")
    assert np.allclose(quiet.u.mean(axis=(1, 2)), reichardt_profile(quiet.y_plus))
    with pytest.raises(ValueError):
        ChannelLES3D(cfg).initialise(noise=0.1, profile="blasius")

def test_statistics_integrates_the_bulk_velocity_correctly():
    """Wall units and the bulk average are defined by the same normalisation."""
    cfg = small(re_tau=1.0, nx=8, ny=16, nz=8, smagorinsky=0.0)
    solver = ChannelLES3D(cfg)
    y = solver.yc[:, None, None]
    solver.u = (y * (2.0 - y)) * np.ones_like(solver.u)     # u_bulk = 2/3 for this shape
    solver.initialise(noise=0.0)                            # resets to the parabola
    solver.u = (y * (2.0 - y)) * np.ones_like(solver.u)
    solver.sample()
    stats = solver.statistics()
    # trapezoidal quadrature on the stretched grid: O(h^2) from the exact 2/3
    assert abs(solver.bulk_velocity() - 2.0 / 3.0) < 1e-2
    assert abs(stats['u_bulk_plus'] - 2.0 / 3.0) < 1e-2