"""Tests for the implicit wall-normal diffusion and the time integrator of the
3-D channel LES (:mod:`tensorfvm.channel_les3d`).

Why these tests exist
---------------------
The production route of ``CASE-FVM-CHANNEL-KMM180`` (3-D LES at
``Re_tau = 180``, ``64 x 64 x 32``) could not hold a solution: the committed
solver diverged after ~10 steps.  Two diagnoses had been recorded elsewhere and
both are exercised here, so that neither can be re-asserted without evidence:

* "the wall-normal diffusion is stiff (``nu_eff dt/dy^2 ~ 2-7`` on the wall
  row), so the explicit update is unstable" -- true, and fixed by
  :meth:`ChannelLES3D._thomas`; the fix alone does *not* stabilise the run;
* "the divergence comes from the ``nu_t`` feedback" -- falsified: the
  divergence survives with ``smagorinsky = 0`` (see
  ``test_divergence_is_independent_of_the_sgs_model``).

Each test below asserts an invariant (an operator identity, an analytic
solution, a stability bound or a measured trend), never merely that the code
runs.
"""
import math

import numpy as np
import pytest

from tensorfvm.channel_les3d import ChannelLES3D, ChannelLESConfig


def small(**kwargs):
    base = dict(re_tau=180.0, nx=16, ny=24, nz=10, lx=2.0 * math.pi, lz=math.pi,
                smagorinsky=0.0, dt=2.0e-3)
    base.update(kwargs)
    return ChannelLESConfig(**base)


# --------------------------------------------------------------------------- #
# 1. The implicit operator is the explicit stencil, not a second discretisation
# --------------------------------------------------------------------------- #
def test_wall_normal_operator_matches_the_explicit_stencil():
    """Applying ``M_y`` must equal the ``y`` part of ``diffusion()`` exactly.

    A field that varies only in ``y`` has zero ``x``/``z`` flux, so
    ``diffusion()`` returns exactly the wall-normal operator; comparing it with
    ``_y_apply`` catches a wrong sign, a missing wall term, a ``dy``/``dy_cc``
    swap or an index slip, for the cell-centred (``u``/``w``) and the face
    (``v``) placements alike.
    """
    cfg = small()
    solver = ChannelLES3D(cfg)
    rng = np.random.default_rng(3)
    nut = 0.3 * rng.random((cfg.ny, cfg.nx, cfg.nz))
    cell, face = solver._y_coefficients(solver.nu + nut)

    profile = rng.standard_normal(cfg.ny)
    solver.u = np.repeat(profile[:, None, None], cfg.nx, axis=1).repeat(cfg.nz, axis=2)
    solver.w = solver.u.copy()
    solver.v = np.zeros((cfg.ny + 1, cfg.nx, cfg.nz))
    solver.v[1:-1] = profile[:-1][:, None, None]
    lap_u, lap_v, lap_w = solver.diffusion(nut)

    assert np.abs(lap_u - solver._y_apply(cell, solver.u)).max() < 1e-12
    assert np.abs(lap_w - solver._y_apply(cell, solver.w)).max() < 1e-12
    assert np.abs(lap_v - solver._y_apply(face, solver.v[1:-1])).max() < 1e-12


def test_thomas_solves_the_same_system_as_a_dense_solver():
    """``_thomas`` must reproduce ``numpy.linalg.solve`` on ``(I - dt M)``."""
    cfg = small(ny=12)
    solver = ChannelLES3D(cfg)
    rng = np.random.default_rng(5)
    nut = rng.random((cfg.ny, cfg.nx, cfg.nz))
    cell, _ = solver._y_coefficients(solver.nu + nut)
    lower, diag, upper = cell
    dt = 3.0e-3
    rhs = rng.standard_normal((cfg.ny, cfg.nx, cfg.nz))
    x = solver._thomas(cell, rhs, dt)

    for j, k in ((0, 0), (3, 7), (cfg.nx - 1, cfg.nz - 1)):
        matrix = np.diag(1.0 - dt * diag[:, j, k])
        matrix += np.diag(-dt * lower[1:, j, k], -1)
        matrix += np.diag(-dt * upper[:-1, j, k], 1)
        assert np.allclose(matrix @ x[:, j, k], rhs[:, j, k], atol=1e-10)


def test_implicit_update_is_unconditionally_stable_where_explicit_is_not():
    """A diffusion number of 10 must still be integrated, and integrated right.

    The wall row of the production grid carries ``nu_eff dt/dy^2 ~ 2-7``; the
    explicit update amplifies such a mode instead of damping it.  The test uses
    the highest wall-normal wavenumber (a sign-alternating field), which is the
    mode the explicit scheme is unstable for: the implicit solve must damp it
    for *any* ``dt``, and must equal the exact inverse ``(I - dt M)^-1``.
    """
    cfg = small(ny=64, nx=8, nz=8, smagorinsky=0.0, dt=1.0e-2)
    solver = ChannelLES3D(cfg)
    shape = (cfg.ny, cfg.nx, cfg.nz)
    nut = np.zeros(shape)                           # molecular stiff: dt nu/dy^2 ~ 7
    cell, _ = solver._y_coefficients(solver.nu + nut)
    alternating = ((-1.0) ** np.arange(cfg.ny))[:, None, None] * np.ones(shape)
    explicit = alternating + cfg.dt * solver._y_apply(cell, alternating)
    implicit = solver._thomas(cell, alternating, cfg.dt)
    assert np.abs(implicit).max() < np.abs(alternating).max()
    assert np.abs(explicit).max() > np.abs(alternating).max()

    # the same statement against a dense solve, for a random field
    rng = np.random.default_rng(7)
    field = rng.standard_normal(shape)
    got = solver._thomas(cell, field, cfg.dt)
    lower, diag, upper = cell
    j, k = 2, 3
    matrix = np.diag(1.0 - cfg.dt * diag[:, j, k])
    matrix += np.diag(-cfg.dt * lower[1:, j, k], -1)
    matrix += np.diag(-cfg.dt * upper[:-1, j, k], 1)
    assert np.allclose(matrix @ got[:, j, k], field[:, j, k], atol=1e-10)


# --------------------------------------------------------------------------- #
# 2. Geometry conventions that the SGS model and the initial profile depend on
# --------------------------------------------------------------------------- #
def test_wall_distance_is_measured_to_the_nearest_wall():
    """``y+`` must be symmetric about the channel centre.

    The grid spans the full channel ``y in [0, 2]``; measuring the wall distance
    as ``y`` (the historical definition) gave ``y+ = 2 Re_tau`` at the top wall,
    switching van Driest damping off in the upper half and initialising the
    mean profile from the half-channel law evaluated at twice its argument.
    """
    solver = ChannelLES3D(small(ny=32))
    assert solver.y_plus[0] < 2.0 and solver.y_plus[-1] < 2.0
    assert np.allclose(solver.y_plus, solver.y_plus[::-1])
    assert abs(solver.y_plus[-1] - solver.config.re_tau * (2.0 - solver.yc[-1])) < 1e-9
    # the initial profile is therefore symmetric, not monotone over the channel
    solver.initialise(profile='reichardt', noise=0.0)
    mean = solver.u.mean(axis=(1, 2))
    assert np.allclose(mean, mean[::-1], atol=1e-10)
    assert mean.argmax() in (solver.config.ny // 2 - 1, solver.config.ny // 2)


def test_second_order_wall_shear_is_exact_on_the_laminar_parabola():
    """Both wall-shear readings must recover the analytic slope of the parabola."""
    cfg = small(re_tau=40.0, smagorinsky=0.0)
    solver = ChannelLES3D(cfg)
    y = solver.yc
    solver.u = (solver.body_force * y * (2.0 - y) / (2.0 * solver.nu))[:, None, None] \
        * np.ones((cfg.ny, cfg.nx, cfg.nz))
    analytic = solver.body_force * 1.0          # tau_w = G * delta at the wall
    assert abs(solver.wall_shear('second_order') / analytic - 1.0) < 1e-6
    # the half-cell reading is what the operator uses and is only first order
    assert abs(solver.wall_shear('half_cell') / analytic - 1.0) > 1e-3


# --------------------------------------------------------------------------- #
# 3. Integrator: consistency, and what it does *not* fix
# --------------------------------------------------------------------------- #
def test_rk3_and_euler_agree_in_the_small_time_step_limit():
    """Both integrators approximate the same ODE: the difference vanishes as ``dt``."""
    diffs = []
    for dt in (2.0e-3, 5.0e-4):
        states = []
        for scheme in ('euler', 'rk3'):
            solver = ChannelLES3D(small(dt=dt, smagorinsky=0.05, time_scheme=scheme))
            solver.initialise(profile='reichardt', noise=0.05)
            solver.step()
            states.append(solver.u.copy())
        diffs.append(np.abs(states[0] - states[1]).max())
    assert diffs[1] < diffs[0]
    assert diffs[1] / diffs[0] < 0.5


def test_projection_keeps_the_field_divergence_free_for_any_time_step():
    """The projection is exact: the time step and the integrator cannot leak mass."""
    for dt, scheme in ((2.0e-3, 'rk3'), (1.0e-2, 'rk3'), (1.0e-2, 'euler')):
        solver = ChannelLES3D(small(dt=dt, smagorinsky=0.1, time_scheme=scheme))
        solver.initialise(profile='reichardt', noise=0.1)
        for _ in range(5):
            solver.step()
        assert np.abs(solver.divergence()).max() < 1e-10


def _steps_until_divergence(**kwargs):
    solver = ChannelLES3D(small(nx=16, ny=16, nz=8, **kwargs))
    solver.initialise(profile='reichardt', noise=0.15)
    for _ in range(4000):
        try:
            solver.step()
        except FloatingPointError:
            return solver.steps + 1
    return None


def test_divergence_is_independent_of_the_sgs_model():
    """``Cs = 0`` diverges too -- the ``nu_t`` feedback is not the cause.

    Both runs are started from the same state and differ only in the SGS
    constant.  If the instability were driven by the eddy viscosity, the laminar
    (``Cs = 0``) run would have to stay finite; it does not.
    """
    with_sgs = _steps_until_divergence(smagorinsky=0.1, dt=2.0e-3)
    without_sgs = _steps_until_divergence(smagorinsky=0.0, dt=2.0e-3)
    assert with_sgs is not None and without_sgs is not None
    assert with_sgs < 2 * without_sgs


def test_divergence_time_scales_with_the_time_step():
    """A per-step amplification, not a per-unit-time one: halving ``dt`` doubles the steps.

    This is the signature that separates a discrete amplification factor
    ``|G| = 1 + O(dt)`` from a physical growth rate, and it is the reason the
    remaining divergence cannot be removed by lowering ``dt`` (a documented
    open item of the anchor, see ``docs/case-fvm-channel-kmm180-les3d-stability``).
    """
    coarse = _steps_until_divergence(dt=4.0e-3)
    fine = _steps_until_divergence(dt=2.0e-3)
    assert coarse is not None and fine is not None
    assert 1.5 < fine / coarse < 3.0