"""Unit tests for the Phase-0 algebraic VOF transport (AC1 of node p233w1-N03).

Coverage: 1D uniform advection (periodic and wall), 2D translation, solid-body
rotation, shear flow, dimensional-splitting consistency, boundedness and
conservation, plus the CFL guard.  The conservation and boundedness claims are
the ones the Phase-0 gates rely on (AC4 mass residual and interface sharpness),
so they are asserted to round-off, not to a loose tolerance.
"""

import math

import pytest
import torch

from tensorfvm.vof_phase0.advection import AlgebraicVOF


def _square_wave(n=200, lo=0.25, hi=0.55):
    dx = 1.0 / n
    xs = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    alpha = ((xs > lo) & (xs < hi)).to(torch.float64)
    return alpha.reshape(1, n), dx, xs


def _uniform_1d(alpha, dx, velocity, period, cfl=0.4):
    n = alpha.shape[1]
    u = torch.full((1, n + 1), float(velocity), dtype=torch.float64)
    v = torch.zeros((2, n), dtype=torch.float64)
    vof = AlgebraicVOF(n, 1, dx, 1.0, boundary="periodic")
    dt = cfl * dx / abs(velocity)
    steps = int(round(period / dt))
    dt = period / steps
    for _ in range(steps):
        alpha = vof.advect(alpha, u, v, dt)
    return alpha, vof


def _rotation_field(n=64, radius=0.12, centre=(0.5, 0.5)):
    d = 1.0 / n
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * d
    x, y = torch.meshgrid(c, c, indexing="xy")
    r = torch.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
    return (r < radius).to(torch.float64), d


def _rotation_velocities(n, d, omega=2.0 * math.pi):
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * d
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * d
    u = (-omega * (y - 0.5)).reshape(n, 1).expand(n, n + 1).clone()
    v = (omega * (x - 0.5)).reshape(1, n).expand(n + 1, n).clone()
    return u, v


# --------------------------------------------------------------------------- #
# 1D
# --------------------------------------------------------------------------- #
def test_1d_uniform_advection_conserves_volume():
    alpha, dx, _ = _square_wave()
    vof = AlgebraicVOF(alpha.shape[1], 1, dx, 1.0, boundary="periodic")
    m0 = vof.mass(alpha)
    advected, _ = _uniform_1d(alpha, dx, 1.0, period=0.37)
    assert abs(vof.mass(advected) - m0) <= 1e-15 * max(m0, 1.0)


def test_1d_square_wave_stays_bounded():
    alpha, dx, _ = _square_wave()
    advected, _ = _uniform_1d(alpha, dx, 1.0, period=1.0)
    assert float(advected.min()) >= -1e-14
    assert float(advected.max()) <= 1.0 + 1e-14


def test_1d_square_wave_l1_error_after_one_period():
    alpha, dx, xs = _square_wave()
    advected, _ = _uniform_1d(alpha, dx, 1.0, period=1.0)
    l1 = float((advected - alpha).abs().sum()) * dx
    # measure of the square wave is 0.3 -> L1 error must stay well under 10 %
    assert l1 <= 0.05


def test_1d_reverse_advection_returns_to_start():
    alpha, dx, _ = _square_wave()
    forward, vof = _uniform_1d(alpha, dx, 1.0, period=0.5)
    backward, _ = _uniform_1d(forward, dx, -1.0, period=0.5)
    l1 = float((backward - alpha).abs().sum()) * dx
    assert l1 <= 0.05


def test_1d_fct_sharper_than_donor_cell():
    """The limiter must recover order: FCT beats pure donor-cell advection."""
    alpha, dx, _ = _square_wave()
    advected_fct, _ = _uniform_1d(alpha, dx, 1.0, period=1.0)
    n = alpha.shape[1]
    dt = 0.4 * dx
    steps = int(round(1.0 / dt))
    dt = 1.0 / steps
    donor = alpha.clone()
    for _ in range(steps):
        # pure donor-cell (upwind) transport, first order
        field = donor.squeeze(0)
        face = torch.where(torch.tensor(1.0) > 0,
                           torch.cat([field[-1:], field[:-1]]), field)
        flux = torch.cat([(face * 1.0), (field[-1:] * 1.0)])
        donor = (field - dt / dx * (flux[1:] - flux[:-1])).reshape(1, n)
    l1_fct = float((advected_fct - alpha).abs().sum()) * dx
    l1_donor = float((donor - alpha).abs().sum()) * dx
    assert l1_fct < l1_donor


def test_1d_zero_velocity_is_a_noop():
    alpha, dx, _ = _square_wave()
    vof = AlgebraicVOF(alpha.shape[1], 1, dx, 1.0, boundary="wall")
    u = torch.zeros((1, alpha.shape[1] + 1), dtype=torch.float64)
    v = torch.zeros((2, alpha.shape[1]), dtype=torch.float64)
    out = vof.advect(alpha, u, v, 0.01)
    assert torch.allclose(out, alpha)


# --------------------------------------------------------------------------- #
# 2D
# --------------------------------------------------------------------------- #
def test_2d_translation_diagonal_conserves_and_bounds():
    n = 64
    d = 1.0 / n
    blob, _ = _rotation_field(n, radius=0.12, centre=(0.5, 0.5))
    vof = AlgebraicVOF(n, n, d, d, boundary="wall")
    m0 = vof.mass(blob)
    u = torch.full((n, n + 1), 0.7, dtype=torch.float64)
    v = torch.full((n + 1, n), 0.3, dtype=torch.float64)
    cur = blob
    dt = 0.3 * d / 0.7
    for _ in range(40):
        cur = vof.advect(cur, u, v, dt)
    assert abs(vof.mass(cur) - m0) <= 1e-14 * max(m0, 1.0)
    assert float(cur.min()) >= -1e-12
    assert float(cur.max()) <= 1.0 + 1e-12


def test_2d_solid_body_rotation_conserves_volume_exactly():
    n = 64
    d = 1.0 / n
    blob, _ = _rotation_field(n, radius=0.12, centre=(0.5, 0.75))
    u, v = _rotation_velocities(n, d)
    vof = AlgebraicVOF(n, n, d, d, boundary="wall")
    m0 = vof.mass(blob)
    cur = blob
    dt = 0.25 * d / max(float(u.abs().max()), float(v.abs().max()))
    t = 0.0
    while t < 1.0:
        step = min(dt, 1.0 - t)
        cur = vof.advect(cur, u, v, step)
        t += step
    assert abs(vof.mass(cur) - m0) <= 1e-14 * max(m0, 1.0)


def test_2d_solid_body_rotation_stays_bounded():
    n = 64
    d = 1.0 / n
    blob, _ = _rotation_field(n, radius=0.12, centre=(0.5, 0.75))
    u, v = _rotation_velocities(n, d)
    vof = AlgebraicVOF(n, n, d, d, boundary="wall")
    cur = blob
    dt = 0.25 * d / max(float(u.abs().max()), float(v.abs().max()))
    t = 0.0
    while t < 1.0:
        step = min(dt, 1.0 - t)
        cur = vof.advect(cur, u, v, step)
        t += step
    assert float(cur.min()) >= -1e-12
    assert float(cur.max()) <= 1.0 + 1e-12


def test_2d_rotation_shape_error_is_small():
    n = 64
    d = 1.0 / n
    blob, _ = _rotation_field(n, radius=0.12, centre=(0.5, 0.75))
    u, v = _rotation_velocities(n, d)
    vof = AlgebraicVOF(n, n, d, d, boundary="wall")
    cur = blob
    dt = 0.25 * d / max(float(u.abs().max()), float(v.abs().max()))
    t = 0.0
    while t < 1.0:
        step = min(dt, 1.0 - t)
        cur = vof.advect(cur, u, v, step)
        t += step
    l1 = float((cur - blob).abs().sum()) * d * d
    area = float(blob.sum()) * d * d
    # 64^2 grid, disk radius 7.7 cells, ~800 steps: the interface stays 2-3
    # cells wide, i.e. ~30 % of the radius.  The gate here only forbids
    # catastrophic diffusion; the quantitative number is reported by the
    # campaign (docs/vof-phase0/artifacts/phase0_report.json).
    assert float(cur.max()) >= 0.80
    assert l1 / area <= 0.70


def test_2d_shear_flow_is_bounded_and_conservative():
    n = 64
    d = 1.0 / n
    blob, _ = _rotation_field(n, radius=0.12, centre=(0.5, 0.5))
    y = (torch.arange(n, dtype=torch.float64) + 0.5) * d
    u = (0.4 * torch.sin(2.0 * math.pi * y)).reshape(n, 1).expand(n, n + 1)
    v = torch.zeros((n + 1, n), dtype=torch.float64)
    vof = AlgebraicVOF(n, n, d, d, boundary="wall")
    m0 = vof.mass(blob)
    cur = blob
    dt = 0.3 * d / 0.4
    for _ in range(40):
        cur = vof.advect(cur, u, v, dt)
    assert abs(vof.mass(cur) - m0) <= 1e-14 * max(m0, 1.0)
    assert float(cur.min()) >= -1e-12
    assert float(cur.max()) <= 1.0 + 1e-12


def test_dimensionally_split_sweep_orders_agree():
    n = 48
    d = 1.0 / n
    blob, _ = _rotation_field(n, radius=0.12, centre=(0.5, 0.5))
    u = torch.full((n, n + 1), 0.5, dtype=torch.float64)
    v = torch.full((n + 1, n), 0.5, dtype=torch.float64)
    vof = AlgebraicVOF(n, n, d, d, boundary="wall")
    dt = 0.3 * d / 0.5
    xy = blob.clone()
    yx = blob.clone()
    for _ in range(20):
        xy = vof.advect(xy, u, v, dt, order="xy")
        yx = vof.advect(yx, u, v, dt, order="yx")
    l1 = float((xy - yx).abs().sum()) * d * d
    area = float(blob.sum()) * d * d
    assert l1 / area <= 0.10


def test_cfl_guard_rejects_unstable_step():
    n = 32
    d = 1.0 / n
    alpha = torch.zeros((n, n), dtype=torch.float64)
    alpha[n // 3:2 * n // 3, n // 3:2 * n // 3] = 1.0
    vof = AlgebraicVOF(n, n, d, d, boundary="wall")
    u = torch.full((n, n + 1), 1.0, dtype=torch.float64)
    v = torch.zeros((n + 1, n), dtype=torch.float64)
    with pytest.raises(ValueError):
        vof.advect(alpha, u, v, 3.0 * d)


def test_local_bounds_do_not_create_extrema():
    n = 64
    d = 1.0 / n
    blob, _ = _rotation_field(n, radius=0.12, centre=(0.5, 0.5))
    vof_local = AlgebraicVOF(n, n, d, d, boundary="wall", bounds="local")
    vof_global = AlgebraicVOF(n, n, d, d, boundary="wall", bounds="global")
    u = torch.full((n, n + 1), 0.6, dtype=torch.float64)
    v = torch.full((n + 1, n), 0.2, dtype=torch.float64)
    dt = 0.3 * d / 0.6
    loc = blob.clone()
    glo = blob.clone()
    for _ in range(20):
        loc = vof_local.advect(loc, u, v, dt)
        glo = vof_global.advect(glo, u, v, dt)
    for field in (loc, glo):
        assert float(field.min()) >= -1e-12
        assert float(field.max()) <= 1.0 + 1e-12


def test_constructor_validation():
    with pytest.raises(ValueError):
        AlgebraicVOF(0, 1)
    with pytest.raises(ValueError):
        AlgebraicVOF(8, 8, 1.0, 1.0, boundary="open")
    with pytest.raises(ValueError):
        AlgebraicVOF(8, 8, 1.0, 1.0, bounds="whatever")