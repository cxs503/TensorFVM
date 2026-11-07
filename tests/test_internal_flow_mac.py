"""Unit tests of the staggered MAC solver and of its FFT cosine transforms."""

import math

import pytest
import torch

from tensorfvm.internal_flow._dct import dct2, dct_matrix, idct2
from tensorfvm.internal_flow.mac import MACConfig, StaggeredMAC


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 17, 64])
def test_fft_dct_matches_dense_matrix(n):
    """The FFT path must reproduce the documented DCT-II basis exactly."""
    torch.manual_seed(1234 + n)
    matrix = dct_matrix(n)
    x = torch.randn(n, dtype=torch.float64)
    forward = dct2(x[:, None], 0)[:, 0]
    inverse = idct2(x[:, None], 0)[:, 0]
    assert torch.allclose(forward, matrix @ x, atol=1e-12)
    assert torch.allclose(inverse, matrix.t() @ x, atol=1e-12)
    assert torch.allclose(idct2(forward[:, None], 0)[:, 0], x, atol=1e-12)


def test_fft_dct_two_dimensional_axes():
    torch.manual_seed(7)
    x = torch.randn(9, 13, dtype=torch.float64)
    matrix_x, matrix_y = dct_matrix(9), dct_matrix(13)
    assert torch.allclose(dct2(x, 0), matrix_x @ x, atol=1e-11)
    assert torch.allclose(dct2(x, 1), x @ matrix_y.t(), atol=1e-11)
    assert torch.allclose(idct2(x, 0), matrix_x.t() @ x, atol=1e-11)
    assert torch.allclose(idct2(x, 1), x @ matrix_y, atol=1e-11)


def developed_channel(cells_per_half_height=20, reynolds=10.0, length=4.0):
    """Straight channel with a fully developed parabolic inlet (an exact solution)."""
    height = 2.0
    ny = 2 * cells_per_half_height
    hy = height / ny
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * hy
    profile = tuple(float(1.5 * (1.0 - (value - 1.0) ** 2)) for value in y)
    config = MACConfig(
        nx=int(round(length / hy)), ny=ny, length=length, height=height,
        viscosity=height / reynolds, left="inlet", right="outlet",
        inlet_profile=profile, cfl=0.4, max_steps=4000, tolerance=1e-7,
        check_every=200)
    return StaggeredMAC(config), torch.as_tensor(profile)


def test_developed_channel_keeps_its_profile():
    """A parabolic profile entering a straight channel is a steady FV solution."""
    solver, profile = developed_channel()
    # the developed profile is an exact steady solution of these equations, so the
    # test starts from it and checks that the discrete operator leaves it alone
    solver.u.copy_(profile[None, :].expand_as(solver.u))
    solver.apply_boundary_conditions()
    ledger = solver.run(max_steps=1500, tolerance=1e-8)
    assert ledger["converged"]
    _, u = solver.sample_u_profile(2.0)
    assert float((u - profile).abs().max()) < 5e-3
    # mass balance and a discretely divergence-free field
    assert math.isclose(float(solver.u[0, :].sum()), float(solver.u[solver.nx, :].sum()),
                        rel_tol=1e-12)
    assert float(solver.divergence().abs().max()) < 1e-10
    # wall friction of the developed profile: C_f = 12 / Re with U_ref = 1
    friction = solver.bottom_wall_friction()
    analytic = 12.0 / 10.0
    assert abs(float(friction[solver.nx // 2]) - analytic) / analytic < 0.02


def test_closed_box_stays_at_rest():
    """A closed cavity with no forcing must not acquire motion."""
    config = MACConfig(nx=12, ny=12, length=1.0, height=1.0, viscosity=0.1,
                       max_steps=50, check_every=10)
    solver = StaggeredMAC(config)
    solver.run(max_steps=50, tolerance=0.0)
    assert float(solver.u.abs().max()) < 1e-14
    assert float(solver.v.abs().max()) < 1e-14


def test_prolongation_preserves_a_linear_profile():
    """Prolongation from a coarse mesh reproduces a resolved, smooth field."""
    coarse = StaggeredMAC(MACConfig(nx=10, ny=8, length=2.0, height=1.0, viscosity=0.1,
                                    max_steps=4))
    fine = StaggeredMAC(MACConfig(nx=20, ny=16, length=2.0, height=1.0, viscosity=0.1,
                                  max_steps=4))
    coarse.u.copy_(torch.linspace(0.0, 1.0, coarse.u.shape[0])[:, None].expand_as(coarse.u))
    fine.prolong_from(coarse)
    restored = torch.linspace(0.0, 1.0, fine.u.shape[0])[:, None].expand_as(fine.u)
    # interior nodes only: the closed-box boundary condition owns the end faces
    assert float((fine.u[1:-1] - restored[1:-1]).abs().max()) < 0.06
    assert float(fine.u[-1].abs().max()) == 0.0
    assert float(fine.u[0].abs().max()) == 0.0
    with pytest.raises(ValueError):
        other = StaggeredMAC(MACConfig(nx=20, ny=16, length=3.0, height=1.0, viscosity=0.1,
                                       max_steps=4))
        other.prolong_from(coarse)


def test_configuration_validation():
    base = dict(nx=8, ny=8, length=1.0, height=1.0, viscosity=0.1)
    with pytest.raises(ValueError):
        MACConfig(**base, left="inlet")                       # inlet without profile
    with pytest.raises(ValueError):
        MACConfig(**base, left="inlet", inlet_profile=(1.0,) * 7)
    with pytest.raises(ValueError):
        MACConfig(**base, right="pressure")
    with pytest.raises(ValueError):
        MACConfig(**base, energy=True)                        # energy without diffusivity
    with pytest.raises(ValueError):
        MACConfig(**dict(base, viscosity=-1.0))
    with pytest.raises(ValueError):
        MACConfig(**base, inlet_profile=(1.0,) * 8)           # profile without inlet