"""Tests for the wall-bounded-span 3-D lid-driven cavity.

The module exists because the periodic-span solver in
:mod:`tensorfvm.lid_cavity_3d` deliberately excludes the end-wall geometry
("no reference table ... and no anchor here").  These tests pin the pieces that
make the new solver trustworthy *and* keep the defect it uncovered closed:

* free-slip end walls + span-invariant lid leave the span inert, so the 3-D
  solution must equal the 2-D solver to round-off;
* no-slip end walls must actually change the answer (negative control) and the
  end-wall faces must never leak;
* the projection stays divergence-free and the discrete solution keeps the
  mid-plane symmetry of the setup -- this is what caught the off-by-one cell in
  the inherited wall-bounded MUSCL branch (symmetry broke at 1e-3);
* a linear profile must be reconstructed *exactly* on the interior faces by the
  wall-bounded reconstruction (regression test for that off-by-one);
* a span-modulated lid excites ``w != 0`` and still preserves the mid-plane
  antisymmetry.
"""

import torch

import pytest

from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver
from tensorfvm.lid_cavity_3d import LidCavity3DSolver
from tensorfvm.lid_cavity_3d_endwalls import (
    LidCavity3DEndWallConfig,
    LidCavity3DEndWallSolver,
    score_endwall_against_ghia,
)


def _two_d(n=8, dt=0.01, steps=20):
    solver = LidCavitySolver(LidCavityConfig(n=n, reynolds=100.0, time_step=dt))
    for _ in range(steps):
        solver.step(dt)
    return solver


def _three_d(span_bc="no_slip", n=8, n_z=4, dt=0.01, steps=20, **kw):
    solver = LidCavity3DEndWallSolver(
        LidCavity3DEndWallConfig(n_x=n, n_z=n_z, reynolds=100.0, time_step=dt,
                                 span_bc=span_bc, **kw))
    for _ in range(steps):
        solver.step(dt)
    return solver


def test_config_rejects_bad_input():
    with pytest.raises(ValueError):
        LidCavity3DEndWallConfig(n_x=8, span_bc="slip")
    with pytest.raises(ValueError):
        LidCavity3DEndWallConfig(n_x=8, n_z=5)      # odd span -> no mid-plane face
    with pytest.raises(ValueError):
        LidCavity3DEndWallConfig(n_x=8, n_y=6)      # non-cubic cells
    with pytest.raises(TypeError):
        LidCavity3DEndWallSolver(LidCavityConfig(n=8))   # wrong config family


def test_free_slip_span_reduces_exactly_to_two_d():
    """The span is inert with free-slip walls: the operators must be identical."""
    two_d = _two_d()
    three_d = _three_d(span_bc="free_slip")
    assert float((three_d.u[:, :, 0] - two_d.u).abs().max()) <= 1e-12
    assert float((three_d.v[:, :, 0] - two_d.v).abs().max()) <= 1e-12
    assert float((three_d.p[:, :, 0] - two_d.p).abs().max()) <= 1e-11
    assert three_d.endwall_speed() <= 1e-14          # w stays at zero
    assert three_d.wall_face_leak() == 0.0


def test_no_slip_end_walls_change_the_solution():
    """Negative control: if no-slip did nothing, the BC was not applied."""
    two_d = _two_d()
    three_d = _three_d(span_bc="no_slip")
    delta = float((three_d.u[:, :, 0] - two_d.u).abs().max())
    assert delta > 1e-3, delta          # the free-slip reduction is at 1e-16
    assert three_d.endwall_speed() > 1e-6            # end walls drive w != 0


def test_projection_is_divergence_free_and_keeps_end_walls_closed():
    solver = _three_d(span_bc="no_slip", n=8, n_z=6, steps=30)
    assert float(solver.divergence().abs().max()) <= 1e-12
    assert solver.wall_face_leak() == 0.0


def test_midplane_symmetry_is_exact():
    """Setup and domain are symmetric about z = L_z/2 -> the discrete solution is."""
    solver = _three_d(span_bc="no_slip", n=8, n_z=8, steps=40)
    asym = solver.midplane_asymmetry()
    assert asym["u"] <= 1e-12
    assert asym["v"] <= 1e-12
    assert asym["w_midplane"] <= 1e-14


def test_wall_bounded_reconstruction_is_exact_on_a_linear_profile():
    """Regression test: the inherited wall branch was one whole cell off."""
    m, h = 8, 0.125
    f = ((torch.arange(m, dtype=torch.float64) + 0.5) * h).view(m, 1, 1)
    exact = (torch.arange(m + 1, dtype=torch.float64) * h).view(m + 1, 1, 1)
    for vel_value in (-1.0, 1.0):
        vel = torch.full((m + 1, 1, 1), vel_value, dtype=torch.float64)
        got = LidCavity3DEndWallSolver._wall_muscl(f, vel, axis=0)
        # faces 2..m-2 sit between two interior cells whose limited slope is exact;
        # face 1 / face m-1 are the boundary cells, where the ghost-extrapolated
        # slope collapses to one-sided (h/2 closure) -- that it is *not* a whole h
        # off is exactly what the parent fix repaired.
        assert float((got[2:m - 1] - exact[2:m - 1]).abs().max()) <= 1e-14
        assert float((got[1:m] - exact[1:m]).abs().max()) <= 0.5 * h + 1e-14
        # the same call on the (fixed) periodic-span solver
        got_parent = LidCavity3DSolver._muscl_faces(f, vel, axis=0, periodic=False)
        assert float((got_parent[2:m - 1] - exact[2:m - 1]).abs().max()) <= 1e-14
        assert float((got_parent[1:m] - exact[1:m]).abs().max()) <= 0.5 * h + 1e-14


def test_modulated_lid_excites_w_and_keeps_antisymmetry():
    solver = _three_d(span_bc="no_slip", n=8, n_z=8, steps=40,
                      lid_spanwise_amplitude=0.2, lid_spanwise_wavenumber=1)
    assert solver.endwall_speed() > 1e-3
    assert float(solver.divergence().abs().max()) <= 1e-12
    asym = solver.midplane_asymmetry()
    assert asym["u"] <= 1e-10 and asym["v"] <= 1e-10


def test_centrelines_and_scoring_contract():
    solver = _three_d(span_bc="no_slip", n=8, n_z=6, steps=10)
    y, u = solver.centreline_u()
    x, v = solver.centreline_v()
    assert y.shape == (solver.n_y,) and u.shape == (solver.n_y,)
    assert x.shape == (solver.n_x,) and v.shape == (solver.n_x,)
    block = score_endwall_against_ghia(solver, reynolds=100.0)
    assert "u" in block and "v" in block
    assert block["span_bc"] == "no_slip"
    assert block["wall_face_leak"] == 0.0
    with pytest.raises(KeyError):
        score_endwall_against_ghia(solver, reynolds=250.0)


def test_span_length_and_span_profile():
    solver = _three_d(span_bc="no_slip", n=8, n_z=12, dt=0.005, steps=5)
    assert abs(solver.span_length - 12 * solver.h) < 1e-15
    z, profile = solver.span_profile_u()
    assert z.shape == (solver.n_z,) and profile.shape == (solver.n_z,)
    # the end walls damp the tangential velocity: strongest at the mid-height lid side
    assert float(profile.abs().max()) > 0.0