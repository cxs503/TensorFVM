"""Verification tests for the 3-D lid-driven cavity (periodic span).

``scripts/verify_lid_cavity_3d.py`` used to import a module that did not exist
(``tensorfvm.lid_cavity_3d``), so the 3-D entry of the verification catalogue was
a dead link.  These tests pin the pieces that make the replacement trustworthy:

* the 3-D discretisation reduces **identically** to the verified 2-D solver
  (:mod:`tensorfvm.lid_cavity`) on a span-invariant state with ``w = 0`` -- the
  reason the Ghia tables can be used as an accuracy anchor at all,
* the projection stays divergence-free on a prism whose span mode is genuinely
  excited (``w != 0``), where the pure-Neumann DCT is combined with a periodic
  FFT and an index slip between divergence and gradient would hide,
* a span-invariant lid keeps ``w`` at exactly zero (discrete symmetry, not
  "small"),
* the spanwise response of a weakly modulated lid is linear in the modulation
  amplitude (self-consistency of the z-flux discretisation, the only part of the
  operator that has no 2-D counterpart),
* the end-to-end Ghia comparison at Re = 100 on a 32 x 32 x 2 prism lands inside
  the same band the 2-D solver is held to.
"""

import math

import pytest
import torch

from tensorfvm.lid_cavity import LidCavityConfig, LidCavitySolver
from tensorfvm.lid_cavity_3d import (
    LidCavity3DConfig,
    LidCavity3DSolver,
    score_against_ghia_3d,
)


def test_config_rejects_bad_input():
    with pytest.raises(ValueError):
        LidCavity3DConfig(n_x=16, n_y=12)            # non-cubic cells unsupported
    with pytest.raises(ValueError):
        LidCavity3DConfig(n_x=15)                    # odd -> no mid-plane grid line
    with pytest.raises(ValueError):
        LidCavity3DConfig(n_x=16, lid_spanwise_amplitude=1.0)
    with pytest.raises(ValueError):
        LidCavity3DConfig(n_x=16, lid_spanwise_wavenumber=0)
    with pytest.raises(ValueError):
        LidCavity3DConfig(n_x=16, cfl=-0.1)


def test_field_shapes_and_no_slip_walls():
    n, nz = 8, 3
    solver = LidCavity3DSolver(LidCavity3DConfig(n_x=n, n_z=nz, reynolds=100.0))
    assert solver.u.shape == (n + 1, n, nz)
    assert solver.v.shape == (n, n + 1, nz)
    assert solver.w.shape == (n, n, nz)
    assert solver.p.shape == (n, n, nz)
    assert float(solver.u[0].abs().max()) == 0.0
    assert float(solver.u[n].abs().max()) == 0.0
    assert float(solver.v[:, 0].abs().max()) == 0.0
    # the lid is a tangential condition on the ghost, not on a stored face
    assert float(solver.v[:, n].abs().max()) == 0.0


def test_two_d_reduction_matches_two_d_solver():
    """Span-invariant state + w = 0 must reproduce the 2-D solver bit-for-bit."""
    n, nz, re, dt, steps = 16, 3, 100.0, 0.02, 30
    two_d = LidCavitySolver(LidCavityConfig(n=n, reynolds=re, time_step=dt,
                                            tolerance=1e-30, max_steps=4))
    three_d = LidCavity3DSolver(LidCavity3DConfig(n_x=n, n_z=nz, reynolds=re,
                                                  time_step=dt, tolerance=1e-30,
                                                  max_steps=4))
    for _ in range(steps):
        two_d.step()
        three_d.step()

    for k in range(nz):                              # every span slice identical
        assert torch.allclose(three_d.u[:, :, k], two_d.u, atol=1e-12, rtol=0)
        assert torch.allclose(three_d.v[:, :, k], two_d.v, atol=1e-12, rtol=0)
    assert torch.allclose(three_d.p[:, :, 0], two_d.p, atol=1e-12, rtol=0)
    assert three_d.spanwise_uniformity() == 0.0      # exact, not approximately


def test_span_invariant_lid_keeps_spanwise_velocity_at_zero():
    solver = LidCavity3DSolver(LidCavity3DConfig(n_x=16, n_z=4, reynolds=100.0,
                                                 time_step=0.005, max_steps=4))
    for _ in range(40):
        solver.step()
    assert solver.spanwise_uniformity() == 0.0
    assert float(solver.divergence().abs().max()) < 1e-12


def test_spanwise_modulated_lid_drives_spanwise_flow():
    solver = LidCavity3DSolver(LidCavity3DConfig(n_x=16, n_z=4, reynolds=100.0,
                                                 time_step=0.002, max_steps=4,
                                                 lid_spanwise_amplitude=0.05))
    for _ in range(60):
        solver.step()
    # genuinely three-dimensional: the spanwise velocity is not zero ...
    assert solver.spanwise_uniformity() > 1e-3
    # ... and the projection still annihilates the discrete divergence exactly
    assert float(solver.divergence().abs().max()) < 1e-12
    # the lid field itself has the imposed modulation, sampled at cell centres
    lid = solver.lid_velocity_field()
    node = math.cos(math.pi / solver.n_z)          # cell-centre sampling factor
    assert float(lid.max() - lid.min()) == pytest.approx(2 * 0.05 * node, rel=1e-6)


def test_spanwise_response_scales_linearly_with_lid_amplitude():
    """The z-flux operator must respond linearly to a weak spanwise modulation."""
    def run(amplitude: float) -> float:
        solver = LidCavity3DSolver(LidCavity3DConfig(
            n_x=16, n_z=4, reynolds=100.0, time_step=0.002, max_steps=4,
            lid_spanwise_amplitude=amplitude))
        for _ in range(100):
            solver.step()
        return solver.spanwise_uniformity()

    w_small, w_half = run(0.02), run(0.01)
    ratio = w_small / w_half
    assert 1.95 < ratio < 2.05, (w_small, w_half, ratio)


def test_ghia_reduction_re100_matches_two_d_accuracy():
    """Converged 32 x 32 x 2 prism scored against the 2-D Ghia tables."""
    solver = LidCavity3DSolver(LidCavity3DConfig(n_x=32, n_z=2, reynolds=100.0,
                                                 tolerance=1e-7, max_steps=20_000,
                                                 check_every=250))
    info = solver.run()
    assert info["converged"], info["history"][-3:]
    assert info["max_divergence"] < 1e-12

    result = score_against_ghia_3d(solver)
    assert result["u"]["max_abs_error"] < 0.03, result["u"]["max_abs_error"]
    assert result["v"]["max_abs_error"] < 0.03, result["v"]["max_abs_error"]
    assert result["spanwise_max"] == 0.0