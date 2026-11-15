"""Unit tests for the closure-free channel turbulence seed (CASE-FVM-CHANNEL-KMM180).

The seed itself is an initial condition, so the tests assert *invariants* rather
than a trajectory: the injected field carries no dilatation of its own, it is
inside the solver's amplitude scale, it is reproducible, and the admission gates
accept a turbulent state while rejecting a laminar one.  Nothing here claims the
seeded run stays turbulent — that is what the long run reports.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfvm.verification.channel_spectral_dns import (
    ChannelConfig,
    SpectralChannel,
    StatisticsAccumulator,
)
from tensorfvm.verification.channel_turbulent_seed import (
    add_streamwise_rolls,
    bulk_velocity_plus,
    flow_throughs,
    roll_shape,
    seed_turbulent_state,
    turbulence_gates,
)

NX, NY, NZ = 16, 32, 16


def make_solver(**kwargs) -> SpectralChannel:
    config = ChannelConfig(re_tau=180.0, nx=NX, ny=NY, nz=NZ,
                           length_x=2.0 * math.pi, length_z=math.pi,
                           **kwargs)
    return SpectralChannel(config)


# --------------------------------------------------------------------------- #
# Shape function
# --------------------------------------------------------------------------- #
def test_roll_shape_vanishes_at_both_walls_and_peaks_at_centre():
    hw = 1.0
    Y = np.linspace(0.0, 2.0 * hw, 129)
    f, df = roll_shape(Y, hw)
    assert abs(f[0]) < 1e-12 and abs(f[-1]) < 1e-12
    assert df[0] > 0.0 and df[-1] < 0.0
    i = int(np.argmax(np.abs(f)))
    assert abs(Y[i] - hw) < 0.1 * hw


# --------------------------------------------------------------------------- #
# The injected field
# --------------------------------------------------------------------------- #
def test_rolls_inject_no_dilatation_of_their_own():
    solver = make_solver()
    info = add_streamwise_rolls(solver, amplitude=1.5, spanwise_mode=2,
                                sinusity=0.0)
    scale = 1.5 * info['kz']
    # The (v, w) pair is divergence-free in (y, z) analytically; what is left is
    # the O(dy^2) discretisation of dv/dy on the wall-normal grid, three orders
    # below the injected amplitude*kz.
    assert info['divergence_residual'] < 1e-2 * scale
    assert solver.v[0].max() == 0.0 and solver.v[-1].min() == 0.0


def test_sinuous_streak_term_is_the_only_divergent_part_and_projection_removes_it():
    solver = make_solver()
    with_wobble = add_streamwise_rolls(solver, amplitude=1.5, spanwise_mode=2,
                                       sinusity=0.4)
    # u ~ sin(kx x) carries du/dx, so this term is *not* divergence-free by
    # design; the solver's projection is what makes the seeded state admissible.
    assert with_wobble['divergence_residual'] > 1e-2
    solver.project(dt=min(1e-3, solver.time_step))
    assert np.max(np.abs(solver.divergence())) < 1e-10


def test_rolls_are_counter_rotating_pairs_with_zero_plane_mean():
    solver = make_solver()
    before = solver.v.copy()
    add_streamwise_rolls(solver, amplitude=1.0, spanwise_mode=2, phase=0.0)
    added = solver.v - before
    # cos(kz z) with an integer mode over one period has zero spanwise mean.
    assert np.max(np.abs(added.mean(axis=2))) < 1e-12
    # The wall-normal velocity changes sign in z (a vortex pair).
    mid = NY // 2
    assert added[mid].max() > 0.0 > added[mid].min()


def test_seed_is_deterministic_and_divergence_free_after_projection():
    a, b = make_solver(), make_solver()
    info_a = seed_turbulent_state(a, amplitude=1.5, noise=0.2, seed=7)
    seed_turbulent_state(b, amplitude=1.5, noise=0.2, seed=7)
    assert info_a['divergence_after_projection'] < 1e-10
    assert np.allclose(a.u, b.u) and np.allclose(a.v, b.v) and np.allclose(a.w, b.w)


def test_seed_amplitude_is_in_wall_units():
    solver = make_solver()
    small = seed_turbulent_state(solver, amplitude=0.5, noise=0.0, seed=1)
    rms_small = small['v_rms']
    solver = make_solver()
    large = seed_turbulent_state(solver, amplitude=1.0, noise=0.0, seed=1)
    assert large['v_rms'] == pytest.approx(2.0 * rms_small, rel=1e-9)
    # A u_tau-scale seed: within an order of magnitude of 1 wall unit.
    assert 0.05 < small['v_rms'] < 1.0


def test_seed_rejects_invalid_parameters():
    solver = make_solver()
    with pytest.raises(ValueError):
        add_streamwise_rolls(solver, amplitude=-1.0)
    with pytest.raises(ValueError):
        add_streamwise_rolls(solver, amplitude=1.0, spanwise_mode=0)
    with pytest.raises(ValueError):
        add_streamwise_rolls(solver, amplitude=1.0, sinusity=1.5)


def test_seeded_state_stays_finite_over_a_short_run():
    solver = make_solver()
    seed_turbulent_state(solver, amplitude=1.5, noise=0.2, seed=3)
    acc = StatisticsAccumulator(solver)
    for _ in range(10):
        solver.step()
        acc.add()
    for field in (solver.u, solver.v, solver.w):
        assert np.all(np.isfinite(field))
    assert solver.divergence_max < 1e-8


# --------------------------------------------------------------------------- #
# Admission gates
# --------------------------------------------------------------------------- #
def _stats(y_plus, u_plus, uv_plus, du_plus, re_tau=180.0, time_span=10.0):
    return SimpleNamespace(y_plus=np.asarray(y_plus, float),
                           u_plus=np.asarray(u_plus, float),
                           uv_plus=np.asarray(uv_plus, float),
                           du_plus_dy_plus=np.asarray(du_plus, float),
                           re_tau=re_tau, time_span=time_span)


def test_gates_reject_the_laminar_state():
    y_plus = np.linspace(1.0, 180.0, 90)
    target = 1.0 - y_plus / 180.0
    laminar = _stats(y_plus, y_plus, np.zeros_like(y_plus), target)
    gates = turbulence_gates(laminar, length_x=2.0 * math.pi)
    assert gates['uv_share'] == 0.0
    assert gates['verdict']['C1'] is False
    assert gates['admissible'] is False


def test_gates_accept_a_balanced_turbulent_state():
    y_plus = np.linspace(1.0, 180.0, 90)
    target = 1.0 - y_plus / 180.0
    uv = -0.9 * target                 # Reynolds stress carries 90 % of the shear
    u_plus = 15.0 + np.log(np.maximum(y_plus, 1.0)) / 0.41
    turbulent = _stats(y_plus, u_plus, uv, target + uv, time_span=30.0)
    gates = turbulence_gates(turbulent, length_x=2.0 * math.pi)
    assert gates['uv_share'] == pytest.approx(0.9, rel=1e-6)
    assert gates['balance_residual'] == pytest.approx(0.0, abs=1e-12)
    assert gates['admissible'] is True


def test_gates_flag_a_momentum_imbalance_even_with_reynolds_stress():
    y_plus = np.linspace(1.0, 180.0, 90)
    target = 1.0 - y_plus / 180.0
    uv = -0.9 * target
    # Total shear short of the applied one by 0.7 u_tau^2 (the m20573 signature).
    broken = _stats(y_plus, 15.0 + np.log(np.maximum(y_plus, 1.0)) / 0.41,
                    uv, target + uv - 0.7, time_span=30.0)
    gates = turbulence_gates(broken, length_x=2.0 * math.pi)
    assert gates['verdict']['C2'] is False
    assert gates['admissible'] is False


def test_bulk_velocity_and_flow_through_bookkeeping():
    y_plus = np.linspace(0.0, 180.0, 181)
    laminar = _stats(y_plus, y_plus, np.zeros_like(y_plus), np.ones_like(y_plus),
                     time_span=1.0)
    assert bulk_velocity_plus(laminar) == pytest.approx(90.0, rel=1e-3)
    # u_tau = h = 1, Re_tau = 180: t+ = t * Re_tau.
    assert flow_throughs(laminar, length_x=math.pi) == pytest.approx(
        90.0 / math.pi, rel=1e-3)