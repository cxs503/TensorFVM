"""Gates for :mod:`tensorfvm.mixing_layer` (temporal mixing layer case).

The published bundle lives in ``docs/repro-mixing-layer-temporal/`` and is produced by
``scripts/verify_mixing_layer.py``.  These tests pin the four independent checks that
make that bundle trustworthy:

* the discrete projection is exact (divergence-free to round-off) and the free-slip
  base flow reproduces the exact heat-kernel solution of the same initial profile;
* the viscous (Orr--Sommerfeld) solve reproduces the inviscid (Rayleigh) eigenvalue as
  Re grows, and the Rayleigh solve is grid-converged -- i.e. the reference is a solved
  eigenproblem, not a transcribed number;
* the simulation's linear growth rate matches the Orr--Sommerfeld eigenvalue of the
  measured instantaneous layer within the declared 10% band;
* the momentum-thickness diagnostic returns the analytic value 0.5 delta for the
  ``tanh`` profile it is defined on.
"""
import math

import numpy as np
import pytest

from tensorfvm.mixing_layer import (
    MixingLayerConfig,
    TemporalMixingLayer,
    heat_kernel_profile,
    growth_rate_fit,
    layer_metrics,
    orr_sommerfeld_modes,
    rayleigh_modes,
)


def _solver(nx=64, ny=192, alpha=0.5, visc=2e-3):
    return TemporalMixingLayer(MixingLayerConfig(nx=nx, ny=ny, wavelength=2 * math.pi / alpha,
                                                 viscosity=visc))


def test_config_rejects_short_domain():
    with pytest.raises(ValueError):
        MixingLayerConfig(height=3.0)


def test_momentum_thickness_of_tanh_profile():
    """theta = int (1/4 - (u/dU)^2) dy = delta/2 for u = U tanh(y/delta), dU = 2U."""
    s = _solver()
    metrics = layer_metrics(s.base_profile(s.yc), s.yc, s.dy)
    assert metrics['momentum_thickness'] == pytest.approx(0.5, rel=1e-3)
    assert metrics['vorticity_thickness'] == pytest.approx(2.0, rel=5e-3)  # FD gradient on the cell centres


def test_projection_is_exact_and_base_flow_is_steady():
    s = _solver()
    for _ in range(4):
        s.step(0.02)
    divergence = s.divergence(s.u, s.v)
    assert np.abs(divergence).max() < 1e-12
    # the undisturbed two-dimensional base flow is an exact solution of the heat
    # equation, so a run without a perturbation must follow the heat kernel
    s2 = _solver()
    for _ in range(10):
        s2.step(0.02)
    exact = heat_kernel_profile(s2.yc, s2.time)
    assert np.abs(s2.mean_profile() - exact).max() < 2e-4 * 0.02


def test_rayleigh_grid_convergence_and_inviscid_limit():
    coarse = rayleigh_modes(0.5, nodes=301, keep=1)[0].growth
    fine = rayleigh_modes(0.5, nodes=601, keep=1)[0].growth
    assert abs(fine / coarse - 1) < 5e-3
    viscous = orr_sommerfeld_modes(0.5, 1.0e5, nodes=301, keep=1)[0].growth
    assert abs(viscous / fine - 1) < 1e-2
    # viscous damping must slow the mode, not accelerate it
    damped = orr_sommerfeld_modes(0.5, 500.0, nodes=301, keep=1)[0].growth
    assert 0 < damped < fine


def test_eigenfunction_seed_shape():
    s = _solver(nx=32, ny=96)
    mode = rayleigh_modes(0.5, nodes=301, keep=1)[0]
    with pytest.raises(ValueError):
        s.seed_mode(1e-5, 0.5, np.zeros(3), np.zeros(3))
    s.seed_stability_mode(1e-5, mode)
    assert np.abs(s.u - s.base_profile(s.yc)[None, :]).max() > 0
    assert np.abs(s.v).max() > 0


def test_linear_growth_matches_orr_sommerfeld():
    """Simulated growth rate vs the eigen-solve of the measured instantaneous layer."""
    alpha = 0.5
    mode = rayleigh_modes(alpha, nodes=301, keep=1)[0]
    s = _solver(nx=64, ny=192, alpha=alpha, visc=2e-3)
    s.seed_stability_mode(1e-5, mode)
    horizon = 6.0
    history = []
    while s.time < horizon - 1e-12:
        s.step(min(s.stable_time_step(), horizon - s.time))
        history.append(s.diagnostics())
    times = [h['time'] for h in history]
    amplitude = [h['fundamental_amplitude'] for h in history]
    rate, _ = growth_rate_fit(times, amplitude, (2.0, horizon))
    index = int(np.argmin(np.abs(np.array(times) - 4.0)))
    delta = history[index]['vorticity_thickness'] / 2
    reference = orr_sommerfeld_modes(alpha, delta / 2e-3, thickness=delta, nodes=301,
                                     keep=1)[0].growth
    assert abs(rate / reference - 1) < 0.10
    assert max(h['max_divergence'] for h in history) < 1e-12
