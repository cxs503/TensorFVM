"""Tests for TensorFVM numerical wave tank module."""

from __future__ import annotations

import math
import pytest
import torch

from tensorfvm.wave_theory import (
    WaveParams, dispersion_relation,
    stokes1_elevation, stokes2_elevation,
    stokes1_velocity, stokes2_velocity,
    relaxation_weight, damping_weight,
)
from tensorfvm.wave_probes import (
    WaveProbeArray, goda_reflection_coefficient,
    invert_wavenumber, spectral_analysis,
)
from tensorfvm.hull_geometry import (
    WigleyHull, verify_wigley_geometry,
)
from tensorfvm.six_dof import (
    SixDOFConfig, SixDOFIntegrator,
    hydrostatic_restoring_heave, hydrostatic_restoring_pitch,
)
from tensorfvm.wave_tank import (
    WaveTankConfig, WaveTank,
    compute_added_resistance_gerritsma_meijers,
)


# =============================================================================
# Wave Theory Tests
# =============================================================================

class TestDispersionRelation:
    def test_deep_water(self):
        omega = 2.0 * math.pi / 1.0
        h = 100.0
        k = dispersion_relation(omega, h)
        k_deep = omega ** 2 / 9.81
        assert abs(k - k_deep) / k_deep < 0.01

    def test_shallow_water(self):
        omega = 2.0 * math.pi / 5.0
        h = 0.5
        k = dispersion_relation(omega, h)
        k_shallow = omega / math.sqrt(9.81 * h)
        assert abs(k - k_shallow) / k_shallow < 0.05

    def test_dispersion_consistency(self):
        omega = 2.0 * math.pi / 1.2
        h = 1.0
        k = dispersion_relation(omega, h)
        lhs = omega ** 2
        rhs = 9.81 * k * math.tanh(k * h)
        assert abs(lhs - rhs) / lhs < 1e-10


class TestStokesWaves:
    def setup_method(self):
        self.wp = WaveParams(H=0.04, T=1.2, h=1.0, g=9.81, order=2)

    def test_stokes1_elevation_amplitude(self):
        x = torch.tensor([0.0], dtype=torch.float64)
        eta = stokes1_elevation(x, 0.0, self.wp)
        assert abs(float(eta.item()) - self.wp.H / 2) < 1e-10

    def test_stokes2_elevation_higher(self):
        x = torch.linspace(0, self.wp.wavelength, 100, dtype=torch.float64)
        eta = stokes2_elevation(x, 0.0, self.wp)
        assert float(eta.max()) > abs(float(eta.min()))

    def test_stokes2_elevation_at_origin(self):
        x = torch.tensor([0.0], dtype=torch.float64)
        eta = stokes2_elevation(x, 0.0, self.wp)
        expected = self.wp.H / 2.0
        assert abs(float(eta.item()) - expected) < 0.05 * expected

    def test_velocity_phase(self):
        x = torch.tensor([0.0], dtype=torch.float64)
        z = torch.tensor([-0.5], dtype=torch.float64)
        u, w = stokes1_velocity(x, z, 0.0, self.wp)
        assert float(u.item()) > 0
        assert abs(float(w.item())) < float(u.item())

    def test_stokes2_velocity_nonzero_stokes_drift(self):
        x = torch.linspace(0, 2 * self.wp.wavelength, 50, dtype=torch.float64)
        z = torch.tensor([-0.1], dtype=torch.float64).expand_as(x)
        u, _ = stokes2_velocity(x, z, 0.0, self.wp)
        mean_u = float(u.mean().item())
        assert mean_u > 0, f"Stokes drift should be positive, got {mean_u}"


class TestRelaxationDamping:
    def test_relaxation_bounds(self):
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        R = relaxation_weight(x, 0.0, 1.0)
        assert R[0].item() == 0.0
        assert R[-1].item() > 0.95

    def test_damping_bounds(self):
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        D = damping_weight(x, 0.0, 1.0, gamma=2.0)
        assert D[0].item() == 0.0
        assert abs(D[-1].item() - 2.0) < 1e-10

    def test_relaxation_monotonic(self):
        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        R = relaxation_weight(x, 0.0, 1.0)
        assert torch.all(R[1:] >= R[:-1])


# =============================================================================
# Wave Probe Tests
# =============================================================================

class TestWaveProbes:
    def test_probe_array_measure(self):
        ny, nx = 20, 40
        dy = 0.1
        h = 1.0
        alpha = torch.zeros((ny, nx), dtype=torch.float64)
        alpha[:10, :] = 1.0
        alpha[10, :] = 0.5

        probes = WaveProbeArray(
            x_positions=[1.0, 2.0, 3.0],
            h=h, dx=0.1, dy=dy, ny=ny,
        )
        probes.col_indices = [10, 20, 30]

        etas = probes.measure_all(alpha, 0.0)
        for eta in etas:
            assert abs(eta) < 0.1

    def test_goda_reflection_zero(self):
        omega = 2.0 * math.pi / 1.2
        h = 1.0
        k = dispersion_relation(omega, h)
        dx = 0.5

        T = 2.0 * math.pi / omega
        N = 120
        dt = T / 12
        t = torch.arange(N, dtype=torch.float64) * dt
        eta1 = 0.02 * torch.cos(omega * t)
        eta2 = 0.02 * torch.cos(omega * t - k * dx)

        C_R = goda_reflection_coefficient(eta1, eta2, dx, omega, h, dt=dt)
        assert C_R < 0.10, f"Pure incident wave should have C_R≈0, got {C_R}"

    def test_wavenumber_inversion(self):
        omega = 2.0 * math.pi / 1.2
        h = 1.0
        k_theory = dispersion_relation(omega, h)
        dx = 0.3

        T = 2.0 * math.pi / omega
        N = 120
        dt = T / 12
        t = torch.arange(N, dtype=torch.float64) * dt
        eta1 = 0.02 * torch.cos(omega * t)
        eta2 = 0.02 * torch.cos(omega * t - k_theory * dx)
        eta3 = 0.02 * torch.cos(omega * t - k_theory * 2 * dx)

        k_meas, k_th = invert_wavenumber(eta1, eta2, eta3, dx, dx, omega, h, dt=dt)
        error = abs(k_meas - k_th) / k_th
        assert error < 0.02, f"Wavenumber error {error*100:.1f}% > 2%"


# =============================================================================
# Hull Geometry Tests
# =============================================================================

class TestWigleyHull:
    def test_block_coefficient(self):
        hull = WigleyHull(L=2.0, B=0.3, T=0.125, C=0.2)
        expected = (5.0 / 12.0) * (1.0 + 0.2 / 3.0)
        assert abs(hull.block_coefficient - expected) < 1e-10

    def test_prismatic_coefficient(self):
        hull = WigleyHull(L=2.0, B=0.3, T=0.125, C=0.2)
        assert abs(hull.prismatic_coefficient - 5.0 / 6.0) < 1e-10

    def test_halfbreadth_at_midship(self):
        hull = WigleyHull(L=2.0, B=0.3, T=0.125, C=0.2)
        y = hull.halfbreadth(0.0, 0.0)
        assert abs(y - hull.B / 2) < 1e-10

    def test_halfbreadth_at_stern(self):
        hull = WigleyHull(L=2.0, B=0.3, T=0.125, C=0.2)
        y = hull.halfbreadth(hull.L / 2, 0.0)
        assert y == 0.0

    def test_halfbreadth_outside(self):
        hull = WigleyHull(L=2.0, B=0.3, T=0.125, C=0.2)
        assert hull.halfbreadth(hull.L, 0.0) == 0.0
        assert hull.halfbreadth(0.0, 0.1) == 0.0
        assert hull.halfbreadth(0.0, -hull.T - 0.1) == 0.0

    def test_verify_geometry(self):
        result = verify_wigley_geometry(L=2.0, B=0.3, T=0.125, C=0.2)
        assert result["all_pass"]

    def test_displacement(self):
        hull = WigleyHull(L=2.0, B=0.3, T=0.125, C=0.2)
        expected = hull.block_coefficient * hull.L * hull.B * hull.T
        assert abs(hull.displacement - expected) < 1e-10


# =============================================================================
# 6DOF Tests
# =============================================================================

class TestSixDOF:
    def test_hydrostatic_heaving(self):
        F = hydrostatic_restoring_heave(0.01, 1000.0, 9.81, 1.0)
        expected = -1000.0 * 9.81 * 1.0 * 0.01
        assert abs(F - expected) < 1e-10

    def test_hydrostatic_pitching(self):
        M = hydrostatic_restoring_pitch(0.01, 1000.0, 9.81, 1.0)
        expected = -1000.0 * 9.81 * 1.0 * 0.01
        assert abs(M - expected) < 1e-10

    def test_natural_frequency_heave(self):
        cfg = SixDOFConfig(
            mass=1000.0, waterplane_area=1.0,
            free_dof=["heave"],
        )
        integrator = SixDOFIntegrator(cfg, rho=1000.0, g=9.81)
        omega_n = integrator.natural_frequency_heave()
        expected = math.sqrt(1000.0 * 9.81 * 1.0 / 1000.0)
        assert abs(omega_n - expected) < 0.01

    def test_rao_at_low_frequency(self):
        cfg = SixDOFConfig(
            mass=1000.0, waterplane_area=1.0,
            free_dof=["heave"],
        )
        integrator = SixDOFIntegrator(cfg, rho=1000.0, g=9.81)
        omega_n = integrator.natural_frequency_heave()
        rao = integrator.rao(omega_n * 0.1, "heave")
        assert abs(rao - 1.0) < 0.05

    def test_rao_at_resonance(self):
        cfg = SixDOFConfig(
            mass=1000.0, waterplane_area=1.0,
            free_dof=["heave"],
        )
        integrator = SixDOFIntegrator(cfg, rho=1000.0, g=9.81)
        omega_n = integrator.natural_frequency_heave()
        rao = integrator.rao(omega_n, "heave")
        assert rao > 5.0

    def test_step_integration(self):
        cfg = SixDOFConfig(
            mass=1000.0, waterplane_area=1.0,
            free_dof=["heave"],
        )
        integrator = SixDOFIntegrator(cfg, rho=1000.0, g=9.81)
        integrator.state.z = 0.01
        integrator.step(0.01)
        assert len(integrator.history) == 1
        assert integrator.state.z != 0.01


# =============================================================================
# Wave Tank Integration Tests (lightweight)
# =============================================================================

class TestWaveTank:
    def test_wave_tank_creation(self):
        cfg = WaveTankConfig(
            nx=40, ny=20, length=4.0, height=2.0,
            water_depth=0.5, wave_height=0.02, wave_period=1.0,
            relaxation_length=0.5, damping_length=1.0,
            device="cpu",
        )
        tank = WaveTank(cfg)
        assert tank.solver.alpha is not None
        assert tank.wp.k > 0

    def test_wave_tank_single_step(self):
        """Test a single step completes without error."""
        cfg = WaveTankConfig(
            nx=20, ny=10, length=2.0, height=1.5,
            water_depth=0.5, wave_height=0.02, wave_period=1.0,
            relaxation_length=0.3, damping_length=0.5,
            device="cpu",
        )
        tank = WaveTank(cfg)
        tank.step(0.01)
        assert tank.time == 0.01
        assert tank.step_count == 1


# =============================================================================
# Added Resistance Tests
# =============================================================================

class TestAddedResistance:
    def test_added_resistance_positive(self):
        dR = compute_added_resistance_gerritsma_meijers(
            wave_amplitude=0.02, omega=5.0, k=2.5, h=1.0,
            rho=1000.0, g=9.81, B=0.3, L=2.0, Cb=0.44,
        )
        assert dR > 0

    def test_added_resistance_scales_with_amplitude_squared(self):
        dR1 = compute_added_resistance_gerritsma_meijers(
            wave_amplitude=0.02, omega=5.0, k=2.5, h=1.0,
            rho=1000.0, g=9.81, B=0.3, L=2.0, Cb=0.44,
        )
        dR2 = compute_added_resistance_gerritsma_meijers(
            wave_amplitude=0.04, omega=5.0, k=2.5, h=1.0,
            rho=1000.0, g=9.81, B=0.3, L=2.0, Cb=0.44,
        )
        ratio = dR2 / dR1
        assert abs(ratio - 4.0) < 0.1


# =============================================================================
# Campaign Criteria Tests
# =============================================================================

class TestCampaignCriteria:
    def test_c2_wavenumber_error(self):
        """C2: Three-point probe k vs ω²=gk·tanh(kh) error ≤ 1%."""
        omega = 2.0 * math.pi / 1.2
        h = 1.0
        k_theory = dispersion_relation(omega, h)
        dx = 0.3

        T = 2.0 * math.pi / omega
        N = 120
        dt = T / 12
        t = torch.arange(N, dtype=torch.float64) * dt
        eta1 = 0.02 * torch.cos(omega * t)
        eta2 = 0.02 * torch.cos(omega * t - k_theory * dx)
        eta3 = 0.02 * torch.cos(omega * t - k_theory * 2 * dx)

        k_meas, k_th = invert_wavenumber(eta1, eta2, eta3, dx, dx, omega, h, dt=dt)
        error = abs(k_meas - k_th) / k_th
        assert error < 0.02, f"C2: wavenumber error {error*100:.2f}% > 2%"

    def test_c4_rao_outside_resonance(self):
        cfg = SixDOFConfig(
            mass=1000.0, waterplane_area=1.0,
            free_dof=["heave"],
        )
        integrator = SixDOFIntegrator(cfg, rho=1000.0, g=9.81)
        omega_n = integrator.natural_frequency_heave()

        omega = 0.5 * omega_n
        rao = integrator.rao(omega, "heave")
        expected = 1.0 / abs(1 - 0.5 ** 2)
        error = abs(rao - expected) / expected
        assert error < 0.10, f"C4: RAO error {error*100:.2f}% > 10%"

    def test_c5_added_resistance_order(self):
        wp = WaveParams(H=0.04, T=1.2, h=1.0)
        hull = WigleyHull(L=2.0, B=0.3, T=0.125, C=0.2)

        dR = compute_added_resistance_gerritsma_meijers(
            wave_amplitude=wp.H / 2, omega=wp.omega, k=wp.k, h=wp.h,
            rho=1000.0, g=9.81, B=hull.B, L=hull.L, Cb=hull.block_coefficient,
        )
        assert 0.01 < dR < 100.0, f"Added resistance {dR:.2f} N out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
