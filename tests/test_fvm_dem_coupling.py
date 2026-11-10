"""Pytest tests for FVM-DEM two-phase coupling verification.

Covers all verification criteria:
  C1: Single floe drift (3 velocities)
  C2: Momentum conservation
  C3: Newton's 3rd law
  C4: Multi-floe interaction
  C5: Grid convergence
  C6: Time step convergence
  C7: Negative control
  C8: ≥18 test cases
"""

import math
import pytest
import torch

from tensorfvm.fvm_dem_coupling import (
    FloeProperties,
    FloeState,
    FlowField,
    CouplingConfig,
    DEMSolver,
    FVMFlowSolver,
    FVMDEMCoupling,
    verify_single_floe_drift,
    verify_momentum_conservation,
    verify_newton_third_law,
    verify_multi_floe_interaction,
    verify_grid_convergence,
    verify_timestep_convergence,
    verify_negative_control,
    run_all_verifications,
)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    yield


# ============================================================================
# Unit tests: FloeProperties
# ============================================================================

class TestFloeProperties:
    """Test floe physical property calculations."""

    def test_mass_calculation(self):
        props = FloeProperties(radius_m=0.5, thickness_m=0.3, density_kg_m3=917.0)
        expected = math.pi * 0.5**2 * 0.3 * 917.0
        assert abs(props.mass_kg - expected) < 1e-10

    def test_added_mass(self):
        props = FloeProperties(radius_m=0.5, thickness_m=0.3, added_mass_coeff=1.0)
        expected = 1.0 * 1025.0 * math.pi * 0.5**2 * 0.3
        assert abs(props.added_mass_kg - expected) < 1e-10

    def test_total_inertia(self):
        props = FloeProperties(radius_m=0.5, thickness_m=0.3)
        assert props.total_inertia_kg == props.mass_kg + props.added_mass_kg

    def test_projected_area(self):
        props = FloeProperties(radius_m=0.5, thickness_m=0.3)
        expected = 2.0 * 0.5 * 0.3
        assert abs(props.projected_area_m2 - expected) < 1e-10


# ============================================================================
# Unit tests: DEMSolver
# ============================================================================

class TestDEMSolver:
    """Test DEM solver mechanics."""

    def test_add_floe(self):
        props = FloeProperties()
        config = CouplingConfig()
        dem = DEMSolver(props, config)
        fid = dem.add_floe(5.0, 2.0)
        assert fid == 0
        assert len(dem.floes) == 1
        assert float(dem.floes[0].position[0]) == 5.0

    def test_add_multiple_floes(self):
        props = FloeProperties()
        config = CouplingConfig()
        dem = DEMSolver(props, config)
        dem.add_floe(1.0, 1.0)
        dem.add_floe(2.0, 2.0)
        dem.add_floe(3.0, 3.0)
        assert len(dem.floes) == 3

    def test_add_floes_uniform(self):
        props = FloeProperties()
        config = CouplingConfig()
        dem = DEMSolver(props, config)
        dem.add_floes_uniform(10, x_range=(1.0, 9.0), y_range=(1.0, 3.0))
        assert len(dem.floes) == 10
        for floe in dem.floes:
            assert 1.0 <= float(floe.position[0]) <= 9.0
            assert 1.0 <= float(floe.position[1]) <= 3.0

    def test_drag_force_direction(self):
        """Drag should be in direction of relative velocity."""
        props = FloeProperties(radius_m=0.3, drag_coefficient=2.0)
        config = CouplingConfig(water_density=1025.0)
        dem = DEMSolver(props, config)
        dem.add_floe(5.0, 2.0, vx=0.0, vy=0.0)

        floe = dem.floes[0]
        u_fluid = torch.tensor([1.0, 0.0], dtype=torch.float64)
        F_D = dem.compute_drag_force(floe, u_fluid)

        # Drag should be positive x (in direction of flow)
        assert F_D[0] > 0
        assert abs(F_D[1]) < 1e-12

    def test_drag_force_zero_relative_velocity(self):
        """Zero drag when floe moves with fluid."""
        props = FloeProperties()
        config = CouplingConfig()
        dem = DEMSolver(props, config)
        dem.add_floe(5.0, 2.0, vx=1.0, vy=0.0)

        floe = dem.floes[0]
        u_fluid = torch.tensor([1.0, 0.0], dtype=torch.float64)
        F_D = dem.compute_drag_force(floe, u_fluid)

        assert float(torch.norm(F_D)) < 1e-10

    def test_drag_force_magnitude(self):
        """Check drag magnitude: F = 0.5 * ρ * C_D * A * |u-v|²."""
        props = FloeProperties(radius_m=0.5, thickness_m=0.3, drag_coefficient=2.0)
        config = CouplingConfig(water_density=1025.0)
        dem = DEMSolver(props, config)
        dem.add_floe(5.0, 2.0, vx=0.0, vy=0.0)

        floe = dem.floes[0]
        u_fluid = torch.tensor([2.0, 0.0], dtype=torch.float64)
        F_D = dem.compute_drag_force(floe, u_fluid)

        A = props.projected_area_m2
        expected = 0.5 * 1025.0 * 2.0 * A * 2.0**2
        assert abs(F_D[0].item() - expected) / expected < 1e-10

    def test_step_advances_position(self):
        """Step should advance floe position."""
        props = FloeProperties()
        config = CouplingConfig()
        dem = DEMSolver(props, config)
        dem.add_floe(5.0, 2.0, vx=1.0, vy=0.0)

        # Apply a force to create acceleration
        dem.forces[0] = torch.tensor([100.0, 0.0], dtype=torch.float64)
        dem.step(dt=0.01)

        # Position should have changed
        assert float(dem.floes[0].position[0]) != 5.0

    def test_kinetic_energy(self):
        """Kinetic energy = 0.5 * m * v²."""
        props = FloeProperties(radius_m=0.5, thickness_m=0.3)
        config = CouplingConfig()
        dem = DEMSolver(props, config)
        dem.add_floe(5.0, 2.0, vx=2.0, vy=0.0)

        ke = dem.get_kinetic_energy()
        expected = 0.5 * props.total_inertia_kg * 4.0
        assert abs(ke - expected) / expected < 1e-10

    def test_total_momentum(self):
        """Total momentum = Σ m*v."""
        props = FloeProperties()
        config = CouplingConfig()
        dem = DEMSolver(props, config)
        dem.add_floe(1.0, 1.0, vx=1.0, vy=0.0)
        dem.add_floe(2.0, 2.0, vx=0.0, vy=2.0)

        p = dem.get_total_momentum()
        m = props.total_inertia_kg
        assert abs(p[0].item() - m * 1.0) < 1e-10
        assert abs(p[1].item() - m * 2.0) < 1e-10


# ============================================================================
# Unit tests: FVMFlowSolver
# ============================================================================

class TestFVMFlowSolver:
    """Test FVM flow solver."""

    def test_initialization(self):
        config = CouplingConfig(nx=20, ny=10, inlet_velocity=1.0)
        fvm = FVMFlowSolver(config)
        assert fvm.u.shape == (10, 20)
        assert float(fvm.u[5, 10]) == 1.0

    def test_interpolate_velocity_center(self):
        """Interpolation at cell center should return cell value."""
        config = CouplingConfig(nx=20, ny=10, inlet_velocity=2.0)
        fvm = FVMFlowSolver(config)
        # At center of domain
        x_mid = config.length_m / 2
        y_mid = config.height_m / 2
        u_interp = fvm.interpolate_velocity(x_mid, y_mid)
        assert abs(u_interp[0].item() - 2.0) < 0.1

    def test_blocking_force_application(self):
        """Blocking force should modify source terms."""
        config = CouplingConfig(nx=20, ny=10)
        fvm = FVMFlowSolver(config)
        fvm.add_blocking_force(5.0, 2.0, torch.tensor([-10.0, 0.0], dtype=torch.float64))
        assert float(torch.sum(torch.abs(fvm.source_x))) > 0

    def test_reset_sources(self):
        config = CouplingConfig(nx=20, ny=10)
        fvm = FVMFlowSolver(config)
        fvm.add_blocking_force(5.0, 2.0, torch.tensor([-10.0, 0.0], dtype=torch.float64))
        fvm.reset_sources()
        assert float(torch.sum(torch.abs(fvm.source_x))) == 0.0

    def test_boundary_conditions(self):
        """After step, BCs should be enforced."""
        config = CouplingConfig(nx=20, ny=10, inlet_velocity=1.5)
        fvm = FVMFlowSolver(config)
        fvm.step(dt=0.01)
        # Inlet should be uniform
        assert abs(float(fvm.u[5, 0]) - 1.5) < 1e-10
        # Walls should be no-slip
        assert abs(float(fvm.u[0, 10])) < 1e-10
        assert abs(float(fvm.u[-1, 10])) < 1e-10


# ============================================================================
# Integration tests: FVMDEMCoupling
# ============================================================================

class TestFVMDEMCoupling:
    """Test coupled solver."""

    def test_coupling_initialization(self):
        config = CouplingConfig()
        props = FloeProperties()
        coupling = FVMDEMCoupling(config, props)
        assert coupling.time == 0.0
        assert coupling.step_count == 0

    def test_single_step(self):
        config = CouplingConfig(nx=20, ny=10, dt=0.01)
        props = FloeProperties()
        coupling = FVMDEMCoupling(config, props)
        coupling.dem.add_floe(5.0, 2.0)

        diag = coupling.step()
        assert "time" in diag
        assert "total_drag_x" in diag
        assert "newton3_error" in diag
        assert coupling.step_count == 1

    def test_newton_third_law_exact(self):
        """Newton's 3rd law: F_block = -F_D exactly (explicit coupling)."""
        config = CouplingConfig(nx=20, ny=10, dt=0.01)
        props = FloeProperties()
        coupling = FVMDEMCoupling(config, props)
        coupling.dem.add_floe(5.0, 2.0)

        diag = coupling.step()
        # In explicit coupling, blocking = -drag exactly
        assert diag["newton3_error"] < 1e-6

    def test_floe_accelerates_in_flow(self):
        """Floe should accelerate in flow direction."""
        config = CouplingConfig(nx=30, ny=15, dt=0.01, inlet_velocity=1.0)
        props = FloeProperties(radius_m=0.3, thickness_m=0.2)
        coupling = FVMDEMCoupling(config, props)
        coupling.dem.add_floe(5.0, 2.5)

        for _ in range(50):
            coupling.step()

        vx = float(coupling.dem.floes[0].velocity[0])
        assert vx > 0.1, f"Floe should accelerate, got vx={vx}"


# ============================================================================
# Verification criteria tests (C1-C7)
# ============================================================================

class TestVerificationCriteria:
    """Test all verification criteria."""

    def test_C1_single_floe_drift_u05(self):
        """C1: Single floe drift at u=0.5 m/s, error ≤10%."""
        result = verify_single_floe_drift(u_inlet=0.5, total_time=8.0, dt=0.01)
        assert result["passed"], f"C1 u=0.5 failed: error={result['relative_error']:.4f}"

    def test_C1_single_floe_drift_u10(self):
        """C1: Single floe drift at u=1.0 m/s, error ≤10%."""
        result = verify_single_floe_drift(u_inlet=1.0, total_time=8.0, dt=0.01)
        assert result["passed"], f"C1 u=1.0 failed: error={result['relative_error']:.4f}"

    def test_C1_single_floe_drift_u15(self):
        """C1: Single floe drift at u=1.5 m/s, error ≤10%."""
        result = verify_single_floe_drift(u_inlet=1.5, total_time=8.0, dt=0.01)
        assert result["passed"], f"C1 u=1.5 failed: error={result['relative_error']:.4f}"

    def test_C2_momentum_conservation(self):
        """C2: Momentum conservation residual ≤3%."""
        result = verify_momentum_conservation(n_steps=5)
        assert result["passed"], f"C2 failed: residual={result['relative_residual']:.6f}"

    def test_C3_newton_third_law_explicit(self):
        """C3: Newton's 3rd law error ≤1e-6 (explicit)."""
        result = verify_newton_third_law("explicit")
        assert result["passed"], f"C3 failed: max_error={result['max_error']:.2e}"

    def test_C4_multi_floe_interaction(self):
        """C4: Multi-floe clearing rate error ≤15%."""
        result = verify_multi_floe_interaction(n_floes=10)
        assert result["passed"], f"C4 failed: error={result['relative_error']:.4f}"

    def test_C5_grid_convergence(self):
        """C5: Grid refinement changes v_drift ≤5%."""
        result = verify_grid_convergence()
        assert result["passed"], f"C5 failed: change={result['relative_change']:.4f}"

    def test_C6_timestep_convergence(self):
        """C6: Time step halving changes v_drift ≤3%."""
        result = verify_timestep_convergence()
        assert result["passed"], f"C6 failed: change={result['relative_change']:.4f}"

    def test_C7_negative_control(self):
        """C7: No flow → floe stationary (v < 0.01 m/s)."""
        result = verify_negative_control()
        assert result["passed"], f"C7 failed: speed={result['final_floe_speed']:.6f}"


# ============================================================================
# Integration: run_all_verifications
# ============================================================================

class TestRunAll:
    """Integration test: run all verifications."""

    def test_all_verifications(self, tmp_path):
        summary = run_all_verifications(output_dir=tmp_path)
        assert "all_passed" in summary
        assert "criteria" in summary
        # Check output files
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "C1_single_floe_drift.json").exists()
        assert (tmp_path / "C2_momentum_conservation.json").exists()
        assert (tmp_path / "C7_negative_control.json").exists()
