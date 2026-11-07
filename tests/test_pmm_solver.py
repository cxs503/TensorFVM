"""Tests for SectionNS2D (2D Navier-Stokes solver with IBM)."""
import numpy as np
import pytest

from tensorfvm.pmm_solver import SectionNS2D
from tensorfvm.pmm_cfd import KVLCC2Hull


class TestSectionNS2D:
    """Test SectionNS2D solver."""
    
    @pytest.fixture
    def hull_section(self):
        """Generate a KVLCC2 hull cross-section."""
        hull = KVLCC2Hull()
        # Use midship section
        y, z = hull.section_outline(hull.L / 2, n=121)
        return np.column_stack([y, z])
    
    @pytest.fixture
    def solver(self, hull_section):
        """Create a SectionNS2D solver instance."""
        return SectionNS2D(
            hull_section=hull_section,
            grid_size=(64, 64),
            domain_size=(2.0, 2.0),
            reynolds=1000.0,
            time_step=0.01,
        )
    
    def test_solver_initialization(self, solver):
        """Test solver initialization."""
        assert solver.grid_size == (50, 50)
        assert solver.domain_size == (2.0, 2.0)
        assert solver.reynolds == 1000.0
        assert solver.time_step == 0.01
    
    def test_mask_creation(self, solver):
        """Test mask creation for hull cross-section."""
        mask = solver._create_mask()
        assert mask.shape == solver.grid_size
        # Check that some cells are fluid and some are solid
        assert mask.sum() > 0
        assert (~mask).sum() > 0
    
    def test_point_in_polygon(self, solver):
        """Test point-in-polygon test."""
        polygon = solver.hull_section
        # Test a point inside the hull
        # Midship section center is approximately (0, -T/2)
        inside = solver._point_in_polygon(0.0, -0.171, polygon)
        assert inside is True
        
        # Test a point outside the hull
        outside = solver._point_in_polygon(1.0, 0.0, polygon)
        assert outside is False
    
    def test_body_velocity_sway(self, solver):
        """Test body velocity update for pure sway."""
        solver._update_body_velocity(0.0, "sway", 0.05, 2 * np.pi / 10.0)
        # At t=0, v = A*w*cos(0) = A*w
        expected_v = 0.05 * 2 * np.pi / 10.0
        assert solver.body_velocity[0] == pytest.approx(expected_v, rel=1e-6)
        assert solver.body_velocity[1] == 0.0
    
    def test_body_velocity_yaw(self, solver):
        """Test body velocity update for pure yaw."""
        solver._update_body_velocity(0.0, "yaw", np.radians(3.0), 2 * np.pi / 10.0)
        # Simplified: yaw velocity is 0 for now
        assert solver.body_velocity[0] == 0.0
        assert solver.body_velocity[1] == 0.0
    
    def test_force_computation(self, solver):
        """Test force computation."""
        # Create dummy fields
        ny, nx = solver.grid_size
        u = torch.zeros((ny, nx), dtype=torch.float64)
        v = torch.zeros((ny, nx), dtype=torch.float64)
        p = torch.zeros((ny, nx), dtype=torch.float64)
        mask = solver._create_mask()
        
        Y, N = solver._compute_forces(u, v, p, mask)
        assert Y == 0.0
        assert N == 0.0
    
    def test_pmm_test_sway(self, solver):
        """Test pure sway PMM test."""
        # Run a short test
        results = solver.run_pmm_test(
            mode="sway",
            amplitude=0.05,
            omega=2 * np.pi / 10.0,
            n_cycles=1,
        )
        
        assert "time" in results
        assert "Y" in results
        assert "N" in results
        assert len(results["time"]) > 0
        assert len(results["Y"]) == len(results["time"])
        assert len(results["N"]) == len(results["time"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
