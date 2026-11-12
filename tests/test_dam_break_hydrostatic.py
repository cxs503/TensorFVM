"""Tests for hydrostatic pressure initialization from arbitrary VOF fields."""

import pytest
import torch

from tensorfvm.verification.dam_break_hydrostatic import hydrostatic_pressure_from_alpha


class TestHydrostaticPressureFromAlpha:
    """Verify the hydrostatic pressure initializer against known cases."""

    def test_uniform_liquid_column(self):
        """A full column of liquid should give p = rho * g * (H - y)."""
        ny, nx = 20, 10
        dy = 0.01
        g = 9.81
        rho_l = 1000.0
        rho_g = 1.2

        alpha = torch.ones(ny, nx, dtype=torch.float64)
        p = hydrostatic_pressure_from_alpha(alpha, dy, g, rho_l, rho_g)

        H = ny * dy
        p_bottom_expected = rho_l * g * (H - dy)

        assert float(p[0, 0]) == pytest.approx(p_bottom_expected, rel=1e-10)
        assert float(p[-1, 0]) == pytest.approx(0.0, abs=1e-15)

    def test_uniform_gas(self):
        """A full column of gas should give near-zero pressure."""
        ny, nx = 20, 10
        dy = 0.01
        g = 9.81
        rho_l = 1000.0
        rho_g = 1.2

        alpha = torch.zeros(ny, nx, dtype=torch.float64)
        p = hydrostatic_pressure_from_alpha(alpha, dy, g, rho_l, rho_g)

        assert float(p.max()) < rho_g * g * ny * dy * 1.1

    def test_stratified_interface(self):
        """A horizontal interface should produce correct hydrostatic profile."""
        ny, nx = 40, 10
        dy = 0.005
        g = 9.81
        rho_l = 1000.0
        rho_g = 1.2
        interface_row = 20

        alpha = torch.ones(ny, nx, dtype=torch.float64)
        alpha[interface_row:, :] = 0.0

        p = hydrostatic_pressure_from_alpha(alpha, dy, g, rho_l, rho_g)

        # Top should be ~0
        assert float(p[-1, 0]) < 1.0
        # Bottom should be significant (liquid column weight)
        h_liquid = interface_row * dy
        p_bottom_expected = rho_g * g * (ny - interface_row) * dy + rho_l * g * h_liquid
        assert float(p[0, 0]) == pytest.approx(p_bottom_expected, rel=0.02)

    def test_dam_break_geometry(self):
        """Dam break: liquid column [0, W] x [0, H] in larger domain."""
        ny, nx = 30, 60
        dy = 0.01
        dx = 0.01
        g = 9.81
        rho_l = 1000.0
        rho_g = 1.2
        W = 0.1
        H = 0.2

        cx = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        cy = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        x, y = torch.meshgrid(cx, cy, indexing="xy")
        alpha = ((x < W) & (y < H)).to(torch.float64)

        p = hydrostatic_pressure_from_alpha(alpha, dy, g, rho_l, rho_g)

        # Inside liquid column, pressure ~ rho_l * g * (H - y)
        i_center = int(W / dx / 2)
        j_center = int(H / dy / 4)
        y_center = (j_center + 0.5) * dy
        p_expected = rho_l * g * (H - y_center)

        assert float(p[j_center, i_center]) == pytest.approx(p_expected, rel=0.15)

        # Outside liquid column, pressure should be small
        i_outside = int(W / dx * 1.5)
        j_inside = int(H / dy / 2)
        assert float(p[j_inside, i_outside]) < 10.0

    def test_pressure_gradient_balances_gravity(self):
        """The discrete pressure gradient should balance gravity in the liquid."""
        ny, nx = 40, 10
        dy = 0.005
        g = 9.81
        rho_l = 1000.0
        rho_g = 1.2

        alpha = torch.ones(ny, nx, dtype=torch.float64)
        p = hydrostatic_pressure_from_alpha(alpha, dy, g, rho_l, rho_g)

        for j in range(ny - 2):
            dpdy = (p[j, 0] - p[j + 1, 0]) / dy
            assert float(dpdy) == pytest.approx(rho_l * g, rel=1e-10)

    def test_device_and_dtype_preserved(self):
        """Output should match input device and dtype."""
        alpha = torch.ones(10, 10, dtype=torch.float32)
        p = hydrostatic_pressure_from_alpha(alpha, 0.01, 9.81, 1000.0, 1.2)
        assert p.dtype == alpha.dtype
        assert p.device == alpha.device
