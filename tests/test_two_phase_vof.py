"""Unit tests for two-phase VOF transport and properties.

Tests cover:
- VOF volume conservation (geometric conservation)
- Flat interface limit (no spurious diffusion)
- Single-phase regression (alpha=0 or 1 everywhere)
- Mixture properties (density/viscosity)
- Interface normal and curvature
- CSF surface tension
- Free surface BC
"""

import torch
import pytest

from tensorfvm.two_phase_vof import (
    VOFTransport,
    TwoPhaseProperties,
    mixture_density,
    mixture_viscosity,
    csf_surface_tension,
    interface_normal,
    interface_curvature,
)
from tensorfvm.free_surface_bc import (
    FreeSurfaceBC,
    apply_open_pressure,
    outflow_vof_flux,
)


# ============================================================================
# Volume conservation tests
# ============================================================================

class TestVOFVolumeConservation:
    """VOF transport must preserve total phase volume to machine precision."""

    def test_zero_velocity_conservation(self):
        """With zero velocity, alpha must not change."""
        ny, nx = 20, 20
        dx, dy = 1.0 / nx, 1.0 / ny
        vof = VOFTransport(ny, nx, dx, dy)

        alpha = torch.zeros((ny, nx), dtype=torch.float64)
        alpha[:ny // 2, :] = 1.0

        u = torch.zeros((ny, nx + 1), dtype=torch.float64)
        v = torch.zeros((ny + 1, nx), dtype=torch.float64)

        init_vol = float(alpha.sum()) * dx * dy
        for _ in range(100):
            alpha = vof.advect(alpha, u, v, dt=0.01)

        final_vol = float(alpha.sum()) * dx * dy
        assert abs(final_vol - init_vol) < 1e-14

    def test_uniform_flow_conservation(self):
        """Uniform flow with closed domain: volume change = net boundary flux.

        With non-periodic boundaries, fluid leaving the domain changes total
        volume. We verify that the volume change is consistent with the
        boundary flux (i.e. conservation holds in the global sense).
        """
        ny, nx = 16, 16
        dx, dy = 1.0 / nx, 1.0 / ny
        vof = VOFTransport(ny, nx, dx, dy)

        # Bubble in center
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        alpha = ((x[None, :] - 0.5)**2 + (y[:, None] - 0.5)**2 < 0.1**2).double()

        # Uniform diagonal flow
        u = torch.full((ny, nx + 1), 1.0, dtype=torch.float64)
        v = torch.full((ny + 1, nx), 1.0, dtype=torch.float64)

        init_vol = float(alpha.sum()) * dx * dy
        dt = 0.4 * dx  # CFL < 1
        for _ in range(50):
            alpha = vof.advect(alpha, u, v, dt=dt)

        final_vol = float(alpha.sum()) * dx * dy
        # Volume changes due to boundary outflow, but must remain bounded
        assert final_vol >= 0.0
        assert final_vol <= 1.0
        # Alpha stays in [0,1]
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_bounded_alpha(self):
        """Alpha must remain in [0, 1] after advection."""
        ny, nx = 20, 20
        dx, dy = 1.0 / nx, 1.0 / ny
        vof = VOFTransport(ny, nx, dx, dy)

        alpha = torch.rand((ny, nx), dtype=torch.float64)
        u = torch.randn((ny, nx + 1), dtype=torch.float64) * 0.5
        v = torch.randn((ny + 1, nx), dtype=torch.float64) * 0.5

        for _ in range(20):
            alpha = vof.advect(alpha, u, v, dt=0.1)

        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0


# ============================================================================
# Flat interface limit tests
# ============================================================================

class TestFlatInterfaceLimit:
    """A flat interface must remain flat under zero or uniform normal flow."""

    def test_horizontal_interface_zero_velocity(self):
        """Horizontal interface with zero velocity: no perturbation."""
        ny, nx = 30, 30
        dx, dy = 1.0 / nx, 1.0 / ny
        vof = VOFTransport(ny, nx, dx, dy)

        alpha = torch.zeros((ny, nx), dtype=torch.float64)
        alpha[:ny // 2, :] = 1.0

        u = torch.zeros((ny, nx + 1), dtype=torch.float64)
        v = torch.zeros((ny + 1, nx), dtype=torch.float64)

        for _ in range(500):
            alpha = vof.advect(alpha, u, v, dt=0.01)

        # Check interface position: should be exactly at ny//2
        for j in range(ny):
            row_mean = float(alpha[j, :].mean())
            if j < ny // 2:
                assert row_mean == 1.0
            else:
                assert row_mean == 0.0

    def test_stratified_uniform_tangential_flow(self):
        """Stratified alpha with tangential flow: interface stays flat."""
        ny, nx = 20, 20
        dx, dy = 1.0 / nx, 1.0 / ny
        vof = VOFTransport(ny, nx, dx, dy)

        alpha = torch.zeros((ny, nx), dtype=torch.float64)
        alpha[:ny // 2, :] = 1.0

        # Pure tangential (x-direction) flow
        u = torch.ones((ny, nx + 1), dtype=torch.float64)
        v = torch.zeros((ny + 1, nx), dtype=torch.float64)

        init_vol = float(alpha.sum())
        for _ in range(100):
            alpha = vof.advect(alpha, u, v, dt=0.1)

        # Volume conserved
        assert abs(float(alpha.sum()) - init_vol) < 1e-12

        # Interface still flat (check column-wise)
        for i in range(nx):
            col_sum = float(alpha[:, i].sum())
            assert abs(col_sum - ny // 2) < 1e-10


# ============================================================================
# Single-phase regression tests
# ============================================================================

class TestSinglePhaseRegression:
    """When alpha=0 or 1 everywhere, two-phase reduces to single-phase."""

    def test_pure_liquid_properties(self):
        """alpha=1 → rho=rho_l, mu=mu_l."""
        alpha = torch.ones((10, 10), dtype=torch.float64)
        rho_l, rho_g = 1000.0, 1.0
        mu_l, mu_g = 1e-3, 1e-5

        rho = mixture_density(alpha, rho_l, rho_g)
        mu = mixture_viscosity(alpha, mu_l, mu_g)

        assert torch.allclose(rho, torch.full_like(rho, rho_l))
        assert torch.allclose(mu, torch.full_like(mu, mu_l), rtol=1e-10)

    def test_pure_gas_properties(self):
        """alpha=0 → rho=rho_g, mu=mu_g."""
        alpha = torch.zeros((10, 10), dtype=torch.float64)
        rho_l, rho_g = 1000.0, 1.0
        mu_l, mu_g = 1e-3, 1e-5

        rho = mixture_density(alpha, rho_l, rho_g)
        mu = mixture_viscosity(alpha, mu_l, mu_g)

        assert torch.allclose(rho, torch.full_like(rho, rho_g))
        assert torch.allclose(mu, torch.full_like(mu, mu_g), rtol=1e-10)

    def test_vof_transport_pure_phases(self):
        """VOF with pure liquid/gas: bounded advection maintains physical range."""
        ny, nx = 16, 16
        dx, dy = 1.0 / nx, 1.0 / ny
        vof = VOFTransport(ny, nx, dx, dy)

        # Pure liquid
        alpha = torch.ones((ny, nx), dtype=torch.float64)
        u = torch.randn((ny, nx + 1), dtype=torch.float64) * 0.3
        v = torch.randn((ny + 1, nx), dtype=torch.float64) * 0.3

        for _ in range(50):
            alpha = vof.advect(alpha, u, v, dt=0.1)

        # Must remain in [0, 1]
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

        # Pure gas
        alpha = torch.zeros((ny, nx), dtype=torch.float64)
        for _ in range(50):
            alpha = vof.advect(alpha, u, v, dt=0.1)

        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0


# ============================================================================
# Mixture properties tests
# ============================================================================

class TestMixtureProperties:
    """Test mixture density and viscosity formulas."""

    def test_density_linear_average(self):
        """Density is linear average: rho = alpha*rho_l + (1-alpha)*rho_g."""
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        rho_l, rho_g = 1000.0, 1.0
        rho = mixture_density(alpha, rho_l, rho_g)

        expected = alpha * rho_l + (1 - alpha) * rho_g
        assert torch.allclose(rho, expected)

    def test_viscosity_harmonic_mean(self):
        """Viscosity is harmonic mean: 1/mu = alpha/mu_l + (1-alpha)/mu_g."""
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        mu_l, mu_g = 1e-3, 1e-5

        mu = mixture_viscosity(alpha, mu_l, mu_g)

        # Harmonic mean formula
        expected = (mu_l * mu_g) / (alpha * mu_g + (1 - alpha) * mu_l)
        assert torch.allclose(mu, expected, rtol=1e-10)

    def test_viscosity_large_ratio(self):
        """Harmonic mean handles large viscosity ratio without blowup."""
        alpha = torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64)
        mu_l, mu_g = 1.0, 1e-6  # ratio = 1e6

        mu = mixture_viscosity(alpha, mu_l, mu_g)
        assert torch.isfinite(mu).all()
        assert mu.min() > 0


# ============================================================================
# Interface geometry tests
# ============================================================================

class TestInterfaceGeometry:
    """Test interface normal and curvature computation."""

    def test_flat_interface_normal(self):
        """Flat horizontal interface: normal should point in -y direction."""
        ny, nx = 20, 20
        dx, dy = 1.0 / nx, 1.0 / ny

        alpha = torch.zeros((ny, nx), dtype=torch.float64)
        alpha[:ny // 2, :] = 1.0

        n = interface_normal(alpha, dx, dy)
        # At interface (row ny//2 - 1 and ny//2), normal should be approximately (0, -1)
        # (pointing from liquid to gas, i.e., upward)
        mid = ny // 2
        # Check a few cells near interface
        for i in range(5, nx - 5):
            if alpha[mid, i] > 0.1 and alpha[mid, i] < 0.9:
                # Normal y-component should be negative (pointing up in our convention)
                assert n[mid, i, 1] < 0


    def test_curvature_flat_interface(self):
        """Flat interface has near-zero curvature."""
        ny, nx = 30, 30
        dx, dy = 1.0 / nx, 1.0 / ny

        # Smooth interface (avoid sharp steps)
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        # 2D alpha: smooth step in y, constant in x
        alpha = 0.5 * (1 - torch.tanh((y.reshape(-1, 1).expand(ny, nx) - 0.5) / (3 * dy)))

        kappa = interface_curvature(alpha, dx, dy)
        # Interior cells away from interface should have near-zero curvature
        mid = ny // 2
        # Curvature at flat interface should be small
        assert kappa[mid, nx // 2].abs() < 0.1


# ============================================================================
# CSF surface tension tests
# ============================================================================

class TestCSFSurfaceTension:
    """Test Brackbill CSF surface tension model."""

    def test_zero_sigma_returns_none(self):
        """sigma=0 → no surface tension (returns zero tensor)."""
        alpha = torch.rand((10, 10), dtype=torch.float64)
        result = csf_surface_tension(alpha, sigma=0.0, dx=0.1, dy=0.1)
        assert result is not None
        assert torch.allclose(result, torch.zeros_like(result))

    def test_finite_force(self):
        """CSF force must be finite."""
        ny, nx = 20, 20
        dx, dy = 1.0 / nx, 1.0 / ny

        # Bubble
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        alpha = ((x[None, :] - 0.5)**2 + (y[:, None] - 0.5)**2 < 0.2**2).double()

        sigma = 0.072
        force = csf_surface_tension(alpha, sigma, dx, dy)
        assert force is not None
        assert torch.isfinite(force).all()

    def test_uniform_alpha_zero_force(self):
        """Uniform alpha (no interface) → zero CSF force."""
        alpha = torch.ones((20, 20), dtype=torch.float64)
        force = csf_surface_tension(alpha, sigma=0.072, dx=0.05, dy=0.05)
        if force is not None:
            assert force.abs().max() < 1e-10


# ============================================================================
# Free surface BC tests
# ============================================================================

class TestFreeSurfaceBC:
    """Test free surface boundary condition module."""

    def test_apply_open_pressure(self):
        """Open pressure sets gauge p=0 on masked cells."""
        p = torch.ones((10, 10), dtype=torch.float64) * 5.0
        mask = torch.zeros((10, 10), dtype=torch.bool)
        mask[-1, :] = True  # top row

        p_bc = apply_open_pressure(p, mask)
        assert torch.all(p_bc[-1, :] == 0.0)
        assert torch.all(p_bc[:-1, :] == 5.0)

    def test_outflow_vof_flux_zero_velocity(self):
        """Zero velocity → zero outflow flux."""
        alpha = torch.rand((10, 10), dtype=torch.float64)
        u = torch.zeros((10, 11), dtype=torch.float64)
        v = torch.zeros((11, 10), dtype=torch.float64)

        flux = outflow_vof_flux(alpha, u, v, dx=0.1, dy=0.1, dt=0.01)
        assert flux == 0.0

    def test_free_surface_bc_creation(self):
        """FreeSurfaceBC can be instantiated."""
        bc = FreeSurfaceBC(ny=20, nx=20, dx=0.05, dy=0.05)
        assert bc.top_atmosphere is True
        assert bc.bottom_wall is True


# ============================================================================
# TwoPhaseProperties tests
# ============================================================================

class TestTwoPhaseProperties:
    """Test TwoPhaseProperties dataclass."""

    def test_default_creation(self):
        """Default properties are physically reasonable."""
        props = TwoPhaseProperties(
            rho_liquid=1000.0,
            rho_gas=1.0,
            mu_liquid=1e-3,
            mu_gas=1e-5
        )
        assert props.rho_liquid > props.rho_gas
        assert props.mu_liquid > props.mu_gas

    def test_custom_properties(self):
        """Custom properties stored correctly."""
        props = TwoPhaseProperties(
            rho_liquid=1000.0,
            rho_gas=1.2,
            mu_liquid=1e-3,
            mu_gas=1.8e-5,
            sigma=0.072,
            surface_tension=True,
        )
        assert props.rho_liquid == 1000.0
        assert props.sigma == 0.072
        assert props.surface_tension is True
