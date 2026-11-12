"""Tests for the Hysing rising-bubble verification leg."""

import json
import math
import pytest

from tensorfvm.twophase_gap.hysing_gate import (
    HYSING_REF_UT,
    HYSING_REF_CIRCULARITY,
    HYSING_REF_YC,
    TERMINAL_VELOCITY_MAX,
    CIRCULARITY_MAX,
    MASS_DRIFT_MAX,
    CONVERGENCE_MAX,
    HysingCase,
    bubble_metrics,
    compute_terminal_velocity,
    convergence_order,
    run_hysing_leg,
)


class TestHysingConstants:
    """Reference values and gate thresholds are sane."""

    def test_ref_terminal_velocity(self):
        assert 0.2 < HYSING_REF_UT < 0.3

    def test_ref_circularity(self):
        assert 0.4 < HYSING_REF_CIRCULARITY < 0.5

    def test_ref_yc(self):
        assert 1.5 < HYSING_REF_YC < 2.5

    def test_gate_thresholds_positive(self):
        assert TERMINAL_VELOCITY_MAX > 0
        assert CIRCULARITY_MAX > 0
        assert MASS_DRIFT_MAX > 0
        assert CONVERGENCE_MAX > 0


class TestHysingCase:
    """HysingCase parameter computation."""

    def test_sigma(self):
        case = HysingCase()
        assert abs(case.sigma - 0.1) < 1e-12  # 1/Eo = 1/10

    def test_mu_liquid_positive(self):
        case = HysingCase()
        assert case.mu_liquid > 0

    def test_mu_gas_less_than_liquid(self):
        case = HysingCase()
        assert case.mu_gas < case.mu_liquid

    def test_rho_gas_less_than_liquid(self):
        case = HysingCase()
        assert case.rho_gas < 1.0  # liquid is 1.0

    def test_solve_config(self):
        case = HysingCase(cells_per_diameter=32)
        grid, tp = case.solve_config()
        assert grid.nx == 32
        assert grid.ny == 64  # domain_height/domain_width = 4/2 = 2
        assert tp.enabled is True
        assert tp.surface_tension is True
        assert tp.gravity == 1.0

    def test_initial_alpha_shape(self):
        case = HysingCase(cells_per_diameter=16)
        grid, tp = case.solve_config()
        alpha = case.initial_alpha(grid.nx, grid.ny, grid.length / grid.nx,
                                   grid.height / grid.ny)
        assert alpha.shape == (grid.ny, grid.nx)
        # Bubble is gas (alpha=0), so some cells should be 0
        assert (alpha == 0.0).any()
        # Most cells are liquid (alpha=1)
        assert (alpha == 1.0).sum() > alpha.numel() * 0.5


class TestBubbleMetrics:
    """bubble_metrics function on synthetic fields."""

    def test_full_liquid(self):
        import torch
        alpha = torch.ones((10, 10), dtype=torch.float64)
        m = bubble_metrics(alpha, 0.1, 0.1)
        # bubble = 1 - alpha = 0 everywhere
        assert m["volume"] == 0.0
        assert m["circularity"] == 0.0

    def test_circular_blob(self):
        import torch
        # Create a roughly circular blob in the center
        alpha = torch.ones((20, 20), dtype=torch.float64)
        cx, cy = 10, 10
        r = 5
        for i in range(20):
            for j in range(20):
                if (i - cy) ** 2 + (j - cx) ** 2 < r ** 2:
                    alpha[i, j] = 0.0  # gas
        dx = dy = 0.1
        m = bubble_metrics(alpha, dx, dy)
        assert m["volume"] > 0
        # circularity should be close to 1 for a circle
        assert 0.5 < m["circularity"] <= 1.0

    def test_centroid_centered(self):
        import torch
        # Single cell bubble in the center
        alpha = torch.ones((5, 5), dtype=torch.float64)
        alpha[2, 2] = 0.0  # gas cell
        dx = dy = 1.0
        m = bubble_metrics(alpha, dx, dy)
        assert abs(m["xc"] - 2.5) < 0.5
        assert abs(m["yc"] - 2.5) < 0.5


class TestComputeTerminalVelocity:
    """Terminal velocity estimation from history."""

    def test_empty_history(self):
        assert compute_terminal_velocity([]) == 0.0

    def test_short_history(self):
        h = [{"time": 0.0, "yc": 0.0}, {"time": 1.0, "yc": 0.5}]
        assert compute_terminal_velocity(h) == 0.0  # < 4 points

    def test_constant_velocity(self):
        h = [
            {"time": 0.0, "yc": 0.0},
            {"time": 1.0, "yc": 0.2},
            {"time": 2.0, "yc": 0.4},
            {"time": 3.0, "yc": 0.6},
        ]
        ut = compute_terminal_velocity(h, window_frac=0.5)
        assert abs(ut - 0.2) < 0.01  # dy/dt = 0.2

    def test_accelerating(self):
        h = [
            {"time": 0.0, "yc": 0.0},
            {"time": 1.0, "yc": 0.1},
            {"time": 2.0, "yc": 0.3},
            {"time": 3.0, "yc": 0.6},
        ]
        ut = compute_terminal_velocity(h, window_frac=0.5)
        # Last window: t=2->3, dy=0.3, dt=1.0
        assert abs(ut - 0.3) < 0.01


class TestConvergenceOrder:
    """Observed order from two terminal velocities."""

    def test_first_order(self):
        # halving grid spacing halves the error
        assert abs(convergence_order(0.1, 0.05, ratio=2.0) - 1.0) < 0.01

    def test_second_order(self):
        assert abs(convergence_order(0.1, 0.025, ratio=2.0) - 2.0) < 0.01

    def test_zero_order(self):
        # no improvement
        assert convergence_order(0.1, 0.1, ratio=2.0) == 0.0


class TestHysingLegFast:
    """Fast integration test: very coarse grid, short time."""

    @pytest.mark.slow
    def test_run_hysing_leg_cpd16(self):
        """Run at cpd=16 for a short time to verify the pipeline works."""
        result = run_hysing_leg(cells_per_diameter=16, end_time=1.0)
        rec = result["record"]
        assert rec["case"] == "hysing"
        assert rec["cells_per_diameter"] == 16
        assert rec["steps"] > 0
        assert rec["diverged"] is False
        assert rec["terminal_velocity"] >= 0
        assert rec["min_circularity"] >= 0
        assert rec["worst_step_drift"] >= 0
        # Reference values are present
        assert rec["reference"]["terminal_velocity"] == HYSING_REF_UT
        assert rec["reference"]["min_circularity"] == HYSING_REF_CIRCULARITY
