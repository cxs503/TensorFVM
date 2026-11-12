"""Tests for the Hysing rising-bubble verification leg."""

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
        # cells_per_diameter is a resolution: the Hysing box is 2d x 4d, so a
        # 32-cells-per-diameter grid is 64 x 128 (historic bug: nx = cpd).
        assert grid.nx == 64
        assert grid.ny == 128
        assert abs(grid.length / grid.nx - case.diameter / 32) < 1e-12
        assert tp.enabled is True
        assert tp.surface_tension is True
        assert tp.gravity == 1.0

    @pytest.mark.parametrize("cpd", [8, 16, 32])
    def test_bubble_spans_cells_per_diameter(self, cpd):
        case = HysingCase(cells_per_diameter=cpd)
        grid, _ = case.solve_config()
        alpha = case.initial_alpha(grid.nx, grid.ny, grid.length / grid.nx,
                                   grid.height / grid.ny)
        # cells crossed by the gas phase across the bubble's horizontal diameter
        dy = grid.height / grid.ny
        row = alpha[int(round(case.start_y / dy - 0.5)), :]
        span = float((row < 0.5).sum()) * (grid.length / grid.nx)
        assert abs(span - case.diameter) <= 2.0 * grid.length / grid.nx

    def test_bubble_clear_of_wall(self):
        case = HysingCase()
        # Hysing places the d=0.5 bubble at (0.5, 0.5) in [0,1]x[0,2] -> (1d, 1d)
        assert abs(case.start_y - 1.0) < 1e-12
        assert case.start_y - case.bubble_radius > 0.4

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
        # circularity close to 1; this blob is only ~5 cells in radius, and the
        # marching-squares contour is exact for a smooth field but carries a few
        # per-cent discretisation wobble for a binary one (see _interface_perimeter)
        assert 0.9 < m["circularity"] <= 1.06

    def test_metric_orientation_not_double_inverted(self):
        """A gas blob near the *top* must report a high yc.

        ``bubble_metrics`` takes the liquid fraction and inverts it itself; the
        historic caller passed ``1 - alpha`` which made the helper measure the
        *liquid* region, so a rising bubble appeared to sink (recorded FAIL:
        U_T = -0.022 vs 0.2342, circularity 5.62 > 1).
        """
        import torch
        alpha = torch.ones((20, 20), dtype=torch.float64)
        alpha[15:18, 8:12] = 0.0  # gas near the top
        m = bubble_metrics(alpha, 0.1, 0.1)
        assert m["yc"] > 1.5  # top half of the 2.0 domain
        in_liquid = bubble_metrics(1.0 - alpha, 0.1, 0.1)
        assert in_liquid["yc"] < 1.5  # double inversion puts the centroid low

    def test_analytic_disc_circularity(self):
        import numpy as np
        import torch

        n, h = 96, 1.0 / 32.0
        xs = (np.arange(n) + 0.5) * h
        a = np.ones((n, n))
        xc, yc, r = n * h / 2.0, n * h / 2.0, 0.75
        a[(xs[None, :] - xc) ** 2 + (xs[:, None] - yc) ** 2 < r ** 2] = 0.0
        m = bubble_metrics(torch.from_numpy(a), h, h)
        assert abs(m["circularity"] - 1.0) < 0.02
        assert abs(m["volume"] - math.pi * r ** 2) / (math.pi * r ** 2) < 0.03

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


class TestConvergenceGateB4:
    """B4 must be a *relative change* gate, not an observed-order gate.

    Historic defect (m16901): the B4 gate compared the observed order
    ``log(U_c/U_f)/log 2`` against ``CONVERGENCE_MAX = 0.05`` with ``>=``.
    That is satisfied by any coarse->fine change above ~3.5 %, so the gate
    could not fail on the divergence it was written to catch.
    """

    @staticmethod
    def _legs(coarse_ut: float, fine_ut: float) -> dict:
        base = {
            "case": "hysing", "eotvos": 10.0, "morton": 1e-3,
            "end_time": 3.0, "min_circularity": 0.5,
            "worst_step_drift": 1e-8, "diverged": False, "steps": 10,
            "peak_velocity": fine_ut,
            "circularity_rel_error": 0.0,
            "terminal_velocity_rel_error": 0.0,
        }
        coarse = dict(base, cells_per_diameter=8, terminal_velocity=coarse_ut,
                      declared=False)
        fine = dict(base, cells_per_diameter=16, terminal_velocity=fine_ut,
                    declared=True)
        return {"hysing_8": coarse, "hysing_16": fine}

    @staticmethod
    def _gate(legs: dict, gid: str) -> dict:
        from tensorfvm.twophase_gap.campaign import evaluate_gates

        gates, _ = evaluate_gates(legs)
        return next(g for g in gates if g["id"] == gid)

    def test_b4_fails_on_a_20_percent_change(self):
        gate = self._gate(self._legs(coarse_ut=0.30, fine_ut=0.24), "B4")
        assert gate["metric"] == "hysing_convergence_change"
        assert gate["got"] == pytest.approx(0.25, rel=1e-6)  # |0.24-0.30|/0.24
        assert gate["passed"] is False

    def test_b4_passes_on_a_converged_pair(self):
        gate = self._gate(self._legs(coarse_ut=0.240, fine_ut=0.245), "B4")
        assert gate["got"] == pytest.approx(0.005 / 0.245, rel=1e-6)
        assert gate["passed"] is True

    def test_b4_not_run_without_a_declared_leg(self):
        legs = self._legs(coarse_ut=0.30, fine_ut=0.24)
        legs["hysing_16"]["declared"] = False
        gate = self._gate(legs, "B4")
        assert gate["got"] is None
        assert gate["passed"] is False
        assert gate["status"] == "not_run"
