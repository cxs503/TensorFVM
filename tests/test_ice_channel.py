"""Pytest tests for ice-under channel flow (slip / no-slip two-state)."""

import json
import pytest
import torch

from tensorfvm.ice_channel import (
    IceChannelConfig,
    analytic_u,
    solve_noslip_fvm,
    solve_slip_1d,
    compute_metrics_fvm,
    compute_metrics_1d,
    run_all,
)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)
    yield


class TestAnalyticProfiles:
    """Verify analytic velocity profiles satisfy BCs and bulk constraint."""

    def test_noslip_profile_bc(self):
        cfg = IceChannelConfig(top_bc="no_slip")
        y = torch.linspace(0, cfg.height_m, 100, dtype=torch.float64)
        u = analytic_u(y, cfg)
        assert abs(float(u[0])) < 1e-12, "Bottom no-slip violated"
        assert abs(float(u[-1])) < 1e-12, "Top no-slip violated"

    def test_slip_profile_bc(self):
        cfg = IceChannelConfig(top_bc="slip")
        y = torch.linspace(0, cfg.height_m, 100, dtype=torch.float64)
        u = analytic_u(y, cfg)
        assert abs(float(u[0])) < 1e-12, "Bottom no-slip violated"
        # Top: du/dy = 0 analytically. Check at y=H exactly.
        u_top = analytic_u(torch.tensor([cfg.height_m], dtype=torch.float64), cfg)
        u_near_top = analytic_u(torch.tensor([cfg.height_m - 1e-8], dtype=torch.float64), cfg)
        dudY_top = (float(u_top[0]) - float(u_near_top[0])) / 1e-8
        assert abs(dudY_top) < 1e-3, f"Top slip (du/dy=0) violated: {dudY_top}"

    def test_noslip_bulk_velocity(self):
        cfg = IceChannelConfig(top_bc="no_slip")
        y = torch.linspace(0.0001, cfg.height_m - 0.0001, 1000, dtype=torch.float64)
        u = analytic_u(y, cfg)
        u_bulk = float(u.mean())
        assert abs(u_bulk - cfg.bulk_velocity_m_s) / cfg.bulk_velocity_m_s < 0.01

    def test_slip_bulk_velocity(self):
        cfg = IceChannelConfig(top_bc="slip")
        y = torch.linspace(0.0001, cfg.height_m - 0.0001, 1000, dtype=torch.float64)
        u = analytic_u(y, cfg)
        u_bulk = float(u.mean())
        assert abs(u_bulk - cfg.bulk_velocity_m_s) / cfg.bulk_velocity_m_s < 0.01

    def test_slip_dpdx_half_of_noslip(self):
        """Slip pressure gradient should be 1/4 of no-slip (not 1/2)."""
        cfg_ns = IceChannelConfig(top_bc="no_slip")
        cfg_sl = IceChannelConfig(top_bc="slip")
        ratio = cfg_ns.ref_dpdx / cfg_sl.ref_dpdx
        assert abs(ratio - 4.0) < 0.01, f"Expected ratio 4, got {ratio}"


class TestSlip1DSolver:
    """Test the 1-D tridiagonal solver for slip-top channel."""

    def test_convergence(self):
        cfg = IceChannelConfig(top_bc="slip")
        result = solve_slip_1d(cfg, ny=32)
        assert result["u"].shape[0] == 32

    def test_velocity_profile_accuracy(self):
        cfg = IceChannelConfig(top_bc="slip")
        result = solve_slip_1d(cfg, ny=64)
        metrics = compute_metrics_1d(result, cfg)
        assert metrics["velocity_l2"] < 0.01, f"u_l2={metrics['velocity_l2']}"

    def test_bulk_velocity(self):
        cfg = IceChannelConfig(top_bc="slip")
        result = solve_slip_1d(cfg, ny=64)
        metrics = compute_metrics_1d(result, cfg)
        assert metrics["bulk_velocity_error"] < 0.01

    def test_pressure_gradient(self):
        cfg = IceChannelConfig(top_bc="slip")
        result = solve_slip_1d(cfg, ny=64)
        metrics = compute_metrics_1d(result, cfg)
        assert metrics["dpdx_error"] < 0.01

    def test_wall_shear_bottom(self):
        cfg = IceChannelConfig(top_bc="slip")
        result = solve_slip_1d(cfg, ny=64)
        metrics = compute_metrics_1d(result, cfg)
        assert metrics["tau_bottom_error"] < 0.03

    def test_grid_convergence(self):
        """Finer grid should give smaller error."""
        cfg = IceChannelConfig(top_bc="slip")
        errors = []
        for ny in [16, 32, 64]:
            result = solve_slip_1d(cfg, ny=ny)
            metrics = compute_metrics_1d(result, cfg)
            errors.append(metrics["velocity_l2"])
        assert errors[0] > errors[1] > errors[2], "No grid convergence"


class TestNoSlipFVM:
    """Test the no-slip case using the existing SimpleSolver."""

    def test_convergence(self):
        cfg = IceChannelConfig(top_bc="no_slip")
        result = solve_noslip_fvm(cfg, ny=12)
        assert result.converged

    def test_velocity_accuracy(self):
        cfg = IceChannelConfig(top_bc="no_slip")
        result = solve_noslip_fvm(cfg, ny=12)
        metrics = compute_metrics_fvm(result, cfg)
        assert metrics["velocity_l2"] < 0.03

    def test_pressure_gradient(self):
        cfg = IceChannelConfig(top_bc="no_slip")
        result = solve_noslip_fvm(cfg, ny=12)
        metrics = compute_metrics_fvm(result, cfg)
        assert metrics["dpdx_error"] < 0.03


class TestRunAll:
    """Integration test: run all cases."""

    def test_all_pass(self, tmp_path):
        summary = run_all(output_dir=tmp_path)
        assert summary["all_passed"], "Not all cases passed"
        assert (tmp_path / "summary.json").exists()

    def test_output_structure(self, tmp_path):
        summary = run_all(output_dir=tmp_path)
        assert "no_slip" in summary["results"]
        assert "slip" in summary["results"]
        assert len(summary["results"]["no_slip"]) >= 1
        assert len(summary["results"]["slip"]) >= 1
