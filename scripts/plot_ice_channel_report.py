"""Generate figures for the ice-channel cross-comparison report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from tensorfvm.ice_channel import IceChannelConfig, analytic_u, solve_noslip_fvm, solve_slip_1d


def plot_velocity_profiles(output_dir: Path = Path("artifacts/figures")):
    """Plot velocity profiles for no-slip and slip cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    # No-slip case
    cfg_ns = IceChannelConfig(top_bc="no_slip")
    result_ns = solve_noslip_fvm(cfg_ns, ny=12)
    u_ns, _ = result_ns.cell_center_velocity()
    y_ns = result_ns.y
    x_ns = result_ns.x
    # Take mid-channel profile
    mid_idx = len(x_ns) // 2
    u_ns_mid = u_ns[:, mid_idx]

    y_analytic = torch.linspace(0, cfg_ns.height_m, 100, dtype=torch.float64)
    u_ns_exact = analytic_u(y_analytic, cfg_ns)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # No-slip profile
    axes[0].plot(u_ns_mid.numpy(), y_ns.numpy(), 'bo-', label='FVM (ny=12)', markersize=4)
    axes[0].plot(u_ns_exact.numpy(), y_analytic.numpy(), 'r-', label='Analytic', linewidth=2)
    axes[0].set_xlabel('u (m/s)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title('No-slip top (frozen ice)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Slip case
    cfg_sl = IceChannelConfig(top_bc="slip")
    result_sl = solve_slip_1d(cfg_sl, ny=32)
    u_sl = result_sl["u"]
    y_sl = result_sl["y"]
    u_sl_exact = analytic_u(y_sl, cfg_sl)

    axes[1].plot(u_sl.numpy(), y_sl.numpy(), 'go-', label='FV 1D (ny=32)', markersize=4)
    axes[1].plot(u_sl_exact.numpy(), y_sl.numpy(), 'r--', label='Analytic', linewidth=2)
    axes[1].set_xlabel('u (m/s)')
    axes[1].set_ylabel('y (m)')
    axes[1].set_title('Slip top (sliding ice)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "velocity_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'velocity_profiles.png'}")


def plot_grid_convergence(output_dir: Path = Path("artifacts/figures")):
    """Plot grid convergence for slip case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    cfg = IceChannelConfig(top_bc="slip")
    ny_list = [8, 16, 32, 64]
    errors = []

    for ny in ny_list:
        result = solve_slip_1d(cfg, ny=ny)
        y = result["y"]
        u = result["u"]
        u_exact = analytic_u(y, cfg)
        err = float(torch.linalg.norm(u - u_exact) / torch.linalg.norm(u_exact))
        errors.append(err)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ny_list, errors, 'bo-', linewidth=2, markersize=8)
    # Reference slope (2nd order)
    ny_ref = torch.tensor(ny_list, dtype=torch.float64)
    err_ref = errors[0] * (ny_list[0] / ny_ref) ** 2
    ax.loglog(ny_list, err_ref, 'r--', label='O(h^2) reference', linewidth=1.5)
    ax.set_xlabel('Number of cells (ny)')
    ax.set_ylabel('Velocity L2 relative error')
    ax.set_title('Grid Convergence: Slip-top Channel')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / "grid_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'grid_convergence.png'}")


def plot_cross_comparison(output_dir: Path = Path("artifacts/figures")):
    """Plot FVM vs LBM comparison (bar chart of errors)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # FVM results
    fvm_data = {
        "ny=12": {"u_l2": 0.00635, "dpdx": 0.0137},
    }

    # LBM results (from extended reference)
    lbm_data = {
        "ny=48": {"u_l2": 0.00738, "dpdx": 0.00695},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity L2 error
    categories = list(fvm_data.keys()) + list(lbm_data.keys())
    u_errors = [fvm_data[k]["u_l2"] for k in fvm_data] + [lbm_data[k]["u_l2"] for k in lbm_data]
    colors = ['blue'] * len(fvm_data) + ['green'] * len(lbm_data)

    bars = axes[0].bar(categories, u_errors, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.03, color='red', linestyle='--', label='3% threshold')
    axes[0].set_ylabel('Velocity L2 relative error')
    axes[0].set_title('Velocity Profile Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Pressure gradient error
    dpdx_errors = [fvm_data[k]["dpdx"] for k in fvm_data] + [lbm_data[k]["dpdx"] for k in lbm_data]
    bars = axes[1].bar(categories, dpdx_errors, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0.03, color='red', linestyle='--', label='3% threshold')
    axes[1].set_ylabel('Pressure gradient relative error')
    axes[1].set_title('Pressure Gradient Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "cross_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'cross_comparison.png'}")


if __name__ == "__main__":
    plot_velocity_profiles()
    plot_grid_convergence()
    plot_cross_comparison()
    print("All figures generated.")
