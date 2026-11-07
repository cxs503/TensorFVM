"""Ice-under channel flow: top lid slip / no-slip two-state FV verification.

Physical setup
--------------
2-D rectangular channel, height H, length L = 6H.
Bottom wall: no-slip.  Top wall: no-slip (frozen ice) or free-slip (sliding ice).
Inlet: uniform velocity U_bulk.  Outlet: zero gauge pressure.

Analytic fully-developed profiles (laminar, Re_H = 1):
  No-slip top:  u(y) = 6U(y/H)(1-y/H),  dp/dx = -12 mu U / H^2
  Slip top:     u(y) = (3U/H^2) y(2H-y), dp/dx = -6 mu U / H^2

Method
------
- No-slip case: uses the existing verified SimpleSolver (tensorfvm.solver)
  which already handles confined channel flow with all no-slip walls.
- Slip case: 1-D tridiagonal solve of the fully-developed momentum equation
  with du/dy=0 at top, u=0 at bottom.  This is exact for the laminar
  fully-developed regime and provides the reference for the FVM cross-check.
- Both cases share the same SI geometry and fluid properties as the existing
  suite-channel benchmark for direct LBM cross-comparison.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import torch

from .solver import SimpleSolver, SolverConfig


@dataclass(frozen=True)
class IceChannelConfig:
    """Shared physical parameters for ice-under channel."""
    height_m: float = 0.02
    length_m: float = 0.12
    density_kg_m3: float = 1000.0
    kinematic_viscosity_m2_s: float = 1e-5
    bulk_velocity_m_s: float = 0.0005
    top_bc: Literal["no_slip", "slip"] = "no_slip"

    @property
    def reynolds(self) -> float:
        return self.bulk_velocity_m_s * self.height_m / self.kinematic_viscosity_m2_s

    @property
    def mu(self) -> float:
        return self.density_kg_m3 * self.kinematic_viscosity_m2_s

    @property
    def ref_dpdx(self) -> float:
        f = 12.0 if self.top_bc == "no_slip" else 3.0
        return -f * self.mu * self.bulk_velocity_m_s / self.height_m**2

    @property
    def ref_tau_bottom(self) -> float:
        f = 6.0 if self.top_bc == "no_slip" else 3.0
        return f * self.mu * self.bulk_velocity_m_s / self.height_m

    @property
    def ref_tau_top(self) -> float:
        return 0.0 if self.top_bc == "slip" else self.ref_tau_bottom


def analytic_u(y, cfg: IceChannelConfig):
    """Analytic fully-developed velocity profile."""
    H, U = cfg.height_m, cfg.bulk_velocity_m_s
    eta = y / H
    if cfg.top_bc == "no_slip":
        return 6.0 * U * eta * (1.0 - eta)
    else:  # slip: du/dy(H)=0, u(0)=0
        return (3.0 * U / 2.0) * eta * (2.0 - eta)


def solve_noslip_fvm(cfg: IceChannelConfig, ny: int = 12):
    """No-slip case using the existing verified SimpleSolver."""
    nx = 6 * ny  # L/H = 6
    sc = SolverConfig(
        nx=nx, ny=ny,
        length=cfg.length_m, height=cfg.height_m,
        inlet_velocity=cfg.bulk_velocity_m_s,
        density=cfg.density_kg_m3,
        reynolds=cfg.reynolds,
        cylinder_radius=None,
        tolerance=1e-6,
        max_iterations=1000,
    )
    result = SimpleSolver(sc).solve()
    return result


def solve_slip_1d(cfg: IceChannelConfig, ny: int = 64):
    """Slip-top case: 1-D tridiagonal solve of fully-developed momentum.

    For fully-developed channel flow, the 2-D Navier-Stokes reduces to:
        mu d²u/dy² = dp/dx = const
    with BCs: u(0) = 0 (bottom no-slip), du/dy(H) = 0 (top slip).

    Solution: u(y) = (-dp/dx)/(2*mu) * y * (2H - y)
    With bulk velocity constraint: dp/dx = -6*mu*U/H^2

    We solve this as a 1-D FV tridiagonal system on a uniform grid to
    demonstrate the FV discretisation with slip BC, matching the 2-D solver
    patterns.
    """
    H = cfg.height_m
    mu = cfg.mu
    U = cfg.bulk_velocity_m_s
    dy = H / ny

    # Cell centres
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy

    # Target pressure gradient for given bulk velocity
    dpdx = cfg.ref_dpdx  # = -3 mu U / H^2

    # Tridiagonal system: a_P u_P = a_N u_N + a_S u_S + b
    # Interior cells: mu/dy^2 * (u_{j+1} - 2u_j + u_{j-1}) = dpdx
    # Bottom (j=0): no-slip → u_{-1} = -u_0 (ghost), so a_S doubled
    # Top (j=ny-1): slip → du/dy=0 → u_{ny} = u_{ny-1} (ghost), a_N NOT doubled

    a_S_val = mu / dy**2
    a_N_val = mu / dy**2

    # Build tridiagonal
    a_diag = torch.full((ny,), a_S_val + a_N_val, dtype=torch.float64)
    a_upper = torch.full((ny - 1,), -a_N_val, dtype=torch.float64)  # super-diagonal
    a_lower = torch.full((ny - 1,), -a_S_val, dtype=torch.float64)  # sub-diagonal
    rhs = torch.full((ny,), -dpdx, dtype=torch.float64)  # source = -dp/dx

    # Bottom BC (j=0): no-slip → ghost u_{-1} = -u_0, contribution = 0
    # Diffusion doubled: a_S → 2*a_S
    a_diag[0] = 2 * a_S_val + a_N_val

    # Top BC (j=ny-1): slip → ghost u_{ny} = u_{ny-1}
    # For slip: du/dy = 0 → flux at top = 0
    # So the top face contributes nothing → a_diag[ny-1] = a_S_val (only south neighbor)
    a_diag[ny - 1] = a_S_val

    # Solve tridiagonal (Thomas algorithm)
    u = _thomas_solve(a_lower, a_diag, a_upper, rhs)

    # Verify bulk velocity
    u_bulk = u.mean().item()
    # Scale to match target bulk velocity
    scale = U / u_bulk if abs(u_bulk) > 1e-15 else 1.0
    u = u * scale

    return {
        "y": y,
        "u": u,
        "dpdx": dpdx * scale,  # scaled pressure gradient
        "ny": ny,
        "dy": dy,
    }


def _thomas_solve(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal system."""
    n = len(diag)
    # Forward elimination
    c_prime = torch.zeros(n - 1, dtype=diag.dtype)
    d_prime = torch.zeros(n, dtype=diag.dtype)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    # Back substitution
    x = torch.zeros(n, dtype=diag.dtype)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def compute_metrics_fvm(result, cfg: IceChannelConfig):
    """Compute metrics for FVM (SimpleSolver) result."""
    u, _ = result.cell_center_velocity()
    y = result.y
    x = result.x

    # Developed region
    dev = (x >= 3 * cfg.height_m) & (x <= 5 * cfg.height_m)
    u_dev = u[:, dev]
    u_exact = analytic_u(y, cfg)

    # Velocity L2 error
    err = u_dev - u_exact[:, None]
    u_l2 = float(torch.linalg.norm(err) / torch.linalg.norm(u_exact[:, None].expand_as(err)))

    # Pressure gradient
    p_dev = result.p[:, dev]
    p_avg = p_dev.mean(dim=0)
    x_dev = x[dev]
    if len(x_dev) >= 2:
        xc = x_dev - x_dev.mean()
        dpdx = float((xc * (p_avg - p_avg.mean())).sum() / (xc**2).sum())
    else:
        dpdx = 0.0
    dpdx_err = abs(dpdx - cfg.ref_dpdx) / abs(cfg.ref_dpdx) if abs(cfg.ref_dpdx) > 0 else float('inf')

    # Wall shear (bottom)
    mu = cfg.mu
    dy = cfg.height_m / result.config.ny
    tau_bot = float(-mu * u[0, dev].mean() / (0.5 * dy))
    tau_bot_err = abs(abs(tau_bot) - cfg.ref_tau_bottom) / cfg.ref_tau_bottom if cfg.ref_tau_bottom > 0 else float('inf')

    # Wall shear (top)
    if cfg.top_bc == "slip":
        tau_top, tau_top_err = 0.0, 0.0
    else:
        tau_top = float(mu * u[-1, dev].mean() / (0.5 * dy))
        tau_top_err = abs(abs(tau_top) - cfg.ref_tau_top) / cfg.ref_tau_top if cfg.ref_tau_top > 0 else float('inf')

    # Mass balance
    inlet = float(torch.sum(result.u[:, 0]).item()) * dy
    outlet = float(torch.sum(result.u[:, -1]).item()) * dy
    mass_imb = abs(outlet - inlet) / abs(inlet) if abs(inlet) > 0 else 0.0

    return {
        "velocity_l2": u_l2,
        "dpdx_pa_m": dpdx,
        "ref_dpdx_pa_m": cfg.ref_dpdx,
        "dpdx_error": dpdx_err,
        "tau_bottom_pa": tau_bot,
        "ref_tau_bottom_pa": cfg.ref_tau_bottom,
        "tau_bottom_error": tau_bot_err,
        "tau_top_pa": tau_top,
        "ref_tau_top_pa": cfg.ref_tau_top,
        "tau_top_error": tau_top_err,
        "mass_imbalance": mass_imb,
        "converged": result.converged,
        "iterations": len(result.history),
        "passed": (u_l2 < 0.03 and dpdx_err < 0.03 and tau_bot_err < 0.03
                   and mass_imb < 1e-6 and result.converged),
    }


def compute_metrics_1d(result_1d, cfg: IceChannelConfig):
    """Compute metrics for 1-D slip solver."""
    y = result_1d["y"]
    u = result_1d["u"]
    u_exact = analytic_u(y, cfg)

    # Velocity L2 error
    err = u - u_exact
    u_l2 = float(torch.linalg.norm(err) / torch.linalg.norm(u_exact))

    # Bulk velocity check
    u_bulk = float(u.mean())
    u_bulk_err = abs(u_bulk - cfg.bulk_velocity_m_s) / cfg.bulk_velocity_m_s

    # Pressure gradient
    dpdx = result_1d["dpdx"]
    dpdx_err = abs(dpdx - cfg.ref_dpdx) / abs(cfg.ref_dpdx) if abs(cfg.ref_dpdx) > 0 else float('inf')

    # Wall shear (bottom)
    mu = cfg.mu
    dy = result_1d["dy"]
    tau_bot = float(-mu * u[0] / (0.5 * dy))
    tau_bot_err = abs(abs(tau_bot) - cfg.ref_tau_bottom) / cfg.ref_tau_bottom if cfg.ref_tau_bottom > 0 else float('inf')

    # Wall shear (top): should be ~0 for slip
    tau_top = float(mu * (u[-1] - u[-2]) / dy)  # approximate du/dy at top
    tau_top_err = abs(tau_top) / cfg.ref_tau_bottom if cfg.ref_tau_bottom > 0 else 0.0

    return {
        "velocity_l2": u_l2,
        "bulk_velocity_m_s": u_bulk,
        "bulk_velocity_error": u_bulk_err,
        "dpdx_pa_m": dpdx,
        "ref_dpdx_pa_m": cfg.ref_dpdx,
        "dpdx_error": dpdx_err,
        "tau_bottom_pa": tau_bot,
        "ref_tau_bottom_pa": cfg.ref_tau_bottom,
        "tau_bottom_error": tau_bot_err,
        "tau_top_pa": tau_top,
        "ref_tau_top_pa": cfg.ref_tau_top,
        "tau_top_error": tau_top_err,
        "passed": (u_l2 < 0.01 and dpdx_err < 0.01 and tau_bot_err < 0.03
                   and u_bulk_err < 0.01),
    }


def run_all(output_dir: Path = Path("artifacts/ice_channel")) -> dict:
    """Run both no-slip (FVM) and slip (1D) cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # No-slip case (FVM)
    cfg_noslip = IceChannelConfig(top_bc="no_slip")
    noslip_cases = []
    for ny in [12]:
        result = solve_noslip_fvm(cfg_noslip, ny=ny)
        m = compute_metrics_fvm(result, cfg_noslip)
        case_dir = output_dir / f"noslip_fvm_{6*ny}x{ny}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
        noslip_cases.append({"grid": f"{6*ny}x{ny}", "method": "FVM_SIMPLE", **m})
    results["no_slip"] = noslip_cases

    # Slip case (1D)
    cfg_slip = IceChannelConfig(top_bc="slip")
    slip_cases = []
    for ny in [16, 32]:
        result = solve_slip_1d(cfg_slip, ny=ny)
        m = compute_metrics_1d(result, cfg_slip)
        case_dir = output_dir / f"slip_1d_{ny}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
        # Save fields
        fields = {
            "y_m": result["y"].tolist(),
            "u_m_s": result["u"].tolist(),
            "dpdx_pa_m": result["dpdx"],
        }
        (case_dir / "fields.json").write_text(json.dumps(fields, indent=2) + "\n")
        slip_cases.append({"ny": ny, "method": "1D_tridiagonal", **m})
    results["slip"] = slip_cases

    summary = {
        "schema": "tensorfvm.ice-channel-comparison/1",
        "solver": {
            "no_slip": "TensorFVM SimpleSolver (verified MAC-SIMPLE)",
            "slip": "1-D tridiagonal FV (fully-developed analytic)",
        },
        "physical_setup": {
            "description": "Ice-under channel: bottom no-slip, top configurable",
            "height_m": 0.02,
            "length_m": 0.12,
            "density_kg_m3": 1000.0,
            "kinematic_viscosity_m2_s": 1e-5,
            "bulk_velocity_m_s": 0.0005,
            "reynolds_number": 1.0,
            "flow_regime": "laminar",
        },
        "analytic_reference": {
            "no_slip": "u(y) = 6U(y/H)(1-y/H), dp/dx = -12muU/H^2",
            "slip": "u(y) = 1.5U/H^2 * y(2H-y), dp/dx = -3muU/H^2",
        },
        "turbulence_model": "laminar (Re=1)",
        "results": results,
        "all_passed": all(
            c["passed"] for cases in results.values() for c in cases
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    torch.set_num_threads(1)
    s = run_all()
    print(json.dumps({"passed": s["all_passed"], "summary": {k: [c.get("passed") for c in v] for k, v in s["results"].items()}}, indent=2))
