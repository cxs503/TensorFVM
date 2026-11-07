"""Simplified hull wake baseline for FVM cross-comparison.

Physical setup
--------------
A 2-D simplified hull cross-section: a flat-bottom rectangular body (barge-like)
in a channel. The body has:
  - Length L_body = 0.06 m (half the channel length)
  - Draft D = 0.008 m (40% of channel height)
  - Positioned at x = 0.02 to 0.08 m, y = 0 to D

Channel: H = 0.02 m, L = 0.16 m (L/H = 8 for wake development)
Inlet: uniform velocity U = 0.001 m/s
Outlet: zero gauge pressure
Walls: no-slip (top, bottom, and body surface)
Re_H = 2.0 (laminar)

The wake behind the body is the quantity of interest: velocity deficit profile
at several downstream stations.

Method
------
Uses the existing SimpleSolver with cylinder_radius=None and a stair-step
body representation via the fluid mask. Since SimpleSolver only supports a
circular cylinder, we approximate the hull as a flat plate (zero thickness)
at y = D/2 to create a wake, or use the body_fitted module if available.

For this baseline, we use a simpler approach: a confined flat plate at mid-height
creating a symmetric wake. This is a standard benchmark for wake development.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .solver import SimpleSolver, SolverConfig


@dataclass(frozen=True)
class HullWakeConfig:
    """Configuration for simplified hull wake baseline."""
    channel_height_m: float = 0.02
    channel_length_m: float = 0.16
    body_length_m: float = 0.06
    body_draft_m: float = 0.008
    body_x_start_m: float = 0.02
    density_kg_m3: float = 1000.0
    kinematic_viscosity_m2_s: float = 1e-5
    inlet_velocity_m_s: float = 0.001
    nx: int = 80
    ny: int = 10

    @property
    def reynolds(self) -> float:
        return self.inlet_velocity_m_s * self.channel_height_m / self.kinematic_viscosity_m2_s


def run_hull_wake(cfg: HullWakeConfig, output_dir: Path) -> dict:
    """Run the hull wake baseline.

    Since SimpleSolver only supports circular cylinders, we approximate the
    hull wake using a confined cylinder at the body location. The cylinder
    radius is chosen to match the body draft.
    """
    # Use a cylinder to represent the hull cross-section
    # Cylinder at mid-channel, radius = draft/2
    cylinder_x = cfg.body_x_start_m + cfg.body_length_m / 2
    cylinder_y = cfg.channel_height_m / 2
    cylinder_r = cfg.body_draft_m / 2

    sc = SolverConfig(
        nx=cfg.nx, ny=cfg.ny,
        length=cfg.channel_length_m, height=cfg.channel_height_m,
        cylinder_x=cylinder_x, cylinder_y=cylinder_y,
        cylinder_radius=cylinder_r,
        inlet_velocity=cfg.inlet_velocity_m_s,
        density=cfg.density_kg_m3,
        reynolds=cfg.reynolds,
        tolerance=1e-6,
        max_iterations=2000,
    )
    result = SimpleSolver(sc).solve()

    # Extract wake profiles at downstream stations
    u, _ = result.cell_center_velocity()
    x = result.x
    y = result.y

    # Wake stations: behind the body at x = 0.10, 0.12, 0.14 m
    stations = [0.10, 0.12, 0.14]
    wake_profiles = {}
    for xs in stations:
        idx = int(torch.argmin(torch.abs(x - xs)).item())
        profile = {
            "x_m": float(x[idx]),
            "y_m": y.tolist(),
            "u_m_s": u[:, idx].tolist(),
            "deficit_m_s": (cfg.inlet_velocity_m_s - u[:, idx]).tolist(),
        }
        wake_profiles[f"station_{xs:.2f}m"] = profile

    # Compute drag (approximate from pressure + shear on cylinder)
    # For this baseline, we report the wake deficit as the primary quantity
    max_deficit = max(
        float(max(p["deficit_m_s"])) for p in wake_profiles.values()
    )

    metrics = {
        "converged": result.converged,
        "iterations": len(result.history),
        "max_wake_deficit_m_s": max_deficit,
        "reynolds_number": cfg.reynolds,
        "wake_stations": list(wake_profiles.keys()),
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "wake_profiles.json").write_text(json.dumps(wake_profiles, indent=2) + "\n")

    return metrics


def run_all(output_dir: Path = Path("artifacts/hull_wake")) -> dict:
    """Run the hull wake baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = HullWakeConfig()
    metrics = run_hull_wake(cfg, output_dir)

    summary = {
        "schema": "tensorfvm.hull-wake-baseline/1",
        "solver": "TensorFVM SimpleSolver (MAC-SIMPLE with cylinder)",
        "physical_setup": {
            "description": "Simplified hull wake: confined cylinder representing barge cross-section",
            "channel_height_m": cfg.channel_height_m,
            "channel_length_m": cfg.channel_length_m,
            "body_draft_m": cfg.body_draft_m,
            "inlet_velocity_m_s": cfg.inlet_velocity_m_s,
            "reynolds_number": cfg.reynolds,
            "flow_regime": "laminar",
        },
        "turbulence_model": "laminar",
        "metrics": metrics,
        "passed": metrics["converged"] and metrics["max_wake_deficit_m_s"] > 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    torch.set_num_threads(1)
    s = run_all()
    print(json.dumps({"passed": s["passed"], "metrics": s["metrics"]}, indent=2))
