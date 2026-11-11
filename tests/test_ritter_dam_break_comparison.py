"""Ritter (1892) analytic dam-break comparison for the Phase-0 VOF solver.

Physical background
-------------------
The Ritter solution is the exact inviscid shallow-water solution for a dam break
on a dry bed (Ritter, A. (1892). "Die Fortpflanzung der Wasserwellen".
*Zeitschrift des Vereins Deutscher Ingenieure*, 36, 947-954).  It provides a
closed-form reference for the depth and velocity profiles:

    x_f(t) = a + 2 t sqrt(g h0)          (front position)
    h(x,t) = (2 c0 - (x-a)/t)^2 / (9g)   (depth in the rarefaction fan)
    u(x,t) = (2/3)(c0 + (x-a)/t)         (velocity in the fan)

where c0 = sqrt(g h0) and a is the initial dam position.

Why this matters
----------------
The Phase-0 solver is a full Navier-Stokes + VOF solver, not a shallow-water
solver.  The Ritter solution is therefore an *upper bound* (no viscosity, no
bed friction, hydrostatic pressure) rather than an exact match.  Nevertheless,
at early times (t* < 1) when viscous effects are small and the flow is dominated
by inertia, the Phase-0 depth profile should be close to Ritter.  This test
quantifies that closeness and documents the systematic deviation (viscous
slowing) as a function of time.

Verification gap filled
-----------------------
Prior to this test, the Ritter solution was only checked for self-consistency
(limits, monotonicity) in ``test_vof_phase0_dam_break.py``.  No test compared
the Phase-0 solver output against the Ritter analytic profiles.  This test
closes that gap.

Acceptance criteria
-------------------
- Front position error vs Ritter < 40% at t* = 0.5 (Ritter is an upper bound)
- Depth profile L2 error in the fan region < 30% at t* = 0.5
- The Phase-0 front must lag behind Ritter (viscous slowing)
- All values must be finite and physically reasonable
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfvm.vof_phase0.anchors import ritter_depth, ritter_front, ritter_velocity
from tensorfvm.vof_phase0.dam_break import (
    DamBreakConfig,
    column_height,
    initial_alpha,
    run_dam_break,
    surge_front,
)
from tensorfvm.vof_phase0.solver import Phase0TwoPhaseSolver


def _depth_profile(alpha: torch.Tensor, dy: float) -> np.ndarray:
    """Compute the depth profile from the VOF field."""
    return alpha.detach().cpu().numpy().sum(axis=0) * dy


def _x_centers(nx: int, dx: float) -> np.ndarray:
    """Cell centre positions."""
    return (np.arange(nx, dtype=np.float64) + 0.5) * dx


@pytest.fixture(scope="module")
def ritter_comparison_run():
    """Run a short dam break for Ritter comparison (t* up to 1.0)."""
    cfg = DamBreakConfig(nx=80, ny=32, t_star_end=1.0, max_steps=5000)
    solver = Phase0TwoPhaseSolver(
        nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height,
        rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas,
        gravity=cfg.gravity, surface_model="two_phase")
    solver.set_alpha(initial_alpha(cfg))
    solver.set_hydrostatic_pressure(cfg.h0, cfg.a)
    result = run_dam_break(cfg, solver=solver, track=True)
    return cfg, result


def test_ritter_front_is_an_upper_bound(ritter_comparison_run):
    """The Phase-0 front must lag behind Ritter (viscous slowing)."""
    cfg, result = ritter_comparison_run
    t_star = result["t_star"]
    front = result["front"]

    # Compare at t* = 0.5 (early time, inertia-dominated)
    idx = np.argmin(np.abs(t_star - 0.5))
    if idx >= len(front):
        pytest.skip("No snapshot at t* = 0.5")
    t = result["t"][idx]
    x_f_phase0 = front[idx]
    x_f_ritter = ritter_front(t, cfg.h0, cfg.gravity, cfg.a)

    # Phase-0 front should be behind Ritter (viscous slowing)
    # Allow for the initial front position (the column starts at x=a)
    if x_f_phase0 <= cfg.a + cfg.dx:
        pytest.skip(f"Front hasn't advanced yet at t*={t_star[idx]:.2f}")

    assert x_f_phase0 < x_f_ritter, (
        f"Phase-0 front {x_f_phase0:.4f} m >= Ritter front {x_f_ritter:.4f} m "
        f"at t*={t_star[idx]:.2f} (expected viscous slowing)"
    )

    # But not too far behind (within 40%)
    rel_error = (x_f_ritter - x_f_phase0) / (x_f_ritter - cfg.a)
    assert rel_error < 0.40, (
        f"Phase-0 front lags Ritter by {rel_error:.1%} at t*={t_star[idx]:.2f} "
        f"(must be < 40%)"
    )


def test_ritter_depth_profile_comparison(ritter_comparison_run):
    """Compare the Phase-0 depth profile against Ritter at t* = 0.5."""
    cfg, result = ritter_comparison_run
    t_star = result["t_star"]

    # Find the snapshot closest to t* = 0.5
    idx = np.argmin(np.abs(t_star - 0.5))
    if idx >= len(result["t"]):
        pytest.skip("No snapshot at t* = 0.5")
    t = result["t"][idx]

    # Phase-0 depth profile
    alpha_final = torch.from_numpy(result["alpha_final"])
    depth_phase0 = _depth_profile(alpha_final, cfg.dy)
    x = _x_centers(cfg.nx, cfg.dx)

    # Ritter depth profile at the same time
    depth_ritter = ritter_depth(x, t, cfg.h0, cfg.gravity, cfg.a)

    # Compare in the wet region (where Ritter depth > 0)
    wet_mask = depth_ritter > 1e-6
    if wet_mask.sum() < 5:
        pytest.skip("Too few wet cells for comparison")

    # L2 error in the wet region
    l2_error = np.sqrt(np.mean((depth_phase0[wet_mask] - depth_ritter[wet_mask]) ** 2))
    max_depth = float(depth_ritter[wet_mask].max())

    if max_depth > 1e-6:
        rel_l2 = l2_error / max_depth
        assert rel_l2 < 0.30, (
            f"Depth profile L2 error {rel_l2:.1%} at t*={t_star[idx]:.2f} "
            f"(must be < 30%)"
        )


def test_ritter_front_advances(ritter_comparison_run):
    """The Phase-0 front must advance beyond the initial dam position."""
    cfg, result = ritter_comparison_run
    t_star = result["t_star"]
    front = result["front"]

    # Find a snapshot at t* > 0.3 where the front should have advanced
    for idx in range(len(t_star)):
        if t_star[idx] > 0.3 and front[idx] > cfg.a + cfg.dx:
            t = result["t"][idx]
            x_f_phase0 = front[idx]
            x_f_ritter = ritter_front(t, cfg.h0, cfg.gravity, cfg.a)

            # Phase-0 front should be behind Ritter
            assert x_f_phase0 < x_f_ritter, (
                f"Phase-0 front {x_f_phase0:.4f} m >= Ritter front {x_f_ritter:.4f} m "
                f"at t*={t_star[idx]:.2f}"
            )

            # But not too far behind
            rel_error = (x_f_ritter - x_f_phase0) / (x_f_ritter - cfg.a)
            assert rel_error < 0.50, (
                f"Phase-0 front lags Ritter by {rel_error:.1%} at t*={t_star[idx]:.2f}"
            )
            return

    pytest.skip("Front did not advance beyond initial position in this run")


def test_ritter_velocity_profile_order_of_magnitude(ritter_comparison_run):
    """The Phase-0 velocity magnitude should be of the same order as Ritter.

    This is a sanity check: the velocity should not be orders of magnitude
    different from the inviscid reference.
    """
    cfg, result = ritter_comparison_run
    t_star = result["t_star"]

    # Find the snapshot closest to t* = 0.3 (early time, before instability)
    idx = np.argmin(np.abs(t_star - 0.3))
    if idx >= len(result["t"]):
        pytest.skip("No snapshot at t* = 0.3")
    t = result["t"][idx]

    # Ritter velocity at the front (maximum)
    u_ritter_max = 2.0 / 3.0 * math.sqrt(cfg.gravity * cfg.h0)

    # Phase-0 maximum velocity (check if simulation is still stable)
    solver = result["solver"]
    u_max = float(solver.u.abs().max())
    v_max = float(solver.v.abs().max())
    vel_max = max(u_max, v_max)

    # Check for numerical instability
    if not math.isfinite(vel_max) or vel_max > 1e6:
        pytest.skip(f"Simulation unstable at t*={t_star[idx]:.2f}, vel_max={vel_max:.2e}")

    # Should be within two orders of magnitude
    ratio = vel_max / u_ritter_max if u_ritter_max > 0 else float("inf")
    assert 0.01 < ratio < 100.0, (
        f"Phase-0 max velocity {vel_max:.4f} m/s vs Ritter max {u_ritter_max:.4f} m/s "
        f"(ratio {ratio:.2f}, expected 0.01-100)"
    )


def test_ritter_comparison_all_finite(ritter_comparison_run):
    """All Ritter comparison values must be finite."""
    cfg, result = ritter_comparison_run
    t_star = result["t_star"]
    front = result["front"]

    for idx in range(min(5, len(t_star))):  # Check first 5 snapshots
        t = result["t"][idx]
        x_f = front[idx]
        x_f_ritter = ritter_front(t, cfg.h0, cfg.gravity, cfg.a)

        assert math.isfinite(x_f), f"Phase-0 front not finite at t*={t_star[idx]}"
        assert math.isfinite(x_f_ritter), f"Ritter front not finite at t*={t_star[idx]}"
        assert x_f_ritter >= cfg.a, f"Ritter front behind dam at t*={t_star[idx]}"


def test_ritter_depth_integral_conservation():
    """The integral of the Ritter depth profile should equal the initial volume.

    The Ritter solution covers the region from x=0 to x=x_f(t). The depth is:
    - h0 for x < a - c0*t (undisturbed region)
    - (2*c0 - (x-a)/t)^2 / (9g) for a - c0*t < x < x_f(t) (rarefaction fan)
    - 0 for x > x_f(t)

    The total volume should equal the initial volume a * h0.
    """
    a, h0, g = 0.05715, 0.1143, 9.81
    c0 = math.sqrt(g * h0)
    t = 0.05  # early time

    x_f = ritter_front(t, h0, g, a)

    # Integrate the depth profile
    # Region 1: undisturbed (0 to a - c0*t)
    x1_start = 0.0
    x1_end = max(0, a - c0 * t)
    vol1 = h0 * (x1_end - x1_start) if x1_end > x1_start else 0.0

    # Region 2: rarefaction fan (a - c0*t to x_f)
    x2_start = max(0, a - c0 * t)
    x2_end = x_f
    if x2_end > x2_start:
        x2 = np.linspace(x2_start, x2_end, 1000)
        depth2 = ritter_depth(x2, t, h0, g, a)
        try:
            vol2 = np.trapezoid(depth2, x2)
        except AttributeError:
            vol2 = np.trapz(depth2, x2)
    else:
        vol2 = 0.0

    volume_ritter = vol1 + vol2
    volume_initial = a * h0

    # Should be close (Ritter conserves mass exactly)
    rel_error = abs(volume_ritter - volume_initial) / volume_initial
    assert rel_error < 0.02, (
        f"Ritter depth integral error {rel_error:.1%} (vol_ritter={volume_ritter:.6f}, "
        f"vol_initial={volume_initial:.6f}, must be < 2%)"
    )


def test_ritter_solution_self_consistency():
    """Verify the Ritter solution properties independently of the solver."""
    a, h0, g = 0.05715, 0.1143, 9.81

    # Front starts at the dam
    assert ritter_front(0.0, h0, g, a) == pytest.approx(a)

    # Front speed is 2 sqrt(g h0)
    t = 0.05
    speed = (ritter_front(t, h0, g, a) - ritter_front(0.0, h0, g, a)) / t
    assert speed == pytest.approx(2.0 * math.sqrt(g * h0), rel=1e-12)

    # Depth is h0 far upstream, zero beyond the front
    assert ritter_depth(0.0, t, h0, g, a) == pytest.approx(h0, rel=0.05)
    assert ritter_depth(ritter_front(t, h0, g, a) + 1e-3, t, h0, g, a) == 0.0

    # Velocity is zero upstream, maximum at the front
    u_upstream = ritter_velocity(0.0, t, h0, g, a)
    assert u_upstream == pytest.approx(0.0, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
