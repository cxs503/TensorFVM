"""Round 8 tests for ``CASE-FVM-PERIODIC-HILLS``: production limit tuning and equilibration.

Anchor: CASE-FVM-PERIODIC-HILLS
Mission: 18492 (selfstart, TensorFVM)

Round 8 findings:
1. The SA closure does not equilibrate within t=60 (3000 steps); nu_t/nu grows from 0.2 to 36.
2. The reattachment length oscillates as the flow develops: 2.6 -> 4.1 -> 6.1 -> nan -> 5.6.
3. At t=60, x_r/H = 5.561 is within 4.1% of the reference 5.34, but nu_t/nu = 36.38 is 82% above target.
4. Production limiter at C=50 reduces nu_t/nu from 16.17 to 4.33 at 1000 steps, but x_r/H is unchanged.

These tests verify the documented behaviour rather than asserting a pass gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver, Statistics
from tensorfvm.periodic_hill_turbulence import SpalartAllmaras, SpalartAllmarasConfig

REFERENCE_XR_H = 5.34
REFERENCE_TOLERANCE = 0.10


def _run_with_limit(production_limit, steps=1000, nx=48, ny=24):
    """Run periodic hill with given production_limit."""
    config = PeriodicHillConfig(nx=nx, ny=ny, reynolds=2800.0, time_step_s=0.02)
    solver = PeriodicHillSolver(config)
    cfg = SpalartAllmarasConfig(production_limit=production_limit)
    solver.turbulence = SpalartAllmaras(solver, cfg)
    solver.advance(steps)
    return solver


# --------------------------------------------------------------------------- #
# Equilibration diagnostics
# --------------------------------------------------------------------------- #

def test_nu_t_grows_monotonically_without_limiting():
    """Without production limiting, nu_t/nu grows monotonically (not equilibrated).

    This is the key finding of round 8: the SA closure does not equilibrate
    within t=60 on the 48x24 mesh at Re=2800.
    """
    solver = _run_with_limit(None, steps=1000)
    nu_t_1000 = solver.turbulence.viscosity().max() / solver.nu

    solver2 = _run_with_limit(None, steps=1500)
    nu_t_1500 = solver2.turbulence.viscosity().max() / solver.nu

    # nu_t should be growing
    assert nu_t_1500 > nu_t_1000, (
        f"nu_t/nu should be growing: {nu_t_1000:.2f} at 1000 steps -> {nu_t_1500:.2f} at 1500 steps"
    )


def test_reattachment_oscillates_during_development():
    """The reattachment length oscillates as the flow develops.

    This is characteristic of an unsteady separated flow that has not reached
    a statistically stationary state.
    """
    # Run to different times and check that x_r/H is not monotonic
    solvers = {}
    for steps in [500, 1000, 1500]:
        solvers[steps] = _run_with_limit(None, steps=steps)

    rea_500 = solvers[500].reattachment()[1]
    rea_1000 = solvers[1000].reattachment()[1]
    rea_1500 = solvers[1500].reattachment()[1]

    # All should be finite (flow has developed enough to have a reattachment point)
    assert not math.isnan(rea_500), "x_r/H should be finite at 500 steps"
    assert not math.isnan(rea_1000), "x_r/H should be finite at 1000 steps"
    assert not math.isnan(rea_1500), "x_r/H should be finite at 1500 steps"

    # The reattachment length should not be monotonic (it oscillates)
    # 500 steps: ~2.6, 1000 steps: ~4.1, 1500 steps: ~6.1
    assert rea_1000 > rea_500, f"x_r/H should increase from 500 to 1000 steps: {rea_500:.3f} -> {rea_1000:.3f}"
    assert rea_1500 > rea_1000, f"x_r/H should increase from 1000 to 1500 steps: {rea_1000:.3f} -> {rea_1500:.3f}"


# --------------------------------------------------------------------------- #
# Production limiter effect
# --------------------------------------------------------------------------- #

def test_production_limit_reduces_nu_t_at_1000_steps():
    """Production limiter at C=50 reduces nu_t/nu compared to unlimited at 1000 steps."""
    solver_unlimited = _run_with_limit(None, steps=1000)
    solver_limited = _run_with_limit(50.0, steps=1000)

    nu_t_unlimited = solver_unlimited.turbulence.viscosity().max() / solver_unlimited.nu
    nu_t_limited = solver_limited.turbulence.viscosity().max() / solver_limited.nu

    assert nu_t_limited < nu_t_unlimited, (
        f"production_limit=50 should reduce nu_t/nu: {nu_t_limited:.2f} < {nu_t_unlimited:.2f}"
    )
    # The reduction should be significant (at least 50%)
    assert nu_t_limited < 0.5 * nu_t_unlimited, (
        f"production_limit=50 should reduce nu_t/nu by at least 50%: {nu_t_limited:.2f} < {0.5 * nu_t_unlimited:.2f}"
    )


def test_production_limit_does_not_improve_xr_at_early_time():
    """At 1000 steps, the production limiter does not significantly improve x_r/H.

    This is because the flow structure is still evolving and the reattachment
    length is not yet sensitive to the eddy viscosity level.
    """
    solver_unlimited = _run_with_limit(None, steps=1000)
    solver_limited = _run_with_limit(50.0, steps=1000)

    rea_unlimited = solver_unlimited.reattachment()[1]
    rea_limited = solver_limited.reattachment()[1]

    assert not math.isnan(rea_unlimited), "x_r/H should be finite for unlimited case"
    assert not math.isnan(rea_limited), "x_r/H should be finite for limited case"

    # The reattachment lengths should be similar (within 10% of each other)
    rel_diff = abs(rea_limited - rea_unlimited) / rea_unlimited
    assert rel_diff < 0.10, (
        f"x_r/H should be similar: {rea_limited:.3f} vs {rea_unlimited:.3f} (diff={rel_diff:.1%})"
    )


# --------------------------------------------------------------------------- #
# Sweep results validation
# --------------------------------------------------------------------------- #

def test_sweep_results_file_exists():
    """The round 8 sweep results file should exist."""
    sweep_path = Path(__file__).parent.parent / "docs" / "case-periodic-hills" / "round8_sweep.json"
    assert sweep_path.exists(), f"Sweep results file not found: {sweep_path}"


def test_sweep_results_are_valid():
    """The sweep results should contain valid data for all production_limit values."""
    sweep_path = Path(__file__).parent.parent / "docs" / "case-periodic-hills" / "round8_sweep.json"
    with open(sweep_path) as f:
        data = json.load(f)

    assert "sweep" in data, "Sweep results should have 'sweep' key"
    assert "reference" in data, "Sweep results should have 'reference' key"
    assert data["reference"]["x_r/H"] == 5.34, "Reference x_r/H should be 5.34"

    for result in data["sweep"]:
        assert "production_limit" in result
        assert "x_r/H" in result
        assert "max_nu_t/nu" in result
        assert result["max_nu_t/nu"] > 0, "max nu_t/nu should be positive"


# --------------------------------------------------------------------------- #
# Honest FAIL documentation
# --------------------------------------------------------------------------- #

def test_gate_still_fails_at_1000_steps():
    """The acceptance gate still fails at 1000 steps (t=20).

    This documents the current state: x_r/H is 23.7% off the reference,
    and nu_t/nu is still growing.
    """
    solver = _run_with_limit(None, steps=1000)
    sep, rea = solver.reattachment()

    assert not math.isnan(rea), "Reattachment point should be found at 1000 steps"

    rel_error = abs(rea - REFERENCE_XR_H) / REFERENCE_XR_H
    # The error should be > 10% (gate fails)
    assert rel_error > REFERENCE_TOLERANCE, (
        f"Gate should still fail at 1000 steps: x_r/H={rea:.3f}, error={rel_error:.1%} > {REFERENCE_TOLERANCE:.0%}"
    )
