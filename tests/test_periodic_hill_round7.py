"""Round 7 tests for ``CASE-FVM-PERIODIC-HILLS``: production limiting.

Anchor: CASE-FVM-PERIODIC-HILLS
Mission: 18063 (selfstart, TensorFVM)

Round 6 diagnosed that the SA closure is over-diffusive on the periodic hill
(``nu_t/nu ≈ 45`` vs the expected ~20 for a turbulent channel at Re = 2800).
Round 7 adds an optional production limiter (``production_limit``) that bounds
the production term by ``production_limit * destruction``, preventing the eddy
viscosity from growing without bound in regions where the wide least-squares
vorticity over-predicts the strain rate.

These tests verify:
1. The production limiter is correctly applied in ``source_terms()``.
2. The limiter reduces the eddy viscosity compared to the unlimited case.
3. The limiter does not affect the solution when production < limit * destruction.
4. Invalid ``production_limit`` values are rejected.
5. Backward compatibility: ``production_limit=None`` reproduces round-6 behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_turbulence import (
    SpalartAllmaras,
    SpalartAllmarasConfig,
    eddy_viscosity,
)


def _make_solver(nx=48, ny=24, reynolds=2800.0):
    """Create a periodic hill solver with SA closure."""
    config = PeriodicHillConfig(nx=nx, ny=ny, reynolds=reynolds)
    solver = PeriodicHillSolver(config)
    solver.advance(50)  # get a non-trivial flow field
    return solver


def _make_closure(solver, production_limit=None, convection_blend=0.0):
    """Create an SA closure with the given production limit."""
    cfg = SpalartAllmarasConfig(
        production_limit=production_limit,
        convection_blend=convection_blend,
    )
    closure = SpalartAllmaras(solver, cfg)
    solver.turbulence = closure
    return closure


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan"), float("inf")])
def test_production_limit_rejects_invalid_values(value):
    """Invalid production_limit values must raise ValueError."""
    with pytest.raises(ValueError):
        SpalartAllmarasConfig(production_limit=value)


def test_production_limit_accepts_none():
    """None means no limiting (backward compatible)."""
    cfg = SpalartAllmarasConfig(production_limit=None)
    assert cfg.production_limit is None


@pytest.mark.parametrize("value", [1.0, 5.0, 10.0, 100.0])
def test_production_limit_accepts_positive_values(value):
    """Positive production_limit values are accepted."""
    cfg = SpalartAllmarasConfig(production_limit=value)
    assert cfg.production_limit == value


# --------------------------------------------------------------------------- #
# Source terms: production limiting is applied
# --------------------------------------------------------------------------- #

def test_production_limiting_reduces_production():
    """When production > limit * destruction, the production term is capped."""
    solver = _make_solver()
    closure = _make_closure(solver, production_limit=1.0)
    src = closure.source_terms()

    # With production_limit=1.0, production should be <= destruction everywhere
    assert np.all(src["production"] <= src["destruction"] + 1e-12), (
        f"production exceeds destruction: max ratio = "
        f"{(src['production'] / np.maximum(src['destruction'], 1e-30)).max():.2f}"
    )


def test_production_limiting_with_higher_limit():
    """With production_limit=10, production should be <= 10 * destruction."""
    solver = _make_solver()
    closure = _make_closure(solver, production_limit=10.0)
    src = closure.source_terms()

    assert np.all(src["production"] <= 10.0 * src["destruction"] + 1e-12), (
        f"production exceeds 10 * destruction: max ratio = "
        f"{(src['production'] / np.maximum(src['destruction'], 1e-30)).max():.2f}"
    )


def test_no_limiting_when_production_below_cap():
    """When production < limit * destruction, the production term is unchanged."""
    solver = _make_solver()

    # First, get the unlimited production
    closure_unlimited = _make_closure(solver, production_limit=None)
    src_unlimited = closure_unlimited.source_terms()

    # Now, with a very high limit, production should be the same
    closure_limited = _make_closure(solver, production_limit=1e6)
    src_limited = closure_limited.source_terms()

    np.testing.assert_allclose(
        src_limited["production"], src_unlimited["production"],
        rtol=1e-10,
        err_msg="production should be unchanged when limit is very high",
    )


# --------------------------------------------------------------------------- #
# Backward compatibility: None reproduces round-6 behaviour
# --------------------------------------------------------------------------- #

def test_backward_compatibility_none_reproduces_round6():
    """production_limit=None must reproduce the round-6 (unlimited) behaviour."""
    solver = _make_solver()

    closure = _make_closure(solver, production_limit=None)
    src = closure.source_terms()

    # The production should be the raw cb1 * Sbar * nu_tilde
    nu_tilde = closure.nu_tilde
    sbar = src["Sbar"]
    expected_production = closure.config.cb1 * sbar * nu_tilde

    np.testing.assert_allclose(
        src["production"], expected_production,
        rtol=1e-10,
        err_msg="production_limit=None should give unlimited production",
    )


# --------------------------------------------------------------------------- #
# Eddy viscosity reduction with limiting
# --------------------------------------------------------------------------- #

def test_limiting_reduces_eddy_viscosity_growth():
    """Production limiting should reduce the eddy viscosity growth rate."""
    solver1 = _make_solver()
    solver2 = _make_solver()

    closure1 = _make_closure(solver1, production_limit=None)
    closure2 = _make_closure(solver2, production_limit=5.0)

    # Run both for the same number of steps
    for _ in range(30):
        closure1.update(solver1.time_step_s)
        closure2.update(solver2.time_step_s)

    nu_t1 = closure1.viscosity()
    nu_t2 = closure2.viscosity()

    # The limited closure should have lower or equal eddy viscosity
    assert nu_t2.max() <= nu_t1.max() + 1e-12, (
        f"limited nu_t max ({nu_t2.max():.4f}) should be <= "
        f"unlimited nu_t max ({nu_t1.max():.4f})"
    )
    # And strictly lower somewhere (otherwise the limiter is doing nothing)
    assert (nu_t2 < nu_t1 - 1e-12).any(), (
        "production limiting should reduce nu_t somewhere"
    )


# --------------------------------------------------------------------------- #
# Source terms invariants with limiting
# --------------------------------------------------------------------------- #

def test_source_terms_remain_finite_with_limiting():
    """All source terms should remain finite with production limiting."""
    solver = _make_solver()
    closure = _make_closure(solver, production_limit=10.0)

    for _ in range(20):
        src = closure.source_terms()
        for key, val in src.items():
            assert np.all(np.isfinite(val)), f"{key} is not finite"
        closure.update(solver.time_step_s)


def test_source_terms_non_negative_with_limiting():
    """Production and destruction should be non-negative with limiting."""
    solver = _make_solver()
    closure = _make_closure(solver, production_limit=5.0)

    for _ in range(20):
        src = closure.source_terms()
        assert np.all(src["production"] >= -1e-12), "production went negative"
        assert np.all(src["destruction"] >= -1e-12), "destruction went negative"
        closure.update(solver.time_step_s)


# --------------------------------------------------------------------------- #
# Integration: solver + closure with limiting
# --------------------------------------------------------------------------- #

def test_solver_advance_with_production_limiting():
    """The solver should advance without error when production limiting is on."""
    config = PeriodicHillConfig(nx=48, ny=24, reynolds=2800.0)
    solver = PeriodicHillSolver(config)
    cfg = SpalartAllmarasConfig(production_limit=10.0)
    solver.turbulence = SpalartAllmaras(solver, cfg)

    # Advance for a moderate number of steps
    solver.advance(100)

    # Check invariants
    assert np.all(np.isfinite(solver.u)), "u is not finite"
    assert np.all(np.isfinite(solver.v)), "v is not finite"
    assert np.all(np.isfinite(solver.turbulence.nu_tilde)), "nu_tilde is not finite"
    assert np.all(solver.turbulence.nu_tilde >= 0.0), "nu_tilde went negative"


def test_production_limiting_stability_over_long_run():
    """Production limiting should keep the solver stable over a longer run.

    Note: the periodic hill flow takes many time units to develop a separation
    bubble (round 6 needed t=120). This test checks stability, not convergence.
    """
    config = PeriodicHillConfig(nx=48, ny=24, reynolds=2800.0)
    solver = PeriodicHillSolver(config)
    cfg = SpalartAllmarasConfig(production_limit=10.0)
    solver.turbulence = SpalartAllmaras(solver, cfg)

    # Advance for 1000 steps (t = 20)
    solver.advance(1000)

    # Check invariants
    assert np.all(np.isfinite(solver.u)), "u is not finite after 1000 steps"
    assert np.all(np.isfinite(solver.v)), "v is not finite after 1000 steps"
    assert np.all(np.isfinite(solver.turbulence.nu_tilde)), "nu_tilde not finite"
    assert np.all(solver.turbulence.nu_tilde >= 0.0), "nu_tilde went negative"

    # The eddy viscosity should be bounded by the production limit effect
    nu_t_ratio = solver.turbulence.viscosity().max() / solver.nu
    # With production_limit=10, nu_t/nu should be significantly lower than
    # the unlimited case (which reached ~45 in round 6)
    assert nu_t_ratio < 30.0, (
        f"nu_t/nu = {nu_t_ratio:.1f} is too high; production limiting may not be working"
    )
