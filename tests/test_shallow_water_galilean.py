"""Tests for the Galilean-boosted dam-break verification.

Anchor: ``CASE-FVM-DAMBREAK-STOKER``.  The classical benchmark only exercises
``u0 = 0``; these tests grade the same solver against the boosted exact
solutions, which is what makes end conditions, source leakage and the
non-zero-velocity HLL branch observable.

See :mod:`tensorfvm.shallow_water.galilean` and
``docs/CASE-FVM-DAMBREAK-STOKER-GALILEAN/REPORT.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water import dam_break, galilean
from tensorfvm.shallow_water.crosscheck import bore_state_newton
from tensorfvm.shallow_water.exact import t_star_to_time

from tensorfvm.shallow_water.galilean import (
    DEFAULT_CASES,
    DRIFT_TOLERANCE,
    TOLERANCE,
    ClosedWall1D,
    GalileanCase,
    boundary_blindness_demo,
    boosted_shock_residual,
    galilean_selfcheck,
    run_case,
)

N = 800


def _case(key: str) -> GalileanCase:
    return next(c for c in DEFAULT_CASES if c.key == key)


# --------------------------------------------------------------------------- #
# Reference fields: analytic identities, no solver involved
# --------------------------------------------------------------------------- #
def test_control_is_bitwise_the_classical_ritter_field():
    """``u0 = 0`` must reproduce ``exact.ritter_state`` exactly (no drift)."""
    g = 9.81
    x = np.linspace(-4.0, 6.0, 501)
    t = t_star_to_time(2.0, 1.0, g)
    h_ref, u_ref = galilean.ritter_state(x, t, 1.0, g)
    h, u = _case("ritter-dry-u0-0.0").exact(x, t)
    assert np.array_equal(h, h_ref)
    assert np.array_equal(u, u_ref)


def test_boost_identity_shifts_the_field_and_the_velocity():
    """``h(x + u0 t, t) == h_rest(x, t)`` and ``u = u_rest + u0``."""
    g, u0 = 9.81, 1.5
    t = t_star_to_time(2.0, 1.0, g)
    x = np.linspace(-4.0, 6.0, 501)
    h_ref, u_ref = galilean.ritter_state(x, t, 1.0, g)
    h, u = _case("ritter-dry-u0-1.5").exact(x + u0 * t, t)
    assert np.max(np.abs(h - h_ref)) == 0.0
    assert np.max(np.abs(u - (u_ref + u0))) == 0.0


def test_volume_rate_is_the_inlet_flux():
    """``dV/dt = h0 u0`` for the boosted dry bed (analytic volume formula)."""
    out = galilean_selfcheck()
    assert out["dry_volume_rate_error"] < 1e-8
    assert out["dry_volume_rate_expected"] == pytest.approx(1.5)
    # the quadrature used for the wet bed reproduces the closed form
    assert out["quadrature_error_on_dry"] < 1e-6


def test_boosted_rankine_hugoniot_residual_is_round_off():
    """The boosted bore still satisfies the jump conditions to round-off."""
    wet = _case("stoker-wet-u0-1.0")
    res = boosted_shock_residual(wet)
    assert res["mass_relative"] < 1e-12
    assert res["momentum_relative"] < 1e-12


def test_wave_positions_shift_with_u0():
    """Head and front translate by ``u0 t``; the fan length is unchanged."""
    t = 0.5
    base = _case("ritter-dry-u0-0.0")
    for u0 in (1.5, -1.5):
        case = _case(f"ritter-dry-u0-{u0}")
        assert case.head(t) == pytest.approx(base.head(t) + u0 * t, rel=0, abs=1e-12)
        assert case.front(t) == pytest.approx(base.front(t) + u0 * t, rel=0, abs=1e-12)
        assert case.front(t) - case.head(t) == pytest.approx(
            base.front(t) - base.head(t), rel=0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Solver grading
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", [c.key for c in DEFAULT_CASES])
def test_every_boosted_case_passes_all_gates(key):
    metrics, solver = run_case(_case(key), n=N)
    failing = {k: v["value"] for k, v in metrics["verdict"].items()
               if isinstance(v, dict) and not v["pass"]}
    assert not failing, f"{key} failed gates: {failing}"
    assert metrics["front_error"] <= TOLERANCE
    assert metrics["quiet_drift_u"] <= DRIFT_TOLERANCE
    assert metrics["clipped_cells"] == 0
    assert solver.cfl_number() < 1.0


def _independent_front(case: GalileanCase, t: float) -> float:
    """Boosted front position rebuilt from an *independent* Stoker solve.

    Wet bed: the bore ratio comes from :func:`crosscheck.bore_state_newton`
    (Newton on ``r = h2/h1``), not from the bisection in ``exact`` that
    :meth:`GalileanCase.front` uses; dry bed: ``x_dam + u0 t + 2 t sqrt(g h0)``
    written out directly.
    """
    if case.kind == "dry":
        return case.x_dam + case.u0 * t + 2.0 * t * math.sqrt(case.gravity * case.h0)
    sol = bore_state_newton(case.h0, case.h1, case.gravity)
    return case.x_dam + (sol["shock_speed"] + case.u0) * t


@pytest.mark.parametrize("key", [c.key for c in DEFAULT_CASES])
def test_front_position_follows_the_boosted_closed_form(key):
    """Graded front vs an independently built boosted position (both routes)."""
    case = _case(key)
    metrics, _ = run_case(case, n=N)
    independent = _independent_front(case, metrics["t"])
    assert metrics["front_exact"] == pytest.approx(independent, rel=1e-10)
    assert abs(metrics["front_measured"] - independent) / independent <= TOLERANCE


def test_uniform_current_is_transported_exactly():
    """The quiet far field keeps ``(h0, u0)`` to round-off at every resolution."""
    for key in ("ritter-dry-u0-1.5", "ritter-dry-u0--1.5"):
        for n in (400, 800):
            metrics, _ = run_case(_case(key), n=n)
            assert metrics["quiet_cells"] >= 16
            # round-off only (measured 0 or ~2e-15), six orders below the gate
            assert metrics["quiet_drift_u"] < 1e-12
            assert metrics["quiet_drift_u"] <= DRIFT_TOLERANCE
            assert metrics["quiet_drift_h"] < 1e-12


def test_control_reproduces_the_rest_frame_benchmark():
    """The ``u0 = 0`` control must agree with the anchor's own headline metric."""
    ref = dam_break.run_case(dam_break.RITTER_DRY, n=N)["metrics"]
    control, _ = run_case(_case("ritter-dry-u0-0.0"), n=N)
    # same grid, same estimator -> identical numbers (rel 1e-12), so the boosted
    # module adds a new reference axis without perturbing the anchor's own
    # headline metric.
    assert control["front_error"] == pytest.approx(ref["front_fan_error"], rel=1e-12)
    assert control["front_measured"] == pytest.approx(ref["front_fan"], rel=1e-12)
    assert control["depth_l1"] == pytest.approx(ref["depth_l1"], rel=1e-12)
    assert control["tv_ratio"] == pytest.approx(ref["tv_ratio"], rel=1e-12)


def test_the_gate_can_fail():
    """Using the *unboosted* reference would be caught by the 5 % gate.

    A gate that cannot fail is not a gate: comparing the boosted run against the
    rest-frame front position must exceed ``TOLERANCE`` by a wide margin, i.e.
    the boosted reference is doing the work, not the tolerance.
    """
    case = _case("ritter-dry-u0-1.5")
    metrics, _ = run_case(case, n=N)
    unboosted = GalileanCase("unboosted", "dry", 1.0, 0.0, 0.0)
    wrong_error = abs(metrics["front_measured"] - unboosted.front(metrics["t"])) \
        / unboosted.front(metrics["t"])
    # measured 24 % vs the 5 % gate -- a 4.8x margin
    assert wrong_error > 4.0 * TOLERANCE, wrong_error


# --------------------------------------------------------------------------- #
# Falsification of the "the classical benchmark is enough" reading
# --------------------------------------------------------------------------- #
def test_boundary_blindness_demo():
    """Closed ends are invisible at ``u0 = 0`` and caught at ``u0 != 0``."""
    demo = boundary_blindness_demo(n=400)
    control, boosted = demo["rows"]
    # control: the two end conditions produce the same reading
    assert control["open_front_error"] == pytest.approx(
        control["wall_front_error"], rel=1e-12, abs=1e-15)
    assert control["wall_all_pass"] is True
    # boosted: the closed end blocks the inlet -> current frozen, budget broken
    assert boosted["open_all_pass"] is True
    assert boosted["wall_all_pass"] is False
    assert boosted["wall_quiet_drift_u"] > 0.1
    assert boosted["wall_volume_error"] > 0.05


def test_closed_wall_variant_leaves_the_uniform_state_untouched():
    """Sanity of the probe itself: at rest, the wall variant is a no-op."""
    case = _case("ritter-dry-u0-0.0")
    solver = ClosedWall1D(n=200, x_min=case.x_min, x_max=case.x_max,
                          gravity=case.gravity)
    h, u = case.exact(solver.x, 0.0)
    solver.set_state(h, u)
    residual_before = solver.rhs(solver.h, solver.hu)
    quiet = solver.x < case.head(1e-9) - 0.5
    assert np.all(np.abs(residual_before[0][quiet]) < 1e-12)


def test_wave_speeds_match_the_exact_fan_edges():
    """``u ± c`` at the fan edges are the exact head/front speeds (sanity)."""
    case = _case("ritter-dry-u0-1.5")
    t = case.t_end
    h_ex, u_ex = case.exact(case.x_dam + case.u0 * t + np.array([-case.c0 * t,
                                                                 2.0 * case.c0 * t]), t)
    assert math.isclose(h_ex[0], case.h0, rel_tol=1e-9)
    assert math.isclose(h_ex[1], 0.0, abs_tol=1e-12)
    head_speed = u_ex[0]
    front_speed = u_ex[1]
    assert head_speed == pytest.approx(case.u0, rel=0, abs=1e-9)
    # at the toe the fan velocity is exactly the front speed u0 + 2 c0
    assert front_speed == pytest.approx(case.u0 + 2.0 * case.c0, rel=0, abs=1e-9)