"""Ritter (1892) / Stoker (1957) dam-break verification of the SWE solver.

Node: CASE-FVM-DAMBREAK-STOKER (TensorFVM shallow-water benchmark).

What is asserted, and why each assertion is evidence rather than decoration:

* the two closed forms are *self-certifying*: the Stoker bore state is the root
  of the rarefaction/bore matching problem and is checked by its relative
  Rankine-Hugoniot residuals (machine precision), the Ritter fan by its
  analytic identities (``h(dam) = 4 h0/9``, ``u(dam) = 2 c0/3``,
  ``int h dx = h0 c0 t``).  No tabulated number is transcribed.
* the finite-volume solution of the shallow-water equations is then compared
  with those closed forms at ``t* = 2`` on a fixed grid: front/bore position,
  post-bore depth, the still-water level step ``h2/h1``, and the whole depth
  profile in L1.  The task tolerance is 5 %.
* the waveform must stay monotone: zero interior local extrema and total
  variation not growing above the exact TV.
* conservation and positivity are asserted separately, because a metric can be
  met by a scheme that quietly clips (that would be a defect, not a result).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water import (
    RITTER_DRY,
    STOKER_WET,
    TOLERANCE,
    ShallowWater1D,
    check_stoker_solution,
    convergence_study,
    exactness_checks,
    ritter_front_position,
    ritter_state,
    run_case,
    stoker_constant_state,
    stoker_state,
    t_star_to_time,
    time_to_t_star,
)

G = 9.81
H0 = 1.0


# --------------------------------------------------------------------------- #
# exact solutions
# --------------------------------------------------------------------------- #
def test_t_star_scaling_and_ritter_front_identity():
    t = t_star_to_time(2.0, H0, G)
    assert t == pytest.approx(2.0 * math.sqrt(H0 / G), rel=0.0, abs=1e-15)
    assert time_to_t_star(t, H0, G) == pytest.approx(2.0, rel=1e-15)
    # Ritter front at t* = 2 is exactly 4 h0 (2 c0 t with c0 t = 2 h0)
    assert ritter_front_position(t, H0, G) == pytest.approx(4.0 * H0, rel=1e-15)


def test_ritter_fan_identities():
    t = t_star_to_time(2.0, H0, G)
    c0 = math.sqrt(G * H0)
    h_dam, u_dam = ritter_state(np.array([0.0]), t, H0, G, 0.0)
    assert h_dam[0] / H0 == pytest.approx(4.0 / 9.0, rel=1e-14)
    assert u_dam[0] / c0 == pytest.approx(2.0 / 3.0, rel=1e-14)
    # mass inside the fan is h0 c0 t (quadrature of the evaluated profile)
    xs = np.linspace(-c0 * t, 2.0 * c0 * t, 400_001)
    h_fan, _ = ritter_state(xs, t, H0, G, 0.0)
    mass = np.trapezoid(h_fan, xs) if hasattr(np, "trapezoid") else np.trapz(h_fan, xs)
    assert mass == pytest.approx(H0 * c0 * t, rel=1e-9)


def test_stoker_bore_state_satisfies_jump_conditions():
    checks = check_stoker_solution(H0, 0.5, G)
    assert checks["riemann_invariant_relative"] < 1e-12
    assert checks["mass_jump_relative"] < 1e-12
    assert checks["momentum_jump_relative"] < 1e-12


def test_stoker_state_continuity_and_limits():
    h1 = 0.5
    t = t_star_to_time(2.0, H0, G)
    sol = stoker_constant_state(H0, h1, G)
    c0, c2, u2 = sol["c0"], sol["c2"], sol["u2"]
    # rarefaction meets the constant state continuously at xi = u2 - c2
    xi = u2 - c2
    h_fan, u_fan = stoker_state(np.array([xi * t * (1.0 - 1e-9)]), t, H0, h1, G, 0.0)
    h_fan2, u_fan2 = stoker_state(np.array([xi * t]), t, H0, h1, G)
    assert h_fan[0] == pytest.approx(sol["h2"], rel=1e-6)
    assert u_fan[0] == pytest.approx(u2, rel=1e-6)
    assert h_fan2[0] == pytest.approx(sol["h2"], rel=1e-14)
    # the bore is faster than the tail and slower than the upstream head
    assert -c0 < xi < sol["shock_speed"]
    # a deeper downstream bed gives a weaker bore: h2 up, u2 down.  The bore
    # speed is NOT monotone in h1 (it dips near h1/h0 ~ 0.4 and grows again both
    # towards the dry limit 2 c0 and towards the acoustic limit c0), so only the
    # physical bound is asserted here.
    weak = stoker_constant_state(H0, 0.8, G)
    assert weak["h2"] > sol["h2"] and weak["u2"] < u2
    assert 0.0 < weak["shock_speed"] < 2.0 * c0
    assert 0.0 < sol["shock_speed"] < 2.0 * c0


def test_exactness_payload_is_machine_precision():
    payload = exactness_checks(G)
    assert payload["ritter"]["h_at_dam_residual"] < 1e-14
    assert payload["ritter"]["u_at_dam_residual"] < 1e-14
    assert payload["ritter"]["fan_mass_relative"] < 1e-10
    assert payload["stoker"]["momentum_jump_relative"] < 1e-12


# --------------------------------------------------------------------------- #
# solver properties
# --------------------------------------------------------------------------- #
def test_uniform_flow_is_an_exact_steady_state():
    solver = ShallowWater1D(100, 0.0, 10.0, gravity=G, order=2)
    solver.set_state(np.ones(100), np.full(100, 1.0))
    solver.advance(1.0)
    assert np.max(np.abs(solver.h - 1.0)) < 1e-13
    assert np.max(np.abs(solver.hu - 1.0)) < 1e-13


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET], ids=["dry", "wet"])
def test_mass_is_conserved_and_depth_stays_positive(case):
    run = run_case(case, n=400)
    metrics = run["metrics"]
    assert metrics["mass_drift"] < 1e-10
    assert run["discretisation"]["clipped_cells"] == 0
    assert metrics["h_min"] >= 0.0
    # both boundaries are still water for the whole run, so the volume is fixed
    assert run["discretisation"]["volume_final"] == pytest.approx(
        run["discretisation"]["volume_initial"], rel=1e-10)


def test_first_order_is_less_accurate_than_second_order():
    coarse = run_case(STOKER_WET, n=400, order=1)["metrics"]["depth_l1"]
    fine = run_case(STOKER_WET, n=400, order=2)["metrics"]["depth_l1"]
    assert fine < coarse


def test_cfl_insensitivity():
    errors = [run_case(STOKER_WET, n=400, cfl=cfl)["metrics"]["step_ratio_error"]
              for cfl in (0.2, 0.8)]
    assert max(errors) < TOLERANCE


# --------------------------------------------------------------------------- #
# the benchmark itself (task acceptance criterion)
# --------------------------------------------------------------------------- #
def test_ritter_dry_front_and_profile_within_tolerance():
    run = run_case(RITTER_DRY, n=800)
    m = run["metrics"]
    assert m["front_wetted_error"] < TOLERANCE, m["front_wetted_error"]
    assert m["depth_l1"] < TOLERANCE
    # the tip is a zero-depth cusp: the scaled level-crossing measure is closer
    assert m["front_level_scaled_error"] < TOLERANCE
    assert m["local_extrema"] == 0
    assert m["tv_ratio"] <= 1.0 + 5e-3
    assert m["h_max_over_h0"] <= 1.0 + 1e-6


def test_stoker_wet_bore_and_level_step_within_tolerance():
    run = run_case(STOKER_WET, n=800)
    m = run["metrics"]
    assert m["shock_error"] < TOLERANCE, m["shock_error"]
    assert m["step_ratio_error"] < TOLERANCE, m["step_ratio_error"]
    assert m["h2_error"] < TOLERANCE
    assert m["depth_l1"] < TOLERANCE
    assert m["local_extrema"] == 0
    assert m["tv_ratio"] <= 1.0 + 5e-3
    assert run["verdict"]["all_pass"] is True


def test_deeper_bore_h0_over_h1_four_stays_within_tolerance():
    from tensorfvm.shallow_water.dam_break import DamBreakCase

    strong = DamBreakCase("stoker-wet-r4", "wet", H0, 0.25, G, 0.0, -4.0, 6.0, 2.0,
                          "Stoker (1957) ch. 10")
    m = run_case(strong, n=800)["metrics"]
    assert m["shock_error"] < TOLERANCE
    assert m["step_ratio_error"] < TOLERANCE
    assert m["local_extrema"] == 0
    assert m["tv_ratio"] <= 1.0 + 5e-3


def test_depth_error_converges_with_grid_refinement():
    study = convergence_study(STOKER_WET, resolutions=(100, 200, 400, 800))
    l1 = [row["depth_l1"] for row in study["rows"]]
    assert all(later < earlier for earlier, later in zip(l1, l1[1:]))
    assert study["observed_order_l1"] > 0.7
    assert l1[-1] < 0.25 * l1[0]