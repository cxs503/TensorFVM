"""Gates for the momentum identity and the velocity-field budget (anchor G7/G8).

These tests pin both the *physics* (the closed form is the exact PDE statement,
not a fit) and the *honest headline* established by this work:

* the global momentum identity closes at machine precision for the solver, and
  the detector is demonstrably non-empty (analytic response, gain 1);
* the anchor's velocity field -- never graded by G1-G6 -- is **honest FAIL at
  n = 800** on the dry bed (> 5 %) and converges with observed order ~0.4
  (sub-linear), because the error is carried by the wetting-tip sliver where the
  exact velocity is ~2 c0 while the solver keeps those cells dry;
* the wet bed (no dry front) converges at first order in the same norm, which
  localises the effect to the wetting front rather than to the scheme at large.

If a future change repairs the front treatment, the sub-linear assertions here
fail loudly and the numbers must be re-derived (they are measurements, not
tolerances of convenience).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET
from tensorfvm.shallow_water.exact import gravity_wave_speed
from tensorfvm.shallow_water.momentum import (
    TIP_FRACTION,
    TOLERANCE,
    defect_injection_control,
    exact_momentum,
    exact_momentum_rate,
    momentum_closure,
    momentum_quadrature,
    similarity_identity_residual,
    similarity_velocity,
    undisturbed_guard,
    velocity_budget,
    velocity_resolution_study,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D


def _profile(case, n):
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity,
                            cfl=0.4, order=2)
    solver.set_state(np.where(solver.x < case.x_dam, case.h0, case.h1))
    solver.advance(case.t_end)
    return solver


# --------------------------------------------------------------- exact side
@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET], ids=["dry", "wet"])
def test_closed_form_matches_independent_quadrature(case):
    """``M(t) = g/2 (h_L^2 - h_R^2) t`` reproduced by quadrature of the profile."""
    closed = exact_momentum(case)
    quad = momentum_quadrature(case)
    assert closed > 0.0
    assert abs(quad - closed) / closed < 1e-5


def test_momentum_rate_uses_the_undisturbed_boundary_depths():
    """The rate is set by the two end faces, not by the initial jump."""
    assert exact_momentum_rate(RITTER_DRY) == pytest.approx(
        0.5 * 9.81 * (1.0 ** 2 - 0.0 ** 2))
    assert exact_momentum_rate(STOKER_WET) == pytest.approx(
        0.5 * 9.81 * (1.0 ** 2 - 0.5 ** 2))
    # dry bed: same left state, no right-hand water -> strictly larger source
    assert exact_momentum_rate(RITTER_DRY) > exact_momentum_rate(STOKER_WET)


def test_the_identity_needs_undisturbed_boundaries_and_says_so():
    """At t* = 2 the waves are 2 m clear of both ends; at t* = 4 they are not."""
    ok = undisturbed_guard(RITTER_DRY)
    assert ok["undisturbed"] and ok["clearance_left"] == pytest.approx(2.0)
    late = undisturbed_guard(RITTER_DRY, t=2.0 * RITTER_DRY.t_end)
    assert not late["undisturbed"]
    assert late["clearance_right"] < 0.0  # the front has left the domain


def test_similarity_velocity_identity_is_exact_on_the_exact_solution():
    """``u = 2 c0 - 2 sqrt(g h)`` holds to machine precision in the dry fan."""
    assert similarity_identity_residual() < 1e-13
    c0 = gravity_wave_speed(RITTER_DRY.h0, RITTER_DRY.gravity)
    assert similarity_velocity(0.0, RITTER_DRY.h0, RITTER_DRY.gravity) == pytest.approx(2 * c0)
    # and it is *not* smooth in h: this is why a depth error becomes a sqrt error
    eps = 1e-6
    slope = (similarity_velocity(eps, RITTER_DRY.h0) - similarity_velocity(0.0, RITTER_DRY.h0)) / eps
    assert slope < -1e3


# ------------------------------------------------------------ numerical side
@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET], ids=["dry", "wet"])
def test_momentum_identity_closes_for_the_solver(case):
    """New gate G7: the conservative solver reproduces the exact momentum."""
    row = momentum_closure(case, n=800)
    assert row["identity_applies"], row["guard"]
    assert abs(row["momentum_exact_closed_form"]) > 1.0
    assert row["closure_error"] < 1e-10
    assert row["quadrature_vs_closed_form"] < 1e-5


def test_momentum_identity_would_flag_a_source_term():
    """Scope guard: with Manning friction the identity no longer applies."""
    solver = ShallowWater1D(400, RITTER_DRY.x_min, RITTER_DRY.x_max,
                            gravity=RITTER_DRY.gravity, cfl=0.4, order=2,
                            manning_n=0.03)
    solver.set_state(np.where(solver.x < RITTER_DRY.x_dam,
                              RITTER_DRY.h0, RITTER_DRY.h1))
    solver.advance(RITTER_DRY.t_end)
    exact = exact_momentum(RITTER_DRY, solver.time)
    assert abs(solver.momentum() - exact) / exact > 1e-2  # friction is a sink


def test_defect_injection_is_detected_with_the_predicted_gain():
    """Both channels respond linearly to an injected defect (criteria not empty)."""
    control = defect_injection_control(RITTER_DRY, n=400,
                                       epsilons=(1e-3, 1e-2, 1e-1))
    assert control["window_cells"] > 0
    for row in control["rows"]:
        assert row["momentum_gain"] == pytest.approx(1.0, rel=0.2)
        assert row["velocity_max_abs_delta"] == pytest.approx(
            row["velocity_predicted"], rel=0.6)
        # 1e-3 is still 9 orders above the solver's own 1e-12 closure floor
        assert row["momentum_measured"] > 1e-6


# ------------------------------------------------------ the honest headline
def test_dry_bed_velocity_field_fails_the_5pct_bar_at_the_accepted_grid():
    """G8 honest FAIL: the anchor's accepted grid n>=800 does not certify u."""
    solver = _profile(RITTER_DRY, 800)
    h, u = solver.primitive()
    row = velocity_budget(RITTER_DRY, solver.x, h, u)
    assert row["u_l1"] > TOLERANCE
    assert not row["pass_5pct"]
    # while the depth field on the same grid is two orders better
    assert row["depth_l1"] < 2e-3
    # the mass weighted norm is *not* the same statement: report both
    assert row["u_l1_mass"] < TOLERANCE


def test_dry_bed_velocity_error_lives_in_the_wetting_tip():
    """> 90 % of the error (bound) is the analytically wet / numerically dry sliver."""
    solver = _profile(RITTER_DRY, 3200)
    h, u = solver.primitive()
    row = velocity_budget(RITTER_DRY, solver.x, h, u)
    assert row["share_missing_tip"] > 0.8
    assert row["cells"]["missing_tip"] > 0
    assert row["u_max_exact"] == pytest.approx(
        2 * gravity_wave_speed(RITTER_DRY.h0, RITTER_DRY.gravity), rel=1e-2)
    assert row["u_max_numeric"] < row["u_max_exact"]


def test_dry_bed_velocity_order_is_sublinear_while_depth_is_first_order():
    """Measured orders: u ~ 0.4 (approaching the sqrt law), h ~ 1.0."""
    study = velocity_resolution_study(RITTER_DRY, resolutions=(800, 1600, 3200))
    u_orders = study["observed_order_u_l1"]
    d_orders = study["observed_order_depth_l1"]
    assert all(o < 0.6 for o in u_orders)
    assert u_orders[-1] > u_orders[0]  # approaches 0.5 from below (sqrt law)
    assert all(0.9 < o < 1.1 for o in d_orders)
    mass_orders = study["observed_order_u_l1_mass"]
    assert all(0.85 < o < 1.15 for o in mass_orders)


def test_wet_bed_velocity_converges_at_first_order_no_dry_front():
    """Control: without a wetting front the same norm is first order at 0.7 %."""
    study = velocity_resolution_study(STOKER_WET, resolutions=(800, 1600, 3200))
    assert all(0.8 < o < 1.2 for o in study["observed_order_u_l1"])
    first = study["rows"][0]
    assert first["u_l1"] < TOLERANCE
    assert first["cells"]["missing_tip"] == 0


def test_tip_fraction_is_the_declared_constant():
    """The region split uses the module-level constant, not a per-call fudge."""
    solver = _profile(RITTER_DRY, 800)
    h, u = solver.primitive()
    row = velocity_budget(RITTER_DRY, solver.x, h, u)
    assert row["tip_fraction"] == TIP_FRACTION
    wide = velocity_budget(RITTER_DRY, solver.x, h, u, tip_fraction=0.2)
    assert wide["cells"]["band"] > row["cells"]["band"]