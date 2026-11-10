"""Verification suite for :mod:`tensorfvm.planing_savitsky`.

Evidence ledger (what each block actually proves):

* **transcription / structural** -- the implemented law reproduces Savitsky's
  published exponents (``tau^1.1``), the ``lam^0.5`` high-speed asymptote and the
  ``0.0065 beta C_L0^0.6`` deadrise decrement; 10 operating points are
  re-evaluated through an independent wetted-area normalisation.  This guards
  against transcription errors; it is *not* an independent physical validation,
  because the closed form is the fit to the very chart it would be compared with
  (the original-figure comparison is declared WITHHELD in the report -- no
  offline access to Savitsky 1964's plotted data).
* **negative controls** -- ``beta = 0`` degenerates to the flat-plate law
  bit-for-bit; ``tau -> 0`` and ``lam -> 0`` drive ``C_L -> 0``; the ``V -> 0``
  limit is shown to leave a speed-independent remnant and is therefore flagged
  out of envelope instead of being mistaken for a validated limit.
* **equilibrium** -- the two running-condition equations are satisfied to
  1e-9 relative by the solver on a 61-point speed sweep (this is what a CFD
  comparison would be run against).
* **resistance** -- component build-up identity, ITTC-57 reference value,
  spray-law scaling.
* **stability** -- the sign criteria (``dC_L/dlam``, ``dM*/dlam``, ``d tau/dC_V``
  and the 2-DOF stiffness determinant) are evaluated, and the *falsification* of
  the rate-free stiffness criterion (indefinite everywhere) is pinned as a test
  so the documented conclusion cannot silently drift.
"""
import math

import pytest

from tensorfvm import planing_savitsky as ps

# reference hull used throughout (8 m / 2.46 t planing craft)
BEAM = 2.0
HULL_LENGTH = 7.0
WEIGHT = 24125.0
BETA = 15.0
LCG_OVER_BEAM = 1.6


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #
def test_speed_coefficient_definition_and_inverse():
    assert ps.speed_coefficient(10.0, 2.5) == pytest.approx(10.0 / math.sqrt(9.80665 * 2.5))
    for C_V in (0.6, 1.0, 2.5, 8.0):
        assert ps.speed_from_coefficient(C_V, BEAM) == pytest.approx(
            C_V * math.sqrt(9.80665 * BEAM), rel=1e-14)
        assert ps.speed_coefficient(ps.speed_from_coefficient(C_V, BEAM), BEAM) == \
            pytest.approx(C_V, rel=1e-13)
    # dynamic pressure scale derived from C_V must agree with the V-based form
    C_V, rho, g = 3.0, 1025.0, 9.80665
    speed = ps.speed_from_coefficient(C_V, BEAM)
    assert ps.dynamic_pressure_scale(C_V, BEAM, rho=rho, g=g) == pytest.approx(
        0.5 * rho * speed ** 2 * BEAM ** 2, rel=1e-13)
    with pytest.raises(ValueError):
        ps.speed_coefficient(1.0, 0.0)


def test_wetted_geometry_identities():
    lam, tau, beta = 2.5, 6.0, 20.0
    assert ps.wetted_area(lam, BEAM) == pytest.approx(lam * BEAM ** 2)
    assert ps.mean_wetted_length(lam, BEAM) == pytest.approx(lam * BEAM)
    assert ps.transom_immersion(lam, tau, BEAM) == pytest.approx(
        lam * BEAM * math.sin(math.radians(tau)))
    keel, chine = ps.chine_wetted_lengths(lam, tau, beta, BEAM)
    assert keel > chine > 0.0
    assert keel - chine == pytest.approx(
        0.5 * BEAM * math.tan(math.radians(beta)) / math.tan(math.radians(tau)), rel=1e-12)
    assert 0.5 * (keel + chine) == pytest.approx(lam * BEAM, rel=1e-12)
    # a deeper deadrise at fixed trim separates the keel/chine waterlines more
    assert ps.chine_wetted_lengths(lam, tau, 30.0, BEAM)[0] > keel


# --------------------------------------------------------------------------- #
# lift law: transcription + structure + negative controls
# --------------------------------------------------------------------------- #
OPERATING_POINTS = [
    (1.0, 2.0, 4.0, 0.0), (2.0, 3.0, 6.0, 10.0), (2.0, 1.5, 8.0, 20.0),
    (3.0, 2.5, 5.0, 15.0), (4.0, 2.0, 4.0, 12.5), (4.0, 3.5, 10.0, 0.0),
    (6.0, 1.0, 6.0, 25.0), (6.0, 4.0, 3.0, 8.0), (8.0, 2.0, 12.0, 18.0),
    (10.0, 2.5, 7.0, 30.0),
]


def test_ten_operating_points_match_wetted_area_form():
    """10 points span C_V 1-10, lam 1-4, tau 3-12 deg, beta 0-30 deg."""
    for C_V, lam, tau, beta in OPERATING_POINTS:
        cl = ps.lift_coefficient(C_V, lam, tau, beta)
        cl0 = ps.lift_coefficient_flat_plate(C_V, lam, tau)
        area_form = tau ** 1.1 * (0.0120 * lam ** -0.5 + 0.0055 * lam ** 1.5 / C_V ** 2)
        if beta > 0.0:
            area_form -= 0.0065 * beta * cl0 ** 0.6 / lam
        assert area_form * lam == pytest.approx(cl, rel=1e-12)
        assert cl > 0.0


def test_trim_exponent_is_one_point_one():
    ratio = (ps.lift_coefficient_flat_plate(2.0, 2.0, 8.0)
             / ps.lift_coefficient_flat_plate(2.0, 2.0, 4.0))
    assert math.log(ratio) / math.log(2.0) == pytest.approx(1.1, abs=1e-12)


def test_high_speed_asymptote_is_sqrt_lambda_law():
    """C_V -> inf leaves C_L0 = tau^1.1 0.0120 lam^0.5 (the 2-D-like branch)."""
    C_V, lam, tau = 1.0e6, 4.0, 6.0
    assert ps.lift_coefficient_flat_plate(C_V, lam, tau) == pytest.approx(
        tau ** 1.1 * 0.0120 * math.sqrt(lam), rel=1e-8)
    # ... and the two branches are both present at finite C_V
    assert (ps.lift_coefficient_flat_plate(1.0, lam, tau)
            > ps.lift_coefficient_flat_plate(1.0e6, lam, tau))


def test_beta_zero_is_exact_flat_plate_negative_control():
    for C_V, lam, tau, _ in OPERATING_POINTS:
        assert ps.lift_coefficient(C_V, lam, tau, 0.0) == \
            ps.lift_coefficient_flat_plate(C_V, lam, tau)
        assert ps.lift_coefficient(C_V, lam, tau) == pytest.approx(
            ps.lift_coefficient_flat_plate(C_V, lam, tau), rel=1e-15)


def test_deadrise_decrement_follows_documented_law_and_is_monotone():
    C_V, lam, tau = 2.0, 3.0, 6.0
    cl0 = ps.lift_coefficient_flat_plate(C_V, lam, tau)
    previous = cl0
    for beta in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        cl = ps.lift_coefficient(C_V, lam, tau, beta)
        assert cl == pytest.approx(cl0 - 0.0065 * beta * cl0 ** 0.6, rel=1e-14)
        assert cl <= previous + 1e-15
        previous = cl
    assert 0.0 < 1.0 - previous / cl0 < 0.35


def test_deadrise_breakdown_threshold():
    for beta in (5.0, 15.0, 30.0):
        cl0 = ps.deadrise_breakdown_lift(beta)
        assert cl0 == pytest.approx((0.0065 * beta) ** 2.5, rel=1e-12)
        assert ps.lift_coefficient(2.0, 3.0, (cl0 / 0.0412) ** (1.0 / 1.1), beta) < 1e-3


def test_lift_vanishes_with_trim_and_wetted_length():
    """(d) negative control: tau -> 0 and lam -> 0 both drive C_L -> 0."""
    assert ps.lift_coefficient_flat_plate(3.0, 2.0, 0.01) == pytest.approx(1.29e-4, rel=5e-3)
    assert ps.lift_coefficient(3.0, 2.0, 0.01, 0.0) > 0.0
    assert ps.lift_coefficient(3.0, 1e-9, 6.0, 0.0) < 1e-3
    assert ps.lift_coefficient(3.0, 2.0, 6.0, 0.0) > ps.lift_coefficient(3.0, 2.0, 3.0, 0.0)


def test_speed_zero_limit_is_a_residual_not_a_zero():
    """(d) the literal V -> 0 limit of the planing fit is *not* zero.

    ``C_L ~ 1/C_V^2``: the law diverges.  The force keeps a speed-independent
    remnant (buoyant equivalent) and the module flags the condition out of
    envelope, which is the only honest handling of this limit.
    """
    assert ps.lift_coefficient_flat_plate(1e-3, 2.0, 6.0) > 1e5
    residual = ps.hydrostatic_residual_force(0.0, BEAM, 2.0, 6.0)
    assert ps.lift_force(0.02, BEAM, 2.0, 6.0) == pytest.approx(residual, rel=1e-2)
    dynamic_part = ps.lift_force(0.01, BEAM, 2.0, 6.0) - ps.hydrostatic_residual_force(
        0.01, BEAM, 2.0, 6.0)
    assert 0.0 < dynamic_part < 1.0                     # -> 0 as C_V -> 0
    assert residual > 1000.0                            # ... but the remnant is not 0
    assert not ps.in_envelope(0.3, 2.0, 6.0, 15.0)      # flagged, not silently used
    assert ps.in_envelope(2.0, 2.0, 6.0, 15.0)


def test_lift_coefficient_magnitude_range():
    """(e) magnitude check: a planing craft in its design band sits in 0.1-0.3."""
    state = ps.running_condition(2.0, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH)
    assert 0.1 <= state.C_L <= 0.3
    assert state.C_L == pytest.approx(0.15, rel=2e-2)


# --------------------------------------------------------------------------- #
# centre of pressure
# --------------------------------------------------------------------------- #
def test_center_of_pressure_asymptote_and_monotonicity():
    lam = 2.5
    assert ps.center_of_pressure_over_beam(1.0e6, lam) == pytest.approx(0.75 * lam, rel=1e-9)
    values = [ps.center_of_pressure_over_beam(2.0, l) for l in (0.5, 1.0, 2.0, 4.0, 5.0)]
    assert values == sorted(values)
    assert all(v < 0.75 * l for v, l in zip(values, (0.5, 1.0, 2.0, 4.0, 5.0)))


def test_center_of_pressure_inverse_roundtrip():
    worst = 0.0
    for C_V in (0.7, 1.0, 2.0, 5.0, 12.0):
        for lam in (0.5, 1.0, 2.0, 3.5, 5.0):
            back = ps.mean_wetted_length_from_center_of_pressure(
                C_V, ps.center_of_pressure_over_beam(C_V, lam))
            worst = max(worst, abs(back - lam) / lam)
    assert worst < 1e-10


def test_pitch_moment_vanishes_at_the_moment_equation():
    C_V, lam = 2.0, 3.0
    lcp = ps.center_of_pressure_over_beam(C_V, lam)
    assert ps.pitch_moment_coefficient(C_V, lam, 6.0, 15.0, lcp) == pytest.approx(0.0, abs=1e-15)
    assert ps.pitch_moment_coefficient(C_V, lam, 6.0, 15.0, lcp - 0.2) > 0.0  # c.p. aft of CG
    assert ps.pitch_moment_coefficient(C_V, lam, 6.0, 15.0, lcp + 0.2) < 0.0


# --------------------------------------------------------------------------- #
# spray drag and friction
# --------------------------------------------------------------------------- #
def test_spray_drag_follows_prescribed_law():
    for C_V in (1.0, 2.0, 4.0, 6.0):
        for b_over_L in (0.1, 0.25):
            assert ps.spray_drag_coefficient(C_V, BEAM, BEAM / b_over_L) == pytest.approx(
                0.003 * math.sqrt(b_over_L) * C_V ** 2, rel=1e-14)
    # force grows as C_V^4 (dC_D already carries C_V^2) -- the documented blow-up
    assert ps.spray_drag_force(4.0, BEAM, HULL_LENGTH) / ps.spray_drag_force(
        2.0, BEAM, HULL_LENGTH) == pytest.approx(16.0, rel=1e-12)
    with pytest.raises(ValueError):
        ps.spray_drag_coefficient(2.0, BEAM, 1.0)      # beam > length


def test_ittc57_friction_reference_value():
    assert ps.ittc57_friction_coefficient(1e9) == pytest.approx(0.075 / 49.0, rel=1e-14)
    assert ps.ittc57_friction_coefficient(1e8) > ps.ittc57_friction_coefficient(1e9)
    r_f, re, cf = ps.frictional_resistance(2.0, 3.0, BEAM)
    speed = ps.speed_from_coefficient(2.0, BEAM)
    assert re == pytest.approx(speed * 3.0 * BEAM / ps.NU_SEA_WATER, rel=1e-12)
    assert r_f == pytest.approx(cf * 0.5 * ps.RHO_SEA_WATER * speed ** 2 * 3.0 * BEAM ** 2,
                                rel=1e-12)


# --------------------------------------------------------------------------- #
# running condition
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def design():
    return ps.running_condition(2.0, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH)


def test_equilibrium_satisfies_lift_and_moment_equations(design):
    assert design.C_L == pytest.approx(design.C_L_required, rel=1e-9)
    assert design.lcp_over_beam == pytest.approx(LCG_OVER_BEAM, rel=1e-9)
    assert design.in_envelope
    # regression pins for the reference hull (C_V = 2)
    assert design.lam == pytest.approx(2.9622, rel=1e-3)
    assert design.tau_deg == pytest.approx(3.9077, abs=5e-3)
    assert design.speed == pytest.approx(8.8574, rel=1e-3)


def test_equilibrium_residuals_over_full_sweep():
    states = ps.trim_curve([1.0 + 0.1 * i for i in range(61)], BEAM, WEIGHT, BETA,
                           LCG_OVER_BEAM, HULL_LENGTH)
    assert len(states) == 61
    assert max(abs(s.C_L - s.C_L_required) for s in states) < 1e-9
    assert max(abs(s.lcp_over_beam - LCG_OVER_BEAM) for s in states) < 1e-9
    lam = [s.lam for s in states]
    tau = [s.tau_deg for s in states]
    assert lam == sorted(lam, reverse=True)             # wetted length shrinks w/ speed
    lam_asymptote = LCG_OVER_BEAM / 0.75                # lambda -> LCG/(0.75 b)
    assert abs(lam[-1] - lam_asymptote) < 0.5
    assert tau[-1] < tau[0]                             # trim falls with speed
    assert all(s.tau_deg > 0.0 for s in states)


def test_resistance_build_up_identity(design):
    expected = (WEIGHT * math.tan(math.radians(design.tau_deg))
                + (design.friction_force + design.spray_force)
                * math.cos(math.radians(design.tau_deg)))
    assert design.resistance == pytest.approx(expected, rel=1e-14)
    assert design.resistance > 0.0
    assert design.friction_force > 0.0 and design.spray_force > 0.0
    assert design.lift_to_drag == pytest.approx(WEIGHT / design.resistance, rel=1e-14)
    assert 4.0 < design.lift_to_drag < 10.0             # planing L/D band
    # spray is a sizeable but not dominant part of the design-point drag
    assert 0.15 < design.spray_force / design.resistance < 0.40
    assert design.resistance > WEIGHT * math.tan(math.radians(design.tau_deg))


def test_envelope_flagging():
    assert ps.envelope_violations(2.0, 2.0, 6.0, 15.0) == []
    assert ps.envelope_violations(0.3, 2.0, 6.0, 15.0) == ["C_V"]
    assert ps.envelope_violations(2.0, 6.0, 6.0, 15.0) == ["lam"]
    assert ps.envelope_violations(2.0, 2.0, 1.0, 15.0) == ["tau"]
    assert ps.envelope_violations(2.0, 2.0, 6.0, 35.0) == ["beta"]
    bad = ps.running_condition(0.4, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH)
    assert not bad.in_envelope and bad.violations


# --------------------------------------------------------------------------- #
# stability / porpoising criteria
# --------------------------------------------------------------------------- #
def test_stability_derivative_signs():
    """dC_L/dlam > 0 (heave restoring) and dM*/dlam > 0 (c.p. migrates aft)."""
    diag = [ps.stability_diagnostics(cv, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH)
            for cv in (1.0, 2.0, 3.0, 5.0)]
    assert all(d.dC_L_dlam > 0.0 for d in diag)
    assert all(d.dM_dlam > 0.0 for d in diag)
    assert all(d.dCL_dlam_positive for d in diag)


def test_quasi_static_stiffness_is_indefinite_everywhere():
    """Falsified criterion, pinned: the rate-free 2-DOF stiffness never discriminates.

    ``J_tt < 0`` because a bow-up perturbation shortens the wetted length and
    walks the c.p. forward of the CG; with no rate terms there is no restoring
    pitch stiffness.  The flag ``quasi_static_unstable`` must therefore stay
    ``True`` for every planing condition -- documenting this is the point of the
    test, so it cannot silently start "predicting" stability.
    """
    diag = [ps.stability_diagnostics(cv, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH)
            for cv in [1.0 + 0.5 * i for i in range(9)]]
    assert all(d.quasi_static_unstable for d in diag)
    assert all(d.j_tt < 0.0 for d in diag)
    assert all(d.stiffness_det < 0.0 for d in diag)
    assert all(d.j_zz > 0.0 for d in diag)


def test_quasi_static_stiffness_is_step_converged():
    a = ps.quasi_static_stiffness(2.0, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH,
                                  delta=1e-5)
    b = ps.quasi_static_stiffness(2.0, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH,
                                  delta=2e-5)
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            assert x == pytest.approx(y, rel=5e-3)


def test_trim_slope_criterion_localises_the_hump_regime():
    """The discriminating criterion flags a finite band, not everything.

    On the reference hull ``d tau/dC_V > 0`` only in the hump/low-planing band
    ``C_V ~ 0.8-2.1``; above it the trim falls with speed (stable) and the
    flag clears.  Also checks that the sign criterion and the flag agree.
    """
    cvs = [0.8 + 0.1 * i for i in range(62)]
    diag = [ps.stability_diagnostics(cv, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH)
            for cv in cvs]
    assert all(d.porpoising_prone == (not d.trim_slope_stable) for d in diag)
    prone = [d.C_V for d in diag if d.porpoising_prone]
    assert prone, "expected a porpoising-prone band in the hump regime"
    assert min(prone) == pytest.approx(0.8)            # sweep floor
    assert 1.8 <= max(prone) <= 2.3
    stable = [d for d in diag if not d.porpoising_prone]
    assert stable and all(d.C_V > max(prone) for d in stable)
    # trim slope must change sign at the band edge (not just flicker)
    assert max(d.trim_slope for d in diag) > 1.0
    assert min(d.trim_slope for d in diag) < -0.3


def test_porpoising_boundary_reports_both_criteria():
    sweep = [0.8 + 0.1 * i for i in range(62)]
    rows = [ps.porpoising_boundary(sweep, beta, BEAM, WEIGHT, LCG_OVER_BEAM, HULL_LENGTH)
            for beta in (5.0, 15.0, 30.0)]
    for row in rows:
        assert row["unstable_interval_stiffness"] == [0.8, pytest.approx(6.9)]
        assert row["n_stiffness_unstable"] == len(sweep)
        lo, hi = row["unstable_interval_trim_slope"]
        assert lo == pytest.approx(0.8) and 1.8 <= hi <= 2.3
    # deadrise amplifies the trim-up slope of the hump band (model result)
    peaks = [row["max_trim_slope"] for row in rows]
    assert peaks == sorted(peaks)
    assert peaks[-1] > 1.5 * peaks[0]


def test_trim_slope_matches_an_independent_finite_difference():
    cv, h = 2.5, 1e-5
    fast = ps.trim_slope(cv, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH)
    slow = ps.trim_slope(cv, BEAM, WEIGHT, BETA, LCG_OVER_BEAM, HULL_LENGTH, h=h)
    assert fast == pytest.approx(slow, rel=1e-6)


def test_porpoising_map_covers_the_grid():
    grid = ps.porpoising_map([1.0, 2.0, 4.0], [10.0, 20.0], BEAM, WEIGHT,
                             LCG_OVER_BEAM, HULL_LENGTH)
    assert len(grid) == 6
    assert {d.beta_deg for d in grid} == {10.0, 20.0}
    assert {d.C_V for d in grid} == {1.0, 2.0, 4.0}
    assert all("porpoising_prone" in d.as_dict() for d in grid)