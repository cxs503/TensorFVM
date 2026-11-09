"""Tests for the manoeuvring / self-propulsion face (node IN63-N02).

The tests encode the *achievements* that the node's report claims and nothing
else: geometry gate, blade-element/momentum equality, the actuator-disk bound,
self-propulsion closure, exact negative controls (``delta = 0``), the analytic
limits of the lifting-surface rudder model, agreement between the closed-form and
finite-difference derivatives, and the correctness of the manoeuvre-metric
extraction on synthetic trajectories (where the answer is known exactly).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.maneuvering import (
    dynamics as D, hull as H, hullforces as HF, propeller as P, resistance as R,
    rudder as RD, selfpropulsion as S,
)
from tensorfvm.maneuvering.reference import (
    KVLCC2_TURNING_BAND, SIMMAN2008_EFD, band_check,
)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_geometry_gate_passes():
    gate = H.geometry_gate()
    assert gate["passed"], gate["checks"]
    cb = gate["checks"]["Cb"]
    assert abs(cb["deviation_pct"]) < 0.5
    assert abs(gate["checks"]["LCB_pct"]["deviation"]) < 0.05


def test_reference_conditions_are_froude_consistent():
    c = H.DEFAULT_CONDITIONS
    assert c.froude == pytest.approx(0.142, abs=1e-4)
    # the farewell value quoted in older documents (0.771 m/s) is Fn = 0.093
    assert c.u_design != pytest.approx(0.771, abs=1e-3)
    assert c.reynolds == pytest.approx(7.23e6, rel=0.02)


def test_rudder_geometry_is_consistent():
    g = H.rudder_geometry()
    assert g.area == pytest.approx(g.span * g.chord)
    assert g.area / (g.length * g.draft) == pytest.approx(0.0169, rel=1e-12)
    assert -0.5 < g.x_r_over_l < -0.4
    assert 0.0 < g.slipstream_fraction < 1.0


# --------------------------------------------------------------------------- #
# propeller / BEMT
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def propeller():
    return P.PropellerGeometry()


def test_expanded_area_ratio_is_exact(propeller):
    assert propeller.achieved_area_ratio() == pytest.approx(0.800, rel=1e-9)


def test_momentum_blade_equality(propeller):
    """The BEMT solution must balance blade-element and momentum forces."""
    for J in (0.5, 0.6, 0.7, 0.8, 0.9):
        point = P.open_water_point(propeller, J, rho=998.2, n_rps=8.0)
        assert point.momentum_residual < 1e-8, (J, point.momentum_residual)


def test_betz_efficiency_bound(propeller):
    for J in (0.3, 0.5, 0.7, 0.9):
        point = P.open_water_point(propeller, J)
        assert point.eta <= P.ideal_efficiency(point.KT, J) + 1e-6


def test_open_water_curve_is_monotone(propeller):
    js = (0.5, 0.6, 0.7, 0.8, 0.9)
    points = [P.open_water_point(propeller, J) for J in js]
    kt = [p.KT for p in points]
    assert all(k1 > k2 for k1, k2 in zip(kt, kt[1:])), kt
    assert kt[-1] < 0.10
    assert 0.4 < max(p.eta for p in points) < 0.8


def test_open_water_zero_advance_has_thrust(propeller):
    point = P.open_water_point(propeller, 0.1)
    assert point.KT > 0.2 and point.KQ > 0.0


# --------------------------------------------------------------------------- #
# resistance / self-propulsion
# --------------------------------------------------------------------------- #
def test_resistance_chain_is_ittc78_and_flagged():
    comp = R.resistance_components()
    assert comp.r_total > 0
    assert comp.c_total == pytest.approx(comp.r_total / (
        0.5 * H.DEFAULT_CONDITIONS.rho * comp.speed ** 2 * comp.wetted_surface))
    assert "WITHHELD" in comp.provenance["wave_term"]
    # the anchor sits below the viscous chain -> the shortfall must be reported
    assert comp.band["c_chain_design"] > comp.band["c_t_anchor"]
    assert any("anchor C_T below" in flag for flag in comp.flags)


def test_resistance_scales_with_speed_squared():
    slow = R.model_scale_resistance(0.5)["R_T"]
    fast = R.model_scale_resistance(1.0)["R_T"]
    # C_F grows as the Reynolds number falls, so the scaling is slightly above u^2
    assert fast == pytest.approx(4.0 * slow, rel=0.20)


def test_self_propulsion_closure():
    point = S.solve_self_propulsion()
    assert abs(point.residual) < 1e-6
    assert point.thrust == pytest.approx(point.thrust_required, rel=1e-6)
    assert 0.0 < point.advance_ratio < 1.0
    assert point.rps > 0.0
    # effective power must be smaller than delivered power
    assert point.effective_power < point.delivered_power


def test_wake_fraction_reduces_shaft_rate():
    """A larger wake fraction lowers the inflow, hence the advance ratio, and so
    the shaft rate needed for the same thrust."""
    low_wake = S.solve_self_propulsion(wake=0.30)
    high_wake = S.solve_self_propulsion(wake=0.40)
    assert high_wake.rps < low_wake.rps
    assert high_wake.advance_ratio < low_wake.advance_ratio


# --------------------------------------------------------------------------- #
# rudder
# --------------------------------------------------------------------------- #
def test_analytic_polar_symmetry_and_slope():
    g = H.rudder_geometry()
    polar = RD.analytic_rudder_polar(g)
    assert polar.cl_at(0.0) == pytest.approx(0.0, abs=1e-12)
    assert polar.cl_at(10.0) == pytest.approx(-polar.cl_at(-10.0), rel=1e-12)
    slope = polar.lift_slope()
    assert 1.5 < slope < 2.0 * math.pi
    # Helmbold limits
    assert HF.lift_slope_finite_aspect(1e6) == pytest.approx(2 * math.pi, rel=1e-3)
    small = HF.lift_slope_finite_aspect(1e-3)
    assert small == pytest.approx(0.5 * math.pi * 1e-3, rel=1e-2)


def test_rudder_negative_control_is_exact():
    """delta = 0 with a symmetric flow gives zero lateral force and moment."""
    c = H.DEFAULT_CONDITIONS
    g = H.rudder_geometry()
    polar = RD.analytic_rudder_polar(g)
    result = RD.rudder_normal_force(0.0, c.u_design, 0.0, 0.0, polar=polar,
                                    geometry=g, conditions=c, KT=0.30, J=0.43)
    assert result["F_y"] == pytest.approx(0.0, abs=1e-12)
    assert result["M_z"] == pytest.approx(0.0, abs=1e-12)


def test_rudder_force_linear_at_small_angle():
    c = H.DEFAULT_CONDITIONS
    g = H.rudder_geometry()
    polar = RD.analytic_rudder_polar(g)
    forces = [RD.rudder_normal_force(math.radians(a), c.u_design, 0.0, 0.0,
                                     polar=polar, geometry=g, conditions=c,
                                     KT=0.30, J=0.43, linear=True)["F_N"]
              for a in (1.0, 2.0)]
    assert forces[1] / forces[0] == pytest.approx(2.0, rel=0.01)


def test_rudder_derivative_magnitudes_are_plausible():
    deriv = RD.rudder_derivatives(KT=0.30, J=0.43)
    assert deriv["F_N_delta_prime_per_rad"] > 0
    assert deriv["F_y_delta_prime_per_rad"] < 0
    assert 0.01 < deriv["F_N_delta_prime_per_rad"] < 0.10


# --------------------------------------------------------------------------- #
# hull forces
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def hull_model():
    return HF.HullForceModelSet()


def test_hull_force_zero_in_straight_ahead(hull_model):
    out = hull_model.force(H.DEFAULT_CONDITIONS.u_design, 0.0, 0.0)
    assert out["Y"] == pytest.approx(0.0, abs=1e-12)
    assert out["N"] == pytest.approx(0.0, abs=1e-12)


def test_hull_force_antisymmetric_in_sway(hull_model):
    u = H.DEFAULT_CONDITIONS.u_design
    plus = hull_model.force(u, 0.1 * u, 0.0)
    minus = hull_model.force(u, -0.1 * u, 0.0)
    assert plus["Y"] == pytest.approx(-minus["Y"], rel=1e-12)
    assert plus["N"] == pytest.approx(-minus["N"], rel=1e-12)
    assert plus["Y"] < 0.0          # drift to starboard -> force to port


def test_linear_derivatives_match_finite_differences(hull_model):
    deriv = HF.linear_derivatives(hull_model, 0.0, 0.0)
    c = H.DEFAULT_CONDITIONS
    q = 0.5 * c.rho * c.u_design ** 2 * c.length * c.draft
    eps = 1e-5
    y_plus = hull_model.force(c.u_design, eps * c.u_design, 0.0)["Y"]
    y_minus = hull_model.force(c.u_design, -eps * c.u_design, 0.0)["Y"]
    fd = (y_plus - y_minus) / (2 * eps) / q
    assert deriv["Y_v"] == pytest.approx(fd, rel=0.02)
    assert deriv["Y_v"] < 0.0 and deriv["N_v"] < 0.0


def test_operating_point_derivatives_differ_from_zero_drift(hull_model):
    zero = HF.linear_derivatives(hull_model, 0.0, 0.0)
    op = HF.linear_derivatives(hull_model, *D.STEADY_TURN_REFERENCE)
    assert op["Y_0"] != pytest.approx(0.0, abs=1e-6)
    assert abs(op["N_r"]) > abs(zero["N_r"])     # cross-flow damping recovered


# --------------------------------------------------------------------------- #
# metric extraction (synthetic trajectories with known answers)
# --------------------------------------------------------------------------- #
def _synthetic_circle(radius: float, length: float, n: int = 4000):
    """Constant-speed circle: heading 0 -> 2 pi, exact metrics known."""
    speed = 1.0
    dt = 2 * math.pi * radius / speed / n
    t = np.arange(n + 1) * dt
    theta = speed * t / radius
    return D.Trajectory(time=t, u=np.full_like(t, speed), v=np.zeros_like(t),
                        r=np.full_like(t, speed / radius),
                        x=radius * np.sin(theta), y=radius * (1 - np.cos(theta)),
                        psi=theta, delta=np.full_like(t, math.radians(35.0)),
                        dt=dt, mode="synthetic", meta={"length": length})


def test_turning_metrics_recover_a_circle():
    radius, length = 11.0, 7.0
    traj = _synthetic_circle(radius, length)
    metrics = D.turning_circle_metrics(traj)
    assert metrics["transfer_over_L"] == pytest.approx(radius / length, rel=0.01)
    assert metrics["tactical_diameter_over_L"] == pytest.approx(2 * radius / length,
                                                                rel=0.02)
    assert metrics["advance_over_L"] == pytest.approx(radius / length, rel=0.01)
    assert metrics["steady_turning_radius_over_L"] == pytest.approx(radius / length,
                                                                    rel=1e-9)


def test_zigzag_metrics_recover_known_overshoot():
    """A prescribed heading record with 4 deg overshoot must be reported as 4 deg."""

    class _Factory:  # minimal object exposing the attributes zigzag_metrics reads
        pass

    t = np.linspace(0.0, 100.0, 5001)
    psi = np.zeros_like(t)
    psi_a = math.radians(10.0)
    overshoot = math.radians(4.0)
    # triangle waves: rise to 10+4 deg, fall to -(10+4) deg, etc.
    for i, ti in enumerate(t):
        phase = (ti // 25.0) % 2
        frac = (ti % 25.0) / 25.0
        amplitude = psi_a + overshoot
        psi[i] = amplitude * (frac if phase == 0 else 1 - frac) * (1 if ti // 25.0 % 2 == 0 else -1)
    traj = D.Trajectory(time=t, u=np.ones_like(t), v=np.zeros_like(t),
                        r=np.zeros_like(t), x=np.zeros_like(t), y=np.zeros_like(t),
                        psi=psi, delta=np.zeros_like(t), dt=t[1] - t[0],
                        mode="synthetic", meta={"length": 7.0})
    metrics = D.zigzag_metrics(traj, psi_a_deg=10.0)
    assert metrics["psi_1_deg"] == pytest.approx(4.0, abs=0.2)


# --------------------------------------------------------------------------- #
# dynamics / negative controls
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def self_propulsion():
    return S.solve_self_propulsion()


def test_zero_rudder_goes_straight(self_propulsion):
    model = D.ManoeuvreModel(self_propulsion=self_propulsion, mode="direct")
    traj = D.simulate_turning_circle(model, 0.0, dt=0.5, t_end=60.0)
    assert abs(traj.y[-1]) < 1e-9
    assert abs(traj.r[-1]) < 1e-12
    assert traj.psi[-1] == pytest.approx(0.0, abs=1e-12)


def test_turning_direction_and_time_step_independence(self_propulsion):
    model = D.ManoeuvreModel(self_propulsion=self_propulsion, mode="direct")
    coarse = D.simulate_turning_circle(model, 35.0, dt=1.0, t_end=400.0)
    fine = D.simulate_turning_circle(model, 35.0, dt=0.25, t_end=400.0)
    m_coarse = D.turning_circle_metrics(coarse)
    m_fine = D.turning_circle_metrics(fine)
    assert m_fine["tactical_diameter_over_L"] > 0.0
    # starboard rudder -> positive yaw and positive lateral displacement
    assert m_fine["steady_r_1_s"] > 0.0
    assert m_fine["transfer_over_L"] > 0.0
    assert m_fine["tactical_diameter_over_L"] == pytest.approx(
        m_coarse["tactical_diameter_over_L"], rel=0.05)


def test_reference_bands_are_flagged_withheld():
    assert SIMMAN2008_EFD["status"] == "withheld"
    band = KVLCC2_TURNING_BAND["tactical_diameter_over_L"]
    assert band.status == "withheld_band"
    check = band_check(3.2, band)
    assert check["inside_band"] and "NOT a validation" in check["verdict"]