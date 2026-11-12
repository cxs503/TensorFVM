"""Tests for :mod:`tensorfvm.rotating_frame` and its verification driver.

The tests assert *invariants* (exact zero, exact solution, frame invariance,
convergence order) rather than merely executing the code, and two of them are
control tests: they assert that a solver which drops (or sign-flips) the
Coriolis term *cannot* pass the anchor.  Without those, a green suite would not
distinguish a correct MRF implementation from an inert one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.rotating_frame import (
    RotatingFrame,
    exact_circular_couette,
    exact_rotating_channel_spanwise,
    exact_rotating_plane_poiseuille,
    solve_circular_couette_frame,
    solve_inert_plane_poiseuille,
    solve_rotating_plane_poiseuille,
)

CH = {"G": 1.0, "rho": 1.0, "nu": 0.05, "omega": 2.0, "h": 1.0}
CO = {"r1": 1.0, "r2": 2.0, "omega_inner": 1.0, "omega_outer": 0.0}


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    den = float(np.linalg.norm(b))
    num = float(np.linalg.norm(a - b))
    return num / den if den else num


# ---------------------------------------------------------------- kinematics
def test_axis_is_normalised_and_zero_axis_rejected():
    frame = RotatingFrame(omega=1.0, axis=(0.0, 0.0, 4.0))
    assert frame.axis == (0.0, 0.0, 1.0)
    assert frame.omega_vector.tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        RotatingFrame(omega=1.0, axis=(0.0, 0.0, 0.0))


def test_inert_frame_has_identically_zero_sources():
    frame = RotatingFrame(omega=0.0)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(17, 3))
    x = rng.normal(size=(17, 3))
    assert np.all(frame.coriolis(v) == 0.0)
    assert np.all(frame.centrifugal(x) == 0.0)
    assert np.all(frame.momentum_sources(v, x, 1.0)["total"] == 0.0)
    p = rng.normal(size=17)
    assert np.array_equal(frame.modified_pressure(p, x, 1000.0), p)


def test_coriolis_component_structure_catches_axis_swap():
    # Omega = omega y_hat, u = (u, 0, w)  =>  2 Omega x u = 2 omega (w, 0, -u)
    frame = RotatingFrame(omega=3.0, axis=(0.0, 1.0, 0.0))
    got = frame.coriolis(np.array([[0.5, 0.0, -0.25]]))[0]
    assert got == pytest.approx([-1.5, 0.0, -3.0])
    # the factor 2 is what hand-written implementations lose: pin it explicitly
    assert frame.coriolis(np.array([[1.0, 0.0, 0.0]]))[0][2] == pytest.approx(-6.0)


def test_omega_sign_flip_equals_axis_flip():
    a = RotatingFrame(omega=2.5, axis=(0.0, 1.0, 0.0))
    b = RotatingFrame(omega=-2.5, axis=(0.0, 1.0, 0.0))
    rng = np.random.default_rng(1)
    v = rng.normal(size=(9, 3))
    x = rng.normal(size=(9, 3))
    assert np.allclose(a.coriolis(v), -b.coriolis(v), rtol=0.0, atol=0.0)
    assert np.allclose(a.centrifugal(x), b.centrifugal(x), rtol=0.0, atol=0.0)


def test_absolute_relative_velocity_roundtrip():
    frame = RotatingFrame(omega=1.3, axis=(0.3, 0.5, 0.8), origin=(1.0, -2.0, 0.5))
    rng = np.random.default_rng(2)
    v = rng.normal(size=(21, 3))
    x = rng.normal(size=(21, 3))
    assert _rel(frame.relative_velocity(frame.absolute_velocity(v, x), x), v) < 1e-15


def test_modified_pressure_roundtrip_and_gradient_identity():
    frame = RotatingFrame(omega=0.9, axis=(0.0, 0.0, 1.0), origin=(0.2, 0.0, 0.0))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(15, 3))
    p = rng.normal(size=15)
    rho = 1025.0
    assert np.allclose(
        frame.pressure_from_modified(frame.modified_pressure(p, x, rho), x, rho),
        p,
        rtol=0.0,
        atol=1e-9,
    )
    # grad(rho/2 |Omega x r|^2) = -rho Omega x (Omega x r), by central differences
    hd = 1e-6
    grad = np.zeros_like(x)
    for k in range(3):
        e = np.zeros(3)
        e[k] = hd
        grad[:, k] = rho * (
            frame.centrifugal_potential(x + e) - frame.centrifugal_potential(x - e)
        ) / (2.0 * hd)
    assert _rel(grad, -rho * frame.centrifugal(x)) < 1e-6


def test_solid_body_rotation_is_an_exact_solution():
    frame = RotatingFrame(omega=1.1, axis=(0.6, 0.0, 0.8), origin=(0.4, 0.0, -1.0))
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 3))
    rho = 1000.0
    p_solid = rho * frame.centrifugal_potential(x)  # inertial solid-body pressure
    assert np.all(frame.coriolis(np.zeros_like(x)) == 0.0)
    assert np.allclose(frame.modified_pressure(p_solid, x, rho), 0.0, rtol=0.0, atol=1e-12)
    reduced = frame.momentum_sources(
        np.zeros_like(x), x, rho, include_centrifugal=False
    )["total"]
    assert np.all(reduced == 0.0)
    # p = rho/2|Omega x r|^2 is exactly the potential whose gradient cancels the
    # centrifugal force: verify on the field, not on the algebra.
    hd = 1e-6
    grad = np.zeros_like(x)
    for k in range(3):
        e = np.zeros(3)
        e[k] = hd
        grad[:, k] = (
            rho * frame.centrifugal_potential(x + e)
            - rho * frame.centrifugal_potential(x - e)
        ) / (2.0 * hd)
    assert np.max(np.abs(-grad / rho - frame.centrifugal(x))) < 1e-6 * np.max(
        np.abs(frame.centrifugal(x))
    )


def test_sources_are_negated_coriolis_plus_optional_centrifugal():
    frame = RotatingFrame(omega=2.0, axis=(0.0, 0.0, 1.0))
    v = np.array([[0.3, -0.2, 0.1]])
    x = np.array([[1.0, 2.0, 0.0]])
    src = frame.momentum_sources(v, x, 1.0)
    assert np.allclose(src["coriolis"], -frame.coriolis(v))
    assert np.allclose(src["total"], src["coriolis"] + src["centrifugal"])
    absorbed = frame.momentum_sources(v, x, 1.0, include_centrifugal=False)
    assert np.allclose(absorbed["total"], -frame.coriolis(v))
    assert np.all(absorbed["centrifugal"] == 0.0)


# ------------------------------------------------- rotating plane Poiseuille
def test_rotating_channel_closed_form_satisfies_the_odes():
    y = np.linspace(0.0, CH["h"], 41)
    ex = exact_rotating_plane_poiseuille(y, **CH)
    a = ex["a"]
    Fp = 1j * CH["G"] / (2.0 * CH["rho"] * CH["omega"])
    ch = np.cosh(a * 0.5 * CH["h"])
    F2 = -(a**2) * Fp * np.cosh(a * (y - 0.5 * CH["h"])) / ch
    u2, w2 = F2.real, -F2.imag
    scale = CH["G"] / CH["rho"]
    assert np.max(np.abs(CH["nu"] * u2 + 2.0 * CH["omega"] * ex["w"] + CH["G"] / CH["rho"])) < 1e-13 * scale
    assert np.max(np.abs(CH["nu"] * w2 - 2.0 * CH["omega"] * ex["u"])) < 1e-13 * scale
    # no-slip is part of the anchor
    assert ex["u"][0] == pytest.approx(0.0, abs=1e-15)
    assert ex["w"][0] == pytest.approx(0.0, abs=1e-15)


def test_omega_zero_branch_reduces_to_parabolic_poiseuille():
    y = np.linspace(0.0, CH["h"], 21)
    ex = exact_rotating_plane_poiseuille(y, **{**CH, "omega": 0.0})
    ref = CH["G"] * y * (CH["h"] - y) / (2.0 * CH["rho"] * CH["nu"])
    assert _rel(ex["u"], ref) < 1e-15
    assert np.all(ex["w"] == 0.0)


def test_finite_volume_converges_to_the_closed_form_at_second_order():
    errs = {}
    for n in (80, 160, 320):
        num = solve_rotating_plane_poiseuille(n, **CH)
        ref = exact_rotating_plane_poiseuille(num["y"], **CH)
        errs[n] = (_rel(num["u"], ref["u"]), _rel(num["w"], ref["w"]))
    assert errs[320][0] < 0.03
    assert errs[320][1] < 0.03
    order_u = math.log2(errs[80][0] / errs[160][0])
    order_w = math.log2(errs[80][1] / errs[160][1])
    assert order_u > 1.8
    assert order_w > 1.8


def test_solver_without_coriolis_cannot_pass_the_secondary_flow_anchor():
    """Control: the anchor must reject a wrong implementation."""
    num = solve_rotating_plane_poiseuille(320, **{**CH, "omega": 0.0})
    ref = exact_rotating_plane_poiseuille(num["y"], **CH)
    assert np.max(np.abs(num["w"])) < 1e-18  # no rotation -> no secondary flow
    assert np.max(np.abs(ref["w"])) > 1e-2  # the anchor's secondary flow is real
    assert _rel(num["w"], ref["w"]) > 0.5  # ... so the error is O(100 %)


def test_secondary_flow_is_linear_in_omega_and_saturates_at_high_omega():
    """Two regimes the anchor must show (both are physics, not artefacts).

    Small ``Omega``: the secondary flow is a linear perturbation of the parabolic
    profile (``w_max ~ Omega``, ``u`` unchanged to O(Omega^2)) -- so a "small
    rotation" run cannot be used to claim the Coriolis term is negligible.
    Large ``Omega``: the streamwise profile collapses as ``u ~ 1/Omega`` (the
    Coriolis term dominates the pressure drive) and ``w/u`` saturates near 3.31.
    """
    y = np.linspace(0.0, CH["h"], 201)

    def amp(omega):
        ex = exact_rotating_plane_poiseuille(y, **{**CH, "omega": omega})
        return np.max(np.abs(ex["u"])), np.max(np.abs(ex["w"]))

    u0, _ = amp(0.0)
    u_small, w_small = amp(1e-3)
    u_ten, w_ten = amp(1e-2)
    u_tiny, w_tiny = amp(1e-4)
    assert w_small / w_tiny == pytest.approx(10.0, rel=1e-2)  # w linear in Omega
    # the streamwise profile is only changed at O(Omega^2) -- measured, not assumed
    assert (u0 - u_ten) / (u0 - u_small) == pytest.approx(100.0, rel=1e-1)
    assert abs(u0 - u_small) / u0 < 1e-4
    u_big, w_big = amp(20.0)
    u_huge, w_huge = amp(100.0)
    assert u_huge / u_big == pytest.approx(20.0 / 100.0, rel=2e-2)  # u ~ 1/Omega
    assert w_huge / u_huge == pytest.approx(3.31, rel=2e-2)  # saturated ratio


# ------------------------------------------------ rotation along the span
def test_spanwise_axis_leaves_the_parabolic_profile_unchanged():
    n = 320
    ys = (np.arange(n) + 0.5) * (CH["h"] / n)
    span = exact_rotating_channel_spanwise(ys, **CH)
    num = solve_inert_plane_poiseuille(
        n, **{k: CH[k] for k in ("G", "rho", "nu", "h")}
    )
    assert _rel(num["u"], span["u"]) < 1e-12


def test_spanwise_axis_pressure_correction_balances_the_coriolis_force():
    ys = np.linspace(0.0, CH["h"], 41)
    span = exact_rotating_channel_spanwise(ys, **CH)
    frame = RotatingFrame(omega=CH["omega"], axis=(0.0, 0.0, 1.0))
    vel = np.stack([span["u"], np.zeros_like(ys), np.zeros_like(ys)], axis=-1)
    cor_y = frame.coriolis(vel)[:, 1]
    resid = cor_y + span["dpdy"] / CH["rho"]
    assert np.max(np.abs(resid)) < 1e-12 * np.max(np.abs(cor_y))
    # control: dropping the Coriolis term violates the balance by O(100 %)
    drop = np.max(np.abs(span["dpdy"] / CH["rho"])) / np.max(np.abs(cor_y))
    assert drop > 0.5


# ------------------------------------------------------- circular Couette MRF
def test_circular_couette_boundary_conditions_and_torque_signs():
    ex = exact_circular_couette(np.array([CO["r1"], CO["r2"]]), **CO)
    assert ex["v_theta"][0] == pytest.approx(CO["omega_inner"] * CO["r1"], rel=1e-14)
    assert ex["v_theta"][1] == pytest.approx(CO["omega_outer"] * CO["r2"], abs=1e-14)
    # inner cylinder faster than the outer: torque on the fluid is negative
    assert ex["torque_per_length_coefficient"] < 0.0


def test_couette_solution_in_a_rotating_frame_is_frame_invariant():
    abs_ref = None
    torque_ref = None
    for omega_frame in (0.0, CO["omega_inner"], 1.7 * CO["omega_inner"]):
        num = solve_circular_couette_frame(200, nu=1e-3, omega_frame=omega_frame, **CO)
        exact = exact_circular_couette(num["r"], **CO)
        assert _rel(num["v_abs"], exact["v_theta"]) < 1e-5
        if abs_ref is None:
            abs_ref, torque_ref = num["v_abs"], num["torque_coefficient"]
        else:
            assert _rel(num["v_abs"], abs_ref) < 1e-11
            assert abs(num["torque_coefficient"] - torque_ref) / abs(torque_ref) < 1e-11


def test_frozen_rotor_makes_inner_wall_relative_velocity_zero():
    num = solve_circular_couette_frame(
        64, nu=1e-3, omega_frame=CO["omega_inner"], **CO
    )
    assert num["v_rel"][0] == pytest.approx(0.0, abs=1e-18)


def test_couette_torque_matches_taylor_closed_form():
    mu = 800.0 * 1e-3
    num = solve_circular_couette_frame(320, nu=1e-3, omega_frame=0.0, **CO)
    exact = exact_circular_couette(num["r"], **CO)
    taylor = mu * exact["torque_per_length_coefficient"]
    assert abs(mu * num["torque_coefficient"] - taylor) / abs(taylor) < 0.03


def test_rotating_frame_radial_balance_uses_the_implemented_operators():
    omega_frame = 1.0
    rr = np.linspace(CO["r1"] * 1.05, CO["r2"] * 0.95, 15)
    exact = exact_circular_couette(rr, **CO)
    frame = RotatingFrame(omega=omega_frame, axis=(0.0, 0.0, 1.0))
    v_rel = exact["v_theta"] - omega_frame * rr
    pts = np.stack([rr, np.zeros_like(rr), np.zeros_like(rr)], axis=-1)
    vel = np.stack([np.zeros_like(rr), v_rel, np.zeros_like(rr)], axis=-1)
    cor = frame.coriolis(vel)[:, 0]
    cen = frame.centrifugal(pts)[:, 0]
    conv = -(v_rel**2) / rr
    residual = conv + cor + exact["dpdr_over_rho"] + cen
    scale = np.max(np.abs(np.stack([conv, cor, exact["dpdr_over_rho"], cen])), axis=0)
    assert np.max(np.abs(residual) / scale) < 1e-14
    # control: the centrifugal term is not optional in the rotating frame
    assert np.max(np.abs(residual - cen) / scale) > 0.05


# ------------------------------------------------------------- driver level
def test_verification_driver_all_criteria_pass():
    from tensorfvm.verification.rotating_frame import run

    summary = run()
    failed = [k for k, c in summary["criteria"].items() if not c["pass"]]
    assert not failed, "failed criteria: %s" % failed
    assert summary["all_pass"] is True


def test_validation_errors_are_loud():
    with pytest.raises(ValueError):
        solve_rotating_plane_poiseuille(2, **CH)
    with pytest.raises(ValueError):
        exact_circular_couette(np.array([1.0]), r1=2.0, r2=1.0, omega_inner=1.0, omega_outer=0.0)
    with pytest.raises(ValueError):
        solve_circular_couette_frame(10, nu=1e-3, r1=1.0, r2=1.0, omega_inner=1.0, omega_outer=0.0)
    frame = RotatingFrame(omega=1.0)
    with pytest.raises(ValueError):
        frame.coriolis(np.zeros((3, 2)))