"""Tests for :mod:`tensorfvm.six_dof` and its verification driver.

The module had no test at all before this file: ``tests/`` contained zero
``six_dof`` tests even though ``wave_tank`` builds its floating-body motion on
it.  Every assertion below is tied to a closed-form anchor, and three of them
are *controls* (a wrong implementation must fail) so the anchors are shown to
have discriminating power.
"""

import math

import numpy as np
import pytest

from tensorfvm.six_dof import (
    SixDOFConfig,
    SixDOFIntegrator,
    hydrostatic_restoring_heave,
    hydrostatic_restoring_pitch,
)
from tensorfvm.verification.six_dof import (
    G,
    RHO,
    barge_config,
    damped_analytic,
    free_decay_peaks,
    harmonic_amplitude_time_domain,
    integrate_explicit_euler,
    run,
)

Z0 = 0.5


def _reference():
    cfg = barge_config()
    integ = SixDOFIntegrator(cfg, rho=RHO, g=G)
    k = RHO * G * cfg.waterplane_area
    return cfg, integ, k, math.sqrt(k / cfg.mass)


def test_restoring_coefficients_are_linear_and_exact():
    cfg = barge_config()
    k_heave = RHO * G * cfg.waterplane_area
    k_pitch = RHO * G * cfg.waterplane_pitch_moi
    for z in (-0.4, 0.0, 0.3):
        assert hydrostatic_restoring_heave(z, RHO, G, cfg.waterplane_area) == -k_heave * z
    for theta in (-0.2, 0.0, 0.1):
        assert hydrostatic_restoring_pitch(theta, RHO, G, cfg.waterplane_pitch_moi) == -k_pitch * theta
    assert hydrostatic_restoring_heave(0.0, RHO, G, cfg.waterplane_area) == 0.0


def test_natural_frequency_includes_added_mass():
    cfg = barge_config(free_dof=("heave", "pitch"), added_mass={"heave": 0.6 * 1e5})
    integ = SixDOFIntegrator(cfg, rho=RHO, g=G)
    expected = math.sqrt(RHO * G * cfg.waterplane_area / (cfg.mass + 0.6 * 1e5))
    assert integ.natural_frequency_heave() == pytest.approx(expected, rel=1e-15)


def test_undamped_free_heave_second_moment_and_order():
    """Symplectic Euler is first order in dt: the ladder must show p ~ 1."""
    cfg, _, _, wn = _reference()
    errors = []
    n = 200
    dt = 3 * (2 * math.pi / wn) / n
    for scale in (1, 2, 4):
        integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
        integ.state.z = Z0
        steps = n * scale
        dtt = dt / scale
        z = np.empty(steps + 1)
        z[0] = Z0
        for i in range(steps):
            integ.step(dtt)
            z[i + 1] = integ.state.z
        t = np.arange(steps + 1) * dtt
        errors.append(np.linalg.norm(z - Z0 * np.cos(wn * t))
                      / np.linalg.norm(Z0 * np.cos(wn * t)))
    order = math.log(errors[-2] / errors[-1]) / math.log(2.0)
    assert 0.85 < order < 1.15, f"observed order {order:.3f} is not first order"
    assert errors[-1] < 3e-2


def test_damped_analytic_has_zero_initial_velocity():
    """Trap found during bring-up: the sine term is required for zdot(0)=0."""
    wn, zeta = 2.0, 0.2
    h = 1e-6
    t = np.array([0.0, h])
    z = damped_analytic(t, Z0, wn, zeta)
    assert z[0] == pytest.approx(Z0, rel=1e-15)
    assert (z[1] - z[0]) / h == pytest.approx(0.0, abs=1e-6)


def test_damped_decay_rate_matches_zeta_wn():
    cfg, _, k, wn = _reference()
    for zeta in (0.05, 0.2):
        c = 2 * zeta * math.sqrt(k * cfg.mass)
        integ = SixDOFIntegrator(barge_config(damping={"heave": c}), rho=RHO, g=G)
        integ.state.z = Z0
        dt = (2 * math.pi / wn) / 1000
        n = int(4 * 1000)
        z = np.empty(n + 1)
        z[0] = Z0
        for i in range(n):
            integ.step(dt)
            z[i + 1] = integ.state.z
        t = np.arange(n + 1) * dt
        tp, zp = free_decay_peaks(z, t)
        measured = -np.polyfit(tp[1:], np.log(zp[1:]), 1)[0]
        assert abs(measured - zeta * wn) / (zeta * wn) < 3e-3
        exact = damped_analytic(t, Z0, wn, zeta)
        one_period = int(round((2 * math.pi / wn) / dt)) + 1
        rel = np.linalg.norm(z[:one_period] - exact[:one_period]) / np.linalg.norm(exact[:one_period])
        assert rel < 3e-2


def test_energy_bounded_for_symplectic_euler_with_explicit_euler_control():
    cfg, _, k, wn = _reference()
    dt = (2 * math.pi / wn) / 400
    n = int(20 * (2 * math.pi / wn) / dt)
    integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
    integ.state.z = Z0
    z = np.empty(n + 1)
    w = np.empty(n + 1)
    z[0], w[0] = Z0, 0.0
    for i in range(n):
        integ.step(dt)
        z[i + 1], w[i + 1] = integ.state.z, integ.state.w
    e0 = 0.5 * k * Z0 ** 2
    drift = np.max(np.abs(0.5 * cfg.mass * w ** 2 + 0.5 * k * z ** 2 - e0)) / e0
    assert drift < 5e-2, f"symplectic energy drift {drift:.2%}"

    z_e = integrate_explicit_euler(k, cfg.mass, Z0, dt, n)
    e_e = 0.5 * cfg.mass * np.gradient(z_e, dt) ** 2 + 0.5 * k * z_e ** 2
    assert np.max(np.abs(e_e - e0)) / e0 > 1.0, "control: explicit Euler must blow up"


def test_rao_resonance_uses_configured_damping():
    """Regression for the pre-fix defect: rao() ignored config.damping."""
    cfg, _, k, wn = _reference()
    zeta = 0.2
    c = 2 * zeta * math.sqrt(k * cfg.mass)
    integ = SixDOFIntegrator(barge_config(damping={"heave": c}), rho=RHO, g=G)
    assert integ.damping_ratio("heave") == pytest.approx(zeta, rel=1e-12)
    assert integ.rao(wn) == pytest.approx(1.0 / (2 * zeta), rel=1e-12)
    # the old hard-coded 5 % is recoverable only by asking for it explicitly
    assert integ.rao(wn, zeta=0.05) == pytest.approx(10.0, rel=1e-12)


def test_rao_default_is_backward_compatible():
    """No configured damping -> historical 5 % placeholder (wave_tank relies on it)."""
    _, integ, _, wn = _reference()
    assert integ.damping_ratio("heave") == pytest.approx(SixDOFIntegrator.DEFAULT_DAMPING_RATIO)
    assert integ.rao(wn) == pytest.approx(10.0, rel=1e-12)
    assert integ.rao(0.1 * wn) == pytest.approx(1.0, abs=5e-2)


def test_rao_limits():
    _, integ, _, wn = _reference()
    assert integ.rao(0.01 * wn) == pytest.approx(1.0, abs=1e-2)
    assert integ.rao(100.0 * wn) < 1e-2
    assert integ.rao(wn, dof="yaw") == 0.0


def test_rao_matches_time_domain_amplitude():
    row = harmonic_amplitude_time_domain(0.05, 1.0, periods=20, steps_per_period=300)
    assert row["rel_error"] < 3e-2


def test_pitch_dof_harmonic():
    cfg = barge_config(free_dof=("pitch",))
    integ = SixDOFIntegrator(cfg, rho=RHO, g=G)
    wn = integ.natural_frequency_pitch()
    theta0 = 0.02
    integ.state.pitch = theta0
    n = 400
    dt = 3 * (2 * math.pi / wn) / n
    th = np.empty(n + 1)
    th[0] = theta0
    for i in range(n):
        integ.step(dt)
        th[i + 1] = integ.state.pitch
    t = np.arange(n + 1) * dt
    rel = np.linalg.norm(th - theta0 * np.cos(wn * t)) / np.linalg.norm(theta0 * np.cos(wn * t))
    assert rel < 3e-2


def test_history_time_follows_nonuniform_dt():
    """Regression: history['time'] used to be len(history)*dt."""
    integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
    dts = [0.1, 0.05, 0.2, 0.025]
    for step in dts:
        integ.step(step)
    times = [h["time"] for h in integ.history]
    assert times == pytest.approx(list(np.cumsum(dts)), rel=1e-15)
    assert integ.time == pytest.approx(sum(dts), rel=1e-15)


def test_verification_suite_all_criteria_pass():
    # Full (not --quick) tolerances: the quick mode coarsens the RAO sweep and
    # the DOP853 cross-check enough to exceed the 3 % gates.
    summary = run()
    failed = {k: v for k, v in summary["criteria"].items() if not v["pass"]}
    assert not failed, f"failed criteria: {failed}"


def test_negative_control_flipped_restoring_rejected():
    class _Flipped(SixDOFIntegrator):
        def compute_forces(self, F_ext=None):
            forces = super().compute_forces(F_ext)
            forces["heave"] = -forces["heave"]
            return forces

    _, _, _, wn = _reference()
    integ = _Flipped(barge_config(), rho=RHO, g=G)
    integ.state.z = Z0
    n = 400
    dt = 3 * (2 * math.pi / wn) / n
    z = np.empty(n + 1)
    z[0] = Z0
    for i in range(n):
        integ.step(dt)
        z[i + 1] = integ.state.z
    t = np.arange(n + 1) * dt
    rel = np.linalg.norm(z - Z0 * np.cos(wn * t)) / np.linalg.norm(Z0 * np.cos(wn * t))
    assert rel > 0.5