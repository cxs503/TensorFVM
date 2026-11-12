"""Verification driver for :mod:`tensorfvm.six_dof` (6DOF rigid-body motion).

Run::

    PYTHONPATH=src python -m tensorfvm.verification.six_dof

Writes ``docs/verification-six-dof/{summary.json,fig_six_dof.png}``.

``tensorfvm.six_dof`` is the rigid-body integrator used by ``wave_tank``
(seakeeping time-domain coupling) and by the maneuvering chain, but until this
driver it had **no test at all** (``tests/`` contained zero ``six_dof`` files).
Everything here is an analytic anchor that can be written down in closed form:

===========================  ==========================================
quantity                     closed form
===========================  ==========================================
heave restoring force        ``F = -rho g A_wp z``
pitch restoring moment       ``M = -rho g I_wp theta``
natural frequency            ``w_n = sqrt(k / m_eff)``
undamped free response       ``z = z0 cos(w_n t)``
damped free response         ``z = z0 e^{-zeta w_n t} cos(w_d t)``
steady harmonic amplitude    ``X = (F0/k) / sqrt((1-r^2)^2 + (2 zeta r)^2)``
===========================  ==========================================

The last row is the *time-domain* statement of the transfer function that
``SixDOFIntegrator.rao`` evaluates in closed form, so comparing the two is a
cross-method check (integrator vs formula), not a restatement of one formula.

Every criterion is pre-declared with its tolerance before it is evaluated; a
failure is reported as ``FAIL`` and never overwritten.  Three *control*
criteria pin down that the anchors actually reject broken implementations.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..six_dof import (
    SixDOFConfig,
    SixDOFIntegrator,
    hydrostatic_restoring_heave,
    hydrostatic_restoring_pitch,
)

__all__ = [
    "RHO",
    "G",
    "BARGE",
    "barge_config",
    "damped_analytic",
    "free_decay_peaks",
    "harmonic_amplitude_time_domain",
    "integrate_explicit_euler",
    "run",
    "plot",
    "main",
]

RHO = 1000.0
G = 9.81

#: Box barge used as the verification object (declared before any run).
#: L x B x T = 10 x 5 x 2 m, even keel, so
#: ``A_wp = L B = 50 m^2``, ``I_wp = B L^3 / 12 = 416.667 m^4``,
#: ``mass = rho L B T = 1.0e5 kg``.
BARGE = {"L": 10.0, "B": 5.0, "T": 2.0}

ARTIFACT_DIR = pathlib.Path(__file__).resolve().parents[3] / "docs" / "verification-six-dof"


def barge_config(
    free_dof: Sequence[str] = ("heave",),
    damping: Optional[Dict[str, float]] = None,
    added_mass: Optional[Dict[str, float]] = None,
) -> SixDOFConfig:
    """Config of the reference box barge (geometry fixed in :data:`BARGE`)."""
    L, B, T = BARGE["L"], BARGE["B"], BARGE["T"]
    return SixDOFConfig(
        mass=RHO * L * B * T,
        Ixx=RHO * L * B * T * (B ** 2 + T ** 2) / 12.0,
        Iyy=RHO * L * B * T * (L ** 2 + T ** 2) / 12.0,
        Izz=RHO * L * B * T * (L ** 2 + B ** 2) / 12.0,
        waterplane_area=L * B,
        waterplane_pitch_moi=B * L ** 3 / 12.0,
        free_dof=list(free_dof),
        damping=damping,
        added_mass=added_mass,
    )


def _criterion(value: float, gate: float, note: str, *, mode: str = "max") -> Dict:
    ok = value <= gate if mode == "max" else value >= gate
    return {"value": float(value), "gate": float(gate), "mode": mode,
            "pass": bool(ok), "note": note}


def damped_analytic(t: np.ndarray, z0: float, wn: float, zeta: float) -> np.ndarray:
    """Exact free response from rest (``z(0)=z0``, ``zdot(0)=0``).

    ``z = z0 e^{-zeta wn t} [cos(wd t) + (zeta wn / wd) sin(wd t)]``.  The sine
    term is *not* optional: dropping it gives a solution whose initial velocity
    is ``-zeta wn z0`` instead of zero, which looks like a 19 % amplitude error
    at ``zeta = 0.2`` (this was a real trap during bring-up).
    """
    wd = wn * math.sqrt(max(1.0 - zeta ** 2, 0.0))
    return z0 * np.exp(-zeta * wn * t) * (
        np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))


def free_decay_peaks(z: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Positive peaks (local maxima of ``z``) as ``(times, values)``."""
    idx = np.where((z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:]) & (z[1:-1] > 0.0))[0] + 1
    return t[idx], z[idx]


def harmonic_amplitude_time_domain(
    zeta: float,
    ratio: float,
    *,
    f0: float = 1.0e4,
    steps_per_period: int = 600,
    periods: int = 40,
    window_fraction: float = 0.25,
) -> Dict[str, float]:
    """Steady-state amplitude of the driven heave response, measured in time.

    The window covers the last ``window_fraction`` of the run so the startup
    transient (time constant ``1/(zeta w_n)``) is below the reported resolution.
    """
    cfg = barge_config(damping={"heave": 0.0})  # placeholder, overwritten below
    wn = SixDOFIntegrator(cfg, rho=RHO, g=G).natural_frequency_heave()
    m_eff = cfg.mass
    k = RHO * G * cfg.waterplane_area
    c = 2.0 * zeta * math.sqrt(k * m_eff)
    cfg = barge_config(damping={"heave": c})
    integ = SixDOFIntegrator(cfg, rho=RHO, g=G)

    omega = ratio * wn
    dt = (2.0 * math.pi / omega) / steps_per_period
    n = int(periods * steps_per_period)
    amp = 0.0
    t = 0.0
    start = int(n * (1.0 - window_fraction))
    for i in range(n):
        integ.step(dt, {"heave": f0 * math.cos(omega * t)})
        t += dt
        if i >= start:
            amp = max(amp, abs(integ.state.z))

    formula = (f0 / k) / math.sqrt((1.0 - ratio ** 2) ** 2 + (2.0 * zeta * ratio) ** 2)
    return {"time_domain": amp, "formula": formula,
            "rel_error": abs(amp - formula) / formula, "omega": omega,
            "dt": dt, "steps": n}


def integrate_explicit_euler(k: float, m: float, z0: float, dt: float, n: int) -> np.ndarray:
    """Non-symplectic (explicit) Euler on the same heave oscillator.

    Negative control for the energy criterion: it uses the *old* state for both
    updates, so the energy grows geometrically instead of staying bounded.
    """
    z, w = z0, 0.0
    out = np.empty(n + 1)
    out[0] = z
    for i in range(n):
        z_new = z + dt * w
        w_new = w - dt * (k / m) * z
        z, w = z_new, w_new
        out[i + 1] = z
    return out


def _heave_energy(z, w, k: float, m: float) -> np.ndarray:
    return 0.5 * m * np.asarray(w) ** 2 + 0.5 * k * np.asarray(z) ** 2


def run(*, quick: bool = False) -> Dict:
    """Evaluate all criteria; return ``{"criteria": ..., "data": ...}``."""
    t_start = time.time()
    crit: Dict[str, Dict] = {}
    data: Dict[str, object] = {}

    cfg = barge_config(free_dof=("heave", "pitch"))
    integ0 = SixDOFIntegrator(cfg, rho=RHO, g=G)
    k_heave = RHO * G * cfg.waterplane_area
    k_pitch = RHO * G * cfg.waterplane_pitch_moi
    wn_heave = math.sqrt(k_heave / cfg.mass)
    wn_pitch = math.sqrt(k_pitch / cfg.Iyy)
    data["reference"] = {
        "A_wp": cfg.waterplane_area, "I_wp": cfg.waterplane_pitch_moi,
        "mass": cfg.mass, "Iyy": cfg.Iyy,
        "C33": k_heave, "C55": k_pitch,
        "omega_n_heave": wn_heave, "omega_n_pitch": wn_pitch,
    }

    # ---------------------------------------------------------------- V1
    # Restoring coefficients: the functions must be *linear* in the displacement
    # with exactly the hydrostatic coefficient, and zero at equilibrium.
    probes = np.array([-0.4, -0.1, 0.0, 0.05, 0.3])
    heave_err = np.max(np.abs(
        np.array([hydrostatic_restoring_heave(z, RHO, G, cfg.waterplane_area) for z in probes])
        + k_heave * probes) / (k_heave * np.max(np.abs(probes))))
    pitch_err = np.max(np.abs(
        np.array([hydrostatic_restoring_pitch(p, RHO, G, cfg.waterplane_pitch_moi) for p in probes])
        + k_pitch * probes) / (k_pitch * np.max(np.abs(probes))))
    crit["V1_restoring_coefficients_exact"] = _criterion(
        max(heave_err, pitch_err), 1e-15,
        "max relative deviation of F=-rho g A_wp z and M=-rho g I_wp theta, "
        "including the zero at equilibrium")

    # ---------------------------------------------------------------- V2
    # Natural frequencies against sqrt(k/m_eff), including added mass.
    added = {"heave": 0.6 * cfg.mass, "pitch": 0.4 * cfg.Iyy}
    cfg_add = barge_config(free_dof=("heave", "pitch"), added_mass=added)
    integ_add = SixDOFIntegrator(cfg_add, rho=RHO, g=G)
    wn_heave_add = math.sqrt(k_heave / (cfg.mass + added["heave"]))
    wn_pitch_add = math.sqrt(k_pitch / (cfg.Iyy + added["pitch"]))
    err_wn = max(
        abs(integ_add.natural_frequency_heave() - wn_heave_add) / wn_heave_add,
        abs(integ_add.natural_frequency_pitch() - wn_pitch_add) / wn_pitch_add,
    )
    crit["V2_natural_frequency_exact"] = _criterion(
        err_wn, 1e-15,
        "sqrt(C33/(m+a33)) and sqrt(C55/(Iyy+a55)) with 60 %/40 % added mass")

    # ---------------------------------------------------------------- V3
    # Undamped free heave vs z0 cos(wn t): first order in dt, so the anchor is
    # the *observed order* (Richardson), plus a fine-grid error bound.
    z0 = 0.5
    periods = 3.0
    ladder: List[Dict[str, float]] = []
    n_base = 100 if not quick else 50
    for nstep in (n_base, 2 * n_base, 4 * n_base, 8 * n_base):
        dt = periods * (2.0 * math.pi / wn_heave) / nstep
        integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
        integ.state.z = z0
        z = np.empty(nstep + 1)
        z[0] = z0
        for i in range(nstep):
            integ.step(dt)
            z[i + 1] = integ.state.z
        t = np.arange(nstep + 1) * dt
        exact = z0 * np.cos(wn_heave * t)
        rel_l2 = float(np.linalg.norm(z - exact) / np.linalg.norm(exact))
        ladder.append({"n": nstep, "dt": dt, "rel_l2": rel_l2})
    order = math.log(ladder[-2]["rel_l2"] / ladder[-1]["rel_l2"]) / math.log(2.0)
    data["heave_ladder"] = ladder
    data["heave_ladder_order"] = order
    crit["V3a_undamped_harmonic_observed_order"] = _criterion(
        order, 0.85, "observed order p from the two finest grids vs z0 cos(wn t)",
        mode="min")
    crit["V3b_undamped_harmonic_fine_grid_error"] = _criterion(
        ladder[-1]["rel_l2"], 3e-2,
        f"relative L2 error at n={ladder[-1]['n']} steps over {periods:g} periods")
    crit["V3c_undamped_harmonic_order_upper"] = _criterion(
        order, 1.15, "observed order must not exceed 1.15: symplectic Euler is "
        "first order, a second-order reading would mean the anchor is not "
        "measuring what it claims")

    # ---------------------------------------------------------------- V4
    # Damped free response: the decay rate measured from the peak envelope must
    # match zeta * wn (logarithmic decrement), and the whole trace must track
    # the closed form.
    decay_rows = []
    worst_decay = 0.0
    worst_trace = 0.0
    for zeta in (0.05, 0.2):
        c = 2.0 * zeta * math.sqrt(k_heave * cfg.mass)
        integ = SixDOFIntegrator(barge_config(damping={"heave": c}), rho=RHO, g=G)
        integ.state.z = z0
        nstep = int((6.0 if not quick else 4.0) * 2000)
        dt = (2.0 * math.pi / wn_heave) / 2000
        z = np.empty(nstep + 1)
        z[0] = z0
        for i in range(nstep):
            integ.step(dt)
            z[i + 1] = integ.state.z
        t = np.arange(nstep + 1) * dt
        tp, zp = free_decay_peaks(z, t)
        slope = float(np.polyfit(tp[1:], np.log(zp[1:]), 1)[0]) if len(tp) > 2 else float("nan")
        measured = -slope
        exact = damped_analytic(t, z0, wn_heave, zeta)
        # A first-order scheme accumulates an O(dt * t) *phase* error, so a
        # multi-period pointwise comparison measures phase drift, not damping.
        # The gated window is therefore one natural period; the multi-period
        # value is reported (reported, not gated) in ``data`` as the phase
        # drift of the scheme.
        one_period = int(round((2.0 * math.pi / wn_heave) / dt)) + 1
        trace = float(np.linalg.norm(z[:one_period] - exact[:one_period])
                      / np.linalg.norm(exact[:one_period]))
        trace_full = float(np.linalg.norm(z - exact) / np.linalg.norm(exact))
        # Peak timing: compare the first interior peak of the numerical trace
        # with the first interior peak of the closed form on the same grid.
        te, ze = free_decay_peaks(exact, t)
        phase_drift = float(tp[0] - te[0]) if (len(tp) > 0 and len(te) > 0) else float("nan")
        decay_rows.append({"zeta": zeta, "measured_decay": measured,
                           "analytic_decay": zeta * wn_heave,
                           "rel_error": abs(measured - zeta * wn_heave) / (zeta * wn_heave),
                           "trace_rel_l2_one_period": trace,
                           "trace_rel_l2_full": trace_full,
                           "first_peak_time_shift_s": phase_drift})
        worst_decay = max(worst_decay, decay_rows[-1]["rel_error"])
        worst_trace = max(worst_trace, trace)
    data["damped_decay"] = decay_rows
    crit["V4a_damped_decay_rate"] = _criterion(
        worst_decay, 3e-3, "peak-envelope decay rate vs zeta*wn (zeta = 0.05, 0.2)")
    crit["V4b_damped_trace_vs_closed_form"] = _criterion(
        worst_trace, 3e-2,
        "relative L2 of the damped trace over the first natural period vs the "
        "closed form (multi-period pointwise comparison is dominated by the "
        "scheme's O(dt*t) phase drift, reported but not gated)")

    # ---------------------------------------------------------------- V5
    # Symplectic property: energy stays bounded with no secular drift.  The
    # explicit-Euler control must violate this, otherwise the criterion has no
    # discriminating power.
    nper = 50
    dt = (2.0 * math.pi / wn_heave) / (400 if not quick else 200)
    nstep = int(nper * (2.0 * math.pi / wn_heave) / dt)
    integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
    integ.state.z = z0
    zs = np.empty(nstep + 1)
    ws = np.empty(nstep + 1)
    zs[0], ws[0] = integ.state.z, integ.state.w
    for i in range(nstep):
        integ.step(dt)
        zs[i + 1], ws[i + 1] = integ.state.z, integ.state.w
    e_symp = _heave_energy(zs, ws, k_heave, cfg.mass)
    z_exp = integrate_explicit_euler(k_heave, cfg.mass, z0, dt, nstep)
    e_exp = _heave_energy(z_exp, np.gradient(z_exp, dt), k_heave, cfg.mass)
    e0 = e_symp[0]
    drift_symp = float(np.max(np.abs(e_symp - e0)) / e0)
    drift_exp = float(np.max(np.abs(e_exp - e0)) / e0)
    q = len(e_symp) // 5
    mean_first = float(np.mean(np.abs(e_symp[:q] - e0)) / e0)
    mean_last = float(np.mean(np.abs(e_symp[-q:] - e0)) / e0)
    data["energy"] = {"symplectic_max_drift": drift_symp,
                      "explicit_euler_max_drift": drift_exp,
                      "window_mean_first": mean_first,
                      "window_mean_last": mean_last,
                      "periods": nper}
    crit["V5a_energy_bounded"] = _criterion(
        drift_symp, 5e-2, f"max |E-E0|/E0 for symplectic Euler over {nper} periods")
    crit["V5b_energy_not_secular"] = _criterion(
        mean_last, 2.0 * max(mean_first, 1e-12),
        "late-window mean energy deviation <= 2x the early-window value "
        "(no growing trend)")
    crit["V5c_control_explicit_euler_grows"] = _criterion(
        drift_exp, 1.0,
        "control: non-symplectic explicit Euler on the same oscillator must grow "
        "by >= 100 % over the same window", mode="min")

    # ---------------------------------------------------------------- V6
    # The pitch DOF must behave like the heave DOF (same integrator path, other
    # state pair): harmonic response vs theta0 cos(wn t) and exact frequency.
    theta0 = 0.02
    integ = SixDOFIntegrator(barge_config(free_dof=("pitch",)), rho=RHO, g=G)
    integ.state.pitch = theta0
    nstep = 800
    dt = periods * (2.0 * math.pi / wn_pitch) / nstep
    th = np.empty(nstep + 1)
    th[0] = theta0
    for i in range(nstep):
        integ.step(dt)
        th[i + 1] = integ.state.pitch
    t = np.arange(nstep + 1) * dt
    rel_pitch = float(np.linalg.norm(th - theta0 * np.cos(wn_pitch * t))
                      / np.linalg.norm(theta0 * np.cos(wn_pitch * t)))
    data["pitch"] = {"rel_l2": rel_pitch, "omega_n": wn_pitch}
    crit["V6_pitch_dof_harmonic"] = _criterion(
        rel_pitch, 3e-2, f"pitch free response vs theta0 cos(wn t), {periods:g} periods")

    # ---------------------------------------------------------------- V7
    # RAO: closed-form transfer function vs the *measured* steady-state
    # amplitude of the driven integrator (cross-method), plus the damping-ratio
    # bookkeeping and the analytic limits.
    rao_rows = []
    worst_rao = 0.0
    ratios = (0.5, 0.9, 1.0, 1.5, 2.0)
    for zeta in (0.05, 0.2):
        for r in ratios:
            row = harmonic_amplitude_time_domain(
                zeta, r, periods=40 if not quick else 20,
                steps_per_period=600 if not quick else 300)
            row["zeta"] = zeta
            row["ratio"] = r
            rao_rows.append(row)
            worst_rao = max(worst_rao, row["rel_error"])
    data["rao_grid"] = rao_rows
    crit["V7a_rao_time_domain_vs_formula"] = _criterion(
        worst_rao, 3e-2,
        "steady-state amplitude from the integrator vs (F0/k)*rao(w) over "
        "zeta in {0.05,0.2} x r in {0.5,0.9,1,1.5,2}")

    zeta_cfg = 0.2
    c = 2.0 * zeta_cfg * math.sqrt(k_heave * cfg.mass)
    integ = SixDOFIntegrator(barge_config(damping={"heave": c}), rho=RHO, g=G)
    zeta_measured = integ.damping_ratio("heave")
    res = integ.rao(integ.natural_frequency_heave())
    crit["V7b_damping_ratio_from_config"] = _criterion(
        abs(zeta_measured - zeta_cfg) / zeta_cfg, 1e-12,
        "damping_ratio() recovers c/(2 sqrt(k m_eff)) from config.damping")
    crit["V7c_resonance_equals_one_over_2zeta"] = _criterion(
        abs(res - 1.0 / (2.0 * zeta_cfg)) / (1.0 / (2.0 * zeta_cfg)), 1e-12,
        "rao(omega_n) = 1/(2 zeta) with the configured damping")
    integ_nodef = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
    low = integ_nodef.rao(0.01 * wn_heave)
    high = integ_nodef.rao(100.0 * wn_heave)
    crit["V7d_rao_limits"] = _criterion(
        max(abs(low - 1.0), high), 1e-2, "rao -> 1 as omega -> 0 and rao -> 0 as omega -> inf")
    # Control: ignoring the configured damping (the historical hard-coded 5 %)
    # must be detectable at resonance, i.e. the anchor rejects the old code path.
    res_hardcoded = integ.rao(integ.natural_frequency_heave(), zeta=0.05)
    crit["V7e_control_ignoring_configured_damping"] = _criterion(
        abs(res_hardcoded - res) / res, 0.5,
        "control: the pre-fix hard-coded zeta=0.05 must disagree with the "
        "configured zeta=0.2 by >= 50 % at resonance", mode="min")

    # ---------------------------------------------------------------- V8
    # Independent integrator (scipy DOP853, rtol 1e-12) as a second opinion on
    # the damped trajectory; the 1st-order symplectic scheme is compared at a
    # fine dt so the discretisation error, not the method, is bounded.
    try:
        from scipy.integrate import solve_ivp  # noqa: WPS433 (optional dependency)

        zeta = 0.1
        c = 2.0 * zeta * math.sqrt(k_heave * cfg.mass)
        integ = SixDOFIntegrator(barge_config(damping={"heave": c}), rho=RHO, g=G)
        integ.state.z = z0
        nstep = 4000 if not quick else 2000
        dt = 5.0 * (2.0 * math.pi / wn_heave) / nstep
        z = np.empty(nstep + 1)
        z[0] = z0
        for i in range(nstep):
            integ.step(dt)
            z[i + 1] = integ.state.z
        t = np.arange(nstep + 1) * dt

        def rhs(_t, y):
            return [y[1], -(c / cfg.mass) * y[1] - (k_heave / cfg.mass) * y[0]]

        sol = solve_ivp(rhs, (0.0, t[-1]), [z0, 0.0], t_eval=t,
                        method="DOP853", rtol=1e-12, atol=1e-14)
        rel_ind = float(np.linalg.norm(z - sol.y[0]) / np.linalg.norm(sol.y[0]))
        data["independent_integrator"] = {"rel_l2": rel_ind, "method": "DOP853",
                                          "rtol": 1e-12, "steps": nstep}
        crit["V8_independent_integrator_crosscheck"] = _criterion(
            rel_ind, 1e-2,
            "damped trajectory vs scipy DOP853 (rtol 1e-12) on the same time grid")
    except ImportError:  # pragma: no cover - scipy is a [dev] extra
        data["independent_integrator"] = {"status": "NOT_RUN", "reason": "scipy missing"}
        crit["V8_independent_integrator_crosscheck"] = {
            "value": float("nan"), "gate": 1e-2, "mode": "max", "pass": False,
            "status": "NOT_RUN", "note": "scipy not installed: reported as NOT_RUN, not PASS"}

    # ---------------------------------------------------------------- V9
    # Bookkeeping: history times must follow the actual (non-uniform) dt.
    integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
    dts = [0.1, 0.05, 0.2, 0.025]
    for step in dts:
        integ.step(step)
    times = np.array([h["time"] for h in integ.history])
    time_err = float(np.max(np.abs(times - np.cumsum(dts))))
    crit["V9_history_time_nonuniform_dt"] = _criterion(
        time_err, 1e-12,
        "history['time'] equals the cumulative sum of the actual dt values "
        "(pre-fix it was len(history)*dt)")

    # --------------------------------------------------------------- V10
    # Control: a sign-flipped restoring term must be rejected by V3.
    class _Flipped(SixDOFIntegrator):
        def compute_forces(self, F_ext=None):
            forces = super().compute_forces(F_ext)
            forces["heave"] = -forces["heave"]
            return forces

    integ = _Flipped(barge_config(), rho=RHO, g=G)
    integ.state.z = z0
    dt = periods * (2.0 * math.pi / wn_heave) / 800
    z = np.empty(801)
    z[0] = z0
    for i in range(800):
        integ.step(dt)
        z[i + 1] = integ.state.z
    t = np.arange(801) * dt
    rel_flip = float(np.linalg.norm(z - z0 * np.cos(wn_heave * t))
                     / np.linalg.norm(z0 * np.cos(wn_heave * t)))
    crit["V10_control_flipped_restoring_rejected"] = _criterion(
        rel_flip, 0.5,
        "control: sign-flipped restoring must miss the harmonic anchor by >= 50 %",
        mode="min")

    return {
        "criteria": crit,
        "data": data,
        "meta": {
            "driver": "tensorfvm.verification.six_dof",
            "runtime_s": time.time() - t_start,
            "quick": quick,
        },
    }


def plot(summary: Dict, path: pathlib.Path) -> pathlib.Path:
    """Four-panel figure: harmonic trace, order ladder, energy, RAO."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, crit = summary["data"], summary["criteria"]
    ref = data["reference"]
    wn = ref["omega_n_heave"]
    k_heave = ref["C33"]
    mass = ref["mass"]
    z0 = 0.5

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) undamped free heave vs closed form
    integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
    integ.state.z = z0
    dt = (2.0 * math.pi / wn) / 200
    n = 400
    z = np.empty(n + 1)
    z[0] = z0
    for i in range(n):
        integ.step(dt)
        z[i + 1] = integ.state.z
    t = np.arange(n + 1) * dt
    ax = axes[0, 0]
    ax.plot(t * wn / (2 * math.pi), z0 * np.cos(wn * t), "k-", lw=1.0,
            label=r"$z_0\cos(\omega_n t)$")
    ax.plot(t * wn / (2 * math.pi), z, "C0--", lw=1.6, label="symplectic Euler")
    ax.set_xlabel(r"$t\,\omega_n/2\pi$ [-]")
    ax.set_ylabel("heave z [m]")
    ax.set_title("(a) undamped free heave vs closed form")
    ax.legend(fontsize=8, loc="lower right")

    # (b) error ladder
    ax = axes[0, 1]
    lad = data["heave_ladder"]
    ldt = np.array([r["dt"] for r in lad])
    lerr = np.array([r["rel_l2"] for r in lad])
    ax.loglog(ldt, lerr, "o-")
    p = data["heave_ladder_order"]
    ax.loglog(ldt, lerr[-1] * (ldt / ldt[-1]) ** 1.0, "k--", lw=0.8, label="slope 1")
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel(r"relative $L_2$ error [-]")
    ax.set_title(f"(b) observed order p = {p:.3f}")
    ax.legend(fontsize=8)

    # (c) energy: symplectic vs explicit Euler
    ax = axes[1, 0]
    nper = 25
    dt = (2.0 * math.pi / wn) / 400
    nstep = int(nper * (2.0 * math.pi / wn) / dt)
    integ = SixDOFIntegrator(barge_config(), rho=RHO, g=G)
    integ.state.z = z0
    zs = np.empty(nstep + 1)
    ws = np.empty(nstep + 1)
    zs[0], ws[0] = z0, 0.0
    for i in range(nstep):
        integ.step(dt)
        zs[i + 1], ws[i + 1] = integ.state.z, integ.state.w
    e_s = _heave_energy(zs, ws, k_heave, mass) / _heave_energy([z0], [0.0], k_heave, mass)[0]
    z_e = integrate_explicit_euler(k_heave, mass, z0, dt, nstep)
    e_e = _heave_energy(z_e, np.gradient(z_e, dt), k_heave, mass) / _heave_energy([z0], [0.0], k_heave, mass)[0]
    tt = np.arange(nstep + 1) * dt * wn / (2 * math.pi)
    ax.plot(tt, e_s, "C0-", lw=1.2, label="symplectic Euler")
    ax.plot(tt, e_e, "C3-", lw=1.2, label="explicit Euler (control)")
    ax.set_xlabel(r"$t\,\omega_n/2\pi$ [-]")
    ax.set_ylabel(r"$E/E_0$ [-]")
    ax.set_yscale("log")
    ax.set_title(f"(c) energy: bounded vs diverging (V5)")
    ax.legend(fontsize=8)

    # (d) RAO: time domain vs formula
    ax = axes[1, 1]
    grid = data["rao_grid"]
    for zeta, marker in ((0.05, "o"), (0.2, "s")):
        rows = [r for r in grid if r["zeta"] == zeta]
        rr = np.array([r["ratio"] for r in rows])
        ax.semilogy(rr, [r["time_domain"] for r in rows], marker, ms=6, ls="none",
                    label=f"time domain zeta={zeta}")
        ax.semilogy(rr, [r["formula"] for r in rows], "k-", lw=1.0)
    r_fine = np.linspace(0.1, 2.5, 240)
    for zeta in (0.05, 0.2):
        ax.semilogy(r_fine, 1.0 / np.sqrt((1 - r_fine ** 2) ** 2 + (2 * zeta * r_fine) ** 2),
                    "k--", lw=0.6)
    ax.set_xlabel(r"$\omega/\omega_n$ [-]")
    ax.set_ylabel("RAO [m/m]")
    ax.set_title("(d) RAO: integrator (markers) vs closed form (lines)")
    ax.legend(fontsize=8)

    fig.suptitle("TensorFVM six_dof verification (box barge 10x5x2 m)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    quick = bool(argv and "--quick" in argv)
    summary = run(quick=quick)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot(summary, ARTIFACT_DIR / "fig_six_dof.png")
    out = {
        "criteria": summary["criteria"],
        "data": summary["data"],
        "meta": summary["meta"],
        "figure": str(fig.relative_to(ARTIFACT_DIR.parent.parent)),
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")

    failed = [k for k, v in summary["criteria"].items() if not v["pass"]]
    for key, val in summary["criteria"].items():
        status = "PASS" if val["pass"] else "FAIL"
        print(f"{status:4s} {key:46s} {val['value']:.4e} (gate {val['gate']:.4e}, {val['mode']})")
    print(f"\n{len(summary['criteria']) - len(failed)}/{len(summary['criteria'])} criteria PASS"
          f"  runtime {summary['meta']['runtime_s']:.1f} s")
    print(f"artifacts: {ARTIFACT_DIR}")
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(__import__("sys").argv[1:]))