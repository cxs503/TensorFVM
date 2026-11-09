"""Reproducible case drivers for the sigma-grid numerical wave tank.

Every ``run_*`` function is a complete experiment: it builds the tank, runs it,
reduces the time series with :mod:`tensorfvm.wave_tank.theory` and returns a
JSON-serialisable dict, so every number in the report is traceable to a run.
The drivers write their evidence under ``docs/wave-tank-td/artifacts/``.

Cases
-----
``run_geometry_gate``       Cb / L/B / B/T / nabla pre-gate (Wigley, KCS, section).
``run_wave_case``           generation -> propagation -> absorption: wave-height
                            error, three-probe wavenumber vs dispersion, Goda
                            two-point reflection coefficient.
``run_reflection_control``  the same case with the absorbing beach removed
                            (negative control: ``C_R`` must jump).
``run_propagation_case``    one wave from an analytic initial field (no
                            wavemaker): pure scheme accuracy for the grid /
                            time-step convergence study.
``run_rao_case``            heave/pitch section in regular head waves: RAO,
                            added resistance (near-field period mean) and the
                            control-surface momentum balance as a second route
                            to the same drift force.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..wave_tank.theory import (added_resistance_energy, goda_two_point,
                                harmonic_fit, three_probe_wavenumber)
from ..wave_tank.geometry import (BargeSection, KCSSimplifiedHull, WigleyHull,
                                  geometry_gate)
from .sigma_tank import SigmaWaveTank, TankConfig
from .wave_target import fit_harmonics

__all__ = [
    "run_geometry_gate", "run_wave_case", "run_reflection_control",
    "run_propagation_case", "run_rao_case", "rectangular_section", "save",
]

ARTIFACT_DIR = Path("docs/wave-tank-td/artifacts")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    return obj


def save(payload: dict, name: str, artifact_dir=None) -> Path:
    d = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------- gate
def run_geometry_gate(artifact_dir=None) -> dict:
    """Geometry pre-gate: every hull must reproduce its own coefficients."""
    wig = geometry_gate(WigleyHull().coefficients())
    kcs = geometry_gate(KCSSimplifiedHull().coefficients())
    strip = BargeSection(length=2.4, draft=0.30).coefficients()
    out = {"wigley": wig, "kcs_simplified": kcs, "strip_2d": strip,
           "passed": bool(wig["passed"] and kcs["passed"])}
    save(out, "geometry_gate.json", artifact_dir)
    return out


# --------------------------------------------------------------- wave cases
def run_wave_case(amp: float = 0.02, period: float = 1.0, nx: int = 200, ns: int = 24,
                  length: float = 12.0, depth: float = 1.0, damping: bool = True,
                  periods: float = 8.0, steps_per_period: int = 100,
                  blend: float = 0.15, name: str = "wave", artifact_dir=None) -> dict:
    """Regular-wave generation / propagation / absorption case."""
    cfg = TankConfig(length=length, depth=depth, nx=nx, ns=ns,
                     advection_blend=blend)
    if not damping:
        cfg.damp_strength = 0.0
    tank = SigmaWaveTank(cfg)
    wave = tank.set_wave(amp, period)
    lam = wave.wavelength
    dt = period / steps_per_period
    x_probe = [(2.0 + 0.25 * i) * lam for i in range(7)]
    x_goda = [2.5 * lam, 2.75 * lam]
    all_x = sorted(set(x_probe + x_goda))
    series = {x: [] for x in all_x}
    ts = []
    div_max = 0.0
    t0 = time.time()
    nsteps = int(round(periods * steps_per_period))
    for _ in range(nsteps):
        t = tank.step(dt)
        ts.append(t)
        for x in all_x:
            series[x].append(float(tank.eta_at(x)))
        div_max = max(div_max, tank.div_residual)
    wall = time.time() - t0
    t = np.asarray(ts)
    keep = t >= max(periods - 4.0, 0.0) * period
    amps, phases, means, resids, amp2 = {}, {}, {}, {}, {}
    for x in all_x:
        fit = fit_harmonics(t[keep], np.asarray(series[x])[keep], wave.omega, n_harm=2)
        amps[x], phases[x] = fit["amp"][0], fit["phase"][0]
        means[x], resids[x] = fit["mean"], fit["residual_rel"]
        amp2[x] = fit["amp"][1]
    a_probe = [amps[x] for x in x_probe]
    a_err = [abs(a / amp - 1.0) for a in a_probe]
    k_fit, _ = three_probe_wavenumber(x_probe, a_probe, [phases[x] for x in x_probe])
    cplx = lambda x: amps[x] * np.exp(1j * phases[x])          # noqa: E731
    A_I, A_R = goda_two_point(x_goda[0], x_goda[1], cplx(x_goda[0]),
                              cplx(x_goda[1]), wave.k)
    c_r = abs(A_R) / max(abs(A_I), 1e-30)
    out = {
        "case": name, "name": name, "damping": bool(damping),
        "config": {k: getattr(cfg, k) for k in
                   ("length", "depth", "nx", "ns", "advection_blend", "relax_waves",
                    "damp_waves", "damp_strength")},
        "wave": wave.as_dict(), "dt": dt, "steps": nsteps,
        "wall_seconds": wall, "ms_per_step": 1000.0 * wall / nsteps,
        "probes_x": x_probe, "probes_x_over_lambda": [x / lam for x in x_probe],
        "amp_first_harmonic": a_probe, "amp_ratio": [a / amp for a in a_probe],
        "amp_error": a_err, "amp_error_max": max(a_err),
        "amp_error_mean": float(np.mean(a_err)),
        "phase": [phases[x] for x in x_probe],
        "residual": [resids[x] for x in x_probe],
        "amp_second_harmonic": [amp2[x] for x in x_probe],
        "amp_second_harmonic_theory": float(
            wave.k * amp * amp / 4.0 * (3.0 - np.tanh(wave.k * depth) ** 2)
            / np.tanh(wave.k * depth) ** 3),
        "mean_elevation": [means[x] for x in x_probe],
        "k_three_probe": k_fit, "k_reference": wave.k,
        "k_error": abs(k_fit - wave.k) / wave.k,
        "goda": {"x1": x_goda[0], "x2": x_goda[1],
                 "k_dx": wave.k * (x_goda[1] - x_goda[0]),
                 "A_I_abs": abs(A_I), "A_R_abs": abs(A_R), "C_R": c_r},
        "volume_drift_rel": tank.volume_drift / max(tank.volume0, 1e-30),
        "div_residual_max": div_max,
        "eta_max_final": float(np.abs(tank.eta).max()),
        "series_x": all_x,
        "series": {"t": ts, **{f"eta_{i}": series[x] for i, x in enumerate(all_x)}},
    }
    out["criteria"] = {
        "wave_height_error_le_3pct": bool(out["amp_error_max"] <= 0.03),
        "dispersion_k_error_le_1pct": bool(out["k_error"] <= 0.01),
        "reflection_C_R_le_5pct": bool(c_r <= 0.05),
    }
    save(out, f"{name}.json", artifact_dir)
    return out


def run_reflection_control(**kw) -> dict:
    """Negative control: identical tank with the absorbing beach removed."""
    kw.setdefault("name", "control_no_beach")
    kw["damping"] = False
    out = run_wave_case(**kw)
    out["criteria"] = {"C_R_ge_20pct_required_for_control":
                       bool(out["goda"]["C_R"] >= 0.20)}
    save(out, f"{out['name']}.json", kw.get("artifact_dir"))
    return out


def run_propagation_case(amp: float = 0.02, period: float = 1.0, nx: int = 200,
                         ns: int = 24, length: float = 14.0, depth: float = 1.0,
                         periods: float = 3.0, steps_per_period: int = 100,
                         blend: float = 0.15, name: str = "propagation",
                         artifact_dir=None) -> dict:
    """Pure propagation from an analytic initial field (scheme accuracy)."""
    cfg = TankConfig(length=length, depth=depth, nx=nx, ns=ns,
                     advection_blend=blend, relax_waves=0.0)
    tank = SigmaWaveTank(cfg)
    wave = tank.set_wave(amp, period)
    tank.initialize_from_wave(0.0)
    lam = wave.wavelength
    dt = period / steps_per_period
    x_probe = [(2.0 + 0.25 * i) * lam for i in range(7)]
    series = {x: [] for x in x_probe}
    ts = []
    nsteps = int(round(periods * steps_per_period))
    for _ in range(nsteps):
        t = tank.step(dt)
        ts.append(t)
        for x in x_probe:
            series[x].append(float(tank.eta_at(x)))
    t = np.asarray(ts)
    keep = t >= max(periods - 2.0, 0.0) * period
    amps, phases, resids = {}, {}, {}
    for x in x_probe:
        fit = fit_harmonics(t[keep], np.asarray(series[x])[keep], wave.omega, n_harm=2)
        amps[x], phases[x] = fit["amp"][0], fit["phase"][0]
        resids[x] = fit["residual_rel"]
    a_list = [amps[x] for x in x_probe]
    k_fit, _ = three_probe_wavenumber(x_probe, a_list, [phases[x] for x in x_probe])
    out = {
        "case": name, "name": name,
        "config": {"length": length, "depth": depth, "nx": nx, "ns": ns,
                   "advection_blend": blend, "relax_waves": 0.0},
        "wave": wave.as_dict(), "dt": dt, "steps": nsteps,
        "probes_x_over_lambda": [x / lam for x in x_probe],
        "amp_ratio": [a / amp for a in a_list],
        "amp_error_max": max(abs(a / amp - 1.0) for a in a_list),
        "residual": [resids[x] for x in x_probe],
        "k_three_probe": k_fit, "k_reference": wave.k,
        "k_error": abs(k_fit - wave.k) / wave.k,
        "div_residual_max": tank.div_residual,
        "volume_drift_rel": tank.volume_drift / max(tank.volume0, 1e-30),
    }
    out["criteria"] = {
        "wave_height_error_le_3pct": bool(out["amp_error_max"] <= 0.03),
        "dispersion_k_error_le_1pct": bool(out["k_error"] <= 0.01),
    }
    save(out, f"{name}.json", artifact_dir)
    return out


# --------------------------------------------------------------------- body
def rectangular_section(length: float, draft: float, x0: float = 0.0):
    """Rectangle section, corners counter-clockwise in (x, z), deck at z = 0."""
    a = length / 2.0
    return np.array([[x0 - a, -draft], [x0 + a, -draft], [x0 + a, 0.0], [x0 - a, 0.0]])


def run_rao_case(amp: float = 0.02, period: float = 0.8, nx: int = 200, ns: int = 24,
                 length: float = 12.0, depth: float = 1.0, body_length: float = 0.6,
                 body_draft: float = 0.2, body_x: float = 5.0,
                 added_mass=(300.0, 3.5), periods: float = 12.0,
                 steps_per_period: int = 120, blend: float = 0.15,
                 name: str = "rao", artifact_dir=None) -> dict:
    """Heave/pitch section in regular head waves: RAO + added resistance."""
    cfg = TankConfig(length=length, depth=depth, nx=nx, ns=ns,
                     advection_blend=blend)
    tank = SigmaWaveTank(cfg)
    wave = tank.set_wave(amp, period)
    poly = rectangular_section(body_length, body_draft, x0=body_x)
    tank.set_body(poly, added_mass=added_mass)
    dt = period / steps_per_period
    x_up = body_x - 1.8 * wave.wavelength
    ts, heave, pitch, fx, eta_up = [], [], [], [], []
    i_up = int(np.searchsorted(tank.x, x_up))
    i_dn = int(np.searchsorted(tank.x, body_x + 1.8 * wave.wavelength))
    flux_up, flux_dn = [], []
    t0 = time.time()
    nsteps = int(round(periods * steps_per_period))
    for _ in range(nsteps):
        t = tank.step(dt)
        ts.append(t)
        heave.append(float(tank.body["xi"][0]))
        pitch.append(float(tank.body["xi"][1]))
        fx.append(float(tank.body["force"][0]))
        eta_up.append(float(tank.eta_at(x_up)))
        flux_up.append(tank.momentum_flux(i_up))
        flux_dn.append(tank.momentum_flux(i_dn))
    wall = time.time() - t0
    t = np.asarray(ts)
    keep = t >= max(periods - 4.0, 0.0) * period
    a3, ph3, _, r3 = harmonic_fit(t[keep], np.asarray(heave)[keep], wave.omega)
    a5, ph5, _, r5 = harmonic_fit(t[keep], np.asarray(pitch)[keep], wave.omega)
    a_eta, ph_eta, _, r_eta = harmonic_fit(t[keep], np.asarray(eta_up)[keep], wave.omega)
    r_aw, se = added_resistance_energy(np.asarray(fx)[keep], t[keep], wave.period)
    dflux = np.asarray(flux_up)[keep] - np.asarray(flux_dn)[keep]
    r_aw_cs = float(np.trapezoid(dflux, t[keep]) / (t[keep][-1] - t[keep][0]))
    section_area = body_length * body_draft
    mass = cfg.density * section_area
    out = {
        "case": name, "name": name,
        "config": {k: getattr(cfg, k) for k in
                   ("length", "depth", "nx", "ns", "advection_blend", "relax_waves",
                    "damp_waves", "damp_strength")},
        "wave": wave.as_dict(), "dt": dt, "steps": nsteps,
        "wall_seconds": wall, "ms_per_step": 1000.0 * wall / nsteps,
        "body": {"section": poly.tolist(), "length": body_length,
                 "draft": body_draft, "x": body_x, "area": section_area,
                 "mass_per_length": mass, "added_mass": list(added_mass)},
        "probe_x_up": x_up, "probe_x_dn": body_x + 1.8 * wave.wavelength,
        "incident_amp_at_probe": a_eta,
        "heave_amp": a3, "pitch_amp": a5,
        "rao_heave": a3 / amp, "rao_pitch": a5 / (wave.k * amp),
        "heave_residual": r3, "pitch_residual": r5, "eta_residual": r_eta,
        "added_resistance": r_aw, "added_resistance_se": se,
        "added_resistance_control_surface": r_aw_cs,
        "added_resistance_rel_diff": (abs(r_aw - r_aw_cs) / max(abs(r_aw), 1e-30)
                                      if r_aw else None),
        "volume_drift_rel": tank.volume_drift / max(tank.volume0, 1e-30),
        "div_residual_max": tank.div_residual,
        "series": {"t": ts, "heave": heave, "pitch": pitch, "fx": fx,
                   "eta_up": eta_up},
    }
    save(out, f"{name}.json", artifact_dir)
    return out