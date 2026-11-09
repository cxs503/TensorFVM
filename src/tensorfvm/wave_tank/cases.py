"""Case drivers for the numerical wave tank (NWT) and their evidence JSON.

Each ``run_*`` function is a *complete, reproducible* experiment: it sets up the
tank, runs it, reduces the time series with :mod:`tensorfvm.wave_tank.theory` and
returns a JSON-serialisable dict.  The campaign node stores those dicts under
``outputs/wave-tank/`` so every number in the report is traceable to a run.

Cases
-----
``run_wave_case``       generation -> propagation -> absorption: wave-height
                        error at the probes, three-probe wavenumber vs dispersion,
                        Goda two-point reflection coefficient.
``run_body_case``       heave/pitch body in regular head waves: RAO (first
                        harmonic), added resistance (period-mean horizontal
                        force, i.e. the near-field Gerritsma-Meijers route) and
                        the calm-water control.
``run_geometry_gate``   Cb / L/B / B/T / nabla pre-gate for the Wigley and the
                        simplified KCS hull plus the 2-D strip used by the tank.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .geometry import BargeSection, KCSSimplifiedHull, WigleyHull, geometry_gate
from .solver import RigidBody2D, TankConfig, WaveTank
from .theory import (added_resistance_energy, goda_two_point, harmonic_fit,
                     rao_from_series, three_probe_wavenumber)

__all__ = ["run_geometry_gate", "run_wave_case", "run_body_case", "save"]


def save(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def run_geometry_gate(artifact_dir: str = "outputs/wave-tank") -> dict:
    """Geometry pre-gate: every body must reproduce its own coefficients."""
    wig = geometry_gate(WigleyHull().coefficients())
    kcs = geometry_gate(KCSSimplifiedHull().coefficients())
    strip = BargeSection(length=2.4, draft=0.30).coefficients()
    out = {"wigley": wig, "kcs_simplified": kcs, "strip_2d": strip,
           "passed": bool(wig["passed"] and kcs["passed"])}
    save(out, Path(artifact_dir) / "geometry_gate.json")
    return out


def run_wave_case(amp: float = 0.02, period: float = 1.0, nx: int = 240, ny: int = 72,
                  length: float = 12.0, height: float = 1.4, depth: float = 1.0,
                  damping: bool = True, periods: float = 8.0, steps_per_period: int = 150,
                  name: str = "wave", artifact_dir: str = "outputs/wave-tank") -> dict:
    """Regular-wave generation/absorption test with three probes.

    ``damping=False`` is the **negative control**: with no absorbing beach the
    wave reflects off the downstream wall, so the Goda two-point reflection
    coefficient must jump -- a reflection analysis that cannot show that is not
    measuring anything.
    """
    cfg = TankConfig(length=length, height=height, depth=depth, nx=nx, ny=ny)
    if not damping:
        cfg.damp_strength = 0.0
    tank = WaveTank(cfg)
    w = tank.set_wave(amp, period)
    lam, om, k = w["lambda"], w["omega"], w["k"]
    dt = min(period / steps_per_period, tank.stable_dt())
    x_probe = [tank.x_relax + 0.35 * lam, tank.x_relax + 1.1 * lam, tank.x_relax + 1.85 * lam]
    gauges = [tank.x_relax + 0.55 * lam, tank.x_relax + 0.55 * lam + 0.25 * lam]
    xs = x_probe + gauges
    series = {x: [] for x in xs}
    ts = []
    t0 = time.time()
    n_steps = int(periods * period / dt)
    for _ in range(n_steps):
        tank.step(dt)
        ts.append(tank.t)
        for x in xs:
            series[x].append(tank.probe(x))
    ts = np.asarray(ts)
    skip = 3.0 * period        # discard the start-up transient
    first = {}
    for x in xs:
        a, ph, mean, resid = harmonic_fit(ts[ts >= skip], np.array(series[x])[ts >= skip], om)
        first[x] = {"amp": a, "phase": ph, "mean": mean, "residual": resid}

    probes = [{"x": float(x), "amp": first[x]["amp"], "phase": first[x]["phase"],
               "amp_over_target": first[x]["amp"] / amp,
               "height_error": abs(2.0 * first[x]["amp"] - 2.0 * amp) / (2.0 * amp),
               "residual": first[x]["residual"]} for x in x_probe]
    k_fit, phase0 = three_probe_wavenumber(x_probe, [p["amp"] for p in probes],
                                           [p["phase"] for p in probes])
    h1 = first[gauges[0]]["amp"] * np.exp(1j * first[gauges[0]]["phase"])
    h2 = first[gauges[1]]["amp"] * np.exp(1j * first[gauges[1]]["phase"])
    try:
        aI, aR = goda_two_point(gauges[0], gauges[1], complex(h1), complex(h2), k)
        c_r = abs(aR) / abs(aI) if abs(aI) > 0 else float("nan")
    except ValueError as exc:              # singular gauge spacing
        aI = aR = 0j
        c_r = float("nan")
        first["goda_error"] = str(exc)
    out = {
        "case": name,
        "wave": w,
        "grid": {"nx": nx, "ny": ny, "dx": tank.dx, "dy": tank.dy,
                 "depth": depth, "length": length, "height": height,
                 "points_per_wavelength": float(lam / tank.dx),
                 "points_per_amplitude": float(amp / tank.dy)},
        "damping_zone": {"enabled": bool(damping),
                         "start_x": tank.x_damp_start,
                         "length_over_lambda": (length - tank.x_damp_start) / lam},
        "relaxation_zone_length_over_lambda": tank.x_relax / lam,
        "time": {"dt": float(dt), "steps": n_steps, "periods": periods,
                 "wall_seconds": time.time() - t0},
        "probes": probes,
        "dispersion": {"k_measured": float(abs(k_fit)), "k_theory": float(k),
                       "relative_error": float(abs(abs(k_fit) - k) / k)},
        "reflection": {"goda_C_R": float(c_r), "gauges": gauges,
                       "gauge_spacing_over_lambda": 0.25,
                       "incident_amp": float(abs(aI)), "reflected_amp": float(abs(aR))},
        "volume_drift": float(tank.diag["volume_drift"]),
        "cg_iterations_last": int(tank.diag["cg_iterations"]),
    }
    save(out, Path(artifact_dir) / f"{name}.json")
    return out


def run_body_case(amp: float = 0.02, period: float = 1.1, nx: int = 240, ny: int = 72,
                  length: float = 12.0, height: float = 1.4, depth: float = 1.0,
                  body_length: float = 2.0, body_draft: float = 0.30,
                  periods: float = 10.0, steps_per_period: int = 150,
                  name: str = "body", artifact_dir: str = "outputs/wave-tank") -> dict:
    """Heave/pitch RAO and added resistance of a 2-D barge strip in head waves."""
    cfg = TankConfig(length=length, height=height, depth=depth, nx=nx, ny=ny)
    body = RigidBody2D(length=body_length, draft=body_draft,
                       x_centre=0.5 * length - 1.6)
    tank = WaveTank(cfg, body=body)
    w = tank.set_wave(amp, period)
    om, k = w["omega"], w["k"]
    dt = min(period / steps_per_period, tank.stable_dt())
    ts, heave, pitch, fx, fy = [], [], [], [], []
    t0 = time.time()
    n_steps = int(periods * period / dt)
    for _ in range(n_steps):
        tank.step(dt)
        ts.append(tank.t)
        heave.append(body.heave)
        pitch.append(body.pitch)
        fx.append(body.force_x)
        fy.append(body.force_y)
    ts = np.asarray(ts)
    skip = 4.0 * period
    sel = ts >= skip
    rao3, ph3, mean3, res3 = rao_from_series(ts[sel], np.asarray(heave)[sel], om, amp)
    rao5, ph5, mean5, res5 = rao_from_series(ts[sel], np.asarray(pitch)[sel], om, amp, scale=k)
    r_aw, r_se = added_resistance_energy(np.asarray(fx), ts, period, skip_periods=4.0)
    out = {
        "case": name, "wave": w,
        "body": {"length": body.length, "draft": body.draft, "mass": body.mass,
                 "inertia": body.inertia, "added_mass_heave": body.added_mass_heave,
                 "added_inertia_pitch": body.added_inertia_pitch,
                 "x_centre": body.x_centre},
        "grid": {"nx": nx, "ny": ny, "dx": tank.dx, "dy": tank.dy},
        "time": {"dt": float(dt), "steps": n_steps, "periods": periods,
                 "wall_seconds": time.time() - t0},
        "RAO_heave": {"value": rao3, "phase": ph3, "mean": mean3, "residual": res3},
        "RAO_pitch": {"value": rao5, "phase": ph5, "mean": mean5, "residual": res5},
        "added_resistance": {"R_aw": r_aw, "std_error": r_se,
                             "method": "near-field pressure integral, period mean"},
        "volume_drift": float(tank.diag["volume_drift"]),
    }
    save(out, Path(artifact_dir) / f"{name}.json")
    return out
