"""Time-domain floating-section campaign node: evidence runs.

Each sub-command runs one case on the sigma-grid tank and writes one JSON into
``docs/wave-tank-td-body/artifacts/``.  ``summary`` turns the artifacts into the
acceptance table (every number read from an artifact, none typed).

    python scripts/wave_tank_td/run_body_cases.py gate
    python scripts/wave_tank_td/run_body_cases.py forced  --period 0.8 --nx 200 --ns 24 --spp 80
    python scripts/wave_tank_td/run_body_cases.py decay   --dof heave  --nx 160 --ns 20 --spp 60
    python scripts/wave_tank_td/run_body_cases.py diffraction --period 1.0 --nx 200 --ns 24 --spp 80
    python scripts/wave_tank_td/run_body_cases.py rao     --period 1.2 --nx 200 --ns 24 --spp 80
    python scripts/wave_tank_td/run_body_cases.py summary

Physics: a 2-D vertical-plane tank (surface-fitted sigma grid, Stokes-II
generation by a relaxation zone, quadratic sponge beach) with a rectangular
floating section, heave and pitch free, surge fixed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.wave_tank.theory import GRAVITY, dispersion_k
from tensorfvm.wave_tank_td import seakeeping as sk
from tensorfvm.wave_tank_td.potential2d import section_hydrodynamics
from tensorfvm.wave_tank_td.sigma_tank import SigmaWaveTank, TankConfig

ART = Path("docs/wave-tank-td-body/artifacts")
SECTION = dict(length=1.2, draft=0.2, freeboard=0.05, x0=4.0)
AMP = 0.02            # incident amplitude [m]
DEPTH = 1.0
TANK_LENGTH = 12.0


# --------------------------------------------------------------------- helpers
def section(**kw) -> sk.SectionGeometry:
    p = dict(SECTION)
    p.update(kw)
    return sk.floating_strip(**p)


def _harmonic(t, y, omega, keep_from):
    """First-harmonic complex amplitude and relative residual of ``y``."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = t >= keep_from
    c = np.polyfit(t[keep], y[keep], 1)
    yd = y - np.polyval(c, t)
    a = np.c_[np.cos(omega * t[keep]), np.sin(omega * t[keep])]
    co, *_ = np.linalg.lstsq(a, yd[keep], rcond=None)
    res = float(np.sqrt(np.mean((a @ co - yd[keep]) ** 2)) / max(np.ptp(yd[keep]), 1e-30))
    return co[0] - 1j * co[1], res, float(c[0])


def potential_reference(sec: sk.SectionGeometry, period: float, depth=DEPTH,
                        span: float = 5.0, nx: int = 240, nz: int = 56) -> dict:
    """Independent potential-flow reference: radiation, excitation, RAO.

    ``potential2d`` takes its moments about the origin, so the section is
    translated to the origin for the solve (a pure translation does not change
    ``A``, ``B``, ``F_exc`` about the section's own reference).  The hydrostatic
    restoring comes from the closed form that was checked against the energy
    Hessian (``seakeeping.hydrostatic_stiffness``), and the 2-DOF response is
    solved by ``seakeeping.reference_rao`` so the mass matrix carries the same
    heave/pitch coupling as the tank's equation of motion.
    """
    omega = 2.0 * math.pi / period
    poly0 = sec.polygon - np.array([sec.x0, 0.0])
    hydro = section_hydrodynamics(poly0, omega, depth=depth, wavelength_span=span,
                                  nx=nx, nz=nz, amp=AMP)
    stiff = sk.hydrostatic_stiffness(sec)
    k = dispersion_k(omega, depth)
    a = np.asarray(hydro["A"], dtype=float)
    b = np.asarray(hydro["B"], dtype=float)
    c35 = float(stiff.get("C35", 0.0))
    c = np.array([[float(stiff["C33"]), c35], [c35, float(stiff["C55"])]])
    force = np.array([complex(*hydro["F_exc"][0]), complex(*hydro["F_exc"][1])])
    d = float(sec.ref[1] - sec.cg[1])
    rao = sk.reference_rao(omega, sec.mass, sec.inertia, d, a, b, c, force,
                           wave_amp=AMP, k=k)
    rao_out = {kk: rao[kk] for kk in ("rao_heave", "rao_pitch", "phase_heave",
                                      "phase_pitch", "natural_heave", "natural_pitch")}
    rao_out["mass_matrix"] = rao["mass_matrix"]
    return {"period": period, "omega": omega, "k": k, "A": a.tolist(),
            "B": b.tolist(), "C": c.tolist(),
            "F_exc": [[force[0].real, force[0].imag],
                      [force[1].real, force[1].imag]],
            "rao": rao_out, "wavelength_span": span,
            "hydrostatics": {kk: float(stiff[kk]) for kk in ("C33", "C55", "C35")}}


def _goda_split(h1, h2, d, k):
    """Two-point incident/reflected split (convention ``eta = Re[H e^{i w t}]``).

    ``H(x) = A e^{-ikx} + B e^{ikx}`` at two gauges ``d`` apart; with
    ``kd -> pi/2`` the system is well conditioned.  The amplitudes returned are
    ``a_i = |A|``, ``a_r = |B|``; a pure progressive wave must give ``a_r = 0``.
    """
    a_inc = (h1 * np.exp(1j * k * d) - h2) / (2j * np.sin(k * d))
    return float(abs(a_inc)), float(abs(h1 - a_inc))


def tank(cfg: TankConfig, sec, *, wave_period, motion="free", forced=None,
         sponge_period=None):
    t = SigmaWaveTank(cfg)
    if wave_period is not None:
        t.set_wave(AMP, wave_period)
        t.initialize_from_wave(0.0)
    elif sponge_period is not None:
        t.set_sponge(sponge_period)
    t.set_body(sec.polygon, mass=sec.mass, inertia=sec.inertia, ref=sec.ref,
               motion=motion, forced=forced)
    return t


def _save(name: str, payload: dict) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return out


# ----------------------------------------------------------------------- cases
def case_gate(name="geometry_gate_body"):
    sec = section()
    stiff = sk.hydrostatic_stiffness(sec)
    num = sk.potential_energy_stiffness(sec.polygon, sec.mass, sec.ref)
    t = SigmaWaveTank(TankConfig(length=TANK_LENGTH, nx=240, ns=24))
    t.set_body(sec.polygon, mass=sec.mass, inertia=sec.inertia, ref=sec.ref)
    t._update_mask()
    fh, mh = t.hydrostatic_force()
    out = {
        "case": name,
        "note": "geometry/hydrostatics pre-gate: no wave run may be interpreted "
                "before this is green",
        "section": sec.as_dict(),
        "stiffness": stiff,
        "stiffness_energy_hessian": num,
        "stiffness_route_difference": {
            "C33_rel": abs(num["C33"] - stiff["C33"]) / stiff["C33"],
            "C55_rel": abs(num["C55"] - stiff["C55"]) / stiff["C55"]},
        "tank_hydrostatic_load": {"f": fh.tolist(), "m": mh,
                                  "weight": sec.mass * GRAVITY,
                                  "force_rel_error": abs(fh[1] - sec.mass * GRAVITY)
                                  / (sec.mass * GRAVITY)},
        "discrete_added_mass_still_water": t.discrete_added_mass().tolist(),
        "mask_blocked_fraction_x": float(t.bfrac_x.sum() * t.Ax.mean()),
        "mask_blocked_fraction_s": float(t.bfrac_s.sum() * t.dx),
        "mask_solid_cells": int(t.solid.sum()),
        "displaced_volume": t.displaced_volume(),
    }
    return _save(name, out)


def case_forced(period, nx, ns, spp, name=None, dof=0, amp_forced=0.01, periods=12.0):
    sec = section()
    name = name or f"forced_{'heave' if dof == 0 else 'pitch'}_T{period:.2f}"
    cfg = TankConfig(length=TANK_LENGTH, nx=nx, ns=ns)
    tank_obj = tank(cfg, sec, wave_period=None, motion="forced", sponge_period=period,
                    forced={"dof": dof, "amp": amp_forced, "omega": 2 * math.pi / period})
    tank_obj._update_mask()
    a_disc = tank_obj.discrete_added_mass()
    dt = period / spp
    t0 = time.time()
    ts, fx, fy, m5, traj = [], [], [], [], []
    for _ in range(int(round(periods * spp))):
        ts.append(tank_obj.step(dt))
        fx.append(float(tank_obj.body["force"][0]))
        fy.append(float(tank_obj.body["force"][1]))
        m5.append(float(tank_obj.body["moment"]))
        traj.append(float(tank_obj.body["xi"][dof]))
    keep_from = 0.5 * periods * period
    omega = 2 * math.pi / period
    ts = np.asarray(ts)
    sig = np.asarray(traj)
    win = sig[ts >= keep_from]
    xamp = 0.5 * float(win.max() - win.min())
    F, res, Fmean = _harmonic(ts, fy, omega, keep_from)
    M, res5, Mmean = _harmonic(ts, m5, omega, keep_from)
    stiff = sk.hydrostatic_stiffness(sec)
    pot = potential_reference(sec, period)
    k33, k55 = stiff["C33"], stiff["C55"]
    # F_fluid = -K x - A a - B v  for a prescribed harmonic motion: with
    # a = -omega^2 x the reactive part is  +omega^2 A x, hence
    #   A = Re(F/(omega^2 x)) + K/omega^2 ,   B = -omega Im(F/(omega^2 x))
    out = {
        "case": name, "dof": dof, "period": period, "omega": omega,
        "config": {"nx": nx, "ns": ns, "spp": spp, "periods": periods,
                   "dx": TANK_LENGTH / nx, "points_per_wavelength":
                       float(2 * math.pi / dispersion_k(omega, DEPTH) / (TANK_LENGTH / nx)),
                   "dt": dt, "wall_seconds": time.time() - t0},
        "forced_amplitude": amp_forced, "measured_amplitude": float(xamp),
        "force_harmonic": {"re": F.real, "im": F.imag, "residual": res},
        "moment_harmonic": {"re": M.real, "im": M.imag, "residual": res5},
        "hydrostatic_stiffness": {"C33": k33, "C55": k55},
        "discrete_added_mass": a_disc.tolist(),
        "potential": pot,
        "radiation": {
            "A_direct": float(np.real(F / (omega ** 2 * xamp)) + k33 / omega ** 2),
            "B_direct": float(-omega * np.imag(F / (omega ** 2 * xamp))),
            "A55_direct": float(np.real(M / (omega ** 2 * xamp)) + k55 / omega ** 2),
            "B55_direct": float(-omega * np.imag(M / (omega ** 2 * xamp))),
            "A_potential": pot["A"][dof][dof], "B_potential": pot["B"][dof][dof],
            "A_rel_error": float(abs(np.real(F / (omega ** 2 * xamp)) + k33 / omega ** 2
                                     - pot["A"][dof][dof]) / pot["A"][dof][dof]),
            "B_rel_error": float(abs(-omega * np.imag(F / (omega ** 2 * xamp))
                                     - pot["B"][dof][dof]) / pot["B"][dof][dof]),
        },
        "numerics": {"div_residual_max": float(tank_obj.div_residual),
                     "volume_drift_rel": float(tank_obj.volume_drift / tank_obj.volume0)},
    }
    return _save(name, out)


def case_decay(dof, nx, ns, spp, name=None, periods=6.0, start=0.01):
    sec = section()
    name = name or f"decay_{'heave' if dof == 0 else 'pitch'}_nx{nx}"
    cfg = TankConfig(length=TANK_LENGTH, nx=nx, ns=ns)
    tank_obj = tank(cfg, sec, wave_period=None, motion="free")
    tank_obj.body["xi"][dof] = start
    tank_obj._update_mask()
    a_disc = tank_obj.discrete_added_mass()
    dt = 0.9 / spp
    t0 = time.time()
    for _ in range(int(round(periods * spp))):
        tank_obj.step(dt)
    ts = np.asarray(tank_obj.body["time_series"]["t"])
    xi = np.asarray(tank_obj.body["time_series"]["heave" if dof == 0 else "pitch"])
    other = np.asarray(tank_obj.body["time_series"]["pitch" if dof == 0 else "heave"])
    sign = np.sign(xi - xi.mean())
    zero = [ts[i] for i in range(1, sign.size) if sign[i - 1] * sign[i] < 0]
    period = float(2.0 * np.mean(np.diff(zero))) if len(zero) > 3 else float("nan")
    stiff = sk.hydrostatic_stiffness(sec)
    inertia = sec.mass if dof == 0 else sec.inertia
    k = stiff["C33"] if dof == 0 else stiff["C55"]
    added = k / (2 * math.pi / period) ** 2 - inertia if np.isfinite(period) else float("nan")
    pot = potential_reference(sec, max(period, 0.5) if np.isfinite(period) else 1.0)
    out = {
        "case": name, "dof": dof, "config": {"nx": nx, "ns": ns, "spp": spp,
                                             "dt": dt, "periods": periods,
                                             "wall_seconds": time.time() - t0},
        "decay_period": period, "amplitude_start": start,
        "amplitude_final": float(np.abs(xi[-int(spp * 1.0):]).max()),
        "cross_dof_max": float(np.abs(other).max()),
        "added_mass_from_frequency": float(added),
        "discrete_added_mass": a_disc.tolist(),
        "hydrostatic_stiffness": {"C33": stiff["C33"], "C55": stiff["C55"]},
        "added_mass_potential": pot["A"][dof][dof],
        "added_mass_rel_error": float(abs(added - pot["A"][dof][dof])
                                      / pot["A"][dof][dof]) if np.isfinite(added) else None,
        "numerics": {"div_residual_max": float(tank_obj.div_residual),
                     "volume_drift_rel": float(tank_obj.volume_drift / tank_obj.volume0)},
    }
    return _save(name, out)


def _run_wave(period, nx, ns, spp, periods, motion, name, lock_pitch=False):
    sec = section()
    cfg = TankConfig(length=TANK_LENGTH, nx=nx, ns=ns)
    tank_obj = tank(cfg, sec, wave_period=period, motion=motion)
    if lock_pitch:
        # diagnostic: a heave-only free section (pitch inertia made enormous)
        tank_obj.body["inertia"] = 1e12
    wave = tank_obj.wave
    lam = wave.wavelength
    x_g1 = sec.x0 - 2.0 * lam
    x_g2 = x_g1 + 0.25 * lam
    x_t = sec.x0 + 1.6 * lam
    dt = period / spp
    keep_from = max(periods - 5.0, 1.0) * period
    t0 = time.time()
    ts, heave, pitch, fx, fy, m5 = [], [], [], [], [], []
    e1, e2, et = [], [], []
    for _ in range(int(round(periods * spp))):
        ts.append(tank_obj.step(dt))
        heave.append(float(tank_obj.body["xi"][0]))
        pitch.append(float(tank_obj.body["xi"][1]))
        fx.append(float(tank_obj.body["force"][0]))
        fy.append(float(tank_obj.body["force"][1]))
        m5.append(float(tank_obj.body["moment"]))
        e1.append(float(tank_obj.eta_at(x_g1)))
        e2.append(float(tank_obj.eta_at(x_g2)))
        et.append(float(tank_obj.eta_at(x_t)))
    omega = wave.omega
    h1, r1, _ = _harmonic(ts, e1, omega, keep_from)
    h2, r2, _ = _harmonic(ts, e2, omega, keep_from)
    ht, rt, _ = _harmonic(ts, et, omega, keep_from)
    hh, rh, _ = _harmonic(ts, heave, omega, keep_from)
    hp, rp, _ = _harmonic(ts, pitch, omega, keep_from)
    Fx, rfx, Fx_mean = _harmonic(ts, fx, omega, keep_from)
    Fy, rfy, _ = _harmonic(ts, fy, omega, keep_from)
    M5, rm5, _ = _harmonic(ts, m5, omega, keep_from)
    k = wave.k
    a_i, a_r, a_t = _goda_split(h1, h2, x_g2 - x_g1, k)[0:2] + (float(abs(ht)),)
    drift_far = sk.far_field_drift_2d(a_i, a_r, a_t, k, DEPTH)
    pot = potential_reference(sec, period)
    out = {
        "case": name, "motion": motion, "period": period, "omega": omega,
        "k": k, "wavelength": lam,
        "config": {"nx": nx, "ns": ns, "spp": spp, "periods": periods, "dt": dt,
                   "dx": TANK_LENGTH / nx,
                   "points_per_wavelength": float(lam / (TANK_LENGTH / nx)),
                   "wall_seconds": time.time() - t0},
        "incident": {"amp_nominal": AMP, "a_i_goda": a_i, "a_r_goda": a_r, "a_t": a_t,
                     "C_R": float(a_r / a_i) if a_i > 0 else float("inf"),
                     "residual_probe1": r1, "residual_probe2": r2},
        "motion": {"heave_amp": float(abs(hh)), "pitch_amp": float(abs(hp)),
                   "rao_heave": float(abs(hh) / a_i), "rao_pitch": float(abs(hp) / (k * a_i)),
                   "phase_heave": float(np.angle(hh)), "phase_pitch": float(np.angle(hp)),
                   "residual_heave": rh, "residual_pitch": rp,
                   "rao_heave_vs_nominal": float(abs(hh) / AMP),
                   "rao_pitch_vs_nominal": float(abs(hp) / (k * AMP))},
        "loads": {"fx_harmonic": [Fx.real, Fx.imag], "fx_residual": rfx,
                  "fx_mean": Fx_mean, "fy_amp": float(abs(Fy)),
                  "fy_residual": rfy, "m5_amp": float(abs(M5)), "m5_residual": rm5,
                  "drift_far_field": float(drift_far),
                  "drift_direct_mean": float(Fx_mean),
                  "drift_rel_error": float(abs(Fx_mean - drift_far) / abs(drift_far))
                  if drift_far else None},
        "excitation_tank": {"fy_re": Fy.real, "fy_im": Fy.imag,
                            "m5_re": M5.real, "m5_im": M5.imag},
        "potential": pot,
        "particle_phase_heave_vs_potential": float(np.angle(hh) - pot["rao"]["phase_heave"]),
        "numerics": {"div_residual_max": float(tank_obj.div_residual),
                     "volume_drift_rel": float(tank_obj.volume_drift / tank_obj.volume0),
                     "eta_max": float(np.abs(tank_obj.eta).max())},
        "series": {"t": ts, "heave": heave, "pitch": pitch, "fx": fx, "fy": fy,
                   "m5": m5, "eta_g1": e1, "eta_g2": e2, "eta_t": et},
    }
    return _save(name, out)


def case_diffraction(period, nx, ns, spp, periods=16.0, name=None):
    """Fixed section in waves: the excitation force is measured, nothing moves."""
    name = name or f"diffraction_T{period:.2f}"
    return _run_wave(period, nx, ns, spp, periods, "fixed", name)


def case_rao(period, nx, ns, spp, periods=16.0, name=None, lock_pitch=False):
    """Free heave+pitch section in head waves: RAO and added resistance."""
    name = name or f"rao_T{period:.2f}"
    return _run_wave(period, nx, ns, spp, periods, "free", name, lock_pitch)


def case_summary(max_rao_periods=None):
    """Acceptance table: every number below is read back from an artifact."""
    def load(n):
        p = ART / f"{n}.json"
        return json.loads(p.read_text()) if p.exists() else None

    gate = load("geometry_gate_body")
    forced = sorted([p.stem for p in ART.glob("forced_*") if "T" in p.stem])
    decays = sorted([p.stem for p in ART.glob("decay_*")])
    raos = sorted([p.stem for p in ART.glob("rao_T*")])
    diffs = sorted([p.stem for p in ART.glob("diffraction_T*")])
    rows = []
    rows.append({"criterion": "geometry gate: hydrostatic load vs weight",
                 "measured": gate["tank_hydrostatic_load"]["force_rel_error"],
                 "threshold": 5e-3, "verdict": "PASS" if
                 gate["tank_hydrostatic_load"]["force_rel_error"] <= 5e-3 else "FAIL"})
    rows.append({"criterion": "geometry gate: K55 closed form vs energy Hessian",
                 "measured": gate["stiffness_route_difference"]["C55_rel"],
                 "threshold": 1e-5, "verdict": "PASS" if
                 gate["stiffness_route_difference"]["C55_rel"] <= 1e-5 else "FAIL"})
    rad = []
    for n in forced:
        d = load(n)
        rad.append({"case": n, "dof": d["dof"], "period": d["period"],
                    "A_direct": d["radiation"]["A_direct"],
                    "A_potential": d["radiation"]["A_potential"],
                    "A_rel_error": d["radiation"]["A_rel_error"],
                    "B_direct": d["radiation"]["B_direct"],
                    "B_potential": d["radiation"]["B_potential"],
                    "B_rel_error": d["radiation"]["B_rel_error"],
                    "dt": d["config"]["dt"], "dx": d["config"]["dx"]})
    decay = [load(n) for n in decays]
    rao = []
    for n in raos:
        d = load(n)
        pot = d["potential"]["rao"]
        rao.append({"case": n, "period": d["period"],
                    "rao_heave": d["motion"]["rao_heave"],
                    "rao_heave_potential": pot["rao_heave"],
                    "rao_heave_rel_error": abs(d["motion"]["rao_heave"] - pot["rao_heave"])
                    / pot["rao_heave"],
                    "rao_pitch": d["motion"]["rao_pitch"],
                    "rao_pitch_potential": pot["rao_pitch"],
                    "rao_pitch_rel_error": abs(d["motion"]["rao_pitch"] - pot["rao_pitch"])
                    / pot["rao_pitch"] if pot["rao_pitch"] > 1e-12 else None,
                    "C_R": d["incident"]["C_R"],
                    "drift_direct": d["loads"]["drift_direct_mean"],
                    "drift_far_field": d["loads"]["drift_far_field"],
                    "drift_rel_error": d["loads"]["drift_rel_error"],
                    "residual_heave": d["motion"]["residual_heave"],
                    "residual_pitch": d["motion"]["residual_pitch"]})
    diff = []
    for n in diffs:
        d = load(n)
        fe = np.array([complex(*d["potential"]["F_exc"][0]),
                       complex(*d["potential"]["F_exc"][1])])
        tank_fe = np.array([complex(d["excitation_tank"]["fy_re"], d["excitation_tank"]["fy_im"]),
                            complex(d["excitation_tank"]["m5_re"], d["excitation_tank"]["m5_im"])])
        diff.append({"case": n, "period": d["period"],
                     "fy_tank": abs(tank_fe[0]), "fy_potential": abs(fe[0]),
                     "fy_rel_error": abs(abs(tank_fe[0]) - abs(fe[0])) / abs(fe[0]),
                     "fy_phase_diff": float(np.angle(tank_fe[0]) - np.angle(fe[0])),
                     "m5_tank": abs(tank_fe[1]), "m5_potential": abs(fe[1]),
                     "m5_rel_error": abs(abs(tank_fe[1]) - abs(fe[1])) / abs(fe[1]),
                     "m5_phase_diff": float(np.angle(tank_fe[1]) - np.angle(fe[1])),
                     "drift_direct": d["loads"]["drift_direct_mean"],
                     "drift_far_field": d["loads"]["drift_far_field"],
                     "drift_rel_error": d["loads"]["drift_rel_error"]})
    payload = {"rows": rows, "radiation": rad, "decay": decay, "rao": rao,
               "diffraction": diff,
               "artifacts": {"gate": "docs/wave-tank-td-body/artifacts/geometry_gate_body.json",
                             "forced": [f"docs/wave-tank-td-body/artifacts/{n}.json" for n in forced],
                             "decay": [f"docs/wave-tank-td-body/artifacts/{n}.json" for n in decays],
                             "rao": [f"docs/wave-tank-td-body/artifacts/{n}.json" for n in raos],
                             "diffraction": [f"docs/wave-tank-td-body/artifacts/{n}.json"
                                             for n in diffs]}}
    return _save("summary", payload)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action")
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ns", type=int, default=24)
    ap.add_argument("--spp", type=int, default=80)
    ap.add_argument("--periods", type=float, default=16.0)
    ap.add_argument("--dof", default="heave", choices=["heave", "pitch"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--amp", type=float, default=0.01)
    ap.add_argument("--lock-pitch", action="store_true")
    args = ap.parse_args(argv)
    if args.action == "gate":
        p = case_gate()
    elif args.action == "forced":
        p = case_forced(args.period, args.nx, args.ns, args.spp, args.name,
                        0 if args.dof == "heave" else 1, args.amp)
    elif args.action == "decay":
        p = case_decay(0 if args.dof == "heave" else 1, args.nx, args.ns, args.spp,
                       args.name)
    elif args.action == "diffraction":
        p = case_diffraction(args.period, args.nx, args.ns, args.spp, args.periods,
                             args.name)
    elif args.action == "rao":
        p = case_rao(args.period, args.nx, args.ns, args.spp, args.periods, args.name,
                     lock_pitch=args.lock_pitch)
    elif args.action == "summary":
        p = case_summary()
    else:
        raise SystemExit(f"unknown action {args.action}")
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())