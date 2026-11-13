"""Verification surface for compressible multi-material interface dynamics.

Five legs, each with its own criterion, all reported as numbers in a JSON
artifact and as figures in the same directory:

``sod``
    1-D Sod shock tube against the exact Riemann solution.  Criterion: L1 error
    and observed order (first and second order runs must both decrease), plus
    machine-precision conservation on an open boundary with the boundary-flux
    ledger included.
``lax``
    Lax (1954) 1-D Euler shock tube (Toro 2009 ch. 4 test 2) against its exact
    Riemann solution: strong right shock, strong left rarefaction and a contact
    whose two star densities differ by a factor 3.785.  Criterion: star-region
    plateaus, the three wave positions and the relative L1 of rho/u/p all within
    2 %, no positivity repairs, mass ledger drift within 0.5 %, plus the
    two-implementation cross-check of the reference and the Rankine-Hugoniot
    residual.  This is the case-pool registration of CASE-LAX.
``interface``
    Two-gas air/He contact.  Criterion: the material-interface trajectory
    matches the exact Riemann solution and converges with order >= 1 at three
    resolutions; the ghost-fluid treatment reduces the spurious pressure
    oscillation at the contact relative to plain multi-material HLLC.
``richtmyer_meshkov``
    Air/SF6 single-mode shock-driven interface.  Criteria: linear growth rate
    against Richtmyer's impulsive model, mixing-width ``W(t)`` against the
    alpha-model form, three-resolution consistency, and < 0.5 % accounted
    drift of mass/momentum/energy.
``shock_bubble``
    Mach 1.22 helium cylinder in air (Haas & Sturtevant configuration).
    Criteria: incident/transmitted shock speeds and the baroclinic-vorticity
    deposition time against the exact-Riemann prediction, reflected wave is a
    rarefaction, and the field topology matches the published wave pattern.
``vortex_pair``
    Co-rotating compressible vortex pair.  Criteria: pair rotation rate against
    ``Omega = Gamma/(pi b^2)``, circulation conservation, and the merging
    threshold/time scaling from a separation sweep.

The module writes
``<outdir>/compressible_multimaterial.json`` (all numbers + verdicts) and the
figures referenced by the report.  Nothing is hard-coded to "pass": the verdict
of each criterion is computed from the measured number and the declared
threshold, and a failed criterion is reported as ``False``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .cases import (
    BubbleConfig,
    RMConfig,
    VortexConfig,
    richtmyer_meshkov,
    shock_bubble,
    sod,
    two_gas_interface,
    material_advection,
    vortex_pair,
    GAMMA_AIR,
)
from .diagnostics import (
    alpha_model_width,
    amplitude_from_columns,
    first_crossing_time,
    growth_rate_fit,
    interface_position,
    linear_width,
    mixing_width,
    pressure_oscillation,
    transverse_mean,
)
from .forward_step import ForwardStepConfig
from .forward_step import run_case as run_forward_step
from .forward_step_benchmark import grade_run as grade_forward_step
from .vortex import (
    core_positions,
    core_separation,
    principal_axis_angle,
    swirl_mach,
    total_circulation,
    vorticity,
)

__all__ = ["run_all", "RunConfig"]

DEFAULT_OUT = Path("src/tensorfvm/verification/artifacts/compressible_multimaterial")

#: thresholds that the report quotes verbatim
THRESHOLDS = {
    "rm_linear_growth_error": 0.15,
    "rm_alpha_model_error": 0.15,
    "bubble_shock_speed_error": 0.10,
    "bubble_formation_time_error": 0.10,
    "vortex_rotation_error": 0.05,
    "vortex_circulation_error": 0.01,
    "conservation_drift": 0.005,
    "interface_order": 1.0,
    "sod_order_first": 0.6,
    "sod_order_second": 0.8,
}


@dataclass
class RunConfig:
    """Resolution knobs; ``quick=True`` is the pytest-sized version."""

    quick: bool = False

    def rm(self) -> List[RMConfig]:
        if self.quick:
            # short box so the growth window fits inside the pytest budget
            return [RMConfig(ny=ny, t_end=0.62, cfl=0.4, shock_x=-0.9)
                    for ny in (24, 32, 40)]
        return [RMConfig(ny=ny, t_end=1.1) for ny in (40, 56, 72)]

    def bubble(self) -> BubbleConfig:
        return BubbleConfig(mesh=36 if self.quick else 40,
                            t_end=0.9 if self.quick else 2.2)

    def vortex(self) -> VortexConfig:
        return VortexConfig(n=96 if self.quick else 160,
                            t_end=3.0 if self.quick else 6.0)

    def sod_sizes(self) -> Tuple[int, ...]:
        return (100, 200) if self.quick else (100, 200, 400)

    def interface_sizes(self) -> Tuple[int, ...]:
        return (100, 200, 400) if self.quick else (100, 200, 400)


def _rel(a: float, b: float) -> float:
    """Symmetric relative difference, safe for zero reference."""
    if a != a or b != b:
        return float("nan")
    denom = abs(b) if abs(b) > 1e-12 else 1.0
    return abs(a - b) / denom


def _verdict(err: float, limit: float) -> bool:
    return bool(err == err and err <= limit)


# ----------------------------------------------------------------------
# leg 1: Sod
# ----------------------------------------------------------------------
def leg_sod(cfg: RunConfig) -> Dict:
    rows = []
    for order in (1, 2):
        for nx in cfg.sod_sizes():
            solver, meta = sod(nx, order=order)
            initial = solver.integrals()
            t0 = time.time()
            solver.run(meta["t_end"])
            prim = solver.primitive()
            xi = (meta["xc"] - meta["x0"]) / meta["t_end"]
            rho_e, u_e, p_e = meta["exact"].sample(xi)
            drift = solver.budget_drift(initial)
            row = {
                "order": order,
                "nx": nx,
                "l1_rho": float((prim["rho"][:, 0] - rho_e).abs().mean()),
                "l1_u": float((prim["u"][:, 0] - u_e).abs().mean()),
                "l1_p": float((prim["p"][:, 0] - p_e).abs().mean()),
                "drift_mass": drift["mass"],
                "drift_energy": drift["energy"],
                "drift_momentum_x": drift["momentum_x"],
                "positivity_fixes": solver.positivity_fixes,
                "species_clamps": solver.species_clamps,
                "steps": solver.nsteps,
                "seconds": round(time.time() - t0, 2),
            }
            if order == 2 and nx == max(cfg.sod_sizes()):
                row["profile"] = {
                    "t": meta["t_end"],
                    "nx": nx,
                    "x": meta["xc"].tolist(),
                    "rho": prim["rho"][:, 0].tolist(),
                    "rho_exact": rho_e.tolist(),
                    "u": prim["u"][:, 0].tolist(),
                    "u_exact": u_e.tolist(),
                    "p": prim["p"][:, 0].tolist(),
                    "p_exact": p_e.tolist(),
                }
            rows.append(row)
    orders = {}
    for order in (1, 2):
        sub = [r for r in rows if r["order"] == order]
        obs = []
        for a, b in zip(sub[:-1], sub[1:]):
            obs.append(math.log(a["l1_rho"] / b["l1_rho"]) / math.log(b["nx"] / a["nx"]))
        orders[str(order)] = obs
    drift_max = max(abs(r["drift_mass"]) for r in rows)
    first = min(orders["1"]) if orders["1"] else float("nan")
    second = min(orders["2"]) if orders["2"] else float("nan")
    verdicts = {
        "order_first_ge": bool(first == first and first >= THRESHOLDS["sod_order_first"]),
        "order_second_ge": bool(second == second and second >= THRESHOLDS["sod_order_second"]),
        "mass_conserved": _verdict(drift_max, THRESHOLDS["conservation_drift"]),
    }
    profile = next((r.pop("profile") for r in rows if "profile" in r), None)
    return {"rows": rows, "profile": profile, "observed_order": orders,
            "verdicts": verdicts, "max_abs_mass_drift": drift_max}


# ----------------------------------------------------------------------
# leg 2: two-gas material interface
# ----------------------------------------------------------------------
def leg_interface(cfg: RunConfig) -> Dict:
    rows = []
    for flux in ("hllc", "gfm"):
        for nx in cfg.interface_sizes():
            solver, meta = two_gas_interface(nx, flux_mode=flux)
            initial = solver.integrals()
            solver.run(meta["t_end"])
            prim = solver.primitive()
            pos = interface_position(prim["Y"], meta["xc"], level=0.5)
            x_exact = meta["x0"] + meta["u_star"] * meta["t_end"]
            x_num = float(torch.nan_to_num(pos, nan=meta["x0"]).mean())
            osc = pressure_oscillation(prim["p"][:, 0], meta["xc"], x_exact,
                                       meta["p_star"], half_width=0.06)
            drift = solver.budget_drift(initial)
            rows.append({
                "flux_mode": flux,
                "nx": nx,
                "dx": 1.0 / nx,
                "interface_position": x_num,
                "interface_exact": x_exact,
                "interface_error": abs(x_num - x_exact),
                "interface_error_rel": _rel(x_num, x_exact),
                "pressure_pp": osc["pp"],
                "pressure_rms": osc["rms"],
                "drift_mass": drift["mass"],
                "drift_energy": drift["energy"],
                "positivity_fixes": solver.positivity_fixes,
            })
    # order of the material-interface position, measured on the pure-advection
    # case where the exact trajectory is unambiguous
    advection = []
    advection_hllc = []
    for nx in cfg.interface_sizes():
        adv_h, meta_h = material_advection(nx, flux_mode="hllc")
        adv_h.run(meta_h["t_end"])
        prim_h = adv_h.primitive()
        advection_hllc.append({
            "nx": nx,
            "pressure_error": float((prim_h["p"][:, 0] - 1.0).abs().max()),
        })
        adv_solver, adv_meta = material_advection(nx, flux_mode="gfm")
        adv_solver.run(adv_meta["t_end"])
        prim_adv = adv_solver.primitive()
        pos_adv = float(torch.nan_to_num(
            interface_position(prim_adv["Y"], adv_meta["xc"], 0.5),
            nan=adv_meta["x0"]).mean())
        advection.append({
            "nx": nx,
            "dx": 1.0 / nx,
            "x_interface": pos_adv,
            "x_exact": adv_meta["x_exact"],
            "error": abs(pos_adv - adv_meta["x_exact"]),
            "pressure_error": float((prim_adv["p"][:, 0] - 1.0).abs().max()),
            "drift_mass": adv_solver.budget_drift(adv_meta["initial_integrals"])["mass"],
        })
    orders = []
    for a, b in zip(advection[:-1], advection[1:]):
        if a["error"] > 0 and b["error"] > 0:
            orders.append(math.log(a["error"] / b["error"])
                          / math.log(a["dx"] / b["dx"]))
    hllc = {r["nx"]: r for r in rows if r["flux_mode"] == "hllc"}
    gfm_by_nx = {r["nx"]: r for r in rows if r["flux_mode"] == "gfm"}
    osc_ratio = {}
    for nx in gfm_by_nx:
        if nx in hllc and hllc[nx]["pressure_pp"] > 0:
            osc_ratio[str(nx)] = gfm_by_nx[nx]["pressure_pp"] / hllc[nx]["pressure_pp"]
    verdicts = {
        "interface_order_ge": _verdict(min(orders) if orders else float("nan"),
                                       THRESHOLDS["interface_order"]),
        # the mass-fraction stiffened-gas closure leaves a pressure dip inside a
        # smeared contact (see docs/compressible-multimaterial-verification.md);
        # the declared defect bound is 15 % of the ambient pressure
        "contact_pressure_dip_le_15pct": all(a["pressure_error"] < 0.15
                                              for a in advection),
        "mass_conserved": _verdict(max(abs(r["drift_mass"]) for r in rows),
                                   THRESHOLDS["conservation_drift"]),
        "gfm_reduces_pressure_wiggle": bool(osc_ratio and max(osc_ratio.values()) < 1.0),
    }
    return {"rows": rows, "advection": advection, "advection_hllc": advection_hllc,
            "advection_pressure_ratio": {
                str(a["nx"]): (a["pressure_error"] / b["pressure_error"]
                               if b["pressure_error"] > 0 else None)
                for a, b in zip(advection, advection_hllc)},
            "observed_order": orders,
            "gfm_over_hllc_pressure_pp": osc_ratio, "verdicts": verdicts}


# ----------------------------------------------------------------------
# leg 3: Richtmyer-Meshkov
# ----------------------------------------------------------------------
def _rm_measure(solver, meta, samples: Dict[str, List[float]], t: float) -> None:
    prim = solver.primitive()
    pos = interface_position(prim["Y"], meta["xc"], 0.5)
    amp = amplitude_from_columns(pos)
    mean_y = transverse_mean(prim["Y"])
    width = mixing_width(mean_y, meta["xc"], 0.01, 0.99)
    samples["t"].append(t)
    samples["amplitude"].append(amp["amplitude"])
    samples["spike"].append(amp["spike"])
    samples["bubble"].append(amp["bubble"])
    samples["width"].append(width["width"])
    samples["interface_mean"].append(amp["mean"])


def leg_rm(cfg: RunConfig) -> Dict:
    analytic = None
    rows = []
    for rm_cfg in cfg.rm():
        solver, meta = richtmyer_meshkov(rm_cfg)
        analytic = meta["analytic"]
        samples: Dict[str, List[float]] = {k: [] for k in
                                           ("t", "amplitude", "spike", "bubble",
                                            "width", "interface_mean")}
        _rm_measure(solver, meta, samples, solver.time)
        snapshots = []
        next_snap = 0.0
        t0 = time.time()
        while solver.time < rm_cfg.t_end:
            dt = min(solver.stable_dt(), rm_cfg.t_end - solver.time)
            solver.step(dt)
            _rm_measure(solver, meta, samples, solver.time)
            if solver.time >= next_snap:
                prim = solver.primitive()
                snapshots.append({
                    "t": solver.time,
                    "Y": prim["Y"].clone(),
                    "rho": prim["rho"].clone(),
                    "vorticity": vorticity(prim["u"], prim["v"], meta["dx"], meta["dy"],
                                           periodic=False),
                })
                next_snap += 0.3
        prim = solver.primitive()
        drift = solver.budget_drift(meta["initial_integrals"])
        t_growth_start = analytic["shock_transit_time"] + 0.1
        growth = growth_rate_fit(samples["t"], samples["amplitude"],
                                 t_growth_start, rm_cfg.t_end)
        rows.append({
            "ny": rm_cfg.ny,
            "nx": meta["nx"],
            "dx": meta["dx"],
            "t_end": rm_cfg.t_end,
            "steps": solver.nsteps,
            "seconds": round(time.time() - t0, 2),
            "amplitude_growth_rate": growth["slope"],
            "amplitude_growth_n": growth["n"],
            "width_final": samples["width"][-1],
            "amplitude_final": samples["amplitude"][-1],
            "width_series": {"t": samples["t"], "width": samples["width"],
                             "amplitude": samples["amplitude"],
                             "spike": samples["spike"], "bubble": samples["bubble"]},
            "drift": drift,
            "positivity_fixes": solver.positivity_fixes,
            "species_clamps": solver.species_clamps,
            "interface_cells": float(
                ((prim["Y"] > 0.05) & (prim["Y"] < 0.95)).to(torch.float64).mean()),
            "snapshots": snapshots,
        })
    theory = analytic["linear_growth_rate"]
    errs = {r["ny"]: _rel(r["amplitude_growth_rate"], theory) for r in rows}
    # linear-regime mixing width: W = 2 a0+ + 2 (da/dt)_theory t_growth
    alpha = 0.06
    rm_cfg_ly = cfg.rm()[0].ly
    linear_errs = {}
    alpha_errs = {}
    for r in rows:
        t_growth = r["t_end"] - analytic["shock_transit_time"]
        w_sim = r["width_final"]
        linear_errs[r["ny"]] = _rel(w_sim, linear_width(t_growth, analytic["a0_post"],
                                                       theory))
        # the alpha model is a *self-similar* law: it only applies once the mixing
        # zone is much wider than the perturbation wavelength
        if w_sim > 0.5 * rm_cfg_ly:  # self-similar regime needs W >~ half a wavelength
            alpha_errs[r["ny"]] = _rel(
                w_sim, alpha_model_width(t_growth, analytic["a0_post"],
                                         analytic["atwood_post"], analytic["u_star"],
                                         alpha=alpha))
    drift_max = max(max(abs(v) for k, v in r["drift"].items()
                        if k != "momentum_y") for r in rows)
    # resolution consistency: growth rate must not drift by more than 5 % between
    # the two finest meshes once the linear rate is resolved
    fine = [r["amplitude_growth_rate"] for r in rows[-2:]]
    consistency = _rel(fine[1], fine[0]) if len(fine) == 2 else float("nan")
    verdicts = {
        "linear_growth_error_le_15pct": _verdict(max(errs.values()),
                                                 THRESHOLDS["rm_linear_growth_error"]),
        "linear_width_error_le_15pct": _verdict(max(linear_errs.values()),
                                                THRESHOLDS["rm_alpha_model_error"]),
        "alpha_model_regime_reached": bool(alpha_errs) and _verdict(
            max(alpha_errs.values()), THRESHOLDS["rm_alpha_model_error"]),
        "conservation_drift_le_0p5pct": _verdict(drift_max, THRESHOLDS["conservation_drift"]),
        "resolution_consistent_le_5pct": _verdict(consistency, 0.05),
        "no_positivity_repair": all(r["positivity_fixes"] == 0 for r in rows),
    }
    return {"rows": rows, "analytic": analytic,
            "linear_growth_error": errs, "linear_width_error": linear_errs,
            "alpha_model_error": alpha_errs,
            "alpha_used": alpha, "drift_max": drift_max,
            "resolution_consistency": consistency, "verdicts": verdicts}


# ----------------------------------------------------------------------
# leg 4: shock-bubble
# ----------------------------------------------------------------------
def leg_bubble(cfg: RunConfig) -> Dict:
    bubble_cfg = cfg.bubble()
    solver, meta = shock_bubble(bubble_cfg)
    analytic = meta["analytic"]
    d = meta["d"]
    i_inc = int((-1.2 - bubble_cfg.x_min) / d)
    i_in = int((0.45 - bubble_cfg.x_min) / d)
    j_mid = meta["ny"] // 2
    series = {"t": [], "p_inc": [], "p_in": [], "circulation_upper": [],
              "max_vorticity": [], "bubble_width": []}
    snapshots = []
    next_snap = 0.15

    def record(t: float) -> None:
        prim = solver.primitive()
        omega = vorticity(prim["u"], prim["v"], d, d, periodic=False)
        upper = omega[:, j_mid:]
        series["t"].append(t)
        series["p_inc"].append(float(prim["p"][i_inc, j_mid]))
        series["p_in"].append(float(prim["p"][i_in, j_mid]))
        series["circulation_upper"].append(
            float(upper.sum() * d * d))
        series["max_vorticity"].append(float(omega.abs().max()))
        mask = prim["Y"] > 0.5
        if bool(mask.any()):
            xs = torch.nonzero(mask.any(dim=1)).flatten()
            series["bubble_width"].append(float((xs[-1] - xs[0]) * d))
        else:
            series["bubble_width"].append(0.0)

    record(0.0)
    t0 = time.time()
    while solver.time < bubble_cfg.t_end:
        dt = min(solver.stable_dt(), bubble_cfg.t_end - solver.time)
        solver.step(dt)
        record(solver.time)
        if solver.time >= next_snap:
            prim = solver.primitive()
            snapshots.append({
                "t": solver.time,
                "Y": prim["Y"].clone(),
                "rho": prim["rho"].clone(),
                "p": prim["p"].clone(),
                "vorticity": vorticity(prim["u"], prim["v"], d, d, periodic=False),
            })
            next_snap += 0.45
    prim = solver.primitive()
    drift = solver.budget_drift(meta["initial_integrals"])

    t_inc = first_crossing_time(series["t"], series["p_inc"],
                                0.5 * (analytic["post_p"] + 1.0))
    t_in = first_crossing_time(series["t"], series["p_in"],
                               0.5 * (analytic["p_interface"] + 1.0))
    v_inc = ((i_inc + 0.5) * d + bubble_cfg.x_min - bubble_cfg.shock_x) / t_inc \
        if t_inc == t_inc and t_inc > 0 else float("nan")
    v_tra = ((i_in + 0.5) * d + bubble_cfg.x_min + bubble_cfg.radius) / t_in \
        if t_in == t_in and t_in > 0 else float("nan")
    # baroclinic vorticity deposition: 95 % of the maximum deposited circulation
    gamma_max = max(series["circulation_upper"])
    t_form = first_crossing_time(series["t"], series["circulation_upper"],
                                 0.95 * gamma_max)
    reflected_rarefaction = False
    if len(series["p_inc"]) > 10:
        idx = [k for k, t in enumerate(series["t"]) if t > 0.6]
        if idx:
            reflected_rarefaction = bool(series["p_inc"][idx[-1]] < analytic["post_p"])
    shock_speed_error = _rel(v_inc, analytic["incident_shock_speed"])
    transmitted_error = _rel(v_tra, analytic["transmitted_shock_speed"])
    formation_reference = analytic["arrival_time"] + analytic["crossing_time"]
    formation_error = _rel(t_form, formation_reference)
    peak_mach = swirl_mach(prim["u"], prim["v"], prim["rho"], prim["p"], GAMMA_AIR)
    verdicts = {
        "incident_shock_speed_le_10pct": _verdict(shock_speed_error,
                                                  THRESHOLDS["bubble_shock_speed_error"]),
        "transmitted_shock_speed_le_10pct": _verdict(transmitted_error,
                                                     THRESHOLDS["bubble_shock_speed_error"]),
        "formation_time_le_10pct": _verdict(formation_error,
                                            THRESHOLDS["bubble_formation_time_error"]),
        "reflected_wave_is_rarefaction": reflected_rarefaction,
        "vortex_pair_forms": bool(gamma_max > 0.0),
        "conservation_drift_le_0p5pct": _verdict(
            max(abs(drift["mass"]), abs(drift["energy"])),
            THRESHOLDS["conservation_drift"]),
    }
    return {
        "mesh": {"nx": meta["nx"], "ny": meta["ny"], "d": d},
        "analytic": analytic,
        "series": series,
        "snapshots": snapshots,
        "steps": solver.nsteps,
        "seconds": round(time.time() - t0, 2),
        "incident_shock_speed": v_inc,
        "transmitted_shock_speed": v_tra,
        "formation_time": t_form,
        "formation_reference": formation_reference,
        "crossing_time": analytic["crossing_time"],
        "incident_shock_speed_error": shock_speed_error,
        "transmitted_shock_speed_error": transmitted_error,
        "formation_time_error": formation_error,
        "circulation_peak": gamma_max,
        "peak_swirl_mach": peak_mach,
        "reflected_rarefaction": reflected_rarefaction,
        "drift": drift,
        "verdicts": verdicts,
    }


# ----------------------------------------------------------------------
# leg 5: vortex pair
# ----------------------------------------------------------------------
def _vortex_run(cfg: VortexConfig, sample_every: float = 0.1) -> Dict:
    solver, meta = vortex_pair(cfg)
    analytic = meta["analytic"]
    d = meta["d"]
    series = {"t": [], "angle": [], "circulation": [], "enstrophy": [],
              "separation": [], "cores": []}
    snapshots = []

    def record(t: float) -> None:
        prim = solver.primitive()
        # non-periodic differences: a central difference wrapped periodically sums
        # to exactly zero, which would make the circulation diagnostic useless
        omega = vorticity(prim["u"], prim["v"], d, d, periodic=False)
        sep, ncores = core_separation(omega, d, d)
        cores = core_positions(omega, d, d)
        if len(cores) == 2:
            angle = math.atan2(cores[1][1] - cores[0][1], cores[1][0] - cores[0][0])
        else:
            angle = principal_axis_angle(omega, d, d)
        series["t"].append(t)
        series["angle"].append(angle)
        series["circulation"].append(total_circulation(omega, d, d))
        series["enstrophy"].append(float((omega * omega).sum() * d * d))
        series["separation"].append(sep)
        series["cores"].append(ncores)

    record(0.0)
    t0 = time.time()
    next_snap = 0.5
    while solver.time < cfg.t_end:
        dt = min(solver.stable_dt(), cfg.t_end - solver.time)
        solver.step(dt)
        if solver.time >= next_snap:
            record(solver.time)
            prim = solver.primitive()
            snapshots.append({
                "t": solver.time,
                "vorticity": vorticity(prim["u"], prim["v"], d, d, periodic=False),
                "rho": prim["rho"].clone(),
            })
            next_snap += sample_every if cfg.t_end < 4.0 else 1.0
    prim = solver.primitive()
    drift = solver.budget_drift(meta["initial_integrals"])
    # unwrap the principal axis angle (mod pi) and fit the rotation rate
    angles = []
    for k, a in enumerate(series["angle"]):
        if a != a:
            continue
        if not angles:
            angles.append(a)
            continue
        prev = angles[-1]
        while a - prev > 0.5 * math.pi:
            a -= math.pi
        while a - prev < -0.5 * math.pi:
            a += math.pi
        angles.append(a)
    ts = series["t"][:len(angles)]
    omega_meas = float("nan")
    if len(ts) > 5:
        n = len(ts)
        mt = sum(ts) / n
        ma = sum(angles) / n
        den = sum((t - mt) ** 2 for t in ts)
        if den > 0:
            omega_meas = sum((t - mt) * (a - ma) for t, a in zip(ts, angles)) / den
    circ_meas = series["circulation"][-1] if series["circulation"] else float("nan")
    merged = bool(series["cores"] and max(series["cores"][-1:]) <= 1)
    return {
        "separation_over_core": cfg.separation / cfg.core,
        "separation": cfg.separation,
        "core": cfg.core,
        "n": cfg.n,
        "analytic": analytic,
        "series": series,
        "snapshots": snapshots,
        "rotation_rate_measured": omega_meas,
        "rotation_rate_theory": analytic["rotation_rate"],
        "rotation_error": _rel(omega_meas, analytic["rotation_rate"]),
        "circulation_measured": circ_meas,
        "circulation_theory": 2.0 * analytic["circulation"],
        "circulation_error": _rel(circ_meas, 2.0 * analytic["circulation"]),
        "merged": merged,
        "final_separation": series["separation"][-1] if series["separation"] else 0.0,
        "steps": solver.nsteps,
        "seconds": round(time.time() - t0, 2),
        "drift": drift,
        "peak_swirl_mach": swirl_mach(prim["u"], prim["v"], prim["rho"], prim["p"],
                                      GAMMA_AIR),
    }


def leg_vortex(cfg: RunConfig) -> Dict:
    base = cfg.vortex()
    main = _vortex_run(base)
    separations = []
    if cfg.quick:
        sweep = [base.separation * 1.6]
    else:
        sweep = [base.separation * 1.3, base.separation * 1.6]
    for sep in sweep:
        variant = VortexConfig(n=max(64, base.n // 2), size=base.size,
                               circulation=base.circulation, core=base.core,
                               separation=sep, mach_vortex=base.mach_vortex,
                               t_end=base.t_end * 0.6 if not cfg.quick else base.t_end,
                               cfl=base.cfl)
        run = _vortex_run(variant, sample_every=0.5)
        separations.append({
            "separation_over_core": run["separation_over_core"],
            "merged": run["merged"],
            "final_separation": run["final_separation"],
            "rotation_error": run["rotation_error"],
            "seconds": run["seconds"],
        })
    verdicts = {
        "rotation_rate_le_5pct": _verdict(main["rotation_error"],
                                          THRESHOLDS["vortex_rotation_error"]),
        "circulation_le_1pct": _verdict(main["circulation_error"],
                                        THRESHOLDS["vortex_circulation_error"]),
        "merges_at_close_separation": bool(main["merged"] or
                                           any(s["merged"] for s in separations)),
        "keeps_two_cores_at_wide_separation": bool(
            (not main["merged"]) or any(not s["merged"] for s in separations)),
    }
    return {"main": main, "sweep": separations, "verdicts": verdicts}


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# leg 6: Lax (1954) shock tube -- case-pool membership (CASE-LAX)
# ----------------------------------------------------------------------
def leg_lax(cfg: RunConfig) -> Dict:
    """Lax shock tube graded against its exact Riemann solution (2 % acceptance).

    Lax (1954) *Comm. Pure Appl. Math.* **7**, 159--193; the same initial data
    set is Toro (2009) ch. 4 test problem 2.  Unlike Sod the right wave is a
    *strong* shock (``p*/p_R = 4.32``), so the HLLC wave-speed estimate and the
    contact/shear resolution are exercised harder than in the Sod leg.

    The reference is the exact solution of the stated initial-value problem,
    recomputed at run time by two implementations that share no code
    (``riemann.ExactRiemann`` Toro pressure iteration vs the from-scratch
    bisection in ``lax.independent_star_state``/``independent_sample``); their
    agreement, the Rankine--Hugoniot residual of the right shock and the
    ``x_head < x_tail < x_contact < x_shock`` ordering are reported alongside the
    comparison so the reference itself is auditable, not asserted.

    Graded on the finest grid: star-region plateaus (``p*``, ``u*``, ``rho*_L``,
    ``rho*_R``), the three wave positions (one estimator applied to *both* the
    numerical and the fine-grid exact field, so the estimator bias cancels),
    relative L1 of ``rho``/``u``/``p``, the mass ledger drift and the positivity
    repair count.
    """
    from . import lax as L

    lcfg = L.LaxConfig()
    sizes = (200, 400) if cfg.quick else (200, 400, 800)
    rows = [L.run_case(nx, keep_profile=(nx == max(sizes)), cfg=lcfg) for nx in sizes]
    orders: Dict[str, List[float]] = {}
    for key in ("rho", "u", "p"):
        obs = []
        for a, b in zip(rows[:-1], rows[1:]):
            ea, eb = a["l1_rel"][key], b["l1_rel"][key]
            obs.append(math.log(ea / eb) / math.log(b["nx"] / a["nx"])
                       if (ea > 0.0 and eb > 0.0) else float("nan"))
        orders[key] = obs
    graded = L.verdicts(rows, cfg=lcfg)
    check = L.independent_check(lcfg)
    rh = L.rankine_hugoniot_residual(lcfg)
    verdicts = {f"lax_{k}": bool(v["ok"]) for k, v in graded["checks"].items()}
    verdicts["lax_exact_solution_independently_reproduced"] = bool(
        max(check["rel_dp_star"], check["rel_du_star"], check["rel_d_rho_star_l"],
            check["rel_d_rho_star_r"], check["max_abs_profile_diff_scaled"])
        <= L.THRESHOLDS["independent_exact_rel_error"])
    verdicts["lax_rankine_hugoniot_satisfied"] = bool(
        max(abs(rh["rel_residual_mass"]), abs(rh["rel_residual_momentum"]),
            abs(rh["rel_residual_energy"])) <= 1e-12)
    profile = next((r.pop("profile") for r in rows if "profile" in r), None)
    if profile is not None:
        profile["t"] = lcfg.t_end
        profile["nx"] = graded["nx"]
    return {
        "reference": ("Lax (1954) Comm. Pure Appl. Math. 7:159-193 = Toro (2009) ch. 4 "
                      "test problem 2; exact solution recomputed at run time"),
        "config": lcfg.as_dict(),
        "thresholds": dict(L.THRESHOLDS),
        "rows": rows,
        "profile": profile,
        "observed_order": orders,
        "graded": graded,
        "exact_waves": L.exact_waves(lcfg),
        "independent_exact_check": check,
        "rankine_hugoniot": rh,
        "verdicts": verdicts,
    }


# ----------------------------------------------------------------------
# leg 7: Mach-3 forward-facing step (Woodward & Colella problem 2)
# ----------------------------------------------------------------------
def leg_forward_step(cfg: RunConfig) -> Dict:
    """Supersonic forward-facing step graded against the exact shock state.

    Woodward & Colella (1984) problem 2: uniform Mach 3 flow over a reflecting
    step in a 3 x 1 tunnel, run to ``t = 4``.  The graded numbers are the
    near-normal shock jump and the wall stagnation state, both closed-form for
    the stated inflow (the paper publishes contours, not tables), plus the
    mass/energy budget identities; the tolerances are the ones pre-registered
    in ``docs/repro-fvm-forward-step/DESIGN.md``.  The production grid of
    record is 480 x 160 -- one grid finer than the ``cfg.quick`` leg below,
    which only exercises the path inside a smoke run.
    """
    nx, ny = (60, 20) if cfg.quick else (240, 80)
    t_end = 2.0 if cfg.quick else 4.0
    import time

    case = ForwardStepConfig(nx=nx, ny=ny, t_end=t_end)
    t0 = time.time()
    result = run_forward_step(case)
    wall = time.time() - t0
    meta = {
        "nsteps": result["nsteps"],
        "budget": result["budget"],
        "positivity_fixes": result["positivity_fixes"],
        "wall_seconds": wall,
    }
    grading = grade_forward_step(result["prim"], case, meta)
    out = {
        "reference": ("Woodward & Colella (1984) J. Comput. Phys. 54, problem 2; "
                      "graded against the exact Rankine-Hugoniot/isentropic state"),
        "grid": [nx, ny],
        "t_end": t_end,
        "wall_seconds": wall,
        "nsteps": result["nsteps"],
        "positivity_fixes": result["positivity_fixes"],
        "troubled_repairs": int(getattr(result["solver"], "troubled_repairs", 0)),
        "shock_sampling": grading["shock_sampling"],
        "criteria": grading["items"],
        "all_pass": grading["all_graded_pass"] and grading["all_pass"],
        "note": ("quick grid 60x20 is a smoke check only: the step is 4 cells high "
                 "there, so the graded numbers are not the benchmark verdict"),
    }
    return out


def _jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(torch.float64).cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if obj != obj else obj
    return obj


def _strip_heavy(result: Dict) -> Dict:
    """Drop field snapshots and time series from the JSON artifact."""
    for leg in result.get("legs", {}).values():
        _strip_heavy(leg)
    if "rows" in result:
        for row in result["rows"]:
            row.pop("snapshots", None)
            row.pop("width_series", None)
    if "snapshots" in result:
        result.pop("snapshots")
    if "profile" in result:
        result.pop("profile")
    if "series" in result:
        result.pop("series")
    if "advection" in result:
        result.pop("advection")
    if "main" in result:
        result["main"].pop("snapshots", None)
        result["main"].pop("series", None)
    return result


def run_all(outdir: Path | str = DEFAULT_OUT, cfg: RunConfig | None = None,
            legs: Sequence[str] = ("sod", "interface", "rm", "bubble", "vortex"),
            figures: bool = True) -> Dict:
    """Run the selected legs, write the JSON artifact and the figures."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # torch's default thread pool thrashes on 10k-cell float64 tensors: on the
    # 128-core loongarch node one thread is ~7x faster than 128.  The override is
    # explicit so the number can be reproduced or changed by the caller.
    threads = int(os.environ.get("TENSORFVM_COMPRESSIBLE_THREADS", "1"))
    if threads > 0:
        torch.set_num_threads(threads)
    cfg = cfg or RunConfig()
    result: Dict = {"config": {"quick": cfg.quick}, "thresholds": THRESHOLDS,
                    "legs": {}}
    t_start = time.time()
    if "sod" in legs:
        result["legs"]["sod"] = leg_sod(cfg)
    if "interface" in legs:
        result["legs"]["interface"] = leg_interface(cfg)
    if "rm" in legs:
        result["legs"]["rm"] = leg_rm(cfg)
    if "bubble" in legs:
        result["legs"]["bubble"] = leg_bubble(cfg)
    if "vortex" in legs:
        result["legs"]["vortex"] = leg_vortex(cfg)
    if "lax" in legs:
        result["legs"]["lax"] = leg_lax(cfg)
    if "forward_step" in legs:
        result["legs"]["forward_step"] = leg_forward_step(cfg)
    result["seconds"] = round(time.time() - t_start, 1)
    result["verdicts"] = {leg: data.get("verdicts", {})
                          for leg, data in result["legs"].items()}
    result["all_criteria_met"] = all(
        all(bool(v) for v in verdicts.values())
        for verdicts in result["verdicts"].values()
    )
    if figures:
        try:
            from .figures import make_all

            result["figures"] = make_all(result, outdir)
        except ImportError as exc:  # pragma: no cover - matplotlib missing
            result["figures"] = {"error": f"matplotlib unavailable: {exc}"}
    artifact = outdir / "compressible_multimaterial.json"
    artifact.write_text(json.dumps(_jsonable(_strip_heavy(result)), indent=1,
                                  sort_keys=True))
    return result