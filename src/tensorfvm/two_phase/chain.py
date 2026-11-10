"""Acceptance chain of the TensorFVM two-phase verification node (P254W1-N01).

Four coupled segments plus one negative control each, one ledger, one verdict
table:

======  ====================================================================
M1      drift-flux void fraction ``alpha(z)`` in a vertical pipe vs the
        Wallis (1969) closed-form solution; control: ``G_g = 0`` degenerates to
        single-phase liquid flow.
M2      PLIC-VOF dam break vs Martin & Moyce (1952) surge front and residual
        column height; control: ``g = 0`` keeps the interface frozen.
M3      Rayleigh-Plesset free oscillation vs the Plesset (1949) linear solution
        and the cavitation void-transport mass ledger; control: ``p_v = p_inf``.
M4      one-phase Stefan front vs the Carslaw & Jaeger similarity solution;
        control: ``L = 0`` has no phase change and reduces to pure conduction.
======  ====================================================================

Every measured number is written to the SQLite provenance ledger
(:mod:`tensorfvm.two_phase.provenance`) together with its criterion
(comparator + limit) and read back with SQL for the report, so the report cannot
contain a hand-copied number.  The ``J5`` (mass closure) and ``J6`` (traceability)
checks are computed from those rows as well.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from . import cavitation_transport as cav
from . import drift_flux as df
from . import stefan as st
from . import vof as vof_mod
from .provenance import ProvenanceDB

__all__ = ["ChainConfig", "run_chain", "CRITERIA", "judge"]


@dataclass
class ChainConfig:
    """Resolution and budget knobs of the chain (tests use the cheap setting)."""

    drift_cells: int = 200
    vof_nx: int = 120
    vof_ny: int = 40
    vof_t_star: float = 4.0
    vof_max_steps: int = 20000
    cavity_periods: float = 6.0
    cavity_cells: int = 200
    stefan_cells: int = 400
    stefan_t_end: float = 600.0
    vof_compare_algebraic: bool = True
    figure_dir: str | None = None
    make_figures: bool = True


#: ``(metric, segment, comparator, limit, criterion)`` -- the machine-readable
#: acceptance table.  ``limit`` is ``None`` for informational metrics.
CRITERIA = [
    # --- M1 drift flux -------------------------------------------------
    ("alpha_max_rel_error", "drift", "<=", 0.10, "J1"),
    ("control_alpha_max", "drift", "<=", 1e-12, "J1"),
    ("control_liquid_velocity_error", "drift", "<=", 1e-12, "J1"),
    ("gas_mass_drift", "drift", "<=", 0.01, "J5"),
    ("liquid_mass_drift", "drift", "<=", 0.01, "J5"),
    # --- M2 PLIC-VOF dam break ----------------------------------------
    ("front_max_rel_error", "vof", "<=", 0.15, "J2"),
    ("height_max_rel_error", "vof", "<=", 0.15, "J2"),
    ("control_max_abs_change", "vof", "<=", 1e-14, "J2"),
    ("volume_drift", "vof", "<=", 0.01, "J5"),
    ("alpha_bound_violation", "vof", "<=", 1e-10, "J5"),
    # --- M3 Rayleigh-Plesset + void transport -------------------------
    ("plesset_max_rel_deviation", "cavitation", "<=", 0.05, "J3"),
    ("frequency_rel_error", "cavitation", "<=", 0.05, "J3"),
    ("control_radius_drift", "cavitation", "<=", 1e-12, "J3"),
    ("bubble_number_conservation", "cavitation", "<=", 1e-10, "J5"),
    ("gas_mass_budget_residual", "cavitation", "<=", 0.01, "J5"),
    # --- M4 Stefan ----------------------------------------------------
    ("front_max_rel_error", "stefan", "<=", 0.03, "J4"),
    ("control_front_max", "stefan", "<=", 1e-12, "J4"),
    ("control_temperature_rel_error", "stefan", "<=", 1e-3, "J4"),
    ("energy_ledger_residual", "stefan", "<=", 0.01, "J5"),
]


def judge(comparator: str, value: float, limit: float) -> bool:
    if comparator == "<=":
        return bool(value <= limit)
    if comparator == "<":
        return bool(value < limit)
    if comparator == ">=":
        return bool(value >= limit)
    raise ValueError(f"unknown comparator {comparator!r}")


def _record(db: ProvenanceDB, run_id: int, segment: str, name: str,
            value: float, unit: str, note: str = "",
            comparator: str | None = None,
            limit: float | None = None) -> bool:
    return db.add_metric(run_id, segment, name, value, unit, comparator, limit,
                         note)


def _criterion(segment: str, name: str) -> tuple[str, float] | tuple[None, None]:
    for metric, seg, comparator, limit, _criterion_id in CRITERIA:
        if seg == segment and metric == name:
            return comparator, limit
    return None, None


# --------------------------------------------------------------------------- #
# M1: drift flux
# --------------------------------------------------------------------------- #
def run_drift_segment(db: ProvenanceDB, cfg: ChainConfig) -> dict:
    pipe = df.DriftFluxConfig(n_cells=cfg.drift_cells)
    ref = df.wallis_profile(pipe)
    num = df.solve_drift_flux(pipe)
    cmp = df.compare_profiles(ref, num)

    run = db.start_run("drift", "vertical pipe, air-water upflow", pipe.__dict__)
    for name, value, unit, note in [
        ("alpha_max_rel_error", cmp["max_rel_error"], "-",
         "max |alpha_FVM - alpha_Wallis|/alpha_Wallis"),
        ("alpha_rms_rel_error", cmp["rms_rel_error"], "-", "rms over cells"),
        ("pressure_max_rel_error", cmp["max_abs_pressure_error"], "-",
         "max |p_FVM - p_reference|/p"),
        ("gas_mass_drift", num.gas_mass_drift, "-",
         "max deviation of the gas mass flux along z"),
        ("liquid_mass_drift", num.liquid_mass_drift, "-",
         "max deviation of the liquid mass flux along z"),
        ("alpha_inlet", float(num.alpha[0]), "-", "inlet void fraction"),
        ("alpha_outlet", float(num.alpha[-1]), "-", "outlet void fraction"),
        ("drift_velocity", pipe.v_dj, "m/s", "Wallis churn-turbulent v_dj"),
        ("steady_residual", float(num.residual), "1/s",
         "relative gas-mass change per second at the end of the march"),
        ("steps", float(num.steps), "-", "time steps to steady state"),
    ]:
        comparator, limit = _criterion("drift", name)
        _record(db, run, "drift", name, value, unit, note, comparator, limit)

    # negative control: no gas at all -> single-phase liquid flow
    zero = df.solve_drift_flux(df.DriftFluxConfig(n_cells=cfg.drift_cells,
                                                  gas_mass_flux=0.0))
    _record(db, run, "drift", "control_alpha_max", float(np.max(zero.alpha)), "-",
            "void fraction with G_g = 0 (must vanish)", *_criterion("drift",
                                                                     "control_alpha_max"))
    v_err = float(np.max(np.abs(zero.v_liquid - pipe.liquid_superficial_velocity))
                  / pipe.liquid_superficial_velocity)
    _record(db, run, "drift", "control_liquid_velocity_error", v_err, "-",
            "|v_l - j_l|/j_l with G_g = 0", *_criterion(
                "drift", "control_liquid_velocity_error"))
    gradient = float((zero.pressure[0] - zero.pressure[-1])
                     / (pipe.length - pipe.dz))
    _record(db, run, "drift", "control_pressure_gradient_rel_error",
            abs(gradient - pipe.rho_liquid * pipe.gravity)
            / (pipe.rho_liquid * pipe.gravity), "-",
            "dp/dz vs rho_l g with G_g = 0 (half-cell offset removed)")

    # no-slip limit: v_dj = 0 must reproduce the homogeneous void fraction
    homogeneous = df.DriftFluxConfig(n_cells=cfg.drift_cells, drift_kind="churn")
    alpha_h = df.homogeneous_void_fraction(
        homogeneous.gas_mass_flux
        / (homogeneous.p_bottom / (homogeneous.r_gas * homogeneous.temperature)),
        homogeneous.liquid_superficial_velocity, homogeneous.distribution)
    closed = df.wallis_void_fraction(
        homogeneous.gas_mass_flux
        / (homogeneous.p_bottom / (homogeneous.r_gas * homogeneous.temperature)),
        homogeneous.liquid_superficial_velocity, 0.0, homogeneous.distribution)
    _record(db, run, "drift", "homogeneous_limit_error",
            abs(closed - alpha_h) / max(alpha_h, 1e-30), "-",
            "v_dj -> 0 limit vs the no-slip closed form")

    # first-order grid convergence of the FVM (documented, not a gate)
    order = []
    for n in (cfg.drift_cells // 4 or 25, cfg.drift_cells // 2 or 50,
              cfg.drift_cells):
        pipe_n = df.DriftFluxConfig(n_cells=n)
        ref_n = df.wallis_profile(pipe_n)
        num_n = df.solve_drift_flux(pipe_n)
        order.append(df.compare_profiles(ref_n, num_n)["max_rel_error"])
    _record(db, run, "drift", "refinement_error_coarse", order[0], "-",
            f"max rel error at n={cfg.drift_cells // 4 or 25}")
    _record(db, run, "drift", "refinement_error_fine", order[-1], "-",
            f"max rel error at n={cfg.drift_cells}")

    return {"run_id": run, "z": ref.z, "alpha_ref": ref.alpha,
            "z_num": num.z, "alpha_num": num.alpha, "p_num": num.pressure,
            "p_ref": ref.pressure, "refinement": order,
            "config": pipe.__dict__, "n_cells": cfg.drift_cells}


# --------------------------------------------------------------------------- #
# M2: PLIC-VOF dam break
# --------------------------------------------------------------------------- #
def run_vof_segment(db: ProvenanceDB, cfg: ChainConfig) -> dict:
    from ..vof_phase0.anchors import martin_moyce_front, martin_moyce_height
    from ..vof_phase0.dam_break import DamBreakConfig

    dam = DamBreakConfig(nx=cfg.vof_nx, ny=cfg.vof_ny,
                         t_star_end=cfg.vof_t_star,
                         max_steps=cfg.vof_max_steps)
    solver = vof_mod.plic_solver(dam)
    out = vof_mod.dam_break_history(dam, solver=solver)
    run = db.start_run("vof", "Martin & Moyce dam break (PLIC-VOF)",
                       {"config": out["config"],
                        "clip_volume": out["vof_clip_volume"]})

    t_star = np.asarray(out["t_star"], dtype=float)
    front_star = np.asarray(out["front_star"], dtype=float)   # (n, thresholds)
    height_star = np.asarray(out["height_star"], dtype=float)
    front_ref = np.array([martin_moyce_front(t) for t in t_star])
    height_ref = np.array([martin_moyce_height(t) for t in t_star])

    window = (t_star > 0.3) & (t_star <= min(cfg.vof_t_star, 8.0))
    front_err = np.abs(front_star[window, 1] - front_ref[window]) / np.maximum(
        np.abs(front_ref[window]), 1e-6)                       # half-cell gate
    lead_err = np.abs(front_star[window, 0] - front_ref[window]) / np.maximum(
        np.abs(front_ref[window]), 1e-6)                       # leading edge
    height_err = np.abs(height_star[window] - height_ref[window]) / np.maximum(
        np.abs(height_ref[window]), 1e-6)
    bands = {"early": (0.3, 1.0), "mid": (1.0, 2.0), "late2": (2.0, 3.0),
             "late": (3.0, 99.0)}
    band_errors = {}
    for key, (lo, hi) in bands.items():
        sel = window & (t_star >= lo) & (t_star < hi)
        band_errors[key] = (float(np.max(np.abs(
            front_star[sel, 1] - front_ref[sel]) / np.maximum(
            np.abs(front_ref[sel]), 1e-6))) if np.any(sel) else float("nan"))
    late = window & (t_star >= 3.0)
    late_err = (np.abs(front_star[late, 1] - front_ref[late])
                / np.maximum(np.abs(front_ref[late]), 1e-6)
                if np.any(late) else np.array([np.nan]))

    alpha = np.asarray(out["alpha_final"])
    values = [
        ("front_max_rel_error", float(np.max(front_err)), "-",
         "max |x_f* - Martin&Moyce|/ref over 0.3 < t* <= t_end, depth > half a cell"),
        ("front_mean_rel_error", float(np.mean(front_err)), "-", "mean over the window"),
        ("front_max_rel_error_leading_edge", float(np.max(lead_err)), "-",
         "same with the depth > 2% of a cell convention (sensitivity)"),
        ("height_max_rel_error", float(np.max(height_err)), "-",
         "max |h/h0 - Martin&Moyce|/ref"),
        ("front_rel_error_late", float(np.max(late_err)), "-",
         "front error for t* >= 3 (asymptotic agreement)"),
        ("volume_drift", float(np.max(np.abs(out["volume_drift"]))), "-",
         "max relative liquid-volume drift"),
        ("alpha_bound_violation", float(max(0.0, alpha.max() - 1.0,
                                            -alpha.min())), "-",
         "max excursion of alpha outside [0,1]"),
        ("vof_clip_volume", float(out["vof_clip_volume"]), "1",
         "liquid fraction redistributed by the bound limiter (accumulated)"),
        ("steps", float(out["steps"]), "-", "time steps"),
        ("divergence_max", float(np.max(out["divergence_max"])), "1/s",
         "max cell divergence of the projected field"),
        ("front_final_star", float(front_star[-1, 1]), "-", "x_f* at the end"),
        ("height_final_star", float(height_star[-1]), "-", "h/h0 at the end"),
        ("mu_liquid", float(dam.mu_liquid), "Pa s", "liquid viscosity"),
        ("dx_mm", float(dam.dx * 1e3), "mm", "cell size"),
    ]
    values += [(f"front_rel_error_band_{key}", value, "-",
                f"front error in the band t* in [{bands[key][0]}, {bands[key][1]})")
               for key, value in band_errors.items()]
    for name, value, unit, note in values:
        comparator, limit = _criterion("vof", name)
        _record(db, run, "vof", name, value, unit, note, comparator, limit)

    # negative control: g = 0 must freeze the interface
    dam0 = DamBreakConfig(nx=max(16, cfg.vof_nx // 2),
                          ny=max(8, cfg.vof_ny // 2), gravity=0.0,
                          pressure_init="zero", t_star_end=1.0)
    control = vof_mod.static_interface_drift(dam0, n_steps=40)
    for name, value, unit in [
            ("control_max_abs_change", control["max_abs_change"], "-"),
            ("control_volume_drift", control["volume_drift"], "-"),
            ("control_speed_max", control["speed_max"], "m/s")]:
        comparator, limit = _criterion("vof", name)
        _record(db, run, "vof", name, value, unit,
                "g = 0 static-interface control", comparator, limit)

    # transport-quality cross-check: PLIC vs the algebraic FCT scheme on the same
    thickness = {"plic": _interface_thickness(alpha)}
    if cfg.vof_compare_algebraic:
        from ..vof_phase0.dam_break import run_dam_break as algebraic_run
        algebraic = algebraic_run(DamBreakConfig(nx=cfg.vof_nx, ny=cfg.vof_ny,
                                                 t_star_end=cfg.vof_t_star,
                                                 max_steps=cfg.vof_max_steps))
        thickness["algebraic"] = _interface_thickness(
            np.asarray(algebraic["alpha_final"]))
        alg_err = np.abs(
            np.interp(t_star[window], algebraic["t_star"],
                      algebraic["front_star"]) - front_ref[window]) / np.maximum(
            np.abs(front_ref[window]), 1e-6)
        _record(db, run, "vof", "algebraic_front_max_rel_error",
                float(np.max(alg_err)), "-",
                "same comparison for the algebraic FCT transport (reference)")
        _record(db, run, "vof", "algebraic_volume_drift",
                float(np.max(np.abs(algebraic["volume_drift"]))), "-",
                "volume drift of the algebraic FCT transport")
    _record(db, run, "vof", "plic_interface_thickness_cells", thickness["plic"],
            "cells", "mean cells spanning 0.1 < alpha < 0.9 in the final field")
    if "algebraic" in thickness:
        _record(db, run, "vof", "algebraic_interface_thickness_cells",
                thickness["algebraic"], "cells",
                "same measure for the algebraic FCT transport")
        _record(db, run, "vof", "interface_thickness_ratio",
                thickness["plic"] / max(thickness["algebraic"], 1e-12), "1",
                "PLIC / algebraic interface thickness (smaller is sharper)")

    return {"run_id": run, "t_star": t_star, "front_star": front_star,
            "height_star": height_star, "window": window,
            "ref_front": front_ref, "ref_height": height_ref,
            "alpha_final": alpha, "config": out["config"],
            "dx": dam.dx, "dy": dam.dy, "h0": dam.h0, "a": dam.a,
            "thickness": thickness}


def _interface_thickness(alpha: np.ndarray) -> float:
    """Mean number of cells between alpha = 0.1 and 0.9, per wetted column."""
    widths = []
    for j in range(alpha.shape[1]):
        column = alpha[:, j]
        inside = np.where((column > 0.1) & (column < 0.9))[0]
        if inside.size:
            widths.append(inside.size)
    return float(np.mean(widths)) if widths else 0.0


# --------------------------------------------------------------------------- #
# M3: Rayleigh-Plesset + void transport
# --------------------------------------------------------------------------- #
def run_cavitation_segment(db: ProvenanceDB, cfg: ChainConfig) -> dict:
    bubble = cav.BubbleConfig()
    eps = 1.0e-3
    t, radius = cav.free_oscillation(bubble, eps=eps,
                                     n_periods=cfg.cavity_periods)
    reference = cav.plesset_solution(bubble, t, eps)
    dev = float(np.max(np.abs(radius - reference)) / bubble.r0)
    omega = cav.linear_frequency(bubble)
    f_meas = cav.measured_frequency(t, radius)
    beta = cav.linear_damping(bubble)
    # envelope decay of the numerical solution vs the analytic e^{-beta t}
    peaks = _peak_indices(radius)
    peak_t = t[peaks]
    peak_amp = (radius[peaks] - bubble.r0) / (eps * bubble.r0)
    fit = np.polyfit(peak_t, np.log(np.maximum(peak_amp, 1e-300)), 1)
    run = db.start_run("cavitation", "Rayleigh-Plesset + void transport",
                       {"bubble": bubble.__dict__, "eps": eps,
                        "n_periods": cfg.cavity_periods})

    values = [
        ("plesset_max_rel_deviation", dev, "-",
         "max |R_num - R_Plesset|/R0 over the sampled window"),
        ("frequency_rel_error", abs(f_meas - omega) / omega, "-",
         "zero-crossing frequency vs Plesset omega"),
        ("damping_rel_error", abs(-fit[0] - beta) / beta, "-",
         "measured envelope decay rate vs beta = 2 mu/(rho R0^2)"),
        ("natural_frequency", omega, "rad/s", "Plesset natural frequency"),
        ("surface_tension_shift", 1.0 - cav.linear_room_frequency(bubble) / omega,
         "1", "relative difference to sqrt(3 gamma p0/(rho R0^2))"),
        ("period_us", 2.0 * math.pi / omega * 1.0e6, "us", "linear period"),
        ("bubble_r0", bubble.r0, "m", "equilibrium radius"),
    ]
    for name, value, unit, note in values:
        comparator, limit = _criterion("cavitation", name)
        _record(db, run, "cavitation", name, value, unit, note, comparator, limit)

    # negative control: p_v = p_inf with the equilibrium gas content
    t_zero, r_zero = cav.integrate_bubble(bubble, bubble.r0,
                                          4.0 * 2.0 * math.pi / omega)
    _record(db, run, "cavitation", "control_radius_drift",
            float(np.max(np.abs(r_zero - bubble.r0)) / bubble.r0), "-",
            "p_inf = p_v + 2 sigma/R0 equilibrium: R must not drift",
            *_criterion("cavitation", "control_radius_drift"))

    # void transport with bubble growth in a low-pressure duct
    transport = cav.CavitationTransportConfig(n_cells=cfg.cavity_cells,
                                              t_end=0.05)
    result = cav.solve_void_transport(transport)
    nb_error = abs(result["nb_total"] - result["nb_total_initial"]) / max(
        result["nb_total_initial"], 1e-300)
    budget = abs(result["gas_mass"] - result["gas_mass_initial"]
                 - result["source_total"]) / max(abs(result["gas_mass"]
                                                     - result["gas_mass_initial"]),
                                                 1e-300)
    _record(db, run, "cavitation", "bubble_number_conservation", nb_error, "-",
            "|n_b(t) - n_b(0)|/n_b(0) under pure advection",
            *_criterion("cavitation", "bubble_number_conservation"))
    _record(db, run, "cavitation", "gas_mass_budget_residual", budget, "-",
            "|d(alpha rho_v) - int m_dot| / d(alpha rho_v)",
            *_criterion("cavitation", "gas_mass_budget_residual"))
    _record(db, run, "cavitation", "void_fraction_max",
            float(np.max(result["void_fraction"])), "-",
            "max void fraction reached in the duct")
    _record(db, run, "cavitation", "radius_ratio_max",
            float(np.max(result["radius"]) / bubble.r0), "-",
            "max R/R0 reached in the duct")

    # Rayleigh asymptotic anchor for the dynamic growth
    # vapour-bubble limit (negligible gas content) in a sub-vapour-pressure
    # region: the inertia-controlled Rayleigh growth is the late-time asymptote
    vapour = cav.BubbleConfig(r0=1.0e-3, p_gas_override=1.0)
    t_ray, r_ray = cav.integrate_bubble(vapour, vapour.r0, 2.0e-2,
                                       p_inf=1.0e3, n_samples=2001)
    ray_ref = cav.rayleigh_growth(vapour, t_ray, 1.0e3)
    mask = ray_ref > 5.0 * vapour.r0
    _record(db, run, "cavitation", "rayleigh_growth_rel_error",
            float(np.max(np.abs(r_ray[mask] - ray_ref[mask]) / ray_ref[mask])),
            "-", "dynamic growth vs sqrt(2 dp/(3 rho)) t for R > 5 R0")

    return {"run_id": run, "t": t, "radius": radius, "plesset": reference,
            "r0": bubble.r0, "z": result["z"], "duct_radius": result["radius"],
            "duct_alpha": result["void_fraction"], "duct_pressure":
            result["pressure"], "t_ray": t_ray, "r_ray": r_ray, "ray": ray_ref}


def _peak_indices(values: np.ndarray) -> np.ndarray:
    idx = []
    for i in range(1, values.size - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            idx.append(i)
    return np.asarray(idx, dtype=int)


# --------------------------------------------------------------------------- #
# M4: Stefan
# --------------------------------------------------------------------------- #
def run_stefan_segment(db: ProvenanceDB, cfg: ChainConfig) -> dict:
    case = st.StefanConfig(n_cells=cfg.stefan_cells, t_end=cfg.stefan_t_end)
    result = st.solve_stefan_1d(case)
    reference = st.neumann_front(result["t"], result["lambda"], case.alpha)
    err = np.abs(result["front"] - reference) / np.maximum(reference, 1e-12)
    run = db.start_run("stefan", "one-phase Stefan solidification",
                       case.__dict__)

    for name, value, unit, note in [
        ("front_max_rel_error", float(np.max(err)), "-",
         "max |s_num - 2 lambda sqrt(alpha t)| / ref"),
        ("front_mean_rel_error", float(np.mean(err)), "-", "mean over samples"),
        ("energy_ledger_residual", float(result["energy_ledger_residual"]), "-",
         "|dH + Q_wall| / |dH|"),
        ("stefan_number", float(case.stefan_number), "-", "c dT / L"),
        ("similarity_lambda", float(result["lambda"]), "-", "lambda(St)"),
        ("front_final", float(result["final_front"]), "m", "s(t_end)"),
        ("analytic_front_final", float(np.max(reference)), "m",
         "2 lambda sqrt(alpha t_end)"),
        ("grid_cells", float(case.n_cells), "-", "cells across the domain"),
        ("time_step", float(result["dt"]), "s", "explicit stability step"),
    ]:
        comparator, limit = _criterion("stefan", name)
        _record(db, run, "stefan", name, value, unit, note, comparator, limit)

    # negative control: no latent heat -> no phase change, pure conduction
    zero_case = st.StefanConfig(n_cells=cfg.stefan_cells, t_end=cfg.stefan_t_end,
                                latent=0.0)
    zero = st.solve_stefan_1d(zero_case)
    analytic = st.conduction_temperature(zero["x"], zero["t"][-1], zero_case.alpha,
                                         zero_case.t_fusion, zero_case.t_wall)
    temp_err = float(np.max(np.abs(zero["temperature"] - analytic))
                     / (zero_case.t_fusion - zero_case.t_wall))
    _record(db, run, "stefan", "control_front_max",
            float(np.max(np.abs(zero["front"]))), "m",
            "front position with L = 0", *_criterion("stefan",
                                                      "control_front_max"))
    _record(db, run, "stefan", "control_temperature_rel_error", temp_err, "-",
            "L = 0 temperature vs the erf conduction solution",
            *_criterion("stefan", "control_temperature_rel_error"))
    _record(db, run, "stefan", "control_energy_error",
            float(zero["energy_ledger_residual"]), "-",
            "L = 0 energy ledger residual")

    # cross-check of the similarity constant against the independent
    # implementation in tensorfvm.cht.stefan (two code paths, one root)
    from ..cht.stefan import neumann_lambda as cht_neumann_lambda
    try:
        other = float(cht_neumann_lambda(case.stefan_number))
        _record(db, run, "stefan", "lambda_cross_check_error",
                abs(other - result["lambda"]) / result["lambda"], "-",
                "lambda vs tensorfvm.cht.stefan.neumann_lambda")
    except Exception as exc:                             # pragma: no cover
        _record(db, run, "stefan", "lambda_cross_check_error", float("nan"), "-",
                f"cross-check unavailable: {exc}")

    # two-phase similarity solution: the one-phase limit must be recovered
    limit = st.stefan_lambda_two_phase(case.stefan_number, 1e-12)
    _record(db, run, "stefan", "two_phase_limit_error",
            abs(limit - result["lambda"]) / result["lambda"], "-",
            "two-phase lambda with St_liquid -> 0 vs one-phase lambda")

    return {"run_id": run, "t": result["t"], "front": result["front"],
            "reference": reference, "x": result["x"],
            "temperature": result["temperature"], "analytic_temperature":
            st.neumann_temperature(result["x"], result["t"][-1],
                                   result["lambda"], case.alpha, case.t_fusion,
                                   case.t_wall),
            "control_temperature": zero["temperature"],
            "control_analytic": analytic, "config": case.__dict__,
            "t_end": zero["t"][-1]}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _figure_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3})
    return plt


def make_figures(data: dict, out_dir: str) -> list[dict]:
    """Render the four report figures; returns ``[{name, path, caption}]``."""
    os.makedirs(out_dir, exist_ok=True)
    plt = _figure_setup()
    figures = []

    # --- M1 ------------------------------------------------------------
    drift = data["drift"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    ax.plot(drift["z"], drift["alpha_ref"] * 100.0, "-",
            label="Wallis closed form + DOP853 pressure")
    ax.plot(drift["z_num"], drift["alpha_num"] * 100.0, "o", ms=3.0,
            markevery=6, label="FVM drift-flux")
    ax.set_xlabel("z [m]")
    ax.set_ylabel(r"void fraction $\alpha$ [%]")
    ax.set_title("M1 drift-flux: void fraction profile")
    ax.legend(loc="upper left", fontsize=7)
    ax = axes[1]
    ax.plot(drift["p_ref"] / 1e3, drift["alpha_ref"] * 100.0, "-",
            label="analytic")
    ax.plot(drift["p_num"] / 1e3, drift["alpha_num"] * 100.0, "s", ms=3.0,
            markevery=6, label="FVM")
    ax.set_xlabel("p [kPa]")
    ax.set_ylabel(r"$\alpha$ [%]")
    ax.set_title("gas expansion along the pipe")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "m1_drift_flux.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": "m1_drift_flux", "path": path,
                    "caption": "M1: void fraction against the Wallis closed-form "
                               "solution and the gas-expansion coupling in a "
                               "3 m vertical pipe (air-water, C0=1.2)."})

    # --- M2 ------------------------------------------------------------
    dam = data["vof"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    field = dam["alpha_final"]
    extent = [0.5 * dam["dx"], field.shape[1] * dam["dx"],
              0.5 * dam["dy"], field.shape[0] * dam["dy"]]
    im = ax.imshow(field, origin="lower", extent=extent, cmap="viridis",
                   vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"M2 PLIC-VOF dam break, liquid fraction at t*={dam['t_star'][-1]:.1f}")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\alpha$")
    ax = axes[1]
    ax.plot(dam["t_star"], dam["front_star"][:, 1], "-",
            label="PLIC-VOF front $x_f^*$")
    ax.plot(dam["t_star"][dam["window"]], dam["ref_front"][dam["window"]], "k^",
            ms=3.5, label="Martin & Moyce (1952)")
    ax.set_xlabel(r"$t^* = t\sqrt{g/a}$")
    ax.set_ylabel(r"$x^*=(x_f-a)/a$")
    ax.set_title("surge-front trajectory")
    ax2 = ax.twinx()
    ax2.plot(dam["t_star"], dam["height_star"], "--", color="tab:red",
             label="$h/h_0$")
    ax2.plot(dam["t_star"][dam["window"]], dam["ref_height"][dam["window"]],
             "rv", ms=3.5, label="M&M height")
    ax2.set_ylabel("$h/h_0$")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "m2_plic_dam_break.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": "m2_plic_dam_break", "path": path,
                    "caption": "M2: PLIC-VOF dam break — final liquid-fraction "
                               "field (left) and the surge front and residual "
                               "column height against Martin & Moyce (1952)."})

    # --- M3 ------------------------------------------------------------
    cavity = data["cavitation"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(cavity["t"] * 1e6, (cavity["radius"] / cavity["r0"] - 1.0) * 1e3,
            "-", label="Rayleigh-Plesset (DOP853)")
    ax.plot(cavity["t"] * 1e6, (cavity["plesset"] / cavity["r0"] - 1.0) * 1e3,
            "--", label="Plesset (1949) linear")
    ax.set_xlabel("t [us]")
    ax.set_ylabel(r"$(R-R_0)/R_0$ [$10^{-3}$]")
    ax.set_title("M3 small-amplitude bubble oscillation")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(cavity["z"] * 1e2, cavity["duct_radius"] * 1e6, "-",
            label="R [um]")
    ax2 = ax.twinx()
    ax2.plot(cavity["z"] * 1e2, cavity["duct_alpha"], "--", color="tab:red",
             label=r"$\alpha$")
    ax2.set_ylabel(r"void fraction $\alpha$")
    ax2.grid(False)
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("bubble radius [um]")
    ax.set_title("void transport with cavitation growth")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "m3_rayleigh_plesset.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": "m3_rayleigh_plesset", "path": path,
                    "caption": "M3: Rayleigh-Plesset free oscillation against the "
                               "Plesset linear solution (left) and the coupled "
                               "void-transport/bubble-growth profile (right)."})

    # --- M4 ------------------------------------------------------------
    stef = data["stefan"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    ax.plot(stef["t"], stef["front"] * 100.0, "o", ms=3.5, label="enthalpy FVM")
    ax.plot(stef["t"], stef["reference"] * 100.0, "-",
            label=r"Carslaw & Jaeger $2\lambda\sqrt{\alpha t}$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("front s(t) [cm]")
    ax.set_title("M4 Stefan front position")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(stef["x"] * 100.0, stef["temperature"] - 273.15, "o", ms=2.5,
            markevery=4, label="enthalpy FVM")
    ax.plot(stef["x"] * 100.0, stef["analytic_temperature"] - 273.15, "-",
            label="similarity solution")
    ax.plot(stef["x"] * 100.0, stef["control_temperature"] - 273.15, ":",
            label="L=0 control (conduction)")
    ax.plot(stef["x"] * 100.0, stef["control_analytic"] - 273.15, "--",
            label="L=0 erf reference")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("T [C]")
    ax.set_title("temperature field at the final time")
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = os.path.join(out_dir, "m4_stefan.png")
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": "m4_stefan", "path": path,
                    "caption": "M4: Stefan front against the similarity solution "
                               "(left) and the temperature field with the "
                               "zero-latent-heat control (right)."})

    return figures


# --------------------------------------------------------------------------- #
# chain driver
# --------------------------------------------------------------------------- #
def run_chain(db_path: str, cfg: ChainConfig | None = None,
              verbose: bool = True) -> dict:
    """Run the four segments, record the ledger and evaluate the criteria."""
    cfg = cfg or ChainConfig()
    db = ProvenanceDB(db_path)
    data: dict = {}
    try:
        if verbose:
            print("[chain] M1 drift-flux ...", flush=True)
        data["drift"] = run_drift_segment(db, cfg)
        if verbose:
            print("[chain] M2 PLIC-VOF dam break ...", flush=True)
        data["vof"] = run_vof_segment(db, cfg)
        if verbose:
            print("[chain] M3 Rayleigh-Plesset + void transport ...", flush=True)
        data["cavitation"] = run_cavitation_segment(db, cfg)
        if verbose:
            print("[chain] M4 Stefan ...", flush=True)
        data["stefan"] = run_stefan_segment(db, cfg)

        # J5 (mass closure) and J6 (traceability) are computed from the rows
        closure = db.query(
            "SELECT segment, name, value, limit_value FROM metrics"
            " WHERE name IN ('gas_mass_drift','liquid_mass_drift',"
            "'volume_drift','bubble_number_conservation',"
            "'gas_mass_budget_residual','energy_ledger_residual')")
        worst = max(closure, key=lambda row: row["value"] / row["limit_value"])
        run_j5 = db.start_run("chain", "J5 mass closure",
                              {"closure_rows": len(closure),
                               "worst": f"{worst['segment']}/{worst['name']}"})
        db.add_metric(run_j5, "chain", "mass_closure_worst_ratio",
                      worst["value"] / worst["limit_value"], "1", "<=", 1.0,
                      f"worst closure metric {worst['segment']}/{worst['name']}"
                      f" = {worst['value']:.3e} against {worst['limit_value']:.3g}")

        rows = db.criteria()
        recomputed = sum(
            1 for row in rows
            if bool(row["passed"]) == judge(row["comparator"], row["value"],
                                            row["limit_value"]))
        run_j6 = db.start_run("chain", "J6 SQL traceability",
                              {"criterion_rows": len(rows)})
        db.add_metric(run_j6, "chain", "verdict_consistency",
                      recomputed / max(len(rows), 1), "1", ">=", 1.0,
                      "verdict recomputed from (value, comparator, limit)")
        db.add_metric(run_j6, "chain", "criterion_rows", float(len(rows)), "-",
                      None, None, "criteria recorded in the ledger")

        if cfg.make_figures and cfg.figure_dir:
            figures = make_figures(data, cfg.figure_dir)
            for fig in figures:
                db.add_figure(None, fig["name"], fig["path"], fig["caption"])
            data["figures"] = figures
        return {"data": data, "db_path": db_path}
    finally:
        db.close()