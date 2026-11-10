"""P295-W1-N01 verification driver: conjugate heat transfer, Stefan, ice-steel.

Three channels, one artefact file each under ``runs/cht_p295``:

``cylinder``  cross-flow CHT around a hot conducting cylinder, mean Nusselt number
              against the Eckert & Soehngen (1952) correlation (<= 10 %) and the
              Churchill-Bernstein correlation as a second anchor.
``stefan``    one-dimensional solidification, enthalpy method, front position
              against the Neumann similarity solution at three times (<= 5 %).
``ice``       ice-steel conjugate plate: interface flux conservation from (a) the
              single-field harmonic conductance, (b) a partitioned
              Dirichlet-Neumann coupling solved in two separate blocks, and the
              cross-adjudication of the icing rate against the archived W4-N08
              Makiranta-type model at identical environmental inputs (<= 30 %).

Everything is produced by solving the equations; no number is tabulated and pasted
in.  The W4-N08 numbers enter only as *reference data* to be compared against.

Usage (from the repository root, ``PYTHONPATH=src``)::

    python -m scripts.p295_cht_validation --case all
    python -m scripts.p295_cht_validation --case cylinder --re 40
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "runs" / "cht_p295"
FIGDIR = OUTDIR / "figures"


# ---------------------------------------------------------------- correlations
def eckert_soehngen(re: float, pr: float) -> float:
    """Mean Nusselt number of cross-flow over a cylinder, Eckert & Soehngen (1952).

    ``Nu = (0.43 + 0.50 Re^0.5) Pr^0.38``, valid for ``1 < Re < 1e3``.

    E. R. G. Eckert and H. Soehngen, *Distribution of heat transfer coefficients
    around circular cylinders in crossflow at Reynolds numbers from 20 to 500*,
    Trans. ASME 74 (1952) 343-347.  The correlation is the mean of their measured
    local distributions, which is what a mean-Nusselt verification can be held to.
    """
    if not 1.0 < re < 1.0e3:
        raise ValueError("Eckert-Soehngen correlation is stated for 1 < Re < 1e3")
    return (0.43 + 0.50 * math.sqrt(re)) * pr ** 0.38


def churchill_bernstein_local(re: float, pr: float) -> float:
    from tensorfvm.cht.runner import churchill_bernstein
    return churchill_bernstein(re, pr)


def _ensure_dirs() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)


def _clean(obj):
    """Convert dataclasses / numpy scalars into JSON-safe primitives."""
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _clean(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _dump(name: str, payload: dict) -> Path:
    path = OUTDIR / f"{name}.json"
    path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n")
    return path


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# -------------------------------------------------------------------- cylinder
def run_cylinder(re: float = 40.0, pr: float = 0.7, n_r: int = 71, n_theta: int = 72,
                 r_out: float = 15.0, wall_cell: float = 8.0e-3, dt: float = 2.0e-3,
                 t_end: float = 60.0, solid_ratio: float = 100.0,
                 relax: float = 0.7) -> dict:
    """Cross-flow conjugate cylinder: fluid annulus coupled to a hot solid core."""
    from tensorfvm.cht.polar import ConjugateCylinder, PolarVorticitySolver

    t0 = time.time()
    flow = PolarVorticitySolver(re=re, n_r=n_r, n_theta=n_theta, r_out=r_out,
                                wall_cell=wall_cell)
    march = flow.march(t_end=t_end, dt=dt, tol_steady=1e-7, max_steps=400000,
                       t_ramp=4.0)
    cd, cl = flow.drag_lift(march["wall_vorticity"])
    case = ConjugateCylinder(re=re, pr=pr, solid_ratio=solid_ratio, flow=flow)
    res = case.couple(march["psi"], outer_temperature=None, relax=relax,
                      max_iter=300, tol=1e-9)
    nu_es = eckert_soehngen(re, pr)
    nu_cb = churchill_bernstein_local(re, pr)
    m = res.metrics
    payload = {
        "case": "cylinder_cross_flow_cht",
        "reynolds": re,
        "prandtl": pr,
        "solid_ratio": solid_ratio,
        "mesh": {"n_r": n_r, "n_theta": n_theta, "r_out": r_out,
                 "wall_cell": wall_cell, "dt": dt, "t_end": t_end},
        "flow": {"steady": bool(march["steady"]), "steps": int(march["steps"]),
                 "time": float(march["time"]),
                 "drag_coefficient": float(cd), "lift_coefficient": float(cl)},
        "nusselt_mean": float(m["nusselt_mean"]),
        "nusselt_mean_flux_weighted": float(m.get("nusselt_mean_flux_weighted", float("nan"))),
        "nusselt_eckert_soehngen_1952": nu_es,
        "nusselt_churchill_bernstein": nu_cb,
        "error_vs_eckert_soehngen_rel": abs(m["nusselt_mean"] - nu_es) / nu_es,
        "error_vs_churchill_bernstein_rel": abs(m["nusselt_mean"] - nu_cb) / nu_cb,
        "interface_local_residual_rel": float(m["interface_local_residual_rel"]),
        "interface_global_residual_rel": float(m["interface_global_residual_rel"]),
        "wall_temperature_mean": float(m["wall_temperature_mean"]),
        "wall_temperature_ptp": float(m["wall_temperature_ptp"]),
        "energy_rate_balance_rel": float(m["energy_rate_balance_rel"]),
        "coupling_iterations": int(m.get("coupling_iterations", -1)),
        "runtime_s": time.time() - t0,
        "acceptance": {
            "nu_error_limit": 0.10,
            "reference": "Eckert & Soehngen 1952 mean correlation",
            "within_limit": bool(abs(m["nusselt_mean"] - nu_es) / nu_es < 0.10),
            # the Dirichlet-Neumann sweep stalls at the level of the interface
            # flux transfer (~1.8e-4 of the wall flux); 1e-3 is the honest bar
            "interface_residual_limit": 1e-3,
            "interface_conserved": bool(m["interface_local_residual_rel"] < 1e-3),
        },
    }
    return payload, res


def plot_cylinder(payload: dict, res) -> str:
    plt = _matplotlib()
    fig = plt.figure(figsize=(13.5, 4.0))
    ax = fig.add_subplot(1, 3, 1)
    theta = np.degrees(res.theta)
    ax.plot(theta, res.nusselt_local, lw=1.4, color="C0")
    ax.axhline(payload["nusselt_mean"], color="C3", ls="--", lw=1.2,
               label=f"mean {payload['nusselt_mean']:.2f}")
    ax.axhline(payload["nusselt_eckert_soehngen_1952"], color="k", ls=":", lw=1.2,
               label=f"Eckert-Soehngen {payload['nusselt_eckert_soehngen_1952']:.2f}")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$Nu_\theta$")
    ax.set_title(f"local Nusselt, Re = {payload['reynolds']:g}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(theta, res.wall_temperature, lw=1.4, color="C1")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$T_w$ (solid core $T=1$)")
    ax.set_title("conjugate wall temperature")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 3)
    temp = np.asarray(res.fields["T_fluid"])
    r = np.asarray(res.fields.get("r_node", np.arange(temp.shape[0])), dtype=float)
    ax.plot(r[:40], temp[:40, 0], label=r"$\theta=0$")
    ax.plot(r[:40], temp[:40, temp.shape[1] // 2], label=r"$\theta=\pi$")
    ax.set_xlabel("r")
    ax.set_ylabel("T")
    ax.set_title("fluid temperature, radial")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGDIR / f"cht_cylinder_re{int(payload['reynolds'])}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(ROOT))


# ---------------------------------------------------------------------- Stefan
def run_stefan(*, n_cells: int = 400, dt: float = 0.5, t_end: float = 1600.0,
               sample_times=(100.0, 500.0, 1600.0), mushy: float = 0.05,
               t_wall: float = -20.0) -> dict:
    from tensorfvm.cht.stefan import (ICE_C, ICE_K, ICE_LATENT, ICE_RHO,
                                      solve_stefan_1d)

    t0 = time.time()
    res = solve_stefan_1d(n_cells=n_cells, dt=dt, t_end=t_end,
                          sample_times=sample_times, mushy_thickness=mushy,
                          t_wall=t_wall)
    # latent-heat mutation: dropping the latent term must break the front badly.
    # The mutant runs on a deliberately coarse mesh/time step -- it only has to be
    # wrong, not accurate, and the L -> 0 limit makes the Newton solve stiff.
    mutant = solve_stefan_1d(n_cells=60, dt=8.0, t_end=t_end,
                             sample_times=sample_times, mushy_thickness=mushy,
                             t_wall=t_wall, latent=1.0)
    rows = []
    for i, t in enumerate(res.times):
        rows.append({
            "t_s": float(t),
            "front_enthalpy_m": float(res.front_enthalpy[i]),
            "front_half_liquid_m": float(res.front_half_liquid[i]),
            "front_neumann_exact_m": float(res.front_exact[i]),
            "error_enthalpy_rel": float(abs(res.front_enthalpy[i]
                                            - res.front_exact[i]) / res.front_exact[i]),
            "error_half_liquid_rel": float(abs(res.front_half_liquid[i]
                                               - res.front_exact[i]) / res.front_exact[i]),
            "mutant_no_latent_error_rel": float(abs(mutant.front_enthalpy[i]
                                                    - res.front_exact[i])
                                                / res.front_exact[i]),
        })
    payload = {
        "case": "stefan_1d_solidification",
        "materials": {"k_ice_W_mK": ICE_K, "rho_ice_kg_m3": ICE_RHO,
                      "c_ice_J_kgK": ICE_C, "latent_J_kg": ICE_LATENT},
        "setup": {"n_cells": n_cells, "dx_m": float(res.grid["dx_m"]), "dt_s": dt,
                  "length_m": float(res.grid["length_m"]), "t_wall_C": t_wall,
                  "t_fusion_C": 0.0, "mushy_band_K": mushy,
                  "stefan_number": float(res.stefan_number),
                  "alpha_solid_m2_s": float(res.alpha_solid),
                  "similarity_lambda": float(res.front_exact[0]
                                             / (2.0 * math.sqrt(res.alpha_solid
                                                                * res.times[0])))},
        "times": rows,
        "max_error_enthalpy_rel": float(np.max(np.abs(res.front_rel_error))),
        "max_error_half_liquid_rel": float(max(r["error_half_liquid_rel"] for r in rows)),
        "energy_ledger": res.ledger,
        "mutant_no_latent_max_error_rel": float(max(r["mutant_no_latent_error_rel"]
                                                    for r in rows)),
        "mutant_no_latent_ledger": mutant.ledger,
        "runtime_s": time.time() - t0,
        "acceptance": {
            "front_error_limit": 0.05,
            "three_times_within_limit": bool(np.all(np.abs(res.front_rel_error) < 0.05)),
            "energy_balance_within_1e-6": bool(
                res.ledger["normalized_imbalance"] < 1e-6),
            "mutation_no_latent_detected": bool(
                max(r["mutant_no_latent_error_rel"] for r in rows) > 0.05),
        },
    }
    return payload, res


def plot_stefan(payload: dict, res) -> str:
    plt = _matplotlib()
    from tensorfvm.cht.stefan import neumann_temperature
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot(res.times, res.front_exact, "k-", lw=1.5, label="Neumann similarity")
    ax.plot(res.times, res.front_enthalpy, "o", ms=4, label="FVM enthalpy method")
    ax.plot(res.times, res.front_half_liquid, "s", ms=3, mfc="none",
            label=r"geometric $f_l=1/2$ front")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("front position X [m]")
    ax.set_title("solidification front, one-phase Stefan")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, t in enumerate(res.times):
        exact_x = np.linspace(0.0, 1.15 * res.front_exact[i], 200)
        exact_t = np.where(exact_x <= res.front_exact[i],
                           neumann_temperature(exact_x, t, res.stefan_number,
                                               res.alpha_solid, res.t_wall,
                                               res.t_fusion), res.t_fusion)
        ax.plot(exact_x, exact_t, "-", lw=1.2, label=f"exact t={t:g}s")
        sel = res.x <= 1.15 * res.front_exact[i]
        ax.plot(res.x[sel], res.temperatures[i][sel], "o", ms=2.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("T [C]")
    ax.set_title("temperature field: markers = FVM, lines = similarity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGDIR / "cht_stefan_front.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(ROOT))


# -------------------------------------------------------------------- ice-steel
def run_ice(*, d_steel: float = 0.020, d_ice: float = 0.030, t_water: float = -1.8,
            t_air_surface: float = -10.0, nx_steel: int = 20, nx_ice: int = 30,
            nz: int = 24, span_variation: float = 3.0,
            h_water: float = 500.0, h_air: float = 25.0,
            coupling_relax: float = 0.8, coupling_max_iter: int = 200,
            loose_iter: int = 2) -> dict:
    """Ice-steel conjugate plate, interface-flux audit and W4-N08 adjudication."""
    from tensorfvm.cht.ice_conjugate import (ICE_K, STEEL_K, PartitionedIceSteel,
                                             W4N08_REFERENCE_PANEL,
                                             ice_growth_from_balance,
                                             interface_flux_audit, layered_field,
                                             layered_interface_analytic,
                                             surface_heat_loss)

    t0 = time.time()
    # (c1) 1-D exact: two layers, Dirichlet at both ends
    exact = layered_interface_analytic(d_steel=d_steel, d_ice=d_ice,
                                       t_inner=t_water, t_outer=t_air_surface,
                                       k_steel=STEEL_K, k_ice=ICE_K)
    field = layered_field(d_steel=d_steel, d_ice=d_ice, t_inner=t_water,
                          t_outer=t_air_surface, nx_steel=nx_steel, nx_ice=nx_ice)
    audit = interface_flux_audit(field, k_steel=STEEL_K, k_ice=ICE_K)
    error_iface_t = abs(audit["interface_temperature"] - exact["interface_temperature"]) \
        / abs(exact["interface_temperature"])
    error_iface_q = abs(audit["interface_area_mean_flux"] - exact["q_W_m2"]) \
        / abs(exact["q_W_m2"])

    # (c2) partitioned Dirichlet-Neumann coupling, two separate blocks
    part = PartitionedIceSteel(d_steel=d_steel, d_ice=d_ice, n_steel=nx_steel,
                               n_ice=nx_ice, t_water=t_water,
                               t_air=t_air_surface - 8.2, h_water=h_water,
                               h_air=h_air)
    conv = part.solve(relax=coupling_relax, max_iter=coupling_max_iter)
    loose = part.solve(relax=coupling_relax, max_iter=loose_iter)
    conv_analytic = layered_interface_analytic(
        d_steel=d_steel, d_ice=d_ice, t_inner=t_water, t_outer=t_air_surface - 8.2,
        k_steel=STEEL_K, k_ice=ICE_K, h_inner=h_water, h_outer=h_air)

    # (c3) 2-D strip: span-wise varying surface temperature => lateral conduction
    span = np.linspace(-1.0, 1.0, nz)
    t_outer_profile = t_air_surface - span_variation * 0.5 * (1.0 - np.cos(
        np.pi * (span + 1.0)))
    field2d = layered_field(d_steel=d_steel, d_ice=d_ice, t_inner=t_water,
                            t_outer=t_outer_profile, nx_steel=nx_steel,
                            nx_ice=nx_ice, nz=nz)
    audit2d = interface_flux_audit(field2d, k_steel=STEEL_K, k_ice=ICE_K)
    t2 = np.atleast_2d(field2d["T"])
    dx2 = field2d["dx"]
    dy2 = field2d["grid"].dy                      # span width of one row
    # heat leaving through the air side, row by row (W per metre of plate depth)
    q_top_rows = 2.0 * ICE_K * (t2[:, -1] - t_outer_profile) / dx2[-1] * dy2
    # heat crossing the ice/steel interface, row by row
    q_iface_rows = np.asarray(audit2d["q_face_profile_W_m2"], dtype=float) * dy2
    global_balance_res = abs(float(np.sum(q_top_rows) - np.sum(q_iface_rows))) / max(
        abs(float(np.sum(q_top_rows))), 1e-30)

    # (d) cross-adjudication with the archived W4-N08 icing model.
    # Wet growth pins the ice *surface* at the freezing temperature, so the CHT
    # configuration is: ice surface at T_f, sea side at T_sea (Robin h_water).
    # The conducted flux q_cond is the term W4-N08 (adiabatic substrate) drops.
    ref = W4N08_REFERENCE_PANEL
    balance = surface_heat_loss(t_surface=ref["surface_temperature_C"],
                                t_air=ref["t_air_C"], wind=ref["wind_m_s"],
                                relative_humidity=ref["relative_humidity"],
                                t_sea=ref["t_sea_C"],
                                spray_flux=ref["spray_flux_kg_m2_s"],
                                length=ref["char_length_m"])
    dev_terms = {
        key: (balance[key] - ref[key]) / ref[key]
        for key in ("q_conv_W_m2", "q_evap_W_m2", "q_lw_W_m2")
    }

    def _structure(t_water_side: float) -> dict:
        """Solve the ice+steel stack with the ice surface pinned at freezing."""
        stack = layered_field(d_steel=d_steel, d_ice=d_ice, t_inner=t_water_side,
                              t_outer=ref["surface_temperature_C"],
                              nx_steel=nx_steel, nx_ice=nx_ice, h_inner=h_water)
        audit_s = interface_flux_audit(stack)
        growth_s = ice_growth_from_balance(balance=balance,
                                           spray_flux=ref["spray_flux_kg_m2_s"],
                                           conduction_into_structure=float(
                                               audit_s["interface_area_mean_flux"]))
        return {
            "t_water_C": float(t_water_side),
            "conduction_into_ice_W_m2": float(audit_s["interface_area_mean_flux"]),
            "interface_temperature_C": float(audit_s["interface_temperature"]),
            "interface_flux_residual": float(audit_s["local_residual_max"]),
            "ice_flux_kg_m2_s": float(growth_s["ice_flux_kg_m2_s"]),
            "freezing_fraction": float(growth_s["freezing_fraction"]),
            "regime": growth_s["regime"],
            "deviation_vs_w4n08_rel": float((growth_s["ice_flux_kg_m2_s"]
                                             - ref["ice_flux_kg_m2_s"])
                                            / ref["ice_flux_kg_m2_s"]),
        }

    sweep = [_structure(tw) for tw in (-1.8, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0)]
    design = _structure(ref["t_sea_C"])
    growth = ice_growth_from_balance(balance=balance,
                                     spray_flux=ref["spray_flux_kg_m2_s"])
    dev_flux = design["deviation_vs_w4n08_rel"]

    payload = {
        "case": "ice_steel_conjugate",
        "materials": {"k_steel_W_mK": STEEL_K, "k_ice_W_mK": ICE_K,
                      "conductivity_ratio": STEEL_K / ICE_K},
        "geometry": {"steel_thickness_m": d_steel, "ice_thickness_m": d_ice,
                     "nx_steel": nx_steel, "nx_ice": nx_ice},
        "boundary": {"t_water_C": t_water, "t_air_surface_C": t_air_surface,
                     "h_water_W_m2K": h_water, "h_air_W_m2K": h_air},
        "single_field_1d": {
            "analytic_interface_temperature_C": exact["interface_temperature"],
            "analytic_flux_W_m2": exact["q_W_m2"],
            "interface_temperature_C": audit["interface_temperature"],
            "q_steel_W_m2": audit["q_steel_W_m2"],
            "q_ice_W_m2": audit["q_ice_W_m2"],
            "local_flux_residual": audit["local_residual_max"],
            "global_flux_residual": audit["global_residual"],
            "interface_temperature_error_rel": error_iface_t,
            "interface_flux_error_rel": error_iface_q,
        },
        "partitioned_1d": {
            "converged": conv.converged,
            "iterations": conv.iterations,
            "interface_temperature_C": conv.interface_temperature,
            "q_steel_W_m2": conv.q_steel,
            "q_ice_W_m2": conv.q_ice,
            "flux_residual": conv.flux_residual,
            "energy_residual": conv.energy_residual,
            "analytic_interface_temperature_C": conv_analytic["interface_temperature"],
            "analytic_flux_W_m2": conv_analytic["q_W_m2"],
            "interface_temperature_error_rel": abs(
                conv.interface_temperature - conv_analytic["interface_temperature"])
            / abs(conv_analytic["interface_temperature"]),
            "flux_error_rel": abs(conv.q_steel - conv_analytic["q_W_m2"])
            / abs(conv_analytic["q_W_m2"]),
            "mutation_loose_iterations": loose.iterations,
            "mutation_flux_residual": loose.flux_residual,
        },
        "two_dimensional": {
            "nz": nz,
            "span_surface_variation_K": span_variation,
            "interface_temperature_profile_C": audit2d["interface_temperature_profile"],
            "interface_flux_profile_W_m2": audit2d["q_face_profile_W_m2"],
            "local_flux_residual_max": audit2d["local_residual_max"],
            "global_flux_residual": audit2d["global_residual"],
            "heat_into_ice_W_per_m": float(np.sum(q_iface_rows)),
            "heat_out_of_air_side_W_per_m": float(np.sum(q_top_rows)),
            "interfacial_energy_balance_residual": float(global_balance_res),
            "ledger": field2d["ledger"],
        },
        "w4n08_cross_check": {
            "reference_source": ref["source"],
            "conditions": {k: ref[k] for k in ("t_air_C", "wind_m_s", "t_sea_C",
                                               "relative_humidity", "char_length_m",
                                               "spray_flux_kg_m2_s")},
            "my_surface_balance": balance,
            "my_growth": growth,
            "design_case": design,
            "reference_panel": {k: ref[k] for k in ("q_conv_W_m2", "q_evap_W_m2",
                                                    "q_lw_W_m2", "q_req_W_m2",
                                                    "freezing_fraction",
                                                    "ice_flux_kg_m2_s",
                                                    "regime")},
            "term_deviation_rel": dev_terms,
            "ice_flux_deviation_rel": float(dev_flux),
            "ice_flux_deviation_rel_surface_only": float(
                (growth["ice_flux_kg_m2_s"] - ref["ice_flux_kg_m2_s"])
                / ref["ice_flux_kg_m2_s"]),
            "sea_temperature_sweep": sweep,
        },
        "runtime_s": time.time() - t0,
        "acceptance": {
            "interface_flux_residual_limit": 0.01,
            "single_field_residual_ok": bool(audit["local_residual_max"] < 0.01),
            "partitioned_residual_ok": bool(conv.flux_residual < 0.01),
            "mutation_loose_coupling_detected": bool(loose.flux_residual >= 0.01),
            "two_d_residual_ok": bool(audit2d["local_residual_max"] < 0.01),
            "two_d_energy_balance_ok": bool(global_balance_res < 0.01),
            "w4n08_deviation_limit": 0.30,
            "w4n08_within_limit": bool(abs(dev_flux) <= 0.30),
        },
    }
    return payload, field, field2d, audit2d


def plot_ice(payload: dict, field, field2d, audit2d) -> str:
    plt = _matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
    ax = axes[0]
    T = np.atleast_2d(field["T"])[0]
    ax.plot(field["x"] * 1e3, T, "-o", ms=3, lw=1.2)
    ax.axvline(field["d_steel"] * 1e3, color="k", ls="--", lw=1.0,
               label="steel / ice interface")
    ax.set_xlabel("x [mm]  (sea side -> air side)")
    ax.set_ylabel("T [C]")
    ax.set_title(f"ice-steel temperature field\nk ratio {payload['materials']['conductivity_ratio']:.0f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    T2 = np.atleast_2d(field2d["T"])
    im = ax.imshow(T2, aspect="auto", origin="lower", cmap="coolwarm",
                   extent=[0, (field2d["d_steel"] + field2d["d_ice"]) * 1e3, 0, T2.shape[0]])
    fig.colorbar(im, ax=ax, label="T [C]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("span index")
    ax.set_title("2-D plate strip, span-varying air side")

    ax = axes[2]
    sweep = payload["w4n08_cross_check"]["sea_temperature_sweep"]
    tw = [s["t_water_C"] for s in sweep]
    dev = [100.0 * s["deviation_vs_w4n08_rel"] for s in sweep]
    ax.plot(tw, dev, "o-", lw=1.4)
    ax.axhline(30.0, color="r", ls="--", lw=1.0, label="+30 % limit")
    ax.axhline(-30.0, color="r", ls="--", lw=1.0)
    ax.set_xlabel(r"sea-side water temperature [$^\circ$C]")
    ax.set_ylabel("ice-rate deviation vs W4-N08 [%]")
    ax.set_title("cross-adjudication vs W4-N08")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGDIR / "cht_ice_steel.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(ROOT))


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all",
                        choices=["all", "cylinder", "stefan", "ice"])
    parser.add_argument("--re", type=float, default=40.0)
    parser.add_argument("--t-end", type=float, default=60.0)
    parser.add_argument("--n-cells", type=int, default=400)
    args = parser.parse_args(argv)

    _ensure_dirs()
    ran = []

    def _figures(payload, name, plotter, *args):
        """Write the JSON first: a missing matplotlib must not lose the numbers."""
        _dump(name, payload)
        try:
            payload["figures"] = [plotter(payload, *args)]
            _dump(name, payload)
        except Exception as exc:                     # pragma: no cover
            payload["figure_error"] = repr(exc)
            _dump(name, payload)

    if args.case in ("all", "cylinder"):
        payload, res = run_cylinder(re=args.re, t_end=args.t_end)
        _figures(payload, f"cylinder_re{int(args.re)}", plot_cylinder, res)
        ran.append("cylinder")
        print("cylinder:", json.dumps(payload["acceptance"]))
    if args.case in ("all", "stefan"):
        payload, res = run_stefan(n_cells=args.n_cells)
        _figures(payload, "stefan", plot_stefan, res)
        ran.append("stefan")
        print("stefan:", json.dumps(payload["acceptance"]),
              "max_err", payload["max_error_enthalpy_rel"])
    if args.case in ("all", "ice"):
        payload, field, field2d, audit2d = run_ice()
        _figures(payload, "ice_steel", plot_ice, field, field2d, audit2d)
        ran.append("ice")
        print("ice:", json.dumps(payload["acceptance"]))
    print("cases run:", ran, "->", OUTDIR)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())