"""KVLCC2 self-propulsion + manoeuvring campaign (node IN63-N02).

Runs the whole chain once and writes machine-readable artefacts plus the figures
of the report:

1. geometry gate (Cb/Cwp/LCB) and reference conditions;
2. ITTC-78 resistance chain at the design Froude number + the anchor audit;
3. BEMT open-water curve and the self-propulsion point (closure residual);
4. rudder polar (analytic; the 2-D RANS attempt is loaded if an artefact exists)
   and the rudder force derivative;
5. turning circle (delta = 35 deg) and zig-zag (10/10) with the direct force
   model, plus the time-step convergence (GCI) of the tactical diameter;
6. the MMG (effective-derivative) model on the same manoeuvres for the mutual
   arbitration, including its failures;
7. the negative control (delta = 0) and the criteria table.

Every number written to JSON is computed here -- nothing is transcribed.

Usage::

    PYTHONPATH=src python scripts/run_maneuvering_campaign.py \
        --out docs/maneuvering-selfprop-cfd
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from tensorfvm import __name__ as _package  # noqa: F401  (import check)
from tensorfvm.maneuvering import (
    dynamics as D, hull as H, hullforces as HF, propeller as P, reference as REF,
    resistance as R, rudder as RD, selfpropulsion as S,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rudder-polar", type=Path, default=None,
                        help="optional 2-D RANS polar artefact")
    args = parser.parse_args(argv)
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    started = time.time()
    conditions = H.DEFAULT_CONDITIONS

    # ---------------------------------------------------------------- geometry
    gate = H.geometry_gate()
    planform = HF.planform_integrals()
    rudder = H.rudder_geometry()

    # ------------------------------------------------------------ resistance
    resistance = R.resistance_components()
    resistance_sweep = [
        R.resistance_components(u * conditions.u_design).as_dict()
        for u in (0.4, 0.6, 0.8, 1.0, 1.1)]

    # ------------------------------------------------------------- propeller
    propeller = P.PropellerGeometry()
    js = np.linspace(0.05, 1.0, 20)
    open_water = [P.open_water_point(propeller, float(j)).as_dict() for j in js]
    eta_max = max(point["eta"] for point in open_water)

    # -------------------------------------------------------- self-propulsion
    sp = S.solve_self_propulsion(propeller=propeller)
    interaction = S.hull_interaction(propeller=propeller)
    # sensitivity of the shaft rate to the empirical wake fraction
    wake_sensitivity = {}
    for wake in (0.30, 0.355, 0.40):
        point = S.solve_self_propulsion(propeller=propeller, wake=wake)
        wake_sensitivity[f"wake_{wake:.3f}"] = {
            "rps": point.rps, "J": point.advance_ratio, "KT": point.KT,
            "residual": point.residual}

    # ---------------------------------------------------------------- rudder
    analytic_polar = RD.analytic_rudder_polar(rudder)
    polar = analytic_polar
    polar_source = "analytic (Helmbold finite-AR + smooth stall)"
    if args.rudder_polar and Path(args.rudder_polar).exists():
        polar = RD.load_rudder_polar(args.rudder_polar)
        polar_source = polar.source
    rudder_derivatives = RD.rudder_derivatives(
        polar=polar, geometry=rudder, conditions=conditions,
        KT=sp.KT, J=sp.advance_ratio)
    rudder_derivatives_analytic = RD.rudder_derivatives(
        polar=analytic_polar, geometry=rudder, conditions=conditions,
        KT=sp.KT, J=sp.advance_ratio)

    # ------------------------------------------------------- negative control
    models = {}
    negative_control = {}
    for tag, pol in (("analytic", analytic_polar), ("loaded", polar)):
        model = D.ManoeuvreModel(conditions=conditions, self_propulsion=sp,
                                 polar=pol, mode="direct")
        zero = RD.rudder_normal_force(0.0, conditions.u_design, 0.0, 0.0,
                                      polar=pol, geometry=rudder,
                                      interaction=RD.DEFAULT_INTERACTION,
                                      KT=sp.KT, J=sp.advance_ratio,
                                      conditions=conditions, linear=False)
        straight = model.forces(conditions.u_design, 0.0, 0.0, 0.0)
        negative_control[tag] = {
            "rudder_F_y_N": zero["F_y"], "rudder_M_z_Nm": zero["M_z"],
            "hull_Y_N": straight[1], "hull_N_Nm": straight[2],
            "F_y_over_lateral_force_scale": abs(zero["F_y"]) / max(
                abs(straight[1]) + 1e-12, 1e-12),
        }
        models[tag] = model

    # ------------------------------------------------------ turning circles
    dt_levels = (1.0, 0.5, 0.25)
    turning = {}
    trajectories = {}
    for dt in dt_levels:
        traj = D.simulate_turning_circle(models["analytic"], 35.0, dt=dt,
                                         t_end=400.0)
        metrics = D.turning_circle_metrics(traj)
        turning[f"dt_{dt:g}"] = metrics
        trajectories[f"dt_{dt:g}"] = traj
    from tensorfvm.rans_ship_resistance.speed_sweep import grid_convergence_index
    gci_turning = grid_convergence_index(
        turning["dt_0.25"]["tactical_diameter_over_L"],
        turning["dt_0.5"]["tactical_diameter_over_L"],
        turning["dt_1"]["tactical_diameter_over_L"])

    # rudder-angle sweep with the direct model
    turning_sweep = {}
    for delta in (5.0, 10.0, 20.0, 35.0):
        traj = D.simulate_turning_circle(models["analytic"], delta, dt=0.5,
                                         t_end=500.0)
        turning_sweep[f"delta_{delta:g}"] = D.turning_circle_metrics(traj)

    # ------------------------------------------------------------- zig-zags
    zigzag = {}
    for label, dt in (("dt_0.5", 0.5), ("dt_0.25", 0.25)):
        traj = D.simulate_zigzag(models["analytic"], psi_a_deg=10.0,
                                 delta_deg=10.0, dt=dt, t_end=600.0)
        zigzag[label] = D.zigzag_metrics(traj)

    # ------------------------------------------------- MMG / direct arbitration
    arbitration = {"direct": {}, "mmg": {}, "failures": {}}
    for delta in (10.0, 35.0):
        try:
            mmg_model = D.ManoeuvreModel(conditions=conditions,
                                         self_propulsion=sp, polar=polar,
                                         mode="mmg", rudder_operating_deg=delta)
            mmg_traj = D.simulate_turning_circle(mmg_model, delta, dt=0.25,
                                                 t_end=400.0)
            arbitration["mmg"][f"delta_{delta:g}"] = D.turning_circle_metrics(mmg_traj)
            arbitration["direct"][f"delta_{delta:g}"] = turning_sweep[
                f"delta_{delta:g}"]
        except Exception as exc:                     # noqa: BLE001 - recorded
            arbitration["failures"][f"delta_{delta:g}"] = f"{type(exc).__name__}: {exc}"

    def relative_difference(a, b):
        if a is None or b is None or b == 0:
            return None
        return abs(a - b) / abs(b)

    arbitration_difference = {}
    for key in arbitration["mmg"]:
        mmg_val = arbitration["mmg"][key]["tactical_diameter_over_L"]
        direct_val = arbitration["direct"][key]["tactical_diameter_over_L"]
        arbitration_difference[key] = {
            "T_D_mmg_over_L": mmg_val, "T_D_direct_over_L": direct_val,
            "relative_difference": relative_difference(mmg_val, direct_val)}

    # the CFD-vs-analytic rudder-derivative substitution effect
    derivative_substitution = {
        "analytic": rudder_derivatives_analytic,
        "loaded": rudder_derivatives,
        "relative_difference_F_y_delta": relative_difference(
            rudder_derivatives["F_y_delta_prime_per_rad"],
            rudder_derivatives_analytic["F_y_delta_prime_per_rad"]),
    }

    # --------------------------------------------------- hull derivative set
    hull_zero = HF.linear_derivatives(models["analytic"].hull_model)
    hull_operating = HF.linear_derivatives(models["analytic"].hull_model,
                                           *D.STEADY_TURN_REFERENCE)
    added = models["analytic"].masses

    # -------------------------------------------------------------- criteria
    band_td = REF.KVLCC2_TURNING_BAND["tactical_diameter_over_L"]
    band_adv = REF.KVLCC2_TURNING_BAND["advance_over_L"]
    band_psi1 = REF.KVLCC2_ZIGZAG_BAND["psi_1_deg"]
    td_direct = turning["dt_0.25"]["tactical_diameter_over_L"]
    td_steady = turning["dt_0.25"]["tactical_diameter_2R_over_L"]
    criteria = {
        "a_self_propulsion_closure": {
            "value": abs(sp.residual), "limit": 0.02,
            "unit": "relative", "passed": bool(abs(sp.residual) <= 0.02),
            "note": "closure of the model, not a validation of the resistance level",
        },
        "b_rudder_derivative": {
            "value": rudder_derivatives["F_y_delta_prime_per_rad"],
            "reference": "SIMMAN 2008 EFD", "status": "withheld",
            "passed": None,
            "note": REF.SIMMAN2008_EFD["reason"],
        },
        "c_tactical_diameter": {
            "value_over_L": td_direct, "band": [band_td.low, band_td.high],
            "steady_radius_route_over_L": td_steady,
            "status": "withheld", "passed": None,
            "note": "no provenance-verified EFD table; literature band shown",
        },
        "d_first_overshoot": {
            "value_deg": zigzag["dt_0.25"]["psi_1_deg"],
            "band_deg": [band_psi1.low, band_psi1.high],
            "status": "withheld", "passed": None},
        "e_mmg_arbitration": {
            "results": arbitration_difference, "failures": arbitration["failures"],
            "limit": 0.10,
            "passed": bool(arbitration_difference) and all(
                (item["relative_difference"] is not None
                 and item["relative_difference"] <= 0.10)
                for item in arbitration_difference.values()),
        },
        "f_convergence": {
            "time_step_gci": gci_turning, "limit": 0.05,
            "cfd_mesh": "see artifacts/rudder_section_cfd.json (not converged)",
            "passed": bool(gci_turning.get("gci_fine", 1e9) <= 0.05),
        },
        "g_negative_control": {
            "results": negative_control,
            "passed": bool(all(abs(item["rudder_F_y_N"]) < 1e-12
                               and abs(item["rudder_M_z_Nm"]) < 1e-12
                               and abs(item["hull_Y_N"]) < 1e-12
                               and abs(item["hull_N_Nm"]) < 1e-12
                               for item in negative_control.values())),
        },
    }

    summary = {
        "node": "IN63-N02",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "conditions": {
            "length_m": conditions.length, "beam_m": conditions.beam,
            "draft_m": conditions.draft, "Cb": conditions.block_coefficient,
            "design_speed_m_s": conditions.u_design,
            "froude": conditions.froude, "reynolds": conditions.reynolds,
            "rho": conditions.rho, "nu": conditions.nu,
        },
        "geometry_gate": gate, "planform": planform,
        "rudder_geometry": rudder.as_dict(),
        "resistance": resistance.as_dict(), "resistance_sweep": resistance_sweep,
        "open_water": open_water, "open_water_eta_max": eta_max,
        "self_propulsion": sp.as_dict(), "hull_interaction": interaction,
        "wake_sensitivity": wake_sensitivity,
        "rudder_polar": polar.as_dict(), "rudder_polar_source": polar_source,
        "rudder_derivatives": rudder_derivatives,
        "rudder_derivatives_analytic": rudder_derivatives_analytic,
        "derivative_substitution": derivative_substitution,
        "negative_control": negative_control,
        "turning_circle": turning, "turning_gci": gci_turning,
        "turning_sweep": turning_sweep, "zigzag": zigzag,
        "arbitration": arbitration,
        "arbitration_difference": arbitration_difference,
        "hull_derivatives_zero_drift": hull_zero,
        "hull_derivatives_operating_point": hull_operating,
        "added_masses": added,
        "criteria": criteria,
        "reference_status": {
            "turning": {k: v.as_dict() for k, v in REF.KVLCC2_TURNING_BAND.items()},
            "zigzag": {k: v.as_dict() for k, v in REF.KVLCC2_ZIGZAG_BAND.items()},
            "simmman2008": REF.SIMMAN2008_EFD,
        },
        "wall_seconds": time.time() - started,
    }
    payload = _jsonable(summary)
    (out / "artifacts" / "maneuvering_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"criteria": payload["criteria"],
                      "turning": {k: v for k, v in turning["dt_0.25"].items()
                                  if k != "mode"},
                      "zigzag": zigzag["dt_0.25"],
                      "self_propulsion": {k: payload["self_propulsion"][k]
                                          for k in ("rps", "J", "KT", "eta_open_water",
                                                    "resistance_N", "closure_pct")}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())