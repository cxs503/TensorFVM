"""Run one internal-flow case and return its momentum-ledger record.

The record is a plain JSON-able dict: case description, solver log, the itemised
control-volume momentum budgets, the physical metrics (Cp recovery / reattachment /
Nusselt) and — when requested — a velocity-magnitude contour image.  Every number
of the published report is taken from these records; nothing is transcribed by
hand.
"""

from __future__ import annotations

import os

import torch

from . import cases, ledger, ledger_reference, metrics
from .cases import InternalCase


def _cell_velocity(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])


def _ledger_block(flow, case: InternalCase, box, component: str = "x",
                  source: torch.Tensor | None = None,
                  open_faces: tuple[bool, bool] = (True, True)) -> dict:
    led = ledger.momentum_ledger(flow.u, flow.v, flow.p, case.fluid, case.dx, case.dy,
                                 case.viscosity, component=component, box=box,
                                 source=source, open_faces=open_faces)
    return led.as_dict()


def _save_contour(flow, case: InternalCase, path: str, title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uc, vc = _cell_velocity(flow.u, flow.v)
    speed = torch.sqrt(uc ** 2 + vc ** 2)
    ny, nx = case.shape
    x = (torch.arange(nx) + 0.5) * case.dx
    y = (torch.arange(ny) + 0.5) * case.dy
    field = torch.where(case.fluid, speed, torch.full_like(speed, float("nan")))
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    cs = ax.contourf(x.numpy(), y.numpy(), field.numpy(), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="|U|")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _duct_metrics(flow, case: InternalCase, ny: int) -> dict:
    """Cp recovery, reattachment, ledgers and mass balance of a duct case."""
    st = case.stations
    rho = 1.0
    h_in = case.geometry["h_in"]
    q_in = ledger.station_flux(flow.u, case.fluid, st["inlet_face"], case.dy)
    height_in = ledger.station_height(case.fluid, 0, case.dy)
    u_in = q_in / height_in
    p_in = ledger.station_pressure(flow.p, case.fluid, st["inlet_face"])
    q_exit = ledger.station_flux(flow.u, case.fluid, st["exit_face"], case.dy)
    height_exit = ledger.station_height(case.fluid, st["exit_face"] - 1, case.dy)
    p_exit = ledger.station_pressure(flow.p, case.fluid, st["exit_face"] - 1)
    momentum_scale = rho * u_in ** 2 * height_in

    boxes = {
        "full": (0, ny, st["inlet_face"], st["exit_face"]),
        "redeveloping": (0, ny, st["step_face"], st["exit_face"]),
    }
    rec_face = st.get("recovery_face")
    if rec_face:
        boxes["recovery"] = (0, ny, st["step_face"], rec_face)
    ledgers = {name: _ledger_block(flow, case, box) for name, box in boxes.items()}
    for item in ledgers.values():
        item["relative_inlet_momentum"] = abs(item["residual"]) / momentum_scale
    record = {
        "u_in": u_in, "u_out": q_exit / height_exit, "inlet_flux": q_in,
        "outlet_flux": q_exit, "mass_imbalance": abs(q_in - q_exit) / abs(q_in),
        "momentum_scale": momentum_scale, "p_in": p_in, "p_exit": p_exit,
        "ledgers": ledgers,
        "ledger_y_full": _ledger_block(flow, case, boxes["full"], component="y"),
    }

    if case.family == "expansion" and rec_face:
        led = ledgers["recovery"]
        height_rec = ledger.station_height(case.fluid, rec_face - 1, case.dy)
        u_rec = ledger.station_flux(flow.u, case.fluid, rec_face, case.dy) / height_rec
        p_rec = ledger.station_pressure(flow.p, case.fluid, rec_face - 1)
        p_up = ledger.station_pressure(flow.p, case.fluid, st["step_face"] - 1)
        s = height_in / height_rec
        denom = height_rec * 0.5 * rho * u_in ** 2
        conv = led["convective"]
        conv_uniform = rho * u_in ** 2 * height_in - rho * u_rec ** 2 * height_rec
        terms = led["terms"]
        cp_meas = (p_rec - p_up) / (0.5 * rho * u_in ** 2)
        cp_pred = (p_rec + led["residual"] / height_rec
                   - p_up) / (0.5 * rho * u_in ** 2)
        dc_profile = (conv - conv_uniform) / denom
        dc_step = (terms["p_wall"] - p_up * (height_rec - height_in)) / denom
        dc_visc = led["viscous"] / denom
        cp_exit = (p_exit - p_up) / (0.5 * rho * u_in ** 2)
        record["cp"] = {
            "station": "inlet -> 2 h_out downstream of the step",
            "x_recovery_over_h_out": case.geometry.get("x_recovery", float("nan")) ,
            "cp_measured": cp_meas, "cp_ledger_prediction": cp_pred,
            "cp_relative_error": abs(cp_pred - cp_meas) / max(abs(cp_meas), 1e-30),
            "cp_borda_carnot": ledger_reference.borda_carnot(1.0 / s).value,
            "p_upstream_of_step": p_up,
            "dcp_profile": dc_profile, "dcp_step_face": dc_step,
            "dcp_viscous": dc_visc,
            "decomposition_closure": abs(ledger_reference.borda_carnot(1.0 / s).value
                                         + dc_profile + dc_step + dc_visc - cp_meas),
            "area_ratio": 1.0 / s,
            "u_out_over_u_in": u_rec / u_in, "expected_u_ratio": s,
            "cp_at_exit_station": cp_exit,
            "exit_station_friction_share": (cp_exit - cp_meas) / max(abs(cp_meas), 1e-30),
        }

    scale = h_in if case.family == "expansion" else case.geometry["step_height"]
    rea = metrics.reattachment_length(flow.u, case.fluid, case.dx, case.dy,
                                      case.viscosity, start_face=st["step_face"],
                                      scale=scale)
    record["reattachment"] = {
        "shear": rea["shear"], "velocity": rea["velocity"], "scale": scale,
        "profile_x": rea["x"], "profile_tau": rea["tau"], "min_tau": rea["min_tau"],
    }
    return record


def _cavity_metrics(flow, case: InternalCase, ny: int, nx: int) -> dict:
    alpha = case.conductivity
    nu_ = metrics.wall_nusselt(flow.T, case.dx, case.t_hot, case.t_cold)
    psi = metrics.streamfunction(flow.u, flow.v, case.dy, case.fluid)
    ref = ledger_reference.cavity(case.params["rayleigh"])
    uc, vc = _cell_velocity(flow.u, flow.v)
    source = (flow.T - case.t_ref) * case.buoyancy
    full = _ledger_block(flow, case, (0, ny, 0, nx), component="y", source=source,
                         open_faces=(False, False))
    half = _ledger_block(flow, case, case.stations["left_half"], component="y",
                         source=source, open_faces=(False, False))
    driving = float((source.abs() * case.dx * case.dy).sum())
    for item in (full, half):
        item["relative_to_driving"] = abs(item["residual"]) / max(driving, 1e-30)
        item["relative_residual"] = abs(item["residual"]) / max(driving, 1e-30)
    return {
        "nusselt": nu_,
        "psi_max_scaled": float(psi.abs().max()) / alpha,
        "psi_max_reference": ref.psi_max,
        "u_max_scaled": float(uc.abs().max()) / alpha, "u_max_reference": ref.u_max,
        "v_max_scaled": float(vc.abs().max()) / alpha, "v_max_reference": ref.v_max,
        "nusselt_reference": ref.nusselt,
        "nusselt_rel_error": abs(nu_["nusselt_hot"] - ref.nusselt) / ref.nusselt,
        "nusselt_rel_error_order1": abs(nu_["nusselt_hot_order1"] - ref.nusselt)
                                    / ref.nusselt,
        "driving_force_scale": driving,
        "ledger_y_full": full, "ledger_y_left_half": half,
        "mid_height_u": uc[ny // 2, :].tolist(),
        "mid_width_v": vc[:, nx // 2].tolist(),
        "mid_temperature": flow.T[:, nx // 2].tolist(),
    }


def run_case(case: InternalCase, *, tolerance: float = 1e-6,
             max_iterations: int = 60000, warm: dict | None = None,
             figure_dir: str | None = None, fields_out: dict | None = None) -> dict:
    """Solve one case and pack its ledger record."""
    options = cases.solver_options(case.family, tolerance=tolerance,
                                   max_iterations=max_iterations)
    flow = case.solver(options)
    if warm:
        cases.warm_start(flow, u=warm.get("u"), v=warm.get("v"), p=warm.get("p"),
                         T=warm.get("T"))
    log = flow.solve()
    ny, nx = case.shape
    record: dict[str, object] = {
        "case": case.summary(), "solver": dict(log),
        "residuals": flow.residuals(),
        "divergence_scaled": float(flow.divergence_field().abs().max()) * case.dx,
    }
    uc, vc = _cell_velocity(flow.u, flow.v)
    record["u_max"] = float(uc.abs().max())
    record["v_max"] = float(vc.abs().max())
    if case.family in ("expansion", "step"):
        record.update(_duct_metrics(flow, case, ny))
    else:
        record.update(_cavity_metrics(flow, case, ny, nx))
    record["fields"] = {
        "u": flow.u.tolist() if figure_dir else None,
        "v": flow.v.tolist() if figure_dir else None,
    }
    if fields_out is not None:
        fields_out.update({"u": flow.u.clone(), "v": flow.v.clone(),
                           "p": flow.p.clone(), "T": flow.T.clone()})
    if figure_dir:
        os.makedirs(figure_dir, exist_ok=True)
        path = os.path.join(figure_dir, f"{case.name}_{case.level}_{case.outlet_mode}.png")
        record["figure"] = _save_contour(
            flow, case, path,
            f"{case.name} {case.level} outlet={case.outlet_mode} |U|")
        record["fields"] = {}
    return record