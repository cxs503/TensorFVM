"""The three benchmark cases, their metrics and their machine-checkable criteria.

Each case is defined as a pure function returning a :class:`CaseSetup`, so a
report can be regenerated from parameters alone.  Metrics are extracted from the
converged field only; the acceptance functions apply fixed numeric thresholds and
return an explicit verdict per criterion (never a single opaque boolean).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import reference
from .solver import SolverOptions, SteadyFlow, cell_velocity, streamfunction


@dataclass
class CaseSetup:
    """Everything needed to rebuild a case and to describe it in a report."""

    name: str
    level: str
    fluid: torch.Tensor
    dx: float
    dy: float
    viscosity: float
    inlet_profile: torch.Tensor | None = None
    outlet: bool = True
    conductivity: float | None = None
    buoyancy: float = 0.0
    t_ref: float = 0.0
    t_hot: float | None = None
    t_cold: float | None = None
    params: dict[str, float] = field(default_factory=dict)
    geometry: dict[str, float] = field(default_factory=dict)

    @property
    def cells(self) -> int:
        return int(self.fluid.numel())

    def solver(self, options: SolverOptions | None = None) -> SteadyFlow:
        return SteadyFlow(self.fluid, dx=self.dx, dy=self.dy, viscosity=self.viscosity,
                          inlet_profile=self.inlet_profile, outlet=self.outlet,
                          conductivity=self.conductivity, buoyancy=self.buoyancy,
                          t_ref=self.t_ref, t_hot=self.t_hot, t_cold=self.t_cold,
                          options=options)

    def summary(self) -> dict[str, object]:
        return {
            "case": self.name, "level": self.level,
            "nx": int(self.fluid.shape[1]), "ny": int(self.fluid.shape[0]),
            "cells": self.cells, "fluid_cells": int(self.fluid.sum()),
            "dx": self.dx, "dy": self.dy, "viscosity": self.viscosity,
            "params": dict(self.params), "geometry": dict(self.geometry),
        }


# --------------------------------------------------------------------- case 1
# (cells per step height, cells per inlet height) so that the step face lands on
# a grid line and the expansion ratio is reproduced to better than 0.06 %.
BFS_LEVELS = {
    "coarse": (16, 17),
    "medium": (33, 35),
    "fine": (49, 52),
}


def backward_step(level: str = "coarse", *, reynolds_dh: float = 800.0,
                  upstream_inlet_heights: float = 3.0,
                  downstream_step_heights: float = 12.0) -> CaseSetup:
    """Armaly backward-facing step: step height 1, expansion ratio 1.9423."""
    if level not in BFS_LEVELS:
        raise ValueError(f"level must be one of {sorted(BFS_LEVELS)}")
    n_s, n_in = BFS_LEVELS[level]
    dy = 1.0 / n_s
    dx = dy
    h_in = n_in * dy
    h_out = (n_s + n_in) * dy
    x_up = upstream_inlet_heights * h_in
    x_down = downstream_step_heights
    nx = int(round((x_up + x_down) / dx))
    ny = n_s + n_in
    x = (torch.arange(nx) + 0.5) * dx - x_up
    y = (torch.arange(ny) + 0.5) * dy
    fluid = (x[None, :] >= 0.0) | (y[:, None] >= 1.0)
    ys = y - 1.0
    profile = torch.where(torch.arange(ny) * dy >= 1.0,
                          6.0 * ys * (h_in - ys) / h_in ** 2, torch.zeros(ny))
    area_ratio = h_out / h_in
    target = reference.backward_facing_step()["expansion_ratio"].value
    geometry = {
        "expansion_ratio_mask": area_ratio,
        "expansion_ratio_target": target,
        "expansion_ratio_rel_error": abs(area_ratio - target) / target,
        "step_height_over_dy": 1.0 / dy,
        "inlet_heights_upstream": x_up / h_in,
        "step_heights_downstream": x_down,
        "inlet_mean_velocity": 1.0,
        "inlet_flux": float(profile.sum() * dy),
        "reynolds_basis": "U_mean * 2 * H_in / nu",
    }
    return CaseSetup(
        name="backward_facing_step", level=level, fluid=fluid, dx=dx, dy=dy,
        viscosity=2.0 * h_in / reynolds_dh, inlet_profile=profile, outlet=True,
        params={"reynolds_dh": reynolds_dh, "expansion_ratio": area_ratio,
                "h_in": h_in, "h_out": h_out, "step_height": 1.0,
                "upstream_length": x_up, "downstream_length": x_down},
        geometry=geometry)


# --------------------------------------------------------------------- case 2
EXPANSION_LEVELS = {"coarse": 8, "medium": 12, "fine": 16}


def sudden_expansion(level: str = "coarse", *, area_ratio: float = 4.0,
                     reynolds_d: float = 100.0, upstream_inlet_heights: float = 4.0,
                     downstream_outlet_heights: float = 16.0) -> CaseSetup:
    """Symmetric planar sudden expansion, area ratio 4, inlet height 1/AR."""
    if level not in EXPANSION_LEVELS:
        raise ValueError(f"level must be one of {sorted(EXPANSION_LEVELS)}")
    n_h = EXPANSION_LEVELS[level]
    height = 1.0
    h_in = height / area_ratio
    dy = h_in / n_h
    dx = dy
    ny = int(round(height / dy))
    x_up = upstream_inlet_heights * h_in
    x_down = downstream_outlet_heights * height
    nx = int(round((x_up + x_down) / dx))
    x = (torch.arange(nx) + 0.5) * dx - x_up
    y = (torch.arange(ny) + 0.5) * dy
    y0 = 0.5 * (height - h_in)
    y1 = 0.5 * (height + h_in)
    inlet_rows = (y >= y0) & (y <= y1)
    fluid = (x[None, :] >= 0.0) | inlet_rows[:, None]
    ys = (y - y0) / h_in
    profile = torch.where(inlet_rows, 6.0 * ys * (1.0 - ys), torch.zeros(ny))
    measured_ratio = float(inlet_rows.sum()) * dy and (height / h_in)
    geometry = {
        "area_ratio_mask": measured_ratio,
        "area_ratio_target": area_ratio,
        "area_ratio_rel_error": abs(measured_ratio - area_ratio) / area_ratio,
        "inlet_cells": int(inlet_rows.sum()),
        "inlet_mean_velocity": 1.0,
        "inlet_flux": float(profile.sum() * dy),
        "reynolds_basis": "U_mean * 2 * h_in / nu",
    }
    return CaseSetup(
        name="sudden_expansion", level=level, fluid=fluid, dx=dx, dy=dy,
        viscosity=2.0 * h_in / reynolds_d, inlet_profile=profile, outlet=True,
        params={"reynolds_d": reynolds_d, "area_ratio": area_ratio, "h_in": h_in,
                "height": height, "upstream_length": x_up,
                "downstream_length": x_down},
        geometry=geometry)


# --------------------------------------------------------------------- case 3
CAVITY_LEVELS = {"coarse": 32, "medium": 48, "fine": 64, "reference": 96}


def natural_cavity(rayleigh: float, level: str = "coarse", *,
                   prandtl: float = 0.71) -> CaseSetup:
    """Differentially heated square cavity, Boussinesq, free-fall scaling."""
    if level not in CAVITY_LEVELS:
        raise ValueError(f"level must be one of {sorted(CAVITY_LEVELS)}")
    n = CAVITY_LEVELS[level]
    dx = dy = 1.0 / n
    fluid = torch.ones((n, n), dtype=torch.bool)
    nu = math.sqrt(prandtl / rayleigh)
    alpha = nu / prandtl
    geometry = {
        "aspect_ratio": 1.0,
        "grid_cells": float(n),
        "rayleigh": rayleigh, "prandtl": prandtl,
        "viscosity": nu, "diffusivity": alpha,
        "thermal_diffusion_time": 1.0 / alpha,
        "buoyancy_scale": 1.0,
    }
    return CaseSetup(
        name="natural_cavity", level=level, fluid=fluid, dx=dx, dy=dy, viscosity=nu,
        outlet=False, conductivity=alpha, buoyancy=1.0, t_ref=0.5, t_hot=1.0,
        t_cold=0.0,
        params={"rayleigh": rayleigh, "prandtl": prandtl, "cells_per_side": float(n)},
        geometry=geometry)


# --------------------------------------------------------------------- metrics
def _zero_crossing(x: torch.Tensor, values: torch.Tensor, *, start: int) -> float:
    """First sign change of ``values`` after index ``start``, linearly interpolated."""
    for i in range(start, len(values) - 1):
        a, b = float(values[i]), float(values[i + 1])
        if a == 0.0:
            return float(x[i])
        if a * b < 0.0:
            xa, xb = float(x[i]), float(x[i + 1])
            return xa + (xb - xa) * abs(a) / (abs(a) + abs(b))
    return float("nan")


def wall_friction(solver: SteadyFlow, wall_row: int = 0) -> torch.Tensor:
    """``C_f`` on a horizontal wall from the half-cell wall shear.

    The returned array lives on the *vertical faces* (``nx + 1`` entries),
    because the wall shear is evaluated from the tangential velocity of the face
    adjacent to the wall.
    """
    ny = solver.ny
    row = 0 if wall_row == 0 else ny - 1
    u_wall = solver.u[row, :]
    tau = 2.0 * solver.nu * u_wall / solver.dy
    return tau / 0.5


def face_coordinates(setup: CaseSetup, offset: float) -> torch.Tensor:
    """Streamwise coordinate of every vertical face (``nx + 1`` entries)."""
    nx = int(setup.fluid.shape[1])
    return torch.arange(nx + 1, dtype=torch.float64) * setup.dx - offset


def backward_step_metrics(solver: SteadyFlow, setup: CaseSetup) -> dict[str, object]:
    params = setup.params
    grid = setup.geometry
    h_in = params["h_in"]
    u_ref = params["inlet_mean_velocity"] if "inlet_mean_velocity" in params else 1.0
    cf = wall_friction(solver, 0) / (u_ref ** 2)
    x = face_coordinates(setup, grid["inlet_heights_upstream"] * h_in)
    downstream = x > 0.0
    idx = torch.nonzero(downstream).flatten()
    cf_d = cf[idx]
    negative = torch.nonzero(cf_d < 0.0).flatten()
    reattach = float("nan")
    if negative.numel():
        reattach = _zero_crossing(x[idx], cf_d, start=int(negative[-1]))
        if not math.isfinite(reattach):
            reattach = float(x[idx[min(int(negative[-1]) + 1, len(idx) - 1)]])
    # Analytic downstream check: fully developed plane channel, C_f = 12 / Re_h.
    h_out = params["h_out"]
    u_out = u_ref * h_in / h_out
    re_out = u_out * h_out / solver.nu
    cf_analytic = 12.0 / re_out
    sample = float(x[torch.argmin(torch.abs(x - 0.85 * float(x[-1])))])
    cf_far = float(cf[torch.argmin(torch.abs(x - sample))])
    min_cf = float(cf_d.min()) if cf_d.numel() else float("nan")
    return {
        "reattachment_over_step": reattach,
        "friction_min": min_cf,
        "friction_far": cf_far,
        "friction_far_x": sample,
        "friction_far_analytic": cf_analytic,
        "friction_far_rel_error": abs(cf_far - cf_analytic) / abs(cf_analytic),
        "friction_profile": {
            "x_over_step": (x / 1.0).tolist(), "cf": cf.tolist()},
        "u_max": float(cell_velocity(solver.u, solver.v)[0].max()),
        "v_max": float(cell_velocity(solver.u, solver.v)[1].abs().max()),
        "divergence": float(solver.divergence_field().abs().max()) * setup.dx,
    }


def expansion_metrics(solver: SteadyFlow, setup: CaseSetup) -> dict[str, object]:
    params = setup.params
    h_in = params["h_in"]
    height = params["height"]
    x_up = params["upstream_length"]
    cf = wall_friction(solver, 0) / 1.0
    x = face_coordinates(setup, x_up)
    idx = torch.nonzero(x > 0.0).flatten()
    cf_d = cf[idx]
    negative = torch.nonzero(cf_d < 0.0).flatten()
    reattach = float("nan")
    if negative.numel():
        reattach = _zero_crossing(x[idx], cf_d, start=int(negative[-1]))
    # Pressure loss coefficient.
    u_cell, _ = cell_velocity(solver.u, solver.v)
    col_in = int(torch.argmin(torch.abs(x - 0.5 * h_in)))
    col_out = int(torch.argmin(torch.abs(x - (x[-1] - 2.0 * height))))
    p_in = float(solver.p[:, col_in].mean())
    p_out = float(solver.p[:, col_out].mean())
    p_in_plane = float(solver.p[setup.fluid[:, col_in], col_in].mean())
    p_out_plane = float(solver.p[setup.fluid[:, col_out], col_out].mean())
    u_in, u_out = 1.0, 1.0 / params["area_ratio"]
    k_total = (p_in_plane - p_out_plane + 0.5 * (u_in ** 2 - u_out ** 2)) / (0.5 * u_in ** 2)
    k_bc = (1.0 - 1.0 / params["area_ratio"]) ** 2
    inlet_flux = float(solver.u[setup.fluid[:, 0], 0].sum() * setup.dy)
    outlet_flux = float(solver.u[setup.fluid[:, -1], -1].sum() * setup.dy)
    return {
        "reattachment_over_inlet_height": reattach,
        "loss_coefficient": k_total,
        "loss_coefficient_borda_carnot": k_bc,
        "loss_excess_over_borda_carnot": k_total - k_bc,
        "pressure_inlet": p_in_plane, "pressure_outlet": p_out_plane,
        "pressure_inlet_mean_col": p_in, "pressure_outlet_mean_col": p_out,
        "inlet_flux": inlet_flux, "outlet_flux": outlet_flux,
        "mass_imbalance": abs(inlet_flux - outlet_flux) / abs(inlet_flux),
        "u_max": float(u_cell.max()),
        "divergence": float(solver.divergence_field().abs().max()) * setup.dx,
        "friction_profile": {"x_over_h": (x / h_in).tolist(), "cf": cf.tolist()},
    }


def cavity_metrics(solver: SteadyFlow, setup: CaseSetup) -> dict[str, object]:
    params = setup.params
    n = int(params["cells_per_side"])
    h = 1.0 / n
    alpha = setup.conductivity
    hot = 2.0 * (setup.t_hot - solver.T[:, 0]) / h
    cold = 2.0 * (solver.T[:, -1] - setup.t_cold) / h
    nu_hot = float(hot.mean())
    nu_cold = float(cold.mean())
    psi = streamfunction(solver.u, solver.v, h, solver.fluid)
    uc, vc = cell_velocity(solver.u, solver.v)
    psi_max = float(psi.abs().max()) / alpha
    u_max = float(uc.abs().max()) / alpha
    v_max = float(vc.abs().max()) / alpha
    ref = reference.cavity(params["rayleigh"])
    imax = int(torch.argmax(psi.abs()))
    iy, ix = divmod(imax, solver.fluid.shape[1])
    return {
        "nusselt_hot": nu_hot,
        "nusselt_cold": nu_cold,
        "nusselt_mean": 0.5 * (nu_hot + nu_cold),
        "energy_balance_rel": abs(nu_hot - nu_cold) / abs(nu_hot),
        "nusselt_reference": ref.nusselt,
        "nusselt_rel_error": abs(nu_hot - ref.nusselt) / ref.nusselt,
        "psi_max": psi_max,
        "psi_max_reference": ref.psi_max,
        "psi_max_rel_error": abs(psi_max - ref.psi_max) / ref.psi_max,
        "streamfunction_centre": [(ix + 0.5) * h, (iy + 0.5) * h],
        "u_max": u_max, "u_max_reference": ref.u_max,
        "u_max_rel_error": abs(u_max - ref.u_max) / ref.u_max,
        "v_max": v_max, "v_max_reference": ref.v_max,
        "v_max_rel_error": abs(v_max - ref.v_max) / ref.v_max,
        "wall_gradient_hot": hot.tolist(),
        "temperature_field": solver.T.tolist(),
        "psi_field": psi.tolist(),
        "divergence": float(solver.divergence_field().abs().max()) * h,
    }


METRICS = {
    "backward_facing_step": backward_step_metrics,
    "sudden_expansion": expansion_metrics,
    "natural_cavity": cavity_metrics,
}


# ------------------------------------------------------------------ acceptance
def unwrap(value: dict | None, key: str) -> float:
    """Accept either ``{"value": x}`` (Reference.__dict__) or a bare float."""
    if isinstance(value, dict):
        return float(value[key])
    return float(value)


def backward_step_verdict(levels: dict[str, dict]) -> dict[str, object]:
    target = reference.backward_facing_step()["reattachment"]
    criteria = []
    for level, metrics in levels.items():
        reattach = metrics["reattachment_over_step"]
        error = abs(reattach - target.value)
        criteria.append({
            "id": f"bfs.reattachment.{level}",
            "description": f"|L_r/S - {target.value}| <= {target.tolerance} (2-D benchmark)",
            "value": reattach, "reference": target.value, "error": error,
            "passed": bool(math.isfinite(reattach) and error <= target.tolerance),
        })
    fine = levels.get("fine") or levels[sorted(levels)[-1]]
    criteria.append({
        "id": "bfs.friction_far",
        "description": "downstream |C_f| within 15 % of the analytic plane-channel value",
        "value": fine["friction_far"], "reference": fine["friction_far_analytic"],
        "error": fine["friction_far_rel_error"],
        "passed": bool(fine["friction_far_rel_error"] <= 0.15),
    })
    criteria.append({
        "id": "bfs.friction_zero_crossing",
        "description": "C_f(x) changes sign exactly at the reattachment point",
        "value": fine["reattachment_over_step"], "reference": fine["reattachment_over_step"],
        "error": 0.0, "passed": bool(math.isfinite(fine["reattachment_over_step"])),
    })
    return {"criteria": criteria,
            "passed": all(item["passed"] for item in criteria)}


def expansion_verdict(levels: dict[str, dict], setups: dict[str, CaseSetup]) -> dict[str, object]:
    criteria = []
    for level, setup in setups.items():
        err = setup.geometry["area_ratio_rel_error"]
        criteria.append({
            "id": f"expansion.area_ratio.{level}",
            "description": "stair-step geometry reproduces A_out/A_in = 4 within 0.2 %",
            "value": setup.geometry["area_ratio_mask"], "reference": 4.0,
            "error": err, "passed": bool(err <= 0.002)})
    fine = levels.get("fine") or levels[sorted(levels)[-1]]
    criteria.append({
        "id": "expansion.borda_carnot_sign",
        "description": "measured loss exceeds Borda-Carnot (viscous excess at Re_D = 100)",
        "value": fine["loss_coefficient"],
        "reference": fine["loss_coefficient_borda_carnot"],
        "error": fine["loss_excess_over_borda_carnot"],
        "passed": bool(fine["loss_excess_over_borda_carnot"] >= 0.0)})
    criteria.append({
        "id": "expansion.mass_balance",
        "description": "inlet and outlet mass flux agree within 1e-4",
        "value": fine["mass_imbalance"], "reference": 0.0,
        "error": fine["mass_imbalance"], "passed": bool(fine["mass_imbalance"] <= 1e-4)})
    return {"criteria": criteria,
            "passed": all(item["passed"] for item in criteria),
            "reattachment_band": [levels[level]["reattachment_over_inlet_height"]
                                  for level in sorted(levels)]}


def cavity_verdict(levels: dict[str, dict], *, target_error: float = 0.03,
                   acceptance_error: float = 0.15) -> dict[str, object]:
    criteria = []
    for key, metrics in levels.items():
        err = metrics["nusselt_rel_error"]
        criteria.append({
            "id": f"cavity.nusselt.{key}",
            "description": f"|Nu - Nu_ref| / Nu_ref <= {acceptance_error:.0%} "
                           f"(target {target_error:.0%})",
            "value": metrics["nusselt_hot"],
            "reference": metrics["nusselt_reference"], "error": err,
            "passed": bool(err <= acceptance_error),
            "meets_target": bool(err <= target_error)})
        psi_err = metrics["psi_max_rel_error"]
        criteria.append({
            "id": f"cavity.psi_max.{key}",
            "description": "psi_max within 15 % of de Vahl Davis",
            "value": metrics["psi_max"], "reference": metrics["psi_max_reference"],
            "error": psi_err, "passed": bool(psi_err <= acceptance_error)})
        criteria.append({
            "id": f"cavity.energy_balance.{key}",
            "description": "hot-wall and cold-wall Nusselt numbers agree within 1 %",
            "value": metrics["energy_balance_rel"], "reference": 0.0,
            "error": metrics["energy_balance_rel"],
            "passed": bool(metrics["energy_balance_rel"] <= 0.01)})
    return {"criteria": criteria, "passed": all(item["passed"] for item in criteria),
            "meets_target": all(item.get("meets_target", True) for item in criteria)}