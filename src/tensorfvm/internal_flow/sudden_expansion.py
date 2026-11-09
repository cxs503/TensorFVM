"""Planar sudden expansion, area ratio 4, at ``Re_D = 100``.

Geometry
--------
A plane channel of downstream height ``D = 1`` with a symmetric sudden expansion
from an inlet height ``h_1 = D / AR`` (``AR = 4``): the inlet plane ``x = 0`` is
open over ``y in [0.375, 0.625]`` with a uniform velocity ``U_in = 1`` and is a
solid wall elsewhere (the two step faces).  Both walls are no-slip, the outlet is
zero-gradient.  The upstream plenum is not resolved: a uniform inlet is the
standard idealisation of a sudden expansion, and it makes the Borda-Carnot
momentum control volume (step-face pressure equal to the inlet pressure) a fair
comparison.

Reynolds number
---------------
``Re_D = U_out D / nu = 100`` with ``U_out = U_in / AR`` the developed
downstream mean velocity, i.e. ``nu = U_out D / 100``.  Both velocities are
reported so the value can be re-referenced to an upstream-based Reynolds number
(``Re_h1 = U_in h_1 / nu = Re_D / AR``).

Acceptance quantities
---------------------
* total-pressure loss coefficient ``K = (<p_t>_in - <p_t>_out) / (0.5 rho U_in^2)``
  against the Borda-Carnot value ``(1 - 1/AR)^2 = 0.5625``.  Two variants are
  reported: ``K_raw`` evaluated at the outlet plane (it therefore contains the
  wall friction of the developed part of the channel) and ``K_extrap``, obtained
  by fitting the linear total-pressure decay of the developed region and
  extrapolating it back to the step plane, which removes that friction and is
  the quantity Borda-Carnot refers to.
* static-pressure recovery coefficient ``Cp = (p_out - p_in)/(0.5 rho U_in^2)``
  against the momentum-balance value ``2 (1/AR)(1 - 1/AR) = 0.375``.
* ``k_from_static_recovery``: the same loss coefficient obtained from the exact
  control-volume identity ``K = 1 - 1/AR^2 - Cp``.  The two routes are *not*
  interchangeable: the total-pressure profiles are ``|u|``-weighted column means,
  so ``K`` from them carries the ``<u^3>/<u>`` bias of a non-uniform profile,
  while ``Cp`` is built from the static pressure only.  Both numbers are
  reported; ``docs/verification-expansion-symmetry/REPORT.md`` documents the
  20 % spread between them at ``D/20``.
* reattachment length ``L_r / D`` measured from the step plane on both walls by
  the zero crossing of the wall shear stress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .mac import OPEN_RIGHT, MACConfig, StaggeredMAC
from .reference import SUDDEN_EXPANSION_AREA_RATIO, borda_carnot_loss


@dataclass(frozen=True)
class SuddenExpansionConfig:
    """Mesh and solver controls of the sudden-expansion case."""

    area_ratio: float = SUDDEN_EXPANSION_AREA_RATIO
    reynolds: float = 100.0
    cells_per_downstream_height: int = 40
    length_in_downstream_heights: float = 12.0
    cfl: float = 0.4
    time_step: float | None = None
    max_steps: int = 400_000
    tolerance: float = 1e-9
    check_every: int = 200
    device: str = "cpu"
    #: downstream closure, see :data:`tensorfvm.internal_flow.mac.OPEN_RIGHT`
    right: str = "outlet"

    def __post_init__(self) -> None:
        if not math.isfinite(self.area_ratio) or self.area_ratio <= 1.0:
            raise ValueError("area_ratio must be finite and > 1")
        for name in ("reynolds", "length_in_downstream_heights", "cfl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cells_per_downstream_height < 4:
            raise ValueError("cells_per_downstream_height must be >= 4")
        if self.right not in OPEN_RIGHT:
            raise ValueError("right must be one of the open outlet closures")

    @property
    def downstream_height(self) -> float:
        return 1.0

    @property
    def inlet_height(self) -> float:
        return self.downstream_height / self.area_ratio

    @property
    def inlet_velocity(self) -> float:
        return 1.0

    @property
    def outlet_velocity(self) -> float:
        return self.inlet_velocity / self.area_ratio

    @property
    def viscosity(self) -> float:
        return self.outlet_velocity * self.downstream_height / self.reynolds

    @property
    def mesh(self) -> tuple[int, int]:
        n = self.cells_per_downstream_height
        return int(round(self.length_in_downstream_heights * n)), n


def inlet_profile(config: SuddenExpansionConfig) -> tuple[float, ...]:
    """Uniform inlet velocity over the centred opening, zero over the step faces.

    The opening of height ``H_in = H_out / AR`` is anchored at the inlet plane
    and generally does not coincide with a cell face (``H_in = 0.25`` on a
    ``H_out / 20`` mesh straddles the faces).  Assigning a whole cell to the
    opening -- the obvious ``|y - centre| < H_in/2`` test -- then makes the
    *discrete* area ratio mesh dependent: 4 open cells out of 20 is ``AR = 5``
    and ``Re = 80`` while 10 out of 40 is ``AR = 4``, ``Re = 100``, so the
    nominal ``AR = 4``, ``Re_D = 100`` case was silently solved at two
    different geometries on two meshes and could not converge under refinement.
    The profile below weights each inlet cell by the fraction of it that lies
    inside the opening, which keeps the inlet flux, area ratio and Reynolds
    number exact on every mesh while leaving the geometry symmetric.
    """
    n = config.cells_per_downstream_height
    h = config.downstream_height / n
    half_open = 0.5 * config.inlet_height
    centre = 0.5 * config.downstream_height
    lower, upper = centre - half_open, centre + half_open
    values = []
    for j in range(n):
        y0, y1 = j * h, (j + 1) * h
        overlap = min(y1, upper) - max(y0, lower)
        fraction = min(1.0, max(0.0, overlap / h))
        values.append(config.inlet_velocity * fraction)
    return tuple(values)


class SuddenExpansionCase:
    """Symmetric planar sudden expansion built on :class:`StaggeredMAC`."""

    def __init__(self, config: SuddenExpansionConfig | None = None) -> None:
        self.config = config or SuddenExpansionConfig()
        nx, ny = self.config.mesh
        self.solver = StaggeredMAC(MACConfig(
            nx=nx, ny=ny,
            length=self.config.length_in_downstream_heights * self.config.downstream_height,
            height=self.config.downstream_height,
            viscosity=self.config.viscosity,
            reference_velocity=self.config.inlet_velocity,
            cfl=self.config.cfl,
            time_step=self.config.time_step,
            max_steps=self.config.max_steps,
            tolerance=self.config.tolerance,
            check_every=self.config.check_every,
            left="inlet",
            right=self.config.right,
            inlet_profile=inlet_profile(self.config),
            device=self.config.device,
        ))

    # ------------------------------------------------------------------ driving
    def run(self, **kwargs) -> dict[str, object]:
        return self.solver.run(**kwargs)

    # -------------------------------------------------------------- diagnostics
    def wall_friction(self, wall: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, C_f)`` on the bottom or top wall, ``C_f`` signed by the near-wall u."""
        solver = self.solver
        x = torch.arange(solver.nx + 1, dtype=solver.dtype) * solver.hx
        row = 0 if wall == "bottom" else solver.ny - 1
        tau = solver.rho * solver.nu * 2.0 * solver.u[:, row] / solver.hy
        # both walls use the same streamwise-friction convention: C_f < 0 where the
        # near-wall flow runs back towards the step, so the zero crossing locates
        # the reattachment point on either wall without a sign flip.
        return x, tau / (0.5 * solver.rho * self.config.inlet_velocity ** 2)

    def reattachment(self, wall: str) -> float | None:
        """Distance from the step plane to the wall-shear zero crossing, in ``D``."""
        _, cf = self.wall_friction(wall)
        for i in range(1, self.solver.nx):
            if cf[i - 1] < 0.0 <= cf[i]:
                x0, x1 = (i - 1) * self.solver.hx, i * self.solver.hx
                w = -cf[i - 1] / (cf[i] - cf[i - 1])
                return float(x0 + w * (x1 - x0))
        return None

    def _column_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (weights * values).sum(dim=1) / weights.sum(dim=1)

    def pressure_profiles(self) -> dict[str, torch.Tensor]:
        """Flux-weighted column averages of static and total pressure and of ``u``."""
        solver = self.solver
        u, v = solver.cell_velocity()
        speed2 = u ** 2 + v ** 2
        weight = u.abs() + 1e-30
        x = (torch.arange(solver.nx, dtype=solver.dtype) + 0.5) * solver.hx
        static = self._column_mean(solver.p, weight)
        total = self._column_mean(solver.p + 0.5 * solver.rho * speed2, weight)
        velocity = self._column_mean(u, weight)
        return {"x": x, "static": static, "total": total, "u": velocity}

    def wall_friction_loss(self) -> float:
        """Total-pressure loss caused by wall shear between the inlet and outlet.

        Exact momentum balance, no developed-profile assumption: the friction
        force on the two walls, ``2 * integral(tau_w) dx``, divided by the
        channel area per unit width ``h2`` is the pressure drop that the measured
        total-pressure difference already contains and that Borda-Carnot
        neglects, so subtracting it isolates the expansion loss.
        """
        solver = self.solver
        total = 0.0
        for wall in ("bottom", "top"):
            _, cf = self.wall_friction(wall)
            tau = 0.5 * solver.rho * self.config.inlet_velocity ** 2 * cf
            total += 2.0 * float(torch.trapz(tau, dx=float(solver.hx)))
        drop = total / self.config.downstream_height
        return drop / (0.5 * solver.rho * self.config.inlet_velocity ** 2)

    def _linear_extrapolation(self, x: torch.Tensor, f: torch.Tensor,
                              window: tuple[float, float]) -> tuple[float, float]:
        """Least-squares line through ``f(x)`` inside ``window``; return (f(0), slope)."""
        mask = (x >= window[0]) & (x <= window[1])
        if int(mask.sum()) < 3:
            raise ValueError("the extrapolation window holds fewer than three samples")
        xs = x[mask]
        fs = f[mask]
        matrix = torch.stack([torch.ones_like(xs), xs], dim=1)
        solution = torch.linalg.lstsq(matrix, fs[:, None]).solution[:, 0]
        return float(solution[0]), float(solution[1])

    def diagnostics(self) -> dict[str, object]:
        cfg = self.config
        solver = self.solver
        profiles = self.pressure_profiles()
        reference = self.config.inlet_velocity
        dynamic = 0.5 * solver.rho * reference ** 2
        length = cfg.length_in_downstream_heights * cfg.downstream_height
        window = (0.55 * length, length)
        pt0, pt_slope = self._linear_extrapolation(profiles["x"], profiles["total"], window)
        p0, p_slope = self._linear_extrapolation(profiles["x"], profiles["static"], window)
        pt_in = solver.flux_weighted_total_pressure(0)
        pt_out = solver.flux_weighted_total_pressure(solver.nx)
        p_in = float(profiles["static"][0])
        p_out = float(profiles["static"][-1])
        bottom = self.reattachment("bottom")
        top = self.reattachment("top")
        k_extrap = (pt_in - pt0) / dynamic
        k_raw = (pt_in - pt_out) / dynamic
        k_friction = self.wall_friction_loss()
        cp_extrap = (p0 - p_in) / dynamic
        return {
            "mesh": {"nx": solver.nx, "ny": solver.ny,
                     "cells_per_downstream_height": cfg.cells_per_downstream_height,
                     "h_over_D": float(solver.hx)},
            "area_ratio": cfg.area_ratio,
            "reynolds": cfg.reynolds,
            "reynolds_upstream": cfg.reynolds / cfg.area_ratio ** 2,
            "viscosity": cfg.viscosity,
            "inlet_velocity": cfg.inlet_velocity,
            "outlet_velocity": cfg.outlet_velocity,
            "steps": solver.steps,
            "dt": solver.dt,
            "residual": solver.residual,
            "max_divergence": float(solver.divergence().abs().max()),
            "reattachment_bottom": bottom,
            "reattachment_top": top,
            "reattachment_mean_over_D": None if (bottom is None or top is None)
            else 0.5 * (bottom + top),
            "k_borda_carnot": borda_carnot_loss(cfg.area_ratio),
            "k_raw": k_raw,
            "k_extrapolated": k_extrap,
            "k_friction_correction": k_friction,
            "k_corrected": k_raw - k_friction,
            # Independent route to the loss coefficient: the exact control-volume
            # identity ``K = 1 - 1/AR**2 - Cp`` (p_t loss + static recovery =
            # change of dynamic head).  It is evaluated on the *static* pressure,
            # which is a primitive variable, and therefore does not depend on the
            # ``u``-weighted averaging used by ``pressure_profiles``.
            "k_from_static_recovery_raw": 1.0 - 1.0 / cfg.area_ratio ** 2
            - (p_out - p_in) / dynamic,
            "k_from_static_recovery": 1.0 - 1.0 / cfg.area_ratio ** 2 - cp_extrap,
            "cp_recovery_raw": (p_out - p_in) / dynamic,
            "cp_recovery_extrapolated": cp_extrap,
            "cp_recovery_momentum_balance": 2.0 * (1.0 / cfg.area_ratio) * (1.0 - 1.0 / cfg.area_ratio),
            "total_pressure_slope_per_D": pt_slope * cfg.downstream_height,
            "static_pressure_slope_per_D": p_slope * cfg.downstream_height,
            "extrapolation_window": window,
            "x": profiles["x"].tolist(),
            "static": profiles["static"].tolist(),
            "total": profiles["total"].tolist(),
            "u_column": profiles["u"].tolist(),
        }