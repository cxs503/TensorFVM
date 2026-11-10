"""Rayleigh--Benard convection: nonlinear Boussinesq layer between two plates.

Physics
-------
A ``Gamma x 1`` closed box, plates at ``y = 0`` (hot, ``theta = +dT/2``) and
``y = 1`` (cold, ``-dT/2``), lateral walls no-slip and adiabatic.  The
non-dimensionalisation uses ``H = 1``, ``dT`` and the free-fall velocity
``U = sqrt(g beta dT H)``, so that

    du/dt + div(u u) = -grad p + sqrt(Pr/Ra) lap(u) + theta e_y,
    dtheta/dt + div(u theta) = lap(theta) / sqrt(Pr Ra),
    div(u) = 0,

reproduces ``Ra = g beta dT H^3 / (nu alpha)`` and ``Pr = nu / alpha`` exactly;
:meth:`RayleighBenardCase.rayleigh_check` recomputes both identities from the
discrete inputs so a scaling slip cannot pass unnoticed.  ``theta`` is measured
from the mid-plane temperature, in units of ``dT``.

Implementation
--------------
The momentum/energy discretisation is the audited staggered MAC solver of
:class:`tensorfvm.internal_flow.mac.StaggeredMAC` (van Leer limited conservative
fluxes, DCT-II projection, explicit marching), reused through its
``thermal_walls="plates"`` option rather than re-implemented: the only new
physics in this module is the plate configuration plus the diagnostics below.
The de Vahl Davis cavity of :mod:`tensorfvm.internal_flow.natural_convection`
uses the same class with ``thermal_walls="sides"`` and is unaffected.

Diagnostics
-----------
``Nu`` is the heat transport normalised by pure conduction:

* :meth:`nusselt` -- plate values from a one-sided wall-normal derivative
  (half-cell difference, or the quadratic fit through the wall value and the
  first two cell centres, which is second-order accurate),
* :meth:`nusselt_budget` -- both plates from the *half-cell* difference, i.e.
  the same stencil the discrete Laplacian uses, so a converged solution
  satisfies ``Nu_bottom = Nu_top`` to the level of the marching tolerance.  This
  is the energy-conservation residual used by the campaign gate; the two-plate
  agreement is measured independently for each plate (it is *not* imposed by
  symmetry), and :meth:`flux_profile` shows where in the layer the budget holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..internal_flow.mac import MACConfig, StaggeredMAC

__all__ = ["RayleighBenardConfig", "RayleighBenardCase"]


@dataclass(frozen=True)
class RayleighBenardConfig:
    """Mesh, parametrisation and solver controls of the RB layer."""

    cells: int = 64                 # vertical cells; horizontal = aspect * cells
    aspect: int = 2                 # Lx in units of the layer height
    rayleigh: float = 1.0e5
    prandtl: float = 0.71
    cfl: float = 0.4
    wall_gradient: str = "second_order"
    time_step: float | None = None
    max_steps: int = 4_000_000
    tolerance: float = 1e-10
    check_every: int = 100
    device: str = "cpu"
    temperature_drop: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.cells, bool) or not isinstance(self.cells, int) or self.cells < 8:
            raise ValueError("cells must be an integer >= 8")
        if isinstance(self.aspect, bool) or not isinstance(self.aspect, int) or self.aspect < 1:
            raise ValueError("aspect must be a positive integer")
        if self.wall_gradient not in ("half_cell", "second_order"):
            raise ValueError("wall_gradient must be 'half_cell' or 'second_order'")
        for name in ("rayleigh", "prandtl", "cfl", "temperature_drop"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def viscosity(self) -> float:
        """``sqrt(Pr/Ra)``: the kinematic viscosity of the free-fall scaling."""
        return math.sqrt(self.prandtl / self.rayleigh)

    @property
    def diffusivity(self) -> float:
        """``1/sqrt(Pr Ra)``: the thermal diffusivity of the free-fall scaling."""
        return 1.0 / math.sqrt(self.prandtl * self.rayleigh)


class RayleighBenardCase:
    """Buoyant layer between a hot lower and a cold upper plate."""

    def __init__(self, config: RayleighBenardConfig | None = None) -> None:
        self.config = config or RayleighBenardConfig()
        cfg = self.config
        half = 0.5 * cfg.temperature_drop
        self.solver = StaggeredMAC(MACConfig(
            nx=cfg.aspect * cfg.cells,
            ny=cfg.cells,
            length=float(cfg.aspect),
            height=1.0,
            viscosity=cfg.viscosity,
            reference_velocity=1.0,
            cfl=cfg.cfl,
            time_step=cfg.time_step,
            max_steps=cfg.max_steps,
            tolerance=cfg.tolerance,
            check_every=cfg.check_every,
            left="wall", right="wall",
            energy=True,
            diffusivity=cfg.diffusivity,
            buoyancy=1.0,
            thermal_walls="plates",
            temperature_bottom=half,
            temperature_top=-half,
            device=cfg.device,
        ))

    # ------------------------------------------------------------- parameters
    def rayleigh_check(self) -> float:
        """``g beta dT H^3/(nu alpha)`` recomputed from the discrete inputs."""
        cfg, solver = self.config, self.solver
        height = solver.hy * solver.ny
        return (solver.config.buoyancy * cfg.temperature_drop * height ** 3
                / (solver.nu * cfg.diffusivity))

    def prandtl_check(self) -> float:
        """``Pr = nu/alpha`` recomputed from the discrete inputs."""
        return self.solver.nu / self.config.diffusivity

    # ---------------------------------------------------------------- initial
    def initialise(self, *, mode: str = "conduction", rolls: int = 1,
                   amplitude: float = 1.0e-3, seed: int = 0) -> None:
        """Set the temperature field: conduction profile plus a seed perturbation.

        The discrete conduction state ``theta = dT (1/2 - z)`` is an exact
        equilibrium of the scheme, so a perturbation is mandatory above onset:
        with ``mode="roll"`` a single normal mode ``cos(2 pi m x / Gamma)
        sin(pi z)`` of amplitude ``amplitude * dT`` is added (``m = rolls``),
        which is the mode the linear analysis of
        :mod:`tensorfvm.thermal.stability` says is the first to grow.  With
        ``mode="noise"`` a reproducible random field is used instead.
        """
        cfg, solver = self.config, self.solver
        if mode not in ("conduction", "roll", "noise"):
            raise ValueError("mode must be 'conduction', 'roll' or 'noise'")
        z = (torch.arange(solver.ny, dtype=solver.dtype, device=solver.device)
             + 0.5) * solver.hy
        solver.theta = (cfg.temperature_drop * (0.5 - z))[None, :].expand(
            solver.nx, solver.ny).clone()
        solver.u.zero_()
        solver.v.zero_()
        solver.p.zero_()
        solver.steps = 0
        solver.residual = float("inf")
        solver.apply_boundary_conditions()
        if mode == "conduction":
            return
        if mode == "roll":
            x = (torch.arange(solver.nx, dtype=solver.dtype, device=solver.device)
                 + 0.5) * solver.hx
            length = solver.hx * solver.nx
            shape = (torch.cos(2.0 * math.pi * rolls * x / length)[:, None]
                     * torch.sin(math.pi * z)[None, :])
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            shape = torch.randn(solver.nx, solver.ny, generator=generator,
                                dtype=solver.dtype).to(solver.device)
            shape = shape - shape.mean()
        solver.theta = solver.theta + amplitude * cfg.temperature_drop * shape

    # ------------------------------------------------------------- diagnostics
    def _wall_derivative(self, side: str) -> torch.Tensor:
        """``dtheta/dy`` at a plate, per horizontal cell (``side`` = bottom/top)."""
        solver, cfg = self.solver, self.config
        theta = solver.theta
        h = solver.hy
        wall = (solver.config.temperature_bottom if side == "bottom"
                else solver.config.temperature_top)
        if cfg.wall_gradient == "half_cell":
            if side == "bottom":
                return (theta[:, 0] - wall) / (0.5 * h)
            return (wall - theta[:, -1]) / (0.5 * h)
        if solver.ny < 3:
            raise ValueError("second_order wall gradient needs at least 3 cells")
        if side == "bottom":
            first, second = theta[:, 0], theta[:, 1]
            return (-8.0 * wall + 9.0 * first - second) / (3.0 * h)
        first, second = theta[:, -1], theta[:, -2]
        return (8.0 * wall - 9.0 * first + second) / (3.0 * h)

    def nusselt(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Nu_bottom, Nu_top)`` per horizontal cell from the chosen stencil.

        ``Nu = -dtheta/dy`` at the hot plate and ``+dtheta/dy`` at the cold one,
        both normalised so that pure conduction gives 1.
        """
        drop = self.config.temperature_drop
        # the heat flux is upward everywhere: Nu = -dtheta/dy at *both* plates
        return (-self._wall_derivative("bottom") / drop,
                -self._wall_derivative("top") / drop)

    def nusselt_budget(self) -> dict[str, float]:
        """Mean plate Nusselt numbers on the *half-cell* (Laplacian) stencil.

        Because this is the stencil the discrete diffusion operator uses, the
        two plate fluxes of a converged solution differ only by the accumulated
        ``dtheta/dt``; ``imbalance`` is therefore the discrete energy-conservation
        residual of the steady state.  Both plates are evaluated from their own
        wall values, so the residual is a measurement and not an identity.
        """
        solver, cfg = self.solver, self.config
        h = solver.hy
        drop = cfg.temperature_drop
        flux_bottom = (solver.config.temperature_bottom - solver.theta[:, 0]) / (0.5 * h)
        flux_top = (solver.theta[:, -1] - solver.config.temperature_top) / (0.5 * h)
        bottom = float(flux_bottom.mean()) / drop
        top = float(flux_top.mean()) / drop
        scale = max(abs(bottom), abs(top), 1e-300)
        return {
            "nusselt_bottom": bottom,
            "nusselt_top": top,
            "imbalance": abs(bottom - top) / scale,
            "bottom_spread": float(flux_bottom.std()) / drop,
            "top_spread": float(flux_top.std()) / drop,
        }

    def flux_profile(self) -> dict[str, object]:
        """Total (conductive + convective) heat flux on every interior y-face.

        ``F_j = sqrt(Ra Pr) <v theta>_j - <dtheta/dy>_j`` normalised by the
        conductive flux; at a steady state every interior value equals ``Nu``.
        The spread across the layer is the interior energy-conservation check
        (the plate budget of :meth:`nusselt_budget` only tests the two walls).
        """
        solver, cfg = self.solver, self.config
        theta = solver.theta
        v = solver.v[:, 1:solver.ny]                       # (nx, ny-1) interior faces
        face_theta = 0.5 * (theta[:, :-1] + theta[:, 1:])
        convective = math.sqrt(cfg.rayleigh * cfg.prandtl) * (v * face_theta).mean(dim=0)
        conductive = -((theta[:, 1:] - theta[:, :-1]) / solver.hy).mean(dim=0)
        total = (convective + conductive) / cfg.temperature_drop
        return {
            "y": ((torch.arange(solver.ny - 1, dtype=solver.dtype) + 1.0)
                  * solver.hy).tolist(),
            "flux": total.tolist(),
            "mean": float(total.mean()),
            "spread": float(total.max() - total.min()),
        }

    def flow_summary(self) -> dict[str, float]:
        """Streamfunction extremum and velocity maxima of the current field."""
        solver = self.solver
        psi = solver.streamfunction()
        u, v = solver.cell_velocity()
        return {
            "psi_min": float(psi.min()),
            "psi_max": float(psi.max()),
            "psi_abs_max": float(psi.abs().max()),
            "u_max": float(u.abs().max()),
            "v_max": float(v.abs().max()),
            "divergence_max": float(solver.divergence().abs().max()),
        }

    # ------------------------------------------------------------------ march
    def march(self, *, max_time: float | None = None, max_steps: int | None = None,
              tolerance: float | None = None, check_every: int | None = None,
              average_from: float | None = None) -> dict[str, object]:
        """March in time, sampling ``Nu``; stop at ``max_time`` or at steady state.

        With ``average_from`` set the run is a *statistical* steady state: the
        sampling window starts at that time and the plate Nusselt numbers are
        time-averaged over the rest of the march (the flow of a 2-D layer at
        ``Ra >= 1e6`` is time dependent, so a single snapshot is not a steady
        value).  Without it, the march stops when the sampled ``Nu`` stops
        changing by more than ``tolerance``.
        """
        solver, cfg = self.solver, self.config
        check_every = cfg.check_every if check_every is None else int(check_every)
        tolerance = cfg.tolerance if tolerance is None else float(tolerance)
        max_steps = cfg.max_steps if max_steps is None else int(max_steps)
        samples: list[dict[str, float]] = []
        self.time = 0.0
        for _ in range(max_steps):
            dt = solver.step()
            self.time += dt
            if solver.steps % check_every:
                continue
            bottom, top = self.nusselt()
            samples.append({"time": self.time, "nusselt_bottom": float(bottom.mean()),
                            "nusselt_top": float(top.mean())})
            if average_from is not None and self.time >= (max_time or math.inf):
                break
            if average_from is None and len(samples) >= 6:
                window = samples[-6:]
                reference = window[0]["nusselt_bottom"]
                change = abs(window[-1]["nusselt_bottom"] - reference) / max(abs(reference), 1e-12)
                solver.residual = change
                if change <= tolerance and self.time >= 5.0 * abs(dt) * check_every:
                    break
        return {"samples": samples, "time": self.time, "steps": solver.steps}

    # ------------------------------------------------------------ aggregation
    def statistics(self, march: dict[str, object], *, from_time: float = 0.0
                   ) -> dict[str, float]:
        """Time average and fluctuation of ``Nu`` over the samples after ``from_time``."""
        samples = [s for s in march["samples"] if s["time"] >= from_time]
        if not samples:
            raise ValueError("no samples after from_time")
        bottom = torch.tensor([s["nusselt_bottom"] for s in samples], dtype=torch.float64)
        top = torch.tensor([s["nusselt_top"] for s in samples], dtype=torch.float64)
        budget = self.nusselt_budget()
        flow = self.flow_summary()
        return {
            "nusselt_bottom_mean": float(bottom.mean()),
            "nusselt_top_mean": float(top.mean()),
            "nusselt_fluctuation": float(bottom.std()) if bottom.numel() > 1 else 0.0,
            "nusselt_min": float(bottom.min()),
            "nusselt_max": float(bottom.max()),
            "samples": len(samples),
            "window": float(samples[-1]["time"] - samples[0]["time"]),
            "time": float(march["time"]),
            "steps": int(march["steps"]),
            "imbalance": budget["imbalance"],
            "nusselt_budget_bottom": budget["nusselt_bottom"],
            "nusselt_budget_top": budget["nusselt_top"],
            **flow,
        }