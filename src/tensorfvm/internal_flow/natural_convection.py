"""Differentially heated square cavity at ``Ra = 1e4, 1e5, 1e6`` (``Pr = 0.71``).

Physics
-------
Boussinesq buoyancy in a square box ``[0, 1]^2``: the left wall is hot
(``theta = +dT/2``), the right wall cold, the top and bottom walls adiabatic, and
all walls are no-slip.  The non-dimensionalisation uses the free-fall velocity
``U = sqrt(g beta dT H)`` and the cavity height ``H = 1`` as units, so

    du/dt + div(u u) = -grad p + sqrt(Pr/Ra) lap(u) + theta e_y,
    dtheta/dt + div(u theta) = lap(theta) / sqrt(Pr Ra),

which reproduces ``Ra = g beta dT H^3/(nu alpha)`` and ``Pr = nu/alpha``
exactly; :meth:`NaturalConvectionCase.rayleigh_check` verifies that identity on
the discrete inputs so a scaling slip cannot pass unnoticed.

Acceptance quantities
---------------------
Mean Nusselt number on the hot wall (``Nu = -dtheta/dx`` integrated, normalised
by ``dT/H``), plus the extrema of ``Nu``, the streamfunction extremum and the
mid-plane velocity maxima, compared with de Vahl Davis (1983) for ``Pr = 0.71``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .mac import MACConfig, StaggeredMAC
from .reference import DE_VAHL_DAVIS_NU, DE_VAHL_DAVIS_VELOCITY


@dataclass(frozen=True)
class NaturalConvectionConfig:
    """Mesh and solver controls of the buoyant-cavity case."""

    n: int = 64
    rayleigh: float = 1e4
    prandtl: float = 0.71
    cfl: float = 0.4
    time_step: float | None = None
    max_steps: int = 600_000
    tolerance: float = 1e-9
    check_every: int = 200
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 8:
            raise ValueError("n must be an integer >= 8")
        for name in ("rayleigh", "prandtl", "cfl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def viscosity(self) -> float:
        return math.sqrt(self.prandtl / self.rayleigh)

    @property
    def diffusivity(self) -> float:
        return 1.0 / math.sqrt(self.prandtl * self.rayleigh)

    @property
    def temperature_drop(self) -> float:
        return 1.0


class NaturalConvectionCase:
    """Buoyant square cavity built on :class:`StaggeredMAC`."""

    def __init__(self, config: NaturalConvectionConfig | None = None) -> None:
        self.config = config or NaturalConvectionConfig()
        cfg = self.config
        self.solver = StaggeredMAC(MACConfig(
            nx=cfg.n, ny=cfg.n,
            length=1.0, height=1.0,
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
            temperature_left=0.5 * cfg.temperature_drop,
            temperature_right=-0.5 * cfg.temperature_drop,
            device=cfg.device,
        ))

    # ------------------------------------------------------------------ driving
    def run(self, **kwargs) -> dict[str, object]:
        return self.solver.run(**kwargs)

    # ------------------------------------------------------------- physics check
    def rayleigh_check(self) -> float:
        """``g beta dT H^3 / (nu alpha)`` recomputed from the discrete inputs."""
        cfg = self.config
        solver = self.solver
        gravity_beta = solver.config.buoyancy
        height = solver.hy * solver.ny
        return (gravity_beta * cfg.temperature_drop * height ** 3
                / (solver.nu * cfg.diffusivity))

    # -------------------------------------------------------------- diagnostics
    def nusselt_profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(y, Nu_local)`` on the hot (left) wall from the wall-normal gradient."""
        cfg = self.config
        solver = self.solver
        y = (torch.arange(solver.ny, dtype=solver.dtype) + 0.5) * solver.hy
        gradient = (0.5 * cfg.temperature_drop - solver.theta[0, :]) / (0.5 * solver.hx)
        return y, gradient

    def diagnostics(self) -> dict[str, object]:
        cfg = self.config
        solver = self.solver
        y, nu_local = self.nusselt_profile()
        nu_mean = float(nu_local.mean())
        u_cell, v_cell = solver.cell_velocity()
        alpha = cfg.diffusivity
        psi = solver.streamfunction() / alpha
        i_mid = solver.nx // 2
        j_mid = solver.ny // 2
        u_max = float(u_cell[i_mid, :].abs().max()) / alpha
        v_max = float(v_cell[:, j_mid].abs().max()) / alpha
        reference = DE_VAHL_DAVIS_NU.get(cfg.rayleigh)
        velocity_reference = DE_VAHL_DAVIS_VELOCITY.get(cfg.rayleigh, {})
        idx = int(torch.argmax(psi))
        i_psi, j_psi = divmod(idx, solver.ny + 1)
        return {
            "n": cfg.n,
            "rayleigh": cfg.rayleigh,
            "prandtl": cfg.prandtl,
            "rayleigh_discrete": self.rayleigh_check(),
            "viscosity": cfg.viscosity,
            "diffusivity": cfg.diffusivity,
            "steps": solver.steps,
            "dt": solver.dt,
            "residual": solver.residual,
            "max_divergence": float(solver.divergence().abs().max()),
            "nu_mean": nu_mean,
            "nu_max": float(nu_local.max()),
            "nu_min": float(nu_local.min()),
            "nu_reference": reference,
            "nu_error_percent": None if reference is None
            else 100.0 * (nu_mean - reference) / reference,
            "psi_max": float(psi.max()),
            "psi_min": float(psi.min()),
            "psi_max_location": [float((i_psi + 0.5) * solver.hx),
                                 float(j_psi * solver.hy)],
            "u_max_over_alpha": u_max,
            "v_max_over_alpha": v_max,
            "u_max_reference": velocity_reference.get("u_max"),
            "v_max_reference": velocity_reference.get("v_max"),
            "y": y.tolist(),
            "nu_local": nu_local.tolist(),
        }