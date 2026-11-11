"""Conjugate heat transfer over a flat plate with a conducting substrate.

Geometry (non-dimensionalised by the plate length ``L``, the free-stream speed
``U`` and the temperature difference ``T_bottom - T_inf``)::

    fluid   x in [x_start, 1],  y in [0, height]      u = Blasius(x, y)
    solid   x in [x_start, 1],  y in [-thickness, 0]  u = 0

The substrate is driven from below (``T = 1`` at ``y = -thickness``) and cooled by
the boundary layer above, so the wall temperature ``T_w(x)`` is an *output* of the
coupled solve rather than an imposed isothermal condition -- that is the
conjugate part.  ``solid_ratio = k_s / k_f`` controls the coupling: for large
ratios the wall becomes isothermal and the classical isothermal-wall Pohlhausen
solution applies; for small ratios the wall temperature sags downstream.

Velocity field: the Blasius streamfunction is evaluated on the grid *corners* and
the face volume fluxes are corner differences of ``psi``.  The discrete
divergence then telescopes to zero exactly (machine precision) whatever the grid
stretching -- mass conservation is not an approximation here, and the energy
ledger cannot be polluted by a spurious mass source.

Reference: ``Nu_x / sqrt(Re_x) = theta'(0)`` from :mod:`tensorfvm.cht.similarity`,
evaluated with the *local* wall temperature ``T_w(x)`` so that the comparison
stays meaningful at finite ``k_s/k_f``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .energy import Grid2D, solve_steady_energy
from .similarity import (BlasiusSolution, classical_correlation,
                         classical_uniform_flux_correlation, pohlhausen_nusselt,
                         solve_blasius, solve_uniform_flux, unheated_start_factor)

__all__ = ["FlatPlateCHT", "FlatPlateResult"]


@dataclass
class FlatPlateResult:
    x: np.ndarray
    nu_x: np.ndarray
    nu_reference: np.ndarray
    nu_correlation: np.ndarray
    t_wall: np.ndarray
    q_wall: np.ndarray
    ledger: dict
    metrics: dict
    fields: dict = field(default_factory=dict)

    @property
    def nu_error_rel(self) -> np.ndarray:
        return self.nu_x / self.nu_reference - 1.0


class FlatPlateCHT:
    """Coupled fluid/solid energy solve on a Cartesian grid straddling the wall."""

    def __init__(self, *, re_l: float = 1.0e5, pr: float = 0.7, solid_ratio: float = 100.0,
                 thickness: float = 0.01, height: float = 0.08, length: float = 1.0,
                 nx: int = 240, ny_fluid: int = 90, ny_solid: int = 16,
                 x_start: float = 1.0e-4, y_power: float = 2.0,
                 blasius: BlasiusSolution | None = None,
                 wall_flux: float | None = None):
        if re_l <= 0 or pr <= 0 or solid_ratio <= 0:
            raise ValueError("re_l, pr and solid_ratio must be positive")
        self.re_l = float(re_l)
        self.pr = float(pr)
        self.solid_ratio = float(solid_ratio)
        self.thickness = float(thickness)
        self.height = float(height)
        self.length = float(length)
        self.nx = int(nx)
        self.ny_fluid = int(ny_fluid)
        self.ny_solid = int(ny_solid)
        self.x_start = float(x_start)
        self.y_power = float(y_power)
        self.blasius = blasius if blasius is not None else solve_blasius()
        # ``wall_flux`` switches the substrate bottom from Dirichlet (T=1) to a
        # prescribed heat flux q'' into the plate.  That is the configuration for
        # which the uniform-heat-flux similarity solution is the exact reference
        # (``Nu_x = sqrt(Re_x)/phi(0)``); with a Dirichlet bottom the wall flux is
        # an output and only the quasi-isothermal Pohlhausen solution applies.
        self.wall_flux = None if wall_flux is None else float(wall_flux)
        if self.wall_flux is not None and self.ny_solid == 0:
            raise ValueError("a prescribed wall flux needs a solid substrate")

        self.u_inf = 1.0
        self.nu = self.u_inf * self.length / self.re_l
        self.kappa_fluid = self.nu / self.pr
        self.kappa_solid = self.solid_ratio * self.kappa_fluid
        self._build_grid()

    # ------------------------------------------------------------------ grid
    def _build_grid(self):
        xf = np.linspace(self.x_start, self.length, self.nx + 1)
        s = np.linspace(0.0, 1.0, self.ny_fluid + 1) ** self.y_power
        yf_fluid = self.height * s
        if self.ny_solid:
            yf_solid = np.linspace(-self.thickness, 0.0, self.ny_solid + 1)
            y_faces = np.concatenate([yf_solid[:-1], yf_fluid])
        else:                                   # isothermal wall, no substrate
            y_faces = yf_fluid
        self.grid = Grid2D(xf, y_faces)
        self.interface_row = self.ny_solid          # face index in y_faces
        ny = self.grid.ny
        kappa = np.empty((ny, self.grid.nx))
        kappa[: self.ny_solid, :] = self.kappa_solid
        kappa[self.ny_solid:, :] = self.kappa_fluid
        self.kappa = kappa

    # -------------------------------------------------------------- velocity
    def velocity_fluxes(self):
        """Divergence-free face volume fluxes from the Blasius streamfunction."""
        xf = self.grid.x_faces
        yf = self.grid.y_faces
        X, Y = np.meshgrid(xf, yf, indexing="xy")
        psi = np.where(Y > 0.0,
                       self.blasius.psi(X, np.maximum(Y, 1e-300), self.u_inf, self.nu), 0.0)
        flux_e = psi[1:, :] - psi[:-1, :]                  # (ny, nx+1)
        flux_n = -(psi[:, 1:] - psi[:, :-1])               # (ny+1, nx)
        return flux_e, flux_n, psi

    # ----------------------------------------------------------------- solve
    def solve(self, **kwargs) -> FlatPlateResult:
        flux_e, flux_n, psi = self.velocity_fluxes()
        ny = self.grid.ny
        types_west = np.array(["adiabatic"] * self.ny_solid + ["dirichlet"] * self.ny_fluid)
        values_west = np.zeros(ny)
        bc = {
            "west": {"types": types_west, "values": values_west},
            "east": {"type": "outflow", "values": 0.0},
            "south": ({"type": "neumann", "values": self.wall_flux}
                      if self.wall_flux is not None
                      else {"type": "dirichlet", "values": 1.0}),
            "north": {"type": "dirichlet", "values": 0.0},
        }
        T, info = solve_steady_energy(self.grid.dx, self.grid.dy, self.kappa,
                                      flux_e, flux_n, bc, **kwargs)
        return self._postprocess(T, info, flux_e, flux_n, psi)

    # ------------------------------------------------------------ diagnostics
    def _postprocess(self, T, info, flux_e, flux_n, psi) -> FlatPlateResult:
        j = self.interface_row
        xs = self.grid.xc
        dy = self.grid.dy
        kf, ks = self.kappa_fluid, self.kappa_solid
        s_fluid = 0.5 * dy[j]
        T_fluid = T[j, :]
        if self.ny_solid:
            s_solid = 0.5 * dy[j - 1]
            T_solid = T[j - 1, :]
            # single-face conductance: the flux is one number by construction
            r_tot = s_solid / ks + s_fluid / kf
            q = (T_solid - T_fluid) / r_tot                   # positive upward
            t_int = T_solid - q * s_solid / ks
            q_fluid_side = kf * (t_int - T_fluid) / s_fluid
            q_solid_side = ks * (T_solid - t_int) / s_solid
            local_iface_imbalance = q_fluid_side - q_solid_side
        else:                                                 # isothermal wall
            t_int = np.ones_like(xs)
            q = kf * (t_int - T_fluid) / s_fluid
            q_fluid_side = q
            q_solid_side = q
            local_iface_imbalance = np.zeros_like(xs)

        re_x = xs / self.nu
        nu_x = q * xs / (kf * np.maximum(t_int, 1e-300))
        if self.wall_flux is None:
            theta0 = pohlhausen_nusselt(self.pr, blasius=self.blasius)
            nu_ref = np.sqrt(re_x) * theta0
            nu_corr = classical_correlation(self.pr) * np.sqrt(re_x)
        else:                                  # uniform-flux similarity anchor
            uhf = solve_uniform_flux(self.pr, blasius=self.blasius)
            theta0 = uhf.coefficient
            nu_ref = np.sqrt(re_x) * theta0
            nu_corr = classical_uniform_flux_correlation(self.pr) * np.sqrt(re_x)

        mask = (xs >= 0.1) & (xs <= 0.95)
        err = nu_x[mask] / nu_ref[mask] - 1.0
        metrics = {
            "re_l": self.re_l,
            "prandtl": self.pr,
            "solid_ratio": self.solid_ratio,
            "nusselt_similarity_coefficient": theta0,
            "classical_coefficient_0.332": nu_corr[0] / np.sqrt(re_x[0]),
            "wall_flux": self.wall_flux,
            "nu_max_rel_error": float(np.max(np.abs(err))),
            "nu_l2_rel_error": float(np.sqrt(np.mean(err ** 2))),
            "nu_evaluation_band": [0.1, 0.95],
            "wall_temperature_min": float(np.min(t_int[mask])),
            "wall_temperature_max": float(np.max(t_int[mask])),
            "wall_temperature_nonuniformity": float(np.max(t_int[mask]) - np.min(t_int[mask])),
            "interface_local_flux_imbalance_max": float(np.max(np.abs(local_iface_imbalance))),
            "interface_local_flux_imbalance_rel": float(
                np.max(np.abs(local_iface_imbalance)) / max(np.max(np.abs(q)), 1e-300)),
            "unheated_start_factor_at_0.1": unheated_start_factor(0.1 / self.x_start),
            "grid": {"nx": self.nx, "ny_fluid": self.ny_fluid, "ny_solid": self.ny_solid,
                     "x_start": self.x_start, "y_power": self.y_power,
                     "thickness": self.thickness, "height": self.height},
            "picard_iterations": info["picard_iterations"],
        }
        if self.wall_flux is not None:
            # Analytic conjugate reference for a uniformly fluxed plate:
            #   T_w(x) = q'' / (k_f * Nu_x / x)  with  Nu_x = sqrt(Re_x)/phi(0)
            #   T_s(x, y) = T_w(x) - q'' y / k_s   (axial conduction neglected)
            # Nothing from the discrete solution enters this reference: it is the
            # uniform-heat-flux similarity solution plus one-dimensional conduction.
            tw_analytic = self.wall_flux * xs / (kf * nu_ref)
            y_cells = 0.5 * (self.grid.y_faces[:-1] + self.grid.y_faces[1:])
            rows = slice(0, self.ny_solid)
            ts_analytic = tw_analytic[None, :] - self.wall_flux * y_cells[rows, None] / ks
            band = (xs >= 0.1) & (xs <= 0.95)
            ts_num = T[rows, :][:, band]
            ts_ref = ts_analytic[:, band]
            scale = max(float(np.max(np.abs(ts_ref))), 1e-300)
            metrics["wall_temperature_analytic"] = tw_analytic
            metrics["solid_analytic_max_rel_error"] = float(
                np.max(np.abs(ts_num - ts_ref)) / scale)
            metrics["solid_analytic_max_abs_error"] = float(np.max(np.abs(ts_num - ts_ref)))
            metrics["wall_temperature_rel_error_max"] = float(
                np.max(np.abs(t_int[band] - tw_analytic[band]) / tw_analytic[band]))
            q_band = q[band]
            metrics["wall_flux_uniformity_rel"] = float(
                (np.max(q_band) - np.min(q_band)) / np.mean(q_band))
            # global interface conservation: heat supplied at the bottom of the
            # slab must equal the heat entering the fluid through the interface.
            q_in = self.wall_flux * (self.length - self.x_start)
            q_out = float(np.sum(q * self.grid.dx))
            metrics["global_interface_balance_rel"] = abs(q_out - q_in) / abs(q_in)
            metrics["global_interface_heat_rate_in"] = q_in
            metrics["global_interface_heat_rate_out"] = q_out
        dy_div = np.abs(flux_e[:, 1:] - flux_e[:, :-1] + flux_n[1:, :] - flux_n[:-1, :])
        metrics["max_discrete_divergence"] = float(np.max(dy_div))
        fields = {
            "x": xs, "nu_x": nu_x, "nu_reference": nu_ref, "t_wall": t_int, "q_wall": q,
            "T": T, "y_faces": self.grid.y_faces, "x_faces": self.grid.x_faces,
            "y_cells": 0.5 * (self.grid.y_faces[:-1] + self.grid.y_faces[1:]),
            "solid_analytic": (None if self.wall_flux is None else ts_analytic),
            "psi": psi,
        }
        return FlatPlateResult(x=xs, nu_x=nu_x, nu_reference=nu_ref, nu_correlation=nu_corr,
                               t_wall=t_int, q_wall=q, ledger=info["ledger"].as_dict(),
                               metrics=metrics, fields=fields)

    # -------------------------------------------------------- wall-only solve
    @classmethod
    def isothermal(cls, **kwargs):
        """Textbook isothermal-wall plate: no substrate, ``T_wall = 1`` exactly.

        This is the Pohlhausen control case.  It isolates the fluid-side
        discretisation (velocity field + energy operator) from the conjugate
        coupling, so an error in the conjugate loop cannot hide here.
        """
        kwargs.setdefault("ny_solid", 0)
        return cls(**kwargs)