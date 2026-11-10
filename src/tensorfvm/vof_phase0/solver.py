"""Compact two-phase solver used by the Phase-0 dam-break gate.

Two surface models share one momentum/projection kernel, selected by
``surface_model``:

``"void"`` (default, used for the quantitative dam-break comparison)
    The gas is an atmospheric void: the pressure problem is solved on the liquid
    cells only (``alpha > 1/2``) with a ``p = 0`` Dirichlet condition on every
    face that touches a void cell.  This is the classical MAC/VOF free-surface
    formulation (Hirt & Nichols 1981) and it is what makes a 1000:1 density jump
    tractable: the gas neither carries momentum into the pressure operator nor
    stiffens it (the ``1/rho`` contrast inside one elliptic problem is the
    dominant error source of a fully coupled two-phase projection).

``"two_phase"``
    Fully coupled variable-density projection: the Poisson operator is assembled
    over both phases with the harmonic face density ``2/(rho_1+rho_2)`` (the
    elliptically consistent coefficient for a discontinuous operator).  Used for
    the 1000:1 stability/spurious-current evidence (AC3); its dam-break surge
    trajectory is *not* used for the accuracy gate because the gas-liquid
    pressure coupling delays the start-up transient (documented limitation).

Common pieces:

* explicit momentum predictor -- first-order upwind convection, variable
  viscosity (harmonic blend at the interface), gravity, free-slip walls;
* pressure projection ``div((1/rho)_face grad(phi)) = div(u*)/dt`` with Neumann
  walls and a ``phi = 0`` Dirichlet datum half a cell above the top row;
* interface transport by :class:`tensorfvm.vof_phase0.AlgebraicVOF` (exactly
  conservative flux-form FCT) using the projected velocity.
"""

from __future__ import annotations

import math

import torch

from .advection import AlgebraicVOF

__all__ = ["Phase0TwoPhaseSolver", "cell_centres"]


def cell_centres(n: int, h: float, dtype=torch.float64) -> torch.Tensor:
    """Cell-centre coordinates of ``n`` uniform cells of size ``h``."""
    return (torch.arange(n, dtype=dtype) + 0.5) * h


class Phase0TwoPhaseSolver:
    """Incompressible two-phase flow on a staggered MAC grid."""

    def __init__(
        self,
        nx: int,
        ny: int,
        length: float,
        height: float,
        rho_liquid: float = 1000.0,
        rho_gas: float = 1.0,
        mu_liquid: float = 1.0e-3,
        mu_gas: float = 1.0e-5,
        gravity: float = 9.81,
        alpha_cfl: float = 0.4,
        vof_bounds: str = "global",
        surface_model: str = "void",
        liquid_cut: float = 1.0e-6,
        bed: str = "free_slip",
        cg_rtol: float = 1.0e-9,
        cg_max_iter: int = 3000,
    ) -> None:
        if nx < 4 or ny < 4:
            raise ValueError("grid too coarse for a two-phase solve")
        if rho_liquid <= 0.0 or rho_gas <= 0.0:
            raise ValueError("densities must be positive")
        if surface_model not in ("void", "two_phase"):
            raise ValueError(f"unknown surface_model {surface_model!r}")
        if bed not in ("free_slip", "no_slip"):
            raise ValueError(f"unknown bed condition {bed!r}")
        self.nx, self.ny = int(nx), int(ny)
        self.lx, self.ly = float(length), float(height)
        self.dx, self.dy = self.lx / self.nx, self.ly / self.ny
        self.rho_l, self.rho_g = float(rho_liquid), float(rho_gas)
        self.mu_l, self.mu_g = float(mu_liquid), float(mu_gas)
        self.gravity = float(gravity)
        self.alpha_cfl = float(alpha_cfl)
        self.surface_model = surface_model
        self.bed = bed
        self.liquid_cut = float(liquid_cut)
        self.cg_rtol, self.cg_max_iter = float(cg_rtol), int(cg_max_iter)

        self.u = torch.zeros((self.ny, self.nx + 1), dtype=torch.float64)
        self.v = torch.zeros((self.ny + 1, self.nx), dtype=torch.float64)
        self.p = torch.zeros((self.ny, self.nx), dtype=torch.float64)
        self.alpha = torch.zeros((self.ny, self.nx), dtype=torch.float64)
        self.vof = AlgebraicVOF(self.nx, self.ny, self.dx, self.dy,
                                boundary="wall", bounds=vof_bounds)
        self.time = 0.0
        self.steps = 0
        self.cg_last: dict = {}
        self._phi_prev: torch.Tensor | None = None
        self._update_properties()

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def _update_properties(self) -> None:
        self.rho = self.alpha * self.rho_l + (1.0 - self.alpha) * self.rho_g
        self.mu = 1.0 / (self.alpha / self.mu_l + (1.0 - self.alpha) / self.mu_g)
        self.liquid_mask = self.alpha > self.liquid_cut

    def set_alpha(self, alpha: torch.Tensor) -> None:
        self.alpha = alpha.to(torch.float64).clone()
        self._update_properties()

    def set_hydrostatic_pressure(self, h0: float, a: float) -> None:
        """Column hydrostatic head ``p = rho_l g (h0 - y)`` for ``x <= a``."""
        ys = cell_centres(self.ny, self.dy)[:, None]
        column = cell_centres(self.nx, self.dx)[None, :] <= a
        head = torch.clamp(h0 - ys, min=0.0)
        self.p = torch.where(column, self.rho_l * self.gravity * head,
                             torch.zeros_like(head)).clone()
        if self.surface_model == "void":
            self.p = torch.where(self.liquid_mask, self.p, torch.zeros_like(self.p))
        self._phi_prev = None

    def divergence(self) -> torch.Tensor:
        return ((self.u[:, 1:] - self.u[:, :-1]) / self.dx
                + (self.v[1:, :] - self.v[:-1, :]) / self.dy)

    def liquid_volume(self) -> float:
        return float(self.alpha.sum().item()) * self.dx * self.dy

    # ------------------------------------------------------------------ #
    # time step
    # ------------------------------------------------------------------ #
    def stable_dt(self, max_depth: float | None = None) -> float:
        """CFL-limited explicit step (convection, gravity wave, viscosity)."""
        umax = float(self.u.abs().max().item())
        vmax = float(self.v.abs().max().item())
        if max_depth is None:
            max_depth = float(self.alpha.sum(dim=0).max().item() * self.dy)
        wave = math.sqrt(self.gravity * max(max_depth, 1.0e-9))
        speed = max(umax, vmax, wave, 1.0e-6)
        dt = self.alpha_cfl * min(self.dx, self.dy) / speed
        nu = self.mu_l / self.rho_l
        h = min(self.dx, self.dy)
        return float(min(dt, 0.25 * h * h / (2.0 * nu)))

    # ------------------------------------------------------------------ #
    # discrete operators
    # ------------------------------------------------------------------ #
    def _face_inv_rho(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse face density ``(1/rho)_face`` for the pressure operator."""
        if self.surface_model == "void":
            liquid = self.liquid_mask
            inv_l = 1.0 / self.rho_l
            inv_u = torch.where(liquid[:, :-1] | liquid[:, 1:],
                                torch.full_like(self.alpha[:, :-1], inv_l),
                                torch.zeros_like(self.alpha[:, :-1]))
            inv_v = torch.where(liquid[:-1, :] | liquid[1:, :],
                                torch.full_like(self.alpha[:-1, :], inv_l),
                                torch.zeros_like(self.alpha[:-1, :]))
            return inv_u, inv_v
        inv = 1.0 / self.rho
        inv_u = 2.0 * inv[:, :-1] * inv[:, 1:] / (inv[:, :-1] + inv[:, 1:])
        inv_v = 2.0 * inv[:-1, :] * inv[1:, :] / (inv[:-1, :] + inv[1:, :])
        return inv_u, inv_v

    def _advective_accel(self) -> tuple[torch.Tensor, torch.Tensor]:
        u, v, dx, dy = self.u, self.v, self.dx, self.dy
        # ---- u-faces (all rows, interior columns) ----
        u_c = u[:, 1:-1]
        u_w = u[:, :-2]
        u_e = u[:, 2:]
        u_bottom = -u[:1] if self.bed == "no_slip" else u[:1]   # bed ghost
        u_y = torch.cat([u_bottom, u, u[-1:]], dim=0)
        u_s = u_y[:-2, 1:-1]
        u_n = u_y[2:, 1:-1]
        v_at_u = 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])
        dudx = torch.where(u_c >= 0.0, (u_c - u_w) / dx, (u_e - u_c) / dx)
        dudy = torch.where(v_at_u >= 0.0, (u_c - u_s) / dy, (u_n - u_c) / dy)
        adv_u = u_c * dudx + v_at_u * dudy

        # ---- v-faces (interior rows, all columns) ----
        v_c = v[1:-1, :]
        v_s = v[:-2, :]
        v_n = v[2:, :]
        v_x = torch.cat([v[:, :1], v, v[:, -1:]], dim=1)  # free-slip ghosts
        v_w = v_x[1:-1, :-2]
        v_e = v_x[1:-1, 2:]
        u_at_v = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
        dvdy = torch.where(v_c >= 0.0, (v_c - v_s) / dy, (v_n - v_c) / dy)
        dvdx = torch.where(u_at_v >= 0.0, (v_c - v_w) / dx, (v_e - v_c) / dx)
        adv_v = u_at_v * dvdx + v_c * dvdy
        return adv_u, adv_v

    def _viscous(self) -> tuple[torch.Tensor, torch.Tensor]:
        u, v, dx, dy, mu, rho = self.u, self.v, self.dx, self.dy, self.mu, self.rho
        mu_u = 0.5 * (mu[:, :-1] + mu[:, 1:])
        rho_u = 0.5 * (rho[:, :-1] + rho[:, 1:])
        u_bottom = -u[:1] if self.bed == "no_slip" else u[:1]
        u_y = torch.cat([u_bottom, u, u[-1:]], dim=0)
        lap_u = ((u[:, :-2] - 2.0 * u[:, 1:-1] + u[:, 2:]) / dx ** 2
                 + (u_y[:-2, 1:-1] - 2.0 * u_y[1:-1, 1:-1] + u_y[2:, 1:-1]) / dy ** 2)
        visc_u = (mu_u / rho_u) * lap_u

        mu_v = 0.5 * (mu[:-1, :] + mu[1:, :])
        rho_v = 0.5 * (rho[:-1, :] + rho[1:, :])
        v_x = torch.cat([v[:, :1], v, v[:, -1:]], dim=1)
        lap_v = ((v_x[1:-1, :-2] - 2.0 * v_x[1:-1, 1:-1] + v_x[1:-1, 2:]) / dx ** 2
                 + (v[:-2, :] - 2.0 * v[1:-1, :] + v[2:, :]) / dy ** 2)
        visc_v = (mu_v / rho_v) * lap_v
        return visc_u, visc_v

    def _pressure_gradient(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(1/rho)_face grad(p)`` on the interior u- and v-faces."""
        inv_u, inv_v = self._face_inv_rho()
        gx = (self.p[:, 1:] - self.p[:, :-1]) / self.dx
        gy = (self.p[1:, :] - self.p[:-1, :]) / self.dy
        return inv_u * gx, inv_v * gy

    # ------------------------------------------------------------------ #
    # projection
    # ------------------------------------------------------------------ #
    def project(self, dt: float) -> None:
        """Solve the pressure Poisson equation and correct the velocities."""
        inv_u, inv_v = self._face_inv_rho()
        div = self.divergence()
        cell_vol = self.dx * self.dy
        liquid = self.liquid_mask

        ce = inv_u * self.dy / self.dx           # (ny, nx-1)
        cn = inv_v * self.dx / self.dy           # (ny-1, nx)
        ce_full = torch.zeros((self.ny, self.nx))
        ce_full[:, :-1] += ce
        ce_full[:, 1:] += ce
        cn_full = torch.zeros((self.ny, self.nx))
        cn_full[:-1, :] += cn
        cn_full[1:, :] += cn
        diag = ce_full + cn_full
        # open top: Dirichlet datum half a cell above the top row
        top_coef = (1.0 / self.rho_l) * self.dx / (0.5 * self.dy)
        diag[-1, :] += top_coef

        if self.surface_model == "void":
            # liquid cells only; void rows are the identity with phi = 0
            ce_off = torch.where(liquid[:, :-1] & liquid[:, 1:], ce,
                                 torch.zeros_like(ce))
            cn_off = torch.where(liquid[:-1, :] & liquid[1:, :], cn,
                                 torch.zeros_like(cn))
            diag = torch.where(liquid, diag, torch.ones_like(diag))
            b = torch.where(liquid, -div * cell_vol / dt, torch.zeros_like(div))
        else:
            ce_off = ce
            cn_off = cn
            b = -div * cell_vol / dt

        def matvec(q: torch.Tensor) -> torch.Tensor:
            out = diag * q
            out[:, :-1] -= ce_off * q[:, 1:]
            out[:, 1:] -= ce_off * q[:, :-1]
            out[:-1, :] -= cn_off * q[1:, :]
            out[1:, :] -= cn_off * q[:-1, :]
            return out

        norm = torch.linalg.vector_norm(b).item()
        if norm == 0.0:
            self.cg_last = {"iterations": 0, "converged": True,
                            "residual_ratio": 0.0}
            self._phi_prev = None
            q = torch.zeros_like(b)
        else:
            if self._phi_prev is not None and self._phi_prev.shape == b.shape:
                q = self._phi_prev.clone()
            else:
                q = torch.zeros_like(b)
            r = b - matvec(q)
            z = r / diag
            direction = z.clone()
            rz = torch.sum(r * z)
            target = norm * self.cg_rtol
            converged = False
            iterations = 0
            for iterations in range(1, self.cg_max_iter + 1):
                ad = matvec(direction)
                denom = torch.sum(direction * ad)
                if denom.item() <= 0.0 or not torch.isfinite(denom):
                    break
                step = rz / denom
                q = q + step * direction
                r = r - step * ad
                if torch.linalg.vector_norm(r).item() <= target:
                    converged = True
                    break
                z = r / diag
                rz_new = torch.sum(r * z)
                direction = z + (rz_new / rz) * direction
                rz = rz_new
            self._phi_prev = q
            self.cg_last = {
                "iterations": int(iterations),
                "converged": bool(converged),
                "residual_ratio": float(
                    torch.linalg.vector_norm(b - matvec(q)).item() / norm),
            }

        # velocity correction: u -= dt (1/rho)_face grad(phi)
        v_top = self.v[-1, :].clone()
        grad_x = (q[:, 1:] - q[:, :-1]) / self.dx
        grad_y = (q[1:, :] - q[:-1, :]) / self.dy
        self.u[:, 1:-1] -= dt * inv_u * grad_x
        self.v[1:-1, :] -= dt * inv_v * grad_y
        if self.surface_model == "void":
            top_inv_v = torch.where(
                liquid[-1, :],
                torch.full_like(q[-1, :], 1.0 / self.rho_l),
                torch.zeros_like(q[-1, :]))
        else:
            top_inv_v = inv_v[-1, :]
        self.v[-1, :] = v_top - dt * top_inv_v * (0.0 - q[-1, :]) / (0.5 * self.dy)
        self.p = self.p + q
        if self.surface_model == "void":
            self.p = torch.where(liquid, self.p, torch.zeros_like(self.p))

    # ------------------------------------------------------------------ #
    # one step
    # ------------------------------------------------------------------ #
    def step(self, dt: float | None = None) -> dict:
        """Advance one explicit step (momentum, projection, VOF transport)."""
        if dt is None:
            dt = self.stable_dt()
        adv_u, adv_v = self._advective_accel()
        visc_u, visc_v = self._viscous()
        gradp_u, gradp_v = self._pressure_gradient()

        self.u[:, 1:-1] += dt * (-adv_u + visc_u - gradp_u)
        self.v[1:-1, :] += dt * (-adv_v + visc_v - gradp_v - self.gravity)

        # walls: no normal flow
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[0, :] = 0.0

        self.project(dt)
        # divergence of the cells that were constrained by the projection
        # (measured here: the mask moves during the VOF advection below, and
        # cells that only wet during this step were not part of the solve)
        div_liq = float(torch.abs(self.divergence()[self.liquid_mask]).max().item()) \
            if bool(self.liquid_mask.any()) else 0.0

        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[0, :] = 0.0
        if self.surface_model == "void":
            # The void carries no momentum.  Faces with a liquid cell on at
            # least one side keep their (projected) velocity -- those are the
            # interface faces that transport the volume fraction; faces fully
            # inside the void are held at rest so the gas free-fall velocity
            # cannot set the CFL limit or advect the interface.
            void = ~self.liquid_mask
            keep_u = ~(void[:, :-1] & void[:, 1:])
            keep_v = ~(void[:-1, :] & void[1:, :])
            self.u[:, 1:-1] = torch.where(keep_u, self.u[:, 1:-1],
                                          torch.zeros_like(self.u[:, 1:-1]))
            self.v[1:-1, :] = torch.where(keep_v, self.v[1:-1, :],
                                          torch.zeros_like(self.v[1:-1, :]))
            self.v[-1, :] = torch.where(self.liquid_mask[-1, :], self.v[-1, :],
                                        torch.zeros_like(self.v[-1, :]))

        self.alpha = self.vof.advect(self.alpha, self.u, self.v, dt)
        self._update_properties()
        self.time += dt
        self.steps += 1
        return {
            "time": self.time,
            "dt": dt,
            "divergence_max": float(torch.abs(self.divergence()).max().item()),
            "divergence_max_liquid": div_liq,
            "volume": self.liquid_volume(),
            "alpha_min": float(self.alpha.min().item()),
            "alpha_max": float(self.alpha.max().item()),
            "speed_max": max(float(self.u.abs().max().item()),
                             float(self.v.abs().max().item())),
            "cg": dict(self.cg_last),
        }