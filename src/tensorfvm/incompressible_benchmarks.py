"""Incompressible-flow verification surface: backward-facing step, sudden
expansion and natural convection in a square cavity.

Why this module exists
----------------------
TensorFVM's incompressible evidence is dominated by periodic / closed
benchmarks (channel DNS, lid-driven cavity, cylinder) and by the LBM
cross-check.  The three cases implemented here are the standard *open* and
*buoyant* anchors such a surface is expected to carry:

* :func:`run_backward_facing_step` -- laminar separation behind a
  backward-facing step at ``Re_H = 800`` (step height ``H``, mean inlet
  velocity).  Anchor: reattachment length ``L_r / H``, ``6.10`` from the
  finite-element benchmark of Gartling (1990), ``6.0 +- 0.2`` from the
  Armaly et al. (1983) experiment.
* :func:`run_sudden_expansion` -- planar sudden expansion, area ratio
  ``AR = 4``, ``Re_D = 100``.  Anchors: reattachment length ``L_r / D`` and the
  loss coefficient ``K`` against the inviscid Borda-Carnot value
  ``(1 - 1/AR) ** 2 = 0.5625``.
* :func:`run_natural_convection` -- differentially heated square cavity at
  ``Ra = 1e4, 1e5, 1e6``, ``Pr = 0.71``.  Anchors: mean Nusselt number and
  ``psi_max`` against de Vahl Davis (1983).

Numerics
--------
One staggered (MAC) finite-volume discretisation is shared by all cases:

* ``u`` on x-faces ``(nx + 1, ny)``, ``v`` on y-faces ``(nx, ny + 1)``,
  ``p`` and ``theta`` cell-centred, uniform square cells of size ``h``;
* convection in conservative flux form with a van Leer (harmonic, TVD) face
  reconstruction -- the same limiter as :mod:`tensorfvm.lid_cavity`;
* explicit forward-Euler marching of momentum and energy, with an exact
  (non-iterative) pressure solve: orthonormal DCT-II diagonalisation for a
  closed domain (pure Neumann), DCT-II along the wall-normal direction plus one
  tridiagonal solve per mode for an open domain (``dp/dx = 0`` at inflow,
  ``p = 0`` at outflow);
* internal no-slip walls (the step, the upstream walls of the expansion) use a
  volume-penalisation mask: every momentum face in contact with a solid cell is
  relaxed towards zero by ``kappa`` after the momentum update and after the
  pressure correction.  The face sits exactly on the geometric wall, so the
  wall is not smeared; the residual slip is ``O(kappa)`` and is reported as
  ``solid_slip`` rather than hidden.

Boussinesq buoyancy lives on the same grid (``energy=True``): the energy
equation is advected with the same limiter and the vertical momentum receives
``Ra * Pr * theta``.

Everything is deterministic numpy/scipy; no wall-clock or RNG input enters any
reported number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy.linalg import solve_banded

TINY = 1e-300

#: de Vahl Davis (1983) benchmark for the differentially heated square cavity
#: (``Pr = 0.71``).  ``nu`` is the mean Nusselt number on the hot wall,
#: ``psi_max`` the streamfunction extremum (height and thermal diffusivity
#: scaling).
DE_VAHL_DAVIS = {
    1.0e4: {"nu": 2.243, "psi_max": 5.071, "u_max": 16.178, "v_max": 19.617},
    1.0e5: {"nu": 4.519, "psi_max": 9.111, "u_max": 34.73, "v_max": 68.59},
    1.0e6: {"nu": 8.800, "psi_max": 16.750, "u_max": 64.63, "v_max": 219.36},
}

#: Gartling (1990) finite-element reattachment length at ``Re = 800`` in units
#: of the step height (expansion ratio 2, parabolic inlet).
GARTLING_REATTACHMENT = 6.10

#: Armaly et al. (1983) experimental reattachment length at ``Re = 800``.
ARMALY_REATTACHMENT = 6.0
ARMALY_TOLERANCE = 0.2

#: Inviscid Borda-Carnot expansion loss for ``AR = 4``.
BORDA_CARNOT_AR4 = (1.0 - 1.0 / 4.0) ** 2


def _vanleer_face(f: np.ndarray, vel: np.ndarray, axis: int) -> np.ndarray:
    """van Leer limited interface values of ``f`` at the faces along ``axis``.

    ``f`` holds ``M`` samples along ``axis``, ``vel`` the ``M - 1`` normal
    velocities at the interfaces between them; the return value has the shape of
    ``vel``.  The limited slope is the harmonic mean of the upwind and the local
    difference, zeroed where they change sign (TVD, no new extrema).
    """
    fa = np.moveaxis(f, axis, 0)
    va = np.moveaxis(vel, axis, 0)
    d = fa[1:] - fa[:-1]
    up = np.empty_like(d)
    up[1:] = d[:-1]
    up[0] = d[0]
    down = np.empty_like(d)
    down[:-1] = d[1:]
    down[-1] = d[-1]
    def limiter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        prod = a * b
        total = a + b
        safe = np.abs(total) > TINY
        value = 2.0 * prod / np.where(safe, total, np.ones_like(total))
        return np.where(prod > 0.0, value, 0.0)

    forward = fa[:-1] + 0.5 * limiter(up, d)
    backward = fa[1:] - 0.5 * limiter(down, d)
    return np.moveaxis(np.where(va >= 0.0, forward, backward), 0, axis)


@dataclass
class FlowState:
    """Convergence ledger of one steady-state march."""

    steps: int
    dt: float
    residual: float
    converged: bool
    max_divergence: float
    solid_slip: float
    cfl: float
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "dt": self.dt,
            "residual": self.residual,
            "converged": self.converged,
            "max_divergence": self.max_divergence,
            "solid_slip": self.solid_slip,
            "cfl": self.cfl,
        }


class StaggeredFlow:
    """Staggered MAC solver for 2-D incompressible flow (see module docstring)."""

    def __init__(self, nx: int, ny: int, h: float, nu: float, *,
                 inlet: np.ndarray | None = None,
                 outlet: str = "wall",
                 solid: np.ndarray | None = None,
                 kappa: float = 1.0e-4,
                 wall_lo: float = 0.0,
                 wall_hi: float = 0.0,
                 energy: bool = False,
                 rayleigh: float = 0.0,
                 prandtl: float = 0.71,
                 theta_hot: float = 0.5,
                 cfl: float = 0.25) -> None:
        for name, value in (("nx", nx), ("ny", ny)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f"{name} must be an integer >= 4")
        if not (math.isfinite(h) and h > 0.0):
            raise ValueError("h must be a positive finite number")
        if not (math.isfinite(nu) and nu > 0.0):
            raise ValueError("nu must be a positive finite number")
        if outlet not in ("wall", "open"):
            raise ValueError("outlet must be 'wall' or 'open'")
        if inlet is not None:
            inlet = np.asarray(inlet, dtype=float).reshape(-1)
            if inlet.size != ny:
                raise ValueError("inlet profile must have ny entries")
            if outlet != "open":
                raise ValueError("an inlet requires outlet='open'")
        if not (0.0 < kappa < 1.0):
            raise ValueError("kappa must lie in (0, 1)")

        self.nx, self.ny, self.h, self.nu = nx, ny, float(h), float(nu)
        self.inlet = None if inlet is None else inlet.copy()
        self.outlet = outlet
        self.kappa = float(kappa)
        self.wall_lo, self.wall_hi = float(wall_lo), float(wall_hi)
        self.energy_on = bool(energy)
        self.rayleigh = float(rayleigh)
        self.prandtl = float(prandtl)
        self.theta_hot = float(theta_hot)
        self.cfl = float(cfl)
        self.steps = 0

        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        self.p = np.zeros((nx, ny))
        self.theta = np.zeros((nx, ny))
        if self.inlet is not None:
            self.u[0, :] = self.inlet
        if self.energy_on:
            xs = (np.arange(nx) + 0.5) * self.h
            self.theta[:, :] = self.theta_hot * (1.0 - 2.0 * xs / (nx * self.h))[:, None]

        if solid is None:
            self.solid = np.zeros((nx, ny), dtype=bool)
        else:
            solid = np.asarray(solid, dtype=bool)
            if solid.shape != (nx, ny):
                raise ValueError("solid mask must have shape (nx, ny)")
            self.solid = solid.copy()
        # a momentum face is dead when either of its two cells is solid
        su = np.zeros((nx + 1, ny), dtype=bool)
        su[1:nx, :] = self.solid[:-1, :] | self.solid[1:, :]
        sv = np.zeros((nx, ny + 1), dtype=bool)
        sv[:, 1:ny] = self.solid[:, :-1] | self.solid[:, 1:]
        self.su, self.sv = su, sv

        self._lam_x = (2.0 * np.cos(np.pi * np.arange(nx) / nx) - 2.0) / self.h ** 2
        self._lam_y = (2.0 * np.cos(np.pi * np.arange(ny) / ny) - 2.0) / self.h ** 2
        self._lam_psi_x = (2.0 * np.cos(np.pi * (np.arange(nx) + 1) / (nx + 1))
                           - 2.0) / self.h ** 2
        self._lam_psi_y = (2.0 * np.cos(np.pi * (np.arange(ny) + 1) / (ny + 1))
                           - 2.0) / self.h ** 2

    # ------------------------------------------------------------- kinematics
    def divergence(self) -> np.ndarray:
        """Cell-centred discrete divergence (vanishes after projection)."""
        return ((self.u[1:, :] - self.u[:-1, :]) / self.h
                + (self.v[:, 1:] - self.v[:, :-1]) / self.h)

    def cell_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        return (0.5 * (self.u[:-1, :] + self.u[1:, :]),
                0.5 * (self.v[:, :-1] + self.v[:, 1:]))

    def vorticity(self) -> np.ndarray:
        """Cell-centred vorticity ``dv/dx - du/dy`` (wall values replicated)."""
        u_c, v_c = self.cell_velocity()
        du_dy = (u_c[:, 1:] - u_c[:, :-1]) / self.h
        dv_dx = (v_c[1:, :] - v_c[:-1, :]) / self.h
        du_dy = np.concatenate((du_dy[:, :1], du_dy, du_dy[:, -1:]), axis=1)
        dv_dx = np.concatenate((dv_dx[:1, :], dv_dx, dv_dx[-1:, :]), axis=0)
        return 0.5 * (dv_dx[1:, :] + dv_dx[:-1, :]) - 0.5 * (du_dy[:, 1:] + du_dy[:, :-1])

    def streamfunction(self) -> np.ndarray:
        """Cell-centred streamfunction with ``psi = 0`` on all four walls.

        Solves ``lap psi = -omega`` with Dirichlet data by an orthonormal DST-I
        diagonalisation; used for ``psi_max`` of the natural-convection cavity.
        """
        dst = sfft.dstn(-self.vorticity(), type=1, norm="ortho")
        lam = self._lam_psi_x[:, None] + self._lam_psi_y[None, :]
        return sfft.idstn(dst / lam, type=1, norm="ortho")

    # --------------------------------------------------------------- operators
    def _momentum(self, phi: np.ndarray, trans: np.ndarray, mask: np.ndarray,
                  wall_lo: float, wall_hi: float, trans_open: bool = False) -> np.ndarray:
        """Convection + diffusion tendency of the axis-0-normal face field phi.

        ``trans`` is the transverse face field, ``mask`` the dead-face mask with
        the shape of ``phi``; ``wall_lo`` / ``wall_hi`` are the tangential wall
        velocities at the two ends of axis 1.
        """
        n = phi.shape[0] - 1
        m = phi.shape[1]
        h, nu = self.h, self.nu
        conv = np.zeros_like(phi)

        # convection along axis 0 (transported and advecting quantity coincide)
        uc = 0.5 * (phi[:-1, :] + phi[1:, :])
        flux_x = uc * _vanleer_face(phi, uc, axis=0)
        conv[1:n, :] = -(flux_x[1:, :] - flux_x[:-1, :]) / h

        # convection along axis 1 (transported phi, advecting trans)
        t_face = 0.5 * (trans[:-1, :] + trans[1:, :])
        face_phi = _vanleer_face(phi[1:n, :], t_face[:, 1:m], axis=1)
        flux_y = np.zeros((n - 1, m + 1))
        flux_y[:, 1:m] = t_face[:, 1:m] * face_phi
        conv[1:n, :] -= (flux_y[:, 1:] - flux_y[:, :-1]) / h
        conv[mask] = 0.0

        # diffusion along axis 0: the end faces carry prescribed data
        lap = np.zeros_like(phi)
        lap[1:n, :] = (phi[:-2, :] - 2.0 * phi[1:-1, :] + phi[2:, :]) / h ** 2

        # diffusion along axis 1 with mirror / zero-gradient ghosts
        below = phi[:, 0] if trans_open else 2.0 * wall_lo - phi[:, 0]
        above = phi[:, -1] if trans_open else 2.0 * wall_hi - phi[:, -1]
        lap[:, 0] += (below - 2.0 * phi[:, 0] + phi[:, 1]) / h ** 2
        if m > 2:
            lap[:, 1:m - 1] += (phi[:, :-2] - 2.0 * phi[:, 1:-1] + phi[:, 2:]) / h ** 2
        lap[:, -1] += (phi[:, -2] - 2.0 * phi[:, -1] + above) / h ** 2
        lap[mask] = 0.0
        return conv + nu * lap

    def tendencies(self) -> tuple[np.ndarray, np.ndarray]:
        """Momentum right-hand sides without the pressure gradient."""
        tu = self._momentum(self.u, self.v, self.su, self.wall_lo, self.wall_hi)
        # the v-equation is the u-equation on transposed fields
        tv_t = self._momentum(self.v.T.copy(), self.u.T.copy(), self.sv.T,
                              0.0, 0.0, trans_open=self.outlet == "open")
        tv = tv_t.T.copy()
        if self.energy_on and self.rayleigh:
            tv[:, 1:self.ny] += (self.rayleigh * self.prandtl * 0.5
                                 * (self.theta[:, :-1] + self.theta[:, 1:]))
        return tu, tv

    def _energy_tendency(self) -> np.ndarray:
        nx, ny, h = self.nx, self.ny, self.h
        fx = np.zeros((nx + 1, ny))
        fx[1:nx, :] = self.u[1:nx, :] * _vanleer_face(
            self.theta, self.u[1:nx, :], axis=0)
        fy = np.zeros((nx, ny + 1))
        fy[:, 1:ny] = self.v[:, 1:ny] * _vanleer_face(
            self.theta, self.v[:, 1:ny], axis=1)
        div = (fx[1:, :] - fx[:-1, :]) / h + (fy[:, 1:] - fy[:, :-1]) / h
        # diffusion: Dirichlet on the vertical walls, adiabatic on the horizontal
        lap = (np.concatenate((self.theta[:1, :], self.theta[:-1, :]), 0)
               + np.concatenate((self.theta[1:, :], self.theta[-1:, :]), 0)
               + np.concatenate((self.theta[:, :1], self.theta[:, :-1]), 1)
               + np.concatenate((self.theta[:, 1:], self.theta[:, -1:]), 1)
               - 4.0 * self.theta) / h ** 2
        lap[0, :] += 2.0 * (self.theta_hot - self.theta[0, :]) / h ** 2
        lap[-1, :] -= 2.0 * (self.theta[-1, :] + self.theta_hot) / h ** 2
        return -div + lap

    # ---------------------------------------------------------------- pressure
    def solve_pressure(self, rhs: np.ndarray) -> np.ndarray:
        """Exact discrete Poisson solve for the pressure correction."""
        nx, ny, h = self.nx, self.ny, self.h
        if self.outlet == "wall" and self.inlet is None:
            centred = rhs - rhs.mean()
            coeff = sfft.dctn(centred, type=2, norm="ortho")
            lam = self._lam_x[:, None] + self._lam_y[None, :]
            spec = coeff / np.where(lam == 0.0, np.ones_like(lam), lam)
            spec[0, 0] = 0.0
            return sfft.idctn(spec, type=2, norm="ortho")
        # open domain: Neumann at x = 0, Dirichlet p = 0 at x = Lx
        a = 1.0 / h ** 2
        modes = sfft.dct(rhs, type=2, norm="ortho", axis=1)
        out = np.empty_like(modes)
        for k in range(ny):
            diag = np.full(nx, self._lam_y[k] - 2.0 * a)
            diag[0] = self._lam_y[k] - a
            diag[-1] = self._lam_y[k] - 3.0 * a
            ab = np.zeros((3, nx))
            ab[0, 1:] = a
            ab[1, :] = diag
            ab[2, :-1] = a
            out[:, k] = solve_banded((1, 1), ab, modes[:, k])
        return sfft.idct(out, type=2, norm="ortho", axis=1)

    # -------------------------------------------------------------- time step
    def stable_time_step(self) -> float:
        speed = max(float(np.abs(self.u).max()), float(np.abs(self.v).max()),
                    abs(self.wall_lo), abs(self.wall_hi), 1e-8)
        return min(self.cfl * self.h / speed, 0.25 * self.h ** 2 / self.nu)

    def step(self, dt: float | None = None) -> float:
        """One explicit projection step; returns the time step used."""
        nx, ny = self.nx, self.ny
        dt = self.stable_time_step() if dt is None else float(dt)
        tu, tv = self.tendencies()

        us = self.u.copy()
        vs = self.v.copy()
        us[1:nx, :] += dt * tu[1:nx, :]
        vs[:, 1:ny] += dt * tv[:, 1:ny]
        us[self.su] *= self.kappa
        vs[self.sv] *= self.kappa

        if self.inlet is not None:
            us[0, :] = self.inlet
        if self.inlet is None:
            us[nx, :] = 0.0
            vs[:, 0] = 0.0
            vs[:, ny] = 0.0

        div = ((us[1:, :] - us[:-1, :]) / self.h
               + (vs[:, 1:] - vs[:, :-1]) / self.h)
        self.p = self.solve_pressure(div / dt)

        us[1:nx, :] -= dt * (self.p[1:, :] - self.p[:-1, :]) / self.h
        vs[:, 1:ny] -= dt * (self.p[:, 1:] - self.p[:, :-1]) / self.h
        if self.outlet == "open":
            # ghost cell p(nx + 1/2) = -p(nx - 1/2) puts p = 0 on the exit face
            us[nx, :] += 2.0 * dt * self.p[-1, :] / self.h
        us[self.su] *= self.kappa
        vs[self.sv] *= self.kappa
        if self.inlet is not None:
            us[0, :] = self.inlet

        self.u = us
        self.v = vs
        if self.energy_on:
            self.theta = self.theta + dt * self._energy_tendency()
        self.steps += 1
        return dt

    def run_steady(self, *, tolerance: float = 1e-7, max_steps: int = 200_000,
                   max_time: float | None = None, check_every: int = 100,
                   on_check=None) -> FlowState:
        """March until the per-step change of the velocity falls below tol."""
        dt = 0.0
        time = 0.0
        residual = float("inf")
        history: list = []
        converged = False
        self.steps = 0
        for _ in range(max_steps):
            before, before_v = self.u.copy(), self.v.copy()
            dt = self.step()
            time += dt
            if self.steps % check_every == 0:
                change = max(float(np.abs(self.u - before).max()),
                             float(np.abs(self.v - before_v).max())) / dt
                residual = change
                history.append((self.steps, time, change))
                if on_check is not None:
                    on_check(self, self.steps, change)
                if change <= tolerance:
                    converged = True
                    break
            if max_time is not None and time >= max_time:
                break
        return FlowState(
            steps=self.steps,
            dt=dt,
            residual=residual,
            converged=converged,
            max_divergence=float(np.abs(self.divergence()).max()),
            solid_slip=float(max(np.abs(self.u[self.su]).max(initial=0.0),
                                 np.abs(self.v[self.sv]).max(initial=0.0))),
            cfl=self.cfl,
            history=history,
        )

    # ------------------------------------------------------------ diagnostics
    def x_coordinates(self) -> np.ndarray:
        """Cell-centre x coordinates."""
        return (np.arange(self.nx) + 0.5) * self.h

    def x_face_coordinates(self) -> np.ndarray:
        """x coordinates of the u faces (where the wall shear is sampled)."""
        return np.arange(self.nx + 1) * self.h

    def y_coordinates(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.h

    def save_state(self, path) -> None:
        np.savez_compressed(path, u=self.u, v=self.v, p=self.p, theta=self.theta)

    def load_state(self, path) -> None:
        """Warm start from a coarser/previous field snapshot (same grid)."""
        with np.load(path) as data:
            if data["u"].shape != self.u.shape:
                raise ValueError("warm-start field shape does not match the grid")
            self.u = data["u"].copy()
            self.v = data["v"].copy()
            self.p = data["p"].copy()
            if "theta" in data and self.energy_on:
                self.theta = data["theta"].copy()

    #: Wall-shear sign convention shared by both walls (single orientation):
    #: the returned array is the streamwise skin friction ``2 nu u_near / h``,
    #: where ``u_near`` is the first u-face row above the wall.  It is
    #: **positive in attached forward flow and negative inside a recirculation
    #: on both walls**, so a reversal test (``tau < 0``) can be applied to
    #: either wall without an orientation factor.  (Signing the two walls with
    #: opposite factors -- the pre-m6383 code returned ``-2 nu u/h`` on the
    #: lower and ``+2 nu u/h`` on the upper wall -- makes the same test report
    #: the *attached* region as reversed on one of them; see
    #: docs/verification/sudden-expansion-symmetry-audit/report.md.)
    def wall_shear_lower(self) -> np.ndarray:
        """Skin friction on the ``j = 0`` wall from the first u-face row.

        Positive = near-wall flow in ``+x``; see the convention note above.
        """
        return 2.0 * self.nu * self.u[:, 0] / self.h

    def wall_shear_upper(self) -> np.ndarray:
        """Skin friction on the ``j = ny`` wall from the last u-face row.

        Same convention as :meth:`wall_shear_lower` (positive forward), so a
        mirror-symmetric field gives identical arrays on the two walls.
        """
        return 2.0 * self.nu * self.u[:, -1] / self.h


# ---------------------------------------------------------------------------
# case definitions
# ---------------------------------------------------------------------------


def parabolic_profile(ny: int, height: float, offset: float, h: float,
                      mean_velocity: float = 1.0) -> np.ndarray:
    """Fully developed plane-Poiseuille profile sampled on the u-face rows.

    ``height`` is the channel height, ``offset`` the y of the lower wall and
    ``mean_velocity`` the bulk velocity; the maximum is ``1.5 * mean_velocity``.
    """
    y = (np.arange(ny) + 0.5) * h
    s = (y - offset) / height
    profile = 6.0 * mean_velocity * s * (1.0 - s)
    return np.where((s > 0.0) & (s < 1.0), profile, 0.0)


def backward_facing_step(re_h: float = 800.0, cells_per_step: int = 20, *,
                         upstream_length: float = 4.0, downstream_length: float = 12.0,
                         cfl: float = 0.25, kappa: float = 1.0e-4) -> StaggeredFlow:
    """Backward-facing step, expansion ratio 2, ``Re_H`` from the step height.

    The inlet channel occupies ``y in [H, 2H]`` for ``x < 0``; the step face
    sits at ``x = 0`` and the downstream channel is ``2H`` tall.  The mean inlet
    velocity is 1, so ``nu = H * U / Re_H``.
    """
    if cells_per_step < 4:
        raise ValueError("cells_per_step must be >= 4")
    h = 1.0 / cells_per_step
    nx = int(round((upstream_length + downstream_length) * cells_per_step))
    ny = int(round(2.0 * cells_per_step))
    x = (np.arange(nx) + 0.5) * h - upstream_length
    y = (np.arange(ny) + 0.5) * h
    solid = (x[:, None] < 0.0) & (y[None, :] < 1.0)
    inlet = parabolic_profile(ny, 1.0, 1.0, h, mean_velocity=1.0)
    return StaggeredFlow(nx, ny, h, nu=1.0 / re_h, inlet=inlet, outlet="open",
                         solid=solid, kappa=kappa, cfl=cfl)


def sudden_expansion(area_ratio: float = 4.0, re_d: float = 100.0,
                     cells_per_diameter: int = 20, *,
                     upstream_length: float = 3.0, downstream_length: float = 12.0,
                     cfl: float = 0.25, kappa: float = 1.0e-4) -> StaggeredFlow:
    """Planar sudden expansion (upstream height ``D``, downstream ``AR * D``)."""
    h = 1.0 / cells_per_diameter
    nx = int(round((upstream_length + downstream_length) * cells_per_diameter))
    ny = int(round(area_ratio * cells_per_diameter))
    x = (np.arange(nx) + 0.5) * h - upstream_length
    y = (np.arange(ny) + 0.5) * h
    half = 0.5 * (area_ratio - 1.0)
    solid = (x[:, None] < 0.0) & ((y[None, :] < half) | (y[None, :] > half + 1.0))
    inlet = parabolic_profile(ny, 1.0, half, h, mean_velocity=1.0)
    return StaggeredFlow(nx, ny, h, nu=1.0 / re_d, inlet=inlet, outlet="open",
                         solid=solid, kappa=kappa, cfl=cfl)


def natural_convection(rayleigh: float, cells: int = 64, *, prandtl: float = 0.71,
                       cfl: float = 0.5) -> StaggeredFlow:
    """Differentially heated square cavity (Boussinesq, ``H = 1``,\n    ``alpha = 1``)."""
    if cells < 4:
        raise ValueError("cells must be >= 4")
    h = 1.0 / cells
    return StaggeredFlow(cells, cells, h, nu=prandtl, outlet="wall",
                         energy=True, rayleigh=rayleigh, prandtl=prandtl,
                         cfl=cfl)


# ---------------------------------------------------------------------------
# post-processing / scoring
# ---------------------------------------------------------------------------


def reattachment_length(x: np.ndarray, tau: np.ndarray, start: float = 0.0,
                        primary: bool = True, min_attached: float = 1.0) -> float:
    """Downstream reattachment point from the wall-shear zero crossing.

    With ``primary=True`` the *main* recirculation is measured: the longest
    interval of reversed shear downstream of ``start`` is located and its
    downstream end (the last zero crossing of that interval) is returned, so a
    small counter-rotating corner eddy cannot be mistaken for the main bubble.
    ``primary=False`` returns the first crossing instead (the corner-eddy
    attachment, useful as a diagnostic).

    ``min_attached`` (in the units of ``x``) guards the outflow boundary: a
    crossing is only accepted when at least that much *re-attached* wall lies
    downstream of it.  The open-outlet pressure condition contaminates the last
    cells of a truncated domain and bends the near-wall profile back to positive
    there, which otherwise fabricates a "reattachment" wherever the bubble is
    still open (the m6253 sudden-expansion runs reported ``L_r/D = 11.86`` on a
    12 D domain from exactly that artefact).  ``nan`` therefore means *not
    resolved in this domain*, never "zero".
    """
    if x.size != tau.size:
        raise ValueError("x and tau must have the same length")
    idx = np.where(x >= start)[0]
    x, tau = x[idx], tau[idx]
    if primary:
        negative = tau < 0.0
        if not negative.any():
            return float("nan")
        # longest run of reversed shear
        edges = np.flatnonzero(np.diff(np.concatenate(([0], negative.view(np.int8), [0]))))
        spans = list(zip(edges[::2], edges[1::2]))
        i0, i1 = max(spans, key=lambda span: span[1] - span[0])  # noqa: F841
        i = i1 - 1
        if i + 1 >= x.size:                       # bubble still open at the exit
            return float("nan")
        x0, x1 = x[i], x[i + 1]
        t0, t1 = tau[i], tau[i + 1]
        x_attach = float(x0 - t0 * (x1 - x0) / (t1 - t0))
        if x[-1] - x_attach < min_attached:
            return float("nan")                   # crossing sits in the outflow zone
        return x_attach
    sign_change = np.where(np.sign(tau[:-1]) != np.sign(tau[1:]))[0]
    if sign_change.size == 0:
        return float("nan")
    i = int(sign_change[0])
    x0, x1 = x[i], x[i + 1]
    t0, t1 = tau[i], tau[i + 1]
    return float(x0 - t0 * (x1 - x0) / (t1 - t0))


def mean_nusselt(flow: StaggeredFlow) -> float:
    """Mean Nusselt number on the hot (``x = 0``) wall of the cavity."""
    flux = (flow.theta_hot - flow.theta[0, :]) / (0.5 * flow.h)
    return float(np.mean(flux))


def section_pressure(flow: StaggeredFlow, i: int) -> float:
    """Area-averaged static pressure in cell column ``i``."""
    return float(np.mean(flow.p[i, :]))


def loss_coefficient(flow: StaggeredFlow, i_in: int, i_out: int,
                     rho: float = 1.0, u_ref: float = 1.0) -> float:
    """``K = (p_in - p_out) / (0.5 rho u_ref^2)`` between two cell columns."""
    return (section_pressure(flow, i_in) - section_pressure(flow, i_out)) / (
        0.5 * rho * u_ref ** 2)


def total_pressure_flux(flow: StaggeredFlow, i: int, rho: float = 1.0) -> float:
    """Mass-flux weighted total pressure ``(p + rho |V|^2 / 2)`` in column ``i``."""
    u = flow.u[i, :]
    v = 0.5 * (flow.v[i, :-1] + flow.v[i, 1:])
    w = np.abs(u)
    if w.sum() <= 0.0:
        return float("nan")
    total = flow.p[i, :] + 0.5 * rho * (u ** 2 + v ** 2)
    return float(np.sum(w * total) / np.sum(w))


def vorticity_band(flow: StaggeredFlow) -> dict:
    """Peak vorticity and recirculating area downstream of the step/expansion."""
    w = flow.vorticity()
    return {
        "omega_min": float(w.min()),
        "omega_max": float(w.max()),
        "backflow_cells": int(np.sum(flow.u < 0.0)),
        "reversal_fraction": float(np.mean(flow.u < 0.0)),
    }