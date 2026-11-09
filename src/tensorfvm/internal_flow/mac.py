"""Staggered (MAC) finite-volume solver for steady laminar internal flow cases.

Why another 2-D incompressible solver
-------------------------------------
The verification chain in :mod:`tensorfvm.verification` is dominated by the
LBM reference solver and by the body-fitted/SIMPLE production stack.  The three
benchmarks of this package (backward-facing step, sudden expansion, buoyant
cavity) all have *published* reference numbers, so they need a solver that is
small enough to be audited line by line and that reproduces the classical
literature values with mesh refinement.  :class:`StaggeredMAC` is that solver:
a plain uniform Cartesian MAC discretisation with explicit time marching to a
steady state, sharing no code path with the LBM or SIMPLE solvers.

Discretisation
--------------
``nx x ny`` uniform cells over ``[0, Lx] x [0, Ly]`` with ``hx = Lx/nx`` and
``hy = Ly/ny``:

* ``u`` -- normal x-velocity on the ``nx + 1`` vertical faces, shape
  ``(nx + 1, ny)`` at ``y = (j + 1/2) hy``,
* ``v`` -- normal y-velocity on the ``ny + 1`` horizontal faces, shape
  ``(nx, ny + 1)`` at ``x = (i + 1/2) hx``,
* ``p`` -- cell-centred pressure, shape ``(nx, ny)``,
* ``theta`` -- cell-centred temperature (only when ``energy=True``).

Convection is written in conservative flux form with a van Leer (harmonic,
TVD) limited face reconstruction for both velocity components and for the
transported scalar.  Diffusion is a compact central difference; no-slip walls
enter through the usual mirror ghost ``u_ghost = 2 u_wall - u_0``, which also
gives the wall shear used for the friction coefficients.

Pressure
--------
The projection step solves ``lap(phi) = div(u*) / dt`` with homogeneous Neumann
conditions on every boundary, by an orthonormal DCT-II diagonalisation
(``torch`` matmuls, zero mode pinned).  Neumann everywhere requires the discrete
compatibility condition ``sum(div u*) = 0``; for a channel that is enforced by a
uniform correction of the outflow face column (global mass balance).  The
outflow pressure is therefore the standard zero-gradient outlet: it is the same
approximation as a Neumann pressure outlet but keeps one single fast solver for
the closed box and for the open channel.

Boundary options
----------------
``left``  : ``"wall"`` (no-slip, closed box) or ``"inlet"`` (prescribed ``u``
            from ``inlet_profile`` at the ``x = 0`` face; the rest of the
            inlet plane is a no-slip wall, which is how the step face of a
            backward-facing step or the face of a sudden expansion is
            represented),
``right`` : ``"wall"``, ``"outlet"`` (zero-gradient velocity + global flux
            balance), ``"developed"`` (the fully developed plane-Poiseuille
            profile at the exact inlet flux) or ``"convective"`` (first-order
            Orlanski convective outflow relaxation).  The three open closures
            all finish with the same global flux correction, so every one of
            them is mass consistent and the pressure Poisson problem keeps its
            compatibility; they differ only in what they assume about the
            downstream flow, which is what an outlet-sensitivity study has to
            separate,
``bottom``/``top`` are always no-slip walls.

Steady state is reached by explicit forward-Euler marching with a
CFL/viscous-limited time step.  This is the same integration scheme that
:mod:`tensorfvm.lid_cavity` uses for the driven cavity; the mixing of schemes
between benchmarks is intentional so that the classical values are reproduced
by an independent implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._dct import dct2, idct2

TINY = 1e-300

#: Right-hand boundary closures that let mass leave the domain.  ``"outlet"`` is
#: the zero-gradient column used by the published channel benchmarks,
#: ``"developed"`` imposes the analytic plane-Poiseuille profile, and
#: ``"convective"`` relaxes the outflow column towards the interior at the local
#: convective speed (first-order Orlanski).
OPEN_RIGHT = ("outlet", "developed", "convective")


def vanleer_face(f: torch.Tensor, vel: torch.Tensor, axis: int) -> torch.Tensor:
    """van Leer limited interface values of ``f`` at the faces along ``axis``.

    ``f`` holds ``M`` samples along ``axis`` and ``vel`` the ``M - 1`` normal
    velocities at the interfaces between them.  The limited slope is the
    harmonic mean ``2 a b / (a + b)`` of the upwind and local differences,
    zeroed where they change sign (TVD: no new extrema).
    """
    fa = f.movedim(axis, 0)
    va = vel.movedim(axis, 0)
    d = fa[1:] - fa[:-1]
    up = torch.empty_like(d)
    up[1:] = d[:-1]
    up[0] = d[0]
    down = torch.empty_like(d)
    down[:-1] = d[1:]
    down[-1] = d[-1]

    def limiter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        prod = a * b
        total = a + b
        safe = total.abs() > TINY
        value = 2.0 * prod / torch.where(safe, total, torch.ones_like(total))
        return torch.where(prod > 0.0, value, torch.zeros_like(value))

    forward = fa[:-1] + 0.5 * limiter(up, d)
    backward = fa[1:] - 0.5 * limiter(down, d)
    return torch.where(va >= 0.0, forward, backward).movedim(0, axis)


@dataclass(frozen=True)
class MACConfig:
    """Discretisation, boundary and solver controls for :class:`StaggeredMAC`."""

    nx: int
    ny: int
    length: float
    height: float
    viscosity: float
    density: float = 1.0
    reference_velocity: float = 1.0
    cfl: float = 0.4
    time_step: float | None = None
    max_steps: int = 400_000
    tolerance: float = 1e-9
    check_every: int = 100
    left: str = "wall"
    right: str = "wall"
    inlet_profile: tuple[float, ...] | None = None
    energy: bool = False
    diffusivity: float = 0.0
    buoyancy: float = 0.0
    temperature_left: float = 0.0
    temperature_right: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "max_steps", "check_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f"{name} must be an integer >= 4")
        for name in ("length", "height", "viscosity", "density", "reference_velocity", "cfl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if self.time_step is not None and (
                not math.isfinite(self.time_step) or self.time_step <= 0):
            raise ValueError("time_step must be a finite positive number")
        if self.left not in ("wall", "inlet"):
            raise ValueError("left must be wall or inlet")
        if self.right not in OPEN_RIGHT + ("wall",):
            raise ValueError("right must be wall, outlet, developed or convective")
        if self.right == "developed" and self.left != "inlet":
            raise ValueError("a developed outlet needs an inlet to set its flux")
        if self.left == "inlet":
            if self.inlet_profile is None or len(self.inlet_profile) != self.ny:
                raise ValueError("an inlet requires inlet_profile with ny entries")
            if not all(math.isfinite(value) for value in self.inlet_profile):
                raise ValueError("inlet_profile entries must be finite")
        elif self.inlet_profile is not None:
            raise ValueError("inlet_profile is only meaningful with left='inlet'")
        if self.energy:
            if not math.isfinite(self.diffusivity) or self.diffusivity <= 0:
                raise ValueError("energy transport requires a positive diffusivity")
        elif self.diffusivity:
            raise ValueError("diffusivity requires energy=True")
        if not math.isfinite(self.buoyancy):
            raise ValueError("buoyancy must be finite")


class StaggeredMAC:
    """Uniform Cartesian MAC solver (see the module docstring)."""

    def __init__(self, config: MACConfig) -> None:
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.hx = config.length / config.nx
        self.hy = config.height / config.ny
        self.nu = config.viscosity
        self.rho = config.density
        self.dtype = torch.float64
        self.device = torch.device(config.device)
        opts = {"dtype": self.dtype, "device": self.device}
        self.u = torch.zeros(self.nx + 1, self.ny, **opts)
        self.v = torch.zeros(self.nx, self.ny + 1, **opts)
        self.p = torch.zeros(self.nx, self.ny, **opts)
        self.theta = torch.zeros(self.nx, self.ny, **opts)
        if config.energy:
            self.theta.fill_(0.5 * (config.temperature_left + config.temperature_right))
        self.steps = 0
        self.residual = float("inf")
        self.dt = 0.0
        self._cx, self._lam_x = self._neumann_basis(self.nx, self.hx)
        self._cy, self._lam_y = self._neumann_basis(self.ny, self.hy)
        self._inlet = (torch.as_tensor(config.inlet_profile, **opts).clone()
                       if config.inlet_profile is not None else None)
        self.apply_boundary_conditions()

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _neumann_basis(n: int, h: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Orthonormal DCT-II matrix and Neumann eigenvalues of ``d2/dx2``."""
        k = torch.arange(n, dtype=torch.float64)[:, None]
        j = torch.arange(n, dtype=torch.float64)[None, :]
        mat = torch.cos(math.pi * k * (j + 0.5) / n)
        scale = torch.full((n, 1), math.sqrt(2.0 / n), dtype=torch.float64)
        scale[0] = math.sqrt(1.0 / n)
        lam = (2.0 * torch.cos(math.pi * torch.arange(n, dtype=torch.float64) / n) - 2.0) / h ** 2
        return mat * scale, lam

    def solve_pressure(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve ``lap p = rhs`` with homogeneous Neumann BCs on all sides.

        Diagonalisation by the orthonormal DCT-II basis of the cell-centred
        Neumann Laplacian; the transforms run through FFTs
        (:mod:`tensorfvm.internal_flow._dct`), the eigenvalues are separable.
        """
        centred = rhs - rhs.mean()
        coeff = dct2(dct2(centred, 0), 1)
        lam = self._lam_x[:, None] + self._lam_y[None, :]
        spec = coeff / torch.where(lam == 0.0, torch.ones_like(lam), lam)
        spec[0, 0] = 0.0
        return idct2(idct2(spec, 0), 1)

    def divergence(self) -> torch.Tensor:
        return ((self.u[1:, :] - self.u[:-1, :]) / self.hx
                + (self.v[:, 1:] - self.v[:, :-1]) / self.hy)

    def momentum_residual(self) -> dict[str, float]:
        """Discrete control-volume momentum residual of the steady equations.

        The projection step enforces ``u = u* - dt grad p`` with ``u*`` the
        explicit tendency, so the steady-state residual of the x-momentum
        equation at an interior u-face ``(i, j)`` is the tendency (convection
        plus diffusion, exactly the right-hand side of :meth:`_u_tendency`)
        plus the pressure gradient ``(p[i, j] - p[i-1, j]) / hx``.  The same
        holds for the v-faces with the y-gradient.  Boundary faces use the
        single adjacent cell (the pressure is cell-centred and the face is
        between the cell and the wall/outlet).

        The returned dictionary carries the maximum and the volume-averaged
        magnitude, each normalised by the convective scale
        ``U_ref^2 / min(hx, hy)`` so that the numbers are dimensionless and
        comparable across meshes and cases.  A converged steady solution has
        a residual at the level of the time-marching tolerance; the campaign
        acceptance gate is 1 % of the convective scale.
        """
        tu = self._u_tendency()
        tv = self._v_tendency()
        p = self.p
        # x-momentum at the solved u-faces i = 1 .. nx-1: the face sits between
        # cells i-1 and i, so the centred gradient is (p[i] - p[i-1]) / hx.
        grad_px = (p[1:, :] - p[:-1, :]) / self.hx          # (nx-1, ny)
        ru = tu[1:self.nx, :] - grad_px
        # y-momentum at the solved v-faces j = 1 .. ny-1
        grad_py = (p[:, 1:] - p[:, :-1]) / self.hy          # (nx, ny-1)
        rv = tv[:, 1:self.ny] - grad_py
        scale = self.config.reference_velocity ** 2 / min(self.hx, self.hy)
        return {
            "max": float(max(ru.abs().max(), rv.abs().max())) / scale,
            "mean": float(max(ru.abs().mean(), rv.abs().mean())) / scale,
            "x_max": float(ru.abs().max()) / scale,
            "y_max": float(rv.abs().max()) / scale,
            "scale": float(scale),
        }

    def apply_boundary_conditions(self) -> None:
        """Impose the non-evolving velocity/temperature boundary values."""
        cfg = self.config
        if cfg.left == "inlet":
            self.u[0, :] = self._inlet
        else:
            self.u[0, :] = 0.0
        if cfg.right == "wall":
            self.u[self.nx, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, self.ny] = 0.0

    # --------------------------------------------------------------- dynamics
    def _u_tendency(self) -> torch.Tensor:
        nx, ny, hx, hy, nu = self.nx, self.ny, self.hx, self.hy, self.nu
        u, v = self.u, self.v
        conv = torch.zeros_like(u)
        # x-convection between u nodes i and i+1 (i = 0 .. nx-1)
        ux = 0.5 * (u[:-1, :] + u[1:, :])
        fx = ux * vanleer_face(u, ux, axis=0)
        conv[1:nx, :] -= (fx[1:, :] - fx[:-1, :]) / hx
        # y-convection on the vertical faces 1 .. nx-1
        uau = 0.5 * (v[:-1, :] + v[1:, :])                 # (nx-1, ny+1)
        face_u = vanleer_face(u[1:nx, :], uau[:, 1:ny], axis=1)   # (nx-1, ny-1)
        fyv = torch.zeros(nx - 1, ny + 1, dtype=u.dtype, device=u.device)
        fyv[:, 1:ny] = uau[:, 1:ny] * face_u
        conv[1:nx, :] -= (fyv[:, 1:] - fyv[:, :-1]) / hy

        lap = torch.zeros_like(u)
        lap[1:nx, :] = (u[:-2, :] - 2.0 * u[1:-1, :] + u[2:, :]) / hx ** 2
        # no-slip mirror ghosts at y = 0 and y = Ly (tangential wall speed 0)
        lap[:, 0] += (-3.0 * u[:, 0] + u[:, 1]) / hy ** 2
        if ny > 2:
            lap[:, 1:ny - 1] += (u[:, :-2] - 2.0 * u[:, 1:-1] + u[:, 2:]) / hy ** 2
        lap[:, ny - 1] += (u[:, ny - 2] - 3.0 * u[:, ny - 1]) / hy ** 2
        return conv + nu * lap

    def _v_tendency(self) -> torch.Tensor:
        nx, ny, hx, hy, nu = self.nx, self.ny, self.hx, self.hy, self.nu
        u, v = self.u, self.v
        conv = torch.zeros_like(v)
        # y-convection between v nodes j and j+1 (j = 0 .. ny-1)
        vy = 0.5 * (v[:, :-1] + v[:, 1:])
        fy = vy * vanleer_face(v, vy, axis=1)
        conv[:, 1:ny] -= (fy[:, 1:] - fy[:, :-1]) / hy
        # x-convection on the horizontal faces 1 .. ny-1
        uau = torch.zeros(nx + 1, ny + 1, dtype=u.dtype, device=u.device)
        uau[:, 1:ny] = 0.5 * (u[:, :-1] + u[:, 1:])
        adv = uau[1:nx, :]                                 # (nx-1, ny+1)
        face_v = vanleer_face(v, adv, axis=0)              # (nx-1, ny+1)
        fxv = torch.zeros(nx + 1, ny + 1, dtype=u.dtype, device=u.device)
        fxv[1:nx, :] = adv * face_v
        if self.config.right != "wall":
            fxv[nx, :] = uau[nx, :] * v[nx - 1, :]          # upwind outflow flux
        conv -= (fxv[1:, :] - fxv[:-1, :]) / hx

        lap = torch.zeros_like(v)
        lap[:, 1:ny] = (v[:, :-2] - 2.0 * v[:, 1:-1] + v[:, 2:]) / hy ** 2
        # no-slip mirror ghost at x = 0 (wall, or the solid step face of an inlet plane)
        lap[0, :] += (-3.0 * v[0, :] + v[1, :]) / hx ** 2
        if nx > 2:
            lap[1:nx - 1, :] += (v[:-2, :] - 2.0 * v[1:-1, :] + v[2:, :]) / hx ** 2
        if self.config.right != "wall":
            lap[nx - 1, :] += (v[nx - 2, :] - v[nx - 1, :]) / hx ** 2   # zero gradient
        else:
            lap[nx - 1, :] += (v[nx - 2, :] - 3.0 * v[nx - 1, :]) / hx ** 2
        tendency = conv + nu * lap
        if self.config.buoyancy:
            source = torch.zeros_like(v)
            source[:, 1:ny] = 0.5 * (self.theta[:, :-1] + self.theta[:, 1:])
            tendency = tendency + self.config.buoyancy * source
        return tendency

    def _theta_tendency(self) -> torch.Tensor:
        nx, ny, hx, hy = self.nx, self.ny, self.hx, self.hy
        alpha = self.config.diffusivity
        theta = self.theta
        conv = torch.zeros_like(theta)
        if nx > 1:
            face_x = vanleer_face(theta, self.u[1:nx, :], axis=0)   # (nx-1, ny)
            fx = torch.zeros(nx + 1, ny, dtype=theta.dtype, device=theta.device)
            fx[1:nx, :] = self.u[1:nx, :] * face_x
            conv -= (fx[1:, :] - fx[:-1, :]) / hx
        if ny > 1:
            face_y = vanleer_face(theta, self.v[:, 1:ny], axis=1)   # (nx, ny-1)
            fy = torch.zeros(nx, ny + 1, dtype=theta.dtype, device=theta.device)
            fy[:, 1:ny] = self.v[:, 1:ny] * face_y
            conv -= (fy[:, 1:] - fy[:, :-1]) / hy

        lap = torch.zeros_like(theta)
        # Dirichlet walls at x = 0 and x = Lx, adiabatic (Neumann) at y = 0, Ly
        left_ghost = 2.0 * self.config.temperature_left - theta[0, :]
        lap[0, :] += (left_ghost - 2.0 * theta[0, :] + theta[1, :]) / hx ** 2
        if nx > 2:
            lap[1:nx - 1, :] += (theta[:-2, :] - 2.0 * theta[1:-1, :] + theta[2:, :]) / hx ** 2
        right_ghost = 2.0 * self.config.temperature_right - theta[nx - 1, :]
        lap[nx - 1, :] += (theta[nx - 2, :] - 2.0 * theta[nx - 1, :] + right_ghost) / hx ** 2
        lap[:, 0] += (theta[:, 1] - theta[:, 0]) / hy ** 2
        if ny > 2:
            lap[:, 1:ny - 1] += (theta[:, :-2] - 2.0 * theta[:, 1:-1] + theta[:, 2:]) / hy ** 2
        lap[:, ny - 1] += (theta[:, ny - 2] - theta[:, ny - 1]) / hy ** 2
        return conv + alpha * lap

    def tendencies(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Momentum (and temperature) right-hand sides without pressure."""
        tt = self._theta_tendency() if self.config.energy else None
        return self._u_tendency(), self._v_tendency(), tt

    # ------------------------------------------------------------- time step
    def stable_time_step(self) -> float:
        cfg = self.config
        if cfg.time_step is not None:
            return float(cfg.time_step)
        speed = max(float(self.u.abs().max()), float(self.v.abs().max()),
                    cfg.reference_velocity)
        h = min(self.hx, self.hy)
        conv = cfg.cfl * h / speed
        diff = self.nu if not cfg.energy else max(self.nu, cfg.diffusivity)
        # 0.18 instead of the textbook 0.25: the mirror-ghost wall treatment raises
        # the near-wall diagonal to 3 nu/h^2, which lowers the explicit diffusion
        # limit; a straight channel at h^2/(4 nu) is measurably unstable
        # (tests/test_internal_flow_mac.py pins the stable branch).
        visc = 0.18 * h ** 2 / diff
        return min(conv, visc)

    def developed_profile(self) -> torch.Tensor:
        """Plane-Poiseuille profile at the face column, scaled to the inlet flux.

        ``u(eta) = 6 Ubar eta (1 - eta)`` with ``eta = (j + 1/2) / ny`` and
        ``Ubar`` taken so that the *discrete* sum of the column equals the inlet
        flux; the analytic quadrature of the parabola is only exact in the
        continuum, so the discrete rescaling is what makes the closure mass
        consistent on every mesh.
        """
        if self._inlet is None:
            raise ValueError("a developed outlet needs an inlet profile")
        eta = (torch.arange(self.ny, dtype=self.dtype, device=self.device) + 0.5) / self.ny
        shape = 6.0 * eta * (1.0 - eta)
        target = self._inlet.sum()
        return shape * (target / shape.sum())

    def _outlet_column(self, u_out: torch.Tensor, u_in: torch.Tensor,
                       u_inlet: torch.Tensor, dt: float) -> torch.Tensor:
        """Close the outflow face column and enforce the global flux balance."""
        cfg = self.config
        if cfg.right == "developed":
            u_out = self.developed_profile()
        elif cfg.right == "convective":
            # first-order Orlanski: d u / dt + c d u / dx = 0 at the local speed c
            speed = float(torch.clamp(u_in, min=0.0).max())
            courant = speed * dt / self.hx
            u_out = (u_out + courant * u_in) / (1.0 + courant)
        # global mass balance: the Neumann pressure solve needs sum(div u*) = 0
        deficit = (u_inlet.sum() - u_out.sum()) / self.ny
        return u_out + deficit

    def step(self, dt: float | None = None) -> float:
        """Advance one explicit projection step and return the step size used."""
        dt = self.stable_time_step() if dt is None else float(dt)
        cfg = self.config
        tu, tv, tt = self.tendencies()

        u_star = self.u.clone()
        v_star = self.v.clone()
        u_star[1:self.nx, :] += dt * tu[1:self.nx, :]
        v_star[:, 1:self.ny] += dt * tv[:, 1:self.ny]
        if cfg.left == "inlet":
            u_star[0, :] = self._inlet
        else:
            u_star[0, :] = 0.0
        v_star[:, 0] = 0.0
        v_star[:, self.ny] = 0.0
        theta_star = self.theta
        if tt is not None:
            theta_star = self.theta + dt * tt

        if cfg.right == "wall":
            u_star[self.nx, :] = 0.0
        else:
            u_star[self.nx, :] = self._outlet_column(u_star[self.nx, :],
                                                    u_star[self.nx - 1, :],
                                                    u_star[0, :], dt)

        div = ((u_star[1:, :] - u_star[:-1, :]) / self.hx
               + (v_star[:, 1:] - v_star[:, :-1]) / self.hy)
        p = self.solve_pressure(div / dt)

        self.p = p
        self.u[1:self.nx, :] = u_star[1:self.nx, :] - dt * (p[1:, :] - p[:-1, :]) / self.hx
        self.v[:, 1:self.ny] = v_star[:, 1:self.ny] - dt * (p[:, 1:] - p[:, :-1]) / self.hy
        self.u[self.nx, :] = u_star[self.nx, :]
        self.u[0, :] = u_star[0, :]
        self.v[:, 0] = 0.0
        self.v[:, self.ny] = 0.0
        if tt is not None:
            self.theta = theta_star
        self.dt = dt
        self.steps += 1
        return dt

    def run(self, *, max_steps: int | None = None, tolerance: float | None = None,
            check_every: int | None = None, on_check=None) -> dict[str, object]:
        """March to steady state and return a convergence ledger."""
        cfg = self.config
        max_steps = cfg.max_steps if max_steps is None else int(max_steps)
        tolerance = cfg.tolerance if tolerance is None else float(tolerance)
        check_every = cfg.check_every if check_every is None else int(check_every)
        history: list[tuple[int, float]] = []
        self.residual = float("inf")
        for _ in range(max_steps):
            sample = (self.steps + 1) % check_every == 0
            before_u = self.u.clone() if sample else None
            before_v = self.v.clone() if sample else None
            dt = self.step()
            if sample:
                change = max(float((self.u - before_u).abs().max()),
                             float((self.v - before_v).abs().max())) / dt
                self.residual = change
                history.append((self.steps, change))
                if on_check is not None:
                    on_check(self, self.steps, change)
                if change <= tolerance:
                    break
        return {
            "steps": self.steps,
            "dt": self.dt,
            "residual": self.residual,
            "converged": self.residual <= tolerance,
            "history": history,
            "max_divergence": float(self.divergence().abs().max()),
        }

    # ------------------------------------------------------------- sequencing
    def prolong_from(self, coarse: "StaggeredMAC") -> None:
        """Initialise this mesh from the steady state of a coarser mesh.

        Each finer mesh of a grid-convergence study is started from the
        bilinearly indexed (nearest-node) prolongation of the next coarser
        solution.  Only the initial transient is removed: the converged
        solution is whatever the fine-mesh equations produce, so this changes
        the cost of a study, not its answer.
        """
        if abs((coarse.hx * coarse.nx) - (self.hx * self.nx)) > 1e-12 * self.hx * self.nx:
            raise ValueError("the two meshes must cover the same streamwise length")
        if abs((coarse.hy * coarse.ny) - (self.hy * self.ny)) > 1e-12 * self.hy * self.ny:
            raise ValueError("the two meshes must cover the same height")

        def index(fine_count: int, fine_h: float, coarse_count: int, coarse_h: float,
                  shifted: bool) -> torch.Tensor:
            positions = (torch.arange(fine_count, dtype=torch.float64) + (0.5 if shifted else 0.0)) * fine_h
            if shifted:
                values = positions / coarse_h - 0.5
            else:
                values = positions / coarse_h
            values = torch.round(values).long().clamp(0, coarse_count - 1 + (0 if shifted else 1))
            return values

        ix_u = index(self.nx + 1, self.hx, coarse.nx, coarse.hx, False)
        iy_u = index(self.ny, self.hy, coarse.ny, coarse.hy, True)
        self.u = coarse.u.index_select(0, ix_u).index_select(1, iy_u).clone()
        ix_v = index(self.nx, self.hx, coarse.nx, coarse.hx, True)
        iy_v = index(self.ny + 1, self.hy, coarse.ny, coarse.hy, False)
        self.v = coarse.v.index_select(0, ix_v).index_select(1, iy_v).clone()
        ix_c = index(self.nx, self.hx, coarse.nx, coarse.hx, True)
        iy_c = index(self.ny, self.hy, coarse.ny, coarse.hy, True)
        self.theta = coarse.theta.index_select(0, ix_c).index_select(1, iy_c).clone()
        self.apply_boundary_conditions()

    # ------------------------------------------------------------ diagnostics
    def cell_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell-centred ``(u, v)`` from the face values."""
        u = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        v = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        return u, v

    def cell_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = (torch.arange(self.nx, dtype=self.dtype, device=self.device) + 0.5) * self.hx
        y = (torch.arange(self.ny, dtype=self.dtype, device=self.device) + 0.5) * self.hy
        return x, y

    def streamfunction(self) -> torch.Tensor:
        """Streamfunction on the nodes, ``u = dpsi/dy`` and ``v = -dpsi/dx``."""
        u = self.u
        psi = torch.zeros(self.nx + 1, self.ny + 1, dtype=self.dtype, device=self.device)
        psi[:, 1:] = torch.cumsum(self.hy * u, dim=1)
        # remove the (constant) inlet/outlet offset so psi=0 on the bottom wall
        psi = psi - psi[:, 0:1]
        return psi

    def bottom_wall_friction(self) -> torch.Tensor:
        """Fanning friction coefficient along the bottom wall at each vertical face."""
        tau = self.rho * self.nu * 2.0 * self.u[:, 0] / self.hy
        return tau / (0.5 * self.rho * self.config.reference_velocity ** 2)

    def reattachment_length(self) -> float | None:
        """Streamwise distance from the inlet plane to the bottom-wall reattachment.

        The skin friction on the bottom wall is negative inside the primary
        recirculation bubble; the reattachment point is the first interior
        zero crossing after the step, interpolated linearly.  ``None`` means no
        sign change was found on the wall.
        """
        cf = self.bottom_wall_friction()
        for i in range(1, self.nx):
            if cf[i - 1] < 0.0 <= cf[i]:
                x0, x1 = (i - 1) * self.hx, i * self.hx
                w = -cf[i - 1] / (cf[i] - cf[i - 1])
                return float(x0 + w * (x1 - x0))
        return None

    def sample_u_profile(self, x: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate the ``u`` profile onto the cell-row centres at station ``x``."""
        i = int(round(x / self.hx))
        i = max(0, min(self.nx, i))
        y = (torch.arange(self.ny, dtype=self.dtype, device=self.device) + 0.5) * self.hy
        return y, self.u[i, :].clone()

    def flux_weighted_total_pressure(self, column: int) -> float:
        """Mass-flux weighted total pressure over the vertical face column ``column``.

        Only the open part of the face (``u > 0``) contributes, so the same
        expression works for the inlet plane of a step (part of it is a wall)
        and for a full outlet plane.
        """
        u = self.u[column, :]
        p = self.p[column - 1 if column > 0 else 0, :]
        w = torch.clamp(u, min=0.0)
        total = w.sum()
        if float(total) <= 0.0:
            raise ValueError("the selected face column carries no forward flow")
        pt = p + 0.5 * self.rho * u ** 2
        return float((w * pt).sum() / total)