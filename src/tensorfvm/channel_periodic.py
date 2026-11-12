"""Streamwise-periodic plane channel flow: staggered MAC kernel with wall friction.

Why this module exists
----------------------
Before this module the tree could not run the classical wall-bounded verification
case in its *native* form.  ``solver.SimpleSolver`` / ``inc_bench.solver`` drive
the channel through an inlet profile plus a zero-gradient outlet, and
``periodic_mac.PeriodicMACSolver`` is periodic in **all three** directions, so it
has no wall at all.  The textbook case is the opposite split:

* streamwise (``x``) direction **periodic** -- no inflow, no outflow, so the mass
  flux is not imposed by a boundary condition and cannot be tuned,
* wall-normal (``y``) direction **no-slip walls** at ``y=0`` and ``y=H``,
* driven by a constant streamwise body force ``G`` (the kinematic stand-in for the
  mean pressure gradient, ``G = -1/rho dp/dx > 0``).

Both the bulk velocity and the wall shear are therefore *outputs* of the discrete
momentum balance.  That is what makes the case a real test: a solver that leaks
momentum through the periodic wrap, or whose projection does not reach machine
divergence, cannot produce the analytic profile.

Exact solution (every relation is re-derived symbolically by ``symbolic_checks``
before the benchmark quotes a target number):::

    u(y)  = G/(2 nu) * y * (H - y) = 4 Umax (y/H)(1 - y/H) = 6 Ubar (y/H)(1 - y/H)
    Umax  = (3/2) Ubar,      Ubar = G H^2 / (12 nu)
    dp/dx = -rho G,          tau_w = rho G H / 2 = 6 mu Ubar / H
    Po    = f_Darcy * Re_Dh = 96       (parallel plates; circular tube is 64)

Numerics
--------
Staggered MAC placement, second-order central fluxes, arrays in ``(y, x)``:

* ``u`` on the ``nx`` x-faces, ``v`` on the ``ny+1`` y-faces, ``p`` at cell
  centres.  ``x`` wraps (``u[:,-1] -> u[:,0]``), ``v`` is pinned to zero on both
  walls so no wall-normal flux can cross them.
* Diffusion is treated **implicitly** and solved exactly for the discrete
  operator: the ``x`` second difference is circulant, so an FFT diagonalises it
  (eigenvalue ``-4 sin^2(k dx/2)/dx^2``), leaving one tridiagonal system per
  Fourier mode in ``y``.  The result is the exact solution of the implicit
  system, not a split approximation.
* Convection is explicit and in flux form (the transported component is averaged
  onto the advecting face, the ``periodic_mac`` convention).
* The projection solves ``D G p = D u* / dt`` with the *same* discrete
  divergence/gradient pair the momentum step uses, periodic in ``x`` (FFT) and
  Neumann at the walls (``p_ghost = p``, which keeps the wall-normal correction
  -- and therefore ``v = 0`` on the wall -- intact).
* The wall flux of ``u`` is selectable: ``half-cell`` (standard first-order
  distance-to-wall treatment, default) or ``quadratic`` (three-point one-sided,
  second order).  The benchmark measures both: the default keeps a genuine
  truncation error to study, the quadratic variant reaches round-off on this
  parabolic solution.
* The walls may move in the streamwise direction (``step(wall_velocity_m_s=...)``
  / :meth:`PeriodicChannelSolver.set_wall_velocity`).  A non-homogeneous no-slip
  wall is an inhomogeneous ghost value -- ``half-cell``:
  ``u_ghost = 2 u_w - u_0``, ``quadratic``: ``u_ghost = -2 u_0 + u_1/3 + (8/3) u_w``
  (both reduce to the homogeneous formulas above at ``u_w = 0``) -- so the wall
  contribution is a per-step source ``coef * u_w / dy**2`` on the two wall rows of
  the implicit system, with ``coef = 2 nu`` (half-cell) or ``(8/3) nu``
  (quadratic).  ``wall_shear`` / ``dissipation`` subtract the same ``u_w``, which
  keeps the discrete momentum budget closed.  This is what Stokes' second problem
  (oscillating wall) needs; see ``verification.stokes_second_problem``.

Nothing here is a placeholder: every operator below is used by ``step`` /
``solve_steady`` and exercised by ``tests/test_channel_periodic.py``.
"""
from dataclasses import dataclass
import math

import torch

__all__ = [
    "PeriodicChannelConfig",
    "PeriodicChannelSolver",
    "analytic_reference",
    "profile_at",
    "symbolic_checks",
    "WALL_FLUX_CHOICES",
]

WALL_FLUX_CHOICES = ("half-cell", "quadratic")


@dataclass(frozen=True)
class PeriodicChannelConfig:
    """Physical and numerical parameters of the periodic plane channel.

    ``body_force_m_s2`` is the imposed streamwise acceleration ``G``; it equals
    ``-1/rho dp/dx``.  ``viscosity_pa_s`` is the dynamic viscosity ``mu`` and
    ``density_kg_m3`` the density, so ``nu = mu/rho``.
    """

    nx: int = 32
    ny: int = 64
    length_m: float = 2.0
    height_m: float = 1.0
    density_kg_m3: float = 1.0
    viscosity_pa_s: float = 0.01
    body_force_m_s2: float = 0.12
    time_step_s: float = 0.5
    wall_flux: str = "half-cell"
    device: str = "cpu"

    def __post_init__(self):
        for name in ("nx", "ny"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 4:
                raise ValueError(f"invalid {name}: need int >= 4")
        for name in ("length_m", "height_m", "density_kg_m3", "viscosity_pa_s",
                     "body_force_m_s2", "time_step_s"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"invalid {name}: need finite > 0")
        if self.wall_flux not in WALL_FLUX_CHOICES:
            raise ValueError(f"wall_flux must be one of {WALL_FLUX_CHOICES}")

    @property
    def dx(self):
        return self.length_m / self.nx

    @property
    def dy(self):
        return self.height_m / self.ny

    @property
    def nu(self):
        return self.viscosity_pa_s / self.density_kg_m3

    def with_body_force_for_bulk(self, bulk_velocity):
        """Copy of the config whose body force implies ``bulk_velocity``."""
        g = 12 * self.nu * float(bulk_velocity) / self.height_m ** 2
        return PeriodicChannelConfig(**{**self.__dict__, "body_force_m_s2": g})

    def with_re_h(self, re_h):
        """Copy of the config whose body force yields ``Re_H = Ubar H / nu``."""
        return self.with_body_force_for_bulk(float(re_h) * self.nu / self.height_m)


def analytic_reference(config, bulk_velocity=None):
    """Exact laminar plane-channel solution for ``config``.

    All entries are algebraic consequences of ``nu u'' = -G`` with
    ``u(0) = u(H) = 0``; ``symbolic_checks`` re-derives them with sympy so no
    number here is a hand-copied table value.
    """
    h = config.height_m
    nu = config.nu
    rho = config.density_kg_m3
    g = config.body_force_m_s2
    ubar = g * h ** 2 / (12 * nu) if bulk_velocity is None else float(bulk_velocity)
    umax = 1.5 * ubar
    tau = rho * g * h / 2.0
    re_h = ubar * h / nu
    re_dh = 2.0 * re_h
    f_darcy = 8.0 * tau / (rho * ubar ** 2)
    return dict(
        bulk_velocity_m_s=ubar,
        peak_velocity_m_s=umax,
        peak_over_bulk=umax / ubar,
        dpdx_pa_m=-rho * g,
        tau_w_pa=tau,
        cf_fanning=tau / (0.5 * rho * ubar ** 2),
        f_darcy=f_darcy,
        re_h=re_h,
        re_dh=re_dh,
        poiseuille=f_darcy * re_dh,
        profile="u(y) = 6 Ubar (y/H)(1 - y/H) = G/(2 nu) y (H - y)",
    )


def profile_at(config, y, bulk_velocity=None):
    """Analytic streamwise velocity at wall-normal positions ``y``."""
    h = config.height_m
    yy = torch.as_tensor(y, dtype=torch.float64)
    if bulk_velocity is not None:
        yn = yy / h
        return 6.0 * float(bulk_velocity) * yn * (1 - yn)
    return config.body_force_m_s2 / (2 * config.nu) * yy * (h - yy)


def symbolic_checks(config):
    """Re-derive the quoted targets with sympy; returns ``{name: bool}``.

    A report may only quote a target number when the matching check is ``True``.
    """
    import sympy as sp

    y, h, mu, rho, g = sp.symbols("y H mu rho G", positive=True)
    nu = mu / rho            # the constitutive relation the solver actually uses
    u = g / (2 * nu) * y * (h - y)
    checks = {}
    checks["profile_solves_stokes_equation"] = sp.simplify(nu * sp.diff(u, y, 2) + g) == 0
    checks["profile_satisfies_both_walls"] = (sp.simplify(u.subs(y, 0)) == 0
                                              and sp.simplify(u.subs(y, h)) == 0)
    ubar = sp.integrate(u, (y, 0, h)) / h
    umax = u.subs(y, h / 2)
    checks["peak_over_bulk_is_three_halves"] = sp.simplify(umax / ubar - sp.Rational(3, 2)) == 0
    tau = sp.simplify(mu * sp.diff(u, y).subs(y, 0))
    checks["wall_shear_equals_half_body_force_times_height"] = sp.simplify(tau - rho * g * h / 2) == 0
    checks["wall_shear_equals_six_mu_ubar_over_h"] = sp.simplify(tau - 6 * mu * ubar / h) == 0
    checks["nu_is_mu_over_rho"] = sp.simplify(nu - mu / rho) == 0
    re_dh = 2 * ubar * h / nu
    f = 8 * tau / (rho * ubar ** 2)
    checks["poiseuille_is_96"] = sp.simplify(f * re_dh - 96) == 0
    checks["bulk_velocity_is_g_h2_over_12_nu"] = sp.simplify(ubar - g * h ** 2 / (12 * nu)) == 0
    ref = analytic_reference(config)
    checks["quoted_peak_over_bulk"] = abs(ref["peak_over_bulk"] - 1.5) < 1e-12
    checks["quoted_poiseuille"] = abs(ref["poiseuille"] - 96.0) < 1e-9
    return checks


class PeriodicChannelSolver:
    """Semi-implicit staggered MAC solver, ``x`` periodic, ``y`` wall bounded."""

    def __init__(self, config, device=None):
        self.config = config
        self.device = torch.device(device or config.device)
        cfg = config
        self.u = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float64, device=self.device)
        self.v = torch.zeros((cfg.ny + 1, cfg.nx), dtype=torch.float64, device=self.device)
        self.pressure = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float64, device=self.device)
        k = 2 * math.pi * torch.fft.fftfreq(cfg.nx, d=cfg.dx, device=self.device, dtype=torch.float64)
        self.lam_x = -4.0 * torch.sin(k * cfg.dx / 2.0) ** 2 / cfg.dx ** 2
        self.time_step_s = float(cfg.time_step_s)   # mutable: pseudo-time acceleration
        self.wall_velocity_m_s = (0.0, 0.0)         # (lower, upper) moving-wall speeds
        self.history = []
        self.time = 0.0

    # -- geometry -----------------------------------------------------------
    def wall_coordinates(self):
        cfg = self.config
        yc = (torch.arange(cfg.ny, device=self.device, dtype=torch.float64) + 0.5) * cfg.dy
        yf = torch.arange(cfg.ny + 1, device=self.device, dtype=torch.float64) * cfg.dy
        xc = (torch.arange(cfg.nx, device=self.device, dtype=torch.float64) + 0.5) * cfg.dx
        return yc, yf, xc

    # -- discrete operators -------------------------------------------------
    def divergence(self):
        """Cell divergence from face fluxes; periodic wrap in x, walls in y."""
        cfg = self.config
        du = (torch.roll(self.u, -1, dims=-1) - self.u) / cfg.dx
        dv = (self.v[1:] - self.v[:-1]) / cfg.dy
        return du + dv

    def gradient_cell_to_u(self, pressure=None):
        """``dp/dx`` at x-faces from cell-centred pressure (periodic wrap)."""
        cfg = self.config
        p = self.pressure if pressure is None else pressure
        return (p - torch.roll(p, 1, dims=-1)) / cfg.dx

    def gradient_cell_to_v(self, pressure=None):
        """``dp/dy`` at y-faces; Neumann at both walls (``p_ghost = p``)."""
        cfg = self.config
        p = self.pressure if pressure is None else pressure
        lower = torch.cat([p[:1], p], dim=0)
        upper = torch.cat([p, p[-1:]], dim=0)
        return (upper - lower) / cfg.dy

    def _u_on_y_faces(self):
        """``u`` averaged onto y-faces; zero on both walls."""
        u = self.u
        mid = 0.5 * (u[:-1] + u[1:])
        zero = torch.zeros_like(u[:1])
        return torch.cat([zero, mid, zero], dim=0)

    def convection_u(self):
        """Flux-form convective term for ``u`` on x-faces."""
        cfg = self.config
        uc = 0.5 * (self.u + torch.roll(self.u, -1, dims=-1))
        duu = (uc ** 2 - torch.roll(uc ** 2, 1, dims=-1)) / cfg.dx
        vx = 0.5 * (self.v + torch.roll(self.v, -1, dims=-1))
        ut = self._u_on_y_faces()
        dvu = (vx[1:] * ut[1:] - vx[:-1] * ut[:-1]) / cfg.dy
        return duu + dvu

    def convection_v(self):
        """Flux-form convective term for ``v`` on interior y-faces."""
        cfg = self.config
        vc = 0.5 * (self.v[:-1] + self.v[1:])            # v at cell centres
        dvv = torch.zeros((cfg.ny + 1, cfg.nx), dtype=self.v.dtype, device=self.device)
        dvv[1:-1] = (vc[1:] ** 2 - vc[:-1] ** 2) / cfg.dy   # walls never use these rows
        vx = 0.5 * (self.v + torch.roll(self.v, 1, dims=-1))     # v on x-faces
        u = self.u
        zero = torch.zeros_like(u[:1])
        uy = torch.cat([zero, 0.5 * (u[:-1] + u[1:]), zero], dim=0)   # u on y-faces
        flux = uy * vx
        duv = (flux - torch.roll(flux, 1, dims=-1)) / cfg.dx
        return dvv + duv

    # -- batched tridiagonal solve -----------------------------------------
    @staticmethod
    def _tridiagonal_solve(lower, diag, upper, rhs):
        """Solve one tridiagonal system per column (arrays ``(n, m)``) through LAPACK.

        Same system as :meth:`_thomas`, assembled as a batch of ``m`` dense
        ``n x n`` matrices and handed to ``torch.linalg.solve``.  The banded
        forward/back substitution below is ``O(n)`` *python-level* kernel
        launches, which dominated the step cost on small grids (measured 12 ms
        per call at ``n = 64``) and made unsteady campaigns -- tens of
        thousands of steps, e.g. ``verification.stokes_second_problem`` --
        needlessly expensive.  The dense route costs ``O(m n^3)`` flops but a
        handful of launches: 0.9 ms per call at ``n = 64``.  The two agree to
        round-off; ``tests/test_channel_periodic_tridiagonal.py`` checks that
        directly.  Measured crossover is near ``n ~ 300`` (``n = 256``: 34 ms
        dense vs 49 ms banded; ``n = 512``: 235 ms vs 107 ms), so tall systems
        keep the banded route.
        """
        n, m = diag.shape
        if n > 256:
            return PeriodicChannelSolver._thomas(lower, diag, upper, rhs)
        mat = torch.diag_embed(diag.transpose(0, 1).to(rhs.dtype))
        if n > 1:
            rows = torch.arange(n, device=diag.device)
            mat[:, rows[:-1], rows[1:]] = upper.transpose(0, 1).to(rhs.dtype)[:, :-1]
            mat[:, rows[1:], rows[:-1]] = lower.transpose(0, 1).to(rhs.dtype)[:, 1:]
        solved = torch.linalg.solve(mat, rhs.transpose(0, 1).unsqueeze(-1))
        return solved.squeeze(-1).transpose(0, 1)

    @staticmethod
    def _thomas(lower, diag, upper, rhs):
        """Reference banded solve, one tridiagonal system per column ``(n, m)``.

        Kept as the independent implementation the vectorised
        :meth:`_tridiagonal_solve` is checked against (see its docstring).
        """
        n = diag.shape[0]
        cp = torch.empty_like(diag)
        dp = torch.empty_like(rhs)
        cp[0] = upper[0] / diag[0]
        dp[0] = rhs[0] / diag[0]
        for j in range(1, n):
            m = diag[j] - lower[j] * cp[j - 1]
            cp[j] = upper[j] / m
            dp[j] = (rhs[j] - lower[j] * dp[j - 1]) / m
        out = torch.empty_like(rhs)
        out[-1] = dp[-1]
        for j in range(n - 2, -1, -1):
            out[j] = dp[j] - cp[j] * out[j + 1]
        return out

    def _implicit_diffusion(self, rhs, boundary, wall_velocity=None):
        """Solve ``(I/dt - nu L) phi = rhs`` exactly for the discrete operator.

        ``boundary='ghost'``    : wall at half a cell (``u``), ``phi_ghost = -phi``.
        ``boundary='dirichlet'``: the walls are faces carrying ``phi = 0`` (``v``).
        """
        cfg = self.config
        nu, dt, n, nx = cfg.nu, self.time_step_s, cfg.ny, cfg.nx
        wall_velocity = ((0.0, 0.0) if wall_velocity is None
                         else (float(wall_velocity[0]), float(wall_velocity[1])))
        if boundary == "ghost" and (wall_velocity[0] or wall_velocity[1]):
            # Inhomogeneous no-slip: the moving wall enters the wall rows as a
            # constant source once the ghost value is eliminated (see module doc).
            coefficient = (8.0 / 3.0 if cfg.wall_flux == "quadratic" else 2.0) * nu / cfg.dy ** 2
            rhs = rhs.clone()
            rhs[0] = rhs[0] + coefficient * wall_velocity[0]
            rhs[-1] = rhs[-1] + coefficient * wall_velocity[1]
        n_rows = n if boundary == "ghost" else n + 1
        rhs_hat = torch.fft.fft(rhs, dim=-1)
        lam = self.lam_x.unsqueeze(0).expand(n_rows, nx)
        lower = torch.full_like(lam, -nu / cfg.dy ** 2)
        upper = torch.full_like(lam, -nu / cfg.dy ** 2)
        diag = 1.0 / dt + 2 * nu / cfg.dy ** 2 + nu * lam
        if boundary == "ghost":
            if cfg.wall_flux == "quadratic":
                # three-point one-sided wall flux (9 phi_0 - phi_1)/(3 dy)
                diag[0] = 1.0 / dt + 4 * nu / cfg.dy ** 2 + nu * lam[0]
                upper[0] = -4 * nu / (3 * cfg.dy ** 2)
                diag[-1] = 1.0 / dt + 4 * nu / cfg.dy ** 2 + nu * lam[-1]
                lower[-1] = -4 * nu / (3 * cfg.dy ** 2)
            else:
                diag[0] = 1.0 / dt + 3 * nu / cfg.dy ** 2 + nu * lam[0]
                diag[-1] = 1.0 / dt + 3 * nu / cfg.dy ** 2 + nu * lam[-1]
            lower[0] = 0.0
            upper[-1] = 0.0
            return torch.fft.ifft(self._tridiagonal_solve(lower, diag, upper, rhs_hat),
                                  dim=-1).real
        diag[0] = 1.0
        upper[0] = 0.0
        lower[0] = 0.0
        diag[-1] = 1.0
        lower[-1] = 0.0
        upper[-1] = 0.0
        rhs_hat = rhs_hat.clone()
        rhs_hat[0] = 0.0
        rhs_hat[-1] = 0.0
        return torch.fft.ifft(self._tridiagonal_solve(lower, diag, upper, rhs_hat),
                              dim=-1).real

    def pressure_poisson(self, rhs):
        """Solve ``D G p = rhs`` with zero mean; periodic x, Neumann walls."""
        cfg = self.config
        n = cfg.ny
        rhs = rhs - rhs.mean()
        rhs_hat = torch.fft.fft(rhs, dim=-1)
        lam = self.lam_x.unsqueeze(0).expand(n, cfg.nx)
        lower = torch.full_like(lam, 1.0 / cfg.dy ** 2)
        upper = torch.full_like(lam, 1.0 / cfg.dy ** 2)
        diag = lam - 2.0 / cfg.dy ** 2
        diag[0] = lam[0] - 1.0 / cfg.dy ** 2
        diag[-1] = lam[-1] - 1.0 / cfg.dy ** 2
        lower[0] = 0.0
        upper[-1] = 0.0
        # The x-mean (k = 0) block is the wall-normal Neumann operator, which is
        # singular: its null space is the constant pressure.  It therefore cannot
        # go through the general (LU) solve below -- that would abort the whole
        # batch -- and is solved in the minimum-norm sense afterwards instead
        # (any exact solution differs from another only by a constant).  Leaving
        # that column out of the batch keeps the solve well posed.
        solved = torch.zeros_like(rhs_hat)
        if cfg.nx > 1:
            solved[:, 1:] = self._tridiagonal_solve(lower[:, 1:], diag[:, 1:],
                                                    upper[:, 1:], rhs_hat[:, 1:])
        n = cfg.ny
        mat = torch.zeros((n, n), dtype=torch.float64, device=self.device)
        idx = torch.arange(n, device=self.device)
        mat[idx, idx] = diag[:, 0]
        mat[idx[1:], idx[:-1]] = lower[1:, 0]
        mat[idx[:-1], idx[1:]] = upper[:-1, 0]
        mean_solution = torch.linalg.lstsq(mat, rhs_hat[:, 0].real.unsqueeze(-1)).solution.squeeze(-1)
        solved[:, 0] = mean_solution.to(torch.complex128)
        return torch.fft.ifft(solved, dim=-1).real

    # -- projection and step ------------------------------------------------
    def project(self, u_star, v_star, dt=None):
        """Discrete exact projection; returns the corrected fields and pressure."""
        cfg = self.config
        dt = self.time_step_s if dt is None else dt
        div = ((torch.roll(u_star, -1, dims=-1) - u_star) / cfg.dx
               + (v_star[1:] - v_star[:-1]) / cfg.dy)
        p = self.pressure_poisson(div / dt)
        u = u_star - dt * self.gradient_cell_to_u(p)
        v = v_star - dt * self.gradient_cell_to_v(p)
        v[0] = 0.0
        v[-1] = 0.0
        return u, v, p, dict(divergence_before_s_inv=float(div.abs().max()),
                             divergence_after_s_inv=float(((torch.roll(u, -1, dims=-1) - u) / cfg.dx
                                                           + (v[1:] - v[:-1]) / cfg.dy).abs().max()))

    def set_time_step(self, time_step_s):
        """Change the (pseudo-)time step; the converged steady state does not depend on it."""
        if not math.isfinite(time_step_s) or time_step_s <= 0:
            raise ValueError("time step must be finite and positive")
        self.time_step_s = float(time_step_s)
        return self.time_step_s

    @staticmethod
    def _wall_velocity_pair(value):
        """Normalise a wall-velocity specification to ``(lower, upper)`` in m/s."""
        if isinstance(value, bool):
            raise ValueError("wall velocity must be a number or a (lower, upper) pair")
        if isinstance(value, (int, float)):
            pair = (float(value), 0.0)
        else:
            if len(value) != 2:
                raise ValueError("wall_velocity_m_s needs a scalar or a (lower, upper) pair")
            pair = (float(value[0]), float(value[1]))
        if not all(math.isfinite(v) for v in pair):
            raise ValueError("wall velocity must be finite")
        return pair

    def set_wall_velocity(self, lower_m_s=0.0, upper_m_s=0.0):
        """Prescribe the streamwise no-slip wall speeds (m/s); default both at rest."""
        self.wall_velocity_m_s = self._wall_velocity_pair((lower_m_s, upper_m_s))
        return self.wall_velocity_m_s

    def step(self, body_force_m_s2=None, wall_velocity_m_s=None):
        """One semi-implicit step: explicit convection, implicit diffusion, projection.

        ``body_force_m_s2`` overrides ``config.body_force_m_s2`` for this step only,
        which is what unsteady (pulsatile) forcing needs; ``None`` keeps the
        configured constant value and therefore the steady-state behaviour.

        ``wall_velocity_m_s`` is the streamwise no-slip wall speed: a scalar (lower
        wall; the upper wall stays at rest) or ``(lower, upper)``.  ``None`` keeps
        the last prescribed value, so a driver may set it once
        (:meth:`set_wall_velocity`); the initial state is both walls at rest, which
        reproduces the historical homogeneous-wall behaviour bit for bit.
        """
        cfg = self.config
        dt = self.time_step_s
        forcing = cfg.body_force_m_s2 if body_force_m_s2 is None else float(body_force_m_s2)
        walls = (self.wall_velocity_m_s if wall_velocity_m_s is None
                 else self._wall_velocity_pair(wall_velocity_m_s))
        u_old = self.u.clone()
        rhs_u = self.u / dt - self.convection_u() + forcing
        rhs_v = self.v / dt - self.convection_v()
        u_star = self._implicit_diffusion(rhs_u, "ghost", wall_velocity=walls)
        v_star = self._implicit_diffusion(rhs_v, "dirichlet")
        v_star[0] = 0.0
        v_star[-1] = 0.0
        u, v, p, info = self.project(u_star, v_star, dt)
        self.u, self.v, self.pressure = u, v, p
        self.wall_velocity_m_s = walls
        self.time += dt
        record = dict(time_s=self.time,
                      body_force_m_s2=forcing,
                      wall_velocity_m_s=walls,
                      bulk_velocity_m_s=self.bulk_velocity(),
                      max_divergence_s_inv=info["divergence_after_s_inv"],
                      velocity_change_m_s=float((self.u - u_old).abs().max()))
        self.history.append(record)
        return record

    # -- diagnostics --------------------------------------------------------
    def bulk_velocity(self):
        """Discrete bulk velocity ``(1/H) integral u dy`` for an x-uniform field."""
        return float(self.u.mean())

    def max_divergence(self):
        return float(self.divergence().abs().max())

    def convection_at_solution(self):
        """Magnitude of the convective term, scaled by the body force."""
        return float(self.convection_u().abs().max() / self.config.body_force_m_s2)

    def wall_shear(self, estimator="flux"):
        """Wall shear stress (Pa) on both walls.

        ``flux``      : the scheme's own wall flux, ``mu * 2 (u_0 - u_w) / dy``
                        (``u_w`` the moving-wall speed, zero for a static wall).
                        This is the quantity closing the discrete momentum
                        balance, so it agrees with ``rho G H / 2`` by
                        conservation.
        ``quadratic`` : three-point one-sided fit through the wall and the two
                        nearest centres, ``mu (9 u_0 - u_1 - 8 u_w)/(3 dy)``; it
                        does not reuse the flux the scheme applied, so it
                        measures truncation error rather than restating
                        conservation.  Both estimators refer to the same sign
                        convention: positive into the wall, so for a static wall
                        they are the two wall-flux magnitudes that add up to the
                        momentum input.
        """
        cfg = self.config
        mu = cfg.viscosity_pa_s
        u = self.u
        wall_lower, wall_upper = self.wall_velocity_m_s
        if estimator == "flux":
            lower = mu * 2.0 * (float(u[0].mean()) - wall_lower) / cfg.dy
            upper = mu * 2.0 * (float(u[-1].mean()) - wall_upper) / cfg.dy
        elif estimator == "quadratic":
            lower = mu * (9.0 * float(u[0].mean()) - float(u[1].mean())
                          - 8.0 * wall_lower) / (3.0 * cfg.dy)
            upper = mu * (9.0 * float(u[-1].mean()) - float(u[-2].mean())
                          - 8.0 * wall_upper) / (3.0 * cfg.dy)
        else:
            raise ValueError("estimator must be 'flux' or 'quadratic'")
        return float(lower), float(upper)

    def momentum_balance(self):
        """``rho G H`` versus the sum of the two *scheme* wall fluxes."""
        cfg = self.config
        estimator = "quadratic" if cfg.wall_flux == "quadratic" else "flux"
        lower, upper = self.wall_shear(estimator)
        flux_sum = lower + upper
        target = cfg.density_kg_m3 * cfg.body_force_m_s2 * cfg.height_m
        return dict(body_force_per_area_N_m2=float(target),
                    wall_flux_sum_N_m2=float(flux_sum),
                    relative_residual=float(abs(flux_sum - target) / target))

    def kinetic_energy(self):
        """Discrete kinetic energy per unit depth ``1/2 rho sum (u^2+v^2) dV``."""
        cfg = self.config
        cell_u = 0.5 * (self.u + torch.roll(self.u, -1, dims=-1))
        cell_v = 0.5 * (self.v[:-1] + self.v[1:])
        vol = cfg.dx * cfg.dy
        return float(0.5 * cfg.density_kg_m3 * vol * (cell_u ** 2 + cell_v ** 2).sum())

    def dissipation(self):
        """Viscous dissipation rate per unit depth, ``mu * integral (du/dy)^2 dy``.

        The streamwise velocity varies only across the channel, so this is the
        whole dissipation to leading order; the wall derivative is taken with the
        same estimator the scheme used, which keeps the budget closed.
        """
        cfg = self.config
        mu = cfg.viscosity_pa_s
        wall_lower, wall_upper = self.wall_velocity_m_s
        if cfg.wall_flux == "quadratic":
            lower = (9.0 * self.u[0] - self.u[1] - 8.0 * wall_lower) / (3.0 * cfg.dy)
            upper = (9.0 * self.u[-1] - self.u[-2] - 8.0 * wall_upper) / (3.0 * cfg.dy)
        else:
            lower = 2.0 * (self.u[0] - wall_lower) / cfg.dy
            upper = 2.0 * (wall_upper - self.u[-1]) / cfg.dy
        interior = (self.u[1:] - self.u[:-1]) / cfg.dy
        edges = torch.cat([lower.reshape(1, -1), interior, upper.reshape(1, -1)], dim=0)
        at_cells = 0.5 * (edges[1:] + edges[:-1])
        return float(mu * cfg.dx * cfg.dy * (at_cells ** 2).sum())

    def sample_profile(self):
        """Row-averaged ``u`` at the y cell centres (x-uniform by construction)."""
        yc, _, _ = self.wall_coordinates()
        return yc, self.u.mean(dim=-1)

    # -- drivers ------------------------------------------------------------
    def solve_steady(self, tolerance=1e-12, max_steps=200000, min_steps=10):
        """Pseudo-time march until ``max |u^{n+1} - u^n| <= tolerance``."""
        converged = False
        change = float("inf")
        for _ in range(max_steps):
            previous = self.u.clone()
            self.step()
            change = float((self.u - previous).abs().max())
            if len(self.history) >= min_steps and change <= tolerance and self.max_divergence() < 1e-10:
                converged = True
                break
        return dict(converged=converged, steps=len(self.history), final_change=change,
                    time_s=self.time)

    def set_velocity_profile(self, values_m_s):
        """Overwrite ``u`` with an x-uniform wall-normal profile."""
        u = torch.as_tensor(values_m_s, dtype=torch.float64, device=self.device).reshape(-1, 1)
        if u.shape[0] != self.config.ny:
            raise ValueError("profile needs one value per y cell")
        self.u = u.expand(-1, self.config.nx).clone()
        self.v = torch.zeros_like(self.v)

    def add_streamwise_wave(self, amplitude_m_s, mode=1):
        """Add ``A sin(2 pi mode x/L) sin(pi y/H)`` to ``u`` and re-project."""
        cfg = self.config
        _, _, xc = self.wall_coordinates()
        yc, _, _ = self.wall_coordinates()
        kx = 2 * math.pi * mode / cfg.length_m
        wave = amplitude_m_s * torch.sin(kx * xc).unsqueeze(0) * torch.sin(math.pi * yc / cfg.height_m).unsqueeze(-1)
        self.u = self.u + wave
        info = self.project(self.u, self.v, self.time_step_s)
        self.u, self.v, self.pressure = info[0], info[1], info[2]
        return float(wave.abs().max())

    def streamwise_wave_amplitude(self, mode=1):
        """Amplitude of the streamwise Fourier mode ``mode`` of ``u``."""
        cfg = self.config
        mean_x = self.u.mean(dim=-1, keepdim=True)
        spec = torch.fft.fft(self.u - mean_x, dim=-1)
        return float(2.0 * spec[:, mode].abs().max() / cfg.nx)