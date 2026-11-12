"""Three-dimensional lid-driven cavity on a staggered MAC grid (periodic span).

Why this module exists (verification debt)
------------------------------------------
``scripts/verify_lid_cavity_3d.py`` was committed against a module that did not
exist: ``tensorfvm.lid_cavity_3d`` was never implemented, so the 3-D entry of
the verification catalogue was a dead link
(``docs/verification-status-summary-2026-11-11.md`` §3.2 item 4).  This module
closes that hole with a real solver rather than deleting the claim.

What is implemented
-------------------
The 2-D staggered discretisation of :mod:`tensorfvm.lid_cavity` (MAC placement,
van Leer limited convective fluxes, DCT-II pure-Neumann pressure solve, explicit
marching to a steady state) is extruded to a prism ``n_x x n_y x n_z`` with

* ``u`` on x-faces ``(n_x + 1, n_y, n_z)``,
* ``v`` on y-faces ``(n_x, n_y + 1, n_z)``,
* ``w`` on z-faces ``(n_x, n_y, n_z)`` -- the span is **periodic**, so the
  "face between cell k-1 and cell k" numbering is used and no end walls exist,
* walls on all four sides ``x = 0, L`` and ``y = 0, L``: no-slip, with the lid
  ``u = U(z)`` on ``y = L``; ``U`` may carry a spanwise modulation
  ``U (1 + a cos(2 pi m z / L_z))`` so the module can be driven into a genuinely
  three-dimensional state (``w != 0``) without leaving the bench,
* pressure solved by DCT-II in ``x``/``y`` (homogeneous Neumann) **and** an FFT
  in the periodic ``z``; the zero mode is pinned, so the pressure datum is
  arbitrary as in 2-D.

The x/y operators are copied from the 2-D solver term by term (same index
convention, same limiter), so a span-invariant initial state with ``w = 0``
reduces to the 2-D discretisation *identically* -- not just to the same order.
That is what makes the 2-D Ghia tables usable as an accuracy anchor for a 3-D
code, and it is asserted in ``tests/test_native_lid_cavity_3d.py``.

The spanwise (z) convective fluxes use the same van Leer limiter with MUSCL
face reconstruction (``_muscl_faces``: periodic or wall-bounded), so no
direction is silently first-order while the others are second-order.

Honest scope
------------
* What is verified here: exact 2-D reduction, discrete mass conservation
  (``max|div u|`` at round-off), spanwise symmetry preservation (``w`` stays at
  machine zero for a span-invariant lid), Ghia centreline accuracy at Re = 100 /
  400, and linearity of the spanwise response for a weakly modulated lid.
* What is **not** verified: a full 3-D cavity with spanwise end walls (Ku,
  Hirsh & Taylor 1989 style geometry) has no reference table in this repository
  and no anchor here -- the periodic span makes the 2-D limit exact but it is
  *not* a substitute for that benchmark.  Turbulent or high-Re 3-D states are
  outside the steady explicit-solver envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .lid_cavity import TINY, hermite_interpolate, score_profiles
from .lid_cavity_reference import SUPPORTED_REYNOLDS

__all__ = [
    "LidCavity3DConfig",
    "LidCavity3DSolver",
    "score_against_ghia_3d",
]


@dataclass(frozen=True)
class LidCavity3DConfig:
    """Discretisation and solver controls for :class:`LidCavity3DSolver`.

    ``n_y = 0`` copies ``n_x`` (square cross-section with cubic cells).  The
    span length is ``n_z * h`` with ``h = L / n_x``.
    """

    n_x: int = 32
    n_y: int = 0
    n_z: int = 4
    reynolds: float = 100.0
    lid_velocity: float = 1.0
    density: float = 1.0
    cfl: float = 0.15
    time_step: float | None = None
    max_steps: int = 200_000
    tolerance: float = 1e-7
    check_every: int = 250
    lid_spanwise_amplitude: float = 0.0
    lid_spanwise_wavenumber: int = 1
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.n_y == 0:
            object.__setattr__(self, "n_y", self.n_x)
        for name in ("n_x", "n_y", "n_z", "max_steps", "check_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if self.n_x != self.n_y:
            raise ValueError("n_x must equal n_y: cubic cells and one h are assumed")
        if self.n_y % 2 or self.n_x % 2:
            raise ValueError("n_x must be even so the mid-plane sits on a grid line")
        for name in ("reynolds", "lid_velocity", "density", "cfl", "tolerance"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.time_step is not None:
            if not math.isfinite(self.time_step) or self.time_step <= 0:
                raise ValueError("time_step must be a positive finite number")
        if not 0.0 <= self.lid_spanwise_amplitude < 1.0:
            raise ValueError("lid_spanwise_amplitude must lie in [0, 1)")
        if isinstance(self.lid_spanwise_wavenumber, bool) or self.lid_spanwise_wavenumber < 1:
            raise ValueError("lid_spanwise_wavenumber must be an integer >= 1")

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity from ``Re = U L / nu`` with ``L = 1``."""
        return self.lid_velocity / self.reynolds


class LidCavity3DSolver:
    """Steady-state 3-D lid cavity: ``n_x x n_y x n_z`` prism, periodic span."""

    def __init__(self, config: LidCavity3DConfig | None = None) -> None:
        self.config = config or LidCavity3DConfig()
        cfg = self.config
        self.n_x, self.n_y, self.n_z = cfg.n_x, cfg.n_y, cfg.n_z
        self.h = 1.0 / cfg.n_x
        self.nu = cfg.viscosity
        self.dtype, self.device = cfg.dtype, cfg.device

        self.u = torch.zeros(self.n_x + 1, self.n_y, self.n_z,
                             dtype=self.dtype, device=self.device)
        self.v = torch.zeros(self.n_x, self.n_y + 1, self.n_z,
                             dtype=self.dtype, device=self.device)
        self.w = torch.zeros(self.n_x, self.n_y, self.n_z,
                             dtype=self.dtype, device=self.device)
        self.p = torch.zeros(self.n_x, self.n_y, self.n_z,
                             dtype=self.dtype, device=self.device)

        self._dct_x = self._dct_matrix(self.n_x)
        self._dct_y = self._dct_matrix(self.n_y)
        self._lambda_x = self._neumann_eigenvalues(self.n_x)
        self._lambda_y = self._neumann_eigenvalues(self.n_y)
        self._lambda_z = self._periodic_eigenvalues(self.n_z)

        self.steps = 0
        self.residual = float("inf")

    # ------------------------------------------------------------- discretisation
    def _neumann_eigenvalues(self, n: int) -> torch.Tensor:
        k = torch.arange(n, dtype=self.dtype, device=self.device)
        return (2.0 * torch.cos(math.pi * k / n) - 2.0) / self.h ** 2

    def _periodic_eigenvalues(self, n: int) -> torch.Tensor:
        """Eigenvalues of the centred second difference, in FFT order."""
        k = torch.arange(n, dtype=self.dtype, device=self.device)
        mode = torch.where(k <= n // 2, k, k - n)
        return (2.0 * torch.cos(2.0 * math.pi * mode / n) - 2.0) / self.h ** 2

    def _dct_matrix(self, n: int) -> torch.Tensor:
        """Orthonormal DCT-II matrix ``C[k, j] = a_k cos(pi k (j+1/2) / n)``."""
        k = torch.arange(n, dtype=self.dtype, device=self.device)[:, None]
        j = torch.arange(n, dtype=self.dtype, device=self.device)[None, :]
        mat = torch.cos(math.pi * k * (j + 0.5) / n)
        scale = torch.full((n, 1), math.sqrt(2.0 / n), dtype=self.dtype, device=self.device)
        scale[0] = math.sqrt(1.0 / n)
        return mat * scale

    # ------------------------------------------------------------------ operators
    @staticmethod
    def _limiter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        prod = a * b
        total = a + b
        safe = total.abs() > TINY
        value = 2.0 * prod / torch.where(safe, total, torch.ones_like(total))
        return torch.where(prod > 0.0, value, torch.zeros_like(value))

    @classmethod
    def _vanleer_face(cls, f: torch.Tensor, vel: torch.Tensor, axis: int) -> torch.Tensor:
        """van Leer limited interface values of ``f`` between ``M - 1`` interfaces.

        ``f`` holds ``M`` samples along ``axis``, ``vel`` the ``M - 1`` normal
        velocities at the interfaces between consecutive samples.  This is the
        same routine (and the same indexing) as :mod:`tensorfvm.lid_cavity`, so
        the x/y operators are bit-compatible with the 2-D solver.
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
        forward = fa[:-1] + 0.5 * cls._limiter(up, d)
        backward = fa[1:] - 0.5 * cls._limiter(down, d)
        out = torch.where(va >= 0.0, forward, backward)
        return out.movedim(0, axis)

    @classmethod
    def _muscl_faces(cls, f: torch.Tensor, vel: torch.Tensor, axis: int,
                     periodic: bool) -> torch.Tensor:
        """Limited (van Leer) face values of a **cell-centred** field.

        ``f`` holds ``M`` cell samples along ``axis``; ``vel`` holds the ``M + 1``
        normal face velocities for the wall-bounded case (``vel[0] = vel[M] = 0``
        are then ignored: the transported value is 0 there) or the ``M`` periodic
        face velocities for the periodic case (face ``k`` sits between cell
        ``k - 1`` and cell ``k``).  Returns the ``M + 1`` face values (walls) or
        the ``M`` periodic face values.
        """
        fa = f.movedim(axis, 0)
        va = vel.movedim(axis, 0)
        if periodic:
            back = fa - torch.roll(fa, 1, 0)            # f[c] - f[c-1]
            fwd = torch.roll(fa, -1, 0) - fa            # f[c+1] - f[c]
            slope = cls._limiter(fwd, back)
            from_left = torch.roll(fa + 0.5 * slope, 1, 0)      # upwind cell k-1
            from_right = fa - 0.5 * slope                       # upwind cell k
            out = torch.where(va >= 0.0, from_left, from_right)
            return out.movedim(0, axis)

        m = fa.shape[0]
        padded = torch.cat((fa[:1], fa, fa[-1:]), 0)
        back = padded[1:m + 1] - padded[0:m]
        fwd = padded[2:m + 2] - padded[1:m + 1]
        slope = cls._limiter(fwd, back)
        out = torch.zeros(m + 1, *fa.shape[1:], dtype=fa.dtype, device=fa.device)
        vi = va[1:m]
        from_left = fa + 0.5 * slope        # face i, upwind cell i-1 (its right face)
        from_right = fa - 0.5 * slope       # face i, upwind cell i (its left face)
        # face i lies between cells i-1 and i: the positive branch takes the
        # right face of cell i-1, the negative branch the LEFT face of cell i
        # (``from_right[0:m-1]`` was one cell upstream-wrong: a linear profile
        # was reconstructed with a uniform offset of h on every inflow face).
        out[1:m] = torch.where(vi >= 0.0, from_left[0:m - 1], from_right[1:m])
        return out.movedim(0, axis)

    # ---------------------------------------------------------------- diagnostics
    def divergence(self) -> torch.Tensor:
        """Cell-centred discrete divergence (vanishes after projection)."""
        return ((self.u[1:] - self.u[:-1]) / self.h
                + (self.v[:, 1:] - self.v[:, :-1]) / self.h
                + (torch.roll(self.w, -1, 2) - self.w) / self.h)

    def spanwise_uniformity(self) -> float:
        """``max|w|`` -- zero for a span-invariant lid, by symmetry."""
        return float(self.w.abs().max())

    def lid_velocity_field(self) -> torch.Tensor:
        """Lid speed as a function of the span (shape ``(n_z,)``)."""
        cfg = self.config
        if cfg.lid_spanwise_amplitude == 0.0:
            return torch.full((self.n_z,), cfg.lid_velocity,
                              dtype=self.dtype, device=self.device)
        z = (torch.arange(self.n_z, dtype=self.dtype, device=self.device) + 0.5) * self.h
        span = self.n_z * self.h
        return cfg.lid_velocity * (1.0 + cfg.lid_spanwise_amplitude
                                   * torch.cos(2.0 * math.pi
                                               * cfg.lid_spanwise_wavenumber * z / span))

    # ------------------------------------------------------------------ momentum
    def _momentum_tendency(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                           wall_lo, wall_hi) -> torch.Tensor:
        """Convection + diffusion tendency of the axis-0 face field ``u``.

        The x/y terms are the 2-D operator of :mod:`tensorfvm.lid_cavity` verbatim
        (one extra span axis); the z terms are added and are identically zero for
        span-invariant fields with ``w = 0``.
        """
        n_x, n_y, n_z, h, nu = self.n_x, self.n_y, self.n_z, self.h, self.nu

        # -- convection in x: transported and advecting quantity are both u
        ux = 0.5 * (u[:-1] + u[1:])
        flux_x = ux * self._vanleer_face(u, ux, axis=0)
        conv = torch.zeros_like(u)
        conv[1:n_x] = -(flux_x[1:] - flux_x[:-1]) / h

        # -- convection in y: transported u, advecting v; wall fluxes are zero
        v_face = 0.5 * (v[0:n_x - 1] + v[1:n_x])
        face_u = self._vanleer_face(u[1:n_x], v_face[:, 1:n_y], axis=1)
        flux_y = torch.zeros(n_x - 1, n_y + 1, n_z, dtype=u.dtype, device=u.device)
        flux_y[:, 1:n_y] = v_face[:, 1:n_y] * face_u
        conv[1:n_x] -= (flux_y[:, 1:] - flux_y[:, :-1]) / h

        # -- convection in z (periodic): w interpolated onto the u locations
        w_at_u = 0.5 * (w[:-1] + w[1:])
        transported = self._muscl_faces(u[1:n_x], w_at_u, axis=2, periodic=True)
        flux_z = w_at_u * transported
        conv[1:n_x] -= (torch.roll(flux_z, -1, 2) - flux_z) / h

        # -- diffusion in x: u = 0 (normal velocity) on both side walls
        lap = torch.zeros_like(u)
        lap[1:n_x] = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / h ** 2

        # -- diffusion in y with mirror ghosts carrying the tangential wall speed
        below = 2.0 * wall_lo - u[:, 0]
        above = 2.0 * wall_hi - u[:, n_y - 1]
        lap[:, 0] += (below - 2.0 * u[:, 0] + u[:, 1]) / h ** 2
        if n_y > 2:
            lap[:, 1:n_y - 1] += (u[:, :-2] - 2.0 * u[:, 1:-1] + u[:, 2:]) / h ** 2
        lap[:, n_y - 1] += (u[:, n_y - 2] - 2.0 * u[:, n_y - 1] + above) / h ** 2

        # -- diffusion in z (periodic, spans identical -> exactly zero)
        lap += (torch.roll(u, -1, 2) + torch.roll(u, 1, 2) - 2.0 * u) / h ** 2

        return conv + nu * lap

    def _spanwise_tendency(self) -> torch.Tensor:
        """Tendency of the z-face field ``w`` (periodic in z, no-slip in x/y)."""
        n_x, n_y, n_z, h, nu = self.n_x, self.n_y, self.n_z, self.h, self.nu
        w, u, v = self.w, self.u, self.v

        # -- convection in z: w lives on z-faces, reconstructed from cell averages
        w_cell = 0.5 * (w + torch.roll(w, -1, 2))
        transported = self._muscl_faces(w, w_cell, axis=2, periodic=True)
        flux_z = w_cell * transported
        conv = -(torch.roll(flux_z, -1, 2) - flux_z) / h

        # -- convection in x: u at x-faces advects the cell-centred w
        u_cell = 0.5 * (u[:-1] + u[1:])
        transported_x = self._muscl_faces(w, u, axis=0, periodic=False)
        flux_x = u * transported_x          # (n_x + 1, n_y, n_z), zero at the walls
        conv -= (flux_x[1:] - flux_x[:-1]) / h

        # -- convection in y
        transported_y = self._muscl_faces(w, v, axis=1, periodic=False)
        flux_y = v * transported_y
        conv -= (flux_y[:, 1:] - flux_y[:, :-1]) / h

        # -- diffusion in x and y with no-slip mirror ghosts (w = 0 on the walls)
        lap = torch.zeros_like(w)
        inner_x = (w[:-2] - 2.0 * w[1:-1] + w[2:]) / h ** 2
        lap[1:n_x - 1] += inner_x
        lap[0] += (-2.0 * w[0] + w[1]) / h ** 2          # ghost = -w[0] (w_wall = 0)
        lap[n_x - 1] += (w[n_x - 2] - 2.0 * w[n_x - 1]) / h ** 2
        inner_y = (w[:, :-2] - 2.0 * w[:, 1:-1] + w[:, 2:]) / h ** 2
        lap[:, 1:n_y - 1] += inner_y
        lap[:, 0] += (-2.0 * w[:, 0] + w[:, 1]) / h ** 2
        lap[:, n_y - 1] += (w[:, n_y - 2] - 2.0 * w[:, n_y - 1]) / h ** 2
        lap += (torch.roll(w, -1, 2) + torch.roll(w, 1, 2) - 2.0 * w) / h ** 2

        return conv + nu * lap

    def tendencies(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Momentum right-hand sides without the pressure gradient."""
        lid = self.lid_velocity_field().view(1, self.n_z)
        tu = self._momentum_tendency(self.u, self.v, self.w,
                                     torch.zeros_like(lid), lid)
        tv_t = self._momentum_tendency(self.v.transpose(0, 1).contiguous(),
                                       self.u.transpose(0, 1).contiguous(),
                                       self.w.transpose(0, 1).contiguous(),
                                       0.0, 0.0)
        tw = self._spanwise_tendency()
        return tu, tv_t.transpose(0, 1).contiguous(), tw

    # ------------------------------------------------------------------ pressure
    def solve_pressure(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve ``lap p = rhs``: Neumann in x/y (DCT-II), periodic in z (FFT)."""
        centred = rhs - rhs.mean()
        coeff = torch.einsum("ij,jkl->ikl", self._dct_x, centred)
        coeff = torch.einsum("jk,ikl->ijl", self._dct_y, coeff)
        spec = torch.fft.fft(coeff, dim=2)
        lam = (self._lambda_x[:, None, None] + self._lambda_y[None, :, None]
               + self._lambda_z[None, None, :])
        zero = lam.abs() < 1e-300
        spec = spec / torch.where(zero, torch.ones_like(lam), lam)
        spec[0, 0, 0] = 0.0
        field = torch.fft.ifft(spec, dim=2).real
        field = torch.einsum("ji,jkl->ikl", self._dct_x, field)
        field = torch.einsum("kj,ikl->ijl", self._dct_y, field)
        return field

    # ----------------------------------------------------------------- time step
    def stable_time_step(self) -> float:
        cfg = self.config
        if cfg.time_step is not None:
            return float(cfg.time_step)
        speed = max(float(self.u.abs().max()), float(self.v.abs().max()),
                    float(self.w.abs().max()), cfg.lid_velocity)
        conv = cfg.cfl * self.h / speed
        visc = self.h ** 2 / (6.0 * self.nu)          # 3-D explicit diffusion limit
        return min(conv, visc)

    def step(self, dt: float | None = None) -> float:
        """Advance one explicit projection step and return the step size used."""
        dt = self.stable_time_step() if dt is None else float(dt)
        tu, tv, tw = self.tendencies()

        n_x, n_y, n_z = self.n_x, self.n_y, self.n_z
        u_star = self.u.clone()
        v_star = self.v.clone()
        w_star = self.w.clone()
        u_star[1:n_x] += dt * tu[1:n_x]
        v_star[:, 1:n_y] += dt * tv[:, 1:n_y]
        w_star += dt * tw

        div = ((u_star[1:] - u_star[:-1]) / self.h
               + (v_star[:, 1:] - v_star[:, :-1]) / self.h
               + (torch.roll(w_star, -1, 2) - w_star) / self.h)
        self.p = self.solve_pressure(div / dt)

        self.u[1:n_x] = u_star[1:n_x] - dt * (self.p[1:] - self.p[:-1]) / self.h
        self.v[:, 1:n_y] = v_star[:, 1:n_y] - dt * (self.p[:, 1:] - self.p[:, :-1]) / self.h
        self.w = w_star - dt * (self.p - torch.roll(self.p, 1, 2)) / self.h
        self.steps += 1
        return dt

    def run(self, *, max_steps: int | None = None, tolerance: float | None = None,
            check_every: int | None = None, on_check=None) -> dict[str, object]:
        """March to steady state and return a convergence ledger."""
        cfg = self.config
        max_steps = cfg.max_steps if max_steps is None else int(max_steps)
        tolerance = cfg.tolerance if tolerance is None else float(tolerance)
        check_every = cfg.check_every if check_every is None else int(check_every)

        dt = 0.0
        history: list[tuple[int, float]] = []
        for _ in range(max_steps):
            before_u, before_v = self.u.clone(), self.v.clone()
            dt = self.step()
            if self.steps % check_every == 0:
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
            "iterations": self.steps,
            "dt": dt,
            "residual": self.residual,
            "converged": self.residual <= tolerance,
            "history": history,
            "max_divergence": float(self.divergence().abs().max()),
            "spanwise_max": self.spanwise_uniformity(),
        }

    # --------------------------------------------------------------- centrelines
    def centreline_u(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(y, u)`` on the mid-plane ``x = L/2``, averaged over the span."""
        i = self.n_x // 2
        y = (torch.arange(self.n_y, dtype=self.dtype) + 0.5) * self.h
        return y, self.u[i].mean(dim=1)

    def centreline_v(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, v)`` on the mid-plane ``y = L/2``, averaged over the span."""
        j = self.n_y // 2
        x = (torch.arange(self.n_x, dtype=self.dtype) + 0.5) * self.h
        return x, self.v[:, j].mean(dim=1)


def score_against_ghia_3d(solver: LidCavity3DSolver, reynolds: float | None = None) -> dict:
    """Score the span-averaged centrelines against the Ghia et al. (1982) tables."""
    from .lid_cavity import score_profiles

    re = float(solver.config.reynolds if reynolds is None else reynolds)
    if re not in SUPPORTED_REYNOLDS:
        raise KeyError(f"no Ghia table for Re={re}; available {SUPPORTED_REYNOLDS}")

    y, u_prof = solver.centreline_u()
    x, v_prof = solver.centreline_v()
    out: dict = {"reynolds": re, "n_x": solver.n_x, "n_z": solver.n_z,
                 "steps": solver.steps, "residual": solver.residual,
                 "lid_velocity": solver.config.lid_velocity}
    out["u"] = score_profiles(y, u_prof, re, "u")
    out["v"] = score_profiles(x, v_prof, re, "v")
    out["max_divergence"] = float(solver.divergence().abs().max())
    out["spanwise_max"] = solver.spanwise_uniformity()
    return out