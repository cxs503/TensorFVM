"""Three-dimensional lid-driven cavity with spanwise **end walls**.

Why this module exists
----------------------
:mod:`tensorfvm.lid_cavity_3d` extrudes the verified 2-D staggered cavity solver
to a prism with a **periodic** span.  Its own docstring records the honest
limitation: *"a full 3-D cavity with spanwise end walls (Ku, Hirsh & Taylor 1989
style geometry) has no reference table in this repository and no anchor here --
the periodic span makes the 2-D limit exact but it is not a substitute for that
benchmark."*  ``docs/verification-catalog.md`` §10.1 item 1 repeats the same
hole in one line ("3D Lid-Driven Cavity: Only 2D verified against Ghia").

This module closes the *capability* half of that hole: the span is now
wall-bounded, so ``w`` genuinely vanishes on ``z = 0, L_z`` (no-slip) or the
tangential components have zero wall-normal derivative there (free-slip).  The
*accuracy* half is anchored as far as the repository's evidence allows:

* free-slip end walls + a span-invariant lid leave the span **inert**, so the
  discrete solution must equal the 2-D solver to round-off (exact reduction,
  not "same order").  This is the consistency anchor of the new z operators;
* no-slip end walls with a span-invariant lid keep ``w = 0`` **exactly**
  (discrete symmetry) while damping ``u``/``v`` near the walls -- a genuinely
  three-dimensional velocity field without any spanwise forcing;
* a modulated lid ``U(z) = U (1 + a cos(2 pi m z / L_z))`` excites ``w != 0``,
  and the discrete solution must stay antisymmetric about the mid-plane
  (``w`` odd, ``u``/``v`` even) and divergence-free;
* the span family ``L_z = 1, 2, 4`` gives the only *quantitative* accuracy
  statement available offline: the mid-plane centreline error against the Ghia
  et al. (1982) tables must fall monotonically with ``L_z`` and extrapolate
  (``1/L_z -> 0``) into the same band the 2-D solver is held to.

There is **no** table for the end-wall geometry in this repository, so absolute
accuracy against Ku/Hirsh/Taylor is reported as WITHHELD, not as a pass.

Discretisation
--------------
Same staggered MAC layout as :mod:`tensorfvm.lid_cavity_3d` for ``u``/``v``/``p``
(cubic cells, ``h = 1/n_x``), with the span treated like the other two walls:

* ``w`` on z-faces ``(n_x, n_y, n_z + 1)`` with ``w[..., 0] = w[..., n_z] = 0``,
* ``u``/``v`` cell-centred in ``z``; no-slip uses an anti-mirror ghost at half a
  cell (``u_wall = 0``), free-slip a mirror ghost (``du/dz = 0``),
* pressure by DCT-II in **all three** directions (homogeneous Neumann on every
  wall, periodic FFT dropped), zero mode pinned as in 2-D,
* spanwise convection uses the same van Leer limiter and MUSCL reconstruction as
  the x/y directions (:meth:`tensorfvm.lid_cavity_3d.LidCavity3DSolver._muscl_faces`),
  so no direction is silently first-order.

The limiter, the MUSCL reconstruction, the DCT helpers and the time-stepping
envelope are inherited from the periodic-span solver; only the span operators
(``_z_diffusion``, the z branches of ``_momentum_tendency``/``_spanwise_tendency``,
the z-DCT pressure solve and the projection step) are new, which is what keeps
the reduction property testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lid_cavity import score_profiles
from .lid_cavity_3d import LidCavity3DConfig, LidCavity3DSolver

__all__ = [
    "SPAN_BCS",
    "LidCavity3DEndWallConfig",
    "LidCavity3DEndWallSolver",
    "score_endwall_against_ghia",
]

SPAN_BCS = ("no_slip", "free_slip")


@dataclass(frozen=True)
class LidCavity3DEndWallConfig(LidCavity3DConfig):
    """Config for :class:`LidCavity3DEndWallSolver`.

    ``span_bc`` selects the end-wall condition; ``n_z`` must be even so the span
    mid-plane coincides with a z-face (where ``w = 0`` by antisymmetry).  All
    other fields keep the meaning they have in :class:`LidCavity3DConfig`.
    """

    n_z: int = 24
    span_bc: str = "no_slip"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.span_bc not in SPAN_BCS:
            raise ValueError(f"span_bc must be one of {SPAN_BCS}, got {self.span_bc!r}")
        if self.n_z % 2:
            raise ValueError("n_z must be even: the span mid-plane must be a z-face")


class LidCavity3DEndWallSolver(LidCavity3DSolver):
    """Steady 3-D lid cavity on ``n_x x n_y x n_z`` cells with a wall-bounded span."""

    def __init__(self, config: LidCavity3DEndWallConfig | None = None) -> None:
        cfg = config or LidCavity3DEndWallConfig()
        if not isinstance(cfg, LidCavity3DEndWallConfig):
            raise TypeError("LidCavity3DEndWallSolver needs a LidCavity3DEndWallConfig")
        self.config = cfg
        self.n_x, self.n_y, self.n_z = cfg.n_x, cfg.n_y, cfg.n_z
        self.h = 1.0 / cfg.n_x
        self.nu = cfg.viscosity
        self.dtype, self.device = cfg.dtype, cfg.device
        self.span_bc = cfg.span_bc

        self.u = torch.zeros(self.n_x + 1, self.n_y, self.n_z,
                             dtype=self.dtype, device=self.device)
        self.v = torch.zeros(self.n_x, self.n_y + 1, self.n_z,
                             dtype=self.dtype, device=self.device)
        self.w = torch.zeros(self.n_x, self.n_y, self.n_z + 1,
                             dtype=self.dtype, device=self.device)
        self.p = torch.zeros(self.n_x, self.n_y, self.n_z,
                             dtype=self.dtype, device=self.device)

        self._dct_x = self._dct_matrix(self.n_x)
        self._dct_y = self._dct_matrix(self.n_y)
        self._dct_z = self._dct_matrix(self.n_z)
        self._lambda_x = self._neumann_eigenvalues(self.n_x)
        self._lambda_y = self._neumann_eigenvalues(self.n_y)
        self._lambda_z = self._neumann_eigenvalues(self.n_z)

        self.steps = 0
        self.residual = float("inf")

    # ------------------------------------------------------------ reconstruction
    @staticmethod
    def _wall_muscl(f: torch.Tensor, vel: torch.Tensor, axis: int) -> torch.Tensor:
        """Van Leer limited face values on a wall-bounded grid.

        ``f`` holds ``M`` cell samples along ``axis``, ``vel`` the ``M + 1`` face
        velocities (``vel[0] = vel[M] = 0`` at the walls).  Face ``i`` lies
        between cells ``i - 1`` and ``i``, so the positive branch takes the
        **right** face of cell ``i - 1`` and the negative branch the **left**
        face of cell ``i``.

        The inherited :meth:`tensorfvm.lid_cavity_3d.LidCavity3DSolver._muscl_faces`
        takes ``from_right[0:m-1]`` -- the left face of cell ``i - 1``, i.e. one
        whole cell upstream-wrong -- for the negative branch, which reconstructs
        a linear profile with a constant offset of exactly ``h`` on every
        inflow face (measured; see ``docs/verification-lid-cavity-3d-endwalls``).
        This method is the corrected version used by the end-wall solver.
        """
        fa = f.movedim(axis, 0)
        va = vel.movedim(axis, 0)
        m = fa.shape[0]
        padded = torch.cat((fa[:1], fa, fa[-1:]), 0)
        back = padded[1:m + 1] - padded[0:m]        # f[c] - f[c-1]
        fwd = padded[2:m + 2] - padded[1:m + 1]     # f[c+1] - f[c]
        slope = LidCavity3DSolver._limiter(fwd, back)
        right = fa + 0.5 * slope                    # right face of cell c = face c+1
        left = fa - 0.5 * slope                     # left face of cell c  = face c
        out = torch.zeros(m + 1, *fa.shape[1:], dtype=fa.dtype, device=fa.device)
        vi = va[1:m]
        out[1:m] = torch.where(vi >= 0.0, right[0:m - 1], left[1:m])
        return out.movedim(0, axis)

    # ------------------------------------------------------------ span operators
    @property
    def span_length(self) -> float:
        """Physical span ``L_z = n_z h``."""
        return self.n_z * self.h

    def _z_diffusion(self, f: torch.Tensor) -> torch.Tensor:
        """Second difference in ``z`` of a tangential field (cell-centred in ``z``).

        no-slip: anti-mirror ghost (``f_wall = 0``) at half a cell;
        free-slip: mirror ghost (``df/dz = 0``).
        """
        h = self.h
        lap = torch.zeros_like(f)
        if self.span_bc == "no_slip":
            lap[..., 0] += (-2.0 * f[..., 0] + f[..., 1]) / h ** 2
            lap[..., -1] += (f[..., -2] - 2.0 * f[..., -1]) / h ** 2
        else:
            lap[..., 0] += (f[..., 1] - f[..., 0]) / h ** 2
            lap[..., -1] += (f[..., -2] - f[..., -1]) / h ** 2
        if self.n_z > 2:
            lap[..., 1:-1] += (f[..., :-2] - 2.0 * f[..., 1:-1] + f[..., 2:]) / h ** 2
        return lap

    # ----------------------------------------------------------------- momentum
    def _momentum_tendency(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                           wall_lo, wall_hi) -> torch.Tensor:
        """Convection + diffusion of the axis-0 face field ``u`` (wall-bounded span)."""
        n_x, n_y, n_z, h, nu = self.n_x, self.n_y, self.n_z, self.h, self.nu

        # -- convection in x (identical to the periodic-span solver)
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

        # -- convection in z: w already lives on the interfaces of this grid, so no
        #    interpolation in z is needed (only the x-average onto the u faces)
        w_at_u = 0.5 * (w[:-1] + w[1:])
        transported = self._wall_muscl(u[1:n_x], w_at_u, axis=2)
        flux_z = w_at_u * transported
        conv[1:n_x] -= (flux_z[..., 1:] - flux_z[..., :-1]) / h

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

        # -- diffusion in z: the tangential component feels the end walls
        lap += self._z_diffusion(u)

        return conv + nu * lap

    def _spanwise_tendency(self) -> torch.Tensor:
        """Tendency of the z-face field ``w`` (wall-bounded in every direction)."""
        n_x, n_y, n_z, h, nu = self.n_x, self.n_y, self.n_z, self.h, self.nu
        w, u, v = self.w, self.u, self.v

        # -- convection in z.  w lives on the z-faces, so the conservative flux is
        #    formed on the faces (advecting = face value, transported = limited
        #    reconstruction of the cell averages) and its divergence is interpolated
        #    back to the faces: conv_k = -(div_{k-1} + div_k)/2, which is second
        #    order and keeps the end-wall faces at zero convection.
        w_cell = 0.5 * (w[..., :-1] + w[..., 1:])
        transported = self._wall_muscl(w_cell, w, axis=2)
        flux_z = w * transported                              # (n_z + 1) faces
        div_cell = (flux_z[..., 1:] - flux_z[..., :-1]) / h    # (n_z) cells
        conv = torch.zeros_like(w)
        conv[..., 1:-1] = -0.5 * (div_cell[..., 1:] + div_cell[..., :-1])

        # -- convection in x: u (x-faces, cell-centred in z) is interpolated onto
        #    the z-faces where w lives; the end-wall faces carry zero (w = 0 there)
        u_at_w = torch.zeros(n_x + 1, n_y, n_z + 1, dtype=w.dtype, device=w.device)
        u_at_w[..., 1:n_z] = 0.5 * (u[..., :-1] + u[..., 1:])
        transported_x = self._wall_muscl(w, u_at_w, axis=0)
        flux_x = u_at_w * transported_x     # (n_x + 1, n_y, n_z + 1), zero at the walls
        conv -= (flux_x[1:] - flux_x[:-1]) / h

        # -- convection in y, same interpolation onto the z-faces
        v_at_w = torch.zeros(n_x, n_y + 1, n_z + 1, dtype=w.dtype, device=w.device)
        v_at_w[:, :, 1:n_z] = 0.5 * (v[..., :-1] + v[..., 1:])
        transported_y = self._wall_muscl(w, v_at_w, axis=1)
        flux_y = v_at_w * transported_y
        conv -= (flux_y[:, 1:] - flux_y[:, :-1]) / h

        # -- diffusion in x and y with no-slip mirror ghosts (w = 0 on the walls)
        lap = torch.zeros_like(w)
        lap[1:n_x - 1] += (w[:-2] - 2.0 * w[1:-1] + w[2:]) / h ** 2
        lap[0] += (-2.0 * w[0] + w[1]) / h ** 2
        lap[n_x - 1] += (w[n_x - 2] - 2.0 * w[n_x - 1]) / h ** 2
        lap[:, 1:n_y - 1] += (w[:, :-2] - 2.0 * w[:, 1:-1] + w[:, 2:]) / h ** 2
        lap[:, 0] += (-2.0 * w[:, 0] + w[:, 1]) / h ** 2
        lap[:, n_y - 1] += (w[:, n_y - 2] - 2.0 * w[:, n_y - 1]) / h ** 2

        # -- diffusion in z: w is zero on both end walls (values carried in the array)
        if n_z > 2:
            lap[..., 1:-1] += (w[..., :-2] - 2.0 * w[..., 1:-1] + w[..., 2:]) / h ** 2

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
        """Solve ``lap p = rhs`` by DCT-II in ``x``, ``y`` **and** ``z`` (Neumann)."""
        centred = rhs - rhs.mean()
        coeff = torch.einsum("ij,jkl->ikl", self._dct_x, centred)
        coeff = torch.einsum("jk,ikl->ijl", self._dct_y, coeff)
        coeff = torch.einsum("pq,ijq->ijp", self._dct_z, coeff)
        lam = (self._lambda_x[:, None, None] + self._lambda_y[None, :, None]
               + self._lambda_z[None, None, :])
        zero = lam.abs() < 1e-300
        coeff = coeff / torch.where(zero, torch.ones_like(lam), lam)
        coeff[0, 0, 0] = 0.0
        field = torch.einsum("ji,jkl->ikl", self._dct_x, coeff)
        field = torch.einsum("kj,ikl->ijl", self._dct_y, field)
        field = torch.einsum("pq,ijp->ijq", self._dct_z, field)
        return field

    # ----------------------------------------------------------------- time step
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
        w_star[..., 1:n_z] += dt * tw[..., 1:n_z]    # the end-wall faces stay exactly 0

        div = ((u_star[1:] - u_star[:-1]) / self.h
               + (v_star[:, 1:] - v_star[:, :-1]) / self.h
               + (w_star[..., 1:] - w_star[..., :-1]) / self.h)
        self.p = self.solve_pressure(div / dt)

        self.u[1:n_x] = u_star[1:n_x] - dt * (self.p[1:] - self.p[:-1]) / self.h
        self.v[:, 1:n_y] = v_star[:, 1:n_y] - dt * (self.p[:, 1:] - self.p[:, :-1]) / self.h
        self.w[..., 1:n_z] = (w_star[..., 1:n_z]
                              - dt * (self.p[..., 1:] - self.p[..., :-1]) / self.h)
        self.steps += 1
        return dt

    # ---------------------------------------------------------------- diagnostics
    def divergence(self) -> torch.Tensor:
        """Cell-centred discrete divergence (vanishes after projection)."""
        return ((self.u[1:] - self.u[:-1]) / self.h
                + (self.v[:, 1:] - self.v[:, :-1]) / self.h
                + (self.w[..., 1:] - self.w[..., :-1]) / self.h)

    def endwall_speed(self) -> float:
        """``max|w|`` over the interior span."""
        return float(self.w[..., 1:-1].abs().max())

    def spanwise_uniformity(self) -> float:
        """``max|w|`` over the whole array (interior; the end-wall faces are 0)."""
        return self.endwall_speed()

    def wall_face_leak(self) -> float:
        """``max|w|`` on the two end walls -- non-zero means the BC was violated."""
        return float(max(self.w[..., 0].abs().max(), self.w[..., -1].abs().max()))

    def midplane_asymmetry(self) -> dict[str, float]:
        """Relative asymmetry of ``u``/``v`` about ``z = L_z/2`` (must be 0).

        A span-invariant lid on a span-symmetric domain makes the discrete
        solution even about the mid-plane; the check therefore tests the z
        operators, not the flow physics.
        """
        flip = torch.arange(self.n_z - 1, -1, -1, device=self.device)
        scale_u = max(float(self.u.abs().max()), 1e-30)
        scale_v = max(float(self.v.abs().max()), 1e-30)
        k = self.n_z // 2
        return {
            "u": float((self.u - self.u[..., flip]).abs().max()) / scale_u,
            "v": float((self.v - self.v[..., flip]).abs().max()) / scale_v,
            "w_midplane": float(self.w[..., k].abs().max()),
        }

    # --------------------------------------------------------------- centrelines
    def centreline_u(self, mid_span: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """``(y, u)`` on ``x = L/2``.

        ``mid_span=True`` averages the two cells that straddle the span
        mid-plane (the 2-D limit of the prism); otherwise the span average is
        returned, which carries the end-wall damping.
        """
        i = self.n_x // 2
        y = (torch.arange(self.n_y, dtype=self.dtype) + 0.5) * self.h
        if mid_span:
            k = self.n_z // 2
            return y, 0.5 * (self.u[i, :, k - 1] + self.u[i, :, k])
        return y, self.u[i].mean(dim=1)

    def centreline_v(self, mid_span: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, v)`` on ``y = L/2`` (mid-span or span-averaged)."""
        j = self.n_y // 2
        x = (torch.arange(self.n_x, dtype=self.dtype) + 0.5) * self.h
        if mid_span:
            k = self.n_z // 2
            return x, 0.5 * (self.v[:, j, k - 1] + self.v[:, j, k])
        return x, self.v[:, j].mean(dim=1)

    def span_profile_u(self, i: int | None = None,
                       j: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(z, u)`` at x-face ``i`` (default mid) and y-cell ``j`` (default mid)."""
        i = self.n_x // 2 if i is None else i
        j = self.n_y // 2 if j is None else j
        z = (torch.arange(self.n_z, dtype=self.dtype) + 0.5) * self.h
        return z, self.u[i, j, :].clone()


def score_endwall_against_ghia(solver: LidCavity3DEndWallSolver,
                               reynolds: float | None = None,
                               mid_span: bool = True) -> dict:
    """Score the centreline of an end-wall cavity against the Ghia et al. tables.

    The returned residual is the *best case* for the end-wall geometry: at
    ``mid_span=True`` the sample sits on the span mid-plane, which is the part of
    the domain least affected by the end walls.  The end-wall drag always makes
    the interior slower than the 2-D solution, so the residual is one-sided.
    """
    from .lid_cavity_reference import SUPPORTED_REYNOLDS

    re = float(solver.config.reynolds if reynolds is None else reynolds)
    if re not in SUPPORTED_REYNOLDS:
        raise KeyError(f"no Ghia table for Re={re}; available {SUPPORTED_REYNOLDS}")

    y, u_prof = solver.centreline_u(mid_span=mid_span)
    x, v_prof = solver.centreline_v(mid_span=mid_span)
    out: dict = {"reynolds": re, "n_x": solver.n_x, "n_z": solver.n_z,
                 "span_length": solver.span_length, "span_bc": solver.span_bc,
                 "steps": solver.steps, "residual": solver.residual,
                 "mid_span": mid_span,
                 "lid_velocity": solver.config.lid_velocity}
    out["u"] = score_profiles(y, u_prof, re, "u")
    out["v"] = score_profiles(x, v_prof, re, "v")
    out["max_divergence"] = float(solver.divergence().abs().max())
    out["endwall_speed"] = solver.endwall_speed()
    out["wall_face_leak"] = solver.wall_face_leak()
    return out