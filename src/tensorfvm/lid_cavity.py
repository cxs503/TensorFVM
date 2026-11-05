"""Incompressible lid-driven cavity in 2-D on a staggered MAC finite-volume grid.

Why this module exists (verification debt)
------------------------------------------
TensorFVM carries plenty of channel / cylinder / airfoil / overset evidence but
no closed-cavity case at all, even though the lid-driven cavity is the single
most reproduced incompressible benchmark (Ghia, Ghia & Shin 1982) and the
platform gap tracker lists ``lid_driven_cavity_3d`` as an escalated, unclosed
gap.  A closed box is also the hardest boundary-condition problem for a
projection solver: every wall is a wall, so the pressure Poisson problem is pure
Neumann and the discrete compatibility condition ``sum(div u*) == 0`` must hold
to machine precision instead of being absorbed by an outlet.

What is implemented
-------------------
A single-rank staggered (MAC) finite-volume discretisation of

    du/dt + div(u u) = -grad p / rho + nu lap u,    div u = 0

on ``n x n`` uniform cells covering ``[0, L]^2`` with

* ``u`` (normal x-velocity) on vertical faces, shape ``(n + 1, n)``,
* ``v`` (normal y-velocity) on horizontal faces, shape ``(n, n + 1)``,
* ``p`` cell-centred with a pure-Neumann Poisson problem solved by an
  orthonormal DCT-II matrix diagonalisation (torch matmul, zero mode pinned, so
  the pressure datum is arbitrary),
* no-slip on three walls and a tangential lid velocity ``U`` on ``y = L``
  imposed through the standard mirror ghost ``2 U - u_wall-adjacent``,
* convection in conservative flux form with a van Leer (harmonic, TVD) limited
  face reconstruction, which stays bounded at cell Reynolds numbers
  ``Re_h = O(10)`` where plain central differencing oscillates,
* explicit forward-Euler marching to a *steady state* (the benchmark is steady)
  with a CFL / viscous-limited time step.  The default ``cfl = 0.15`` is tighter
  than the usual 0.5 because the van Leer face reconstruction is unstable
  against the moving-lid shear layer at ``cfl = 0.4``: at ``n = 64`` and
  ``Re = 100`` that Courant number produces a steady checkerboard flip in the
  lid-adjacent cells (``max|du| = 1.04`` per step, ``|u|max = 1.27 > U``) instead
  of decaying, while ``cfl = 0.15`` stays quiet (``max|du| = 3.6e-4``).  The
  converged steady state is the same, so only the cost changes.

The y-momentum operator is obtained by calling the same routine with the fields
transposed: on a square grid the v-equation is the u-equation in ``(y, x)`` with
``(v, u)``, which removes a whole class of copy/paste index bugs.

Reference data and scoring live in :mod:`tensorfvm.lid_cavity_reference`; the
literature comparison is driven by ``scripts/verify_lid_cavity_ghia.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from .lid_cavity_reference import SUPPORTED_REYNOLDS, table

TINY = 1e-300


@dataclass(frozen=True)
class LidCavityConfig:
    """Discretisation and solver controls for :class:`LidCavitySolver`."""

    n: int = 96
    reynolds: float = 1000.0
    lid_velocity: float = 1.0
    density: float = 1.0
    cfl: float = 0.15
    time_step: float | None = None
    max_steps: int = 400_000
    tolerance: float = 1e-9
    check_every: int = 100
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("n", "max_steps", "check_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f"{name} must be an integer >= 4")
        for name in ("reynolds", "lid_velocity", "density", "cfl", "tolerance"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.time_step is not None:
            if not math.isfinite(self.time_step) or self.time_step <= 0:
                raise ValueError("time_step must be a positive finite number")
        if self.n % 2:
            raise ValueError("n must be even so the mid-plane sits on a grid line")

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity from ``Re = U L / nu`` with ``L = 1``."""
        return self.lid_velocity / self.reynolds


class LidCavitySolver:
    """Steady lid-driven cavity on a staggered FV grid (see module docstring)."""

    def __init__(self, config: LidCavityConfig | None = None) -> None:
        self.config = config or LidCavityConfig()
        cfg = self.config
        self.n = cfg.n
        self.h = 1.0 / cfg.n
        self.nu = cfg.viscosity
        self.dtype = torch.float64
        self.device = torch.device(cfg.device)
        self.u = torch.zeros(cfg.n + 1, cfg.n, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(cfg.n, cfg.n + 1, dtype=self.dtype, device=self.device)
        self.p = torch.zeros(cfg.n, cfg.n, dtype=self.dtype, device=self.device)
        self.steps = 0
        self.residual = float("inf")
        self._lambda = self._neumann_eigenvalues()
        self._dct = self._dct_matrix()
        self._dirichlet_lambda = self._dirichlet_eigenvalues()
        # static "does this face lie inside the domain" masks are implicit in the
        # slicing below; nothing else has to be rebuilt per step.

    # ---------------------------------------------------------------- geometry
    def _neumann_eigenvalues(self) -> torch.Tensor:
        k = torch.arange(self.n, dtype=self.dtype, device=self.device)
        return (2.0 * torch.cos(math.pi * k / self.n) - 2.0) / self.h ** 2

    def _dct_matrix(self) -> torch.Tensor:
        """Orthonormal DCT-II matrix ``C[k, j] = a_k cos(pi k (j+1/2) / n)``."""
        k = torch.arange(self.n, dtype=self.dtype, device=self.device)[:, None]
        j = torch.arange(self.n, dtype=self.dtype, device=self.device)[None, :]
        mat = torch.cos(math.pi * k * (j + 0.5) / self.n)
        scale = torch.full((self.n, 1), math.sqrt(2.0 / self.n), dtype=self.dtype,
                           device=self.device)
        scale[0] = math.sqrt(1.0 / self.n)
        return mat * scale

    def divergence(self) -> torch.Tensor:
        """Cell-centred discrete divergence (vanishes after projection)."""
        return ((self.u[1:, :] - self.u[:-1, :]) / self.h
                + (self.v[:, 1:] - self.v[:, :-1]) / self.h)

    def cell_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell-centred velocity components (arithmetic face averages)."""
        return (0.5 * (self.u[:-1, :] + self.u[1:, :]),
                0.5 * (self.v[:, :-1] + self.v[:, 1:]))

    def vorticity(self) -> torch.Tensor:
        """Cell-centred vorticity ``dv/dx - du/dy`` from cell-centre velocities.

        Face derivatives are first averaged to the cell centres; at the walls the
        outermost interface value is replicated (the wall vorticity itself is a
        one-sided quantity and is not used for any acceptance metric).
        """
        u_c, v_c = self.cell_velocity()
        du_dy = (u_c[:, 1:] - u_c[:, :-1]) / self.h     # interfaces y = h..(n-1)h
        dv_dx = (v_c[1:, :] - v_c[:-1, :]) / self.h     # interfaces x = h..(n-1)h
        du_dy = torch.cat((du_dy[:, :1], du_dy, du_dy[:, -1:]), dim=1)   # y = 0..n
        dv_dx = torch.cat((dv_dx[:1, :], dv_dx, dv_dx[-1:, :]), dim=0)   # x = 0..n
        return 0.5 * (dv_dx[1:, :] + dv_dx[:-1, :]) - 0.5 * (du_dy[:, 1:] + du_dy[:, :-1])

    def neumann_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Discrete Laplacian with zero-flux (mirror) ghosts on all four sides."""
        lap = (torch.cat((field[:1, :], field[:-1, :]), 0)
               + torch.cat((field[1:, :], field[-1:, :]), 0)
               + torch.cat((field[:, :1], field[:, :-1]), 1)
               + torch.cat((field[:, 1:], field[:, -1:]), 1)
               - 4.0 * field) / self.h ** 2
        return lap

    def _dst_matrix(self) -> torch.Tensor:
        """Orthonormal DST-I matrix ``S[k, j] = b sin(pi (k+1)(j+1)/(n+1))``."""
        k = torch.arange(self.n, dtype=self.dtype, device=self.device)[:, None]
        j = torch.arange(self.n, dtype=self.dtype, device=self.device)[None, :]
        return (torch.sin(math.pi * (k + 1) * (j + 1) / (self.n + 1))
                * math.sqrt(2.0 / (self.n + 1)))

    def _dirichlet_eigenvalues(self) -> torch.Tensor:
        k = torch.arange(self.n, dtype=self.dtype, device=self.device)
        return (2.0 * torch.cos(math.pi * (k + 1) / (self.n + 1)) - 2.0) / self.h ** 2

    def streamfunction(self) -> torch.Tensor:
        """Cell-centred streamfunction with ``psi = 0`` on all walls.

        Solves ``lap psi = -omega`` (Dirichlet) with a DST-I diagonalisation, so
        the primary vortex can be located at the minimum of ``psi`` instead of at
        the minimum of the speed (which sits in a stagnant corner cell).
        """
        dst = self._dst_matrix()
        rhs = -self.vorticity()
        coeff = dst @ rhs @ dst.t()
        lam = (self._dirichlet_lambda[:, None] + self._dirichlet_lambda[None, :])
        return dst.t() @ (coeff / lam) @ dst

    # --------------------------------------------------------------- operators
    @staticmethod
    def _vanleer_face(f: torch.Tensor, vel: torch.Tensor, axis: int) -> torch.Tensor:
        """van Leer limited interface values of ``f`` at the faces along ``axis``.

        ``f`` holds ``M`` samples along ``axis``; ``vel`` holds the ``M - 1``
        normal velocities at the interfaces between them.  The limited slope is
        the harmonic mean ``2 a b / (a + b)`` of the upwind and local
        differences, zeroed where they change sign (TVD, no new extrema).
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
        out = torch.where(va >= 0.0, forward, backward)
        return out.movedim(0, axis)

    def _momentum_tendency(self, u: torch.Tensor, v: torch.Tensor,
                           wall_lo: float, wall_hi: float) -> torch.Tensor:
        """Convection + diffusion tendency of the axis-0 face field ``u``.

        ``wall_lo`` / ``wall_hi`` are the tangential wall velocities at the low
        and high ends of axis 1 (``y = 0`` and ``y = L``).
        """
        n, h, nu = self.n, self.h, self.nu

        # convection in x: transported and advecting quantity are both u
        ux = 0.5 * (u[:-1, :] + u[1:, :])                        # (n, n)
        flux_x = ux * self._vanleer_face(u, ux, axis=0)           # face i + 1/2
        conv = torch.zeros_like(u)
        conv[1:n, :] = -(flux_x[1:, :] - flux_x[:-1, :]) / h

        # convection in y: transported u, advecting v; wall fluxes are zero
        v_face = 0.5 * (v[0:n - 1, :] + v[1:n, :])                # rows i = 1..n-1
        face_u = self._vanleer_face(u[1:n, :], v_face[:, 1:n], axis=1)
        flux_y = torch.zeros(n - 1, n + 1, dtype=u.dtype, device=u.device)
        flux_y[:, 1:n] = v_face[:, 1:n] * face_u
        conv[1:n, :] -= (flux_y[:, 1:] - flux_y[:, :-1]) / h

        # diffusion in x: u = 0 (normal velocity) on both side walls
        lap = torch.zeros_like(u)
        lap[1:n, :] = (u[:-2, :] - 2.0 * u[1:-1, :] + u[2:, :]) / h ** 2

        # diffusion in y with mirror ghosts carrying the tangential wall speed
        below = 2.0 * wall_lo - u[:, 0]
        above = 2.0 * wall_hi - u[:, n - 1]
        lap[:, 0] += (below - 2.0 * u[:, 0] + u[:, 1]) / h ** 2
        if n > 2:
            lap[:, 1:n - 1] += (u[:, :-2] - 2.0 * u[:, 1:-1] + u[:, 2:]) / h ** 2
        lap[:, n - 1] += (u[:, n - 2] - 2.0 * u[:, n - 1] + above) / h ** 2

        return conv + nu * lap

    def tendencies(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Momentum right-hand sides without the pressure gradient."""
        lid = self.config.lid_velocity
        tu = self._momentum_tendency(self.u, self.v, 0.0, lid)
        tv_t = self._momentum_tendency(self.v.t().contiguous(),
                                       self.u.t().contiguous(), 0.0, 0.0)
        return tu, tv_t.t().contiguous()

    # ---------------------------------------------------------------- pressure
    def solve_pressure(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve ``lap p = rhs`` with homogeneous Neumann BCs (DCT-II)."""
        centred = rhs - rhs.mean()            # discrete compatibility
        coeff = self._dct @ centred @ self._dct.t()
        lam = self._lambda[:, None] + self._lambda[None, :]
        spec = coeff / torch.where(lam == 0.0, torch.ones_like(lam), lam)
        spec[0, 0] = 0.0                      # arbitrary pressure datum
        return self._dct.t() @ spec @ self._dct

    # -------------------------------------------------------------- time step
    def stable_time_step(self) -> float:
        cfg = self.config
        if cfg.time_step is not None:
            return float(cfg.time_step)
        speed = max(float(self.u.abs().max()), float(self.v.abs().max()),
                    cfg.lid_velocity)
        conv = cfg.cfl * self.h / speed
        visc = 0.25 * self.h ** 2 / self.nu
        return min(conv, visc)

    def step(self, dt: float | None = None) -> float:
        """Advance one explicit projection step and return the step size used."""
        dt = self.stable_time_step() if dt is None else float(dt)
        tu, tv = self.tendencies()

        u_star = self.u.clone()
        v_star = self.v.clone()
        u_star[1:self.n, :] += dt * tu[1:self.n, :]
        v_star[:, 1:self.n] += dt * tv[:, 1:self.n]

        div = ((u_star[1:, :] - u_star[:-1, :]) / self.h
               + (v_star[:, 1:] - v_star[:, :-1]) / self.h)
        self.p = self.solve_pressure(div / dt)

        self.u[1:self.n, :] = u_star[1:self.n, :] - dt * (self.p[1:, :] - self.p[:-1, :]) / self.h
        self.v[:, 1:self.n] = v_star[:, 1:self.n] - dt * (self.p[:, 1:] - self.p[:, :-1]) / self.h
        self.steps += 1
        return dt

    def run(self, *, max_steps: int | None = None, tolerance: float | None = None,
            check_every: int | None = None, on_check=None) -> dict[str, object]:
        """March to steady state and return a small convergence ledger."""
        cfg = self.config
        max_steps = cfg.max_steps if max_steps is None else int(max_steps)
        tolerance = cfg.tolerance if tolerance is None else float(tolerance)
        check_every = cfg.check_every if check_every is None else int(check_every)

        dt = 0.0
        history: list[tuple[int, float]] = []
        for _ in range(max_steps):
            before, before_v = self.u.clone(), self.v.clone()
            dt = self.step()
            if self.steps % check_every == 0:
                change = max(float((self.u - before).abs().max()),
                             float((self.v - before_v).abs().max())) / dt
                self.residual = change
                history.append((self.steps, change))
                if on_check is not None:
                    on_check(self, self.steps, change)
                if change <= tolerance:
                    break
        return {
            "steps": self.steps,
            "dt": dt,
            "residual": self.residual,
            "converged": self.residual <= tolerance,
            "history": history,
            "max_divergence": float(self.divergence().abs().max()),
        }

    # ------------------------------------------------------------- centrelines
    def centreline_u(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(y, u)`` on the vertical mid-plane ``x = L/2`` (u lives on faces)."""
        j = self.n // 2
        y = (torch.arange(self.n, dtype=self.dtype) + 0.5) * self.h
        return y, self.u[j, :].clone()

    def centreline_v(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, v)`` on the horizontal mid-plane ``y = L/2``."""
        i = self.n // 2
        x = (torch.arange(self.n, dtype=self.dtype) + 0.5) * self.h
        return x, self.v[:, i].clone()

    def vortex_centre(self) -> tuple[float, float]:
        """Primary-vortex centre = interior minimum of the streamfunction."""
        psi = self.streamfunction()
        mask = torch.zeros_like(psi, dtype=torch.bool)
        mask[1:-1, 1:-1] = True
        idx = int(torch.argmin(psi.masked_fill(~mask, float("inf"))))
        i, j = divmod(idx, self.n)
        return ((i + 0.5) * self.h, (j + 0.5) * self.h)

    def streamfunction_extrema(self) -> dict[str, float]:
        """``psi`` at the primary vortex and on the wall, for the report table."""
        psi = self.streamfunction()
        mask = torch.zeros_like(psi, dtype=torch.bool)
        mask[1:-1, 1:-1] = True
        return {"psi_min": float(psi.min()), "psi_max_interior": float(psi.masked_fill(~mask, float("-inf")).max())}


def hermite_interpolate(x: torch.Tensor, f: torch.Tensor,
                        targets: Iterable[float]) -> torch.Tensor:
    """Catmull-Rom cubic interpolation of ``(x, f)`` at ``targets``.

    Implemented locally so the scoring path needs numpy/torch only.
    """
    import numpy as np

    xs = np.asarray(x.detach().cpu().double(), dtype=float)
    fs = np.asarray(f.detach().cpu().double(), dtype=float)
    slopes = np.empty_like(fs)
    slopes[1:-1] = (fs[2:] - fs[:-2]) / (xs[2:] - xs[:-2])
    slopes[0] = (fs[1] - fs[0]) / (xs[1] - xs[0])
    slopes[-1] = (fs[-1] - fs[-2]) / (xs[-1] - xs[-2])
    t = np.asarray(list(targets), dtype=float)
    idx = np.clip(np.searchsorted(xs, t) - 1, 0, len(xs) - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    f0, f1 = fs[idx], fs[idx + 1]
    m0, m1 = slopes[idx], slopes[idx + 1]
    s = (t - x0) / (x1 - x0)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    out = h00 * f0 + h10 * (x1 - x0) * m0 + h01 * f1 + h11 * (x1 - x0) * m1
    return torch.as_tensor(out, dtype=f.dtype)


def score_profiles(coords, values, reynolds: float, which: str) -> dict:
    """Score one computed centreline against its published table.

    ``coords``/``values`` are the discrete centreline sample (``x`` or ``y`` and the
    velocity component).  The returned block reports the residual on *every*
    published point - that is the historical, reproducible number - and separately on
    the points that survived the reference audit
    (:data:`tensorfvm.lid_cavity_reference.SUSPECT_ENTRIES`).  The two are reported
    side by side so an excluded point can never be mistaken for a fixed solver.
    """
    from .lid_cavity_reference import consistent_points, suspect_entries

    pts = table(reynolds, which)
    targets = [p[0] for p in pts]
    ref = torch.tensor([p[1] for p in pts], dtype=values.dtype)
    got = hermite_interpolate(coords, values, targets)
    delta = got - ref
    jmin = int(torch.argmin(ref))
    jmax = int(torch.argmax(ref))

    flagged = suspect_entries(reynolds, which)
    keep = [j for j, p in enumerate(pts) if p[0] not in flagged]
    band = [float(delta[j]) for j in keep]
    consistent_error = max((abs(v) for v in band), default=0.0)
    consistent_rms = (sum(v * v for v in band) / len(band)) ** 0.5 if band else 0.0

    return {
        "profiles": {"coord": targets, "reference": ref.tolist(),
                     "computed": got.tolist(), "delta": delta.tolist()},
        "max_abs_error": float(delta.abs().max()),
        "rms_error": float(torch.sqrt((delta ** 2).mean())),
        "max_abs_error_consistent": float(consistent_error),
        "rms_error_consistent": float(consistent_rms),
        "consistent_point_count": len(keep),
        "suspect_points": [
            {"coord": p[0], "reference": p[1], "computed": float(got[j]),
             "delta": float(delta[j]), **flagged[p[0]]}
            for j, p in enumerate(pts) if p[0] in flagged
        ],
        "extremum": {
            "min_reference": [pts[jmin][0], pts[jmin][1]],
            "min_computed": [pts[jmin][0], float(got[jmin])],
            "max_reference": [pts[jmax][0], pts[jmax][1]],
            "max_computed": [pts[jmax][0], float(got[jmax])],
        },
    }


def score_against_ghia(solver: LidCavitySolver, reynolds: float | None = None) -> dict:
    """Compare the centreline profiles with the Ghia et al. (1982) tables.

    Returns per-quantity error metrics (``max`` / ``rms``, both in units of the
    lid speed) over all published points, the same metrics over the
    audit-consistent subset, the audit-rejected points, the computed extrema and
    their locations, the primary-vortex centre, and the discretisation data needed
    to interpret them.
    """
    re = float(solver.config.reynolds if reynolds is None else reynolds)
    if re not in SUPPORTED_REYNOLDS:
        raise KeyError(f"no Ghia table for Re={re}; available {SUPPORTED_REYNOLDS}")

    y, u_prof = solver.centreline_u()
    x, v_prof = solver.centreline_v()
    out: dict = {"reynolds": re, "n": solver.n, "steps": solver.steps,
                 "residual": solver.residual, "lid_velocity": solver.config.lid_velocity}
    out["u"] = score_profiles(y, u_prof, re, "u")
    out["v"] = score_profiles(x, v_prof, re, "v")

    vx, vy = solver.vortex_centre()
    out["vortex_centre"] = {"computed": [vx, vy]}
    out["max_divergence"] = float(solver.divergence().abs().max())
    return out


def _registered_disputes() -> dict:
    """Group the reference dispute register by ``(Re, component)`` -> ``{x: record}``."""
    from .lid_cavity_reference import DISPUTES

    grouped: dict = {}
    for (reynolds, which, x), record in DISPUTES.items():
        grouped.setdefault((reynolds, which), {})[round(x, 4)] = record
    return grouped


def ghia_acceptance(result: dict, *, max_error: float = 0.03,
                    min_relative: float = 0.15,
                    allow_registered_disputes: bool = False) -> dict:
    """Turn :func:`score_against_ghia` output into explicit pass/fail facts.

    Two independent criteria, both reported so a partial pass cannot hide:

    ``max_error``    pointwise ``max|u_computed - u_Ghia|`` on both centrelines
                     must stay below ``max_error`` (units of the lid speed);
    ``min_relative`` relative error of the *primary extremum* (the deepest
                     reverse-flow u or the strongest v) must stay below
                     ``min_relative`` (the platform's 15 % acceptance band).

    ``allow_registered_disputes`` drops the *registered* reference entries
    (:data:`tensorfvm.lid_cavity_reference.DISPUTES`) from the pointwise gate.
    It is off by default so the published gate keeps failing; when it is on the
    ignored points are reported back with their published value, the solver's
    delta and the evidence that registered them, so a pass can never hide one.
    """
    from .lid_cavity_reference import VORTEX_CENTRE

    re = float(result["reynolds"])
    checks: dict = {"reynolds": re, "criteria": {}}
    worst = 0.0
    # Both mechanisms: dispute register (main) and excluded points (m3195)
    dispute_register = _registered_disputes() if allow_registered_disputes else {}
    checks["registered_disputes"] = []
    excluded: list = []
    for which in ("u", "v"):
        block = result[which]
        disputed = dispute_register.get((re, which), {})
        # Support legacy result dicts without "profiles" (m3195 compatibility)
        if "profiles" in block:
            profile = block["profiles"]
            pointwise, ignored = [], []
            for coord, delta in zip(profile["coord"], profile["delta"]):
                x = round(float(coord), 4)
                if x in disputed:
                    ignored.append({"x": x, "delta": float(delta),
                                    "published": disputed[x]["published"],
                                    "evidence": disputed[x]["evidence"]})
                else:
                    pointwise.append(abs(float(delta)))
            value = max(pointwise) if pointwise else float("nan")
        else:
            # Legacy path: use max_abs_error directly, no dispute filtering
            value = block["max_abs_error"]
            ignored = []
        worst = max(worst, value)
        checks["criteria"][f"{which}_max_abs_error"] = {
            "value": value, "limit": max_error,
            "points": len(pointwise) if "profiles" in block else 0,
            "ignored": len(ignored),
            "pass": bool(value <= max_error),
        }
        # Registered disputes (main)
        if ignored:
            checks["registered_disputes"].extend(
                [dict(item, component=which) for item in ignored])
        # Consistent-point criterion (m3195)
        # When disputes are ignored, use the non-disputed value for consistency
        if ignored:
            consistent = value  # already excludes disputed points
        else:
            consistent = block.get("max_abs_error_consistent", block["max_abs_error"])
        checks["criteria"][f"{which}_max_abs_error_consistent"] = {
            "value": consistent, "limit": max_error,
            "pass": consistent <= max_error,
        }
        for item in block.get("suspect_points", []):
            excluded.append({"which": which, "coord": item["coord"],
                             "reference": item["reference"],
                             "delta": item["delta"]})
    checks["disputes_declared"] = bool(checks["registered_disputes"]) if allow_registered_disputes else True
    if excluded:
        checks["excluded_reference_points"] = excluded
    ext = result["u"]["extremum"]
    u_ref, u_got = ext["min_reference"][1], ext["min_computed"][1]
    u_rel = abs(u_got - u_ref) / abs(u_ref)
    extv = result["v"]["extremum"]
    v_ref, v_got = extv["max_reference"][1], extv["max_computed"][1]
    v_rel = abs(v_got - v_ref) / abs(v_ref)
    checks["criteria"]["u_min_relative"] = {
        "reference": u_ref, "computed": u_got, "value": u_rel,
        "limit": min_relative, "pass": u_rel <= min_relative,
    }
    checks["criteria"]["v_max_relative"] = {
        "reference": v_ref, "computed": v_got, "value": v_rel,
        "limit": min_relative, "pass": v_rel <= min_relative,
    }
    if re in VORTEX_CENTRE:
        ref_centre = VORTEX_CENTRE[re]
        got = result["vortex_centre"]["computed"]
        distance = math.hypot(got[0] - ref_centre[0], got[1] - ref_centre[1])
        checks["criteria"]["vortex_centre_distance"] = {
            "reference": list(ref_centre), "computed": got,
            "value": distance, "limit": 0.05, "pass": distance <= 0.05,
        }
    checks["max_abs_error"] = worst
    checks["pass"] = all(item["pass"] for item in checks["criteria"].values())
    return checks