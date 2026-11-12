"""Spalart--Allmaras one-equation closure on the periodic-hill finite-volume mesh.

Mission 16880 (selfstart, TensorFVM), round 5 of ``CASE-FVM-PERIODIC-HILLS``.

Round 4 stabilised the Re = 2800 hill run and then showed that its remaining
10 %-gate failure is a **model-class** problem, not a discretisation problem:
refining 48x24 -> 96x48 moved the mean reattachment point by 0.025 % while the
value stayed 52 % too long (``x_r/H = 8.137`` against Breuer et al. 2009's
5.34).  A 2D laminar Navier--Stokes solver keeps the separation bubble attached
far downstream; the reference is a turbulent simulation.  This module adds the
missing ingredient -- an eddy-viscosity closure -- so the same solver can be run
as 2D URANS.

Model
-----
Spalart & Allmaras (1992, *AIAA J.* 32(7):1598, doi:10.2514/3.12149), the
"original" form without the trip term and without the ``ft2`` correction (both
irrelevant here: no transition is being modelled and the run is fully turbulent
from the start):

    D nu~ / Dt = cb1 S~ nu~  -  cw1 fw (nu~ / d)^2
                 + (1/sigma) [ div((nu + nu~) grad nu~) + cb2 |grad nu~|^2 ]

    nu_t = nu~ fv1,   fv1 = chi^3 / (chi^3 + cv1^3),   chi = nu~ / nu
    S~   = |omega| + nu~ fv2 / (kappa^2 d^2),          fv2 = 1 - chi/(1 + chi fv1)
    r    = nu~ / (S~ kappa^2 d^2),  g = r + cw2 (r^6 - r)
    fw   = g [ (1 + cw3^6) / (g^6 + cw3^6) ]^(1/6)
    cw1  = cb1/kappa^2 + (1 + cb2)/sigma

Discretisation on the terrain-following mesh
--------------------------------------------
* convection: the solver's own conservative face fluxes (``_convection`` with
  ``blend = 0``, i.e. first-order upwind -- a bounded choice for a transported
  scalar that must stay non-negative);
* diffusion: the same face-coefficient operator as momentum, with the
  *variable* coefficient ``nu + nu~`` face-averaged; both walls are Dirichlet
  ``nu~ = 0``;
* destruction term: implicit.  ``cw1 fw nu~^2/d^2`` is linearised as
  ``(cw1 fw nu~^n/d^2) nu~^{n+1}`` on the diagonal, which is what keeps the
  near-wall cell stable (the destruction time scale there is ``d^2/nu~``, far
  shorter than the flow time step once ``nu~ >> nu``);
* production and the ``cb2`` term: explicit, from the previous iterate.

The wall distance is an exact point-to-segment distance to the hill contour
(plus the flat top wall), with the streamwise periodic images included, so it is
correct on the sloped hill surface where a vertical distance would be wrong by
``1/cos(theta)``.

Everything here is a *closure provider*: :class:`PeriodicHillSolver` accepts one
through ``PeriodicHillSolver(config, turbulence=SpalartAllmaras(solver))`` and
otherwise behaves exactly as in round 4.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "SpalartAllmarasConfig",
    "SpalartAllmaras",
    "wall_distance",
    "eddy_viscosity",
]


@dataclass(frozen=True)
class SpalartAllmarasConfig:
    """Standard SA constants (Spalart & Allmaras 1992, table 1)."""

    cb1: float = 0.1355
    cb2: float = 0.622
    sigma: float = 2.0 / 3.0
    kappa: float = 0.41
    cw2: float = 0.3
    cw3: float = 2.0
    cv1: float = 7.1
    #: initial ``nu~ / nu`` in the free stream (the customary 3)
    freestream_ratio: float = 3.0
    #: numerical guard: ``nu_t`` is capped at this multiple of ``nu``
    max_viscosity_ratio: float = 1.0e4
    #: r is clipped to this value (standard SA practice, keeps ``fw`` bounded)
    r_clip: float = 10.0

    def __post_init__(self):
        for name in ("cb1", "cb2", "sigma", "kappa", "cw2", "cw3", "cv1"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid SA constant {name}: {value!r}")

    @property
    def cw1(self) -> float:
        return self.cb1 / self.kappa ** 2 + (1.0 + self.cb2) / self.sigma


def wall_distance(mesh, length_m: float, channel_top_m: float) -> np.ndarray:
    """Exact distance from every cell centre to the nearest wall.

    The bottom wall is the polyline ``(x_vertices, y_vertices[0])`` (periodic in
    ``x``, so the +-L images are searched as well); the top wall is the plane
    ``y = channel_top_m``.  On the flat part of the contour this reduces to the
    vertical distance to round-off, which is what the plane-channel test checks.
    """
    xb = np.asarray(mesh.x_vertices, dtype=float)
    yb = np.asarray(mesh.y_vertices[0], dtype=float)
    x0, x1 = xb[:-1], xb[1:]
    y0, y1 = yb[:-1], yb[1:]
    sx = (x1 - x0)[None, None, :]
    sy = (y1 - y0)[None, None, :]
    seg_len2 = sx ** 2 + sy ** 2
    if np.any(seg_len2 <= 0.0):
        raise ValueError("degenerate wall segment in wall_distance")
    X = mesh.xc
    Y = mesh.yc
    best = np.full(X.shape, np.inf)
    for shift in (-length_m, 0.0, length_m):
        px = X[:, :, None] + shift - x0[None, None, :]
        py = Y[:, :, None] - y0[None, None, :]
        t = np.clip((px * sx + py * sy) / seg_len2, 0.0, 1.0)
        ddx = px - t * sx
        ddy = py - t * sy
        best = np.minimum(best, np.sqrt(ddx ** 2 + ddy ** 2).min(axis=2))
    d_top = np.asarray(channel_top_m, dtype=float) - Y
    dist = np.minimum(best, d_top)
    if np.any(dist <= 0.0):
        raise ValueError("mesh cell centre on or through a wall")
    return dist


def eddy_viscosity(nu_tilde, nu: float, config: SpalartAllmarasConfig):
    """``nu_t = nu~ fv1(chi)`` plus the intermediates used by the source terms."""
    nu_tilde = np.asarray(nu_tilde, dtype=float)
    chi = nu_tilde / nu
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + config.cv1 ** 3)
    nu_t = np.minimum(nu_tilde * fv1, config.max_viscosity_ratio * nu)
    return nu_t, chi, fv1


class SpalartAllmaras:
    """Eddy-viscosity closure state for one :class:`PeriodicHillSolver`."""

    def __init__(self, solver, config: SpalartAllmarasConfig | None = None):
        self.solver = solver
        self.config = config or SpalartAllmarasConfig()
        c = solver.config
        self.distance = wall_distance(solver.mesh, c.length_m, c.channel_top_m)
        # free-stream start; the wall BC is imposed through the transport solve
        self.nu_tilde = np.full_like(self.distance, self.config.freestream_ratio * solver.nu)
        self.steps = 0
        self.last = {}

    # ------------------------------------------------------------ properties
    @property
    def nu(self) -> float:
        return self.solver.nu

    def chi(self) -> np.ndarray:
        return self.nu_tilde / self.nu

    def viscosity(self) -> np.ndarray:
        """Current ``nu_t`` field."""
        return eddy_viscosity(self.nu_tilde, self.nu, self.config)[0]

    def total_viscosity(self) -> np.ndarray:
        return self.nu + self.viscosity()

    def vorticity(self) -> np.ndarray:
        """``|omega| = |dv/dx - du/dy|`` from the wide least-squares gradient."""
        gux, guy = self.solver.gradient(self.solver.u)
        gvx, gvy = self.solver.gradient(self.solver.v)
        return np.abs(gvx - guy)

    # ------------------------------------------------------------ operators
    def _diffusion_operator(self, mu_cell):
        """``(1/Vol) sum_f c_f mu_f (phi_N - phi_P)`` with Dirichlet 0 at walls."""
        from scipy import sparse

        m = self.solver.mesh
        ny, nx = self.solver.ny, self.solver.nx
        n = self.solver.n
        c_e = self.solver.c_e
        c_n = self.solver.c_n
        vols = m.volume
        mu = np.asarray(mu_cell, dtype=float)
        mue = 0.5 * (mu + mu[:, m.ip])
        muw = 0.5 * (mu + mu[:, m.im])
        rows, cols, vals = [], [], []
        diag = np.zeros((ny, nx))
        for j in range(ny):
            p = np.arange(j * nx, (j + 1) * nx)
            ce_e = c_e[j] * mue[j]
            ce_w = c_e[j, m.im] * muw[j]
            cn_s = c_n[j] * mu[j]
            cn_n = c_n[j + 1] * (mu[j] if j == ny - 1 else 0.5 * (mu[j] + mu[j + 1]))
            rows.append(p); cols.append(self.solver._cell(j, np.arange(nx) + 1))
            vals.append(ce_e / vols[j])
            rows.append(p); cols.append(self.solver._cell(j, np.arange(nx) - 1))
            vals.append(ce_w / vols[j])
            if j > 0:
                rows.append(p); cols.append(self.solver._cell(j - 1, np.arange(nx)))
                vals.append(cn_s / vols[j])
            if j < ny - 1:
                rows.append(p); cols.append(self.solver._cell(j + 1, np.arange(nx)))
                vals.append(cn_n / vols[j])
            diag[j] = (ce_e + ce_w + cn_s + cn_n) / vols[j]
        off = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        return off - sparse.diags(diag.ravel())

    def source_terms(self):
        """Spalart--Allmaras coefficients at the current state (all per cell)."""
        cfg = self.config
        d = self.distance
        nu = self.nu
        nu_tilde = self.nu_tilde
        nu_t, chi, fv1 = eddy_viscosity(nu_tilde, nu, cfg)
        omega = self.vorticity()
        fv2 = 1.0 - chi / (1.0 + chi * fv1)
        sbar = omega + nu_tilde / (cfg.kappa ** 2 * d ** 2) * fv2
        sbar = np.maximum(sbar, 1.0e-12)
        r = np.minimum(nu_tilde / (sbar * cfg.kappa ** 2 * d ** 2), cfg.r_clip)
        g = r + cfg.cw2 * (r ** 6 - r)
        fw = g * ((1.0 + cfg.cw3 ** 6) / (g ** 6 + cfg.cw3 ** 6)) ** (1.0 / 6.0)
        production = cfg.cb1 * sbar * nu_tilde
        destruction = cfg.cw1 * fw * nu_tilde ** 2 / d ** 2
        # implicit (linearised) destruction coefficient: d(destruction)/d nu~
        destruction_linear = cfg.cw1 * fw * nu_tilde / d ** 2
        return {
            "nu_t": nu_t,
            "omega": omega,
            "Sbar": sbar,
            "r": r,
            "fw": fw,
            "production": production,
            "destruction": destruction,
            "destruction_linear": destruction_linear,
        }

    # ---------------------------------------------------------------- update
    def update(self, dt: float) -> np.ndarray:
        """One backward-Euler step of the SA transport equation; returns ``nu_t``."""
        from scipy.sparse import diags, identity
        from scipy.sparse.linalg import spsolve

        s = self.solver
        cfg = self.config
        nu = self.nu
        d = self.distance
        phi = self.nu_tilde
        if s._flux_e is None or s._flux_steps != s.steps:
            s.refresh_fluxes()
        flux_e, flux_n = s._flux_e, s._flux_n

        src = self.source_terms()
        mu = nu + np.maximum(src["nu_t"], 0.0)
        laplacian = self._diffusion_operator(mu)

        grad_x, grad_y = s.gradient(phi)
        conv = s._convection(phi, flux_e, flux_n, blend=0.0)
        n = s.n
        lhs = (diags(np.full(n, 1.0 / dt))
               - (1.0 / cfg.sigma) * laplacian
               + diags(src["destruction_linear"].ravel()))
        rhs = (phi.ravel() / dt - conv.ravel() + src["production"].ravel()
               + (cfg.cb2 / cfg.sigma) * (grad_x ** 2 + grad_y ** 2).ravel())
        new = spsolve(lhs.tocsc(), rhs).reshape(phi.shape)
        if not np.all(np.isfinite(new)):
            raise FloatingPointError("SA transport produced a non-finite nu~")
        bad = int(np.count_nonzero(new < 0.0))
        new = np.maximum(new, 0.0)
        cap = cfg.max_viscosity_ratio * nu
        capped = int(np.count_nonzero(new > cap))
        if capped:
            new = np.minimum(new, cap)
        self.nu_tilde = new
        self.steps += 1
        self.last = {
            "dt": float(dt),
            "negative_clipped_cells": bad,
            "capped_cells": capped,
            "max_nu_tilde": float(new.max()),
            "max_nu_t": float(src["nu_t"].max()),
            "min_distance": float(d.min()),
        }
        return src["nu_t"]