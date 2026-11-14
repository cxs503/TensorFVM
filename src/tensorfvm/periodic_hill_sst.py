"""SST k-omega two-equation closure on the periodic-hill finite-volume mesh.

Round 10 of ``CASE-FVM-PERIODIC-HILLS`` (mission 20351, selfstart, TensorFVM);
round 12 (mission 20920) fixed three discretisation defects in the *transport*
of ``k``/``omega`` -- see the "Discretisation" list and ``ROUND12.md``.

Rounds 5--9 attached a Spalart--Allmaras one-equation closure to the Re = 2800
periodic-hill run and found it *saturated*: ``nu_t/nu`` plateaus at ~65 (vs the
~26 expected for a turbulent channel at this Re) with the separation bubble
still open, so the mean reattachment length ``x_r/H`` does not exist in the
time-averaged field and every ``u/U_b`` profile station at ``y/H = 0.5`` is
>70 % off Breuer et al. (2009).  Round 9's scoping note (``ROUND9.md`` §7.1)
names the next step explicitly: **a different closure, not more tuning** --
SST ``k-omega`` (Menter 1994) or a Reynolds-stress transport model.  This module
adds the SST ``k-omega`` closure as a second eddy-viscosity provider so the same
``PeriodicHillSolver`` can be run as 2-D URANS with either closure.

Model
-----
Menter (1994, *AIAA J.* 32(8):1598, doi:10.2514/6.1993-2906), the 2003
revision (Menter, Kuntz & Langtry 2003) as implemented in
:func:`tensorfvm.native_sst3d.sst2003_closure` (the same algebraic closure the
3-D native solver uses, so the 2-D hill closure and the 3-D native solver share
one source of truth for the F1/F2 blending, ``nu_t``, production and
destruction):

    D k / Dt     = P_k - beta* k omega + (nu + sigma_k nu_t) laplacian k
    D omega / Dt = gamma P_k/nu_t - beta omega^2
                   + (nu + sigma_w nu_t) laplacian omega
                   + 2 (1 - F1) sigma_w2 (grad k . grad omega) / omega

    nu_t = a1 k / max(a1 omega, sqrt(S2) F2)

with the standard 2003 constants (``BETA_STAR = 0.09``, ``BETA1 = 0.075``,
``BETA2 = 0.0828``, ``SIGMA_K1 = 0.85``, ``SIGMA_K2 = 1.0``,
``SIGMA_W1 = 0.5``, ``SIGMA_W2 = 0.856``, ``GAMMA1 = 5/9``, ``GAMMA2 = 0.44``,
``A1 = 0.31``) and the ``CD_komega`` cross-diffusion floor ``1e-10``.

Discretisation on the terrain-following mesh
--------------------------------------------
* convection: the solver's own conservative face fluxes (``_convection`` with
  ``blend``), first-order upwind by default (a bounded choice for ``k`` and
  ``omega`` that must stay non-negative);
* diffusion: the same face-coefficient operator as momentum, with the
  *variable* coefficient ``nu + sigma nu_t`` face-averaged (``sigma_k`` /
  ``sigma_omega`` are the **F1-blended** closure fields, and the coefficient is
  applied exactly once -- round 12 removed a second, spurious ``1/sigma`` that
  rounds 10--11 carried); both walls are Dirichlet ``k = 0`` and
  ``omega = wall_omega`` (the low-Re ``60 nu/(beta1 d^2)`` wall condition from
  :func:`tensorfvm.native_sst3d.wall_omega`, imposed through the inhomogeneous
  Dirichlet source :meth:`SSTKOmega.wall_source` -- round 12; before it the
  operator silently used ``omega = 0`` at the wall);
* production of ``omega``: ``gamma P_k / nu_t`` with the F1-blended ``gamma``,
  written as ``gamma * (P_k / nu_t)`` so it stays finite where ``nu_t -> 0``;
* destruction: implicit.  ``beta* k omega`` and ``beta omega^2`` are linearised
  on the diagonal (``beta* omega`` for ``k``, ``2 beta omega`` for ``omega``),
  which keeps the near-wall cell stable;
* production and the ``cb2``-type cross-diffusion term: explicit, from the
  previous iterate.

The wall distance is the same exact point-to-segment distance as the SA closure
(:func:`tensorfvm.periodic_hill_turbulence.wall_distance`), correct on the
sloped hill surface.

Everything here is a *closure provider*: :class:`PeriodicHillSolver` accepts one
through ``PeriodicHillSolver(config, turbulence=SSTKOmega(solver))`` and
otherwise behaves exactly as in round 4.  The two-equation transport is
deliberately kept in the same backward-Euler / implicit-destruction shape as the
SA closure so the two can be compared on identical grids and budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .native_sst3d import sst2003_closure, wall_omega
from .periodic_hill_turbulence import wall_distance

__all__ = [
    "SSTKOmegaConfig",
    "SSTKOmega",
    "velocity_gradient_tensor",
]


@dataclass(frozen=True)
class SSTKOmegaConfig:
    """Standard SST-2003 constants (Menter 1994 / 2003 revision).

    The numerical values are the same ones :func:`native_sst3d.sst2003_closure`
    uses, so the 2-D hill closure and the 3-D native solver share one source of
    truth for the model constants.
    """

    beta_star: float = 0.09
    beta1: float = 0.075
    beta2: float = 0.0828
    sigma_k1: float = 0.85
    sigma_k2: float = 1.0
    sigma_w1: float = 0.5
    sigma_w2: float = 0.856
    gamma1: float = 5.0 / 9.0
    gamma2: float = 0.44
    a1: float = 0.31
    cd_floor: float = 1.0e-10
    #: freestream turbulence intensity ``I`` for the ``k`` seed:
    #: ``k = (I * U_b)^2`` (a small fully-turbulent seed, ~5 % of the bulk speed)
    freestream_intensity: float = 0.05
    #: initial ``nu_t / nu`` in the free stream (so ``omega = k / (ratio * nu)``)
    freestream_nut_ratio: float = 3.0
    #: numerical guard: ``nu_t`` is capped at this multiple of ``nu``
    max_viscosity_ratio: float = 1.0e4
    #: convection blend of the ``k``/``omega`` transport: 0 = first-order
    #: upwind (bounded), 1 = second-order (``beta`` blend)
    convection_blend: float = 0.0
    #: optional production limiter on ``P_k`` (``production_limit * k_destruction``);
    #: ``None`` (default) means no limiting
    production_limit: float | None = None

    def __post_init__(self):
        for name in ("beta_star", "beta1", "beta2", "sigma_k1", "sigma_k2",
                     "sigma_w1", "sigma_w2", "gamma1", "gamma2", "a1"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid SST constant {name}: {value!r}")
        blend = float(self.convection_blend)
        if not (math.isfinite(blend) and 0.0 <= blend <= 1.0):
            raise ValueError(f"invalid convection_blend: {blend!r} (must be in [0, 1])")
        if self.production_limit is not None:
            pl = float(self.production_limit)
            if not math.isfinite(pl) or pl <= 0.0:
                raise ValueError(
                    f"invalid production_limit: {pl!r} (must be positive or None)")


def velocity_gradient_tensor(gux, guy, gvx, gvy):
    """Build the 3x3 velocity-gradient tensor from the 2-D cell gradients.

    ``gux, guy`` are ``du/dx, du/dy``; ``gvx, gvy`` are ``dv/dx, dv/dy``.  The
    out-of-plane components are zero (the hill case is 2-D), which is the
    standard 2-D URANS reduction of the strain/vorticity tensors.
    """
    gux = np.asarray(gux, dtype=float)
    guy = np.asarray(guy, dtype=float)
    gvx = np.asarray(gvx, dtype=float)
    gvy = np.asarray(gvy, dtype=float)
    shape = gux.shape
    G = np.zeros(shape + (3, 3), dtype=float)
    G[..., 0, 0] = gux
    G[..., 0, 1] = guy
    G[..., 1, 0] = gvx
    G[..., 1, 1] = gvy
    return G


class SSTKOmega:
    """SST ``k-omega`` closure provider for :class:`PeriodicHillSolver`.

    Interface-compatible with :class:`periodic_hill_turbulence.SpalartAllmaras`
    (``viscosity`` / ``total_viscosity`` / ``source_terms`` / ``update``), so it
    plugs into the solver through ``turbulence=SSTKOmega(solver)``.
    """

    def __init__(self, solver, config: SSTKOmegaConfig | None = None):
        self.solver = solver
        self.config = config if config is not None else SSTKOmegaConfig()
        self.distance = wall_distance(
            solver.mesh, solver.config.length_m, solver.config.channel_top_m)
        # free-stream start from the bulk velocity (a small fully-turbulent seed):
        #   k = (I * U_b)^2,  omega = k / (nu_t_ratio * nu)  ->  nu_t = ratio * nu
        ub = solver.config.bulk_velocity
        k0 = (self.config.freestream_intensity * ub) ** 2
        self.k = np.full_like(self.distance, k0)
        self.omega = np.full_like(
            self.distance, k0 / (self.config.freestream_nut_ratio * solver.nu))
        self.steps = 0
        self.last = {}

    # ------------------------------------------------------------ properties
    @property
    def nu(self) -> float:
        return self.solver.nu

    def _velocity_gradient(self):
        """3x3 velocity-gradient tensor from the wide least-squares gradient."""
        gux, guy = self.solver.gradient(self.solver.u)
        gvx, gvy = self.solver.gradient(self.solver.v)
        return velocity_gradient_tensor(gux, guy, gvx, gvy)

    def _grad_k(self):
        return self.solver.gradient(self.k)

    def _grad_omega(self):
        return self.solver.gradient(self.omega)

    def closure(self):
        """Full SST-2003 algebraic closure at the current ``k``/``omega`` state."""
        G = self._velocity_gradient()
        gkx, gky = self._grad_k()
        gwx, gwy = self._grad_omega()
        gk = np.stack([gkx, gky, np.zeros_like(gkx)], axis=-1)
        gw = np.stack([gwx, gwy, np.zeros_like(gwx)], axis=-1)
        return sst2003_closure(
            self.k, self.omega, self.distance, self.nu, G, gk, gw,
            rho=1.0, precision="float64")

    def viscosity(self) -> np.ndarray:
        """Current ``nu_t`` field."""
        return self.closure()["nu_t"]

    def total_viscosity(self) -> np.ndarray:
        return self.nu + self.viscosity()

    # ------------------------------------------------------------ operators
    def _diffusion_operator(self, mu_cell, wall_value=None):
        """``(1/Vol) sum_f c_f mu_f (phi_N - phi_P)``, Dirichlet at both walls.

        ``wall_value`` is ``None`` for a homogeneous condition (``k = 0``, the
        correct wall data for the ``k`` equation).  For the ``omega`` equation
        pass a ``(2, nx)`` array of wall values -- row 0 is the hill/bottom wall,
        row 1 the top wall -- and the corresponding inhomogeneous Dirichlet
        contribution is *not* in the returned matrix but in
        :meth:`transport_matrix`'s ``rhs`` companion (:meth:`wall_source`).
        """
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
            # face-averaged coefficients: the south face coefficient must be the
            # same one row ``j - 1`` uses for its north face, or the discrete
            # diffusion operator is neither conservative nor symmetric when
            # ``mu = nu + sigma nu_t`` varies by orders of magnitude (round 12)
            mu_s = mu[j] if j == 0 else 0.5 * (mu[j] + mu[j - 1])
            mu_n = mu[j] if j == ny - 1 else 0.5 * (mu[j] + mu[j + 1])
            cn_s = c_n[j] * mu_s
            cn_n = c_n[j + 1] * mu_n
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
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n))
        return off - sparse.diags(diag.ravel())

    def source_terms(self):
        """SST ``k``/``omega`` coefficients at the current state (all per cell).

        Returns the closure dict plus the implicit (linearised) destruction
        coefficients used to keep the near-wall cells stable.
        """
        cfg = self.config
        cl = self.closure()
        nu = self.nu
        k = self.k
        omega = self.omega
        nu_t = cl["nu_t"]
        # implicit (linearised) destruction coefficients:
        #   d(beta* k omega)/d k   = beta* omega
        #   d(beta omega^2)/d omega = 2 beta omega
        k_destruction_linear = cfg.beta_star * omega
        omega_destruction_linear = 2.0 * cl["beta"] * omega
        # production (per unit mass, already limited by the closure)
        production = cl["production_limited"]
        if cfg.production_limit is not None:
            cap = cfg.production_limit * cl["k_destruction"]
            production = np.minimum(production, cap)
        return {
            **cl,
            "nu_t": nu_t,
            "production": production,
            "k_destruction": cl["k_destruction"],
            "omega_destruction": cl["omega_destruction"],
            "k_destruction_linear": k_destruction_linear,
            "omega_destruction_linear": omega_destruction_linear,
            "cross_diffusion_source": cl["cross_diffusion_source"],
        }

    def omega_wall_values(self, mu=None):
        """Low-Re ``omega`` wall data ``60 nu / (beta1 d^2)`` at both walls.

        ``d`` is the first-cell wall distance of the terrain-following mesh, so
        the value follows the sloped hill surface.  Row 0 is the bottom
        (hill) wall, row 1 the flat top wall; shape ``(2, nx)``.
        """
        d = self.distance
        return np.stack([wall_omega(self.nu, d[0]), wall_omega(self.nu, d[-1])])

    # ------------------------------------------------------------- operators
    def wall_source(self, mu_cell, wall_value):
        """Inhomogeneous-Dirichlet contribution ``c_f mu_f phi_wall / Vol`` per cell.

        Returns a ``(ny, nx)`` array to be added to the right-hand side of
        ``I/dt - D`` (the operator itself keeps the same diagonal, because the
        face conductance is unchanged -- only the boundary *value* moves).
        """
        if wall_value is None:
            return np.zeros((self.solver.ny, self.solver.nx))
        wall = np.asarray(wall_value, dtype=float)
        if wall.shape != (2, self.solver.nx):
            raise ValueError(
                f"wall_value must have shape (2, nx), got {wall.shape}")
        m = self.solver.mesh
        mu = np.asarray(mu_cell, dtype=float)
        vols = m.volume
        out = np.zeros((self.solver.ny, self.solver.nx))
        out[0] += self.solver.c_n[0] * mu[0] * wall[0] / vols[0]
        out[-1] += self.solver.c_n[self.solver.ny] * mu[-1] * wall[1] / vols[-1]
        return out

    def transport_matrix(self, mu, src_dest_lin, dt):
        """Implicit SST scalar operator ``A = I/dt - D(mu) + diag(dest_lin)``.

        ``mu`` is the **full** diffusion coefficient of the transported scalar,
        ``nu + sigma nu_t`` with the F1-blended ``sigma`` of
        :func:`native_sst3d.sst2003_closure`.  It is applied once: there is no
        further ``1/sigma`` scaling (round 12 fix; rounds 10--11 solved
        ``D((nu + sigma nu_t)/sigma)`` instead).
        """
        from scipy.sparse import diags

        laplacian = self._diffusion_operator(mu)
        n = self.solver.n
        return (diags(np.full(n, 1.0 / dt))
                - laplacian
                + diags(np.asarray(src_dest_lin).ravel()))

    def _transport_step(self, phi, mu, src_prod, src_dest_lin, src_extra,
                        dt, wall_value=None):
        """One backward-Euler step of a transported scalar; returns new ``phi``."""
        from scipy.sparse.linalg import spsolve

        s = self.solver
        if s._flux_e is None or s._flux_steps != s.steps:
            s.refresh_fluxes()
        flux_e, flux_n = s._flux_e, s._flux_n
        conv = s._convection(phi, flux_e, flux_n, blend=self.config.convection_blend)
        lhs = self.transport_matrix(mu, src_dest_lin, dt)
        rhs = (phi.ravel() / dt - conv.ravel() + src_prod.ravel()
               + src_extra.ravel() + self.wall_source(mu, wall_value).ravel())
        new = spsolve(lhs.tocsc(), rhs).reshape(phi.shape)
        if not np.all(np.isfinite(new)):
            raise FloatingPointError("SST transport produced a non-finite field")
        return new

    def update(self, dt: float) -> np.ndarray:
        """One backward-Euler step of the ``k``/``omega`` transport; returns ``nu_t``."""
        s = self.solver
        cfg = self.config
        nu = self.nu
        src = self.source_terms()
        nu_t = np.maximum(src["nu_t"], 0.0)
        # full diffusion coefficients nu + sigma nu_t, sigmas F1-blended by the
        # closure (round 12: applied once, no further 1/sigma)
        mu_k = nu + np.asarray(src["sigma_k"]) * nu_t
        mu_w = nu + np.asarray(src["sigma_omega"]) * nu_t

        # k transport: production - beta* k omega  (k = 0 at both walls)
        new_k = self._transport_step(
            self.k, mu_k, src["production"], src["k_destruction_linear"],
            np.zeros_like(self.k), dt)
        # omega transport: gamma P_k/nu_t - beta omega^2 + cross-diffusion.
        # P_k = nu_t * (P_k/nu_t), so the ratio form stays finite as nu_t -> 0.
        ratio = np.divide(src["production"], nu_t, out=np.zeros_like(nu_t),
                          where=nu_t > 0.0)
        omega_prod = np.asarray(src["gamma"]) * ratio
        new_omega = self._transport_step(
            self.omega, mu_w, omega_prod, src["omega_destruction_linear"],
            src["cross_diffusion_source"], dt,
            wall_value=self.omega_wall_values(mu_w))

        # positivity + cap
        bad_k = int(np.count_nonzero(new_k < 0.0))
        new_k = np.maximum(new_k, 0.0)
        bad_w = int(np.count_nonzero(new_omega <= 0.0))
        new_omega = np.maximum(new_omega, 1e-12)
        cap = cfg.max_viscosity_ratio * nu
        # cap nu_t via k (nu_t = a1 k / denom, denom >= a1 omega)
        k_cap = cap * np.maximum(cfg.a1 * new_omega, 1e-12) / cfg.a1
        capped = int(np.count_nonzero(new_k > k_cap))
        new_k = np.minimum(new_k, k_cap)

        self.k = new_k
        self.omega = new_omega
        self.steps += 1
        self.last = {"neg_k": bad_k, "neg_omega": bad_w, "cap_k": capped, "dt": dt}
        return self.closure()["nu_t"]
