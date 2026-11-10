"""VOF ice-water two-phase free-surface solver with analytic anchors.

Physical setting
----------------
An isothermal, immiscible three-field system on a uniform Cartesian grid:

===============  ==========================================================
field            meaning
===============  ==========================================================
``alpha``        water volume fraction  (0 = air/ice, 1 = pure water)
``ice``          ice volume fraction    (0 = fluid,      1 = pure ice)
``1-alpha-ice``  air                    ("void", or a real light fluid)
===============  ==========================================================

Ice is not a fluid here: it is an **immiscible rigid phase** (a floe) whose
volume fraction is transported by the same VOF scheme as the water (so a moving
floe can be followed), while the momentum equation sees the ice as a rigid
region -- velocities are zero on every face that touches an ice cell and the
pressure Poisson problem is closed with a homogeneous Neumann condition there
(no penetration).  This is the standard "rigid marker + VOF transport" model for
ice-floe problems: ice inertia is not resolved, ice motion has to be prescribed
(see limitations in ``docs/FVM_VOF_ICE_WATER.md``).

Numerics
--------
* Staggered MAC grid; cell-centred scalars, face-centred velocities.
* Second-order central diffusion with face-averaged viscosity; first-order
  upwind momentum advection in flux form (documented: this is the accuracy
  limiter of the node).
* Incremental projection with the **reduced pressure** ``p_rgh`` (interFoam
  formulation): the hydrostatic head of a *reference* density ``rho_ref`` is
  removed analytically, so a motionless water body is an exact discrete solution
  instead of free-falling.  Body force ``(rho - rho_ref)/rho * g``.
* Variable-density pressure Poisson
  ``div((1/rho_f) grad phi) = div(u*)/dt`` solved by Jacobi-preconditioned CG
  with a warm start; Dirichlet ``phi = 0`` in every void/gas cell (the free
  surface), homogeneous Neumann on walls and ice, identity rows elsewhere so the
  CG never sees a singular system.
* Interface transport: PLIC (:mod:`tensorfvm.vof_plic`) or THINC
  (:mod:`tensorfvm.vof.thinc`), both flux form and therefore conservative.

Analytic anchors shipped with the module
----------------------------------------
* :func:`ritter_dam_break` -- Ritter (1892) exact shallow-water dam-break
  solution on a dry bed: analytic front position and surface profile (different
  reference from the digitised Martin & Moyce curve already used by
  ``verification/dam_break.py``).
* :func:`standing_wave_elevation` -- linear standing-wave elevation with the
  exact finite-depth dispersion relation.
* :func:`subcell_front` -- sub-cell front location from the hydrograph, so the
  front position is not quantised to ``dx``.

Units: SI (m, s, kg).  All tensors are ``float64``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..vof_plic import advect_plic
from .thinc import advect_thinc

__all__ = [
    "IceWaterConfig",
    "IceWaterVOF",
    "ritter_dam_break",
    "standing_wave_elevation",
    "column_height",
    "subcell_front",
]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class IceWaterConfig:
    """Grid + phase properties + numerical knobs (all SI)."""

    nx: int = 96
    ny: int = 48
    length: float = 2.0
    height: float = 0.4
    rho_water: float = 998.0
    rho_ice: float = 917.0
    rho_air: float = 1.2
    mu_water: float = 1.0e-3
    mu_ice: float = 1.0
    mu_air: float = 1.8e-5
    gravity: float = 9.81
    #: ``"void"``: gas is a passive void (p = 0 Dirichlet, no momentum) -- the
    #: standard free-surface model at a 1:1000 density ratio.
    #: ``"two_fluid"``: gas is a real light fluid with its own momentum; needed
    #: for the density-ratio negative control.
    air_model: str = "void"
    #: Buoyancy density floor: ``rho_eff = max(rho, floor*rho_water)`` in the
    #: body-force term only.  Caps the accelerations in light cells and is
    #: documented as a numerical limiter (``1.0`` disables it).
    buoyancy_density_floor: float = 1.0
    vof_scheme: str = "plic"  # "plic" | "thinc"
    thinc_sharpness: float = 3.0
    cfl: float = 0.35
    top_boundary: str = "wall"  # "wall" | "open"
    poisson_tol: float = 1.0e-10
    poisson_max_iter: int = 400
    warm_start: bool = True
    slip: str = "no_slip"  # "no_slip" | "free_slip"
    #: Transport the ice volume fraction with the same VOF scheme (moving floe).
    ice_advection: bool = True

    def __post_init__(self) -> None:
        if self.nx < 4 or self.ny < 4:
            raise ValueError("nx, ny must be >= 4")
        if self.length <= 0 or self.height <= 0:
            raise ValueError("length, height must be positive")
        if min(self.rho_water, self.rho_ice, self.rho_air) <= 0:
            raise ValueError("densities must be positive")
        if min(self.mu_water, self.mu_ice, self.mu_air) < 0:
            raise ValueError("viscosities must be non-negative")
        if self.air_model not in ("void", "two_fluid"):
            raise ValueError(f"air_model must be 'void' or 'two_fluid', got {self.air_model!r}")
        if self.vof_scheme not in ("plic", "thinc"):
            raise ValueError(f"vof_scheme must be 'plic' or 'thinc', got {self.vof_scheme!r}")
        if not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must be in (0, 1]")
        if self.top_boundary not in ("wall", "open"):
            raise ValueError(f"top_boundary must be 'wall' or 'open', got {self.top_boundary!r}")
        if self.slip not in ("no_slip", "free_slip"):
            raise ValueError(f"slip must be 'no_slip' or 'free_slip', got {self.slip!r}")
        if self.buoyancy_density_floor <= 0:
            raise ValueError("buoyancy_density_floor must be positive")


# --------------------------------------------------------------------------- #
# analytic anchors
# --------------------------------------------------------------------------- #
def ritter_dam_break(x, t: float, h0: float, gravity: float = 9.81, x_dam: float = 0.0):
    """Ritter (1892) exact shallow-water dam-break solution on a dry bed.

    Reservoir at depth ``h0`` for ``x < x_dam``, dry downstream, dam removed at
    ``t = 0``.  With ``c0 = sqrt(g h0)``:

        h(x,t) = h0                                   for x <= x_dam - c0 t
        h(x,t) = (2 c0 - (x-x_dam)/t)^2 / (9 g)       for x_dam - c0 t < x < x_dam + 2 c0 t
        h(x,t) = 0                                    for x >= x_dam + 2 c0 t

    and ``u = (2/3)(c0 + (x-x_dam)/t)`` in the rarefaction fan.  The front sits
    at ``x_front = x_dam + 2 c0 t`` and travels exactly at ``2 c0``.

    Returns ``(h, u, x_front)`` with ``h``/``u`` shaped like ``x``.  Validity:
    shallow water (``h0 << L``) and ``t < L/c0`` (rarefaction must not have
    reached the back wall).
    """
    if h0 <= 0 or gravity <= 0:
        raise ValueError("h0 and gravity must be positive")
    if t <= 0:
        raise ValueError("t must be positive")
    xa = torch.as_tensor(x, dtype=torch.float64)
    c0 = math.sqrt(gravity * h0)
    s = (xa - x_dam) / t
    h = torch.where(
        xa <= x_dam - c0 * t,
        torch.full_like(xa, float(h0)),
        torch.where(
            xa < x_dam + 2.0 * c0 * t,
            (2.0 * c0 - s).clamp(min=0.0) ** 2 / (9.0 * gravity),
            torch.zeros_like(xa),
        ),
    )
    u = torch.where(
        (xa > x_dam - c0 * t) & (xa < x_dam + 2.0 * c0 * t),
        (2.0 / 3.0) * (c0 + s),
        torch.zeros_like(xa),
    )
    return h, u, x_dam + 2.0 * c0 * t


def standing_wave_elevation(
    x, t: float, mean_depth: float, amplitude: float, wavelength: float,
    gravity: float = 9.81,
):
    """Linear standing wave ``eta = a cos(kx) cos(omega t)``.

    ``k = 2*pi/wavelength`` and ``omega`` from the finite-depth dispersion
    relation ``omega^2 = g k tanh(k H)``.  Returns ``(eta, omega)``.
    """
    if mean_depth <= 0 or amplitude <= 0 or wavelength <= 0:
        raise ValueError("mean_depth, amplitude, wavelength must be positive")
    xa = torch.as_tensor(x, dtype=torch.float64)
    k = 2.0 * math.pi / wavelength
    omega = math.sqrt(gravity * k * math.tanh(k * mean_depth))
    return amplitude * torch.cos(k * xa) * math.cos(omega * t), omega


def column_height(alpha: torch.Tensor, dy: float) -> torch.Tensor:
    """Water depth per column ``H_j = dy * sum_i alpha[j, i]`` (hydrograph)."""
    a = torch.as_tensor(alpha, dtype=torch.float64)
    return dy * a.sum(dim=0)


def subcell_front(height, dx: float, threshold: float, x_origin: float = 0.0) -> float:
    """Rightmost sub-cell crossing of the hydrograph through ``threshold``.

    ``height`` is piecewise linear between cell centres, so the crossing is
    interpolated instead of being quantised to a cell.  Returns ``x_origin`` if
    no cell exceeds the threshold.
    """
    h = torch.as_tensor(height, dtype=torch.float64)
    above = (h > threshold).nonzero().reshape(-1)
    if above.numel() == 0:
        return float(x_origin)
    j = int(above[-1].item())
    if j == h.numel() - 1:
        return float(x_origin + (j + 0.5) * dx)
    h_here = float(h[j])
    h_next = float(h[j + 1])
    if h_next >= threshold or h_here == h_next:
        return float(x_origin + (j + 0.5) * dx)
    frac = (h_here - threshold) / (h_here - h_next)
    return float(x_origin + (j + 0.5 + frac) * dx)


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
class IceWaterVOF:
    """Variable-density projection solver + VOF free surface + rigid ice."""

    def __init__(self, config: IceWaterConfig | None = None):
        self.cfg = cfg = config or IceWaterConfig()
        self.dx = cfg.length / cfg.nx
        self.dy = cfg.height / cfg.ny
        if abs(self.dx - self.dy) > 1e-9 * self.dx:
            raise ValueError(
                f"cells must be square: dx={self.dx:.6g} dy={self.dy:.6g} "
                f"(choose length/height = nx/ny)")
        self.alpha = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float64)
        self.ice = torch.zeros_like(self.alpha)
        self.u = torch.zeros((cfg.ny, cfg.nx + 1), dtype=torch.float64)
        self.v = torch.zeros((cfg.ny + 1, cfg.nx), dtype=torch.float64)
        self.p = torch.zeros_like(self.alpha)
        self.step_count = 0
        self.time = 0.0
        self.diagnostics: list[dict] = []

    # -- construction ------------------------------------------------------ #
    def set_water(self, alpha) -> None:
        a = torch.as_tensor(alpha, dtype=torch.float64)
        if a.shape != self.alpha.shape:
            raise ValueError(f"alpha must be {tuple(self.alpha.shape)}, got {tuple(a.shape)}")
        if float(a.min()) < -1e-12 or float(a.max()) > 1.0 + 1e-12:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = a.clone()

    def set_ice(self, ice) -> None:
        i = torch.as_tensor(ice, dtype=torch.float64)
        if i.shape != self.alpha.shape:
            raise ValueError(f"ice must be {tuple(self.alpha.shape)}, got {tuple(i.shape)}")
        if float(i.min()) < -1e-12 or float(i.max()) > 1.0 + 1e-12:
            raise ValueError("ice must lie in [0, 1]")
        self.ice = i.clone()

    def add_ice_block(self, x0: float, x1: float, y0: float, y1: float) -> None:
        """Axis-aligned rigid ice block aligned to the grid."""
        jx0, jx1 = int(round(x0 / self.dx)), int(round(x1 / self.dx))
        jy0, jy1 = int(round(y0 / self.dy)), int(round(y1 / self.dy))
        if not (0 <= jx0 < jx1 <= self.cfg.nx and 0 <= jy0 < jy1 <= self.cfg.ny):
            raise ValueError("ice block outside the domain")
        self.ice[jy0:jy1, jx0:jx1] = 1.0

    # -- derived fields ---------------------------------------------------- #
    @property
    def solid_mask(self) -> torch.Tensor:
        return self.ice >= 0.5

    @property
    def void_mask(self) -> torch.Tensor:
        """Cells carrying no momentum (rigid ice, and void air in ``"void"``)."""
        if self.cfg.air_model == "void":
            return (self.alpha + self.ice < 0.5) | self.solid_mask
        return self.solid_mask

    def density(self) -> torch.Tensor:
        cfg = self.cfg
        air = (1.0 - self.alpha - self.ice).clamp(0.0, 1.0)
        return cfg.rho_water * self.alpha + cfg.rho_ice * self.ice + cfg.rho_air * air

    def viscosity(self) -> torch.Tensor:
        cfg = self.cfg
        air = (1.0 - self.alpha - self.ice).clamp(0.0, 1.0)
        return cfg.mu_water * self.alpha + cfg.mu_ice * self.ice + cfg.mu_air * air

    def water_volume(self) -> float:
        return float(self.alpha.sum()) * self.dx * self.dy

    def ice_volume(self) -> float:
        return float(self.ice.sum()) * self.dx * self.dy

    def max_speed(self) -> float:
        return max(float(self.u.abs().max()), float(self.v.abs().max()))

    def kinetic_energy(self) -> float:
        rho = self.density()
        uc = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        vc = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        return float((0.5 * rho * (uc ** 2 + vc ** 2)).sum()) * self.dx * self.dy

    def momentum(self) -> tuple[float, float]:
        rho = self.density()
        uc = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        vc = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        cell = self.dx * self.dy
        return float((rho * uc).sum()) * cell, float((rho * vc).sum()) * cell

    def hydrograph(self) -> torch.Tensor:
        return column_height(self.alpha, self.dy)

    def front_position(self, fraction: float = 0.01, h0: float | None = None) -> float:
        """Front location with threshold ``fraction * h_ref``."""
        h = self.hydrograph()
        ref = float(h.max()) if h0 is None else h0
        return subcell_front(h, self.dx, fraction * ref)

    # -- discrete operators ------------------------------------------------ #
    def _adv_x(self, f: torch.Tensor, uf: torch.Tensor) -> torch.Tensor:
        """``d(f u)/dx`` on x-faces, first-order upwind donors."""
        donor = torch.where(
            uf > 0,
            torch.cat([f[:, :1], f[:, :-1]], dim=1),
            torch.cat([f[:, 1:], f[:, -1:]], dim=1),
        )
        flux = uf * donor
        out = torch.zeros_like(uf)
        out[:, 1:-1] = (flux[:, 1:-1] - flux[:, :-2]) / self.dx
        return out

    def _adv_y(self, f: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
        """``d(f v)/dy`` on y-faces, first-order upwind donors."""
        donor = torch.where(
            vf > 0,
            torch.cat([f[:1, :], f[:-1, :]], dim=0),
            torch.cat([f[1:, :], f[-1:, :]], dim=0),
        )
        flux = vf * donor
        out = torch.zeros_like(vf)
        out[1:-1, :] = (flux[1:-1, :] - flux[:-2, :]) / self.dy
        return out

    def _to_u_faces(self, f: torch.Tensor) -> torch.Tensor:
        """Average a cell field onto x-faces, ``(ny, nx+1)``."""
        return 0.5 * (torch.cat([f[:, :1], f], dim=1) + torch.cat([f, f[:, -1:]], dim=1))

    def _to_v_faces(self, f: torch.Tensor) -> torch.Tensor:
        """Average a cell field onto y-faces, ``(ny+1, nx)``."""
        return 0.5 * (torch.cat([f[:1, :], f], dim=0) + torch.cat([f, f[-1:, :]], dim=0))

    def _predict(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit momentum predictor: advection + diffusion + buoyancy."""
        cfg = self.cfg
        rho, mu = self.density(), self.viscosity()
        rho_u, rho_v = self._rho_u, self._rho_v
        mu_u, mu_v = self._to_u_faces(mu), self._to_v_faces(mu)

        # ---- u momentum: x-flux + y-flux -------------------------------- #
        adv_u = self._adv_x(self.u, self.u)
        v_at_u = torch.cat(
            [self.v[:, :1], 0.5 * (self.v[:, :-1] + self.v[:, 1:]), self.v[:, -1:]],
            dim=1)  # (ny+1, nx+1): v at x-face abscissae
        adv_u = adv_u + self._flux_y_of(self.u, v_at_u) / self.dy
        ux = torch.cat([self.u[:, :1], self.u, self.u[:, -1:]], dim=1)
        uy = torch.cat([self.u[:1, :], self.u, self.u[-1:, :]], dim=0)
        d2u = (ux[:, 2:] - 2.0 * ux[:, 1:-1] + ux[:, :-2]) / self.dx ** 2
        d2uy = (uy[2:, :] - 2.0 * uy[1:-1, :] + uy[:-2, :]) / self.dy ** 2
        mup = torch.cat([mu_u[:1, :], mu_u, mu_u[-1:, :]], dim=0)
        mu_uy = 0.5 * (mup[2:, :] + mup[:-2, :])
        visc_u = (mu_u[:, 1:-1] * d2u[:, 1:-1]
                  + mu_uy[:, 1:-1] * d2uy[:, 1:-1])
        u_star = self.u.clone()
        u_star[:, 1:-1] = self.u[:, 1:-1] + dt * (
            -adv_u[:, 1:-1] + visc_u / rho_u[:, 1:-1])

        # ---- v momentum --------------------------------------------------- #
        adv_v = self._adv_y(self.v, self.v)
        u_at_v = torch.cat(
            [self.u[:1, :], 0.5 * (self.u[:-1, :] + self.u[1:, :]), self.u[-1:, :]],
            dim=0)  # (ny+1, nx+1): u at y-face abscissae
        adv_v = adv_v + self._flux_x_of(self.v, u_at_v) / self.dx
        vx = torch.cat([self.v[:, :1], self.v, self.v[:, -1:]], dim=1)
        vy = torch.cat([self.v[:1, :], self.v, self.v[-1:, :]], dim=0)
        d2v = (vx[:, 2:] - 2.0 * vx[:, 1:-1] + vx[:, :-2]) / self.dx ** 2
        d2vy = (vy[2:, :] - 2.0 * vy[1:-1, :] + vy[:-2, :]) / self.dy ** 2
        mupv = torch.cat([mu_v[:, :1], mu_v, mu_v[:, -1:]], dim=1)
        mu_vx = 0.5 * (mupv[:, 2:] + mupv[:, :-2])
        visc_v = (mu_vx[1:-1, :] * d2v[1:-1, :]
                  + mu_v[1:-1, :] * d2vy[1:-1, :])
        # Total-pressure projection: gravity is the body force and the
        # hydrostatic head is carried by the pressure field (see the class
        # docstring); no density ratio enters the *acceleration*.
        buo_v = torch.full_like(rho_v, -cfg.gravity)
        v_star = self.v.clone()
        v_star[1:-1, :] = self.v[1:-1, :] + dt * (
            -adv_v[1:-1, :] + visc_v / rho_v[1:-1, :] + buo_v[1:-1, :])
        return u_star, v_star

    def _flux_y_of(self, f: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
        """Divergence (in y) of ``f*v`` at x-face abscissae, shape ``(ny, nx+1)``."""
        donor = torch.where(
            vf > 0,
            torch.cat([f[:1, :], f], dim=0),
            torch.cat([f, f[-1:, :]], dim=0),
        )
        flux = vf * donor  # (ny+1, nx+1)
        return flux[1:, :] - flux[:-1, :]

    def _flux_x_of(self, f: torch.Tensor, uf: torch.Tensor) -> torch.Tensor:
        """Divergence (in x) of ``f*u`` at y-face abscissae, shape ``(ny+1, nx)``."""
        donor = torch.where(
            uf > 0,
            torch.cat([f[:, :1], f], dim=1),
            torch.cat([f, f[:, -1:]], dim=1),
        )
        flux = uf * donor  # (ny+1, nx+1)
        return flux[:, 1:] - flux[:, :-1]

    # -- pressure ---------------------------------------------------------- #
    def _column_surface(self) -> torch.Tensor:
        """Water-surface elevation per column ``y_s = dy * sum_j alpha`` (m)."""
        return self.dy * self.alpha.sum(dim=0)

    def free_surface_pressure(self) -> torch.Tensor:
        """Column-hydrostatic pressure at every cell centre (audit reference).

        ``p = rho_w g (y_s(x) - y)``: positive below the local free surface and
        negative above it.  It is the exact discrete reference for a motionless
        pool and is exported for the tests; the solver does **not** impose it.
        """
        cfg = self.cfg
        y_s = self._column_surface()
        yy = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * self.dy
        return cfg.rho_water * cfg.gravity * (y_s.unsqueeze(0) - yy.unsqueeze(1))

    def _face_distances(self):
        """Effective pressure-stencil distance of every face (sub-cell surface).

        Fluid-fluid faces keep the cell size.  A face between a fluid cell and a
        masked (void/rigid) cell uses the distance from the *fluid cell centre to
        the free surface*, clamped to a quarter cell (ghost-fluid / cut-cell
        treatment of the free surface).  With that distance the discrete
        pressure gradient across the surface face of a hydrostatic column equals
        exactly ``-rho g``: the balanced-force condition that keeps a motionless
        pool motionless, and the reason a dry bed does not see a fictitious
        negative pressure (the atmosphere is p = 0 at the *interface*, not at the
        neighbouring cell centre).
        """
        cfg, free = self.cfg, self._free
        y_s = self._column_surface()
        yy = (torch.arange(cfg.ny, dtype=torch.float64) + 0.5) * self.dy
        d_cell = (y_s.unsqueeze(0) - yy.unsqueeze(1)).abs().clamp(min=0.25 * self.dy)
        dx_eff = torch.full((cfg.ny, cfg.nx + 1), 0.5 * self.dx, dtype=torch.float64)
        dy_eff = torch.full((cfg.ny + 1, cfg.nx), self.dy, dtype=torch.float64)
        fl, fr = free[:, :-1] > 0, free[:, 1:] > 0
        dx_eff[:, 1:-1] = torch.where(fl & fr, torch.full_like(dx_eff[:, 1:-1], self.dx),
                                      torch.full_like(dx_eff[:, 1:-1], 0.5 * self.dx))
        fb, ft = free[:-1, :] > 0, free[1:, :] > 0
        dy_eff[1:-1, :] = torch.where(
            fb & ft, torch.full_like(dy_eff[1:-1, :], self.dy),
            torch.where(fb, d_cell[:-1, :], torch.where(ft, d_cell[1:, :],
                                                        torch.full_like(dy_eff[1:-1, :], self.dy))))
        return dx_eff, dy_eff

    def _face_scalar(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        """Face values of a cell field with the *fluid-side* one-sided rule.

        On a fluid-void (or fluid-ice) face the average of water and air is
        meaningless: the momentum flux and the pressure stencil there are
        governed by the fluid side only.  Using the mixture average would make a
        hydrostatic surface face over-correct by the density ratio -- measured as
        a spurious 1.6 m/s in a motionless pool before this rule was applied.
        """
        free = self._free
        if axis == 1:
            avg = 0.5 * (f[:, :-1] + f[:, 1:])
            fl, fr = free[:, :-1] > 0, free[:, 1:] > 0
            val = torch.where(fl & fr, avg,
                              torch.where(fl, f[:, :-1], torch.where(fr, f[:, 1:], avg)))
            return torch.cat([val[:, :1], val, val[:, -1:]], dim=1)
        avg = 0.5 * (f[:-1, :] + f[1:, :])
        fb, ft = free[:-1, :] > 0, free[1:, :] > 0
        val = torch.where(fb & ft, avg,
                          torch.where(fb, f[:-1, :], torch.where(ft, f[1:, :], avg)))
        return torch.cat([val[:1, :], val, val[-1:, :]], dim=0)

    def _build_faces(self) -> None:
        """Face densities / inverse densities used by momentum and pressure."""
        rho = self.density()
        self._rho_u = self._face_scalar(rho, 1)
        self._rho_v = self._face_scalar(rho, 0)
        self._invrho_u = 1.0 / self._rho_u
        self._invrho_v = 1.0 / self._rho_v

    def _build_poisson(self) -> None:
        # 1. unknown set (fluid cells) -- needed by the one-sided face rule
        self._free = self._is_source_cell().to(torch.float64)
        # A closed two-fluid tank has no Dirichlet surface: the Neumann problem
        # is singular and is made unique by projecting onto mean-zero fields
        # (pinning one cell instead would corrupt the hydrostatic field and
        # generate a spurious flow; measured 0.97 m/s in the density-ratio
        # negative control before this was fixed).
        self._neumann_only = self.cfg.air_model == "two_fluid"
        free = self._free
        self._build_faces()
        # 2. coefficients: off-diagonal only between two fluid cells
        solid = self.solid_mask
        # interior faces only (1..nx-1 and 1..ny-1); the face arrays are already
        # one-sided where the neighbour is masked
        self._invrho_x = self._invrho_u[:, 1:-1]
        self._invrho_y = self._invrho_v[1:-1, :]
        cx = self._invrho_x / self.dx ** 2
        cy = self._invrho_y / self.dy ** 2
        cx = torch.where(~(solid[:, :-1] & solid[:, 1:]), cx, torch.zeros_like(cx))
        cy = torch.where(~(solid[:-1, :] & solid[1:, :]), cy, torch.zeros_like(cy))
        self._dx_eff, self._dy_eff = self._face_distances()
        fl, fr = free[:, :-1] > 0, free[:, 1:] > 0
        fb, ft = free[:-1, :] > 0, free[1:, :] > 0
        self._cx = torch.where(fl & fr, cx, torch.zeros_like(cx))
        self._cy = torch.where(fb & ft, cy, torch.zeros_like(cy))
        # 3. fluid-masked faces: surface Dirichlet p = 0 folded into the diagonal
        diag_x = torch.where((fl ^ fr),
                             self._invrho_x / (self._dx_eff[:, 1:-1] * self.dx),
                             torch.zeros_like(cx))
        diag_y = torch.where((fb ^ ft),
                             self._invrho_y / (self._dy_eff[1:-1, :] * self.dy),
                             torch.zeros_like(cy))
        extra = torch.zeros_like(self.alpha)
        extra[:, :-1] += diag_x
        extra[:, 1:] += diag_x
        extra[:-1, :] += diag_y
        extra[1:, :] += diag_y
        self._diag_extra = extra * free

    def _laplacian(self, q: torch.Tensor) -> torch.Tensor:
        """Positive operator ``sum_f c_f (q_c - q_nb)`` on the fluid subspace."""
        qq = q * self._free
        out = torch.zeros_like(q)
        out[:, :-1] += self._cx * (qq[:, :-1] - qq[:, 1:])
        out[:, 1:] += self._cx * (qq[:, 1:] - qq[:, :-1])
        out[:-1, :] += self._cy * (qq[:-1, :] - qq[1:, :])
        out[1:, :] += self._cy * (qq[1:, :] - qq[:-1, :])
        out = out + self._diag_extra * qq
        return out * self._free + (1.0 - self._free) * q

    def _diag(self) -> torch.Tensor:
        d = torch.zeros_like(self.alpha)
        d[:, :-1] += self._cx
        d[:, 1:] += self._cx
        d[:-1, :] += self._cy
        d[1:, :] += self._cy
        d = d + self._diag_extra
        return d * self._free + (1.0 - self._free)

    def _solve_pressure(self, rhs: torch.Tensor) -> torch.Tensor:
        """Jacobi-preconditioned CG on the variable-density Poisson problem."""
        cfg = self.cfg
        b = rhs * self._free
        if getattr(self, "_neumann_only", False):
            b = b - b.mean()
        x = (self.p.clone() if cfg.warm_start else torch.zeros_like(b)) * self._free
        if getattr(self, "_neumann_only", False):
            x = x - x.mean()
        r = b - self._laplacian(x)
        diag = self._diag().clamp(min=1e-300)
        z = r / diag
        d = z.clone()
        rz = float((r * z).sum())
        b_norm = max(float(b.abs().max()), 1e-300)
        iterations = 0
        for iterations in range(1, cfg.poisson_max_iter + 1):
            if float(r.abs().max()) < cfg.poisson_tol * b_norm:
                break
            ad = self._laplacian(d)
            denom = float((d * ad).sum())
            if abs(denom) < 1e-300:
                break
            step = rz / denom
            x = x + step * d
            r = r - step * ad
            z = r / diag
            rz_new = float((r * z).sum())
            if abs(rz) < 1e-300:
                break
            d = z + (rz_new / rz) * d
            rz = rz_new
        self.last_poisson = dict(iterations=iterations,
                                 residual=float(r.abs().max()) / b_norm)
        return x * self._free

    # -- boundaries -------------------------------------------------------- #
    def _project_velocity(self, u_star, v_star, dt):
        div = ((u_star[:, 1:] - u_star[:, :-1]) / self.dx
               + (v_star[1:, :] - v_star[:-1, :]) / self.dy)
        # Aim of the solve: A phi = -div(u*)/dt, because the corrector
        # u = u* - dt (1/rho) grad(phi) removes the divergence in this sign
        # convention (A is the *positive* operator sum c_f (phi_c - phi_nb)).
        # A rigid or void cell cannot receive fluid: drop its divergence row.
        rhs = -div / dt
        rhs = torch.where(self._is_source_cell(), rhs, torch.zeros_like(rhs))
        phi = self._solve_pressure(rhs)
        invrho_u, invrho_v = self._invrho_u, self._invrho_v
        u = u_star.clone()
        v = v_star.clone()
        full = torch.where(self._free > 0.0, phi, torch.zeros_like(phi))
        u[:, 1:-1] -= dt * invrho_u[:, 1:-1] * (full[:, 1:] - full[:, :-1]) / self._dx_eff[:, 1:-1]
        v[1:-1, :] -= dt * invrho_v[1:-1, :] * (full[1:, :] - full[:-1, :]) / self._dy_eff[1:-1, :]
        return u, v, full

    def _is_source_cell(self) -> torch.Tensor:
        """Cells whose divergence is imposed to be zero (fluid cells)."""
        if self.cfg.air_model == "void":
            return (self.alpha + self.ice >= 0.5) & ~self.solid_mask
        return ~self.solid_mask

    @staticmethod
    def _face_blocked(cell_mask: torch.Tensor, axis: int) -> torch.Tensor:
        """True on a face when either adjacent cell is masked (walls count)."""
        if axis == 1:  # x-faces: (ny, nx) -> (ny, nx+1)
            pad = torch.cat([torch.ones_like(cell_mask[:, :1]), cell_mask,
                             torch.ones_like(cell_mask[:, :1])], dim=1)
            return pad[:, :-1] | pad[:, 1:]
        pad = torch.cat([torch.ones_like(cell_mask[:1, :]), cell_mask,
                         torch.ones_like(cell_mask[:1, :])], dim=0)
        return pad[:-1, :] | pad[1:, :]

    def _apply_bounds(self, u: torch.Tensor, v: torch.Tensor):
        cfg = self.cfg
        solid = self.solid_mask
        u = torch.where(self._face_blocked(solid, 1), torch.zeros_like(u), u)
        v = torch.where(self._face_blocked(solid, 0), torch.zeros_like(v), v)
        if cfg.air_model == "void":
            # A *fluid-void* face must keep its velocity: it is the flux that
            # moves the interface (the void has no momentum of its own, but the
            # water crossing the face does).  Only faces fully inside void or
            # touching ice are dead.
            void = self.void_mask & ~solid
            vx = torch.cat([torch.ones_like(void[:, :1]), void,
                            torch.ones_like(void[:, :1])], dim=1)
            vy = torch.cat([torch.ones_like(void[:1, :]), void,
                            torch.ones_like(void[:1, :])], dim=0)
            dead_u = vx[:, :-1] & vx[:, 1:]
            dead_v = vy[:-1, :] & vy[1:, :]
            u = torch.where(dead_u, torch.zeros_like(u), u)
            v = torch.where(dead_v, torch.zeros_like(v), v)
        u = u.clone()
        v = v.clone()
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        v[0, :] = 0.0
        if cfg.top_boundary == "wall":
            v[-1, :] = 0.0
        if cfg.slip == "free_slip":
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
            v[0, :] = v[1, :]
            if cfg.top_boundary == "wall":
                v[-1, :] = v[-2, :]
        return u, v

    # -- VOF transport ----------------------------------------------------- #
    def _advect(self, field: torch.Tensor, dt: float) -> torch.Tensor:
        cfg = self.cfg
        if cfg.vof_scheme == "plic":
            return advect_plic(field, self.u, self.v, dt, self.dx, self.dy,
                               order="xy" if self.step_count % 2 == 0 else "yx",
                               clip=True, conservative_bounds=True)
        return advect_thinc(field, self.u, self.v, dt, self.dx, self.dy,
                            order="xy" if self.step_count % 2 == 0 else "yx",
                            sharpness=cfg.thinc_sharpness)

    # -- time step --------------------------------------------------------- #
    def stable_dt(self) -> float:
        cfg = self.cfg
        h = min(self.dx, self.dy)
        # Advective bound from the bulk speed *and* the gravity-wave speed: at
        # rest ``max_speed() == 0`` must not produce an unbounded dt.
        h_water = float(self.hydrograph().max()) if bool(self.alpha.any()) else 0.0
        speed = max(self.max_speed(), math.sqrt(max(cfg.gravity, 0.0) * max(h_water, 0.0)), 1e-8)
        dt_adv = cfg.cfl * h / speed
        # Local kinematic viscosity, not the worst-case pair of a global mu_max
        # and rho_min: mixing the two (water mu with air rho) shrank dt by ~30x
        # in the ice-floe cross-check and made the run 1403 steps instead of 46.
        nu_max = float((self.viscosity() / self.density()).max())
        dt_visc = 0.25 * h ** 2 / max(nu_max, 1e-30)
        return float(min(dt_adv, dt_visc))

    def step(self, dt: float | None = None) -> dict:
        """Advance one time step; returns a diagnostics record."""
        cfg = self.cfg
        self._build_poisson()
        if dt is None:
            dt = self.stable_dt()
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be positive and finite, got {dt!r}")
        v_before = self.water_volume()
        u_star, v_star = self._predict(dt)
        u, v, phi = self._project_velocity(u_star, v_star, dt)
        u, v = self._apply_bounds(u, v)
        self.u, self.v, self.p = u, v, phi
        self.alpha = self._advect(self.alpha, dt)
        if float(self.ice.max()) > 0.0 and cfg.ice_advection:
            self.ice = self._advect(self.ice, dt)
        self.step_count += 1
        self.time += dt
        div = ((self.u[:, 1:] - self.u[:, :-1]) / self.dx
               + (self.v[1:, :] - self.v[:-1, :]) / self.dy)
        fluid = self._is_source_cell()
        div_max = float(div[fluid].abs().max()) if bool(fluid.any()) else 0.0
        rec = dict(
            step=self.step_count,
            time=self.time,
            dt=dt,
            water_volume=self.water_volume(),
            water_drift=(self.water_volume() - v_before) / max(v_before, 1e-300),
            ice_volume=self.ice_volume(),
            max_speed=self.max_speed(),
            kinetic_energy=self.kinetic_energy(),
            divergence=div_max,
            poisson_iterations=self.last_poisson["iterations"],
            poisson_residual=self.last_poisson["residual"],
            front=self.front_position() if self.alpha.any() else 0.0,
        )
        self.diagnostics.append(rec)
        return rec

    def drag_on_ice(self) -> tuple[float, float]:
        """Pressure force on the rigid ice (``N/m`` in 2-D, unit depth).

        Normal-only: the hydrostatic part dominates for a tank-mounted floe and
        the shear contribution is reported separately by :meth:`drag_on_ice_viscous`
        so the two are not silently mixed.
        """
        p, solid = self.p, self.solid_mask
        fx = fy = 0.0
        # x-faces: fluid on the left, ice on the right -> +x force on ice
        m = (~solid[:, :-1]) & solid[:, 1:]
        fx += float((p[:, :-1] * m).sum()) * self.dy
        m = solid[:, :-1] & (~solid[:, 1:])
        fx -= float((p[:, 1:] * m).sum()) * self.dy
        m = (~solid[:-1, :]) & solid[1:, :]
        fy += float((p[:-1, :] * m).sum()) * self.dx
        m = solid[:-1, :] & (~solid[1:, :])
        fy -= float((p[1:, :] * m).sum()) * self.dx
        return fx, fy

    def drag_on_ice_viscous(self) -> tuple[float, float]:
        """First-order wall-shear estimate on the ice faces (diagnostic only)."""
        solid, mu = self.solid_mask, self.viscosity()
        fx = fy = 0.0
        # x-faces, fluid left of ice: sample first interior fluid face velocity
        m = (~solid[:, :-1]) & solid[:, 1:]
        if bool(m.any()):
            idx = m.nonzero()
            j, i = idx[:, 0], idx[:, 1]
            u_in = self.u[j.clamp(max=self.cfg.nx - 1), (i - 2).clamp(min=0)]
            fx += float((mu[j, i - 1] * u_in / (0.5 * self.dx)).sum()) * self.dy
        m = solid[:, :-1] & (~solid[:, 1:])
        if bool(m.any()):
            idx = m.nonzero()
            j, i = idx[:, 0], idx[:, 1]
            u_in = self.u[j, (i + 1).clamp(max=self.cfg.nx)]
            fx -= float((mu[j, i] * u_in / (0.5 * self.dx)).sum()) * self.dy
        m = (~solid[:-1, :]) & solid[1:, :]
        if bool(m.any()):
            idx = m.nonzero()
            j, i = idx[:, 0], idx[:, 1]
            v_in = self.v[(j - 2).clamp(min=0), i]
            fy += float((mu[j - 1, i] * v_in / (0.5 * self.dy)).sum()) * self.dx
        m = solid[:-1, :] & (~solid[1:, :])
        if bool(m.any()):
            idx = m.nonzero()
            j, i = idx[:, 0], idx[:, 1]
            v_in = self.v[(j + 1).clamp(max=self.cfg.ny), i]
            fy -= float((mu[j, i] * v_in / (0.5 * self.dy)).sum()) * self.dx
        return fx, fy