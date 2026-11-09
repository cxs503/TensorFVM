"""Two-phase VOF solver: variable-property SIMPLE with algebraic VOF transport.

This module extends the staggered MAC SIMPLE framework of
:class:`~tensorfvm.solver.SimpleSolver` to handle two-phase (liquid/gas) flows
with volume-of-fluid interface capturing.

Design principles
-----------------
1. **Config-gated single-phase regression**: when ``two_phase=False`` the solver
   delegates to the original :class:`SimpleSolver` path, guaranteeing
   bit-for-bit identical results to the single-phase baseline.

2. **Variable-property momentum**: density and viscosity are computed from the
   volume fraction field ``alpha`` at every SIMPLE step (linear mean for ρ,
   harmonic mean for μ).

3. **VOF transport**: the volume fraction is advected with a donor-cell
   conservative scheme (see :class:`~tensorfvm.two_phase_vof.VOFTransport`)
   after each velocity correction.  Transport is bounded for CFL ≤ 1.

4. **Gravity & hydrostatics**: a body-force source ``ρ·g`` is added to the
   vertical momentum equation, enabling the hydrostatic-pressure and
   static-interface anchors.

5. **Surface tension**: the Brackbill CSF term (see
   :func:`~tensorfvm.two_phase_vof.csf_surface_tension`) is an optional source,
   gated by ``surface_tension`` in :class:`~tensorfvm.two_phase_vof.TwoPhaseProperties`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .solver import _neighbors
from .two_phase_vof import (
    VOFTransport,
    TwoPhaseProperties,
    balanced_csf_face_forces,
    mixture_density,
    mixture_viscosity,
)
from .free_surface_bc import FreeSurfaceBC
from .vof_plic import advect_plic


@dataclass
class GridConfig:
    """Minimal grid configuration for the two-phase solver.

    This avoids the single-phase :class:`SolverConfig` constraint that
    ``inlet_velocity > 0``, which is inappropriate for closed-domain
    two-phase problems (static interface, hydrostatics, bubble rise).
    """

    nx: int = 40
    ny: int = 40
    length: float = 1.0
    height: float = 1.0
    device: str = "cpu"
    velocity_relaxation: float = 0.7
    pressure_relaxation: float = 0.3
    max_iterations: int = 1000
    tolerance: float = 1e-6
    time_step: float | None = None
    inlet_velocity: float = 0.0  # 0 for closed domains
    cylinder_radius: float | None = None
    cylinder_x: float = 0.5
    cylinder_y: float = 0.5
    density: float = 1.0  # reference density (not used when two_phase enabled)

    def __post_init__(self):
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must be >= 2")
        if self.length <= 0 or self.height <= 0:
            raise ValueError("length and height must be positive")


@dataclass
class TwoPhaseConfig:
    """Configuration for the two-phase VOF solver.

    When ``enabled=False``, the solver is single-phase and all two-phase
    fields are dormant — the base :class:`SimpleSolver` path is used.
    """

    enabled: bool = False
    # Phase properties
    rho_liquid: float = 1000.0
    rho_gas: float = 1.0
    mu_liquid: float = 1.0e-3
    mu_gas: float = 1.0e-5
    sigma: float = 0.0  # surface tension coefficient
    surface_tension: bool = False
    # Gravity (m/s²), pointing in −y
    gravity: float = 0.0
    # Free-surface boundary conditions
    top_atmosphere: bool = True
    bottom_wall: bool = True
    side_outlet: bool = False
    # VOF transport
    vof_cfl_limit: float = 0.5
    #: Interface transport scheme.  ``"upwind"`` is the historical donor-cell
    #: transport of :class:`~tensorfvm.two_phase_vof.VOFTransport`;
    #: ``"plic"`` uses the geometric PLIC reconstruction of
    #: :mod:`tensorfvm.vof_plic` (sharp interface, conservative, bounded).
    vof_scheme: str = "upwind"
    #: Momentum advection (``u·∇u``) in the transient projection predictor.
    #: The historical predictor carried only diffusion + pressure + gravity +
    #: CSF, i.e. it integrated a Stokes-like momentum balance.  That is
    #: adequate for hydrostatics, capillary and dam-break anchors, but it
    #: cannot represent the inertia that sets the terminal velocity of a rising
    #: bubble (it would return the creeping-flow value instead).  ``False``
    #: (default) keeps every existing caller bit-for-bit unchanged.
    momentum_advection: bool = False
    #: How the body force and the pressure are split.
    #:
    #: * ``"total"`` (historical default): the momentum equation carries the
    #:   *total* pressure ``p`` and the full gravitational acceleration ``-g``.
    #: * ``"reduced"``: the momentum equation carries the *dynamic* pressure
    #:   ``p_dyn = p + ρ_ref g y`` and the buoyancy acceleration
    #:   ``((ρ - ρ_ref)/ρ)·g`` with ``ρ_ref = ρ_liquid``.
    #:
    #: The two are algebraically equivalent for the continuous equations, but
    #: not for the discrete incremental-projection scheme used here: the
    #: corrector solves ``∇·(1/ρ ∇φ) = ∇·u*/Δt`` and the update ``p ← p + φ``
    #: annihilates every pressure field whose ``(1/ρ)∇p`` is discretely
    #: divergence-free.  A hydrostatic head over a *uniform* liquid column is
    #: exactly such a field, so with ``"total"`` a pool at rest is not
    #: supported: it free-falls (measured: ``|v|`` grows linearly from rest,
    #: see ``tests/test_bubble_rise.py::test_hydrostatic_pool_stays_at_rest``).
    #: ``"reduced"`` moves the head out of the pressure and leaves a body force
    #: that vanishes in a uniform liquid, so a static pool is an exact solution
    #: and the pressure stays O(dynamic).  Any caller with ``gravity != 0`` and
    #: a free surface should use ``"reduced"``.
    pressure_form: str = "total"
    #: Reuse the previous step's pressure correction as the initial guess of
    #: the CG solve.  Two-phase Poisson systems (1/rho contrast of 10-1000)
    #: need several hundred Jacobi-preconditioned CG iterations from a zero
    #: guess; consecutive time steps differ little, so the warm start removes
    #: most of that cost (measured on a 120x240 bubble grid: 780 -> 45
    #: iterations).  The iterate is *the same* linear system either way — the
    #: flag only changes how many iterations are spent getting there, and the
    #: solve still runs to its own residual target.  ``False`` (default) keeps
    #: every existing caller bit-for-bit unchanged.
    warm_start_pressure: bool = False
    # Surface-tension discretisation.  ``csf_balance="balanced"`` samples the
    # CSF force on the MAC faces with the same two-point alpha difference the
    # pressure gradient uses (Francois et al. 2006); ``"legacy"`` averages the
    # cell-centred force, which leaves a residual curl that drives spurious
    # currents growing with resolution (see docs/two-phase-csf-balance/).
    csf_balance: str = "balanced"
    #: Brackbill--Kothe--Zemach smoothing passes applied to alpha before the
    #: normals/curvature are differentiated (0 = legacy, unsmoothed).
    csf_smoothing: int = 3
    #: Interface-curvature estimator used by the balanced CSF force:
    #: ``"divergence"`` (default, ``kappa = div n`` on the smoothed colour
    #: field -- existing behaviour, bit-for-bit) or ``"height_function"``
    #: (second-order height-function curvature, see
    #: :mod:`tensorfvm.two_phase_curvature` and
    #: ``docs/height-function-curvature/report.md``).
    csf_curvature: str = "divergence"
    # Domain boundary type for two-phase (not using inlet/outlet channel)
    # "closed" = all walls except top atmosphere; "channel" = inlet+outlet
    domain_type: str = "closed"

    def __post_init__(self):
        if self.vof_scheme not in ("upwind", "plic"):
            raise ValueError(
                f"vof_scheme must be 'upwind' or 'plic', got {self.vof_scheme!r}")
        if self.rho_liquid <= 0 or self.rho_gas <= 0:
            raise ValueError("phase densities must be positive")
        if self.mu_liquid < 0 or self.mu_gas < 0:
            raise ValueError("phase viscosities must be non-negative")
        if self.pressure_form not in ("total", "reduced"):
            raise ValueError(
                f"pressure_form must be 'total' or 'reduced', got {self.pressure_form!r}")


class TwoPhaseSolver:
    """Variable-property SIMPLE + VOF on a staggered MAC grid.

    The solver mirrors :class:`SimpleSolver` for the velocity-pressure
    coupling but replaces the constant ρ, μ with volume-fraction-weighted
    fields and adds gravity / surface-tension sources.

    Parameters
    ----------
    config : GridConfig
        Grid configuration (resolution, domain size, relaxation, device).
    tp_config : TwoPhaseConfig
        Two-phase parameters.  If ``tp_config.enabled`` is False, the solver
        operates in *single-phase* mode with uniform properties.  In
        single-phase mode the VOF, gravity, and surface-tension code paths
        are fully bypassed — guaranteeing no change to the baseline physics.
    """

    def __init__(self, config: GridConfig, tp_config: TwoPhaseConfig | None = None):
        self.config = config
        self.tp = tp_config or TwoPhaseConfig()
        c = config
        self.dx, self.dy = c.length / c.nx, c.height / c.ny
        opts = {"dtype": torch.float64, "device": c.device}

        ny, nx = c.ny, c.nx
        self.ny, self.nx = ny, nx

        # Velocity and pressure fields (same layout as SimpleSolver)
        self.u = torch.zeros((ny, nx + 1), **opts)
        self.v = torch.zeros((ny + 1, nx), **opts)
        self.p = torch.zeros((ny, nx), **opts)

        # Fluid mask (all fluid; optional cylinder obstacle)
        self.fluid = torch.ones((ny, nx), dtype=torch.bool, device=c.device)
        if c.cylinder_radius:
            x = (torch.arange(nx, **opts) + 0.5) * self.dx
            y = (torch.arange(ny, **opts) + 0.5) * self.dy
            self.fluid = ((x[None, :] - c.cylinder_x) ** 2
                          + (y[:, None] - c.cylinder_y) ** 2
                          > c.cylinder_radius ** 2)

        # VOF / two-phase fields (dormant in single-phase mode)
        self.alpha = None
        self.rho_field = None
        self.mu_field = None
        self.vof = None
        self.fs_bc = None
        self.props = None

        if self.tp.enabled:
            self.alpha = torch.zeros((ny, nx), **opts)
            self.props = TwoPhaseProperties(
                rho_liquid=self.tp.rho_liquid,
                rho_gas=self.tp.rho_gas,
                mu_liquid=self.tp.mu_liquid,
                mu_gas=self.tp.mu_gas,
                sigma=self.tp.sigma,
                surface_tension=self.tp.surface_tension,
            )
            self._update_properties()
            self.vof = VOFTransport(ny, nx, self.dx, self.dy, device=c.device)
            self.fs_bc = FreeSurfaceBC(
                ny, nx, self.dx, self.dy, device=c.device,
                top_atmosphere=self.tp.top_atmosphere,
                bottom_wall=self.tp.bottom_wall,
                side_outlet=self.tp.side_outlet,
            )
        else:
            # Single-phase: uniform properties (no VOF, no gravity, no CSF)
            self.rho_field = torch.full((ny, nx), float(c.density), **opts)
            self.mu_field = torch.full((ny, nx), 1.0, **opts)

        # Active masks for u and v faces
        self._u_active = torch.ones((ny, nx + 1), dtype=torch.bool, device=c.device)
        self._u_active[:, 0] = False  # inlet prescribed (or zero)
        self._v_active = torch.ones((ny + 1, nx), dtype=torch.bool, device=c.device)
        self._v_active[0, :] = False  # bottom wall
        if self.tp.enabled:
            if not self.tp.top_atmosphere:
                self._v_active[-1, :] = False
        else:
            self._v_active[-1, :] = False  # single-phase: top wall

        self.u[:, 0] = float(c.inlet_velocity)
        self.history = []
        self.time = 0.0
        self._plic_sweeps = 0
        self.plic_diagnostics: dict = {}
        self.cg_diagnostics: dict = {}
        self._last_dp = None
        self._current_dt = c.time_step  # may be None

    def _face_mask_u(self):
        """Return boolean mask of interior u-faces (always true for full domain)."""
        return torch.ones((self.ny, self.nx + 1), dtype=torch.bool,
                          device=self.config.device)

    def _update_properties(self):
        """Recompute density and viscosity fields from alpha."""
        self.rho_field = mixture_density(self.alpha, self.props.rho_liquid, self.props.rho_gas)
        self.mu_field = mixture_viscosity(self.alpha, self.props.mu_liquid, self.props.mu_gas)

    def set_alpha(self, alpha: torch.Tensor):
        """Set the volume fraction field and update properties."""
        self.alpha = alpha.to(dtype=self.alpha.dtype, device=self.alpha.device)
        self._update_properties()

    def initialize_hydrostatic_pressure(self, interface_y: float = 0.5):
        """Initialise pressure to the analytical hydrostatic profile.

        For a stratified two-phase column with the interface at height
        ``interface_y``:

        * Liquid (y < interface_y):  p = ρ_l · g · (interface_y − y)
        * Gas    (y ≥ interface_y):  p = ρ_g · g · (height   − y)

        The top boundary is set to zero (atmospheric gauge).
        """
        c = self.config
        g = self.tp.gravity
        rho_l, rho_g = self.tp.rho_liquid, self.tp.rho_gas
        y = (torch.arange(self.ny, dtype=self.p.dtype, device=c.device)
             + 0.5).reshape(-1, 1) * self.dy
        y_grid = y.expand(self.ny, self.nx)
        # Pressure is *continuous* across the interface: the liquid branch must
        # carry the weight of the gas column above it, p_i = rho_g g (H - h_i).
        # Dropping that offset leaves a jump of rho_g g (H - h_i) at the
        # interface, i.e. a spurious discrete pressure gradient across the
        # interface cell that drives the resting column.
        p_iface = rho_g * g * (c.height - interface_y)
        self.p = torch.where(
            y_grid < interface_y,
            p_iface + rho_l * g * (interface_y - y_grid),
            rho_g * g * (c.height - y_grid),
        ).to(dtype=self.p.dtype, device=c.device)
        # No cell-centre pinning here: the atmospheric datum p = 0 is imposed on
        # the top *face* by the projection pressure operator, so the top cell
        # centre carries the discrete-consistent value rho_g g dy/2.

    def hydrostatic_balance_residual(self) -> dict:
        """Compute the force balance residual at each v-face.

        For a true hydrostatic equilibrium the v-momentum equation reduces
        to:  ∂p/∂y = −ρ·g   (pressure gradient balances gravity).

        Returns the max and RMS of |dp/dy + ρ·g| · dx (per-face residual).
        """
        g = self.tp.gravity
        rho_v = self._rho_face_v()
        # Discrete pressure gradient at v-faces: dp/dy ≈ (p_N - p_S) / dy
        dpdy = torch.zeros_like(self.v)
        dpdy[1:-1, :] = (self.p[1:, :] - self.p[:-1, :]) / self.dy
        # Residual: dp/dy + rho*g should be 0 for hydrostatic balance
        residual = dpdy + rho_v * g
        # Only evaluate at active interior v-faces (dpdy is not defined on the
        # two boundary rows, where it is left at zero and would report a
        # spurious rho g residual)
        interior = torch.zeros_like(self.v, dtype=torch.bool)
        interior[1:-1, :] = True
        active = self._v_active & interior
        res_active = residual[active]
        return {
            "max_residual": float(res_active.abs().max()) if active.any() else 0.0,
            "rms_residual": float(
                torch.sqrt(torch.mean(res_active ** 2))) if active.any() else 0.0,
        }

    def _rho_face_u(self) -> torch.Tensor:
        """Density at u-faces (east faces, ny×(nx+1)): average of left/right cell."""
        rho = self.rho_field
        # face j between cell j-1 and j; boundary = one-sided
        rho_face = torch.zeros_like(self.u)
        rho_face[:, 1:-1] = 0.5 * (rho[:, :-1] + rho[:, 1:])
        rho_face[:, 0] = rho[:, 0]
        rho_face[:, -1] = rho[:, -1]
        return rho_face

    def _rho_face_v(self) -> torch.Tensor:
        """Density at v-faces (north faces, (ny+1)×nx): average of bottom/top cell.

        At the liquid-gas interface the arithmetic average creates a
        pressure-gravity imbalance because the discrete pressure jump spans
        two cells with different densities.  Using the *upwind* (below)
        density at the interface face eliminates this artefact for a
        stratified column at rest.
        """
        rho = self.rho_field
        rho_face = torch.zeros_like(self.v)
        rho_face[1:-1, :] = 0.5 * (rho[:-1, :] + rho[1:, :])
        rho_face[0, :] = rho[0, :]
        rho_face[-1, :] = rho[-1, :]
        # At sharp interfaces (alpha jumps), use downwind-corrected density
        # that matches the pressure gradient: if the cell below is liquid
        # (rho_l) and above is gas (rho_g), the v-face density should match
        # the pressure gradient which transitions across the interface.
        # For hydrostatic balance: dp/dy = -rho_face * g, and the discrete
        # p jump is p[j] - p[j-1] where p follows the piecewise profile.
        # Using (rho_below + rho_above)/2 is the standard VOF approach;
        # the residual is a discretisation error O(dx) at the interface.
        return rho_face

    def _mu_face_u(self) -> torch.Tensor:
        """Viscosity at u-faces."""
        mu = self.mu_field
        mu_face = torch.zeros_like(self.u)
        mu_face[:, 1:-1] = 0.5 * (mu[:, :-1] + mu[:, 1:])
        mu_face[:, 0] = mu[:, 0]
        mu_face[:, -1] = mu[:, -1]
        return mu_face

    def _mu_face_v(self) -> torch.Tensor:
        """Viscosity at v-faces."""
        mu = self.mu_field
        mu_face = torch.zeros_like(self.v)
        mu_face[1:-1, :] = 0.5 * (mu[:-1, :] + mu[1:, :])
        mu_face[0, :] = mu[0, :]
        mu_face[-1, :] = mu[-1, :]
        return mu_face

    def _momentum_u(self) -> tuple:
        """Build u-momentum: diagonal, 4 neighbour coeffs, pressure, source, active."""
        dx, dy = self.dx, self.dy
        field = self.u
        rho_face = self._rho_face_u()
        mu_face = self._mu_face_u()
        e, w, _, _ = _neighbors(field)
        # Convective fluxes (upwind implicit via first-order)
        fe = rho_face * dy * (field + e) / 2
        fw = rho_face * dy * (field + w) / 2
        fe[:, -1] = rho_face[:, -1] * dy * field[:, -1]
        # Transverse velocities at u-faces
        vn = torch.zeros_like(field)
        vs = torch.zeros_like(field)
        vn[:, 1:-1] = 0.5 * (self.v[1:, :-1] + self.v[1:, 1:])
        vs[:, 1:-1] = 0.5 * (self.v[:-1, :-1] + self.v[:-1, 1:])
        width = torch.full_like(field, dx)
        fn = rho_face * width * vn
        fs = rho_face * width * vs
        # Diffusion
        de = mu_face * dy / dx
        dw = de.clone()
        dn = mu_face * width / dy
        ds = dn.clone()
        # Pressure gradient: -dp/dx
        pressure = torch.zeros_like(field)
        pressure[:, 1:-1] = dy * (self.p[:, :-1] - self.p[:, 1:])
        pressure[:, -1] = dy * self.p[:, -1]
        # Gravity source (u-component = 0 for vertical gravity)
        source = torch.zeros_like(field)
        active = self._u_active
        # Wall correction for tangential diffusion
        neighbor_open = _neighbors(torch.ones_like(field, dtype=torch.bool))
        fluxes, diffusion = [fe, fw, fn, fs], [de, dw, dn, ds]
        for k in range(4):
            if k >= 2:  # tangential (n, s)
                wall = ~neighbor_open[k]
                diffusion[k] = diffusion[k] * torch.where(wall, 2.0, 1.0)
                fluxes[k] = fluxes[k].masked_fill(wall, 0)
        fe, fw, fn, fs = fluxes
        diagonal = sum(diffusion) + fe.clamp_min(0) + (-fw).clamp_min(0) \
            + fn.clamp_min(0) + (-fs).clamp_min(0)
        coeffs = [diffusion[0] + (-fe).clamp_min(0),
                  diffusion[1] + fw.clamp_min(0),
                  diffusion[2] + (-fn).clamp_min(0),
                  diffusion[3] + fs.clamp_min(0)]
        coeffs = [a.masked_fill(~active, 0) for a in coeffs]
        diagonal = torch.where(active, diagonal, torch.ones_like(diagonal))
        return diagonal, coeffs, pressure, source, active

    def _momentum_v(self) -> tuple:
        """Build v-momentum with gravity body force."""
        dx, dy = self.dx, self.dy
        field = self.v
        rho_face = self._rho_face_v()
        mu_face = self._mu_face_v()
        _, _, n, s = _neighbors(field)
        fn = rho_face * dx * (field + n) / 2
        fs = rho_face * dx * (field + s) / 2
        # Cross velocities
        ue = torch.zeros_like(field)
        uw = torch.zeros_like(field)
        ue[1:-1] = 0.5 * (self.u[:-1, 1:] + self.u[1:, 1:])
        uw[1:-1] = 0.5 * (self.u[:-1, :-1] + self.u[1:, :-1])
        fe = rho_face * dy * ue
        fw = rho_face * dy * uw
        # Diffusion
        de = mu_face * dy / dx
        dw = de.clone()
        dn = mu_face * dx / dy
        ds = dn.clone()
        # Pressure gradient: -dp/dy
        pressure = torch.zeros_like(field)
        pressure[1:-1] = dx * (self.p[:-1] - self.p[1:])
        # Gravity body force: source = -rho*g*vol (gravity in -y, v positive up)
        source = torch.zeros_like(field)
        if self.tp.gravity != 0.0:
            source = -rho_face * self.tp.gravity * dx * dy
        active = self._v_active
        neighbor_open = _neighbors(torch.ones_like(field, dtype=torch.bool))
        fluxes, diffusion = [fe, fw, fn, fs], [de, dw, dn, ds]
        for k in range(2):  # tangential: e, w
            wall = ~neighbor_open[k]
            diffusion[k] = diffusion[k] * torch.where(wall, 2.0, 1.0)
            fluxes[k] = fluxes[k].masked_fill(wall, 0)
        fe, fw, fn, fs = fluxes
        diagonal = sum(diffusion) + fe.clamp_min(0) + (-fw).clamp_min(0) \
            + fn.clamp_min(0) + (-fs).clamp_min(0)
        coeffs = [diffusion[0] + (-fe).clamp_min(0),
                  diffusion[1] + fw.clamp_min(0),
                  diffusion[2] + (-fn).clamp_min(0),
                  diffusion[3] + fs.clamp_min(0)]
        coeffs = [a.masked_fill(~active, 0) for a in coeffs]
        diagonal = torch.where(active, diagonal, torch.ones_like(diagonal))
        return diagonal, coeffs, pressure, source, active

    def _predict(self, component: str) -> torch.Tensor:
        """Jacobi-iterated momentum prediction. Returns d (velocity/relaxed-diagonal)."""
        if component == "u":
            diagonal, coeffs, pressure, source, active = self._momentum_u()
        else:
            diagonal, coeffs, pressure, source, active = self._momentum_v()
        field = self.u if component == "u" else self.v
        old = field.clone()
        alpha_relax = self.config.velocity_relaxation
        # Transient inertia: always active in two-phase mode (pseudo-transient)
        # Uses dt from config or the step() argument.
        dt_eff = self._current_dt
        if dt_eff is not None and dt_eff > 0:
            rho_face = self._rho_face_u() if component == "u" else self._rho_face_v()
            width = self.dy if component == "u" else self.dx
            inertia = rho_face * width * self.dx / dt_eff
            diagonal = diagonal + inertia
            pressure = pressure + inertia * old
        relaxed = diagonal / alpha_relax
        rhs = pressure + source + (1 - alpha_relax) / alpha_relax * diagonal * old
        inner_tol = 1e-8
        for _ in range(200):
            update = (rhs + self._neighbor_sum(field, coeffs)) / relaxed
            error = torch.max(torch.abs(update[active] - field[active]))
            field[active] = 0.9 * update[active] + 0.1 * field[active]
            if error.item() < inner_tol:
                break
        distance_area = self.dy if component == "u" else self.dx
        return torch.where(active, distance_area / relaxed, 0)

    @staticmethod
    def _neighbor_sum(field, coeffs):
        return sum(a * value for a, value in zip(coeffs, _neighbors(field)))

    def _csf_face_forces(self):
        """Brackbill CSF force (N/m³) interpolated to the interior MAC faces.

        Returns ``(f_u, f_v)`` with shapes ``(ny, nx-1)`` and ``(ny-1, nx)``,
        or ``(None, None)`` when surface tension is switched off so that the
        single-phase / sigma=0 behaviour is bit-for-bit unchanged.
        """
        if (self.props is None or not self.props.surface_tension
                or self.props.sigma == 0.0):
            return None, None
        if getattr(self.tp, 'csf_balance', 'balanced') == 'balanced':
            return balanced_csf_face_forces(
                self.alpha, self.props.sigma, self.dx, self.dy,
                smoothing=getattr(self.tp, 'csf_smoothing', 3),
                curvature=getattr(self.tp, 'csf_curvature', 'divergence'))
        f = self.props.csf(self.alpha, self.dx, self.dy)      # (ny, nx, 2)
        f_u = 0.5 * (f[:, :-1, 0] + f[:, 1:, 0])
        f_v = 0.5 * (f[:-1, :, 1] + f[1:, :, 1])
        return f_u, f_v

    def _divergence(self) -> torch.Tensor:
        return ((self.u[:, 1:] - self.u[:, :-1]) / self.dx
                + (self.v[1:] - self.v[:-1]) / self.dy)

    def _pressure_correction(self, du, dv) -> torch.Tensor:
        """Pressure correction equation (Poisson).

        Uses a *reference density* (liquid density) for the pressure-velocity
        coupling coefficients.  This is standard practice for variable-density
        SIMPLE: the pressure correction is a Laplacian-like equation whose
        role is to drive the velocity field toward divergence-free, not to
        transport density.  Using a constant reference density avoids
        ill-conditioning at the density interface.
        """
        if self.tp.enabled and self.props is not None:
            rho_ref = self.props.rho_liquid
        else:
            rho_ref = float(self.config.density)

        cu = rho_ref * self.dy * du
        cv = rho_ref * self.dx * dv
        ce, cw, cn, cs = (torch.zeros_like(self.p) for _ in range(4))
        ce[:, :-1], cw[:, 1:] = cu[:, 1:-1], cu[:, 1:-1]
        cn[:-1], cs[1:] = cv[1:-1], cv[1:-1]

        # Boundary conditions for pressure correction:
        # - Top atmosphere: p'[-1, :] = 0 → remove north coefficient at top row
        # - Side outlet (if any): p'[:, -1] = 0 → remove east coefficient at outlet
        # - Closed domain: include all interior face fluxes
        if self.tp.top_atmosphere:
            cn[-1, :] = 0  # no north neighbour at top (p' fixed = 0)
        if self.tp.side_outlet:
            ce[:, -1] = 0  # no east neighbour at outlet
            # include outlet flux via cu[:, -1] in the correction
        else:
            # Closed domain: east-most face connects to last cell only
            ce[:, -1] = cu[:, -1]  # half-volume at boundary

        diagonal = ce + cw + cn + cs
        # For isolated closed cells (no open faces), set diagonal = 1
        diagonal = torch.where(
            diagonal > 0,
            diagonal,
            torch.ones_like(diagonal))
        diagonal = torch.where(self.fluid, diagonal, torch.ones_like(diagonal))

        coeffs = [ce.clone(), cw, cn, cs]

        def matvec(q):
            return diagonal * q - self._neighbor_sum(q, coeffs)

        # Mass imbalance: b = -rho_ref * div(u*area)
        b = -(rho_ref * self.dx * self.dy * self._divergence()).masked_fill(~self.fluid, 0)

        q = torch.zeros_like(b)
        r = b.clone()
        norm = torch.linalg.vector_norm(b).item()
        if norm == 0:
            return q
        z = r / diagonal
        direction, rz = z.clone(), torch.sum(r * z)
        target = max(norm * 1e-10, 1e-14)
        for _ in range(max(200, self.config.nx * self.config.ny)):
            ad = matvec(direction)
            denominator = torch.sum(direction * ad)
            if denominator.item() <= 0 or not torch.isfinite(denominator):
                break
            step = rz / denominator
            q += step * direction
            r -= step * ad
            self.cg_diagnostics["residual"] = torch.linalg.vector_norm(r).item()
            if self.cg_diagnostics["residual"] <= target:
                self.cg_diagnostics["converged"] = True
                return q
            z = r / diagonal
            new_rz = torch.sum(r * z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        return q

    def _boundaries(self, projection: bool = False):
        """Apply boundary conditions.

        ``projection=True`` is used by the transient Chorin path: the
        atmospheric pressure datum is then imposed on the top *face* inside the
        Poisson operator, so the top cell-centre level must not be overwritten.
        """
        # Inlet
        self.u[:, 0] = self.config.inlet_velocity if self.config.inlet_velocity != 0 else 0.0
        # Bottom wall
        if self.tp.bottom_wall:
            self.v[0, :] = 0.0
        # Top atmosphere
        if self.tp.top_atmosphere:
            if not projection:
                self.v[-1, :] = self.v[-2, :]  # zero gradient (SIMPLE path)
                self.p[-1, :] = 0.0  # atmospheric (SIMPLE path gauge)
            # In the projection path the top face velocity is *corrected* by the
            # pressure datum p=0 imposed on that face; overwriting it with a
            # zero-gradient copy would leave the top cell's divergence
            # uncorrectable and poison the whole elliptic solve.
        # Side outlet
        if self.tp.side_outlet:
            self.u[:, -1] = self.u[:, -2]
            self.p[:, -1] = 0.0
            self.p[:, 0] = 0.0
        # Mask inactive
        self.u.masked_fill_(~self._u_active, 0)
        self.v.masked_fill_(~self._v_active, 0)

    def _advective_accel(self):
        """Advective acceleration ``u·∇u`` on the interior MAC faces (upwind).

        Written in *advective* (non-conservative) form because that is the form
        of the single-fluid variable-density momentum balance the predictor
        already integrates: ``∂u/∂t = (∇·τ - ∇p)/ρ + g - u·∇u``.  First-order
        upwind is used deliberately: the predictor is explicit, so upwind keeps
        the convection bounded for CFL ≤ 1 and cannot create a new extremum
        next to a large density jump.

        Index conventions (staggered MAC, ``dx``/``dy`` may differ):

        * ``u[j, i]`` sits at ``(x = i·dx, y = (j + 1/2)·dy)``;
        * ``v[j, i]`` sits at ``(x = (i + 1/2)·dx, y = j·dy)``.

        Returns
        -------
        (adv_u, adv_v)
            ``adv_u`` has the shape of the updated ``u`` block
            ``(ny-2, nx-1)``, ``adv_v`` that of the updated ``v`` block
            ``(ny-1, nx-2)`` — exactly the blocks the predictor adds to.
        """
        ny, nx = self.ny, self.nx
        u, v = self.u, self.v
        dx, dy = self.dx, self.dy

        # --- u-momentum: interior faces rows 1..ny-2, cols 1..nx-1 ---
        ri, ci = slice(1, ny - 1), slice(1, nx)
        u_c = u[ri, ci]
        u_w = u[ri, slice(0, nx - 1)]
        u_e = u[ri, slice(2, nx + 1)]
        u_s = u[slice(0, ny - 2), ci]
        u_n = u[slice(2, ny), ci]
        # v interpolated to the u-face: it lies between v rows j, j+1 and
        # between v cols i-1, i.
        v_at_u = 0.25 * (
            v[slice(1, ny - 1), slice(0, nx - 1)] + v[slice(1, ny - 1), slice(1, nx)]
            + v[slice(2, ny), slice(0, nx - 1)] + v[slice(2, ny), slice(1, nx)]
        )
        dudx = torch.where(u_c >= 0.0, (u_c - u_w) / dx, (u_e - u_c) / dx)
        dudy = torch.where(v_at_u >= 0.0, (u_c - u_s) / dy, (u_n - u_c) / dy)
        adv_u = u_c * dudx + v_at_u * dudy

        # --- v-momentum: interior faces rows 1..ny-1, cols 1..nx-2 ---
        rj, cj = slice(1, ny), slice(1, nx - 1)
        v_c = v[rj, cj]
        v_s = v[slice(0, ny - 1), cj]
        v_n = v[slice(2, ny + 1), cj]
        v_w = v[rj, slice(0, nx - 2)]
        v_e = v[rj, slice(2, nx)]
        # u interpolated to the v-face: between u rows j-1, j and between
        # u cols i, i+1.
        u_at_v = 0.25 * (
            u[slice(0, ny - 1), slice(1, nx - 1)] + u[slice(0, ny - 1), slice(2, nx)]
            + u[slice(1, ny), slice(1, nx - 1)] + u[slice(1, ny), slice(2, nx)]
        )
        dvdy = torch.where(v_c >= 0.0, (v_c - v_s) / dy, (v_n - v_c) / dy)
        dvdx = torch.where(u_at_v >= 0.0, (v_c - v_w) / dx, (v_e - v_c) / dx)
        adv_v = u_at_v * dvdx + v_c * dvdy
        return adv_u, adv_v

    def stable_time_step(
        self,
        cfl: float = 0.4,
        capillary_safety: float = 0.5,
        viscous_safety: float = 0.25,
    ) -> float:
        """Largest explicit step this discretisation tolerates right now.

        The transient predictor is explicit in convection, diffusion and
        surface tension, so the step is the minimum of three limits:

        * CFL on the MAC faces: ``dt ≤ cfl·h/|u|_max``;
        * explicit viscous diffusion: ``dt ≤ 0.25·h²/ν_max`` with
          ``ν = μ/ρ`` taken *per cell* (the gas phase is the restrictive one in
          an air/water pair, its kinematic viscosity is ~30× the liquid's);
        * capillary wave: ``dt ≤ c·sqrt(ρ_l h³/(2πσ))`` (Brackbill et al. 1992).

        ``h = min(dx, dy)``.  Returns a positive finite float; callers that step
        manually must respect it or the interface transport raises (PLIC
        rejects directional CFL > 1).
        """
        h = min(self.dx, self.dy)
        umax = max(float(self.u.abs().max()), float(self.v.abs().max()))
        dt_cfl = cfl * h / umax if umax > 0.0 else math.inf
        dt_visc = math.inf
        if self.props is not None and self.mu_field is not None:
            nu_max = float((self.mu_field / self.rho_field.clamp_min(1e-12)).max())
            if nu_max > 0.0:
                dt_visc = viscous_safety * h * h / nu_max
        dt_cap = math.inf
        if (self.props is not None and self.props.surface_tension
                and self.props.sigma > 0.0):
            dt_cap = capillary_safety * math.sqrt(
                self.props.rho_liquid * h ** 3 / (2.0 * math.pi * self.props.sigma))
        dt = min(dt_cfl, dt_visc, dt_cap)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("no finite stable step (check velocities and properties)")
        return float(dt)

    def _advect_vof(self, dt: float):
        """Advect the volume fraction with the configured interface scheme."""
        if getattr(self.tp, "vof_scheme", "upwind") == "plic":
            u, v = self.u, self.v
            if not self.tp.side_outlet:
                # The geometric sweeps index-wrap (periodic stencil).  A closed
                # domain is reproduced exactly by zeroing the boundary-normal
                # face velocities: no flux crosses a wall either way, and the
                # wrapped donor cell is then never asked for a flux.
                u = u.clone()
                u[:, 0] = 0.0
                u[:, -1] = 0.0
                v = v.clone()
                v[0, :] = 0.0
                v[-1, :] = 0.0
            # Alternate the sweep order so the Lie splitting is not biased
            # towards +x every step.
            order = "xy" if self._plic_sweeps % 2 == 0 else "yx"
            self._plic_sweeps += 1
            self.plic_diagnostics = {}
            self.alpha = advect_plic(
                self.alpha, u, v, dt, self.dx, self.dy, order=order,
                conservative_bounds=True,
                diagnostics=self.plic_diagnostics)
        else:
            self.alpha = self.vof.advect(self.alpha, self.u, self.v, dt)
        self._update_properties()

    def _projection_pressure(self, u_star, v_star, dt):
        """Chorin projection pressure solve for variable-density flow.

        Solves:  ∇·(1/ρ ∇φ) = (1/dt) ∇·u*

        Returns the pressure *correction* φ (not the pressure): the caller
        updates p ← p + φ, so the scheme is the standard incremental-pressure
        projection.  The 1/ρ weighting is the key: it naturally handles density
        jumps.  A Jacobi-preconditioned CG iteration is used.
        """
        ny, nx = self.ny, self.nx
        # Face inverse density (harmonic mean for sharp interfaces)
        inv_rho_u = 2.0 / (self.rho_field[:, :-1] + self.rho_field[:, 1:]).clamp_min(1e-10)
        inv_rho_v = 2.0 / (self.rho_field[:-1, :] + self.rho_field[1:, :]).clamp_min(1e-10)

        # Laplacian coefficients
        ce = inv_rho_u * self.dy / self.dx  # (ny, nx-1)
        cn = inv_rho_v * self.dx / self.dy  # (ny-1, nx)

        # Build diagonal and off-diagonals properly
        # For cell (j,i):
        #   diag = (ce_east if has east) + (ce_west if has west)
        #        + (cn_north if has north) + (cn_south if has south)
        #   east_off = -ce[j,i]   (coeff of q[j,i+1])
        #   west_off = -ce[j,i-1] (coeff of q[j,i-1])
        #   north_off = -cn[j,i]  (coeff of q[j+1,i])
        #   south_off = -cn[j-1,i](coeff of q[j-1,i])

        # Use padding to build full-size arrays
        ce_e = torch.zeros((ny, nx), device=self.config.device, dtype=self.p.dtype)
        ce_e[:, :-1] = ce        # east face of cell (j,i) for i < nx-1
        ce_w = torch.zeros((ny, nx), device=self.config.device, dtype=self.p.dtype)
        ce_w[:, 1:] = ce         # west face of cell (j,i) for i > 0
        cn_n = torch.zeros((ny, nx), device=self.config.device, dtype=self.p.dtype)
        cn_n[:-1, :] = cn        # north face of cell (j,i) for j < ny-1
        cn_s = torch.zeros((ny, nx), device=self.config.device, dtype=self.p.dtype)
        cn_s[1:, :] = cn         # south face of cell (j,i) for j > 0

        diag = ce_e + ce_w + cn_n + cn_s
        # matvec below is  diag*q - sum(off*neighbour); for the standard
        # five-point -Laplacian (diag*q - ce_e*q_E - ...) the off-diagonals
        # must be *positive* here.  Negating them (the previous code) turned
        # the operator into sum(ce*(q + q_neighbour)), a positive-definite
        # Helmholtz-like matrix whose inverse is not the projection: the
        # correction then *amplified* divergence by (1 + 1/(dx dy)).
        off_e = ce_e    # coefficient of east neighbor
        off_w = ce_w    # coefficient of west neighbor
        off_n = cn_n    # coefficient of north neighbor
        off_s = cn_s    # coefficient of south neighbor

        # Boundary conditions.
        # A sealed box has no Dirichlet datum: the pressure operator is a pure
        # Neumann Laplacian, singular up to an additive constant.  Refuse the
        # configuration instead of silently returning an arbitrary pressure
        # level (downstream this produced a non-finite iterate).
        if not self.tp.top_atmosphere and not self.tp.side_outlet:
            raise ValueError(
                "two-phase transient projection needs a pressure datum: set "
                "top_atmosphere=True (p=0 on the top face) or "
                "side_outlet=True (p=0 on the side columns). A fully sealed "
                "domain gives an all-Neumann Poisson problem whose pressure "
                "level is not unique."
            )
        # Top atmosphere: p = 0 on the top *face*, i.e. half a cell above the
        # top cell centre.  The half-cell distance is what makes the discrete
        # hydrostatic gradient exact on the topmost interior face:
        #   (p_top - p_below)/dy = -rho_g g   with  p_top = rho_g g dy/2.
        if self.tp.top_atmosphere:
            inv_rho_top = 1.0 / self.rho_field[-1, :].clamp_min(1e-10)
            diag[-1, :] = diag[-1, :] + 2.0 * inv_rho_top * self.dx / self.dy

        # Bottom wall: no flux (Neumann) → south face already 0 at j=0 (cn_s[0]=0)
        # Side walls: no flux → ce_e[:,-1]=0 and ce_w[:,0]=0 already

        # Replace zero diagonal with 1 (isolated/solid cells)
        diag = torch.where(diag > 0, diag, torch.ones_like(diag))
        diag = torch.where(self.fluid, diag, torch.ones_like(diag))

        coeffs_proj = [off_e, off_w, off_n, off_s]

        def matvec_proj(q):
            return diag * q - self._neighbor_sum(q, coeffs_proj)

        # RHS: (1/dt) * div(u*) * cell volume.  The volume factor is not
        # cosmetic: the operator columns carry face areas (dy/dx, dx/dy), so
        # omitting V from the right-hand side scales phi by 1/V and the
        # velocity correction over-shoots by a factor 1/(dx dy) = 64 (n=8) --
        # the projection then *amplifies* divergence instead of removing it.
        div = ((u_star[:, 1:] - u_star[:, :-1]) / self.dx
               + (v_star[1:] - v_star[:-1]) / self.dy)
        # Sign: u_new = u* - dt (1/rho) grad(phi) must *remove* the
        # divergence, i.e. matvec(phi) = -V div(u*)/dt with the operator
        # matvec = sum_faces ce (phi - phi_neighbour).  Using +V div/dt
        # doubles the divergence instead (measured amplification 2.000).
        b = (-div * (self.dx * self.dy) / dt).masked_fill(~self.fluid, 0)

        # CG solve (Jacobi preconditioned)
        norm = torch.linalg.vector_norm(b).item()
        if norm == 0:
            self._last_dp = None
            return torch.zeros_like(b)
        if (self.tp.warm_start_pressure and self._last_dp is not None
                and self._last_dp.shape == b.shape):
            q = self._last_dp.clone()
            r = b - matvec_proj(q)
        else:
            q = torch.zeros_like(b)
            r = b.clone()
        z = r / diag
        direction, rz = z.clone(), torch.sum(r * z)
        target = max(norm * 1e-10, 1e-14)
        self.cg_diagnostics = {"rhs_norm": norm, "iterations": 0, "residual": norm,
                               "converged": False, "residual_history": []}
        for _ in range(max(500, nx * ny)):
            self.cg_diagnostics["iterations"] += 1
            ad = matvec_proj(direction)
            denominator = torch.sum(direction * ad)
            if denominator.item() <= 0 or not torch.isfinite(denominator):
                break
            step = rz / denominator
            q += step * direction
            r -= step * ad
            residual = torch.linalg.vector_norm(r).item()
            self.cg_diagnostics["residual"] = residual
            self.cg_diagnostics["residual_history"].append(residual / norm)
            if residual <= target:
                self.cg_diagnostics["converged"] = True
                self._last_dp = q
                return q
            z = r / diag
            new_rz = torch.sum(r * z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        self._last_dp = q
        return q

    def _velocity_correction(self, u_star, v_star, dp, dt: float) -> tuple:
        """Return ``(u, v)`` after the projection correction with ``dp``.

        ``u = u* - dt (1/rho)_f grad(dp)``, with the *same* face density that the
        operator in :meth:`_projection_pressure` assembles.  Keeping the
        correction as a pure function of ``(u*, v*, dp)`` lets the operator and
        the correction be checked against each other independently (see
        ``tests/test_two_phase_projection_operator.py``); when they disagree the
        solve returns ``dp`` for a different operator and divergence is only
        partly removed.

        Index conventions (staggered MAC grid):
        ``u`` is ``(ny, nx+1)`` with interior faces ``u[:, 1:-1]`` sitting
        between cells ``i-1`` and ``i``; ``v`` is ``(ny+1, nx)`` with interior
        faces ``v[1:-1, :]`` between cells ``j-1`` and ``j``.  The top face
        ``v[-1, :]`` is corrected across the half-cell distance to the
        atmospheric pressure datum.
        """
        inv_rho_u = 2.0 / (self.rho_field[:, :-1] + self.rho_field[:, 1:]).clamp_min(1e-10)
        inv_rho_v = 2.0 / (self.rho_field[:-1, :] + self.rho_field[1:, :]).clamp_min(1e-10)

        u = u_star.clone()
        v = v_star.clone()
        u[:, 1:-1] = u_star[:, 1:-1] - dt * inv_rho_u * (dp[:, 1:] - dp[:, :-1]) / self.dx
        v[1:-1, :] = v_star[1:-1, :] - dt * inv_rho_v * (dp[1:, :] - dp[:-1, :]) / self.dy
        # Open top boundary: the atmospheric datum lives on the face, half a
        # cell above the top cell centre, so the top face velocity is corrected
        # with the half-cell gradient.  This is what closes the divergence of
        # the top cell and makes the projection an exact projection.
        if self.tp.top_atmosphere:
            inv_rho_top = 1.0 / self.rho_field[-1, :].clamp_min(1e-10)
            v[-1, :] = v_star[-1, :] + dt * inv_rho_top * dp[-1, :] / (0.5 * self.dy)
        return u, v

    def _projection_step(self, dt: float) -> dict:
        """Chorin projection step for two-phase flow on staggered MAC grid.

        1. Predict velocity (diffusion + gravity + CSF - ∇p^n/ρ)
        2. Solve the pressure-correction Poisson: ∇·(1/ρ ∇φ) = (1/dt) ∇·u*
        3. Correct: u = u* - dt/ρ ∇φ, then p ← p + φ
        4. Advect VOF

        Velocity arrays are MAC staggered:
          u: shape (ny, nx+1)  — east face, column i sits between cell (i-1) and i
          v: shape (ny+1, nx)  — north face, row j sits between cell (j-1) and j
        """
        self._boundaries(projection=True)

        rho = self.rho_field      # (ny, nx)
        mu = self.mu_field        # (ny, nx)

        u_old = self.u.clone()
        v_old = self.v.clone()
        # Advective acceleration is evaluated once, from the step-start velocity
        # field, so the u- and v-predictors see the same convection operator.
        adv_u, adv_v = (
            self._advective_accel() if getattr(self.tp, "momentum_advection", False)
            else (None, None))

        # --- Step 1: Predict velocity (explicit Euler, diffusion + gravity) ---
        # We update interior faces only; boundaries handled by _boundaries().
        #
        # u: shape (ny, nx+1). Interior u-faces: rows 1..ny-2, cols 1..nx-1
        #   i.e. u[1:-1, 1:-1] has shape (ny-2, nx-1)
        #   d2u/dx2: (u[j,i-1] - 2*u[j,i] + u[j,i+1]) / dx² → cols [0:-2],[1:-1],[2:]
        #   d2u/dy2: (u[j-1,i] - 2*u[j,i] + u[j+1,i]) / dy² → rows [0:-2],[1:-1],[2:]
        #   mu/rho at u-faces: average of left/right cell → mu[:, :-1] etc.
        d2udx2 = (self.u[:, :-2] - 2 * self.u[:, 1:-1] + self.u[:, 2:]) / self.dx**2
        d2udy2_full = (self.u[:-2, :] - 2 * self.u[1:-1, :] + self.u[2:, :]) / self.dy**2
        # Interior: rows 1..ny-2 of d2udy2, cols 1..nx-1 → [:-1, 1:-1]? No.
        # d2udy2_full is (ny-2, nx+1). We need [all, 1:-1] for interior cols.
        # But we update u[1:-1, 1:-1] which is (ny-2, nx-1).
        # d2udx2 is (ny, nx-1). For interior rows: d2udx2[1:-1, :] is (ny-2, nx-1). ✓
        # d2udy2_full[:, 1:-1] is (ny-2, nx-1). ✓
        mu_u_face = 0.5 * (mu[:, :-1] + mu[:, 1:])       # (ny, nx-1) at all u-faces
        rho_u_face = 0.5 * (rho[:, :-1] + rho[:, 1:])   # (ny, nx-1)
        # Pressure gradient at the *current* level plus the surface-tension
        # source.  The scheme is incremental (p <- p + phi), so the predictor
        # must carry -grad(p^n)/rho: without it the hydrostatic head is not
        # balanced and the projection manufactures a pressure impulse every
        # step.  The CSF term is likewise absent from the SIMPLE-only path.
        dpdx_old = (self.p[:, 1:] - self.p[:, :-1]) / self.dx   # (ny, nx-1)
        dpdy_old = (self.p[1:, :] - self.p[:-1, :]) / self.dy   # (ny-1, nx)
        f_u, f_v = self._csf_face_forces()
        du_dt = (mu_u_face[1:-1, :] * (d2udx2[1:-1, :] + d2udy2_full[:, 1:-1])
                 - dpdx_old[1:-1, :]) / rho_u_face[1:-1, :]
        if f_u is not None:
            du_dt = du_dt + f_u[1:-1, :] / rho_u_face[1:-1, :]
        if adv_u is not None:
            du_dt = du_dt - adv_u
        self.u[1:-1, 1:-1] = u_old[1:-1, 1:-1] + dt * du_dt

        # v: shape (ny+1, nx). Interior v-faces: rows 1..ny-1, cols 1..nx-2
        #   i.e. v[1:-1, 1:-1] has shape (ny-1, nx-2)
        #   d2v/dx2: (v[j,i-1] - 2*v[j,i] + v[j,i+1]) / dx² → cols [0:-2],[1:-1],[2:]
        #   d2v/dy2: (v[j-1,i] - 2*v[j,i] + v[j+1,i]) / dy² → rows [0:-2],[1:-1],[2:]
        #   d2vdx2_full = (ny+1, nx-2); d2vdy2_full = (ny-1, nx)
        d2vdx2_full = (self.v[:, :-2] - 2 * self.v[:, 1:-1] + self.v[:, 2:]) / self.dx**2
        d2vdy2_full = (self.v[:-2, :] - 2 * self.v[1:-1, :] + self.v[2:, :]) / self.dy**2
        # Interior: v[1:-1, 1:-1] is (ny-1, nx-2)
        # d2vdy2_full[1:-1, 1:-1] is (ny-1, nx-2). But d2vdy2_full is (ny-1, nx),
        # so [1:-1, 1:-1] would be (ny-2, nx-2). That's wrong.
        # We need d2vdy2 for rows 1..ny-1, which is d2vdy2_full[0:ny-1, :] = d2vdy2_full[0:ny-1, :].
        # Wait: d2vdy2_full[j-1] corresponds to v[j]. For v[1:-1] (rows 1..ny-1),
        # d2vdy2_full for row j is: (v[j-1] - 2*v[j] + v[j+1])/dy².
        # d2vdy2_full has shape (ny-1, nx) where row k corresponds to v[k+1].
        # For interior v (rows 1..ny-1): d2vdy2_full[0:ny-1, :] = d2vdy2_full[:].
        # But we only need cols 1..nx-2: d2vdy2_full[:, 1:-1] → (ny-1, nx-2). ✓
        # d2vdx2_full for rows 1..ny-1: d2vdx2_full[1:ny, :] → (ny-1, nx-2). Hmm.
        # d2vdx2_full is (ny+1, nx-2). Rows 1..ny-1: [1:-1] → (ny-1, nx-2). ✓
        mu_v_face = 0.5 * (mu[:-1, :] + mu[1:, :])       # (ny-1, nx) at all v-faces
        rho_v_face = 0.5 * (rho[:-1, :] + rho[1:, :])   # (ny-1, nx)
        dv_dt = (mu_v_face[:, 1:-1]
                 * (d2vdx2_full[1:-1, :] + d2vdy2_full[:, 1:-1])
                 - dpdy_old[:, 1:-1]) / rho_v_face[:, 1:-1]
        # Body force.  With the reduced pressure the gravitational head is not
        # in the pressure any more, so what remains is the buoyancy
        # acceleration (rho - rho_ref)/rho * g on the v-faces; for a uniform
        # liquid that term vanishes and a pool at rest is an exact solution.
        rho_v_cell = rho_v_face[:, 1:-1]
        if self.tp.pressure_form == "reduced" and self.props is not None:
            buoy = ((rho_v_cell - self.props.rho_liquid) / rho_v_cell) * (-self.tp.gravity)
            dv_dt = dv_dt + buoy
        else:
            dv_dt = dv_dt - self.tp.gravity
        if f_v is not None:
            dv_dt = dv_dt + f_v[:, 1:-1] / rho_v_face[:, 1:-1]
        if adv_v is not None:
            dv_dt = dv_dt - adv_v
        self.v[1:-1, 1:-1] = v_old[1:-1, 1:-1] + dt * dv_dt

        self._boundaries(projection=True)

        # --- Step 2: Solve pressure-correction Poisson ---
        u_star = self.u.clone()
        v_star = self.v.clone()
        dp = self._projection_pressure(u_star, v_star, dt)

        # --- Step 3: Velocity correction ---
        self.u, self.v = self._velocity_correction(u_star, v_star, dp, dt)
        self.p = self.p + dp
        self._boundaries(projection=True)

        # --- Step 4: VOF transport ---
        if dt > 0:
            self._advect_vof(dt)
            self.time += dt

        # Metrics
        div_max = torch.max(torch.abs(self._divergence())).item()
        vel_change = max(
            torch.max(torch.abs(self.u - u_old)).item(),
            torch.max(torch.abs(self.v - v_old)).item(),
        )
        v_drift = 0.0
        if self.tp.enabled and self.vof is not None:
            v_drift = self.vof.volume_drift(
                self.alpha, self.vof.total_volume(self.alpha))

        item = {
            "iteration": len(self.history) + 1,
            "time": self.time,
            "continuity": div_max,
            "velocity_change": vel_change,
            "volume_drift": v_drift,
        }
        self.history.append(item)

        if not torch.isfinite(self.u).all() or not torch.isfinite(self.v).all():
            raise RuntimeError("non-finite iterate")
        return item

    @torch.no_grad()
    def step(self, dt: float | None = None) -> dict:
        """Perform one step (single-phase or two-phase).

        For two-phase mode, uses a Chorin projection method with
        variable-density pressure Poisson equation.  For single-phase
        mode, uses the original SIMPLE iteration.
        """
        c = self.config
        dt_eff = dt if dt is not None else (c.time_step if c.time_step else 0.0)
        self._current_dt = dt_eff if dt_eff > 0 else None

        if self.tp.enabled and dt_eff > 0:
            return self._projection_step(dt_eff)

        # --- Single-phase SIMPLE path (unchanged from baseline) ---
        self._boundaries()

        old_u = self.u.clone()
        old_v = self.v.clone()

        du = self._predict("u")
        dv = self._predict("v")
        correction = self._pressure_correction(du, dv)

        self.u[:, 1:-1] += du[:, 1:-1] * (correction[:, :-1] - correction[:, 1:])
        if self.tp.side_outlet:
            self.u[:, -1] += du[:, -1] * correction[:, -1]
        self.v[1:-1] += dv[1:-1] * (correction[:-1] - correction[1:])

        self.p += c.pressure_relaxation * correction
        self._boundaries()

        if self.tp.enabled and dt_eff > 0:
            self._advect_vof(dt_eff)
            self.time += dt_eff

        div_max = torch.max(torch.abs(self._divergence())).item()
        vel_change = max(
            torch.max(torch.abs(self.u - old_u)).item(),
            torch.max(torch.abs(self.v - old_v)).item(),
        )
        v_drift = 0.0
        if self.tp.enabled and self.vof is not None:
            v_drift = self.vof.volume_drift(
                self.alpha, self.vof.total_volume(self.alpha))

        item = {
            "iteration": len(self.history) + 1,
            "time": self.time,
            "continuity": div_max,
            "velocity_change": vel_change,
            "volume_drift": v_drift,
        }
        self.history.append(item)

        if not torch.isfinite(self.u).all() or not torch.isfinite(self.v).all():
            raise RuntimeError("non-finite iterate")
        return item

    def solve(self):
        """Run to convergence (steady) or max_iterations."""
        for _ in range(self.config.max_iterations):
            self.step()
            if self.converged:
                break
        return self

    @property
    def converged(self) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        return last["velocity_change"] < self.config.tolerance
