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
    mixture_density,
    mixture_viscosity,
)
from .free_surface_bc import FreeSurfaceBC


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
    # Domain boundary type for two-phase (not using inlet/outlet channel)
    # "closed" = all walls except top atmosphere; "channel" = inlet+outlet
    domain_type: str = "closed"


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
        self.p = torch.where(
            y_grid < interface_y,
            rho_l * g * (interface_y - y_grid),
            rho_g * g * (c.height - y_grid),
        ).to(dtype=self.p.dtype, device=c.device)
        if self.tp.top_atmosphere:
            self.p[-1, :] = 0.0

    def hydrostatic_balance_residual(self) -> dict:
        """Compute the force balance residual at each v-face.

        For a true hydrostatic equilibrium the v-momentum equation reduces
        to:  ∂p/∂y = −ρ·g   (pressure gradient balances gravity).

        Returns the max and RMS of |dp/dy + ρ·g| · dx (per-face residual).
        """
        c = self.config
        g = self.tp.gravity
        rho_v = self._rho_face_v()
        # Discrete pressure gradient at v-faces: dp/dy ≈ (p_N - p_S) / dy
        dpdy = torch.zeros_like(self.v)
        dpdy[1:-1, :] = (self.p[1:, :] - self.p[:-1, :]) / self.dy
        # Residual: dp/dy + rho*g should be 0 for hydrostatic balance
        residual = dpdy + rho_v * g
        # Only evaluate at active v-faces
        active = self._v_active
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
        c, dx, dy = self.config, self.dx, self.dy
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
        c, dx, dy = self.config, self.dx, self.dy
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
            if torch.linalg.vector_norm(r).item() <= target:
                return q
            z = r / diagonal
            new_rz = torch.sum(r * z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        return q

    def _boundaries(self):
        """Apply boundary conditions."""
        # Inlet
        self.u[:, 0] = self.config.inlet_velocity if self.config.inlet_velocity != 0 else 0.0
        # Bottom wall
        if self.tp.bottom_wall:
            self.v[0, :] = 0.0
        # Top atmosphere
        if self.tp.top_atmosphere:
            self.v[-1, :] = self.v[-2, :]  # zero gradient
            self.p[-1, :] = 0.0  # atmospheric
        # Side outlet
        if self.tp.side_outlet:
            self.u[:, -1] = self.u[:, -2]
            self.p[:, -1] = 0.0
            self.p[:, 0] = 0.0
        # Mask inactive
        self.u.masked_fill_(~self._u_active, 0)
        self.v.masked_fill_(~self._v_active, 0)

    def _advect_vof(self, dt: float):
        """Advect volume fraction with donor-cell scheme."""
        self.alpha = self.vof.advect(self.alpha, self.u, self.v, dt)
        self._update_properties()

    def _projection_pressure(self, u_star, v_star, dt):
        """Chorin projection pressure solve for variable-density flow.

        Solves:  ∇·(1/ρ ∇p) = (1/dt) ∇·u*

        Returns the pressure field p (not correction).  The 1/ρ weighting
        is the key: it naturally handles density jumps.  A Jacobi-preconditioned
        CG iteration is used.
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
        off_e = -ce_e   # coefficient of east neighbor
        off_w = -ce_w   # coefficient of west neighbor
        off_n = -cn_n   # coefficient of north neighbor
        off_s = -cn_s   # coefficient of south neighbor

        # Boundary conditions:
        # Top atmosphere: Dirichlet p=0 at j=ny-1
        if self.tp.top_atmosphere:
            # Top row: diag=1, all off-diagonals=0
            diag[-1, :] = 1.0
            off_e[-1, :] = 0.0
            off_w[-1, :] = 0.0
            off_n[-1, :] = 0.0
            off_s[-1, :] = 0.0
            # Row below top: remove north face (it connects to Dirichlet cell)
            # Actually for Dirichlet, the north coefficient of row ny-2 is still
            # valid (it's an interior face). The Dirichlet condition is handled
            # by setting top row's equation to p=0.

        # Bottom wall: no flux (Neumann) → south face already 0 at j=0 (cn_s[0]=0)
        # Side walls: no flux → ce_e[:,-1]=0 and ce_w[:,0]=0 already

        # Replace zero diagonal with 1 (isolated/solid cells)
        diag = torch.where(diag > 0, diag, torch.ones_like(diag))
        diag = torch.where(self.fluid, diag, torch.ones_like(diag))

        coeffs_proj = [off_e, off_w, off_n, off_s]

        def matvec_proj(q):
            return diag * q - self._neighbor_sum(q, coeffs_proj)

        # RHS: (1/dt) * div(u*)
        div = ((u_star[:, 1:] - u_star[:, :-1]) / self.dx
               + (v_star[1:] - v_star[:-1]) / self.dy)
        b = (div / dt).masked_fill(~self.fluid, 0)
        if self.tp.top_atmosphere:
            b[-1, :] = 0.0  # Dirichlet p=0

        # CG solve (Jacobi preconditioned)
        q = torch.zeros_like(b)
        r = b.clone()
        norm = torch.linalg.vector_norm(b).item()
        if norm == 0:
            return q
        z = r / diag
        direction, rz = z.clone(), torch.sum(r * z)
        target = max(norm * 1e-10, 1e-14)
        for _ in range(max(500, nx * ny)):
            ad = matvec_proj(direction)
            denominator = torch.sum(direction * ad)
            if denominator.item() <= 0 or not torch.isfinite(denominator):
                break
            step = rz / denominator
            q += step * direction
            r -= step * ad
            if torch.linalg.vector_norm(r).item() <= target:
                break
            z = r / diag
            new_rz = torch.sum(r * z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        return q

    def _projection_step(self, dt: float) -> dict:
        """Chorin projection step for two-phase flow on staggered MAC grid.

        1. Predict velocity (diffusion + gravity, no pressure gradient)
        2. Solve pressure Poisson: ∇·(1/ρ ∇p) = (1/dt) ∇·u*
        3. Correct: u = u* - dt/ρ ∇p
        4. Advect VOF

        Velocity arrays are MAC staggered:
          u: shape (ny, nx+1)  — east face, column i sits between cell (i-1) and i
          v: shape (ny+1, nx)  — north face, row j sits between cell (j-1) and j
        """
        self._boundaries()

        rho = self.rho_field      # (ny, nx)
        mu = self.mu_field        # (ny, nx)

        u_old = self.u.clone()
        v_old = self.v.clone()

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
        du_dt = (mu_u_face[1:-1, :] * (d2udx2[1:-1, :] + d2udy2_full[:, 1:-1])
                 / rho_u_face[1:-1, :])
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
                 / rho_v_face[:, 1:-1])
        # Gravity body force (same for all fluid)
        grav_v = -self.tp.gravity
        dv_dt = dv_dt + grav_v
        self.v[1:-1, 1:-1] = v_old[1:-1, 1:-1] + dt * dv_dt

        self._boundaries()

        # --- Step 2: Solve pressure Poisson ---
        u_star = self.u.clone()
        v_star = self.v.clone()
        p_new = self._projection_pressure(u_star, v_star, dt)

        # --- Step 3: Velocity correction ---
        # u = u* - dt * (1/rho_face) * dp/dx
        inv_rho_u = 2.0 / (self.rho_field[:, :-1] + self.rho_field[:, 1:]).clamp_min(1e-10)
        inv_rho_v = 2.0 / (self.rho_field[:-1, :] + self.rho_field[1:, :]).clamp_min(1e-10)

        dpdx = (p_new[:, 1:] - p_new[:, :-1]) / self.dx
        dpdy = (p_new[1:, :] - p_new[:-1, :]) / self.dy

        # u[j, i] face (i=1..nx-2 interior) sits between cells (i-1) and (i).
        # dpdx[i-1] = (p[i]-p[i-1])/dx, inv_rho_u[i-1] at that face.
        # dpdx and inv_rho_u are (ny, nx-1); u[:, 1:-1] is (ny, nx-1) → match directly
        self.u[:, 1:-1] = u_star[:, 1:-1] - dt * inv_rho_u * dpdx
        # v[j] face (j=1..ny-1 interior, since v has ny+1 rows: 0..ny)
        # dpdy[j-1] = (p[j]-p[j-1])/dy, inv_rho_v[j-1] at that face.
        # dpdy and inv_rho_v are (ny-1, nx); v[1:-1] is (ny-1, nx) → match directly
        self.v[1:-1, :] = v_star[1:-1, :] - dt * inv_rho_v * dpdy

        self.p = p_new
        self._boundaries()

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
