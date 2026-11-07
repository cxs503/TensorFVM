"""Steady staggered-MAC finite-volume kernels for the incompressible benchmark set.

Self-contained on purpose: nothing here shares mesh, projection or pressure code
with :mod:`tensorfvm.lid_cavity` (DCT projection, closed box) or
:mod:`tensorfvm.solver` (cylinder SIMPLE), so the evidence it produces is an
*independent* FVM verification surface instead of a second reading of an existing
one.

Layout (staggered / MAC, uniform Cartesian grid)::

    u     (ny, nx + 1)   normal velocity on vertical faces
    v     (ny + 1, nx)   normal velocity on horizontal faces
    p, T  (ny, nx)       cell centred

A boolean ``fluid`` mask makes any cell a stair-step solid.  Convection is
first-order upwind inside the implicit operator plus a bounded second-order
deferred-correction flux (van Leer weighted) applied conservatively, diffusion is
central, and wall distances are honoured on closed faces (half a cell in the
transverse direction, a full cell along the face-normal direction) so that wall
shear -- and therefore a friction coefficient -- is meaningful.  Pressure-velocity
coupling is SIMPLE (Patankar): the under-relaxed momentum diagonal supplies the
pressure-correction mobility, and the symmetric positive-definite correction
system is solved by Jacobi-preconditioned conjugate gradients.

Non-dimensional equations solved::

    div(u)   = 0
    div(u u) = -grad p + nu lap(u) + buoyancy (T - t_ref) e_y
    div(u T) = alpha lap(T)

``nu`` and ``alpha`` are supplied by the caller, so Reynolds and Rayleigh
conventions live in :mod:`tensorfvm.inc_bench.cases`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

EPS = 1e-300


def shift(a: torch.Tensor, axis: int, k: int) -> torch.Tensor:
    """``out[i] = a[i - k]`` along ``axis``; zero padded, same shape."""
    if k == 0:
        return a.clone()
    out = torch.zeros_like(a)
    n = a.shape[axis]
    if abs(k) >= n:
        return out
    if k > 0:
        out.narrow(axis, k, n - k).copy_(a.narrow(axis, 0, n - k))
    else:
        d = -k
        out.narrow(axis, 0, n - d).copy_(a.narrow(axis, d, n - d))
    return out


def divergence(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Cell-centred divergence of a staggered velocity field."""
    return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy


def cell_velocity(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell-centred velocity components, each ``(ny, nx)``."""
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])


def streamfunction(u: torch.Tensor, v: torch.Tensor, dy: float,
                   fluid: torch.Tensor) -> torch.Tensor:
    """``psi = int_0^y u dy`` (closed domain, bottom wall is a streamline)."""
    uc, _ = cell_velocity(u, v)
    psi = torch.zeros_like(uc)
    running = torch.zeros_like(uc[0])
    for i in range(uc.shape[0]):
        running = running + uc[i] * dy
        psi[i] = running
    return torch.where(fluid, psi, torch.zeros_like(psi))


def limiter_weight(upwind_slope: torch.Tensor, downstream_slope: torch.Tensor
                   ) -> torch.Tensor:
    """van Leer weight in ``[0, 1]`` from upwind and downstream slopes."""
    ratio = upwind_slope / torch.clamp(downstream_slope.abs(), min=1e-13)
    weight = (ratio + ratio.abs()) / (1.0 + ratio.abs())
    return torch.clamp(weight, 0.0, 1.0)


@dataclass
class SolverOptions:
    """Under-relaxation, iteration budget and discretisation switches."""

    alpha_u: float = 0.7
    alpha_p: float = 0.3
    alpha_t: float = 0.8
    momentum_sweeps: int = 3
    energy_sweeps: int = 4
    pressure_solver: str = "cg"
    pressure_sweeps: int = 16
    pressure_omega: float = 1.7
    pressure_iterations: int = 300
    pressure_tolerance: float = 1.0e-8
    max_iterations: int = 40000
    tolerance: float = 1e-7
    min_iterations: int = 120
    check_every: int = 25
    high_order: bool = True
    correction_ramp: float = 1.0
    ramp_iterations: int = 200
    warmup_iterations: int = 300
    alpha_u_warmup: float = 0.4
    alpha_p_warmup: float = 0.15

    def __post_init__(self) -> None:
        for name in ("alpha_u", "alpha_p", "alpha_t"):
            value = getattr(self, name)
            if not math.isfinite(value) or not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_iterations < 1 or self.min_iterations < 0:
            raise ValueError("max_iterations must be positive, min_iterations >= 0")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.momentum_sweeps < 1 or self.energy_sweeps < 1:
            raise ValueError("sweep counts must be positive")


class SteadyFlow:
    """Steady SIMPLE solver on a masked staggered grid.

    Parameters
    ----------
    fluid:
        ``(ny, nx)`` boolean mask, ``True`` for fluid cells.
    viscosity:
        kinematic viscosity ``nu`` (density is unity).
    inlet_profile:
        optional ``(ny,)`` prescribed ``u`` on the west boundary, applied where
        the adjacent cell is fluid; ``None`` makes the west boundary a wall.
    outlet:
        ``True`` gives zero-gradient velocity and zero gauge pressure on the east
        boundary; ``False`` makes it a wall.
    conductivity:
        thermal diffusivity ``alpha``; ``None`` disables the energy equation.
    buoyancy:
        transverse momentum source ``buoyancy * (T - t_ref)`` (volume weighted).
    t_hot, t_cold:
        Dirichlet wall temperatures on the west / east wall (``None`` =
        adiabatic).
    """

    def __init__(self, fluid: torch.Tensor, *, dx: float, dy: float,
                 viscosity: float, density: float = 1.0,
                 inlet_profile: torch.Tensor | None = None, outlet: bool = True,
                 conductivity: float | None = None, buoyancy: float = 0.0,
                 t_ref: float = 0.0, t_hot: float | None = None,
                 t_cold: float | None = None,
                 options: SolverOptions | None = None) -> None:
        if fluid.dim() != 2 or fluid.dtype is not torch.bool:
            raise ValueError("fluid must be a 2-D boolean mask")
        if dx <= 0 or dy <= 0 or viscosity <= 0 or density <= 0:
            raise ValueError("dx, dy, viscosity and density must be positive")
        if conductivity is not None and conductivity <= 0:
            raise ValueError("conductivity must be positive")
        self.fluid = fluid
        self.ny, self.nx = int(fluid.shape[0]), int(fluid.shape[1])
        self.dx, self.dy = float(dx), float(dy)
        self.nu, self.rho = float(viscosity), float(density)
        self.opts = options or SolverOptions()
        self.buoyancy, self.t_ref = float(buoyancy), float(t_ref)
        self.alpha_T = None if conductivity is None else float(conductivity)
        self.t_hot, self.t_cold = t_hot, t_cold
        opts = {"dtype": torch.float64}
        f, ny, nx = self.fluid, self.ny, self.nx
        self.u = torch.zeros((ny, nx + 1), **opts)
        self.v = torch.zeros((ny + 1, nx), **opts)
        self.p = torch.zeros((ny, nx), **opts)
        self.T = torch.full((ny, nx), self.t_ref, **opts)
        self.u_dof = torch.zeros((ny, nx + 1), dtype=torch.bool)
        self.u_dof[:, 1:nx] = f[:, :nx - 1] & f[:, 1:]
        self.v_dof = torch.zeros((ny + 1, nx), dtype=torch.bool)
        self.v_dof[1:ny] = f[:ny - 1] & f[1:]
        self.u_known = torch.zeros((ny, nx + 1), **opts)
        self.v_known = torch.zeros((ny + 1, nx), **opts)
        self.u_inlet = torch.zeros((ny, nx + 1), dtype=torch.bool)
        self.u_outlet = torch.zeros((ny, nx + 1), dtype=torch.bool)
        self.inlet_profile = None
        if inlet_profile is not None:
            profile = torch.as_tensor(inlet_profile, **opts).reshape(-1)
            if profile.numel() != ny:
                raise ValueError("inlet_profile needs one entry per row")
            self.inlet_profile = profile
            self.u_inlet[:, 0] = f[:, 0]
        if outlet:
            self.u_outlet[:, nx] = f[:, nx - 1]
        self.u_open = self.u_dof | self.u_inlet | self.u_outlet
        self.v_open = self.v_dof
        self._dp_guess: torch.Tensor | None = None
        self.alpha_u = self.opts.alpha_u
        self.alpha_p = self.opts.alpha_p
        self.history: list[dict[str, float]] = []
        self.converged = False
        self.iterations = 0
        self._refresh_known()
        if self.inlet_profile is not None:
            # Mass-consistent start: the inlet flux spread over the outlet area,
            # which keeps the initial continuity defect small and is markedly more
            # robust on fine grids than a zero field.
            outlet_area = float(self.fluid[:, -1].sum()) * self.dy
            if outlet_area > 0:
                mean = float(self.inlet_profile.sum() * self.dy) / outlet_area
                self.u = torch.where(self.u_open, torch.full_like(self.u, mean), self.u)

    # ------------------------------------------------------------------ helpers
    def _refresh_known(self) -> None:
        if self.inlet_profile is not None:
            self.u_known[:, 0] = torch.where(self.u_inlet[:, 0], self.inlet_profile,
                                             torch.zeros_like(self.inlet_profile))
            self.u[:, 0] = self.u_known[:, 0]
        if bool(self.u_outlet.any()):
            self.u_known[:, -1] = self.u[:, -2]
            self.u[:, -1] = self.u_known[:, -1]
        self.u = torch.where(self.u_open, self.u, torch.zeros_like(self.u))
        self.v = torch.where(self.v_open, self.v, torch.zeros_like(self.v))

    def _transverse_averages(self, trans: torch.Tensor, field: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Average a staggered transverse velocity onto a normal-face CV."""
        shape = field.shape
        left = torch.zeros(shape, dtype=field.dtype)
        right = torch.zeros(shape, dtype=field.dtype)
        left[:, 1:] = trans[:-1, :]
        right[:, :trans.shape[1]] = trans[:-1, :]
        t_s = 0.5 * (left + right)
        left = torch.zeros(shape, dtype=field.dtype)
        right = torch.zeros(shape, dtype=field.dtype)
        left[:, 1:] = trans[1:, :]
        right[:, :trans.shape[1]] = trans[1:, :]
        t_n = 0.5 * (left + right)
        return t_n, t_s

    def _face_fluxes(self, field: torch.Tensor, dof: torch.Tensor,
                     known: torch.Tensor, opened: torch.Tensor, dx: float, dy: float
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        fld = torch.where(dof, field, known)
        nbr_e = torch.where(shift(dof, 1, -1), shift(fld, 1, -1), shift(known, 1, -1))
        nbr_w = torch.where(shift(dof, 1, 1), shift(fld, 1, 1), shift(known, 1, 1))
        fe = self.rho * dy * 0.5 * (field + nbr_e) * shift(opened, 1, -1).to(fld.dtype)
        fw = self.rho * dy * 0.5 * (field + nbr_w) * shift(opened, 1, 1).to(fld.dtype)
        return fe, fw

    def _assembly(self, field: torch.Tensor, trans: torch.Tensor,
                  dof: torch.Tensor, known: torch.Tensor, opened: torch.Tensor,
                  pressure_force: torch.Tensor, source: torch.Tensor,
                  dx: float, dy: float, extra: torch.Tensor
                  ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Momentum coefficients of the ``u``-oriented normal-face problem."""
        mu, rho = self.nu, self.rho
        fld = torch.where(dof, field, known)
        fe, fw = self._face_fluxes(field, dof, known, opened, dx, dy)
        t_n, t_s = self._transverse_averages(trans, field)
        fn, fs = rho * dx * t_n, rho * dx * t_s
        de, dn = mu * dy / dx, mu * dx / dy
        dof_e = shift(dof, 1, -1).to(fld.dtype)
        dof_w = shift(dof, 1, 1).to(fld.dtype)
        nbr_e = torch.where(shift(dof, 1, -1), shift(fld, 1, -1), shift(known, 1, -1))
        nbr_w = torch.where(shift(dof, 1, 1), shift(fld, 1, 1), shift(known, 1, 1))
        aE = de + torch.clamp(-fe, min=0) * dof_e
        aW = de + torch.clamp(fw, min=0) * dof_w
        b = pressure_force + source + extra
        b = b + de * (1.0 - dof_e) * nbr_e + de * (1.0 - dof_w) * nbr_w
        dof_n = shift(dof, 0, -1).to(fld.dtype)
        dof_s = shift(dof, 0, 1).to(fld.dtype)
        aN = torch.where(dof_n > 0, dn + torch.clamp(-fn, min=0), 2 * dn)
        aS = torch.where(dof_s > 0, dn + torch.clamp(fs, min=0), 2 * dn)
        b = b + (1.0 - dof_n) * 2 * dn * shift(fld, 0, -1)
        b = b + (1.0 - dof_s) * 2 * dn * shift(fld, 0, 1)
        aP = aE + aW + aN + aS
        aP = torch.where(dof, torch.clamp(aP, min=de), torch.ones_like(aP))
        return aP, [aE, aW, aN, aS], b

    def _deferred_flux(self, field: torch.Tensor, dof: torch.Tensor,
                       known: torch.Tensor, opened: torch.Tensor, dx: float, dy: float
                       ) -> torch.Tensor:
        """Conservative high-order flux correction over the east faces.

        A *hybrid* blend is used: the correction moves the upwind face value
        towards the central one by ``beta`` and ``beta`` decays with the local
        face Peclet number, so the correction is second order where the grid
        resolves the shear and vanishes where upwinding is needed for stability
        (this is the deferred form of Patankar's hybrid scheme, which is far more
        robust than a MUSCL limiter in recirculating flow).
        """
        fld = torch.where(dof, field, known)
        fe, _ = self._face_fluxes(field, dof, known, opened, dx, dy)
        downstream = torch.where(shift(dof, 1, -1), shift(fld, 1, -1),
                                 shift(known, 1, -1))
        face_open = shift(opened, 1, -1)
        peclet = torch.abs(fe) / max(self.nu * dy / dx, EPS)
        beta = 1.0 / (1.0 + (peclet / 2.0) ** 2)
        central = 0.5 * (fld + downstream)
        upwind_value = torch.where(fe >= 0, fld, downstream)
        correction = -fe * beta * (central - upwind_value)
        return torch.where(face_open, correction, torch.zeros_like(field))

    @staticmethod
    def _gauss_seidel(field: torch.Tensor, aP: torch.Tensor, coeffs: list[torch.Tensor],
                      b: torch.Tensor, dof: torch.Tensor, sweeps: int) -> torch.Tensor:
        aE, aW, aN, aS = coeffs
        ny, nxx = field.shape
        ii = torch.arange(ny).view(-1, 1)
        jj = torch.arange(nxx).view(1, -1)
        parity = (ii + jj) % 2
        colours = [parity == 0, parity == 1]
        value = torch.where(dof, field, torch.zeros_like(field))
        safe = torch.clamp(aP, min=EPS)
        for _ in range(sweeps):
            for colour in colours:
                rhs = (aE * shift(value, 1, -1) + aW * shift(value, 1, 1)
                       + aN * shift(value, 0, -1) + aS * shift(value, 0, 1) + b)
                value = torch.where(dof & colour, rhs / safe, value)
        return torch.where(dof, value, field)

    # ------------------------------------------------------------------- energy
    def _energy_coefficients(self, T: torch.Tensor
                             ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        dx, dy, f = self.dx, self.dy, self.fluid
        g = self.alpha_T
        fd = f.to(T.dtype)
        fe = self.rho * dy * self.u[:, 1:] * fd * shift(f, 1, -1).to(T.dtype)
        fw = self.rho * dy * self.u[:, :-1] * fd * shift(f, 1, 1).to(T.dtype)
        fn = self.rho * dx * self.v[1:, :] * fd * shift(f, 0, -1).to(T.dtype)
        fs = self.rho * dx * self.v[:-1, :] * fd * shift(f, 0, 1).to(T.dtype)
        de, dn = g * dy / dx, g * dx / dy
        aE = (de + torch.clamp(-fe, min=0)) * shift(f, 1, -1).to(T.dtype)
        aW = (de + torch.clamp(fw, min=0)) * shift(f, 1, 1).to(T.dtype)
        aN = (dn + torch.clamp(-fn, min=0)) * shift(f, 0, -1).to(T.dtype)
        aS = (dn + torch.clamp(fs, min=0)) * shift(f, 0, 1).to(T.dtype)
        b = torch.zeros_like(T)
        if self.t_hot is not None:
            aW = aW.clone()
            aW[:, 0] = 2 * de
            b[:, 0] = b[:, 0] + 2 * de * self.t_hot
        if self.t_cold is not None:
            aE = aE.clone()
            aE[:, -1] = 2 * de
            b[:, -1] = b[:, -1] + 2 * de * self.t_cold
        aP = aE + aW + aN + aS
        aP = torch.where(f, torch.clamp(aP, min=de), torch.ones_like(aP))
        return aP, [aE, aW, aN, aS], b

    def _energy_deferred(self, T: torch.Tensor) -> torch.Tensor:
        """Conservative limited correction for cell-centred convection."""
        f = self.fluid
        fd = f.to(T.dtype)
        g = self.alpha_T
        dx, dy = self.dx, self.dy
        total = torch.zeros_like(T)
        for axis, (flux, area) in enumerate((
                (self.u[:, 1:], self.rho * dy),
                (self.v[1:, :], self.rho * dx))):
            if axis == 0:
                face_flux = flux * fd * shift(f, 1, -1).to(T.dtype) * area
                downstream = shift(T, 1, -1)
                opened = shift(f, 1, -1) & f
            else:
                face_flux = flux * fd * shift(f, 0, -1).to(T.dtype) * area
                downstream = shift(T, 0, -1)
                opened = shift(f, 0, -1) & f
            peclet = torch.abs(face_flux) / max(g * float(area), EPS)
            beta = 1.0 / (1.0 + (peclet / 2.0) ** 2)
            central = 0.5 * (T + downstream)
            upwind_value = torch.where(face_flux >= 0, T, downstream)
            gamma = torch.where(opened, -face_flux * beta * (central - upwind_value),
                                torch.zeros_like(T))
            total = total + gamma - shift(gamma, axis, 1)
        return torch.where(f, total, torch.zeros_like(total))

    # --------------------------------------------------------------- pressure
    def _pressure_sor(self, rhs: torch.Tensor, diagonal: torch.Tensor,
                      coeffs: list[torch.Tensor], fluid: torch.Tensor) -> torch.Tensor:
        """Red-black SOR sweeps on the pressure-correction system.

        SIMPLE only needs an approximate correction, and a fixed number of
        red-black SOR sweeps scales far better than a Jacobi-preconditioned CG
        when the grid is refined (the CG iteration count grows like the grid
        width, and under-resolved CG iterations destabilise the outer loop).  The
        sweeps are warm started from the previous correction.
        """
        ce, cw, cn, cs = coeffs
        ny, nx = rhs.shape
        ii = torch.arange(ny).view(-1, 1)
        jj = torch.arange(nx).view(1, -1)
        parity = (ii + jj) % 2
        colours = [parity == 0, parity == 1]
        if self._dp_guess is not None and self._dp_guess.shape == rhs.shape:
            q = torch.where(fluid, self._dp_guess, torch.zeros_like(rhs))
        else:
            q = torch.zeros_like(rhs)
        safe = torch.clamp(diagonal, min=EPS)
        omega = self.opts.pressure_omega
        pinned = not bool(self.u_outlet.any())
        for _ in range(self.opts.pressure_sweeps):
            for colour in colours:
                neighbour = (ce * shift(q, 1, -1) + cw * shift(q, 1, 1)
                             + cn * shift(q, 0, -1) + cs * shift(q, 0, 1))
                updated = (rhs + neighbour) / safe
                q = torch.where(fluid & colour, q + omega * (updated - q), q)
            if pinned:
                q[0, 0] = 0.0
        self._dp_guess = torch.where(fluid, q, torch.zeros_like(rhs))
        return self._dp_guess

    def _pressure_correction(self, d_u: torch.Tensor, d_v: torch.Tensor,
                             defect: torch.Tensor) -> torch.Tensor:
        f = self.fluid
        ny, nx = self.ny, self.nx
        ce = torch.zeros_like(defect)
        cw = torch.zeros_like(defect)
        cn = torch.zeros_like(defect)
        cs = torch.zeros_like(defect)
        ce[:, :nx - 1] = d_u[:, 1:nx]
        cw[:, 1:] = d_u[:, 1:nx]
        cn[:ny - 1, :] = d_v[1:ny, :]
        cs[1:, :] = d_v[1:ny, :]
        diagonal = ce + cw + cn + cs
        if bool(self.u_outlet.any()):
            diagonal = diagonal.clone()
            diagonal[:, -1] = diagonal[:, -1] + d_u[:, -1]
        diagonal = torch.clamp(diagonal, min=EPS)
        diagonal = torch.where(f, diagonal, torch.ones_like(diagonal))
        ce_off, cw_off, cn_off, cs_off = ce.clone(), cw.clone(), cn.clone(), cs.clone()
        rhs = torch.where(f, -defect, torch.zeros_like(defect))
        if bool(self.u_outlet.any()):
            ce_off[:, -1] = 0.0
        else:
            ce_off[0, 0] = cw_off[0, 0] = cn_off[0, 0] = cs_off[0, 0] = 0.0
            diagonal = diagonal.clone()
            diagonal[0, 0] = 1.0
            rhs = rhs.clone()
            rhs[0, 0] = 0.0

        def matvec(q: torch.Tensor) -> torch.Tensor:
            out = diagonal * q
            out = out - ce_off * shift(q, 1, -1) - cw_off * shift(q, 1, 1)
            out = out - cn_off * shift(q, 0, -1) - cs_off * shift(q, 0, 1)
            return torch.where(f, out, torch.zeros_like(out))

        if self.opts.pressure_solver == "sor":
            return self._pressure_sor(rhs, diagonal, [ce_off, cw_off, cn_off, cs_off], f)
        q = torch.zeros_like(rhs)
        if self._dp_guess is not None and self._dp_guess.shape == rhs.shape:
            q = torch.where(f, self._dp_guess, torch.zeros_like(rhs))
        r = torch.where(f, rhs - matvec(q), torch.zeros_like(rhs))
        norm = float(torch.linalg.vector_norm(r))
        if norm == 0.0:
            self._dp_guess = q
            return q
        z = r / diagonal
        direction = z.clone()
        rz = torch.sum(r * z)
        target = max(norm * self.opts.pressure_tolerance, 1e-18)
        for _ in range(self.opts.pressure_iterations):
            ad = matvec(direction)
            denominator = torch.sum(direction * ad)
            if float(denominator) <= 0.0 or not math.isfinite(float(denominator)):
                break
            step = rz / denominator
            q = q + step * direction
            r = r - step * ad
            if float(torch.linalg.vector_norm(r)) <= target:
                break
            z = r / diagonal
            new_rz = torch.sum(r * z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        self._dp_guess = q
        return q

    # ---------------------------------------------------------------- momentum
    def _solve_component(self, field: torch.Tensor, trans: torch.Tensor,
                         dof: torch.Tensor, known: torch.Tensor, opened: torch.Tensor,
                         pressure_force: torch.Tensor, source: torch.Tensor,
                         dx: float, dy: float, ramp: float
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.opts
        extra = torch.zeros_like(field)
        if o.high_order and ramp > 0.0:
            gamma = self._deferred_flux(field, dof, known, opened, dx, dy)
            extra = ramp * (gamma - shift(gamma, 1, 1))
        aP, coeffs, b = self._assembly(field, trans, dof, known, opened, pressure_force,
                                       source, dx, dy, extra)
        relaxed = aP / self.alpha_u
        b = b + (1.0 - self.alpha_u) / self.alpha_u * aP * torch.where(dof, field,
                                                                torch.zeros_like(field))
        solved = self._gauss_seidel(field, relaxed, coeffs, b, dof, o.momentum_sweeps)
        return aP, solved

    def step(self, ramp: float = 1.0) -> None:
        o = self.opts
        self._refresh_known()
        aP_u, u_new = self._solve_component(
            self.u, self.v, self.u_dof, self.u_known, self.u_open,
            self._pressure_force_u(), torch.zeros_like(self.u), self.dx, self.dy, ramp)
        self.u = torch.where(self.u_open, u_new, self.u)
        self._refresh_known()
        pT = self.p.T.contiguous()
        dp_v = torch.zeros((self.nx, self.ny + 1), dtype=self.p.dtype)
        dp_v[:, 1:self.ny] = (pT[:, :self.ny - 1] - pT[:, 1:self.ny]) * self.dx
        source_v = torch.zeros((self.nx, self.ny + 1), dtype=self.p.dtype)
        if self.buoyancy:
            cell = (self.T.T.contiguous() - self.t_ref) * self.buoyancy * self.dx * self.dy
            source_v[:, 1:self.ny] = 0.5 * (cell[:, :self.ny - 1] + cell[:, 1:self.ny])
        aP_v_t, v_new_t = self._solve_component(
            self.v.T.contiguous(), self.u.T.contiguous(), self.v_dof.T.contiguous(),
            self.v_known.T.contiguous(), self.v_open.T.contiguous(), dp_v, source_v,
            self.dy, self.dx, ramp)
        self.v = torch.where(self.v_open, v_new_t.T.contiguous(), self.v)
        self._refresh_known()
        # Pressure correction (continuity).
        d_u = torch.where(self.u_open, self.dy * self.alpha_u / torch.clamp(aP_u, min=EPS),
                          torch.zeros_like(aP_u)) * self.rho
        aP_v = aP_v_t.T.contiguous()
        d_v = torch.where(self.v_open, self.dx * self.alpha_u / torch.clamp(aP_v, min=EPS),
                          torch.zeros_like(aP_v)) * self.rho
        defect = self.rho * torch.where(
            self.fluid,
            (self.u[:, 1:] - self.u[:, :-1]) * self.dy
            + (self.v[1:, :] - self.v[:-1, :]) * self.dx,
            torch.zeros_like(self.p))
        correction = self._pressure_correction(d_u, d_v, defect)
        pad_col = torch.zeros(self.ny, 1, dtype=correction.dtype)
        self.u = self.u + d_u * torch.cat([pad_col, correction], dim=1) \
            - d_u * torch.cat([correction, pad_col], dim=1)
        pad_row = torch.zeros(1, self.nx, dtype=correction.dtype)
        self.v = self.v + d_v * torch.cat([pad_row, correction], dim=0) \
            - d_v * torch.cat([correction, pad_row], dim=0)
        self.p = self.p + self.alpha_p * correction
        if bool(self.u_outlet.any()):
            self.p = self.p.clone()
            self.p[:, -1] = 0.0
        self._refresh_known()
        if self.alpha_T is not None:
            aP, coeffs, b = self._energy_coefficients(self.T)
            if o.high_order and ramp > 0.0:
                b = b + ramp * self._energy_deferred(self.T)
            relaxed = aP / o.alpha_t
            b = b + (1.0 - o.alpha_t) / o.alpha_t * aP * torch.where(
                self.fluid, self.T, torch.zeros_like(self.T))
            self.T = self._gauss_seidel(self.T, relaxed, coeffs, b, self.fluid,
                                        o.energy_sweeps)
        self.iterations += 1

    def _pressure_force_u(self) -> torch.Tensor:
        force = torch.zeros_like(self.u)
        force[:, 1:self.nx] = (self.p[:, :self.nx - 1] - self.p[:, 1:self.nx]) * self.dy
        force[:, -1] = self.p[:, -1] * self.dy
        return force

    # ------------------------------------------------------------- diagnostics
    def divergence_field(self) -> torch.Tensor:
        return torch.where(self.fluid, divergence(self.u, self.v, self.dx, self.dy),
                           torch.zeros_like(self.p))

    def residuals(self) -> dict[str, float]:
        uc, vc = cell_velocity(self.u, self.v)
        div = self.divergence_field()
        return {
            "divergence": float(div.abs().max()) * max(self.dx, self.dy),
            "u_max": float(uc.abs().max()),
            "v_max": float(vc.abs().max()),
            "p_range": float(self.p.max() - self.p.min()),
        }

    @torch.no_grad()
    def solve(self, max_iterations: int | None = None,
              tolerance: float | None = None) -> dict[str, object]:
        """Iterate to a steady state; returns a convergence record."""
        o = self.opts
        limit = o.max_iterations if max_iterations is None else int(max_iterations)
        tol = o.tolerance if tolerance is None else float(tolerance)
        scale = 1.0
        if self.inlet_profile is not None:
            scale = max(1.0, float(self.inlet_profile.abs().max()))
        change = float("inf")
        while self.iterations < limit:
            previous = (self.u.clone(), self.v.clone(), self.p.clone(),
                        None if self.alpha_T is None else self.T.clone())
            ramp = 0.0
            if o.high_order:
                ramp = o.correction_ramp * min(
                    1.0, max(0.0, self.iterations - o.warmup_iterations) / o.ramp_iterations)
            fraction = min(1.0, self.iterations / max(1, o.warmup_iterations))
            self.alpha_u = o.alpha_u_warmup + fraction * (o.alpha_u - o.alpha_u_warmup)
            self.alpha_p = o.alpha_p_warmup + fraction * (o.alpha_p - o.alpha_p_warmup)
            self.step(ramp)
            change = max(float((self.u - previous[0]).abs().max()),
                         float((self.v - previous[1]).abs().max()))
            if previous[3] is not None:
                delta_t = float((self.T - previous[3]).abs().max())
            else:
                delta_t = 0.0
            p_change = float((self.p - previous[2]).abs().max())
            record = self.residuals()
            record.update({"iteration": float(self.iterations), "change": change / scale,
                           "temperature_change": delta_t, "pressure_change": p_change})
            if self.iterations % o.check_every == 0 or change / scale < tol:
                self.history.append(record)
            if self.iterations >= o.min_iterations and change / scale < tol \
                    and delta_t < max(tol, 1e-9):
                self.converged = True
                break
            if not math.isfinite(change):
                raise RuntimeError("non-finite SIMPLE iterate")
        return {"iterations": self.iterations, "converged": self.converged,
                "change": change / scale,
                "final": self.history[-1] if self.history else self.residuals()}