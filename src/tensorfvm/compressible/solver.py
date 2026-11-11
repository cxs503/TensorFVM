"""Two-dimensional multi-material Euler solver (MUSCL-Hancock + HLLC/GFM).

This is the compressible core added to TensorFVM for the multi-material
interface-dynamics surface.  Design notes, in the order the decisions matter:

State
    Five conserved fields per cell ``U = [rho, rho u, rho v, E, rho Y]`` on a
    uniform Cartesian grid, ``Y`` the mass fraction of material 2.  The mixture
    is closed with the stiffened-gas rules of :mod:`tensorfvm.compressible.eos`;
    the two materials may have different ratios of specific heats, which is what
    makes air/SF6 and air/He interfaces (Richtmyer--Meshkov, shock--bubble)
    representable on one mesh without a separate interface grid.

Discretisation
    MUSCL-Hancock (van Leer 1979; Toro ch. 14): limited primitive slopes, a
    half-time predictor built from the flux difference of the cell's own two
    reconstructed face values, then HLLC (:mod:`tensorfvm.compressible.riemann`)
    on the predicted states.  Second order in space and time, TVD in the
    scalar sense; ``flux_mode="hllc"`` plus ``order=1`` gives the first-order
    donor-cell reference used for order-of-accuracy measurements.

Material interface
    Two modes.  ``"hllc"`` relies on the contact wave of the approximate solver
    and needs nothing else.  ``"gfm"`` adds the ghost-fluid treatment (Fedkiw et
    al. 1999; Liu, Khoo & Yeo 2003): at faces where the material fraction jumps,
    the exact two-material Riemann problem is solved for ``(p_I, u_I)`` and both
    sides are reset to a single interface pressure and normal velocity with
    densities extrapolated along each material's own isentrope.  Because the
    interface pressure is unique, the pressure-oscillation error that a diffuse
    contact produces in a multi-material HLLC flux is removed at the source; the
    two-gas shock-tube test measures the difference.

Conservation
    The update is a pure flux difference, so the *interior* budget is conserved
    to round-off.  Whatever leaves through the domain boundary is accumulated in
    ``self.boundary_outflow`` every step, so an open-boundary run can report the
    accounted drift ``U(t) + outflow(t) - U(0)`` instead of pretending that mass
    and energy are conserved inside an open box.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from .eos import StiffenedGasMixture
from .reconstruct import limited_slope
from .riemann import ExactRiemann, hllc_flux

__all__ = ["EulerSolver", "PRIMITIVE_NAMES"]

PRIMITIVE_NAMES = ("rho", "u", "v", "p", "Y")
_NG = 2  # ghost layers: 2 are needed for a second-order stencil at a boundary face


def _as_tuple(bc: Sequence) -> Tuple:
    return tuple(bc)


class EulerSolver:
    """Multi-material Euler solver on a uniform Cartesian grid.

    Parameters
    ----------
    nx, ny, lx, ly:
        Cell counts and physical extent.  ``ny == 1`` with periodic ``y``
        boundaries degenerates to a one-dimensional solve (used by the
        shock-tube verification).
    eos:
        :class:`~tensorfvm.compressible.eos.StiffenedGasMixture` with two
        materials.
    bc:
        Boundary conditions in the order ``(left, right, bottom, top)``.  Each
        entry is ``"outflow"`` (zero-gradient), ``"reflective"`` (slip wall),
        ``"periodic"``, or a tuple ``("inflow", state)`` with ``state`` a dict
        holding ``rho, u, v, p, Y``.
    flux_mode:
        ``"gfm"`` (ghost fluid at material faces) or ``"hllc"``.
    order:
        2 for MUSCL-Hancock, 1 for first-order donor cell.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        eos: StiffenedGasMixture,
        bc: Sequence = ("outflow", "outflow", "outflow", "outflow"),
        flux_mode: str = "gfm",
        limiter: str = "mc",
        order: int = 2,
        cfl: float = 0.4,
        gfm_threshold: float = 0.2,
        interface_solver: str = "acoustic",
        rho_floor: float = 1e-10,
        p_floor: float = 1e-12,
        solid: torch.Tensor | None = None,
        project_wall_flux: bool = True,
    ) -> None:
        if nx < 4 or ny < 1:
            raise ValueError("grid must be at least 4 cells in x and 1 in y")
        if lx <= 0 or ly <= 0:
            raise ValueError("domain extents must be positive")
        if interface_solver not in ("acoustic", "exact"):
            raise ValueError("interface_solver must be 'acoustic' or 'exact'")
        if flux_mode not in ("gfm", "hllc"):
            raise ValueError("flux_mode must be 'gfm' or 'hllc'")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if eos.nmat != 2:
            raise ValueError("the solver currently supports exactly two materials")
        self.nx, self.ny, self.lx, self.ly = nx, ny, lx, ly
        self.dx, self.dy = lx / nx, ly / ny
        self.eos = eos
        self.bc = _as_tuple(bc)
        if len(self.bc) != 4:
            raise ValueError("bc must be (left, right, bottom, top)")
        self.flux_mode = flux_mode
        self.limiter = limiter
        self.order = order
        self.cfl = cfl
        self.gfm_threshold = gfm_threshold
        self.interface_solver = interface_solver
        self.rho_floor = rho_floor
        self.p_floor = p_floor
        self.U = torch.zeros((5, nx, ny), dtype=torch.float64)
        self.time = 0.0
        self.nsteps = 0
        self.boundary_outflow = torch.zeros(5, dtype=torch.float64)
        self.positivity_fixes = 0
        self.species_clamps = 0
        self.solid = None
        self.project_wall_flux = bool(project_wall_flux)
        self._solid_cells = None
        self._solid_interior: list = []
        self._solid_padded: list = []
        if solid is not None:
            self.set_solid(solid)

    # ------------------------------------------------------------------
    # embedded solid (masked) walls
    # ------------------------------------------------------------------
    def set_solid(self, solid: torch.Tensor) -> None:
        """Install an embedded-wall mask: ``True`` marks a solid cell.

        A solid cell carries no state of its own.  Before every flux evaluation
        its primitive values are mirrored from a neighbouring fluid cell with
        the wall-normal velocity component negated, which is the classical
        reflection ("ghost-cell") treatment of a slip wall: at the face between
        the fluid cell and its mirror the two Riemann states share a pressure
        and have equal and opposite normal velocity, so the HLLC star velocity
        is zero and no mass crosses the wall.  For a solid region that is a
        rectangle aligned with the grid the wall is represented *exactly* (no
        staircase error); for an oblique wall the mask is a staircase and the
        mirror fills every solid cell from the nearest already-filled cell, so
        a run of solid cells reproduces a reflection about the fluid-facing
        face of the block.
        """
        mask = torch.as_tensor(solid, dtype=torch.bool)
        if tuple(mask.shape) != (self.nx, self.ny):
            raise ValueError(f"solid mask must have shape {(self.nx, self.ny)}, "
                             f"got {tuple(mask.shape)}")
        if bool(mask.all()):
            raise ValueError("the solid mask cannot cover the whole domain")
        ng = _NG
        padded = torch.zeros((self.nx + 2 * ng, self.ny + 2 * ng), dtype=torch.bool)
        padded[ng:ng + self.nx, ng:ng + self.ny] = mask
        # a ghost cell inherits the mask of the nearest interior cell, so the
        # solid block is continued past the domain boundary instead of being
        # reflected into fluid
        padded[:ng, :] = padded[ng:ng + 1, :]
        padded[ng + self.nx:, :] = padded[ng + self.nx - 1:ng + self.nx, :]
        padded[:, :ng] = padded[:, ng:ng + 1]
        padded[:, ng + self.ny:] = padded[:, ng + self.ny - 1:ng + self.ny]
        self.solid = mask
        self._solid_cells = mask.reshape(-1).nonzero().squeeze(-1)
        self._solid_interior = self._solid_fill_layers(mask)
        self._solid_padded = self._solid_fill_layers(padded)
        # a wall face separates a fluid cell from a solid one: the mirrored
        # Riemann states make its normal velocity vanish only to truncation
        # error, so the wall fluxes are projected exactly (see ``step``)
        self._wall_x = padded[:-1, :] != padded[1:, :]
        self._wall_y = padded[:, :-1] != padded[:, 1:]

    def _solid_fill_layers(self, mask: torch.Tensor) -> list:
        """Breadth-first fill order for a mask as ``(index, parent, axis)``.

        ``index``/``parent`` are flat cell indices of the (possibly padded)
        array, ``axis`` is ``0`` when the state is mirrored about a face normal
        to x (flip ``u``) and ``1`` when it is mirrored about a face normal to
        y (flip ``v``).
        """
        nxp, nyp = mask.shape
        solid = mask.numpy().copy()
        filled = ~solid
        if not bool(solid.any()):
            return []
        # prefer the vertical mirror: a block roof is the usual fluid-facing face
        dirs = ((0, 1, 1), (0, -1, 1), (-1, 0, 0), (1, 0, 0))
        layers: list = []
        remaining = solid
        while bool(remaining.any()):
            groups: dict = {}
            progressed = filled.copy()
            idx_i, idx_j = remaining.nonzero()
            for i, j in zip(idx_i.tolist(), idx_j.tolist()):
                for di, dj, axis in dirs:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nxp and 0 <= jj < nyp and filled[ii, jj]:
                        groups.setdefault(axis, ([], []))
                        groups[axis][0].append(i * nyp + j)
                        groups[axis][1].append(ii * nyp + jj)
                        progressed[i, j] = True
                        break
            if not groups:
                raise ValueError("solid region is not reachable from any fluid cell")
            for axis in sorted(groups):
                idx, par = groups[axis]
                layers.append((torch.tensor(idx, dtype=torch.long),
                               torch.tensor(par, dtype=torch.long), axis))
            filled = progressed
            remaining = solid & ~filled
        return layers

    def _project_wall_fluxes(self, fx, gy):
        """Enforce an impermeable adiabatic wall on fluid--solid faces.

        The reflected states already give a star velocity of zero up to
        truncation error; the residual (a fraction of a per cent of the inlet
        mass flux) is not acceptable for a conservation budget, so the mass and
        energy fluxes through wall faces are set to zero exactly.  The normal
        momentum flux is left untouched: it is the pressure force that the wall
        exerts on the fluid (the convective part is already zero to the same
        order).
        """
        fx = list(fx)
        gy = list(gy)
        for fluxes, wall in ((fx, self._wall_x), (gy, self._wall_y)):
            for k in (0, 3, 4):
                fluxes[k] = torch.where(wall, torch.zeros_like(fluxes[k]), fluxes[k])
        return tuple(fx), tuple(gy)

    @staticmethod
    def _mirror_solid(prim: Dict[str, torch.Tensor],
                      layers: list) -> Dict[str, torch.Tensor]:
        """Overwrite solid cells with the reflected neighbouring fluid state."""
        for idx, par, axis in layers:
            for name in PRIMITIVE_NAMES:
                q = prim[name].reshape(-1)
                q[idx] = q[par]
            comp = "u" if axis == 0 else "v"
            qc = prim[comp].reshape(-1)
            qc[idx] = -qc[par]
        return prim

    # ------------------------------------------------------------------
    # state handling
    # ------------------------------------------------------------------
    def set_uniform(self, rho: float, u: float, v: float, p: float, Y: float = 0.0) -> None:
        self.set_primitive({"rho": float(rho), "u": float(u), "v": float(v),
                            "p": float(p), "Y": min(max(float(Y), 0.0), 1.0)})

    def set_primitive(self, prim: Dict[str, torch.Tensor]) -> None:
        """Install a primitive state; the dict values ``(nx, ny)`` or broadcastable."""
        shape = (self.nx, self.ny)
        fit = lambda v: self._fit(v, shape)  # noqa: E731
        rho, u, v, p, Y = (fit(prim[k]) for k in ("rho", "u", "v", "p", "Y"))
        if torch.any(rho <= 0) or torch.any(p <= 0):
            raise ValueError("initial state must have positive density and pressure")
        if torch.any(Y < 0) or torch.any(Y > 1):
            raise ValueError("mass fraction must lie in [0, 1]")
        self.U = self._conserved_from_prim({"rho": rho, "u": u, "v": v, "p": p, "Y": Y})

    @staticmethod
    def _fit(v, shape) -> torch.Tensor:
        """Broadcast a scalar / (nx,) / (nx, ny) input onto the grid shape."""
        t = torch.as_tensor(v, dtype=torch.float64)
        if tuple(t.shape) == tuple(shape):
            return t.clone()
        if t.dim() == 1 and shape[1] == 1 and t.shape[0] == shape[0]:
            return t.reshape(shape).clone()
        try:
            return t.expand(shape).clone()
        except RuntimeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"cannot broadcast a state of shape {tuple(t.shape)} "
                             f"onto the grid {tuple(shape)}") from exc

    def primitive(self) -> Dict[str, torch.Tensor]:
        """Primitive fields of the interior cells."""
        return self._primitive_from_conserved(self.U)

    def _conserved_from_prim(self, prim: Dict[str, torch.Tensor]) -> torch.Tensor:
        rho, u, v, p, Y = prim["rho"], prim["u"], prim["v"], prim["p"], prim["Y"]
        E = self.eos.energy_density(rho, u, v, p, Y)
        return torch.stack([rho, rho * u, rho * v, E, rho * Y], dim=0)

    def _primitive_from_conserved(self, U: torch.Tensor) -> Dict[str, torch.Tensor]:
        rho = torch.clamp(U[0], min=self.rho_floor)
        u = U[1] / rho
        v = U[2] / rho
        Y = torch.clamp(U[4] / rho, 0.0, 1.0)
        p = torch.clamp(self.eos.pressure_from_conserved(rho, u, v, U[3], Y),
                        min=self.p_floor)
        return {"rho": rho, "u": u, "v": v, "p": p, "Y": Y}

    # ------------------------------------------------------------------
    # boundary conditions
    # ------------------------------------------------------------------
    def _pad(self, prim: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pad primitive fields with ghost cells according to ``self.bc``."""
        ng, nx, ny = _NG, self.nx, self.ny
        out = {}
        for name in PRIMITIVE_NAMES:
            q = torch.zeros((nx + 2 * ng, ny + 2 * ng), dtype=torch.float64)
            q[ng:ng + nx, ng:ng + ny] = prim[name]
            out[name] = q
        self._fill_x(out, ng, nx, ny)
        self._fill_y(out, ng, nx, ny)
        if self.solid is not None:
            self._mirror_solid(out, self._solid_padded)
        return out

    def _fill_x(self, out, ng, nx, ny) -> None:
        left, right = self.bc[0], self.bc[1]
        for name, q in out.items():
            for k in range(ng):
                # layer index ng-1-k mirrors interior column ng+k
                if left == "periodic":
                    q[ng - 1 - k, ng:ng + ny] = q[ng + nx - 1 - k, ng:ng + ny]
                elif left == "reflective":
                    q[ng - 1 - k, ng:ng + ny] = q[ng + k, ng:ng + ny]
                elif isinstance(left, tuple):
                    q[ng - 1 - k, ng:ng + ny] = left[1][name]
                else:  # outflow
                    q[ng - 1 - k, ng:ng + ny] = q[ng, ng:ng + ny]
                if right == "periodic":
                    q[ng + nx + k, ng:ng + ny] = q[ng + k, ng:ng + ny]
                elif right == "reflective":
                    q[ng + nx + k, ng:ng + ny] = q[ng + nx - 1 - k, ng:ng + ny]
                elif isinstance(right, tuple):
                    q[ng + nx + k, ng:ng + ny] = right[1][name]
                else:
                    q[ng + nx + k, ng:ng + ny] = q[ng + nx - 1, ng:ng + ny]
        # reflective walls flip the wall-normal velocity
        for side, idx in ((left, "l"), (right, "r")):
            if side != "reflective":
                continue
            u = out["u"]
            for k in range(ng):
                if idx == "l":
                    u[ng - 1 - k, ng:ng + ny] = -u[ng + k, ng:ng + ny]
                else:
                    u[ng + nx + k, ng:ng + ny] = -u[ng + nx - 1 - k, ng:ng + ny]

    def _fill_y(self, out, ng, nx, ny) -> None:
        bottom, top = self.bc[2], self.bc[3]
        for name, q in out.items():
            for k in range(ng):
                if bottom == "periodic":
                    q[:, ng - 1 - k] = q[:, ng + ny - 1 - k]
                elif bottom == "reflective":
                    q[:, ng - 1 - k] = q[:, ng + k]
                elif isinstance(bottom, tuple):
                    q[:, ng - 1 - k] = bottom[1][name]
                else:
                    q[:, ng - 1 - k] = q[:, ng]
                if top == "periodic":
                    q[:, ng + ny + k] = q[:, ng + k]
                elif top == "reflective":
                    q[:, ng + ny + k] = q[:, ng + ny - 1 - k]
                elif isinstance(top, tuple):
                    q[:, ng + ny + k] = top[1][name]
                else:
                    q[:, ng + ny + k] = q[:, ng + ny - 1]
        for side, idx in ((bottom, "b"), (top, "t")):
            if side != "reflective":
                continue
            v = out["v"]
            for k in range(ng):
                if idx == "b":
                    v[:, ng - 1 - k] = -v[:, ng + k]
                else:
                    v[:, ng + ny + k] = -v[:, ng + ny - 1 - k]

    # ------------------------------------------------------------------
    # fluxes
    # ------------------------------------------------------------------
    def _hllc_state(self, prim: Dict[str, torch.Tensor], normal: str = "x") -> dict:
        rho, u, v, p, Y = prim["rho"], prim["u"], prim["v"], prim["p"], prim["Y"]
        un, ut = (u, v) if normal == "x" else (v, u)
        beta = self.eos.beta(Y)
        pi = self.eos.pi(Y)
        gamma = 1.0 + 1.0 / beta
        pinf = pi / (gamma * beta)
        return {
            "rho": rho,
            "un": un,
            "ut": ut,
            "p": p,
            "E": p * beta + pi + 0.5 * rho * (u * u + v * v),
            "rhoY": rho * Y,
            "c": torch.sqrt(torch.clamp(gamma * (p + pinf) / rho, min=1e-30)),
            "pinf": pinf,
        }

    def _physical_flux(self, prim: Dict[str, torch.Tensor], normal: str):
        st = self._hllc_state(prim, normal)
        un = st["un"]
        if normal == "x":
            return (
                st["rho"] * un,
                st["rho"] * un * un + st["p"],
                st["rho"] * un * st["ut"],
                un * (st["E"] + st["p"]),
                st["rhoY"] * un,
            )
        return (
            st["rho"] * un,
            st["rho"] * un * st["ut"],
            st["rho"] * un * un + st["p"],
            un * (st["E"] + st["p"]),
            st["rhoY"] * un,
        )

    def _gfm_states(self, prim_l, prim_r, normal: str = "x"):
        """Ghost-fluid interface states on material faces.

        ``prim_l``/``prim_r`` are the two sides of a set of faces, flattened to
        one dimension.  Where the material fraction jumps by more than
        ``gfm_threshold`` the exact two-material Riemann problem is solved and
        both sides are reset to ``(p_I, u_I)`` with densities extrapolated along
        each material's own isentrope.
        """
        nkey = "u" if normal == "x" else "v"
        Yl, Yr = prim_l["Y"], prim_r["Y"]
        if self.interface_solver == "acoustic":
            return self._gfm_states_acoustic(prim_l, prim_r, normal)
        mask = torch.abs(Yl - Yr) > self.gfm_threshold
        prim_l = {k: v.clone() for k, v in prim_l.items()}
        prim_r = {k: v.clone() for k, v in prim_r.items()}
        if not bool(mask.any()):
            return prim_l, prim_r
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        Yl_f, Yr_f = Yl[idx], Yr[idx]
        rho_l, rho_r = prim_l["rho"][idx], prim_r["rho"][idx]
        p_l, p_r = prim_l["p"][idx], prim_r["p"][idx]
        gamma_l, gamma_r = self.eos.gamma_eff(Yl_f), self.eos.gamma_eff(Yr_f)
        pinf_l, pinf_r = self.eos.pinf_eff(Yl_f), self.eos.pinf_eff(Yr_f)
        riem = ExactRiemann(
            rho_l, prim_l[nkey][idx], p_l, gamma_l,
            rho_r, prim_r[nkey][idx], p_r, gamma_r, pinf_l, pinf_r,
        )
        p_star, u_star = riem.solve()
        # isentropic extrapolation of each material to the interface pressure
        rho_l_star = rho_l * torch.pow(
            (p_star + pinf_l) / (p_l + pinf_l), 1.0 / gamma_l)
        rho_r_star = rho_r * torch.pow(
            (p_star + pinf_r) / (p_r + pinf_r), 1.0 / gamma_r)
        for prim, rho_s in ((prim_l, rho_l_star), (prim_r, rho_r_star)):
            prim["rho"][idx] = rho_s
            prim[nkey][idx] = u_star
            prim["p"][idx] = p_star
        return prim_l, prim_r

    def _gfm_states_acoustic(self, prim_l, prim_r, normal: str = "x"):
        """Ghost-fluid interface states from the *linearised* interface Riemann solve.

        The two-material linearised (acoustic) Riemann problem gives the unique
        interface pressure and normal velocity

        .. math::

            u_I = \\frac{Z_L u_L + Z_R u_R + p_L - p_R}{Z_L + Z_R}, \\qquad
            p_I = \\frac{Z_R p_L + Z_L p_R + Z_L Z_R (u_L - u_R)}{Z_L + Z_R},

        with ``Z = rho c`` per material.  It is the standard ghost-fluid interface
        state and costs one algebraic evaluation per face instead of an iterative
        exact solve; the exact (Toro) solver remains available through
        ``interface_solver="exact"`` and is what the 1-D verification uses.
        """
        nkey = "u" if normal == "x" else "v"
        mask = torch.abs(prim_l["Y"] - prim_r["Y"]) > self.gfm_threshold
        prim_l = {k: v.clone() for k, v in prim_l.items()}
        prim_r = {k: v.clone() for k, v in prim_r.items()}
        if not bool(mask.any()):
            return prim_l, prim_r
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        Yl_f, Yr_f = prim_l["Y"][idx], prim_r["Y"][idx]
        rho_l, rho_r = prim_l["rho"][idx], prim_r["rho"][idx]
        p_l, p_r = prim_l["p"][idx], prim_r["p"][idx]
        gamma_l, gamma_r = self.eos.gamma_eff(Yl_f), self.eos.gamma_eff(Yr_f)
        pinf_l, pinf_r = self.eos.pinf_eff(Yl_f), self.eos.pinf_eff(Yr_f)
        c_l = torch.sqrt(gamma_l * (p_l + pinf_l) / rho_l)
        c_r = torch.sqrt(gamma_r * (p_r + pinf_r) / rho_r)
        z_l, z_r = rho_l * c_l, rho_r * c_r
        u_l, u_r = prim_l[nkey][idx], prim_r[nkey][idx]
        denom = z_l + z_r
        u_star = (z_l * u_l + z_r * u_r + p_l - p_r) / denom
        p_star = (z_r * p_l + z_l * p_r + z_l * z_r * (u_l - u_r)) / denom
        p_star = torch.clamp(p_star, min=self.p_floor)
        rho_l_star = rho_l * torch.pow((p_star + pinf_l) / (p_l + pinf_l), 1.0 / gamma_l)
        rho_r_star = rho_r * torch.pow((p_star + pinf_r) / (p_r + pinf_r), 1.0 / gamma_r)
        for prim, rho_s in ((prim_l, rho_l_star), (prim_r, rho_r_star)):
            prim["rho"][idx] = rho_s
            prim[nkey][idx] = u_star
            prim["p"][idx] = p_star
        return prim_l, prim_r

    def _face_flux(self, prim_l: Dict[str, torch.Tensor], prim_r: Dict[str, torch.Tensor],
                   normal: str = "x"):
        """HLLC (optionally ghost-fluid) flux for a flattened set of faces."""
        if self.flux_mode == "gfm":
            prim_l, prim_r = self._gfm_states(prim_l, prim_r, normal)
        return hllc_flux(self._hllc_state(prim_l, normal),
                         self._hllc_state(prim_r, normal))

    # ------------------------------------------------------------------
    # one time step
    # ------------------------------------------------------------------
    def max_speed(self) -> Tuple[float, float]:
        prim = self.primitive()
        c = self.eos.sound_speed(prim["rho"], prim["p"], prim["Y"])
        return (
            float(torch.max(torch.abs(prim["u"]) + c)),
            float(torch.max(torch.abs(prim["v"]) + c)),
        )

    def stable_dt(self) -> float:
        sx, sy = self.max_speed()
        rate_x = sx / self.dx if self.nx > 1 else 0.0
        rate_y = sy / self.dy if self.ny > 1 and self.ly > 0 else 0.0
        rate = rate_x + rate_y
        if rate <= 0.0:
            return float("inf")
        return self.cfl / rate

    def step(self, dt: float) -> None:
        """Advance one time step of size ``dt`` (MUSCL-Hancock + HLLC/GFM)."""
        if dt <= 0 or not torch.isfinite(torch.tensor(dt)):
            raise ValueError("dt must be positive and finite")
        ng, nx, ny = _NG, self.nx, self.ny
        prim = self.primitive()
        if self.solid is not None:
            prim = self._mirror_solid(prim, self._solid_interior)
        prim = self._pad(prim)
        U_pad = self._conserved_from_prim(prim)

        if self.order == 2:
            slopes = {
                name: (
                    limited_slope(prim[name], 0, self.limiter),
                    limited_slope(prim[name], 1, self.limiter),
                )
                for name in PRIMITIVE_NAMES
            }
            # right-face (plus) and left-face (minus) reconstructed states
            plus_x = {n: prim[n] + 0.5 * slopes[n][0] for n in PRIMITIVE_NAMES}
            minus_x = {n: prim[n] - 0.5 * slopes[n][0] for n in PRIMITIVE_NAMES}
            plus_y = {n: prim[n] + 0.5 * slopes[n][1] for n in PRIMITIVE_NAMES}
            minus_y = {n: prim[n] - 0.5 * slopes[n][1] for n in PRIMITIVE_NAMES}
            fxl = self._physical_flux(plus_x, "x")
            fxr = self._physical_flux(minus_x, "x")
            gyb = self._physical_flux(minus_y, "y")
            gyt = self._physical_flux(plus_y, "y")
            U_half = U_pad.clone()
            for k in range(5):
                U_half[k] = (
                    U_pad[k]
                    - dt / (2.0 * self.dx) * (fxl[k] - fxr[k])
                    - dt / (2.0 * self.dy) * (gyb[k] - gyt[k])
                )
            half = self._primitive_from_conserved(U_half)
            left_x = {n: half[n] + 0.5 * slopes[n][0] for n in PRIMITIVE_NAMES}
            right_x = {n: half[n] - 0.5 * slopes[n][0] for n in PRIMITIVE_NAMES}
            left_y = {n: half[n] + 0.5 * slopes[n][1] for n in PRIMITIVE_NAMES}
            right_y = {n: half[n] - 0.5 * slopes[n][1] for n in PRIMITIVE_NAMES}
        else:
            left_x = right_x = left_y = right_y = prim
        # the reconstructed mass fraction must stay physical: a slope overshoot
        # here is what produces out-of-range Y after the update
        for fields in (left_x, right_x, left_y, right_y):
            if fields is prim:
                continue
            fields["Y"] = torch.clamp(fields["Y"], 0.0, 1.0)

        # face k sits between padded cells k-1 and k
        fx = self._face_flux(
            {n: left_x[n][:-1, :] for n in PRIMITIVE_NAMES},
            {n: right_x[n][1:, :] for n in PRIMITIVE_NAMES},
            "x",
        )
        gy_raw = self._face_flux(
            {n: left_y[n][:, :-1] for n in PRIMITIVE_NAMES},
            {n: right_y[n][:, 1:] for n in PRIMITIVE_NAMES},
            "y",
        )
        # the normal/tangential ordering of the y-face flux is (mass, mom_y, mom_x,
        # energy, species); store it as (mass, mom_x, mom_y, energy, species)
        gy = (gy_raw[0], gy_raw[2], gy_raw[1], gy_raw[3], gy_raw[4])
        if self.solid is not None and self.project_wall_flux:
            fx, gy = self._project_wall_fluxes(fx, gy)

        # face element j sits between padded cells j and j+1, so interior cell i
        # (padded index i+ng) owns face elements i+ng-1 (left) and i+ng (right)
        fx_l = [f[ng - 1:ng - 1 + nx, ng:ng + ny] for f in fx]
        fx_r = [f[ng:ng + nx, ng:ng + ny] for f in fx]
        gy_b = [g[ng:ng + nx, ng - 1:ng - 1 + ny] for g in gy]
        gy_t = [g[ng:ng + nx, ng:ng + ny] for g in gy]

        for k in range(5):
            self.U[k] = (
                self.U[k]
                - dt / self.dx * (fx_r[k] - fx_l[k])
                - dt / self.dy * (gy_t[k] - gy_b[k])
            )

        # open-boundary ledger: outward flux through the four domain faces.
        # Only the fluid part of a face is open: where an embedded solid block
        # reaches the boundary the cell there is a mirror, not a reservoir, and
        # counting it would corrupt the budget.
        outward = torch.zeros(5, dtype=torch.float64)
        open_r = open_l = open_t = open_b = None
        if self.solid is not None:
            open_l, open_r = ~self.solid[0, :], ~self.solid[-1, :]
            open_b, open_t = ~self.solid[:, 0], ~self.solid[:, -1]
        for k in range(5):
            fr = fx_r[k][-1, :]
            fl = fx_l[k][0, :]
            gt = gy_t[k][:, -1]
            gb = gy_b[k][:, 0]
            if self.solid is not None:
                fr = fr * open_r
                fl = fl * open_l
                gt = gt * open_t
                gb = gb * open_b
            outward[k] += (fr.sum() - fl.sum()) * self.dy
            outward[k] += (gt.sum() - gb.sum()) * self.dx
        self.boundary_outflow = self.boundary_outflow + dt * outward

        self._enforce_positivity()
        self._sync_solid_state()
        self.time += dt
        self.nsteps += 1

    def _sync_solid_state(self) -> None:
        """Replace the conserved state of solid cells by the mirrored fluid state.

        Solid cells take part in the flux sum like any other cell, so their own
        storage accumulates the wall momentum flux and would otherwise drift
        without bound (which shows up as a collapsing time step through
        :meth:`max_speed`).  Their state carries no information: only the
        reflected value used at the wall face does.  Overwriting it keeps
        :meth:`primitive` and :meth:`integrals` finite; the fluid-domain budget
        must still exclude the solid cells.
        """
        if self.solid is None:
            return
        prim = self._mirror_solid(self.primitive(), self._solid_interior)
        U = self._conserved_from_prim(prim)
        self.U[:, self.solid] = U[:, self.solid]

    def _enforce_positivity(self) -> None:
        """Repair non-positive density/pressure (counted, non-conservative)."""
        rho = self.U[0]
        bad = rho <= self.rho_floor
        if bool(bad.any()):
            self.positivity_fixes += int(bad.sum())
            self.U[0] = torch.clamp(self.U[0], min=self.rho_floor)
        prim = self._primitive_from_conserved(self.U)
        if bool((prim["p"] <= self.p_floor * 1.0000001).any()):
            self.positivity_fixes += 1
            pe = self.eos.pi(prim["Y"]) + self.p_floor * self.eos.beta(prim["Y"])
            ke = 0.5 * (self.U[1] ** 2 + self.U[2] ** 2) / self.U[0]
            self.U[3] = torch.maximum(self.U[3], ke + pe)
        yraw = self.U[4] / self.U[0]
        n_clamp = int(((yraw < -1e-12) | (yraw > 1.0 + 1e-12)).sum())
        if n_clamp:
            self.species_clamps += n_clamp
            self.U[4] = torch.clamp(yraw, 0.0, 1.0) * self.U[0]

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    def run(
        self,
        t_end: float,
        dt_max: float | None = None,
        callback=None,
        max_steps: int = 2_000_000,
    ) -> None:
        """Integrate to ``t_end`` with the CFL-limited step."""
        if t_end <= self.time:
            raise ValueError("t_end must exceed the current time")
        while self.time < t_end and self.nsteps < max_steps:
            dt = self.stable_dt()
            if dt_max is not None:
                dt = min(dt, dt_max)
            dt = min(dt, t_end - self.time)
            if not torch.isfinite(torch.tensor(dt)) or dt <= 0:
                raise RuntimeError("non-positive time step")
            self.step(dt)
            if callback is not None:
                callback(self)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def integrals(self) -> Dict[str, float]:
        """Volume integrals of the conserved fields (unit depth)."""
        cell = self.dx * self.dy
        U = self.U
        return {
            "mass": float(U[0].sum() * cell),
            "mass_mat1": float((U[0] - U[4]).sum() * cell),
            "mass_mat2": float(U[4].sum() * cell),
            "momentum_x": float(U[1].sum() * cell),
            "momentum_y": float(U[2].sum() * cell),
            "energy": float(U[3].sum() * cell),
        }

    def budget_drift(self, initial: Dict[str, float]) -> Dict[str, float]:
        """Relative drift of the *accounted* budget ``U(t) + outflow(t) - U(0)``.

        ``outflow`` is the time integral of the outward boundary flux, so this
        is the honest measure for open-boundary runs; for closed or periodic
        boundaries ``outflow`` is zero and the number is the interior
        conservation error.
        """
        now = self.integrals()
        out = {
            "mass": float(self.boundary_outflow[0]),
            "mass_mat2": float(self.boundary_outflow[4]),
            "momentum_x": float(self.boundary_outflow[1]),
            "momentum_y": float(self.boundary_outflow[2]),
            "energy": float(self.boundary_outflow[3]),
        }
        drift = {}
        for key in ("mass", "momentum_x", "momentum_y", "energy"):
            ref = initial[key]
            if abs(ref) < 1e-300:
                drift[key] = float("nan")
            else:
                drift[key] = (now[key] + out[key] - ref) / abs(ref)
        m1_now = (self.U[0] - self.U[4]).sum() * self.dx * self.dy
        m1_0 = initial["mass"] - initial["mass_mat2"]
        drift["mass_mat1"] = float(
            (m1_now + out["mass"] - out["mass_mat2"] - m1_0) / abs(m1_0)
            if abs(m1_0) > 1e-300 else float("nan"))
        return drift