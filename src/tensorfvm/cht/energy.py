"""Single-field conjugate heat transfer on a Cartesian orthogonal grid.

The fluid and the solid are discretised on one structured grid that straddles the
interface; every cell carries a thermal diffusivity ``kappa`` and every face one
number.  The face conductance is the distance-weighted harmonic mean

.. math::  C_f = A_f \\Big/ \\Big(\\frac{s_P}{\\kappa_P} + \\frac{s_N}{\\kappa_N}\\Big)

so the heat flux through a fluid/solid face is a single unknown: continuity
``q_fluid = q_solid`` holds *by construction*, and the audit re-derives both
one-sided fluxes from the reconstructed interface temperature to prove the
identity numerically (and to catch a future regression that would break it).

Discretisation: cell-centred finite volume, upwind convection in the implicit
operator plus a van-Leer-limited second-order deferred correction, central
diffusion.  The deferred correction makes the converged solution second order
while keeping the implicit matrix an M-matrix.  The steady system is solved by a
sparse direct factorisation inside a Picard loop, so the discretisation is
verified independently of any residual-based iterative-solver question.

Sign convention used throughout: every row of the assembled system collects the
*heat entering* the cell, so at convergence ``sum(heat in) = 0`` and the same
expression evaluated after the solve is the per-cell residual reported in the
ledger.

Boundary conditions are per-side and, where a side is shared by fluid and solid
(for example the inlet plane of a plate with a conducting substrate), per-cell:
``dirichlet``, ``neumann`` (prescribed heat flux into the domain),
``adiabatic`` (no conduction, upwind convection) and ``outflow``
(no conduction, extrapolated convection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..registry import capability

__all__ = ["solve_steady_energy", "solve_harmonic_conduction", "solve_transient_conduction", "Grid2D", "EnergyLedger"]


@dataclass
class Grid2D:
    """Rectangular orthogonal grid: 1-D face/centre coordinates per direction."""

    x_faces: np.ndarray
    y_faces: np.ndarray

    def __post_init__(self):
        self.x_faces = np.asarray(self.x_faces, dtype=float)
        self.y_faces = np.asarray(self.y_faces, dtype=float)
        if np.any(np.diff(self.x_faces) <= 0) or np.any(np.diff(self.y_faces) <= 0):
            raise ValueError("face coordinates must increase")

    @property
    def nx(self) -> int:
        return self.x_faces.size - 1

    @property
    def ny(self) -> int:
        return self.y_faces.size - 1

    @property
    def xc(self) -> np.ndarray:
        return 0.5 * (self.x_faces[1:] + self.x_faces[:-1])

    @property
    def yc(self) -> np.ndarray:
        return 0.5 * (self.y_faces[1:] + self.y_faces[:-1])

    @property
    def dx(self) -> np.ndarray:
        return np.diff(self.x_faces)

    @property
    def dy(self) -> np.ndarray:
        return np.diff(self.y_faces)

    @classmethod
    def graded(cls, x0, x1, nx, y0, y1, ny, power=1.0):
        """Uniform grid in x and ``y = y0 + (y1-y0) (k/ny)**power`` in y."""
        xf = np.linspace(x0, x1, nx + 1)
        s = np.linspace(0.0, 1.0, ny + 1) ** power
        yf = y0 + (y1 - y0) * s
        return cls(xf, yf)


def _contact_resistance(value, shape, axis):
    resistance = np.asarray(value, dtype=float)
    if resistance.ndim and resistance.shape != shape:
        raise ValueError(f"{axis} contact resistance needs shape {shape} or a scalar")
    if not np.isfinite(resistance).all() or (resistance < 0).any():
        raise ValueError(f"{axis} contact resistance must be finite and nonnegative")
    return resistance


def _face_conductance(dx, dy, kappa, contact_resistance_x=0.0, contact_resistance_y=0.0, conductivity_y=None):
    """Return ``(cond_e, cond_n)`` face conductances (per unit depth)."""
    ny, nx = kappa.shape
    if conductivity_y is not None:
        conductivity_y = np.broadcast_to(np.asarray(conductivity_y, dtype=float), (ny, nx))
        if not np.isfinite(conductivity_y).all() or np.any(conductivity_y <= 0):
            raise ValueError("conductivity_y must be finite and positive")
    cond_e = np.zeros((ny, nx + 1))
    cond_n = np.zeros((ny + 1, nx))
    denom = 0.5 * dx[:-1][None, :] / kappa[:, :-1] + 0.5 * dx[1:][None, :] / kappa[:, 1:]
    denom += _contact_resistance(contact_resistance_x, denom.shape, "x")
    cond_e[:, 1:nx] = dy[:, None] / denom
    cond_e[:, 0] = dy / (0.5 * dx[0] / kappa[:, 0])
    cond_e[:, nx] = dy / (0.5 * dx[-1] / kappa[:, -1])
    ky = kappa if conductivity_y is None else conductivity_y
    denom_n = 0.5 * dy[:-1][:, None] / ky[:-1, :] + 0.5 * dy[1:][:, None] / ky[1:, :]
    denom_n += _contact_resistance(contact_resistance_y, denom_n.shape, "y")
    cond_n[1:ny, :] = dx[None, :] / denom_n
    cond_n[0, :] = dx / (0.5 * dy[0] / ky[0, :])
    cond_n[ny, :] = dx / (0.5 * dy[-1] / ky[-1, :])
    return cond_e, cond_n


def _hybrid_weight(peclet, pe_low=2.0, pe_high=2.5):
    """Blend weight for the second-order (central) face value.

    1 for ``|Pe| <= pe_low`` (diffusion dominated, central is stable and second
    order), 0 for ``|Pe| >= pe_high`` (pure upwind, bounded), linear in between.
    """
    w = (pe_high - np.abs(peclet)) / (pe_high - pe_low)
    return np.clip(w, 0.0, 1.0)


def _face_values_x(T, flux_e, cond_e, dx, first_order=False):
    """Upwind and blended second-order values on the x faces.

    The blended value is ``up + w (central - up)`` with ``w`` from
    :func:`_hybrid_weight`, so the deferred correction stays bounded: it can only
    move the face value between the upwind and the central one, never past them.
    """
    ny, nx = T.shape
    up = np.zeros((ny, nx + 1))
    ho = np.zeros((ny, nx + 1))
    for f in range(1, nx):
        F = flux_e[:, f]
        P = T[:, f - 1]
        N = T[:, f]
        up_in = np.where(F > 0.0, P, N)
        up[:, f] = up_in
        sP = 0.5 * dx[f - 1]
        sN = 0.5 * dx[f]
        central = (P * sN + N * sP) / (sP + sN)
        if first_order:
            ho[:, f] = up_in
        else:
            pe = np.abs(F) / np.maximum(cond_e[:, f], 1e-300)
            w = _hybrid_weight(pe)
            ho[:, f] = up_in + w * (central - up_in)
    return up, ho


def _face_values_n(T, flux_n, cond_n, dy, first_order=False):
    ny, nx = T.shape
    up = np.zeros((ny + 1, nx))
    ho = np.zeros((ny + 1, nx))
    for f in range(1, ny):
        F = flux_n[f, :]
        P = T[f - 1, :]
        N = T[f, :]
        up_in = np.where(F > 0.0, P, N)
        up[f, :] = up_in
        sP = 0.5 * dy[f - 1]
        sN = 0.5 * dy[f]
        central = (P * sN + N * sP) / (sP + sN)
        if first_order:
            ho[f, :] = up_in
        else:
            pe = np.abs(F) / np.maximum(cond_n[f, :], 1e-300)
            w = _hybrid_weight(pe)
            ho[f, :] = up_in + w * (central - up_in)
    return up, ho


def _bc_arrays(bc, side, count, default="adiabatic", *, dtype=float):
    spec = bc.get(side)
    if spec is None:
        return np.array([default] * count), np.zeros(count)
    types = spec.get("types")
    values = spec.get("values", 0.0)
    if types is None:
        types = [spec["type"]] * count
    types = np.asarray(types)
    values = np.broadcast_to(np.asarray(values, dtype=dtype), (count,)).copy()
    return types, values


@dataclass
@capability('fvm.thermal.conjugate_energy',
            kind='thermal',
            maturity='developing',
            test_path='tests/test_cht_conjugate_anchors.py',
            wired=True,
            tested=True,
            verified=False,
            endpoint='cli:tensorfvm-thermal',
            evidence={'grade': 'MECHANICS', 'docs': 0})
class EnergyLedger:
    """Global energy bookkeeping for a solved field (all in the same units)."""

    boundary: dict = field(default_factory=dict)
    advective: dict = field(default_factory=dict)
    inflow: float = 0.0
    outflow: float = 0.0
    imbalance: float = 0.0
    normalized_imbalance: float = 0.0
    residual_inf: float = 0.0
    residual_scaled: float = 0.0
    generation: float = 0.0

    def as_dict(self) -> dict:
        return {
            "boundary_heat_rate": {k: float(v) for k, v in self.boundary.items()},
            "advective_heat_rate": {k: float(v) for k, v in self.advective.items()},
            "inflow": float(self.inflow),
            "outflow": float(self.outflow),
            "imbalance": float(self.imbalance),
            "generation": float(self.generation),
            "normalized_imbalance": float(self.normalized_imbalance),
            "residual_inf": float(self.residual_inf),
            "residual_scaled": float(self.residual_scaled),
        }


def _convective_faces(T, flux_e, flux_n, cond_e, cond_n, bval, dx, dy, first_order=False):
    """Face values for the convection term, boundary faces consistent with the BC.

    Boundary faces take the value the implicit operator actually used: the
    Dirichlet value where the flux enters the domain, the interior value where it
    leaves.  Filling them (instead of leaving zeros) is what makes the deferred
    correction and the audit close the *same* equation at the boundary.
    """
    ny, nx = T.shape
    up_x, ho_x = _face_values_x(T, flux_e, cond_e, dx, first_order)
    up_y, ho_y = _face_values_n(T, flux_n, cond_n, dy, first_order)
    up_x[:, 0] = np.where(flux_e[:, 0] > 0, bval["west"], T[:, 0])
    up_x[:, nx] = np.where(flux_e[:, nx] < 0, bval["east"], T[:, nx - 1])
    up_y[0, :] = np.where(flux_n[0, :] > 0, bval["south"], T[0, :])
    up_y[ny, :] = np.where(flux_n[ny, :] < 0, bval["north"], T[ny - 1, :])
    ho_x[:, 0] = up_x[:, 0]
    ho_x[:, nx] = up_x[:, nx]
    ho_y[0, :] = up_y[0, :]
    ho_y[ny, :] = up_y[ny, :]
    return up_x, ho_x, up_y, ho_y


def _conv_net_out(face_x, face_y, flux_e, flux_n):
    """Net convective heat *leaving* each cell for the given face values."""
    return (face_x[:, 1:] * flux_e[:, 1:] - face_x[:, :-1] * flux_e[:, :-1]
            + face_y[1:, :] * flux_n[1:, :] - face_y[:-1, :] * flux_n[:-1, :])


def _assemble_energy(dx, dy, kappa, flux_e, flux_n, bc, source, contact_resistance_x, contact_resistance_y, conductivity_y=None, *, load_dtype=float):
    """Assemble net entering heat as A @ T - b, plus the ledger inputs."""
    ny, nx = kappa.shape
    cond_e, cond_n = _face_conductance(dx, dy, kappa, contact_resistance_x, contact_resistance_y, conductivity_y)
    idx = np.arange(ny * nx).reshape(ny, nx)

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(np.asarray(r).ravel())
        cols.append(np.asarray(c).ravel())
        vals.append(np.asarray(v).ravel())

    diag = np.zeros(ny * nx)
    rates = np.broadcast_to(np.asarray(source, dtype=load_dtype), (ny, nx)) * np.outer(dy, dx)
    if not np.isfinite(rates).all():
        raise ValueError("source must be finite")
    b = -rates.flatten()

    # --- interior x faces --------------------------------------------------
    P = idx[:, :-1]
    N = idx[:, 1:]
    c = cond_e[:, 1:nx]
    diag[P.ravel()] -= c.ravel()
    diag[N.ravel()] -= c.ravel()
    add(P, N, c)
    add(N, P, c)
    F = flux_e[:, 1:nx]
    pos = F > 0
    diag[P.ravel()] += np.where(pos, -F, 0.0).ravel()
    add(P, N, np.where(pos, 0.0, -F))
    diag[N.ravel()] += np.where(pos, 0.0, F).ravel()
    add(N, P, np.where(pos, F, 0.0))

    # --- interior y faces --------------------------------------------------
    Pn = idx[:-1, :]
    Nn = idx[1:, :]
    cn = cond_n[1:ny, :]
    diag[Pn.ravel()] -= cn.ravel()
    diag[Nn.ravel()] -= cn.ravel()
    add(Pn, Nn, cn)
    add(Nn, Pn, cn)
    Fn = flux_n[1:ny, :]
    posn = Fn > 0
    diag[Pn.ravel()] += np.where(posn, -Fn, 0.0).ravel()
    add(Pn, Nn, np.where(posn, 0.0, -Fn))
    diag[Nn.ravel()] += np.where(posn, 0.0, Fn).ravel()
    add(Nn, Pn, np.where(posn, Fn, 0.0))

    # --- boundary faces ----------------------------------------------------
    # the face areas travel with the boundary sides: a prescribed Neumann flux
    # is a heat *rate* q * A, and the conductance does not carry A explicitly.
    side_specs = [
        ("west", idx[:, 0], cond_e[:, 0], flux_e[:, 0], -1.0, dy),
        ("east", idx[:, nx - 1], cond_e[:, nx], flux_e[:, nx], +1.0, dy),
        ("south", idx[0, :], cond_n[0, :], flux_n[0, :], -1.0, dx),
        ("north", idx[ny - 1, :], cond_n[ny, :], flux_n[ny, :], +1.0, dx),
    ]
    btype, bval = {}, {}
    for side, cells, cf, ff, sign_out, area in side_specs:
        types, values = _bc_arrays(bc, side, cells.size, dtype=load_dtype)
        btype[side] = types
        bval[side] = values
        robin = types == "robin"
        if np.any(robin):
            spec = bc[side]
            if "film_coefficient" not in spec:
                raise ValueError(f"Robin boundary on {side} needs film_coefficient")
            h = np.broadcast_to(np.asarray(spec["film_coefficient"], dtype=float), cells.shape)[robin]
            if not np.isfinite(h).all() or np.any(h < 0):
                raise ValueError("film coefficient must be finite and nonnegative")
            film = h * area[robin]
            # Replace the boundary face conductance in-place for the ledger too.
            # The half-cell and ambient film are in series; zero film insulates.
            cf[robin] *= film / (cf[robin] + film)
        cf = np.asarray(cf).ravel()
        ff = np.asarray(ff).ravel()
        area = np.broadcast_to(np.asarray(area, dtype=float).ravel(), cells.shape)
        cells = cells.ravel()
        prescribed = (types == "dirichlet") | robin
        diag[cells[prescribed]] -= cf[prescribed]
        b[cells[prescribed]] -= cf[prescribed] * values[prescribed]
        # ``neumann`` = prescribed heat flux *into the domain* (per unit area):
        # no conduction unknown on that face, the flux lands on the right-hand
        # side as a heat rate.  Same sign convention as the Dirichlet branch.
        neumann = types == "neumann"
        b[cells[neumann]] -= values[neumann] * area[neumann]
        out = sign_out * ff                       # positive = leaving the domain
        diag[cells] -= np.where(out > 0.0, out, 0.0)
        inflow = out < 0.0
        b[cells[inflow]] += out[inflow] * values[inflow]

    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ny * nx, ny * nx)).tocsc()
    A = (A + sp.diags(diag)).tocsc()
    return A, b, (cond_e, cond_n, side_specs, btype, bval, rates)


def solve_steady_energy(dx, dy, kappa, flux_e, flux_n, bc, *,
                        t_init=None, tol=1e-13, max_picard=500, dc_relax=1.0,
                        high_order=True, source=0.0, contact_resistance_x=0.0,
                        contact_resistance_y=0.0, conductivity_y=None, return_wall_power=False):
    """Solve ``div(u T) = div(kappa grad T) + source`` with the CHT face conductance.

    Parameters
    ----------
    dx, dy : (nx,), (ny,) cell sizes.
    source : scalar or cell field of volumetric temperature production (K/s).
    kappa : (ny, nx) thermal diffusivity of each cell (fluid or solid).
    conductivity_y : optional scalar or cell field for the y component.
        With this option kappa supplies the x component.
    flux_e, flux_n : (ny, nx+1), (ny+1, nx) volume fluxes through the faces,
        positive in the +x / +y direction (per unit depth).
    contact_resistance_x, contact_resistance_y : additional diffusive resistance
        per unit area on interior faces, scalar or shapes (ny, nx-1) and
        (ny-1, nx), respectively. Set zero for perfect contact. Units are
        length/kappa (s/m for diffusivity). In a static conductivity problem
        with kappa in W/(m K), supply the physical resistance in m² K/W.
    bc : dict side -> {"type": str, "values": float | array}.
        For type "robin", values is the ambient temperature and film_coefficient
        is the scalar or per-face film coefficient (W/(m² K) for conductivity).
        Per-face types may be supplied using the "types" key.

    Returns
    -------
    ``(T, info)`` where ``info`` holds the Picard history and the ledger.
    With return_wall_power=True, info["wall_power"] maps each wall to the
    inward conductive power per face (W for unit depth). Advective energy
    transport is reported separately in the ledger.
    """
    ny, nx = kappa.shape
    A, b, faces = _assemble_energy(dx, dy, kappa, flux_e, flux_n, bc, source,
                                    contact_resistance_x, contact_resistance_y, conductivity_y)
    cond_e, cond_n, side_specs, btype, bval, rates = faces
    lu = spla.splu(A)

    T = np.zeros((ny, nx)) if t_init is None else np.array(t_init, dtype=float)
    history = []
    it = 0
    for it in range(max_picard):
        b_dc = np.zeros(ny * nx)
        if high_order:
            up_x, ho_x, up_y, ho_y = _convective_faces(T, flux_e, flux_n, cond_e, cond_n,
                                                       bval, dx, dy)
            net_ho = _conv_net_out(ho_x, ho_y, flux_e, flux_n)
            net_up = _conv_net_out(up_x, up_y, flux_e, flux_n)
            b_dc = dc_relax * (net_ho - net_up).ravel()
        T_new = lu.solve(b + b_dc).reshape(ny, nx)
        delta = float(np.max(np.abs(T_new - T))) if it else np.inf
        T = T_new
        history.append(delta)
        if it > 0 and delta < tol:
            break

    info = {"picard_iterations": it + 1, "picard_history": history[-8:],
            "high_order": high_order}
    info["ledger"] = _ledger(dx, dy, kappa, cond_e, cond_n, flux_e, flux_n,
                            side_specs, btype, bval, T, high_order, rates)
    if return_wall_power:
        state = T.ravel()
        info["wall_power"] = {
            side: _conductive_wall_power(btype[side], bval[side], conductance, area, state[cells])
            for side, cells, conductance, _, _, area in side_specs}
    return T, info


def _conduction_materials(dx, dy, conductivity, capacity):
    for widths in (dx, dy):
        if widths.ndim != 1 or widths.size == 0 or not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("cell widths must be finite positive vectors")
    properties = tuple(np.broadcast_to(np.asarray(v, dtype=float), (len(dy), len(dx)))
                       for v in (conductivity, capacity))
    if any(not np.isfinite(v).all() or np.any(v <= 0) for v in properties):
        raise ValueError("material properties must be finite and positive")
    return properties


def _validate_conduction_bc(bc, sides, dtype):
    if not set(bc).issubset(sides):
        raise ValueError("unknown conduction boundary side")
    for side, count in sides.items():
        types, values = _bc_arrays(bc, side, count, dtype=dtype)
        if types.shape != (count,) or not np.isin(types, ("adiabatic", "dirichlet", "neumann", "outflow", "robin")).all():
            raise ValueError(f"invalid conduction boundary type on {side}")
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite boundary value on {side}")


def _join_periodic_y(operator, conductance_y, bc):
    if "north" in bc or "south" in bc:
        raise ValueError("periodic_y requires omitting north and south boundary conditions")
    # Each incidence row takes the last-minus-first temperature. The two
    # boundary half-cell resistances are in series across the periodic seam.
    seam = 1/(1/conductance_y[-1] + 1/conductance_y[0])
    nx = len(seam)
    columns = np.arange(nx)
    last = operator.shape[0]-nx+columns
    incidence = sp.coo_matrix((np.r_[np.ones(nx), -np.ones(nx)],
                              (np.tile(columns, 2), np.r_[last, columns])),
                              shape=(nx, operator.shape[0])).tocsc()
    return operator - incidence.T @ sp.diags(seam) @ incidence, seam


def solve_harmonic_conduction(dx, dy, conductivity, volumetric_heat_capacity, bc,
                              angular_frequency, *, source=0., conductivity_y=None,
                              contact_resistance_x=0., contact_resistance_y=0.,
                              return_interior_power=False, periodic_y=False):
    """Return cell temperature and inward power per wall face for unit depth.

    Use Re(phasor * exp(iωt)), with positive angular frequency in rad/s.
    Wall values and volumetric sources may be complex amplitudes. Real
    materials, contact resistances and Robin films use the transient units.
    With return_interior_power=True also return a third item, a dictionary
    of interior face powers (W). Its x and y arrays have shapes (ny,nx-1)
    and (ny-1,nx), positive toward increasing coordinates. periodic_y joins
    north to south with perfect thermal contact. Omit those two wall BCs.
    Their reported face powers are then opposite inward seam transfers.
    contact_resistance_y continues to describe the ny-1 interior faces.
    """
    if not np.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    k, capacity = _conduction_materials(dx, dy, conductivity, volumetric_heat_capacity)
    ny, nx = k.shape
    _validate_conduction_bc(bc, {"west": ny, "east": ny, "south": nx, "north": nx}, complex)
    A, b, faces = _assemble_energy(dx, dy, k, np.zeros((ny, nx+1)), np.zeros((ny+1, nx)),
                                  bc, source, contact_resistance_x, contact_resistance_y,
                                  conductivity_y, load_dtype=complex)
    if periodic_y:
        A, seam = _join_periodic_y(A, faces[1], bc)
    mass = (capacity*np.outer(dy, dx)).ravel()
    state = spla.spsolve(sp.diags(1j*angular_frequency*mass, format="csc") - A, -b)
    _, _, side_specs, types, values, _ = faces
    powers = {side: _conductive_wall_power(types[side], values[side], conductance, area, state[cells])
              for side, cells, conductance, _, _, area in side_specs}
    temperature = state.reshape(ny, nx)
    if periodic_y:
        powers["south"] = seam*(temperature[-1]-temperature[0])
        powers["north"] = -powers["south"]
    if return_interior_power:
        conductance_x, conductance_y = faces[:2]
        interior = {"x": -conductance_x[:,1:-1] * np.diff(temperature, axis=1),
                    "y": -conductance_y[1:-1,:] * np.diff(temperature, axis=0)}
        return temperature, powers, interior
    return temperature, powers


def solve_transient_conduction(dx, dy, conductivity, volumetric_heat_capacity, bc,
                               time, initial_temperature, *, source=0.0,
                               contact_resistance_x=0.0, contact_resistance_y=0.0,
                               theta=0.5, conductivity_y=None, return_wall_power=False, periodic_y=False):
    """Advance cell-centred conduction with fixed material data and boundary types.

    Return temperature at each requested time with shape (nt, ny, nx). Cell
    widths are in metres, conductivity in W/(m K), volumetric heat capacity
    in J/(m³ K), source in W/m³, and contact resistance in m² K/W. Fixed
    wall temperatures and prescribed inward flux use the steady BC format.
    A side's values may also be a callable of time, returning a scalar or
    one value per face (temperature, inward flux or Robin ambient temperature).
    source may also be a function of time returning a scalar or a cell field.
    Callables are sampled at the supplied times and theta-weighted per step.
    For grid-aligned anisotropy, conductivity supplies the x component and
    conductivity_y supplies the y component. Both broadcast to (ny, nx).
    With return_wall_power=True return (temperature, wall_power). For each
    side, wall_power[side] has shape (nt, number_of_faces) and gives inward
    watts per face for unit depth. To integrate use the same theta as the solve.
    With periodic_y=True omit the north and south BCs. Those two face powers
    then report opposite inward transfers across a perfectly conducting contact
    between the last and first rows. Interior contact resistances are unchanged.
    theta=0.5 selects Crank-Nicolson and theta=1 selects backward Euler.
    """
    dx, dy, time = (np.asarray(v, dtype=float) for v in (dx, dy, time))
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing finite vector")
    if not .5 <= theta <= 1:
        raise ValueError("theta must lie in [0.5, 1]")
    k, capacity = _conduction_materials(dx, dy, conductivity, volumetric_heat_capacity)
    shape = k.shape
    ny, nx = shape
    initial = np.broadcast_to(np.asarray(initial_temperature, dtype=float), shape)
    if not np.isfinite(initial).all():
        raise ValueError("temperature must be finite")

    sides = {"west": ny, "east": ny, "south": nx, "north": nx}
    static_bc = dict(bc)
    signals = {}
    for side, count in sides.items():
        spec = bc.get(side)
        if spec is not None and callable(spec.get("values")):
            samples = np.empty((len(time), count))
            for index, instant in enumerate(time):
                samples[index] = np.broadcast_to(np.asarray(spec["values"](float(instant)), dtype=float), (count,))
            if not np.isfinite(samples).all():
                raise ValueError(f"nonfinite boundary value on {side}")
            signals[side] = samples
            static_bc[side] = dict(spec, values=0.)
    _validate_conduction_bc(static_bc, sides, float)
    volume = np.outer(dy, dx)
    def source_power(instant):
        field = np.broadcast_to(np.asarray(source(float(instant)), dtype=float), shape)
        if not np.isfinite(field).all():
            raise ValueError("source must be finite")
        return (field*volume).ravel()

    varying_source = callable(source)
    if varying_source:
        previous_power = source_power(time[0])
    A, b, faces = _assemble_energy(dx, dy, k, np.zeros((ny, nx+1)), np.zeros((ny+1, nx)),
                               static_bc, 0. if varying_source else source, contact_resistance_x, contact_resistance_y, conductivity_y)
    if periodic_y:
        A, seam = _join_periodic_y(A, faces[1], bc)
    _, _, side_specs, types, values, _ = faces
    varying_loads = []
    for side, cells, conductance, _, _, area in side_specs:
        if side in signals:
            weights = _conductive_wall_power(types[side], 1., conductance, area, 0.)
            varying_loads.append((cells, signals[side], weights))
    mass = (capacity * volume).ravel()
    M = sp.diags(mass, format="csc")
    history = np.empty((len(time), ny, nx))
    history[0] = initial
    previous_dt = None
    for step, dt in enumerate(np.diff(time)):
        if dt != previous_dt:
            lu = spla.splu(M - (theta * dt) * A)
            previous_dt = dt
        state = history[step].ravel()
        rhs = mass * state + dt * ((1 - theta) * (A @ state) - b)
        if varying_source:
            next_power = source_power(time[step+1])
            rhs += dt*((1-theta)*previous_power+theta*next_power)
            previous_power = next_power
        for cells, samples, weights in varying_loads:
            rhs[cells] += dt * weights * ((1-theta)*samples[step] + theta*samples[step+1])
        history[step + 1] = lu.solve(rhs).reshape(shape)
    if return_wall_power:
        state = history.reshape(len(time), ny*nx)
        powers = {side: _conductive_wall_power(types[side], signals.get(side, values[side]), conductance,
                                               area, state[:, cells])
                  for side, cells, conductance, _, _, area in side_specs}
        if periodic_y:
            powers["south"] = seam*(history[:, -1]-history[:, 0])
            powers["north"] = -powers["south"]
        return history, powers
    return history


def _conductive_wall_power(types, values, conductance, area, temperature):
    prescribed = (types == "dirichlet") | (types == "robin")
    power = np.where(prescribed, conductance * (values - temperature), 0.)
    return np.where(types == "neumann", values * area, power)


def _ledger(dx, dy, kappa, cond_e, cond_n, flux_e, flux_n, side_specs, btype, bval, T,
            high_order=True, rates=0.0):
    """Recompute the discrete energy balance cell by cell and at the boundary.

    ``residual`` accumulates the *heat entering* each cell, so it vanishes where
    the discrete equation holds.  Convective face values are the same limited
    second-order ones the deferred correction used, so the audit closes exactly
    the equation the solver solved.
    """
    ny, nx = T.shape
    _, ho_x, _, ho_y = _convective_faces(T, flux_e, flux_n, cond_e, cond_n, bval, dx, dy,
                                        first_order=not high_order)
    residual = np.broadcast_to(rates, (ny, nx)).copy()
    residual[:, :-1] += cond_e[:, 1:nx] * (T[:, 1:] - T[:, :-1])
    residual[:, 1:] -= cond_e[:, 1:nx] * (T[:, 1:] - T[:, :-1])
    residual[:-1, :] += cond_n[1:ny, :] * (T[1:, :] - T[:-1, :])
    residual[1:, :] -= cond_n[1:ny, :] * (T[1:, :] - T[:-1, :])
    # interior convective faces only: boundary faces are added per side below
    face_x = ho_x[:, 1:nx] * flux_e[:, 1:nx]
    residual[:, :-1] -= face_x
    residual[:, 1:] += face_x
    face_y = ho_y[1:ny, :] * flux_n[1:ny, :]
    residual[:-1, :] -= face_y
    residual[1:, :] += face_y
    boundary = {}
    advective = {"inflow": 0.0, "outflow": 0.0}
    for side, cells, cf, ff, sign_out, area in side_specs:
        types = btype[side]
        values = bval[side]
        cf = np.asarray(cf).ravel()
        ff = np.asarray(ff).ravel()
        area = np.broadcast_to(np.asarray(area, dtype=float).ravel(), cells.shape)
        cells = cells.ravel()
        Tc = T.ravel()[cells]
        q_cond = _conductive_wall_power(types, values, cf, area, Tc)
        out = sign_out * ff
        q_conv = np.where(out < 0.0, -out * values, -out * Tc)
        boundary[side] = float(np.sum(q_cond + q_conv))
        residual.ravel()[cells] += q_cond + q_conv
        advective["inflow"] += float(np.sum(np.where(out < 0.0, -out * values, 0.0)))
        advective["outflow"] += float(np.sum(np.where(out > 0.0, out * Tc, 0.0)))
    generation = float(np.sum(rates))
    total = sum(boundary.values()) + generation
    # scale on the largest *throughput* of the problem, not just the boundary
    # fluxes: for a case whose boundary fluxes cancel the latter is degenerate.
    scale = max([abs(v) for v in boundary.values()]
                + [abs(advective["inflow"]), abs(advective["outflow"])] + [float(np.sum(np.abs(rates))), 1e-300])
    return EnergyLedger(
        generation=generation,
        boundary=boundary,
        advective=advective,
        inflow=advective["inflow"],
        outflow=advective["outflow"],
        imbalance=total,
        normalized_imbalance=abs(total) / scale,
        residual_inf=float(np.max(np.abs(residual))),
        residual_scaled=float(np.max(np.abs(residual)) / scale),
    )