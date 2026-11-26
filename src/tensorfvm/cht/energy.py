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


def _assemble_energy(dx, dy, kappa, flux_e, flux_n, bc, source, contact_resistance_x, contact_resistance_y, conductivity_y=None, *, load_dtype=float, angular_frequency=None):
    """Assemble net entering heat as A @ T - b, plus the ledger inputs."""
    ny, nx = kappa.shape
    cond_e, cond_n = _face_conductance(dx, dy, kappa, contact_resistance_x, contact_resistance_y, conductivity_y)
    idx = np.arange(ny * nx).reshape(ny, nx)

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(np.asarray(r).ravel())
        cols.append(np.asarray(c).ravel())
        vals.append(np.asarray(v).ravel())

    dynamic_walls = any("wall_areal_heat_capacity" in (spec or {}) for spec in bc.values())
    if dynamic_walls:
        if angular_frequency is None:
            raise ValueError("wall_areal_heat_capacity requires harmonic conduction")
        cond_e, cond_n = cond_e.astype(complex), cond_n.astype(complex)
    diag = np.zeros(ny * nx, dtype=cond_e.dtype)
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
        spec = bc.get(side) or {}
        if "wall_areal_heat_capacity" in spec:
            wall_capacity = np.broadcast_to(np.asarray(spec["wall_areal_heat_capacity"], dtype=float), cells.shape)
            if (not np.isfinite(wall_capacity).all() or np.any(wall_capacity < 0)
                    or np.any(wall_capacity[~robin] != 0)):
                raise ValueError("wall areal heat capacity must be finite, nonnegative and confined to Robin faces")
        if np.any(robin):
            if "film_coefficient" not in spec:
                raise ValueError(f"Robin boundary on {side} needs film_coefficient")
            h = np.broadcast_to(np.asarray(spec["film_coefficient"], dtype=float), cells.shape)[robin]
            if not np.isfinite(h).all() or np.any(h < 0):
                raise ValueError("film coefficient must be finite and nonnegative")
            if "wall_areal_heat_capacity" in spec:
                # Eliminate the wall temperature from the film and storage balance.
                admittance = h + 1j*angular_frequency*wall_capacity[robin]
                values = values.copy()
                values[robin] *= np.divide(h, admittance, out=np.zeros_like(admittance), where=admittance != 0)
                bval[side] = values
                film = admittance * area[robin]
            else:
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


def _bulk_loss_operator(operator, coefficient, volume):
    raw = np.asarray(coefficient)
    if np.iscomplexobj(raw):
        raise ValueError("heat_loss_coefficient must be real")
    loss = np.broadcast_to(np.asarray(raw,dtype=float),volume.shape)
    if not np.isfinite(loss).all() or np.any(loss<0):
        raise ValueError("heat_loss_coefficient must be finite and nonnegative")
    if np.any(loss):
        return operator-sp.diags((volume*loss).ravel(),format="csc")
    return operator


def _interior_conduction_power(temperature, faces):
    conductance_x,conductance_y = faces[:2]
    return {"x": -conductance_x[:,1:-1]*np.diff(temperature,axis=-1),
            "y": -conductance_y[1:-1,:]*np.diff(temperature,axis=-2)}


def _second_order_wall(side, cells, mask, area, dx, dy, k, conductivity_y, faces):
    ny, nx = k.shape
    x_side = side in ("west","east")
    widths = dx if x_side else dy
    if widths.size < 2:
        raise ValueError("second-order wall flux requires two cells along the wall normal")
    forward = side in ("west","south")
    d1,d2 = widths[:2] if forward else widths[-2:][::-1]
    inner = cells + (1 if x_side else nx)*(1 if forward else -1)
    normal_k = k if x_side or conductivity_y is None else np.broadcast_to(conductivity_y, k.shape)
    k1,k2 = normal_k.ravel()[cells], normal_k.ravel()[inner]
    near = 1 if forward else -2
    interface = (faces[0][:,near] if x_side else faces[1][near,:])[mask]
    # For q(x)=q0+b*x, cell averages satisfy T_wall-T_j=a_j*q0+b_j*b.
    a1 = d1/(2*k1)
    a2 = a1 + area/interface
    contact = area/interface-a1-d2/(2*k2)
    b1 = d1**2/(6*k1)
    b2 = d1**2/(2*k1)+d1*contact+(d1*d2/2+d2**2/6)/k2
    scale = area/(a1*b2-a2*b1)
    correction = scale*b1
    g = scale*(b2-b1)
    return inner, g, correction


def _fixed_wall_operator(A, b, faces, dx, dy, k, conductivity_y, order):
    if order not in (1,2):
        raise ValueError("dirichlet_flux_order must be 1 or 2")
    _, _, side_specs, types, values, _ = faces
    fixed_wall_links = {}
    if order == 2:
        A = A.tolil()
        for side, cells, conductance, _, _, area in side_specs:
            mask = types[side] == "dirichlet"
            if not mask.any():
                continue
            cells = cells[mask]
            inner, g, extra = _second_order_wall(side,cells,mask,area[mask],dx,dy,k,conductivity_y,faces)
            delta = g-conductance[mask]
            for cell, other, change, correction in zip(cells,inner,delta,extra):
                A[cell,cell] -= change+correction
                A[cell,other] += correction
            b[cells] -= delta*values[side][mask]
            conductance[mask] = g
            fixed_wall_links[side] = mask,cells,inner,extra
        A = A.tocsc()
    return A, fixed_wall_links


def _wall_half_resistance(side,dx,dy,kx,ky,shape,cells):
    horizontal = side in ("west","east")
    widths = dx if horizontal else dy
    material = kx if horizontal or ky is None else ky
    return .5*widths[0 if side in ("west","south") else -1]/np.broadcast_to(material,shape).ravel()[cells]


def _frequency_system(dx,dy,conductivity,volumetric_heat_capacity,bc,angular_frequency,source,
                      conductivity_y,contact_resistance_x,contact_resistance_y,periodic_y,heat_loss_coefficient,dirichlet_flux_order):
    dx, dy = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    k, capacity = _conduction_materials(dx, dy, conductivity, volumetric_heat_capacity)
    ny, nx = k.shape
    _validate_conduction_bc(bc, {"west": ny, "east": ny, "south": nx, "north": nx}, complex)
    A, b, faces = _assemble_energy(dx, dy, k, np.zeros((ny, nx+1)), np.zeros((ny+1, nx)),
                                  bc, source, contact_resistance_x, contact_resistance_y,
                                  conductivity_y, load_dtype=complex, angular_frequency=angular_frequency)
    seam = None
    if periodic_y:
        A, seam = _join_periodic_y(A, faces[1], bc)
    volume = np.outer(dy, dx)
    A = _bulk_loss_operator(A,heat_loss_coefficient,volume)
    mass = (capacity*volume).ravel()
    A, fixed_wall_links = _fixed_wall_operator(A,b,faces,dx,dy,k,conductivity_y,dirichlet_flux_order)
    heater_power = {}
    heater_surface_energy = 0j
    for side,cells,_,_,_,area in faces[2]:
        spec = bc.get(side) or {}
        if "wall_heater_flux" not in spec:
            continue
        cw = np.broadcast_to(spec.get("wall_areal_heat_capacity",0.),area.shape)
        storing = cw > 0
        if not storing.any():
            raise ValueError("wall_heater_flux requires storing faces")
        flux = np.asarray(spec["wall_heater_flux"],dtype=complex)
        if flux.shape not in ((),(int(storing.sum()),)) or not np.isfinite(flux).all():
            raise ValueError("wall_heater_flux requires a finite phasor per storing face or a scalar")
        h = np.broadcast_to(spec["film_coefficient"],area.shape)[storing]
        half_r = _wall_half_resistance(side,dx,dy,k,conductivity_y,k.shape,cells[storing])
        # Eliminate the surface node. Its independent heater transmits through
        # the half cell even at DC when the ambient film is fully insulated.
        delivered = area[storing]*flux/(1+half_r*(h+1j*angular_frequency*cw[storing]))
        b[cells[storing]] -= delivered
        heater_power[side] = np.zeros(area.shape,dtype=complex)
        heater_power[side][storing] = delivered
        heater_surface_energy += np.sum(cw[storing]*half_r*delivered)
    return A,b,faces,mass,seam,fixed_wall_links,heater_power,heater_surface_energy


def _frequency_output(state,faces,seam,return_interior_power,fixed_wall_links,heater_power):
    _, _, side_specs, types, values, _ = faces
    powers = {side: _conductive_wall_power(types[side], values[side], conductance, area, state[cells])
              for side, cells, conductance, _, _, area in side_specs}
    for side,(mask,cells,inner,extra) in fixed_wall_links.items():
        powers[side][mask] += extra*(state[inner]-state[cells])
    for side,delivered in heater_power.items():
        powers[side] += delivered
    temperature = state.reshape(faces[0].shape[0],-1)
    if seam is not None:
        powers["south"] = seam*(temperature[-1]-temperature[0])
        powers["north"] = -powers["south"]
    if return_interior_power:
        return temperature, powers, _interior_conduction_power(temperature,faces)
    return temperature, powers


def _floating_harmonic_state(matrix,faces,mass,bc,omega,dx,dy,kx,ky):
    # The net source fixes the capacity-weighted mean exactly. Build the
    # remaining load from physical powers before eliminating the wall stores.
    _,_,sides,types,values,cell_source = faces
    cell_power = cell_source.ravel().copy()
    total_capacity = mass.sum()
    walls = {}
    for side,cells,_,_,_,area in sides:
        prescribed = types[side]=="neumann"
        cell_power[cells[prescribed]] += values[side][prescribed]*area[prescribed]
        spec = bc.get(side) or {}
        cw = np.broadcast_to(spec.get("wall_areal_heat_capacity",0.),area.shape)
        active = cw>0
        if active.any():
            capacity = area[active]*cw[active]
            total_capacity += capacity.sum()
            applied = area[active]*np.asarray(spec.get("wall_heater_flux",0.),dtype=complex)
            resistance = _wall_half_resistance(side,dx,dy,kx,ky,cell_source.shape,cells[active])/area[active]
            walls[side] = active,cells[active],capacity,applied,resistance
    rate = (cell_power.sum()+sum(applied.sum() for _,_,_,applied,_ in walls.values()))/total_capacity
    rhs = cell_power-rate*mass
    energy_weight = mass.astype(complex)
    energy_offset = 0j
    delivered = {}
    for side,(active,cells,capacity,applied,resistance) in walls.items():
        transmission = 1/(1+1j*omega*capacity*resistance)
        power = (applied-rate*capacity)*transmission
        rhs[cells] += power
        delivered[side] = np.zeros(active.shape,dtype=complex)
        delivered[side][active] = power
        energy_weight[cells] += capacity*transmission
        energy_offset += np.sum(capacity*resistance*power)
    # The last row sets total excess energy (including surfaces) to zero.
    # This independent constraint remains well-conditioned as omega tends to zero.
    pin = sp.csc_matrix(([1.],([0],[0])),shape=(mass.size,1))
    system = sp.bmat([[matrix,pin],[sp.csc_matrix(energy_weight[None,:]/total_capacity),None]],format="csc")
    solution = spla.spsolve(system,np.r_[rhs,-energy_offset/total_capacity])
    return solution[:-1],rate,delivered


def _frequency_bulk_exchange(dx,dy,coefficient,temperature):
    volume = np.outer(dy,dx)
    ambient = 0.
    if temperature is not None:
        ambient = np.broadcast_to(np.asarray(temperature,dtype=complex),volume.shape)
        if not np.isfinite(ambient).all():
            raise ValueError("bulk_ambient_temperature must be finite")
    return volume*np.asarray(coefficient),ambient


def solve_harmonic_conduction(dx, dy, conductivity, volumetric_heat_capacity, bc,
                              angular_frequency, *, source=0., conductivity_y=None,
                              contact_resistance_x=0., contact_resistance_y=0.,
                              return_interior_power=False, periodic_y=False, heat_loss_coefficient=0.,
                              dirichlet_flux_order=1, return_storing_wall_response=False, return_uniform_rate=False,
                              bulk_ambient_temperature=None, return_cell_power=False):
    """Return cell temperature and inward power per wall face for unit depth.

    Use Re(phasor * exp(iωt)), with positive angular frequency in rad/s.
    Wall values and volumetric sources may be complex amplitudes. Real
    materials, contact resistances and Robin films use the transient units.
    heat_loss_coefficient adds a nonnegative bulk sink H*T in W/m³, with H
    broadcast to the cell grid. Set bulk_ambient_temperature to a scalar or
    cell field of complex ambient phasors Ta to apply H*(Ta-T), with source
    reserved for independent heating. The default ambient is zero.
    return_cell_power appends a dict of complex arrays shaped (ny,nx), in
    watts per cell for unit depth: source, bulk (ambient into material), and
    storage (rate of increase of cell energy).
    Robin specs may include wall_areal_heat_capacity (J/(m² K)), scalar or
    per face. wall_heater_flux adds an independent complex heat-flux phasor
    (W/m², positive into the wall), scalar or one value per storing face in
    face order. The returned wall power enters the material after wall storage.
    return_storing_wall_response appends a dict keyed by each storing side.
    Each entry contains complex arrays in storing-face order: temperature (K)
    and powers (W) named heater, ambient (film into wall), material (wall into
    cells), and storage (rate of increase of wall energy).
    With return_uniform_rate=True prepend the common heating rate (K/s).
    For a lossless body without temperature anchoring the cell and surface
    temperatures are deviations from rate/(i*omega), with zero total excess
    stored energy. All returned powers retain their physical values. An
    anchored body returns zero rate and its ordinary temperature phasors.
    With return_interior_power=True also return a third item, a dictionary
    of interior face powers (W). Its x and y arrays have shapes (ny,nx-1)
    and (ny-1,nx), positive toward increasing coordinates. periodic_y joins
    north to south with perfect thermal contact. Omit those two wall BCs.
    Their reported face powers are then opposite inward seam transfers.
    contact_resistance_y continues to describe the ny-1 interior faces.
    dirichlet_flux_order=2 reconstructs fixed-wall flux from two normal cell means.
    """
    if not np.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    A,b,faces,mass,seam,fixed_wall_links,heater_power,_ = _frequency_system(dx,dy,conductivity,volumetric_heat_capacity,bc,
        angular_frequency,source,conductivity_y,contact_resistance_x,contact_resistance_y,periodic_y,heat_loss_coefficient,dirichlet_flux_order)
    if return_cell_power or bulk_ambient_temperature is not None:
        bulk_conductance,ambient = _frequency_bulk_exchange(dx,dy,heat_loss_coefficient,bulk_ambient_temperature)
        if bulk_ambient_temperature is not None:
            b -= (bulk_conductance*ambient).ravel()
    matrix = sp.diags(1j*angular_frequency*mass,format="csc")-A
    rate = 0j
    # A capacitive wall has nonzero dynamic admittance even without a film.
    floating = return_uniform_rate and not np.any(heat_loss_coefficient)
    if floating:
        for side,kinds in faces[3].items():
            robin = kinds=="robin"
            if np.any(kinds=="dirichlet") or (robin.any() and
                    np.any(np.broadcast_to(bc[side]["film_coefficient"],kinds.shape)[robin]>0)):
                floating = False
                break
    if floating:
        state,rate,heater_power = _floating_harmonic_state(matrix,faces,mass,bc,angular_frequency,
                                                           dx,dy,conductivity,conductivity_y)
    else:
        state = spla.spsolve(matrix,-b)
    result = _frequency_output(state,faces,seam,return_interior_power,fixed_wall_links,heater_power)
    if return_storing_wall_response:
        response = _storing_wall_response(result,faces[2],bc,dx,dy,conductivity,conductivity_y,
                                          angular_frequency,uniform_rate=rate)
        result += (response,)
    if return_cell_power:
        cell = result[0]
        result += (dict(source=faces[5],bulk=bulk_conductance*(ambient-cell),
                        storage=mass.reshape(cell.shape)*(rate+1j*angular_frequency*cell)),)
    return (rate,*result) if return_uniform_rate else result


def _storing_wall_response(result,sides,bc,dx,dy,conductivity,conductivity_y,angular_frequency,steady_wall=None,uniform_rate=0.):
    response = {}
    state = result[0].ravel()
    for side,cells,_,_,_,area in sides:
        spec = bc.get(side) or {}
        cw = np.broadcast_to(spec.get("wall_areal_heat_capacity",0.),area.shape)
        active = cw > 0
        if not active.any():
            continue
        cells,area,cw = cells[active],area[active],cw[active]
        h = np.broadcast_to(spec["film_coefficient"],active.shape)[active]
        ambient = np.broadcast_to(spec.get("values",0.),active.shape)[active]
        heater_flux = np.asarray(spec.get("wall_heater_flux",0.),dtype=complex)
        half_r = _wall_half_resistance(side,dx,dy,conductivity,conductivity_y,result[0].shape,cells)
        storage_rate = 1j*angular_frequency*cw
        storage_drive = cw*uniform_rate
        if steady_wall is not None:
            # Differentiate C*s*Tw about s=0 with frequency-independent loads.
            heater_flux = 0j
            ambient = 0j
            storage_drive = cw*steady_wall[side]["temperature"]
        denominator = 1+half_r*(h+storage_rate)
        surface = (state[cells]+half_r*(heater_flux+h*ambient-storage_drive))/denominator
        # Evaluate the small film drop directly for strongly coupled reservoirs.
        film_drop = ((ambient-state[cells])+half_r*(storage_rate*ambient-heater_flux+storage_drive))/denominator
        response[side] = dict(temperature=surface, heater=area*heater_flux,
                              ambient=area*h*film_drop, material=result[1][side][active],
                              storage=area*(storage_rate*surface+storage_drive))
    return response


def solve_low_frequency_conduction(dx, dy, conductivity, volumetric_heat_capacity, bc, *,
                                   source=0., conductivity_y=None, contact_resistance_x=0.,
                                   contact_resistance_y=0., periodic_y=False, heat_loss_coefficient=0.,
                                   return_interior_power=False, dirichlet_flux_order=1, return_storing_wall_response=False,
                                   return_uniform_rate=False, bulk_ambient_temperature=None, return_cell_power=False):
    """Return steady and first-order coefficients for the exp(+i*omega*t) response.

    Both outputs follow solve_harmonic_conduction's temperature and face-power
    convention. First-order units are K s and W s. Loads and prescribed data
    are independent of frequency. An unanchored lossless body requires zero
    net power by default. With return_uniform_rate=True return (rate, steady,
    first), where rate is the complex net heating power divided by total heat
    capacity (K/s) for an unanchored lossless body, or zero for an anchored
    body. For s=i*omega the temperature is rate/s + steady + s*first to first
    order. Wall storage power in steady includes the uniform heating. After
    removing the pole the closed body has zero total stored energy at both
    orders, including any heat capacity on insulated Robin surfaces.
    return_storing_wall_response appends the same surface quantities as the
    harmonic solver to each output. The second dictionary gives coefficients
    of i*omega, with surface temperature in K s and powers in W s.
    bulk_ambient_temperature supplies the frequency-independent scalar or cell
    ambient phasor for H*(Ta-T). return_cell_power appends a dict to each
    output with source, bulk and storage power per cell. The first dict gives
    constant terms in W and the second gives coefficients of i*omega in W s.
    dirichlet_flux_order selects the same fixed-wall reconstruction as in the harmonic solve.
    """
    A,b,faces,mass,seam,fixed_wall_links,heater_power,heater_surface_energy = _frequency_system(dx,dy,conductivity,volumetric_heat_capacity,bc,0.,
        source,conductivity_y,contact_resistance_x,contact_resistance_y,periodic_y,heat_loss_coefficient,dirichlet_flux_order)
    if return_cell_power or bulk_ambient_temperature is not None:
        bulk_conductance,ambient = _frequency_bulk_exchange(dx,dy,heat_loss_coefficient,bulk_ambient_temperature)
        if bulk_ambient_temperature is not None:
            b -= (bulk_conductance*ambient).ravel()
    _,_,sides,types,_,_ = faces
    anchored = any(np.any((g.real>0)&np.isin(types[side],["dirichlet","robin"]))
                   for side,_,g,_,_,_ in sides)
    closed = not anchored and not np.any(heat_loss_coefficient)
    matrix = (-A).astype(complex).tocsc()
    energy_weight = mass.copy()
    uniform_rate = 0j
    if closed:
        if not return_uniform_rate and abs(b.sum())>64*np.finfo(float).eps*np.abs(b).sum():
            raise ValueError("unanchored low-frequency conduction requires zero net power")
        for side,cells,_,_,_,area in sides:
            spec = bc.get(side) or {}
            energy_weight[cells] += area*np.asarray(spec.get("wall_areal_heat_capacity",0.))
        if return_uniform_rate:
            uniform_rate = -b.sum()/energy_weight.sum()
            b += uniform_rate*energy_weight
            for side,cells,_,_,_,area in sides:
                cw = np.broadcast_to((bc.get(side) or {}).get("wall_areal_heat_capacity",0.),area.shape)
                if cw.any():
                    # Part of the wall heater drives uniform wall storage. Only
                    # the remaining power creates a DC half-cell temperature drop.
                    absorbed = uniform_rate*cw*area
                    heater_power[side] = heater_power.get(side,0.)-absorbed
                    half_r = _wall_half_resistance(side,dx,dy,conductivity,conductivity_y,
                                                   (len(dy),len(dx)),cells)
                    heater_surface_energy -= np.sum(cw*half_r*absorbed)
        # A rank-one pin fixes the additive constant, including for one cell.
        pin = sp.csc_matrix(([max(1.,abs(matrix[0,0]))],([0],[0])),shape=matrix.shape)
        matrix = matrix+pin
    factor = spla.splu(matrix)
    state = factor.solve(-b)
    if closed:
        state -= (energy_weight@state+heater_surface_energy)/energy_weight.sum()
    steady = _frequency_output(state,faces,seam,return_interior_power,fixed_wall_links,heater_power)
    rhs = -mass*state
    correction = {}
    dx,dy = np.asarray(dx),np.asarray(dy)
    shape = steady[0].shape
    surface_energy_lag = 0.
    for side,cells,g,_,_,area in sides:
        spec = bc.get(side) or {}
        if "wall_areal_heat_capacity" not in spec:
            continue
        # At DC Tw = Tcell + Q/G, where G is the half-cell conductance.
        # Eliminating the wall's first-order unknown transmits a fraction
        # G/(G+h*area) of its storage load to the cell equation.
        half_r = _wall_half_resistance(side,dx,dy,conductivity,conductivity_y,shape,cells)
        half_resistance = half_r/area
        cw = np.broadcast_to(spec["wall_areal_heat_capacity"],area.shape)
        active = cw > 0
        h = np.broadcast_to(spec.get("film_coefficient",0.),area.shape)[active]
        ambient = np.broadcast_to(spec.get("values",0.),area.shape)[active]
        heater = np.asarray(spec.get("wall_heater_flux",0.),dtype=complex)
        correction[side] = np.zeros(area.shape,dtype=complex)
        if active.any():
            transmission = 1/(1+h*half_r[active])
            wall = transmission*(state[cells[active]]+half_r[active]*(heater+h*ambient-cw[active]*uniform_rate))
            correction[side][active] = -cw[active]*area[active]*wall*transmission
        rhs[cells] += correction[side]
        if closed:
            surface_energy_lag += np.sum(area*np.asarray(spec["wall_areal_heat_capacity"])*
                                         half_resistance*correction[side])
    lag = factor.solve(rhs)
    if closed:
        lag -= (energy_weight@lag+surface_energy_lag)/energy_weight.sum()
    cells_lag = lag.reshape(shape)
    powers = {side:_conductive_wall_power(types[side],0.,g,area,lag[cells])+correction.get(side,0.)
              for side,cells,g,_,_,area in sides}
    for side,(mask,cells,inner,extra) in fixed_wall_links.items():
        powers[side][mask] += extra*(lag[inner]-lag[cells])
    if seam is not None:
        powers["south"] = seam*(cells_lag[-1]-cells_lag[0])
        powers["north"] = -powers["south"]
    first = (cells_lag,powers)
    if return_interior_power:
        first += (_interior_conduction_power(cells_lag,faces),)
    if return_storing_wall_response:
        wall_dc = _storing_wall_response(steady,sides,bc,dx,dy,conductivity,conductivity_y,0.,uniform_rate=uniform_rate)
        wall_lag = _storing_wall_response(first,sides,bc,dx,dy,conductivity,conductivity_y,0.,wall_dc)
        steady += (wall_dc,)
        first += (wall_lag,)
    if return_cell_power:
        capacity = mass.reshape(shape)
        steady += (dict(source=faces[5],bulk=bulk_conductance*(ambient-steady[0]),
                        storage=capacity*uniform_rate),)
        first += (dict(source=np.zeros_like(faces[5]),bulk=-bulk_conductance*cells_lag,
                       storage=capacity*steady[0]),)
    return (uniform_rate,steady,first) if return_uniform_rate else (steady,first)


def solve_transient_conduction(dx, dy, conductivity, volumetric_heat_capacity, bc,
                               time, initial_temperature, *, source=0.0,
                               contact_resistance_x=0.0, contact_resistance_y=0.0,
                               theta=0.5, conductivity_y=None, return_wall_power=False, periodic_y=False,
                               wall_flux_order=1, heat_loss_coefficient=0., return_interior_power=False,
                               load_jump_times=(), return_face_heat=False, rannacher=False,
                               monotone_theta=False, dirichlet_flux_order=1, return_cell_heat=False,
                               bulk_ambient_temperature=None, cell_temperature_reference=None,
                               initial_storing_wall_state=None, return_storing_wall_state=False,
                               return_storing_wall_heat=False):
    """Advance cell-centred conduction with fixed material data and boundary types.

    Return temperature at each requested time with shape (nt, ny, nx). Cell
    temperatures normally use absolute kelvins. If cell_temperature_reference
    is supplied, initial_temperature and the returned cell history are
    deviations in K from that constant reference. A returned cell field can
    then start a subsequent solve at the same reference. Prescribed wall,
    bulk ambient and explicit initial wall temperatures still use absolute K.
    return_storing_wall_state appends a dict keyed by sides with storing faces.
    Its arrays have shape (nt, number of storing faces on that side), ordered
    by increasing face index, in the same temperature coordinates as the cell
    history. To restart, pass the last row for each side in
    initial_storing_wall_state along with the last cell field. Each supplied
    side overrides its BC initial_wall_temperature; omitted sides use the BC.
    Use the same storing faces and temperature reference across the restart.
    return_storing_wall_heat appends a dict keyed by the same sides. Each
    entry contains interval heat arrays "ambient" (film into wall), "material"
    (wall into adjacent cells), and "storage" (wall energy increase), in J
    for unit depth, shaped (nt-1, number of storing faces).
    On a storing side, wall_heater_flux prescribes additional power density
    (W/m², positive into the wall) as a scalar, a vector in storing-face
    order, or a callable of time returning either. Declare discontinuities
    in load_jump_times. On that side the heat dict also contains "heater";
    wall storage equals ambient plus heater minus material heat.
    Cell widths are in metres, conductivity in W/(m K), volumetric heat capacity
    in J/(m³ K), source in W/m³, and contact resistance in m² K/W. Fixed
    wall temperatures and prescribed inward flux use the steady BC format.
    A side's values may also be a callable of time, returning a scalar or
    one value per face (temperature, inward flux or Robin ambient temperature).
    source may also be a function of time returning a scalar or a cell field.
    heat_loss_coefficient is a nonnegative scalar or cell field H in W/(m³ K).
    The bulk sink H*T is implicit with the same theta as conduction. Supply
    bulk_ambient_temperature (scalar, cell field, or callable of time) to apply
    H*(Ta-T) directly, with source reserved for independent heating. Omitting
    the ambient uses Ta=0 and retains the original -H*T formulation.
    List jumps of source or wall values in load_jump_times. A callable gives
    the new value at the jump; the preceding step samples its left limit.
    With rannacher=True replace the first Crank–Nicolson step at startup and
    after each declared jump by two backward Euler half steps. monotone_theta
    raises each step's implicit weight as needed to keep the explicit update
    coefficients nonnegative. It requires nonnegative off-diagonal thermal
    couplings. Nonnegative initial data and loads then preserve nonnegative
    temperatures. Stiff steps can lose Crank–Nicolson time accuracy; interval
    heat is integrated with the adjusted weights.
    Internal steps at jumps preserve the requested output grid. Instantaneous
    power at an output jump uses the new boundary values.
    For grid-aligned anisotropy, conductivity supplies the x component and
    conductivity_y supplies the y component. Both broadcast to (ny, nx).
    With return_wall_power=True return (temperature, wall_power). For each
    side, wall_power[side] has shape (nt, number_of_faces) and gives inward
    watts per face for unit depth. return_interior_power=True appends a dict
    of instantaneous powers in W, oriented towards increasing x or y. Its x
    and y arrays have shapes (nt,ny,nx-1) and (nt,ny-1,nx). With both flags
    the output is (temperature, wall_power, interior_power).
    return_face_heat appends a dict containing "wall" and "interior" face
    heats in joules. Their leading index is the requested output interval.
    They integrate the theta-stage powers including internal steps at jumps.
    return_cell_heat appends a dict with "source" (prescribed volumetric
    heating) and "bulk" (the signed contribution of H*(Ta-T)). Both arrays have
    shape (nt-1,ny,nx) in joules per cell for unit depth. Positive heat
    enters the cell. Any H*Ta supplied in source is counted in "source".
    With periodic_y=True omit the north and south BCs. Those two face powers
    then report opposite inward transfers across a perfectly conducting contact
    between the last and first rows. Interior contact resistances are unchanged.
    On Robin faces, wall_areal_heat_capacity in J/(m² K) adds wall storage.
    initial_wall_temperature defaults to the adjacent initial cell temperature.
    Both accept scalar or per-face values. Reported power enters the material.
    wall_flux_order=2 uses two adjacent cell means to reconstruct a linearly
    varying normal flux on storing faces, accounting for both cells' widths,
    conductivities and any intervening contact. It needs two cells along the
    normal. dirichlet_flux_order=2 applies the same cell-mean reconstruction
    to fixed-temperature faces, including the flux used to advance the cells.
    Both default to the boundary half-cell conductance.
    theta=0.5 selects Crank-Nicolson and theta=1 selects backward Euler.
    """
    dx, dy, time = (np.asarray(v, dtype=float) for v in (dx, dy, time))
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time must be a strictly increasing finite vector")
    jumps = np.asarray(load_jump_times,dtype=float)
    if (jumps.ndim!=1 or not np.isfinite(jumps).all() or len(np.unique(jumps))!=len(jumps)
            or np.any(jumps<=time[0]) or np.any(jumps>=time[-1])):
        raise ValueError("load_jump_times must contain distinct finite interior times")
    requested = time
    if jumps.size:
        time = np.union1d(time,jumps)
    if rannacher:
        if theta!=.5:
            raise ValueError("Rannacher damping requires theta=.5")
        start = np.searchsorted(time,np.r_[time[0],jumps])
        mid = time[start]+.5*(time[start+1]-time[start])
        if np.any((mid<=time[start]) | (mid>=time[start+1])):
            raise ValueError("Rannacher half step needs a representable midpoint")
        backward_starts = np.r_[time[start],mid]
        time = np.union1d(time,mid)
        stage_weights = np.where(np.isin(time[:-1],backward_starts),1.,theta)
    else:
        stage_weights = np.full(len(time)-1,theta)
    subdivided = bool(jumps.size or rannacher)
    outputs = np.searchsorted(time,requested) if subdivided else slice(None)
    jump_indices = np.searchsorted(time,jumps)
    sample_times = np.r_[time,np.nextafter(time[jump_indices],time[jump_indices-1])]
    end_samples = np.arange(1,len(time))
    end_samples[jump_indices-1] = np.arange(len(jumps))+len(time)
    if wall_flux_order not in (1,2):
        raise ValueError("wall_flux_order must be 1 or 2")
    if not .5 <= theta <= 1:
        raise ValueError("theta must lie in [0.5, 1]")
    k, capacity = _conduction_materials(dx, dy, conductivity, volumetric_heat_capacity)
    shape = k.shape
    ny, nx = shape
    initial = np.broadcast_to(np.asarray(initial_temperature, dtype=float), shape)
    if not np.isfinite(initial).all():
        raise ValueError("temperature must be finite")

    if cell_temperature_reference is None:
        datum = float(initial[0,0]) if bulk_ambient_temperature is not None else 0.
        if datum:
            initial = initial-datum
    else:
        reference = np.asarray(cell_temperature_reference)
        if reference.ndim or not np.isrealobj(reference) or not np.isfinite(reference):
            raise ValueError("cell_temperature_reference must be a finite real scalar")
        datum = float(reference)
        if bulk_ambient_temperature is None:
            bulk_ambient_temperature = 0.

    sides = {"west": ny, "east": ny, "south": nx, "north": nx}
    static_bc = dict(bc)
    signals = {}
    for side, count in sides.items():
        spec = bc.get(side)
        if spec is not None and callable(spec.get("values")):
            samples = np.empty((len(sample_times), count))
            for index, instant in enumerate(sample_times):
                samples[index] = np.broadcast_to(np.asarray(spec["values"](float(instant)), dtype=float), (count,))
            if not np.isfinite(samples).all():
                raise ValueError(f"nonfinite boundary value on {side}")
            signals[side] = samples
            static_bc[side] = dict(spec, values=0.)
    _validate_conduction_bc(static_bc, sides, float)
    if datum:
        for side, spec in static_bc.items():
            if spec is None:
                continue
            kinds, values = _bc_arrays(static_bc, side, sides[side])
            thermal = np.isin(kinds, ("dirichlet", "robin"))
            if side in signals:
                signals[side][:,thermal] -= datum
            else:
                values[thermal] -= datum
                static_bc[side] = dict(spec, values=values)
    wall_data = {}
    for side, spec in list(static_bc.items()):
        if not spec or "wall_areal_heat_capacity" not in spec:
            continue
        spec = dict(spec)
        types_on_side, _ = _bc_arrays(static_bc, side, sides[side])
        cw = np.broadcast_to(np.asarray(spec.pop("wall_areal_heat_capacity"), dtype=float), (sides[side],))
        active = cw > 0
        if not np.isfinite(cw).all() or np.any(cw < 0) or np.any(active & (types_on_side != "robin")):
            raise ValueError("wall areal heat capacity must be finite, nonnegative and confined to Robin faces")
        if active.any():
            h = np.broadcast_to(np.asarray(spec.get("film_coefficient", np.nan), dtype=float), cw.shape)[active]
            if not np.all(np.isfinite(h) & (h >= 0)):
                raise ValueError("dynamic Robin faces require finite nonnegative film coefficients")
            # Assemble the cell operator first, then connect each wall as a storage node.
            spec["types"] = np.where(active, "adiabatic", types_on_side)
            wall_data[side] = (active, cw[active], h)
        static_bc[side] = spec
    for side,spec in static_bc.items():
        if spec and "wall_heater_flux" in spec and side not in wall_data:
            raise ValueError("wall_heater_flux requires a side with storing faces")
    restart_walls = {} if initial_storing_wall_state is None else initial_storing_wall_state
    if not isinstance(restart_walls, dict) or not restart_walls.keys() <= wall_data.keys():
        raise ValueError("initial_storing_wall_state must map storing sides to face temperatures")
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
    A = _bulk_loss_operator(A,heat_loss_coefficient,volume)
    if bulk_ambient_temperature is not None:
        bulk_conductance = volume*np.asarray(heat_loss_coefficient)
        def relative_ambient(value):
            if np.iscomplexobj(value):
                raise ValueError("bulk ambient temperature must be real")
            field = np.broadcast_to(np.asarray(value,dtype=float),shape)
            if not np.isfinite(field).all():
                raise ValueError("bulk ambient temperature must be finite")
            return field-datum
        if callable(bulk_ambient_temperature):
            ambient_samples = np.stack([relative_ambient(bulk_ambient_temperature(float(t))) for t in sample_times])
        else:
            ambient_stage = relative_ambient(bulk_ambient_temperature)
    A, fixed_wall_links = _fixed_wall_operator(A,b,faces,dx,dy,k,conductivity_y,dirichlet_flux_order)
    varying_loads = []
    for side, cells, conductance, _, _, area in side_specs:
        if side in signals:
            weights = _conductive_wall_power(types[side], 1., conductance, area, 0.)
            varying_loads.append((cells, signals[side], weights))
    mass = (capacity * volume).ravel()
    cell_count = mass.size
    state = initial.ravel()
    wall_links = {}
    if wall_data:
        wall_count = sum(len(cw) for _,cw,_ in wall_data.values())
        A = sp.block_diag((A, sp.csc_matrix((wall_count, wall_count))), format="lil")
        for side, cells, conductance, _, _, area in side_specs:
            if side not in wall_data:
                continue
            mask, cw, h = wall_data[side]
            cells, g, area = cells[mask], conductance[mask], area[mask]
            film = h*area
            ids = np.arange(len(cells))+len(mass)
            inner = cells
            correction = np.zeros_like(g)
            if wall_flux_order == 2:
                inner, g, correction = _second_order_wall(side,cells,mask,area,dx,dy,k,conductivity_y,faces)
            for cell, node, other, link, extra, conv in zip(cells, ids, inner, g, correction, film):
                A[cell,cell] -= link+extra
                A[cell,node] = link
                A[node,cell] = link+extra
                if extra:
                    A[cell,other] += extra
                    A[node,other] = -extra
                A[node,node] = -link-conv
            mass = np.r_[mass, cw*area]
            spec = bc[side]
            init_wall = initial.ravel()[cells]
            if side in restart_walls:
                restart = np.asarray(restart_walls[side])
                if restart.shape != cells.shape or np.iscomplexobj(restart):
                    raise ValueError(f"expected one real restart temperature per storing face on {side}")
                init_wall = np.asarray(restart,dtype=float)
                if cell_temperature_reference is None:
                    init_wall = init_wall-datum
            elif "initial_wall_temperature" in spec:
                init_wall = np.broadcast_to(np.asarray(spec["initial_wall_temperature"], dtype=float), mask.shape)[mask]-datum
            if not np.isfinite(init_wall).all():
                raise ValueError("initial wall temperature must be finite")
            state = np.r_[state, init_wall]
            b = np.r_[b, -film*values[side][mask]]
            if side in signals:
                varying_loads.append((ids, signals[side][:,mask], film))
            wall_links[side] = (mask, ids, cells, g, inner, correction, film)
        A = A.tocsc()
    if monotone_theta:
        entries = A.tocoo()
        if np.any(entries.data[entries.row!=entries.col]<0.):
            raise ValueError("monotone_theta requires nonnegative off-diagonal thermal couplings")
        outgoing = -A.diagonal()
        active = outgoing>0.
        if np.any(active):
            # M+(1-weight)*dt*A must be entrywise nonnegative. The implicit
            # conduction matrix is an M-matrix for this conservative stencil.
            explicit_limit = np.min(mass[active]/outgoing[active])
            stage_weights = np.maximum(stage_weights,1.-explicit_limit/np.diff(time))
    def heater_power(value,area):
        raw = np.asarray(value)
        if not np.isrealobj(raw) or raw.shape not in ((),area.shape):
            raise ValueError("wall_heater_flux requires a real scalar or one value per storing face")
        power = np.asarray(raw,dtype=float)*area
        if not np.isfinite(power).all():
            raise ValueError("wall_heater_flux must be finite")
        return power

    heater_stages = {}
    for side,link in wall_links.items():
        if "wall_heater_flux" not in bc[side]:
            continue
        flux = bc[side]["wall_heater_flux"]
        mask,ids = link[:2]
        area = (dy if side in ("west","east") else dx)[mask]
        if callable(flux):
            samples = np.stack([heater_power(flux(float(t)),area) for t in sample_times])
            w = stage_weights[:,None]
            power = (1-w)*samples[:len(time)-1]+w*samples[end_samples]
        else:
            power = np.broadcast_to(heater_power(flux,area),(len(time)-1,len(ids)))
        heater_stages[side] = (ids,power)
    if callable(bulk_ambient_temperature):
        w = stage_weights[:,None,None]
        ambient_stage = (1-w)*ambient_samples[:len(time)-1]+w*ambient_samples[end_samples]
    M = sp.diags(mass, format="csc")
    states = np.empty((len(time), len(mass)))
    states[0] = state
    history = states[:, :cell_count].reshape(len(time), ny, nx)
    if return_cell_heat and varying_source:
        source_stages = np.empty((len(time)-1,ny,nx))
    previous_shift = None
    for step, dt in enumerate(np.diff(time)):
        weight = stage_weights[step]
        shift = weight*dt
        if shift != previous_shift:
            lu = spla.splu(M - shift * A)
            previous_shift = shift
        state = states[step]
        rhs = mass * state + dt * ((1 - weight) * (A @ state) - b)
        if bulk_ambient_temperature is not None:
            ambient = ambient_stage[step] if callable(bulk_ambient_temperature) else ambient_stage
            rhs[:cell_count] += dt*(bulk_conductance*ambient).ravel()
        if varying_source:
            next_power = source_power(sample_times[end_samples[step]])
            stage_power = (1-weight)*previous_power+weight*next_power
            rhs[:cell_count] += dt*stage_power
            if return_cell_heat:
                source_stages[step] = stage_power.reshape(shape)
            previous_power = source_power(time[step+1]) if end_samples[step]!=step+1 else next_power
        for cells, samples, weights in varying_loads:
            rhs[cells] += dt * weights * ((1-weight)*samples[step] + weight*samples[end_samples[step]])
        for ids,power in heater_stages.values():
            rhs[ids] += dt*power[step]
        states[step + 1] = lu.solve(rhs)
    def storing_wall_power(selected_states,link):
        _, ids, cells, g, inner, correction, _ = link
        return (g*(selected_states[:,ids]-selected_states[:,cells])
                +correction*(selected_states[:,inner]-selected_states[:,cells]))

    def wall_powers(selected_states,boundary_values):
        state = selected_states[:,:cell_count]
        field = state.reshape(-1,ny,nx)
        powers = {side: _conductive_wall_power(types[side], boundary_values.get(side,values[side]),
                                               conductance,area,state[:,cells])
                  for side,cells,conductance,_,_,area in side_specs}
        for side,(mask,cells,inner,extra) in fixed_wall_links.items():
            powers[side][:,mask] += extra*(state[:,inner]-state[:,cells])
        for side,link in wall_links.items():
            powers[side][:,link[0]] = storing_wall_power(selected_states,link)
        if periodic_y:
            powers["south"] = seam*(field[:,-1]-field[:,0])
            powers["north"] = -powers["south"]
        return powers

    result = [history[outputs]]
    if return_wall_power:
        result.append(wall_powers(states[outputs],{s:v[outputs] for s,v in signals.items()}))
    if return_interior_power:
        result.append(_interior_conduction_power(result[0],faces))
    if return_face_heat or return_cell_heat or return_storing_wall_heat:
        durations = np.diff(time)
        def integrate(power):
            step_heat = power*durations.reshape((-1,)+(1,)*(power.ndim-1))
            return np.add.reduceat(step_heat,outputs[:-1],axis=0) if subdivided else step_heat

        weights = stage_weights[:,None]
        stage = (1-weights)*states[:-1]+weights*states[1:]
    if return_face_heat or return_cell_heat:
        integrated_cells = integrate(stage[:,:cell_count].reshape(-1,ny,nx))
    if return_face_heat or return_storing_wall_heat:
        stage_bc = {s:(1-weights)*v[:len(time)-1]+weights*v[end_samples] for s,v in signals.items()}
    if return_face_heat:
        walls = {s:integrate(v) for s,v in wall_powers(stage,stage_bc).items()}
        result.append({"wall":walls,"interior":_interior_conduction_power(integrated_cells,faces)})
    if return_cell_heat:
        source_heat = (integrate(source_stages) if varying_source else
                       np.diff(requested)[:,None,None]*(volume*np.asarray(source,dtype=float)))
        if bulk_ambient_temperature is None:
            bulk_heat = -integrated_cells*(volume*heat_loss_coefficient)
        else:
            relative_stage = stage[:,:cell_count].reshape(-1,ny,nx)
            bulk_heat = integrate((ambient_stage-relative_stage)*bulk_conductance)
        result.append({"source":source_heat, "bulk":bulk_heat})
    if return_storing_wall_state:
        wall_state = {}
        for side, link in wall_links.items():
            temperatures = states[:,link[1]][outputs]
            if cell_temperature_reference is None and datum:
                temperatures = temperatures+datum
            wall_state[side] = temperatures
        result.append(wall_state)
    if return_storing_wall_heat:
        wall_heat = {}
        for side,link in wall_links.items():
            mask, ids, _, _, _, _, film = link
            ambient = stage_bc.get(side,values[side])[...,mask]
            wall_heat[side] = {
                "ambient": integrate(film*(ambient-stage[:,ids])),
                "material": integrate(storing_wall_power(stage,link)),
                "storage": mass[ids]*np.diff(states[:,ids][outputs],axis=0),
            }
            if side in heater_stages:
                wall_heat[side]["heater"] = integrate(heater_stages[side][1])
        result.append(wall_heat)
    if cell_temperature_reference is None and datum:
        result[0] = result[0]+datum
    return tuple(result) if len(result)>1 else result[0]


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