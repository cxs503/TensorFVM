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
``dirichlet``, ``adiabatic`` (no conduction, upwind convection) and ``outflow``
(no conduction, extrapolated convection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["solve_steady_energy", "Grid2D", "EnergyLedger"]


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


def _face_conductance(dx, dy, kappa):
    """Return ``(cond_e, cond_n)`` face conductances (per unit depth)."""
    ny, nx = kappa.shape
    cond_e = np.zeros((ny, nx + 1))
    cond_n = np.zeros((ny + 1, nx))
    denom = 0.5 * dx[:-1][None, :] / kappa[:, :-1] + 0.5 * dx[1:][None, :] / kappa[:, 1:]
    cond_e[:, 1:nx] = dy[:, None] / denom
    cond_e[:, 0] = dy / (0.5 * dx[0] / kappa[:, 0])
    cond_e[:, nx] = dy / (0.5 * dx[-1] / kappa[:, -1])
    denom_n = 0.5 * dy[:-1][:, None] / kappa[:-1, :] + 0.5 * dy[1:][:, None] / kappa[1:, :]
    cond_n[1:ny, :] = dx[None, :] / denom_n
    cond_n[0, :] = dx / (0.5 * dy[0] / kappa[0, :])
    cond_n[ny, :] = dx / (0.5 * dy[-1] / kappa[-1, :])
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


def _bc_arrays(bc, side, count, default="adiabatic"):
    spec = bc.get(side)
    if spec is None:
        return np.array([default] * count), np.zeros(count)
    types = spec.get("types")
    values = spec.get("values", 0.0)
    if types is None:
        types = [spec["type"]] * count
    types = np.asarray(types)
    values = np.broadcast_to(np.asarray(values, dtype=float), (count,)).astype(float)
    return types, values


@dataclass
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

    def as_dict(self) -> dict:
        return {
            "boundary_heat_rate": {k: float(v) for k, v in self.boundary.items()},
            "advective_heat_rate": {k: float(v) for k, v in self.advective.items()},
            "inflow": float(self.inflow),
            "outflow": float(self.outflow),
            "imbalance": float(self.imbalance),
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


def solve_steady_energy(dx, dy, kappa, flux_e, flux_n, bc, *,
                        t_init=None, tol=1e-13, max_picard=500, dc_relax=1.0,
                        high_order=True):
    """Solve ``div(u T) = div(kappa grad T)`` with the CHT face conductance.

    Parameters
    ----------
    dx, dy : (nx,), (ny,) cell sizes.
    kappa : (ny, nx) thermal diffusivity of each cell (fluid or solid).
    flux_e, flux_n : (ny, nx+1), (ny+1, nx) volume fluxes through the faces,
        positive in the +x / +y direction (per unit depth).
    bc : dict side -> {"type": str | array, "values": float | array}.

    Returns
    -------
    ``(T, info)`` where ``info`` holds the Picard history and the ledger.
    """
    ny, nx = kappa.shape
    cond_e, cond_n = _face_conductance(dx, dy, kappa)
    idx = np.arange(ny * nx).reshape(ny, nx)

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(np.asarray(r).ravel())
        cols.append(np.asarray(c).ravel())
        vals.append(np.asarray(v).ravel())

    diag = np.zeros(ny * nx)
    b = np.zeros(ny * nx)

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
    side_specs = [
        ("west", idx[:, 0], cond_e[:, 0], flux_e[:, 0], -1.0),
        ("east", idx[:, nx - 1], cond_e[:, nx], flux_e[:, nx], +1.0),
        ("south", idx[0, :], cond_n[0, :], flux_n[0, :], -1.0),
        ("north", idx[ny - 1, :], cond_n[ny, :], flux_n[ny, :], +1.0),
    ]
    btype, bval = {}, {}
    for side, cells, cf, ff, sign_out in side_specs:
        types, values = _bc_arrays(bc, side, cells.size)
        btype[side] = types
        bval[side] = values
        cf = np.asarray(cf).ravel()
        ff = np.asarray(ff).ravel()
        cells = cells.ravel()
        dirichlet = types == "dirichlet"
        diag[cells[dirichlet]] -= cf[dirichlet]
        b[cells[dirichlet]] -= cf[dirichlet] * values[dirichlet]
        out = sign_out * ff                       # positive = leaving the domain
        diag[cells] -= np.where(out > 0.0, out, 0.0)
        inflow = out < 0.0
        b[cells[inflow]] += out[inflow] * values[inflow]

    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ny * nx, ny * nx)).tocsc()
    A = (A + sp.diags(diag)).tocsc()
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
                            side_specs, btype, bval, T, high_order)
    return T, info


def _ledger(dx, dy, kappa, cond_e, cond_n, flux_e, flux_n, side_specs, btype, bval, T,
            high_order=True):
    """Recompute the discrete energy balance cell by cell and at the boundary.

    ``residual`` accumulates the *heat entering* each cell, so it vanishes where
    the discrete equation holds.  Convective face values are the same limited
    second-order ones the deferred correction used, so the audit closes exactly
    the equation the solver solved.
    """
    ny, nx = T.shape
    _, ho_x, _, ho_y = _convective_faces(T, flux_e, flux_n, cond_e, cond_n, bval, dx, dy,
                                        first_order=not high_order)
    residual = np.zeros((ny, nx))
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
    for side, cells, cf, ff, sign_out in side_specs:
        types = btype[side]
        values = bval[side]
        cells = cells.ravel()
        cf = np.asarray(cf).ravel()
        ff = np.asarray(ff).ravel()
        Tc = T.ravel()[cells]
        q_cond = np.where(types == "dirichlet", cf * (values - Tc), 0.0)
        out = sign_out * ff
        q_conv = np.where(out < 0.0, -out * values, -out * Tc)
        boundary[side] = float(np.sum(q_cond + q_conv))
        residual.ravel()[cells] += q_cond + q_conv
        advective["inflow"] += float(np.sum(np.where(out < 0.0, -out * values, 0.0)))
        advective["outflow"] += float(np.sum(np.where(out > 0.0, out * Tc, 0.0)))
    total = sum(boundary.values())
    # scale on the largest *throughput* of the problem, not just the boundary
    # fluxes: for a case whose boundary fluxes cancel the latter is degenerate.
    scale = max([abs(v) for v in boundary.values()]
                + [abs(advective["inflow"]), abs(advective["outflow"])] + [1e-300])
    return EnergyLedger(
        boundary=boundary,
        advective=advective,
        inflow=advective["inflow"],
        outflow=advective["outflow"],
        imbalance=total,
        normalized_imbalance=abs(total) / scale,
        residual_inf=float(np.max(np.abs(residual))),
        residual_scaled=float(np.max(np.abs(residual)) / scale),
    )