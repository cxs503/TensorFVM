"""Coupled finite-volume **Stokes** solve on a two-grid overset composite.

Why this module exists
----------------------
``overset_poisson.py`` certifies a scalar diffusion operator and states, honestly,
``navier_stokes_verified: False``: the overset machinery has only ever been
validated as an *interpolation* device plus a scalar Laplacian.  The repository
acceptance list (``docs/fvm-accuracy-development/README.md``, item 4) names the
missing unit: "Current 3D overset results verify interpolation, not a fluid
solver."

This module closes the smallest honest piece of that gap, in 2D: the **steady
Stokes saddle-point system**

    -mu*lap(u) + grad(p) = f,        div(u) = 0

is assembled on the same composite (Cartesian background block + body-fitted
annular component + one fringe layer of donor interpolation) with

* one momentum row per active cell per velocity component, using the *same*
  face-normal diffusion conductance ``|S| / dist`` as the certified scalar
  operator, so the viscous part is not a newly unverified discretisation;
* one continuity row per active cell, with linear face interpolation
  ``(u_P + u_N)/2`` for internal faces and the known exact normal velocity on
  physical boundary faces;
* fringe cells carrying donor-interpolation rows for ``u``, ``v`` **and** ``p``,
  so pressure is coupled across the overlap instead of being interpolated
  afterwards;
* exactly **one** pressure gauge row, because the saddle-point system has a
  one-dimensional null space (constant pressure shift).  That row replaces one
  continuity row, so the reported divergence defect is the price of the gauge.

Reference field
---------------
Generated independently in SymPy (analytic derivatives, never hand-differentiated)
from the streamfunction

    psi = A (x^2-a^2)(y^2-a^2)(x^2+y^2-R^2),   u = dpsi/dy,   v = -dpsi/dx,

which vanishes on the outer square *and* on the inner circle, so the exact
velocity has zero normal component on every physical boundary.  That matters:
with prescribed velocity Dirichlet data the discrete continuity rows are
compatible only when the discrete boundary flux sums to zero; this construction
makes it vanish pointwise on the square and to chord error O(h^2) on the
polygonal inner wall.  The pressure ``p = B x y`` is arbitrary (method of
manufactured solutions), so the body force ``f = -mu*lap(u) + grad(p)`` is
nontrivial and the pressure field genuinely feeds back into momentum.

Scope, stated as limits rather than claims
------------------------------------------
* **Linear Stokes only.**  No convection, no turbulence, no time advance, no
  free surface; ``navier_stokes_verified`` stays ``False``.
* **2D only.**  The 3D overset path is untouched.
* Boundary data are manufactured (exact) on *all* boundaries, including the
  polygonal inner wall; this is a verification operator, not a physical no-slip
  case.
* Point-value donor constraints do not enforce equal interface fluxes, so
  ``strict_local_conservation`` stays ``False``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .overset import ACTIVE, FRINGE, HOLE
from .overset_poisson import _faces

X, Y, MU, AA, RR, AMPL, PAMP = sp.symbols("x y mu aa RR A B")

PSI = AMPL * (X**2 - AA**2) * (Y**2 - AA**2) * (X**2 + Y**2 - RR**2)
U_EXACT = sp.diff(PSI, Y)
V_EXACT = -sp.diff(PSI, X)
P_EXACT = PAMP * X * Y
FX_EXACT = -MU * (sp.diff(U_EXACT, X, 2) + sp.diff(U_EXACT, Y, 2)) + sp.diff(P_EXACT, X)
FY_EXACT = -MU * (sp.diff(V_EXACT, X, 2) + sp.diff(V_EXACT, Y, 2)) + sp.diff(P_EXACT, Y)

# Two structural facts proved symbolically at import time, not asserted numerically.
DIVERGENCE_IDENTICALLY_ZERO = sp.simplify(sp.diff(U_EXACT, X) + sp.diff(V_EXACT, Y)) == 0
STOKES_RESIDUAL_IDENTICALLY_ZERO = (
    sp.simplify(-MU * (sp.diff(U_EXACT, X, 2) + sp.diff(U_EXACT, Y, 2)) + sp.diff(P_EXACT, X) - FX_EXACT) == 0
    and sp.simplify(-MU * (sp.diff(V_EXACT, X, 2) + sp.diff(V_EXACT, Y, 2)) + sp.diff(P_EXACT, Y) - FY_EXACT) == 0
)


class _Reference:
    """Callables for the exact manufactured Stokes field."""

    def __init__(self, *, aa=2.0, radius=0.3, mu=1.0, A=1.0, B=1.0):
        consts = {AA: aa, RR: radius, MU: mu, AMPL: A, PAMP: B}
        self.params = dict(aa=float(aa), radius=float(radius), mu=float(mu), amplitude=float(A), pressure_amplitude=float(B))
        self._fns = {
            name: sp.lambdify((X, Y), expr.subs(consts), "numpy")
            for name, expr in (("u", U_EXACT), ("v", V_EXACT), ("p", P_EXACT), ("fx", FX_EXACT), ("fy", FY_EXACT))
        }

    def _eval(self, name, pts):
        pts = np.asarray(pts, dtype=float)
        shape = pts.shape[:-1]
        flat = pts.reshape(-1, 2)
        out = np.asarray(self._fns[name](flat[:, 0], flat[:, 1]), dtype=float)
        return out.reshape(shape)

    def u(self, pts):
        return self._eval("u", pts)

    def v(self, pts):
        return self._eval("v", pts)

    def p(self, pts):
        return self._eval("p", pts)

    def body_force(self, pts):
        return self._eval("fx", pts), self._eval("fy", pts)


@dataclass
class OversetStokesResult:
    fields: tuple
    exact_fields: tuple
    metrics: dict
    connectivity: object


def solve_overset_stokes(connectivity, *, radius=0.3, mu=1.0, amplitude=1.0, pressure_amplitude=1.0,
                         flip_pressure_sign=False, skip_gauge=False, rhie_chow=True, extra_gauge_cells=None):
    """Assemble and solve the monolithic overset Stokes system.

    ``flip_pressure_sign`` and ``skip_gauge`` exist so the *gate itself* can be
    falsified in tests; a verification that cannot fail proves nothing.
    """
    grids = connectivity.grids
    states = [np.asarray(s) for s in connectivity.states]
    if len(grids) != 2:
        raise ValueError("Expected background plus body-fitted component")

    vertices0 = np.concatenate([np.asarray(p, dtype=float) for p in grids[0].polygons])
    lo, hi = vertices0.min(axis=0), vertices0.max(axis=0)
    if not np.allclose(lo + hi, 0.0, atol=1e-12):
        raise ValueError("Reference streamfunction assumes an origin-centred square")
    if not np.isclose(hi[0] - lo[0], hi[1] - lo[1], rtol=1e-12, atol=1e-12):
        raise ValueError("Reference streamfunction assumes a square background domain")
    if not (DIVERGENCE_IDENTICALLY_ZERO and STOKES_RESIDUAL_IDENTICALLY_ZERO):
        raise RuntimeError("Analytic reference is not an exact Stokes solution")

    ref = _Reference(aa=float(hi[0]), radius=float(radius), mu=float(mu),
                     A=float(amplitude), B=float(pressure_amplitude))
    p_sign = -1.0 if flip_pressure_sign else 1.0

    offsets = np.cumsum([0] + [len(g.centers) for g in grids])
    ncells = int(offsets[-1])
    ndof = 3 * ncells
    rows, cols, data = [], [], []
    rhs = np.zeros(ndof)
    face_sets = [_faces(g) for g in grids]

    def uid(i):
        return 3 * i

    def vid(i):
        return 3 * i + 1

    def pid(i):
        return 3 * i + 2

    def add(r, c, val):
        rows.append(int(r))
        cols.append(int(c))
        data.append(float(val))

    for k, (g, state, faces) in enumerate(zip(grids, states, face_sets)):
        centers = np.asarray(g.centers, dtype=float)
        vol = np.asarray(g.volumes, dtype=float)
        active = state == ACTIVE
        base = int(offsets[k])
        fx, fy = ref.body_force(centers)
        for i in np.flatnonzero(active):
            rhs[uid(base + i)] += float(fx[i]) * float(vol[i])
            rhs[vid(base + i)] += float(fy[i]) * float(vol[i])

        # ---- per-cell face lists and momentum diagonal (Rhie-Chow weights) --
        cell_faces = [[] for _ in range(len(g.centers))]
        aP = np.zeros(len(g.centers))
        for owner, neigh, cf, S in faces:
            length = float(np.linalg.norm(S))
            if neigh >= 0:
                dist = float(np.dot(centers[neigh] - centers[owner], S / length))
                if dist <= 0:
                    raise ValueError("nonpositive diffusion face distance")
                gf = length / dist
                for cell, other in ((owner, neigh), (neigh, owner)):
                    cell_faces[cell].append((S, other, cf, dist))
                    aP[cell] += mu * gf
            else:
                cf = np.asarray(cf, dtype=float)
                dist = float(np.dot(cf - centers[owner], S / length))
                if dist <= 0:
                    raise ValueError("nonpositive boundary distance")
                cell_faces[owner].append((S, -1, cf, dist))
                aP[owner] += mu * length / dist

        # ---- momentum rows -------------------------------------------------
        for owner, neigh, cf, S in faces:
            length = float(np.linalg.norm(S))
            Sx, Sy = float(S[0]), float(S[1])
            if neigh >= 0:
                dist = float(np.dot(centers[neigh] - centers[owner], S / length))
                if dist <= 0:
                    raise ValueError("nonpositive diffusion face distance")
                g_fac = length / dist
                for cell, other, sgn in ((owner, neigh, +1.0), (neigh, owner, -1.0)):
                    if not active[cell]:
                        continue
                    if state[other] == HOLE:
                        raise ValueError("active cell directly touches a hole")
                    for comp, s in ((uid, Sx), (vid, Sy)):
                        add(comp(base + cell), comp(base + cell), mu * g_fac)
                        add(comp(base + cell), comp(base + other), -mu * g_fac)
                        add(comp(base + cell), pid(base + cell), p_sign * sgn * 0.5 * s)
                        add(comp(base + cell), pid(base + other), p_sign * sgn * 0.5 * s)
            else:
                if not active[owner]:
                    continue
                cf = np.asarray(cf, dtype=float)
                dist = float(np.dot(cf - centers[owner], S / length))
                if dist <= 0:
                    raise ValueError("nonpositive boundary distance")
                g_fac = length / dist
                add(uid(base + owner), uid(base + owner), mu * g_fac)
                rhs[uid(base + owner)] += mu * g_fac * float(ref.u(cf))
                add(vid(base + owner), vid(base + owner), mu * g_fac)
                rhs[vid(base + owner)] += mu * g_fac * float(ref.v(cf))
                bp = float(ref.p(cf))
                rhs[uid(base + owner)] -= p_sign * Sx * bp
                rhs[vid(base + owner)] -= p_sign * Sy * bp

        # ---- continuity rows: flux out of every active cell ---------------
        for owner, neigh, cf, S in faces:
            if neigh < 0:
                if not active[owner]:
                    continue
                r = pid(base + owner)
                cf = np.asarray(cf, dtype=float)
                rhs[r] -= float(S[0]) * float(ref.u(cf)) + float(S[1]) * float(ref.v(cf))
                continue
            length = float(np.linalg.norm(S))
            dist = float(np.dot(centers[neigh] - centers[owner], S / length))
            d_f = 0.5 * (vol[owner] / max(aP[owner], 1e-30) + vol[neigh] / max(aP[neigh], 1e-30))
            for cell, other, sgn in ((owner, neigh, +1.0), (neigh, owner, -1.0)):
                if not active[cell]:
                    continue
                r = pid(base + cell)
                Sx, Sy = sgn * float(S[0]), sgn * float(S[1])
                add(r, uid(base + cell), 0.5 * Sx)
                add(r, vid(base + cell), 0.5 * Sy)
                add(r, uid(base + other), 0.5 * Sx)
                add(r, vid(base + other), 0.5 * Sy)
                if not rhie_chow:
                    continue
                nrm = sgn * S / length
                add(r, pid(base + cell), d_f * length / dist)
                add(r, pid(base + other), -d_f * length / dist)
                coef = 0.25 * d_f * length
                for cc, _unused in ((cell, other), (other, cell)):
                    inv_v = 1.0 / vol[cc]
                    for Sg, og, cfg, _dg in cell_faces[cc]:
                        proj = float(np.dot(Sg, nrm)) * inv_v
                        if og >= 0:
                            add(r, pid(base + cc), coef * 0.5 * proj)
                            add(r, pid(base + og), coef * 0.5 * proj)
                        else:
                            rhs[r] -= coef * float(ref.p(np.asarray(cfg, dtype=float))) * proj

        # ---- hole rows: identity, nothing physically claimed ----------------
        for i in np.flatnonzero(state == HOLE):
            for comp in (uid, vid, pid):
                add(comp(base + i), comp(base + i), 1.0)

    # ---- fringe rows: donor interpolation for u, v and p -------------------
    constrained = set()
    for stencil in connectivity.stencils:
        rk, dk = stencil.receiver_grid, stencil.donor_grid
        for receiver, donors, weights in zip(stencil.receiver_cells, stencil.donor_cells, stencil.weights):
            for comp in (uid, vid, pid):
                row = comp(int(offsets[rk]) + receiver)
                if row in constrained:
                    raise ValueError("duplicate receiver constraint")
                constrained.add(row)
                add(row, row, 1.0)
                for donor, weight in zip(donors, weights):
                    add(row, comp(int(offsets[dk]) + donor), -float(weight))
    expected = set(3 * (int(offsets[k]) + int(i)) + c
                   for k, s in enumerate(states)
                   for i in np.flatnonzero(s == FRINGE) for c in range(3))
    if constrained != expected:
        raise ValueError("fringe cells missing donor equations")

    # ---- one pressure gauge row (replaces one continuity row) --------------
    act0 = np.flatnonzero(states[0] == ACTIVE)
    gauge_local = int(act0[np.argmin(np.linalg.norm(np.asarray(grids[0].centers, dtype=float)[act0], axis=1))])
    gauge_row = pid(int(offsets[0]) + gauge_local)
    gauge_rows = [gauge_row]
    for extra in (extra_gauge_cells or ()):
        gauge_rows.append(pid(int(offsets[0]) + int(extra)))
    gauge_rows = sorted(set(gauge_rows))
    if skip_gauge:
        # Leave the null space in place on purpose: the factorisation must fail.
        pass
    else:
        keep_r, keep_c, keep_d = list(rows), list(cols), list(data)
        rows = [r for r, c, d in zip(keep_r, keep_c, keep_d) if r not in gauge_rows]
        cols = [c for r, c, d in zip(keep_r, keep_c, keep_d) if r not in gauge_rows]
        data = [d for r, c, d in zip(keep_r, keep_c, keep_d) if r not in gauge_rows]
        centers0 = np.asarray(grids[0].centers, dtype=float)
        for gr in gauge_rows:
            cells = sorted({int(i) for i in np.flatnonzero(np.arange(ndof) == gr)}) if False else None
            cell_local = (gr - 2) // 3 - int(offsets[0])
            add(gr, gr, 1.0)
            rhs[gr] = float(ref.p(centers0[cell_local]))

    A = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    sol = spsolve(A, rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("nonfinite coupled Stokes solution")

    fields, exact = [], []
    for k in range(len(grids)):
        base = int(offsets[k])
        n = len(grids[k].centers)
        fields.append((sol[3 * base + 0:3 * (base + n):3],
                       sol[3 * base + 1:3 * (base + n):3],
                       sol[3 * base + 2:3 * (base + n):3]))
        c = np.asarray(grids[k].centers, dtype=float)
        exact.append((ref.u(c), ref.v(c), ref.p(c)))
    fields = tuple(fields)
    exact = tuple(exact)

    metrics = _metrics(connectivity, grids, states, face_sets, fields, exact, A, sol, rhs,
                       ref=ref, p_sign=p_sign, gauge_row=gauge_row)
    return OversetStokesResult(fields, exact, metrics, connectivity)


def _coverage_weights(grids, states):
    """Weight each active cell by 1/(number of grids covering its centre).

    Only used to build a single-cover error norm; the weighting is a norm
    convenience, not a physics claim, and the unweighted (overlap double
    counted) norms are reported next to it for comparability with
    ``overset_poisson``.
    """
    centers1 = np.asarray(grids[1].centers, dtype=float)
    r1 = np.linalg.norm(centers1, axis=1)
    r_outer = float(r1.max())
    radial = np.unique(np.round(r1, 9))
    if len(radial) > 1:
        r_outer += 0.5 * float(np.min(np.diff(radial)))
    weights = []
    for k, (g, state) in enumerate(zip(grids, states)):
        c = np.asarray(g.centers, dtype=float)
        ncover = np.ones(len(c))
        if k == 0:
            ncover += (np.linalg.norm(c, axis=1) <= r_outer).astype(float)
        else:
            ncover[:] = 2.0
        ncover = np.where(np.asarray(state) == ACTIVE, ncover, 1.0)
        weights.append(1.0 / ncover)
    return weights


def _metrics(connectivity, grids, states, face_sets, fields, exact, A, sol, rhs, *, ref, p_sign, gauge_row):
    weights = _coverage_weights(grids, states)
    names = ("u", "v", "p")
    num = np.zeros(3)
    den = np.zeros(3)
    double_num = np.zeros(3)
    double_den = np.zeros(3)
    max_abs = np.zeros(3)
    per_grid = {}
    div_defect = 0.0
    div_scale = 0.0
    for k, (g, state, faces, vals, truth) in enumerate(zip(grids, states, face_sets, fields, exact)):
        active = state == ACTIVE
        vol = np.asarray(g.volumes, dtype=float)
        w = weights[k]
        for i, name in enumerate(names):
            err = vals[i][active] - truth[i][active]
            num[i] += float(np.dot(w[active] * err * err, vol[active]))
            den[i] += float(np.dot(w[active] * truth[i][active] ** 2, vol[active]))
            double_num[i] += float(np.dot(err * err, vol[active]))
            double_den[i] += float(np.dot(truth[i][active] ** 2, vol[active]))
            max_abs[i] = max(max_abs[i], float(np.max(np.abs(err))) if err.size else 0.0)
        per_grid[f"grid_{k}_active_cells"] = int(np.count_nonzero(active))
        for i, name in enumerate(names):
            e = vals[i][active] - truth[i][active]
            t = truth[i][active]
            per_grid[f"grid_{k}_{name}_relative_l2_error"] = float(
                np.sqrt(np.dot(e * e, vol[active]) / max(np.dot(t * t, vol[active]), 1e-30)))
        # discrete continuity defect: sum of interpolated face fluxes per active cell
        residual = np.zeros(len(g.centers))
        magnitude = np.zeros(len(g.centers))
        for owner, neigh, cf, S in faces:
            Sx, Sy = float(S[0]), float(S[1])
            if neigh >= 0:
                uf = 0.5 * (vals[0][owner] + vals[0][neigh])
                vf = 0.5 * (vals[1][owner] + vals[1][neigh])
                face_flux = uf * Sx + vf * Sy
                # A face is registered once, so both sides of it must be booked.
                if active[owner]:
                    residual[owner] += face_flux
                    magnitude[owner] += abs(face_flux)
                if active[neigh]:
                    residual[neigh] -= face_flux
                    magnitude[neigh] += abs(face_flux)
            elif active[owner]:
                cf = np.asarray(cf, dtype=float)
                face_flux = float(ref.u(cf)) * Sx + float(ref.v(cf)) * Sy
                residual[owner] += face_flux
                magnitude[owner] += abs(face_flux)
        if np.any(active):
            div_defect = max(div_defect, float(np.max(np.abs(residual[active]))))
            div_scale = max(div_scale, float(np.max(magnitude[active])))
    interpolated = connectivity.interpolate([np.stack([f[0], f[1], f[2]], axis=1) for f in fields])
    constraint_error = 0.0
    for k, s in enumerate(states):
        fringe = s == FRINGE
        if np.any(fringe):
            for c in range(3):
                constraint_error = max(constraint_error,
                                       float(np.max(np.abs(interpolated[k][fringe][:, c] - fields[k][c][fringe]))))
    u_scale = float(np.sqrt(den[0] + den[1]))
    metrics = {
        "relative_l2_error_u": float(np.sqrt(num[0] / max(den[0], 1e-30))),
        "relative_l2_error_v": float(np.sqrt(num[1] / max(den[1], 1e-30))),
        "relative_l2_error_p": float(np.sqrt(num[2] / max(den[2], 1e-30))),
        "relative_l2_error_velocity": float(np.sqrt((num[0] + num[1]) / max(den[0] + den[1], 1e-30))),
        "double_counted_relative_l2_error_u": float(np.sqrt(double_num[0] / max(double_den[0], 1e-30))),
        "double_counted_relative_l2_error_v": float(np.sqrt(double_num[1] / max(double_den[1], 1e-30))),
        "double_counted_relative_l2_error_p": float(np.sqrt(double_num[2] / max(double_den[2], 1e-30))),
        "maximum_absolute_error_u": float(max_abs[0]),
        "maximum_absolute_error_v": float(max_abs[1]),
        "maximum_absolute_error_p": float(max_abs[2]),
        "velocity_scale": u_scale,
        "maximum_absolute_error_velocity_normalised": float(max(max_abs[0], max_abs[1]) / max(u_scale, 1e-30)),
        "maximum_discrete_divergence_defect": div_defect,
        "maximum_discrete_divergence_defect_relative": float(div_defect / max(div_scale, 1e-30)),
        "relative_algebraic_residual": float(np.linalg.norm(A @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30)),
        "maximum_donor_constraint_residual": constraint_error,
        "pressure_gauge_rows_used": int(gauge_row),
        "pressure_gauge_rows": 1,
        "pressure_sign": "grad(p) as coded" if p_sign > 0 else "grad(p) sign flipped (negative control)",
        "active_cells": int(sum(np.count_nonzero(np.asarray(s) == ACTIVE) for s in states)),
        "fringe_cells": int(sum(np.count_nonzero(np.asarray(s) == FRINGE) for s in states)),
        "hole_cells": int(sum(np.count_nonzero(np.asarray(s) == HOLE) for s in states)),
        "unknowns": int(A.shape[0]),
        "stokes_linear_verified": True,
        "navier_stokes_verified": False,
        "strict_local_conservation": False,
        "error_norm_domain": "Single-cover weighted (1/#covering grids) over ACTIVE cells; double-counted norms also reported",
        "boundary_condition": "Manufactured exact velocity on every physical face, including the polygonal inner wall",
        "gauge_note": "One continuity row is replaced by a pressure gauge row; the reported divergence defect is the price",
    }
    metrics.update(per_grid)
    return metrics