"""Verification debt: the Forchheimer (``beta > 0``) porous regime had no quantitative reference.

What existed and what it could not detect
-----------------------------------------
The shipped porous extension (:class:`tensorfvm.porous_drag.PorousChannelSolver`)
treats the quadratic sink ``-rho*beta*|u|u`` with a *lagged* Picard coefficient
(``diagonal += volume*rho*beta*|u_old|``).  Its only ``beta > 0`` acceptance gate
was ``|u_bulk - forchheimer_bulk_velocity(dp/dx)| / u < 5%`` --- a volume-averaged
identity that is blind to the profile shape and that passes even if the sink is
mis-weighted or frozen at a stale field, because every existing check reads the
solver's own residual history, which is zero at convergence **by construction**
for whatever equation the code happened to assemble.

Two independent references, neither of which reads solver internals:

``continuum_solution``
    The true nonlinear boundary-value problem of the volume-averaged momentum
    equation for fully developed channel flow,

        mu u'' - (mu/K) u - rho beta |u| u = dp/dx,   u(0) = u(H) = 0,

    with ``dp/dx`` fixed by the prescribed bulk velocity
    ``(1/H) int u dy = U_b`` (incompressibility).  Solved by Picard iteration
    with continuation in ``beta`` on a fine finite-difference grid.  Being a
    *continuum* answer, it grades the finite-volume result including its
    discretisation error.

``discrete_profile``
    An independent NumPy replica of the finite-volume momentum balance that the
    MAC/SIMPLE discretisation of this repository is *supposed* to assemble at a
    u-face in fully developed flow: pressure source ``dx*dy*dp/dx``, half-cell
    wall diffusion ``2 mu dx/dy``, interior diffusion ``mu dx/dy`` and the
    implicit sink ``dx*dy*(mu/K + rho beta |u|)`` on the diagonal.  Agreement with
    the converged SIMPLE field audits coefficient placement and the claim
    "lagged sink at convergence == nonlinear sink" --- a statement no
    residual-based test can make.

Both are validated against the closed forms already shipped in the package
(Darcy--Brinkman for ``beta = 0``, plane Poiseuille for ``K -> inf``); see
:func:`reference_self_check`.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from ..porous_drag import (PorousChannelSolver, PorousMedia, brinkman_pressure_gradient,
                           brinkman_profile)

__all__ = ["CHANNEL", "continuum_solution", "discrete_profile", "flow_case",
           "implementation_agreement", "reference_self_check", "study"]

# Channel shared with ``scripts/verify_porous_drag.py`` (same geometry and fluid).
CHANNEL = dict(length=0.12, height=0.02, inlet_velocity=5e-4, density=1000.0,
               reynolds=1.0, viscosity=1e-2)


def _case(case, overrides):
    cfg = dict(CHANNEL)
    cfg.update(case or {})
    cfg.update(overrides or {})
    cfg.setdefault("form_drag_1_per_m", 0.0)
    return cfg


def _solve_profile(dpdx, cfg, y, tolerance=1e-13, max_iterations=200):
    """Newton solve of ``mu u'' - (mu/K + rho beta |u|) u = dpdx``, u(0)=u(H)=0.

    The residual scale is ``mu/dy**2 * max|u|``, so the convergence test is
    scaled; an absolute test is unreachable on a fine grid.  Damped Newton with
    backtracking is used because the quadratic sink makes plain Picard diverge
    when ``|dp/dx|`` is an order of magnitude above the Darcy--Brinkman value
    (which happens inside the ``dp/dx`` bracketing).
    """
    height, mu, rho = cfg["height"], cfg["viscosity"], cfg["density"]
    permeability, beta = cfg["permeability_m2"], cfg["form_drag_1_per_m"]
    n = len(y) - 1
    dy = y[1] - y[0]
    laplace = mu / dy ** 2
    u = brinkman_profile(y, cfg["inlet_velocity"], height, permeability).numpy().copy()

    def residual(field):
        r = np.empty(n + 1)
        r[1:-1] = (laplace * (field[:-2] - 2.0 * field[1:-1] + field[2:])
                   - (mu / permeability + rho * beta * np.abs(field[1:-1])) * field[1:-1]
                   - dpdx)
        r[0], r[-1] = field[0], field[-1]
        return r

    scale = max(laplace * float(np.max(np.abs(u))), abs(dpdx), 1e-300)
    for _ in range(max_iterations):
        r = residual(u)
        norm = float(np.max(np.abs(r)))
        if norm <= tolerance * scale:
            return u
        ab = np.empty((3, n + 1))
        ab[0, 1:] = laplace
        ab[2, :-1] = laplace
        ab[1, :] = -2.0 * laplace - (mu / permeability
                                     + 2.0 * rho * beta * np.abs(u))
        ab[0, 1] = ab[2, -2] = 0.0
        ab[1, 0] = ab[1, -1] = 1.0
        step = solve_banded((1, 1), ab, -r, check_finite=False)
        factor = 1.0
        for _ in range(40):
            trial = u + factor * step
            if float(np.max(np.abs(residual(trial)))) <= (1.0 - 1e-4 * factor) * norm:
                break
            factor *= 0.5
        u = u + factor * step
    raise RuntimeError("continuum Newton reference did not converge")


def _root_dpdx(bulk_of_dpdx, cfg):
    """Fix ``dp/dx`` so that the bulk velocity equals the prescribed inlet value."""
    height, mu, u_bulk = cfg["height"], cfg["viscosity"], cfg["inlet_velocity"]
    scale = abs(brinkman_pressure_gradient(u_bulk, height, cfg["permeability_m2"], mu))
    lo, hi = -0.25 * scale, -scale * (1.0 + 90.0)
    while bulk_of_dpdx(hi) - u_bulk < 0.0:
        hi *= 2.0
        if abs(hi) > 1e12 * scale:  # pragma: no cover - defensive
            raise RuntimeError("could not bracket dp/dx")
    return float(brentq(lambda d: bulk_of_dpdx(d) - u_bulk, lo, hi,
                        xtol=1e-30 * scale, rtol=1e-15, maxiter=300))


def continuum_solution(case=None, grid_points=2001, **overrides):
    """Exact nonlinear channel solution with prescribed bulk velocity."""
    cfg = _case(case, overrides)
    height = cfg["height"]
    y = np.linspace(0.0, height, grid_points)

    def bulk(dpdx):
        return float(np.trapezoid(_solve_profile(dpdx, cfg, y), y) / height)

    dpdx = _root_dpdx(bulk, cfg)
    u = _solve_profile(dpdx, cfg, y)
    return dict(dpdx=dpdx, y_m=y, u_m_s=u, bulk=float(np.trapezoid(u, y) / height),
                permeability_m2=cfg["permeability_m2"],
                form_drag_1_per_m=cfg["form_drag_1_per_m"],
                grid_points=int(grid_points - 1))


def discrete_profile(case=None, ny=16, **overrides):
    """Independent replica of the staggered finite-volume u-face balance.

    The discrete operator is positive definite (``diag = 2 mu dx/dy + volume*sink``,
    negative off-diagonals), and ``_momentum`` writes the pressure source
    ``dy*(p_{j-1} - p_j) = -volume*dp/dx``, hence ``rhs = -volume*dp/dx > 0`` for a
    favourable (negative) gradient.  Faces sit at ``y_j = (j + 1/2) dy``.  Wall rows carry the doubled half-cell
    diffusion ``2 mu dx/dy`` that ``SimpleSolver._momentum`` writes on its
    diagonal; the sink is implicit, i.e. the converged state of the shipped
    lagged linearisation.  ``dx = 3 H/ny`` mirrors production ``nx = 3 ny``.
    """
    cfg = _case(case, overrides)
    height, mu, rho = cfg["height"], cfg["viscosity"], cfg["density"]
    permeability, beta = cfg["permeability_m2"], cfg["form_drag_1_per_m"]
    dy = height / ny
    dx = 3.0 * height / ny
    volume = dx * dy
    diffusion = mu * dx / dy
    y_face = (np.arange(ny) + 0.5) * dy
    y_ref = np.linspace(0.0, height, 2001)
    initial = np.interp(y_face, y_ref,
                        brinkman_profile(y_ref, cfg["inlet_velocity"], height,
                                         permeability).numpy())

    def profile(dpdx):
        u = initial.copy()
        for step in range(1, 11):
            sink_beta = beta * step / 10.0
            for _ in range(5000):
                ab = np.empty((3, ny))
                ab[0, 1:] = -diffusion
                ab[2, :-1] = -diffusion
                sink = mu / permeability + rho * sink_beta * np.abs(u)
                diag = 2.0 * diffusion + volume * sink
                wall = 3.0 * diffusion + volume * sink[[0, -1]]
                diag[0], diag[-1] = wall[0], wall[1]
                ab[1, :] = diag
                new = solve_banded((1, 1), ab, np.full(ny, -dpdx * volume),
                                   check_finite=False)
                delta = float(np.max(np.abs(new - u)))
                u = new
                if delta <= 1e-13 * max(float(np.max(np.abs(u))), 1e-30):
                    break
            else:  # pragma: no cover - defensive
                raise RuntimeError("replica Picard did not converge")
        return u

    dpdx = _root_dpdx(lambda d: float(np.mean(profile(d))), cfg)
    u = profile(dpdx)
    # On the MAC grid the u-faces and the cell centres share the same y stations,
    # so the exported cell-centre profile is compared without interpolation.
    return dict(dpdx=dpdx, y_m=(np.arange(ny) + 0.5) * dy,
                u_m_s=u, u_face_m_s=u,
                bulk=float(np.mean(u)), permeability_m2=permeability,
                form_drag_1_per_m=beta, ny=int(ny))


def reference_self_check():
    """Both references must reproduce the closed forms already in the package."""
    height, u_bulk, mu = CHANNEL["height"], CHANNEL["inlet_velocity"], CHANNEL["viscosity"]
    out = {}
    for permeability in (1e-6, 1e-5):
        for beta in (0.0,):
            case = dict(permeability_m2=permeability, form_drag_1_per_m=beta)
            cont = continuum_solution(case)
            exact = brinkman_profile(cont["y_m"], u_bulk, height, permeability).numpy()
            exact_dpdx = brinkman_pressure_gradient(u_bulk, height, permeability, mu)
            out[f"continuum_K{permeability:g}_beta{beta:g}"] = dict(
                profile_linf=float(np.max(np.abs(cont["u_m_s"] - exact))
                                   / float(np.max(np.abs(exact)))),
                dpdx_relative=abs(cont["dpdx"] - exact_dpdx) / abs(exact_dpdx))
            disc = discrete_profile(case, ny=64)
            exact_c = brinkman_profile(disc["y_m"], u_bulk, height, permeability).numpy()
            out[f"discrete_K{permeability:g}_beta{beta:g}"] = dict(
                profile_linf=float(np.max(np.abs(disc["u_m_s"] - exact_c))
                                   / float(np.max(np.abs(exact_c)))),
                dpdx_relative=abs(disc["dpdx"] - exact_dpdx) / abs(exact_dpdx))
    cont = continuum_solution(dict(permeability_m2=1.0, form_drag_1_per_m=0.0))
    y = cont["y_m"]
    poiseuille = 6.0 * u_bulk * y / height * (1.0 - y / height)
    exact_dpdx = -12.0 * mu * u_bulk / height ** 2
    out["continuum_poiseuille"] = dict(
        profile_linf=float(np.max(np.abs(cont["u_m_s"] - poiseuille))
                           / float(np.max(poiseuille))),
        dpdx_relative=abs(cont["dpdx"] - exact_dpdx) / abs(exact_dpdx))
    return out


def config_for(ny):
    from ..solver import SolverConfig
    return SolverConfig(nx=3 * ny, ny=ny, length=CHANNEL["length"], height=CHANNEL["height"],
                        inlet_velocity=CHANNEL["inlet_velocity"],
                        density=CHANNEL["density"], reynolds=CHANNEL["reynolds"],
                        cylinder_radius=None, tolerance=1e-8, max_iterations=4000,
                        velocity_relaxation=0.7)


def flow_case(ny=16, permeability=1e-6, form_drag=0.0):
    """Run the shipped solver and export the fully developed cell-centre field."""
    cfg = config_for(ny)
    started = time.perf_counter()
    solver = PorousChannelSolver(cfg, PorousMedia(permeability_m2=permeability,
                                                 form_drag_1_per_m=form_drag))
    result = solver.solve()
    elapsed = time.perf_counter() - started
    u, _ = result.cell_center_velocity()
    u, p = u.numpy(), result.p.numpy()
    x, y = result.x.numpy(), result.y.numpy()
    height = CHANNEL["height"]
    region = (x >= 2.0 * height) & (x <= 0.98 * cfg.length)
    profile = u[:, region].mean(axis=1)
    px = p[:, region].mean(axis=0)
    xr = x[region] - x[region].mean()
    return dict(ny=int(ny), converged=bool(result.converged), iterations=len(result.history),
                elapsed_s=elapsed, y_m=y, u_m_s=profile,
                dpdx=float(np.dot(xr, px) / np.dot(xr, xr)),
                residual=dict(result.history[-1]) if result.history else {},
                permeability_m2=permeability, form_drag_1_per_m=form_drag)


def implementation_agreement(ny=32, permeability=1e-6, form_drag=2e4):
    """Converged SIMPLE field vs the independent finite-volume replica, same grid."""
    sim = flow_case(ny=ny, permeability=permeability, form_drag=form_drag)
    rep = discrete_profile(dict(permeability_m2=permeability,
                                form_drag_1_per_m=form_drag), ny=ny)
    peak = float(np.max(np.abs(rep["u_m_s"])))
    return dict(
        ny=int(ny), permeability_m2=permeability, form_drag_1_per_m=form_drag,
        profile_linf=float(np.max(np.abs(sim["u_m_s"] - rep["u_m_s"])) / peak),
        profile_l2=float(np.sqrt(np.mean((sim["u_m_s"] - rep["u_m_s"]) ** 2)) / peak),
        dpdx_sim=sim["dpdx"], dpdx_replica=rep["dpdx"],
        dpdx_relative=abs(sim["dpdx"] - rep["dpdx"]) / abs(rep["dpdx"]),
        converged=sim["converged"], iterations=sim["iterations"],
        y_m=sim["y_m"].tolist(), sim_profile=sim["u_m_s"].tolist(),
        replica_profile=rep["u_m_s"].tolist())


def study(levels=(16, 32), form_drags=(0.0, 2e3, 2e4), permeability=1e-6, ny_audit=32):
    """Grid ladder against the continuum reference plus the discrete audit."""
    rows = []
    for beta in form_drags:
        case = dict(permeability_m2=permeability, form_drag_1_per_m=beta)
        exact = continuum_solution(case)
        for ny in levels:
            sim = flow_case(ny=ny, permeability=permeability, form_drag=beta)
            y = sim["y_m"]
            exact_c = np.interp(y, exact["y_m"], exact["u_m_s"])
            peak = float(np.max(np.abs(exact_c)))
            rows.append(dict(
                ny=int(ny), form_drag_1_per_m=beta, permeability_m2=permeability,
                layer_ratio=float((CHANNEL["height"] / ny) / math.sqrt(permeability)),
                profile_linf=float(np.max(np.abs(sim["u_m_s"] - exact_c)) / peak),
                profile_l2=float(np.sqrt(np.mean((sim["u_m_s"] - exact_c) ** 2)) / peak),
                dpdx_sim=sim["dpdx"], dpdx_exact=exact["dpdx"],
                dpdx_relative=abs(sim["dpdx"] - exact["dpdx"]) / abs(exact["dpdx"]),
                bulk_sim=float(np.mean(sim["u_m_s"])), bulk_exact=exact["bulk"],
                converged=sim["converged"], iterations=sim["iterations"],
                elapsed_s=sim["elapsed_s"],
                y_m=y.tolist(), u_sim=sim["u_m_s"].tolist(), u_exact=exact_c.tolist()))
    orders = {}
    for beta in form_drags:
        sel = [r for r in rows if r["form_drag_1_per_m"] == beta]
        e = [r["profile_linf"] for r in sel]
        orders[f"{beta:g}"] = [float(math.log(e[i] / e[i + 1]) / math.log(2.0))
                              for i in range(len(e) - 1)]
    audit = implementation_agreement(ny=ny_audit, permeability=permeability,
                                     form_drag=form_drags[-1])
    return dict(channel=CHANNEL, levels=list(levels), form_drags=list(form_drags),
                permeability_m2=permeability, self_check=reference_self_check(),
                rows=rows, orders=orders, audit=audit)


if __name__ == "__main__":  # pragma: no cover - manual runner
    import json
    result = study()
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    for row in result["rows"]:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()
               if k not in ("y_m", "u_sim", "u_exact")})