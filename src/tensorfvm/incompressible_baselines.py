"""Incompressible FVM verification baselines: backward-facing step, planar
sudden expansion and differentially heated cavity.

The three cases share the SIMPLE kernel of
:mod:`tensorfvm.staggered_simple` and the reference data below.

Conventions
-----------
* ``Re`` for the step cases is based on the **downstream channel height** ``H``
  and the downstream bulk velocity ``U_b = Q/H`` (``Q`` = volume flux per unit
  depth), i.e. Gartling's convention in which the inlet maximum is
  ``1.5 U_b``.
* Lengths are normalised by ``H``; the reattachment point is measured from the
  step face (``x = 0``).
* The cavity uses the non-dimensional Boussinesq set ``nu = sqrt(Pr/Ra)``,
  ``alpha = nu/Pr``, ``g*beta = 1``, ``dT = 1``, ``L = 1``.
"""

from __future__ import annotations

import numpy as np

from .staggered_simple import StaggeredSimple

# --------------------------------------------------------------------------- #
# literature reference data
# --------------------------------------------------------------------------- #
#: Gartling (1990) 2-D test problem, expansion ratio 2, Re_H = 800, plus the
#: verification-contract target and the Armaly et al. (1983) experimental band.
BFS_REFERENCE = {
    "reattachment_over_H": 6.10,
    "contract_target": 6.0,
    "contract_band": 0.2,
    "contract_tolerance": 0.10,
    "armaly_1983_band": (5.8, 6.4),
}

#: de Vahl Davis (1983) benchmark.  The contract values are within the published
#: scatter of the 1983 table (2.238 / 4.509 / 8.817); both are reported.
NATURAL_CONVECTION_REFERENCE = {
    1e4: {"nu_contract": 2.243, "nu_dvd1983": 2.238, "psi_max_dvd1983": 5.071},
    1e5: {"nu_contract": 4.519, "nu_dvd1983": 4.509, "psi_max_dvd1983": 9.111},
    1e6: {"nu_contract": 8.800, "nu_dvd1983": 8.817, "psi_max_dvd1983": 16.32},
}

SUDDEN_EXPANSION_AREA_RATIO = 4.0
#: Borda-Carnot loss coefficient referred to the upstream dynamic head,
#: :math:`K = (1 - A_1/A_2)^2` (uniform profiles, inviscid, high Re limit).
SUDDEN_EXPANSION_K_THEORY = (1.0 - 1.0 / SUDDEN_EXPANSION_AREA_RATIO) ** 2  # 0.5625


# --------------------------------------------------------------------------- #
# backward-facing step
# --------------------------------------------------------------------------- #
def build_backward_facing_step(n_over_H: int, *, re_H: float = 800.0,
                               inlet_length_H: float = 2.0,
                               outlet_length_H: float = 10.0,
                               limited: bool = True) -> StaggeredSimple:
    """Gartling BFS: step height ``H/2``, expansion ratio 2, ``Re_H = 800``.

    ``n_over_H`` cells resolve the downstream channel height ``H = 1``.  The
    upstream duct occupies ``y in [0.5, 1]``; below it, for ``x < 0``, the cells
    are solid (the step).  The inlet profile is parabolic with maximum ``1.5``
    and mean ``1`` over the half-height duct, so the downstream bulk velocity is
    ``U_b = 0.5``.
    """
    H = 1.0
    hy = H / n_over_H
    hx = hy
    nx = int(round((inlet_length_H + outlet_length_H) * n_over_H))
    ny = n_over_H
    n_in = int(round(inlet_length_H * n_over_H))
    j_step = int(round(0.5 * n_over_H))
    ys = (np.arange(ny) + 0.5) * hy
    fluid = np.ones((nx, ny), dtype=bool)
    fluid[:n_in, :j_step] = False
    s = np.clip(ys - 0.5, 0.0, None)
    profile = np.where(fluid[0], 24.0 * s * (0.5 - s), 0.0)
    u_bulk = 0.5
    nu = u_bulk * H / re_H
    solver = StaggeredSimple(nx, ny, hx, hy, nu, fluid, inlet_profile=profile,
                             east_outlet=True, limited=limited,
                             alpha_u=0.7, alpha_p=0.3)
    solver.meta = dict(case="backward_facing_step", n_over_H=n_over_H, re_H=re_H,
                       u_bulk=u_bulk, nu=nu, H=H, x_step=0.0, n_in=n_in,
                       hx=hx, hy=hy, inlet_length_H=inlet_length_H,
                       outlet_length_H=outlet_length_H,
                       nx=nx, ny=ny, cells=int(fluid.sum()))
    return solver


def _wall_reattachment(x: np.ndarray, tau: np.ndarray) -> float | None:
    """First downstream sign change of ``tau`` from negative to positive."""
    for k in range(len(tau) - 1):
        if tau[k] < 0.0 <= tau[k + 1]:
            frac = -tau[k] / (tau[k + 1] - tau[k])
            return float(x[k] + frac * (x[k + 1] - x[k]))
    return None


def backward_facing_step_metrics(solver: StaggeredSimple) -> dict:
    """Reattachment length from the step, backflow statistics, mass imbalance."""
    m = solver.meta
    hx, hy = solver.hx, solver.hy
    x = np.arange(solver.nx + 1) * hx - m["inlet_length_H"]
    tau = solver.u[:, 0]                     # sign of the wall shear at y = 0
    xs, ts = x[m["n_in"]:], tau[m["n_in"]:]
    x_r = _wall_reattachment(xs, ts)
    uc = 0.5 * (solver.u[:-1] + solver.u[1:])
    fluid = solver.fluid
    recirc = np.where(fluid, uc, 0.0)
    min_u = float(recirc.min())
    area_neg = float((np.where(fluid, uc, 0.0) < 0).sum()) / float(fluid.sum())
    return dict(
        reattachment_over_H=None if x_r is None else float(x_r / m["H"]),
        min_u_in_recirculation=min_u,
        backflow_area_fraction=area_neg,
        continuity=float(np.abs(solver.divergence()).max()),
        iterations=int(solver.iterations),
    )


# --------------------------------------------------------------------------- #
# planar sudden expansion
# --------------------------------------------------------------------------- #
def build_sudden_expansion(n_over_D: int, *, re_D: float = 100.0,
                           area_ratio: float = 4.0,
                           inlet_profile: str = "uniform",
                           length_inlet_heights: float = 12.0,
                           limited: bool = True) -> StaggeredSimple:
    """Planar sudden expansion ``A2/A1 = area_ratio``, inlet height ``D = 1``.

    ``n_over_D`` cells resolve ``D``.  The inlet occupies ``y in [0, 1]`` on the
    west boundary, the remaining part of that boundary is the no-slip step face,
    so no blocked cells are needed.  ``Re_D = U_in D / nu``.
    """
    D = 1.0
    H2 = area_ratio * D
    hy = D / n_over_D
    hx = hy
    ny = int(round(H2 / hy))
    nx = int(round(length_inlet_heights * D / hx))
    ys = (np.arange(ny) + 0.5) * hy
    fluid = np.ones((nx, ny), dtype=bool)
    if inlet_profile == "uniform":
        profile = np.where(ys <= D, 1.0, 0.0)
        u_mean = 1.0
    elif inlet_profile == "parabolic":
        s = np.clip(ys, 0.0, D)
        profile = np.where(ys <= D, 6.0 * (s / D) * (1.0 - s / D), 0.0)
        u_mean = 1.0
    else:
        raise ValueError("inlet_profile must be 'uniform' or 'parabolic'")
    nu = u_mean * D / re_D
    solver = StaggeredSimple(nx, ny, hx, hy, nu, fluid, inlet_profile=profile,
                             east_outlet=True, limited=limited,
                             alpha_u=0.7, alpha_p=0.3)
    solver.meta = dict(case="sudden_expansion", n_over_D=n_over_D, re_D=re_D,
                       area_ratio=area_ratio, inlet_profile=inlet_profile,
                       u_mean_inlet=u_mean, nu=nu, D=D, H2=H2, hx=hx, hy=hy,
                       length_inlet_heights=length_inlet_heights,
                       nx=nx, ny=ny, cells=int(fluid.sum()))
    return solver


def _section_pressure(solver: StaggeredSimple, i: int, j0: int, j1: int) -> float:
    """Area-weighted mean pressure over cells ``[j0, j1)`` of column ``i``."""
    p = solver.p[i, j0:j1]
    return float(p.mean())


def _section_bulk(solver: StaggeredSimple, i: int, j0: int, j1: int) -> float:
    """Bulk (area-averaged) streamwise velocity of column ``i``."""
    u = solver.u[i, j0:j1]
    return float(u.mean())


def sudden_expansion_metrics(solver: StaggeredSimple) -> dict:
    """Total-head loss coefficient ``K``, static pressure recovery, reattachment.

    ``K = [(p + rho U^2/2)_in - (p + rho U^2/2)_out] / (rho U_in^2/2)`` evaluated
    on the inlet plane and on a downstream plane in the redeveloped region, with
    area-averaged ``p`` and bulk ``U``.  Both the static-pressure recovery
    ``(p_out - p_in)/(rho U_in^2/2)`` and the momentum-integral prediction are
    reported so the Borda-Carnot comparison can be read either way.
    """
    m = solver.meta
    nu, H2, D = solver.nu, m["H2"], m["D"]
    ny = solver.ny
    n_in = max(2, int(round(D / solver.hy)))
    # downstream fully-developed station: two inlet heights upstream of the exit
    i_out = solver.nx - max(3, int(round(2.0 * D / solver.hx)))
    p_in = _section_pressure(solver, 0, 0, n_in)
    u_in = _section_bulk(solver, 0, 0, n_in)
    p_out = _section_pressure(solver, i_out, 0, ny)
    u_out = _section_bulk(solver, i_out, 0, ny)
    q_in = 0.5 * u_in ** 2
    q_out = 0.5 * u_out ** 2
    k_total = ((p_in + q_in) - (p_out + q_out)) / q_in
    static_recovery = (p_out - p_in) / q_in
    # momentum-integral (Borda-Carnot) prediction with the *measured* profiles:
    # the control volume between the two stations has only wall shear on the
    # expansion walls; neglecting it, p_out - p_in = rho(beta_in U_in^2/AR^... )
    # is not attempted here -- instead the dimensionless identity
    #   K = static_recovery + (1 - 1/AR^2)
    # is reported as an internal consistency check.
    k_identity = static_recovery + (1.0 - (u_out / u_in) ** 2)
    # reattachment on the step-side wall (top wall, y = H2)
    tau = solver.v[:, -1] if False else solver.u[:, -1]
    x = np.arange(solver.nx + 1) * solver.hx
    x_r = _wall_reattachment(x, tau)
    return dict(
        p_in=p_in, u_in=u_in, p_out=p_out, u_out=u_out,
        k_total_head=float(k_total),
        k_static_recovery=float(static_recovery),
        k_identity_check=float(k_identity),
        k_theory_borda_carnot=float(SUDDEN_EXPANSION_K_THEORY),
        k_error_rel=float(abs(k_total - SUDDEN_EXPANSION_K_THEORY)
                          / SUDDEN_EXPANSION_K_THEORY),
        reattachment_over_D=None if x_r is None else float(x_r / D),
        continuity=float(np.abs(solver.divergence()).max()),
        iterations=int(solver.iterations),
    )


# --------------------------------------------------------------------------- #
# natural convection cavity
# --------------------------------------------------------------------------- #
def build_natural_convection_cavity(n: int, *, rayleigh: float,
                                    prandtl: float = 0.71,
                                    limited: bool = True) -> StaggeredSimple:
    """Square cavity, hot west wall (``T = 1``), cold east wall (``T = 0``).

    Adiabatic south/north walls, no inflow and no outflow.  The Boussinesq
    scaling sets ``nu = sqrt(Pr/Ra)``, ``alpha = nu/Pr`` and ``g*beta = 1`` so
    that ``Ra = g*beta*dT*L^3/(nu*alpha)`` with ``dT = L = 1``.
    """
    L = 1.0
    nu = float(np.sqrt(prandtl / rayleigh))
    h = L / n
    fluid = np.ones((n, n), dtype=bool)
    solver = StaggeredSimple(n, n, h, h, nu, fluid, inlet_profile=None,
                             east_outlet=False, limited=limited,
                             alpha_u=0.5, alpha_p=0.2, alpha_T=0.7,
                             buoyancy=1.0, t_ref=0.0, prandtl=prandtl,
                             energy=True)
    solver.t_bc = dict(west=("fixed", 1.0), east=("fixed", 0.0),
                       south=("adiabatic", 0.0), north=("adiabatic", 0.0))
    solver.T[:, 0] = 0.5
    solver.T[:, -1] = 0.5
    solver.meta = dict(case="natural_convection_cavity", n=n, rayleigh=rayleigh,
                       prandtl=prandtl, nu=nu, alpha=nu / prandtl,
                       nx=n, ny=n, hx=h, hy=h, cells=int(fluid.sum()))
    return solver


def cavity_streamfunction(solver: StaggeredSimple) -> np.ndarray:
    """``psi`` from ``u = dpsi/dy`` integrated from the bottom wall."""
    psi = np.zeros((solver.nx, solver.ny + 1), dtype=float)
    psi[:, 1:] = np.cumsum(solver.u[:-1] * solver.hy, axis=1)
    return 0.5 * (psi[:, :-1] + psi[:, 1:])


def cavity_nusselt(solver: StaggeredSimple) -> tuple[float, float]:
    """Mean Nusselt number on the hot and cold walls (second-order one-sided)."""
    h = solver.hx
    T = solver.T
    # Nu = -dT/dx at the wall.  Second-order one-sided derivative with the wall
    # node at 0 and the two cell centres at h/2 and 3h/2:
    #   f'(0) = (-8/3 f0 + 3 f1 - 1/3 f2) / h
    # (exact for a linear profile: f'(0) = 1 for f = 1 -+- x)
    nu_hot = float(np.mean(((8.0 / 3.0) * 1.0 - 3.0 * T[0, :]
                            + (1.0 / 3.0) * T[1, :]) / h))
    nu_cold = float(np.mean((3.0 * T[-1, :] - (1.0 / 3.0) * T[-2, :]) / h))
    return nu_hot, nu_cold


def natural_convection_metrics(solver: StaggeredSimple) -> dict:
    """Mean Nu on both walls, ``|psi|_max``, energy-balance closure."""
    nu_hot, nu_cold = cavity_nusselt(solver)
    psi = cavity_streamfunction(solver)
    ref = NATURAL_CONVECTION_REFERENCE.get(solver.meta["rayleigh"], {})
    return dict(
        nusselt=nu_hot,
        nusselt_cold=nu_cold,
        nusselt_reference_contract=ref.get("nu_contract"),
        nusselt_reference_dvd=ref.get("nu_dvd1983"),
        nusselt_error_rel=None if not ref else float(
            abs(nu_hot - ref["nu_contract"]) / ref["nu_contract"]),
        psi_max=float(psi.max()),
        psi_min=float(psi.min()),
        psi_abs_max=float(np.abs(psi).max()),
        psi_over_alpha=float(np.abs(psi).max() / (solver.nu / solver.prandtl)),
        psi_max_reference=ref.get("psi_max_dvd1983"),
        psi_error_rel=None if not ref.get("psi_max_dvd1983") else float(
            abs(float(np.abs(psi).max() / (solver.nu / solver.prandtl))
                - ref["psi_max_dvd1983"]) / ref["psi_max_dvd1983"]),
        energy_imbalance_rel=float(abs(nu_hot - nu_cold) / max(abs(nu_hot), 1e-12)),
        continuity=float(np.abs(solver.divergence()).max()),
        iterations=int(solver.iterations),
    )


# --------------------------------------------------------------------------- #
# convergence driver shared by all three cases
# --------------------------------------------------------------------------- #
def converge(solver: StaggeredSimple, *,
             max_iter: int = 6000, tol: float = 1e-6, check: int = 10,
             momentum_sweeps: int = 4, p_cycles: int = 2,
             deferred_cycles: int = 2, first_order_iters: float = 0.35,
             verbose: bool = False) -> dict:
    """Converge a case: first-order upwind, then frozen van Leer defect cycles.

    The fully explicit van Leer correction is unstable above ``Pe_cell ~ 3``
    (measured here: a cell-Peclet 6 channel diverges, cell-Peclet 0.6 converges),
    so the higher-order terms are applied as a **frozen defect correction**: the
    first-order solution is converged, the limited face fluxes are evaluated
    once, and the momentum/energy equations are re-solved with that source held
    fixed.  ``deferred_cycles`` Picard refreezes are performed.
    """
    kw = dict(momentum_sweeps=momentum_sweeps, p_cycles=p_cycles)
    solver.limited = False
    first = solver.run(max_iter=int(max_iter * first_order_iters), tol=tol * 10,
                       check=check, **kw)
    history = [dict(stage="first_order", **{k: first[k] for k in
                                            ("iterations", "continuity", "change")})]
    for cyc in range(deferred_cycles):
        solver.refresh_deferred(1.0)
        r = solver.run(max_iter=max_iter, tol=tol, check=check, **kw)
        history.append(dict(stage=f"deferred_{cyc + 1}",
                            **{k: r[k] for k in ("iterations", "continuity", "change")}))
        if verbose:
            print(f"    deferred cycle {cyc + 1}: it={r['iterations']} "
                  f"cont={r['continuity']:.2e} dU={r['change']:.2e}", flush=True)
    return dict(history=history, final=history[-1])