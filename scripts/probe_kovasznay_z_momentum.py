#!/usr/bin/env python3
"""Locate the O(h) driver of the spurious spanwise (w) velocity in the
Kovasznay (1948) extruded order study.

Background / the debt being audited
-----------------------------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger, still open) asks for an observed
L2 convergence order >= 1.8 on the Kovasznay analytic solution at Re = 40.
The delivered round measured 1.684 for the *total* velocity norm and recorded
an honest FAIL.  ``scripts/audit_kovasznay_order_components.py`` (m16396)
split that norm: the in-plane (u,v) budget passes at 1.808, while the spanwise
(w) budget -- whose exact value is identically zero -- is only first order
(1.106) and its squared-error share *grows* with refinement (8.7% -> 20.1%).
The spanwise cell DOF carries a real momentum row (n=8 probe: rowsum ~0.4,
zero rows 0), so the w-velocity is *driven* by the discrete coupling, not a
massless artefact.

This script answers the open question from m16396: **which term of the discrete
momentum equations produces that O(h) driver?**  It evaluates the full
3-component momentum residual at the *exact* solution (w == 0 identically,
u/v = the analytic field) and decomposes it into the discrete contributions:

    R = (viscous + convection) @ state_exact + source
        - (divergence.T @ p_exact) - load

Key finding (n=8): the **z-momentum** residual at the exact solution is
machine-zero -- the z-momentum equation is trivially satisfied for w = 0.
The nonzero w the solver emits is therefore *not* driven by the z-momentum
equation; it is the response of the coupled system to a defect in the
**in-plane (x,y) momentum** equations, which couple into w through the
symmetric-deviatoric viscous stress and the face-mean reconstruction.  The
script reports the per-component defect so the guilty in-plane term is named.

Usage (n=8 is the cheap structural probe; n=16/32 confirm the scaling):

    PYTHONPATH=src python scripts/probe_kovasznay_z_momentum.py \
        --output runs/kovasznay-zprobe/n8 --n 8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.verification.native_hybrid_ns import kovasznay_fields, kovasznay_solver


def _exact_hybrid_state(mesh, solver, exact):
    """Construct the hybrid state that *is* the exact solution.

    Cell unknowns = the analytic velocity at cell centres (w == 0).
    Face unknowns = the exact boundary means on Dirichlet faces, and the
    exact face-centre field on free faces (O(h^2) consistent with the exact
    cell field -- the probe only needs the *pattern* of the driver).
    """
    N, F = len(mesh.cells), len(mesh.owner)
    state = np.zeros((N + F, 3))
    state[:N] = exact['velocity']
    dirichlet = solver.viscous.dirichlet_components
    state[N:][dirichlet] = solver.prescribed[3 * N:].reshape(F, 3)[dirichlet]
    free = ~dirichlet
    if free.any():
        state[N:][free] = kovasznay_fields(mesh.face_centers, 40.)['velocity'][free]
    return state.ravel()


def _stats(a):
    return dict(max=float(np.max(np.abs(a))), rms=float(np.sqrt(np.mean(a * a))),
                mean=float(np.mean(a)), l1=float(np.sum(np.abs(a))))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--reynolds', type=float, default=40.)
    p.add_argument('--outer-relaxation', type=float, default=.7)
    p.add_argument('--tolerance', type=float, default=1e-8)
    args = p.parse_args()

    tick = time.perf_counter()
    mesh, solver, exact = kovasznay_solver(
        args.n, args.reynolds, symmetry_extrusion=True,
        convection_order=2, coupled_linear_method='reused_lu')
    build_seconds = time.perf_counter() - tick
    N, F = len(mesh.cells), len(mesh.owner)

    solve_tick = time.perf_counter()
    result = solver.solve(algorithm='coupled', outer_relaxation=args.outer_relaxation,
                          tolerance=args.tolerance, max_outer_iterations=200,
                          callback=lambda row: print('outer', row, flush=True))
    solve_seconds = time.perf_counter() - solve_tick

    full, source = solver._current_system()
    state = solver.state
    p = solver.p
    load = solver.load

    # --- Full 3-component momentum residual at the CONVERGED state ----------
    r_cell = (full @ state + source - solver.divergence.T @ p - load).reshape(N + F, 3)[:N]
    visc = (solver.viscous.full_matrix @ state).reshape(N + F, 3)[:N]
    conv_sys = solver.convection.hybrid_system(
        solver.volume_flux(state), solver.face_mean.matrix,
        solver.velocity_gradient(state))
    conv = (conv_sys[0] @ state + conv_sys[1]).reshape(N + F, 3)[:N]
    pres = -(solver.divergence.T @ p).reshape(N + F, 3)[:N]

    # --- Full 3-component momentum residual at the EXACT solution -----------
    state_exact = _exact_hybrid_state(mesh, solver, exact)
    p_exact = exact['pressure']
    r_exact = (full @ state_exact + source - solver.divergence.T @ p_exact - load).reshape(N + F, 3)[:N]
    visc_exact = (solver.viscous.full_matrix @ state_exact).reshape(N + F, 3)[:N]
    conv_sys_exact = solver.convection.hybrid_system(
        solver.volume_flux(state_exact), solver.face_mean.matrix,
        solver.velocity_gradient(state_exact))
    conv_exact = (conv_sys_exact[0] @ state_exact + conv_sys_exact[1]).reshape(N + F, 3)[:N]
    pres_exact = -(solver.divergence.T @ p_exact).reshape(N + F, 3)[:N]

    # --- How the w-velocity is coupled into the in-plane momentum -----------
    # The symmetric-deviatoric viscous stress couples w into the x/y momentum
    # through tau_xz = mu (dw/dx + du/dz) and tau_yz = mu (dw/dy + dv/dz), and
    # through the trace term (2/3) mu div(u) on the diagonal.  Measure the
    # viscous x/y reaction that *changes* when w is turned on.
    w_on = state.copy(); w_off = state.copy()
    w_on[:3 * N:3] = 0  # placeholder; we instead measure the w-column coupling
    # viscous coupling of the w DOF into the x/y momentum rows:
    w_dof = np.zeros_like(state); w_dof[2::3] = state[2::3]  # w cell+face DOFs
    visc_w_coupling_xy = (solver.viscous.full_matrix @ w_dof).reshape(N + F, 3)[:N, :2]

    row = dict(
        n=args.n, h=1. / args.n, reynolds=args.reynolds,
        cells=N, faces=F,
        numerically_converged=bool(result['numerically_converged']),
        max_abs_w=float(np.max(np.abs(result['velocity'][:, 2]))),
        # converged-state momentum balance (should be ~0; terms balance)
        balance=dict(
            x=_stats(r_cell[:, 0]), y=_stats(r_cell[:, 1]), z=_stats(r_cell[:, 2]),
            viscous_x=_stats(visc[:, 0]), viscous_y=_stats(visc[:, 1]),
            convection_x=_stats(conv[:, 0]), convection_y=_stats(conv[:, 1]),
            pressure_x=_stats(pres[:, 0]), pressure_y=_stats(pres[:, 1]),
        ),
        # exact-state momentum defect (the O(h) driver, if nonzero)
        defect=dict(
            x=_stats(r_exact[:, 0]), y=_stats(r_exact[:, 1]), z=_stats(r_exact[:, 2]),
            viscous_x=_stats(visc_exact[:, 0]), viscous_y=_stats(visc_exact[:, 1]),
            convection_x=_stats(conv_exact[:, 0]), convection_y=_stats(conv_exact[:, 1]),
            pressure_x=_stats(pres_exact[:, 0]), pressure_y=_stats(pres_exact[:, 1]),
            # z-momentum is trivially satisfied at the exact solution
            z_exact_is_zero=bool(np.max(np.abs(r_exact[:, 2])) < 1e-14),
            # which in-plane term is the dominant defect
            dominant_inplane_term=(
                'viscous' if _stats(visc_exact[:, :2])['max'] >= max(
                    _stats(conv_exact[:, :2])['max'], _stats(pres_exact[:, :2])['max'])
                else 'convection' if _stats(conv_exact[:, :2])['max'] >= _stats(pres_exact[:, :2])['max']
                else 'pressure'),
            inplane_defect_scale_vs_w=float(
                np.max(np.abs(r_exact[:, :2])) / max(np.max(np.abs(result['velocity'][:, 2])), 1e-30)),
        ),
        # viscous coupling of the w DOF into the in-plane (x,y) momentum rows
        w_viscous_coupling_xy=_stats(visc_w_coupling_xy),
        build_seconds=build_seconds, solve_seconds=solve_seconds,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'cases.json').write_text(json.dumps([row], indent=2) + '\n')
    np.savez_compressed(args.output / 'zprobe.npz',
                        r_cell=r_cell, visc=visc, conv=conv, pres=pres,
                        r_exact=r_exact, visc_exact=visc_exact,
                        conv_exact=conv_exact, pres_exact=pres_exact,
                        w=result['velocity'][:, 2],
                        visc_w_coupling_xy=visc_w_coupling_xy,
                        cell_centers=np.asarray(mesh.cell_centers))
    print('RESULT ' + json.dumps({k: v for k, v in row.items()}, default=str), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
