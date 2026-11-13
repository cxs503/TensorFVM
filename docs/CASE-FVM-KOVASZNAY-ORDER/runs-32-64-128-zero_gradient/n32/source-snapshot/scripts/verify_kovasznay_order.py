#!/usr/bin/env python3
"""Kovasznay (1948) exact-solution mesh-refinement order study.

One grid per invocation so the three grids of the acceptance criterion can run
as independent jobs:

    python scripts/verify_kovasznay_order.py --output runs/order/n32 --n 32

Per grid the runner records the volume-weighted relative L2 error of the
computed velocity and pressure against the analytic field evaluated at cell
centres, together with the discrete conservation identities (energy balance,
interface traction defect, global momentum imbalance, net flux) so the log-log
slope computed later by ``summarize_kovasznay_order.py`` is bound to a solution
that is actually converged and conservative.

Reference: Kovasznay, L. I. G. (1948), "Laminar flow behind a two-dimensional
grid", Proc. Cambridge Philos. Soc. 44:58-62, doi:10.1017/S0305004100023999.
"""
import argparse
from pathlib import Path
import time
import numpy as np

from tensorfvm.verification.native_hybrid_ns import kovasznay_fields, kovasznay_solver
from tensorfvm.verification.native_run import NativeRunEvidence, write_json
from tensorfvm.native_manufactured3d import volume_weighted_error


def sampling_bias(n, reynolds, subsamples=6):
    """Relative L2 gap between the pointwise and the cell-averaged exact field.

    A cell-centred finite-volume unknown approximates a *cell average*, so the
    point value at the centroid is only a legitimate comparator while that gap
    decays at least at the measured order.  Measuring it keeps the reported
    slopes from being an artefact of the sampling rule itself.
    """
    mesh, _, exact = kovasznay_solver(n, reynolds, symmetry_extrusion=True)
    centres = np.asarray(mesh.cell_centers)
    h = 1. / n
    offsets = (np.arange(subsamples) + .5) / subsamples - .5
    gap = {}
    for key in ('velocity', 'pressure'):
        average = np.zeros_like(exact[key])
        for dx in offsets:
            for dy in offsets:
                average += kovasznay_fields(centres + np.array([dx * h, dy * h, 0.]), reynolds)[key]
        average /= subsamples ** 2
        gap[key] = float(np.linalg.norm(average - exact[key]) / np.linalg.norm(exact[key]))
    return gap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--reynolds', type=float, default=40.)
    p.add_argument('--convection-order', type=int, default=2)
    p.add_argument('--outer-relaxation', type=float, default=.7)
    p.add_argument('--tolerance', type=float, default=1e-8)
    p.add_argument('--max-outer-iterations', type=int, default=120)
    p.add_argument('--max-iterations', type=int, default=6000)
    p.add_argument('--symmetry-closure', choices=('free', 'zero_gradient'), default='free',
                   help="closure of the two extrusion planes: 'free' (historical, leaves a "
                        "spurious O(h) spanwise velocity) or 'zero_gradient' (the symmetry "
                        "condition du/dz=dv/dz=0, see tensorfvm.verification.kovasznay_symmetry)")
    p.add_argument('--skip-sampling-check', action='store_true')
    args = p.parse_args()
    if args.n < 4:
        p.error('n must be at least 4')
    evidence = NativeRunEvidence(args.output, __file__)
    out = args.output
    tick = time.perf_counter()

    mesh, solver, exact = kovasznay_solver(args.n, args.reynolds, symmetry_extrusion=True,
                                           convection_order=args.convection_order,
                                           coupled_linear_method='reused_lu',
                                           symmetry_closure=args.symmetry_closure)
    build_seconds = time.perf_counter() - tick
    volumes = mesh.cell_volumes
    total_volume = float(volumes.sum())
    if not np.isclose(total_volume, 1., atol=1e-12):
        raise RuntimeError(f'unit-domain volume closure violated: {total_volume!r}')
    exterior = np.asarray(mesh.neighbor) < 0

    solve_tick = time.perf_counter()
    result = solver.solve(algorithm='coupled', outer_relaxation=args.outer_relaxation,
                          tolerance=args.tolerance,
                          max_outer_iterations=args.max_outer_iterations,
                          max_iterations=args.max_iterations,
                          callback=lambda row: print('outer', row, flush=True))
    solve_seconds = time.perf_counter() - solve_tick

    velocity, pressure, flux = result['velocity'], result['pressure'], np.asarray(result['flux'])
    # ``flux`` is the discrete face flux vector; its conservation is reported via the
    # solver's own divergence_max residual rather than a hand-rolled surface sum.
    row = dict(
        n=args.n, h=1. / args.n, reynolds=args.reynolds,
        cells=len(mesh.cells), faces=len(mesh.owner), total_volume=total_volume,
        convection_order=args.convection_order, coupled_linear_method='reused_lu',
        symmetry_closure=args.symmetry_closure,
        algorithm='coupled', outer_relaxation=args.outer_relaxation, tolerance=args.tolerance,
        outer_iterations=len(result['outer_history']),
        inner_iterations=len(result['history']),
        numerically_converged=bool(result['numerically_converged']),
        final_residuals=result['outer_history'][-1],
        energy_balance_error=result['energy_balance_error'],
        convection_energy_identity_error=result['convection_energy_identity_error'],
        interface_traction_defect_max=result['interface_traction_defect_max'],
        global_momentum_imbalance=float(np.max(np.abs(
            result['momentum_flux_integrated'][exterior].sum(0)))),
        divergence_max=float(result['outer_history'][-1]['divergence_max']),
        max_abs_z_velocity=float(np.max(np.abs(velocity[:, 2]))),
        velocity_l2=volume_weighted_error(velocity, exact['velocity'], volumes),
        pressure_l2=volume_weighted_error(pressure, exact['pressure'], volumes),
        pressure_mean_free_l2=volume_weighted_error(pressure, exact['pressure'], volumes,
                                                    remove_mean=True),
        build_seconds=build_seconds, solve_seconds=solve_seconds,
    )
    if not args.skip_sampling_check:
        row['sampling_bias'] = sampling_bias(args.n, args.reynolds)
    write_json(out / 'cases.json', [row])
    for kind, data in [('outer', result['outer_history']), ('inner', result['history']),
                       ('linear', solver.coupled_linear_history)]:
        write_json(out / f'n{args.n}-{kind}.json', data)
    np.savez_compressed(out / f'n{args.n}-fields.npz',
                        cell_centers=np.asarray(mesh.cell_centers), cell_volumes=volumes,
                        velocity=velocity, pressure=pressure,
                        exact_velocity=exact['velocity'], exact_pressure=exact['pressure'],
                        flux=flux,
                        total_traction_integrated=result['total_traction_integrated'])
    evidence.verify_sources()
    write_json(out / 'summary.json', dict(n=args.n, reynolds=args.reynolds,
                                          symmetry_closure=args.symmetry_closure,
                                          numerically_converged=row['numerically_converged'],
                                          runtime_seconds=time.perf_counter() - tick,
                                          source_integrity_verified=True))
    print('RESULT', row, flush=True)
    return 0 if row['numerically_converged'] else 2


if __name__ == '__main__':
    raise SystemExit(main())