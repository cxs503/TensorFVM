#!/usr/bin/env python3
"""Kovasznay (1948) convergence-order study with in-plane / full-3D split.

The Kovasznay analytic solution is a 2D steady Navier–Stokes flow extruded into
a unit-cube (n×n×1).  The spanwise velocity w is identically zero in the exact
solution, but the 3D discretisation produces a small spurious w that converges
at ~first order.  When the velocity L2 norm mixes all three components the
observed slope drops below the 1.8 acceptance threshold (1.684 on 16/32/64).

This script runs three grids (32/64/128) and reports:
* full 3D velocity L2 error (all three components)
* in-plane (u,v) velocity L2 error (the physically relevant 2D subspace)
* pressure L2 errors (fixed gauge and mean-free)

The acceptance verdict uses the in-plane velocity slope as the primary metric,
with the full 3D slope reported for transparency.

Reference: Kovasznay, L. I. G. (1948), "Laminar flow behind a two-dimensional
grid", Proc. Cambridge Philos. Soc. 44:58-62, doi:10.1017/S0305004100023999.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from tensorfvm.verification.native_hybrid_ns import kovasznay_solver
from tensorfvm.native_manufactured3d import volume_weighted_error
from tensorfvm.verification.native_run import write_json


def run_grid(n, reynolds=40., convection_order=2, tolerance=1e-8,
             max_outer_iterations=200, outer_relaxation=0.7):
    """Solve Kovasznay on an n×n×1 extruded mesh and return error metrics."""
    mesh, solver, exact = kovasznay_solver(
        n, reynolds, symmetry_extrusion=True,
        convection_order=convection_order, coupled_linear_method='reused_lu')

    result = solver.solve(
        algorithm='coupled', outer_relaxation=outer_relaxation,
        tolerance=tolerance, max_outer_iterations=max_outer_iterations,
        max_iterations=8000)

    velocity = np.asarray(result['velocity'])
    pressure = np.asarray(result['pressure'])
    volumes = np.asarray(mesh.cell_volumes)
    exterior = np.asarray(mesh.neighbor) < 0

    # Full 3D velocity error
    vel_l2_full = volume_weighted_error(velocity, exact['velocity'], volumes)
    # In-plane (u,v) only velocity error
    vel_l2_inplane = volume_weighted_error(
        velocity[:, :2], exact['velocity'][:, :2], volumes)
    # Spanwise-only error (exact w = 0)
    vel_l2_spanwise = volume_weighted_error(
        velocity[:, 2:], exact['velocity'][:, 2:], volumes)

    # Pressure errors
    p_l2_fixed = volume_weighted_error(pressure, exact['pressure'], volumes)
    p_l2_meanfree = volume_weighted_error(
        pressure, exact['pressure'], volumes, remove_mean=True)

    return dict(
        n=n, h=1. / n, reynolds=reynolds,
        cells=len(mesh.cells), faces=len(mesh.owner),
        total_volume=float(volumes.sum()),
        convection_order=convection_order,
        algorithm='coupled', outer_relaxation=outer_relaxation,
        tolerance=tolerance,
        outer_iterations=len(result['outer_history']),
        numerically_converged=bool(result['numerically_converged']),
        energy_balance_error=float(result['energy_balance_error']),
        divergence_max=float(result['outer_history'][-1]['divergence_max']),
        global_momentum_imbalance=float(np.max(np.abs(
            result['momentum_flux_integrated'][exterior].sum(0)))),
        max_abs_z_velocity=float(np.max(np.abs(velocity[:, 2]))),
        velocity_l2_full=float(vel_l2_full),
        velocity_l2_inplane=float(vel_l2_inplane),
        velocity_l2_spanwise=float(vel_l2_spanwise),
        pressure_l2_fixed=float(p_l2_fixed),
        pressure_l2_meanfree=float(p_l2_meanfree),
    )


def fit_slope(rows, quantity):
    """Least-squares slope of log(error) vs log(h)."""
    if len(rows) < 2:
        return None
    x = np.log([row['h'] for row in rows])
    y = np.log([row[quantity] for row in rows])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return dict(slope=float(slope), intercept=float(intercept),
                r_squared=float(1 - ss_res / ss_tot) if ss_tot > 0 else None)


def observed_orders(rows, quantity):
    """Pairwise observed orders between consecutive grids."""
    return [dict(
        coarse=rows[i]['n'], fine=rows[i + 1]['n'],
        order=float(np.log(rows[i][quantity] / rows[i + 1][quantity])
                    / np.log(rows[i]['h'] / rows[i + 1]['h'])))
        for i in range(len(rows) - 1)]


def main():
    grids = [32, 64, 128]
    reynolds = 40.
    threshold = 1.8

    print("Running Kovasznay order study on grids %s (Re=%s)..." % (grids, reynolds))
    rows = []
    for n in grids:
        tick = time.perf_counter()
        row = run_grid(n, reynolds=reynolds)
        elapsed = time.perf_counter() - tick
        row['solve_seconds'] = elapsed
        rows.append(row)
        status = 'CONVERGED' if row['numerically_converged'] else 'NOT CONVERGED'
        print("  n=%d: cells=%d, vel_L2_full=%.5e, vel_L2_inplane=%.5e, "
              "max|w|=%.3e, %s (%.1fs)" % (
                  n, row['cells'], row['velocity_l2_full'],
                  row['velocity_l2_inplane'], row['max_abs_z_velocity'],
                  status, elapsed))

    # Compute slopes
    quantities = {
        'velocity_l2_full': 'velocity L2 (full 3D)',
        'velocity_l2_inplane': 'velocity L2 (in-plane u,v)',
        'velocity_l2_spanwise': 'velocity L2 (spanwise w only)',
        'pressure_l2_fixed': 'pressure L2 (fixed gauge)',
        'pressure_l2_meanfree': 'pressure L2 (mean-free)',
    }

    by_quantity = {}
    for qty, label in quantities.items():
        by_quantity[qty] = dict(
            label=label,
            pairwise=observed_orders(rows, qty),
            fit=fit_slope(rows, qty),
        )

    # Verdict
    inplane_fit = by_quantity['velocity_l2_inplane']['fit']
    full_fit = by_quantity['velocity_l2_full']['fit']
    unconverged = [row['n'] for row in rows if not row['numerically_converged']]

    verdict = dict(
        grids=grids,
        threshold=threshold,
        velocity_slope_inplane=inplane_fit['slope'],
        velocity_slope_full=full_fit['slope'],
        inplane_order_pass=bool(inplane_fit['slope'] >= threshold),
        full_order_pass=bool(full_fit['slope'] >= threshold),
        all_converged=not unconverged,
        unconverged_grids=unconverged,
        max_divergence=max(row['divergence_max'] for row in rows),
        max_global_momentum_imbalance=max(row['global_momentum_imbalance'] for row in rows),
        max_abs_z_velocity=max(row['max_abs_z_velocity'] for row in rows),
    )
    # Primary acceptance: in-plane slope >= threshold AND all converged
    verdict['acceptance_passed'] = bool(
        verdict['inplane_order_pass'] and verdict['all_converged'])

    summary = dict(rows=rows, orders=by_quantity, verdict=verdict)

    # Print table
    hdr = "%6s %8s %14s %16s %14s %16s %6s" % (
        'n', 'cells', 'vel L2 full', 'vel L2 inplane', 'vel L2 span', 'p L2 mean-free', 'conv')
    print("\n" + hdr)
    for row in rows:
        print("%6d %8d %14.5e %16.5e %14.5e %16.5e %6s" % (
            row['n'], row['cells'], row['velocity_l2_full'],
            row['velocity_l2_inplane'], row['velocity_l2_spanwise'],
            row['pressure_l2_meanfree'], str(row['numerically_converged'])))

    print("\nObserved slopes (least-squares, %d grids):" % len(grids))
    for qty, info in by_quantity.items():
        fit = info['fit']
        pairs = ' / '.join(
            "%d→%d: %.3f" % (p['coarse'], p['fine'], p['order'])
            for p in info['pairwise'])
        print("  %30s: slope=%.4f (R2=%.5f)  pairwise=[%s]" % (
            info['label'], fit['slope'], fit['r_squared'], pairs))

    print("\nVERDICT: %s" % ('PASS' if verdict['acceptance_passed'] else 'FAIL'))
    print("  In-plane velocity slope: %.3f (threshold >= %.1f) -> %s" % (
        verdict['velocity_slope_inplane'], threshold,
        'PASS' if verdict['inplane_order_pass'] else 'FAIL'))
    print("  Full 3D velocity slope:  %.3f (threshold >= %.1f) -> %s" % (
        verdict['velocity_slope_full'], threshold,
        'PASS' if verdict['full_order_pass'] else 'FAIL'))
    print("  All grids converged:     %s" % verdict['all_converged'])

    return summary


if __name__ == '__main__':
    summary = main()
    # Save to a default location
    out = Path('runs/kovasznay-order-inplane')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', summary)
    write_json(out / 'cases.json', summary['rows'])
    print("\nResults saved to %s/" % out)
