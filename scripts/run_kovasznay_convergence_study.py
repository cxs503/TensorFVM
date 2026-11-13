#!/usr/bin/env python3
"""CASE-FVM-KOVASZNAY-ORDER: Full convergence study on 32/64/128 grids.

Runs the Kovasznay analytic solution (Re=40) on three grid levels,
computes volume-weighted L2 errors, fits convergence order, and
produces machine-checkable artifacts with sha256 traceability.

Usage:
    PYTHONPATH=src python scripts/run_kovasznay_convergence_study.py \
        --output runs/kovasznay-order-m17078
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.verification.native_hybrid_ns import kovasznay_solver


def sha256_of(obj):
    """Compute sha256 of a JSON-serializable object for traceability."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def volume_weighted_l2(computed, exact, volumes, remove_mean=False):
    """Volume-weighted relative L2 error."""
    w = volumes.reshape((len(volumes),) + (1,) * (computed.ndim - 1))
    e = computed - exact
    if remove_mean:
        mean_e = np.sum(w * e) / np.sum(w)
        e = e - mean_e
    return float(np.sqrt(np.sum(w * np.square(e)) / np.sum(w * np.square(exact))))


def observed_order(e_coarse, e_fine, h_ratio=2.0):
    """Two-level observed order: log(e_coarse/e_fine) / log(h_ratio)."""
    if e_coarse <= 0 or e_fine <= 0 or not (np.isfinite(e_coarse) and np.isfinite(e_fine)):
        return None
    return float(np.log(e_coarse / e_fine) / np.log(h_ratio))


def fit_order(h_vals, e_vals):
    """Least-squares slope of log(error) vs log(h)."""
    usable = [(h, e) for h, e in zip(h_vals, e_vals) if np.isfinite(e) and e > 0]
    if len(usable) < 2:
        return None
    h = np.log([u[0] for u in usable])
    e = np.log([u[1] for u in usable])
    slope, intercept = np.polyfit(h, e, 1)
    fitted = slope * h + intercept
    residual = float(np.max(np.abs(np.asarray(e) - fitted)))
    return dict(slope=float(slope), max_log_residual=residual, levels=len(usable))


def run_level(n, reynolds=40., tolerance=1e-8, relaxation=0.7, max_outer=200):
    """Solve Kovasznay on n×n×1 mesh and return error metrics."""
    started = time.time()
    mesh, solver, exact = kovasznay_solver(
        n, reynolds, symmetry_extrusion=True, symmetry_closure='zero_gradient',
        convection_order=2, coupled_linear_method='reused_lu')
    result = solver.solve(
        algorithm='coupled', outer_relaxation=relaxation, tolerance=tolerance,
        max_outer_iterations=max_outer)
    elapsed = time.time() - started

    velocity = np.asarray(result['velocity'])
    pressure = np.asarray(result['pressure'])
    volumes = np.asarray(mesh.cell_volumes)

    # Velocity errors
    vel_l2_full = volume_weighted_l2(velocity, exact['velocity'], volumes)
    vel_l2_inplane = volume_weighted_l2(velocity[:, :2], exact['velocity'][:, :2], volumes)
    # Spanwise: absolute error since exact w = 0
    w_err = velocity[:, 2]  # exact w = 0
    vel_l2_spanwise = float(np.sqrt(np.sum(volumes * w_err**2) / np.sum(volumes)))
    max_abs_w = float(np.max(np.abs(w_err)))

    # Pressure error (mean-free)
    p_err = pressure - exact['pressure']
    mean_p_err = np.sum(volumes * p_err) / np.sum(volumes)
    p_err = p_err - mean_p_err
    p_l2 = float(np.sqrt(np.sum(volumes * p_err**2) / np.sum(volumes * exact['pressure']**2)))

    outer = solver.outer_history[-1] if solver.outer_history else {}

    record = dict(
        n=int(n), h=1.0 / n, reynolds=reynolds,
        cells=int(solver.N), faces=int(solver.F),
        converged=bool(result['numerically_converged']),
        outer_iterations=len(solver.outer_history),
        final_divergence_max=float(outer.get('divergence_max', float('nan'))),
        final_momentum_residual=float(outer.get('current_momentum_residual',
                                                  outer.get('momentum_relative_residual', float('nan')))),
        seconds=round(elapsed, 2),
        velocity_l2_full=vel_l2_full,
        velocity_l2_inplane=vel_l2_inplane,
        velocity_l2_spanwise=vel_l2_spanwise,
        max_abs_w=max_abs_w,
        pressure_l2_meanfree=p_l2,
    )
    record['sha256'] = sha256_of({k: v for k, v in record.items() if k != 'sha256'})
    return record


def generate_plot(records, output_dir):
    """Generate log-log convergence plot and save as PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    h_vals = [r['h'] for r in records]
    keys_labels = [
        ('velocity_l2_inplane', 'velocity L2 (in-plane u,v)', 's-', 'tab:blue'),
        ('velocity_l2_full', 'velocity L2 (full 3D)', 'o-', 'tab:orange'),
        ('pressure_l2_meanfree', 'pressure L2 (mean-free)', '^--', 'tab:green'),
    ]

    for key, label, style, color in keys_labels:
        e_vals = [r[key] for r in records]
        ax.loglog(h_vals, e_vals, style, color=color, label=label, linewidth=2, markersize=8)

    # Reference lines
    h_ref = np.array([1/128, 1/32])
    e_ref = records[-1]['velocity_l2_inplane']
    ax.loglog(h_ref, e_ref * (h_ref / h_ref[-1])**1, 'k--', alpha=0.3, label='1st order')
    ax.loglog(h_ref, e_ref * (h_ref / h_ref[-1])**2, 'k-.', alpha=0.3, label='2nd order')

    ax.set(xlabel='h = 1/n', ylabel='volume-weighted relative L2 error',
           title='Kovasznay Re=40 convergence order study\n(zero_gradient symmetry closure)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

    # Add slope annotation
    orders = pair_orders(records)
    if orders:
        finest = orders[-1]
        ax.text(0.02, 0.02,
                'In-plane slope (64→128): %.3f\nTotal slope (64→128): %.3f' % (
                    finest.get('inplane', 0), finest.get('total', 0)),
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    out_path = output_dir / 'convergence_loglog.png'
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def pair_orders(records):
    """Two-level observed orders for consecutive pairs."""
    rows = []
    for coarse, fine in zip(records, records[1:]):
        ratio = coarse['h'] / fine['h']
        rows.append(dict(
            pair="%d->%d" % (coarse['n'], fine['n']),
            h_ratio=float(ratio),
            total=observed_order(coarse['velocity_l2_full'], fine['velocity_l2_full'], ratio),
            inplane=observed_order(coarse['velocity_l2_inplane'], fine['velocity_l2_inplane'], ratio),
            spanwise=observed_order(coarse['velocity_l2_spanwise'], fine['velocity_l2_spanwise'], ratio),
            pressure=observed_order(coarse['pressure_l2_meanfree'], fine['pressure_l2_meanfree'], ratio),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Kovasznay convergence study')
    parser.add_argument('--levels', type=int, nargs='+', default=[32, 64, 128],
                        help='Grid levels (default: 32 64 128)')
    parser.add_argument('--output', type=str, default='runs/kovasznay-order-m17078',
                        help='Output directory')
    parser.add_argument('--reynolds', type=float, default=40.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CASE-FVM-KOVASZNAY-ORDER Convergence Study")
    print("Re = %.1f, levels = %s" % (args.reynolds, args.levels))
    print("=" * 60)

    records = []
    for n in args.levels:
        print("\n--- Solving n=%d ---" % n)
        record = run_level(n, reynolds=args.reynolds)
        records.append(record)
        print("  cells=%d, converged=%s, time=%.1fs" % (record['cells'], record['converged'], record['seconds']))
        print("  vel_l2_full=%.6e, vel_l2_inplane=%.6e, pres_l2=%.6e" % (
            record['velocity_l2_full'], record['velocity_l2_inplane'], record['pressure_l2_meanfree']))
        print("  max|w|=%.2e" % record['max_abs_w'])

        # Save individual level record
        level_path = output_dir / ('level_n%d.json' % n)
        level_path.write_text(json.dumps(record, indent=2, default=float) + '\n')
        print("  Saved: %s (sha256: %s)" % (level_path.name, record['sha256'][:16]))

    # Compute orders
    orders = pair_orders(records)
    fits = {
        'velocity_l2_full': fit_order([r['h'] for r in records], [r['velocity_l2_full'] for r in records]),
        'velocity_l2_inplane': fit_order([r['h'] for r in records], [r['velocity_l2_inplane'] for r in records]),
        'velocity_l2_spanwise': fit_order([r['h'] for r in records], [r['velocity_l2_spanwise'] for r in records]),
        'pressure_l2_meanfree': fit_order([r['h'] for r in records], [r['pressure_l2_meanfree'] for r in records]),
    }

    # Generate plot
    plot_path = generate_plot(records, output_dir)
    print("\nPlot saved: %s" % plot_path)

    # Summary
    summary = dict(
        mission='m17078',
        case='CASE-FVM-KOVASZNAY-ORDER',
        reynolds=args.reynolds,
        levels=args.levels,
        records=records,
        pair_orders=orders,
        fitted_orders=fits,
        acceptance_threshold=1.8,
    )

    # Verdict
    finest_order = orders[-1]['inplane'] if orders else None
    lsq_slope = (fits['velocity_l2_inplane'] or {}).get('slope')
    verdict = 'PASS' if (finest_order is not None and finest_order >= 1.8) else 'FAIL'

    summary['verdict'] = verdict
    summary['finest_pair_order'] = finest_order
    summary['lsq_slope'] = lsq_slope

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + '\n')
    summary['sha256'] = sha256_of({k: v for k, v in summary.items() if k != 'sha256'})
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + '\n')

    print("\n" + "=" * 60)
    print("CONVERGENCE ORDER RESULTS")
    print("=" * 60)
    print("\nPair orders:")
    for o in orders:
        print("  %s: total=%.3f, inplane=%.3f, spanwise=%.3f, pressure=%.3f" % (
            o['pair'], o['total'], o['inplane'], o['spanwise'], o['pressure']))

    print("\nFitted slopes (least-squares):")
    for key, fit in fits.items():
        if fit:
            print("  %s: %.3f (residual=%.4e, levels=%d)" % (key, fit['slope'], fit['max_log_residual'], fit['levels']))

    print("\nVerdict: %s (in-plane order on finest pair = %.3f, threshold = 1.8)" % (
        verdict, finest_order if finest_order else 'N/A'))

    if verdict == 'FAIL':
        print("\n*** HONEST FAIL ***")
        print("Error attribution:")
        print("  - Spanwise w artifact: max|w| = %.2e (exact w = 0)" % records[-1]['max_abs_w'])
        print("  - Spanwise L2 error: %.6e" % records[-1]['velocity_l2_spanwise'])
        print("  - Spanwise share of total squared error: %.1f%%" % (
            100 * records[-1]['velocity_l2_spanwise']**2 / records[-1]['velocity_l2_full']**2))
        print("  - Convection term: 2nd-order hybrid scheme")
        print("  - Pressure-velocity coupling: coupled algorithm with reused LU")
        print("  - Boundary treatment: zero_gradient symmetry closure")

    print("\nArtifacts:")
    print("  %s/summary.json" % output_dir)
    print("  %s/convergence_loglog.png" % output_dir)
    for n in args.levels:
        print("  %s/level_n%d.json" % (output_dir, n))

    return summary


if __name__ == '__main__':
    main()
