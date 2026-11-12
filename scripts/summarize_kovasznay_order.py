#!/usr/bin/env python3
"""Summarise the Kovasznay grid-refinement study into observed orders.

Reads every ``cases.json`` produced by ``verify_kovasznay_order.py``, sorts by
mesh size, and reports

* pairwise observed orders ``p = log(e_coarse / e_fine) / log(h_coarse / h_fine)``,
* the least-squares slope of ``log e`` against ``log h`` over the requested
  grids, for velocity and for pressure (fixed gauge and mean-free gauge),
* the acceptance verdict against a threshold declared on the command line,
* optionally a markdown results section and a convergence figure, both written
  straight from the same numbers so the report can never drift from the data.

Usage:

    python scripts/summarize_kovasznay_order.py --runs runs/kovasznay-order/n32 \
        runs/kovasznay-order/n64 runs/kovasznay-order/n128 \
        --output runs/kovasznay-order/summary \
        --markdown docs/case-kovasznay-order-order-verification/RESULTS.md \
        --figure docs/case-kovasznay-order-order-verification/convergence.png
"""
import argparse
import json
from pathlib import Path
import numpy as np

from tensorfvm.verification.native_run import write_json

QUANTITIES = ('velocity_l2', 'pressure_l2', 'pressure_mean_free_l2')
LABELS = {'velocity_l2': 'velocity', 'pressure_l2': 'pressure (fixed gauge)',
          'pressure_mean_free_l2': 'pressure (mean-free)'}


def load(paths):
    rows = []
    for path in paths:
        directory = Path(path)
        candidate = directory / 'cases.json' if directory.is_dir() else directory
        rows.extend(json.loads(candidate.read_text()))
    return sorted(rows, key=lambda row: row['n'])


def observed_orders(rows, quantity):
    return [dict(coarse=rows[i]['n'], fine=rows[i + 1]['n'], order=float(
        np.log(rows[i][quantity] / rows[i + 1][quantity])
        / np.log(rows[i]['h'] / rows[i + 1]['h'])))
        for i in range(len(rows) - 1)]


def fit_slope(rows, quantity):
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


def markdown(summary, extra):
    verdict = summary['verdict']
    rows = summary['rows']
    lines = ['## 3. 计算结果（由 summarize_kovasznay_order.py 程序化注入，零手抄）', '',
             'Re = 40，κ = 2π，λ = %.6f，单位平方域 n×n×1（对称挤出），'
             '二阶对流，`reused_lu` 耦合直接解，Picard 松弛 0.7，tol 1e-8。' % extra['lambda'],
             '',
             '| n | h | cells | velocity L2 | pressure L2 (fixed gauge) | pressure L2 (mean-free) '
             '| outer it | converged |',
             '|---:|---:|---:|---:|---:|---:|---:|:--|']
    for row in rows:
        lines.append('| %d | %.5f | %d | %.5e | %.5e | %.5e | %d | %s |' % (
            row['n'], row['h'], row['cells'], row['velocity_l2'], row['pressure_l2'],
            row['pressure_mean_free_l2'], row['outer_iterations'],
            'yes' if row['numerically_converged'] else 'NO'))
    lines += ['', '### 观测阶（成对）与最小二乘斜率', '',
              '| quantity | 成对观测阶 | 三网格最小二乘斜率 | R² |', '|---|:--|---:|---:|']
    for quantity in QUANTITIES:
        block = summary['orders'][quantity]
        pairs = ' / '.join('%d→%d: %.3f' % (p['coarse'], p['fine'], p['order'])
                           for p in block['pairwise'])
        lines.append('| %s | %s | %.3f | %s |' % (
            LABELS[quantity], pairs, block['fit']['slope'],
            'n/a' if block['fit']['r_squared'] is None else '%.5f' % block['fit']['r_squared']))
    lines += ['', '### 判据逐条', '',
              '| 判据 | 阈值 | 实测 | 判定 |', '|---|---:|---:|:--|',
              '| 速度 L2 log-log 斜率（32/64/128） | ≥ %.1f | %.3f | %s |' % (
                  verdict['threshold'], verdict['velocity_slope'],
                  'PASS' if verdict['velocity_order_pass'] else 'FAIL'),
              '| 三套网格全部数值收敛 | 全真 | %s | %s |' % (
                  '全真' if verdict['all_converged'] else '不收敛网格 %s' % verdict['unconverged_grids'],
                  'PASS' if verdict['all_converged'] else 'FAIL'),
              '| 最大离散散度 | 越小越好（仅报告） | %.3e | n/a |' % verdict['max_divergence'],
              '| 全局动量不平衡 | 越小越好（仅报告） | %.3e | n/a |'
              % verdict['max_global_momentum_imbalance'],
              '| 挤出方向伪速度 max |w| | 越小越好（仅报告） | %.3e | n/a |'
              % verdict['max_abs_z_velocity'],
              '', '**总判定：%s**（velocity slope %.3f vs 阈值 %.1f；判据原文要求"L2 收敛阶 >=1.8"）' % (
                  'PASS' if verdict['acceptance_passed'] else 'FAIL',
                  verdict['velocity_slope'], verdict['threshold']), '']
    if verdict['sampling_bias']:
        lines += ['### 采样一致性控制（点值 vs 单元平均的精确解偏差）', '',
                  '| n | velocity bias | pressure bias |', '|---:|---:|---:|']
        for row in verdict['sampling_bias']:
            if 'velocity' in row:
                lines.append('| %d | %.3e | %.3e |' % (row['n'], row['velocity'], row['pressure']))
        lines += ['', '单元中心未知量是单元平均，此处偏差须比被测误差衰减更快，否则斜率是采样假象。', '']
    return '\n'.join(lines) + '\n'


def figure(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['rows']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    for quantity, style in zip(QUANTITIES, ('o-', 's--', '^:')):
        h = [row['h'] for row in rows]
        e = [row[quantity] for row in rows]
        fit = summary['orders'][quantity]['fit']
        axes[0].loglog(h, e, style, label='%s (slope %.2f)' % (LABELS[quantity], fit['slope']))
    axes[0].set(xlabel='h = 1/n', ylabel='volume-weighted relative L2 error',
                title='Kovasznay Re=40 order study')
    axes[0].grid(True, which='both', alpha=.3); axes[0].legend(fontsize=7)
    n = rows[-1]['n']
    fields = np.load(Path(summary['field_source']) / ('n%d-fields.npz' % n))
    colours = {0: 'tab:blue', 1: 'tab:orange'}
    interior = slice(1, -1)
    for axis, key, exact_key, title, label in (
            (1, 'velocity', 'exact_velocity', 'u_x: computed - exact (n=%d)' % n, 'u_x'),
            (2, 'pressure', 'exact_pressure', 'p: computed - exact (n=%d)' % n, 'p')):
        values = fields[key][:, 0] if key == 'velocity' else fields[key]
        reference = fields[exact_key][:, 0] if key == 'velocity' else fields[exact_key]
        grid = (values - reference).reshape(n, n)
        lim = np.max(np.abs(grid))
        image = axes[axis].imshow(grid, origin='lower', extent=(0, 1, 0, 1), cmap='coolwarm',
                                  vmin=-lim, vmax=lim, interpolation='nearest')
        axes[axis].set(title=title, xlabel='x', ylabel='y')
        fig.colorbar(image, ax=axes[axis], label='error in ' + label)
    del colours, interior
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=1.8)
    p.add_argument('--grids', type=int, nargs='*', default=[32, 64, 128])
    p.add_argument('--markdown', type=Path)
    p.add_argument('--figure', type=Path)
    args = p.parse_args()

    all_rows = load(args.runs)
    rows = [row for row in all_rows if row['n'] in set(args.grids)]
    if len(rows) < 3:
        raise SystemExit(f'need three grids from {args.grids}, got {[r["n"] for r in rows]}')

    by_quantity = {quantity: dict(pairwise=observed_orders(rows, quantity),
                                  fit=fit_slope(rows, quantity)) for quantity in QUANTITIES}
    unconverged = [row['n'] for row in rows if not row['numerically_converged']]
    velocity_fit = by_quantity['velocity_l2']['fit']
    verdict = dict(
        grids=[row['n'] for row in rows],
        threshold=args.threshold,
        velocity_slope=velocity_fit['slope'],
        velocity_order_pass=bool(velocity_fit['slope'] >= args.threshold),
        all_converged=not unconverged,
        unconverged_grids=unconverged,
        max_divergence=max(row['divergence_max'] for row in rows),
        max_global_momentum_imbalance=max(row['global_momentum_imbalance'] for row in rows),
        max_abs_z_velocity=max(row['max_abs_z_velocity'] for row in rows),
        sampling_bias=[dict(n=row['n'], **row.get('sampling_bias', {})) for row in rows],
    )
    verdict['acceptance_passed'] = bool(verdict['velocity_order_pass'] and verdict['all_converged'])
    summary = dict(rows=rows, orders=by_quantity, verdict=verdict,
                   field_source=str(Path(args.runs[-1]).resolve()))
    write_json(args.output / 'orders.json', summary)

    print(f'{"n":>6} {"cells":>8} {"vel L2":>12} {"p L2":>12} {"p mean-free":>12} {"conv":>6}')
    for row in rows:
        print(f'{row["n"]:>6} {row["cells"]:>8} {row["velocity_l2"]:>12.5e} '
              f'{row["pressure_l2"]:>12.5e} {row["pressure_mean_free_l2"]:>12.5e} '
              f'{str(row["numerically_converged"]):>6}')
    for quantity in QUANTITIES:
        fit = by_quantity[quantity]['fit']
        print(f'{LABELS[quantity]:>22}: slope {fit["slope"]:.4f} (R2 {fit["r_squared"]}) '
              f'pairwise {[round(q["order"], 3) for q in by_quantity[quantity]["pairwise"]]}')
    if args.markdown:
        kappa = 2 * np.pi
        reynolds = rows[0]['reynolds']
        lam = -2 * kappa ** 2 / (reynolds + np.sqrt(reynolds ** 2 + 4 * kappa ** 2))
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary, {'lambda': lam}))
    if args.figure:
        figure(summary, args.figure)
    print('VERDICT', json.dumps(verdict, indent=2))
    return 0 if verdict['acceptance_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())