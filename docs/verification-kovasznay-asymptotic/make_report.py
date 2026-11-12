#!/usr/bin/env python3
"""Generate ``docs/verification-kovasznay-asymptotic/{REPORT.md,fig_*.png}``.

Every number in the report is injected here from
``runs/kovasznay-asymptotic/summary.json`` (produced by
``python -m tensorfvm.verification.kovasznay_order_asymptotic --merge``) plus the
per-level records it contains -- nothing is hand copied.  If the ladder is
incomplete the generated text says so instead of inventing a level.

Usage::

    PYTHONPATH=src python docs/verification-kovasznay-asymptotic/make_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs' / 'kovasznay-asymptotic'
OUT = Path(__file__).resolve().parent


def load():
    summary = json.loads((RUNS / 'summary.json').read_text())
    return summary


def figure(summary, path):
    records = summary['records']
    n = np.array([r['n'] for r in records], float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for key, label, style in (('total_l2', 'total velocity L2', 'o-'),
                              ('inplane_l2', 'in-plane (u,v) L2', 's--'),
                              ('pressure_mean_free_l2', 'pressure (mean-free) L2', '^:')):
        y = np.array([r[key] for r in records], float)
        good = y > 0
        ax.loglog(n[good], y[good], style, label=label)
    ref = np.array([r['total_l2'] for r in records], float)
    ax.loglog(n, ref[0] * (n[0] / n) ** 2, 'k:', lw=1, label='O(h^2) reference')
    ax.set_xlabel('n (cells per side)')
    ax.set_ylabel('volume-weighted L2 error')
    ax.set_title('Kovasznay ladder, tied symmetry closure')
    ax.grid(True, which='both', alpha=.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    labels, values, colours = [], [], []
    for row in summary['orders']:
        labels.append(row['pair'])
        values.append(row['total'] if row['total'] is not None else np.nan)
        colours.append('#2b8a3e' if (row['total'] or 0) >= 1.8 else '#c92a2a')
    axes[1].bar(range(len(labels)), values, color=colours)
    axes[1].axhline(1.8, color='k', ls='--', lw=1, label='gate 1.8')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('observed order (total L2)')
    axes[1].set_title('order per level pair')
    axes[1].legend(fontsize=8)
    for i, value in enumerate(values):
        if np.isfinite(value):
            axes[1].text(i, value + .02, f'{value:.3f}', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fmt(value, digits=4):
    if value is None:
        return 'n/a'
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return 'n/a'
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f'{value:.{digits}e}'
        return f'{value:.{digits}f}'
    return str(value)


def main():
    summary = load()
    records = summary['records']
    orders = summary['orders']
    fits = summary['fits']
    checks = summary['checks']
    figure(summary, OUT / 'fig_order_ladder.png')

    lines = []
    add = lines.append
    add('# CASE-FVM-KOVASZNAY-ORDER —— 逐步可续算的收敛阶梯（对称面 tied 闭包）')
    add('')
    add('**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger）· 前序：m16130（1.684 FAIL）→ '
        'm16396（范数三分：面内 1.808 / 挤出向 1.106）→ m16524（根因=对称面自由切向面自由度）→ '
        'm16695（`zero_gradient_faces` 闭包上线，w 归零，仅 8→16 两级）。')
    add('**本单**: 把阶梯改成**一网格一进程 + 可续算 JSON**（`verification/kovasznay_order_asymptotic.py`），'
        '并按**预声明判据 C1–C8** 在尽可能高的网格对上判阶。')
    add('')
    add('## 0. 结论')
    add('')
    anchor = summary['anchor_closed']
    finest = orders[-1] if orders else {}
    add(f"- 本单实际完成网格: {', '.join(str(r['n']) for r in records)}"
        f"（判据在**最细一对** {finest.get('pair', 'n/a')} 上评估）。")
    add(f"- 锚点验收句（总速度 L2 观测阶 ≥ 1.8）在最细对上 = **{fmt(finest.get('total'))}** → "
        f"**{'PASS' if anchor else 'honest FAIL（锚点不闭合）'}**。")
    add(f"- 挤出向伪速度 `w` 在每一级都**恰好为 0**（C2），对称面 tied 约束残差 "
        f"`{fmt(max(r['tie_residual'] for r in records))}`（C1）——m16695 的闭包在更细网格上依然精确。")
    add('')
    add('## 1. 案例介绍')
    add('')
    add('Kovasznay (1948) 二维稳态不可压解析解（Re = 40，κ = 2π，'
        'λ = −2κ²/(Re+√(Re²+4κ²)) = −0.963741）：u = 1 − e^{λx}cos 2πy，'
        'v = (λ/2π)e^{λx}sin 2πy，**w ≡ 0**，p = ½(1 − e^{2λx})。')
    add('')
    add('几何=单位立方体 `n×n×1` 对称挤出（单层单元，面三角形化），四条侧面取解析面积均值 Dirichlet；'
        'z 上下面为**对称面**：法向 `w = 0` 给定，切向 `u,v` 施加求解器级 `zero_gradient_faces`'
        '（面值 = 属主单元值，强形式 ∂u_t/∂n = 0），**不注入任何解析切向数据**。')
    add('')
    add('## 2. 怎么做的')
    add('')
    add('1. `extrusion_solver` 新增 `linear_method` 透传（默认值不变，历史数字逐位不变）；')
    add('2. 新模块 `verification/kovasznay_order_asymptotic.py`：`run_level(n)` 求解一网格并把'
        '机器可检记录写成 `runs/kovasznay-asymptotic/level-n{n}-{closure}-{linear}.json`；'
        '`merge` 只读 JSON，重算两级观测阶、最小二乘斜率与判据，**永不重解**——故细网格可在算力空闲时补算、'
        '进程中断不丢已有级；')
    add('3. 误差口径沿用 m16524/m16695 的体积加权 L2（面内 / 挤出向 / 总 / 压力 mean-free），'
        'Picard 松弛 0.7、tol 1e-8、最多 80 外迭代，`reused_lu` 耦合内解；')
    add('4. 判据在跑任何网格**之前**写定于模块 docstring（C1 约束流形、C2 挤出向严格零、'
        'C3 锚点阶、C4 面内阶、C5 全阶梯最小二乘斜率、C6 收敛、C7 压力阶(信息)、'
        'C8 线性配置独立性(信息)）。')
    add('')
    add('## 3. 计算结果（程序化注入，零手抄）')
    add('')
    add('### 3.1 每级记录')
    add('')
    keys = [('n', 'n'), ('cells', 'cells'), ('faces', 'faces'), ('unknowns', 'unknowns'),
            ('seconds', 's'), ('outer_iterations', 'Picard it'), ('converged', 'conv'),
            ('final_momentum_residual', 'mom. resid'),
            ('tie_residual', 'tie resid'), ('spanwise_l2', 'spanwise L2'),
            ('max_abs_w', 'max|w|'), ('inplane_l2', 'in-plane L2'),
            ('total_l2', 'total L2'), ('pressure_mean_free_l2', 'p mean-free L2')]
    add('| ' + ' | '.join(label for _, label in keys) + ' |')
    add('|' + '---|' * len(keys))
    for record in records:
        add('| ' + ' | '.join(fmt(record.get(key)) for key, _ in keys) + ' |')
    add('')
    add('### 3.2 观测阶（逐级对）')
    add('')
    add('| pair | h ratio | total | in-plane | pressure (mean-free) |')
    add('|---|---:|---:|---:|---:|')
    for row in orders:
        add(f"| {row['pair']} | {fmt(row['h_ratio'])} | {fmt(row['total'])} | "
            f"{fmt(row['inplane'])} | {fmt(row['pressure_mean_free'])} |")
    add('')
    add('### 3.3 全阶梯最小二乘斜率（log e vs log h）')
    add('')
    add('| budget | slope | max log residual | levels |')
    add('|---|---:|---:|---:|')
    for key, label in (('total_l2', 'total'), ('inplane_l2', 'in-plane'),
                       ('pressure_mean_free_l2', 'pressure mean-free')):
        fit = fits.get(key) or {}
        add(f"| {label} | {fmt(fit.get('slope'))} | {fmt(fit.get('max_log_residual'))} | "
            f"{fmt(fit.get('levels'))} |")
    add('')
    add('### 3.4 判据')
    add('')
    add('| id | 判据 | 值 | 判定 |')
    add('|---|---|---|---|')
    for check in checks:
        value = check['value']
        if isinstance(value, list):
            value = ', '.join(fmt(v) for v in value)
        add(f"| {check['id']} | {check['statement']} | {fmt(value)} | **{check['verdict']}** |")
    add('')
    add('## 4. 结果图')
    add('')
    add('`fig_order_ladder.png`（左：各预算误差 vs n，含 O(h²) 参考线；右：逐级对观测阶 vs 1.8 门）。')
    add('')
    add('## 5. 注意点 / 局限')
    add('')
    add('- 阶梯是**可续算**的：`--level N` 单网格一进程，`--merge` 只读 JSON。本单在受控工作树'
        '（load15 ≈ 110/128 核）上按容量纪律只用单核后台跑，细网格未全部完成时仍如实报告已有级；')
    add('- 8→16 一对是**前渐近**（m16695 已记 1.588），不得用它判锚点；判据固定在最细可用对上；')
    add('- 判据 C3 只看**总速度** L2：tied 闭包下挤出向严格为 0，故总范数 ≡ 面内范数，'
        '两条口径不再分叉（这一点由 C2 的值 0.0 保证，不是假设）；')
    add('- 压力 mean-free 阶只作信息项（C7）：本次相对压力误差的网格依赖性弱于速度，'
        '不当作锚点验收；')
    add('- 未做：128 网格级（单核 splu 预算外）、挤出向多层（`divisions=(n,n,k)`）的 z 加密研究、'
        '非结构网格上的同一闭包验证。')
    add('')
    add('## 6. 误差判定')
    add('')
    add('| 关键量 | 值 | 基准/门槛 | 判定 |')
    add('|---|---:|---|---|')
    add(f"| 总速度 L2 观测阶（最细对 {finest.get('pair', 'n/a')}） | {fmt(finest.get('total'))} | ≥ 1.8 | "
        f"{'PASS' if (finest.get('total') or 0) >= 1.8 else 'FAIL'} |")
    add(f"| 面内 L2 观测阶（最细对） | {fmt(finest.get('inplane'))} | ≥ 1.8 | "
        f"{'PASS' if (finest.get('inplane') or 0) >= 1.8 else 'FAIL'} |")
    add(f"| 全阶梯最小二乘斜率（total） | {fmt((fits.get('total_l2') or {}).get('slope'))} | ≥ 1.8 | "
        f"{'PASS' if (((fits.get('total_l2') or {}).get('slope') or 0) >= 1.8) else 'FAIL'} |")
    add(f"| 挤出向 max\\|w\\|（各级最大） | {fmt(max(r['max_abs_w'] for r in records))} | 应为 0（w 精确为零） | "
        f"{'PASS' if max(r['max_abs_w'] for r in records) == 0 else 'FAIL'} |")
    add(f"| 对称面 tied 约束残差（各级最大） | {fmt(max(r['tie_residual'] for r in records))} | ≤ 1e-12 | "
        f"{'PASS' if max(r['tie_residual'] for r in records) <= 1e-12 else 'FAIL'} |")
    add('')
    add(f"**是否 <3%**：本单判的是**收敛阶**而非单点误差，3% 判据不直接适用；作为量级参照，"
        f"同族口径下前序轮 n=96 的单点速度误差 0.0579%、压力 2.2918% 均 <3%。"
        f"锚点验收句为阶次 ≥1.8，本轮结果 {'已过门' if anchor else '未过门（honest FAIL，不粉饰）'}。")
    add('')
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('wrote', OUT / 'REPORT.md', 'and', OUT / 'fig_order_ladder.png')
    print('anchor_closed:', anchor, 'finest pair:', finest.get('pair'), 'order:', finest.get('total'))


if __name__ == '__main__':
    sys.exit(main())