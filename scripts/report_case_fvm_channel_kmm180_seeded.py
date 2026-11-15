#!/usr/bin/env python
"""Assemble the seeded closure-free channel report for ``CASE-FVM-CHANNEL-KMM180``.

Reads the ``summary.json`` / ``profiles.csv`` / ``transition.csv`` artefacts of
one or more seeded runs and writes ``REPORT.md`` plus figures with every number
injected programmatically (no hand-copied digits).

    PYTHONPATH=src python scripts/report_case_fvm_channel_kmm180_seeded.py \
        --runs runs/long-0 runs/long-1 --out docs/case-fvm-channel-kmm180-seeded
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.verification.kmm1987 import load_kmm1987_reference  # noqa: E402

LOW_Y_PLUS = 30.0


def load_run(directory: Path) -> dict:
    summary = json.loads((directory / 'summary.json').read_text())
    summary['_dir'] = str(directory)
    profiles = np.genfromtxt(directory / 'profiles.csv', delimiter=',', names=True)
    summary['_profiles'] = profiles
    transition = directory / 'transition.csv'
    summary['_transition'] = (np.genfromtxt(transition, delimiter=',', names=True)
                             if transition.exists() and transition.stat().st_size
                             else None)
    return summary


def dns_shear(dns: dict) -> dict:
    """DNS total shear and implied ``<u'v'>+ = du+/dy+ - (1 - y+/Re_tau)``."""
    y_plus = np.asarray(dns['y_plus'], float)
    du = np.asarray(dns['dU_plus_dy_plus'], float)
    target = 1.0 - y_plus / float(dns['re_tau'])
    return dict(y_plus=y_plus, du_plus_dy_plus=du, target=target,
                uv_plus=du - target)


def figure_profiles(runs, dns, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    shear = dns_shear(dns)
    ax = axes[0]
    ax.semilogx(shear['y_plus'], dns['u_plus'], 'k-', lw=2.0,
                label='DNS (Lee & Moser 2015)')
    for run in runs:
        p = run['_profiles']
        ax.semilogx(p['y_plus'], p['u_plus'], lw=1.4,
                    label=f"{run['label']} ({run['verdict']['anchor']})")
    ax.axvspan(LOW_Y_PLUS, 0.2 * dns['re_tau'], color='0.85', zorder=0,
               label='graded window')
    ax.set(xlabel=r'$y^+$', ylabel=r'$u^+$', title='mean velocity, wall units',
           xlim=(1.0, 200.0))
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.plot(shear['y_plus'], shear['uv_plus'], 'k-', lw=2.0, label='DNS')
    for run in runs:
        p = run['_profiles']
        ax.plot(p['y_plus'], p['uv_plus'], lw=1.4, label=run['label'])
    ax.set(xlabel=r'$y^+$', ylabel=r"$\langle u'v'\rangle^+$",
           title='Reynolds shear stress')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(shear['y_plus'], shear['target'], 'k--', lw=1.5,
            label=r'$1-Y/h$ (exact)')
    for run in runs:
        p = run['_profiles']
        ax.plot(p['y_plus'], p['total_shear_plus'], lw=1.4, label=run['label'])
    ax.set(xlabel=r'$y^+$', ylabel=r'$du^+/dy^+ - \langle u\'v\'\rangle^+$',
           title='mean momentum balance')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / 'fig_profiles.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_transition(runs, out: Path) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    plotted = False
    for run in runs:
        t = run['_transition']
        if t is None or t.size == 0:
            continue
        plotted = True
        axes[0].plot(t['time_plus'], np.abs(t['uv_share_max']), marker='o',
                     ms=2.5, lw=1.0, label=run['label'])
        axes[1].semilogy(t['time_plus'], np.maximum(t['identity_residual_max'], 1e-16),
                         marker='o', ms=2.5, lw=1.0, label=run['label'])
    if not plotted:
        plt.close(fig)
        return None
    axes[0].axhline(0.5, color='k', ls='--', lw=1.0, label='C1 limit (0.50)')
    axes[0].set(xlabel=r'$t^+$', ylabel=r'$\max\ -\langle u\'v\'\rangle^+ / (1-Y/h)$',
                title='Reynolds-stress share of the applied shear')
    axes[1].axhline(0.05, color='k', ls='--', lw=1.0, label='C2 limit (0.05)')
    axes[1].set(xlabel=r'$t^+$', ylabel='max momentum-balance residual',
                title='closure-free momentum balance')
    for ax in axes:
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / 'fig_transition.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_deviation(runs, dns, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y_plus = np.asarray(dns['y_plus'], float)
    u_ref = np.asarray(dns['u_plus'], float)
    for run in runs:
        p = run['_profiles']
        u_dns = np.interp(p['y_plus'], y_plus, u_ref)
        ax.plot(p['y_plus'], 100.0 * (p['u_plus'] - u_dns) / u_dns, lw=1.4,
                label=run['label'])
    ax.axvspan(LOW_Y_PLUS, 0.2 * dns['re_tau'], color='0.85', zorder=0,
               label='graded window')
    ax.axhline(0.0, color='k', lw=0.8)
    ax.axhspan(-5.0, 5.0, color='tab:green', alpha=0.08, zorder=0,
               label='±5 % anchor tolerance')
    ax.set(xlabel=r'$y^+$', ylabel=r'relative deviation $u^+$ [%]',
           title='signed deviation from the DNS reference')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / 'fig_deviation.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def table_runs(runs) -> list[str]:
    rows = ['| run | cells | steps (seed + average) | $T^+$ | flow-throughs | '
            '$uv$ share (C1) | balance (C2) | anchor RMS (legacy / log layer) | verdict |',
            '|---|---|---|---|---|---|---|---|---|']
    for run in runs:
        g = run['gates']
        rows.append(
            f"| {run['label']} | {run['resolution']['cells']} | "
            f"{run['steps_transient']} + {run['steps_average']} | "
            f"{g['time_plus']:.0f} | {g['flow_throughs']:.2f} | "
            f"{g['uv_share']:.3f} ({'PASS' if g['verdict']['C1'] else 'FAIL'}) | "
            f"{g['balance_residual']:.3f} ({'PASS' if g['verdict']['C2'] else 'FAIL'}) | "
            f"{100 * run['anchor']['legacy']['rms_relative']:.2f} % / "
            f"{100 * run['anchor']['log_layer']['rms_relative']:.2f} % | "
            f"{run['verdict']['published']} |")
    return rows


def render(runs, dns, out: Path, figs, notes: str) -> str:
    primary = runs[0]
    g = primary['gates']
    r = primary['resolution']
    gl = primary['globals']
    rr = primary['resolution_report']
    lines: list[str] = []
    add = lines.append

    add('# CASE-FVM-CHANNEL-KMM180 —— 有种子、长时间、无闭合槽道跑：把 m20573 的 VACUOUS_PASS 变成可判定的验证')
    add('')
    add('> 本文件所有数字由 `scripts/report_case_fvm_channel_kmm180_seeded.py` 从各 run 的 '
        '`summary.json` / `profiles.csv` 程序化注入，零手抄。')
    add(f"> 主运行：`{primary['_dir']}`；判决：锚点 **{primary['verdict']['anchor']}**，"
        f"准入 **{primary['verdict']['admissible']}**，发布判定 "
        f"**{primary['verdict']['published']}**。")
    add('')
    add('## 1. 案例介绍')
    add('')
    add('**物理背景**：`Re_tau = 180` 充分发展槽道湍流，流向/展向周期、两壁无滑移，'
        '恒定体积力驱动；`u_tau^2 = G h` 使壁面单位**由构造**确定，'
        '因此解出的流向速度本身就是 `u+`，全程不经过壁面剪切估计器，也不经过任何湍流闭合。')
    add('')
    add('| 项目 | 取值 |')
    add('|---|---|')
    add(f"| $Re_\\tau$ | {primary['re_tau']:.1f} |")
    add(f"| 网格 | {r['nx']}x{r['ny']}x{r['nz']} = {r['cells']} 单元 |")
    add(f"| 计算域 | $L_x$ = {r['length_x']:.4f}, $L_z$ = {r['length_z']:.4f}"
        f" (${rr['length_x_plus']:.0f}^+$, ${rr['length_z_plus']:.0f}^+$) |")
    add(f"| 壁面单位间距 | $\\Delta y^+_{{min}}$ = {rr['dy_plus_min']:.2f}, "
        f"$\\Delta y^+_{{max}}$ = {rr['dy_plus_centre']:.2f}, "
        f"$\\Delta x^+$ = {rr['dx_plus']:.1f}, $\\Delta z^+$ = {rr['dz_plus']:.1f} |")
    add(f"| 时间步 | $\\Delta t$ = {primary['dt']:.6g} ($\\Delta t^+$ = {primary['dt_plus']:.3f}) |")
    add(f"| 统计窗口 | 弃置 {primary['steps_transient']} 步 + 采样 {primary['steps_average']} 步, "
        f"$T^+$ = {g['time_plus']:.0f}, {g['flow_throughs']:.2f} 个流通时间 |")
    add(f"| 参考 | Lee & Moser 2015 DNS, $Re_\\tau$ = {dns['re_tau']:.3f} "
        '(仓内 `docs/rans-turbulence-validation-matrix/dns/0180/`)；KMM1987 原表仍 WITHHELD |')
    add('')
    add('**补的是哪一条欠账**：`docs/audit-m20573-closurefree-pass` 判定同一路线的首个'
        '无闭合跑为 `VACUOUS_PASS`——初值几乎不演化（`<u\'v\'>+ ~ 1e-6`、动量平衡残差 0.708），'
        '`u+` 的 1.6 % 读数判的是**初值**不是求解器；审计明确列出未做项：'
        '「重跑该路线的长时版本，需 $T^+ \\gtrsim 10^3$」。本单就是这一步。')
    add('')
    add('## 2. 怎么做的')
    add('')
    add('1. **种子**（`src/tensorfvm/verification/channel_turbulent_seed.py`）：'
        '在混合平均型面上叠加**反向旋转流向涡对** '
        '$v = A f(Y)\\cos(k_z z)$, $w = -(A f\'(Y)/k_z)\\sin(k_z z)$'
        '（解析无散），再加一个**正弦型条带摆动**项与少量白噪声；'
        '种子只作初始条件（等价于实验里的绊线），不是模型项。')
    add(f"   主运行种子参数：$A$ = {primary['seed']['amplitude']}, "
        f"$n_z$ = {primary['seed']['spanwise_mode']}, "
        f"摆动 = {primary['seed']['sinusity']}, 噪声 = {primary['seed']['noise']}, "
        f"seed = {primary['seed']['seed']}；投影后散度 "
        f"{primary['seed']['divergence_after_projection']:.2e}。")
    add('2. **准入判据先冻结**（与 `kmm180_closurefree_audit.CRITERIA` 同一常量）：'
        'C1 $-\\langle u\'v\'\\rangle^+/(1-Y/h) \\ge 0.50$、'
        'C2 $|du^+/dy^+ - \\langle u\'v\'\\rangle^+ - (1-Y/h)| \\le 0.05$、'
        'C3 流通时间 $\\ge 1.0$。**先过准入，锚点 5 % 读数才算证据**。')
    add('3. **求解器不变**：`SpectralChannel`（流向/展向 Fourier、壁法向交错 FV、SSP-RK3、'
        '`nu_t = 0`），只换初值与时长。')
    add('4. 全部数字由报告脚本注入；图由 `matplotlib` 生成后随本报告入库。')
    add('')
    add('## 3. 计算结果')
    add('')
    add('### 3.1 准入判据与锚点判据（逐 run）')
    add('')
    lines.extend(table_runs(runs))
    add('')
    add('### 3.2 主运行全局量')
    add('')
    add('| 量 | 计算 | 参考 (LM2015) | 相对偏差 |')
    add('|---|---|---|---|')
    add(f"| $U_b^+$ | {gl['bulk_plus']:.3f} | {gl['bulk_plus_reference']:.3f} | "
        f"{100 * (gl['bulk_plus'] / gl['bulk_plus_reference'] - 1):+.2f} % |")
    add(f"| $C_f$ | {gl['Cf']:.6f} | {gl['Cf_reference']:.6f} | "
        f"{100 * (gl['Cf'] / gl['Cf_reference'] - 1):+.2f} % |")
    add(f"| 壁面剪切（剖面估计，应=1） | {gl['wall_shear']:.4f} | 1.0000 | "
        f"{100 * (gl['wall_shear'] - 1):+.2f} % |")
    add(f"| 对数层 $\\langle u\'v\'\\rangle^+$ 最小值 | {gl['uv_plus_min']:.3f} | "
        'DNS ≈ -0.72 | — |')
    add(f"| 动量平衡残差最大（对数层） | {gl['identity_residual_log']:.4f} | 0 | — |")
    add('')
    add('### 3.3 锚点量 $u^+(y^+)$')
    add('')
    add('| 窗口 | 点数 | RMS 相对偏差 | 最大相对偏差 | 判定 (<5 %) |')
    add('|---|---|---|---|---|')
    for key, label in (('legacy', 'y+ ∈ [30, 最近参考站]（锚点字面）'),
                       ('log_layer', 'y+ ∈ [30, 0.2 Re_tau]（物理对数层）')):
        a = primary['anchor'][key]
        add(f"| {label} | {a['points']} | {100 * a['rms_relative']:.3f} % | "
            f"{100 * a['max_relative']:.3f} % | "
            f"{'PASS' if a['rms_relative'] < primary['anchor']['tolerance'] else 'FAIL'} |")
    add('')
    add('## 4. 结果图')
    add('')
    add('![fig_profiles.png](fig_profiles.png)')
    add('')
    add('左：$u^+(y^+)$ 与 DNS；阴影为锚点评分窗口。中：雷诺剪应力 '
        '$\\langle u\'v\'\\rangle^+$（DNS 由自身动量平衡反推）。右：无闭合动量平衡'
        ' $du^+/dy^+ - \\langle u\'v\'\\rangle^+$ 与精确的 $1-Y/h$。')
    add('')
    if figs.get('transition'):
        add('![fig_transition.png](fig_transition.png)')
        add('')
        add('过渡诊断：左为雷诺应力承担的剪切份额（C1），右为动量平衡残差（C2），'
            '横轴为壁面时间 $t^+$。判据线为虚线。')
        add('')
    add('![fig_deviation.png](fig_deviation.png)')
    add('')
    add('带符号的相对偏差：$-5\\%$–$+5\\%$ 绿带为锚点容差。')
    add('')
    add('## 5. 注意点（坑、局限、适用范围）')
    add('')
    add(notes)
    add('')
    add('## 6. 误差判定（逐项 vs 基准）')
    add('')
    add('| 关键量 | 基准 | 实测 | 误差 | <3 % ？ |')
    add('|---|---|---|---|---|')
    lg = primary['anchor']['log_layer']
    add(f"| $u^+$ 对数层 RMS（锚点量） | DNS 表 | {100 * lg['rms_relative']:.3f} % | "
        f"{100 * lg['rms_relative']:.2f} % | {'是' if lg['rms_relative'] < 0.03 else '否'} |")
    add(f"| 对数层 $\\langle u\'v\'\\rangle^+$ | DNS ≈ -0.72 | {gl['uv_plus_min']:.3f} | "
        f"{100 * abs(gl['uv_plus_min'] / -0.72 - 1):.1f} % | "
        f"{'是' if abs(gl['uv_plus_min'] / -0.72 - 1) < 0.03 else '否'} |")
    add(f"| 动量平衡残差 | 0 | {gl['identity_residual_log']:.4f} | — | "
        f"{'是' if gl['identity_residual_log'] < 0.05 else '否'} |")
    add(f"| $U_b^+$ | {gl['bulk_plus_reference']:.3f} | {gl['bulk_plus']:.3f} | "
        f"{100 * abs(gl['bulk_plus'] / gl['bulk_plus_reference'] - 1):.2f} % | "
        f"{'是' if abs(gl['bulk_plus'] / gl['bulk_plus_reference'] - 1) < 0.03 else '否'} |")
    add(f"| $C_f$ | {gl['Cf_reference']:.6f} | {gl['Cf']:.6f} | "
        f"{100 * abs(gl['Cf'] / gl['Cf_reference'] - 1):.2f} % | "
        f"{'是' if abs(gl['Cf'] / gl['Cf_reference'] - 1) < 0.03 else '否'} |")
    add('')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-seeded')
    parser.add_argument('--notes-file', default=None)
    args = parser.parse_args(argv)

    runs = [load_run(Path(r)) for r in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dns = load_kmm1987_reference()

    figs = dict(profiles=figure_profiles(runs, dns, out),
                deviation=figure_deviation(runs, dns, out),
                transition=figure_transition(runs, out))
    default_notes = '\n'.join([
        '* 种子是**初始条件**，不是模型项：它是否存活必须由 C1/C2 判定，'
        '不能因为锚点读数漂亮就宣布路线成立（这正是 m20573 的教训）。',
        '* 统计窗口以 $T^+$ 与流通时间双记账；$T^+ \\gtrsim 10^3$ 是槽道 DNS 的常规量级，'
        '低于此的读数只能当趋势。',
        '* 参考是 Lee & Moser 2015 DNS（$Re_\\tau = 182.088$），不是 KMM1987 原表；'
        '代理替换的代价在 `docs/case-fvm-channel-kmm180-refband` 里量过。',
        '* 网格仍是 $64\\times128\\times64$，$\\Delta z^+ \\approx 8.8$：'
        '高于 DNS 通常要求的 $\\Delta z^+ \\lesssim 5$，故小尺度统计量的误差不可忽略，'
        '但锚点量（平均速度）主要由对数层大尺度决定。',
    ])
    notes = Path(args.notes_file).read_text() if args.notes_file else default_notes
    (out / 'REPORT.md').write_text(render(runs, dns, out, figs, notes))
    print(json.dumps(dict(
        out=str(out),
        runs=[r['label'] for r in runs],
        verdicts=[r['verdict'] for r in runs],
        figures=[str(p) for p in figs.values() if p]), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())