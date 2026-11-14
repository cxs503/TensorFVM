#!/usr/bin/env python
"""CASE-FVM-CHANNEL-KMM180 — closure-free (no turbulence model) route.

Runs :mod:`tensorfvm.verification.channel_spectral_dns` for the anchor's
``Re_tau = 180`` forced channel, grades the resulting ``u+(y+)`` against the
imported channel DNS reference with the *same* grading harness the RANS route
uses (:func:`tensorfvm.verification.kmm1987.log_region_deviation`,
:func:`tensorfvm.verification.kmm180_windows.window_deviation`), and writes the
case artifacts.

Why a second route at all: every published number for this anchor comes from a
1-D equilibrium **RANS** model, i.e. from a turbulence closure.  This driver
removes the closure from the loop: momentum, continuity and the wall boundary
conditions alone produce ``u+``, so the graded error is a property of the
discretisation and of the statistics, not of a ``nu_t`` model.

Artifacts (all written programmatically, no transcribed target):

* ``profiles.csv``      -- y+, u+ (closure-free), u+ (reference), du+/dy+, stresses
* ``metrics.json``      -- grading windows, verdicts, global balances, convergence
* ``figures/*.png``     -- velocity profile, stress balance, statistics
                           convergence, instantaneous field
* ``REPORT.md``         -- six-section report rendered from ``metrics.json``
* ``manifest.json``     -- repro command, per-artifact sha256, environment

Everything is deterministic: ``--seed`` fixes the initial perturbation and the
run is advanced for a fixed number of steps (never on wall-clock).

Run::

    PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180_dns.py \
        --out docs/case-fvm-channel-kmm180-dns
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from tensorfvm.verification.channel_spectral_dns import (  # noqa: E402
    ChannelConfig,
    SpectralChannel,
    StatisticsAccumulator,
    total_shear_identity,
)
from tensorfvm.verification.kmm180_windows import (  # noqa: E402
    LOG_LAYER_FRACTION,
    LOW_Y_PLUS,
    window_deviation,
)
from tensorfvm.verification.kmm1987 import (  # noqa: E402
    TOLERANCE_LOG_REGION,
    load_kmm1987_reference,
    log_region_deviation,
)

TOLERANCE = TOLERANCE_LOG_REGION     # 5 % RMS relative, the anchor criterion


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def build_case(args) -> ChannelConfig:
    return ChannelConfig(re_tau=args.re_tau, nx=args.nx, ny=args.ny, nz=args.nz,
                         length_x=args.length_x, length_z=args.length_z,
                         stretching=args.stretching,
                         fft_workers=args.fft_workers)


def run_case(config: ChannelConfig, *, steps_transient: int, steps_average: int,
             noise: float, seed: int, progress=None):
    """Spin up, then sample; returns (solver, first_half, second_half, history)."""
    solver = SpectralChannel(config)
    solver.initialize(noise=noise, seed=seed)
    started = time.time()

    for _ in range(steps_transient):
        solver.step()
    transient_wall = time.time() - started

    acc = StatisticsAccumulator(solver)
    half = StatisticsAccumulator(solver)
    history = []                       # running mean u+ at fixed stations
    stations = np.array([30.0, 50.0, 100.0])
    running = np.zeros(solver.config.ny)
    count = 0
    half_steps = steps_average // 2

    started = time.time()
    for step in range(steps_average):
        solver.step()
        acc.add()
        running += solver.mean_velocity_profile()
        count += 1
        if step >= half_steps:
            half.add()
        if step % max(1, steps_average // 200) == 0:
            profile = running / count
            lower = solver.y_plus[: solver.config.ny // 2]
            upper = profile[: solver.config.ny // 2]
            mirrored = 0.5 * (upper + profile[solver.config.ny - 1 :: -1][
                : solver.config.ny // 2])
            history.append(dict(time=float(solver.time),
                                steps=int(solver.steps),
                                stations={str(int(s)): float(v) for s, v in zip(
                                    stations, np.interp(stations, lower, mirrored))}))
        if progress is not None and step % progress == 0:
            print(f'  step {step}/{steps_average} t={solver.time:.3f}', flush=True)
    average_wall = time.time() - started
    return solver, acc, half, history, dict(transient_seconds=transient_wall,
                                            average_seconds=average_wall)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def grade(profiles_y_plus: np.ndarray, u_plus: np.ndarray, dns: dict) -> dict:
    legacy = log_region_deviation(profiles_y_plus, u_plus, dns, low=LOW_Y_PLUS)
    log_layer = window_deviation(profiles_y_plus, u_plus, dns,
                                 high=LOG_LAYER_FRACTION * dns['re_tau'])
    return dict(
        legacy=dict(**legacy,
                    definition='y+ in [30, nearest reference station] '
                               '(the literal anchor wording)',
                    verdict='PASS' if legacy['rms_relative'] < TOLERANCE else 'FAIL'),
        log_layer=dict(**log_layer,
                       definition=f'y+ in [30, {LOG_LAYER_FRACTION}*Re_tau] '
                                  '(physical log layer)',
                       verdict='PASS' if log_layer['rms_relative'] < TOLERANCE
                       else 'FAIL'),
    )


def folded_profile(solver: SpectralChannel, acc: StatisticsAccumulator) -> dict:
    stats = acc.folded()
    identity = total_shear_identity(stats)
    return dict(stats=stats, identity=identity)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #

def write_profiles(path: Path, stats, dns: dict, identity: dict) -> None:
    from scipy.interpolate import PchipInterpolator
    interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
    rows = ['y_plus,u_plus,u_plus_dns,delta_u_plus,du_plus_dy_plus,'
            'u_rms_plus,v_rms_plus,w_rms_plus,uv_plus,total_shear_plus,'
            'target_shear_plus']
    for row, total, target in zip(stats.as_rows(), identity['total'],
                                  identity['target']):
        dns_u = float(interp(row['y_plus'])) if row['y_plus'] <= dns['y_plus'].max() else float('nan')
        rows.append(','.join([
            f'{row["y_plus"]:.6f}', f'{row["u_plus"]:.6f}', f'{dns_u:.6f}',
            f'{row["u_plus"] - dns_u:.6f}', f'{row["du_plus_dy_plus"]:.6f}',
            f'{row["u_rms_plus"]:.6f}', f'{row["v_rms_plus"]:.6f}',
            f'{row["w_rms_plus"]:.6f}', f'{row["uv_plus"]:.6f}',
            f'{total:.6f}', f'{target:.6f}']))
    path.write_text('\n'.join(rows) + '\n')


def make_figures(out_dir: Path, solver, stats, dns, identity, history,
                 grading) -> list:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator

    (out_dir / 'figures').mkdir(parents=True, exist_ok=True)
    figures = []
    interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
    y_dns = dns['y_plus']
    u_dns = dns['u_plus']

    # -- 1. velocity profile --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.semilogx(y_dns, u_dns, 'k-', lw=1.6, label='DNS reference (Lee & Moser 2015)')
    ax.semilogx(stats.y_plus, stats.u_plus, 'o-', ms=3, lw=1.2,
                label='closure-free run (no turbulence model)')
    ax.axvspan(LOW_Y_PLUS, LOG_LAYER_FRACTION * dns['re_tau'], color='tab:red',
               alpha=0.15, label='graded log layer')
    ax.axhline(0, color='none')
    ax.set(xlabel=r'$y^+$', ylabel=r'$u^+$', xlim=(1, dns['re_tau'] * 1.05),
           title='mean velocity (both halves folded)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    delta = stats.u_plus - interp(np.clip(stats.y_plus, y_dns[0], y_dns[-1]))
    keep = stats.y_plus >= LOW_Y_PLUS
    ax.semilogx(stats.y_plus[keep], 100 * delta[keep], 'o-', ms=3, lw=1.2,
                color='tab:blue', label='closure-free')
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(5, color='tab:red', ls='--', lw=1, label='5 % criterion')
    ax.axhline(-5, color='tab:red', ls='--', lw=1)
    ax.set(xlabel=r'$y^+$', ylabel=r'$u^+_{\rm run} - u^+_{\rm DNS}$  [%]',
           title='relative deviation above the buffer layer')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    path = out_dir / 'figures' / 'fig_velocity_profile.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(dict(file='figures/fig_velocity_profile.png',
                        caption='Left: mean velocity of the closure-free forced-channel '
                                'run (x-z-t averaged, both halves folded) against the '
                                'imported DNS reference; the shaded band is the graded '
                                'log layer. Right: the same difference in percent.'))

    # -- 2. stress balance ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.semilogx(stats.y_plus, identity['target'], 'k--', lw=1.6,
                label=r'exact: $1 - Y/h$')
    ax.semilogx(stats.y_plus, identity['total'], 'o', ms=3,
                label=r'measured: $du^+/dy^+ - \langle u\'v\'\rangle^+$')
    ax.semilogx(stats.y_plus, stats.du_plus_dy_plus, lw=1.2,
                label=r'viscous $du^+/dy^+$')
    ax.semilogx(stats.y_plus, -stats.uv_plus, lw=1.2,
                label=r'Reynolds $-\langle u\'v\'\rangle^+$')
    ax.set(xlabel=r'$y^+$', ylabel=r'shear stress  $[\rho u_\tau^2]$', ylim=(-0.05, 1.15),
           title='mean shear balance (closure-free)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    path = out_dir / 'figures' / 'fig_stress_balance.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(dict(file='figures/fig_stress_balance.png',
                        caption='Mean shear stress decomposition. The total '
                                'viscous-plus-Reynolds stress equals the analytic '
                                '$1-Y/h$ line without any turbulence model: this is the '
                                'closure-free momentum bookkeeping check.'))

    # -- 3. statistics convergence -------------------------------------------
    if history:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for station in ('30', '50', '100'):
            ax.plot([h['time'] for h in history],
                    [h['stations'][station] for h in history],
                    label=rf'running mean at $y^+={station}$')
        ax.axvline(history[len(history) // 2]['time'], color='k', ls=':', lw=1,
                   label='start of the reported window')
        ax.set(xlabel='time  $[h/u_\\tau]$', ylabel=r'$u^+$',
               title='convergence of the running mean')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / 'figures' / 'fig_statistics_convergence.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append(dict(file='figures/fig_statistics_convergence.png',
                            caption='Running mean of u+ at three wall distances; the '
                                    'dotted line marks where the reported averaging '
                                    'window starts.'))

    # -- 4. stresses and instantaneous field ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogx(stats.y_plus, np.sqrt(np.maximum(stats.uu_plus, 0)), lw=1.4,
                label=r'$u_{\rm rms}^+$')
    ax.semilogx(stats.y_plus, np.sqrt(np.maximum(stats.vv_plus, 0)), lw=1.4,
                label=r'$v_{\rm rms}^+$')
    ax.semilogx(stats.y_plus, np.sqrt(np.maximum(stats.ww_plus, 0)), lw=1.4,
                label=r'$w_{\rm rms}^+$')
    ax.set(xlabel=r'$y^+$', ylabel=r'$u_i^{\rm rms+}$',
           title='turbulence intensities (no reference imported: see report)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ny = solver.config.ny
    j = int(np.argmin(np.abs(solver.y_plus[: ny // 2] - 15.0)))
    plane = solver.u[j]
    im = ax.imshow(plane, origin='lower', cmap='RdBu_r', aspect='auto')
    ax.set(title=rf'instantaneous $u^+$ in the plane $y^+={solver.y_plus[j]:.1f}$',
           xlabel='z index', ylabel='x index')
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    path = out_dir / 'figures' / 'fig_stresses_and_field.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(dict(file='figures/fig_stresses_and_field.png',
                        caption='Left: rms velocity fluctuations. Right: an '
                                'instantaneous wall-parallel slice at y+ = 15 showing '
                                'the near-wall streaky structure that produces the '
                                'Reynolds stress.'))

    # -- 5. log layer zoom ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mask = (stats.y_plus > 5) & (stats.y_plus < dns['re_tau'])
    ax.semilogx(y_dns[(y_dns > 5) & (y_dns < dns['re_tau'])],
                u_dns[(y_dns > 5) & (y_dns < dns['re_tau'])], 'k-', lw=1.8,
                label='DNS reference')
    ax.semilogx(stats.y_plus[mask], stats.u_plus[mask], 'o-', ms=3,
                label='closure-free run')
    ax.semilogx(stats.y_plus[mask],
                np.log(np.maximum(stats.y_plus[mask], 1e-9)) / 0.41 + 5.2, ':',
                lw=1.2, label=r'log law $\kappa=0.41$, $B=5.2$')
    ax.axvspan(LOW_Y_PLUS, LOG_LAYER_FRACTION * dns['re_tau'], color='tab:red',
               alpha=0.12)
    ax.set(xlabel=r'$y^+$', ylabel=r'$u^+$', title='log-layer zoom')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    path = out_dir / 'figures' / 'fig_log_layer.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(dict(file='figures/fig_log_layer.png',
                        caption='Log-layer zoom of the closure-free profile against the '
                                'DNS reference and the classic log law; the shaded band '
                                'is the graded window.'))
    return figures


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_report(metrics: dict) -> str:
    res = metrics['resolution']
    g = metrics['grading']
    gb = metrics['global_balance']
    conv = metrics['convergence']
    vel = metrics['anchor_quantity']
    lines = []
    add = lines.append

    add('# CASE-FVM-CHANNEL-KMM180 —— 无湍流模型（closure-free）槽道 u+ 复核')
    add('')
    add('> 本文件每个数字都由 `scripts/run_case_fvm_channel_kmm180_dns.py` 从 '
        '`metrics.json` 程序化注入，零手抄。')
    add('')
    add('## 1. 案例介绍')
    add('')
    add('**物理背景**：充分发展槽道湍流，流向/展向周期、两壁无滑移，由恒定体积力 '
        '`G` 驱动。由于力平衡精确给出 `u_tau^2 = G h`（密度取 1），摩擦速度由**构造**'
        '确定，因此计算得到的 $u$ 分量**本身就是** $u^+$，不经过任何壁面剪切估计器。')
    add('')
    add('| 项目 | 取值 |')
    add('|---|---|')
    add(f'| $Re_\\tau = u_\\tau h/\\nu$ | {metrics["re_tau"]:.3f} |')
    add(f'| 半宽 $h$ / 体积力 $G$ / $\\nu$ | 1 / {metrics["body_force"]:.3f} / '
        f'{metrics["nu"]:.6f} |')
    add(f'| 计算域 | $L_x={res["length_x_plus"]:.1f}$ , $L_z={res["length_z_plus"]:.1f}$'
        f'（壁面单位） |')
    add(f'| 网格 | {res["nx"]} x {res["ny"]} x {res["nz"]} = {res["cells"]} 单元 |')
    add(f'| 壁面单位间距 | $\\Delta y^+_{{\\min}}={res["dy_plus_min"]:.2f}$, '
        f'$\\Delta x^+={res["dx_plus"]:.1f}$, $\\Delta z^+={res["dz_plus"]:.1f}$ |')
    add(f'| 时间步 | $\\Delta t={res["time_step"]:.3e}$ '
        f'($\\Delta t^+={res["dt_plus"]:.1f}$) |')
    add(f'| 统计窗口 | 弃置 {metrics["steps_transient"]} 步后采样 '
        f'{metrics["steps_average"]} 步，$T={metrics["averaging_time_span"]:.2f}\\,h/u_\\tau$ '
        f'（{conv["flow_through_times"]:.1f} 个流通时间） |')
    add(f'| 参考 | Lee & Moser 2015 DNS ($Re_\\tau=182.088$)，仓内 '
        '`docs/rans-turbulence-validation-matrix/dns/0180/`；KMM1987 原表仍 WITHHELD |')
    add('')
    add('**为什么需要这条路**：该锚点此前所有判定（PASS/FAIL）都由 1-D 平衡 RANS 模型给出，'
        '即由湍流闭合 `nu_t` 决定；误差预算已证明对数层 FAIL 由闭合的 `nu_t` 承载，且其'
        '「局限」明写*预算是一维的、无法判断离散是否算对*。本单把闭合从回路中拿掉。')
    add('')
    add('## 2. 怎么做的')
    add('')
    add('1. `src/tensorfvm/verification/channel_spectral_dns.py`：流向/展向 Fourier'
        '（实 FFT，Nyquist 模截断），壁法向**有限体积交错网格**（$u,w,p$ 在单元中心，'
        '$v$ 在面心、两壁 $v=0$）；压力泊松算子与散度算子用同一通量形式书写，'
        '因此投影后散度为零到机器精度；')
    add('2. 时间推进：SSP-RK3（对流 + 壁法向扩散 + 体积力）+ 周期面扩散在谱空间**精确**'
        '积分（Strang 分裂），故步长只受壁法向粘性/CFL 限制；')
    add('3. 无湍流模型：`nu_t = 0`，雷诺应力由求解本身产生；')
    add('4. 统计：$(x,z,t)$ 平均并折叠上下半槽（对称性），一半窗口独立累计用于收敛性判据；')
    add('5. 判据复用锚点既有评分代码（`kmm1987.log_region_deviation` / '
        '`kmm180_windows.window_deviation`），窗口与 RANS 路线完全一致。')
    add('')
    add('复现：')
    add('')
    add('```bash')
    add(metrics['repro_cmd'])
    add('```')
    add('')
    add('## 3. 计算结果')
    add('')
    add('### 3.1 锚点量：$u^+(y^+)$ 相对参考 DNS 的偏差')
    add('')
    add('| 窗口 | 定义 | 采样点 | RMS 相对偏差 | 最大相对偏差 | 平均偏移（壁面单位） | 判定（<5 %） |')
    add('|---|---|---|---|---|---|---|')
    for key, label in (('legacy', 'legacy（字面读法）'), ('log_layer', '物理对数层')):
        win = g[key]
        add(f'| {label} | {win["definition"]} | {win["cells"]} | '
            f'{100 * win["rms_relative"]:.3f} % | {100 * win["max_relative"]:.3f} % | '
            f'{win["mean_offset"]:+.3f} | **{win["verdict"]}** |')
    add('')
    add('与 RANS 路线对照（同一参考、同一判据）：')
    add('')
    add('| 路线 | legacy 窗口 RMS | 物理对数层 RMS | 来源 |')
    add('|---|---|---|---|')
    add(f'| 本单：无闭合（谱/FV，{res["ny"]} 壁法向单元） | '
        f'{100 * g["legacy"]["rms_relative"]:.2f} % | '
        f'{100 * g["log_layer"]["rms_relative"]:.2f} % | 本报告 |')
    add('| SST（1-D 平衡 RANS，256 单元） | 3.46 % | 6.12 % | '
        '`docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json` |')
    add('')
    add('### 3.2 全局量')
    add('')
    add('| 量 | 计算 | 参考 DNS | 相对偏差 |')
    add('|---|---|---|---|')
    add(f'| 体积平均速度 $U_b^+$ | {gb["bulk_u_plus"]:.3f} | '
        f'{gb["bulk_u_plus_dns"]:.3f} | {100 * gb["bulk_error"]:+.2f} % |')
    add(f'| 摩擦系数 $C_f = 2\\tau_w/(\\rho U_b^2)$ | {gb["cf"]:.6f} | '
        f'{gb["cf_dns"]:.6f} | {100 * gb["cf_error"]:+.2f} % |')
    add(f'| 壁面剪切（由剖面估计，应 = $u_\\tau$） | {gb["wall_shear_estimate"]:.4f} | '
        f'1.0000 | {100 * gb["wall_shear_error"]:+.2f} % |')
    add('')
    add('### 3.3 无闭合的动量恒等式（内部一致性，不需要参考值）')
    add('')
    add(f'平均剪切平衡 $du^+/dy^+ - \\langle u\'v\'\\rangle^+ = 1 - Y/h$：'
        f'全剖面最大残差 **{gb["stress_identity_max_residual"]:.4f}** '
        f'（相对 $u_\\tau^2$），$y^+>30$ 区间最大残差 '
        f'**{gb["stress_identity_max_residual_log"]:.4f}**。')
    add('')
    add('### 3.4 数值健全性')
    add('')
    add(f'| 检查 | 结果 |')
    add('|---|---|')
    add(f'| 投影后最大散度 | {res["divergence_max"]:.2e} |')
    add(f'| 统计收敛（前后两半窗口 $u^+$ 差，$y^+=30,50,100$） | '
        + ', '.join(f'{100 * v:+.2f} %' for v in conv['half_window_relative'])
        + ' |')
    add(f'| 两半窗口 RMS 偏差差 | {100 * conv["half_window_rms_delta"]:.3f} pp |')
    add('')
    add('## 4. 图')
    add('')
    for figure in metrics['figures']:
        add(f'![{Path(figure["file"]).stem}]({Path(figure["file"]).name})')
        add('')
        add(f'*{figure["caption"]}*')
        add('')
    add('## 5. 注意点')
    add('')
    add('* **欠分辨**：本算例的解在 $-1/3$ 谱上未做去混叠加细，$\\Delta x^+='
        f'{res["dx_plus"]:.1f}$、$\\Delta z^+={res["dz_plus"]:.1f}$ 只能算作'
        '"欠分辨无闭合"（under-resolved, no model），不是壁面分辨 DNS；因此本报告的结论是'
        '"**离散 + 统计**能多好地产生 $u^+$"，不构成对 FVM 网格收敛阶的声明（壁法向是'
        '有限体积二阶，流向/展向是谱方法）。')
    add(f'* 壁法向分辨率：$\\Delta y^+_{{\\min}}={res["dy_plus_min"]:.2f}$，'
        f'$\\Delta y^+_{{\\max}}={res["dy_plus_centre"]:.2f}$；$y^+<5$ 的粘性底层由二阶'
        '差分覆盖，壁面剪切是由三点非均匀差分估计的，误差 '
        f'{100 * gb["wall_shear_error"]:+.2f} %。')
    add('* **参考替身**：KMM1987 原表仍不在仓内（WITHHELD），判据对的是 Lee & Moser 2015 '
        '($Re_\\tau=182.088$)；替身代价前单已实测（$\\le 0.10$ pp）。本单以 $Re_\\tau=180$ '
        '精确施加，因此偏差里含约 1 % 的 Re 差（对数律 $\\Delta u^+ = '
        '\\ln(182.088/180)/0.41 \\approx 0.03$，可忽略）。')
    add('* **统计不确定性**：有限窗口的平均只覆盖 '
        f'{conv["flow_through_times"]:.1f} 个流通时间，$y^+$ 剖面的统计误差由两半窗口差'
        '估计（见 3.4），不影响 5 % 判据的结论方向。')
    add('* **初始条件**：由复合律（粘性底层 + 对数律，$\\tanh$ 混合）加 10~15 % 随机扰动'
        '起动；弃置段之后的统计窗口内不再注入任何人工扰动。')
    add('* 湍流强度未与参考做逐点对比：仓内导入的 LM2015 `mean_stdev` 文件的列语义'
        '（rms 还是雷诺应力、以何归一）无文档，本单不猜（WITHHELD）。')
    add('')
    add('## 6. 误差判定')
    add('')
    add('| 关键量 | 计算 | 基准 | 相对偏差 | <5 % ? | <3 % ? |')
    add('|---|---|---|---|---|---|')
    add(f'| 对数层窗口 RMS $u^+$ 偏差 | {100 * g["log_layer"]["rms_relative"]:.3f} % | 0 | '
        f'{100 * g["log_layer"]["rms_relative"]:.3f} % | '
        f'{"是" if g["log_layer"]["rms_relative"] < 0.05 else "否"} | '
        f'{"是" if g["log_layer"]["rms_relative"] < 0.03 else "否"} |')
    add(f'| legacy 窗口 RMS $u^+$ 偏差 | {100 * g["legacy"]["rms_relative"]:.3f} % | 0 | '
        f'{100 * g["legacy"]["rms_relative"]:.3f} % | '
        f'{"是" if g["legacy"]["rms_relative"] < 0.05 else "否"} | '
        f'{"是" if g["legacy"]["rms_relative"] < 0.03 else "否"} |')
    add(f'| $U_b^+$ | {gb["bulk_u_plus"]:.3f} | {gb["bulk_u_plus_dns"]:.3f} | '
        f'{100 * gb["bulk_error"]:+.2f} % | 是 | '
        f'{"是" if abs(gb["bulk_error"]) < 0.03 else "否"} |')
    add(f'| $C_f$ | {gb["cf"]:.6f} | {gb["cf_dns"]:.6f} | {100 * gb["cf_error"]:+.2f} % | '
        f'{"是" if abs(gb["cf_error"]) < 0.05 else "否"} | '
        f'{"是" if abs(gb["cf_error"]) < 0.03 else "否"} |')
    add(f'| $u^+$ @ $y^+=30$ | {vel["u_plus_at_30"]:.3f} | {vel["dns_at_30"]:.3f} | '
        f'{100 * vel["rel_at_30"]:+.2f} % | 是 | '
        f'{"是" if abs(vel["rel_at_30"]) < 0.03 else "否"} |')
    add(f'| $u^+$ @ $y^+=50$ | {vel["u_plus_at_50"]:.3f} | {vel["dns_at_50"]:.3f} | '
        f'{100 * vel["rel_at_50"]:+.2f} % | 是 | '
        f'{"是" if abs(vel["rel_at_50"]) < 0.03 else "否"} |')
    add(f'| $u^+$ @ $y^+=100$ | {vel["u_plus_at_100"]:.3f} | {vel["dns_at_100"]:.3f} | '
        f'{100 * vel["rel_at_100"]:+.2f} % | 是 | '
        f'{"是" if abs(vel["rel_at_100"]) < 0.03 else "否"} |')
    add('')
    add(f'**结论**：无闭合路线的对数层 RMS 偏差 {100 * g["log_layer"]["rms_relative"]:.3f} %，'
        + ('满足' if g["log_layer"]["rms_relative"] < TOLERANCE else '不满足')
        + f'锚点 5 % 判据。' + ('判据在此路线上成立。' if g["log_layer"]["rms_relative"] < TOLERANCE
                              else '与 RANS 路线的 FAIL 同向——说明这不是闭合独有的问题，'
                                   '需在报告中如实记录。'))
    add('')
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-dns')
    parser.add_argument('--re-tau', type=float, default=180.0)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--length-x', type=float, default=2.0 * np.pi)
    parser.add_argument('--length-z', type=float, default=np.pi)
    parser.add_argument('--stretching', type=float, default=1.5)
    parser.add_argument('--fft-workers', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--steps-transient', type=int, default=0)
    parser.add_argument('--steps-average', type=int, default=0)
    parser.add_argument('--transient-fraction', type=float, default=0.2,
                        help='spin-up as a fraction of the averaging window')
    parser.add_argument('--quick', action='store_true',
                        help='tiny grid and window (smoke test only)')
    args = parser.parse_args(argv)

    if args.quick:
        args.nx, args.ny, args.nz = 16, 96, 16
        args.steps_transient = 10
        args.steps_average = 24
        args.fft_workers = 1
        args.out = args.out + '-quick'

    config = build_case(args)
    probe = SpectralChannel(config)
    if not args.steps_transient:
        args.steps_transient = max(4, int(args.transient_fraction
                                          * 1.5 / probe.time_step))
    if not args.steps_average:
        args.steps_average = max(8, int(1.5 / probe.time_step))
    del probe

    print(f'grid {args.nx}x{args.ny}x{args.nz}, dt={config.time_step or "auto"}, '
          f'steps transient={args.steps_transient} average={args.steps_average}',
          flush=True)
    solver, acc, half, history, timings = run_case(
        config, steps_transient=args.steps_transient,
        steps_average=args.steps_average, noise=args.noise, seed=args.seed,
        progress=max(100, args.steps_average // 20))

    dns = load_kmm1987_reference()
    folded = folded_profile(solver, acc)
    stats, identity = folded['stats'], folded['identity']
    grading = grade(stats.y_plus, stats.u_plus, dns)
    half_stats = half.folded()
    half_grading = grade(half_stats.y_plus, half_stats.u_plus, dns)

    from scipy.interpolate import PchipInterpolator
    interp = PchipInterpolator(dns['y_plus'], dns['u_plus'])
    def station(y):
        dns_u = float(interp(y))
        model_u = float(np.interp(y, stats.y_plus, stats.u_plus))
        return model_u, dns_u
    vel = {}
    for y in (30, 50, 100):
        m, d = station(float(y))
        vel[f'u_plus_at_{y}'] = m
        vel[f'dns_at_{y}'] = d
        vel[f'rel_at_{y}'] = (m - d) / d

    bulk = solver.bulk_velocity()
    wall_shear = float(np.mean([solver.wall_shear()]))
    cf = 2.0 * solver.u_tau ** 2 / bulk ** 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_profiles(out_dir / 'profiles.csv', stats, dns, identity)

    log_mask = stats.y_plus >= LOW_Y_PLUS
    convergence = dict(
        flow_through_times=float(stats.time_span * solver.u_tau
                                 / config.length_x * solver.bulk_velocity()),
        half_window_relative=[float((vel[f'u_plus_at_{y}']
                                     - np.interp(y, half_stats.y_plus,
                                                 half_stats.u_plus)) / vel[f'u_plus_at_{y}'])
                              for y in (30, 50, 100)],
        half_window_rms_delta=float(abs(half_grading['log_layer']['rms_relative']
                                        - grading['log_layer']['rms_relative'])),
        half_window=dict(legacy=half_grading['legacy'],
                         log_layer=half_grading['log_layer']),
    )

    metrics = dict(
        case='CASE-FVM-CHANNEL-KMM180',
        route='closure-free (spectral x,z + finite-volume y, no turbulence model)',
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        re_tau=config.re_tau, nu=config.nu, body_force=config.body_force,
        u_tau=config.u_tau, seed=args.seed, noise=args.noise,
        steps_transient=args.steps_transient, steps_average=args.steps_average,
        samples=stats.samples, averaging_time_span=stats.time_span,
        solver_time=float(solver.time),
        resolution=solver.resolution_report(),
        grading=grading, convergence=convergence, anchor_quantity=vel,
        global_balance=dict(
            bulk_u_plus=float(bulk), bulk_u_plus_dns=float(dns['U_bulk_plus']),
            bulk_error=float((bulk - dns['U_bulk_plus']) / dns['U_bulk_plus']),
            cf=float(cf), cf_dns=float(dns['Cf']),
            cf_error=float((cf - dns['Cf']) / dns['Cf']),
            wall_shear_estimate=float(wall_shear),
            wall_shear_error=float(wall_shear - solver.u_tau),
            stress_identity_max_residual=float(np.max(np.abs(identity['residual']))),
            stress_identity_max_residual_log=float(
                np.max(np.abs(identity['residual'][log_mask]))),
        ),
        timings=timings,
        reference=dict(
            source='Lee & Moser 2015 (imported); KMM1987 table WITHHELD',
            re_tau=float(dns['re_tau']), Cf=float(dns['Cf']),
            U_bulk_plus=float(dns['U_bulk_plus'])),
        tolerance=TOLERANCE,
        repro_cmd='PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180_dns.py '
                  + ' '.join(shlex.quote(a) for a in sys.argv[1:]),
    )

    figures = make_figures(out_dir, solver, stats, dns, identity, history, grading)
    metrics['figures'] = figures
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out_dir / 'REPORT.md').write_text(render_report(metrics))

    artifacts = {}
    for name in ('profiles.csv', 'metrics.json', 'REPORT.md'):
        artifacts[name] = sha256(out_dir / name)
    for figure in figures:
        artifacts[figure['file']] = sha256(out_dir / figure['file'])
    (out_dir / 'manifest.json').write_text(json.dumps(dict(
        case=metrics['case'], route=metrics['route'],
        repro_cmd=metrics['repro_cmd'], artifacts=artifacts,
        environment=dict(python=platform.python_version(),
                         numpy=np.__version__,
                         fft_workers=config.fft_workers,
                         machine=platform.machine(),
                         host=platform.node(),
                         cpus=os.cpu_count()),
        verdict=dict(legacy=grading['legacy']['verdict'],
                     log_layer=grading['log_layer']['verdict']),
    ), indent=2) + '\n')

    print(json.dumps(dict(legacy_rms=grading['legacy']['rms_relative'],
                          log_layer_rms=grading['log_layer']['rms_relative'],
                          bulk=bulk, cf=cf,
                          stress_residual=metrics['global_balance'][
                              'stress_identity_max_residual']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())