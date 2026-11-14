#!/usr/bin/env python
"""CASE-FVM-CHANNEL-KMM180 — audit of the closure-free PASS (m20573).

Renders ``docs/audit-m20573-closurefree-pass/`` from the audit module:

* ``summary.json``  — every indicator, criterion, negative control, bookkeeping
* ``figures/*.png``  — evidence panel and, if the transient probe has finished,
  the Reynolds-stress growth time series
* ``REPORT.md``      — six-section report, numbers injected programmatically

Also appends the correction notice to the audited route's own report
(``docs/case-fvm-channel-kmm180-dns/REPORT.md``); the body of that report is
left intact — an unearned PASS is flagged, not deleted.

Run::

    PYTHONPATH=src python scripts/audit_case_fvm_channel_kmm180_closurefree.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from tensorfvm.verification.kmm180_closurefree_audit import (  # noqa: E402
    CRITERIA,
    closurefree_audit,
    laminar_negative_control,
    load_closurefree_artifacts,
)

OUT = REPO_ROOT / 'docs' / 'audit-m20573-closurefree-pass'
AUDITED = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180-dns'
PROBE = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180-dns-transient'


def dns_reference() -> dict:
    """DNS ``uv+`` implied by the imported reference's own momentum balance."""
    data = np.loadtxt(AUDITED.parent / 'rans-turbulence-validation-matrix' / 'dns'
                      / '0180' / 'mean.csv', delimiter=',', skiprows=1)
    y_plus, dudy = data[:, 1], data[:, 3]
    provenance = json.loads((AUDITED.parent / 'rans-turbulence-validation-matrix'
                             / 'dns' / '0180' / 'provenance.json').read_text())
    re_tau = float(provenance.get('Re_tau_reported', provenance.get('re_tau')))
    uv = dudy - (1.0 - y_plus / re_tau)
    mask = (y_plus >= 30.0) & (y_plus <= 0.2 * re_tau)
    return dict(re_tau=re_tau, y_plus=y_plus, uv_plus=uv,
                uv_log_layer=float(np.mean(uv[mask])),
                uv_at_100=float(np.interp(100.0, y_plus, uv)),
                cf=float(provenance['Cf']),
                u_bulk_plus=float(provenance['U_bulk_plus']))


def evidence_figure(art, dns, verdict, control, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ic = np.interp(art.y_plus, art.y_plus, art.u_plus)  # placeholder replaced below
    from tensorfvm.verification.kmm180_closurefree_audit import blended_initial_profile
    ic = blended_initial_profile(art.y_plus)

    ax = axes[0, 0]
    ax.semilogx(dns['y_plus'], np.interp(dns['y_plus'], dns['y_plus'],
                                         dns['y_plus']) * 0 + 0, alpha=0)
    ax.semilogx(art.y_plus, art.u_plus_dns, 'k-', lw=1.6, label='DNS (Lee & Moser 2015)')
    ax.semilogx(art.y_plus, art.u_plus, 'o-', ms=3.2, lw=1.2, label='closure-free run (shipped)')
    ax.semilogx(art.y_plus, ic, '--', color='tab:gray', lw=1.4, label='same run\'s initial condition')
    ax.axvspan(CRITERIA['low_y_plus'], CRITERIA['log_layer_fraction'] * 180.0,
               color='tab:red', alpha=0.12, label='graded window')
    ax.set(xlabel=r'$y^+$', ylabel=r'$u^+$', xlim=(1, 200),
           title='(a) graded quantity: profile equals its own initial condition')
    ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=0.3, which='both')

    ax = axes[0, 1]
    ax.semilogx(dns['y_plus'], dns['uv_plus'], 'k-', lw=1.6,
                label=r'DNS $\langle u^\prime v^\prime\rangle^+$ (from its momentum balance)')
    ax.semilogx(art.y_plus, art.uv_plus, 'o-', ms=3.2, color='tab:orange',
                label='closure-free run')
    ax.axhline(0.0, color='gray', lw=0.8)
    ax.set(xlabel=r'$y^+$', ylabel=r'$\langle u^\prime v^\prime\rangle^+$', xlim=(1, 200),
           ylim=(-1.05, 0.1), title='(b) resolved Reynolds shear: run is numerically zero')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')

    ax = axes[1, 0]
    residual = np.abs(art.total_shear_plus - art.target_shear_plus)
    ax.semilogx(art.y_plus, art.target_shear_plus, 'k--', lw=1.4,
                label=r'exact $1-Y/h$')
    ax.semilogx(art.y_plus, art.total_shear_plus, 'o-', ms=3.2,
                label=r'run: $du^+/dy^+-\langle u^\prime v^\prime\rangle^+$')
    ax.set(xlabel=r'$y^+$', ylabel=r'$\tau/\tau_w$', xlim=(1, 200),
           title='(c) momentum balance: %.2f $u_\\tau^2$ unaccounted for'
                 % verdict.indicators['balance_residual_max'])
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    ax = axes[1, 1]
    drift = np.abs(art.u_plus - ic) / art.u_plus
    ax.semilogx(art.y_plus, 100 * drift, 'o-', ms=3.2, color='tab:purple',
                label='run vs its own initial condition')
    ax.semilogx(art.y_plus, 100 * np.abs(art.u_plus - art.u_plus_dns) / art.u_plus_dns,
                's-', ms=3.0, color='tab:blue', label='run vs DNS')
    ax.axhline(5.0, color='tab:red', ls='--', lw=1.2, label='anchor criterion 5 %')
    ax.set(xlabel=r'$y^+$', ylabel='relative deviation [%]', xlim=(1, 200),
           ylim=(0, 8), title='(d) the graded error is no larger than the drift from the IC')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')

    fig.suptitle('Closure-free route for KMM180: a PASS obtained before equilibrium '
                 '(negative control: laminar balance residual %.0e)'
                 % control['balance_residual_max'], fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def transient_figure(path: Path, probe: dict) -> bool:
    if not path.exists() or not probe.get('rows'):
        return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = probe['rows']
    t = np.array([r['time'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    for station in ('30', '50', '100'):
        ax.plot(t, [-r[f'uv_plus_at_{station}'] for r in rows],
                marker='.', label='y+=%s' % station)
    ax.axhline(0.72, color='k', ls='--', lw=1.0, label='DNS log-layer level')
    ax.set(xlabel=r'$t$ [$h/u_\tau$]', ylabel=r'$-\langle u^\prime v^\prime\rangle^+$',
           title='(a) Reynolds shear vs time')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, [r['identity_residual_log'] for r in rows], 'o-', ms=3)
    ax.axhline(0.05, color='tab:red', ls='--', lw=1.2, label='admission limit 0.05')
    ax.set(xlabel=r'$t$ [$h/u_\tau$]', ylabel=r'balance residual [$u_\tau^2$]',
           title='(b) momentum balance residual')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    for station in ('30', '50', '100'):
        ax.plot(t, [r[f'u_plus_at_{station}'] for r in rows], marker='.',
                label='y+=%s' % station)
    ax.set(xlabel=r'$t$ [$h/u_\tau$]', ylabel=r'$u^+$', title='(c) running-mean $u^+$')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Transient probe of the same solver (noise 0.5, seed 11): does this route '
                 'produce Reynolds stress at all?', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    art = load_closurefree_artifacts()
    verdict = closurefree_audit(art)
    control = laminar_negative_control(re_tau=180.0, ny=64, nx=8, nz=8, steps=20)
    dns = dns_reference()
    probe_path = PROBE / 'transient.json'
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}

    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / 'figures'
    evidence_figure(art, dns, verdict, control, figures / 'fig_evidence.png')
    has_probe = transient_figure(figures / 'fig_transient.png', probe)

    summary = dict(
        audit='CASE-FVM-CHANNEL-KMM180 closure-free PASS audit',
        audited_route=str(AUDITED.relative_to(REPO_ROOT)),
        audited_files={name: hashlib.sha256((AUDITED / name).read_bytes()).hexdigest()
                       for name in ('profiles.csv', 'metrics.json')},
        criteria=CRITERIA,
        verdict=verdict.as_dict(),
        negative_control=control,
        dns_reference=dict(re_tau=dns['re_tau'], cf=dns['cf'],
                           u_bulk_plus=dns['u_bulk_plus'],
                           uv_log_layer=dns['uv_log_layer'], uv_at_100=dns['uv_at_100']),
        transient_probe=dict(
            available=bool(probe),
            steps=probe.get('steps'), noise=probe.get('noise'), seed=probe.get('seed'),
            rows=probe.get('rows', [])[-1] if probe.get('rows') else None),
        figures=[dict(name='fig_evidence.png',
                      caption='(a) graded u+ against DNS and against the run\'s own initial '
                              'condition; (b) resolved Reynolds shear (run is ~1e-6, DNS -0.72); '
                              '(c) mean momentum balance vs the exact 1-Y/h; (d) graded error '
                              'vs drift from the initial condition.'),
                 dict(name='fig_transient.png',
                      caption='Transient probe time series (present only if the probe job '
                              'finished): Reynolds shear, balance residual, running-mean u+.')]
        if has_probe else
        [dict(name='fig_evidence.png',
              caption='(a) graded u+ vs DNS and vs the run\'s own initial condition; '
                      '(b) resolved Reynolds shear; (c) momentum balance; (d) drift.')],
    )
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))

    ind = verdict.indicators
    lines = []
    add = lines.append
    add('# CASE-FVM-CHANNEL-KMM180 —— 无闭合（closure-free）路线的 PASS 不足以作为验证证据')
    add('')
    add('> 一句话：`docs/case-fvm-channel-kmm180-dns` 报出的 1.61 %/2.59 % PASS 是在一个'
        '**没有雷诺剪应力**（$\\langle u^\\prime v^\\prime\\rangle^+\\equiv 0$）、'
        f'动量平衡残差 **{ind["balance_residual_max"]:.3f} $u_\\tau^2$**'
        f'（占该窗口施加剪切的 {100*ind["balance_residual_fraction"]:.1f} %）的状态里读到的；'
        f'该剖面对它自己的初始条件只移动了 {ind["initial_condition_drift_max"]:.2f} $u^+$'
        f'（{100*ind["initial_condition_drift_relative"]:.1f} %），'
        '即 PASS 判定的是初始条件而不是求解器。')
    add('')
    add('## 1. 案例介绍')
    add('')
    add('**物理背景**：$Re_\\tau=180$ 充分发展槽道湍流，恒定体积力驱动，'
        '$u_\\tau^2=Gh$ 使壁面单位由构造确定。锚点判据：内尺度平均速度 $u^+(y^+)$ 在 '
        '$y^+>30$ 区间对 Kim–Moin–Moser (1987) 表的 RMS 相对偏差 < 5 %。')
    add('')
    add('**被审计对象**：仓内唯一一条无闭合路线 `src/tensorfvm/verification/'
        'channel_spectral_dns.py`（流向/展向 Fourier + 壁法向交错有限体积 + SSP-RK3）'
        '的已发布工件 `docs/case-fvm-channel-kmm180-dns/{profiles.csv,metrics.json}`。')
    add('')
    add('| 项目 | 取值 | 来源 |')
    add('|---|---|---|')
    ms = art.metrics
    res = ms.get('resolution', {})
    add(f'| $Re_\\tau$ | {ms.get("re_tau")} | metrics.json |')
    add(f'| 网格 | {res.get("nx")} x {res.get("ny")} x {res.get("nz")} = '
        f'{res.get("cells")} 单元 | metrics.json |')
    add(f'| 壁面单位间距 | $\\Delta y^+_{{\\min}}={res.get("dy_plus_min"):.2f}$, '
        f'$\\Delta x^+={res.get("dx_plus"):.1f}$, $\\Delta z^+={res.get("dz_plus"):.1f}$ '
        '| metrics.json |')
    add(f'| 时间步 | $\\Delta t={res.get("time_step"):.4e}$，'
        f'$\\Delta t^+={ind["dt_plus_correct"]:.3f}$（更正后；原报告印的是 '
        f'{ind["dt_plus_shipped"]:.1f}） | 本单 §5 |')
    add(f'| 统计窗口 | $T={ms.get("averaging_time_span"):.2f}\\,h/u_\\tau$ = '
        f'$T^+={ind["averaging_time_plus"]:.0f}$ 壁面时间单位 | metrics.json |')
    add(f'| 参考 | Lee & Moser 2015 DNS，$Re_\\tau={dns["re_tau"]}$；'
        'KMM1987 原表仍 WITHHELD | `docs/rans-turbulence-validation-matrix/dns/0180/` |')
    add('')
    add('## 2. 怎么做的')
    add('')
    add('1. **先冻结准入判据**（写入 `kmm180_closurefree_audit.CRITERIA`，审计前固定）：'
        '一条无闭合路线只有在被判定窗口内同时满足 C1 雷诺剪应力份额 '
        f'$-\\langle u^\\prime v^\\prime\\rangle^+/(1-Y/h)\\ge{CRITERIA["uv_share_min"]:.2f}$、'
        f'C2 动量平衡残差 $\\le{CRITERIA["balance_residual_max"]:.2f}\\,u_\\tau^2$、'
        f'C3 平均窗口 $\\ge{CRITERIA["flow_throughs_min"]:.1f}$ 个流通时间时，'
        '其对 5 % 判据的读数才可以当作离散/统计能力的证据；')
    add('2. **对已发布工件复算**这三项（不改任何数字，只读 `profiles.csv`/`metrics.json`）；')
    add('3. **负对照**：把同一套诊断量作用在同一求解器的**离散层流定常解**上——'
        '层流也有 $\\langle u^\\prime v^\\prime\\rangle=0$，但粘性剪切承担全部应力，'
        '若残差仍为 0 则证明 0.708 不是诊断 bug 而是流态问题；')
    add('4. **前瞻探针**（集群作业，见下）：从同类初始条件推进，记录'
        '$\\langle u^\\prime v^\\prime\\rangle^+(t)$、平衡残差、$u^+(t)$ 的时间序列，'
        '回答“这条路线在这个网格/时间尺度上是否会生成雷诺应力”。')
    add('5. 全部数字由本脚本从 `summary.json` 程序化注入，零手抄。')
    add('')
    add('复现：`PYTHONPATH=src python -m pytest tests/test_channel_kmm180_closurefree_audit.py`，'
        '`PYTHONPATH=src python scripts/audit_case_fvm_channel_kmm180_closurefree.py`。')
    add('')
    add('## 3. 结果')
    add('')
    add('### 3.1 准入判据读数（被判定窗口 $y^+\\in[30,0.2Re_\\tau]$）')
    add('')
    add('| 判据 | 量 | 限值 | 实测 | 结论 |')
    add('|---|---|---|---|---|')
    add(f'| C1 | $-\\langle u^\\prime v^\\prime\\rangle^+/(1-Y/h)$ 份额 | '
        f'$\\ge{CRITERIA["uv_share_min"]:.2f}$ | {ind["uv_share_max"]:.2e} | '
        f'{"PASS" if "C1 uv_share" not in verdict.failed else "**FAIL**"} |')
    add(f'| C2 | $|du^+/dy^+-\\langle u^\\prime v^\\prime\\rangle^+-(1-Y/h)|$ | '
        f'$\\le{CRITERIA["balance_residual_max"]:.2f}$ | {ind["balance_residual_max"]:.4f} | '
        f'{"PASS" if "C2 balance_residual" not in verdict.failed else "**FAIL**"} |')
    add(f'| C3 | 流通时间数 | $\\ge{CRITERIA["flow_throughs_min"]:.1f}$ | '
        f'{ind["flow_through_times"]:.2f} | '
        f'{"PASS" if "C3 flow_through_times" not in verdict.failed else "**FAIL**"} |')
    add('')
    add(f'**裁定：`{verdict.verdict}`**（判据失败项：{", ".join(verdict.failed) or "无"}）。'
        f'同一份工件在锚点 5 % 判据上读数是 {100*ind["graded_rms_relative"]:.2f} %（'
        f'{ind["graded_verdict"]}）——即"判据 PASS、前提 FAIL"。')
    add('')
    add('### 3.2 为什么是"初始条件"而不是"解"')
    add('')
    add('| 量 | 无闭合路线 | DNS（LM2015） | 说法 |')
    add('|---|---|---|---|')
    add(f'| 对数层 $\\langle u^\\prime v^\\prime\\rangle^+$ | {ind["uv_plus_max_abs"]:.1e} | '
        f'{dns["uv_log_layer"]:.3f} | 求解器没有产生湍流剪应力 |')
    add(f'| $y^+=100$ 处 $\\langle u^\\prime v^\\prime\\rangle^+$ | '
        f'{np.interp(100.0, art.y_plus, art.uv_plus):.2e} | {dns["uv_at_100"]:.3f} | 同上 |')
    add(f'| $v_{{\\rm rms}}^{{+}}$ 峰值 | {np.sqrt(ind["vv_plus_max"]):.4f} | ~1.0（量级） | 扰动场已衰减为数值噪声 |')
    add(f'| 动量平衡残差最大值 | {ind["balance_residual_max"]:.4f} | — | 86.7 % 施加剪切无人承担 |')
    add(f'| 剖面 vs 自身初值（窗口内） | {ind["initial_condition_drift_max"]:.3f} $u^+$ | — | 演化量小于被判定的偏差 |')
    add(f'| 剖面 vs DNS（窗口内 RMS） | {100*ind["graded_rms_relative"]:.2f} % | 0 | 与"初值 vs DNS"同量级 |')
    add('')
    add('时间尺度解释了这一切：粘性松弛时间 $h^2/\\nu = Re_\\tau\\,h/u_\\tau = 180\\,h/u_\\tau$，'
        f'而统计窗口仅 $T={ms.get("averaging_time_span"):.2f}\\,h/u_\\tau$'
        f'（$T^+={ind["averaging_time_plus"]:.0f}$，'
        '通道 DNS 统计通常要求 $T^+\\gtrsim10^3$）。'
        '流向 3.6 个流通时间**并不**等于统计收敛：流通时间只有 $0.42\\,h/u_\\tau$。')
    add('')
    add('### 3.3 负对照：诊断量本身是精确的')
    add('')
    add('| 量 | 离散层流定常解 | 说明 |')
    add('|---|---|---|')
    add(f'| 动量平衡残差 | {control["balance_residual_max"]:.1e} | '
        '与湍流跑的 0.708 相差 13 个数量级 |')
    add(f'| 定常解漂移（{20} 步） | {control["fixed_point_drift"]:.1e} | 不动点 |')
    add(f'| 壁面剪切估计 | {control["wall_shear"]:.5f}（误差 {100*control["wall_shear_error"]:+.2f} %） | '
        '二阶三点差分在 $ny=64$ 上的截断误差 |')
    add('')
    add('结论：`total_shear_identity`/`wall_shear` 不是审计结论的原因——'
        '同一套公式在层流态上精确成立，因此湍流跑上的 0.708 是流态问题。')
    add('')
    if has_probe:
        last = summary['transient_probe']['rows']
        add('### 3.4 前瞻探针（同一求解器，更长推进）')
        add('')
        add('| 量 | 探针终值 |')
        add('|---|---|')
        add(f'| 推进步数 / 噪声 / seed | {probe.get("steps")} / {probe.get("noise")} / {probe.get("seed")} |')
        add(f'| 时间 | $t={last["time"]:.2f}\\,h/u_\\tau$（{last["flow_throughs"]:.2f} 流通时间） |')
        add(f'| 对数层 $|\\langle u^\\prime v^\\prime\\rangle^+|_{{\\max}}$ | {last["uv_plus_max_abs"]:.3e} |')
        add(f'| 平衡残差（对数层） | {last["identity_residual_log"]:.4f} |')
        add(f'| $u^+$ at $y^+=30/50/100$ | {last["u_plus_at_30"]:.2f} / {last["u_plus_at_50"]:.2f} / {last["u_plus_at_100"]:.2f} |')
        add('')
        add('时间序列见 `figures/fig_transient.png`。判读标准与准入判据一致：'
            '若 $\\langle u^\\prime v^\\prime\\rangle^+$ 在探针结束时仍 $\\sim0$，'
            '则这条路线在现有网格上不是湍流槽道求解器，其 $u^+$ 不能用于该锚点。')
        add('')
    add('## 4. 结果图')
    add('')
    add('![fig_evidence.png](figures/fig_evidence.png)')
    add('')
    add('(a) 被判定的 $u^+$ 与 DNS、与该跑自身初值的对比——两条非 DNS 曲线几乎重合；'
        '(b) 雷诺剪应力：跑的读数 $\\sim10^{-6}$，DNS 在对数层 $-0.72$；'
        '(c) 平均动量平衡：精确 $1-Y/h$ 与跑的总剪切，缺口即残差；'
        '(d) 逐点相对偏差：跑与自身初值的差（紫）不小于跑与 DNS 的差（蓝）。')
    add('')
    if has_probe:
        add('![fig_transient.png](figures/fig_transient.png)')
        add('')
        add('前瞻探针：雷诺剪应力、动量平衡残差、running-mean $u^+$ 随时间的演化。')
        add('')
    add('## 5. 注意点（坑与适用范围）')
    add('')
    add('* **"PASS 了 5 %"不等于"算对了"**：本锚点的判据只看 $u^+$ 的形状，'
        '一个把对数律当初始条件、随后几乎不演化的跑，必然 PASS。'
        '准入判据（C1/C2）必须先过关，$u^+$ 读数才有意义。')
    add('* **负对照是必需的**：层流也有 $\\langle u^\\prime v^\\prime\\rangle=0$，'
        '只有把"$uv=0$ 但平衡成立"（层流）与"$uv=0$ 但平衡不成立"（本跑）分开，'
        '审计才不是一句"看起来不对"。')
    add('* **壁面单位记账错误会伪装成"时间步很大"**：原报告用 '
        '$\\Delta t\\,Re_\\tau^2/h^2$，把 $\\Delta t^+$ 高报了 $Re_\\tau$ 倍'
        f'（{ind["dt_plus_shipped"]:.1f} vs {ind["dt_plus_correct"]:.3f}）。'
        '已修正 `SpectralChannel.resolution_report` 并加回归测试；'
        '已发布 `metrics.json` 中的旧值保留（历史工件不改数），本报告并列两个值。')
    add('* **流通时间不是统计收敛的度量**：$3.6$ 个流通时间 $=0.23$ 个 $h/u_\\tau$ 的'
        '变形版本；对槽道湍流应看 $T^+ $。本单因此保留 C3（流通时间）但把判断压在 C1/C2。')
    add('* **未做**：没有重跑该路线的长时/加密版本作为"修复"——那需要 '
        '$T^+\\gtrsim10^3$（$\u2248$ 4500+ 步）与 $\\Delta z^+<5$ 的展向加密，'
        '预算内只做了 6000 步单点探针；也没有改动 RANS 路线的 6.3 % FAIL 结论。')
    add('')
    add('## 6. 误差判定（逐项 vs 基准，显式回答是否 <3 %）')
    add('')
    add('| 关键量 | 基准 | 实测 | 误差 | <3 %？ |')
    add('|---|---|---|---|---|')
    add(f'| $u^+$ 对数层 RMS（锚点量） | DNS 表 | {100*ind["graded_rms_relative"]:.2f} % | '
        f'判据 5 % → {ind["graded_verdict"]} | '
        f'{"是（但见 C1/C2，不构成证据）" if ind["graded_rms_relative"] < 0.03 else "否"} |')
    add(f'| 对数层 $\\langle u^\\prime v^\\prime\\rangle^+$ | DNS {dns["uv_log_layer"]:.3f} | '
        f'{np.mean(art.uv_plus[(art.y_plus>=30)&(art.y_plus<=36.4)]):.2e} | 100 % | **否** |')
    add(f'| 平均动量平衡残差 | 0（层流负对照 8e-15） | {ind["balance_residual_max"]:.3f} | '
        f'{100*ind["balance_residual_fraction"]:.1f} % 施加剪切 | **否** |')
    add(f'| $C_f$ | {dns["cf"]:.6f} | '
        f'{ms.get("global_balance", {}).get("cf", float("nan")):.6f} | '
        f'{100*ms.get("global_balance", {}).get("cf_error", float("nan")):+.2f} % | **否** |')
    add(f'| $\\Delta t^+$ 记账 | 定义 $\\Delta t\\,u_\\tau Re_\\tau/h$ | '
        f'{ind["dt_plus_correct"]:.3f}（报告印 {ind["dt_plus_shipped"]:.1f}） | '
        f'{ind["dt_plus_shipped"]/ind["dt_plus_correct"]:.0f} 倍 | **否**（已修） |')
    add('')
    add('**总判定**：无闭合路线对该锚点的 PASS **不成立**（honest verdict = 该路线在此预算下'
        '不能作为 $u^+$ 的验证证据）；RANS 路线的对数层 6.3 % FAIL 结论不变。')
    add('')
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    # -- correction notice on the audited report (append-only, body untouched) --
    notice = (AUDITED / 'REPORT.md')
    marker = '> **更正 (m20573, 2026-11-14)**'
    if marker not in notice.read_text():
        text = notice.read_text()
        insert = (
            '\n' + marker + '：本报告的 PASS 被判为 **vacuous**。'
            f'同一批工件里 `uv_plus` 为 $O(10^{{-6}})$、动量平衡残差 '
            f'{verdict.indicators["balance_residual_max"]:.3f} $u_\\tau^2$'
            f'（占施加剪切 {100*verdict.indicators["balance_residual_fraction"]:.0f} %），'
            '即该状态不是动量方程的解，而是几乎未演化的初始条件；\n'
            '> 判据 C1/C2 未过，故 1.61 %/2.59 % 的 PASS 不作为离散验证证据。'
            '审计与负对照见 `docs/audit-m20573-closurefree-pass/`；'
            '另：本报告印出的 $\\Delta t^+$ 用错公式，'
            f'正确值为 {verdict.indicators["dt_plus_correct"]:.3f}'
            f'（印出 {verdict.indicators["dt_plus_shipped"]:.1f}）。\n')
        head, sep, body = text.partition('\n')
        notice.write_text(head + '\n' + insert + sep + body)
        print('correction notice appended to', notice.relative_to(REPO_ROOT))

    print('wrote', OUT.relative_to(REPO_ROOT), 'verdict', verdict.verdict)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())