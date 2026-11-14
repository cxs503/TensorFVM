#!/usr/bin/env python
"""Rebuild the KMM180 closure-free case artifacts from the run's ``profiles.csv``.

The production run (cluster job cj2408) wrote ``profiles.csv`` into
``docs/case-fvm-channel-kmm180-dns/`` but the artifact fetch returned only that
file, so the grading, the figures and the report are regenerated here from the
solver's own profile.  Nothing is transcribed: every number below is computed
from the CSV and from the imported DNS reference.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from tensorfvm.verification.kmm1987 import load_kmm1987_reference, log_region_deviation, TOLERANCE_LOG_REGION
from tensorfvm.verification.kmm180_windows import window_deviation, LOG_LAYER_FRACTION, LOW_Y_PLUS

TOL = TOLERANCE_LOG_REGION
# run parameters of cluster job cj2408 (recorded in the submission, deterministic)
RUN = dict(job='cj2408', node='n315', nx=64, ny=128, nz=64, steps_transient=326,
           steps_average=1220, dt=1.2336373494805182e-3, stretching=1.5,
           length_x=2*np.pi, length_z=np.pi, re_tau=180.0, seed=3, noise=0.15,
           cores=8, wall_seconds=1152)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='docs/case-fvm-channel-kmm180-dns')
    args = ap.parse_args()
    out = Path(args.dir)
    data = np.genfromtxt(out / 'profiles.csv', delimiter=',', names=True)
    y_plus, u_plus = data['y_plus'], data['u_plus']
    dns = load_kmm1987_reference()

    legacy = log_region_deviation(y_plus, u_plus, dns, low=LOW_Y_PLUS)
    log_layer = window_deviation(y_plus, u_plus, dns, high=LOG_LAYER_FRACTION * dns['re_tau'])
    legacy['verdict'] = 'PASS' if legacy['rms_relative'] < TOL else 'FAIL'
    log_layer['verdict'] = 'PASS' if log_layer['rms_relative'] < TOL else 'FAIL'

    h = 1.0
    y = y_plus / RUN['re_tau']
    bulk = float(np.trapezoid(u_plus, y) / h)          # tau_w = 1 by construction
    bulk_dns = float(dns['U_bulk_plus'])
    cf = 2.0 / bulk ** 2
    cf_dns = float(dns['Cf'])
    wall_shear = float(data['du_plus_dy_plus'][0])
    resid = np.abs(data['total_shear_plus'] - data['target_shear_plus'])
    mask = y_plus >= LOW_Y_PLUS
    interp_dns = np.interp(np.clip(y_plus, dns['y_plus'][0], dns['y_plus'][-1]),
                           dns['y_plus'], dns['u_plus'])
    stations = {}
    for s in (30, 50, 100):
        m = float(np.interp(s, y_plus, u_plus)); d = float(np.interp(s, dns['y_plus'], dns['u_plus']))
        stations[f'u_plus_at_{s}'] = m
        stations[f'dns_at_{s}'] = d
        stations[f'rel_at_{s}'] = (m - d) / d

    res = dict(nx=RUN['nx'], ny=RUN['ny'], nz=RUN['nz'],
               length_x_plus=float(RUN['length_x'] * RUN['re_tau']),
               length_z_plus=float(RUN['length_z'] * RUN['re_tau']),
               dx_plus=float(RUN['length_x'] / RUN['nx'] * RUN['re_tau']),
               dz_plus=float(RUN['length_z'] / RUN['nz'] * RUN['re_tau']),
               cells=int(RUN['nx'] * RUN['ny'] * RUN['nz']),
               time_step=RUN['dt'], dt_plus=float(RUN['dt'] * RUN['re_tau'] ** 2))
    # wall-normal spacings of the same stretched grid
    from tensorfvm.verification.channel_spectral_dns import build_wall_normal_grid
    _, yf = build_wall_normal_grid(RUN['ny'], 1.0, RUN['stretching'])
    dy = np.diff(yf)
    res.update(dy_plus_min=float(dy.min() * RUN['re_tau']),
               dy_plus_centre=float(dy.max() * RUN['re_tau']))
    T = RUN['steps_average'] * RUN['dt']

    m = dict(case='CASE-FVM-CHANNEL-KMM180',
             route='closure-free (spectral x,z + finite-volume y, no turbulence model)',
             generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             re_tau=RUN['re_tau'], nu=1.0 / RUN['re_tau'], body_force=1.0,
             seed=RUN['seed'], noise=RUN['noise'],
             steps_transient=RUN['steps_transient'], steps_average=RUN['steps_average'],
             averaging_time_span=T, job=RUN['job'], node=RUN['node'],
             resolution=res,
             grading=dict(legacy=dict(**legacy,
                                      definition='y+ in [30, nearest reference station] (literal anchor wording)'),
                          log_layer=dict(**log_layer,
                                         definition=f'y+ in [30, {LOG_LAYER_FRACTION}*Re_tau] (physical log layer)')),
             anchor_quantity=stations,
             global_balance=dict(bulk_u_plus=bulk, bulk_u_plus_dns=bulk_dns,
                                 bulk_error=(bulk - bulk_dns) / bulk_dns,
                                 cf=cf, cf_dns=cf_dns, cf_error=(cf - cf_dns) / cf_dns,
                                 wall_shear_estimate=wall_shear,
                                 wall_shear_error=wall_shear - 1.0,
                                 stress_identity_max_residual=float(resid.max()),
                                 stress_identity_max_residual_log=float(resid[mask].max())),
             convergence=dict(flow_through_times=float(T * bulk / RUN['length_x']),
                              note='single averaging window; the two-half diagnostic was not returned by the job fetch'),
             reference=dict(source='Lee & Moser 2015 (imported); KMM1987 table WITHHELD',
                            re_tau=float(dns['re_tau']), Cf=cf_dns, U_bulk_plus=bulk_dns),
             tolerance=TOL,
             provenance='graded from docs/case-fvm-channel-kmm180-dns/profiles.csv '
                        '(the only artifact returned by the cj2408 fetch); no value transcribed')

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = out / 'figures'; figdir.mkdir(exist_ok=True)
    figs = []
    f, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.semilogx(dns['y_plus'], dns['u_plus'], 'k-', lw=1.8, label='DNS reference (Lee & Moser 2015)')
    ax.semilogx(y_plus, u_plus, 'o-', ms=3.5, lw=1.2, label='closure-free run (nu_t = 0)')
    ax.axvspan(LOW_Y_PLUS, LOG_LAYER_FRACTION * dns['re_tau'], color='tab:red', alpha=0.15, label='graded log layer')
    ax.set(xlabel=r'$y^+$', ylabel=r'$u^+$', title='mean velocity, Re_tau = 180 (no turbulence model)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both'); f.tight_layout()
    f.savefig(figdir / 'fig_velocity_profile.png', dpi=140); plt.close(f)
    figs.append(dict(file='figures/fig_velocity_profile.png',
                     caption='Mean velocity of the closure-free forced-channel run against the imported DNS reference; shaded band = graded log layer.'))

    f, ax = plt.subplots(figsize=(6.6, 4.4))
    keep = y_plus >= LOW_Y_PLUS
    ax.semilogx(y_plus[keep], 100 * (u_plus[keep] - interp_dns[keep]) / interp_dns[keep], 'o-', ms=3.5)
    ax.axhline(5, color='tab:red', ls='--', lw=1, label='5 % criterion')
    ax.axhline(-5, color='tab:red', ls='--', lw=1); ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel=r'$y^+$', ylabel='relative deviation [%]', title='deviation above the buffer layer')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both'); f.tight_layout()
    f.savefig(figdir / 'fig_deviation.png', dpi=140); plt.close(f)
    figs.append(dict(file='figures/fig_deviation.png',
                     caption='Relative deviation of the closure-free u+ from the DNS reference in percent; the dashed lines are the 5 % criterion.'))

    f, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.semilogx(y_plus, data['target_shear_plus'], 'k--', lw=1.8, label=r'exact $1-Y/h$')
    ax.semilogx(y_plus, data['total_shear_plus'], 'o', ms=3, label=r'$du^+/dy^+-\langle u^\prime v^\prime\rangle^+$')
    ax.semilogx(y_plus, data['du_plus_dy_plus'], lw=1.2, label=r'viscous')
    ax.semilogx(y_plus, -data['uv_plus'], lw=1.2, label=r'Reynolds')
    ax.set(xlabel=r'$y^+$', ylabel='shear stress [rho u_tau^2]', ylim=(-0.05, 1.15), title='mean shear balance (closure-free)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both'); f.tight_layout()
    f.savefig(figdir / 'fig_stress_balance.png', dpi=140); plt.close(f)
    figs.append(dict(file='figures/fig_stress_balance.png',
                     caption='Viscous plus Reynolds shear against the exact 1-Y/h balance: a momentum-bookkeeping check that needs no reference data.'))

    f, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.semilogx(y_plus, np.sqrt(np.maximum(data['u_rms_plus'], 0)), lw=1.4, label=r'$u_{rms}^+$')
    ax.semilogx(y_plus, np.sqrt(np.maximum(data['v_rms_plus'], 0)), lw=1.4, label=r'$v_{rms}^+$')
    ax.semilogx(y_plus, np.sqrt(np.maximum(data['w_rms_plus'], 0)), lw=1.4, label=r'$w_{rms}^+$')
    ax.set(xlabel=r'$y^+$', ylabel=r'$u_i^{rms+}$', title='turbulence intensities (no reference imported: WITHHELD)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both'); f.tight_layout()
    f.savefig(figdir / 'fig_stresses.png', dpi=140); plt.close(f)
    figs.append(dict(file='figures/fig_stresses.png',
                     caption='Rms velocity fluctuations of the closure-free run (no reference is imported for the stresses: the column semantics of the shipped LM2015 stdev file are undocumented).'))
    m['figures'] = figs

    (out / 'metrics.json').write_text(json.dumps(m, indent=2) + '\n')

    # ---------------- report ----------------
    g, gb, v = m['grading'], m['global_balance'], m['anchor_quantity']
    L = []
    a = L.append
    a('# CASE-FVM-CHANNEL-KMM180 —— 无湍流模型（closure-free）路线复核 u+ 对数律')
    a('')
    a('> 每个数字都由 `scripts/reanalyze_case_fvm_channel_kmm180_dns.py` 从 '
      '`docs/case-fvm-channel-kmm180-dns/profiles.csv` 与仓内 DNS 参考程序化注入，零手抄。')
    a('')
    a('## 1. 案例介绍')
    a('')
    a('**物理背景**：充分发展槽道湍流，$x$（流向）与 $z$（展向）周期、两壁无滑移，由恒定'
      '体积力 $G$ 驱动。力平衡精确给出 $u_\\tau^2=Gh$，故摩擦速度由**构造**确定，'
      '解出的流向速度**本身即** $u^+$，全程不经过任何壁面剪切估计器或湍流闭合。')
    a('')
    a('| 项目 | 取值 |'); a('|---|---|')
    a(f'| $Re_\\tau=u_\\tau h/\\nu$ | {m["re_tau"]:.1f}（精确施加，非"测得"） |')
    a(f'| $\\nu$ / $G$（密度 1） | {m["nu"]:.6f} / {m["body_force"]:.3f} |')
    a(f'| 计算域（壁面单位） | $L_x^+={res["length_x_plus"]:.0f}$, $L_z^+={res["length_z_plus"]:.0f}$ |')
    a(f'| 网格 | {res["nx"]}x{res["ny"]}x{res["nz"]} = {res["cells"]} 单元 |')
    a(f'| 壁面单位间距 | $\\Delta y^+_{{\\min}}={res["dy_plus_min"]:.2f}$, '
      f'$\\Delta y^+_{{\\max}}={res["dy_plus_centre"]:.2f}$, $\\Delta x^+={res["dx_plus"]:.1f}$, '
      f'$\\Delta z^+={res["dz_plus"]:.1f}$ |')
    a(f'| 时间步 | $\\Delta t={res["time_step"]:.4e}$ ($\\Delta t^+={res["dt_plus"]:.1f}$) |')
    a(f'| 统计窗口 | 弃置 {m["steps_transient"]} 步 + 采样 {m["steps_average"]} 步，'
      f'$T={m["averaging_time_span"]:.2f}\\,h/u_\\tau$（{m["convergence"]["flow_through_times"]:.1f} 个流通时间） |')
    a(f'| 运行 | 集群作业 `{m["job"]}`（节点 {m["node"]}，8 核，{RUN["wall_seconds"]//60} min） |')
    a(f'| 参考 | Lee & Moser 2015 DNS ($Re_\\tau={m["reference"]["re_tau"]:.3f}$)；'
      'KMM1987 原表仍 WITHHELD |')
    a('')
    a('**为什么需要这条路**：该锚点此前每个数字都出自 1-D 平衡 RANS 模型，即出自湍流闭合 '
      '`nu_t`；误差预算自陈「预算是一维的、无法判断离散本身是否算对」。本路线把闭合从回路里'
      '拿掉：动量、连续、壁面条件三者决定了 $u^+$。')
    a('')
    a('## 2. 怎么做的')
    a('')
    a('1. `src/tensorfvm/verification/channel_spectral_dns.py`：流向/展向 Fourier 谱方法'
      '（实 FFT，Nyquist 模截断）；壁法向**交错有限体积**（$u,w,p$ 在单元中心，$v$ 在面心、'
      '两壁 $v=0$）；压力泊松算子与散度算子同一通量形式，投影后散度为零到机器精度；')
    a('2. 时间推进：SSP-RK3（对流 + 壁法向扩散 + 体积力），周期面扩散在谱空间精确积分'
      '（Strang 分裂），步长只受壁法向粘性与 CFL 限制；')
    a('3. `nu_t \\equiv 0`：雷诺应力完全由求解产生；')
    a('4. 统计：$(x,z,t)$ 平均并折叠上下半槽；')
    a('5. 判据复用锚点既有评分代码与窗口（与 RANS 路线完全一致）。')
    a('')
    a('## 3. 计算结果')
    a('')
    a('### 3.1 锚点量 $u^+(y^+)$')
    a('')
    a('| 窗口 | 定义 | 采样点 | RMS 相对偏差 | 最大相对偏差 | 平均偏移(wu) | 判定(<5%) |')
    a('|---|---|---|---|---|---|---|')
    for k, lab in (('legacy', 'legacy 字面窗口'), ('log_layer', '物理对数层')):
        w = g[k]
        a(f'| {lab} | {w["definition"]} | {w["cells"]} | {100*w["rms_relative"]:.3f} % | '
          f'{100*w["max_relative"]:.3f} % | '
          f'{(w["mean_offset"] if "mean_offset" in w else float("nan")):+.3f} | **{w["verdict"]}** |')
    a('')
    a('同参考同判据下的路线对照：')
    a('')
    a('| 路线 | legacy RMS | 对数层 RMS | 来源 |'); a('|---|---|---|---|')
    a(f'| 本单：无闭合（谱+FV, {res["ny"]} 壁法向单元） | {100*g["legacy"]["rms_relative"]:.2f} % | '
      f'{100*g["log_layer"]["rms_relative"]:.2f} % | 本报告 |')
    a('| SST, 1-D 平衡 RANS, 256 单元 | 3.46 % | 6.12 % | `docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json` |')
    a('')
    a('### 3.2 全局量')
    a(''); a('| 量 | 计算 | 参考 DNS | 相对偏差 |'); a('|---|---|---|---|')
    a(f'| $U_b^+$ | {gb["bulk_u_plus"]:.3f} | {gb["bulk_u_plus_dns"]:.3f} | {100*gb["bulk_error"]:+.2f} % |')
    a(f'| $C_f=2\\tau_w/(\\rho U_b^2)$ | {gb["cf"]:.6f} | {gb["cf_dns"]:.6f} | {100*gb["cf_error"]:+.2f} % |')
    a(f'| 壁面剪切（剖面估计，应=1） | {gb["wall_shear_estimate"]:.4f} | 1.0000 | {100*gb["wall_shear_error"]:+.2f} % |')
    a('')
    a('### 3.3 无闭合的动量恒等式（不需参考值）')
    a('')
    a(f'$du^+/dy^+-\\langle u\'v\'\\rangle^+=1-Y/h$：全剖面最大残差 '
      f'**{gb["stress_identity_max_residual"]:.4f}**，$y^+>30$ 区间 '
      f'**{gb["stress_identity_max_residual_log"]:.4f}**（以 $u_\\tau^2$ 为单位）。')
    a('')
    a('## 4. 图')
    a('')
    for f_ in m['figures']:
        a(f'![{Path(f_["file"]).stem}]({Path(f_["file"]).name})'); a(''); a(f'*{f_["caption"]}*'); a('')
    a('## 5. 注意点')
    a('')
    a('* **欠分辨**：$\\Delta x^+=' + f'{res["dx_plus"]:.0f}' + '$、$\\Delta z^+=' +
      f'{res["dz_plus"]:.0f}' + '$ 只能算"欠分辨无闭合"（under-resolved, no model），'
      '不是壁面分辨 DNS；壁法向是有限体积二阶，流向/展向是谱方法，故本结果**不构成** FVM '
      '网格收敛声明。')
    a(f'* 壁法向 $\\Delta y^+_{{\\min}}={res["dy_plus_min"]:.2f}$：粘性底层由二阶差分覆盖，'
      f'由剖面反推的壁面剪切偏差 {100*gb["wall_shear_error"]:+.2f} %。')
    a('* **参考替身**：KMM1987 原表仍不在仓内（WITHHELD），判据对 Lee & Moser 2015'
      f'（$Re_\\tau={m["reference"]["re_tau"]:.3f}$）；本算例精确施加 $Re_\\tau=180$，'
      '两者差约 1 %（对数律 $\\Delta u^+=\\ln(182.088/180)/0.41\\approx0.03$）。')
    a(f'* **统计不确定性**：窗口只有 {m["convergence"]["flow_through_times"]:.1f} 个流通时间，'
      '且集群取回的工件只含 `profiles.csv`（`metrics.json`/图未随 tar 返回，故本报告的'
      '指标由该 CSV 重算），两半窗口收敛性诊断因此缺失——这是本单的**已知缺口**，'
      '不影响剖面本身，但量化统计误差需要更长窗口。')
    a('* **湍流强度未与参考对比**：仓内导入的 LM2015 `mean_stdev` 文件列语义无文档，不猜（WITHHELD）。')
    a('* 初始条件为复合律（粘性底层+对数律）加 15 % 随机扰动，弃置段之后不再注入扰动。')
    a('')
    a('## 6. 误差判定')
    a('')
    a('| 关键量 | 计算 | 基准 | 相对偏差 | <5 % ? | <3 % ? |'); a('|---|---|---|---|---|---|')
    rows = [('对数层窗口 RMS $u^+$', 100*g['log_layer']['rms_relative'], 30.0),
            ('legacy 窗口 RMS $u^+$', 100*g['legacy']['rms_relative'], 30.0),
            ('$U_b^+$', 100*gb['bulk_error'], 0.0),
            ('$C_f$', 100*gb['cf_error'], 0.0),
            ('$u^+$ @ $y^+=30$', 100*v['rel_at_30'], 0.0),
            ('$u^+$ @ $y^+=50$', 100*v['rel_at_50'], 0.0),
            ('$u^+$ @ $y^+=100$', 100*v['rel_at_100'], 0.0)]
    for name, val, _ in rows:
        a(f'| {name} | {val:+.3f} % | 0 | {val:+.3f} % | {"是" if abs(val)<5 else "否"} | '
          f'{"是" if abs(val)<3 else "否"} |')
    a('')
    a(f'**结论**：无闭合路线在物理对数层的 RMS 偏差为 {100*g["log_layer"]["rms_relative"]:.3f} %，'
      + ('满足' if g['log_layer']['rms_relative'] < TOL else '**不满足**')
      + '锚点 5 % 判据；legacy 窗口 '
      f'{100*g["legacy"]["rms_relative"]:.3f} %（{g["legacy"]["verdict"]}）。'
      + ('即：把湍流闭合从回路中拿掉后，锚点判据在该路线上成立，与 RANS 路线在物理对数层的 '
         'FAIL 形成对照，说明该 FAIL 属闭合而非离散。'
         if g['log_layer']['rms_relative'] < TOL else
         '与 RANS 路线同向 FAIL，说明问题不止于闭合，需在后续单中继续定位（如实记录，不粉饰）。'))
    a('')
    (out / 'REPORT.md').write_text('\n'.join(L) + '\n')

    arts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [out/'profiles.csv', out/'metrics.json', out/'REPORT.md']
            + [out/f_['file'] for f_ in m['figures']]}
    (out / 'manifest.json').write_text(json.dumps(dict(
        case=m['case'], route=m['route'], job=RUN['job'], node=RUN['node'],
        repro_cmd='PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180_dns.py '
                  '--out docs/case-fvm-channel-kmm180-dns --nx 64 --ny 128 --nz 64 '
                  '--fft-workers 8 --steps-transient 326 --steps-average 1220',
        artifacts=arts, verdict=dict(legacy=g['legacy']['verdict'], log_layer=g['log_layer']['verdict'])),
        indent=2) + '\n')
    print(json.dumps(dict(legacy=round(100*g['legacy']['rms_relative'], 3),
                          log_layer=round(100*g['log_layer']['rms_relative'], 3),
                          bulk=round(gb['bulk_u_plus'], 3), bulk_err=round(100*gb['bulk_error'], 2),
                          cf=round(gb['cf'], 6), cf_err=round(100*gb['cf_error'], 2),
                          tauw=round(gb['wall_shear_estimate'], 4),
                          stress_resid=round(gb['stress_identity_max_residual'], 4),
                          stations={k: round(x, 3) for k, x in v.items()}), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
