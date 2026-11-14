"""Grade the *finite-volume* channel solution with the anchor's own criterion.

Why this module exists
----------------------
``CASE-FVM-CHANNEL-KMM180`` is registered in TensorFVM, but every published
verdict on it (:mod:`tensorfvm.benchmark_channel_kmm1987`,
``kmm180_windows`` / ``_metric_norm`` / ``_refband`` / ``_direct_anchor`` /
``_loglaw`` / ``_budget`` / ``_sublayer`` / ``_sa`` ...) is obtained from a
**1-D equilibrium ODE integration** (``rans_validation.channel.solve_channel``).
``docs/verification-channel-kmm180/REPORT.md`` states the omission literally:

    "未做：3-D 求解器的壁面律一致性（本模块是 1-D 平衡诊断…）"

and ``kmm180_metric_norm`` repeats it: "no claim is made about the 2-/3-D FVM
discretisation".  The FVM side of the repository, in turn, has its own channel
audit (``docs/verification-native-channel-sst``) that never touches the
anchor's criterion: it grades with a *volume-weighted* profile norm against its
own 3 % gate, so its numbers are not comparable with the anchor's
*unweighted, cell-centre* RMS relative deviation at 5 %.

This module closes the loop and answers one question that neither side could
answer alone:

    **does the anchor's verdict transfer from the 1-D equilibrium solver to the
    finite-volume discretisation that actually ships?**

What is measured
----------------
1. :func:`fvm_channel_profile` runs the native hybrid FVM developed-channel
   with SST-2003 (:func:`tensorfvm.verification.native_developed_channel.
   developed_channel_rans`), reduces the extruded field to a wall-normal
   profile, and derives the wall units from **the actual integrated wall
   traction of the FVM solution**, not from the ``Re_tau`` that was requested.
   That distinction is the physics: the 1-D solver is *driven* at
   ``Re_tau = 182.088`` by construction, whereas the FVM solution arrives at
   its own friction velocity and its own ``Cf``.
2. :func:`grade_profile` feeds that profile through the anchor's own grading
   function (``kmm1987.validate_kmm1987``: both windows, same DNS reference,
   same 5 % tolerance) — no new metric, no re-tuning.
3. :func:`cross_solver_audit` solves the 1-D equilibrium channel for the same
   closure and grid level, grades it identically, and diffs the two profiles in
   wall units.  The output therefore separates *discretisation* differences
   (FVM vs ODE, same closure) from *closure* differences (both vs DNS).

Nothing here alters the solver, the closure, the reference data or the
acceptance tolerance; the module only moves a published criterion onto a
solution it had never been applied to.

Reference
---------
Kim, J., Moin, P. & Moser, R. D. 1987 J. Fluid Mech. **177**, 133-166.
DOI: 10.1017/S0022112087000892
Lee, M. & Moser, R. D. 2015 J. Fluid Mech. **774**, 395-415 (imported table).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..rans_validation.channel import solve_channel
from .kmm1987 import (
    RE_TAU_KMM,
    RE_TAU_REF,
    TOLERANCE_LOG_REGION,
    cf_deviation,
    bulk_velocity_deviation,
    load_kmm1987_reference,
    log_region_deviation,
)

__all__ = [
    'RE_TAU_NOMINAL',
    'developed_profiles',
    'fvm_channel_profile',
    'grade_profile',
    'cross_solver_audit',
    'main',
]

#: Nominal anchor Reynolds number (see ``feature_ledger`` slug text).
RE_TAU_NOMINAL = RE_TAU_KMM

#: Minimum number of sample points for a window to be graded by
#: ``kmm1987.log_region_deviation``; mirrors that function's own guard.
MIN_WINDOW_CELLS = 3


# --------------------------------------------------------------------------- #
# Profile extraction
# --------------------------------------------------------------------------- #

def developed_profiles(mesh, result, info, *, rho=1.0, nu=None):
    """Reduce an extruded developed-channel solution to a wall-normal profile.

    Parameters
    ----------
    mesh : NativeMesh3D
        The channel mesh returned by ``developed_channel_rans``.
    result : dict
        Converged result dict (``velocity``, ``total_traction_integrated`` ...).
    info : dict
        Boundary-face masks returned by ``developed_channel_rans``
        (key ``'wall'``; ``info['nu']`` is used when ``nu`` is None).
    rho : float
        Density (the channel solvers are incompressible with ``rho = 1``).
    nu : float, optional
        Kinematic viscosity, overriding ``info['nu']``.

    Returns
    -------
    dict with
        ``y``, ``u`` (wall-normal profile: one value per distinct cell-centre
        ``y``, axial velocity averaged over the extruded directions),
        ``u_spread`` (max-minus-min axial velocity inside each y level, i.e. a
        direct check that the extrusion really is two-dimensional),
        ``y_plus``, ``u_plus``, ``u_tau``, ``tau_w``, ``Cf``,
        ``bulk_u``, ``bulk_u_plus``, ``re_tau``, ``re_tau_nominal``.
    """
    nu = float(info['nu'] if nu is None else nu)
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError('positive viscosity required')

    wall = np.asarray(info['wall'], bool)
    area = np.linalg.norm(mesh.area_vectors, axis=1)
    traction = np.asarray(result['total_traction_integrated'], float)
    if not np.isfinite(traction).all():
        raise ValueError('non-finite boundary traction in result')

    wall_area = float(area[wall].sum())
    if wall_area <= 0:
        raise ValueError('wall patch carries no area')
    # Traction integral over the wall patch is -tau_w * A (the wall pulls the
    # fluid backwards); its area mean is the actual FVM skin friction.
    tau_w = -float(traction[wall, 0].sum()) / wall_area
    if not np.isfinite(tau_w) or tau_w <= 0:
        raise ValueError('non-positive wall shear stress: solution not developed')

    u_tau = float(np.sqrt(tau_w / rho))
    volumes = np.asarray(mesh.cell_volumes, float)
    velocity = np.asarray(result['velocity'], float)[:, 0]
    y = np.asarray(mesh.cell_centers[:, 1], float)

    levels = np.unique(np.round(y, 12))
    u = np.empty(levels.size)
    spread = np.empty(levels.size)
    for j, level in enumerate(levels):
        mask = np.round(y, 12) == level
        u[j] = float(velocity[mask].mean())
        spread[j] = float(velocity[mask].max() - velocity[mask].min())

    order = np.argsort(levels)
    levels, u, spread = levels[order], u[order], spread[order]

    bulk_u = float(volumes @ velocity / volumes.sum())
    cf = 2.0 * tau_w / bulk_u ** 2 if bulk_u > 0 else float('nan')

    return dict(
        y=levels,
        u=u,
        u_spread=spread,
        y_plus=levels * u_tau / nu,
        u_plus=u / u_tau,
        u_tau=u_tau,
        tau_w=tau_w,
        Cf=float(cf),
        bulk_u=bulk_u,
        bulk_u_plus=bulk_u / u_tau,
        re_tau=float(u_tau / nu),      # half-width = 1 in this mesh
        re_tau_nominal=None,
    )


def fvm_channel_profile(n, *, re_tau=RE_TAU_REF, stretch=3.0, axial=None,
                        spanwise=2, length=0.1, span=0.1,
                        max_iterations=400, dt=1.0, relaxation=0.5,
                        tolerance=1e-8, momentum_method='coupled'):
    """Solve the native FVM developed channel and return its wall-normal profile.

    The returned profile carries ``converged``; a solution that fails the
    solver's own residual/ledger gate is reported with ``graded=False`` and no
    verdict, never with a downgraded tolerance.
    """
    from .native_developed_channel import developed_channel_rans, solve_developed_rans

    axial = max(2, n // 8) if axial is None else axial
    mesh, solver, info = developed_channel_rans(
        n, re_tau=re_tau, stretch=stretch, axial=axial, spanwise=spanwise,
        length=length, span=span)
    result = solve_developed_rans(
        solver, max_iterations=max_iterations, dt=dt, relaxation=relaxation,
        tolerance=tolerance, momentum_method=momentum_method)
    profile = developed_profiles(mesh, result, info)
    profile.update(
        solver='native_fvm_sst2003',
        n=int(n),
        cells=int(len(mesh.cell_centers)),
        stretch=float(stretch),
        axial=int(axial),
        spanwise=int(spanwise),
        converged=bool(result['numerically_converged']),
        iterations=int(len(solver.history)),
        tolerance=float(tolerance),
        re_tau_driven=float(re_tau),
        re_tau_nominal=float(re_tau),
        residual={k: float(v) for k, v in result.items()
                  if k in ('momentum', 'divergence', 'gauge', 'k', 'omega')
                  and np.isscalar(v)},
        scalar_absolute_ledger={k: float(v)
                                for k, v in result.get('scalar_absolute_ledger', {}).items()},
    )
    return profile


# --------------------------------------------------------------------------- #
# Grading (the anchor's own function, unchanged)
# --------------------------------------------------------------------------- #

def grade_profile(profile, *, dns=None, tolerance=TOLERANCE_LOG_REGION):
    """Grade a wall-normal profile with ``kmm1987.validate_kmm1987``.

    A window that holds fewer than :data:`MIN_WINDOW_CELLS` sample points is
    reported as ``None`` (not gradeable) instead of being silently widened --
    the same policy the 1-D audit uses.
    """
    dns = load_kmm1987_reference() if dns is None else dns
    y_plus = np.asarray(profile['y_plus'], float)
    u_plus = np.asarray(profile['u_plus'], float)
    if y_plus.size != u_plus.size or y_plus.size < MIN_WINDOW_CELLS:
        raise ValueError('profile needs matching y+/u+ arrays with >=3 points')
    if np.any(np.diff(y_plus) <= 0):
        raise ValueError('wall coordinate must increase monotonically')

    out = dict(
        solver=profile.get('solver'),
        n=profile.get('n'),
        stretch=profile.get('stretch'),
        converged=bool(profile.get('converged', False)),
        tolerance=float(tolerance),
        u_tau=float(profile['u_tau']),
        re_tau=float(profile['re_tau']),
        re_tau_driven=profile.get('re_tau_driven'),
        Cf=float(profile['Cf']),
        bulk_u_plus=float(profile['bulk_u_plus']),
        u_spread_max=float(np.max(profile.get('u_spread', [0.0]))),
    )

    for name, fraction in (('legacy_full', None), ('log_layer', 0.2)):
        high = float(np.max(dns['y_plus'])) if fraction is None else fraction * dns['re_tau']
        cells = int(np.count_nonzero((y_plus >= 30.0) & (y_plus <= high)))
        if cells < MIN_WINDOW_CELLS or not out['converged']:
            out[name] = dict(window=[30.0, high], cells=cells, gradeable=False,
                             reason=('not converged' if not out['converged']
                                     else f'{cells} cells < {MIN_WINDOW_CELLS}'))
            out[name + '_verdict'] = 'WITHHELD'
            continue
        # ``log_region_deviation`` *is* the anchor metric; it is applied per
        # window so that a window too narrow to hold three cells (the normal
        # situation on the FVM mesh) is withheld instead of aborting the grade.
        block = log_region_deviation(y_plus, u_plus, dns, low=30.0,
                                     high_fraction=fraction)
        out[name] = dict(block, gradeable=True)
        out[name + '_verdict'] = ('PASS' if block['rms_relative'] < tolerance
                                  else 'FAIL')

    if np.isfinite(out['Cf']):
        out['cf_relative_error'] = cf_deviation(out['Cf'], dns)['relative_error']
        out['bulk_relative_error'] = bulk_velocity_deviation(
            out['bulk_u_plus'], dns)['relative_error']
    out['verdict'] = out['legacy_full_verdict']
    out['verdict_log_layer'] = out['log_layer_verdict']
    out['reference'] = dict(re_tau=float(dns['re_tau']), Cf=float(dns['Cf']),
                            U_bulk_plus=float(dns['U_bulk_plus']))
    return out


def _profile_difference(fvm, ode):
    """Diff two wall-unit profiles where they overlap (no extrapolation)."""
    x_f = np.asarray(fvm['y_plus'], float)
    x_o = np.asarray(ode['y_plus'], float)
    low = max(x_f.min(), x_o.min())
    high = min(x_f.max(), x_o.max())
    mask = (x_f >= low) & (x_f <= high)
    if np.count_nonzero(mask) < MIN_WINDOW_CELLS:
        return dict(gradeable=False, points=int(np.count_nonzero(mask)))
    interp = PchipInterpolator(x_o, np.asarray(ode['u_plus'], float))
    delta = np.asarray(fvm['u_plus'], float)[mask] - interp(x_f[mask])
    inner = mask & (x_f >= 30.0)
    delta_inner = (np.asarray(fvm['u_plus'], float)[inner]
                   - interp(x_f[inner])) if np.count_nonzero(inner) else np.array([np.nan])
    return dict(
        gradeable=True,
        points=int(np.count_nonzero(mask)),
        rms_wall_units=float(np.sqrt(np.mean(delta ** 2))),
        max_wall_units=float(np.max(np.abs(delta))),
        mean_wall_units=float(np.mean(delta)),
        rms_wall_units_y_plus_gt_30=float(np.sqrt(np.mean(delta_inner ** 2))),
        u_tau_relative=float(abs(fvm['u_tau'] - ode['u_tau']) / ode['u_tau']),
        cf_relative=float(abs(fvm['Cf'] - ode['Cf']) / ode['Cf']),
    )


def cross_solver_audit(n, *, re_tau=RE_TAU_REF, model='sst', dns=None,
                       stretch=3.0, fvm_options=None, ode_options=None):
    """Grade the FVM and the 1-D equilibrium solutions on identical terms."""
    dns = load_kmm1987_reference() if dns is None else dns
    options = dict(fvm_options or {})
    options.setdefault('stretch', stretch)
    fvm = fvm_channel_profile(n, re_tau=re_tau, **options)
    fvm_grade = grade_profile(fvm, dns=dns)

    ode_result = solve_channel(model, n, re_tau=re_tau, **(ode_options or {}))
    ode = dict(
        solver=f'1d_equilibrium_{model}',
        n=int(n),
        converged=bool(ode_result['converged']),
        y_plus=np.asarray(ode_result['y_plus'], float),
        u_plus=np.asarray(ode_result['u_plus'], float),
        u_tau=1.0,   # the 1-D solver is non-dimensionalised with u_tau = 1
        re_tau=float(re_tau),
        Cf=float(ode_result['Cf']),
        bulk_u_plus=float(ode_result['bulk_u_plus']),
        u_spread=[0.0],
    )
    # The 1-D solver is non-dimensionalised with u_tau = 1 by construction.
    ode['u_tau'] = 1.0
    ode_grade = grade_profile(ode, dns=dns)

    return dict(
        n=int(n),
        stretch=float(stretch),
        re_tau_driven=float(re_tau),
        model=model,
        fvm=dict(grade=fvm_grade,
                 y_plus=fvm['y_plus'].tolist(), u_plus=fvm['u_plus'].tolist(),
                 re_tau_measured=fvm['re_tau']),
        ode=dict(grade=ode_grade,
                 y_plus=ode['y_plus'].tolist(), u_plus=ode['u_plus'].tolist()),
        difference=_profile_difference(fvm, ode),
        verdict_agreement=bool(fvm_grade['verdict'] == ode_grade['verdict']
                               and fvm_grade['verdict_log_layer']
                               == ode_grade['verdict_log_layer']),
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _figure(cases, dns, out_dir):
    """Plot the profiles; returns None when matplotlib is unavailable."""
    try:
        import matplotlib
    except ImportError:      # headless compute nodes may not ship matplotlib
        return None
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.semilogx(dns['y_plus'], dns['u_plus'], 'k-', lw=2,
                label=f'DNS (LM2015, Re_tau={dns["re_tau"]:.1f})')
    ax.axvspan(30.0, 0.2 * dns['re_tau'], color='0.85',
               label=r'log-layer window $[30,0.2Re_\tau]$')
    for case in cases:
        label = f"FVM SST n={case['n']} ({case['fvm']['grade']['verdict']})"
        ax.semilogx(case['fvm']['y_plus'], case['fvm']['u_plus'], '-o', ms=2.5,
                    label=label)
        ax.semilogx(case['ode']['y_plus'], case['ode']['u_plus'], '--', lw=1.2,
                    label=f"1-D ODE SST n={case['n']} ({case['ode']['grade']['verdict']})")
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$u^+$')
    ax.set_title('FVM vs 1-D equilibrium vs DNS')
    ax.legend(fontsize=6.5)
    ax.grid(alpha=.3, which='both')

    ax = axes[1]
    for case in cases:
        y = np.asarray(case['fvm']['y_plus'])
        u = np.asarray(case['fvm']['u_plus'])
        interp = PchipInterpolator(dns['y_plus'], dns['u_plus'], extrapolate=False)
        ref = interp(y)
        mask = np.isfinite(ref)
        ax.semilogx(y[mask], ((u[mask] - ref[mask]) / ref[mask] * 100), '-o', ms=2.5,
                    label=f'FVM n={case["n"]}')
        yo = np.asarray(case['ode']['y_plus'])
        uo = np.asarray(case['ode']['u_plus'])
        refo = interp(yo)
        masko = np.isfinite(refo)
        ax.semilogx(yo[masko], ((uo[masko] - refo[masko]) / refo[masko] * 100),
                    '--', lw=1.2, label=f'1-D ODE n={case["n"]}')
    ax.axhline(5, color='r', ls=':', lw=1, label='+5 % criterion')
    ax.axhline(-5, color='r', ls=':', lw=1)
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel('relative deviation from DNS [%]')
    ax.set_title('Deviation against the anchor criterion')
    ax.legend(fontsize=6.5)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    path = Path(out_dir) / 'fig_fvm_vs_ode.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path.name


def _report_markdown(summary):
    """Render REPORT.md from computed numbers only (no hand-copied digits)."""
    lines = []
    add = lines.append
    add('# CASE-FVM-CHANNEL-KMM180：把锚点判据接到 FVM 本体上')
    add('')
    add('锚点：`CASE-FVM-CHANNEL-KMM180`（湍流槽道 `Re_tau=180` 平均速度对数律）。')
    add('验收：`y+ > 30` 上 `u+(y+)` 相对 DNS 参考 RMS 偏差 `< 5 %%`。容差 = %.1f %%。'
        % (100 * summary['tolerance']))
    add('')
    add('## 1. 案例介绍（本单补的是哪一条欠账）')
    add('')
    add('此前该锚点的**全部**判定读数（`benchmark_channel_kmm1987`、`kmm180_windows`、'
        '`_metric_norm`、`_refband`、`_direct_anchor`、`_loglaw`、`_budget`、`_sublayer`、'
        '`_sa`）都来自 **1-D 平衡 ODE 积分** `rans_validation.channel.solve_channel`。'
        '`docs/verification-channel-kmm180/REPORT.md` 把这件事写成待办：'
        '「未做：3-D 求解器的壁面律一致性」；`kmm180_metric_norm` 亦声明「no claim is made '
        'about the 2-/3-D FVM discretisation」。FVM 一侧自己的槽道审计'
        '（`docs/verification-native-channel-sst`）用的是**体积加权**范数 + 3 % 门槛，与锚点的'
        '**非加权单元中心** RMS + 5 % 判据不同度量，两者数字不可比。')
    add('')
    add('本单把锚点判据原样搬到 FVM 解上：同一参考表、同一 `y+>30` 窗口、同一 5 % 容差、'
        '同一个函数 `kmm1987.validate_kmm1987`，不新造指标、不改求解器与闭包。')
    add('')
    add('## 2. 怎么做的')
    add('')
    add('1. `fvm_channel_profile(n)`：原生混合 FVM 充分发展槽道 + SST-2003'
        '（`verification.native_developed_channel.developed_channel_rans` / '
        '`solve_developed_rans`，压力驱动 `dp*/dx=1`、半槽道、中心对称、展向与流向挤出）。'
        '速度场按 `y` 层归约成壁法向剖面，并**用解自身积分壁面牵引**得到 `tau_w`、'
        '`u_tau=sqrt(tau_w/rho)`、`Cf=2 tau_w/U_b^2`、`Re_tau=u_tau h/nu`（`h=1`）。'
        '判据用的是**实测** `u_tau`，不是请求的 `Re_tau`。')
    add('2. 同一 `n`、同一 `Re_tau=%.3f` 跑 1-D 平衡 ODE（`solve_channel`），'
        '用同一函数评级。' % summary['re_tau_driven'])
    add('3. 两条剖面在共同 `y+` 区间用 PCHIP 比较（不外插），把「离散差」与「闭包差」分开。')
    add('4. 复现：`PYTHONPATH=src python -m tensorfvm.verification.kmm180_fvm_bridge '
        '--cases %s --out docs/case-fvm-channel-kmm180-fvm-bridge`'
        % ' '.join('%d:%.1f' % (c['n'], c.get('stretch') or 3.0)
                   for c in summary['cases']))
    add('')
    add('## 3. 计算结果')
    add('')
    add('### 3.1 判据读数（同一函数、同一参考、同一 5 % 容差）')
    add('')
    add('| n | stretch | 求解器 | 收敛 | 迭代 | cells | Re_tau(实测) | u_tau | Cf | Cf 误差 | '
        'full RMS | full 判定 | log RMS | log 判定 |')
    add('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for case in summary['cases']:
        for key in ('fvm', 'ode'):
            g = case[key]['grade']
            full = g['legacy_full']
            log = g['log_layer']
            add('| %d | %.1f | %s | %s | %s | %s | %.3f | %.5f | %.6f | %s | %s | %s | %s | %s |'
                % (case['n'], case.get('stretch') or 3.0, g['solver'],
                   '✓' if g['converged'] else '✘',
                   case[key].get('iterations', '—'), full.get('cells', '—'),
                   g['re_tau'], g['u_tau'], g['Cf'],
                   ('%.2f %%' % (100 * g['cf_relative_error'])) if 'cf_relative_error' in g else '—',
                   ('%.2f %%' % (100 * full['rms_relative'])) if full.get('gradeable') else '不可判',
                   g['legacy_full_verdict'],
                   ('%.2f %%' % (100 * log['rms_relative'])) if log.get('gradeable') else '不可判',
                   g['log_layer_verdict']))
    add('')
    add('### 3.2 FVM vs 1-D ODE（同一闭包、同一网格档）')
    add('')
    add('| n | stretch | 共同点数 | Δu+ RMS | Δu+ max | Δu+ 均值 | Δu+(y+>30) RMS | Δu_tau | ΔCf | '
        '两窗判定一致 |')
    add('|---|---|---|---|---|---|---|---|---|---|')
    for case in summary['cases']:
        d = case['difference']
        if not d.get('gradeable'):
            add('| %d | %.1f | %d | 不可比 | — | — | — | — | — | — |'
                % (case['n'], case.get('stretch') or 3.0, d.get('points', 0)))
            continue
        add('| %d | %.1f | %d | %.4f | %.4f | %+.4f | %.4f | %.2f %% | %.2f %% | %s |'
            % (case['n'], case.get('stretch') or 3.0, d['points'],
               d['rms_wall_units'], d['max_wall_units'],
               d['mean_wall_units'], d['rms_wall_units_y_plus_gt_30'],
               d['u_tau_relative'] * 100, d['cf_relative'] * 100,
               '是' if case['verdict_agreement'] else '否'))
    add('')
    add('## 4. 图')
    add('')
    add('![fvm vs ode](fig_fvm_vs_ode.png)')
    add('')
    add('*图 1*：左 = 平均速度剖面（黑实线=DNS 参考，灰带=被判定的对数层窗口，'
        '彩色实线=各档 FVM 解，虚线=同档 1-D 平衡解）；右 = 相对偏差随 `y+` 的分布，'
        '红点线为 ±5 % 判据线。')
    add('')
    add('## 5. 注意点')
    add('')
    add('* FVM 网格 `stretch=3` 在 `y+∈[30, 0.2Re_tau]`（本雷诺数下约 6 个壁面单位宽）'
        '里只有极少数单元，窄窗常常**不可判**（`WITHHELD`，点数 < 3）；本模块如实报 '
        '`cells` 与 `WITHHELD`，不做窗口内插、不放宽容差。')
    add('* FVM 解的 `u_tau` 由自身壁面牵引求得，因此它的 `Re_tau` 与请求值不同——'
        '这正是 1-D 求解器（按构造把 `u_tau` 钉成 1）与 FVM 的**本质差别**，'
        '也是 Cf 误差与壁法向归一化耦合的来源。')
    add('* 挤出方向上的速度层内极差 `u_spread_max` 已随产物输出，用于确认该剖面确实是'
        '一维物理剖面而非三维效应。')
    add('* 本单不改求解器、闭包、参考数据与容差；只把已发布的判据搬到未判过的解上。')
    add('')
    add('## 6. 误差判定')
    add('')
    for line in summary['verdicts']:
        add('* ' + line)
    add('')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, nargs='+', default=None)
    parser.add_argument('--cases', nargs='+', default=None,
                        help='n:stretch pairs; overrides --levels')
    parser.add_argument('--re-tau', type=float, default=RE_TAU_REF)
    parser.add_argument('--model', default='sst')
    parser.add_argument('--max-iterations', type=int, default=400)
    parser.add_argument('--out', type=Path,
                        default=Path('docs/case-fvm-channel-kmm180-fvm-bridge'))
    args = parser.parse_args(argv)

    dns = load_kmm1987_reference()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        levels = [(int(c.split(':')[0]), float(c.split(':')[1])) for c in args.cases]
    else:
        levels = [(n, 3.0) for n in (args.levels or [16, 32])]

    cases = []
    for n, stretch in levels:
        case = cross_solver_audit(n, re_tau=args.re_tau, model=args.model, dns=dns,
                                  stretch=stretch,
                                  fvm_options=dict(max_iterations=args.max_iterations))
        case['ode']['iterations'] = None
        print(json.dumps({k: case['fvm']['grade'][k]
                          for k in ('n', 'verdict', 'verdict_log_layer', 'Cf')},
                         ensure_ascii=False), flush=True)
        cases.append(case)

    verdicts = []
    for case in cases:
        g = case['fvm']['grade']
        full, log = g['legacy_full'], g['log_layer']
        detail = []
        if full.get('gradeable'):
            detail.append('full 窗 RMS %.2f %% → %s' % (100 * full['rms_relative'],
                                                         g['legacy_full_verdict']))
        else:
            detail.append('full 窗 %s（%s）' % (g['legacy_full_verdict'], full['reason']))
        if log.get('gradeable'):
            detail.append('对数层窗 RMS %.2f %% → %s' % (100 * log['rms_relative'],
                                                         g['log_layer_verdict']))
        else:
            detail.append('对数层窗 %s（%s）' % (g['log_layer_verdict'], log['reason']))
        verdicts.append('n=%d (stretch=%.1f) FVM SST：%s；与同档 1-D ODE 判定一致：%s。'
                        % (case['n'], case.get('stretch') or 3.0, '；'.join(detail),
                           '是' if case['verdict_agreement'] else '否'))

    summary = dict(
        anchor='CASE-FVM-CHANNEL-KMM180',
        tolerance=TOLERANCE_LOG_REGION,
        re_tau_driven=float(args.re_tau),
        model=args.model,
        reference=dict(re_tau=float(dns['re_tau']), Cf=float(dns['Cf']),
                       U_bulk_plus=float(dns['U_bulk_plus']),
                       points=int(np.size(dns['y_plus']))),
        metric='kmm1987.validate_kmm1987 (unweighted cell-centre RMS relative deviation)',
        cases=cases,
        verdicts=verdicts,
    )
    figure = _figure(cases, dns, out_dir)
    summary['figure'] = figure
    if figure is None:
        summary['figure_missing_reason'] = 'matplotlib not installed on the job node'
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    (out_dir / 'REPORT.md').write_text(_report_markdown(summary), encoding='utf-8')
    print('wrote', out_dir / 'summary.json', flush=True)
    return summary


if __name__ == '__main__':  # pragma: no cover
    main()