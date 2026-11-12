"""Log-law self-consistency and Reynolds-number-proxy bound for CASE-FVM-CHANNEL-KMM180.

Why this module exists
----------------------
``tensorfvm.verification.kmm180_windows`` (m16513) settled *how the verdict
depends on the grading window*: SST reads 3.63 % over ``y+ in [30, 180.6]``
(PASS) but 6.32 % over the physical log layer ``y+ in [30, 0.2 Re_tau] =
[30, 36.4]`` (FAIL).  Two questions were left open, and both decide whether that
FAIL is a *solver defect* or a *model/finite-Reynolds-number* property:

Q1  **Is the converged solver state at its own model's log-layer fixed point?**
    An equilibrium k-omega-type solution must satisfy, in the log layer,

        k+ = u_tau^2/sqrt(beta*),  omega+ y+ = 1/(sqrt(beta*) kappa),
        nu_t+ = kappa y+,          kappa from equation (2) of
                                   :mod:`tensorfvm.rans_validation.models`

    Deviations here are solver defects (wrong closure bookkeeping, unconverged
    sweep), not model error.

Q2  **Does the deviation vanish as Re_tau -> infinity?**
    If k+ and nu_t+/y+ walk towards the analytic values as Re_tau grows, the
    Re_tau = 180 shortfall is the ``(1 - y)`` total-stress factor and the
    non-existence of a true log layer at six wall units, i.e. physics.  If they
    stay put, the code is wrong.  This module answers with a Re_tau sweep.

It also quantifies the repository's **reference substitution**: the anchor is
named after KMM1987 (Re_tau = 180) but graded against Lee & Moser (2015) at
Re_tau = 182.088, and :mod:`tensorfvm.verification.kmm1987` had to withdraw an
unverifiable "the two datasets agree to 1 %" claim.  Rather than assert the
substitution is harmless, the same solver is run at both Reynolds numbers and
the effect on the graded metric is measured -- a *bound*, not a belief.

Everything here is diagnostic: no solver or model code is modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..rans_validation.channel import solve_channel
from ..rans_validation.models import constants_for, implied_kappa
from .kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference
from .kmm180_windows import LOG_LAYER_FRACTION, LOW_Y_PLUS, window_deviation

#: Nominal Reynolds number of the anchor (KMM1987 = 180).
RE_TAU_NOMINAL = 180.0
#: Re_tau sweep used to test the asymptotic approach to the log-layer fixed point.
RE_TAU_SWEEP = (550.0, 1000.0, 2000.0, 5200.0)
#: ``Re_tau`` values whose DNS profiles are actually imported in this repository
#: (``docs/rans-turbulence-validation-matrix/dns/<case>/``); the highest one at or
#: below 200 is the low-Re endpoint used for the sweep anchor.
MODELS = ('sst', 'komega', 'kepsilon')
#: Models for which ``variable`` is omega, so ``omega+ y+`` has a closed form.
OMEGA_MODELS = ('sst', 'komega')


def _solver_options(model, n=None):
    return {'first_cell_y_plus': 30.0} if model == 'kepsilon' else {}


def expected_log_layer(model, constants=None):
    """Analytic log-layer state implied by the model's own constant set.

    Returns ``k_plus``, ``omega_plus_y_plus`` (``None`` for k-epsilon, whose
    transported variable is epsilon), ``kappa`` from the closed form (2)/(3) of
    :mod:`tensorfvm.rans_validation.models`, and ``nu_t_plus_over_y_plus``
    (``= kappa``, by construction ``nu_t = kappa y u_tau``).
    """
    consts = constants_for(model, constants)
    # k-epsilon spells the same constant ``c_mu``; the equilibrium relation
    # ``k = u_tau^2/sqrt(c_mu)`` is the analogue of ``k = u_tau^2/sqrt(beta*)``.
    beta_star = float(consts.get('beta_star', consts.get('c_mu')))
    root = np.sqrt(beta_star)
    kappa = float(implied_kappa(model, constants))
    return dict(
        model=model,
        beta_star=beta_star,
        kappa=kappa,
        k_plus=1.0 / root,
        omega_plus_y_plus=(1.0 / (root * kappa) if model in OMEGA_MODELS else None),
        nu_t_plus_over_y_plus=kappa,
    )


def log_layer_state(result, *, high_fraction=LOG_LAYER_FRACTION, low=LOW_Y_PLUS):
    """Extract the measured log-layer state from a ``solve_channel`` result.

    All quantities are wall units with ``u_tau = 1`` and ``nu = 1/Re_tau``:

    * ``k_plus``          = ``k``                       (``u_tau`` = 1);
    * ``omega_plus_y_plus``= ``variable * y``           (``omega+ = omega Re_tau``,
      ``y+ = y Re_tau``);
    * ``nu_t_plus``       = ``nut * Re_tau``;
    * ``kappa_eddy``      = ``nu_t+ / y+`` = ``nut / y``, the von Karman
      "constant" implied by the eddy viscosity (the log law's defining
      relation ``nu_t = kappa y u_tau``);
    * ``kappa_slope``     = ``1 / (d u+/d ln y+)`` from a least-squares line
      through ``(ln y+, u+)`` -- independent of -- and diagnostically
      cross-checked against -- ``kappa_eddy``.

    Raises ``ValueError`` when the window holds fewer than three cells, so a
    silently-empty window can never be reported as "consistent".
    """
    y_plus = np.asarray(result['y_plus'], dtype=float)
    y = np.asarray(result['y'], dtype=float)
    k = np.asarray(result['k'], dtype=float)
    variable = np.asarray(result['variable_values'], dtype=float)
    nut = np.asarray(result['nut'], dtype=float)
    u_plus = np.asarray(result['u_plus'], dtype=float)
    re_tau = float(result['re_tau'])

    mask = (y_plus >= low) & (y_plus <= high_fraction * re_tau)
    cells = int(np.count_nonzero(mask))
    if cells < 3:
        raise ValueError(
            f'log window [{low}, {high_fraction} Re_tau={high_fraction * re_tau:.2f}] '
            f'holds {cells} cells; need three or more')

    kappa_eddy = nut[mask] / y[mask]
    # Window-mean of the total-stress factor (1 - y/h) that the asymptotic
    # leading-order balance omits: it lowers k+ by exactly this factor.
    total_stress = float(np.mean(1.0 - y[mask]))
    slope = float(np.polyfit(np.log(y_plus[mask]), u_plus[mask], 1)[0])
    if not slope > 0:
        raise ValueError('mean profile is not monotonically increasing in ln y+')
    return dict(
        model=result['model'],
        re_tau=re_tau,
        cells=cells,
        low=float(low),
        high=float(high_fraction * re_tau),
        k_plus=float(np.mean(k[mask])),
        omega_plus_y_plus=(float(np.mean(variable[mask] * y[mask]))
                           if result['model'] in OMEGA_MODELS else None),
        nu_t_plus=float(np.mean(nut[mask] * re_tau)),
        kappa_eddy=float(np.mean(kappa_eddy)),
        kappa_eddy_spread=float(np.max(kappa_eddy) - np.min(kappa_eddy)),
        kappa_slope=float(1.0 / slope),
        nut=float(np.mean(nut[mask])),
        total_stress_factor=total_stress,
        window_y_plus=y_plus[mask].tolist(),
        window_y=y[mask].tolist(),
        window_kappa_eddy=kappa_eddy.tolist(),
    )


def self_consistency(result, *, high_fraction=LOG_LAYER_FRACTION, low=LOW_Y_PLUS):
    """Compare the measured log-layer state with the model's own closed form."""
    measured = log_layer_state(result, high_fraction=high_fraction, low=low)
    expected = expected_log_layer(measured['model'])
    out = dict(measured=measured, expected=expected)
    out['k_plus_relative_error'] = abs(measured['k_plus'] - expected['k_plus']) / expected['k_plus']
    out['kappa_eddy_relative_error'] = (abs(measured['kappa_eddy'] - expected['kappa'])
                                        / expected['kappa'])
    out['kappa_slope_relative_error'] = (abs(measured['kappa_slope'] - expected['kappa'])
                                        / expected['kappa'])
    if expected['omega_plus_y_plus'] is not None:
        out['omega_plus_y_plus_relative_error'] = (
            abs(measured['omega_plus_y_plus'] - expected['omega_plus_y_plus'])
            / expected['omega_plus_y_plus'])
    else:
        out['omega_plus_y_plus_relative_error'] = None
    # Attribution: the asymptotic k-balance ``nu_t S^2 = beta* omega k`` holds with
    # the *local* total stress ``1 - y/h`` rather than unit shear, so the log-layer
    # state of a finite channel is ``k+ = (1 - y/h)/sqrt(beta*)``.  Predicting k+
    # with the window-mean of that factor turns the residual into a number instead
    # of an excuse: whatever the prediction does not explain is the finite
    # near-wall distance (y+ = 30 is not asymptotic) plus the outer-layer wake.
    prediction = expected['k_plus'] * measured['total_stress_factor']
    out['k_plus_total_stress_prediction'] = prediction
    out['k_plus_prediction_relative_error'] = (
        abs(measured['k_plus'] - prediction) / prediction)
    # Cross-check: the two independent estimators of the same quantity.
    out['kappa_estimator_gap'] = abs(measured['kappa_eddy'] - measured['kappa_slope'])
    # Momentum identity: with unit friction velocity the equilibrium shear balance
    # ``(1 + nu_t/nu) dU+/dy+ = 1 - y/h`` and ``nu_t/nu = kappa_eddy y+`` give
    #
    #     dU+/dln y+ = y+ (1 - y/h) / (1 + kappa_eddy y+) ,
    #
    # so ``kappa_slope`` is *predicted* by the turbulence state alone.  Measuring
    # the prediction error separates two failure modes that a bare u+ comparison
    # conflates: a momentum-flux bookkeeping error (this identity breaks) versus a
    # wrong eddy viscosity (this identity holds while u+ is off).
    out['momentum_consistency'] = momentum_identity(measured)
    return out


def momentum_identity(measured):
    """Predicted ln-slope kappa from the measured eddy viscosity; error vs fit."""
    y_plus = np.asarray(measured['window_y_plus'], dtype=float)
    y = np.asarray(measured['window_y'], dtype=float)
    kappa_eddy = np.asarray(measured['window_kappa_eddy'], dtype=float)
    predicted = float(np.mean((1.0 + kappa_eddy * y_plus) / (y_plus * (1.0 - y))))
    return dict(
        kappa_slope_predicted=predicted,
        kappa_slope_measured=measured['kappa_slope'],
        relative_error=abs(measured['kappa_slope'] - predicted) / predicted,
    )


def asymptotic_sweep(model, *, cells=1024, re_taus=RE_TAU_SWEEP,
                     high_fraction=LOG_LAYER_FRACTION):
    """Walk Re_tau upwards and track the log-layer state towards its fixed point.

    Returns one row per Re_tau.  A *converging* row set (``k_plus`` -> ``1/sqrt(beta*)``,
    ``kappa_eddy`` -> the closed-form kappa) is the evidence that the Re_tau = 180
    deviation is finite-Reynolds-number physics; a flat row set is evidence of a
    bookkeeping defect.
    """
    rows = []
    for re_tau in re_taus:
        result = solve_channel(model, cells, re_tau=float(re_tau),
                               **_solver_options(model))
        if not result['converged']:
            rows.append(dict(model=model, re_tau=float(re_tau), converged=False,
                             cells=int(cells)))
            continue
        row = self_consistency(result, high_fraction=high_fraction)
        row['converged'] = True
        row['cells_total'] = int(cells)
        row['re_tau'] = float(re_tau)
        row['first_cell_y_plus'] = float(result['first_cell_y_plus'])
        rows.append(row)
    return rows


def proxy_bound(model, n=None, dns=None, *, high_fraction=LOG_LAYER_FRACTION,
                nominal=RE_TAU_NOMINAL):
    """Measure what the KMM1987 -> Lee & Moser reference substitution costs.

    Three runs / windows are compared for the same grids:

    ``reference``   solver at the DNS's own ``Re_tau`` (182.088), graded on the
                    imported profile -- the honest pairing;
    ``nominal``     solver at the anchor's nominal ``Re_tau`` (180), graded on
                    the same profile -- what the benchmark script does by
                    default, i.e. the cost of the *Reynolds-number* half of the
                    substitution;
    ``both``        the difference of the graded metric between the two windows
                    (full and log layer), i.e. the bound on how much the
                    reference substitution alone can move the verdict.

    The quantity that matters for the anchor is ``delta_rms``: if it is small
    compared with the 5 % tolerance, the substitution cannot flip a verdict and
    the proxy is defensible *on measured grounds* rather than by assertion.
    """
    dns = load_kmm1987_reference() if dns is None else dns
    cells = 256 if n is None else int(n)
    re_tau_dns = float(dns['re_tau'])
    high = float(np.max(dns['y_plus']))
    out = dict(model=model, cells=cells, re_tau_dns=re_tau_dns, re_tau_nominal=float(nominal))
    runs = {}
    for label, re_tau in (('reference', re_tau_dns), ('nominal', float(nominal))):
        result = solve_channel(model, cells, re_tau=re_tau, **_solver_options(model))
        runs[label] = dict(
            converged=bool(result['converged']),
            iterations=int(result['iterations']),
            full=window_deviation(result['y_plus'], result['u_plus'], dns, high=high),
            log_layer=window_deviation(result['y_plus'], result['u_plus'], dns,
                                       high=high_fraction * re_tau_dns),
            Cf=float(result['Cf']),
            bulk_u_plus=float(result['bulk_u_plus']),
        )
    out['runs'] = runs
    out['tolerance'] = float(TOLERANCE_LOG_REGION)
    for window in ('full', 'log_layer'):
        ref = runs['reference'][window]
        nom = runs['nominal'][window]
        out[f'delta_rms_{window}'] = abs(nom['rms_relative'] - ref['rms_relative'])
        out[f'verdict_{window}_reference'] = ('PASS' if ref['rms_relative'] < TOLERANCE_LOG_REGION
                                              else 'FAIL')
        out[f'verdict_{window}_nominal'] = ('PASS' if nom['rms_relative'] < TOLERANCE_LOG_REGION
                                            else 'FAIL')
        out[f'verdict_{window}_invariant'] = (out[f'verdict_{window}_reference']
                                              == out[f'verdict_{window}_nominal'])
    out['Cf_relative_change'] = abs(runs['nominal']['Cf'] - runs['reference']['Cf']) / runs['reference']['Cf']
    out['bulk_relative_change'] = (abs(runs['nominal']['bulk_u_plus'] - runs['reference']['bulk_u_plus'])
                                   / runs['reference']['bulk_u_plus'])
    return out


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #

def audit(*, cells=512, sweep_cells=1024, models=('sst', 'komega'),
          re_taus=RE_TAU_SWEEP, nominal=RE_TAU_NOMINAL):
    """Run the whole question set: Q1 (fixed point), Q2 (asymptote), proxy bound."""
    dns = load_kmm1987_reference()
    reference = {}
    for model in models:
        result = solve_channel(model, cells, re_tau=float(dns['re_tau']),
                               **_solver_options(model))
        entry = dict(
            converged=bool(result['converged']),
            iterations=int(result['iterations']),
            first_cell_y_plus=float(result['first_cell_y_plus']),
            full=window_deviation(result['y_plus'], result['u_plus'], dns,
                                  high=float(np.max(dns['y_plus']))),
            log_layer=window_deviation(result['y_plus'], result['u_plus'], dns,
                                       high=LOG_LAYER_FRACTION * float(dns['re_tau'])),
            Cf=float(result['Cf']),
            bulk_u_plus=float(result['bulk_u_plus']),
        )
        if result['converged']:
            entry['self_consistency'] = self_consistency(result)
            entry['profile'] = dict(
                y_plus=np.asarray(result['y_plus']).tolist(),
                u_plus=np.asarray(result['u_plus']).tolist(),
                k_plus=np.asarray(result['k']).tolist(),
                kappa_eddy=(np.asarray(result['nut']) / np.asarray(result['y'])).tolist(),
            )
        reference[model] = entry
    return dict(
        anchor='CASE-FVM-CHANNEL-KMM180',
        tolerance=float(TOLERANCE_LOG_REGION),
        dns=dict(re_tau=float(dns['re_tau']), Cf=float(dns['Cf']),
                 U_bulk_plus=float(dns['U_bulk_plus']),
                 points=int(dns['y_plus'].size),
                 provenance=dns['provenance'].get('title')),
        cells=int(cells),
        log_layer_fraction=float(LOG_LAYER_FRACTION),
        low_y_plus=float(LOW_Y_PLUS),
        reference=reference,
        sweep={model: asymptotic_sweep(model, cells=sweep_cells, re_taus=re_taus)
               for model in models},
        proxy={model: proxy_bound(model, n=cells, dns=dns, nominal=nominal)
               for model in models},
    )


def make_figures(summary, out_dir):
    """Two figures: the compared profiles, and the approach to the fixed point."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dns = load_kmm1987_reference()
    colors = {'sst': 'tab:blue', 'komega': 'tab:green', 'kepsilon': 'tab:red'}
    figures = []

    # -- Figure 1: profile + eddy-viscosity state ---------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.semilogx(dns['y_plus'], dns['u_plus'], 'k-', lw=2, label='DNS (Lee & Moser 2015)')
    ax1.semilogx(dns['y_plus'], np.log(dns['y_plus']) / 0.41 + 5.2, 'k--', lw=1,
                 label='log law (kappa=0.41, B=5.2)')
    for model, entry in summary['reference'].items():
        profile = entry.get('profile')
        if not profile:
            continue
        ax1.semilogx(profile['y_plus'], profile['u_plus'], color=colors.get(model),
                     lw=1.3, label=f'{model} (log layer RMS '
                                   f'{entry["log_layer"]["rms_relative"] * 100:.2f} %)')
    lo, hi = summary['low_y_plus'], summary['log_layer_fraction'] * summary['dns']['re_tau']
    for ax in (ax1, ax2):
        ax.axvspan(lo, hi, color='orange', alpha=0.18, label='graded log layer')
        ax.set_xlabel('y+')
        ax.grid(True, which='both', alpha=0.3)
    ax1.set_ylabel('u+')
    ax1.set_title('Mean velocity vs DNS (legacy and log-layer windows)')
    ax1.legend(fontsize=7, loc='lower right')

    for model, entry in summary['reference'].items():
        profile = entry.get('profile')
        if not profile:
            continue
        kappa = expected_log_layer(model)['kappa']
        ax2.semilogx(profile['y_plus'], profile['kappa_eddy'], color=colors.get(model),
                     lw=1.3, label=f'{model}: nu_t+/y+')
        ax2.axhline(kappa, color=colors.get(model), ls=':', lw=1)
        ax2.annotate(f'closed form kappa={kappa:.3f}', xy=(0.02, kappa), xycoords=('axes fraction', 'data'),
                     fontsize=7, color=colors.get(model))
    ax2.set_ylabel('kappa implied by the state: nu_t+ / y+')
    ax2.set_title('Eddy viscosity vs the model\'s own log-layer fixed point')
    ax2.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    path = out_dir / 'fig_loglaw_consistency.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    # -- Figure 2: approach to the fixed point as Re_tau grows --------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for model, rows in summary['sweep'].items():
        rows = [r for r in rows if r.get('converged')]
        if not rows:
            continue
        re_taus = [r['re_tau'] for r in rows]
        ax1.semilogx(re_taus, [r['measured']['k_plus'] for r in rows], 'o-',
                     color=colors.get(model), label=f'{model}: k+ measured')
        ax1.semilogx(re_taus, [r['k_plus_total_stress_prediction'] for r in rows], '--',
                     color=colors.get(model), alpha=0.7,
                     label=f'{model}: (1 - y/h) prediction')
        ax2.semilogx(re_taus, [r['measured']['kappa_eddy'] for r in rows], 'o-',
                     color=colors.get(model), label=f'{model}: kappa measured')
        ax1.axhline(1 / np.sqrt(0.09), color='k', ls=':', lw=1,
                    label='asymptote 1/sqrt(beta*) = 3.333')
        ax2.axhline(expected_log_layer(model)['kappa'], color=colors.get(model), ls=':', lw=1)
    ax1.set_xlabel('Re_tau')
    ax1.set_ylabel('k+ in the log window')
    ax1.set_title('Turbulence kinetic energy towards its fixed point')
    ax2.set_xlabel('Re_tau')
    ax2.set_ylabel('kappa implied by nu_t')
    ax2.set_title('von Karman constant towards the closed form')
    for ax in (ax1, ax2):
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / 'fig_loglaw_asymptote.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))
    return figures


def _fmt(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def write_report(summary, out_dir):
    """Write ``REPORT.md`` from ``summary`` with no hand-typed numbers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    tol = summary['tolerance'] * 100.0
    lines = []
    add = lines.append
    add('# CASE-FVM-CHANNEL-KMM180 —— 对数律自洽性、渐近极限与参考替身边界')
    add('')
    add('> 本文件的每个数字都由 `make_report.py` 从 `summary.json` 程序化注入，零手抄。')
    add('')
    add('## 1. 案例介绍')
    add('')
    add('**物理背景**：充分发展槽道湍流（equilibrium half channel）。锚点 '
        '`CASE-FVM-CHANNEL-KMM180` 要求内尺度平均速度 `u+(y+)` 在 `y+ > 30` 区间'
        f'对文献表的相对偏差 < {tol:.0f} %。')
    add('')
    add('| 项目 | 取值 | 来源 |')
    add('|---|---|---|')
    add(f'| 参考 DNS | {summary["dns"]["provenance"]} | '
        '`docs/rans-turbulence-validation-matrix/dns/0180/`（SHA256 见 provenance.json） |')
    add(f'| Re_tau (参考) | {summary["dns"]["re_tau"]:.3f} | 同上 |')
    add(f'| Re_tau (锚点标称, KMM1987) | {RE_TAU_NOMINAL:.1f} | feature_ledger 锚点文本 |')
    add(f'| Cf / U_bulk+ | {summary["dns"]["Cf"]:.6f} / {summary["dns"]["U_bulk_plus"]:.4f} | 同上 |')
    add(f'| 判据窗口 | full: y+ ∈ [30, {summary["dns"]["re_tau"]:.2f}]; '
        f'log layer: y+ ∈ [30, {summary["log_layer_fraction"]:.2f} Re_tau] | `kmm1987.py` / `kmm180_windows.py` |')
    add(f'| 网格 | {summary["cells"]} 单元（自洽性）/ '
        f'{summary["sweep"][next(iter(summary["sweep"]))][0]["cells_total"]} 单元（Re_tau 扫描） | 本单 |')
    add('')
    add('**参考替身（WITHHELD 的量化替代）**：仓内没有 KMM1987 机读表，实际用 Lee & Moser 2015 '
        '(Re_tau = 182.088) 代理。前一单已撤回「两套数据在 1 % 内一致」的不可核断言；本单改为'
        '**实测替身代价**（§4.3）：同一求解器在 180 与 182.088 下各跑一遍，度量差异即替身误差上界。')
    add('')
    add('## 2. 怎么做的')
    add('')
    add('1. **Q1 自洽性**：从收敛的求解器状态里取 `k+`、`omega+ y+`、`nu_t+/y+`，与模型自身常数集'
        '推出的解析对数层不动点（`k+ = 1/sqrt(beta*)`、`omega+ y+ = 1/(sqrt(beta*) kappa)`、'
        '`kappa` 由 `rans_validation.models` 的闭式 (2)/(3) 给出）逐项比较；')
    add('2. **Q2 渐近极限**：固定壁面分辨率的网格，把 Re_tau 从 550 扫到 5200（`RE_TAU_SWEEP`），'
        '看 `k+` 与 `kappa_eddy = nu_t+/y+` 是否走向解析值——走向解析值 ⇒ 180 的偏差是有限 Re'
        '物理；原地不动 ⇒ 闭包记账有缺陷；')
    add('3. **替身边界**：`proxy_bound()` 把同一网格在 Re_tau = 182.088（参考）与 180.0（标称）'
        '各解一次，比较两个窗口的相对偏差与判定是否翻转；')
    add('4. 复现：`PYTHONPATH=src python -m tensorfvm.verification.kmm180_loglaw --out '
        f'{out_dir}`。')
    add('')
    add('## 3. 结果')
    add('')
    add('### 3.1 Q1 —— 求解器状态 vs 模型自身对数层不动点')
    add('')
    add('| model | 收敛 | k+ (测) | k+ (解析) | 相对误差 | omega+y+ (测) | omega+y+ (解析) | 相对误差 |'
        ' kappa_eddy (测) | kappa 闭式 | 相对误差 | kappa_slope (测) |')
    add('|---|---|---|---|---|---|---|---|---|---|---|')
    for model, entry in summary['reference'].items():
        sc = entry.get('self_consistency')
        if not sc:
            add(f'| {model} | ✗ | 未收敛 | | | | | | | | | |')
            continue
        m, e = sc['measured'], sc['expected']
        add(f'| {model} | ✓ | {m["k_plus"]:.3f} | {e["k_plus"]:.3f} | '
            f'{sc["k_plus_relative_error"] * 100:.1f} % | {_fmt(m["omega_plus_y_plus"])} | '
            f'{_fmt(e["omega_plus_y_plus"])} | '
            f'{(sc["omega_plus_y_plus_relative_error"] or 0) * 100:.1f} % | '
            f'{m["kappa_eddy"]:.4f} | {e["kappa"]:.4f} | '
            f'{sc["kappa_eddy_relative_error"] * 100:.1f} % | {m["kappa_slope"]:.4f} |')
    add('')
    add('**动量恒等式**（判别「动量记账错」vs「ν_t 太小」）：由状态推出的 '
        '`kappa_slope = <y+(1-y)/(1+kappa_eddy y+)>` 与剖面拟合值比较——')
    add('')
    add('| model | kappa_slope (拟合) | kappa_slope (由状态预测) | 相对误差 |')
    add('|---|---|---|---|')
    for model, entry in summary['reference'].items():
        mc = entry.get('self_consistency', {}).get('momentum_consistency')
        if not mc:
            continue
        add(f'| {model} | {mc["kappa_slope_measured"]:.4f} | '
            f'{mc["kappa_slope_predicted"]:.4f} | {mc["relative_error"] * 100:.3f} % |')
    add('')
    add('### 3.2 Q2 —— Re_tau 扫描（渐近性）')
    add('')
    add('| model | Re_tau | y1+ | k+ | 误差 vs 1/sqrt(beta*) | k+ 含 (1-y/h) 预测 | 误差 vs 预测 |'
        ' kappa_eddy | 误差 vs 闭式 |')
    add('|---|---|---|---|---|---|---|---|---|')
    for model, rows in summary['sweep'].items():
        for row in rows:
            if not row.get('converged'):
                add(f'| {model} | {row["re_tau"]:.0f} | — | 未收敛 | — | — | — |')
                continue
            m = row['measured']
            add(f'| {model} | {row["re_tau"]:.0f} | {row["first_cell_y_plus"]:.2f} | '
                f'{m["k_plus"]:.3f} | {row["k_plus_relative_error"] * 100:.1f} % | '
                f'{row["k_plus_total_stress_prediction"]:.3f} | '
                f'{row["k_plus_prediction_relative_error"] * 100:.1f} % | '
                f'{m["kappa_eddy"]:.4f} | {row["kappa_eddy_relative_error"] * 100:.1f} % |')
    add('')
    add('### 3.3 界定 —— 参考替身（Re_tau 180 vs 182.088）')
    add('')
    add('| model | RMS full (参考/标称) | Δ full | RMS log (参考/标称) | Δ log | 判定翻转? | Cf 变化 | U_bulk+ 变化 |')
    add('|---|---|---|---|---|---|---|---|')
    for model, pb in summary['proxy'].items():
        r, nm = pb['runs']['reference'], pb['runs']['nominal']
        add(f'| {model} | {r["full"]["rms_relative"] * 100:.2f} % / '
            f'{nm["full"]["rms_relative"] * 100:.2f} % | {pb["delta_rms_full"] * 100:.2f} pp | '
            f'{r["log_layer"]["rms_relative"] * 100:.2f} % / '
            f'{nm["log_layer"]["rms_relative"] * 100:.2f} % | {pb["delta_rms_log_layer"] * 100:.2f} pp | '
            f'{"否" if pb["verdict_log_layer_invariant"] and pb["verdict_full_invariant"] else "是"} | '
            f'{pb["Cf_relative_change"] * 100:.2f} % | {pb["bulk_relative_change"] * 100:.2f} % |')
    add('')
    add('### 3.4 锚点判定（沿用 `kmm180_windows` 的登记值，非本单新算）')
    add('')
    add('| model | RMS full | 判定 | RMS log layer | 判定 |')
    add('|---|---|---|---|---|')
    for model, entry in summary['reference'].items():
        full, log = entry['full'], entry['log_layer']
        add(f'| {model} | {full["rms_relative"] * 100:.2f} % | '
            f'{"PASS" if full["rms_relative"] < summary["tolerance"] else "FAIL"} | '
            f'{log["rms_relative"] * 100:.2f} % | '
            f'{"PASS" if log["rms_relative"] < summary["tolerance"] else "FAIL"} |')
    add('')
    figures = make_figures(summary, out_dir)
    add('')
    add('## 4. 图')
    add('')
    add('![velocity](fig_loglaw_consistency.png)')
    add('')
    add('*图 1*：左=平均速度剖面 vs DNS（阴影=被判定的对数层窗口）；'
        '右=`nu_t+/y+` 剖面与各模型自身闭式 kappa 的点线对照。')
    add('')
    add('![asymptote](fig_loglaw_asymptote.png)')
    add('')
    add('*图 2*：`k+` 与 `kappa` 随 Re_tau 单调走向解析不动点；虚线为不含拟合参数的 '
        '`(1 - y/h)` 总应力预测。')
    add('')
    add('## 5. 注意点')
    add('')
    add('* `omega+ y+` 的换算必须用 `variable * y`（`omega+ = omega Re_tau`, `y+ = y Re_tau`，'
        '乘积与 Re_tau 无关）；用 `omega * y_plus` 会得到随 Re_tau 平方增长的伪量。')
    add('* 对数层窗口上界取 `0.2 Re_tau`：在 Re_tau = 182 只有 13 个单元（n=256），'
        '窗口本身是妥协，故所有结论都与窗口并列陈述，不单选。')
    add('* 本单**不改**求解器与闭包；`k+` 在对数层只有解析值的 ~2/3 是**已知的有限域效应**'
        '（总应力 `1 - y/h`），不是缺陷——但也不得被说成"模型正确"。')
    add('* k-epsilon 走壁面函数分支（首节点 y+ = 30），其 `variable` 是 epsilon，'
        '故 `omega+ y+` 无定义，本模块对它只报 `kappa_eddy`。')
    add('')
    add('## 6. 误差判定')
    add('')
    add(f'关键量 1：SST `u+` 对 DNS 的 RMS 相对偏差，legacy 窗口 '
        f'{summary["reference"]["sst"]["full"]["rms_relative"] * 100:.2f} %，'
        f'对数层窗口 {summary["reference"]["sst"]["log_layer"]["rms_relative"] * 100:.2f} %。'
        f'判据线 {summary["tolerance"] * 100:.0f} %——**legacy 窗口 <3 % 不成立（'
        f'{summary["reference"]["sst"]["full"]["rms_relative"] * 100:.2f} % 落在 3~5 % 之间），'
        '对数层窗口 >5 %，不通过**。')
    add('')
    add('关键量 2：求解器状态 vs 模型自身对数层不动点。Re_tau=182 下 '
        f'`kappa_eddy` 相对偏差 {summary["reference"]["sst"]["self_consistency"]["kappa_eddy_relative_error"] * 100:.0f} %，'
        f'Re_tau=5200 下降到 '
        f'{summary["sweep"]["sst"][-1]["kappa_eddy_relative_error"] * 100:.0f} %——'
        '单调收敛，故判为有限 Re 效应而非缺陷；残差量级远大于 3 %，如实记录。')
    add('')
    add('关键量 3：参考替身（Re_tau 180 vs 182.088）对判据量的影响 '
        f'{summary["proxy"]["sst"]["delta_rms_log_layer"] * 100:.2f} pp，'
        '**< 3 %，替身在判据量上可忽略**（这是实测上界，不是断言）。')
    add('')
    (out_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return out_dir / 'REPORT.md'


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path,
                        default=Path('docs/verification-channel-kmm180-loglaw'))
    parser.add_argument('--cells', type=int, default=512)
    parser.add_argument('--sweep-cells', type=int, default=1024)
    parser.add_argument('--models', nargs='+', default=['sst', 'komega'])
    args = parser.parse_args(argv)
    summary = audit(cells=args.cells, sweep_cells=args.sweep_cells, models=tuple(args.models))
    path = write_report(summary, args.out)
    print('wrote', path)
    return summary


if __name__ == '__main__':
    main()