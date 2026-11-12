"""Verification driver: equivalent sand-grain roughness in the channel closure.

Mission 16801 (selfstart, TensorFVM).  The capability under verification is
``rans_validation.roughness`` together with the ``k_s`` hook of
``rans_validation.channel.solve_channel``: a wall value of ``omega`` capped by
Wilcox's (1988) sand-grain condition.

Reference
---------
Nikuradse (1933) sand-roughened pipe/channel data, in law-of-the-wall form:
smooth ``u^+ = (1/kappa) ln y^+ + 5.2`` and fully rough
``u^+ = (1/kappa) ln(y^+/k_s^+) + 8.5``.  The *difference* of the two is the
quantity this driver grades,

    du^+(k_s^+) = 3.3 - (1/kappa) ln k_s^+ ,        k_s^+ = k_s u_tau / nu

because the offset ``3.3 = 8.5 - 5.2`` is the classical constant while the slope
``-1/kappa`` is not free.  At fixed ``Re_tau`` (imposed by the solve) a
log-law shift integrates to the same shift of the bulk velocity, so
``dU_b^+ = U_b^+(rough) - U_b^+(smooth)`` is a *differential* measurement that
cancels every smooth-wall modelling error of the closure -- which is exactly
what makes it a usable anchor for a RANS model whose own ``kappa`` is 0.408
(SST) or 0.434 (Wilcox), not 0.41.

Pre-declared criteria (fixed before the sweep was run; graded by ``run``)
------------------------------------------------------------------------
R1 smooth limit is exact: ``k_s^+`` below the crossover recovers the smooth
   wall value, so the profile is *identical* (max |du^+| = 0) and the result
   reports ``branch == 'smooth'``.
R2 Nikuradse shift at the converged mesh (``y_1^+ <= 0.5``): for every
   ``k_s^+`` in {30, 70, 100, 300, 1000},
   |dU_b^+ - (3.3 - ln(k_s^+)/kappa_model)| <= 0.30 wall units.
R3 roughness-function slope: least-squares ``d dU_b^+ / d ln k_s^+`` over
   ``k_s^+`` in [70, 1000] within 5 % of ``-1/kappa_model``.
R4 the shift is a *shift*: over the fitted log band, spread of
   ``du^+(y^+)`` (max - min) <= 0.30 wall units at ``k_s^+ = 1000``.
R5 intercept difference: ``B_rough - B_smooth`` measured with the model's own
   ``kappa`` (fixed-slope intercepts) within 0.30 of ``8.5 - 5.2 = 3.3``.
R6 Reynolds-number independence: at ``k_s^+ = 100`` and the same ``y_1^+``,
   ``|dU_b^+(Re_tau = 2000) - dU_b^+(Re_tau = 20000)| <= 0.30`` wall units.
R7 mesh sensitivity is *registered*, not hidden: at ``y_1^+ = 3`` the shift
   error exceeds 1.0 wall unit (the BC is calibrated for a resolved near-wall
   layer), and a mesh whose smooth wall value would mask the roughness
   (``y_1^+ = 10`` at ``k_s^+ = 100``) raises instead of returning smooth.
R8 monotonicity: ``dU_b^+`` decreases strictly with ``k_s^+``.
R9 scope is explicit: the wall-function k-epsilon model raises
   ``NotImplementedError`` for ``k_s`` instead of returning a smooth answer.
R10 SST is *not* certified: the slope of ``dU_b^+`` against ``ln k_s^+`` for
   SST-2003 deviates from ``-1/kappa_model`` by more than R3's 5 % band, i.e.
   Wilcox's ``S_R`` transfers to the SST constant set **only** in the
   transitional range.  Registered as an honest FAIL of the extrapolation, so a
   downstream user cannot read "SST + k_s" as validated.

Everything here is a verification driver; the only production-code change it
relies on is the ``k_s`` keyword of ``solve_channel`` and the cap in
:mod:`tensorfvm.rans_validation.roughness`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..rans_validation.channel import solve_channel
from ..rans_validation.roughness import (B_ROUGH, B_SMOOTH, ROUGHNESS_MODELS,
                                         crossover_k_s_plus, nikuradse_shift)
from ..rans_validation.models import implied_kappa

#: Base Reynolds number of the sweep.  Large enough that ``0.2 Re_tau`` clears
#: ``3 k_s^+`` for the whole ladder, so every band holds a real log layer.
RE_TAU = 20000.
#: Wall-normal cells of the base sweep.
CELLS = 512
#: First-cell location used for every graded case (``y_1^+ <= 0.5``, R2).
FIRST_CELL_Y_PLUS = 0.4
#: Roughness ladder, in wall units.
LADDER = (30., 70., 100., 300., 1000.)
#: Sub-ladder of the coarse-mesh control (R7).
MESH_CONTROL = (3.,)
#: ``k_s^+`` values used for the Reynolds-number control (R6).
REYNOLDS_CONTROL = (100.,)
#: Lower ``Re_tau`` of the Reynolds control: the smallest value at which the
#: graded band ``[3 k_s^+, 0.2 Re_tau]`` still holds a log layer for ``k_s^+=100``.
REYNOLDS_LOW_RE_TAU = 2000.
#: ``k_s^+`` at which the fitted shift is compared for R4 (lowest noise).
SHIFT_FIT_K_S_PLUS = 1000.
#: Nikuradse intercept difference, ``B_ROUGH - B_SMOOTH``.
B_JUMP = B_ROUGH - B_SMOOTH
#: Tolerances of R2/R4/R5 in wall units, and of R3 in relative terms.
SHIFT_TOLERANCE = 0.30
SLOPE_TOLERANCE = 0.05
MESH_CONTROL_MIN_ERROR = 1.0


def _band(k_s_plus, re_tau):
    return max(3. * k_s_plus, 50.), 0.2 * re_tau


def _shift(baseline, rough, k_s_plus):
    """``(bulk shift, mean/spread of du^+(y^+), fitted slope/intercepts)``."""
    low, high = _band(k_s_plus, rough['re_tau'])
    mask = (rough['y_plus'] >= low) & (rough['y_plus'] <= high)
    if np.count_nonzero(mask) < 3:
        # No log layer survives above the requested roughness at this Re_tau:
        # keep the bulk shift (always defined) and mark the profile items
        # not-measured instead of quietly dropping the row.
        return dict(k_s_plus=float(k_s_plus),
                    bulk_shift=float(rough['bulk_u_plus'] - baseline['bulk_u_plus']),
                    shift_mean=float('nan'), shift_spread=float('nan'), points=0,
                    band=[float(low), float(high)], kappa_model=float(implied_kappa(rough['model'])),
                    kappa_fit=float('nan'), b_rough=float('nan'), b_smooth=float('nan'),
                    intercept_jump=float('nan'), analytic=float('nan'), error=float('nan'))
    u_plus = rough['u_plus'][mask]
    smooth_at_same_y = np.interp(rough['y_plus'][mask], baseline['y_plus'], baseline['u_plus'])
    shift = u_plus - smooth_at_same_y
    kappa = float(implied_kappa(rough['model']))
    x = np.log(rough['y_plus'][mask] / k_s_plus)
    slope, _ = np.polyfit(x, u_plus, 1)
    b_rough = float(np.mean(u_plus - x / kappa))
    y_smooth = np.log(baseline['y_plus'][mask])
    b_smooth = float(np.mean(baseline['u_plus'][mask] - y_smooth / kappa))
    return dict(
        k_s_plus=float(k_s_plus),
        bulk_shift=float(rough['bulk_u_plus'] - baseline['bulk_u_plus']),
        shift_mean=float(np.mean(shift)),
        shift_spread=float(np.max(shift) - np.min(shift)),
        points=int(np.count_nonzero(mask)),
        band=list((float(low), float(high))),
        kappa_model=kappa,
        kappa_fit=float(1. / slope),
        b_rough=b_rough,
        b_smooth=b_smooth,
        intercept_jump=float(b_rough - b_smooth),
        analytic=float(nikuradse_shift(k_s_plus, kappa=kappa)),
        error=float(np.mean(shift) - float(nikuradse_shift(k_s_plus, kappa=kappa))),
    )


def _sweep(model, *, re_tau=RE_TAU, cells=CELLS, first_cell_y_plus=FIRST_CELL_Y_PLUS,
           ladder=LADDER, constants=None):
    baseline = solve_channel(model, cells, re_tau=re_tau,
                             first_cell_y_plus=first_cell_y_plus, constants=constants)
    rows = []
    for k_s_plus in ladder:
        rough = solve_channel(model, cells, re_tau=re_tau,
                              first_cell_y_plus=first_cell_y_plus,
                              constants=constants, k_s=k_s_plus / re_tau)
        row = _shift(baseline, rough, k_s_plus)
        row['branch'] = rough['roughness']['branch']
        row['converged'] = bool(rough['converged'])
        rows.append(row)
    return dict(model=model, re_tau=float(re_tau), cells=int(cells),
                first_cell_y_plus=float(baseline['first_cell_y_plus']),
                smooth_bulk_u_plus=float(baseline['bulk_u_plus']),
                kappa_model=float(implied_kappa(model, constants)),
                rows=rows)


def _slope(rows, kappa_model):
    """Fitted ``d dU_b^+ / d ln k_s^+`` versus ``-1/kappa_model``."""
    usable = [r for r in rows if r['k_s_plus'] >= 70.]
    if len(usable) < 2:
        return None
    x = np.log([r['k_s_plus'] for r in usable])
    y = np.array([r['bulk_shift'] for r in usable])
    slope = float(np.polyfit(x, y, 1)[0])
    target = -1. / kappa_model
    return dict(k_s_plus=[float(v) for v in np.exp(x)], slope=slope,
                target=target, relative=float(abs(slope - target) / abs(target)))


def _smooth_limit(model, *, re_tau=RE_TAU, cells=CELLS,
                  first_cell_y_plus=FIRST_CELL_Y_PLUS, k_s_plus=None):
    if k_s_plus is None:
        cross = crossover_k_s_plus(model, re_tau=re_tau,
                                   first_cell_y_plus=first_cell_y_plus)
        k_s_plus = 0.5 * cross
    baseline = solve_channel(model, cells, re_tau=re_tau,
                             first_cell_y_plus=first_cell_y_plus)
    rough = solve_channel(model, cells, re_tau=re_tau, first_cell_y_plus=first_cell_y_plus,
                          k_s=k_s_plus / re_tau)
    return dict(k_s_plus=float(k_s_plus),
                branch=rough['roughness']['branch'],
                max_profile_difference=float(np.max(np.abs(rough['u_plus'] - baseline['u_plus']))),
                bulk_difference=float(abs(rough['bulk_u_plus'] - baseline['bulk_u_plus'])),
                cross_value=rough['roughness']['omega_w_rough'] >= rough['roughness']['omega_w_smooth'])


def _mesh_control(model, y1_plus, *, k_s_plus=100., re_tau=RE_TAU, cells=CELLS):
    baseline = solve_channel(model, cells, re_tau=re_tau, first_cell_y_plus=y1_plus)
    try:
        rough = solve_channel(model, cells, re_tau=re_tau, first_cell_y_plus=y1_plus,
                              k_s=k_s_plus / re_tau)
    except ValueError as exc:
        return dict(requested_first_cell_y_plus=float(y1_plus), raised=True, message=str(exc))
    row = _shift(baseline, rough, k_s_plus)
    row['branch'] = rough['roughness']['branch']
    row['requested_first_cell_y_plus'] = float(y1_plus)
    row['raised'] = False
    return row


def _wall_function_scope(re_tau=RE_TAU, cells=64):
    try:
        solve_channel('kepsilon', cells, re_tau=re_tau, k_s=1e-3)
    except NotImplementedError as exc:
        return dict(raised=True, message=str(exc))
    return dict(raised=False, message='k-epsilon accepted k_s -- scope leak')


def run(*, quick=False, include_reynolds=True, **overrides):
    """Run the graded sweep and return the machine-readable summary."""
    cells = 128 if quick else CELLS
    re_tau = RE_TAU if not quick else 4000.
    ladder = LADDER if not quick else (30., 100.)
    first_cell = FIRST_CELL_Y_PLUS
    specs = dict(re_tau=re_tau, cells=cells, first_cell_y_plus=first_cell, ladder=ladder)
    specs.update(overrides)
    summary = dict(re_tau=specs['re_tau'], cells=specs['cells'],
                   first_cell_y_plus_requested=specs['first_cell_y_plus'],
                   ladder=list(specs['ladder']))
    sweeps = {}
    for model in ROUGHNESS_MODELS:
        sweeps[model] = _sweep(model, **specs)
        sweeps[model]['slope'] = _slope(sweeps[model]['rows'], sweeps[model]['kappa_model'])
        if sweeps[model]['slope'] is None:
            sweeps[model]['slope'] = dict(k_s_plus=[], slope=float('nan'),
                                          target=-1. / sweeps[model]['kappa_model'],
                                          relative=float('nan'))
        summary[model] = sweeps[model]
    summary['crossovers'] = {}
    for model in ROUGHNESS_MODELS:
        summary['crossovers'][model] = crossover_k_s_plus(
            model, re_tau=specs['re_tau'],
            first_cell_y_plus=sweeps[model]['first_cell_y_plus'])
    summary['smooth_limit'] = {model: _smooth_limit(model, re_tau=specs['re_tau'],
                                                    cells=specs['cells'],
                                                    first_cell_y_plus=specs['first_cell_y_plus'])
                               for model in ROUGHNESS_MODELS}
    summary['mesh_control'] = _mesh_control('komega', 3.0, re_tau=specs['re_tau'],
                                            cells=specs['cells'])
    summary['coarse_control'] = _mesh_control('komega', 10.0, re_tau=specs['re_tau'],
                                              cells=specs['cells'])
    summary['wall_function_scope'] = _wall_function_scope(re_tau=specs['re_tau'])
    summary['criteria'] = grade(summary)
    if include_reynolds:
        summary['reynolds'] = reynolds_control(
            'komega', k_s_plus=REYNOLDS_CONTROL[0],
            high_re_tau=max(specs['re_tau'], 2 * REYNOLDS_LOW_RE_TAU),
            low_re_tau=REYNOLDS_LOW_RE_TAU, cells=specs['cells'],
            first_cell_y_plus=specs['first_cell_y_plus'])
        finish_criteria(summary, summary['reynolds'])
    else:
        finish_criteria(summary)
    return summary


def grade(summary):
    """Evaluate R1-R10 against the pre-declared thresholds."""
    komega = summary['komega']
    sst = summary['sst']
    checks = {}

    limit = summary['smooth_limit']['komega']
    checks['R1'] = dict(
        passed=bool(limit['max_profile_difference'] == 0.0 and limit['branch'] == 'smooth'),
        detail='k_s^+=%.2f branch=%s max|du+|=%.3e'
               % (limit['k_s_plus'], limit['branch'], limit['max_profile_difference']))

    errors = {round(row['k_s_plus']): row['error'] for row in komega['rows']
              if np.isfinite(row['error'])}
    if errors:
        worst = max(abs(v) for v in errors.values())
        checks['R2'] = dict(passed=bool(worst <= SHIFT_TOLERANCE),
                            detail='max |error| = %.3f w.u. (tol %.2f); ' % (worst, SHIFT_TOLERANCE)
                                   + ', '.join('%d: %+.3f' % (k, v) for k, v in sorted(errors.items())))
    else:
        checks['R2'] = dict(passed=None, detail='not_run: no graded band in this configuration')

    slope = komega['slope']
    if np.isfinite(slope['relative']):
        checks['R3'] = dict(passed=bool(slope['relative'] <= SLOPE_TOLERANCE),
                            detail='dD/dln k_s^+ = %.4f vs %.4f (%.2f %%)'
                                   % (slope['slope'], slope['target'], 100 * slope['relative']))
    else:
        checks['R3'] = dict(passed=None, detail='not_run: the ladder holds fewer than two points above k_s^+=70')

    target_row = [r for r in komega['rows'] if r['k_s_plus'] == SHIFT_FIT_K_S_PLUS]
    if target_row:
        spread = target_row[0]['shift_spread']
        checks['R4'] = dict(passed=bool(spread <= SHIFT_TOLERANCE),
                            detail='spread of du^+(y^+) = %.3f w.u. over band' % spread)
    else:
        checks['R4'] = dict(passed=None, detail='not_run: no k_s^+=%g row in this configuration'
                            % SHIFT_FIT_K_S_PLUS)

    if target_row:
        jump = target_row[0]['intercept_jump']
        checks['R5'] = dict(passed=bool(abs(jump - B_JUMP) <= SHIFT_TOLERANCE),
                            detail='B_rough - B_smooth = %.3f vs Nikuradse %.2f' % (jump, B_JUMP))
    else:
        checks['R5'] = dict(passed=None, detail='not_run: no k_s^+=%g row in this configuration'
                            % SHIFT_FIT_K_S_PLUS)

    checks['R6'] = dict(passed=None, detail='filled by the Reynolds control run')
    checks['R7'] = dict(passed=None, detail='filled from the mesh controls')
    checks['R8'] = dict(
        passed=bool(all(a['bulk_shift'] > b['bulk_shift']
                        for a, b in zip(komega['rows'], komega['rows'][1:]))),
        detail='bulk shifts ' + ', '.join('%d: %+.3f' % (r['k_s_plus'], r['bulk_shift'])
                                          for r in komega['rows']))

    scope = summary['wall_function_scope']
    checks['R9'] = dict(passed=bool(scope['raised']), detail=scope['message'])

    sst_slope = sst['slope']
    if np.isfinite(sst_slope['relative']):
        checks['R10'] = dict(passed=bool(sst_slope['relative'] > SLOPE_TOLERANCE),
                             detail='SST slope %.4f vs %.4f (%.1f %% off) -> extrapolation not certified'
                                    % (sst_slope['slope'], sst_slope['target'],
                                       100 * sst_slope['relative']))
    else:
        checks['R10'] = dict(passed=None, detail='not_run: the ladder holds fewer than two points above k_s^+=70')
    return checks


def finish_criteria(summary, reynolds=None):
    """Fill R6/R7 once the control runs (Reynolds, mesh) have been executed."""
    checks = summary['criteria']
    if reynolds is not None:
        checks['R6']['passed'] = bool(abs(reynolds['difference']) <= SHIFT_TOLERANCE)
        checks['R6']['detail'] = ('dU_b^+(Re_tau=%g) - dU_b^+(Re_tau=%g) = %+.3f w.u. at k_s^+=%g'
                                  % (reynolds['low_re_tau'], reynolds['high_re_tau'],
                                     reynolds['difference'], reynolds['k_s_plus']))
    mesh = summary['mesh_control']
    coarse = summary['coarse_control']
    checks['R7']['passed'] = bool(abs(mesh['error']) >= MESH_CONTROL_MIN_ERROR
                                 and mesh['branch'] == 'rough' and coarse['raised'])
    checks['R7']['detail'] = ('y1+=3: shift error %+.3f w.u. (>= %.1f registered); '
                              'y1+=10: %s'
                              % (mesh['error'], MESH_CONTROL_MIN_ERROR,
                                 'raised' if coarse['raised'] else 'returned a value'))
    return checks


def reynolds_control(model='komega', *, k_s_plus=100., low_re_tau=2000., high_re_tau=RE_TAU,
                     cells=CELLS, first_cell_y_plus=FIRST_CELL_Y_PLUS):
    """Same ``k_s^+`` at two ``Re_tau``: the roughness function must not move."""
    shifts = {}
    for re_tau in (low_re_tau, high_re_tau):
        baseline = solve_channel(model, cells, re_tau=re_tau,
                                 first_cell_y_plus=first_cell_y_plus)
        rough = solve_channel(model, cells, re_tau=re_tau,
                              first_cell_y_plus=first_cell_y_plus, k_s=k_s_plus / re_tau)
        row = _shift(baseline, rough, k_s_plus)
        shifts[re_tau] = row['shift_mean'] if np.isfinite(row['shift_mean']) else row['bulk_shift']
    return dict(model=model, k_s_plus=float(k_s_plus), low_re_tau=float(low_re_tau),
                high_re_tau=float(high_re_tau),
                low_shift=float(shifts[low_re_tau]), high_shift=float(shifts[high_re_tau]),
                difference=float(shifts[low_re_tau] - shifts[high_re_tau]))


def write(summary, directory, *, filename='summary.json'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return path


def table(summary):
    """Markdown table of the graded sweep (used by the report generator)."""
    lines = ['| model | k_s^+ | band (y^+) | points | dU_b^+ | analytic | error (w.u.) | kappa_fit | B_rough-B_smooth | branch |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for model in ROUGHNESS_MODELS:
        for row in summary[model]['rows']:
            lines.append('| %s | %.0f | %.0f-%.0f | %d | %+.3f | %+.3f | %+.3f | %.4f | %.3f | %s |'
                         % (model, row['k_s_plus'], row['band'][0], row['band'][1], row['points'],
                            row['bulk_shift'], row['analytic'], row['error'], row['kappa_fit'],
                            row['intercept_jump'], row['branch']))
    return '\n'.join(lines)