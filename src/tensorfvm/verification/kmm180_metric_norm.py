"""Norm (measure) audit of the number that grades CASE-FVM-CHANNEL-KMM180.

Why this module exists
----------------------
The anchor ``CASE-FVM-CHANNEL-KMM180`` is graded by a *single scalar*: the RMS
relative deviation of the mean velocity profile ``u+(y+)`` from the imported DNS
reference over ``y+ >= 30``.  Three earlier nodes settled how that scalar moves
with

* the **window**            -- :mod:`tensorfvm.verification.kmm180_windows`
  (``legacy_full`` ``[30, 180.6]`` vs ``log_layer`` ``[30, 0.2 Re_tau] =
  [30, 36.4]``),
* the **model / grid**      -- ``benchmark_channel_kmm1987`` matrix plus the
  grid-convergence ladder,
* the **reference choice**  -- LM2015 ``Re_tau = 182.088`` standing in for the
  KMM1987 ``Re_tau = 180`` table,

but the **measure itself** was never questioned: ``kmm1987.log_region_deviation``
computes an *unweighted* RMS over the solver cell centres,

    e_rms = sqrt( (1/N) * sum_i [ (u+_i - u+_DNS(y+_i)) / u+_DNS(y+_i) ]^2 ),   (1)

i.e. every cell in the window counts once regardless of how much wall-normal
volume it represents.  That is *not* the metric the repository's own DNS
provenance declares.  ``docs/rans-turbulence-validation-matrix/dns/0180/
provenance.json`` records

    metric_profile = sqrt( sum(V * (U_calc - U_DNS)^2) / sum(V * U_DNS^2) ),    (2)

a wall-normal **volume-weighted** norm.  On a wall-clustered mesh the two are not
the same functional, and the log-layer window at this Reynolds number is only
``[30, 36.42]`` -- 6.4 wall units -- which holds as few as **3 cells** at
``n = 64``.  A metric whose sample count, spacing and weights are all an artefact
of the mesh is a weak grading instrument; this module measures how weak.

What is computed
----------------
For every ``(model, n)`` case of the published matrix, over each of the two
windows, four definitions of "the deviation in the log region" are evaluated on
the *same* solution (so the difference is purely the measure, not the physics):

``rms_rel_unweighted``
    Equation (1).  The implemented legacy metric; reproduced here bit-for-bit so
    that the audit is demonstrably grading the same object.
``rms_rel_weighted``
    Volume-weighted RMS relative deviation, ``sqrt(sum(V e^2)/sum(V))`` --
    the numerator structure of (2) with the relative denominator kept.
``mean_rel_weighted``
    Volume-weighted *signed* mean relative deviation.  An equilibrium RANS
    model errs mostly by a near-constant ``u+`` offset, so this separates
    "wrong slope" from "wrong additive constant".
``integral_rel``
    ``| int u+ dy - int u+_DNS dy | / int u+_DNS dy`` over the window: the
    wall-normal *mass-flux* deficit, which is what an integral quantity such as
    ``C_f`` or ``U_bulk+`` actually feels.  Insensitive to cell placement.

Every number is emitted with the tolerance of the anchor (5 %) and its own
verdict, plus the implied uniform ``u+`` offset in wall units, so a reader can
see whether a FAIL is a 0.06 w.u. measurement artefact or a 0.9 w.u. model
error.  Nothing in the solver, the models or the legacy grading is modified:
this module is diagnostic and additive.

Findings are written by ``scripts/run_case_fvm_channel_kmm180_metric_norm.py``
into ``docs/verification-channel-kmm180-metric-norm/`` (``summary.json`` +
``REPORT.md`` + figure).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..rans_validation.channel import solve_channel
from .kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference
from .kmm180_windows import LOG_LAYER_FRACTION, LOW_Y_PLUS

__all__ = [
    'NORM_CHOICES',
    'TOLERANCE',
    'trapezoid_weights',
    'window_volumes',
    'cell_weights',
    'window_mask',
    'window_metric',
    'case_metrics',
    'run_matrix',
    'mesh_phase_sweep',
    'first_cell_limit',
    'PHASE_FIRST_CELL_Y_PLUS',
    'LIKE_FOR_LIKE_MODELS',
    'LIKE_FOR_LIKE_RESOLUTIONS',
]

#: The tolerance of the anchor (RMS relative deviation < 5 %).
TOLERANCE = TOLERANCE_LOG_REGION

#: Every norm implemented here.  All four are "deviation in the log region";
#: they differ only in the measure.
NORM_CHOICES = (
    'rms_rel_unweighted',
    'rms_rel_weighted',
    'mean_rel_weighted',
    'integral_rel',
)

#: Models and resolutions of the published matrix (``benchmark_channel_kmm1987``).
LIKE_FOR_LIKE_MODELS = ('sst', 'komega', 'kepsilon')
LIKE_FOR_LIKE_RESOLUTIONS = (64, 128, 256)


# --------------------------------------------------------------------------- #
# Measure
# --------------------------------------------------------------------------- #

def trapezoid_weights(y):
    """Normalised trapezoidal quadrature weights for samples ``y`` = y+.

    The weights are the wall-normal measure each cell centre represents,
    ``w_i / sum(w) = V_i / sum(V)``, obtained from the sample abscissae alone so
    that the same routine can weight a solver mesh and a DNS table without
    knowing either grid.  Interior weights are ``0.5 (y_{i+1} - y_{i-1})``;
    the two ends take the half interval they actually span.  ``sum(w) == 1``
    by construction, which is what makes the weighted RMS comparable to the
    unweighted one.

    Parameters
    ----------
    y : array-like
        Strictly increasing sample abscissae (here ``y+``).

    Returns
    -------
    ndarray of the same length, non-negative, summing to 1.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError('need a 1-D array of at least two samples')
    if not np.all(np.isfinite(y)):
        raise ValueError('sample coordinates must be finite')
    if np.any(np.diff(y) <= 0):
        raise ValueError('sample coordinates must increase strictly')

    gaps = np.diff(y)
    w = np.empty_like(y)
    w[0] = 0.5 * gaps[0]
    w[-1] = 0.5 * gaps[-1]
    w[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return w / w.sum()


def window_volumes(faces, y_plus, low=LOW_Y_PLUS, high=None):
    """Exact wall-normal measure of every cell centre inside the window.

    ``faces`` are the cell faces of the solver mesh, so the measure of cell
    ``i`` clipped to ``[low, high]`` is the overlap of ``[faces[i], faces[i+1]]``
    with the window.  This is the ``V`` of the metric declared by the DNS
    provenance, not an approximation: a cell centre near the window edge keeps
    only the part of its control volume that lies inside, and the weights are
    normalised to sum to one so the weighted and unweighted deviations are
    directly comparable.

    Returns the weights of the *masked* cells, in the order given by
    :func:`window_mask`.
    """
    faces = np.asarray(faces, dtype=float)
    y_plus = np.asarray(y_plus, dtype=float)
    if faces.ndim != 1 or faces.size != y_plus.size + 1:
        raise ValueError('need len(faces) == len(y_plus) + 1')
    if np.any(np.diff(faces) <= 0):
        raise ValueError('faces must increase strictly')

    mask = window_mask(y_plus, low, high)
    upper = float(faces[-1]) if high is None else min(float(high), float(faces[-1]))
    overlap = np.clip(np.minimum(faces[1:], upper) - np.maximum(faces[:-1], low),
                      0.0, None)
    volume = overlap[mask]
    if np.any(volume <= 0):
        raise ValueError('a cell centre inside the window has no overlapping '
                         'control volume; the window and the mesh disagree')
    return volume / volume.sum()


def cell_weights(y_plus, faces=None, *, low=LOW_Y_PLUS, high=None):
    """Volume weights of the window: exact if ``faces`` is given, else trapezoid.

    ``faces=None`` falls back to :func:`trapezoid_weights`, the only option for
    a reference table that publishes cell centres without faces; that fallback
    assigns the two window ends half a cell each, a first-order error the tests
    bound at 5 % on a clustered mesh.
    """
    if faces is not None:
        return window_volumes(faces, y_plus, low, high)
    mask = window_mask(y_plus, low, high)
    return trapezoid_weights(np.asarray(y_plus, dtype=float)[mask])


def window_mask(y_plus, low=LOW_Y_PLUS, high=None, *, minimum=3):
    """Boolean mask of the cells inside ``[low, high]``.

    ``minimum`` guards the grading: a window holding fewer than three cells
    cannot support an RMS with a resolvable spread and raises instead of
    returning a number that is really two numbers.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    if high is None:
        high = float(np.max(y_plus))
    mask = (y_plus >= low) & (y_plus <= high)
    if np.count_nonzero(mask) < minimum:
        raise ValueError(
            'window [%.6g, %.6g] holds %d cells, fewer than the %d required'
            % (low, high, np.count_nonzero(mask), minimum)
        )
    return mask


def window_metric(y_plus, u_plus, dns, *, low=LOW_Y_PLUS, high=None,
                  faces=None, tolerance=TOLERANCE):
    """Grade one profile in one window under all four norms.

    Parameters
    ----------
    y_plus, u_plus : array-like
        Model profile in wall units; ``y_plus`` must increase strictly.
    dns : dict
        Output of :func:`tensorfvm.verification.kmm1987.load_kmm1987_reference`.
    low, high : float
        Window bounds in wall units.  ``high=None`` uses the model centreline.
    faces : array-like, optional
        Cell faces of the model mesh.  When given the weights are the exact
        clipped control volumes (:func:`window_volumes`); otherwise trapezoidal
        weights over the sample abscissae are used.
    tolerance : float
        The anchor's threshold, applied to each norm independently.

    Returns
    -------
    dict
        ``cells``, ``window``, the four norm values with per-norm verdicts,
        ``implied_offset_u_plus`` (volume-weighted mean absolute error in wall
        units) and its size relative to the window-mean DNS ``u+``.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    if y_plus.shape != u_plus.shape:
        raise ValueError('y_plus and u_plus must have the same length')
    if np.any(np.diff(y_plus) <= 0):
        raise ValueError('model wall coordinate must increase strictly')

    mask = window_mask(y_plus, low, high)
    yw = y_plus[mask]
    uw = u_plus[mask]

    dns_u = PchipInterpolator(dns['y_plus'], dns['u_plus'])(yw)
    if np.any(dns_u <= 0):
        raise ValueError('DNS reference must be positive in the window')

    weights = cell_weights(y_plus, faces, low=low, high=high)
    absolute = uw - dns_u
    relative = absolute / dns_u

    # (1) legacy: every cell counts once, relative denominator
    rms_unweighted = float(np.sqrt(np.mean(relative ** 2)))
    # (2) volume weighted, same relative denominator
    rms_weighted = float(np.sqrt(np.sum(weights * relative ** 2)))
    # signed bias: is the residual an offset or a slope error?
    mean_weighted = float(np.sum(weights * relative))
    # mass-flux (integral) deficit, insensitive to cell placement
    integral_model = float(np.sum(weights * uw))
    integral_dns = float(np.sum(weights * dns_u))
    integral = float(abs(integral_model - integral_dns) / abs(integral_dns))

    values = dict(
        rms_rel_unweighted=rms_unweighted,
        rms_rel_weighted=rms_weighted,
        mean_rel_weighted=mean_weighted,
        integral_rel=integral,
    )
    verdicts = {name: ('PASS' if abs(value) < tolerance else 'FAIL')
                for name, value in values.items()}

    half_high = float(high) if high is not None else float(yw[-1])
    return dict(
        window=dict(low=float(low), high=half_high, cells=int(yw.size)),
        tolerance=float(tolerance),
        values=values,
        verdicts=verdicts,
        unanimous=len(set(verdicts.values())) == 1,
        implied_offset_u_plus=float(np.sum(weights * absolute)),
        implied_offset_relative=float(np.sum(weights * absolute) / integral_dns),
        window_mean_u_plus_dns=integral_dns,
        cell_spacing=dict(
            min=float(np.min(np.diff(yw))) if yw.size > 1 else float('nan'),
            max=float(np.max(np.diff(yw))) if yw.size > 1 else float('nan'),
            span=float(yw[-1] - yw[0]),
        ),
        profile=dict(y_plus=yw.tolist(), u_plus=uw.tolist(),
                     u_plus_dns=dns_u.tolist(), weights=weights.tolist()),
    )


# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #

def _faces_in_wall_units(solution):
    """Solver cell faces converted to wall units.

    ``solve_channel`` returns ``faces`` in half-height units (``0 <= y <= 1``)
    while ``y_plus`` is in wall units, so a weight built from the raw faces
    would be compared against the wrong abscissa.  The conversion is done once,
    here, so no caller has to remember it.
    """
    return np.asarray(solution['faces'], dtype=float) * float(solution['re_tau'])


def case_metrics(model, n, *, dns=None, re_tau=180.0, solution=None,
                 windows=('legacy_full', 'log_layer')):
    """Solve (or reuse) one ``(model, n)`` case and grade it in both windows.

    ``solution`` accepts an existing :func:`solve_channel` result so that a
    caller which already owns the matrix does not pay for a second solve; with
    ``None`` the case is solved here at the anchor's nominal ``Re_tau = 180``.
    """
    if dns is None:
        dns = load_kmm1987_reference()
    if solution is None:
        solution = solve_channel(model, n=n, re_tau=re_tau)
    if not solution['converged']:
        raise RuntimeError('channel solution for %s n=%d did not converge'
                           % (model, n))

    y_plus = np.asarray(solution['y_plus'], dtype=float)
    u_plus = np.asarray(solution['u_plus'], dtype=float)
    centreline = float(np.max(dns['y_plus']))

    bounds = {'legacy_full': centreline,
              'log_layer': LOG_LAYER_FRACTION * float(dns['re_tau'])}
    out = dict(model=model, n=int(n), re_tau=float(re_tau),
               iterations=int(solution['iterations']),
               Cf=float(solution['Cf']), bulk_u_plus=float(solution['bulk_u_plus']),
               first_cell_y_plus=float(solution['first_cell_y_plus']),
               wall_treatment=solution['wall_treatment'])
    for name in windows:
        if name not in bounds:
            raise ValueError('unknown window %r (have %s)'
                             % (name, ', '.join(sorted(bounds))))
        out[name] = window_metric(y_plus, u_plus, dns,
                                  low=LOW_Y_PLUS, high=bounds[name],
                                  faces=_faces_in_wall_units(solution))
    return out


def run_matrix(models=LIKE_FOR_LIKE_MODELS, resolutions=LIKE_FOR_LIKE_RESOLUTIONS,
               *, dns=None, re_tau=180.0, repeat=1):
    """Grade the whole ``models x resolutions`` matrix under all four norms.

    ``repeat > 1`` solves each case more than once and records the spread of the
    graded number, which separates solver reproducibility from metric
    sensitivity (the solver is deterministic here, so it should be ~0).
    """
    if dns is None:
        dns = load_kmm1987_reference()
    if repeat < 1:
        raise ValueError('repeat must be >= 1')

    cases = []
    for model in models:
        for n in resolutions:
            first = case_metrics(model, n, dns=dns, re_tau=re_tau)
            spread = {}
            if repeat > 1:
                for window in ('legacy_full', 'log_layer'):
                    key = 'rms_rel_unweighted'
                    again = [case_metrics(model, n, dns=dns, re_tau=re_tau)
                             [window]['values'][key] for _ in range(repeat - 1)]
                    first_value = first[window]['values'][key]
                    spread[window] = float(max([first_value] + again)
                                           - min([first_value] + again))
            first['repeat_spread'] = spread
            cases.append(first)

    summary = dict(
        schema='tensorfvm.case-fvm-channel-kmm180/metric-norms/1',
        re_tau_nominal=float(re_tau),
        re_tau_reference=float(dns['re_tau']),
        tolerance=TOLERANCE,
        windows=dict(
            legacy_full='y+ in [30, centreline]; literal anchor wording',
            log_layer='y+ in [30, %.6g Re_tau]; the only interval that is '
                      'logarithmic at this Re_tau' % LOG_LAYER_FRACTION,
        ),
        norms=dict(
            rms_rel_unweighted='sqrt(mean(e^2)), one vote per cell centre '
                               '(implemented legacy metric)',
            rms_rel_weighted='sqrt(sum(V e^2)/sum(V)), wall-normal volume '
                             'weighted (declared metric of the DNS provenance)',
            mean_rel_weighted='sum(V e)/sum(V), signed bias under the same measure',
            integral_rel='|int u+ dy - int u+_DNS dy| / int u+_DNS dy (mass flux)',
        ),
        cases=cases,
    )
    summary['flips'] = _norm_flips(cases)
    summary['headline'] = _headline(cases)
    return summary


def _norm_flips(cases):
    """(case, window) pairs whose verdict is *not* the same under all four norms."""
    flips = []
    for case in cases:
        for window in ('legacy_full', 'log_layer'):
            block = case[window]
            if block['unanimous']:
                continue
            flips.append(dict(
                model=case['model'], n=case['n'], window=window,
                verdicts=block['verdicts'],
                values=block['values'],
                cells=block['window']['cells'],
            ))
    return flips


def _headline(cases):
    """Per (model, window, norm): the range of the graded number over the ladder.

    No single number is quoted alone anywhere downstream: for every model and
    window the table carries the best and the worst norm value seen across the
    grid ladder, and whether even the *best* one fails the 5 % criterion.
    """
    acc = {}
    for case in cases:
        for window in ('legacy_full', 'log_layer'):
            for norm, value in case[window]['values'].items():
                key = (case['model'], window, norm)
                lo, hi = acc.get(key, (value, value))
                acc[key] = (min(lo, value), max(hi, value))

    table = []
    for (model, window, norm), (lo, hi) in sorted(acc.items()):
        table.append(dict(model=model, window=window, norm=norm,
                          best=float(lo), worst=float(hi),
                          worst_pass=bool(hi < TOLERANCE),
                          best_pass=bool(lo < TOLERANCE)))
    return table


# --------------------------------------------------------------------------- #
# Mesh-phase (clustering) sensitivity
# --------------------------------------------------------------------------- #

#: Wall-normal first-cell positions (in wall units) used by the phase sweep.
#: Every value is reachable at every resolution of the ladder:
#: ``first_cell_y_plus <= Re_tau/n`` is the geometric limit of
#: :func:`tensorfvm.rans_validation.channel.stretch_for_first_center`.
PHASE_FIRST_CELL_Y_PLUS = (0.10, 0.15, 0.20, 0.25, 0.30)


def first_cell_limit(re_tau, n):
    """Largest reachable first-cell centre in wall units, ``Re_tau/n``.

    The stretching function can place the first centre anywhere in
    ``(0, 1/n]`` of the half height, so ``y1+ <= Re_tau/n``.  A phase sweep that
    ignores this silently raises instead of returning a number.
    """
    return float(re_tau) / float(n)


def mesh_phase_sweep(model='sst', resolutions=LIKE_FOR_LIKE_RESOLUTIONS,
                     first_cell_y_plus=PHASE_FIRST_CELL_Y_PLUS, *,
                     dns=None, re_tau=180.0):
    """Move the wall clustering at fixed ``n`` and watch the graded number.

    The log-layer window ``[30, 36.42]`` holds only a handful of cells and its
    width is a fraction of the local cell spacing at coarse ``n``.  Which cells
    land inside it is therefore a *mesh choice*, not physics.  This sweep holds
    the resolution, the model and the reference fixed and varies only
    ``first_cell_y_plus``, so any movement of the graded number is metric
    noise floor -- the resolution below which a PASS cannot be distinguished
    from a FAIL.
    """
    if dns is None:
        dns = load_kmm1987_reference()
    high = LOG_LAYER_FRACTION * float(dns['re_tau'])

    rows = []
    for n in resolutions:
        limit = first_cell_limit(re_tau, n)
        for y1 in first_cell_y_plus:
            if not 0 < y1 <= limit:
                raise ValueError(
                    'first cell centre y+=%.3g is unreachable at n=%d '
                    '(limit Re_tau/n = %.4g)' % (y1, n, limit))
            solution = solve_channel(model, n=n, re_tau=re_tau,
                                     first_cell_y_plus=y1)
            block = window_metric(solution['y_plus'], solution['u_plus'], dns,
                                  low=LOW_Y_PLUS, high=high,
                                  faces=_faces_in_wall_units(solution))
            rows.append(dict(
                model=model, n=int(n), first_cell_y_plus_requested=float(y1),
                first_cell_y_plus_actual=float(solution['first_cell_y_plus']),
                cells=block['window']['cells'],
                rms_rel_unweighted=block['values']['rms_rel_unweighted'],
                rms_rel_weighted=block['values']['rms_rel_weighted'],
                integral_rel=block['values']['integral_rel'],
                implied_offset_u_plus=block['implied_offset_u_plus'],
                verdict=block['verdicts']['rms_rel_unweighted'],
            ))

    groups = {}
    for row in rows:
        groups.setdefault(row['n'], []).append(row)
    spread = {}
    for n, group in groups.items():
        values = [g['rms_rel_unweighted'] for g in group]
        spread[str(n)] = dict(
            min=float(min(values)), max=float(max(values)),
            ratio=float(max(values) / min(values)),
            verdicts=sorted({g['verdict'] for g in group}),
            verdict_unstable=len({g['verdict'] for g in group}) > 1,
            cells=sorted({g['cells'] for g in group}),
        )
    return dict(model=model, window=dict(low=LOW_Y_PLUS, high=high),
                tolerance=TOLERANCE, rows=rows, spread_by_resolution=spread)


def write_summary(summary, path):
    """Write the audit summary as JSON (stable key order, UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8')
    return path