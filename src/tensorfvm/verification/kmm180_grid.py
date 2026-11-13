"""Grid-refinement (discretisation-uncertainty) audit of the KMM180 anchor.

``CASE-FVM-CHANNEL-KMM180`` grades the RMS relative deviation of ``u+(y+)``
against the DNS on three grids (n = 64, 128, 256 cells) for three models and
publishes a PASS/FAIL per (model, grid, window).  A verdict that changes with
the grid is a property of *the grid*, not of the model -- the coarsest cell
count can pass while both refined ones fail.  This module closes that hole with
the standard Roache grid-convergence machinery, so the anchor can state whether
each of its verdicts is grid-robust or grid-decided, and where the
``h -> 0`` limit of the graded quantity actually sits.

Conventions
-----------
* ``values`` are ordered **(coarse, medium, fine)** and the grid is refined
  uniformly, ``r = n_fine/n_medium = n_medium/n_coarse`` (r = 2 here).
* Richardson, ``f(h) = f0 + a h^p``::

      d1 = f_medium - f_coarse,   d2 = f_fine - f_medium
      p  = ln(d1/d2)/ln(r)
      f0 = f_fine + d2/(r^p - 1)

* Roache GCI on the fine grid with safety factor ``Fs = 1.25``::

      GCI_abs = Fs |d2| / (r^p - 1),      GCI_rel = GCI_abs/|f_fine|

  The band quoted for the limit is ``f0 +- GCI_abs``: the width of the fine
  grid's own discretisation uncertainty, propagated to the limit.

A study has an *asymptotic range* only when the two differences keep their sign,
shrinking in magnitude (``|d2| < |d1|``), and the observed order lands inside
``ORDER_BOUNDS``.  A flat series whose differences are at the level of the
solver's own scatter (sign changes, |d2| >= |d1|) carries no order information
and must not be extrapolated; it is classified as a *plateau* instead.

Decision rule (pre-declared, applied mechanically by :func:`classify`)
---------------------------------------------------------------------
``grid_decided``
    the published verdict changes with the grid, or the Richardson limit
    together with its GCI band sits on the other side of the tolerance than
    the published verdict -> the verdict cannot be quoted without its grid.
``grid_robust``
    identical verdict on every grid *and* the limit plus band stay on that
    side of the tolerance -> refinement cannot overturn it.
``plateau_robust``
    no asymptotic range, but the whole series is flat (spread
    ``<= PLATEAU_FRACTION * tolerance``) and lies entirely on one side of the
    tolerance -> the verdict is set by the model bias, not by the grid.
``inconclusive``
    no asymptotic range and the series is not flat -> the grids are too coarse
    to decide either way.

PASS means ``value < tolerance`` everywhere in the repository (see
:func:`tensorfvm.verification.kmm180_windows.grade_window`), so the comparison
here is strict as well.
"""
from __future__ import annotations

import math

#: Uniform refinement ratio of the published grid set (64 -> 128 -> 256).
REFINEMENT = 2.0
#: Roache's recommended safety factor for a three-grid study.
SAFETY_FACTOR = 1.25
#: Plausible band for the observed order of a second-order FVM solver: if the
#: three grids are inside the asymptotic range, p lands near 2; the window is
#: deliberately loose on both ends (a coarsened mesh can show p ~ 1) but still
#: rejects the noise-driven orders (|p| < 0.2 or p > 3) that a flat series
#: produces.
ORDER_BOUNDS = (0.2, 3.0)
#: A series counts as a plateau when its full spread is at most this fraction
#: of the tolerance (5 % of 5 % = 0.25 pp here).
PLATEAU_FRACTION = 0.05
#: The two grading stations of the anchor.
STATIONS = ('legacy_full', 'log_layer')

CLASSIFICATIONS = ('grid_robust', 'grid_decided', 'plateau_robust', 'inconclusive')


def refinement_ratio(cells):
    """Uniform refinement ratio of a grid sequence, validated."""
    cells = [float(c) for c in cells]
    if len(cells) < 3:
        raise ValueError('a grid study needs at least three grids, got %d' % len(cells))
    ratios = [cells[i + 1] / cells[i] for i in range(len(cells) - 1)]
    if max(ratios) - min(ratios) > 1e-9 * max(ratios):
        raise ValueError('grid sequence is not uniformly refined: %r' % (ratios,))
    if ratios[0] <= 1:
        raise ValueError('grid sequence must be coarsening-ordered (coarse first)')
    return ratios[0]


def observed_order(values, refinement=REFINEMENT, order_bounds=ORDER_BOUNDS):
    """Observed order of accuracy from three grids, with its validity flag.

    Returns a dict with ``d1``/``d2`` (the two grid differences), ``ratio``
    (d1/d2), ``order`` (None when undefined), ``valid`` and ``reason`` (empty
    when valid).
    """
    coarse, medium, fine = (float(v) for v in values)
    d1 = medium - coarse
    d2 = fine - medium
    out = dict(d1=d1, d2=d2, ratio=None, order=None, valid=False, reason='')
    if d1 == 0.0 or d2 == 0.0:
        out['reason'] = 'no measurable refinement signal (a grid difference is zero)'
        return out
    if d1 * d2 < 0:
        out['reason'] = 'grid differences change sign: no asymptotic range'
        return out
    ratio = d1 / d2
    out['ratio'] = ratio
    if abs(d2) >= abs(d1):
        out['reason'] = 'grid differences do not shrink with refinement: no asymptotic range'
        return out
    order = math.log(ratio) / math.log(float(refinement))
    out['order'] = order
    low, high = order_bounds
    if not low <= order <= high:
        out['reason'] = 'observed order %.3f outside [%.2f, %.2f]' % (order, low, high)
        return out
    out['valid'] = True
    return out


def richardson_limit(values, order, refinement=REFINEMENT):
    """Extrapolated ``h -> 0`` value of a three-grid series."""
    fine = float(values[-1])
    medium = float(values[-2])
    if not order > 0:
        raise ValueError('Richardson extrapolation needs a positive order, got %r' % (order,))
    divisor = float(refinement) ** float(order) - 1.0
    if divisor <= 0:
        raise ValueError('refinement %r admits no extrapolation' % (refinement,))
    return fine + (fine - medium) / divisor


def gci(values, order, refinement=REFINEMENT, safety=SAFETY_FACTOR):
    """Roache grid-convergence index of the fine grid, absolute and relative."""
    fine = float(values[-1])
    medium = float(values[-2])
    if not order > 0:
        raise ValueError('GCI needs a positive order, got %r' % (order,))
    band = safety * abs(fine - medium) / (float(refinement) ** float(order) - 1.0)
    return dict(absolute=band, relative=(band / abs(fine)) if fine else None,
                safety=float(safety))


def _trend(values):
    """Direction of the graded quantity under refinement."""
    first, last = float(values[0]), float(values[-1])
    if last > first:
        return 'increasing'
    if last < first:
        return 'decreasing'
    return 'flat'


def classify(values, verdicts, tolerance, order_info, limit=None, band=None,
             plateau_fraction=PLATEAU_FRACTION):
    """Apply the pre-declared decision rule; returns ``(classification, reason)``."""
    tolerance = float(tolerance)
    verdicts = [str(v).upper() for v in verdicts]
    values = [float(v) for v in values]
    if len(set(verdicts)) > 1:
        return 'grid_decided', ('the published verdict changes with the grid: %s'
                                % '/'.join(verdicts))
    verdict = verdicts[0]
    published_pass = verdict == 'PASS'
    if order_info.get('valid') and limit is not None and band is not None:
        low, high = limit - band, limit + band
        limit_pass_high = high < tolerance          # the whole band passes
        limit_fail_low = low >= tolerance           # the whole band fails
        if published_pass and limit_pass_high:
            return 'grid_robust', ('Richardson limit %.6g +- %.6g stays below the '
                                   'tolerance %.6g' % (limit, band, tolerance))
        if (not published_pass) and limit_fail_low:
            return 'grid_robust', ('Richardson limit %.6g +- %.6g stays above the '
                                   'tolerance %.6g' % (limit, band, tolerance))
        return 'grid_decided', ('the Richardson limit %.6g +- %.6g crosses the '
                                'tolerance %.6g, so refinement can overturn the '
                                'published %s' % (limit, band, tolerance, verdict))
    spread = max(values) - min(values)
    flat = spread <= plateau_fraction * tolerance
    same_side = all((v < tolerance) == published_pass for v in values)
    if flat and same_side:
        return 'plateau_robust', ('no asymptotic range, but the series is flat '
                                  '(spread %.6g <= %.6g) and stays on one side of '
                                  'the tolerance' % (spread, plateau_fraction * tolerance))
    if flat and not same_side:
        # flat but a member disagrees: the verdict is decided by a difference
        # smaller than the plateau threshold, i.e. by the grid.
        return 'grid_decided', 'the verdict changes inside the plateau scatter'
    return 'inconclusive', ('no asymptotic range and the series is not flat '
                            '(spread %.6g > %.6g)' % (spread, plateau_fraction * tolerance))


def station_study(cells, values, verdicts, tolerance, refinement=None,
                  order_bounds=ORDER_BOUNDS, safety=SAFETY_FACTOR,
                  plateau_fraction=PLATEAU_FRACTION):
    """Full grid study of one graded quantity (one model, one station)."""
    cells = list(cells)
    values = [float(v) for v in values]
    verdicts = [str(v).upper() for v in verdicts]
    if not len(cells) == len(values) == len(verdicts):
        raise ValueError('cells/values/verdicts must have the same length')
    ratio = float(refinement) if refinement is not None else refinement_ratio(cells)
    spread = max(values) - min(values)
    info = observed_order(values, ratio, order_bounds)
    # A series whose whole spread is below the plateau threshold carries no
    # order information: any fitted slope is noise.  Refuse to extrapolate it,
    # and say so instead of quoting a meaningless order.
    if spread <= plateau_fraction * float(tolerance):
        info = dict(info, valid=False,
                    reason=('series is flat (spread %.6g <= %.6g): the fitted order '
                            '%s is noise and is not extrapolated'
                            % (spread, plateau_fraction * float(tolerance),
                               'undefined' if info['order'] is None
                               else '%.3f' % info['order'])))
    study = dict(
        cells=cells,
        values=values,
        verdicts=verdicts,
        tolerance=float(tolerance),
        refinement=ratio,
        trend=_trend(values),
        spread=spread,
        verdict_consistent=len(set(verdicts)) == 1,
        observed_order=info['order'] if info['valid'] else None,
        rejected_order=(info['order'] if (not info['valid'] and info['order'] is not None) else None),
        order_valid=info['valid'],
        order_reason=info['reason'],
        grid_differences=[info['d1'], info['d2']],
        limit=None,
        gci=None,
        limit_band=None,
    )
    if info['valid']:
        limit = richardson_limit(values, info['order'], ratio)
        band = gci(values, info['order'], ratio, safety)
        study['limit'] = limit
        study['gci'] = band
        study['limit_band'] = [limit - band['absolute'], limit + band['absolute']]
    classification, reason = classify(values, verdicts, tolerance, info,
                                      study['limit'], band['absolute'] if study['gci'] else None,
                                      plateau_fraction)
    study['classification'] = classification
    study['reason'] = reason
    return study


def grid_studies(benchmark, stations=STATIONS):
    """Per-model, per-station grid studies derived from a benchmark record.

    ``benchmark`` is the parsed ``docs/case-fvm-channel-kmm180/benchmark.json``;
    nothing is retyped, every number comes from its ``cases`` list.
    """
    tolerance = float(benchmark['tolerance'])
    cases = list(benchmark['cases'])
    if not cases:
        raise ValueError('benchmark record holds no cases')
    studies = {}
    for model in sorted({case['model'] for case in cases}):
        rows = sorted((c for c in cases if c['model'] == model), key=lambda c: c['n'])
        for station in stations:
            try:
                graded = [row[station] for row in rows]
            except KeyError:
                continue
            studies['%s/%s' % (model, station)] = station_study(
                [row['n'] for row in rows],
                [window['rms_relative'] for window in graded],
                [window['verdict'] for window in graded],
                tolerance,
            )
    if not studies:
        raise ValueError('benchmark record holds no graded station')
    return studies


def grid_decided(studies):
    """Keys of the studies whose verdict is decided by the grid."""
    return [key for key, study in sorted(studies.items())
            if study['classification'] == 'grid_decided']


def not_robust(studies):
    """Keys of the studies whose verdict is *not* grid-robust.

    ``inconclusive`` belongs here -- the grid is too coarse to decide either
    way, which is not a robust verdict -- but it is *not* reported as
    ``grid_decided``: the difference between "the grid flips the verdict" and
    "we cannot tell" must survive into the headline.
    """
    return [key for key, study in sorted(studies.items())
            if study['classification'] in ('grid_decided', 'inconclusive')]


def headline(studies):
    """One-line summary of the audit, generated rather than written."""
    counts = {name: 0 for name in CLASSIFICATIONS}
    for study in studies.values():
        counts[study['classification']] += 1
    decided = grid_decided(studies)
    undecided = [key for key in not_robust(studies) if key not in decided]
    parts = ['%d of %d graded verdicts are grid-decided' % (len(decided), len(studies))]
    if decided:
        parts.append(' (' + ', '.join(decided) + ')')
    if undecided:
        parts.append(', %d inconclusive (%s)' % (len(undecided), ', '.join(undecided)))
    parts.append('; classification counts: ' + ', '.join(
        '%s=%d' % (name, counts[name]) for name in CLASSIFICATIONS))
    for key in decided:
        study = studies[key]
        if study['limit'] is not None:
            parts.append('; %s: published verdicts %s, Richardson limit %.4f '
                         '+- %.4f vs tolerance %.4f'
                         % (key, '/'.join(study['verdicts']), study['limit'],
                            study['gci']['absolute'], study['tolerance']))
    return ''.join(parts) + '.'