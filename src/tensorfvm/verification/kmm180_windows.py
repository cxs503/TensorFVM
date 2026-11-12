"""Adversarial window audit of the CASE-FVM-CHANNEL-KMM180 validation claim.

Why this module exists
----------------------
``tensorfvm.verification.kmm1987`` decides the PASS/FAIL verdict of the anchor
``CASE-FVM-CHANNEL-KMM180`` on the RMS *relative* deviation of ``u+`` from the
imported DNS profile over ``y+ >= 30 `` **up to the centreline**.  That window
does not describe a log layer at ``Re_tau ~ 180``: the log layer spans roughly
``30 <= y+ <= 0.2 Re_tau ~ 36``, i.e. an interval three cells wide on the
solver grid, while the upper part of the accepted window is the outer/wake
layer where ``u+`` reaches its maximum (~18.9).

Because the deviation of an equilibrium RANS model from DNS in a low
Reynolds-number channel is close to a *constant offset* in ``u+`` (the model
reproduces the slope but not the additive constant), the RMS *relative* error
scales like ``offset / u+``: extending the window towards the centreline
divides the metric by the larger ``u+`` values and can flip a FAIL into a
PASS without any change in the physics.  This module quantifies that
sensitivity, decomposes the residual into slope (kappa) and intercept (B)
errors, and reports per-window verdicts so the claim cannot hide behind the
window choice.

Everything here is diagnostic: no solver code is modified and the legacy
full-range metric is kept intact (it is reported as ``full`` / legacy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..rans_validation.channel import solve_channel
from .kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference

#: Fraction-of-Re_tau upper bounds used for the window sweep.
WINDOW_FRACTIONS = (0.20, 0.25, 0.30, 0.40, 0.50)
#: The window used by the validation matrix as "the log layer" (primary_fraction).
LOG_LAYER_FRACTION = 0.20
#: Lower bound of every window, fixed by the task specification of the anchor.
LOW_Y_PLUS = 30.0
#: Wall-law constant used only to express a slope error as an equivalent kappa.
KAPPA_REF = 0.41


def _window_profile(y_plus, u_plus, dns, high):
    """Model and DNS ``u+`` restricted to ``LOW_Y_PLUS <= y+ <= high``."""
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus >= LOW_Y_PLUS) & (y_plus <= high)
    if np.count_nonzero(mask) < 3:
        raise ValueError(
            'window [%.6g, %.6g] holds fewer than three model cells (found %d)'
            % (LOW_Y_PLUS, high, np.count_nonzero(mask))
        )
    dns_u = PchipInterpolator(dns['y_plus'], dns['u_plus'])(y_plus[mask])
    return y_plus[mask], u_plus[mask], dns_u


def window_deviation(y_plus, u_plus, dns, *, high):
    """RMS/max/mean *relative* deviation plus the absolute ``u+`` offset.

    ``mean_offset`` is the mean of ``u+_model - u+_DNS`` in *wall units* (not a
    percentage).  For an offset-dominated error it is window-independent while
    ``rms_relative`` shrinks as the window is widened -- which is exactly the
    dilution mechanism this audit measures.
    """
    y_w, u_w, dns_w = _window_profile(y_plus, u_plus, dns, high)
    relative = (u_w - dns_w) / dns_w
    return dict(
        low=float(LOW_Y_PLUS),
        high=float(high),
        cells=int(y_w.size),
        rms_relative=float(np.sqrt(np.mean(relative ** 2))),
        max_relative=float(np.max(np.abs(relative))),
        mean_relative=float(np.mean(relative)),
        mean_offset=float(np.mean(u_w - dns_w)),
        min_y_plus=float(np.min(y_w)),
        max_y_plus=float(np.max(y_w)),
        mean_u_plus=float(np.mean(dns_w)),
    )


def decompose_residual(y_plus, u_plus, dns, *, high):
    """Split ``u+_model - u+_DNS`` into offset and slope parts on the window.

    Two independent pieces of information are reported:

    * the **local log-slope** of each profile over the window,
      ``d u+ / d ln(y+)`` from a least-squares line through ``(ln y+, u+)``.
      Their ratio isolates a von Karman-constant error from an additive error
      (a pure offset leaves the ratio at 1);
    * the **residual regression** ``u+_model - u+_DNS = a ln(y+) + b`` whose
      mean value ``residual_mean`` is the additive error in wall units and
      whose ``offset_share`` (= ``|residual_mean| / rms(residual)``) says how
      much of the discrepancy a pure shift explains.
    """
    y_w, u_w, dns_w = _window_profile(y_plus, u_plus, dns, high)
    x = np.log(y_w)
    residual = u_w - dns_w
    rms = float(np.sqrt(np.mean(residual ** 2)))
    mean = float(np.mean(residual))
    a, b = np.polyfit(x, residual, 1)
    fitted = a * x + b
    ss_tot = float(np.sum((residual - mean) ** 2))
    ss_res = float(np.sum((residual - fitted) ** 2))
    model_slope = float(np.polyfit(x, u_w, 1)[0])
    dns_slope = float(np.polyfit(x, dns_w, 1)[0])
    return dict(
        residual_mean=mean,
        residual_rms=rms,
        offset_share=float(abs(mean) / rms) if rms > 0 else float('nan'),
        slope_a=float(a),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
        model_slope=model_slope,
        dns_slope=dns_slope,
        slope_ratio=float(model_slope / dns_slope),
        kappa_local_model=float(1.0 / model_slope),
        kappa_local_dns=float(1.0 / dns_slope),
        points=int(y_w.size),
        low=float(LOW_Y_PLUS),
        high=float(high),
    )


def log_law_reference_deviation(dns, *, high, kappa=KAPPA_REF, B=5.2):
    """How far the *DNS itself* sits from the classic log law in the window.

    Context for the 5 % target: at ``Re_tau ~ 180`` the DNS profile is not a
    perfect log law either, so a model cannot be expected to be much better
    than the log law's own local error.
    """
    y_plus = np.asarray(dns['y_plus'], dtype=float)
    u_plus = np.asarray(dns['u_plus'], dtype=float)
    mask = (y_plus >= LOW_Y_PLUS) & (y_plus <= high)
    if np.count_nonzero(mask) < 3:
        raise ValueError('window holds fewer than three DNS points')
    log_law = np.log(y_plus[mask]) / kappa + B
    relative = (u_plus[mask] - log_law) / u_plus[mask]
    return dict(
        kappa=kappa, B=B,
        cells=int(np.count_nonzero(mask)),
        rms_relative=float(np.sqrt(np.mean(relative ** 2))),
        mean_offset=float(np.mean(u_plus[mask] - log_law)),
        high=float(high),
    )


def estimate_dilution(offset, mean_u_log, mean_u_full):
    """Expected RMS-relative ratio between the full and the log-layer window.

    For a pure constant offset ``du`` the ratio of the two RMS relative errors
    is ``mean_u_full / mean_u_log`` (up to the spread of ``u+`` inside each
    window), i.e. ``> 1``: the full-range metric is *smaller*.
    """
    return dict(
        offset=float(offset),
        predicted_ratio=float(mean_u_full / mean_u_log),
        predicted_full_rms=float(abs(offset) / mean_u_full),
        predicted_log_rms=float(abs(offset) / mean_u_log),
    )


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #

MODELS = ('sst', 'komega', 'kepsilon')
GRIDS = (64, 128, 256, 512)


def _solver_options(model):
    return {'first_cell_y_plus': 30.0} if model == 'kepsilon' else {}


def audit(model, n, dns=None):
    """Solve one (model, grid) case and grade every window."""
    dns = load_kmm1987_reference() if dns is None else dns
    result = solve_channel(model, n, re_tau=dns['re_tau'], **_solver_options(model))
    re_tau = float(dns['re_tau'])
    full = window_deviation(result['y_plus'], result['u_plus'], dns,
                            high=float(np.max(dns['y_plus'])))
    windows = {'full': full}
    for fraction in WINDOW_FRACTIONS:
        windows['%.2f' % fraction] = window_deviation(
            result['y_plus'], result['u_plus'], dns, high=fraction * re_tau)
    log_layer = windows['%.2f' % LOG_LAYER_FRACTION]
    log_layer_high = LOG_LAYER_FRACTION * re_tau
    decomposition = decompose_residual(result['y_plus'], result['u_plus'], dns,
                                      high=log_layer_high)
    dns_vs_log_law = log_law_reference_deviation(dns, high=log_layer_high)
    dilution = estimate_dilution(log_layer['mean_offset'],
                                 log_layer['mean_u_plus'],
                                 full['mean_u_plus'])
    return dict(
        model=model,
        cells=int(n),
        converged=bool(result['converged']),
        iterations=int(result['iterations']),
        first_cell_y_plus=float(result['y_plus'][0]),
        Cf=float(result['Cf']),
        bulk_u_plus=float(result['bulk_u_plus']),
        windows=windows,
        log_layer=grade_window(log_layer),
        divergence_vs_full=dict(
            rms_full=full['rms_relative'],
            rms_log_layer=log_layer['rms_relative'],
            ratio=full['rms_relative'] / log_layer['rms_relative'],
        ),
        decomposition=decomposition,
        dns_vs_log_law=dns_vs_log_law,
        offset_dilution=dilution,
        verdict_full='PASS' if full['rms_relative'] < TOLERANCE_LOG_REGION else 'FAIL',
        verdict_log_layer=('PASS' if log_layer['rms_relative'] < TOLERANCE_LOG_REGION
                           else 'FAIL'),
    )


def grade_window(window):
    """Verdict helper for one window dict (kept tiny to stay testable)."""
    return dict(
        rms_relative=window['rms_relative'],
        mean_relative=window['mean_relative'],
        mean_offset=window['mean_offset'],
        cells=window['cells'],
        high=window['high'],
        mean_u_plus=window['mean_u_plus'],
        verdict=('PASS' if window['rms_relative'] < TOLERANCE_LOG_REGION else 'FAIL'),
    )


def run(models=MODELS, grids=GRIDS):
    dns = load_kmm1987_reference()
    rows = [audit(model, n, dns=dns) for model in models for n in grids]
    log_layer_high = LOG_LAYER_FRACTION * float(dns['re_tau'])
    return dict(
        anchor='CASE-FVM-CHANNEL-KMM180',
        tolerance=TOLERANCE_LOG_REGION,
        low_y_plus=LOW_Y_PLUS,
        window_fractions=list(WINDOW_FRACTIONS),
        log_layer_fraction=LOG_LAYER_FRACTION,
        reference=dict(
            re_tau=float(dns['re_tau']),
            Cf=float(dns['Cf']),
            U_bulk_plus=float(dns['U_bulk_plus']),
            points=int(len(dns['y_plus'])),
            max_y_plus=float(np.max(dns['y_plus'])),
            dns_vs_log_law=log_law_reference_deviation(dns, high=log_layer_high),
        ),
        cases=rows,
    )


def summarise(report):
    """Tabular summary used by the report generator and the tests."""
    lines = ['\t'.join(('model', 'n', 'rms_full%', 'rms_log%', 'ratio',
                        'verdict_full', 'verdict_log', 'residual_mean',
                        'slope_ratio', 'offset_share'))]
    for row in report['cases']:
        lines.append('\t'.join([
            row['model'], str(row['cells']),
            '%.3f' % (100 * row['windows']['full']['rms_relative']),
            '%.3f' % (100 * row['log_layer']['rms_relative']),
            '%.2f' % row['divergence_vs_full']['ratio'],
            row['verdict_full'], row['verdict_log_layer'],
            '%+.3f' % row['decomposition']['residual_mean'],
            '%.4f' % row['decomposition']['slope_ratio'],
            '%.2f' % row['decomposition']['offset_share'],
        ]))
    return '\n'.join(lines)


def main(out_dir=None):
    report = run()
    out_dir = Path(out_dir) if out_dir is not None else (
        Path(__file__).resolve().parents[3] / 'docs' / 'verification-channel-kmm180')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(report, indent=1) + '\n')
    print(summarise(report))
    print('\nwrote %s' % (out_dir / 'summary.json'))
    return report


if __name__ == '__main__':  # pragma: no cover
    main()