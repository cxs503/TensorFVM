"""Independent (non-circular) verification of the Kovasznay 1948 reference field.

The acceptance gate ``tests/test_kovasznay_32_64_128_acceptance.py`` recomputes the
L2 errors from the committed ``n<N>-fields.npz`` dumps, but it recomputes them with
the *same* helper the producer used (``tensorfvm.native_manufactured3d.
volume_weighted_error``) and it trusts the ``exact_velocity`` / ``exact_pressure``
arrays stored in those dumps.  That leaves two holes:

1. if the exact-solution arrays were generated from a wrong (e.g. fitted or
   self-consistent) field, every downstream number would still be "reproducible";
2. the normalisation convention of the weighted norm is only checked against itself.

This script closes both holes without importing TensorFVM at all.  It

* reconstructs the Kovasznay (1948) field from the published formulae and the
  Reynolds number alone, solving ``lam^2 - Re*lam = (2*pi)^2`` independently of
  the closed-form root used anywhere else;
* proves the reconstructed field really is a steady Navier-Stokes solution by
  evaluating div(u) and the momentum residual with 4th-order central differences;
* compares the reconstruction with the ``exact_*`` arrays in every committed
  field dump (round-off agreement, pressure up to the additive gauge);
* recomputes the relative L2 errors and the log-log slopes with its own numpy
  quadrature / least squares and compares them with the published summary.

Usage
-----
    PYTHONPATH=src python scripts/verify_kovasznay_reference_independent.py \
        --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
        --grids 32,64,128 --json docs/.../independent_reference_check.json

Reference: Kovasznay, L. I. G. (1948), "Laminar flow behind a two-dimensional
grid", Proc. Cambridge Philos. Soc. 44:58-62, doi:10.1017/S0305004100023999.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REYNOLDS = 40.0
TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# reference solution, reconstructed from the paper (no project imports)
# --------------------------------------------------------------------------- #
def decay_rate(reynolds: float = REYNOLDS) -> float:
    """Stable root of ``lam^2 - Re*lam - (2*pi)^2 = 0`` (the paper's lam < 0).

    Solved by the quadratic formula written in its cancellation-free form, which
    is then checked against a bracketed Newton/bisection iteration so the value
    does not depend on one formula being remembered correctly.
    """
    stable = 0.5 * (reynolds - np.sqrt(reynolds ** 2 + 4.0 * TWO_PI ** 2))
    # independent bracketed solve of lam^2 - Re*lam - 4*pi^2 = 0 on [-5, 0]
    lo, hi = -5.0, 0.0

    def g(x):
        return x * x - reynolds * x - TWO_PI ** 2

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    bisected = 0.5 * (lo + hi)
    if abs(bisected - stable) > 1e-12:
        raise SystemExit(f'quadratic root {stable!r} != bisection root {bisected!r}')
    return float(stable)


def reference(x, y, z=None, lam=None):
    """Kovasznay 1948 steady solution (mesh units: unit square, u=(1,0) far upstream)."""
    lam = decay_rate() if lam is None else lam
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    e = np.exp(lam * x)
    u = 1.0 - e * np.cos(TWO_PI * y)
    v = (lam / TWO_PI) * e * np.sin(TWO_PI * y)
    w = np.zeros_like(u)
    p = 0.5 * (1.0 - np.exp(2.0 * lam * x))
    return u, v, w, p


def _residual(x, y, lam, re):
    """div u and the two momentum residuals, 4th-order central differences."""
    h = 2e-5

    def f(xx, yy):
        return reference(xx, yy, lam=lam)[:2]

    def d(axis):
        if axis == 0:
            fp2, fm2 = f(x + 2 * h, y), f(x - 2 * h, y)
            fp1, fm1 = f(x + h, y), f(x - h, y)
        else:
            fp2, fm2 = f(x, y + 2 * h), f(x, y - 2 * h)
            fp1, fm1 = f(x, y + h), f(x, y - h)
        out = []
        for comp in range(2):
            out.append((-fp2[comp] + 8 * fp1[comp] - 8 * fm1[comp] + fm2[comp]) / (12 * h))
        return out

    ux, vx = d(0)
    uy, vy = d(1)
    u, v = f(x, y)
    # second derivatives, 4th order (per component)
    lap = []
    for comp in range(2):
        if comp == 0:
            fp2, fm2 = f(x + 2 * h, y)[0], f(x - 2 * h, y)[0]
            fp1, fm1 = f(x + h, y)[0], f(x - h, y)[0]
            dxx = (-fp2 + 16 * fp1 - 30 * u + 16 * fm1 - fm2) / (12 * h * h)
            fp2, fm2 = f(x, y + 2 * h)[0], f(x, y - 2 * h)[0]
            fp1, fm1 = f(x, y + h)[0], f(x, y - h)[0]
            dyy = (-fp2 + 16 * fp1 - 30 * u + 16 * fm1 - fm2) / (12 * h * h)
        else:
            fp2, fm2 = f(x + 2 * h, y)[1], f(x - 2 * h, y)[1]
            fp1, fm1 = f(x + h, y)[1], f(x - h, y)[1]
            dxx = (-fp2 + 16 * fp1 - 30 * v + 16 * fm1 - fm2) / (12 * h * h)
            fp2, fm2 = f(x, y + 2 * h)[1], f(x, y - 2 * h)[1]
            fp1, fm1 = f(x, y + h)[1], f(x, y - h)[1]
            dyy = (-fp2 + 16 * fp1 - 30 * v + 16 * fm1 - fm2) / (12 * h * h)
        lap.append(dxx + dyy)
    # pressure gradient from the closed form of p = (1 - exp(2 lam x))/2
    e2 = np.exp(2.0 * lam * x)
    px = -lam * e2
    py = 0.0
    momentum_x = u * ux + v * uy + px - lap[0] / re
    momentum_y = u * vx + v * vy + py - lap[1] / re
    return ux + vy, momentum_x, momentum_y


# --------------------------------------------------------------------------- #
# independent norm / slope estimators
# --------------------------------------------------------------------------- #
def weighted_rel_l2(actual, reference_field, volumes, remove_gauge=False, norm=None):
    """sqrt(sum V (a-r)^2 / sum V r^2), with an optional volume-weighted gauge fix.

    ``norm`` overrides the denominator when the reference field is identically
    zero (the spanwise velocity budget, which is normalised by the in-plane
    reference norm instead -- see ``spanwise_share``).
    """
    a = np.asarray(actual, dtype=float)
    r = np.asarray(reference_field, dtype=float)
    v = np.asarray(volumes, dtype=float)
    if remove_gauge:
        a = a - (a * v).sum() / v.sum()
        r = r - (r * v).sum() / v.sum()
    w = v if a.ndim == 1 else v[:, None]
    if norm is None:
        denominator = (w * r ** 2).sum()
    else:
        denominator = float(norm)
    if denominator <= 0.0:
        raise ValueError('nonzero reference norm required')
    return float(np.sqrt((w * (a - r) ** 2).sum() / denominator))


def spanwise_share(velocity, exact_velocity, volumes):
    """rms(w) share of the in-plane reference norm (exact w is identically zero)."""
    v = np.asarray(volumes, dtype=float)
    denom = float(np.sum(v * np.sum(np.asarray(exact_velocity) ** 2, axis=1)))
    return float(np.sqrt(np.sum(v * np.asarray(velocity)[:, 2] ** 2) / denom))


def loglog_slope(h, err):
    """Ordinary least squares of log(error) against log(h)."""
    return float(np.polyfit(np.log(np.asarray(h, float)), np.log(np.asarray(err, float)), 1)[0])


def pairwise_order(h, err):
    return [float(np.log(err[i] / err[i + 1]) / np.log(h[i] / h[i + 1])) for i in range(len(h) - 1)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--grids', default='32,64,128')
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args(argv)

    grids = [int(token) for token in args.grids.split(',')]
    lam = decay_rate()

    # 1. the reconstructed field must solve the incompressible N-S equations
    rng = np.random.default_rng(20261114)
    xs = rng.uniform(0.05, 0.95, 8)
    ys = rng.uniform(0.05, 0.95, 8)
    div, mom_x, mom_y = _residual(xs, ys, lam, REYNOLDS)
    residual = {'lam': lam, 'quadratic_identity': float(lam ** 2 - REYNOLDS * lam - TWO_PI ** 2),
                'max_abs_divergence': float(np.max(np.abs(div))),
                'max_abs_momentum_residual': float(max(np.max(np.abs(mom_x)),
                                                       np.max(np.abs(mom_y))))}

    # 2. compare against the exact_* arrays stored in every committed dump
    levels, field_check = [], []
    for n in grids:
        npz = np.load(args.runs / f'n{n}' / f'n{n}-fields.npz')
        centres = npz['cell_centers']
        u, v, w, p = reference(centres[:, 0], centres[:, 1], lam=lam)
        exact_v = np.column_stack([u, v, w])
        dv = np.max(np.abs(npz['exact_velocity'] - exact_v))
        dp = npz['exact_pressure'] - p
        gauge = float(np.mean(dp))
        dp_centred = np.max(np.abs(dp - gauge))
        field_check.append({'n': n, 'max_abs_velocity_diff': float(dv),
                            'pressure_gauge_offset': gauge,
                            'max_abs_pressure_diff_after_gauge': float(dp_centred)})

        h = 1.0 / n
        vol = npz['cell_volumes']
        vel = npz['velocity']
        levels.append({
            'n': n, 'h': h, 'cells': int(vol.size), 'total_volume': float(vol.sum()),
            'velocity_full': weighted_rel_l2(vel, exact_v, vol),
            'velocity_inplane': weighted_rel_l2(vel[:, :2], exact_v[:, :2], vol),
            'velocity_spanwise': spanwise_share(vel, exact_v, vol),
            'pressure_fixed': weighted_rel_l2(npz['pressure'], p, vol),
            'pressure_meanfree': weighted_rel_l2(npz['pressure'], p, vol, remove_gauge=True),
        })

    h = [row['h'] for row in levels]
    independent = {}
    for key in ('velocity_full', 'velocity_inplane', 'velocity_spanwise',
                'pressure_fixed', 'pressure_meanfree'):
        err = [row[key] for row in levels]
        independent[key] = {'slope': loglog_slope(h, err), 'per_level': err,
                            'pairwise': pairwise_order(h, err)}

    # 3. Richardson / GCI on the independent errors: how much of the finest-grid
    #    error is not resolved by the triple, and does the observed order survive
    #    the pairwise consistency check?
    fm = 1.25  # Roache safety factor for a three-grid study
    richardson = {}
    for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
        err = independent[key]['per_level']
        p_hat = independent[key]['slope']
        ratio = h[-2] / h[-1]
        gain = ratio ** p_hat - 1.0
        richardson[key] = {
            'observed_order': p_hat,
            'pairwise_spread': max(independent[key]['pairwise'])
                              - min(independent[key]['pairwise']),
            'refinement_ratio': ratio,
            'order_gain_factor': gain,
            'remaining_error_estimate_finest': err[-1] / gain,
            'gci_finest': fm * err[-1] / gain,
        }

    # 4. centre-sampling bias: it is an O(h^2) truncation, so it cannot bend the
    #    slope.  Measured from the per-level budgets the producer recorded.
    sampling = None
    if (args.runs / 'summary.json').is_file():
        published_levels = json.loads((args.runs / 'summary.json').read_text())['levels']
        bias = [float(level['sampling_bias']['velocity']) for level in published_levels]
        sampling = {'per_level': bias, 'order': loglog_slope(h, bias)}

    # 5. cross-check against the published summary (no shared code path)
    published = None
    summary_path = args.runs / 'summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        published = {}
        for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
            fit = summary['fits'][key]
            published[key] = {'slope': float(fit['slope']),
                              'slope_diff': abs(float(fit['slope']) - independent[key]['slope'])}
        verdict = summary['verdict']
        published['verdict_slope_diff'] = max(
            abs(verdict['velocity_full_slope'] - independent['velocity_full']['slope']),
            abs(verdict['velocity_inplane_slope'] - independent['velocity_inplane']['slope']),
            abs(verdict['pressure_meanfree_slope'] - independent['pressure_meanfree']['slope']))

    payload = {'reference': 'Kovasznay 1948 doi:10.1017/s0305004100023999',
               'reynolds': REYNOLDS, 'runs': str(args.runs), 'grids': grids,
               'residual_check': residual, 'exact_field_check': field_check,
               'levels': levels, 'independent_estimates': independent,
               'richardson': richardson, 'sampling_bias': sampling,
               'cross_check_vs_summary': published}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + '\n')
    return payload


if __name__ == '__main__':
    main()