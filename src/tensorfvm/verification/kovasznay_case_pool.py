"""Machine-adjudicable case-pool entry for ``CASE-FVM-KOVASZNAY-ORDER``.

Why this module exists
----------------------
The ledger row is

    CASE-FVM-KOVASZNAY-ORDER | 算例池·定常 Kovasznay 流收敛阶验证
    ref: Kovasznay 1948 Proc. Camb. Phil. Soc. 44:58
    doi:10.1017/s0305004100023999
    容差: L2 收敛阶 >= 1.8 (名义二阶) | 指标: log-log 斜率 on 32/64/128 三套网格

The word 算例池 (case pool) asks for a *record a machine can adjudicate*, not for
another prose report.  Before this module the anchor existed as nine documents,
a set of committed ``runs-32-64-128*`` artifact trees and a gate test that
re-read them; nothing in one place answered, without a human reading a table:

* which quantity is graded, on which grids, with which comparator and threshold;
* whether the graded reference field really is the Kovasznay (1948) closed form
  the ledger names, or a look-alike field that merely satisfies the wall data;
* which discrete configuration the pool admits, and whether every level of the
  ladder was solved with *that* configuration;
* whether a PASS survives the obvious adversarial moves (pre-asymptotic ladder,
  a genuinely first-order scheme, a mixed sequence, a mislabelled closure);
* whether the entry can be rejected at all.

This module answers those five questions and nothing else.  It adds **no new
physics, no new solver run and no new tolerance**: every number is recomputed
from already-committed artifacts, and the order threshold is the ledger's
``1.8`` (``ORDER_MIN``), cross-checked by the tests against the value the
summariser and the acceptance gate already use.

Inputs (all committed, read-only)
---------------------------------
``docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n{32,64,128}/``
    The admitted ladder: ``cases.json`` (recorded errors + configuration) and
    ``n<N>-fields.npz`` (cell centres, volumes, velocity, pressure, exact
    velocity, exact pressure).  Every graded error is **recomputed** from the
    npz; a disagreement with ``cases.json`` beyond ``RECOMPUTE_RTOL`` raises, so
    a hand-edited record cannot be admitted.
``docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-firstorder/``
    Second-order → first-order convection control (``convection_order = 1``).
``docs/CASE-FVM-KOVASZNAY-ORDER/selftest-4-8-16/``
    Pre-asymptotic control: the same format on grids far outside the asymptotic
    range.  Both controls must **fail** the order criterion; that is what makes
    the admission verdict falsifiable.

Graded quantities
-----------------
``velocity_full``
    Volume-weighted relative L2 of all three velocity components against the
    exact field.  This is the primary graded quantity: the admitted closure
    (``du/dz = dv/dz = 0`` on the extrusion planes) drives the spurious spanwise
    mode to round-off, so no in-plane decomposition defence is needed.
``pressure_mean_free``
    Diagnostic.  The momentum equations fix only the pressure gradient, so the
    gauge-consistent (mean-free) pressure is graded and reported; the fixed-gauge
    error grows like ``h^0`` by construction and is published as such.
``velocity_spanwise``
    Diagnostic bounded by ``SPANWISE_ROUNDOFF``: it is the measured evidence for
    the declared closure, not an accuracy measure.

Reference: Kovasznay, L. I. G. (1948), "Laminar flow behind a two-dimensional
grid", Proc. Cambridge Philos. Soc. 44:58-62,
doi:10.1017/s0305004100023999.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tensorfvm.native_manufactured3d import volume_weighted_error

#: Ledger threshold for the graded quantity (nominal second order).
ORDER_MIN = 1.8
#: A three-level log-log fit has no residual degrees of freedom, so the two
#: consecutive pair orders must also clear the threshold and agree to within
#: this band; a wider spread means the coarse pair is pre-asymptotic and the
#: fitted slope is an average of two regimes, not the order of the scheme.
PAIR_ORDER_BAND = 0.20
#: Recorded errors must be reproducible from the dumped fields to this rtol.
RECOMPUTE_RTOL = 1e-10
#: ``zero_gradient`` keeps the spurious spanwise mode at round-off; ``free``
#: leaves an O(h) mode (13 orders of magnitude larger on the same grids).  The
#: threshold sits inside that gap, so the closure is *measured*, not trusted.
SPANWISE_ROUNDOFF = 1e-12
#: Grids of the adjudicated ladder (ledger contract).
GRIDS = (32, 64, 128)
REFERENCE_REF = 'Kovasznay 1948, Proc. Camb. Phil. Soc. 44:58-62'
REFERENCE_DOI = '10.1017/s0305004100023999'
#: Settings that must be identical on every level for a slope to be the order of
#: a *scheme* rather than an average of different schemes.
CONFIG_KEYS = ('reynolds', 'symmetry_closure', 'convection_order', 'algorithm')
#: The configuration the pool admits, and the only one it admits.
ADMITTED_CONFIG = {'reynolds': 40.0, 'symmetry_closure': 'zero_gradient',
                   'convection_order': 2, 'algorithm': 'coupled'}
#: Names the artifacts use for the quantities that are also recorded in
#: ``cases.json``.  The in-plane and spanwise budgets are recomputed-only: the
#: record never carried them, so they cannot be drift-checked, only published.
RECORDED_KEYS = {'velocity_full': 'velocity_l2',
                 'pressure_fixed': 'pressure_l2',
                 'pressure_mean_free': 'pressure_mean_free_l2'}
#: Adversarial controls: ``(directory, grids, expected failure)``.
CONTROLS = (('runs-32-64-128-firstorder', (32, 64, 128), 'first_order_convection'),
            ('selftest-4-8-16', (4, 8, 16), 'pre_asymptotic'))


class CasePoolError(RuntimeError):
    """Raised when a committed artifact cannot support the claim it makes."""


def case_root(root=None):
    """Locate ``docs/CASE-FVM-KOVASZNAY-ORDER`` in a source checkout."""
    if root is not None:
        return Path(root)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
        if candidate.is_dir():
            return candidate
    return Path.cwd() / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'


# ----------------------------------------------------------------------------
# Reference field: Kovasznay (1948), re-derived here so the audit does not go
# through the code path that produced the committed ``exact_*`` arrays.
# ----------------------------------------------------------------------------
def lam_1948(reynolds):
    """Decay exponent of the 1948 solution, from the quadratic root form.

    The closed form is built on the identity ``lam^2 - Re*lam = 4*pi^2``, so
    ``lam = (Re - sqrt(Re^2 + 16*pi^2)) / 2``.  The solver-side module writes
    the same constant as ``-2*k^2/(Re + sqrt(Re^2 + 4*k^2))`` with ``k = 2*pi``;
    both algebraic forms agree to round-off (checked in :func:`lambda_identity`).
    """
    reynolds = float(reynolds)
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError('positive finite Reynolds number required')
    return 0.5 * (reynolds - np.sqrt(reynolds ** 2 + 16.0 * np.pi ** 2))


def exact_fields(points, reynolds=40.):
    """Kovasznay (1948) velocity and pressure at ``points`` (``[x, y, z]``)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be (N, 3)')
    lam = lam_1948(reynolds)
    k = 2.0 * np.pi
    x, y, _ = points.T
    e = np.exp(lam * x)
    c, s = np.cos(k * y), np.sin(k * y)
    velocity = np.column_stack((1.0 - e * c, lam / k * e * s, np.zeros_like(x)))
    pressure = 0.5 * (1.0 - e * e)
    return velocity, pressure


def lambda_identity(reynolds=40.):
    """Relative residual of the two algebraic forms of the 1948 exponent identity."""
    reynolds = float(reynolds)
    k = 2.0 * np.pi
    lam = lam_1948(reynolds)
    other = -2.0 * k * k / (reynolds + np.sqrt(reynolds ** 2 + 4.0 * k * k))
    identity = (lam ** 2 - reynolds * lam) / (k * k) - 1.0
    return {'lambda': lam, 'lambda_rationalised': other,
            'form_gap': abs(lam - other) / abs(lam),
            'identity_residual': abs(identity)}


def momentum_residual(points, reynolds=40.):
    """Steady, unforced Navier--Stokes residual of the closed form (analytic).

    The point of this check is that the reference field needs **no body force**:
    a look-alike field fitted to the wall data would not satisfy the momentum
    equations identically.
    """
    points = np.asarray(points, dtype=float)
    lam = lam_1948(reynolds)
    k = 2.0 * np.pi
    nu = 1.0 / float(reynolds)
    x, y, _ = points.T
    e, c, s = np.exp(lam * x), np.cos(k * y), np.sin(k * y)
    u, v = 1.0 - e * c, lam / k * e * s
    p_x = -lam * e * e
    u_x, u_y = -lam * e * c, k * e * s
    u_xx, u_yy = -lam ** 2 * e * c, k * k * e * c
    v_x, v_y = lam ** 2 / k * e * s, lam * e * c
    v_xx, v_yy = lam ** 3 / k * e * s, -lam * k * e * s
    res_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    res_v = u * v_x + v * v_y - nu * (v_xx + v_yy)
    return np.column_stack((res_u, res_v, np.zeros_like(res_u)))


# ----------------------------------------------------------------------------
# Artifact loading (recomputation is the only source of graded numbers)
# ----------------------------------------------------------------------------
def _first_row(path):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        if len(payload) != 1:
            raise CasePoolError(f'{path}: expected exactly one case row, got {len(payload)}')
        return payload[0]
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_level(directory, n):
    """One ladder level: recorded row + errors recomputed from the field dump."""
    directory = Path(directory)
    level_dir = directory / f'n{n}'
    row = _first_row(level_dir / 'cases.json')
    if int(row['n']) != int(n):
        raise CasePoolError(f'{level_dir}: recorded n={row["n"]} but directory says n={n}')
    fields = np.load(level_dir / f'n{n}-fields.npz')
    volumes = fields['cell_volumes']
    velocity, exact_velocity = fields['velocity'], fields['exact_velocity']
    pressure, exact_pressure = fields['pressure'], fields['exact_pressure']
    recomputed = {
        'velocity_full': volume_weighted_error(velocity, exact_velocity, volumes),
        'velocity_inplane': volume_weighted_error(velocity[:, :2], exact_velocity[:, :2], volumes),
        'velocity_spanwise': float(np.sqrt(np.sum(volumes * velocity[:, 2] ** 2)
                                           / np.sum(volumes * np.sum(exact_velocity ** 2, axis=1)))),
        'pressure_fixed': volume_weighted_error(pressure, exact_pressure, volumes),
        'pressure_mean_free': volume_weighted_error(pressure, exact_pressure, volumes,
                                                   remove_mean=True),
    }
    recorded = {}
    for name, value in recomputed.items():
        key = RECORDED_KEYS.get(name)
        if key is None:
            continue
        if key not in row:
            raise CasePoolError(f'{level_dir}/cases.json: missing recorded {key!r}')
        recorded[key] = float(row[key])
        if not np.isclose(recorded[key], value, rtol=RECOMPUTE_RTOL, atol=0.0):
            raise CasePoolError(f'{level_dir}: recorded {key}={recorded[key]!r} is not reproducible '
                                f'from the dump (recomputed {value!r})')
    return {
        'n': int(n),
        'h': float(row['h']),
        'cells': int(row['cells']),
        'errors': recomputed,
        'recorded': recorded,
        'max_abs_z_velocity': abs(float(row['max_abs_z_velocity'])),
        'config': {key: row.get(key) for key in CONFIG_KEYS},
        'discrete_conservation': {
            'energy_balance_error': float(row['energy_balance_error']),
            'global_momentum_imbalance': float(row['global_momentum_imbalance']),
            'interface_traction_defect_max': float(row['interface_traction_defect_max']),
            'divergence_max': float(row['divergence_max']),
            'total_volume_error': abs(float(row['total_volume']) - 1.0),
        },
        'numerically_converged': bool(row.get('numerically_converged', False)),
        'outer_iterations': int(row.get('outer_iterations', 0)),
        'paths': {'cases': str(level_dir / 'cases.json'),
                  'fields': str(level_dir / f'n{n}-fields.npz')},
    }


def load_sequence(directory, grids=GRIDS):
    """All levels of a ladder, coarsest first."""
    return [load_level(directory, n) for n in grids]


def lsq_slope(levels, key='velocity_full'):
    """Log-log least-squares slope of ``errors[key]`` against ``h``."""
    h = np.array([level['h'] for level in levels], dtype=float)
    e = np.array([level['errors'][key] for level in levels], dtype=float)
    if np.any(e <= 0) or np.any(h <= 0):
        raise CasePoolError(f'{key}: non-positive error or spacing cannot be fitted on log-log axes')
    log_h, log_e = np.log(h), np.log(e)
    slope, intercept = np.polyfit(log_h, log_e, 1)
    residual = log_e - (slope * log_h + intercept)
    total = log_e - log_e.mean()
    r_squared = 1.0 - float(np.sum(residual ** 2) / np.sum(total ** 2)) if np.sum(total ** 2) else 1.0
    return {'slope': float(slope), 'intercept': float(intercept), 'r_squared': r_squared}


def pair_orders(levels, key='velocity_full'):
    """Order of each consecutive pair, fine -> coarse labelling as observed."""
    orders = []
    for fine, coarse in zip(levels, levels[1:]):
        orders.append(float(np.log(fine['errors'][key] / coarse['errors'][key])
                            / np.log(fine['h'] / coarse['h'])))
    return orders


# ----------------------------------------------------------------------------
# Criteria
# ----------------------------------------------------------------------------
CRITERIA = (
    {'id': 'order_velocity_full', 'quantity': 'velocity_full', 'primary': True,
     'comparator': '>=', 'threshold': ORDER_MIN, 'unit': 'log-log LSQ slope',
     'source': 'ledger acceptance: L2 收敛阶 >= 1.8 on 32/64/128'},
    {'id': 'pair_order_consistency', 'quantity': 'velocity_full', 'primary': True,
     'comparator': 'both pairs >= threshold; spread <= band',
     'threshold': PAIR_ORDER_BAND, 'unit': 'order',
     'source': 'a three-level slope has no residual dof; independent pairs must agree'},
    {'id': 'reference_is_kovasznay1948', 'quantity': 'exact fields vs closed form',
     'primary': True, 'comparator': '<=', 'threshold': 1e-12, 'unit': 'max abs',
     'source': REFERENCE_REF + ' doi:' + REFERENCE_DOI},
    {'id': 'configuration_homogeneous', 'quantity': 'reynolds/closure/order/algorithm',
     'primary': True, 'comparator': 'identical on all levels and == admitted',
     'threshold': None, 'unit': 'config', 'source': 'a slope is only the order of one scheme'},
    {'id': 'closure_label_self_consistent', 'quantity': 'max_abs_z_velocity',
     'primary': True, 'comparator': '<=', 'threshold': SPANWISE_ROUNDOFF, 'unit': 'abs',
     'source': 'zero_gradient closure keeps the spurious spanwise mode at round-off'},
    {'id': 'first_order_control_fails', 'quantity': 'velocity_full',
     'primary': True, 'comparator': '< threshold', 'threshold': ORDER_MIN,
     'unit': 'log-log LSQ slope',
     'source': 'convection_order=1 ladder must not appear to be second order'},
    {'id': 'pre_asymptotic_control_fails', 'quantity': 'velocity_full',
     'primary': True, 'comparator': '< threshold', 'threshold': ORDER_MIN,
     'unit': 'log-log LSQ slope',
     'source': '4/8/16 ladder must fail: the gate has to be able to say no'},
    {'id': 'pressure_mean_free_order', 'quantity': 'pressure_mean_free', 'primary': False,
     'comparator': '>=', 'threshold': ORDER_MIN, 'unit': 'log-log LSQ slope',
     'source': 'diagnostic: pressure is graded in its gauge-consistent norm'},
    {'id': 'pressure_fixed_gauge', 'quantity': 'pressure_fixed', 'primary': False,
     'comparator': 'published', 'threshold': None, 'unit': 'log-log LSQ slope',
     'source': 'not an accuracy measure: the fixed gauge is O(1) by construction'},
    {'id': 'spanwise_spurious_share', 'quantity': 'velocity_spanwise', 'primary': False,
     'comparator': '<=', 'threshold': SPANWISE_ROUNDOFF, 'unit': 'fraction',
     'source': 'diagnostic: measured evidence for the declared closure'},
    {'id': 'discrete_conservation', 'quantity': 'energy/momentum/divergence defects',
     'primary': False, 'comparator': '<=', 'threshold': 1e-8, 'unit': 'relative',
     'source': 'diagnostic: the discrete solution is converged and conservative'},
)


def _row(criterion_id, value, passed, note=''):
    for criterion in CRITERIA:
        if criterion['id'] == criterion_id:
            row = dict(criterion)
            row.update({'value': value,
                        'passed': None if passed is None else bool(passed),
                        'note': note})
            return row
    raise KeyError(criterion_id)


def validate_sequence(levels, admitted=None):
    """Configuration + closure-label consistency across the ladder."""
    admitted = dict(ADMITTED_CONFIG if admitted is None else admitted)
    violations = []
    for key in CONFIG_KEYS:
        values = [level['config'][key] for level in levels]
        if any(value is None for value in values):
            violations.append(f'{key}: not recorded on every level')
            continue
        if len({json.dumps(value) for value in values}) != 1:
            violations.append(f'{key}: mixed sequence {values}')
            continue
        if admitted.get(key) is not None and json.dumps(values[0]) != json.dumps(admitted[key]):
            violations.append(f'{key}: sequence uses {values[0]!r}, admitted configuration is '
                              f'{admitted[key]!r}')
    for level in levels:
        closure = level['config']['symmetry_closure']
        if closure is None:
            violations.append(f"n={level['n']}: closure not recorded")
        elif closure == 'zero_gradient' and level['max_abs_z_velocity'] > SPANWISE_ROUNDOFF:
            violations.append(f"n={level['n']}: labelled zero_gradient but max|w|="
                              f"{level['max_abs_z_velocity']:.3e} > {SPANWISE_ROUNDOFF:g}")
        elif closure == 'free' and level['max_abs_z_velocity'] <= SPANWISE_ROUNDOFF:
            violations.append(f"n={level['n']}: labelled free but max|w|="
                              f"{level['max_abs_z_velocity']:.3e} is at round-off")
        elif closure not in ('zero_gradient', 'free'):
            violations.append(f"n={level['n']}: unknown closure {closure!r}")
    for level in levels:
        if not level['numerically_converged']:
            violations.append(f"n={level['n']}: solver did not converge")
    if violations:
        raise CasePoolError('ladder refused: ' + '; '.join(violations))
    return {'configuration': dict(levels[0]['config']), 'admitted': admitted}


def reference_audit(level):
    """Independent re-derivation of the graded reference field for one level."""
    fields = np.load(level['paths']['fields'])
    points = fields['cell_centers']
    reynolds = float(level['config']['reynolds'])
    velocity, pressure = exact_fields(points, reynolds)
    dumped_velocity, dumped_pressure = fields['exact_velocity'], fields['exact_pressure']
    identity = lambda_identity(reynolds)
    residual = momentum_residual(points, reynolds)
    span = np.ptp(points, axis=0)
    # Scale the momentum residual by the convection magnitude so the check is
    # dimensionless and does not depend on the grid.
    scale = float(np.max(np.abs(dumped_velocity[:, :2]))) ** 2 / np.max(span[:2])
    return {
        'reynolds': reynolds,
        'lambda': identity['lambda'],
        'lambda_form_gap': identity['form_gap'],
        'lambda_identity_residual': identity['identity_residual'],
        'velocity_max_abs_gap': float(np.max(np.abs(velocity - dumped_velocity))),
        'pressure_max_abs_gap': float(np.max(np.abs(pressure - dumped_pressure))),
        'momentum_residual_max_scaled': float(np.max(np.abs(residual)) / scale),
        'points_checked': int(len(points)),
        'source_doi': REFERENCE_DOI,
    }


def control_ladder(directory, grids):
    """Order of an adversarial control ladder (must fail the criterion)."""
    levels = load_sequence(directory, grids)
    return {'levels': [level['n'] for level in levels],
            'velocity_full_slope': lsq_slope(levels)['slope'],
            'pair_orders': pair_orders(levels),
            'configuration': dict(levels[0]['config']),
            'errors': {level['n']: level['errors']['velocity_full'] for level in levels}}


def evaluate(root=None):
    """Adjudicate the pooled ladder and return the full admission payload."""
    base = case_root(root)
    admitted_dir = base / 'runs-32-64-128-zero_gradient'
    levels = load_sequence(admitted_dir, GRIDS)
    sequence = validate_sequence(levels)
    reference = reference_audit(levels[-1])

    slope = lsq_slope(levels)
    orders = pair_orders(levels)
    spread = float(max(orders) - min(orders))
    pressure_slope = lsq_slope(levels, 'pressure_mean_free')
    pressure_fixed_slope = lsq_slope(levels, 'pressure_fixed')
    spanwise = max(level['errors']['velocity_spanwise'] for level in levels)
    conservation = max(max(level['discrete_conservation'].values()) for level in levels)

    controls = {}
    for name, grids, label in CONTROLS:
        control = control_ladder(base / name, grids)
        control['label'] = label
        control['expected'] = f'velocity_full_slope < {ORDER_MIN}'
        control['passed'] = control['velocity_full_slope'] < ORDER_MIN
        controls[name] = control

    reference_gap = max(reference['velocity_max_abs_gap'], reference['pressure_max_abs_gap'])
    reference_ok = (reference_gap <= 1e-12
                    and reference['lambda_form_gap'] <= 1e-12
                    and reference['lambda_identity_residual'] <= 1e-12
                    and reference['momentum_residual_max_scaled'] <= 1e-12)

    rows = [
        _row('order_velocity_full', slope['slope'], slope['slope'] >= ORDER_MIN,
             f"R^2={slope['r_squared']:.6f} on {list(GRIDS)}"),
        _row('pair_order_consistency',
             {'pair_orders': orders, 'spread': spread},
             all(order >= ORDER_MIN for order in orders) and spread <= PAIR_ORDER_BAND),
        _row('reference_is_kovasznay1948', reference_gap, reference_ok,
             f"lambda identity residual {reference['lambda_identity_residual']:.2e}, "
             f"scaled momentum residual {reference['momentum_residual_max_scaled']:.2e}"),
        _row('configuration_homogeneous', sequence['configuration'], True,
             'all levels identical and equal to the admitted configuration'),
        _row('closure_label_self_consistent', spanwise, spanwise <= SPANWISE_ROUNDOFF,
             f"max|w| over the ladder; free closure would be O(h)"),
        _row('first_order_control_fails', controls['runs-32-64-128-firstorder']['velocity_full_slope'],
             controls['runs-32-64-128-firstorder']['passed'],
             'same format, convection_order=1'),
        _row('pre_asymptotic_control_fails', controls['selftest-4-8-16']['velocity_full_slope'],
             controls['selftest-4-8-16']['passed'],
             'same format on 4/8/16, outside the asymptotic range'),
        _row('pressure_mean_free_order', pressure_slope['slope'],
             pressure_slope['slope'] >= ORDER_MIN, f"R^2={pressure_slope['r_squared']:.6f}"),
        _row('pressure_fixed_gauge', pressure_fixed_slope['slope'], None,
             'published for completeness; fixed-gauge error is O(1) by construction'),
        _row('spanwise_spurious_share', spanwise, spanwise <= SPANWISE_ROUNDOFF),
        _row('discrete_conservation', conservation, conservation <= 1e-8,
             'worst of energy balance / momentum imbalance / traction defect / divergence / volume'),
    ]
    primary = [row for row in rows if row['primary']]
    failures = [row['id'] for row in primary if not row['passed']]
    return {
        'anchor': 'CASE-FVM-KOVASZNAY-ORDER',
        'title': '算例池·定常 Kovasznay 流收敛阶验证',
        'reference': {'ref': REFERENCE_REF, 'doi': REFERENCE_DOI,
                      'solution': 'u = 1 - exp(lam x) cos(2 pi y), v = lam/(2 pi) exp(lam x) '
                                  'sin(2 pi y), p = (1 - exp(2 lam x))/2',
                      'lambda_identity': 'lam^2 - Re*lam = 4*pi^2'},
        'admitted_configuration': sequence['configuration'],
        'grids': list(GRIDS),
        'levels': [{key: level[key] for key in
                    ('n', 'h', 'cells', 'errors', 'max_abs_z_velocity', 'discrete_conservation',
                     'outer_iterations')} for level in levels],
        'fits': {'velocity_full': slope, 'pressure_mean_free': pressure_slope,
                 'pressure_fixed': pressure_fixed_slope,
                 'velocity_inplane': lsq_slope(levels, 'velocity_inplane')},
        'pair_orders': orders,
        'reference_audit': reference,
        'controls': controls,
        'criteria': rows,
        'verdict': {'status': 'PASS' if not failures else 'FAIL',
                    'failures': failures,
                    'graded_quantity': 'velocity_full',
                    'threshold': ORDER_MIN,
                    'observed': slope['slope']},
        'evidence': {'admitted_runs': str(admitted_dir),
                     'digests': {str(path.relative_to(base)): _sha256(path)
                                 for path in ([admitted_dir / 'summary.json']
                                              + [Path(level['paths']['cases'])
                                                 for level in levels])}},
    }


def write_admission(output, root=None):
    """Write the admission record to ``output`` and return the payload."""
    payload = evaluate(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', default='docs/case-fvm-kovasznay-order-pool/'
                                           'case_pool_entry.json')
    parser.add_argument('--root', default=None)
    args = parser.parse_args(argv)
    payload = write_admission(args.output, args.root)
    verdict = payload['verdict']
    print(f"{payload['anchor']}: {verdict['status']} "
          f"(velocity_full slope {verdict['observed']:.4f} vs {verdict['threshold']})")
    print(f"written: {args.output}")
    return 0 if verdict['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())