"""Budget-level audit tests for CASE-FVM-CHANNEL-KMM180.

Two classes of check, mirroring ``test_channel_kmm180_windows``:

* synthetic -- the parsers, the two-route shear recovery and the integral
  balance must work on manufactured data whose answer is known, and must
  *reject* files that break the archive's conventions (wrong column count,
  ``y+`` that is not ``(y/delta) Re_tau``, a rescaled dissipation column);
* registration -- the real imported Lee & Moser archive and the real closures,
  so that a silent change in either (or in the wall-unit conventions used to
  interpret them) fails the suite instead of passing quietly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.verification.kmm180_budget import (
    BUDGET_COLUMNS,
    closure_identity,
    compare_to_dns,
    global_balance,
    load_budget,
    model_budget,
    parse_header,
    production_identity,
    shear_from_momentum,
    shear_from_production,
    window_stats,
    windows,
)
from tensorfvm.verification.kmm1987 import DNS_ROOT, load_kmm1987_reference

RE_TAU = 182.088


# --------------------------------------------------------------------------- #
# Synthetic archive
# --------------------------------------------------------------------------- #

def write_case(root, *, rows=16, re_tau=RE_TAU, scale=1.0, reverse=False,
               columns=None, gap_in_dissipation=0.0):
    """Write a minimal but conventional archive under ``root``.

    The manufactured profiles satisfy ``P+ = -u'v'>+ dU+/dy+`` and
    ``int P+ d(y/h) = int eps+ d(y/h)`` so that the audit's own answers are
    known: a correct implementation recovers zero residual, and each argument
    that perturbs the file must be detectable.
    """
    labels = tuple(BUDGET_COLUMNS if columns is None else columns)
    cache = Path(root) / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    case_dir = Path(root) / '0180'
    case_dir.mkdir(parents=True, exist_ok=True)
    half = np.linspace(0.0, 1.0, rows)
    y_plus = half * re_tau
    slope = 1.0 / (1.0 + 40.0 * half) ** 2          # dU+/dy+, positive, decaying
    shear = 1.0 - half - slope                       # momentum balance
    production = shear * slope                       # P+ = -u'v'>+ dU+/dy+
    dissipation = production * (1.0 + gap_in_dissipation)
    data = np.column_stack([half, (y_plus[::-1] if reverse else y_plus),
                            production, np.zeros(rows), np.zeros(rows),
                            np.zeros(rows), np.zeros(rows), dissipation,
                            np.zeros(rows)])
    if scale != 1.0:
        data[:, 7] *= scale
    header = [
        '% Filename : LM_Channel_0180_RSTE_k_prof.dat ',
        '%         y/delta                    y^+                   '
        + '          '.join(labels),
        '%' + '-' * 40,
    ]
    (cache / 'LM_Channel_0180_RSTE_k_prof.dat').write_text(
        '\n'.join(header) + '\n'
        + '\n'.join('    ' + '    '.join('%.16e' % value for value in row)
                    for row in data) + '\n')
    (case_dir / 'provenance.json').write_text(
        json.dumps({'Re_tau_reported': re_tau}))
    return dict(path=cache / 'LM_Channel_0180_RSTE_k_prof.dat',
                y_over_delta=half, y_plus=y_plus, slope=slope, shear=shear,
                production=production, dissipation=dissipation)


def synthetic_dns(fixture, re_tau=RE_TAU):
    """The mean-profile view of the same manufactured state."""
    return dict(y_plus=fixture['y_plus'], y_over_delta=fixture['y_over_delta'],
                dU_plus_dy_plus=fixture['slope'], u_plus=np.zeros_like(fixture['y_plus']),
                re_tau=float(re_tau))


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

def test_parse_header_reads_the_declared_column_order():
    labels = parse_header('%   y/delta   y^+   ' + '   '.join(BUDGET_COLUMNS))
    assert labels == ('y/delta', 'y^+') + tuple(BUDGET_COLUMNS)


@pytest.mark.parametrize('text', [
    '% no labels here\n',
    '%   y/delta   y^+   Production\n',                       # too few columns
    '%   y/delta   y^+   ' + '   '.join(BUDGET_COLUMNS[:-1]) + '\n',
])
def test_parse_header_rejects_nonconforming_headers(text):
    with pytest.raises(ValueError):
        parse_header(text)


def test_load_budget_accepts_the_conventional_file(tmp_path):
    fixture = write_case(tmp_path)
    budget = load_budget('0180', 'k', root=tmp_path)
    assert budget['rows'] == 16
    assert budget['re_tau'] == pytest.approx(RE_TAU)
    assert np.allclose(budget['y_plus'], fixture['y_plus'])
    assert len(budget['sha256']) == 64


def test_load_budget_rejects_a_wall_coordinate_that_is_not_y_plus(tmp_path):
    write_case(tmp_path, reverse=True)
    with pytest.raises(ValueError):
        load_budget('0180', 'k', root=tmp_path)


def test_load_budget_rejects_a_nonconforming_column_count(tmp_path):
    write_case(tmp_path, columns=BUDGET_COLUMNS[:-1])
    with pytest.raises(ValueError):
        load_budget('0180', 'k', root=tmp_path)


def test_load_budget_rejects_a_missing_kind():
    with pytest.raises(ValueError):
        load_budget('0180', 'nope')


# --------------------------------------------------------------------------- #
# Two routes to the Reynolds shear stress
# --------------------------------------------------------------------------- #

def test_production_route_inverts_the_budget_definition(tmp_path):
    fixture = write_case(tmp_path)
    budget = load_budget('0180', 'k', root=tmp_path)
    dns = synthetic_dns(fixture)
    recovered = shear_from_production(dns, budget)
    assert np.allclose(recovered, fixture['shear'], rtol=1e-12, atol=1e-14)
    assert np.allclose(shear_from_momentum(dns), fixture['shear'], atol=1e-14)


def test_production_route_refuses_a_non_positive_slope(tmp_path):
    fixture = write_case(tmp_path)
    budget = load_budget('0180', 'k', root=tmp_path)
    dns = synthetic_dns(fixture)
    dns['dU_plus_dy_plus'] = -np.abs(fixture['slope'])
    assert np.all(np.isnan(shear_from_production(dns, budget)))


def test_identity_is_exact_on_consistent_data(tmp_path):
    fixture = write_case(tmp_path)
    budget = load_budget('0180', 'k', root=tmp_path)
    identity = production_identity(synthetic_dns(fixture), budget)
    assert identity['comparable_cells'] > 0
    assert identity['rms_relative'] < 1e-12
    assert identity['max_relative'] < 1e-12


def test_identity_detects_a_rescaled_production_column(tmp_path):
    fixture = write_case(tmp_path)
    budget = load_budget('0180', 'k', root=tmp_path)
    budget['production'] = budget['production'] * 1.10
    identity = production_identity(synthetic_dns(fixture), budget)
    assert identity['rms_relative'] == pytest.approx(0.10, rel=1e-6)


def test_identity_excludes_the_ill_conditioned_near_wall_points(tmp_path):
    fixture = write_case(tmp_path)
    budget = load_budget('0180', 'k', root=tmp_path)
    dns = synthetic_dns(fixture)
    identity = production_identity(dns, budget, floor_fraction=0.5)
    expected = int(np.count_nonzero(np.abs(identity['shear_momentum'])
                                    >= 0.5 * identity['stress_scale']))
    assert identity['comparable_cells'] == expected
    assert identity['comparable_cells'] < np.size(identity['shear_momentum'])


# --------------------------------------------------------------------------- #
# Integral balance and window statistics
# --------------------------------------------------------------------------- #

def test_global_balance_closes_on_consistent_data(tmp_path):
    fixture = write_case(tmp_path)
    budget = load_budget('0180', 'k', root=tmp_path)
    totals = global_balance(budget, synthetic_dns(fixture))
    assert totals['production'] == pytest.approx(totals['dissipation'], rel=1e-12)
    assert abs(totals['gap_relative']) < 1e-12


def test_global_balance_detects_a_rescaled_dissipation(tmp_path):
    fixture = write_case(tmp_path, gap_in_dissipation=0.10)
    budget = load_budget('0180', 'k', root=tmp_path)
    totals = global_balance(budget, synthetic_dns(fixture))
    assert totals['gap_relative'] == pytest.approx(-0.10, rel=1e-6)


def test_windows_tile_the_channel():
    bands = windows(RE_TAU)
    assert bands[0][1] == 0.0
    assert bands[-1][2] == pytest.approx(RE_TAU)
    for earlier, later in zip(bands, bands[1:]):
        assert earlier[2] == pytest.approx(later[1])


def test_window_stats_reports_an_empty_band_instead_of_dropping_it():
    y_plus = np.array([40.0, 60.0, 80.0])
    stats = window_stats(y_plus, 2.0 * y_plus, y_plus, re_tau=RE_TAU)
    assert stats['viscous']['cells'] == 0
    assert stats['viscous']['rms_relative'] is None
    assert stats['log']['cells'] == 0
    assert stats['outer']['cells'] == 3
    assert stats['outer']['rms_relative'] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Real archive (registration)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def real():
    if not (Path(DNS_ROOT).parent / 'cache' / 'LM_Channel_0180_RSTE_k_prof.dat').is_file():
        pytest.skip('imported Lee & Moser budget files are not present')
    dns = load_kmm1987_reference()
    return dns, load_budget('0180', 'k', re_tau=float(dns['re_tau']))


def test_real_budget_grid_matches_the_mean_profile(real):
    dns, budget = real
    assert budget['rows'] == np.size(dns['y_plus'])
    assert np.max(np.abs(budget['y_plus'] - dns['y_plus'])) == 0.0


def test_real_two_route_shear_agrees_below_half_a_percent(real):
    dns, budget = real
    identity = production_identity(dns, budget)
    assert identity['comparable_cells'] > 0.8 * np.size(dns['y_plus'])
    assert identity['rms_relative'] < 5e-3
    assert identity['max_relative'] < 1e-2
    for band in ('log', 'outer'):
        assert identity['bands'][band]['rms_relative'] < 5e-3


def test_real_dns_closes_its_own_global_balance(real):
    dns, budget = real
    totals = global_balance(budget, dns)
    assert totals['production'] > 0.0
    assert totals['dissipation'] > 0.0
    assert abs(totals['gap_relative']) < 1e-2
    assert totals['pointwise_balance_relative'] < 5e-2


def test_real_dissipation_is_positive_everywhere(real):
    _, budget = real
    assert np.all(budget['dissipation'] > 0.0)
    assert np.all(budget['production'] >= 0.0)


# --------------------------------------------------------------------------- #
# Closures
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('model', ['sst', 'komega', 'kepsilon'])
def test_closure_identity_holds_for_every_model(model):
    result = solve_channel(model, 64, re_tau=RE_TAU)
    state = model_budget(result)
    identity = closure_identity(state)
    assert identity['cells'] > 0
    # SST's stress limiter is allowed to clip nu_t, so it is checked at 5 %.
    tolerance = 5e-2 if model == 'sst' else 1e-9
    assert identity['max_relative'] < tolerance


@pytest.mark.parametrize('model', ['sst', 'komega', 'kepsilon'])
def test_model_budget_is_positive_and_finite(model):
    state = model_budget(solve_channel(model, 64, re_tau=RE_TAU))
    assert np.all(np.isfinite(state['production_plus']))
    assert np.all(state['production_plus'] >= 0.0)
    assert np.all(state['dissipation_plus'] > 0.0)
    assert np.all(state['slope_plus'] > 0.0)


def test_model_comparison_registers_the_shear_split(real):
    """The total stress is imposed by momentum, the split is not.

    This is the substantive claim of the module, so it is registered against the
    real archive rather than asserted in prose.
    """
    dns, budget = real
    state = model_budget(solve_channel('sst', 128, re_tau=float(dns['re_tau'])))
    comparison = compare_to_dns(state, dns, budget)
    log_band = comparison['shear_plus']['log']
    assert log_band['cells'] >= 2
    # Total stress matches to a few percent ...
    assert log_band['rms_relative'] < 5e-2
    # ... while the eddy viscosity that carries it does not.
    assert comparison['nut_plus']['log']['rms_relative'] > 5e-2