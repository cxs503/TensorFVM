"""Tests for the proxy-free (direct-anchor) grading of CASE-FVM-CHANNEL-KMM180.

Five layers, on purpose:

1. **Reference integrity** -- the committed MKM1999 profile must reproduce its
   own provenance record (digest, ``Cf``, ``U_bulk+``, ``Re_tau``) and must
   *reject* a tampered copy.  A grading table that cannot detect modification
   is not a reference.
2. **Importer identities** -- ``U_bulk+``/``Cf`` are recomputed from the profile
   by half-channel symmetry; on a synthetic profile whose integral is known by
   hand the values must be exact.  A wrong digest must be refused.
3. **Window statistics** -- exact values on synthetic profiles (identical
   profile -> 0, constant offset -> the offset), and refusal of windows thinner
   than three cells.
4. **The substitution measurement** -- the claim that used to be *withheld*
   ("the two DNS sets agree within 1 %") is now a number; it is pinned here so
   a silent data swap cannot keep the claim true by accident.
5. **Grading contract** -- the direct anchor verdict must agree with the
   published proxy verdict to well inside the 5 % tolerance, and an ungradable
   window must report ``NOT_GRADED`` rather than pass.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import kmm180_direct_anchor as da
from tensorfvm.verification.kmm1987 import load_kmm1987_reference

sys_scripts = Path(__file__).resolve().parents[1] / 'scripts'


# --------------------------------------------------------------------------- #
# 1. Reference integrity
# --------------------------------------------------------------------------- #

def test_reference_reproduces_its_provenance():
    dns = da.load_mkm1999_reference()
    provenance = dns['provenance']
    payload = (da.REFERENCE_DIR / 'mean.csv').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == provenance['files']['mean']['csv_sha256']
    assert dns['re_tau'] == pytest.approx(178.12, abs=1e-9)
    assert dns['Cf'] == pytest.approx(provenance['Cf'], rel=0, abs=1e-12)
    assert dns['U_bulk_plus'] == pytest.approx(provenance['U_bulk_plus'], rel=0, abs=1e-9)


def test_reference_declares_the_kmm1987_method():
    """The whole point of the substitution: this set is the anchor's lineage."""
    provenance = da.load_mkm1999_reference()['provenance']
    assert 'Kim, Moin' in provenance['header_facts']['numerical_method']
    assert provenance['header_facts']['re_tau'] == pytest.approx(178.12)
    assert provenance['source']['tarball_sha256']
    assert provenance['source']['member_sha256']


def test_tampered_reference_is_rejected(tmp_path):
    directory = tmp_path / 'ref'
    directory.mkdir()
    shutil.copy(da.REFERENCE_DIR / 'provenance.json', directory / 'provenance.json')
    rows = (da.REFERENCE_DIR / 'mean.csv').read_text().splitlines()
    rows[5] = rows[5].replace(rows[5].split(',')[2], '99.0')
    (directory / 'mean.csv').write_text('\n'.join(rows) + '\n')
    with pytest.raises(ValueError, match='digest'):
        da.load_mkm1999_reference(directory)


# --------------------------------------------------------------------------- #
# 2. Importer identities
# --------------------------------------------------------------------------- #

def _import_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'import_mkm1999_channel', sys_scripts / 'import_mkm1999_channel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bulk_and_friction_on_analytic_profile():
    """``u+ = a - b*(y/h)^2`` has ``U_bulk+ = a - b/3`` and ``Cf = 2/U_bulk+^2``."""
    importer = _import_module()
    a, b = 18.0, 9.0
    y = np.linspace(0.0, 1.0, 2001)
    u = a - b * y ** 2
    data = np.zeros((y.size, 7))
    data[:, 0] = y
    data[:, 1] = y * 178.12
    data[:, 2] = u
    bulk, cf = importer.bulk_and_friction(data)
    # the trapezoidal rule is exact for a quadratic only up to O(h^2)
    assert bulk == pytest.approx(a - b / 3.0, rel=1e-7)
    assert cf == pytest.approx(2.0 / (a - b / 3.0) ** 2, rel=1e-7)

    # ... and exact for a linear profile, which pins the quadrature itself
    linear = data.copy()
    linear[:, 2] = 3.0 + 5.0 * y
    bulk_linear, cf_linear = importer.bulk_and_friction(linear)
    assert bulk_linear == pytest.approx(3.0 + 2.5, rel=1e-13)
    assert cf_linear == pytest.approx(2.0 / 5.5 ** 2, rel=1e-13)


def test_parse_means_reads_header_and_rejects_missing_method():
    importer = _import_module()
    rows = ['0.00e+00 0.000e+00 0.000e+00 1.7812e+02 0.0 2.3e-01 6.2e-13']
    for k in range(1, 9):
        eta = k / 8.0
        rows.append(f'{eta:.5e} {eta * 178.12:.5e} {18.3 * eta:.5e} '
                    f'{1.0 - eta:.1e} 0.0 0.0 0.0')
    good = '\n'.join([
        '# Numerical Method: Kim, Moin & Moser, 1987, J. Fluid Mech. vol 177, 133-166',
        '# Re_tau = 178.12',
        '# ny = 129,  Re = 178.12',
    ] + rows)
    parsed = importer.parse_means(good)
    assert parsed['re_tau'] == pytest.approx(178.12)
    assert parsed['data'].shape == (9, 7)

    bad = good.replace('Kim, Moin & Moser', 'some other scheme')
    with pytest.raises(ValueError, match='KMM1987 numerical method'):
        importer.parse_means(bad)


def test_wrong_archive_digest_is_refused(tmp_path):
    importer = _import_module()
    with pytest.raises(ValueError, match='digest'):
        importer.build(tmp_path, tarball=da.REFERENCE_DIR / 'mean.csv',
                       expected_tarball_sha256='0' * 64)


# --------------------------------------------------------------------------- #
# 3. Window statistics
# --------------------------------------------------------------------------- #

def synthetic_dns(y_plus, u_plus, re_tau=None):
    return dict(name='synthetic', y_plus=np.asarray(y_plus, dtype=float),
                u_plus=np.asarray(u_plus, dtype=float),
                re_tau=float(np.max(y_plus) if re_tau is None else re_tau))


def test_identical_profile_deviates_by_zero():
    dns = synthetic_dns(np.linspace(0, 180, 91), 2.44 * np.log(np.linspace(1, 181, 91)) + 5.2)
    stats = da.window_stats(dns['y_plus'], dns['u_plus'], dns)
    assert stats['rms_relative'] == pytest.approx(0.0, abs=1e-14)
    assert stats['offset'] == pytest.approx(0.0, abs=1e-12)


def test_constant_relative_offset_is_returned_exactly():
    y = np.linspace(0, 180, 91)
    truth = 2.44 * np.log(y + 1.0) + 5.2
    dns = synthetic_dns(y, truth)
    stats = da.window_stats(y, truth * 1.02, dns)
    # relative deviation is (1.02*truth - truth)/truth = 2 % exactly
    assert stats['rms_relative'] == pytest.approx(0.02, rel=1e-12)
    assert stats['mean_relative'] == pytest.approx(0.02, rel=1e-12)


def test_too_thin_window_raises_instead_of_returning_a_number():
    y = np.array([0.0, 20.0, 40.0, 60.0])
    dns = synthetic_dns(y, 2.0 * np.log(y + 1.0) + 1.0)
    with pytest.raises(ValueError, match='fewer than three'):
        da.window_stats(y, dns['u_plus'], dns, low=30.0, high=42.0)


# --------------------------------------------------------------------------- #
# 4. The substitution, measured
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def substitution():
    return da.reference_substitution()


def test_substitution_is_below_the_previously_withheld_bound(substitution):
    """The withdrawn claim was "agreement within 1 %".  Now measured, not asserted."""
    for window in ('full', 'log_layer'):
        entry = substitution['windows'][window]
        assert entry['rms_relative'] < 0.01, (window, entry['rms_relative'])
        assert entry['max_relative'] < 0.01, (window, entry['max_relative'])


def test_substitution_costs_a_small_share_of_the_acceptance_budget(substitution):
    assert substitution['windows']['full']['rms_relative'] < 0.25 * da.TOLERANCE_LOG_REGION
    assert 0.0 < substitution['budget_fraction'] < 0.25


def test_substitution_records_both_integral_quantities(substitution):
    assert substitution['delta_Re_tau'] == pytest.approx(182.088 - 178.12, abs=1e-6)
    # the lower-Reynolds-number run carries the larger skin friction
    assert substitution['anchor']['Cf'] > substitution['proxy']['Cf']
    assert substitution['delta_Cf_relative'] < 0.005
    assert substitution['delta_U_bulk_plus_relative'] < 0.005


def test_offset_in_wall_units_is_reported_not_only_relative():
    sub = da.reference_substitution()
    entry = sub['windows']['full']
    # a sub-percent relative difference still means a finite u+ offset
    assert entry['mean_offset'] != 0.0
    assert abs(entry['max_offset']) < 0.25


# --------------------------------------------------------------------------- #
# 5. Grading contract
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def graded_case():
    anchor = da.load_mkm1999_reference()
    proxy = load_kmm1987_reference()
    run = da.solve('sst', 128)
    return dict(anchor=da.grade(run['y_plus'], run['u_plus'], anchor),
                proxy=da.grade(run['y_plus'], run['u_plus'], proxy),
                run=run)


def test_direct_anchor_verdict_matches_the_proxy_verdict(graded_case):
    anchor, proxy = graded_case['anchor'], graded_case['proxy']
    for window in ('full', 'log_layer'):
        assert anchor[window]['verdict'] == proxy[window]['verdict']
        delta = abs(anchor[window]['rms_relative'] - proxy[window]['rms_relative'])
        assert delta < 0.01, (window, delta)


def test_published_sst_log_layer_failure_is_reproduced(graded_case):
    """The honest verdict of the anchor: SST fails the log layer, passes the widened window."""
    anchor = graded_case['anchor']
    assert anchor['log_layer']['verdict'] == 'FAIL'
    assert anchor['log_layer']['rms_relative'] > da.TOLERANCE_LOG_REGION
    assert anchor['full']['verdict'] == 'PASS'


def test_ungradable_window_is_not_reported_as_pass():
    y = np.linspace(0.0, 180.0, 65)
    dns = synthetic_dns(y, 2.5 * np.log(y + 1.0) + 5.0, re_tau=178.12)
    graded = da.grade(y, dns['u_plus'] * 1.3, dns)
    assert graded['log_layer']['verdict'] == 'NOT_GRADED'
    assert graded['overall_verdict'] == 'NOT_GRADED'


def test_windows_follow_each_reference_reynolds_number():
    anchor = da.load_mkm1999_reference()
    proxy = load_kmm1987_reference()
    windows_anchor = da.reference_windows(anchor)
    windows_proxy = da.reference_windows(proxy)
    assert windows_anchor['log_layer']['high'] < windows_proxy['log_layer']['high']
    assert windows_anchor['full']['high'] < windows_proxy['full']['high']
    assert windows_anchor['log_layer']['high'] == pytest.approx(
        da.LOG_LAYER_FRACTION * anchor['re_tau'])


def test_reference_name_falls_back_for_the_legacy_loader():
    """The LM2015 loader predates the ``name`` key; naming must not KeyError."""
    assert da.reference_name(load_kmm1987_reference())
    assert da.reference_name(da.load_mkm1999_reference()) != 'reference'