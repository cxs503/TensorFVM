"""Tests for the KMM180 log-law self-consistency audit and the registration gate.

Three layers, in increasing cost:

* **synthetic** -- ``log_layer_state`` / ``expected_log_layer`` must recover a
  manufactured log-layer state exactly, so a defect in the extraction cannot be
  mistaken for a physical finding;
* **registration** -- the real solver's Re_tau = 182 numbers and the Re_tau
  sweep are pinned, so a silent model/closure change fails here instead of
  quietly moving the anchor's verdict;
* **catalog** -- ``docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json`` must quote
  *both* grading windows and must not re-introduce an unqualified PASS, and the
  numbers it quotes must match a recomputation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.rans_validation.models import implied_kappa
from tensorfvm.verification.kmm180_loglaw import (
    RE_TAU_NOMINAL,
    asymptotic_sweep,
    expected_log_layer,
    log_layer_state,
    proxy_bound,
    self_consistency,
)
from tensorfvm.verification.kmm180_windows import LOG_LAYER_FRACTION, window_deviation
from tensorfvm.verification.kmm1987 import TOLERANCE_LOG_REGION, load_kmm1987_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / 'docs' / 'oss_solutions' / 'CASE-FVM-CHANNEL-KMM180.json'


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #

def manufactured_result(model='komega', *, kappa=None, n=64, re_tau=1000.0):
    """A result dict carrying the *exact* log-layer state for ``kappa``."""
    expected = expected_log_layer(model)
    kappa = expected['kappa'] if kappa is None else kappa
    beta_star = expected['beta_star']
    root = np.sqrt(beta_star)
    y = np.linspace(35.0 / re_tau, 0.2, n)              # y+ from 35 to 0.2 Re_tau
    k = np.full(n, 1.0 / root)                          # k+ = 1/sqrt(beta*)
    omega = 1.0 / (root * kappa * y)                    # omega+ y+ = 1/(sqrt(beta*) kappa)
    u_plus = np.log(y * re_tau) / kappa + 5.2
    return dict(model=model, re_tau=re_tau, y=y, y_plus=y * re_tau, k=k,
                variable_values=omega, nut=k / omega, u_plus=u_plus,
                Cf=0.0, bulk_u_plus=0.0, converged=True)


def test_expected_log_layer_is_the_closed_form():
    for model in ('sst', 'komega', 'kepsilon'):
        expected = expected_log_layer(model)
        assert expected['k_plus'] == pytest.approx(1.0 / np.sqrt(0.09), rel=1e-12)
        assert expected['kappa'] == pytest.approx(implied_kappa(model), rel=1e-12)
        if model in ('sst', 'komega'):
            assert expected['omega_plus_y_plus'] == pytest.approx(
                1.0 / (np.sqrt(0.09) * implied_kappa(model)), rel=1e-12)
        else:
            assert expected['omega_plus_y_plus'] is None


def test_log_layer_state_recovers_a_manufactured_state():
    kappa = expected_log_layer('komega')['kappa']
    result = manufactured_result()
    state = log_layer_state(result, high_fraction=0.2)
    assert state['kappa_eddy'] == pytest.approx(kappa, rel=1e-10)
    assert state['kappa_slope'] == pytest.approx(kappa, rel=1e-10)
    assert state['k_plus'] == pytest.approx(1.0 / np.sqrt(0.09), rel=1e-12)
    assert state['omega_plus_y_plus'] == pytest.approx(
        1.0 / (np.sqrt(0.09) * kappa), rel=1e-3)
    # nu_t = k/omega by construction, so the two kappa estimators must coincide.
    assert state['kappa_eddy'] == pytest.approx(state['kappa_slope'], rel=1e-10)


def test_log_layer_state_rejects_an_empty_window():
    result = manufactured_result(n=8)
    with pytest.raises(ValueError, match='holds'):
        log_layer_state(result, high_fraction=1e-4)


def test_self_consistency_flags_a_halved_eddy_viscosity():
    """A 50 % eddy-viscosity defect is reported as exactly that, not as noise."""
    result = manufactured_result()
    result['nut'] = result['nut'] * 0.5
    result['variable_values'] = 2.0 * result['variable_values']
    sc = self_consistency(result)
    assert sc['kappa_eddy_relative_error'] == pytest.approx(0.5, rel=1e-9)
    assert sc['omega_plus_y_plus_relative_error'] == pytest.approx(1.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# Registration (real solver)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def dns():
    return load_kmm1987_reference()


@pytest.fixture(scope='module')
def sst_result(dns):
    return solve_channel('sst', 256, re_tau=float(dns['re_tau']))


def test_sst_reference_run_matches_the_registered_verdict(sst_result, dns):
    """The 3.63 % / 6.32 % pair of ``m16513`` must stay put."""
    full = window_deviation(sst_result['y_plus'], sst_result['u_plus'], dns,
                            high=float(np.max(dns['y_plus'])))
    log = window_deviation(sst_result['y_plus'], sst_result['u_plus'], dns,
                           high=LOG_LAYER_FRACTION * float(dns['re_tau']))
    assert full['rms_relative'] < TOLERANCE_LOG_REGION          # legacy window passes
    assert log['rms_relative'] > TOLERANCE_LOG_REGION           # physical log layer fails
    assert log['rms_relative'] > full['rms_relative']           # dilution direction
    # Window dilution, pinned: widening the window must not shrink a real defect.
    assert full['rms_relative'] == pytest.approx(0.0346, abs=3e-3)


def test_sst_eddy_viscosity_is_half_its_own_log_law_at_re_tau_180(sst_result):
    """The measured finding: nu_t+ is ~0.20 y+ where the model's closed form says 0.41 y+.

    This is the Q1 answer recorded for the anchor.  It is *not* asserted to be a
    defect here -- the sweep test below shows it relaxes towards the closed form
    as Re_tau grows -- only that it does not silently change.
    """
    state = log_layer_state(sst_result)
    assert 0.18 < state['kappa_eddy'] < 0.23
    assert state['kappa_eddy'] < 0.75 * implied_kappa('sst')


def test_momentum_identity_closes(sst_result):
    """The mean profile is the exact integral of ``(1 - y/h)/(1 + nu_t/nu)``.

    A 0.1 % closure means the u+ deficit cannot be a momentum-flux bookkeeping
    error: it is carried entirely by ``nu_t``.
    """
    sc = self_consistency(sst_result)
    mc = sc['momentum_consistency']
    assert mc['relative_error'] < 1e-3
    # Because nu_t is far below its own log law, the profile's nominal kappa is
    # between the eddy-viscosity kappa and the closed form -- not above either.
    assert sc['measured']['kappa_eddy'] < sc['measured']['kappa_slope'] < sc['expected']['kappa']


def test_the_residual_gap_shrinks_towards_the_total_stress_prediction():
    """Q2: the Re_tau = 180 shortfall is finite-Re physics, not a closure defect.

    Two claims, both falsifiable: k+ must *rise* towards ``1/sqrt(beta*)`` with
    Re_tau, and the distance to the ``(1 - y/h)``-corrected prediction -- which
    uses no fitted parameter -- must *fall*.  A bookkeeping defect would keep
    both flat.
    """
    rows = asymptotic_sweep('sst', cells=256, re_taus=(550.0, 2000.0))
    assert all(row['converged'] for row in rows)
    low, high = rows
    assert high['measured']['k_plus'] > low['measured']['k_plus']
    assert high['measured']['kappa_eddy'] > low['measured']['kappa_eddy']
    assert high['k_plus_prediction_relative_error'] < low['k_plus_prediction_relative_error']
    # The prediction must not be vacuous: it has to land within 15 % of the state.
    assert high['k_plus_prediction_relative_error'] < 0.15


def test_reference_substitution_is_bounded(monkeypatch, dns):
    """Re_tau = 180 vs 182.088 must not flip either verdict, and stay small."""
    bound = proxy_bound('sst', n=128, dns=dns, nominal=RE_TAU_NOMINAL)
    assert bound['verdict_full_invariant']
    assert bound['verdict_log_layer_invariant']
    assert bound['delta_rms_log_layer'] < 0.005      # < 0.5 percentage points
    assert bound['delta_rms_full'] < 0.005
    assert bound['Cf_relative_change'] < 0.01
    assert bound['bulk_relative_change'] < 0.01
    # A bogus nominal Reynolds number must be detected, not smoothed over.
    coarse = proxy_bound('sst', n=128, dns=dns, nominal=0.5 * RE_TAU_NOMINAL)
    assert coarse['delta_rms_log_layer'] > bound['delta_rms_log_layer']


# --------------------------------------------------------------------------- #
# Catalog registration gate
# --------------------------------------------------------------------------- #

def _catalog():
    return json.loads(CATALOG.read_text())


def test_catalog_entry_quotes_both_windows():
    entry = _catalog()
    windows = entry['grading_windows']
    assert set(windows) >= {'legacy_full', 'log_layer'}
    assert windows['log_layer']['definition'].startswith('y+ in [30, 0.2*Re_tau]')
    for window in windows.values():
        assert isinstance(window['sst_rms_relative'], float)
        assert window['criterion'].endswith('%')


def test_catalog_entry_does_not_claim_an_unqualified_pass():
    entry = _catalog()
    verdicts = entry['grading_windows']
    assert entry['verdict_qualified'] is True
    # The log layer must be recorded as failing, and the headline verdict must not
    # read as a bare PASS.
    assert verdicts['log_layer']['sst_verdict'] == 'FAIL'
    assert verdicts['legacy_full']['sst_verdict'] == 'PASS'
    assert 'PASS' not in entry['verdict'] or 'qualified' in entry['verdict'].lower()
    assert entry['tolerance'] == pytest.approx(TOLERANCE_LOG_REGION)


def test_catalog_numbers_match_a_recomputation(sst_result, dns):
    """Zero hand-typed numbers: the catalog's two RMS values are re-derived here."""
    windows = _catalog()['grading_windows']
    full = window_deviation(sst_result['y_plus'], sst_result['u_plus'], dns,
                            high=float(np.max(dns['y_plus'])))
    log = window_deviation(sst_result['y_plus'], sst_result['u_plus'], dns,
                           high=LOG_LAYER_FRACTION * float(dns['re_tau']))
    assert full['rms_relative'] == pytest.approx(
        windows['legacy_full']['sst_rms_relative'], abs=3e-3)
    assert log['rms_relative'] == pytest.approx(
        windows['log_layer']['sst_rms_relative'], abs=3e-3)


def test_catalog_evidence_pointers_exist():
    entry = _catalog()
    pointers = entry['impl_files'] + entry['artifacts']
    missing = [p for p in pointers if not (REPO_ROOT / p).exists()]
    assert not missing, f'dead evidence pointers: {missing}'
    assert entry['repro_cmd'], 'a reproducer must be quoted'
    assert entry['test_cmd'], 'a test command must be quoted'