"""Reference-law, DNS-import and metric checks.

The DNS numbers quoted here are the ones in the tarball imported by
``scripts/import_channel_dns.py`` (Lee & Moser 2015, case ``0180``): the header
carries ``Re_tau = 182.088`` and ``Cf = 0.0081233``.  They are read back from the
imported CSV/provenance pair, never retyped as a reference value.
"""
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.reference import (dns_deviation, load_dns_profile, log_law,
                                                 log_law_indicator, log_region_deviation,
                                                 reichardt)

DNS_ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'rans-turbulence-validation-matrix' / 'dns'


def test_reichardt_and_log_law_agree_in_the_overlap_layer():
    y = np.logspace(np.log10(50.), np.log10(200.), 40)
    relative = reichardt(y) / log_law(y) - 1
    assert np.max(np.abs(relative)) < .05
    assert np.all(np.diff(reichardt(y)) > 0)


def test_reichardt_reaches_the_documented_viscous_limit():
    assert reichardt(np.array([1e-8]))[0] == pytest.approx(0., abs=1e-6)
    assert reichardt(np.array([5.]))[0] == pytest.approx(5., abs=.35)


@pytest.mark.parametrize('bad', [-1., np.nan])
def test_reference_laws_reject_invalid_wall_coordinates(bad):
    with pytest.raises(ValueError):
        reichardt(np.array([bad]))
    with pytest.raises(ValueError):
        log_law(np.array([bad]))


def test_log_region_deviation_is_zero_against_its_own_values():
    y = np.linspace(31., 110., 50)
    deviation = log_region_deviation(y, log_law(y), log_law, re_tau=590.)
    assert deviation['rms_relative'] < 1e-15 and deviation['max_relative'] < 1e-15
    assert deviation['cells'] == 50
    assert deviation['high'] == pytest.approx(.2 * 590.)


def test_log_region_deviation_detects_a_constant_offset():
    y = np.linspace(31., 110., 50)
    deviation = log_region_deviation(y, log_law(y) * 1.1, log_law, re_tau=590.)
    assert deviation['rms_relative'] == pytest.approx(.1, rel=1e-12)


def test_log_region_deviation_rejects_a_window_with_too_few_cells():
    with pytest.raises(ValueError):
        log_region_deviation(np.array([1., 2., 3.]), np.array([1., 2., 3.]), log_law, re_tau=590.)


def test_log_law_indicator_returns_the_slope_of_a_logarithmic_profile():
    y = np.linspace(31., 110., 60)
    indicator = log_law_indicator(y, log_law(y), re_tau=590.)
    assert indicator['kappa_local'] == pytest.approx(.41, rel=2e-3)
    assert indicator['xi_spread'] < 5e-3


@pytest.mark.skipif(not (DNS_ROOT / '0180' / 'mean.csv').is_file(), reason='DNS case 0180 not imported')
def test_dns_profile_carries_provenance_and_header_parameters():
    profile = load_dns_profile(DNS_ROOT / '0180')
    assert profile['re_tau'] == pytest.approx(182.088, abs=1e-3)
    assert profile['Cf'] == pytest.approx(.0081233, abs=1e-6)
    assert len(profile['raw_sha256']) == 64
    assert profile['upstream'].endswith('LM_Channel_0180_mean_prof.dat')
    assert np.all(np.diff(profile['y_plus']) > 0)
    assert profile['y_plus'][-1] <= profile['re_tau'] + 1e-6


@pytest.mark.skipif(not (DNS_ROOT / '0180' / 'mean.csv').is_file(), reason='DNS case 0180 not imported')
def test_dns_deviation_is_zero_for_the_dns_profile_itself():
    profile = load_dns_profile(DNS_ROOT / '0180')
    inside = (profile['y_plus'] > 30.) & (profile['y_plus'] < .2 * profile['re_tau'])
    deviation = dns_deviation(profile['y_plus'][inside], profile['u_plus'][inside], profile)
    assert deviation['rms_relative'] < 1e-12
    assert deviation['dns_Cf'] == pytest.approx(profile['Cf'])


@pytest.mark.skipif(not (DNS_ROOT / '0180' / 'mean.csv').is_file(), reason='DNS case 0180 not imported')
def test_dns_deviation_rejects_windows_outside_the_sample():
    profile = load_dns_profile(DNS_ROOT / '0180')
    y = np.array([10., 20., 25.])
    with pytest.raises(ValueError):
        dns_deviation(y, np.ones_like(y), profile)