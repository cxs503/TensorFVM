"""DNS reference profile vs classic log-law and Reichardt composite law.

Why this file exists
--------------------
The CASE-FVM-CHANNEL-KMM180 anchor grades RANS models against the Lee & Moser
(2015) DNS at Re_tau = 182.088, but the repository never quantified how well
the DNS *itself* matches the classic logarithmic law ``U+ = ln(y+)/kappa + B``
or the Reichardt (1951) composite law.  Without this number, a reader cannot
tell whether a 6 % RANS-vs-DNS deviation is mostly model error or mostly the
DNS's own finite-Reynolds-number departure from the asymptotic log law.

This module fills that verification debt by:

1. Loading the imported DNS profile (Lee & Moser 2015, case 0180);
2. Comparing it against the classic log law (kappa = 0.41, B = 5.2) in the
   physical log-layer window ``y+ in [30, 0.2*Re_tau]``;
3. Comparing it against Reichardt's composite law over the same window;
4. Fitting ``U+ = ln(y+)/kappa + B`` to the DNS data to extract the *effective*
   kappa and B that the DNS implies at this Reynolds number;
5. Computing the log-law indicator ``Xi = y+ dU+/dy+`` whose reciprocal is the
   local von Karman constant.

All numbers are derived from the imported CSV; nothing is hand-typed.
"""
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.reference import (
    fit_log_law,
    load_dns_profile,
    log_law,
    log_law_indicator,
    log_region_deviation,
    reichardt,
)
from tensorfvm.wall_functions import KAPPA

DNS_ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'rans-turbulence-validation-matrix' / 'dns'


@pytest.fixture(scope='module')
def dns():
    return load_dns_profile(DNS_ROOT / '0180')


# --------------------------------------------------------------------------- #
# DNS vs classic log law
# --------------------------------------------------------------------------- #

def test_dns_vs_classic_log_law_in_log_region(dns):
    """The DNS deviates from the classic log law in the physical log layer.

    At Re_tau = 182 the log layer is only ~6 wall units wide (30 to 36.4),
    so the DNS cannot be expected to match the asymptotic law exactly.
    This test records the actual deviation so it is not mistaken for RANS error.
    """
    y_plus = dns['y_plus']
    u_plus = dns['u_plus']
    re_tau = dns['re_tau']

    # Physical log layer: y+ in [30, 0.2*Re_tau]
    deviation = log_region_deviation(y_plus, u_plus, log_law, re_tau=re_tau,
                                     low=30., high_fraction=0.2)
    # The DNS deviates ~3 % from the classic log law in this narrow window.
    # This is finite-Reynolds-number physics, not a data defect.
    assert 0.02 < deviation['rms_relative'] < 0.05, (
        f'DNS-vs-loglaw RMS {deviation["rms_relative"]:.4f} outside expected range')
    assert deviation['cells'] >= 3
    # The DNS mean is slightly above the classic log law in this window
    # (the finite-Re wake effect pushes u+ up).
    assert deviation['mean_relative'] > 0


def test_dns_vs_reichardt_in_log_region(dns):
    """Reichardt's composite law fits the DNS better than the pure log law.

    Reichardt (1951) was fitted to channel/pipe data and includes the viscous
    sublayer transition, so it should be closer to the DNS than the asymptotic
    log law.
    """
    y_plus = dns['y_plus']
    u_plus = dns['u_plus']
    re_tau = dns['re_tau']

    deviation_log = log_region_deviation(y_plus, u_plus, log_law, re_tau=re_tau,
                                         low=30., high_fraction=0.2)
    deviation_reich = log_region_deviation(y_plus, u_plus, reichardt, re_tau=re_tau,
                                           low=30., high_fraction=0.2)
    # Reichardt should be closer to the DNS than the pure log law.
    assert deviation_reich['rms_relative'] < deviation_log['rms_relative']
    assert deviation_reich['rms_relative'] < 0.03


# --------------------------------------------------------------------------- #
# Effective log-law parameters from DNS
# --------------------------------------------------------------------------- #

def test_dns_fitted_kappa_is_below_the_classic_value(dns):
    """The DNS implies a lower effective kappa than 0.41 at Re_tau = 182.

    A least-squares fit of ``U+ = ln(y+)/kappa + B`` to the DNS data in the
    log region yields the effective slope.  At low Re_tau the finite-domain
    effect (1 - y/h) reduces the apparent slope.
    """
    fit = fit_log_law(dns['y_plus'], dns['u_plus'], re_tau=dns['re_tau'],
                      low=30., high_fraction=0.2)
    # The fitted kappa should be below 0.41 (classic value) at this Re_tau.
    assert fit['kappa'] < KAPPA
    # At Re_tau=182 the effective kappa is around 0.32-0.35 due to finite-Re effects.
    assert 0.30 < fit['kappa'] < 0.40
    assert fit['cells'] >= 3


def test_dns_fitted_log_law_has_small_residual(dns):
    """The fitted log law has a small RMS residual in the log window.

    Even though the effective kappa differs from 0.41, the DNS data in the
    log window is well approximated by *some* log law -- the question is which
    kappa and B.
    """
    fit = fit_log_law(dns['y_plus'], dns['u_plus'], re_tau=dns['re_tau'],
                      low=30., high_fraction=0.2)
    # The RMS residual should be small (the DNS is approximately logarithmic).
    assert fit['rms_residual'] < 0.5  # less than 0.5 wall units


# --------------------------------------------------------------------------- #
# Log-law indicator (local slope)
# --------------------------------------------------------------------------- #

def test_dns_log_law_indicator_shows_finite_re_effect(dns):
    """The local slope indicator Xi = y+ dU+/dy+ is not constant at Re_tau = 182.

    In a true logarithmic layer, Xi = 1/kappa = 2.44 everywhere.  At low
    Re_tau the outer-layer wake and the finite-domain factor (1 - y/h) make
    Xi vary across the log window.
    """
    indicator = log_law_indicator(dns['y_plus'], dns['u_plus'], re_tau=dns['re_tau'],
                                  low=30., high_fraction=0.2)
    # The effective kappa from the indicator should be below 0.41.
    assert indicator['kappa_local'] < KAPPA
    # The spread should be measurable (not a perfect log layer).
    assert indicator['xi_spread'] > 0.05
    assert indicator['cells'] >= 4


# --------------------------------------------------------------------------- #
# Full-range comparison (y+ > 30 to centreline)
# --------------------------------------------------------------------------- #

def test_dns_vs_classic_log_law_full_range(dns):
    """Over the full range y+ > 30, the DNS-vs-loglaw deviation is ~3.7 %.

    The wider window includes the outer/wake region where the wake effect
    pushes u+ above the log law.  At Re_tau = 182 the full-range RMS is
    slightly *larger* than the narrow log-layer RMS because the wake deviation
    dominates the narrow log-layer deviation.
    """
    y_plus = dns['y_plus']
    u_plus = dns['u_plus']

    # Full range: y+ in [30, max(y+)]
    mask_full = (y_plus >= 30.)
    ref_full = log_law(y_plus[mask_full])
    rel_full = (u_plus[mask_full] - ref_full) / ref_full
    rms_full = float(np.sqrt(np.mean(rel_full ** 2)))

    # The full-range deviation is around 3.7% (DNS is above log law in wake).
    assert 0.03 < rms_full < 0.05
    # The mean is positive (DNS above log law).
    assert np.mean(rel_full) > 0
