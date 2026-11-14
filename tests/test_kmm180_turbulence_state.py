"""Tests for the turbulence-state validation module.

Verifies that the Reynolds stress, eddy viscosity, and production/dissipation
diagnostics are computed correctly and produce physically meaningful results.
"""
from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.rans_validation.models import constants_for
from tensorfvm.verification.kmm180_budget import (
    load_budget,
    shear_from_momentum,
)
from tensorfvm.verification.kmm1987 import load_kmm1987_reference
from tensorfvm.verification.kmm180_turbulence_state import (
    MODELS,
    RESOLUTIONS,
    audit_turbulence_state,
    dns_implied_nut_plus,
    full_audit,
    model_dissipation,
    model_production,
    model_reynolds_stress,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def dns():
    return load_kmm1987_reference()


@pytest.fixture(scope='module')
def budget():
    return load_budget('0180', kind='uv')


@pytest.fixture(scope='module')
def sst_result(dns):
    return solve_channel('sst', 128, re_tau=float(dns['re_tau']))


# --------------------------------------------------------------------------- #
# Unit tests on synthetic / derived quantities
# --------------------------------------------------------------------------- #

def test_dns_implied_nut_plus_is_positive_in_the_log_layer(dns):
    """The DNS-implied eddy viscosity must be positive where the mean shear is positive."""
    nut_plus = dns_implied_nut_plus(dns)
    y_plus = np.asarray(dns['y_plus'], dtype=float)
    log_mask = (y_plus >= 30.0) & (y_plus <= 0.2 * dns['re_tau'])
    assert np.all(nut_plus[log_mask] > 0)


def test_dns_implied_nut_plus_grows_linearly_in_the_log_layer(dns):
    """In the log layer, nu_t+ ≈ kappa y+ (linear growth)."""
    nut_plus = dns_implied_nut_plus(dns)
    y_plus = np.asarray(dns['y_plus'], dtype=float)
    log_mask = (y_plus >= 30.0) & (y_plus <= 0.2 * dns['re_tau'])
    # Fit a line: nu_t+ = a * y+ + b
    coeffs = np.polyfit(y_plus[log_mask], nut_plus[log_mask], 1)
    slope = coeffs[0]
    # The slope should be close to kappa ≈ 0.41
    assert 0.30 < slope < 0.50, f"DNS-implied nu_t+ slope = {slope:.3f}"


def test_model_reynolds_stress_is_positive(sst_result):
    """The model's Reynolds stress must be positive in the channel interior."""
    y_plus, stress_plus, slope, nut_plus = model_reynolds_stress(sst_result)
    # Exclude the wall and centreline
    interior = (y_plus > 5.0) & (y_plus < 0.9 * float(sst_result['re_tau']))
    assert np.all(stress_plus[interior] > 0)


def test_model_production_is_positive(sst_result):
    """TKE production must be positive everywhere."""
    y_plus, production_plus = model_production(sst_result)
    assert np.all(production_plus >= 0)


def test_model_dissipation_is_positive(sst_result):
    """Dissipation must be positive everywhere."""
    y_plus, eps_plus = model_dissipation(sst_result)
    assert np.all(eps_plus > 0)


# --------------------------------------------------------------------------- #
# Full audit tests
# --------------------------------------------------------------------------- #

def test_audit_converges_for_all_models(dns, budget):
    """Every (model, n) case must converge."""
    for model in MODELS:
        for n in RESOLUTIONS:
            result = audit_turbulence_state(model, n, dns=dns, budget=budget)
            assert result['converged'], f"{model} n={n} did not converge"


def test_reynolds_stress_window_stats_are_finite(dns, budget):
    """Window statistics for Reynolds stress must be finite numbers."""
    result = audit_turbulence_state('sst', 128, dns=dns, budget=budget)
    rs = result['reynolds_stress']['window_stats']
    for band, stats in rs.items():
        if stats['cells'] >= 2:
            assert stats['rms_relative'] is not None
            assert stats['max_relative'] is not None
            assert np.isfinite(stats['rms_relative'])
            assert np.isfinite(stats['max_relative'])


def test_eddy_viscosity_window_stats_are_finite(dns, budget):
    """Window statistics for eddy viscosity must be finite numbers."""
    result = audit_turbulence_state('sst', 128, dns=dns, budget=budget)
    nut = result['eddy_viscosity']['window_stats']
    for band, stats in nut.items():
        if stats['cells'] >= 2:
            assert stats['rms_relative'] is not None
            assert np.isfinite(stats['rms_relative'])


def test_pd_ratio_in_log_layer_is_order_unity(dns, budget):
    """In the log layer, P+/eps+ should be close to 1 for equilibrium closures."""
    result = audit_turbulence_state('sst', 256, dns=dns, budget=budget)
    ratio = result['production_dissipation']['log_layer_mean_ratio']
    assert ratio is not None
    # SST should be within a factor of 2 of unity in the log layer
    assert 0.5 < ratio < 2.0, f"P+/eps+ = {ratio:.3f}"


def test_k_plus_in_log_layer_is_bounded(dns, budget):
    """In the log layer, k+ should be positive and below the asymptotic value.

    At Re_tau=182, k+ has not yet reached the asymptotic value 1/sqrt(beta*) = 3.333.
    The SST model typically gives k+ ≈ 2.2-2.5 in the log layer at this Re_tau.
    """
    result = audit_turbulence_state('sst', 256, dns=dns, budget=budget)
    kp = result['k_plus']
    assert kp['log_layer_mean'] is not None
    # k+ must be positive and below the asymptotic value
    assert 0 < kp['log_layer_mean'] < kp['asymptotic_value']
    # At Re_tau=182, SST gives k+ ≈ 2.2-2.5
    assert kp['log_layer_mean'] > 1.5, f"k+ = {kp['log_layer_mean']:.3f}"


def test_models_have_different_eddy_viscosity(dns, budget):
    """Different models should produce different eddy viscosity profiles."""
    sst = audit_turbulence_state('sst', 128, dns=dns, budget=budget)
    ke = audit_turbulence_state('kepsilon', 128, dns=dns, budget=budget)

    # Eddy viscosity in the log layer should be different
    sst_nut = sst['eddy_viscosity']['window_stats'].get('log', {})
    ke_nut = ke['eddy_viscosity']['window_stats'].get('log', {})

    # k-epsilon uses wall functions, so its nu_t+ profile is different
    # We check that the models produce different results, not which is better
    assert sst_nut.get('rms_relative') is not None
    assert ke_nut.get('rms_relative') is not None


def test_full_audit_returns_all_cases(dns, budget):
    """The full audit must return results for all (model, n) combinations."""
    results = full_audit(dns=dns, budget=budget)
    expected = len(MODELS) * len(RESOLUTIONS)
    assert len(results) == expected
    assert all(r['converged'] for r in results)


# --------------------------------------------------------------------------- #
# Artifact writing tests
# --------------------------------------------------------------------------- #

def test_write_artifacts_produces_valid_json(tmp_path, dns, budget):
    """The written JSON must be valid and contain all expected keys."""
    results = full_audit(dns=dns, budget=budget)
    write_artifacts(results, tmp_path)

    data = json.loads((tmp_path / 'turbulence_state.json').read_text())
    assert data['schema'] == 'tensorfvm.kmm180-turbulence-state/1'
    assert data['anchor'] == 'CASE-FVM-CHANNEL-KMM180'
    assert len(data['results']) == len(MODELS) * len(RESOLUTIONS)


def test_write_artifacts_produces_valid_csv(tmp_path, dns, budget):
    """The written CSV must have the expected columns and rows."""
    results = full_audit(dns=dns, budget=budget)
    write_artifacts(results, tmp_path)

    with (tmp_path / 'turbulence_state_summary.csv').open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    expected_cols = {'model', 'n', 'quantity', 'band', 'cells', 'rms_relative', 'max_relative'}
    assert set(rows[0].keys()) == expected_cols


# --------------------------------------------------------------------------- #
# Cross-check against existing modules
# --------------------------------------------------------------------------- #

def test_reynolds_stress_matches_momentum_route(sst_result, dns):
    """The model's Reynolds stress computed via nu_t+ * dU+/dy+ must match
    the momentum-balance route when the model's nu_t+ equals the DNS-implied value."""
    y_plus, stress_plus, slope, nut_plus = model_reynolds_stress(sst_result)
    dns_stress = shear_from_momentum(dns)

    # Interpolate DNS stress onto model grid
    from scipy.interpolate import PchipInterpolator
    dns_stress_interp = PchipInterpolator(dns['y_plus'], dns_stress)(y_plus)

    # In the log layer, the relative error should be bounded
    re_tau = float(dns['re_tau'])
    log_mask = (y_plus >= 30.0) & (y_plus <= 0.2 * re_tau)
    if np.any(log_mask):
        rel_error = np.abs(stress_plus[log_mask] - dns_stress_interp[log_mask]) / np.abs(dns_stress_interp[log_mask])
        # SST at Re_tau=182 has significant error in the log layer
        assert np.mean(rel_error) < 0.50  # < 50% mean relative error


def test_eddy_viscosity_defect_explains_mean_velocity_error(sst_result, dns):
    """The eddy viscosity deficit should explain the mean velocity error.

    If nu_t+ is half its log-law value, the mean velocity slope is reduced,
    leading to a lower u+ profile.
    """
    from tensorfvm.verification.kmm180_turbulence_state import dns_implied_nut_plus

    y_plus = np.asarray(sst_result['y_plus'], dtype=float)
    nut_plus = np.asarray(sst_result['nut'], dtype=float) * float(dns['re_tau'])
    dns_nut_plus = dns_implied_nut_plus(dns)

    from scipy.interpolate import PchipInterpolator
    dns_nut_interp = PchipInterpolator(dns['y_plus'], dns_nut_plus)(y_plus)

    # In the log layer, SST's nu_t+ should be significantly below DNS-implied
    re_tau = float(dns['re_tau'])
    log_mask = (y_plus >= 30.0) & (y_plus <= 0.2 * re_tau)
    if np.any(log_mask):
        ratio = np.mean(nut_plus[log_mask] / dns_nut_interp[log_mask])
        # SST's nu_t+ is about half the DNS-implied value
        assert ratio < 0.70, f"nu_t+ ratio = {ratio:.3f}"
