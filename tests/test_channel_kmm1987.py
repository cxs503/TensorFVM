"""Tests for KMM1987 channel-flow DNS validation.

Verifies that the SST turbulence model reproduces the KMM1987 DNS mean-velocity
profile within the 5 % tolerance in the log region (y+ > 30).
"""
import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.verification.kmm1987 import (
    load_kmm1987_reference,
    log_region_deviation,
    cf_deviation,
    bulk_velocity_deviation,
    validate_kmm1987,
    TOLERANCE_LOG_REGION,
)


class TestKMM1987Reference:
    """Tests for the reference data loading."""

    def test_load_reference(self):
        dns = load_kmm1987_reference()
        assert dns['re_tau'] == pytest.approx(182.088, rel=1e-3)
        assert dns['Cf'] > 0
        assert dns['U_bulk_plus'] > 0
        assert len(dns['y_plus']) == len(dns['u_plus'])
        assert dns['y_plus'][0] >= 0
        assert np.all(np.diff(dns['y_plus']) > 0)

    def test_reference_viscous_sublayer(self):
        """In the viscous sublayer U+ ≈ y+ (within 3 % for y+ < 3)."""
        dns = load_kmm1987_reference()
        # Focus on y+ < 3 where the linear law is most accurate
        mask = (dns['y_plus'] >= 1.0) & (dns['y_plus'] < 3)
        if np.count_nonzero(mask) > 0:
            err = np.abs(dns['u_plus'][mask] - dns['y_plus'][mask]) / dns['y_plus'][mask]
            assert np.max(err) < 0.03  # < 3 % error in deep viscous sublayer


class TestLogRegionDeviation:
    """Tests for the log-region deviation metric."""

    def test_perfect_match(self):
        dns = load_kmm1987_reference()
        dev = log_region_deviation(dns['y_plus'], dns['u_plus'], dns)
        assert dev['rms_relative'] < 1e-10

    def test_constant_offset(self):
        dns = load_kmm1987_reference()
        # 10 % offset everywhere
        u_plus = dns['u_plus'] * 1.1
        dev = log_region_deviation(dns['y_plus'], u_plus, dns)
        assert dev['rms_relative'] == pytest.approx(0.10, rel=1e-6)

    def test_too_few_cells(self):
        dns = load_kmm1987_reference()
        # Only 2 points in log region
        y_plus = np.array([30.0, 35.0])
        u_plus = np.array([14.0, 14.5])
        with pytest.raises(ValueError, match='fewer than three'):
            log_region_deviation(y_plus, u_plus, dns)


class TestSSTValidation:
    """Tests that SST passes the KMM1987 validation."""

    @pytest.mark.parametrize('n', [64, 128, 256])
    def test_sst_passes_log_region(self, n):
        """SST model should have < 5 % RMS error in y+ > 30 region."""
        result = solve_channel('sst', n, re_tau=180.)
        assert result['converged']

        validation = validate_kmm1987(
            result['y_plus'], result['u_plus'],
            result['Cf'], result['bulk_u_plus'],
        )
        assert validation['verdict'] == 'PASS', (
            f"SST n={n} failed: log RMS = {validation['log_region']['rms_relative']:.1%}"
        )
        assert validation['log_region']['rms_relative'] < TOLERANCE_LOG_REGION

    def test_sst_grid_convergence(self):
        """SST log-region error should be grid-independent (within factor of 2)."""
        coarse = solve_channel('sst', 64, re_tau=180.)
        fine = solve_channel('sst', 256, re_tau=180.)

        dns = load_kmm1987_reference()
        dev_coarse = log_region_deviation(coarse['y_plus'], coarse['u_plus'], dns)
        dev_fine = log_region_deviation(fine['y_plus'], fine['u_plus'], dns)

        # Both should pass, and the ratio should be reasonable
        assert dev_coarse['rms_relative'] < TOLERANCE_LOG_REGION
        assert dev_fine['rms_relative'] < TOLERANCE_LOG_REGION
        # Finer grid may have slightly higher RMS due to more points in the window
        assert dev_fine['rms_relative'] < dev_coarse['rms_relative'] * 2


class TestOtherModels:
    """Tests for k-omega and k-epsilon (expected to have larger errors)."""

    def test_komega_log_region_error(self):
        """k-omega has known bias in the log region."""
        result = solve_channel('komega', 128, re_tau=180.)
        assert result['converged']

        dns = load_kmm1987_reference()
        dev = log_region_deviation(result['y_plus'], result['u_plus'], dns)
        # k-omega typically has ~10 % error in the log region
        assert dev['rms_relative'] > 0.05  # Fails the 5 % criterion

    def test_kepsilon_log_region_error(self):
        """k-epsilon with wall functions has known bias in the log region."""
        result = solve_channel('kepsilon', 128, re_tau=180., first_cell_y_plus=30.)
        assert result['converged']

        dns = load_kmm1987_reference()
        dev = log_region_deviation(result['y_plus'], result['u_plus'], dns)
        # k-epsilon typically has ~7 % error in the log region
        assert dev['rms_relative'] > 0.05  # Fails the 5 % criterion
