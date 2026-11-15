"""SST-2003 ZPG flat plate benchmark bound to anchor FEAT-TURB-SST-GATE.

Tests the SST-2003 closure against empirical correlations for the
zero-pressure-gradient flat plate.

Anchor: FEAT-TURB-SST-GATE

**STATUS: HONEST FAIL** - The flat plate benchmark requires a proper 2-D
boundary-layer solver with streamwise development. The current implementation
uses the SST channel flow solver as a proxy, which is not physically equivalent
to a flat plate boundary layer.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.benchmark_sst_flat_plate import (
    schultz_grunow_cf,
    solve_flat_plate_sst,
    ANCHOR_SLUG,
    RE_L,
)


@pytest.fixture(scope='module')
def sst_flat_plate_cases():
    """Run SST flat plate solves at three resolutions."""
    cases = {}
    for n in [64, 128, 256]:
        t0 = time.monotonic()
        result = solve_flat_plate_sst(n, re_l=RE_L)
        elapsed = time.monotonic() - t0
        cases[n] = dict(result=result, elapsed=elapsed)
    return cases


class TestSSTFlatPlateConvergence:
    """Test that SST flat plate solves converge at all resolutions."""

    def test_n64_converges(self, sst_flat_plate_cases):
        case = sst_flat_plate_cases[64]
        assert case['result']['converged'], f"n=64 did not converge"

    def test_n128_converges(self, sst_flat_plate_cases):
        case = sst_flat_plate_cases[128]
        assert case['result']['converged'], f"n=128 did not converge"

    def test_n256_converges(self, sst_flat_plate_cases):
        case = sst_flat_plate_cases[256]
        assert case['result']['converged'], f"n=256 did not converge"


class TestSSTFlatPlateWallResolution:
    """Test first-cell y+ sensitivity."""

    def test_first_cell_yplus_n64(self, sst_flat_plate_cases):
        yp = sst_flat_plate_cases[64]['result']['first_cell_y_plus']
        assert yp < 1.0, f"n=64 first cell y+ = {yp:.3f} >= 1.0"

    def test_first_cell_yplus_n128(self, sst_flat_plate_cases):
        yp = sst_flat_plate_cases[128]['result']['first_cell_y_plus']
        assert yp < 1.0, f"n=128 first cell y+ = {yp:.3f} >= 1.0"

    def test_first_cell_yplus_n256(self, sst_flat_plate_cases):
        yp = sst_flat_plate_cases[256]['result']['first_cell_y_plus']
        assert yp < 1.0, f"n=256 first cell y+ = {yp:.3f} >= 1.0"

    def test_yplus_refinement_monotonic(self, sst_flat_plate_cases):
        """First cell y+ should decrease with grid refinement."""
        yp64 = sst_flat_plate_cases[64]['result']['first_cell_y_plus']
        yp128 = sst_flat_plate_cases[128]['result']['first_cell_y_plus']
        yp256 = sst_flat_plate_cases[256]['result']['first_cell_y_plus']
        assert yp64 > yp128 > yp256, "First cell y+ should decrease with refinement"


class TestSSTFlatPlateProxyLimitation:
    """Document the proxy limitation honestly."""

    def test_channel_cf_much_higher_than_flat_plate(self, sst_flat_plate_cases):
        """The channel flow Cf is ~3.5x the flat plate Cf.

        This is expected: channel flow and flat plate are fundamentally
        different flows. The SST closure is validated against DNS in the
        channel flow benchmark; the flat plate benchmark failure is due to
        the proxy design, not the closure.
        """
        cf_ref = schultz_grunow_cf(RE_L)
        for n in [64, 128, 256]:
            result = sst_flat_plate_cases[n]['result']
            cf_channel = result['cf_channel']
            ratio = cf_channel / cf_ref
            # Document the ratio - it should be around 3.5x
            assert 2.0 < ratio < 5.0, f"n={n} Cf ratio = {ratio:.1f}x (expected ~3.5x)"

    def test_overall_verdict_is_fail(self, sst_flat_plate_cases):
        """The flat plate benchmark should honestly report FAIL."""
        for n in [64, 128, 256]:
            result = sst_flat_plate_cases[n]['result']
            assert result['overall_verdict'] == 'FAIL', \
                f"n={n} should report FAIL due to proxy limitation"

    def test_fail_reason_documented(self, sst_flat_plate_cases):
        """The fail reason should explain the proxy limitation."""
        for n in [64, 128, 256]:
            result = sst_flat_plate_cases[n]['result']
            assert 'proxy' in result.get('fail_reason', '').lower(), \
                f"n={n} fail reason should mention proxy limitation"


class TestSSTFlatPlateAnchorBinding:
    """Test anchor binding and artifact generation."""

    def test_anchor_slug_declared(self):
        assert ANCHOR_SLUG == "FEAT-TURB-SST-GATE"

    def test_benchmark_artifacts_generated(self, tmp_path):
        """Run the benchmark and verify artifacts."""
        from tensorfvm.benchmark_sst_flat_plate import run_benchmark
        summary = run_benchmark(tmp_path, resolutions=[64])

        assert (tmp_path / "benchmark.json").exists()
        assert (tmp_path / "profiles.csv").exists()
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "REPORT.md").exists()
        assert (tmp_path / "manifest.json").exists()

        # Verify anchor slug in benchmark.json
        with open(tmp_path / "benchmark.json") as f:
            data = json.load(f)
        assert data['anchor_slug'] == ANCHOR_SLUG
        assert data['verdict'] == 'FAIL'


class TestSSTFlatPlateSchultzGrunow:
    """Test the Schultz-Grunow correlation itself."""

    def test_correlation_valid_at_re_l(self):
        """Schultz-Grunow correlation is valid for Re_x > 1e6."""
        assert RE_L > 1e6, f"Re_L = {RE_L} is below the valid range"
        cf = schultz_grunow_cf(RE_L)
        assert 0.001 < cf < 0.01, f"Cf = {cf} is outside expected range"

    def test_correlation_decreases_with_re(self):
        """Cf should decrease with increasing Reynolds number."""
        cf1 = schultz_grunow_cf(1e6)
        cf2 = schultz_grunow_cf(1e7)
        cf3 = schultz_grunow_cf(1e8)
        assert cf1 > cf2 > cf3, "Cf should decrease with Re"
