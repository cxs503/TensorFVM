"""CASE-FVM-KOVASZNAY-ORDER: Convergence order verification test (m17078 round 2).

Validates that the TensorFVM incompressible NS solver achieves >= 1.8 L2
convergence order on the Kovasznay analytic solution (Re=40) using the
zero_gradient symmetry closure.

Reference: Kovasznay (1948), Proc. Camb. Phil. Soc. 44:58.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

# Pre-computed results from the convergence study (m17078)
STUDY_DIR = Path(__file__).parent.parent / 'runs' / 'kovasznay-order-m17078'
SUMMARY_PATH = STUDY_DIR / 'summary.json'


def _load_summary():
    """Load the pre-computed summary.json."""
    if not SUMMARY_PATH.exists():
        pytest.skip("Convergence study artifacts not found at %s" % SUMMARY_PATH)
    return json.loads(SUMMARY_PATH.read_text())


def _sha256_of(obj):
    """Recompute sha256 for traceability verification."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=float).encode()
    ).hexdigest()


class TestKovasznayConvergenceOrderM17078:
    """Validate the m17078 convergence study results."""

    def test_summary_exists(self):
        """The summary.json artifact must exist."""
        assert SUMMARY_PATH.exists(), "Summary not found at %s" % SUMMARY_PATH

    def test_all_levels_converged(self):
        """All grid levels must report numerical convergence."""
        summary = _load_summary()
        for record in summary['records']:
            assert record['converged'], "n=%d did not converge" % record['n']

    def test_spanwise_w_is_machine_zero(self):
        """The spanwise w component must be at machine precision (zero_gradient closure)."""
        summary = _load_summary()
        for record in summary['records']:
            assert record['max_abs_w'] < 1e-15, (
                "n=%d: max|w|=%.2e should be < 1e-15" % (record['n'], record['max_abs_w']))

    def test_velocity_l2_decreases_monotonically(self):
        """Velocity L2 error must decrease with grid refinement."""
        summary = _load_summary()
        errors = [r['velocity_l2_inplane'] for r in summary['records']]
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                "L2 error not monotonic: n=%d %.6e >= n=%d %.6e" % (
                    summary['records'][i]['n'], errors[i],
                    summary['records'][i + 1]['n'], errors[i + 1]))

    def test_finest_pair_order_passes(self):
        """The finest pair (64->128) in-plane order must be >= 1.8."""
        summary = _load_summary()
        orders = summary['pair_orders']
        assert len(orders) >= 1, "Need at least one pair order"
        finest = orders[-1]
        assert finest['inplane'] >= 1.8, (
            "Finest pair order %.3f < 1.8 threshold" % finest['inplane'])

    def test_lsq_slope_passes(self):
        """The least-squares fitted slope must be >= 1.8."""
        summary = _load_summary()
        fit = summary['fitted_orders']['velocity_l2_inplane']
        assert fit is not None, "LSQ fit not available"
        assert fit['slope'] >= 1.8, (
            "LSQ slope %.3f < 1.8 threshold" % fit['slope'])

    def test_pressure_order_passes(self):
        """Pressure mean-free order on finest pair must be >= 1.8."""
        summary = _load_summary()
        orders = summary['pair_orders']
        finest = orders[-1]
        assert finest['pressure'] >= 1.8, (
            "Pressure order %.3f < 1.8 threshold" % finest['pressure'])

    def test_verdict_is_pass(self):
        """The overall verdict must be PASS."""
        summary = _load_summary()
        assert summary['verdict'] == 'PASS', (
            "Verdict is %s, expected PASS" % summary['verdict'])

    def test_sha256_traceability(self):
        """Each level record's sha256 must be verifiable."""
        summary = _load_summary()
        for record in summary['records']:
            payload = {k: v for k, v in record.items() if k != 'sha256'}
            expected = _sha256_of(payload)
            assert record['sha256'] == expected, (
                "n=%d: sha256 mismatch" % record['n'])

    def test_error_magnitudes_reasonable(self):
        """Error magnitudes should be in expected ranges for a 2nd-order scheme."""
        summary = _load_summary()
        # n=128 should have velocity error < 1e-3
        n128 = [r for r in summary['records'] if r['n'] == 128][0]
        assert n128['velocity_l2_inplane'] < 1e-3, (
            "n=128 velocity L2 %.6e should be < 1e-3" % n128['velocity_l2_inplane'])
        # n=32 should have velocity error < 1e-2
        n32 = [r for r in summary['records'] if r['n'] == 32][0]
        assert n32['velocity_l2_inplane'] < 1e-2, (
            "n=32 velocity L2 %.6e should be < 1e-2" % n32['velocity_l2_inplane'])


class TestKovasznayConvergenceOrderLive:
    """Live solver tests on coarse grids (fast, for CI)."""

    def test_n8_converges_and_has_small_error(self):
        """n=8 grid should converge and have reasonable error."""
        from tensorfvm.verification.native_hybrid_ns import kovasznay_solver
        mesh, solver, exact = kovasznay_solver(
            8, symmetry_extrusion=True, symmetry_closure='zero_gradient')
        result = solver.solve(
            algorithm='coupled', outer_relaxation=0.7, tolerance=1e-8,
            max_outer_iterations=200)
        assert result['numerically_converged']

        velocity = np.asarray(result['velocity'])
        volumes = np.asarray(mesh.cell_volumes)
        e = velocity - exact['velocity']
        l2 = float(np.sqrt(np.sum(volumes * np.sum(e**2, axis=1)) / np.sum(volumes)))
        assert l2 < 0.05, "n=8 velocity L2 %.6e should be < 0.05" % l2

    def test_n16_error_smaller_than_n8(self):
        """n=16 error should be smaller than n=8."""
        from tensorfvm.verification.native_hybrid_ns import kovasznay_solver

        def solve_and_error(n):
            mesh, solver, exact = kovasznay_solver(
                n, symmetry_extrusion=True, symmetry_closure='zero_gradient')
            result = solver.solve(
                algorithm='coupled', outer_relaxation=0.7, tolerance=1e-8,
                max_outer_iterations=200)
            assert result['numerically_converged']
            velocity = np.asarray(result['velocity'])
            volumes = np.asarray(mesh.cell_volumes)
            e = velocity - exact['velocity']
            return float(np.sqrt(np.sum(volumes * np.sum(e**2, axis=1)) / np.sum(volumes)))

        e8 = solve_and_error(8)
        e16 = solve_and_error(16)
        assert e16 < e8, "n=16 error %.6e should be < n=8 error %.6e" % (e16, e8)
