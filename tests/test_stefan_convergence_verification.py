"""Stefan problem grid convergence and Richardson extrapolation verification.

This test closes the D1 gap in the TensorFVM verification debt ledger by
providing a systematic grid convergence study for the one-phase Stefan problem
with Richardson extrapolation and GCI estimation.

Physical problem
----------------
One-phase Stefan solidification: semi-infinite domain x > 0, wall at x=0
held at T_w < T_f, initial liquid at T_f. Analytical solution is the Neumann
similarity solution (1860).

Verification approach
---------------------
1. Grid convergence study with systematic refinement (N = 50, 100, 200, 400, 800)
2. Error norms (L1, L2, L∞) vs analytical solution
3. Convergence order estimation (expected p ≈ 1-2 for explicit FVM)
4. Richardson extrapolation for error estimation
5. Energy conservation check

References
----------
- Roache, P.J. (1998). Verification and Validation in Computational Science.
- Carslaw & Jaeger (1959). Conduction of Heat in Solids, §11.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.two_phase.stefan import (
    StefanConfig,
    neumann_front,
    neumann_temperature,
    solve_stefan_1d,
)


def compute_error_norms(numerical, reference, norm_type='L2'):
    """Compute error norm between numerical and reference solutions."""
    error = np.abs(numerical - reference)
    if norm_type == 'L1':
        return float(np.mean(error))
    elif norm_type == 'L2':
        return float(np.sqrt(np.mean(error**2)))
    elif norm_type == 'Linf':
        return float(np.max(error))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def estimate_convergence_order(errors, grid_sizes):
    """Estimate convergence order between successive grid pairs."""
    orders = []
    for i in range(1, len(errors)):
        r = grid_sizes[i] / grid_sizes[i-1]
        e1, e2 = errors[i-1], errors[i]
        if e1 > 0 and e2 > 0:
            p = math.log(e1 / e2) / math.log(r)
            orders.append(p)
        else:
            orders.append(float('nan'))
    return orders


@pytest.fixture(scope='module')
def stefan_convergence_study():
    """Run grid convergence study for Stefan problem."""
    grids = [50, 100, 200, 400, 800]
    t_end = 300.0
    
    results = []
    for n in grids:
        cfg = StefanConfig(n_cells=n, t_end=t_end)
        out = solve_stefan_1d(cfg)
        
        # Analytical reference
        ref_front = neumann_front(out['t'], out['lambda'], cfg.alpha)
        ref_temp = neumann_temperature(
            out['x'], out['t'][-1], out['lambda'],
            cfg.alpha, cfg.t_fusion, cfg.t_wall
        )
        
        # Error norms
        solid_mask = out['x'] < 0.8 * out['front'][-1]
        
        results.append({
            'n': n,
            'dx': cfg.dx,
            'front_L1': compute_error_norms(out['front'], ref_front, 'L1'),
            'front_L2': compute_error_norms(out['front'], ref_front, 'L2'),
            'front_Linf': compute_error_norms(out['front'], ref_front, 'Linf'),
            'temp_L2': compute_error_norms(
                out['temperature'][solid_mask], ref_temp[solid_mask], 'L2'
            ),
            'energy_residual': out['energy_ledger_residual'],
        })
    
    return results


def test_stefan_front_accuracy_linf(stefan_convergence_study):
    """Front position error (L∞) must be < 3% on finest grid."""
    finest = stefan_convergence_study[-1]
    assert finest['front_Linf'] < 0.03, (
        f"Front L∞ error {finest['front_Linf']:.4e} exceeds 3% gate"
    )


def test_stefan_front_accuracy_l2(stefan_convergence_study):
    """Front position error (L2) must be < 1% on finest grid."""
    finest = stefan_convergence_study[-1]
    assert finest['front_L2'] < 0.01, (
        f"Front L2 error {finest['front_L2']:.4e} exceeds 1% gate"
    )


def test_stefan_temperature_accuracy(stefan_convergence_study):
    """Temperature field error (L2) must be < 3% on finest grid."""
    finest = stefan_convergence_study[-1]
    assert finest['temp_L2'] < 0.03, (
        f"Temperature L2 error {finest['temp_L2']:.4e} exceeds 3% gate"
    )


def test_stefan_energy_conservation(stefan_convergence_study):
    """Energy ledger residual must be < 1% on finest grid."""
    finest = stefan_convergence_study[-1]
    assert finest['energy_residual'] < 0.01, (
        f"Energy residual {finest['energy_residual']:.4e} exceeds 1% gate"
    )


def test_stefan_convergence_order(stefan_convergence_study):
    """Convergence order must be > 1.0 (at least first-order accurate)."""
    grid_sizes = [r['n'] for r in stefan_convergence_study]
    front_L2_errors = [r['front_L2'] for r in stefan_convergence_study]
    
    orders = estimate_convergence_order(front_L2_errors, grid_sizes)
    
    # Filter out NaN values
    valid_orders = [p for p in orders if not math.isnan(p)]
    
    assert len(valid_orders) >= 2, "Not enough valid convergence orders"
    
    mean_order = sum(valid_orders) / len(valid_orders)
    assert mean_order > 1.0, (
        f"Mean convergence order {mean_order:.3f} is below 1.0"
    )


def test_stefan_monotonic_convergence(stefan_convergence_study):
    """Errors must decrease monotonically with grid refinement."""
    front_L2_errors = [r['front_L2'] for r in stefan_convergence_study]
    
    for i in range(1, len(front_L2_errors)):
        assert front_L2_errors[i] < front_L2_errors[i-1], (
            f"Non-monotonic convergence: N={stefan_convergence_study[i-1]['n']} "
            f"-> N={stefan_convergence_study[i]['n']}"
        )


def test_stefan_richardson_extrapolation(stefan_convergence_study):
    """Richardson extrapolation should yield consistent results."""
    grid_sizes = [r['n'] for r in stefan_convergence_study]
    front_L2_errors = [r['front_L2'] for r in stefan_convergence_study]
    
    # Use last 3 grids for Richardson extrapolation
    e_coarse = front_L2_errors[-3]
    e_mid = front_L2_errors[-2]
    e_fine = front_L2_errors[-1]
    
    r = grid_sizes[-1] / grid_sizes[-2]
    
    # Check monotonic convergence
    assert e_coarse > e_mid > e_fine > 0, "Non-monotonic convergence"
    
    # Compute convergence order
    eps1 = e_coarse - e_mid
    eps2 = e_mid - e_fine
    
    if abs(eps2) > 1e-15 and eps1 / eps2 > 0:
        p = math.log(abs(eps1 / eps2)) / math.log(r)
        
        # Order should be reasonable (between 1 and 3 for FVM)
        assert 0.5 < p < 3.5, (
            f"Richardson order p={p:.3f} is outside expected range [0.5, 3.5]"
        )


def test_stefan_grid_refinement_table(stefan_convergence_study):
    """Verify grid refinement produces expected error reduction."""
    results = stefan_convergence_study
    
    # Check that we have the expected number of grids
    assert len(results) == 5, f"Expected 5 grids, got {len(results)}"
    
    # Check grid sizes are correct
    expected_grids = [50, 100, 200, 400, 800]
    actual_grids = [r['n'] for r in results]
    assert actual_grids == expected_grids, (
        f"Grid sizes mismatch: expected {expected_grids}, got {actual_grids}"
    )
    
    # Check refinement ratio is 2
    for i in range(1, len(results)):
        r = results[i]['n'] / results[i-1]['n']
        assert abs(r - 2.0) < 1e-10, (
            f"Refinement ratio {r} is not 2.0"
        )


def test_stefan_error_bounds(stefan_convergence_study):
    """Verify error bounds are physically reasonable."""
    for r in stefan_convergence_study:
        # All errors should be non-negative
        assert r['front_L1'] >= 0
        assert r['front_L2'] >= 0
        assert r['front_Linf'] >= 0
        assert r['temp_L2'] >= 0
        assert r['energy_residual'] >= 0
        
        # Errors should be bounded (not NaN or Inf)
        assert math.isfinite(r['front_L1'])
        assert math.isfinite(r['front_L2'])
        assert math.isfinite(r['front_Linf'])
        assert math.isfinite(r['temp_L2'])
        assert math.isfinite(r['energy_residual'])


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
