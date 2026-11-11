"""Stefan problem comprehensive verification with convergence analysis.

This module provides a rigorous verification of the Stefan problem solver
using grid convergence study, Richardson extrapolation, and comparison with
the analytical similarity solution.

Physical problem
----------------
One-phase Stefan solidification: semi-infinite domain x > 0, wall at x=0
held at T_w < T_f, initial liquid at T_f. The interface s(t) moves according
to the energy balance:

    ρ L ds/dt = k ∂T/∂x|_{x=s}

Similarity solution (Neumann 1860; Carslaw & Jaeger §11.2):
    s(t) = 2 λ √(α t)
    √π λ exp(λ²) erf(λ) = St = c_s (T_f - T_w) / L

Numerical method
----------------
Explicit enthalpy-method FVM with exact enthalpy inversion (no mushy band).
Conservative cell-centered finite volumes, explicit time integration.

Verification approach
---------------------
1. Grid convergence study with systematic refinement
2. Error norms (L1, L2, L∞) vs analytical solution
3. Convergence order estimation
4. Energy conservation check
5. Comparison with existing test gates (J4, J5)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.special import erf

from tensorfvm.two_phase.stefan import (
    StefanConfig,
    neumann_front,
    neumann_temperature,
    solve_stefan_1d,
    stefan_lambda,
)


def compute_error_norms(
    numerical: np.ndarray,
    reference: np.ndarray,
    norm_type: Literal['L1', 'L2', 'Linf'] = 'L2'
) -> float:
    """Compute error norm between numerical and reference solutions.
    
    Parameters
    ----------
    numerical : np.ndarray
        Numerical solution.
    reference : np.ndarray
        Reference (analytical) solution.
    norm_type : {'L1', 'L2', 'Linf'}
        Type of norm to compute.
    
    Returns
    -------
    float
        Error norm value.
    """
    error = np.abs(numerical - reference)
    
    if norm_type == 'L1':
        return float(np.mean(error))
    elif norm_type == 'L2':
        return float(np.sqrt(np.mean(error**2)))
    elif norm_type == 'Linf':
        return float(np.max(error))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def estimate_convergence_order(
    errors: list[float],
    grid_sizes: list[int],
) -> list[float]:
    """Estimate convergence order between successive grid pairs.
    
    Parameters
    ----------
    errors : list[float]
        Error measures on each grid.
    grid_sizes : list[int]
        Number of cells on each grid.
    
    Returns
    -------
    list[float]
        Convergence order for each grid pair.
    """
    orders = []
    for i in range(1, len(errors)):
        r = grid_sizes[i] / grid_sizes[i-1]
        e1 = errors[i-1]
        e2 = errors[i]
        if e1 > 0 and e2 > 0:
            p = math.log(e1 / e2) / math.log(r)
            orders.append(p)
        else:
            orders.append(float('nan'))
    return orders


def richardson_extrapolation_robust(
    errors: list[float],
    grid_sizes: list[int],
) -> dict:
    """Robust Richardson extrapolation with multiple grid triplets.
    
    Uses all available triplets of successive grids and reports statistics.
    
    Parameters
    ----------
    errors : list[float]
        Error measures on each grid (finest last).
    grid_sizes : list[int]
        Number of cells on each grid.
    
    Returns
    -------
    dict
        Richardson extrapolation results with statistics.
    """
    if len(errors) < 3:
        return {'available': False}
    
    results = []
    
    # Use all triplets (coarse, mid, fine)
    for i in range(len(errors) - 2):
        e_coarse = errors[i]
        e_mid = errors[i+1]
        e_fine = errors[i+2]
        
        r1 = grid_sizes[i+1] / grid_sizes[i]
        r2 = grid_sizes[i+2] / grid_sizes[i+1]
        
        # Check for monotonic convergence
        if not (e_coarse > e_mid > e_fine > 0):
            continue
        
        # Observed order
        eps1 = e_coarse - e_mid
        eps2 = e_mid - e_fine
        
        if abs(eps2) < 1e-15 or eps1 / eps2 <= 0:
            continue
        
        p = math.log(abs(eps1 / eps2)) / math.log(r1)
        
        # Richardson extrapolation
        if r2**p - 1.0 > 1e-10:
            f_exact = e_fine + (e_fine - e_mid) / (r2**p - 1.0)
        else:
            continue
        
        results.append({
            'grids': (grid_sizes[i], grid_sizes[i+1], grid_sizes[i+2]),
            'order_p': p,
            'f_exact': f_exact,
        })
    
    if not results:
        return {'available': False}
    
    # Statistics
    orders = [r['order_p'] for r in results]
    f_exacts = [r['f_exact'] for r in results]
    
    return {
        'available': True,
        'n_triplets': len(results),
        'order_mean': float(np.mean(orders)),
        'order_std': float(np.std(orders)),
        'order_min': float(np.min(orders)),
        'order_max': float(np.max(orders)),
        'f_exact_mean': float(np.mean(f_exacts)),
        'f_exact_std': float(np.std(f_exacts)),
        'triplets': results,
    }


def run_stefan_verification(
    grids: list[int] = [50, 100, 200, 400, 800],
    t_end: float = 300.0,
) -> dict:
    """Run comprehensive Stefan problem verification.
    
    Parameters
    ----------
    grids : list[int]
        List of grid sizes to test.
    t_end : float
        End time for simulation.
    
    Returns
    -------
    dict
        Comprehensive verification results.
    """
    
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
        
        # Front position errors (relative)
        front_rel_error = np.abs(out['front'] - ref_front) / np.maximum(ref_front, 1e-12)
        
        # Temperature field errors (relative, inside solid region)
        temp_scale = cfg.t_fusion - cfg.t_wall
        solid_mask = out['x'] < 0.8 * out['front'][-1]
        temp_rel_error = np.abs(out['temperature'][solid_mask] - ref_temp[solid_mask]) / temp_scale
        
        results.append({
            'n': n,
            'dx': cfg.dx,
            'front_L1': compute_error_norms(out['front'], ref_front, 'L1'),
            'front_L2': compute_error_norms(out['front'], ref_front, 'L2'),
            'front_Linf': compute_error_norms(out['front'], ref_front, 'Linf'),
            'temp_L1': compute_error_norms(out['temperature'][solid_mask], ref_temp[solid_mask], 'L1'),
            'temp_L2': compute_error_norms(out['temperature'][solid_mask], ref_temp[solid_mask], 'L2'),
            'temp_Linf': compute_error_norms(out['temperature'][solid_mask], ref_temp[solid_mask], 'Linf'),
            'energy_residual': out['energy_ledger_residual'],
            'final_front': out['final_front'],
            'ref_final_front': float(ref_front[-1]),
            'stefan_number': cfg.stefan_number,
            'lambda_analytical': out['lambda'],
        })
    
    # Convergence analysis
    grid_sizes = [r['n'] for r in results]
    
    analysis = {
        'front_L1_orders': estimate_convergence_order([r['front_L1'] for r in results], grid_sizes),
        'front_L2_orders': estimate_convergence_order([r['front_L2'] for r in results], grid_sizes),
        'front_Linf_orders': estimate_convergence_order([r['front_Linf'] for r in results], grid_sizes),
        'temp_L1_orders': estimate_convergence_order([r['temp_L1'] for r in results], grid_sizes),
        'temp_L2_orders': estimate_convergence_order([r['temp_L2'] for r in results], grid_sizes),
        'temp_Linf_orders': estimate_convergence_order([r['temp_Linf'] for r in results], grid_sizes),
    }
    
    # Richardson extrapolation
    analysis['richardson_front_L2'] = richardson_extrapolation_robust(
        [r['front_L2'] for r in results], grid_sizes
    )
    analysis['richardson_front_Linf'] = richardson_extrapolation_robust(
        [r['front_Linf'] for r in results], grid_sizes
    )
    
    return {
        'results': results,
        'analysis': analysis,
        'config': {
            'grids': grids,
            't_end': t_end,
        },
    }


def generate_verification_report(verification: dict) -> dict:
    """Generate structured verification report.
    
    Parameters
    ----------
    verification : dict
        Output from run_stefan_verification.
    
    Returns
    -------
    dict
        Structured report suitable for JSON serialization.
    """
    
    results = verification['results']
    analysis = verification['analysis']
    
    # Finest grid results
    finest = results[-1]
    
    # Verification verdict
    verdict = {
        'front_accuracy_Linf': 'PASS' if finest['front_Linf'] < 0.03 else 'FAIL',
        'front_accuracy_L2': 'PASS' if finest['front_L2'] < 0.01 else 'FAIL',
        'energy_conservation': 'PASS' if finest['energy_residual'] < 0.01 else 'FAIL',
    }
    
    # Convergence order verdict
    if analysis['richardson_front_L2']['available']:
        p_mean = analysis['richardson_front_L2']['order_mean']
        verdict['convergence_order'] = 'PASS' if p_mean > 1.0 else 'MARGINAL'
    else:
        verdict['convergence_order'] = 'UNKNOWN'
    
    report = {
        'title': 'Stefan Problem Comprehensive Verification',
        'version': '1.0',
        'physical_problem': {
            'name': 'One-phase Stefan solidification',
            'description': 'Semi-infinite domain with wall cooling',
            'analytical_solution': 'Neumann similarity solution (1860)',
            'parameters': {
                'stefan_number': finest['stefan_number'],
                'lambda_analytical': finest['lambda_analytical'],
                't_end': verification['config']['t_end'],
            },
        },
        'numerical_method': {
            'scheme': 'Explicit enthalpy-method FVM',
            'interface_tracking': 'Exact enthalpy inversion',
            'conservation': 'Cell-integrated enthalpy balance',
        },
        'grid_convergence': {
            'grids': [r['n'] for r in results],
            'refinement_ratio': 2.0,
            'front_errors': {
                'L1': [r['front_L1'] for r in results],
                'L2': [r['front_L2'] for r in results],
                'Linf': [r['front_Linf'] for r in results],
            },
            'temp_errors': {
                'L1': [r['temp_L1'] for r in results],
                'L2': [r['temp_L2'] for r in results],
                'Linf': [r['temp_Linf'] for r in results],
            },
            'energy_residuals': [r['energy_residual'] for r in results],
        },
        'convergence_orders': {
            'front_L1': analysis['front_L1_orders'],
            'front_L2': analysis['front_L2_orders'],
            'front_Linf': analysis['front_Linf_orders'],
        },
        'richardson_extrapolation': {
            'front_L2': analysis['richardson_front_L2'],
            'front_Linf': analysis['richardson_front_Linf'],
        },
        'verification_verdict': verdict,
        'finest_grid_results': finest,
    }
    
    return report


def main():
    """Run verification and generate report."""
    
    print("=" * 80)
    print("Stefan Problem Comprehensive Verification")
    print("=" * 80)
    
    # Run verification
    print("\nRunning grid convergence study...")
    verification = run_stefan_verification()
    
    # Print results table
    print("\nGrid convergence results:")
    print("-" * 100)
    print(f"{'N':>6s} {'dx':>8s} {'front_L1':>12s} {'front_L2':>12s} {'front_Linf':>12s} {'temp_L2':>12s} {'energy_res':>12s}")
    print("-" * 100)
    for r in verification['results']:
        print(f"{r['n']:6d} {r['dx']:8.5f} {r['front_L1']:12.4e} {r['front_L2']:12.4e} {r['front_Linf']:12.4e} {r['temp_L2']:12.4e} {r['energy_residual']:12.4e}")
    
    # Print convergence orders
    print("\nConvergence orders (front position):")
    print("-" * 60)
    analysis = verification['analysis']
    for i, (L1, L2, Linf) in enumerate(zip(
        analysis['front_L1_orders'],
        analysis['front_L2_orders'],
        analysis['front_Linf_orders']
    )):
        grids = f"N={verification['results'][i]['n']}->{verification['results'][i+1]['n']}"
        print(f"  {grids:>20s}: L1={L1:.3f}, L2={L2:.3f}, Linf={Linf:.3f}")
    
    # Richardson extrapolation
    print("\nRichardson extrapolation (front L2 norm):")
    print("-" * 60)
    rx = analysis['richardson_front_L2']
    if rx['available']:
        print(f"  Number of triplets: {rx['n_triplets']}")
        print(f"  Order p (mean±std): {rx['order_mean']:.3f} ± {rx['order_std']:.3f}")
        print(f"  Order p range:      [{rx['order_min']:.3f}, {rx['order_max']:.3f}]")
        print(f"  f_exact (mean±std): {rx['f_exact_mean']:.4e} ± {rx['f_exact_std']:.4e}")
    else:
        print("  Not available (non-monotonic convergence)")
    
    # Generate report
    report = generate_verification_report(verification)
    
    # Save report
    output_dir = Path(__file__).parent / 'artifacts'
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / 'stefan_verification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    # Verification verdict
    print("\n" + "=" * 80)
    print("Verification Verdict:")
    print("=" * 80)
    for key, verdict in report['verification_verdict'].items():
        symbol = '✓' if verdict == 'PASS' else ('?' if verdict == 'MARGINAL' else '✗')
        print(f"  {symbol} {key}: {verdict}")
    
    # Summary
    finest = report['finest_grid_results']
    print("\n" + "=" * 80)
    print("Summary (finest grid N={}):".format(finest['n']))
    print("=" * 80)
    print(f"  Front position error (L∞):  {finest['front_Linf']:.4e} ({finest['front_Linf']*100:.3f}%)")
    print(f"  Front position error (L2):  {finest['front_L2']:.4e} ({finest['front_L2']*100:.3f}%)")
    print(f"  Temperature error (L2):     {finest['temp_L2']:.4e} ({finest['temp_L2']*100:.3f}%)")
    print(f"  Energy conservation:        {finest['energy_residual']:.4e}")
    
    print("\nConclusion:")
    all_pass = all(v == 'PASS' for v in report['verification_verdict'].values())
    if all_pass:
        print("  ✓ Stefan problem verification PASSED all gates")
    else:
        print("  ⚠ Stefan problem verification has marginal or failed gates")
    
    return report


if __name__ == '__main__':
    main()
