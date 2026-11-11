"""Stefan problem grid convergence study with Richardson extrapolation.

This script performs a systematic grid refinement study for the one-phase Stefan
problem (M4 in the two-phase chain) and quantifies the convergence order using
Richardson extrapolation and the Grid Convergence Index (GCI).

Physical problem
----------------
Semi-infinite solidification: wall at x=0 held at T_w < T_f, initial liquid at T_f.
Similarity solution (Neumann 1860; Carslaw & Jaeger §11.2):
  s(t) = 2 λ √(α t)
  √π λ exp(λ²) erf(λ) = St = c_s (T_f - T_w) / L

Numerical method
----------------
Explicit enthalpy-method FVM with exact enthalpy inversion (no mushy band).
Conservative cell-centered finite volumes, backward Euler time integration.

Verification metrics
--------------------
1. Front position error vs analytical solution
2. Temperature field error vs similarity profile
3. Energy ledger residual (conservation)
4. Convergence order p via Richardson extrapolation
5. GCI (Grid Convergence Index) for safety factor estimation

References
----------
- Roache, P.J. (1998). Verification and Validation in Computational Science and
  Engineering. Hermosa Publishers.
- Carslaw, H.S. & Jaeger, J.C. (1959). Conduction of Heat in Solids, 2nd ed.
  Oxford University Press.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erf

from tensorfvm.two_phase.stefan import (
    StefanConfig,
    neumann_front,
    neumann_temperature,
    solve_stefan_1d,
    stefan_lambda,
)


def richardson_extrapolation(fine, mid, coarse, r: float = 2.0):
    """Estimate converged value and convergence order from three grids.
    
    Parameters
    ----------
    fine, mid, coarse : float
        Error measures on three successively coarser grids.
    r : float
        Grid refinement ratio (default 2.0 for uniform doubling).
    
    Returns
    -------
    f_exact : float
        Richardson extrapolated value (estimate of exact solution).
    p : float
        Observed order of convergence.
    """
    if fine <= 0 or mid <= 0 or coarse <= 0:
        return np.nan, np.nan
    
    # Observed order: p = ln((f_coarse - f_mid) / (f_mid - f_fine)) / ln(r)
    eps_mid_coarse = mid - fine
    eps_fine_mid = coarse - mid
    
    if abs(eps_fine_mid) < 1e-15 or eps_mid_coarse / eps_fine_mid <= 0:
        return np.nan, np.nan
    
    p = math.log(abs(eps_mid_coarse / eps_fine_mid)) / math.log(r)
    
    # Richardson extrapolation: f_exact = f_fine + (f_fine - f_mid) / (r^p - 1)
    f_exact = fine + (fine - mid) / (r**p - 1.0)
    
    return f_exact, p


def compute_gci(fine, mid, r: float = 2.0, p: float = 2.0, Fs: float = 1.25):
    """Compute Grid Convergence Index (GCI).
    
    Parameters
    ----------
    fine, mid : float
        Error measures on two successive grids.
    r : float
        Grid refinement ratio.
    p : float
        Observed order of convergence.
    Fs : float
        Safety factor (1.25 for 3 grids, 3.0 for 2 grids per Roache 1998).
    
    Returns
    -------
    gci : float
        Grid Convergence Index (percentage).
    """
    if fine <= 0 or p <= 0:
        return np.nan
    
    eps = abs((mid - fine) / fine)
    gci = Fs * eps / (r**p - 1.0) * 100.0
    return gci


def run_convergence_study():
    """Run grid convergence study for Stefan problem."""
    
    # Grid sizes (uniform refinement)
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
        
        # Front position error
        front_err = np.abs(out['front'] - ref_front) / np.maximum(ref_front, 1e-12)
        front_max_err = float(np.max(front_err))
        front_mean_err = float(np.mean(front_err))
        
        # Temperature field error (inside solid region)
        temp_scale = cfg.t_fusion - cfg.t_wall
        solid_mask = out['x'] < 0.8 * out['front'][-1]
        temp_err = np.abs(out['temperature'][solid_mask] - ref_temp[solid_mask]) / temp_scale
        temp_max_err = float(np.max(temp_err))
        
        results.append({
            'n': n,
            'dx': cfg.dx,
            'front_max_err': front_max_err,
            'front_mean_err': front_mean_err,
            'temp_max_err': temp_max_err,
            'energy_residual': out['energy_ledger_residual'],
            'final_front': out['final_front'],
            'ref_final_front': float(ref_front[-1]),
            'stefan_number': cfg.stefan_number,
            'lambda_analytical': out['lambda'],
        })
    
    return results


def analyze_convergence(results: list[dict]) -> dict:
    """Analyze convergence order and GCI from grid study."""
    
    analysis = {
        'grids': [r['n'] for r in results],
        'front_max_errors': [r['front_max_err'] for r in results],
        'front_mean_errors': [r['front_mean_err'] for r in results],
        'temp_max_errors': [r['temp_max_err'] for r in results],
        'energy_residuals': [r['energy_residual'] for r in results],
    }
    
    # Richardson extrapolation for front position (last 3 grids)
    if len(results) >= 3:
        f_fine = results[-1]['front_max_err']
        f_mid = results[-2]['front_max_err']
        f_coarse = results[-3]['front_max_err']
        
        f_exact, p = richardson_extrapolation(f_fine, f_mid, f_coarse, r=2.0)
        gci = compute_gci(f_fine, f_mid, r=2.0, p=p, Fs=1.25)
        
        analysis['richardson_front'] = {
            'f_exact': f_exact,
            'order_p': p,
            'gci_percent': gci,
            'grids_used': [results[-3]['n'], results[-2]['n'], results[-1]['n']],
        }
    
    # Convergence order between successive grids
    orders = []
    for i in range(1, len(results)):
        r = results[i]['n'] / results[i-1]['n']
        e1 = results[i-1]['front_max_err']
        e2 = results[i]['front_max_err']
        if e1 > 0 and e2 > 0:
            p = math.log(e1 / e2) / math.log(r)
            orders.append({
                'grids': f"N={results[i-1]['n']}->{results[i]['n']}",
                'order_p': p,
            })
    
    analysis['convergence_orders'] = orders
    
    return analysis


def generate_report(results: list[dict], analysis: dict) -> dict:
    """Generate comprehensive verification report."""
    
    report = {
        'title': 'Stefan Problem Grid Convergence Study',
        'physical_problem': {
            'description': 'One-phase Stefan solidification problem',
            'geometry': 'Semi-infinite domain x > 0, wall at x=0',
            'boundary_conditions': 'T(0,t) = T_w < T_f, T(∞,t) = T_f',
            'initial_condition': 'Liquid at fusion temperature T_f',
            'parameters': {
                'stefan_number': results[0]['stefan_number'],
                'lambda_analytical': results[0]['lambda_analytical'],
                't_end': 300.0,
            },
            'analytical_solution': 's(t) = 2 λ √(α t), Neumann (1860)',
        },
        'numerical_method': {
            'scheme': 'Explicit enthalpy-method FVM',
            'time_integration': 'Backward Euler with Picard iteration',
            'interface_tracking': 'Exact enthalpy inversion (no mushy band)',
            'conservation': 'Cell-integrated enthalpy balance',
        },
        'grid_convergence': {
            'grids': analysis['grids'],
            'refinement_ratio': 2.0,
            'front_max_errors': analysis['front_max_errors'],
            'front_mean_errors': analysis['front_mean_errors'],
            'temp_max_errors': analysis['temp_max_errors'],
            'energy_residuals': analysis['energy_residuals'],
        },
        'richardson_extrapolation': analysis.get('richardson_front', {}),
        'convergence_orders': analysis.get('convergence_orders', []),
        'verification_verdict': {
            'front_accuracy': 'PASS' if results[-1]['front_max_err'] < 0.03 else 'FAIL',
            'energy_conservation': 'PASS' if results[-1]['energy_residual'] < 0.01 else 'FAIL',
            'convergence_order': 'PASS' if analysis.get('richardson_front', {}).get('order_p', 0) > 1.0 else 'MARGINAL',
        },
    }
    
    return report


def main():
    """Run convergence study and generate report."""
    
    print("=" * 70)
    print("Stefan Problem Grid Convergence Study")
    print("=" * 70)
    
    # Run convergence study
    print("\nRunning grid refinement study...")
    results = run_convergence_study()
    
    print("\nGrid convergence results:")
    print(f"{'N':>6s} {'dx':>8s} {'front_max_err':>14s} {'front_mean_err':>14s} {'temp_max_err':>14s} {'energy_res':>12s}")
    print("-" * 70)
    for r in results:
        print(f"{r['n']:6d} {r['dx']:8.5f} {r['front_max_err']:14.4e} {r['front_mean_err']:14.4e} {r['temp_max_err']:14.4e} {r['energy_residual']:12.4e}")
    
    # Analyze convergence
    print("\nAnalyzing convergence order...")
    analysis = analyze_convergence(results)
    
    if 'richardson_front' in analysis:
        rx = analysis['richardson_front']
        print(f"\nRichardson extrapolation (front position):")
        print(f"  Extrapolated value: {rx['f_exact']:.4e}")
        print(f"  Observed order p:   {rx['order_p']:.3f}")
        print(f"  GCI:                {rx['gci_percent']:.3f} %")
        print(f"  Grids used:         {rx['grids_used']}")
    
    print("\nConvergence orders between successive grids:")
    for co in analysis.get('convergence_orders', []):
        print(f"  {co['grids']}: p = {co['order_p']:.3f}")
    
    # Generate report
    report = generate_report(results, analysis)
    
    # Save report
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / 'stefan_convergence_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    # Verification verdict
    print("\n" + "=" * 70)
    print("Verification Verdict:")
    print("=" * 70)
    for key, verdict in report['verification_verdict'].items():
        print(f"  {key}: {verdict}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"  Finest grid (N={results[-1]['n']}):")
    print(f"    Front max error:     {results[-1]['front_max_err']:.4e} ({results[-1]['front_max_err']*100:.2f}%)")
    print(f"    Energy residual:     {results[-1]['energy_residual']:.4e}")
    if 'richardson_front' in analysis:
        print(f"    Convergence order:   {analysis['richardson_front']['order_p']:.3f}")
        print(f"    GCI:                 {analysis['richardson_front']['gci_percent']:.3f}%")
    
    print("\nConclusion:")
    if report['verification_verdict']['front_accuracy'] == 'PASS':
        print("  ✓ Stefan problem verification PASSED: front accuracy < 3%")
    else:
        print("  ✗ Stefan problem verification FAILED: front accuracy >= 3%")
    
    if report['verification_verdict']['energy_conservation'] == 'PASS':
        print("  ✓ Energy conservation PASSED: residual < 1%")
    else:
        print("  ✗ Energy conservation FAILED: residual >= 1%")
    
    return report


if __name__ == '__main__':
    main()
