"""Convergence-order verification for CASE-FVM-KOVASZNAY-ORDER.

Acceptance criterion (feature_ledger):
    L2 convergence order >= 1.8 (nominal second order)
    log-log slope on 32/64/128 three grid sets

Reference: Kovasznay, L. I. G. (1948). Laminar flow behind a two-dimensional
grid. Proc. Camb. Phil. Soc. 44, 58. DOI: 10.1017/s0305004100023999

The analytic solution satisfies div(u)=0 and (u·∇)u + ∇p − μΔu = 0 exactly
with no body force, making it an ideal manufactured-solution benchmark for
Navier–Stokes solver verification.

Coarse-grid tests (4/8/16) run in CI; the full 32/64/128 study is executed
on the HPC cluster via ``scripts/run_kovasznay_32_64_128.py`` and its results
are consumed by ``test_kovasznay_full_convergence_order_from_cluster``.
"""

import json
import pathlib

import numpy as np
import pytest

from tensorfvm.verification.native_hybrid_ns import kovasznay_solver
from tensorfvm.native_manufactured3d import volume_weighted_error


def _observed_order(errors, refinement_ratio=2.0):
    """Compute observed convergence order from a sequence of errors."""
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0:
            orders.append(np.log(errors[i] / errors[i + 1]) / np.log(refinement_ratio))
    return orders


# Coarse grids for CI-speed execution
COARSE_NS = (4, 8, 16)


@pytest.mark.parametrize('convection_order', [1, 2])
def test_kovasznay_velocity_error_decreases_with_refinement(convection_order):
    """Velocity L2 error must decrease monotonically with grid refinement."""
    errors = []
    for n in COARSE_NS:
        mesh, solver, exact = kovasznay_solver(
            n, 40., symmetry_extrusion=True, convection_order=convection_order)
        result = solver.solve(
            algorithm='coupled', outer_relaxation=0.7, tolerance=1e-8,
            max_outer_iterations=80)
        assert result['numerically_converged'], f'n={n} did not converge'
        err = volume_weighted_error(result['velocity'], exact['velocity'], mesh.cell_volumes)
        errors.append(err)

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f'Error did not decrease: n={COARSE_NS[i]} err={errors[i]:.6e} '
            f'-> n={COARSE_NS[i+1]} err={errors[i+1]:.6e}')


def test_kovasznay_second_order_convergence_on_coarse_grids():
    """Second-order convection must show observed order approaching 2.

    On coarse grids (4/8/16), the asymptotic regime is not fully reached,
    so we require order > 1.5 as a conservative CI gate.  The full
    32/64/128 study (order >= 1.8) is validated separately.
    """
    errors = []
    for n in COARSE_NS:
        mesh, solver, exact = kovasznay_solver(
            n, 40., symmetry_extrusion=True, convection_order=2)
        result = solver.solve(
            algorithm='coupled', outer_relaxation=0.7, tolerance=1e-8,
            max_outer_iterations=80)
        assert result['numerically_converged']
        err = volume_weighted_error(result['velocity'], exact['velocity'], mesh.cell_volumes)
        errors.append(err)

    orders = _observed_order(errors)
    # On coarse grids, require at least 1.5 (asymptotic 2.0 not yet reached)
    assert orders[-1] > 1.5, (
        f'Observed order {orders[-1]:.3f} on coarse grids below 1.5 threshold; '
        f'errors={[f"{e:.6e}" for e in errors]}')


def test_kovasznay_pressure_error_decreases_with_refinement():
    """Pressure L2 error must also decrease with refinement."""
    errors = []
    for n in COARSE_NS:
        mesh, solver, exact = kovasznay_solver(
            n, 40., symmetry_extrusion=True, convection_order=2)
        result = solver.solve(
            algorithm='coupled', outer_relaxation=0.7, tolerance=1e-8,
            max_outer_iterations=80)
        err = volume_weighted_error(result['pressure'], exact['pressure'], mesh.cell_volumes)
        errors.append(err)

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f'Pressure error did not decrease: n={COARSE_NS[i]} err={errors[i]:.6e} '
            f'-> n={COARSE_NS[i+1]} err={errors[i+1]:.6e}')


@pytest.mark.slow
def test_kovasznay_32_grid_meets_3_percent_threshold():
    """At n=32, both velocity and pressure errors must be < 3%.

    Marked slow because n=32 takes >60s under load.
    """
    mesh, solver, exact = kovasznay_solver(
        32, 40., symmetry_extrusion=True, convection_order=2)
    result = solver.solve(
        algorithm='coupled', outer_relaxation=0.7, tolerance=1e-8,
        max_outer_iterations=120)
    assert result['numerically_converged']

    vel_err = volume_weighted_error(result['velocity'], exact['velocity'], mesh.cell_volumes)
    pres_err = volume_weighted_error(result['pressure'], exact['pressure'], mesh.cell_volumes)

    assert vel_err < 0.03, f'Velocity error {vel_err:.6f} exceeds 3% threshold'
    assert pres_err < 0.03, f'Pressure error {pres_err:.6f} exceeds 3% threshold'


def test_kovasznay_analytic_solution_satisfies_ns_equations():
    """The Kovasznay analytic solution must satisfy the continuous N-S equations."""
    points = np.random.default_rng(42).uniform(0.05, 0.95, (50, 3))
    re = 40.0
    from tensorfvm.verification.native_hybrid_ns import kovasznay_fields

    f = kovasznay_fields(points, re)

    # Verify divergence-free
    h = 1e-5
    grad_u = np.zeros((len(points), 3, 3))
    for j in range(3):
        plus = kovasznay_fields(points + np.eye(3)[j] * h, re)['velocity']
        minus = kovasznay_fields(points - np.eye(3)[j] * h, re)['velocity']
        grad_u[:, :, j] = (plus - minus) / (2 * h)

    div_u = np.trace(grad_u, axis1=1, axis2=2)
    assert np.max(np.abs(div_u)) < 1e-6, f'Divergence max: {np.max(np.abs(div_u))}'

    # Verify momentum balance: (u·∇)u + ∇p - (1/Re)Δu = 0
    lap_u = np.zeros_like(f['velocity'])
    for j in range(3):
        plus = kovasznay_fields(points + np.eye(3)[j] * h, re)['velocity']
        minus = kovasznay_fields(points - np.eye(3)[j] * h, re)['velocity']
        center = f['velocity']
        lap_u += (plus - 2 * center + minus) / h ** 2

    grad_p = np.zeros((len(points), 3))
    for j in range(3):
        plus = kovasznay_fields(points + np.eye(3)[j] * h, re)['pressure']
        minus = kovasznay_fields(points - np.eye(3)[j] * h, re)['pressure']
        grad_p[:, j] = (plus - minus) / (2 * h)

    conv = np.einsum('cij,cj->ci', grad_u, f['velocity'])
    residual = conv + grad_p - lap_u / re
    assert np.max(np.abs(residual)) < 1e-5, f'Momentum residual max: {np.max(np.abs(residual))}'


# --------------------------------------------------------------------------- #
# Full 32/64/128 convergence order test (reads cluster artifacts)
# --------------------------------------------------------------------------- #

CLUSTER_RUN_DIR = pathlib.Path(__file__).parents[1] / 'runs' / 'kovasznay-32-64-128'


def _load_cluster_results():
    """Load cases.json from each grid directory."""
    results = []
    for n in (32, 64, 128):
        cases_file = CLUSTER_RUN_DIR / f'n{n}' / 'cases.json'
        if not cases_file.exists():
            return None
        with open(cases_file) as f:
            cases = json.load(f)
        results.append(cases[0])
    return sorted(results, key=lambda r: r['n'])


def _least_squares_slope(rows, quantity):
    """Compute log-log slope via least squares."""
    if len(rows) < 2:
        return None
    x = np.log([row['h'] for row in rows])
    y = np.log([row[quantity] for row in rows])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


@pytest.mark.slow
def test_kovasznay_full_convergence_order_from_cluster():
    """CASE-FVM-KOVASZNAY-ORDER acceptance gate.

    Reads the 32/64/128 cluster run results and verifies:
    1. All three grids converged numerically
    2. Velocity L2 log-log slope >= 1.8 (nominal second order)
    3. Pressure L2 log-log slope >= 1.5 (pressure converges slower)

    This is the definitive acceptance test for the feature_ledger entry.
    """
    results = _load_cluster_results()
    if results is None:
        pytest.skip(
            f'Cluster run artifacts not found at {CLUSTER_RUN_DIR}; '
            'run scripts/run_kovasznay_32_64_128.py first')

    # Check all grids converged
    for r in results:
        assert r['numerically_converged'], (
            f"n={r['n']} did not converge: {r['final_residuals']}")

    # Velocity convergence order
    vel_slope = _least_squares_slope(results, 'velocity_l2')
    assert vel_slope is not None
    assert abs(vel_slope) >= 1.8, (
        f'Velocity L2 log-log slope = {abs(vel_slope):.3f}, '
        f'required >= 1.8; errors={[r["velocity_l2"] for r in results]}')

    # Pressure convergence order (more lenient threshold)
    pres_slope = _least_squares_slope(results, 'pressure_l2')
    assert pres_slope is not None
    assert abs(pres_slope) >= 1.5, (
        f'Pressure L2 log-log slope = {abs(pres_slope):.3f}, '
        f'required >= 1.5; errors={[r["pressure_l2"] for r in results]}')

    # Report the results for visibility
    print(f'\nKovasznay 32/64/128 convergence order:')
    for r in results:
        print(f"  n={r['n']}: vel_L2={r['velocity_l2']:.6e}, "
              f"pres_L2={r['pressure_l2']:.6e}")
    print(f'  Velocity slope: {abs(vel_slope):.3f} (threshold: 1.8)')
    print(f'  Pressure slope: {abs(pres_slope):.3f} (threshold: 1.5)')
