"""Sod shock tube (Sod 1978) verification of the compressible Euler solver.

Anchor: REPRO-FVM-SOD1978
Reference: Sod, G. A. (1978). "A Survey of Several Finite Difference Methods
for Systems of Nonlinear Hyperbolic Conservation Laws." J. Comput. Phys.
27:1-31. doi:10.1016/0021-9991(78)90023-2

Acceptance criterion (from feature_ledger):
  L1 relative error of (rho, u, p) vs exact Riemann solution < 3% at
  nx=400, second-order MUSCL-Hancock + HLLC, t_end=0.2.

The exact solution is computed by the package's own ExactRiemann solver
(no tabulated numbers are transcribed).
"""
import numpy as np
import pytest

from tensorfvm.compressible import cases


@pytest.fixture(scope="module")
def sod_result():
    """Run Sod shock tube once for all tests."""
    solver, cfg = cases.sod()
    solver.run(t_end=0.2, max_steps=5000)
    return solver, cfg


def test_sod_completes(sod_result):
    solver, cfg = sod_result
    assert solver.step > 0, "Solver did not advance"


def test_sod_density_l1_error(sod_result):
    solver, cfg = sod_result
    rho = solver.rho.cpu().numpy()
    j = solver.ny // 2
    rho_1d = rho[:, j]
    xc = cfg['xc'].cpu().numpy()
    exact = cfg['exact']
    exact_state = exact.solve(xc, 0.2)
    l1_rel = float(np.mean(np.abs(rho_1d - exact_state['rho'])) / np.mean(exact_state['rho']))
    assert l1_rel < 0.03, f"Sod density L1 error {l1_rel:.4f} exceeds 3%"


def test_sod_velocity_l1_error(sod_result):
    solver, cfg = sod_result
    u = solver.u.cpu().numpy()
    j = solver.ny // 2
    u_1d = u[:, j]
    xc = cfg['xc'].cpu().numpy()
    exact = cfg['exact']
    exact_state = exact.solve(xc, 0.2)
    denom = np.mean(np.abs(exact_state['u'])) + 1e-10
    l1_rel = float(np.mean(np.abs(u_1d - exact_state['u'])) / denom)
    assert l1_rel < 0.03, f"Sod velocity L1 error {l1_rel:.4f} exceeds 3%"


def test_sod_pressure_l1_error(sod_result):
    solver, cfg = sod_result
    p = solver.p.cpu().numpy()
    j = solver.ny // 2
    p_1d = p[:, j]
    xc = cfg['xc'].cpu().numpy()
    exact = cfg['exact']
    exact_state = exact.solve(xc, 0.2)
    l1_rel = float(np.mean(np.abs(p_1d - exact_state['p'])) / np.mean(exact_state['p']))
    assert l1_rel < 0.03, f"Sod pressure L1 error {l1_rel:.4f} exceeds 3%"


def test_sod_no_nan_or_inf(sod_result):
    solver, cfg = sod_result
    assert np.all(np.isfinite(solver.rho.cpu().numpy()))
    assert np.all(np.isfinite(solver.u.cpu().numpy()))
    assert np.all(np.isfinite(solver.v.cpu().numpy()))
    assert np.all(np.isfinite(solver.p.cpu().numpy()))


def test_sod_second_order_better_than_first():
    """Second-order MUSCL should be more accurate than first-order."""
    solver2, cfg2 = cases.sod()
    solver2.run(t_end=0.2, max_steps=5000)

    solver1, cfg1 = cases.sod()
    # Override limiter to first-order (minmod with zero slope)
    solver1._limiter_name = 'first_order'
    solver1.run(t_end=0.2, max_steps=5000)

    j = solver2.ny // 2
    xc = cfg2['xc'].cpu().numpy()
    exact = cfg2['exact']
    exact_state = exact.solve(xc, 0.2)

    rho2 = solver2.rho.cpu().numpy()[:, j]
    rho1 = solver1.rho.cpu().numpy()[:, j]
    err2 = np.mean(np.abs(rho2 - exact_state['rho']))
    err1 = np.mean(np.abs(rho1 - exact_state['rho']))
    assert err2 < err1, f"Second order error {err2:.5e} should be < first order {err1:.5e}"
