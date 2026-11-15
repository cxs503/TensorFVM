"""Acceptance tests for the Chebyshev-collocation cavity reference solver.

These tests are the machine check behind ``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL.md``.
They are deliberately cheap (N <= 24, Re <= 100) so they run on the control
plane; the production ladder at Re = 1000 lives in
``scripts/run_cavity_spectral_ref.py`` and its artefacts in
``docs/CASE-FVM-CAVITY-RE1000/runs/spectral/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm import lid_cavity_reference as ref
from tensorfvm.cavity_spectral import (
    SpectralCavity,
    barycentric_matrix,
    differentiation_matrices,
    wall_second_derivative_functional,
)

DATA = Path("docs/CASE-FVM-CAVITY-RE1000/runs/spectral")


def test_derivative_matrices_are_exact_on_polynomials() -> None:
    n = 12
    xi, D1, D2 = differentiation_matrices(n)
    for degree in (3, 7, 11):
        a = np.zeros(degree + 3)
        a[degree] = 1.0
        f = np.polyval(a[::-1], xi)
        df = np.polyval(np.polyder(a[::-1]), xi)
        d2f = np.polyval(np.polyder(a[::-1], 2), xi)
        assert np.allclose(D1 @ f, df, atol=1e-9)
        assert np.allclose(D2 @ f, d2f, atol=1e-8)


def test_barycentric_interpolation_reproduces_polynomials() -> None:
    n = 16
    xi, _, _ = differentiation_matrices(n)
    t = np.linspace(-1.0, 1.0, 37)
    L = barycentric_matrix(t, xi)
    for degree in (1, 9, 15):
        a = np.zeros(degree + 1)
        a[degree] = 1.0
        assert np.allclose(L @ np.polyval(a[::-1], xi),
                           np.polyval(a[::-1], t), atol=1e-10)


def test_wall_functional_honours_the_no_slip_slope() -> None:
    """``psi_nn`` at a wall must come from data with the *imposed* wall slope.

    This is the property whose omission lets ``psi = omega = 0`` satisfy the
    whole system (see ``test_lid_drives_the_flow``).
    """
    n = 16
    xi, _, _ = differentiation_matrices(n)
    X = 0.5 * (1.0 - xi)
    interior = X[1:n]
    for degree in (2, 5, 9):
        a = np.zeros(degree + 1)
        a[degree] = 1.0
        p = lambda t: np.polyval(a[::-1], t)
        dp = lambda t: np.polyval(np.polyder(a[::-1]), t)
        d2p = lambda t: np.polyval(np.polyder(a[::-1], 2), t)
        for sign, wall in ((+1, 1.0), (-1, 0.0)):
            wi, ww, ws = wall_second_derivative_functional(n, interior, sign)
            got = wi @ p(interior) + ww * p(wall) + ws * dp(wall)
            assert got == pytest.approx(d2p(wall), abs=1e-8)


def test_lid_drives_the_flow() -> None:
    """Regression for the trivial-solution defect.

    With a wall closure that ignores the lid slope, ``psi = omega = 0`` is an
    exact solution of the discrete system for *any* lid speed: the solver
    "converges" in zero iterations and reports ``u_min = 0``.  A correct
    implementation must produce a primary vortex of the right sign and size.
    """
    solver = SpectralCavity(16)
    sol = solver.solve(10.0, tol=1e-9)
    assert sol.metrics["u_min"] < -0.15          # Ghia Re = 100: -0.2109
    assert sol.metrics["psi_min"] < -0.05        # primary vortex is clockwise
    assert sol.metrics["kinetic_energy"] > 1e-3
    assert sol.newton["iterations"] > 0


def test_jacobian_matches_finite_differences() -> None:
    solver = SpectralCavity(12)
    X = solver.X
    psi = -0.1 * np.outer(np.sin(np.pi * X), np.sin(np.pi * X))
    omega = -(solver._L @ psi.ravel()).reshape(solver.n1, solver.n1)
    m = solver._interior.size
    state = np.concatenate([omega.ravel()[solver._interior],
                            psi.ravel()[solver._interior]])
    J = solver.jacobian(state, 100.0)
    d = np.random.RandomState(1).standard_normal(state.size)
    eps = 1e-6
    num = (solver.residual(state + eps * d, 100.0)
           - solver.residual(state - eps * d, 100.0)) / (2 * eps)
    assert np.linalg.norm(num - J @ d) / np.linalg.norm(num) < 1e-9


def test_residual_and_wall_slip_close_with_order() -> None:
    """The converged solution must satisfy the PDE and the no-slip wall data."""
    metrics = {}
    for n in (16, 24):
        solver = SpectralCavity(n)
        sol = solver.solve(100.0, tol=1e-10)
        metrics[n] = sol.metrics
        assert sol.metrics["poisson_residual_max"] < 1e-8
        assert sol.metrics["lid_slip_defect"] < 2e-2
        assert sol.metrics["wall_slip_defect"] < 2e-2
    # the wall-slip defect is a discretisation error: it must shrink with N
    assert metrics[24]["lid_slip_defect"] < 0.5 * metrics[16]["lid_slip_defect"]


def test_re100_agrees_with_ghia_within_the_documented_band() -> None:
    """Ghia (1982) Re = 100: u_min = -0.21090, v_max = +0.17527.

    The converged spectral values are -0.214042 / +0.179572, i.e. 1.5 % / 2.5 %
    away from the transcription of Table I/II; the 5 % case-pool band is met and
    the residual offset is attributed in SPECTRAL.md §5 (it is the reference's
    own discretisation, not the spectral solution -- the same offset appears at
    Re = 100 where this scheme is exact to 1e-6).
    """
    solver = SpectralCavity(24)
    sol = solver.solve(100.0, tol=1e-10)
    u_ref = float(np.array(ref.U_CENTRELINE[100.0])[1:-1, 1].min())
    v_ref = float(np.array(ref.V_CENTRELINE[100.0])[1:-1, 1].max())
    assert abs(sol.metrics["u_min"] - u_ref) / abs(u_ref) < 0.05
    assert abs(sol.metrics["v_max"] - v_ref) / abs(v_ref) < 0.05
    # vortex centre (Ghia: 0.6172, 0.7344) must sit inside 1 % of L
    cx, cy = sol.metrics["vortex_centre"]
    assert abs(cx - 0.6172) < 0.01 and abs(cy - 0.7344) < 0.01


def test_solution_is_independent_of_the_corner_rule() -> None:
    """The four lid corners are singular; their closure must not set the answer."""
    fields = {}
    for rule in ("y", "x"):
        solver = SpectralCavity(16, corner_rule=rule)
        sol = solver.solve(100.0, tol=1e-10)
        fields[rule] = sol
    du = np.max(np.abs(fields["y"].u - fields["x"].u))
    dv = np.max(np.abs(fields["y"].v - fields["x"].v))
    assert max(du, dv) < 1e-2 * max(1.0, fields["y"].lid_velocity)


def test_solution_is_independent_of_the_initial_guess() -> None:
    """A warm start only accelerates Newton; it must not move the converged point."""
    solver = SpectralCavity(16)
    cold = solver.solve(100.0, tol=1e-10)
    warm = solver.solve(100.0, psi=1.05 * cold.psi, omega=1.05 * cold.omega, tol=1e-10)
    assert warm.newton["iterations"] <= cold.newton["iterations"]
    assert abs(warm.metrics["u_min"] - cold.metrics["u_min"]) < 1e-8


@pytest.mark.skipif(not (DATA / "cavity_spectral_summary_re100.json").exists(),
                    reason="ladder artefacts not present")
def test_stored_re100_ladder_is_grid_converged() -> None:
    summary = json.loads((DATA / "cavity_spectral_summary_re100.json").read_text())
    recs = sorted(summary["records"], key=lambda r: r["order"])
    u = [r["metrics"]["u_min"] for r in recs]
    slip = [r["metrics"]["lid_slip_defect"] for r in recs]
    assert np.all(np.diff(u) > -1e-6)              # monotone from below
    assert abs(u[-1] - u[-2]) < 1e-4               # converged in N
    assert slip[-1] < slip[0]                      # discretisation error shrinks
    for r in recs:
        assert r["metrics"]["poisson_residual_max"] < 1e-8