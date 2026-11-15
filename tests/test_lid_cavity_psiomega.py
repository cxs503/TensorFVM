"""Unit tests of the independent streamfunction-vorticity cavity solver.

These tests pin the properties the cross-code check relies on:

* the DST-I Poisson solve is a *direct* solve of the discrete Dirichlet problem;
* the face velocity built from ``psi`` is discretely divergence-free by
  construction (this is what makes the second code's mass balance independent of
  any pressure correction);
* the wall-vorticity boundary condition lands on the right wall -- axis 0 is x
  and axis 1 is y, and the lid is the ``j = n`` row.  Getting this backwards
  silently drives the wrong wall, which is exactly the defect this test set was
  written against;
* the Thom balance converges at first order in the wall-normal spacing, i.e. the
  formula is implemented as intended rather than as a plausible-looking variant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.lid_cavity_psiomega import (
    PsiOmegaConfig,
    PsiOmegaSolver,
    poisson_dirichlet,
)


def _discrete_laplacian(field: np.ndarray, h: float) -> np.ndarray:
    """Five-point Laplacian of the interior block of a node field."""
    return (field[2:, 1:-1] + field[:-2, 1:-1] + field[1:-1, 2:] + field[1:-1, :-2]
            - 4.0 * field[1:-1, 1:-1]) / h ** 2


def test_poisson_dirichlet_solves_the_discrete_problem_exactly():
    n, h = 24, 1.0 / 24
    i = np.arange(n + 1)[:, None] * h
    j = np.arange(n + 1)[None, :] * h
    exact = np.sin(2.0 * np.pi * i) * np.sin(np.pi * j) * (1.0 - i) * (1.0 - j)
    exact[0, :] = exact[n, :] = exact[:, 0] = exact[:, n] = 0.0

    rhs = _discrete_laplacian(exact, h)
    solved = poisson_dirichlet(rhs, h)

    assert np.allclose(solved, exact[1:-1, 1:-1], atol=1e-11)


def test_poisson_dirichlet_matrix_fallback_agrees_with_the_transform():
    """The numpy-only fallback must not be a different operator."""
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal((12, 12))
    h = 1.0 / 13
    fast = poisson_dirichlet(rhs, h)

    m = rhs.shape[0]
    s = np.sqrt(2.0 / (m + 1)) * np.sin(
        math.pi * np.outer(np.arange(1, m + 1), np.arange(1, m + 1)) / (m + 1))
    lam = (2.0 * np.cos(math.pi * np.arange(1, m + 1) / (m + 1)) - 2.0) / h ** 2
    slow = s.T @ ((s @ rhs @ s.T) / (lam[:, None] + lam[None, :])) @ s

    assert np.allclose(fast, slow, atol=1e-12)


def test_face_velocity_is_discretely_divergence_free():
    rng = np.random.default_rng(3)
    solver = PsiOmegaSolver(PsiOmegaConfig(n=20))
    psi = rng.standard_normal((21, 21))
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0.0
    solver.psi = psi

    divergence = solver.divergence()
    assert np.abs(divergence).max() < 1e-11
    u, v = solver.face_velocity()
    assert u.shape == (21, 20) and v.shape == (20, 21)
    # the node velocity carries the wall conditions: lid U, other walls no-slip
    u_node, v_node = solver.node_velocity()
    assert np.allclose(u_node[:, -1], 1.0)
    assert np.abs(u_node[:, 0]).max() < 1e-15
    assert np.abs(v_node[0, :]).max() < 1e-15 and np.abs(v_node[-1, :]).max() < 1e-15


def test_wall_vorticity_lands_on_the_correct_wall():
    """Regression: the moving lid is the ``j = n`` row, not the ``i = n`` column."""
    n = 12
    solver = PsiOmegaSolver(PsiOmegaConfig(n=n, lid_velocity=1.0))
    rng = np.random.default_rng(11)
    psi = rng.standard_normal((n + 1, n + 1))
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0.0
    solver.psi = psi
    h = solver.h

    omega = solver.wall_vorticity()
    lid = -2.0 * psi[:, n - 1] / h ** 2 - 2.0 * solver.lid_velocity / h
    right = -2.0 * psi[n - 1, :] / h ** 2
    bottom = -2.0 * psi[:, 1] / h ** 2
    left = -2.0 * psi[1, :] / h ** 2

    # corners are averaged by design, so compare the non-corner entries
    assert np.allclose(omega[1:-1, n], lid[1:-1])
    assert np.allclose(omega[n, 1:-1], right[1:-1])
    assert np.allclose(omega[1:-1, 0], bottom[1:-1])
    assert np.allclose(omega[0, 1:-1], left[1:-1])
    # the lid term must not be applied to a vertical wall
    assert not np.allclose(omega[n, 1:-1], right[1:-1] - 2.0 / h)


def test_thom_balance_first_order_in_the_wall_spacing():
    """``omega_wall = -2 psi_adj/h^2`` for a no-slip wall is O(h) accurate."""
    errors = {}
    for n in (16, 32):
        solver = PsiOmegaSolver(PsiOmegaConfig(n=n))
        h = 1.0 / n
        x = np.arange(n + 1)[:, None] * h
        y = np.arange(n + 1)[None, :] * h
        amp = 0.037
        # psi = 0 on every wall, psi_y = 0 on every wall -> pure no-slip driver
        solver.psi = amp * np.sin(np.pi * x) * (y ** 2) * (y - 1.0) ** 2
        omega_wall = solver.wall_vorticity()[:, 0]
        exact = -2.0 * amp * np.sin(np.pi * x)[:, 0]     # -psi_yy at y = 0
        errors[n] = float(np.abs(omega_wall - exact).max() / np.abs(exact).max())

    ratio = errors[16] / errors[32]
    assert 1.6 < ratio < 2.5, errors
    assert errors[32] < 0.2


@pytest.mark.timeout(180)
def test_solver_reaches_the_ghia_cavity_at_coarse_resolution():
    """n = 32 must already be in the right physical ballpark, and be steady."""
    solver = PsiOmegaSolver(PsiOmegaConfig(n=32, reynolds=1000.0, max_steps=60_000))
    ledger = solver.run()
    functionals = solver.functionals()

    assert ledger["converged"], ledger
    assert ledger["max_divergence"] < 1e-10
    assert ledger["vorticity_defect"] < 1e-8
    # the lid shear has to actually enter the domain: strong negative wall vorticity
    assert solver.omega[:, -1].mean() < -10.0
    assert -0.45 < functionals["u_min"] < -0.25
    assert 0.2 < functionals["v_max"] < 0.45
    assert functionals["psi_min"] < 0.0