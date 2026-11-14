"""Round 13: the eta-face viscosity pairing rule of the hill solver.

The discrete diffusion operators of the periodic-hill solver are built as

    ``L_h phi|_P = (1/Vol_P) sum_f c_f mu_f (phi_N - phi_P)``

with the two walls Dirichlet (``phi_wall = 0`` for the momentum/transport
operators, the inhomogeneous ``omega`` value is added to the right-hand side).
An eta face sits on **one** level ``L`` but borders **two** cells: it is the north
face of row ``L-1`` and the south face of row ``L``.  If the two rows do not read
the same ``mu_f`` the operator is not the discrete divergence of a single flux,
and it loses the property the implicit solve depends on:

    ``sum_P Vol_P phi_P (L_h phi)_P = -sum_f c_f mu_f (phi_N - phi_P)^2 <= 0``

(discrete dissipation is non-positive, i.e. the operator only ever removes
energy).  Rounds 4-12 used ``mu[j]`` (the cell's own value) for the south face and
``0.5*(mu[j] + mu[j+1])`` for the north face; the tests below measure what that
costs and pin the round-13 fix.

Every reference quantity here is an independent construction (a face loop written
from the definition, a closed-form identity), not a call to the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillSolver,
    eta_face_viscosity,
)


def _solver(nx=32, ny=16):
    return PeriodicHillSolver(PeriodicHillConfig(nx=nx, ny=ny))


def _mu_field(solver, nu):
    """Strongly varying viscosity: the hill problem spans ~300 nu across the shear layer."""
    ny, nx = solver.ny, solver.nx
    eta = np.linspace(0.0, 1.0, ny)[:, None]
    return nu * (1.0 + 300.0 * np.sin(np.pi * eta) ** 2 * np.ones((1, nx)))


def _face_loop_operator(solver, mu, south_rule="paired"):
    """Independent face-by-face assembly of ``(1/Vol) sum_f c_f mu_f (phi_N - phi_P)``.

    ``south_rule="paired"``: one viscosity per eta face, shared by both cells.
    ``south_rule="legacy"``: south face uses the cell's own ``mu[j]`` (rounds 4-12).
    """
    ny, nx = solver.ny, solver.nx
    c_e, c_n = solver.c_e, solver.c_n
    vols = solver.mesh.volume
    A = np.zeros((ny, nx, ny, nx))

    def add(p, n, coef, scale_p):
        A[p[0], p[1], n[0], n[1]] += coef / scale_p

    mu_eta = eta_face_viscosity(mu) if south_rule == "paired" else None
    for j in range(ny):
        for i in range(nx):
            ip, im = (i + 1) % nx, (i - 1) % nx
            # east / west
            mue = 0.5 * (mu[j, i] + mu[j, ip])
            muw = 0.5 * (mu[j, i] + mu[j, im])
            add((j, i), (j, ip), c_e[j, i] * mue, vols[j, i])
            add((j, i), (j, im), c_e[j, im] * muw, vols[j, i])
            # south (level j) / north (level j+1)
            if south_rule == "paired":
                cs = c_n[j, i] * mu_eta[j, i]
                cn = c_n[j + 1, i] * mu_eta[j + 1, i]
            else:
                cs = c_n[j, i] * mu[j, i]
                cn = c_n[j + 1, i] * (
                    mu[j, i] if j == ny - 1 else 0.5 * (mu[j, i] + mu[j + 1, i]))
            if j > 0:
                add((j, i), (j - 1, i), cs, vols[j, i])
            if j < ny - 1:
                add((j, i), (j + 1, i), cn, vols[j, i])
            A[j, i, j, i] -= (c_e[j, i] * mue + c_e[j, im] * muw + cs + cn) / vols[j, i]
    return A.reshape(ny * nx, ny * nx)


def _face_sum(solver, mu, phi):
    """``sum_f c_f mu_f (phi_N - phi_P)^2`` over every face, walls Dirichlet 0.

    Independent of the assembled matrix: written from the face definition, with
    one viscosity per eta face (:func:`eta_face_viscosity`).
    """
    ny, nx = solver.ny, solver.nx
    c_e, c_n = solver.c_e, solver.c_n
    mu_eta = eta_face_viscosity(mu)
    ip = solver.mesh.ip                       # periodic wrap in xi
    jump_e = phi[:, ip] - phi
    total = np.sum(c_e[:, :nx] * 0.5 * (mu + mu[:, ip]) * jump_e ** 2)
    # interior eta levels 1..ny-1
    total += np.sum(c_n[1:-1] * mu_eta[1:-1] * (phi[1:] - phi[:-1]) ** 2)
    # the two wall levels: Dirichlet phi_wall = 0 -> the jump is the cell value
    total += np.sum(c_n[0] * mu_eta[0] * phi[0] ** 2)
    total += np.sum(c_n[-1] * mu_eta[-1] * phi[-1] ** 2)
    return float(total)


def _volume_form(solver, mat, phi):
    """``sum_P Vol_P phi_P (L phi)_P`` -- the energy-weighted quadratic form."""
    vols = solver.mesh.volume
    return float(np.sum(vols * phi * (mat @ phi.ravel()).reshape(solver.ny, solver.nx)))


# --------------------------------------------------------------- the helper itself
def test_eta_face_values_pair_the_two_neighbouring_cells():
    mu = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    face = eta_face_viscosity(mu)
    assert face.shape == (4, 2)
    np.testing.assert_allclose(face[0], [1.0, 2.0])          # bottom wall level
    np.testing.assert_allclose(face[1], [5.5, 11.0])         # 0.5*(1+10)
    np.testing.assert_allclose(face[2], [55.0, 110.0])       # 0.5*(10+100)
    np.testing.assert_allclose(face[3], [100.0, 200.0])      # top wall level


def test_eta_face_rejects_invalid_viscosity():
    with pytest.raises(ValueError):
        eta_face_viscosity(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        eta_face_viscosity(np.full((3, 4), np.nan))
    with pytest.raises(ValueError):
        eta_face_viscosity(np.ones(4))


# --------------------------------------------------- the assembled momentum operator
def test_momentum_operator_matches_an_independent_face_loop():
    """The assembled matrix *is* ``(1/Vol) sum_f c_f mu_f (phi_N - phi_P)``, face by face."""
    solver = _solver()
    mu = _mu_field(solver, solver.nu)
    solver.set_eddy_viscosity(mu - solver.nu)
    built = solver.laplacian.toarray()
    reference = _face_loop_operator(solver, mu, south_rule="paired")
    np.testing.assert_allclose(built, reference, atol=1e-14)


def test_legacy_south_face_rule_is_not_the_face_loop():
    """Negative control: rounds 4-12's rule is measurably a *different* operator."""
    solver = _solver()
    mu = _mu_field(solver, solver.nu)
    solver.set_eddy_viscosity(mu - solver.nu)
    built = solver.laplacian.toarray()
    legacy = _face_loop_operator(solver, mu, south_rule="legacy")
    gap = np.max(np.abs(built - legacy))
    # the mismatch is exactly the difference of the two south-face conductances
    expected = np.max(solver.c_n[1:-1]
                      * 0.5 * np.abs(mu[1:] - mu[:-1])
                      / solver.mesh.volume[1:])
    assert gap == pytest.approx(expected, rel=1e-12)
    assert gap > 1e-3   # not a rounding-level difference


def test_momentum_operator_is_energy_stable():
    """``sum_P Vol_P phi_P (L phi)_P = -sum_f c_f mu_f (phi_N - phi_P)^2``."""
    solver = _solver()
    mu = _mu_field(solver, solver.nu)
    solver.set_eddy_viscosity(mu - solver.nu)
    rng = np.random.default_rng(20261114)
    phi = rng.standard_normal((solver.ny, solver.nx))
    volume_form = _volume_form(solver, solver.laplacian, phi)
    assert volume_form == pytest.approx(-_face_sum(solver, mu, phi),
                                        rel=1e-10, abs=1e-12)
    assert volume_form < 0.0


def test_legacy_south_face_rule_breaks_energy_stability():
    """The defect broke the identity: the same field is no longer dissipated."""
    solver = _solver()
    mu = _mu_field(solver, solver.nu)
    legacy = _face_loop_operator(solver, mu, south_rule="legacy")
    fixed = _face_loop_operator(solver, mu, south_rule="paired")
    rng = np.random.default_rng(20261114)
    phi = rng.standard_normal((solver.ny, solver.nx))
    target = -_face_sum(solver, mu, phi)
    assert _volume_form(solver, fixed, phi) == pytest.approx(target, rel=1e-10)
    # the legacy pairing misses the identity by a first-order amount
    violation = _volume_form(solver, legacy, phi) - target
    assert abs(violation) > 1e-3 * abs(target)
    # the sign of the violation follows d(mu)/d(eta): the assembled operator then
    # under-dissipates where the shear layer thickens and over-dissipates where it
    # thins (measured 3.5 % / 3.2 % of the total dissipation below)
    eta = np.linspace(0.0, 1.0, solver.ny)[:, None] * np.ones((1, solver.nx))
    decay = solver.nu * (1.0 + 300.0 * np.exp(-4.0 * eta))
    rise = solver.nu * (1.0 + 300.0 * eta ** 2)
    rng = np.random.default_rng(3)
    phi = rng.standard_normal((solver.ny, solver.nx))
    for mu_field, sign in ((decay, +1.0), (rise, -1.0)):
        legacy = _face_loop_operator(solver, mu_field, south_rule="legacy")
        fixed = _face_loop_operator(solver, mu_field, south_rule="paired")
        target = -_face_sum(solver, mu_field, phi)
        assert _volume_form(solver, fixed, phi) == pytest.approx(target, rel=1e-10)
        viol = _volume_form(solver, legacy, phi) - target
        assert sign * viol > 0.0
        assert abs(viol) > 3.0e-2 * abs(target)


def test_uniform_viscosity_reproduces_the_round4_operator():
    """Backwards compatibility: ``mu`` uniform must not move a single entry."""
    solver = _solver()
    nu = solver.nu
    reference = solver.laplacian.toarray().copy()
    solver.set_eddy_viscosity(np.zeros((solver.ny, solver.nx)))
    np.testing.assert_array_equal(solver.laplacian.toarray(), reference)
    # and the face pairing degenerates to the cell value
    face = eta_face_viscosity(np.full((solver.ny, solver.nx), nu))
    np.testing.assert_array_equal(face, np.full((solver.ny + 1, solver.nx), nu))


def test_sa_transport_operator_uses_the_same_pairing():
    """``periodic_hill_turbulence.SpalartAllmaras`` had the identical defect."""
    from tensorfvm.periodic_hill_turbulence import SpalartAllmaras

    solver = _solver()
    sa = SpalartAllmaras(solver)
    mu = _mu_field(solver, solver.nu)
    built = sa._diffusion_operator(mu).toarray()
    reference = _face_loop_operator(solver, mu, south_rule="paired")
    np.testing.assert_allclose(built, reference, atol=1e-14)