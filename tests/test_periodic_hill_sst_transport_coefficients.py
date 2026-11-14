"""Round 12 of CASE-FVM-PERIODIC-HILLS: the SST transport *coefficients* and the
omega wall Dirichlet data.

Round 10 wired ``SSTKOmega`` into :class:`PeriodicHillSolver` and verified the
*algebraic* closure against :func:`native_sst3d.sst2003_closure`.  What was
never checked is how the two *transported* scalars are discretised.  Two defects
survived that far (found in round 12, both fixed by this round):

1. **Double ``1/sigma`` scaling of the diffusion.**  ``_transport_step`` was
   called with ``mu = nu + sigma * nu_t`` *and* assembled
   ``-(1/sigma) * diffusion_operator(mu)``, i.e. the operator it actually solved
   had the eddy diffusivity ``nu_t`` -- not ``sigma * nu_t`` -- and the molecular
   part scaled by ``1/sigma``.  For ``omega`` that is a factor 2 in the eddy
   diffusivity (``SIGMA_W1 = 0.5``).  The module docstring promises
   ``nu + sigma nu_t``.
2. **The omega wall Dirichlet value was documented but never imposed.**  The
   diffusion operator is homogeneous-Dirichlet (``k = 0`` at the wall, correct);
   the same operator silently used ``omega = 0`` at the wall as well, so
   ``wall_omega(nu, d) = 60 nu / (beta1 d^2)`` -- quoted in the docstring and
   used by the 3-D native solver -- never entered the 2-D hill run.

Both defects push the same way: they inflate the eddy viscosity, which shortens
the separation bubble (round 11 read ``x_r/H = 2.05`` vs the Breuer 2009 value
5.34).  These tests pin the *discretisation identities*, not the hill answer:

* the ``k``- and ``omega``-equation matrices are ``I/dt - D(nu + sigma nu_t)``
  with the **F1-blended** ``sigma`` of the closure, assembled with unit
  coefficient (no second ``1/sigma``);
* the ``omega`` production uses the **F1-blended** ``gamma``;
* a *constant* ``omega`` field equal to the wall value is an exact solution of
  the backward-Euler step with no sources (an inhomogeneous-Dirichlet
  reproduction test -- it fails at ``O(10^-2)`` if the wall data are dropped);
* the wall value actually handed to the operator is ``60 nu / (beta1 d^2)`` with
  ``d`` the first-cell wall distance, at both walls.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import diags

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_sst import SSTKOmega
from tensorfvm.native_sst3d import BETA1, wall_omega


def _solver(nx=24, ny=12, amplitude=1.0):
    cfg = PeriodicHillConfig(nx=nx, ny=ny, reynolds=2800.0,
                             hill_amplitude=amplitude)
    solver = PeriodicHillSolver(cfg)
    solver.turbulence = SSTKOmega(solver)
    solver.set_time_step(0.02)
    # a non-trivial eddy-viscosity state: perturb k/omega so F1, sigma_k,
    # sigma_omega and gamma are not constant over the domain
    turb = solver.turbulence
    turb.k = turb.k * (1.0 + 0.5 * np.cos(2.0 * np.pi *
                                          np.arange(solver.n).reshape(ny, nx) / max(ny, 1)))
    turb.omega = turb.omega * (1.0 + 0.3 * np.sin(2.0 * np.pi *
                                                  np.arange(solver.n).reshape(ny, nx) / max(nx, 1)))
    return solver


def _independent_laplacian(solver, mu):
    """Face-based 5-point ``div(mu grad phi)`` assembled from the mesh metric.

    Written from the discretisation (not by calling the module under test):
    every internal face uses the *same* face-averaged coefficient from both
    sides, both walls are homogeneous Dirichlet.
    """
    m = solver.mesh
    ny, nx = solver.ny, solver.nx
    mu = np.asarray(mu, dtype=float)
    vols = m.volume
    L = np.zeros((ny * nx, ny * nx))

    def idx(j, i):
        return j * nx + i

    for j in range(ny):
        for i in range(nx):
            p = idx(j, i)
            V = vols[j, i]
            mu_e = 0.5 * (mu[j, i] + mu[j, (i + 1) % nx])
            mu_w = 0.5 * (mu[j, i] + mu[j, (i - 1) % nx])
            mu_n = mu[j, i] if j == ny - 1 else 0.5 * (mu[j, i] + mu[j + 1, i])
            mu_s = mu[j, i] if j == 0 else 0.5 * (mu[j, i] + mu[j - 1, i])
            ce = solver.c_e[j, i] * mu_e
            cw = solver.c_e[j, (i - 1) % nx] * mu_w
            cn = solver.c_n[j + 1, i] * mu_n
            cs = solver.c_n[j, i] * mu_s
            L[p, idx(j, (i + 1) % nx)] += ce / V
            L[p, idx(j, (i - 1) % nx)] += cw / V
            L[p, p] -= (ce + cw + cs + cn) / V
            if j < ny - 1:
                L[p, idx(j + 1, i)] += cn / V
            if j > 0:
                L[p, idx(j - 1, i)] += cs / V
    return L


def test_diffusion_operator_is_not_double_scaled():
    """``A_k = I/dt - D(nu + sigma_k nu_t) + diag(dest)`` with unit coefficient."""
    solver = _solver()
    turb = solver.turbulence
    src = turb.source_terms()
    nu_t = np.maximum(src["nu_t"], 0.0)
    sigma_k = np.asarray(src["sigma_k"])
    assert sigma_k.max() - sigma_k.min() > 1.0e-6, "test state must exercise F1 blending"

    dt = 0.02
    mu = solver.nu + sigma_k * nu_t
    A = turb.transport_matrix(mu, src["k_destruction_linear"], dt)
    reference = (diags(np.full(solver.n, 1.0 / dt))
                 - _independent_laplacian(solver, mu)
                 + diags(np.asarray(src["k_destruction_linear"]).ravel()))
    np.testing.assert_allclose(A.toarray(), reference, rtol=1e-12, atol=1e-13)

    # the defective variant -- coefficient (nu + sigma nu_t) / sigma -- must differ
    wrong = (diags(np.full(solver.n, 1.0 / dt))
             - _independent_laplacian(solver, mu / sigma_k)
             + diags(np.asarray(src["k_destruction_linear"]).ravel()))
    assert np.max(np.abs(A.toarray() - wrong)) > 1.0e-6


def test_diffusion_operator_is_conservative():
    """Varying ``nu_t`` must not break the face pairing of the FV operator.

    ``div(mu grad)`` is assembled per cell, so the *volume-weighted* operator
    ``diag(V) L`` is the object that must be symmetric (cell volumes vary in
    ``x`` on the terrain-following mesh), and every interior row must sum to
    zero -- i.e. one face coefficient per face, shared by both neighbours.
    """
    solver = _solver()
    turb = solver.turbulence
    mu = 1.0e-3 * np.exp(np.linspace(0.0, 3.0, solver.n)).reshape(solver.ny, solver.nx)
    L = turb._diffusion_operator(mu).toarray()
    vols = solver.mesh.volume.ravel()
    weighted = vols[:, None] * L
    np.testing.assert_allclose(weighted, weighted.T, rtol=1e-12, atol=1e-15)
    interior = np.ones(solver.n, dtype=bool)
    interior[:solver.nx] = False           # bottom (hill) wall row
    interior[-solver.nx:] = False          # top wall row
    np.testing.assert_allclose(L.sum(axis=1)[interior], 0.0, atol=1e-15)


def test_update_uses_blended_sigma_and_gamma():
    """``update`` hands the blended closure sigmas/gamma to the transport."""
    solver = _solver()
    turb = solver.turbulence
    src = turb.source_terms()
    nu_t = np.maximum(src["nu_t"], 0.0)
    seen = []

    def recorder(phi, mu, prod, dest_lin, extra, dt, wall_value=None):
        seen.append(dict(phi=phi.copy(), mu=np.array(mu, dtype=float),
                         prod=np.array(prod, dtype=float), dest=np.array(dest_lin),
                         extra=np.array(extra), dt=dt, wall_value=wall_value))
        return phi.copy()

    turb._transport_step = recorder
    turb.update(0.02)
    assert len(seen) == 2

    np.testing.assert_allclose(seen[0]["mu"], solver.nu + src["sigma_k"] * nu_t, rtol=1e-13)
    np.testing.assert_allclose(seen[1]["mu"], solver.nu + src["sigma_omega"] * nu_t, rtol=1e-13)
    # omega production carries the blended gamma (and stays finite at nu_t -> 0)
    expected = src["gamma"] * np.divide(src["production"], nu_t,
                                        out=np.zeros_like(nu_t), where=nu_t > 0.0)
    np.testing.assert_allclose(seen[1]["prod"], expected, rtol=1e-12)
    assert np.all(np.isfinite(seen[1]["prod"]))
    assert src["gamma"].max() - src["gamma"].min() > 1.0e-6


def test_omega_wall_dirichlet_is_imposed():
    """A constant ``omega`` equal to the wall value is an exact BE fixed point."""
    solver = _solver(nx=16, ny=10)
    turb = solver.turbulence
    dt = 0.02
    nu_wall = float(wall_omega(solver.nu, float(turb.distance[0, 0])))
    assert nu_wall > 1.0
    phi = np.full((solver.ny, solver.nx), nu_wall)
    mu = np.full((solver.ny, solver.nx), solver.nu + 3.0 * solver.nu)
    wall = np.full((2, solver.nx), nu_wall)
    out = turb._transport_step(phi, mu, np.zeros_like(phi), np.zeros_like(phi),
                               np.zeros_like(phi), dt, wall_value=wall)
    np.testing.assert_allclose(out, phi, rtol=1e-12, atol=1e-12)

    # dropping the wall data must break the reproduction (the old defect)
    broken = turb._transport_step(phi, mu, np.zeros_like(phi), np.zeros_like(phi),
                                  np.zeros_like(phi), dt)
    assert np.max(np.abs(broken - phi)) > 1.0e-3


def test_update_imposes_wall_omega_at_both_walls():
    """``update`` passes ``60 nu / (beta1 d^2)`` (first-cell distance) to the operator."""
    solver = _solver()
    turb = solver.turbulence
    seen = []

    def recorder(phi, mu, prod, dest_lin, extra, dt, wall_value=None):
        seen.append(wall_value)
        return phi.copy()

    turb._transport_step = recorder
    turb.update(0.02)
    assert seen[0] is None, "k has a homogeneous Dirichlet condition at both walls"
    wall = np.asarray(seen[1], dtype=float)
    assert wall.shape == (2, solver.nx)
    d = turb.distance
    np.testing.assert_allclose(wall[0], 60.0 * solver.nu / (BETA1 * d[0] ** 2), rtol=1e-12)
    np.testing.assert_allclose(wall[1], 60.0 * solver.nu / (BETA1 * d[-1] ** 2), rtol=1e-12)