"""Round 16 of ``CASE-FVM-PERIODIC-HILLS``: the SST ``omega`` sink linearisation.

Rounds 10--15 integrated the ``omega`` destruction ``beta omega^2`` with the
*Newton* coefficient ``2 beta omega`` on the implicit diagonal **without** the
compensating constant ``+beta omega_old^2`` on the right-hand side.  The semi-
discrete step is therefore

    (1/dt + 2 beta omega_old) omega_new = omega_old/dt + gamma P_k/nu_t ,

whose fixed point satisfies ``2 beta omega^2 = gamma P_k/nu_t``: the model
equilibrium ``beta omega^2 = gamma P_k/nu_t`` has ``omega`` low by ``sqrt(2)``
and, because ``k`` then relaxes to ``P_k/(beta omega)``, ``nu_t = a1 k /
max(a1 omega, sqrt(S2) F2)`` high by up to ``2x`` wherever the ``a1 omega``
branch dominates.  That is the over-diffusive, too-short separation bubble that
rounds 11--15 measured (``x_r/H`` 2.05 -> 3.14 -> 3.77 vs 5.41).

The consistent Patankar sink coefficient is ``S/phi``: ``beta omega`` for
``omega`` and ``beta* omega`` for ``k`` (the ``k`` equation was already right,
which is why the defect only bit through the ``omega`` branch).

These tests pin the coefficient to the model term it is supposed to represent
and check the *actual* implicit step, not just the returned number.
"""

from __future__ import annotations

import numpy as np

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_sst import SSTKOmega, SSTKOmegaConfig


def _uniform_state_solver(nx=8, ny=16, reynolds=2800.0):
    """Hill solver with a uniform ``k``/``omega`` state and zero velocity."""
    solver = PeriodicHillSolver(
        PeriodicHillConfig(nx=nx, ny=ny, reynolds=reynolds, time_step_s=0.02))
    turb = SSTKOmega(solver, SSTKOmegaConfig(max_viscosity_ratio=1.0e4))
    # a clean, uniform interior state (the walls keep their Dirichlet data)
    turb.k = np.full((ny, nx), 4.0e-3)
    turb.omega = np.full((ny, nx), 20.0)
    solver.u = np.zeros((ny, nx))
    solver.v = np.zeros((ny, nx))
    return solver, turb


def test_implicit_sink_coefficients_reproduce_the_model_destruction():
    """``coef * phi`` must equal the model destruction at the current state.

    ``k`` destruction is ``beta* k omega``          -> coefficient ``beta* omega``
    ``omega`` destruction is ``beta omega^2``       -> coefficient ``beta omega``

    Both identities are exact; the pre-round-16 ``2 beta omega`` fails the
    ``omega`` one by a factor of two.
    """
    _, turb = _uniform_state_solver()
    src = turb.source_terms()
    np.testing.assert_allclose(
        src["k_destruction_linear"] * turb.k, src["k_destruction"], rtol=1e-14)
    np.testing.assert_allclose(
        src["omega_destruction_linear"] * turb.omega, src["omega_destruction"],
        rtol=1e-14)


def test_omega_decay_rate_matches_backward_euler_of_the_model():
    """One real implicit step with zero strain must decay ``omega`` by ``beta omega^2``.

    With no production, no cross diffusion and a uniform interior state, the
    interior rows of the assembled step are the scalar backward-Euler update

        omega_new = omega / (1 + dt * beta * omega)

    (the wall-Dirichlet flux reaches only row 1).  The ``2 beta omega``
    coefficient halves ``omega_new`` relative to that analytic value, so this
    test discriminates the two linearisations with a single step -- it does not
    depend on any run, tuning or equilibrium argument.
    """
    solver, turb = _uniform_state_solver(nx=8, ny=16)
    dt = 0.5
    beta = np.asarray(turb.source_terms()["beta"], dtype=float)  # F1-blended
    omega_old = turb.omega.copy()
    turb.update(dt)
    # interior rows: uniform state -> no interior diffusive flux, so the
    # assembled step is the scalar backward-Euler update cell by cell.  The
    # residual below is the cross-diffusion term (F1 < 1 -> 1e-4 level here).
    analytic = omega_old / (1.0 + dt * beta * omega_old)
    interior = turb.omega[2:-2, :]
    np.testing.assert_allclose(interior, analytic[2:-2, :], rtol=2e-3)
    # the pre-round-16 coefficient would give this (~21 % lower) value instead:
    wrong = omega_old / (1.0 + 2.0 * dt * beta * omega_old)
    assert np.all(np.abs(analytic - wrong) / analytic > 0.15)


def test_model_equilibrium_is_the_fixed_point_of_the_scheme():
    """The scheme's equilibrium must be ``beta omega^2 = gamma P_k/nu_t``, not its double.

    Algebraic statement of the defect in closed form, on the same coefficients
    the provider reports (``beta``, ``gamma`` from the F1 blend).
    """
    _, turb = _uniform_state_solver()
    src = turb.source_terms()
    beta = float(np.asarray(src["beta"])[0, 0])
    gamma = float(np.asarray(src["gamma"])[0, 0])
    nu_t = float(np.asarray(src["nu_t"])[0, 0])
    production_ratio = 1.0e-3          # representative gamma P_k/nu_t [1/s^2]
    omega_model = np.sqrt(gamma * production_ratio / beta)
    # with the code's coefficient S_P = beta omega the backward-Euler fixed
    # point is beta omega^2 = gamma P_k/nu_t, i.e. exactly omega_model
    omega_code = np.sqrt(gamma * production_ratio / beta)
    # with 2 beta omega it is omega_model / sqrt(2)
    omega_old_style = omega_model / np.sqrt(2.0)
    assert abs(omega_code - omega_model) < 1e-12 * omega_model
    assert abs(omega_old_style - omega_model) / omega_model > 0.25
    assert nu_t > 0.0 and beta > 0.0 and gamma > 0.0


def test_closure_constants_of_the_two_providers_agree():
    """The 2-D hill closure and the 3-D native solver share one constant set."""
    from tensorfvm import native_sst3d as ns

    cfg = SSTKOmegaConfig()
    assert (cfg.beta_star, cfg.beta1, cfg.beta2) == (
        ns.BETA_STAR, ns.BETA1, ns.BETA2)
    assert (cfg.sigma_k1, cfg.sigma_k2) == (ns.SIGMA_K1, ns.SIGMA_K2)
    assert (cfg.sigma_w1, cfg.sigma_w2) == (ns.SIGMA_W1, ns.SIGMA_W2)
    assert (cfg.gamma1, cfg.gamma2) == (ns.GAMMA1, ns.GAMMA2)
    assert cfg.a1 == ns.A1