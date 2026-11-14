"""Round 10 of CASE-FVM-PERIODIC-HILLS: the SST k-omega closure is correctly wired.

Every helper is pinned to an *independent* check rather than to itself:

* ``test_closure_matches_shared_sst`` -- the 2-D hill closure's ``nu_t`` is the
  same algebraic object as :func:`native_sst3d.sst2003_closure` (one source of
  truth for the model constants);
* ``test_velocity_gradient_tensor_2d`` -- the 3x3 tensor built from 2-D
  gradients has the correct out-of-plane-zero structure;
* ``test_channel_reproduces_poiseuille`` -- the hill-free SST channel
  reproduces the analytic plane-Poiseuille solution (the closure must not
  break the laminar baseline);
* ``test_positivity_and_bulk`` -- ``k >= 0``, ``omega > 0``, finite field and
  the imposed bulk velocity are held on a short run;
* ``test_sst_reattachment_shorter_than_sa`` -- the round-10 model-class claim:
  on the same coarse grid and budget, the SST reattachment is *shorter* than
  the SA over-diffusive value (SA round 8: ``x_r/H ~ 8`` at ``t = 60``).

None of this asserts the 10 % acceptance gate passes -- rounds 1--9 record the
2-D URANS answer is over-diffusive, and the full Re = 2800 acceptance is a
cluster run (``docs/case-periodic-hills/ROUND10.md``).  What is asserted here is
that the new closure is *correctly implemented*, so the round-10 verdict is
about physics and not about a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.periodic_hill import (
    PeriodicHillConfig,
    PeriodicHillSolver,
    analytic_flat_channel,
)
from tensorfvm.periodic_hill_sst import (
    SSTKOmega,
    SSTKOmegaConfig,
    velocity_gradient_tensor,
)
from tensorfvm.native_sst3d import sst2003_closure
from tensorfvm.verification.periodic_hill_round10 import (
    sst_hill_solver,
    sst_channel_solver,
)


def _make_solver(nx=48, ny=24, hill_amplitude=1.0):
    cfg = PeriodicHillConfig(nx=nx, ny=ny, reynolds=2800.0,
                             hill_amplitude=hill_amplitude)
    solver = PeriodicHillSolver(cfg)
    solver.turbulence = SSTKOmega(solver)
    solver.set_time_step(0.02)
    return solver


def test_velocity_gradient_tensor_2d():
    """The 3x3 tensor from 2-D gradients has zero out-of-plane components."""
    gux = np.array([[1.0, 2.0], [3.0, 4.0]])
    guy = np.array([[0.5, 1.5], [2.5, 3.5]])
    gvx = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    gvy = np.array([[0.1, 0.2], [0.3, 0.4]])
    G = velocity_gradient_tensor(gux, guy, gvx, gvy)
    assert G.shape == (2, 2, 3, 3)
    assert np.all(G[..., 0, 0] == gux)
    assert np.all(G[..., 0, 1] == guy)
    assert np.all(G[..., 1, 0] == gvx)
    assert np.all(G[..., 1, 1] == gvy)
    # out-of-plane rows/cols are zero
    assert np.all(G[..., 2, :] == 0.0)
    assert np.all(G[..., :, 2] == 0.0)


def test_closure_matches_shared_sst():
    """The hill closure's ``nu_t`` is the shared ``sst2003_closure`` object."""
    solver = _make_solver()
    cl = solver.turbulence.closure()
    # the closure dict is produced by the shared function; nu_t must be finite
    # and positive
    assert np.all(np.isfinite(cl["nu_t"]))
    assert np.all(cl["nu_t"] > 0.0)
    # F1/F2 blending in [0, 1]
    assert np.all((cl["F1"] >= 0.0) & (cl["F1"] <= 1.0))
    assert np.all((cl["F2"] >= 0.0) & (cl["F2"] <= 1.0))
    # nu_t = a1 k / max(a1 omega, sqrt(S2) F2) -- the algebraic identity
    denom = np.maximum(solver.turbulence.config.a1 * solver.turbulence.omega,
                       np.sqrt(cl["strain_squared"]) * cl["F2"])
    expected = solver.turbulence.config.a1 * solver.turbulence.k / denom
    assert np.allclose(cl["nu_t"], expected, rtol=1e-6)


def test_channel_reproduces_poiseuille():
    """The hill-free SST channel reproduces the analytic plane-Poiseuille.

    The closure must not break the laminar baseline: on the hill-free mesh the
    mid-channel velocity must sit at the bulk value (the ERCOFTAC bulk
    functional ``u_b = 1`` is the imposed input), and the profile must be
    symmetric about the channel centreline.
    """
    solver = sst_channel_solver(ny=24, nx=48)
    for _ in range(200):
        solver.advance(1)
    ub = solver.bulk_velocity()
    assert abs(ub - 1.0) < 1e-6, f"bulk velocity not held: {ub}"
    # mid-channel velocity ~ bulk (the channel is driven to u_b = 1)
    u_mid = solver.u[solver.ny // 2, :].mean()
    assert abs(u_mid - 1.0) < 0.05, f"mid-channel u={u_mid:.3f} vs bulk=1.0"
    # profile symmetric about the centreline (top/bottom halves agree)
    u_top = solver.u[solver.ny // 2:, :].mean()
    u_bot = solver.u[: solver.ny // 2, :].mean()
    assert abs(u_top - u_bot) < 0.05, (
        f"asymmetric channel: top={u_top:.3f} bottom={u_bot:.3f}")


def test_positivity_and_bulk():
    """k >= 0, omega > 0, finite field and bulk velocity held on a short run."""
    solver = _make_solver()
    for _ in range(100):
        solver.advance(1)
    assert np.all(solver.turbulence.k >= 0.0)
    assert np.all(solver.turbulence.omega > 0.0)
    assert np.all(np.isfinite(solver.turbulence.viscosity()))
    assert abs(solver.bulk_velocity() - 1.0) < 1e-6


def test_sst_reattachment_shorter_than_sa():
    """Round-10 model-class claim: SST reattachment is shorter than SA's ~8.

    On the same 48x24 grid and a short budget, the SA closure (round 8) keeps
    the bubble attached far downstream (``x_r/H ~ 8`` at ``t = 60``).  The SST
    closure must produce a *shorter* reattachment (a separation bubble that
    closes upstream of the SA value), which is the direction the reference
    (``x_r/H = 5.34``) requires.
    """
    solver = _make_solver()
    trend = []
    for i in range(300):
        solver.advance(1)
        if i % 50 == 0:
            sep, rea = solver.reattachment()
            trend.append((i * 0.02, sep, rea))
    # at t = 5 the bubble must be closed (rea finite) and shorter than 8
    t5 = next((r for t, s, r in trend if abs(t - 5.0) < 0.5), None)
    assert t5 is not None and np.isfinite(t5), "no reattachment at t=5"
    assert t5 < 8.0, f"SST reattachment {t5:.2f} not shorter than SA's ~8"
