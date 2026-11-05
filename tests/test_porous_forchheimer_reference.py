"""Independent references for the Forchheimer porous regime (verification debt).

``src/tensorfvm/verification/porous_forchheimer.py`` supplies two references that
do not read solver internals: the nonlinear continuum BVP (Newton) and an
independent replica of the staggered finite-volume balance.  These tests pin

* that both references reproduce the closed forms already shipped in the package,
* that the converged production SIMPLE field equals the independent replica to
  machine precision --- i.e. the lagged Picard sink really does converge to the
  nonlinear discrete equation, with the correct ``dx*dy`` weight and half-cell
  wall diffusion (no residual-based test in this repository can state that), and
* the quantified, honest accuracy gap of the ``beta > 0`` regime on production
  grids (Brinkman layer ``sqrt(K)`` under-resolved), which the previous 5%
  bulk-law gate did not reveal.
"""

import numpy as np
import pytest
import torch

from tensorfvm.verification.porous_forchheimer import (CHANNEL, continuum_solution,
                                                     discrete_profile, flow_case,
                                                     implementation_agreement,
                                                     reference_self_check)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_references_reproduce_shipped_closed_forms():
    check = reference_self_check()
    for key in ("continuum_K1e-06_beta0", "continuum_K1e-05_beta0",
                "continuum_poiseuille"):
        assert check[key]["profile_linf"] < 1e-5, key
        assert check[key]["dpdx_relative"] < 1e-4, key
    # The replica is a finite-volume operator, so it approaches the closed form
    # with grid refinement rather than bit-exactly: 1% at dy/sqrt(K) = 0.31.
    assert check["discrete_K1e-06_beta0"]["profile_linf"] < 5e-2
    assert check["discrete_K1e-06_beta0"]["dpdx_relative"] < 5e-3


def test_converged_simple_field_equals_independent_finite_volume_replica():
    """The lagged sink at convergence must satisfy the nonlinear discrete equation."""
    audit = implementation_agreement(ny=16, permeability=1e-6, form_drag=2e4)
    assert audit["converged"]
    assert audit["profile_linf"] < 1e-8, audit["profile_linf"]
    assert audit["profile_l2"] < 1e-8
    assert audit["dpdx_relative"] < 1e-8, audit["dpdx_relative"]


def test_forchheimer_regime_is_quantified_against_the_nonlinear_reference():
    """Honest gap: at dy/sqrt(K) = 1.25 the profile is ~7% off, dp/dx ~3% off."""
    case = dict(permeability_m2=1e-6, form_drag_1_per_m=2e4)
    exact = continuum_solution(case)
    sim = flow_case(ny=16, permeability=1e-6, form_drag=2e4)
    exact_c = np.interp(sim["y_m"], exact["y_m"], exact["u_m_s"])
    peak = float(np.max(np.abs(exact_c)))
    profile_linf = float(np.max(np.abs(sim["u_m_s"] - exact_c)) / peak)
    dpdx_relative = abs(sim["dpdx"] - exact["dpdx"]) / abs(exact["dpdx"])
    assert 3e-2 < profile_linf < 1.2e-1, profile_linf
    # The 3% physical gate is *not* met for dp/dx at this resolution: the old 5%
    # bulk-law identity hid this because a plug-like core keeps the mean right.
    assert dpdx_relative > 3e-2, dpdx_relative
    assert abs(float(np.mean(sim["u_m_s"])) - CHANNEL["inlet_velocity"]) \
        < 1e-9  # incompressibility is exact, so the mean velocity is not the error


def test_replica_reference_is_grid_convergent():
    """dy/sqrt(K) = 1.25 -> 0.31 must shrink the replica-to-continuum gap."""
    case = dict(permeability_m2=1e-6, form_drag_1_per_m=2e4)
    exact = continuum_solution(case, grid_points=4001)
    errors = []
    for ny in (16, 64):
        replica = discrete_profile(case, ny=ny)
        exact_c = np.interp(replica["y_m"], exact["y_m"], exact["u_m_s"])
        errors.append(float(np.max(np.abs(replica["u_m_s"] - exact_c))
                            / np.max(np.abs(exact_c))))
    assert errors[0] > errors[1]
    assert errors[1] < 0.6 * errors[0]