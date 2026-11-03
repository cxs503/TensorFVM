"""Anchors for the Darcy--Forchheimer porous resistance (``porous_drag``).

``TensorFVM`` had no porous-media capability before this module (no ``darcy`` /
``forchheimer`` token anywhere in ``src``, ``tests`` or ``scripts``), so the
extension is verified against closed-form references rather than against itself:

* ``porous=None`` must leave the production ``SimpleSolver`` bitwise unchanged;
* Darcy--Brinkman limit must reproduce the exact channel profile
  ``u = A[1 - cosh((y-h)/l)/cosh(h/l)]`` and ``dp/dx = -mu A/K``;
* ``K -> infinity`` must recover plane Poiseuille flow;
* Forchheimer regime must be monotone in ``beta`` and obey the bulk law.

Full-resolution evidence and figures: ``scripts/verify_porous_drag.py`` ->
``docs/porous-darcy-forchheimer/``.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_porous_drag import (darcy_brinkman_case, inert_extension_check,  # noqa: E402
                                no_resistance_limit)
from tensorfvm.porous_drag import (PorousChannelSolver, PorousMedia,  # noqa: E402
                                   brinkman_pressure_gradient, brinkman_profile,
                                   forchheimer_bulk_velocity)
from tensorfvm.solver import SolverConfig  # noqa: E402


def test_porous_media_rejects_unphysical_input():
    with pytest.raises(ValueError):
        PorousMedia(permeability_m2=0.0)
    with pytest.raises(ValueError):
        PorousMedia(permeability_m2=-1e-6)
    with pytest.raises(ValueError):
        PorousMedia(permeability_m2=1e-6, form_drag_1_per_m=-3.0)


def test_analytic_helpers_hit_their_own_limits():
    """Darcy limit checks and the closed-form Forchheimer root are self-consistent."""
    y = torch.linspace(0.0, 0.02, 11)
    # K = 1e-6 m^2: ell/h = 0.1, so the profile is a near plug with an 11 % deficit
    # at the wall layers; no cosh overflow (h/ell = 10).
    profile = brinkman_profile(y, 5e-4, 0.02, 1e-6)
    assert float(profile.max()) <= 5e-4 * 1.2
    assert float(profile.min()) >= 0.0
    assert float((profile[1:-1] - 5e-4).abs().max()) / 5e-4 < 0.15
    # Mid-channel velocity exceeds the mean (wall layers carry less).
    assert float(profile[5]) > 5e-4
    # Darcy's law for that profile is one sign-definite large gradient.
    assert brinkman_pressure_gradient(5e-4, 0.02, 1e-6, 1e-2) < -1.0
    # Forchheimer law root satisfies the quadratic by construction.
    u = forchheimer_bulk_velocity(-7.0, 1e-2, 1000.0, 1e-6, 2e4)
    residual = abs(1e-2 / 1e-6 * u + 1000.0 * 2e4 * u * u - 7.0) / 7.0
    assert residual < 1e-9; residual


def test_extension_is_inert_without_media():
    """``porous=None`` reproduces the production solver bit for bit."""
    report = inert_extension_check(ny=8)
    assert report["bitwise_identical"]
    assert report["max_abs_u"] == 0.0 and report["max_abs_p"] == 0.0


def test_darcy_brinkman_profile_matches_exact_solution():
    """Exact Darcy--Brinkman channel: profile and pressure gradient within 6 %."""
    case = darcy_brinkman_case(ny=16, permeability=4e-6)
    values = {k: case[k] for k in ("converged", "profile_l2_relative_error",
                                   "profile_linf_relative_error", "dpdx_relative_error",
                                   "dpdx_simulated", "dpdx_exact")}
    assert case["converged"], values
    assert float(case["profile_l2_relative_error"]) < 0.06, values
    assert float(case["profile_linf_relative_error"]) < 0.10, values
    assert float(case["dpdx_relative_error"]) < 0.06, values


def test_large_permeability_recovers_poiseuille():
    """K -> infinity: negligible resistance, plane Poiseuille is recovered."""
    limit = no_resistance_limit(ny=12, permeability=1.0)
    assert limit["poiseuille_profile_l2_relative_error"] < 0.05
    assert limit["poiseuille_dpdx_relative_error"] < 0.05


def test_forchheimer_regime_is_monotone_and_obeys_bulk_law():
    """Raising beta raises |dp/dx| and the central-region velocity follows the law."""
    darcy = darcy_brinkman_case(ny=12, permeability=1e-6, form_drag=0.0)
    strong = darcy_brinkman_case(ny=12, permeability=1e-6, form_drag=2e4)
    assert strong["converged"] and darcy["converged"]
    assert abs(strong["dpdx_simulated"]) > abs(darcy["dpdx_simulated"])
    assert strong["forchheimer_bulk_law_residual"] < 0.05
    # Mass conservation pins the mean velocity to the inlet value: the extra
    # resistance appears as pressure gradient, never as a mass sink.
    assert abs(strong["mean_velocity"] - 5e-4) / 5e-4 < 5e-3


def test_solver_rejects_nonphysical_configuration():
    """A configured porous solver must still honour SolverConfig validation."""
    with pytest.raises(ValueError):
        PorousChannelSolver(SolverConfig(nx=4, ny=4, cylinder_radius=0.2),
                            PorousMedia(permeability_m2=1e-6))