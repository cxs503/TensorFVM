"""Lax (1954) shock-tube gate: exact-solution cross-check + <2 % acceptance.

Two independent claims are gated:

1. the reference is the *exact* solution -- the in-repo Toro iteration and a
   from-scratch bisection re-derivation agree to machine precision, and the
   right-shock jump satisfies the Rankine--Hugoniot relations;
2. the production scheme (MUSCL-Hancock + MC limiter + HLLC) reproduces that
   reference: star-region plateaus and wave positions within 2 %, pressure field
   within 2 % in relative L1, no positivity clipping.

The numbers quoted in ``docs/lax-shock-tube/report.md`` come from
``scripts/lax_shock_tube_benchmark.py``; this test is the fast, always-on gate.
"""

from __future__ import annotations

import pytest

from tensorfvm.compressible import lax

TOL = lax.THRESHOLDS


@pytest.fixture(scope="module")
def cfg():
    return lax.LaxConfig()


def test_exact_solution_is_independently_reproduced(cfg):
    """Two implementations that share no code must give the same exact solution."""
    check = lax.independent_check(cfg)
    assert check["rel_dp_star"] < TOL["independent_exact_rel_error"]
    assert check["rel_du_star"] < TOL["independent_exact_rel_error"]
    assert check["rel_d_rho_star_l"] < TOL["independent_exact_rel_error"]
    assert check["rel_d_rho_star_r"] < TOL["independent_exact_rel_error"]
    # the whole self-similar profile, not just the star state
    assert check["max_abs_profile_diff_scaled"] < TOL["independent_exact_rel_error"]


def test_rankine_hugoniot_is_satisfied(cfg):
    """First-principles check of the shock: S(U*-U) = F*-F to machine precision."""
    res = lax.rankine_hugoniot_residual(cfg)
    assert res["rel_residual_mass"] < 1e-12
    assert res["rel_residual_momentum"] < 1e-12
    assert res["rel_residual_energy"] < 1e-12


def test_lax_solution_structure(cfg):
    """Lax is a left rarefaction + contact + *right shock* (unlike Sod's weak shock)."""
    waves = lax.exact_waves(cfg)
    assert waves["left_wave"] == "rarefaction"
    assert waves["right_wave"] == "shock"
    # strong shock: p*/p_R > 4 and rho*_R/rho_R > 2.5 is what makes this case
    # a harder test than Sod for the HLLC wave-speed estimate
    assert waves["p_star"] / cfg.p_r > 4.0
    assert waves["rho_star_r"] / cfg.rho_r > 2.5
    assert waves["x_head"] < waves["x_tail"] < waves["x_contact"] < waves["x_shock"]


def test_lax_matches_exact_solution_within_two_percent(cfg):
    """nx = 400 acceptance: plateaus, wave positions and L1 fields under 2 %."""
    row = lax.run_case(400, cfg=cfg)
    for key, err in row["plateau_rel_error"].items():
        assert err < TOL["plateau_rel_error"], f"plateau {key}: {err:.4%}"
    for key, err in row["position_rel_error"].items():
        assert err is not None, f"wave feature {key} could not be located"
        assert err < TOL["wave_position_rel_error"], f"position {key}: {err:.4%}"
    for key, err in row["l1_rel"].items():
        assert err < TOL["l1_rel_error"], f"L1 {key}: {err:.4%}"
    assert row["positivity_fixes"] == 0
    assert abs(row["drift"]["mass"]) < TOL["conservation_drift"]


def test_convergence_improves_with_resolution(cfg):
    """The L1 error must fall monotonically between two grid levels."""
    coarse = lax.run_case(200, cfg=cfg)
    fine = lax.run_case(400, cfg=cfg)
    for key in ("rho", "u", "p"):
        assert fine["l1_rel"][key] < coarse["l1_rel"][key]