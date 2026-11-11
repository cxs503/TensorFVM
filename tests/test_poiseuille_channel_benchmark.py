"""Tests for the plane-channel Poiseuille benchmark (analytic targets, metrics)."""

import math

import pytest
import torch

from tensorfvm.benchmark_poiseuille_channel import (
    PLAN,
    TOLERANCE,
    observed_order,
    reference_values,
    run_case,
    summarise,
    symbolic_reference,
    wall_shear,
)


def test_symbolic_derivation_reproduces_tabulated_targets():
    """The closed form, the 3/2 peak ratio and Po = 96 come from sympy, not a table."""
    derivation = symbolic_reference()
    if not derivation.get("available"):  # sympy is an optional dependency
        pytest.skip(f"sympy unavailable: {derivation.get('reason')}")
    assert derivation["all_checks"] is True, derivation["checks"]
    assert derivation["peak_over_bar"] == "3/2"
    assert derivation["poiseuille"] == "96"
    assert derivation["cf_times_re_h"] == "12"


def test_reference_values_are_the_exact_plane_channel_relations():
    ref = reference_values(100.0)
    assert ref["nu"] == pytest.approx(0.01, rel=0)
    assert ref["dpdx"] == pytest.approx(-0.12, rel=1e-15)
    assert ref["u_max_over_bar"] == 1.5
    assert ref["tau_w"] == pytest.approx(0.06, rel=1e-15)
    assert ref["cf_fanning_times_re_h"] == pytest.approx(12.0, rel=1e-15)
    assert ref["poiseuille"] == 96.0
    # Re_dh = 2 Re_H for a plane channel of height H (D_h = 2H)
    assert ref["re_dh"] == pytest.approx(2.0 * ref["re_h"], rel=0)


def test_wall_shear_estimators_recover_the_parabola():
    """A synthetic exact profile must give tau_w = 6 mu ubar / H to round-off."""
    ny, height, mu, ubar = 32, 1.0, 0.01, 1.0
    dy = height / ny
    faces = torch.tensor([(j + 0.5) * dy for j in range(ny)], dtype=torch.float64)
    ufaces = (6.0 * ubar * faces / height * (1.0 - faces / height)).reshape(-1, 1)
    shear = wall_shear(ufaces, dy, mu, 0, height)
    exact = 6.0 * mu * ubar / height
    assert shear["tau_w_quadratic"] == pytest.approx(exact, rel=1e-13)
    # the half-cell stencil carries the documented first-order bias dy/(2H)
    assert shear["tau_w_half_cell"] == pytest.approx(exact * (1.0 - 0.5 * dy / height),
                                                     rel=1e-13)


def test_observed_order_recovers_a_known_convergence_rate():
    assert observed_order(1.0, 0.25) == pytest.approx(2.0, rel=1e-13)
    assert observed_order(0.3, 0.3 / 2 ** 1.5, 2.0) == pytest.approx(1.5, rel=1e-13)
    with pytest.raises(ValueError):
        observed_order(1.0, 0.0)
    with pytest.raises(ValueError):
        observed_order(1.0, 0.5, ratio=1.0)


def test_summarise_flags_the_finest_mesh_as_the_production_case():
    def fake(name, error):
        return {
            "case": name,
            "ny": int(name.split("ny")[-1]),
            "scored_metrics": ["velocity_l2"],
            "metrics": {"velocity_l2": error, "passed": error < TOLERANCE,
                        "pressure_gradient_relative_error": error},
        }

    results = [fake("mesh-ny8", 0.05), fake("mesh-ny16", 0.01), fake("mesh-ny32", 0.002)]
    payload = summarise(results)
    assert payload["mesh_sequence"] == ["mesh-ny8", "mesh-ny16", "mesh-ny32"]
    assert payload["production_case"] == "mesh-ny32"
    assert payload["production_passed"] is True
    assert payload["all_cases_passed"] is False  # coarse end reported honestly
    assert payload["case_verdicts"]["mesh-ny8"] is False
    assert payload["observed_orders"][0]["velocity_l2_order"] == pytest.approx(
        math.log2(5.0), rel=1e-12)


def test_tiny_case_runs_conserves_mass_and_scores_itself():
    """Smoke test on a 4x16 grid: the kernel conserves discrete mass exactly."""
    result = run_case("smoke", 4, 100.0, 4.0, max_iterations=4000, tolerance=1e-7)
    assert result["converged"] is True
    assert math.isfinite(result["metrics"]["velocity_l2"])
    assert result["metrics"]["flux_conservation_error"] < 1e-6
    assert result["metrics"]["relative_mass_imbalance"] < 1e-6
    assert len(result["profile"]["numerical"]) == result["ny"]
    assert {case[0] for case in PLAN} >= {"mesh-ny8", "mesh-ny16"}