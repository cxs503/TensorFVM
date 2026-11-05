"""Guards for the SST-2003 constant algebra audit.

The audit under ``scripts/audit_sst_log_law_constants.py`` claims that the
implemented constant families imply a log-law slope within 0.5 % of the 0.41
that the wall functions assume.  These tests keep the claim honest:

* the symbolic identity must close (it is re-derived, not quoted);
* the numeric leg must reproduce the closed form through the *repository
  closure* and converge at the discretisation order;
* a 1 % edit of any single constant must move the implied slope detectably,
  otherwise the audit could never fail;
* the checked-in evidence must match a fresh run (no hand-copied numbers).
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import audit_sst_log_law_constants as audit  # noqa: E402
from tensorfvm import native_sst3d as sst  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return audit.run()


def test_k_budget_of_the_log_layer_ansatz_closes_identically():
    identity = audit.symbolic_identity()
    assert identity["k_budget_closes_identically"] is True
    assert identity["identity_closes_symbolically"] is True


def test_closed_form_is_the_one_solved_from_the_omega_budget():
    """kappa^2 == (beta/sqrt(beta*) - gamma sqrt(beta*))/sigma_omega."""
    expected = (sst.BETA1 / np.sqrt(sst.BETA_STAR) - sst.GAMMA1 * np.sqrt(sst.BETA_STAR)) / sst.SIGMA_W1
    kappa, kappa_squared = audit.closed_form_kappa(sst.BETA1, sst.GAMMA1, sst.SIGMA_W1)
    assert kappa_squared == pytest.approx(expected, rel=1e-15)
    assert kappa == pytest.approx(np.sqrt(expected), rel=1e-15)


def test_repository_closure_eddy_viscosity_is_kappa_u_tau_y():
    assert audit.nu_t_matches_kappa_u_tau_y() < 1e-12


def test_numeric_root_matches_the_closed_form(result):
    numeric = result["numeric"]
    assert numeric["root_kappa"] == pytest.approx(numeric["closed_form_kappa"], abs=1e-3)
    assert abs(numeric["closed_form_minus_root"]) < 1e-3


def test_numeric_root_converges_at_the_difference_order(result):
    gaps = [abs(row["gap_to_closed_form"]) for row in result["numeric"]["convergence_ladder"]]
    assert all(later < earlier for earlier, later in zip(gaps, gaps[1:])), gaps
    orders = [np.log2(earlier / later) for earlier, later in zip(gaps, gaps[1:])]
    assert all(order > 1.5 for order in orders), orders


def test_families_bracket_the_wall_function_slope(result):
    inner = result["families"]["inner_k_omega"]["kappa"]
    outer = result["families"]["outer_k_epsilon"]["kappa"]
    wall = result["constants"]["wall_kappa"]
    assert inner < wall < outer
    assert result["family_spread_percent"] < 0.5
    assert abs(100.0 * (inner - wall) / wall) < 0.5


def test_inner_family_governs_the_log_layer(result):
    """grad(k) vanishes there, so the cross diffusion drops out and F1 -> 1."""
    assert result["log_layer"]["f1_min"] > 0.99
    assert result["log_layer"]["governing_family"] == "inner_k_omega"


def test_constants_are_consistent_with_the_spalding_wall_law(result):
    """The mismatch must stay well inside the 3 % acceptance band."""
    rows = result["wall_law_consequence"]
    assert max(abs(row["cf_bias_percent"]) for row in rows) < 3.0
    assert all(row["cf_bias_percent"] > 0 for row in rows)


def test_eddy_viscosity_limiter_keeps_a_positive_margin(result):
    limiter = result["limiter"]
    assert limiter["a1"] > limiter["sqrt_beta_star"]
    assert limiter["ratio"] == pytest.approx(sst.A1 / np.sqrt(sst.BETA_STAR), rel=1e-15)
    assert 0 < limiter["margin_percent"] < 10


def test_audit_detects_a_one_percent_constant_edit(result):
    controls = result["negative_controls"]
    assert len(controls) == 5
    for control in controls:
        assert abs(control["relative_change_percent"]) > 0.1, control


def test_forcing_the_wall_function_slope_breaks_the_omega_budget(result):
    """Evaluating the budget at kappa = 0.41 must be visibly worse than at its root."""
    numeric = result["numeric"]
    assert numeric["relative_residual_at_wall_kappa"] > 1e-3
    assert (numeric["relative_residual_at_wall_kappa"]
            > 5 * numeric["relative_residual_at_root"])


def test_every_gate_in_the_shipped_artifact_passed(result):
    assert all(result["gates"].values()), [k for k, v in result["gates"].items() if not v]


def test_published_evidence_matches_a_fresh_run(result):
    path = os.path.join(ROOT, "docs", "verification-sst-log-law-constants", "audit.json")
    if not os.path.exists(path):
        pytest.skip("evidence not generated in this checkout")
    with open(path) as fh:
        stored = json.load(fh)
    assert stored["schema"] == result["schema"]
    assert stored["constants"] == result["constants"]
    for name in ("inner_k_omega", "outer_k_epsilon"):
        assert stored["families"][name]["kappa"] == pytest.approx(
            result["families"][name]["kappa"], rel=0, abs=0)
    assert stored["numeric"]["root_kappa"] == pytest.approx(
        result["numeric"]["root_kappa"], rel=0, abs=0)


def test_script_is_importable_without_running_the_driver():
    spec = importlib.util.find_spec("audit_sst_log_law_constants")
    assert spec is not None and spec.origin.endswith("audit_sst_log_law_constants.py")