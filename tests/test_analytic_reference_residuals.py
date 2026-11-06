"""Guards for the analytic reference fields that every FVM acceptance band rests on.

These tests audit the *verifiers*: each closed form used as a pass/fail
baseline is re-derived symbolically and compared with the shipped function.
A sign or transcription slip here would invalidate downstream "<3 %" claims
while leaving every solver test green, so the guards are kept separate from
the solver tests.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import audit_analytic_reference_residuals as audit  # noqa: E402

SYMBOLIC_CASES = {
    "native_order3d": audit.case_native_order3d,
    "annular": audit.case_annular,
    "abc": audit.case_abc,
    "advected_shear": audit.case_advected_shear,
    "overset_poisson": audit.case_overset_poisson,
    "curved_channel": audit.case_curved_channel,
    "taylor_green": audit.case_taylor_green,
}


@pytest.fixture(scope="module")
def results():
    return {name: fn() for name, fn in SYMBOLIC_CASES.items()}


@pytest.mark.parametrize("name", sorted(SYMBOLIC_CASES))
def test_symbolic_identity_is_exact(results, name):
    case = results[name]
    zero = {k: v for k, v in case["symbolic"].items()}
    assert case["symbolic_zero"], (name, {k: str(v) for k, v in zero.items() if v != 0})


@pytest.mark.parametrize("name", sorted(SYMBOLIC_CASES))
def test_numeric_containment(results, name):
    case = results[name]
    assert case["numeric_max_rel"] < audit.TOL_NUMERIC, (name, case["numeric_per_component"])


def test_torch_reference_fields_match_symbolic_solution():
    case = audit.case_torch_references()
    assert case["numeric_max_rel"] < audit.TOL_NUMERIC, case["numeric_per_component"]


def test_derivative_limit_is_second_order():
    rows, slopes = audit.derivative_limit()
    # central differences must converge onto the shipped analytic gradient
    assert rows[-1]["max_abs_error"] < rows[0]["max_abs_error"] / 1e3
    assert all(1.8 < s < 2.2 for s in slopes), slopes


def test_published_results_are_in_sync():
    """If the checked-in evidence exists, it must agree with a fresh audit run."""
    path = os.path.join(ROOT, "docs", "verification-analytic-reference-residuals", "results.json")
    if not os.path.exists(path):
        pytest.skip("no checked-in evidence")
    with open(path) as fh:
        stored = json.load(fh)
    fresh = {c["case"]: c for c in (fn() for fn in SYMBOLIC_CASES.values())}
    for entry in stored["cases"]:
        case = fresh.get(entry["case"])
        if case is None:
            continue
        assert case["symbolic_zero"] == entry["symbolic_zero"]
        assert abs(case["numeric_max_rel"] - entry["numeric_max_rel"]) < 1e-12