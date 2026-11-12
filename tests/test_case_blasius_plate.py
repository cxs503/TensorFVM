"""Tests for CASE-FVM-BLASIUS-PLATE (``tensorfvm.case_blasius_plate``)."""

import math

import numpy as np
import pytest

from tensorfvm.case_blasius_plate import (
    ERROR_LIMIT, LITERATURE, REFERENCE_RE_X, blasius_reference,
    boundary_layer_integrals, compare_against_reference, run_fvm_case,
    run_march_case, similarity_crosscheck, wall_shear_profile,
)


def test_literature_reference_values():
    ref = blasius_reference(REFERENCE_RE_X)
    assert ref["cf"] == pytest.approx(0.664 / 10.0 ** 2.5, rel=1e-12)
    assert ref["cf"] == pytest.approx(0.002099752366351804, rel=1e-12)
    assert ref["delta_star"] == pytest.approx(1.7208 / 10.0 ** 2.5, rel=1e-12)
    assert ref["H"] == pytest.approx(2.5916, rel=1e-12)
    # the tabulated H must be the ratio of the tabulated thicknesses
    assert LITERATURE["delta_star_sqrt_re_x"] / LITERATURE["theta_sqrt_re_x"] == \
        pytest.approx(LITERATURE["shape_factor"], rel=1e-4)
    with pytest.raises(ValueError):
        blasius_reference(0.0)


def test_similarity_crosscheck_reproduces_table_digits():
    cross = similarity_crosscheck()
    assert cross["fpp0"] == pytest.approx(0.3320573362, rel=1e-9)
    # tabulated 0.664 and 2.5916 must be reproducible from the ODE here
    assert cross["cf_constant_relative_deviation"] < 1e-3
    assert cross["shape_factor_relative_deviation"] < 1e-3
    assert cross["delta_star_sqrt_re_x"] == pytest.approx(1.7208, rel=1e-3)
    assert cross["theta_sqrt_re_x"] == pytest.approx(0.664, rel=1e-3)


def test_boundary_layer_integrals_on_exact_blasius_profile():
    """Integrating the similarity profile must return the tabulated thicknesses."""
    from tensorfvm.rans_ship_resistance.anchors import blasius_solution

    x, u_inf, nu = 1.0, 1.0, 1e-5
    re_x = u_inf * x / nu
    sol = blasius_solution(eta_max=12.0, n=1201)
    eta = np.asarray(sol["eta"])
    fp = np.asarray(sol["fp"])
    y = eta * math.sqrt(nu * x / u_inf)
    u = fp * u_inf
    got = boundary_layer_integrals(y, u, u_e=u_inf)
    ref = blasius_reference(re_x)
    assert got["delta_star"] == pytest.approx(ref["delta_star"], rel=2e-3)
    assert got["theta"] == pytest.approx(ref["theta"], rel=2e-3)
    assert got["H"] == pytest.approx(ref["H"], rel=2e-3)
    assert got["delta_99"] == pytest.approx(4.91 * math.sqrt(nu * x / u_inf), rel=1e-2)
    # truncating at delta_99 biases theta low (documented ~0.9 % on this profile)
    truncated = boundary_layer_integrals(y, u, u_e=u_inf, truncate_at_delta_99=True)
    assert truncated["theta"] < got["theta"]
    assert truncated["theta"] / ref["theta"] == pytest.approx(0.991, abs=5e-3)


def test_boundary_layer_integrals_rejects_bad_input():
    with pytest.raises(ValueError):
        boundary_layer_integrals([0.0, 0.1], [0.0, 0.5])
    with pytest.raises(ValueError):
        boundary_layer_integrals([0.0, 0.2, 0.1], [0.0, 0.5, 1.0])
    # profile that never reaches 0.99 u_e inside the domain
    with pytest.raises(ValueError):
        boundary_layer_integrals([0.0, 0.1, 0.2], [0.0, 0.1, 0.2], u_e=1.0)


def test_compare_against_reference_gate():
    ref = blasius_reference()
    ok = compare_against_reference(ref["cf"] * 1.01, ref["H"] * 0.99)
    assert ok["cf"]["passed"] and ok["H"]["passed"] and ok["passed"]
    bad = compare_against_reference(ref["cf"] * 1.05, ref["H"])
    assert not bad["cf"]["passed"] and not bad["passed"]
    assert bad["cf"]["relative_error"] == pytest.approx(0.05, rel=1e-9)
    assert ERROR_LIMIT == 0.03


def test_march_case_matches_blasius_within_gate():
    """The affordable rung is cheap; assert the case verdict directly."""
    result = run_march_case()
    assert result["measured"]["re_x"] == pytest.approx(REFERENCE_RE_X, rel=1e-9)
    assert abs(result["comparison"]["cf"]["relative_error"]) < ERROR_LIMIT
    assert abs(result["comparison"]["H"]["relative_error"]) < ERROR_LIMIT
    assert result["comparison"]["passed"]


def test_fvm_smoke_extraction():
    """Tiny grid, one iteration: mechanics of the FVM extraction only."""
    result = run_fvm_case(nx=16, ny=12, max_iterations=1, trace_every=1,
                          threads=1, progress=False)
    assert result["device"] == "fvm"
    assert result["grid"]["wall_faces"] == 16
    # per-face forces must sum to the whole-plate force the solver reports
    assert result["grid"]["force_sum_closure"] < 1e-12
    assert result["measured"]["cf"] > 0.0
    assert result["measured"]["H"] > 1.0
    assert len(result["trace"]) == 1
    assert result["run"]["converged"] is False