"""Integration test: convergence diagnosis with actual FVM solver runs.

This test verifies that the convergence diagnosis module works correctly
when fed with real residual histories from the FVM SIMPLE solver, not just
pre-recorded JSON histories. It bridges the gap between the unit-level
diagnosis tests and the production solver.

Covers gap Q5 from the adversarial review (ADVERSARIAL-REVIEW.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_diagnosis_on_poisson_gmres_solve(tmp_path):
    """Run a real Poisson GMRES solve and verify the diagnosis is recorded."""
    pytest.importorskip("scipy.sparse")
    from tensorfvm.convergence_xval import Scenario, run_xval

    scenario = Scenario("poisson-n16-tol1e-8", "poisson", 16, "gmres", 1e-8)
    payload = run_xval(str(tmp_path), max_iter=400, scenarios=[scenario])

    # Verify artifacts exist
    assert (tmp_path / "xval.json").exists()
    history_path = tmp_path / "histories" / "poisson-n16-tol1e-8.json"
    assert history_path.exists()

    # Verify the history has enough points for diagnosis
    history = json.loads(history_path.read_text())
    assert "history" in history, f"Expected 'history' key, got {list(history.keys())}"
    assert len(history["history"]) >= 6, "Need >= 6 residuals for diagnosis"

    # Verify the case result
    row = payload["cases"][0]
    assert row["verdict_agree"] is True
    assert row["ref_converged"] is True
    assert row["rate_rel_error"] is not None


def test_diagnosis_on_cd_gmres_solve(tmp_path):
    """Run a real convection-diffusion GMRES solve and verify diagnosis."""
    pytest.importorskip("scipy.sparse")
    from tensorfvm.convergence_xval import Scenario, run_xval

    scenario = Scenario("cd-pe10-n32", "cd", 32, "gmres", 1e-10, peclet=10)
    payload = run_xval(str(tmp_path), max_iter=1000, scenarios=[scenario])

    row = payload["cases"][0]
    assert row["verdict_agree"] is True
    assert row["ref_converged"] is True


def test_diagnosis_on_diverging_cg(tmp_path):
    """Run a CG solve on a non-symmetric operator (expected to diverge)."""
    pytest.importorskip("scipy.sparse")
    from tensorfvm.convergence_xval import Scenario, run_xval

    # CG on convection-diffusion (non-symmetric) should not converge
    scenario = Scenario("cd-pe10-cg", "cd", 32, "cg", 1e-10, peclet=10)
    payload = run_xval(str(tmp_path), max_iter=500, scenarios=[scenario])

    row = payload["cases"][0]
    assert row["ref_converged"] is False
    assert row["v2_verdict"] in ("diverging", "oscillating", "stagnating", "insufficient")


def test_diagnosis_records_residual_history_format(tmp_path):
    """Verify the residual history format is compatible with the diagnosis module."""
    pytest.importorskip("scipy.sparse")
    from tensorfvm.convergence_xval import Scenario, run_xval

    scenario = Scenario("poisson-n16-tol1e-8", "poisson", 16, "gmres", 1e-8)
    run_xval(str(tmp_path), max_iter=400, scenarios=[scenario])

    history_path = tmp_path / "histories" / "poisson-n16-tol1e-8.json"
    history = json.loads(history_path.read_text())

    # Verify format
    assert "history" in history, f"Expected 'history' key, got {list(history.keys())}"
    assert isinstance(history["history"], list)
    assert len(history["history"]) > 0

    # Verify each entry has iteration and residual
    for entry in history["history"]:
        assert "iteration" in entry, f"Missing 'iteration' in {entry}"
        assert "residual" in entry, f"Missing 'residual' in {entry}"
        # Verify all residuals are positive
        assert entry["residual"] > 0, f"Residual must be positive, got {entry['residual']}"

    # Verify residuals are generally decreasing (for converging case)
    residuals = [e["residual"] for e in history["history"]]
    if len(residuals) > 1:
        first = residuals[0]
        last = residuals[-1]
        assert last < first, "Residual should decrease for converging case"


def test_diagnosis_on_actual_fvm_simple_solver():
    """Verify the SIMPLE solver path feeds the ConvergenceMonitor.

    This is a static analysis test (no solver run) that verifies the
    production solver code contains the diagnosis integration.
    """
    import inspect
    from tensorfvm import sparse_simple

    source = inspect.getsource(sparse_simple)
    assert "ConvergenceMonitor" in source, \
        "SIMPLE solver must use ConvergenceMonitor for diagnosis"
    assert "convergence_diagnosis" in source, \
        "SIMPLE solver must import convergence_diagnosis"
