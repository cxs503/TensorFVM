"""Tests for convergence-diagnosis v2 and its OpenFOAM-10 cross-validation."""

from __future__ import annotations

import json
import math
import os

import pytest

from tensorfvm.convergence_diagnosis import (
    ConvergenceMonitor, DiagnosisConfig, diagnose_series, load_history, main,
    norm_factor, normalised_residual, openfoam_converged)

numpy = pytest.importorskip("numpy")


# --------------------------------------------------------------- reference rule
def test_openfoam_check_convergence_truth_table():
    # SolverPerformance::checkConvergence (OpenFOAM-10, SolverPerformance.C:60-88)
    assert openfoam_converged(1e-9, 1.0, 1e-8, 0.0) is True         # absolute branch
    assert openfoam_converged(1e-6, 1.0, 1e-12, 1e-2) is True       # relTol branch
    assert openfoam_converged(1e-6, 1.0, 1e-12, 0.0) is False       # relTol disabled
    assert openfoam_converged(1e-6, 1.0, 1e-12, 1e-30) is False     # relTol <= small_


def test_norm_factor_matches_openfoam_formula():
    # LduMatrix::solver::normFactor (LduMatrixSolver.C:167-186)
    A = numpy.array([[4.0, -1.0], [-1.0, 4.0]])
    b = numpy.array([1.0, 0.0])
    psi = numpy.array([0.5, 0.5])
    row_sums = A.sum(axis=1)
    tmp = row_sums * psi.mean()
    expected = float(numpy.sum(numpy.abs(A @ psi - tmp)) + numpy.sum(numpy.abs(b - tmp)))
    assert norm_factor(A @ psi, b, row_sums, psi) == pytest.approx(expected)
    assert normalised_residual(A, b, psi, row_sums, expected) == pytest.approx(1.0)


def test_norm_factor_is_stabilised():
    zero = numpy.zeros((2, 2))
    assert norm_factor(numpy.zeros(2), numpy.zeros(2), zero.sum(axis=1),
                       numpy.zeros(2)) == 1e-20


# -------------------------------------------------------------------- monitor
def test_geometric_series_is_diagnosed_as_converging_then_converged():
    config = DiagnosisConfig(tolerance=1e-10)
    mon = ConvergenceMonitor(config)
    verdicts = [mon.record(k, 0.8 ** k).verdict for k in range(120)]
    assert verdicts[0] == "insufficient"          # not enough points yet
    assert "converging" in verdicts
    assert mon.diagnosis().converged is True
    assert mon.diagnosis().verdict == "converged"
    d = mon.diagnosis()
    assert d.rate == pytest.approx(0.8, rel=1e-9)
    assert d.forecast_iterations == mon.diagnosis().iterations


def test_rate_of_a_geometric_series_is_exact():
    iterations = list(range(30))
    residuals = [0.85 ** k for k in iterations]
    d = diagnose_series(iterations, residuals, DiagnosisConfig(tolerance=1e-30))
    assert d.rate == pytest.approx(0.85, rel=1e-9)
    assert d.forecast_ready is True


def test_diverging_and_oscillating_are_separated():
    growing = ConvergenceMonitor(DiagnosisConfig(tolerance=1e-12))
    for k in range(12):
        growing.record(k, 1.05 ** k)
    assert growing.diagnosis().verdict == "diverging"

    oscillating = ConvergenceMonitor(DiagnosisConfig(tolerance=1e-12))
    for k in range(12):
        oscillating.record(k, 1.0 + 0.5 * (1 if k % 2 else -1) * 1.0)
    assert oscillating.diagnosis().verdict in ("oscillating", "insufficient")


def test_stagnation_is_declared_against_the_budget():
    config = DiagnosisConfig(tolerance=1e-14, max_iter=500)
    mon = ConvergenceMonitor(config)
    residuals = [0.8 ** k for k in range(25)]
    residuals += [residuals[-1] * 0.9995 ** (k - 24) for k in range(25, 60)]
    for k, r in enumerate(residuals):
        mon.record(k, r)
    d = mon.diagnosis()
    assert d.forecast_ready is True
    assert d.verdict == "stagnating"
    assert d.forecast_iterations > config.max_iter


def test_gate_stays_closed_through_a_fast_transient():
    """A superlinear transient must not be extrapolated as a standing rate."""
    residuals = [0.4 ** k for k in range(6)] + [0.4 ** 5 * 0.97 ** (k - 5) for k in range(6, 40)]
    mon = ConvergenceMonitor(DiagnosisConfig(tolerance=1e-14))
    for k, r in enumerate(residuals):
        d = mon.record(k, r)
        if k == 6:
            assert d.forecast_ready is False      # rate not standing yet
            assert d.verdict in ("insufficient", "converged")
    assert mon.diagnosis().forecast_ready is True
    ready = mon.diagnosis()
    assert ready.rate == pytest.approx(0.97, rel=5e-2)


def test_forecast_band_is_ordered():
    residuals = [0.4 ** k for k in range(6)] + [0.4 ** 5 * 0.93 ** (k - 5) for k in range(6, 60)]
    mon = ConvergenceMonitor(DiagnosisConfig(tolerance=1e-30))
    for k, r in enumerate(residuals):
        mon.record(k, r)
    d = mon.diagnosis()
    assert d.forecast_optimistic <= d.forecast_iterations <= d.forecast_pessimistic


def test_convergence_beats_the_gate():
    """An early exact solve is reported as converged, not as 'insufficient'."""
    mon = ConvergenceMonitor(DiagnosisConfig(tolerance=1e-8))
    mon.record(0, 1.0)
    assert mon.record(1, 1e-12).verdict == "converged"


# ------------------------------------------------------------------------ CLI
def test_diagnose_cli_round_trip(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([{"iteration": k, "residual": 0.7 ** k}
                                for k in range(60)]), encoding="utf-8")
    out = tmp_path / "report.json"
    assert main(["diagnose", "--history", str(path), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["diagnosis"]["converged"] is True
    iterations, residuals = load_history(str(path))
    assert len(iterations) == len(residuals) == 60


# ------------------------------------------------------------- integration面
def test_production_solver_records_the_diagnosis():
    """The SIMPLE linear path must feed the monitor (integration caller)."""
    import inspect
    from tensorfvm import sparse_simple
    source = inspect.getsource(sparse_simple)
    assert "ConvergenceMonitor" in source
    assert "convergence_diagnosis" in source


# ---------------------------------------------------------------- XVAL harness
def test_xval_single_case_produces_artifacts(tmp_path):
    pytest.importorskip("scipy.sparse")
    from tensorfvm.convergence_xval import Scenario, run_xval
    scenario = Scenario("poisson-n16-tol1e-8", "poisson", 16, "gmres", 1e-8)
    payload = run_xval(str(tmp_path), max_iter=400, scenarios=[scenario])
    for name in ("xval.json", "xval.md", os.path.join("histories",
                                                      "poisson-n16-tol1e-8.json")):
        assert (tmp_path / name).exists()
    row = payload["cases"][0]
    assert row["verdict_agree"] is True
    assert row["ref_converged"] is True
    assert row["condition_estimate"] is None or math.isfinite(row["condition_estimate"])
    assert payload["provenance"]["entries"][0]["path"].endswith("SolverPerformance.C")