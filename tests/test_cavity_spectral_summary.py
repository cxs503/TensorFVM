"""Gate for spectral summary completeness (CASE-FVM-CAVITY-RE1000).

Ensures the spectral summary JSON carries the extrapolation and comparison
data that the COMPREHENSIVE_REPORT.md and SPECTRAL.md quote, so that a
reader can verify every number without opening ``spectral_report.json``.

    PYTHONPATH=src python -m pytest tests/test_cavity_spectral_summary.py -q
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parent.parent / "docs" / "CASE-FVM-CAVITY-RE1000"
SPECTRAL = CASE_DIR / "runs" / "spectral"
SUMMARY_1000 = SPECTRAL / "cavity_spectral_summary_re1000.json"
REPORT = SPECTRAL / "spectral_report.json"


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY_1000.exists():
        pytest.skip("spectral summary not generated yet")
    return json.loads(SUMMARY_1000.read_text())


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.exists():
        pytest.skip("spectral report not generated yet")
    return json.loads(REPORT.read_text())


def test_summary_has_extrapolation(summary: dict) -> None:
    ext = summary.get("extrapolation")
    assert ext is not None, "extrapolation must not be null"
    for q in ("u_min", "v_max", "psi_min"):
        assert q in ext, f"extrapolation must include {q}"
        assert "limit" in ext[q], f"extrapolation[{q}] must have a limit"
        assert "pairs" in ext[q], f"extrapolation[{q}] must have pairs"
        assert len(ext[q]["pairs"]) >= 3, f"need >= 3 pairs for {q}"


def test_summary_has_vs_ghia(summary: dict) -> None:
    vg = summary.get("vs_ghia")
    assert vg is not None, "vs_ghia must not be null"
    for key in ("u_min_rel_err", "v_max_rel_err", "u_profile_max_abs",
                "v_profile_max_abs", "within_5pct", "finest_order"):
        assert key in vg, f"vs_ghia must include {key}"
    assert vg["within_5pct"] is True, "finest spectral grid must be within 5% of Ghia"


def test_summary_has_vs_fvm(summary: dict) -> None:
    vf = summary.get("vs_fvm")
    assert vf is not None, "vs_fvm must not be null"
    for q in ("u_min", "v_max"):
        assert q in vf, f"vs_fvm must include {q}"
        for key in ("spectral_limit", "fvm_richardson_limit",
                     "spectral_vs_fvm_rel", "spectral_vs_ghia_rel", "fvm_vs_ghia_rel"):
            assert key in vf[q], f"vs_fvm[{q}] must include {key}"


def test_spectral_vs_fvm_agreement(summary: dict) -> None:
    """The headline claim: spectral and FVM limits agree to < 0.5%."""
    vf = summary["vs_fvm"]
    for q in ("u_min", "v_max"):
        rel = vf[q]["spectral_vs_fvm_rel"]
        assert rel < 0.005, (q, f"spectral vs FVM {100*rel:.3f}% > 0.5%")


def test_summary_consistent_with_report(summary: dict, report: dict) -> None:
    """Every number in the summary must match the report (single source of truth)."""
    ext_s = summary["extrapolation"]
    ext_r = report["extrapolation"]
    for q in ("u_min", "v_max", "psi_min"):
        assert math.isclose(ext_s[q]["limit"], ext_r[q]["limit"], rel_tol=1e-10), (q, "limit mismatch")

    vf_s = summary["vs_fvm"]
    vf_r = report["verdict"]
    for q in ("u_min", "v_max"):
        assert math.isclose(vf_s[q]["spectral_limit"], vf_r[q]["spectral_limit"], rel_tol=1e-10)
        assert math.isclose(vf_s[q]["fvm_richardson_limit"], vf_r[q]["fvm_richardson_limit"], rel_tol=1e-10)
        assert math.isclose(vf_s[q]["spectral_vs_fvm_rel"], vf_r[q]["spectral_vs_fvm_rel"], rel_tol=1e-10)


def test_summary_records_match_files(summary: dict) -> None:
    """The files listed in the summary must exist."""
    for fname in summary["files"]:
        assert (SPECTRAL / fname).exists(), f"missing spectral file: {fname}"


def test_summary_orders_match_records(summary: dict) -> None:
    assert summary["orders"] == [r["order"] for r in summary["records"]]


def test_finest_spectral_grid_passes_5pct(summary: dict) -> None:
    """The finest converged spectral grid must be within 5% of Ghia."""
    vg = summary["vs_ghia"]
    assert vg["u_min_rel_err"] < 0.05
    assert vg["v_max_rel_err"] < 0.05
    assert vg["within_5pct"] is True
