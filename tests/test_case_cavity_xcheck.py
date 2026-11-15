"""Gate for the cross-code check of CASE-FVM-CAVITY-RE1000 (``XCHECK.md``).

The gate does not re-run the solver: it re-derives every claim from the stored
artifacts in ``docs/CASE-FVM-CAVITY-RE1000/runs/xcheck/`` and fails when

* a stored grid did not actually converge, is not divergence-free, or its
  vorticity field is not the Laplacian of its streamfunction;
* the sequence is not in an admissible asymptotic range (observed order);
* the two independent discretisations do not reach the same extrapolated limit
  within the tolerance the document claims -- i.e. the headline claim of
  ``XCHECK.md`` is checked, not merely executed;
* ``XCHECK.md`` disagrees with ``xcheck.json`` (hand-typed numbers, stale report).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parent.parent / "docs" / "CASE-FVM-CAVITY-RE1000"
OUT = CASE_DIR / "runs" / "xcheck"
PAYLOAD = OUT / "xcheck.json"
REPORT = CASE_DIR / "XCHECK.md"

MIN_GRIDS = 3
MIN_FINEST = 96          # coarse-grid-only extrapolations are not quoted as evidence
ORDER_BOUNDS = (1.5, 3.0)
CHECKED_QUANTITIES = ("u_min", "v_max")


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip("xcheck payload not generated yet "
                    "(run docs/CASE-FVM-CAVITY-RE1000/xcheck_psiomega.py)")
    return json.loads(PAYLOAD.read_text())


def test_sequence_is_long_enough_to_extrapolate(payload: dict) -> None:
    grids = payload["grids"]
    assert len(grids) >= MIN_GRIDS, grids
    assert max(grids) >= MIN_FINEST, grids
    assert sorted(grids) == grids


@pytest.mark.parametrize("n", [32, 64, 128, 192])
def test_stored_grids_are_converged_and_consistent(payload: dict, n: int) -> None:
    key = str(n)
    if key not in payload["per_grid"]:
        pytest.skip(f"grid n={n} not stored")
    grid = payload["per_grid"][key]
    ledger = grid["ledger"]
    assert ledger["converged"] is True
    assert ledger["residual"] <= 1e-7
    assert ledger["max_divergence"] < 1e-10          # psi-based velocity
    assert ledger["vorticity_defect"] < 1e-6         # omega == -lap(psi)
    f = grid["functionals"]
    assert f["u_min"] < 0.0 < f["v_max"]
    assert f["psi_min"] < 0.0


def test_observed_orders_are_admissible(payload: dict) -> None:
    for q in CHECKED_QUANTITIES:
        blk = payload["psiomega"][q]
        assert blk["data_quality"]["usable"], (q, blk["data_quality"])
        p = blk["observed_order"]
        assert ORDER_BOUNDS[0] <= p <= ORDER_BOUNDS[1], (q, p)


def test_two_discretisations_reach_the_same_limit(payload: dict) -> None:
    """This is the claim of XCHECK.md: it must hold, or the document must say FAIL."""
    tolerance = payload["agreement_tolerance"]
    differences = {}
    for q in CHECKED_QUANTITIES:
        row = payload["comparison"][q]
        assert "cross_code_relative_difference" in row, q
        differences[q] = row["cross_code_relative_difference"]
    worst = max(differences[q] for q in payload["governed_quantities"])
    assert math.isclose(worst, payload["worst_cross_code_difference"], rel_tol=1e-12)
    assert payload["limit_is_code_independent"] is (worst <= tolerance)
    # the non-governed functionals are reported, not hidden
    assert payload["cross_code_difference_per_quantity"]["kinetic"] is not None
    assert worst <= tolerance, f"cross-code disagreement {100 * worst:.3f}% > {100 * tolerance:.1f}%"


def test_reference_offset_is_smaller_than_the_reference_uncertainty_claim(payload: dict) -> None:
    """The residual offset vs Ghia must stay inside the pool tolerance (5 %)."""
    for q in CHECKED_QUANTITIES:
        offset = payload["comparison"][q]["psiomega_extrapolated_vs_reference"]
        assert abs(offset) < 0.05, (q, offset)


def test_report_numbers_are_injected_not_typed(payload: dict) -> None:
    assert REPORT.exists()
    text = REPORT.read_text()
    assert payload["case"] in text
    for q in CHECKED_QUANTITIES:
        value = payload["comparison"][q]["psiomega_extrapolated"]
        assert f"{value:.6f}" in text, (q, value)
        value = payload["comparison"][q]["production_extrapolated"]
        assert f"{value:.6f}" in text, (q, value)
    # the verdict statement must be machine-checkable and must not be reworded by hand
    assert "**判定语句（机械可查）**" in text
    expected = "PASS" if payload["limit_is_code_independent"] else "honest-FAIL"
    assert f"当前判定 = **{expected}**" in text