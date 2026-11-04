"""Reference-table consistency audit for the Ghia (1982) lid-cavity ladder.

The m3003 ladder report (§5.1) recorded one red row: at Re = 400 the v-centreline
residual *grows* under refinement (0.115 / 0.142 / 0.149 for n = 32 / 64 / 128)
while u and both extrema converge.  These tests pin the attribution: the residual
saturates at a grid-independent value at exactly one published point while the same
solver reproduces every other published point, so the entry - not the solver - is
the defective object.

Everything here is derived from the committed ladder bundle, so the tests are
cheap; they skip when the evidence tree is not checked out (``evidence_support``).
Nothing is re-solved, and the Re = 400 verdict is *not* flipped to green by
deleting the number: the digitised value stays, and scoring reports both the
all-point and the consistent-point population.
"""

from __future__ import annotations

import json

import pytest
import torch

from tensorfvm.lid_cavity import ghia_acceptance, score_profiles
from tensorfvm.lid_cavity_reference import (
    SUSPECT_ENTRIES,
    TABLE_PROVENANCE,
    consistent_points,
    self_check,
    suspect_entries,
    table,
)
from tensorfvm.verification.ghia_reference_audit import audit_bundle

from tests.evidence_support import published

BUNDLE = "docs/verification-lid-cavity-ladder/summary.json"


def _bundle() -> dict:
    with open(published(BUNDLE)) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def bundle() -> dict:
    return _bundle()


@pytest.fixture(scope="module")
def audit(bundle) -> dict:
    return audit_bundle(bundle)


def _case(bundle: dict, reynolds: float, n: int) -> dict:
    return {(c["reynolds"], c["n"]): c for c in bundle["cases"]}[(reynolds, n)]


def _midplane_v(bundle: dict, reynolds: float, n: int) -> torch.Tensor:
    """v on y = L/2 from the cached staggered field (axis 0 = x cell, axis 1 = y face)."""
    field = _case(bundle, reynolds, n)["field"]["v"]
    return torch.tensor([row[n // 2] for row in field], dtype=torch.float64)


def _midplane_u(bundle: dict, reynolds: float, n: int) -> torch.Tensor:
    """u on x = L/2 (axis 0 = x face, axis 1 = y cell)."""
    field = _case(bundle, reynolds, n)["field"]["u"]
    return torch.tensor([value for value in field[n // 2]], dtype=torch.float64)


# --------------------------------------------------------------------------- #
# provenance: the transcription is declared, and its blind spot is named
# --------------------------------------------------------------------------- #

def test_provenance_declares_the_transcription_and_its_blind_spot():
    assert "digital copy" in TABLE_PROVENANCE["transcription"]
    assert "interior entries are otherwise unverified" in TABLE_PROVENANCE["self_check_coverage"]
    assert "audit_ghia_reference_consistency" in TABLE_PROVENANCE["consistency_audit"]


def test_self_check_passes_and_is_needed_but_not_sufficient():
    """``self_check`` validates the extrema only, so the bad interior entry passes it."""
    report = self_check()
    assert report["ok"] is True
    bad = SUSPECT_ENTRIES[(400.0, "v")][0.9063]
    # the value is untouched in the table: reproducibility of every older bundle wins
    assert dict(table(400.0, "v"))[0.9063] == bad["as_digitized"] == -0.23827


# --------------------------------------------------------------------------- #
# the audit: exactly one point cannot be explained by resolution
# --------------------------------------------------------------------------- #

def test_audit_reproduces_the_committed_scoring(audit, bundle):
    """The audit's own interpolation must match the published ladder numbers."""
    for re in (100.0, 400.0, 1000.0):
        block = audit["reynolds"][str(re)]
        for which in ("u", "v"):
            n = block["grids"][-1]
            published_error = _case(bundle, re, n)[which]["max_abs_error"]
            assert block["which"][which]["max_abs_error"][str(n)] == pytest.approx(
                published_error, abs=2e-5)


def test_exactly_one_published_point_is_a_reference_defect(audit):
    defects = audit["verdict"]["reference_defects"]
    assert [(f["reynolds"], f["which"], f["x"]) for f in defects] == [(400.0, "v", 0.9063)]
    counts = audit["verdict"]["counts"]
    assert counts.get("consistent", 0) == 101  # 17 points x 2 profiles x 3 Re - 1
    assert counts.get("under_resolved", 0) == 0


def test_defect_residual_is_grid_independent_while_its_neighbours_converge(audit):
    rows = {r["x"]: r for r in audit["reynolds"]["400.0"]["which"]["v"]["points"]}
    bad = rows[0.9063]
    assert bad["ratio_prev_over_finest"] < 1.5, bad["ratio_prev_over_finest"]
    assert bad["residual_over_uncertainty"] > 10.0
    assert bad["convergence"] == "saturated"
    others = [abs(r["delta"]["128"]) for x, r in rows.items() if x != 0.9063]
    assert max(others) < 0.005


def test_the_flagged_curve_is_a_second_order_limit_not_a_spike(audit):
    """-0.3528 / -0.3806 / -0.3876: increments shrink by 4x, i.e. clean 2nd order."""
    row = next(r for r in audit["reynolds"]["400.0"]["which"]["v"]["points"]
               if r["x"] == 0.9063)
    fine, mid, coarse = row["computed"]["128"], row["computed"]["64"], row["computed"]["32"]
    step_coarse, step_fine = abs(mid - coarse), abs(fine - mid)
    assert step_coarse / step_fine > 3.0, step_coarse / step_fine
    assert fine == pytest.approx(-0.3876, abs=2e-3)


def test_shape_control_is_reported_but_cannot_decide_alone(audit):
    """Genuine published near-extrema produce the same slope jumps: no power."""
    rows = {r["x"]: r for r in audit["reynolds"]["400.0"]["shape"]["rows"]}
    assert rows[0.2344]["slope_jump_ratio"] > rows[0.9063]["slope_jump_ratio"]
    assert audit["verdict"]["shape_test_used_for_verdict"] is False


# --------------------------------------------------------------------------- #
# scoring: both populations are reported, none is hidden
# --------------------------------------------------------------------------- #

def test_suspect_entry_is_excluded_from_the_consistent_population():
    assert 0.9063 in suspect_entries(400.0, "v")
    kept = consistent_points(400.0, "v")
    assert len(kept) == 16 and all(p[0] != 0.9063 for p in kept)
    assert consistent_points(400.0, "v", include_suspect=True) == table(400.0, "v")
    # no other table is touched
    for reynolds in (100.0, 1000.0):
        for which in ("u", "v"):
            assert consistent_points(reynolds, which) == table(reynolds, which)


def test_scoring_reports_all_points_and_the_consistent_subset(bundle):
    n = 128
    coords = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    block = score_profiles(coords, _midplane_v(bundle, 400.0, n), 400.0, "v")
    assert block["max_abs_error"] == pytest.approx(0.14931, abs=2e-4)
    assert block["max_abs_error_consistent"] < 0.005
    assert block["consistent_point_count"] == 16
    assert len(block["suspect_points"]) == 1
    flagged = block["suspect_points"][0]
    assert flagged["coord"] == 0.9063
    assert flagged["delta"] == pytest.approx(-0.14931, abs=2e-4)
    assert flagged["evidence"].endswith("results.json")


def test_re400_row_is_red_only_through_the_explained_reference_point(bundle):
    n = 128
    coords = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    case = _case(bundle, 400.0, n)
    result = {
        "reynolds": 400.0,
        "u": score_profiles(coords, _midplane_u(bundle, 400.0, n), 400.0, "u"),
        "v": score_profiles(coords, _midplane_v(bundle, 400.0, n), 400.0, "v"),
        "vortex_centre": case["vortex_centre"],
    }
    checks = ghia_acceptance(result)
    # the historical, all-point verdict stays visible and still fails: the
    # digitised number is never deleted to buy a green row
    assert checks["criteria"]["v_max_abs_error"]["pass"] is False
    assert checks["pass"] is False
    # the consistent population passes, and the exclusion is recorded, not hidden
    assert checks["criteria"]["v_max_abs_error_consistent"]["pass"] is True
    assert len(checks["excluded_reference_points"]) == 1
    # ... and it is the *only* failing criterion: the row is red for one explained reason
    failing = [name for name, item in checks["criteria"].items() if not item["pass"]]
    assert failing == ["v_max_abs_error"]


def test_acceptance_unchanged_for_results_without_the_audit_keys():
    """Backward compatibility: an older result dict scores exactly as before."""
    extrema = {"min_reference": [0.5, -0.2], "min_computed": [0.5, -0.2],
               "max_reference": [0.5, 0.1], "max_computed": [0.5, 0.1]}
    legacy = {"reynolds": 100.0,
              "u": {"max_abs_error": 0.02, "extremum": extrema},
              "v": {"max_abs_error": 0.02, "extremum": extrema},
              "vortex_centre": {"computed": [0.6172, 0.7344]}}
    checks = ghia_acceptance(legacy)
    assert checks["criteria"]["u_max_abs_error_consistent"]["value"] == 0.02
    assert "excluded_reference_points" not in checks