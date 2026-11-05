"""Ghia (1982) lid-cavity reference-table consistency and dispute attribution.

Two complementary audit tracks are maintained here:

1. **Dispute register** (main): the Re = 400 v-centreline discrepancy at x = 0.9063
   is attributed to a defective published entry.  The solver value is grid-converged
   at clean second order and the gap survives Richardson extrapolation, while every
   other point converges.  The audit script ``scripts/audit_ghia_re400_v_attribution.py``
   provides the evidence; the tests here keep the answer pinned.

2. **Consistency audit** (m3195): the full ladder bundle is audited for interior
   consistency.  Exactly one point (400/v/0.9063) cannot be explained by resolution;
   its residual saturates under refinement while all 101 other points converge.
   The scoring path reports both the all-point and the consistent-subset populations.

Both tracks are kept active so a pass can never hide a defect.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
import torch

from tensorfvm.lid_cavity import ghia_acceptance, score_profiles
from tensorfvm.lid_cavity_reference import (
    DISPUTES,
    SUSPECT_ENTRIES,
    TABLE_PROVENANCE,
    V_CENTRELINE,
    consistent_points,
    self_check,
    smoothness_audit,
    suspect_entries,
    table,
)
from tensorfvm.verification.ghia_reference_audit import audit_bundle

from tests.evidence_support import published

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "audit_ghia_re400_v_attribution.py")
BUNDLE = "docs/verification-lid-cavity-ladder/summary.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _audit_module():
    spec = importlib.util.spec_from_file_location("ghia_attribution_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle():
    module = _audit_module()
    path = os.path.join(REPO, "docs", "verification-lid-cavity-ladder", "summary.json")
    try:
        return module, module.load_bundle(path)
    except SystemExit as error:  # pragma: no cover - evidence checkout policy
        pytest.skip(f"ladder bundle unavailable: {error}")


def _json_bundle() -> dict:
    with open(published(BUNDLE)) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def bundle() -> dict:
    return _json_bundle()


@pytest.fixture(scope="module")
def audit(bundle) -> dict:
    return audit_bundle(bundle)


def _case(bundle: dict, reynolds: float, n: int) -> dict:
    return {(c["reynolds"], c["n"]): c for c in bundle["cases"]}[(reynolds, n)]


def _midplane_v(bundle: dict, reynolds: float, n: int) -> torch.Tensor:
    """v on y = L/2 from the cached staggered field."""
    field = _case(bundle, reynolds, n)["field"]["v"]
    return torch.tensor([row[n // 2] for row in field], dtype=torch.float64)


def _midplane_u(bundle: dict, reynolds: float, n: int) -> torch.Tensor:
    """u on x = L/2."""
    field = _case(bundle, reynolds, n)["field"]["u"]
    return torch.tensor([value for value in field[n // 2]], dtype=torch.float64)


# =========================================================================== #
# Track 1: Dispute register (main)
# =========================================================================== #

def test_the_only_corner_in_all_six_tables_is_the_registered_dispute():
    audit = smoothness_audit()
    flagged = {item["key"] for item in audit["flagged"]}
    assert audit["ok"], audit
    assert "400/v/0.9063" in flagged
    assert flagged <= {"400/v/0.9063", "400/v/0.9453"}
    assert audit["unexplained"] == []
    assert (400.0, "v", 0.9063) in DISPUTES


def test_a_corruption_anywhere_else_is_still_flagged(monkeypatch):
    """Negative control: the audit must not be a rubber stamp for one known key."""
    rey, which, x = 1000.0, "v", 0.8594
    table_rows = list(table(rey, which))
    index = [row[0] for row in table_rows].index(x)
    table_rows[index] = (x, -0.51550)
    assert table_rows[index][1] != V_CENTRELINE[rey][index][1]
    monkeypatch.setitem(V_CENTRELINE, rey, tuple(table_rows))
    audit = smoothness_audit()
    assert f"{rey:g}/{which}/{x:.4f}" in {item["key"] for item in audit["flagged"]}
    assert not audit["ok"]
    assert f"{rey:g}/{which}/{x:.4f}" in audit["unexplained"]


def test_the_gap_survives_extrapolation_while_its_neighbours_do_not():
    module, bundle = _bundle()
    cases = module.cases_by_grid(bundle)
    rows = module.pointwise(cases, 400.0, "v")
    disputed = module.observed_order(rows[0.9063]["grid"])
    assert disputed["order"] == pytest.approx(2.0, abs=0.05)
    assert disputed["richardson_limit"] == pytest.approx(-0.38990, abs=5e-4)
    finest = max(rows[0.9063]["grid"])
    gaps = {x: abs(rows[x]["grid"][finest] - rows[x]["reference"])
            for x in (0.9453, 0.8594, 0.8047)}
    disputed_gap = abs(rows[0.9063]["grid"][finest] - rows[0.9063]["reference"])
    assert max(gaps.values()) < 0.03, gaps
    assert disputed_gap > 5 * max(gaps.values()), (disputed_gap, gaps)


def test_the_metric_path_cannot_manufacture_the_gap():
    module, bundle = _bundle()
    rows = module.pointwise(module.cases_by_grid(bundle), 400.0, "v")
    finest = max(rows[0.9063]["grid"])
    bound = module.interpolation_bound(rows, 0.9063, finest)
    gap = abs(rows[0.9063]["grid"][finest] - rows[0.9063]["reference"])
    assert bound["target_inside_sampled_range"] is True
    assert gap / bound["interpolation_error_bound"] > 100


def test_register_record_agrees_with_the_committed_evidence():
    module, bundle = _bundle()
    record = DISPUTES[(400.0, "v", 0.9063)]
    rows = module.pointwise(module.cases_by_grid(bundle), 400.0, "v")
    assert set(record["computed_n"]) == set(rows[0.9063]["grid"])
    for n, value in record["computed_n"].items():
        assert rows[0.9063]["grid"][n] == pytest.approx(value, abs=5e-5), n
    converged = module.observed_order(rows[0.9063]["grid"])
    assert record["observed_order"] == pytest.approx(converged["order"], abs=0.05)
    assert record["richardson_limit"] == pytest.approx(converged["richardson_limit"], abs=5e-4)
    assert record["evidence"]


# =========================================================================== #
# Track 2: Consistency audit (m3195)
# =========================================================================== #

def test_provenance_declares_the_transcription_and_its_blind_spot():
    assert "digital copy" in TABLE_PROVENANCE["transcription"]
    assert "interior entries are otherwise unverified" in TABLE_PROVENANCE["self_check_coverage"]
    assert "audit_ghia_reference_consistency" in TABLE_PROVENANCE["consistency_audit"]


def test_self_check_passes_and_is_needed_but_not_sufficient():
    """``self_check`` validates the extrema only, so the bad interior entry passes it."""
    report = self_check()
    assert report["ok"] is True
    bad = SUSPECT_ENTRIES[(400.0, "v")][0.9063]
    assert dict(table(400.0, "v"))[0.9063] == bad["as_digitized"] == -0.23827


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
    assert counts.get("consistent", 0) == 101
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


def test_suspect_entry_is_excluded_from_the_consistent_population():
    assert 0.9063 in suspect_entries(400.0, "v")
    kept = consistent_points(400.0, "v")
    assert len(kept) == 16 and all(p[0] != 0.9063 for p in kept)
    assert consistent_points(400.0, "v", include_suspect=True) == table(400.0, "v")
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
    assert checks["criteria"]["v_max_abs_error"]["pass"] is False
    assert checks["pass"] is False
    assert checks["criteria"]["v_max_abs_error_consistent"]["pass"] is True
    assert len(checks["excluded_reference_points"]) == 1
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


# =========================================================================== #
# Gate behaviour tests (shared)
# =========================================================================== #

def _synthetic_result(reynolds=400.0, extra_bad=None):
    """A perfect score_against_ghia-shaped result with the disputed point bad."""
    out = {"reynolds": reynolds}
    for which in ("u", "v"):
        rows = table(reynolds, which)
        coord = [row[0] for row in rows]
        reference = [row[1] for row in rows]
        computed = list(reference)
        if which == "v":
            index = coord.index(0.9063)
            computed[index] = reference[index] - 0.15
            if extra_bad is not None:
                other = coord.index(extra_bad)
                computed[other] = reference[other] - 0.2
        delta = [c - r for c, r in zip(computed, reference)]
        jmin = reference.index(min(reference))
        jmax = reference.index(max(reference))
        out[which] = {
            "profiles": {"coord": coord, "reference": reference,
                         "computed": computed, "delta": delta},
            "max_abs_error": max(abs(d) for d in delta),
            "rms_error": 0.0,
            "extremum": {"min_reference": [coord[jmin], reference[jmin]],
                         "min_computed": [coord[jmin], computed[jmin]],
                         "max_reference": [coord[jmax], reference[jmax]],
                         "max_computed": [coord[jmax], computed[jmax]]},
        }
    out["vortex_centre"] = {"computed": [0.5547, 0.6055]}
    return out


def test_default_gate_keeps_failing_on_the_disputed_point():
    result = _synthetic_result()
    verdict = ghia_acceptance(result)
    assert verdict["pass"] is False
    assert verdict["registered_disputes"] == []
    assert verdict["criteria"]["v_max_abs_error"]["value"] > 0.03


def test_registered_dispute_can_be_ignored_but_is_reported():
    verdict = ghia_acceptance(_synthetic_result(), allow_registered_disputes=True)
    assert verdict["pass"] is True
    ignored = verdict["registered_disputes"]
    assert [item["x"] for item in ignored] == [0.9063]
    assert ignored[0]["published"] == -0.23827
    assert ignored[0]["evidence"]
    assert verdict["criteria"]["v_max_abs_error"]["ignored"] == 1
    assert verdict["disputes_declared"] is True


def test_a_second_bad_point_still_fails_the_dispute_aware_gate():
    verdict = ghia_acceptance(_synthetic_result(extra_bad=0.2344),
                              allow_registered_disputes=True)
    assert verdict["pass"] is False
    assert verdict["criteria"]["v_max_abs_error"]["value"] == pytest.approx(0.2, abs=1e-9)
    assert [item["x"] for item in verdict["registered_disputes"]] == [0.9063]
