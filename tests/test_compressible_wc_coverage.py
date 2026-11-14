"""Gates for the CASE-WOODWARD-COLELLA domain-coverage / robustness / provenance audit.

Every test here *recomputes* from the committed profiles and reference tables; the
committed JSON artifacts are only used as a cross-check, never as the assertion
oracle.  The audit itself is described in
``docs/audit-m20339/AUDIT-CASE-WOODWARD-COLELLA-adversarial-review.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WC = REPO / "docs" / "woodward-colella-double-blast"
COV = WC / "coverage"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cr():
    return _load("wc_coverage_robustness", COV / "wc_coverage_robustness.py")


@pytest.fixture(scope="module")
def pm():
    return _load("wc_provenance_manifest", COV / "wc_provenance_manifest.py")


@pytest.fixture(scope="module")
def res(cr):
    cr.figures = lambda *a, **k: []          # analysis only; no plotting in tests
    profiles = cr.asy.load_profiles()
    return {"profiles": profiles,
            "coverage": cr.coverage(profiles),
            "evidence": cr.scheme_and_grid_evidence(profiles),
            "invariants": cr.invariants(),
            "offset_sweep": cr.offset_sweep(profiles),
            "trough_conditioning": cr.trough_conditioning(profiles)}


# ------------------------------------------------------------------ 1. coverage
def test_published_window_mass_share(res):
    c = res["coverage"]
    # 60 % of the length, but ~93.5 % of the mass: the blind strip is light
    assert c["length_share_percent"] == pytest.approx(60.0)
    assert 92.0 < c["window_mass_share_percent"] < 95.0, c["window_mass_share_percent"]
    assert 5.0 < c["left_mass_share_percent"] < 8.0, c["left_mass_share_percent"]
    assert c["blind_strip_has_external_reference"] is False
    for tag, row in c["per_grid"].items():
        assert row["window_mass_share_percent"] > 92.0, tag


def test_artifact_contract_of_stored_profiles(res):
    """Full-domain ladders span [0,1]; the two `artifacts/` profiles are window-only.

    This is an explicit contract: `wc_blast_profile_nx{400,800}.npz` are *not*
    whole-domain profiles despite the name, so nobody may use them for
    blind-strip or conservation analysis.
    """
    p = res["profiles"]
    for tag in ("nx200_hllc", "nx1600_hllc", "nx3200_hllc"):
        assert p[tag]["x"].min() < 0.01 and p[tag]["x"].max() > 0.99, tag
    for tag in ("nx400_hllc", "nx800_hllc"):
        assert p[tag]["x"].min() >= 0.4 and p[tag]["x"].max() <= 1.0, tag
        assert p[tag]["x"].size < p[tag]["nx"], tag


# ------------------------------------------------- 2. reference-free evidence
def test_blind_strip_is_scheme_and_grid_independent(res):
    """The un-referenced strip is bounded even without a literature curve."""
    ev = res["evidence"]
    blind = ev["cross_scheme"]["nx3200"]["blind_left_(0,0.4)"]["percent"]
    window = ev["cross_scheme"]["nx3200"]["window_(0.4,1.0)"]["percent"]
    assert blind < 0.5, blind
    assert ev["cross_grid"]["blind_left_(0,0.4)"]["d_1600_3200"]["percent"] < 1.0
    # the shock-rich window is where the scheme sensitivity actually lives
    assert window > blind


def test_conservation_invariants_are_exact(res):
    inv = res["invariants"]
    assert inv["gamma"] == pytest.approx(1.4)
    rows = inv["rows"]
    assert len(rows) >= 5
    for tag, row in rows.items():
        assert abs(row["mass_drift"]) < 1e-10, (tag, row["mass_drift"])
        assert abs(row["energy_relative_drift"]) < 1e-10, (tag, row["energy_relative_drift"])


# --------------------------------------------- 3. robustness to calibration
def test_acceptance_is_robust_over_the_plausible_calibration_band(res):
    """<10 % must hold for every offset in [-0.022, +0.044], not just the chosen one."""
    sw = res["offset_sweep"]["sweep"]
    assert len(sw) == 7
    for key, per_grid in sw.items():
        for tag in ("nx1600_hllc", "nx3200_hllc"):
            row = per_grid[tag]
            assert row["peak_percent"] < 10.0, (key, tag, row["peak_percent"])
            assert row["l1_percent"] < 10.0, (key, tag, row["l1_percent"])
            assert row["window_mass_percent"] < 10.0, (key, tag, row["window_mass_percent"])
            assert row["verdict"] == "PASS", (key, tag, row["checks"])


def test_offset_sensitivity_ranks_the_metrics(res):
    s = res["offset_sweep"]["sensitivity"]
    # the trough is ~44x more sensitive to the reference offset than the peak
    assert abs(s["trough_percent"]["d_error_percent_per_unit_density"]) > \
        20.0 * abs(s["peak_percent"]["d_error_percent_per_unit_density"])
    # ... which is the quantitative reason it cannot carry the verdict


def test_trough_exclusion_is_independently_justified(res):
    """The excluded metric really does read a ramp edge, not a trough."""
    tc = res["trough_conditioning"]
    e = tc["edge"]
    assert e["argmin_x"] <= 0.43, e["argmin_x"]
    assert e["edge_distance_percent_of_window"] < 5.0, e
    assert e["reference_has_interior_min"] is False
    assert tc["monotone_rise"]["strictly_increasing"] is True
    assert tc["monotone_rise"]["negative_increments"] == 0


def test_honest_record_of_the_raw_reference_is_preserved(res):
    """At delta=0 the trough is 14.45 % -- the honest, unedited number stays."""
    sw = res["offset_sweep"]["sweep"]["+0.000000"]
    assert 13.0 < sw["nx3200_hllc"]["trough_percent"] < 15.0, sw["nx3200_hllc"]
    at_bias = res["offset_sweep"]["sweep"]["+0.022017"]
    assert at_bias["nx3200_hllc"]["trough_percent"] < 1.0


# ------------------------------------------------------------ 4. provenance gates
@pytest.fixture(scope="module")
def prov(pm):
    return pm.analyse()


def test_no_contradicted_reference_identity(prov):
    a = prov["absolute_path_records"]
    assert a["contradicted"] == [], a["contradicted"]


def test_every_recorded_absolute_path_has_a_repo_relative_target(prov):
    for h in prov["absolute_path_records"]["fields"]:
        if h["recorded"].startswith("/tmp/"):
            # transient cache (the publisher PDF is not committed, only its sha256);
            # documented in reference/axis_calibration.json
            continue
        assert h["repo_relative"], h
        assert h["target_exists_in_repo"], h


def test_all_run_records_reproduce_from_the_committed_tree(prov):
    """Decisive check: dangling cross-worktree paths caused no silent contamination."""
    r = prov["record_reproduction"]
    assert r["n_records"] >= 15, r["n_records"]
    bad = [row["record"] for row in r["rows"]
           if "profile" in row and not row["reproduced_from_committed_tree"]]
    assert bad == [], bad
    assert r["n_reproduced"] == r["n_records"]


def test_manifest_matches_a_fresh_scan(prov):
    """The committed manifest must not be stale w.r.t. the bundle."""
    committed = json.loads((COV / "provenance_manifest.json").read_text())
    assert ({h["recorded"] for h in committed["absolute_path_records"]["fields"]} ==
            {h["recorded"] for h in prov["absolute_path_records"]["fields"]})
    assert committed["absolute_path_records"]["count"] == prov["absolute_path_records"]["count"]


def test_committed_analysis_artifact_matches_recomputation(res):
    committed = json.loads((COV / "coverage_robustness.json").read_text())
    assert committed["coverage"]["window_mass_share_percent"] == pytest.approx(
        res["coverage"]["window_mass_share_percent"], abs=1e-9)
    for name in res["evidence"]["cross_scheme"]["nx3200"]:
        assert committed["evidence"]["cross_scheme"]["nx3200"][name]["percent"] == pytest.approx(
            res["evidence"]["cross_scheme"]["nx3200"][name]["percent"], abs=1e-9)
    assert committed["figures"] and set(committed["figures"]) == {
        "fig_wc_domain_coverage.png", "fig_wc_offset_robustness.png"}
    for f in committed["figures"]:
        assert (COV / f).stat().st_size > 20_000, f