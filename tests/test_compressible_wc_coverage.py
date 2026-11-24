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


def test_every_record_has_an_accounted_disposition(prov):
    """reproduced / failed / unverifiable must partition the record set.

    Guards the round-2 defect: a record with nothing to re-score used to be
    reported as "not reproduced", which made the gate red for a reason that had
    nothing to do with provenance.
    """
    r = prov["record_reproduction"]
    assert r["n_reproduced"] + r["n_failed"] + r["n_unverifiable"] == r["n_records"]
    assert r["n_reference_metrics"] + r["n_solver_observables"] <= r["n_records"]
    for row in r["rows"]:
        if row["check_kind"] == "none":
            assert row["reproduced_from_committed_tree"] is None, row["record"]
            assert row.get("reason"), row["record"]
        else:
            assert isinstance(row["reproduced_from_committed_tree"], bool), row["record"]


def test_all_run_records_reproduce_from_the_committed_tree(prov):
    """Decisive check: dangling cross-worktree paths caused no silent contamination."""
    r = prov["record_reproduction"]
    assert r["n_records"] >= 15, r["n_records"]
    # The cfl0.8 weno5 records now carry full reference metrics (low_plateau, etc.)
    # so they are classified as reference_metrics, not solver_observables.
    assert r["n_failed"] == 0, r["failed_records"]
    assert r["n_unverifiable"] == 0, r["unverifiable_records"]
    assert r["n_reproduced"] == r["n_records"]


def test_metric_free_records_are_checked_via_solver_observables(prov, pm):
    """A record without reference error blocks is still checkable, and is checked.

    ``ladder_metrics_nx{12800,25600}_weno5_cfl0.8.json`` now carry full reference
    metrics (low_plateau, etc.) and are classified as reference_metrics.
    They must still be re-derived from the committed profile to ensure no silent
    contamination.
    """
    rows = {r["record"].split("/")[-1]: r for r in prov["record_reproduction"]["rows"]}
    for name in ("ladder_metrics_nx12800_weno5_cfl0.8.json",
                 "ladder_metrics_nx25600_weno5_cfl0.8.json"):
        row = rows[name]
        assert row["check_kind"] == "reference_metrics", row
        assert row["profile_match"] == "exact", row
        assert row["reproduced_from_committed_tree"] is True, row


def test_observable_gate_is_not_vacuous(pm):
    """An off-by-1e-6 profile must be rejected -- the gate has real teeth."""
    prof = pm._profile_for("nx12800_weno5_cfl0.8")
    recorded = {"rho_max": 6.463485157370068, "x_at_rho_max": 0.7784765625}
    obs = pm.observe_profile(prof)
    assert all(v["within_tolerance"] for v in pm._compare_observables(recorded, obs).values())
    pert = {"rho_max": prof["rho"].max() * (1 + 1e-6),
            "x_at_rho_max": prof["x"][int(prof["rho"].argmax())] + 1e-4}
    bad = pm._compare_observables(recorded, pert)
    assert bad["rho_max"]["within_tolerance"] is False, bad["rho_max"]
    assert bad["x_at_rho_max"]["within_tolerance"] is False, bad["x_at_rho_max"]


def test_no_record_silently_uses_a_different_limiter_family(prov):
    """Reproduction must not compare a weno5 record against an HLLC profile.

    That silent fallback is the same defect class as the F3 skip in
    ``test_compressible_wc_calibration.py``; here it would launder a wrong-family
    comparison as an ordinary mismatch.
    """
    r = prov["record_reproduction"]
    assert r["n_family_fallback"] == 0, [row for row in r["rows"]
                                         if row["profile_match"] == "hllc_family_fallback"]
    for row in r["rows"]:
        if row["profile_match"] == "artifact_window_only":
            # documented contract: wc_blast_profile_nx{400,800}.npz are window-only
            assert row["profile"].endswith(("wc_blast_profile_nx400.npz",
                                            "wc_blast_profile_nx800.npz")), row
            assert row["reproduced_from_committed_tree"] is True, row


def test_manifest_matches_a_fresh_scan(prov):
    """The committed manifest must not be stale w.r.t. the bundle.

    Round-2 defect: only the absolute-path field set was compared, so records
    added *after* the audit (which carry no absolute path) left the manifest
    silently stale.  Record identity and classification are compared too.
    """
    committed = json.loads((COV / "provenance_manifest.json").read_text())
    assert ({h["recorded"] for h in committed["absolute_path_records"]["fields"]} ==
            {h["recorded"] for h in prov["absolute_path_records"]["fields"]})
    assert committed["absolute_path_records"]["count"] == prov["absolute_path_records"]["count"]

    c, f = committed["record_reproduction"], prov["record_reproduction"]
    assert c["n_records"] == f["n_records"]
    assert [(r["record"], r["check_kind"], r["reproduced_from_committed_tree"]) for r in c["rows"]] == \
           [(r["record"], r["check_kind"], r["reproduced_from_committed_tree"]) for r in f["rows"]]
    for key in ("n_reproduced", "n_failed", "n_unverifiable", "n_solver_observables",
                "n_family_fallback"):
        assert c[key] == f[key], key
    assert committed["verdict"] == prov["verdict"]


def test_verdict_names_portability_defect_but_no_contamination(prov):
    v = prov["verdict"]
    assert v["portability"] in {"DEFECT", "clean"}
    assert "none found" in v["numerical_contamination"]
    assert prov["absolute_path_records"]["contradicted"] == []


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
