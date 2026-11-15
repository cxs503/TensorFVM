"""Asymptotic-range GCI / uncertainty-budget gate for CASE-WOODWARD-COLELLA.

The anchor's earlier GCI study (``tests/test_wc_gci_analysis.py``) used
nx = 200/400/800, i.e. grids that are *not* in the asymptotic range of a
solution with two colliding shocks: it produced an extrapolated peak of 9.02
(43 % above the published reference).  After the ladder was extended to
nx = 1600/3200, ``wc_asymptotic_analysis.py`` repeats the study on the fine
triplet and builds an uncertainty budget that separates *our* discretisation
error from the error carried by the digitised published reference.

These tests lock in both the numbers and the honest structure of the
conclusion: the residual to the reference must be shown to be larger than the
numerical uncertainty (so nobody can claim it is a discretisation error we
could refine away), and the alternative explanation that the digitisation
polyline clips the peak must stay falsified (it costs 0.05 %, not 3 %).

No solver runs: pure numpy on committed artifacts (< 2 s).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "docs" / "woodward-colella-double-blast" / "ladder"
DATA = LADDER / "data"
REFDIR = ROOT / "docs" / "woodward-colella-double-blast" / "reference"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ana = _load("wc_asymptotic_analysis", LADDER / "wc_asymptotic_analysis.py")


@pytest.fixture(scope="module")
def result():
    profiles = ana.load_profiles()
    import tensorfvm.compressible.wc_reference as wcr

    ref_cal = wcr.load_reference(ana.CALIBRATED)
    ref_raw = wcr.load_reference()
    return {
        "hllc": ana.family_analysis(profiles, "hllc", ref_cal, ref_raw, ana.THEORY_ORDER),
        "cross_code": ana.cross_code(profiles),
    }


def test_fine_ladder_present():
    """nx = 3200 and 1600 profiles must be committed; without them no claim holds."""
    profiles = ana.load_profiles()
    nx = {p["nx"] for p in profiles.values() if p["family"] == "hllc"}
    assert {800, 1600, 3200} <= nx


def test_field_converges_on_fine_grids(result):
    """Successive grids must get closer: D(1600,3200) < D(800,1600)."""
    d = result["hllc"]["field_distance_successive"]
    assert len(d) == 2
    coarse, fine = d[0], d[1]
    assert coarse["grids"] == [800, 1600] and fine["grids"] == [1600, 3200]
    assert fine["percent"] < coarse["percent"]
    assert 0.3 < result["hllc"]["field_order"]["observed_order"] < 3.0
    # both distances are far inside the 10 % acceptance band
    assert coarse["percent"] < 10.0 and fine["percent"] < 10.0


def test_nx3200_profile_matches_calibrated_reference():
    """The finest grid must clear the <10 % acceptance on every well-posed metric."""
    m = json.loads((REFDIR / "metrics_calibrated_nx3200_hllc.json").read_text())
    cal = m["calibrated_reference"]
    assert cal["peak_density"]["rel_error_percent"] < 10.0
    assert cal["l1_relative"]["percent"] < 10.0
    assert cal["low_plateau"]["rel_error_percent"] < 3.0
    assert cal["window_mass"]["rel_error_percent"] < 3.0
    # ... and the *raw* (uncalibrated) reference still shows the 14 % plateau
    # artefact, i.e. the calibration is what changed, not the metric definition.
    raw = m["committed_reference"]
    assert raw["low_plateau"]["rel_error_percent"] > 10.0
    assert raw["peak_density"]["rel_error_percent"] < 10.0


def test_uncertainty_budget_separates_reference_error(result):
    """The residual to the reference must NOT be explainable by our discretisation.

    ``U_num`` is the GCI of the fine triplet.  A residual much larger than
    ``U_num`` means refining our grid cannot close it: it belongs to the
    reference (or to the difference between the two schemes), and the honest
    verdict must say so instead of quoting the residual as solution error.
    """
    budget = result["hllc"]["uncertainty_budget"]
    for key in ("window_mass", "peak_density"):
        b = budget[key]
        assert b["numerical_uncertainty_percent"] < 1.0   # fine triplet is tight
    # window mass: residual ~0.02 % still 1.6x above the (tiny) GCI band
    assert budget["window_mass"]["residual_to_reference_percent"] < 0.1
    assert budget["window_mass"]["residual_over_unum"] > 1.0
    # peak: residual is two orders of magnitude larger than the band
    assert budget["peak_density"]["residual_to_reference_percent"] > 2.0
    assert budget["peak_density"]["residual_over_unum"] > 10.0
    assert budget["peak_density"]["reference_inside_gci_band"] is False


def test_reference_distance_does_not_improve_with_refinement(result):
    """Our distance to the published reference is a *floor*, not a converging error.

    nx = 800 sits 2.0 % away and nx = 3200 sits 3.0 % away: refinement moves the
    solution past the digitised curve instead of onto it.  If this ever starts
    decreasing monotonically the interpretation in the report must be revisited.
    """
    l1 = result["hllc"]["studies"]["l1_vs_calibrated_reference"]["values"]
    assert l1["800"] < l1["3200"]
    assert l1["3200"] < 0.10 and l1["800"] < 0.10


def test_polyline_digitisation_cannot_explain_peak_residual(result):
    """The 'the reference figure clips the peak' explanation must stay falsified."""
    ds = result["hllc"]["digitisation_sampling"]
    worst = ds["cases"][0]        # 174 vertices = the actual path vertex count
    assert worst["vertices"] == 174
    assert worst["peak_loss_percent"] < 0.2
    # the unexplained peak residual is an order of magnitude larger
    budget = result["hllc"]["uncertainty_budget"]["peak_density"]
    assert budget["residual_to_reference_percent"] > 10.0 * worst["peak_loss_percent"]


def test_extrapolated_field_sweep_is_stable(result):
    """Integral quantities survive the order sweep; the shock peak does not.

    The window mass is a smooth functional: Richardson extrapolation of the
    field reproduces the published value to <1 % at every order tried.  The
    peak, by contrast, is a local extremum inside a collision zone; when the
    field is extrapolated with the *shock-limited* order (p ~ 0.8) the peak is
    inflated past the 10 % acceptance band (6.98 vs 6.318).  That is an
    extrapolation artefact, not a solution error: the peak's own fine-triplet
    study (p_obs = 2.74) extrapolates to 6.515, i.e. 3.1 %.  Both facts are
    asserted so the report cannot quietly quote one without the other.
    """
    ef = result["hllc"]["extrapolated_field"]
    ref = ef["reference_mass_calibrated"]
    for row in ef["order_sweep"]:
        assert abs(row["window_mass"] - ref) / ref * 100.0 < 1.0
    peaks = {row["order_used"]: row["peak_density"] for row in ef["order_sweep"]}
    # theory-order (p = 2) extrapolated field respects the acceptance band
    assert abs(peaks[2.0] - 6.3181) / 6.3181 * 100.0 < 10.0
    # the shock-limited field order inflates the peak: recorded, not hidden
    field_order = ef["order_sweep"][1]["order_used"]
    assert 0.2 < field_order < 2.0
    assert abs(peaks[field_order] - 6.3181) / 6.3181 * 100.0 > 10.0
    # the peak's own fine-triplet study is the one that may be quoted as a peak
    study = result["hllc"]["studies"]["peak_density"]["fine"]
    assert abs(study["extrapolated"] - 6.3181) / 6.3181 * 100.0 < 4.0


def test_cross_code_agreement_when_available(result):
    """Two independent families must agree better on finer grids (basis error).

    Skipped while the WENO5 fine-grid profiles are not committed.
    """
    cc = result["cross_code"]
    if not cc.get("available"):
        pytest.skip(f"WENO5 ladder profiles absent: {cc.get('reason')}")
    per = {e["nx"]: e["percent"] for e in cc["per_grid"]}
    if 1600 in per and 3200 in per:
        assert per[3200] < per[1600] < per[200]
        assert per[3200] < 5.0
        agree = cc["extrapolated_field_agreement"]
        assert agree["distance_percent"] < 5.0
        # the family difference must *vanish* with refinement (a basis error bound
        # that holds without any published curve): 4.60 -> 1.48 -> 0.97 -> 0.62 %
        ladder = [per[n] for n in sorted(per)]
        assert all(b < a for a, b in zip(ladder, ladder[1:])), ladder
        assert per[max(per)] < 1.0
        assert agree["distance_percent"] < 2.0
        # integral quantities: family difference tiny against the gap to the reference
        ref_mass = 0.93534261921925
        fam_mass = abs(agree["mass_hllc"] - agree["mass_weno5"]) / agree["mass_weno5"]
        gap_mass = abs(agree["mass_hllc"] - ref_mass) / ref_mass
        assert gap_mass > 5.0 * fam_mass
        # peak: the family difference is the SAME order as the gap to the reference,
        # so the peak residual may not be booked to the reference alone (report ⑤.5 / ⑥)
        fam_peak = abs(agree["peak_hllc"] - agree["peak_weno5"]) / agree["peak_weno5"]
        assert fam_peak > 0.02
        assert agree["peak_hllc"] > 6.318 and agree["peak_weno5"] > 6.318
    else:
        assert min(per.values()) < 10.0


def test_ladder_classification_is_explicit():
    """Family and variant must be *parsed*; no substitution may happen by ordering.

    Regression for a real defect: the loader used to fall back to ``hllc`` for
    every tag that did not contain the substring ``weno5``, so the committed
    ``nx1600_weno7`` profile silently replaced the nx = 1600 HLLC profile and
    ``nx200_hllc_limiterminmod_cfl0.2`` (a different limiter *and* CFL, i.e. not
    a refinement level) replaced the canonical nx = 200 one.  That inflated the
    peak GCI from 0.14 % to 8.0 % and turned this gate red on main.
    """
    profiles = ana.load_profiles()
    # the WENO7 profile is its own family, never a member of the HLLC ladder
    assert profiles["nx1600_weno7"]["family"] == "weno7"
    hllc = ana.ladder_tags(profiles, "hllc")
    assert hllc[1600] == "nx1600_hllc"
    assert hllc[200] == "nx200_hllc"          # canonical, not the cfl0.2 variant
    # the variant is still loaded (nothing is hidden) but flagged non-canonical
    var = profiles["nx200_hllc_limiterminmod_cfl0.2"]
    assert var["variant"] == "limiterminmod_cfl0.2"
    assert all(profiles[t]["variant"] is None for t in hllc.values())
    # the nx = 1600 ladder point really is the HLLC profile, not the WENO7 one
    peak = ana.peak_density(profiles[hllc[1600]]["x"], profiles[hllc[1600]]["rho"])
    assert peak == pytest.approx(6.466034495852625, rel=1e-12)
    assert peak < profiles["nx1600_weno7"]["rho"].max()
    # parsing is strict: an unknown scheme must raise instead of defaulting
    with pytest.raises(ValueError):
        ana.parse_profile_tag("nx800_rusanov")
    assert ana.parse_profile_tag("nx200_hllc_limiterminmod_cfl0.2") == (
        200, "hllc", "limiterminmod_cfl0.2")
    assert ana.parse_profile_tag("nx3200_weno5") == (3200, "weno5", None)


def test_duplicate_canonical_profile_is_rejected():
    """Two canonical profiles on one grid is an error, not a dict-order accident."""
    fake = {
        "nx800_hllc": {"nx": 800, "family": "hllc", "variant": None},
        "nx800_hllc_copy": {"nx": 800, "family": "hllc", "variant": None},
    }
    with pytest.raises(ValueError, match="two canonical profiles"):
        ana.ladder_tags(fake, "hllc")
    ok = {
        "nx800_hllc": {"nx": 800, "family": "hllc", "variant": None},
        "nx800_hllc_v2": {"nx": 800, "family": "hllc", "variant": "v2"},
        "nx1600_weno5": {"nx": 1600, "family": "weno5", "variant": None},
    }
    assert ana.ladder_tags(ok, "hllc") == {800: "nx800_hllc"}
    assert ana.ladder_tags(ok, "weno5") == {1600: "nx1600_weno5"}


def test_coarse_triplet_artefact_is_reproduced_by_the_canonical_ladder():
    """200/400/800 must reproduce the historical 9.02 peak artefact.

    ``wc_gci_analysis.py`` reported an extrapolated peak of 9.02 on that triplet,
    which is why the fine triplet exists at all.  With the canonical nx = 200
    HLLC profile the number comes back (9.019, +42.8 %, GCI 57 %); with the
    limiter/CFL variant wrongly substituted it read 6.80 - i.e. the contamination
    had hidden the artefact it was supposed to expose.  This is a two-script,
    independent cross-check of the ladder hygiene fix.
    """
    profiles = ana.load_profiles()
    import tensorfvm.compressible.wc_reference as wcr

    ref_cal = wcr.load_reference(ana.CALIBRATED)
    res = ana.family_analysis(profiles, "hllc", ref_cal, wcr.load_reference(),
                              ana.THEORY_ORDER)
    coarse = res["studies"]["peak_density"]["coarse"]
    assert coarse["triplet"] == [800, 400, 200]
    assert coarse["extrapolated"] == pytest.approx(9.02, abs=0.01)
    assert coarse["gci_percent"] > 50.0
    assert coarse["asymptotic"] is False
    # the fine triplet must not be the one that produces this artefact
    assert res["studies"]["peak_density"]["fine"]["extrapolated"] < 7.0


def test_json_artifact_matches_recomputation():
    """The committed JSON must be what the committed script produces (no drift)."""
    committed = json.loads((DATA / "asymptotic_analysis.json").read_text())
    profiles = ana.load_profiles()
    import tensorfvm.compressible.wc_reference as wcr

    ref_cal = wcr.load_reference(ana.CALIBRATED)
    fresh = ana.family_analysis(profiles, "hllc", ref_cal, wcr.load_reference(),
                                ana.THEORY_ORDER)
    a = committed["families"]["hllc"]["uncertainty_budget"]
    b = fresh["uncertainty_budget"]
    for key in ("window_mass", "peak_density", "probe_x0.5"):
        assert a[key]["residual_to_reference_percent"] == pytest.approx(
            b[key]["residual_to_reference_percent"], rel=1e-9, abs=1e-12)
    assert (committed["families"]["hllc"]["field_order"]["observed_order"]
            == pytest.approx(fresh["field_order"]["observed_order"], rel=1e-9))
    # every ladder value and both triplets, not just the budget: a re-classification
    # of the input profiles (the weno7 / variant defect) changes exactly these
    for key in ("window_mass", "peak_density", "probe_x0.5"):
        a_vals = committed["families"]["hllc"]["studies"][key]["values"]
        b_vals = fresh["studies"][key]["values"]
        assert sorted(a_vals) == sorted(b_vals)
        for n in a_vals:
            assert a_vals[n] == pytest.approx(b_vals[n], rel=1e-12, abs=1e-15)
        for label in ("fine", "coarse"):
            assert (committed["families"]["hllc"]["studies"][key][label]["extrapolated"]
                    == pytest.approx(fresh["studies"][key][label]["extrapolated"], rel=1e-9))
    # the ladder rule itself is recorded, so a reader can see what entered the study
    rule = committed["ladder_rule"]
    assert rule["fine_triplet"] == [3200, 1600, 800]
    assert rule["canonical_only"] is True
    assert "nx200_hllc_limiterminmod_cfl0.2" in rule["variants_excluded"]
    assert committed["profiles"]["nx1600_weno7"]["family"] == "weno7"