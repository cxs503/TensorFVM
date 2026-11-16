"""Regression gates for the Round-14 closure of anchor XVAL-FVM-M18737-ROUND2.

Round 14 fetched the three pre-registered R13 jobs (``cj4221``/``cj4222``/
``cj4223``), back-filled their artifacts into ``ladder/data/`` and re-ran the
*frozen* R13 pipeline (``r13_certified_rung.py``) on the completed ladder.
The criteria were frozen in ``PREDECLARATION-r13.md`` before any number landed;
these tests pin what the completed ladder must show so a future round cannot
silently re-grade the same rungs under a different meaning:

* the CFL admission controls (E1/E1r) must be *measured* and admitted at the
  frozen 0.2 % threshold -- the nx=12800 rung is a CFL 0.8 variant and is only
  licensed by these two controls, not by how the result looks;
* the graded ladder must stay monotone and the finest rung must clear the
  certified <1 % gate (E3) against the self-certified WENO5@12800 reference;
* the honest ceiling (E4) must keep the reference-side error bar attached:
  four of the five conventions (B0/B1/B2/B4) come in under 1 % at nx=12800
  while the most conservative one (B3, non-asymptotic Fs=3) must still fail --
  an honest FAIL that must not be overwritten;
* the E2 power-law prediction missed in the *favourable* direction (measured
  below the pre-registered band, observed order above the band): the test pins
  that miss so a later "correction" of the prediction is visible;
* the independent re-integration and the shared metric must agree on every rung.

All checks are pure numpy over stored artifacts -- no solver call.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
DRIVER = HERE / "docs" / "XVAL-FVM-M18737-ROUND2" / "r13_certified_rung.py"
DATA = HERE / "docs" / "woodward-colella-double-blast" / "ladder" / "data"

#: the three R13 pre-registered jobs and the tags their artifacts carry
R14_TAGS = ("nx12800_hllc_cfl0.8", "nx6400_hllc_cfl0.8", "nx12800_weno5_cfl0.8")


@pytest.fixture(scope="module")
def r13():
    spec = importlib.util.spec_from_file_location("wc_r13_driver", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def verdict(r13):
    return r13.compute()


def test_r14_artifacts_are_backfilled():
    """The three job artifacts (profile + metrics sidecar) must be in the repo."""
    for tag in R14_TAGS:
        npz = DATA / f"ladder_profile_{tag}.npz"
        js = DATA / f"ladder_metrics_{tag}.json"
        assert npz.exists(), f"missing profile {npz}"
        assert js.exists(), f"missing metrics sidecar {js}"
        m = json.loads(js.read_text())
        assert m.get("cfl") == 0.8, (tag, m.get("cfl"))


def test_cfl_admission_controls_measured_and_admitted(r13, verdict):
    """E1/E1r are measured this round and both pass the frozen 0.2 % bar."""
    assert r13.CFL_ADMISSION == 0.2
    for key in ("E1_cfl_graded", "E1r_cfl_reference"):
        d = verdict[key]
        assert d is not None, f"{key} not measured"
        assert d["l1_percent"] < r13.CFL_ADMISSION, d
        assert d["admitted"] is True, d
    assert verdict["cfl_admission_pass"] is True


def test_rung_12800_is_genuine_cfl08(verdict):
    """E0 on the new rung: real grid geometry, positive density, conserved, cfl matches tag."""
    g = verdict["E0_genuineness"]["12800"]
    assert g["pass"] is True, g
    assert g["metrics_available"] is True
    assert g["cfl"] == 0.8, "tag says cfl0.8; the metrics payload must agree"
    assert g["checks"]["E0c_cfl_matches_tag"] is True
    assert g["mass_drift"] < 1e-10
    assert g["positive"] and g["all_finite"]
    assert verdict["missing_rungs"] == []


def test_ladder_monotone_and_finest_certified(r13, verdict):
    """L1 decreases rung over rung and the finest rung clears the <1 % gate (E3)."""
    ks = sorted(int(k) for k in verdict["ladder"])
    assert ks[-1] == 12800, ks
    es = [verdict["ladder"][str(nx)]["l1_percent"] for nx in ks]
    assert all(a > b > 0.0 for a, b in zip(es, es[1:])), es
    e3 = verdict["E3_certified"]
    assert e3["l1_below_1pct"] is True
    assert e3["peak_below_1pct"] is True
    assert e3["certified_under_1pct"] is True
    # scope discipline: the gate is vs the self-certified reference, not the exact solution
    assert "NOT vs the exact solution" in e3["scope"]


def test_honest_ceiling_four_of_five_conventions(r13, verdict):
    """B0/B1/B2/B4 under 1 %, B3 (non-asymptotic Fs=3) still over -- honest FAIL kept."""
    e4 = verdict["E4_honest_ceiling"]
    if e4.get("status") != "measured":
        pytest.skip("reference-side pair not available in this checkout")
    ub = e4["finest"]["upper_bound_percent"]
    for k in ("B0", "B1", "B2", "B4"):
        assert ub[k] < r13.BAR_CERTIFIED, (k, ub[k])
    assert ub["B3"] > r13.BAR_CERTIFIED, ("B3 must stay an honest FAIL", ub["B3"])
    # the bar must raise the bound, never lower it
    for row in e4["rows"]:
        for k in ("B0", "B1", "B2", "B3", "B4"):
            assert row["upper_bound_percent"][k] > row["l1_percent"], (row["nx"], k)
    # the reference ladder mixes CFL 0.4 (6400) with 0.8 (12800/25600): not licensed
    assert e4["licensed"] is False
    assert "licensed" in e4["license_note"] or "探针" in e4["license_note"]


def test_e2_power_law_missed_in_favourable_direction(r13, verdict):
    """The pre-registered prediction missed: measured below the band, order above 1.15."""
    e2 = verdict["E2_prediction"]
    assert e2["hit"] is False
    lo, hi = e2["band_percent"]
    assert e2["measured_percent"] < lo, (e2["measured_percent"], (lo, hi))
    ks = sorted(int(k) for k in verdict["ladder"])
    p = verdict["orders"][f"{ks[-2]}->{ks[-1]}"]["p"]
    assert p > 1.15, p
    assert e2["order_within_band"] is False


def test_independent_metric_agrees(r13, verdict):
    """Shared metric vs independent re-integration: a metric artefact would have to survive both."""
    diff = verdict["independent_check_max_abs_diff_percent"]
    finest = min(r["l1_percent"] for r in verdict["ladder"].values())
    assert diff < 0.15, diff
    assert diff < 0.25 * finest, (diff, finest)


def test_adjudication_string_is_complete(verdict):
    """The one-line adjudication must carry the certified tag, the floor class and the bar survival."""
    adj = verdict["adjudication"]
    assert "CERTIFIED_UNDER_1PCT_AT_NX12800" in adj
    assert "NO_FLOOR_DETECTED" in adj
    assert "HONEST_UNDER_1PCT_B0_B1_B2_B4" in adj
    assert "B3" not in adj.split("HONEST_UNDER_1PCT_")[-1]


def test_r14_json_matches_recomputed_verdict(r13):
    """The committed r14 JSON must be reproducible from the stored artifacts (no drift)."""
    js = HERE / "docs" / "XVAL-FVM-M18737-ROUND2" / "r14_certified_rung.json"
    if not js.exists():
        pytest.skip("r14_certified_rung.json not generated in this checkout")
    stored = json.loads(js.read_text())
    fresh = r13.compute()
    for key in ("status", "adjudication", "missing_rungs", "cfl_admission_pass",
                "gates_all_pass", "independent_check_max_abs_diff_percent"):
        assert stored[key] == fresh[key], key
    for nx, row in fresh["ladder"].items():
        assert stored["ladder"][nx]["l1_percent"] == pytest.approx(row["l1_percent"], rel=1e-12)
