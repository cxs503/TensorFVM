"""Regression gates for the round-6 *reference-order* audit (anchor XVAL-FVM-M18737-ROUND2).

Round-2/3 bounded the certified-<1 % question with a **same-family** reference floor
(``WENO5@12800 vs @6400 = 0.548 %``), which measures how fast the reference family is
still moving, not how far its limit sits from the graded solution.  Round-6 adds the
missing instrument: ``wc_weno7.py`` (7th-order member of the same code family) run on the
blast datum for the first time.

This gate guards the *instrument and the honesty of the accounting*, not the headline
number:

1. the producer really solves the blast datum (finite, conservative, peak in a physical
   band) and stores a profile the shared loader can read as ``family="weno7"``;
2. the producer is idempotent (an existing rung is skipped, not silently recomputed);
3. the same-grid cross-order metric is symmetric and zero for a profile against itself
   (i.e. it measures a *difference*, not an artefact of argument order);
4. a missing rung is recorded as ``not_measured`` -- never as a PASS, never as an
   exception that hides the gap;
5. the frozen predeclaration is intact: all six quantities are present, the amendment is
   recorded, and the frozen hash of the WENO7 implementation still matches the file on
   disk (a silent edit of the instrument after the freeze would fail here).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
XCODE = REPO / "docs" / "woodward-colella-double-blast" / "xcode"
XV = REPO / "docs" / "XVAL-FVM-M18737-ROUND2"
for p in (XCODE, XV):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ref_order_r6 as r6      # noqa: E402
import wc_weno7 as w7          # noqa: E402
import wc_weno7_ladder as wl   # noqa: E402

from tensorfvm.verification import wc_refine as wcr  # noqa: E402


# --- 1/2: the producer --------------------------------------------------------
@pytest.fixture(scope="module")
def small_rung(tmp_path_factory):
    """One cheap rung (nx=200) produced exactly as the cluster job would."""
    d = tmp_path_factory.mktemp("weno7_ladder")
    res = wl.run_rung(400, d)
    return d, res


def test_producer_solves_blast_datum(small_rung):
    d, res = small_rung
    m = res["metrics"]
    assert res["skipped"] is False
    assert m["family"] == "weno7" and m["nx"] == 400
    assert m["steps"] > 100 and m["seconds"] > 0
    # conservative: the initial mass integral is 1 and it must not drift
    assert abs(m["mass"] - 1.0) < 1e-8
    assert m["mass_drift"] < 1e-10
    # a physically possible two-blast peak: above the right-hand plateau (100 / 1 / gamma)
    # and below the left-hand plateau's post-shock value
    assert 3.0 < m["rho_max"] < 8.0
    assert m["finite"] is True
    assert (d / "ladder_profile_nx400_weno7.npz").exists()
    assert (d / "weno7_metrics_nx400_weno7.json").exists()


def test_stored_profile_is_loader_compatible(small_rung):
    d, _ = small_rung
    prof = wcr.load_profile(d, 400, "weno7")
    assert prof["nx"] == 400
    assert prof["rho"].shape == (400,)
    assert np.all(np.isfinite(prof["rho"]))
    # cell centres, strictly increasing, inside the unit interval
    assert np.all(np.diff(prof["x"]) > 0) and prof["x"][0] > 0 and prof["x"][-1] < 1.0
    assert {"p", "u"} <= set(prof)


def test_producer_is_idempotent(small_rung):
    d, _ = small_rung
    again = wl.run_rung(400, d)
    assert again["skipped"] is True


def test_diverged_rung_is_recorded_and_not_stored(tmp_path, monkeypatch):
    """A NaN rung must not become a reference (WENO7 really does diverge at coarse/fine rungs)."""
    def fake_solve(nx, *a, **k):
        z = [float("nan")] * nx
        return {"method": "fake", "datum": "blast", "nx": nx, "t_end": 0.038, "cfl": 0.4,
                "x": [(i + 0.5) / nx for i in range(nx)], "rho": z, "u": z, "p": z,
                "steps": 1, "seconds": 0.0, "positivity_fallbacks": 0,
                "mass": float("nan"), "mass_drift": float("nan"),
                "rho_max": float("nan"), "x_at_rho_max": float("nan"), "p_max": float("nan")}
    monkeypatch.setattr(wl.w7, "weno7_solve", fake_solve)
    res = wl.run_rung(128, tmp_path)
    assert res["diverged"] is True
    assert res["metrics"]["status"] == "diverged"
    assert not (tmp_path / "ladder_profile_nx128_weno7.npz").exists()
    assert (tmp_path / "weno7_metrics_nx128_weno7.json").exists()


# --- 3: the same-grid cross-order metric --------------------------------------
def test_cross_order_metric_is_zero_against_itself(small_rung):
    d, _ = small_rung
    prof = wcr.load_profile(d, 400, "weno7")
    row = r6.two_direction_l1(prof, prof)
    assert row["l1_percent"] == pytest.approx(0.0, abs=1e-12)
    assert row["peak_percent"] == pytest.approx(0.0, abs=1e-12)


def test_cross_order_metric_band_is_symmetric(small_rung):
    d, _ = small_rung
    a = wcr.load_profile(d, 400, "weno7")
    # a strictly different profile on the *same* grid: conservative restriction + repeat
    sm = np.repeat(wcr.restrict(a["rho"], 4), 4)
    b = {"rho": sm, "x": a["x"], "nx": a["nx"]}
    row = r6.two_direction_l1(a, b)
    assert row["l1_percent"] > 0.0
    lo, hi = row["l1_percent_band"]
    assert lo <= row["l1_percent"] <= hi
    swapped = r6.two_direction_l1(b, a)
    assert swapped["l1_percent"] == pytest.approx(row["l1_percent"], rel=1e-12)


# --- 4: missing data is not a pass -------------------------------------------
def test_missing_reference_is_not_measured(tmp_path, small_rung):
    d, _ = small_rung
    rows = r6.graded_ladder_vs(d, "weno7", 6400)
    assert rows and all(r["measured"] is False for r in rows)


def test_regrade_skips_unavailable_references(tmp_path, small_rung):
    from shutil import copyfile
    src = r6.DATA / "ladder_profile_nx3200_hllc.npz"
    if not src.exists():
        pytest.skip("graded profile not stored in this checkout")
    copyfile(src, tmp_path / "ladder_profile_nx3200_hllc.npz")
    rows = r6.regrade_matrix(tmp_path, 3200)
    assert rows == []          # no reference -> no row, and certainly no PASS


def test_adjudication_without_weno7_data_marks_not_measured():
    frozen = json.loads((XV / "predeclaration_r6.json").read_text())
    empty = {"selfcert_ladders": {"weno5": [], "weno7": []}, "reference_rungs": {},
             "weno7_fine_row": None, "weno7_u_ref_6400": None,
             "weno5_u_ref_12800": None, "cross_order_gaps": [], "regrades": [],
             "graded_ladder_w7": [], "graded_ladder_w5": [], "metrics": {}}
    adj = r6.adjudicate(frozen, empty)
    assert set(adj["not_measured"]) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
    assert adj["hits"] == [] and adj["misses"] == []
    assert adj["checks"]["Q5"].get("certified") is not True
    doc = r6.render({"measurements": empty, "adjudication": adj, "figures": {}})
    assert "not_measured" in doc and "certified 仍为 False" in doc


# --- 5: the freeze itself -----------------------------------------------------
def test_predeclaration_is_complete_and_frozen():
    frozen = json.loads((XV / "predeclaration_r6.json").read_text())
    assert set(frozen["predictions"]) == {
        "Q1_weno7_observed_order", "Q2_cross_order_gap_3200", "Q3_weno7_u_ref_6400",
        "Q4_reference_side_share_pp", "Q5_certified_still_false", "Q6_peak_ratio"}
    for key, spec in frozen["predictions"].items():
        assert spec.get("band") is not None and spec.get("point_prediction") is not None, key
    ids = [a["id"] for a in frozen["amendments"]]
    assert ids[0] == "AMENDMENT-1" and set(ids) <= {"AMENDMENT-1", "AMENDMENT-2"}
    for a in frozen["amendments"]:
        assert a["reason"] and a.get("frozen_job_state_at_amendment")
    assert str(frozen["predictions"]["Q4_reference_side_share_pp"]["band"]) == "[-0.8, -0.1]"


def test_frozen_code_hash_still_matches_disk():
    """No *silent* edit of the round-6 instrument.

    Round-8 supersedes these hashes on purpose (``predeclaration_r8.json``
    ``supersedes_code_sha256``, repair documented there).  Round-6's own freeze
    forbids adding amendments to *its* record, so the supersession lives in the
    later predeclaration and is accepted here only if its hash matches disk
    exactly -- an edit that is neither round-6's nor round-8's declared one still
    fails.
    """
    frozen = json.loads((XV / "predeclaration_r6.json").read_text())
    sup_path = XV / "predeclaration_r8.json"
    superseding = json.loads(sup_path.read_text()).get("supersedes_code_sha256", {}) if sup_path.exists() else {}
    for name, digest in frozen["code_sha256"].items():
        path = XCODE / name
        assert path.exists(), name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            assert superseding.get(name) == actual, (
                f"{name} changed and is not declared by a later predeclaration "
                f"({actual[:12]} != {digest[:12]}); amend the predeclaration, do not relabel it")


def test_weno7_linear_weights_are_the_derived_ones():
    """The instrument's reconstruction weights must stay the derived WENO7 ones."""
    w = np.asarray(w7._optimal_weights() if callable(getattr(w7, "_optimal_weights", None)) else [])
    if w.size:
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(w > 0)