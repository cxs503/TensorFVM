"""Regression gate for round-10: the nx=25600 reference rung on the *frozen* instrument.

Round 5 froze Q1-Q5 waiting for ``WENO5@25600``; round 7 tried to buy the rung back at
CFL 0.8 and admit it through the pre-frozen equivalence gate E1; round 10 runs the rung on
the frozen instrument (CFL 0.4) in a single job instead, so no admission gate is involved.

Three different kinds of claim are pinned here, and they are different in nature:

* **provenance** (E0a/E0b) -- the artifact under the standard name must *be* the frozen
  instrument (``cfl == 0.4``, ``status == ok``, no ``promoted_from``), and the same job's
  re-run of the control rung must reproduce the stored one.  Without this, the new rung is
  not comparable with the older rungs and the ladder arithmetic below is meaningless.
* **arithmetic/consistency invariants** -- honest upper bound is ``point + eps_ref``;
  ``certified`` is exactly ``upper_bound < 1 %``; the recorded verdicts equal the verdicts
  recomputed from the raw artifacts (so the document cannot drift).
* **the superseded route** -- E1 (CFL 0.8) is *recorded, not used*: whether it passed or
  failed, the reference rung in the cube must not carry ``promoted_from``.

Missing artifacts skip (``not_measured`` != pass).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "XVAL-FVM-M18737-ROUND2"
DATA = REPO / "docs" / "woodward-colella-double-blast" / "ladder" / "data"
EVIDENCE = XVAL / "evidence"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorfvm.verification import wc_ref_uncertainty as U  # noqa: E402
from tensorfvm.verification import wc_refine as R  # noqa: E402

EXTENDED = 25600
CONTROL_NX = 6400
FROZEN_CFL = 0.4


def _driver():
    if "_ref25600_r10" in sys.modules:
        return sys.modules["_ref25600_r10"]
    spec = importlib.util.spec_from_file_location("_ref25600_r10", str(XVAL / "ref25600_r10.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ref25600_r10"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def drv():
    return _driver()


@pytest.fixture(scope="module")
def std_metrics():
    path = DATA / f"ladder_metrics_nx{EXTENDED}_weno5.json"
    if not path.exists():
        pytest.skip("frozen-instrument nx=25600 rung absent: still not_measured (not a pass)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def summary():
    path = XVAL / "summary_r10.json"
    if not path.exists():
        pytest.skip("summary_r10.json not generated yet")
    s = json.loads(path.read_text())
    if not s["provenance"].get("profile_present"):
        pytest.skip("nx=25600 rung absent: round-10 verdicts are not_measured")
    return s


# --------------------------------------------------------------------------- #
# E0a - provenance of the admitted rung
# --------------------------------------------------------------------------- #
def test_standard_rung_is_the_frozen_instrument(std_metrics):
    assert std_metrics["cfl"] == pytest.approx(FROZEN_CFL), (
        "the rung under the standard name must be the frozen CFL-0.4 instrument; a relaxed "
        "time step would make it a different instrument than the other ladder rungs")
    assert std_metrics["status"] == "ok"
    assert std_metrics["finite"] is True
    assert "promoted_from" not in std_metrics, "a promoted CFL-0.8 artifact must not pass as frozen"


def test_standard_rung_completed_the_frozen_end_time(std_metrics):
    assert std_metrics["t_end"] == pytest.approx(0.038)
    assert std_metrics["method"] == "fd-weno5-ssprk3-rusanov"
    assert std_metrics["steps"] > 50000, "steps must be the CFL-0.4 count, not the halved CFL-0.8 one"


def test_conservation_of_the_new_rung(std_metrics):
    assert abs(std_metrics["mass_drift"]) < 1e-9


# --------------------------------------------------------------------------- #
# E0b - same job re-run of the control rung
# --------------------------------------------------------------------------- #
def test_rerun_reproduces_the_stored_control_rung(drv):
    rep = drv.reproducibility()
    if not rep.get("rerun_present"):
        pytest.skip("control re-run artifact absent: E0b is not_measured (not a pass)")
    assert rep["same_grid"] is True
    assert rep["rel_delta"] < 1e-12, (
        "the instrument must reproduce the stored rung; otherwise the new rung is not "
        "comparable with the older rungs of the ladder")


# --------------------------------------------------------------------------- #
# reference-side budget
# --------------------------------------------------------------------------- #
def test_ladder_has_five_rungs_and_ends_at_25600(summary):
    pairs = U.selfcert_ladder(DATA, "weno5")
    assert [r["fine_nx"] for r in pairs] == [3200, 6400, 12800, EXTENDED]
    assert all(r["l1_percent"] > 0 for r in pairs)
    assert all(r["u_ref_percent"] is not None for r in pairs)


def test_error_bar_shrinks_with_the_new_rung(summary):
    new = U.u_ref_of_reference(DATA, EXTENDED, "weno5")
    old = U.u_ref_of_reference(DATA, 12800, "weno5")
    assert new["measurable"] is True
    assert new["u_ref_percent"] < old["u_ref_percent"]


def test_error_bar_is_the_richardson_formula(summary):
    for row in U.selfcert_ladder(DATA, "weno5"):
        p = row["order"]
        assert p is not None and p > 0
        assert row["u_ref_percent"] == pytest.approx(row["l1_percent"] / (2.0 ** p - 1.0))


# --------------------------------------------------------------------------- #
# honest grading invariants
# --------------------------------------------------------------------------- #
def test_upper_bound_is_point_plus_error_bar(summary):
    eps = summary["regrade"]["eps_ref_percent"]
    for row in summary["regrade"]["rows"]:
        if not row.get("measured"):
            continue
        assert row["upper_bound_percent"] == pytest.approx(row["l1_percent"] + eps, abs=5e-4)


def test_certified_flag_is_exactly_the_upper_bound_test(summary):
    for row in summary["regrade"]["rows"]:
        if not row.get("measured"):
            continue
        assert row["earned_certified"] == (row["upper_bound_percent"] < R.CERTIFIED_PERCENT)


def test_verdicts_are_recomputed_not_copied(drv, summary):
    """Re-run the driver's own decisions from raw artifacts; the document must not drift."""
    e0a = drv.provenance()
    e0b = drv.reproducibility()
    adj = drv.adjudicate()
    c, ref, g = adj["control"], adj["reference"], adj["regrade"]
    assert summary["verdicts"] == {
        "E0a": e0a.get("E0a"), "E0b": e0b.get("E0b"), "E1": c.get("E1"),
        "E2": ref["E2"].get("verdict"), "E3": ref["E3"].get("verdict"),
        "E4": g["E4"].get("verdict"), "E5": g["E5"].get("verdict"),
    }


def test_e2_verdict_matches_its_own_measurement(summary):
    e2 = summary["reference"]["E2"]
    assert e2["verdict"] != "not_measured"
    inside = e2["band"][0] <= e2["measured"] <= e2["band"][1]
    assert e2["verdict"] == ("pass" if inside else "fail")


def test_e3_verdict_matches_the_measured_bar(summary):
    e3 = summary["reference"]["E3"]
    assert e3["verdict"] == ("pass" if e3["measured"] < e3["bar"] else "fail")


def test_ledger_tolerance_is_met_by_every_measured_quantity(summary):
    """Owner standard: >15 % is unacceptable.  This must hold for the case to ship."""
    fin = summary["regrade"].get("finest")
    assert fin is not None
    assert fin["l1_percent"] < R.TOLERANCE_PERCENT
    assert fin["upper_bound_percent"] < R.TOLERANCE_PERCENT
    assert fin["peak_percent"] < R.TOLERANCE_PERCENT


# --------------------------------------------------------------------------- #
# the superseded CFL-0.8 route
# --------------------------------------------------------------------------- #
def test_e1_is_recorded_and_never_used_to_admit_the_rung(drv, std_metrics):
    """E1 may pass or fail; either way it must not have produced the standard artifact."""
    c = drv.instrument_control()
    assert c["E1"] in {"pass", "fail", "not_measured"}
    assert "promoted_from" not in std_metrics
    if c["E1"] != "pass":
        assert c.get("status") == "measured", (
            "if E1 is not a pass it must at least carry the measured numbers, so the "
            "rejection of the CFL-0.8 route is evidence-backed rather than assumed")


def test_e1_verdict_follows_its_frozen_thresholds(drv):
    c = drv.instrument_control()
    if c.get("status") != "measured":
        pytest.skip("CFL-0.8 control rung absent: E1 is not_measured")
    inside = (c["d_l1_percent"] <= c["bars"]["l1_percent"]
              and c["d_peak_percent"] <= c["bars"]["peak_percent"])
    assert c["E1"] == ("pass" if inside else "fail")


def test_defects_are_finite_and_reported_for_every_pair(summary):
    defects = summary["reference"]["defects"]
    assert len(defects) >= 4
    assert all(d["max_abs_D"] > 0 and math.isfinite(d["max_abs_D"]) for d in defects)
    assert math.isfinite(defects[-1]["order_vs_previous_pair"])