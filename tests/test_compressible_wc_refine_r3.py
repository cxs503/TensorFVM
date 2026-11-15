"""Tests for the round-3 predeclaration adjudication (F1/F2/F3) and the CFL control.

Two layers, deliberately separated:

* **logic tests** drive :func:`wc_refine.check_predeclaration` with synthetic
  ladders, so pass/fail branches are exercised without any stored run.  A
  criterion that can only return ``True`` on the real data is not a criterion.
* **data tests** check the stored ladder (when the nx=6400/12800 MUSCL rungs are
  present) against an *independent* recomputation of the same numbers, so the
  reported verdict cannot drift away from the profiles it claims to summarise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import wc_refine as wr

HERE = Path(__file__).resolve().parents[1] / "docs" / "XVAL-FVM-M18737-ROUND2"


def _row(nx, l1, peak=0.05, mass=0.001):
    return {"nx": nx, "l1_percent": l1, "peak_percent": peak, "mass_percent": mass,
            "measured": True}


def _verdict(rows, ref_l1=0.5437):
    return {"ladder": rows, "not_measured": [n for n in wr.HLLC_RUNGS
                                             if n not in {r["nx"] for r in rows}],
            "reference": {"family": "weno5", "nx": 12800,
                          "self_certification": {"l1_percent": ref_l1,
                                                 "peak_percent": 0.023, "mass_percent": 0.0005,
                                                 "self_certified": True}}}


# --------------------------------------------------------------------------- #
# F1 -- point value inside the predeclared band
# --------------------------------------------------------------------------- #
def test_f1_accepts_measurement_inside_the_frozen_band():
    rows = [_row(1600, 3.2513), _row(3200, 1.7685), _row(6400, 0.95)]
    out = wr.check_predeclaration(_verdict(rows))
    assert out["F1"]["passed"] is True
    assert out["F1"]["band_percent"] == [0.854, 1.023]
    assert 0.854 <= out["F1"]["measured_percent"] <= 1.023


def test_f1_rejects_a_measurement_outside_the_band():
    rows = [_row(1600, 3.2513), _row(3200, 1.7685), _row(6400, 1.30)]
    out = wr.check_predeclaration(_verdict(rows))
    assert out["F1"]["passed"] is False
    # a miss must still carry the numbers, direction included
    assert out["F1"]["measured_percent"] == pytest.approx(1.30)
    assert out["F1"]["predicted_percent"] == pytest.approx(0.9620, abs=5e-3)


def test_f1_is_undecided_when_the_target_rung_was_never_run():
    out = wr.check_predeclaration(_verdict([_row(1600, 3.2513), _row(3200, 1.7685)]))
    assert out["F1"]["passed"] is None
    assert out["F1"]["measured_percent"] is None
    assert out["overall"] == "undecided"


# --------------------------------------------------------------------------- #
# F2 -- sequence order, and the plateau that would falsify the refinement route
# --------------------------------------------------------------------------- #
def test_f2_sequence_order_is_fitted_over_three_rungs_not_one_pair():
    rows = [_row(1600, 3.2513), _row(3200, 1.7685), _row(6400, 0.893)]
    fit = wr.sequence_order_lsq(rows, rungs=(1600, 3200, 6400))
    assert fit["rungs"] == [1600, 3200, 6400]
    two_point = wr.observed_order(3.2513, 0.893, 1600, 6400)
    assert fit["order"] == pytest.approx(two_point, abs=0.08)
    assert wr.check_predeclaration(_verdict(rows))["F2"]["passed"] is True


def test_f2_flags_a_plateau_and_fails_the_order_band():
    # L1 stops falling from 3200 to 6400 -> near-zero order, above the 1.5 % bar
    rows = [_row(1600, 3.2513), _row(3200, 1.7685), _row(6400, 1.76)]
    out = wr.check_predeclaration(_verdict(rows))
    assert out["F2"]["order_in_band"] is False
    assert out["F2"]["plateau"] is True
    assert out["F2"]["passed"] is False
    assert out["overall"] == "fail"


def test_f2_does_not_call_a_still_falling_ladder_a_plateau():
    rows = [_row(1600, 3.2513), _row(3200, 1.7685), _row(6400, 1.60)]
    out = wr.check_predeclaration(_verdict(rows))
    assert out["F2"]["plateau"] is False


# --------------------------------------------------------------------------- #
# F3 -- certified ruling and the honest-FAIL alternative
# --------------------------------------------------------------------------- #
def test_f3_names_the_first_certified_rung():
    rows = [_row(1600, 3.2513), _row(3200, 1.7685),
            _row(6400, 0.893, peak=0.044), _row(12800, 0.51, peak=0.02)]
    out = wr.check_predeclaration(_verdict(rows))
    assert out["F3"]["certified_at_nx"] == 6400
    assert out["F3"]["passed"] is True
    assert out["overall"] == "pass"


def test_f3_requires_the_peak_gate_too():
    rows = [_row(1600, 3.2513), _row(3200, 1.7685),
            _row(6400, 0.5, peak=1.4)]
    out = wr.check_predeclaration(_verdict(rows))
    assert out["F3"]["certified_at_nx"] is None
    assert "honest FAIL" in out["F3"]["ruling"]


def test_f3_honest_fail_reports_the_cost_of_the_unreached_bar():
    rows = [_row(1600, 3.2513), _row(3200, 1.7685), _row(6400, 1.20),
            _row(12800, 1.05)]
    out = wr.check_predeclaration(_verdict(rows))
    assert out["F3"]["certified_at_nx"] is None
    assert out["F3"]["path_exhausted"] is True
    assert np.isfinite(out["F3"]["nx_needed_for_1pct"])
    assert out["F3"]["nx_needed_for_1pct"] > 12800


# --------------------------------------------------------------------------- #
# CFL control on synthetic profiles
# --------------------------------------------------------------------------- #
def _write_profile(dirpath: Path, tag: str, rho: np.ndarray) -> None:
    x = (np.arange(rho.size) + 0.5) / rho.size
    np.savez_compressed(dirpath / f"ladder_profile_{tag}.npz", x=x, rho=rho)


def _field(nx: int, amp: float) -> np.ndarray:
    x = (np.arange(nx) + 0.5) / nx
    return 1.0 + 0.5 * np.sin(2 * np.pi * x) + amp * np.exp(-((x - 0.6) / 0.02) ** 2)


def test_cfl_sensitivity_measures_the_delta_and_keeps_the_decision(tmp_path):
    _write_profile(tmp_path, "nx12800_weno5", _field(12800, 0.0))
    _write_profile(tmp_path, "nx6400_hllc", _field(6400, 0.10))
    _write_profile(tmp_path, "nx6400_hllc_cfl0.2", _field(6400, 0.101))
    out = wr.cfl_sensitivity(tmp_path, 6400, 0.2, ref_nx=12800)
    assert out["l1_percent_baseline"] > 0.0
    assert out["delta_l1_pp"] < 0.05
    assert out["time_error_negligible"] is True
    assert out["decision_invariant"] is True


def test_cfl_sensitivity_reports_a_large_delta_as_not_negligible(tmp_path):
    _write_profile(tmp_path, "nx12800_weno5", _field(12800, 0.0))
    _write_profile(tmp_path, "nx6400_hllc", _field(6400, 0.05))
    _write_profile(tmp_path, "nx6400_hllc_cfl0.2", _field(6400, 0.60))
    out = wr.cfl_sensitivity(tmp_path, 6400, 0.2, ref_nx=12800)
    assert out["delta_l1_pp"] > 0.05
    assert out["time_error_negligible"] is False
    assert out["decision_invariant"] is False


def test_cfl_sensitivity_raises_when_the_control_run_is_missing(tmp_path):
    _write_profile(tmp_path, "nx12800_weno5", _field(12800, 0.0))
    _write_profile(tmp_path, "nx6400_hllc", _field(6400, 0.05))
    with pytest.raises(wr.DataMissing):
        wr.cfl_sensitivity(tmp_path, 6400, 0.2, ref_nx=12800)


# --------------------------------------------------------------------------- #
# stored-ladder consistency (skipped until the nx=6400/12800 rungs exist)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nx", [6400, 12800])
def test_stored_rung_matches_independent_recomputation(nx):
    data = wr.default_data_dir()
    if wr.try_load_profile(data, nx, "hllc") is None:
        pytest.skip(f"nx={nx} MUSCL rung not computed yet")
    got = wr.load_profile(data, nx, "hllc")
    ref = wr.load_profile(data, 12800, "weno5")
    independent = wr.cell_average_l1(got["rho"], ref["rho"])["l1_percent"]
    ladder = {r["nx"]: r for r in wr.audit(data)["ladder"] if r.get("measured")}
    assert ladder[nx]["l1_percent"] == pytest.approx(independent, rel=1e-9)


def test_round3_summary_if_present_agrees_with_a_fresh_audit():
    summary_path = HERE / "summary_r3.json"
    if not summary_path.exists():
        pytest.skip("round-3 summary not rendered yet")
    recorded = json.loads(summary_path.read_text())
    fresh = wr.check_predeclaration(wr.audit(wr.default_data_dir()))
    assert recorded["predeclaration_checks"]["F3"]["certified_at_nx"] == \
        fresh["F3"]["certified_at_nx"]
    assert recorded["predeclaration_checks"]["overall"] == fresh["overall"]