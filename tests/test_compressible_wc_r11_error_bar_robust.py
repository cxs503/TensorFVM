"""Regression gate for round-11: robustness of the *error bar convention* itself.

Round 1-10 pinned the numbers.  Round-11 pins a different kind of claim: given the same
measured pair difference ``e`` and order ``p``, the honest upper bound under any other
legally-motivated convention must be ``L1_graded + eps(convention)`` and the frozen bars
(E4 = 1.5 %, E5 = 1.0 %, ledger = 15 %) must be reported *as they fall*, never upgraded.

Three kinds of claim are pinned:

* **convention arithmetic** -- B0 must reproduce the library's own ``u_ref_percent`` to
  floating-point identity; B2/B3 must be exactly ``Fs * B0``; B1 must bracket B0 (a wider
  order band can only enlarge the bar); B4 must be >= B0 whenever it is licensed.
* **licensing logic** -- ``e/(1-r)`` is only an upper bound when the difference-ratio
  sequence does not increase.  The predicate is exercised on synthetic sequences so the
  test does not merely re-run the one case that happens to be committed.
* **honesty** -- an unlicensed or missing bar must never produce a ``pass``; ``<1 %`` must
  not be claimable while the convention layer says otherwise.  ``not_licensed`` is not a
  pass, and ``not_measured`` is not a pass.

If the extended reference rung lands later, ``compute()`` picks it up automatically and the
thresholds below still apply -- nothing here is pinned to nx=12800.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "XVAL-FVM-M18737-ROUND2"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorfvm.verification import wc_ref_uncertainty as U  # noqa: E402


def _load():
    name = "ref25600_r11_robust"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, XVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R11 = _load()


@pytest.fixture(scope="module")
def summary():
    return R11.compute()


# --------------------------------------------------------------------------- #
# 1. the layer is measured, not skipped
# --------------------------------------------------------------------------- #
def test_status_measured(summary):
    assert summary["status"] == "measured", "no reference pair carries an uncertainty"
    for key in ("B0", "B1", "B2", "B3", "B4"):
        assert key in summary["bars"]


def test_all_six_criteria_reported(summary):
    assert set(summary["verdicts"]) == {"R1", "R2", "R3", "R4", "R5", "R6"}
    allowed = {"pass", "fail", "not_licensed"}
    assert set(summary["verdicts"].values()) <= allowed


# --------------------------------------------------------------------------- #
# 2. convention arithmetic (no convention may be silently cheaper than another)
# --------------------------------------------------------------------------- #
def test_B0_matches_library_u_ref(summary):
    """R1 is a formula-identity gate: B0 must equal the library's Richardson value."""
    assert summary["R1"]["verdict"] == "pass"
    assert summary["R1"]["rel_diff"] is not None
    assert summary["R1"]["rel_diff"] < 1e-12


def test_safety_factor_scaling_exact(summary):
    b0 = summary["bars"]["B0"]["eps_percent"]
    assert summary["bars"]["B2"]["eps_percent"] == pytest.approx(
        R11.FS_GCI * b0, abs=2e-4)
    assert summary["bars"]["B3"]["eps_percent"] == pytest.approx(
        R11.FS_NONASYMPTOTIC * b0, abs=5e-4)
    assert R11.FS_NONASYMPTOTIC > R11.FS_GCI > 1.0


def test_order_band_widens_the_bar(summary):
    """A wider order band cannot *shrink* the uncertainty."""
    b = summary["bars"]
    assert b["B1"]["eps_percent"] >= b["B0"]["eps_percent"]
    assert b["B1"]["eps_lo_percent"] <= b["B0"]["eps_percent"] <= b["B1"]["eps_percent"]
    assert b["B1"]["spread_percent"] == pytest.approx(
        b["B1"]["eps_percent"] - b["B1"]["eps_lo_percent"], abs=2e-4)


def test_order_free_bar_not_below_richardson(summary):
    """For 0<r<1, e/(1-r) >= e/(2^p-1) whenever r >= 2^-p; check the committed case."""
    b = summary["bars"]
    if b["B4"]["eps_percent"] is None:
        pytest.skip("no usable ratio for the order-free bar")
    r, p = b["ratio"], b["order"]
    assert 0.0 < r < 1.0
    assert b["B4"]["eps_percent"] >= b["B0"]["eps_percent"]
    # and the closed form itself
    assert b["B4"]["eps_percent"] == pytest.approx(
        b["e_percent"] / (1.0 - r), rel=1e-4)


# --------------------------------------------------------------------------- #
# 3. B4 licensing logic, exercised on synthetic sequences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rseq,licensed", [
    ([0.53, 0.57], False),          # the committed case: ratio increases
    ([0.57, 0.53], True),           # contracting: geometric bound legal
    ([0.60, 0.60], True),           # constant ratio is non-increasing
    ([0.50, 0.60, 0.55], False),    # any increase poisons the licence
    ([0.55, 0.50, 0.40], True),
    ([], False),                    # no evidence -> not licensed
    ([0.55], True),                 # single pair carries no increase
])
def test_licensing_predicate(rseq, licensed):
    bars = R11.bar_conventions(1.0, 1.0, 0.5, rseq)
    assert bars["B4"]["licensed"] is licensed


def test_unlicensed_bar_is_reported_but_excluded(summary):
    b4 = summary["bars"]["B4"]
    if not b4["licensed"]:
        assert summary["verdicts"]["R5"] == "not_licensed"
        for row in summary["regrade"]["rows"]:
            assert row["upper_bound_percent"]["B4"] is None


# --------------------------------------------------------------------------- #
# 4. honest bounds are exactly L1 + eps, and never pass by omission
# --------------------------------------------------------------------------- #
def test_upper_bound_is_l1_plus_eps(summary):
    bars = summary["bars"]
    for row in summary["regrade"]["rows"]:
        for key in ("B0", "B1", "B2", "B3", "B4"):
            ub = row["upper_bound_percent"][key]
            eps = bars[key].get("eps_percent")
            if ub is None:
                assert key == "B4" and not bars["B4"]["licensed"]
                continue
            assert ub == pytest.approx(row["l1_percent"] + eps, rel=1e-9)


def test_verdicts_follow_from_bounds(summary):
    fin = summary["regrade"]["finest"]
    for key, v in summary["regrade"]["verdicts"].items():
        ub = fin["upper_bound_percent"][key]
        if ub is None:
            assert v["E4"] == "not_licensed" and v["E5"] == "not_licensed"
            continue
        assert v["E4"] == ("pass" if ub < R11.BAR_E4 else "fail")
        assert v["E5"] == ("pass" if ub < R11.BAR_E5 else "fail")
        assert v["ledger"] == ("pass" if ub < R11.BAR_LEDGER else "fail")


def test_no_convention_claims_sub_one_percent(summary):
    """Negative result pinned: no licensed convention proves <1 % on this datum."""
    fin = summary["regrade"]["finest"]
    for key, ub in fin["upper_bound_percent"].items():
        if ub is None:
            continue
        assert ub >= 1.0, f"{key} unexpectedly proves <1 % ({ub}); re-audit before claiming"


# --------------------------------------------------------------------------- #
# 5. asymptotic-range gate and the order inversion
# --------------------------------------------------------------------------- #
def test_asymptotic_gate_is_honest(summary):
    a = summary["asymptotic"]
    assert a["monotone"] is True              # Richardson is at least usable
    assert a["formal_order_weno5"] == 5.0
    assert a["order_ratio"] == pytest.approx(a["order"] / 5.0, abs=2e-4)
    # observed order is an order of magnitude below formal -> must NOT claim asymptotic
    assert a["asymptotic"] is False


def test_critical_order_closed_form(summary):
    crit = summary["critical_order"]
    e = crit["e_percent"]
    l1 = crit["l1_graded_percent"]
    for name, d in crit["bars"].items():
        if d.get("p_star") is None:
            assert d["bar"] - l1 <= 0
            continue
        # round-trip: the bar is exactly met at p*
        assert l1 + e / (2.0 ** d["p_star"] - 1.0) == pytest.approx(d["bar"], rel=2e-4)
        # a stricter bar needs a better (larger) order
    stars = [d["p_star"] for d in crit["bars"].values() if d.get("p_star") is not None]
    if len(stars) == 2:
        assert crit["bars"]["E5"]["p_star"] > crit["bars"]["E4"]["p_star"]


# --------------------------------------------------------------------------- #
# 6. documented numbers must equal recomputed numbers (no hand-copied drift)
# --------------------------------------------------------------------------- #
def test_document_matches_summary():
    import json
    doc = (XVAL / "R11-ERROR-BAR-ROBUSTNESS.md").read_text()
    summary = json.loads((XVAL / "summary_r11.json").read_text())
    b0 = summary["bars"]["B0"]["eps_percent"]
    assert f"{b0:.4f}" in doc
    ub = summary["regrade"]["finest"]["upper_bound_percent"]["B0"]
    assert f"{ub:.4f}" in doc
    p_star = summary["critical_order"]["bars"]["E5"]["p_star"]
    assert f"{p_star:.4f}" in doc


def test_no_stale_not_measured_marker():
    doc = (XVAL / "R11-ERROR-BAR-ROBUSTNESS.md").read_text()
    assert "无参考对" not in doc, "summary says not_measured but a reference pair exists"