"""Round 20 tests: lower-Re sister cases for ``CASE-FVM-PERIODIC-HILLS``.

What these tests are for
------------------------
The round-20 increment is (a) a closure-free branch in the shared production
driver and (b) an *Reynolds-number-aware* reference/scoring path.  Both are
places where a silent fallback would corrupt a verdict rather than crash:

* a typo'd closure name falling back to "laminar" would publish a
  closure-free number labelled as a closure result;
* scoring a ``Re = 700`` artefact against the ``Re = 2800`` reattachment length
  is the exact error the module exists to prevent, and it produces a *plausible*
  number (``5.24`` vs ``5.41`` differ by 3 %), so no exception would ever fire;
* interpolating between two published cases invents a reference that nobody
  published, and the artefact would look no different from a real one.

So the tests assert invariants, not merely that the code runs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import periodic_hill_reference as ref
from tensorfvm.verification.periodic_hill_lower_re import (
    KB_REATTACHMENT_BY_RE,
    attribution,
    attribution_limits,
    reattachment_reference,
    score_case,
    separation_reference,
    sister_table,
)
from tensorfvm.verification.periodic_hill_round11 import (
    CLOSURES,
    _closure_block,
    build_solver,
)

ARTEFACTS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


# --------------------------------------------------------------------------- #
# 1. the reference table is single-sourced and not invented
# --------------------------------------------------------------------------- #
def test_kb_table_is_cross_checked_against_the_source_quote():
    """The five values must be the ones the reference module quotes verbatim.

    Independence: the numbers are re-parsed out of the *quotation string* that
    :mod:`periodic_hill_reference` carries from the ERCOFTAC page, which is a
    different artefact from the dict under test.  A typo or an "adjusted" value
    in either place fails here.
    """
    quote = ref.__doc__
    match = re.search(
        r"x_R/h = 5\.24,\s*\n?\s*5\.19,\s*(\n?\s*)5\.41,\s*5\.09,\s*and\s*4\.69",
        quote)
    assert match is not None, "the verbatim KB quotation is gone from periodic_hill_reference"
    quoted = [5.24, 5.19, 5.41, 5.09, 4.69]
    quoted_re = [700.0, 1400.0, 2800.0, 5600.0, 10595.0]
    assert list(KB_REATTACHMENT_BY_RE) == quoted_re
    assert [KB_REATTACHMENT_BY_RE[k] for k in quoted_re] == quoted
    # the Re=2800 entry is the *same object* as the gate's constant, not a copy
    assert KB_REATTACHMENT_BY_RE[2800.0] == ref.X_R_OVER_H == 5.41


def test_reference_lookup_is_exact_and_never_interpolated():
    for re_value, expected in KB_REATTACHMENT_BY_RE.items():
        got = reattachment_reference(re_value)
        assert got["reynolds"] == re_value
        assert got["x_r_over_H"] == expected
        assert got["band"] == pytest.approx((expected * 0.9, expected * 1.1))
    # Reynolds numbers that no source tabulates must be reported as absent
    assert reattachment_reference(1000.0) is None
    assert reattachment_reference(2100.0) is None
    assert reattachment_reference(3000.0) is None
    assert reattachment_reference(float("nan")) is None
    assert reattachment_reference(float("inf")) is None


def test_separation_reference_is_withheld_where_the_source_has_no_number():
    """Re=2800 has *no* published x_S/H -- WITHHELD, not the legacy 0.227."""
    assert separation_reference(2800.0) is None
    assert separation_reference(1400.0) is None
    got = separation_reference(700.0)
    assert got["x_s_over_H"] == pytest.approx(0.3)
    assert got["approximate"] is True


# --------------------------------------------------------------------------- #
# 2. the closure-free branch of the production driver
# --------------------------------------------------------------------------- #
def test_laminar_is_a_first_class_closure_with_no_provider():
    assert "laminar" in CLOSURES
    solver = build_solver("laminar", nx=24, ny=12, reynolds=700.0, dt=0.02)
    assert solver.turbulence is None
    block = _closure_block("laminar", solver)
    assert block["closure_free"] is True
    assert block["name"] == "laminar"


def test_laminar_run_keeps_nu_t_identically_zero_and_the_bulk_velocity():
    solver = build_solver("laminar", nx=24, ny=12, reynolds=700.0, dt=0.02)
    solver.advance(60)
    assert np.all(solver.nu_t == 0.0), "a closure-free run must not accumulate eddy viscosity"
    assert math.isfinite(float(np.abs(solver.u).max()))
    assert abs(float(solver.bulk_velocity()) - solver.config.bulk_velocity) < 1e-6
    # the momentum operator must be the molecular one only (round-15 eta_face
    # viscosity would otherwise inject a closure's face viscosity here)
    assert solver.nu > 0.0


def test_unknown_closure_still_raises_instead_of_falling_back_to_laminar():
    for bad in ("les", "", "LAMINAR2", "none"):
        with pytest.raises(ValueError):
            build_solver(bad, nx=12, ny=6, reynolds=700.0)


# --------------------------------------------------------------------------- #
# 3. scoring: the Reynolds number of the run decides the reference
# --------------------------------------------------------------------------- #
def _payload(reynolds, x_r, *, diverged=None, x_s=None):
    return {
        "reynolds": reynolds,
        "config": {"reynolds": reynolds},
        "closure": {"name": "laminar"},
        "diverged_at_step": diverged,
        "samples": 100 if diverged is None else 0,
        "time_averaged": {"x_r_over_H": x_r, "x_s_over_H": x_s},
    }


def test_score_uses_the_payload_reynolds_not_the_re2800_default():
    """Re=700 at 5.24 must PASS, not be scored against the 5.41 gate."""
    row = score_case(_payload(700.0, 5.24))
    assert row["reference_x_r_over_H"] == 5.24
    assert row["verdict"] == "PASS"
    assert row["error"] == pytest.approx(0.0, abs=1e-12)
    assert row["below_3_percent"] is True


def test_score_band_edges_are_consistent_with_hand_arithmetic():
    ref700 = 5.24
    for factor, expect in ((1.0, True), (1.09, True), (0.91, True),
                           (1.11, False), (0.89, False)):
        value = ref700 * factor
        row = score_case(_payload(700.0, value))
        assert row["below_10_percent"] is expect, (factor, row)
        assert row["error"] == pytest.approx(abs(factor - 1.0), rel=1e-9)
        assert row["below_3_percent"] is (abs(factor - 1.0) < 0.03)


def test_score_reports_withheld_for_an_untabulated_reynolds_number():
    row = score_case(_payload(1000.0, 5.3))
    assert row["verdict"] == "WITHHELD"
    assert row["below_10_percent"] is None and row["below_3_percent"] is None
    assert "no interpolation" in row["reason"]


def test_divergence_cannot_be_scored_as_physics():
    """A run that diverged is DIVERGED even when its stored number looks fine."""
    row = score_case(_payload(700.0, 5.24, diverged=281))
    assert row["verdict"] == "DIVERGED"
    assert row["below_10_percent"] is False
    assert "divergence at step 281" in row["reason"]


def test_missing_crossing_is_undefined_not_skipped():
    row = score_case(_payload(700.0, None))
    assert row["verdict"] == "UNDEFINED"
    assert row["below_10_percent"] is False


def test_certified_operator_overrides_the_payloads_derived_number():
    """The verdict must come from the certified operator, not the payload field.

    A synthetic mean shear whose sign changes at known abscissae is scored
    against the payload's *contradicting* stored value; the certified number
    must win and say so.
    """
    x = np.linspace(0.0, 9.0, 901)          # dx = 0.01, so sign nodes are 4.99/5.00
    # negative between x=1.0 and x=5.0 -> separation 1.0, reattachment 5.0
    tau = np.where((x > 1.0) & (x < 5.0), -1.0, 1.0)
    row = score_case(_payload(700.0, 9.99), reynolds=700.0,
                     mean_tau_b=tau, x_over_h=x)
    # the certified operator interpolates the crossing *within* the sign-change
    # cell, so the exact value is 4.995 -- the payload's 9.99 is discarded
    assert row["x_r_over_H"] == pytest.approx(4.995, abs=1e-9)
    assert "certified" in row["x_r_provenance"]
    assert row["x_r_over_H"] != 9.99
    assert row["error"] == pytest.approx(abs(4.995 - 5.24) / 5.24, rel=1e-9)


def test_certified_operator_requires_both_arguments():
    with pytest.raises(ValueError):
        score_case(_payload(700.0, 5.0), mean_tau_b=np.zeros(10))


def test_x_s_is_scored_only_where_the_source_publishes_a_number():
    row = score_case(_payload(700.0, 5.24, x_s=0.31))
    assert row["x_s_reference"] == pytest.approx(0.3)
    assert row["x_s_error"] == pytest.approx(abs(0.31 - 0.3) / 0.3)
    assert row["x_s_approximate_reference"] is True
    row2800 = score_case(_payload(2800.0, 5.41, x_s=0.3))
    assert "x_s_reference" not in row2800  # withheld at Re=2800, not 0.0


# --------------------------------------------------------------------------- #
# 4. the attribution rule is decided in advance and cannot flatter the result
# --------------------------------------------------------------------------- #
def _row(reynolds, x_r, reference, verdict, error, closure="laminar"):
    return {"reynolds": reynolds, "closure": closure, "x_r_over_H": x_r,
            "reference_x_r_over_H": reference, "verdict": verdict,
            "error": error, "below_10_percent": verdict == "PASS",
            "below_3_percent": bool(error is not None and error < 0.03)}


def test_attribution_closure_case():
    lam = [_row(700.0, 5.24, 5.24, "PASS", 0.0), _row(2800.0, 5.5, 5.41, "PASS", 0.0166)]
    clo = [_row(2800.0, 7.68, 5.41, "FAIL", 0.42, closure="sst")]
    out = attribution(lam, clo)
    assert out["status"] == "closure_attributable"
    assert out["laminar_2800_error"] == pytest.approx(0.0166)


def test_attribution_framework_case():
    lam = [_row(700.0, 5.2, 5.24, "PASS", 0.0076),
           _row(2800.0, 7.9, 5.41, "FAIL", 0.46)]
    out = attribution(lam, [])
    assert out["status"] == "framework_attributable"
    assert out["sister_max_error"] == pytest.approx(0.0076)


def test_attribution_is_inconclusive_when_effects_are_entangled():
    # closure-free Re=2800 is off by 20 %, but a closure-free sister case is off
    # by 30 % -- the sister case cannot clear the framework, so nothing is
    # attributable and the rule must say so instead of blaming the closure.
    lam = [_row(700.0, 6.81, 5.24, "FAIL", 0.30),
           _row(2800.0, 6.49, 5.41, "FAIL", 0.20)]
    out = attribution(lam, [_row(2800.0, 7.68, 5.41, "FAIL", 0.42, closure="sst")])
    assert out["status"] == "inconclusive"
    assert "not separable" in out["reason"]


def test_attribution_needs_both_rows():
    out = attribution([], [])
    assert out["status"] == "inconclusive"
    assert "no closure-free" in out["reason"]
    # a diverged/undefined sister row carries no error and cannot decide anything
    out2 = attribution([_row(700.0, None, 5.24, "DIVERGED", None),
                        _row(2800.0, 5.5, 5.41, "PASS", 0.0166)], [])
    assert out2["status"] == "inconclusive"


def test_attribution_limits_are_shipped_with_the_rule():
    limits = attribution_limits()
    assert set(limits) == {"agreement_proves", "agreement_does_not_prove",
                           "disagreement_proves", "reference_caveat"}
    assert "3-D" in limits["agreement_does_not_prove"]


def test_sister_table_counts_every_verdict_class():
    payloads = {
        "a": _payload(700.0, 5.24),
        "b": _payload(1400.0, 6.5),
        "c": _payload(1000.0, 5.3),
        "d": _payload(2800.0, None),
        "e": _payload(5600.0, 5.0, diverged=12),
    }
    table = sister_table(payloads)
    assert (table["n_pass"], table["n_fail"], table["n_withheld"],
            table["n_undefined"], table["n_diverged"]) == (1, 1, 1, 1, 1)
    assert [r["reynolds"] for r in table["rows"]] == [700.0, 1000.0, 1400.0, 2800.0, 5600.0]


# --------------------------------------------------------------------------- #
# 5. artefacts (when the round-20 runs have landed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("closure,reynolds,tag", [
    ("laminar", 700.0, "96x48"),
    ("laminar", 1400.0, "96x48"),
    ("laminar", 2800.0, "96x48"),
])
def test_round20_artefact_is_self_consistent(closure, reynolds, tag):
    stem = f"round20_{closure}_Re{reynolds:g}_{tag}"
    path = ARTEFACTS / f"{stem}.json"
    if not path.exists():
        pytest.skip(f"{stem}.json not produced yet (cluster job pending)")
    payload = json.loads(path.read_text())
    assert payload["lower_re"]["reynolds"] == reynolds
    assert payload["lower_re"]["reference_x_r_over_H"] == \
        KB_REATTACHMENT_BY_RE[reynolds]
    assert payload["gate_reference_reynolds"] == 2800.0
    # the closure-free claim must be recorded by the artefact itself
    assert payload["closure"]["closure_free"] is True
    assert payload["lower_re"]["x_r_over_H"] is not None or \
        payload["lower_re"]["verdict"] in ("UNDEFINED", "DIVERGED")