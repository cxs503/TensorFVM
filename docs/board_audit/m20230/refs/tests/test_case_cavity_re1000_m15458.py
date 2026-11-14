"""Anchor ``CASE-CAVITY-RE1000-FVM``: the round-2 assertions, against the current bundle.

History: mission m15458 wrote this test against its own artefact schema
(``verdict_5pct``, ``case["solver"]["max_divergence"]``).  The case is now a real
module (:mod:`tensorfvm.case_cavity_re1000`) with one canonical schema, so the
three invariants the round-2 test pinned are re-asserted here **in that schema**
and nothing else -- the detailed gate lives in
``tests/test_case_cavity_re1000.py`` and the two must not drift into two copies
of the same judgement.

Invariants kept (they were the point of the round-2 test):
1. every reported row is converged and its mass residual is at machine level;
2. the stored verdict is reproducible from the numbers stored beside it;
3. the key quantities sit inside the case-pool's 5 % band, while deliberately
   under-resolved control rows stay reported as failing.
"""

from __future__ import annotations

import json
import os

import pytest

from tensorfvm.case_cavity_re1000 import judge
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "docs", "CASE-CAVITY-RE1000-FVM", "artifacts", "summary.json")
RE = 1000.0
TOL = 0.05


def _summary() -> dict:
    if not os.path.exists(SUMMARY):
        pytest.skip(f"no artefact bundle at {SUMMARY}")
    with open(SUMMARY) as fh:
        return json.load(fh)


def test_every_reported_row_is_converged():
    rows = _summary()["cases"]
    assert rows, "no cases in the bundle"
    for case in rows:
        assert case["solver"]["converged"] is True, f"n={case['n']} not converged"
        assert case["solver"]["residual_rate"] <= 1e-6
        assert case["max_divergence"] < 1e-10


def test_ledger_verdict_matches_its_own_numbers():
    for case in _summary()["cases"]:
        verdict = case["verdict"]
        assert verdict["tolerance"] == TOL
        assert verdict["pass"] is judge(case)["pass"], f"n={case['n']} verdict drifted"
        for check in verdict["checks"]:
            value = check["value"]
            if check["comparator"] == "is":
                ok = value is bool(check["threshold"])
            else:
                ok = float(value) <= float(check["threshold"])
            assert check["pass"] is bool(ok), f"n={case['n']} {check['id']}"


def test_key_quantities_inside_five_percent():
    bundle = _summary()
    ref_u = dict(table(RE, "u"))
    ref_v = dict(table(RE, "v"))
    gx, gy = VORTEX_CENTRE[RE]
    production = [c for c in bundle["cases"]
                  if not c.get("role", "").startswith("control")]
    assert production, "no production-resolution row in the bundle"
    for case in bundle["cases"]:
        if case.get("role", "").startswith("control"):
            # the deliberately under-resolved control must stay reported as failing,
            # so a 5 % pass cannot hide a coarse grid
            assert case["verdict"]["pass"] is False
            continue
        y_ref = case["u"]["extremum"]["min_reference"][0]
        x_ref = case["v"]["extremum"]["max_reference"][0]
        u_min = case["u"]["extremum"]["min_computed"][1]
        v_max = case["v"]["extremum"]["max_computed"][1]
        assert abs(u_min - ref_u[y_ref]) / abs(ref_u[y_ref]) < TOL
        assert abs(v_max - ref_v[x_ref]) / abs(ref_v[x_ref]) < TOL
        vx, vy = case["vortex_centre"]["computed"]
        assert max(abs(vx - gx), abs(vy - gy)) < TOL