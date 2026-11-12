"""Regression guard for the m15554 board audit of mission m15458 (CASE-CAVITY-RE1000-FVM).

The audit claims live in ``docs/audit-m15554/cavity_audit.json`` (built by
``docs/audit-m15554/verify_cavity_m15458.py`` from the stored fields).  This test
pins the claims that the verdict row relies on, so a later edit cannot quietly turn
"converged, sub-percent at n=128" into a pass without the numbers behind it:

* the discrete fields are steady solutions of the shipped operator (re-stepped rate
  at the 1e-7 gate) and divergence-free to round-off;
* the audit's independently re-read centreline errors decrease monotonically with
  refinement, and the finest row is inside the 3 % criterion on every quantity;
* the audit's numbers agree with the delivered ``summary.json`` numbers (the
  delivery is not fabricated);
* the referee registration and the figures that the report cites exist.

Skips cleanly when the audit artefacts are absent (plain checkout without the audit
run), exactly like the delivered evidence test.
"""

from __future__ import annotations

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(ROOT, "docs", "audit-m15554")
AUDIT = os.path.join(AUDIT_DIR, "cavity_audit.json")


def _audit() -> dict:
    if not os.path.exists(AUDIT):
        pytest.skip(f"no audit artefact at {AUDIT}")
    with open(AUDIT) as fh:
        return json.load(fh)


def test_all_three_resolutions_are_steady_discrete_solutions():
    """'converged' re-derived: re-stepping the stored field must stay at the gate."""
    audit = _audit()
    assert sorted(int(k) for k in audit["rows"]) == [32, 64, 128]
    for n, row in audit["rows"].items():
        st = row["audit"]["steadiness"]
        assert st["steady"] is True, f"n={n} re-stepped rate {st['max_abs_change_rate']:.2e}"
        assert st["max_abs_change_rate"] <= st["gate"]
        assert row["audit"]["max_divergence"] < 1e-10, f"n={n} divergence"
        assert st["divergence_after_step"] < 1e-10


def test_error_decreases_monotonically_with_refinement():
    audit = _audit()
    ns = [int(k) for k in sorted(audit["rows"], key=lambda k: int(k))]
    u_err = [audit["rows"][str(n)]["audit"]["u_interior_max_abs_error_pct_of_U"] for n in ns]
    v_err = [audit["rows"][str(n)]["audit"]["v_max_rel_err_pct"] for n in ns]
    assert all(b < a for a, b in zip(u_err, u_err[1:])), u_err
    assert all(b < a for a, b in zip(v_err, v_err[1:])), v_err
    # second-order-ish behaviour on the interior metric, from the audit's own numbers
    for order in audit["convergence"]["observed_order_interior"]:
        assert 1.2 < order < 3.0, order


def test_finest_row_satisfies_the_3_percent_claim():
    audit = _audit()
    finest = audit["rows"]["128"]["audit"]
    assert finest["u_min_rel_err_pct"] < 3.0
    assert finest["v_max_rel_err_pct"] < 3.0
    assert finest["u_interior_max_abs_error_pct_of_U"] < 3.0
    dx, dy = finest["vortex_centre"]["abs_delta"]
    assert max(dx, dy) <= 0.01, finest["vortex_centre"]


def test_audit_numbers_match_the_delivered_summary():
    """An independent re-read must land on the delivered numbers, not near them."""
    audit = _audit()
    for n, row in audit["rows"].items():
        shipped = row["shipped"]
        assert abs(row["audit"]["u_min"] - shipped["u_min"]) < 5e-3, n
        assert abs(row["audit"]["v_max"] - shipped["v_max"]) < 5e-3, n
        assert abs(row["audit"]["vortex_centre"]["computed"][0]
                   - shipped["vortex_centre"][0]) < 2.0 / int(n)
        assert shipped["converged"] is True


def test_referee_registration_and_figures_exist():
    sql = os.path.join(AUDIT_DIR, "register_benchmark_case.sql")
    assert os.path.exists(sql)
    assert "CASE-CAVITY-RE1000-FVM" in open(sql, encoding="utf-8").read()
    for name in ("fig_audit_centrelines.png", "fig_audit_fields.png",
                 "fig_audit_convergence.png"):
        path = os.path.join(AUDIT_DIR, name)
        assert os.path.getsize(path) > 10_000, name