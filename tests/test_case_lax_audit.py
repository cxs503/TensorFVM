"""Audit gates for CASE-LAX: the reference is exact, and the <2 % gate has teeth.

Companion to ``scripts/audit_case_lax_evidence.py`` / ``docs/audit-CASE-LAX/AUDIT.md``.
``tests/test_compressible_lax_leg.py`` pins the *registration* of the Lax leg
(the leg is dispatched, publishes its verdicts, writes the artifact/figures).
This file pins the two things that registration does **not** prove:

1. the reference state really is the exact solution of the Lax initial-value
   problem -- checked here by a third path that shares no algebraic branch with
   the two implementations in ``lax.py`` (shock-Mach parameterisation, 60-digit
   arithmetic) and by the published 5-digit values of the problem;
2. the acceptance gate actually fails when the scheme is degraded -- a
   first-order run and a run graded against the wrong initial-value problem are
   both rejected.  A <2 % criterion that no degraded case can fail would make
   the CASE-LAX PASS uninformative.

All checks are cheap (<= ~10 s total): one first-order run at nx=200 plus
closed-form arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_case_lax_evidence as audit  # noqa: E402  (script, not a package)

from tensorfvm.compressible import lax as L  # noqa: E402


def test_third_path_confirms_the_reference_state():
    """A third, algebraically independent exact solution matches the artifact ref."""
    check = audit.compare_third_path(L.LaxConfig())
    assert check["ok"] is True, check["rel_diff"]
    assert check["max_rel_diff"] < 1e-12
    # the root really was found in 60-digit arithmetic, on the shock branch
    assert check["residual_at_root"] < 1e-40
    assert 1.9 < check["shock_mach"] < 2.0


def test_reference_matches_the_published_lax_values():
    lit = audit.compare_literature(L.LaxConfig())
    assert lit["ok"] is True, {k: v for k, v in lit["checks"].items() if not v["ok"]}
    # p* > 4 p_R is what distinguishes Lax from Sod (strong right shock)
    assert lit["checks"]["p_star"]["recomputed"] > 4.0 * L.LaxConfig().p_r


def test_wave_position_locator_bias_is_below_the_acceptance_scale():
    """The reported position errors are locator-limited; the bias must stay << 2 %."""
    bias = audit.locator_bias(L.LaxConfig())
    for grid in ("200", "400", "800"):
        for feature, value in bias["graded_grids"][grid].items():
            assert value < 0.01, (grid, feature, value)
    # and it must shrink as the sampling is refined (a locator, not a constant)
    assert bias["graded_grids"]["800"]["x_shock"] < bias["graded_grids"]["200"]["x_shock"]


@pytest.fixture(scope="module")
def first_order_run():
    row = L.run_case(200, order=1)
    return row, L.verdicts([row])


def test_gate_rejects_a_first_order_scheme(first_order_run):
    row, graded = first_order_run
    assert graded["all_ok"] is False, "first-order scheme passed the <2 % gate"
    failed = {k for k, v in graded["checks"].items() if not v["ok"]}
    # the resolution-sensitive criteria must be the ones that fail
    assert "l1[rho]" in failed
    assert row["l1_rel"]["rho"] > L.THRESHOLDS["l1_rel_error"]


def test_gate_rejects_a_wrong_reference():
    """Same Lax field, graded against the exact solution of another IVP (sod)."""
    cfg = L.LaxConfig()
    solver, meta = L.lax(100)
    initial = solver.integrals()
    solver.run(meta["t_end"])
    prim = solver.primitive()
    bad = L.LaxConfig(rho_l=1.0, u_l=0.0, p_l=1.0, rho_r=0.125, u_r=0.0, p_r=0.1)
    meta_bad = dict(meta)
    meta_bad["exact"] = L.exact_riemann(bad)
    meta_bad["config"] = bad
    row = audit._row_from(100, prim, meta_bad, bad, initial, solver)
    graded = L.verdicts([row], cfg=bad)
    failed = {k for k, v in graded["checks"].items() if not v["ok"]}
    assert graded["all_ok"] is False
    assert len(failed) >= 6, sorted(failed)
    for key in ("l1[rho]", "l1[u]", "plateau[p_star]", "position[x_shock]"):
        assert key in failed


def test_committed_audit_artifact_is_consistent_with_the_code():
    """The published AUDIT.md/artifact.json must be reproducible from the code."""
    art_path = ROOT / "docs" / "audit-CASE-LAX" / "artifact.json"
    if not art_path.exists():
        pytest.skip("audit artifact not published in this checkout")
    art = json.loads(art_path.read_text())
    fresh = audit.compare_third_path(L.LaxConfig())
    assert art["third_path"]["max_rel_diff"] == pytest.approx(fresh["max_rel_diff"], rel=1e-9)
    assert art["third_path"]["ok"] is True
    assert art["literature"]["ok"] is True
    # the positive control re-grades the CASE-LAX artifact with the same verdict code
    assert art["positive_control"]["all_ok"] is True
    assert art["positive_control"]["regrade_matches_artifact"] is True
    # and the negative control recorded in the artifact must have been rejected
    assert art["negative_controls"]["first_order"]["400"]["all_ok"] is False
    assert art["negative_controls"]["wrong_reference"]["all_ok"] is False