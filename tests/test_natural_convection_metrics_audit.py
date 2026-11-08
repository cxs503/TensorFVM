"""Gates on the natural-convection metric audit.

The archived ladder (``docs/incompressible-baselines/cavity_n0*.json``) is the
evidence behind the "Nu within 2 % of de Vahl Davis" claim.  Two failure modes
are pinned here:

* a *derived* metric field that cannot be recomputed from the primitives stored
  in the same record (silently quoting code that no longer exists).  All eight
  archived records carry such a field -- ``psi_error_rel`` computed against the
  raw, dimensioned ``|psi|`` -- and the audit must identify it by arithmetic
  (``|stored - legacy_formula| == 0``), not by a comment;
* a *reproduction* that quietly uses a different iteration recipe.  Re-running
  ``n = 32, Ra = 1e5`` with the frozen deferred correction (``dc_relax = 1``)
  reproduces the archive to machine precision; the damped path does not converge
  at all, so the recipe is part of the evidence.

The ``doc`` side of the ledger (``audit.json``) is regenerated from the tree, so
the tests also fail if the committed ledger drifts away from the records.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_natural_convection_metrics.py"
LEDGER = ROOT / "docs" / "verification-natural-convection-metrics" / "audit.json"
ARCHIVE = ROOT / "docs" / "incompressible-baselines"


def _audit():
    spec = importlib.util.spec_from_file_location("audit_nc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _audit()


def test_stale_psi_error_is_identified_by_arithmetic():
    """The legacy field must equal the *raw-psi* formula for all eight records."""
    records = [audit.audit_record(str(p)) for p in sorted(ARCHIVE.glob("cavity_n0*.json"))]
    assert len(records) == 8
    for rec in records:
        assert rec["psi_error_rel_is_legacy_raw_psi"], rec["record"]
        # the true (|psi|/alpha) error is 4-53x tighter than the stored one
        assert rec["psi_error_rel_true"] < rec["psi_error_rel_stored"] / 2.0, rec["record"]


def test_other_derived_fields_reconcile_exactly():
    for path in sorted(ARCHIVE.glob("cavity_n0*.json")):
        rec = audit.audit_record(str(path))
        for key, mismatch in rec["max_rel_mismatch"].items():
            assert mismatch < 1e-9, (rec["record"], key, mismatch)


def test_reproduction_recipe_reproduces_the_archive():
    """``n = 32, Ra = 1e5`` on the frozen path == archived record, bit for bit."""
    with open(ROOT / "docs" / "verification-natural-convection-metrics"
              / "reproduced_cavity_n032_ra1e05.json") as fh:
        repro = json.load(fh)
    archived = audit.audit_record(str(ARCHIVE / "cavity_n032_ra1e05.json"))
    assert abs(repro["metrics"]["nusselt"] - archived["nusselt"]) < 1e-12
    assert abs(repro["metrics"]["psi_over_alpha"] - archived["psi_over_alpha"]) < 1e-12
    assert repro["metrics"]["iterations"] == 1630
    assert repro["settings"]["dc_relax"] == 1.0
    assert [h["stage"] for h in repro["convergence"]["history"]] == ["first_order", "deferred_1"]


def test_observed_order_needs_a_monotone_ladder():
    """Non-monotone data must yield ``None``, never a fitted number."""
    assert audit.observed_order([1, 0.5, 1 / 3], [4.590, 4.436, 4.442]) is None
    # synthetic second-order data on the archive's unequal grid spacing
    h = [1.0 / 32, 1.0 / 64, 1.0 / 96]
    f = [1.0 + 0.7 * x ** 2 for x in h]
    assert audit.observed_order(h, f) == pytest.approx(2.0, abs=1e-6)
    # first order, shrinking sequence -- the psi ladder's shape
    f1 = [9.0 + 0.4 * x for x in h]
    assert audit.observed_order(h, f1) == pytest.approx(1.0, abs=1e-6)


def test_richardson_extrapolation_closes_the_psi_anchor_at_ra_1e6():
    """At Ra = 1e6 the first-order psi ladder extrapolates inside 3 % of DVD."""
    ledger = audit.build_ledger(str(ROOT / "docs"))
    ra6 = ledger["ladders"]["Ra=1e+06"]
    assert 0.8 < ra6["observed_order_psi"] < 1.1
    assert ra6["psi_error_rel_finest"] > 0.03           # raw finite-grid error: still open
    assert ra6["psi_richardson_error_rel"] < 0.03       # but the ladder says discretisation


def test_nusselt_order_is_withheld_where_the_ladder_oscillates():
    ledger = audit.build_ledger(str(ROOT / "docs"))
    assert ledger["ladders"]["Ra=1e+05"]["observed_order_nusselt"] is None
    assert ledger["ladders"]["Ra=1e+05"]["nusselt_monotone"] is False
    assert ledger["ladders"]["Ra=1e+05"]["nusselt_verdict"] == "PASS"   # <3 % at n = 96


def test_committed_ledger_matches_the_tree():
    committed = json.loads(LEDGER.read_text())
    fresh = audit.build_ledger(str(ROOT / "docs"))
    assert fresh["legacy_psi_error_records"] == committed["legacy_psi_error_records"]
    assert fresh["summary"] == committed["summary"]
    for key in fresh["ladders"]:
        assert fresh["ladders"][key]["levels"] == committed["ladders"][key]["levels"]
        assert fresh["ladders"][key]["nusselt"] == pytest.approx(
            committed["ladders"][key]["nusselt"], rel=1e-12)


def test_negative_control_stale_ledger_is_detected():
    """A ledger whose legacy list is edited must fail the ratchet."""
    ledger = audit.build_ledger(str(ROOT / "docs"))
    assert ledger["legacy_psi_error_matches_pin"] is True
    tampered = dict(ledger, legacy_psi_error_records=ledger["legacy_psi_error_records"][:-1])
    assert (sorted(tampered["legacy_psi_error_records"]) == sorted(audit.LEGACY_PSI_ERROR)) is False