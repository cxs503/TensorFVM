"""D3 reconciliation invariants (m16314).

The D3 ledger item asked for a status re-check of ``GAP-FVM-TWOPHASE``.  The
verdict was *not* "promote to done": the Hysing leg of the acceptance face
fails its terminal-velocity gate.  These tests pin that verdict to the raw
artifacts so a future edit cannot silently flip the status without new
evidence:

* the recorded Hysing run must still imply a FAIL of the B1 gate
  (|U_T - ref|/ref > 10 %);
* the recorded Zalesak run must still show all five Z gates passing;
* the debt ledger must carry the D3 closure and the open D9 gap.
"""

import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "docs", "GAP-FVM-TWOPHASE", "artifacts")
LEDGER = os.path.join(REPO, "docs", "VERIFICATION_DEBT_LEDGER.md")
RECON = os.path.join(REPO, "docs", "GAP-FVM-TWOPHASE", "D3-LEDGER-RECONCILIATION.md")


def _load(name):
    with open(os.path.join(ART, name)) as fh:
        return json.load(fh)


def test_hysing_artifact_exists():
    assert os.path.exists(os.path.join(ART, "leg_hysing_cpd16.json"))


def test_hysing_evidence_implies_b1_fail():
    """The recorded run must still fail the 10 % terminal-velocity gate."""
    rec = _load("leg_hysing_cpd16.json")
    ref = rec["reference"]["terminal_velocity"]
    assert ref == pytest.approx(0.2342)
    u_t = rec["terminal_velocity"]
    # Bubble sinks: the sign is already wrong, independent of magnitude.
    assert u_t < 0.0
    rel_err = abs(u_t - ref) / ref
    assert rel_err > 0.10
    assert rec["terminal_velocity_rel_error"] == pytest.approx(rel_err, rel=1e-6)
    # Mass conservation of the failing run was fine: the defect is dynamics,
    # not transport.
    assert rec["worst_step_drift"] <= 1e-6
    assert rec["diverged"] is False


def test_zalesak_evidence_implies_pass():
    """The Zalesak leg evidence must still show all five gates passing."""
    data = _load("gap_twophase.json")
    z = {g["id"]: g for g in data["gates"] if g["case"] == "zalesak"}
    assert set(z) == {"Z1", "Z2", "Z3", "Z4", "Z5"}
    for gate in z.values():
        assert gate["passed"] is True, gate["id"]
    # Spot-check the two headline numbers against the gate limits.
    assert z["Z1"]["got"] <= z["Z1"]["limit"]
    assert z["Z5"]["got"] >= z["Z5"]["limit"]


def test_ledger_carries_d3_closure_and_d9():
    text = open(LEDGER, encoding="utf-8").read()
    # D3 closure note present in the open-items section...
    assert "D3 已结（m16314" in text
    # ...and the new open gap D9 with its quantified acceptance line.
    assert "### D9 Hysing 上升气泡腿修复" in text
    assert "U_T vs 0.2342 <10%" in text
    # D3 must also be listed as delivered in the done section.
    assert "D3 总账 GAP-FVM-TWOPHASE 状态复核 —— 已交付（m16314" in text


def test_reconciliation_doc_points_at_evidence():
    text = open(RECON, encoding="utf-8").read()
    for pointer in (
        "leg_hysing_cpd16.json",
        "gap_twophase.json",
        "GAP-FVM-TWOPHASE",
        "109.5%",
    ):
        assert pointer in text, pointer
