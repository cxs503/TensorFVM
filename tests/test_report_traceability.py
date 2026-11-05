"""Gate on the provenance of numbers quoted by the verification reports.

Two failure modes are made loud here:

* a report text or an archived artifact drifts away from the value they shared
  (``doc_claims.verify_registry``), and
* a report quotes a ``scripts/*.py`` reproduce pointer that no longer exists.

The registry is a ratchet: ``claims.json`` may grow, never shrink, so deleting a
report's evidence cannot silently retire the gate.  Negative controls at the
bottom prove the checker actually fails on drift instead of passing vacuously.
"""

from __future__ import annotations

import json
import pathlib

from tensorfvm.verification.doc_claims import (
    audit_traceability, build_registry, scan_report, verify_registry,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE = ROOT / "docs" / "verification-report-traceability"
REGISTRY = json.loads((CASE / "claims.json").read_text())
SUMMARY = json.loads((CASE / "traceability.json").read_text())["summary"]


def test_every_bound_claim_still_holds():
    checked, violations = verify_registry(REGISTRY)
    assert not violations, violations[:3]
    assert checked == REGISTRY["n_claims"]


def test_registry_is_a_ratchet():
    assert REGISTRY["n_claims"] >= 15, REGISTRY["n_claims"]
    assert REGISTRY["reports_hooked"] >= 2, REGISTRY["reports_hooked"]
    fresh = build_registry(str(ROOT / "docs"))
    assert fresh["n_claims"] >= REGISTRY["n_claims"], (
        "claims disappeared: %d -> %d" % (REGISTRY["n_claims"], fresh["n_claims"]))


def test_no_report_quotes_a_missing_reproduce_script():
    rows = audit_traceability(str(ROOT / "docs"))
    dead = [(r["doc"], r["dead_scripts"]) for r in rows if r["dead_scripts"]]
    assert not dead, dead


def test_traceability_baseline_is_recorded_and_does_not_regress():
    rows = audit_traceability(str(ROOT / "docs"))
    from tensorfvm.verification.doc_claims import summarize
    live = summarize(rows)
    assert live["claimable"] == SUMMARY["claimable"], (live["claimable"], SUMMARY["claimable"])
    assert live["traced"] >= SUMMARY["traced"], (live["traced"], SUMMARY["traced"])
    assert live["dead_scripts"] == 0


def test_negative_control_report_edit_is_detected():
    """A one-digit retype in the report must break the printed precision band."""
    from tensorfvm.verification.doc_claims import TOKEN
    claim = REGISTRY["claims"][0]
    token = claim["token"]
    digit = "8" if token[-1] != "8" else "7"
    edited = token[:-1] + digit
    band, printed = claim["band"], float(token)
    assert abs(float(edited) - printed) > band, (token, edited, band)
    tokens = [float(t) for t in TOKEN.findall("value " + edited)]
    assert not any(abs(t - printed) <= band for t in tokens)


def test_negative_control_artifact_edit_is_detected(tmp_path):
    """A regenerated artifact whose leaf moved must fail the gate, not pass silently."""
    from tensorfvm.verification.doc_claims import artifact_numbers, verify_registry
    claim = REGISTRY["claims"][0]
    item = claim["pointers"][0]
    source = pathlib.Path(item["artifact"])
    assert source.suffix == ".json"
    moved = item["value"] * 1.05
    payload = json.loads(source.read_text())
    node, keys = payload, item["pointer"].strip("/").split("/")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = moved
    copy = tmp_path / "regenerated.json"
    copy.write_text(json.dumps(payload))
    assert abs(artifact_numbers(copy)[item["pointer"]] - float(claim["token"])) > claim["band"]
    registry = dict(schema=1, reports_scanned=1, reports_hooked=1, n_claims=1,
                    claims=[dict(claim, pointers=[dict(item, artifact=str(copy))])])
    _, violations = verify_registry(registry)
    assert [v["kind"] for v in violations] == ["artifact_drifted"], violations


def test_negative_control_missing_reproduce_script_is_detected(tmp_path):
    """A report quoting a script that does not exist must be flagged, not skipped."""
    case = tmp_path / "docs" / "case"
    case.mkdir(parents=True)
    (case / "evidence.json").write_text(json.dumps({"metric": 0.123456}))
    report = case / "report.md"
    report.write_text("metric 0.1235 matches; reproduce: `scripts/no_such_script.py`\n")
    rows = audit_traceability(str(tmp_path / "docs"))
    assert [r["dead_scripts"] for r in rows] == [["scripts/no_such_script.py"]], rows
    report.write_text("metric 0.1235 matches; reproduce: `scripts/audit_report_traceability.py`\n")
    assert all(not r["dead_scripts"] for r in audit_traceability(str(tmp_path / "docs")))


def test_scan_report_separates_traced_from_untraced(tmp_path):
    (tmp_path / "evidence.json").write_text(json.dumps({"metric": 0.123456}))
    report = tmp_path / "report.md"
    report.write_text("metric 0.1235 matches, guess 0.98765 does not; gate 1e-10 ignored\n")
    scan = scan_report(report)
    assert [c["token"] for c in scan["claims"]] == ["0.1235"]
    assert scan["untraced"] == ["0.98765"]
    assert scan["claimable"] == 2
