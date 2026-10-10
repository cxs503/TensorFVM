"""Meaningful acceptance/reproducibility regressions for the shared workflow."""
import json
import numpy as np
import pytest
from tensorfvm.verification.core import Metric, evaluate, relative_error, save_run
from tensorfvm.verification.cases import channel
from tensorfvm.verification.audit import audit


def test_accuracy_gate_is_strict_and_finite():
    rows,passed=evaluate([Metric("boundary",.03,.03,"exactly three percent"),
                          Metric("nonfinite",float("nan"),.03,"invalid")])
    assert not passed and not any(row["passed"] for row in rows)
    assert evaluate([Metric("below",.029999,.03,"strictly below")])[1]
    with pytest.raises(ValueError):
        relative_error([1,2],[0,0])
    assert relative_error([0.,2.],[0.,1.])==pytest.approx(1.)  # valid zero crossing


def test_unconverged_run_preserves_actual_fields_and_failure(tmp_path):
    row=channel(12,max_iterations=1)
    assert not row["passed"] and not row["converged"]
    assert any(not m["passed"] for m in row["metrics"])
    u=row["fields"]["u_faces_m_s"].copy()
    save_run(tmp_path,row)
    fields=np.load(tmp_path/"fields.npz",allow_pickle=False)
    assert np.array_equal(fields["u_faces_m_s"],u)
    assert not json.loads((tmp_path/"result.json").read_text())["passed"]
    assert len(json.loads((tmp_path/"history.json").read_text()))==1


def test_audit_cannot_qualify_empty_or_missing_studies(tmp_path):
    summary={"error_limit":.03,"runs":[],"provenance":{"source_sha256":{}}}
    (tmp_path/"summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="refinement"):
        audit(tmp_path,source_check=False)


def test_cli_preserves_existing_evidence(tmp_path):
    from tensorfvm.verification.__main__ import main
    sentinel=tmp_path/"existing-failure.json";sentinel.write_text("preserve")
    with pytest.raises(SystemExit) as exc:
        main(["--output",str(tmp_path)])
    assert exc.value.code==2 and sentinel.read_text()=="preserve"
