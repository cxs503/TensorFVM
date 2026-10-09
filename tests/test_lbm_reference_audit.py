"""Saved independent references retain failed convergence and reject changed evidence."""
import json
from pathlib import Path
import sys
import shutil
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from compare_lbm_channel_reference import map_case
from validate_lbm_channel_reference import audit


def test_low_error_does_not_override_nonsteady():
    raw=json.loads((SCRIPTS.parent/'docs/suite-channel/lbm-reference/H48-raw.json').read_text())
    case=map_case(raw)
    assert case['velocity_l2_relative_error'] < .03
    assert case['pressure_gradient_relative_error'] < .03
    assert not case['steady'] and not case['passed']


def test_raw_artifact_edit_rejected(tmp_path):
    source=SCRIPTS.parent/'docs/suite-channel/lbm-extended-reference'
    shutil.copytree(source,tmp_path/'reference')
    path=tmp_path/'reference/H48-raw.json'
    raw=json.loads(path.read_text());raw['steady']=not raw['steady']
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='Raw artifact changed'):
        audit(tmp_path/'reference')


def test_si_summary_edit_rejected(tmp_path):
    source=SCRIPTS.parent/'docs/suite-channel/lbm-extended-reference'
    shutil.copytree(source,tmp_path/'reference')
    path=tmp_path/'reference/comparison.json'
    raw=json.loads(path.read_text());raw['cases'][0]['dt_s'] *= 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='SI reconstruction mismatch'):
        audit(tmp_path/'reference')


def test_float64_actual_monitor_gate_and_fields():
    from validate_lbm_float64_diagnostic import audit as diagnostic_audit
    report=diagnostic_audit(SCRIPTS.parent/'docs/suite-channel/lbm-float64-diagnostic')
    assert report['audit_passed']
    assert report['reference_passed']
    assert all(c['steady'] and c['last_drift']<1e-5 for c in report['cases'])


def test_monitor_history_edit_rejected_even_with_new_hash(tmp_path):
    import hashlib
    from validate_lbm_float64_diagnostic import audit as diagnostic_audit
    source=SCRIPTS.parent/'docs/suite-channel/lbm-float64-diagnostic'
    shutil.copytree(source,tmp_path/'reference')
    path=tmp_path/'reference/H24-raw.json'
    raw=json.loads(path.read_text());raw['umax_history'][-1]*=2
    path.write_text(json.dumps(raw))
    summary_path=tmp_path/'reference/comparison.json'
    summary=json.loads(summary_path.read_text())
    summary['artifact_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Monitor convergence decision mismatch'):
        diagnostic_audit(tmp_path/'reference')
