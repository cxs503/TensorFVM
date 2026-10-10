"""Protect additional benchmark evidence against missing or falsified raw states."""
import json
import numpy as np
import pytest
import torch
from tensorfvm.verification.core import save_run, write_json, provenance
from tensorfvm.verification.extended_cases import mac_case
from tensorfvm.verification.extended_audit import audit


@pytest.fixture
def evidence(tmp_path):
    torch.set_num_threads(1)
    raw=mac_case('advected-shear',8,end_time=.02)
    raw['directory']='shear';row=save_run(tmp_path/'shear',raw)
    write_json(tmp_path/'summary.json',dict(provenance=provenance(),runs=[row],passed=False,comparison_passed=None,observed_orders={}))
    return tmp_path


def test_missing_refinement_family_cannot_qualify(evidence):
    with pytest.raises(ValueError,match='missing formal refinement family'):audit(evidence)


def test_tampered_accepted_velocity_rejected(evidence):
    path=evidence/'shear/fields.npz'
    with np.load(path) as f:fields={k:f[k] for k in f.files}
    fields['accepted_faces_m_s'][1,0,0,1,2]+=.01
    np.savez_compressed(path,**fields)
    with pytest.raises(ValueError,match='independent reconstruction mismatch'):audit(evidence)


def test_missing_accepted_state_rejected(evidence):
    path=evidence/'shear/fields.npz'
    with np.load(path) as f:fields={k:f[k] for k in f.files}
    fields['accepted_faces_m_s']=fields['accepted_faces_m_s'][:-1]
    np.savez_compressed(path,**fields)
    with pytest.raises(ValueError,match='missing step evidence'):audit(evidence)


def test_loosened_numerical_gate_rejected(evidence):
    p=evidence/'summary.json';s=json.loads(p.read_text())
    next(q for q in s['runs'][0]['metrics'] if q['name']=='energy_defect')['limit']=1.
    write_json(p,s)
    with pytest.raises(ValueError,match='numerical gate altered'):audit(evidence)
