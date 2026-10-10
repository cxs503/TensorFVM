"""Reject corrupt wall-force evidence in the common cylinder adapter."""
import numpy as np
import pytest
import torch
from tensorfvm.verification.core import save_run
from tensorfvm.verification.dfg import run,audit_case


@pytest.fixture
def cylinder(tmp_path):
    torch.set_num_threads(1)
    raw,_=run(16,8,1)
    row=save_run(tmp_path,raw)
    return tmp_path,row


def test_unconverged_cylinder_keeps_failure_badge(cylinder):
    path,row=cylinder
    assert not row['passed']
    assert audit_case(path,row)['physical_passed'] is False
    assert next(q for q in row['metrics'] if q['name']=='convergence_failure')['passed'] is False


def test_tampered_pressure_traction_rejected(cylinder):
    path,row=cylinder
    with np.load(path/'fields.npz') as f:fields={k:f[k] for k in f.files}
    fields['cylinder_pressure_force_n_m'][0,0]+=.01
    np.savez_compressed(path/'fields.npz',**fields)
    with pytest.raises(ValueError,match='traction inconsistency'):audit_case(path,row)


def test_tampered_drag_coefficient_rejected(cylinder):
    path,row=cylinder
    row['computed']['drag']+=1.
    with pytest.raises(ValueError,match='coefficient inconsistency'):audit_case(path,row)
