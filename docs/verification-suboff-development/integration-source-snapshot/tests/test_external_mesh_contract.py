"""Reject Cartesian substitutions and damaged body geometry."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mesh_audit', ROOT/'scripts/audit_external_mesh.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('case', ['cylinder', 'naca'])
def test_saved_mesh_contract_and_rejection(case):
    directory = ROOT/'docs'/f'verification-external-{case}'
    row = json.loads((directory/'summary.json').read_text())['runs'][0]
    with np.load(directory/row['resolution']/'fields.npz') as archive:
        fields = {key: archive[key] for key in archive.files}
    assert module.audit(case, row['config'], fields)['body_fitted']
    with pytest.raises(ValueError, match='require body-fitted'):
        module.audit(case, dict(row['config'], mesh_type='cartesian'), fields)
    fields['vertices_m'] = fields['vertices_m'].copy()
    fields['vertices_m'][0, 2, 0] += .001
    with pytest.raises(ValueError, match='Mesh generation must be repaired'):
        module.audit(case, row['config'], fields)
