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


def saved_evidence(case):
    """Return the published external-flow summary for ``case``.

    ``docs/verification-external-<case>/summary.json`` and the ``fields.npz``
    it indexes are produced by the external-flow benchmark workflow; only the
    manifest and report of those directories are tracked, so the evidence is
    absent from a fresh checkout.  Skip instead of failing: the audit that
    follows is still exercised whenever the artifacts are published, and a
    missing artifact must not be reported as a mesh-contract violation.
    """
    directory = ROOT/'docs'/f'verification-external-{case}'
    summary = directory/'summary.json'
    if not summary.exists():
        pytest.skip(f'published external mesh evidence not shipped: {summary}')
    return directory, json.loads(summary.read_text())


@pytest.mark.parametrize('case', ['cylinder', 'naca'])
def test_saved_mesh_contract_and_rejection(case):
    directory, summary = saved_evidence(case)
    row = summary['runs'][0]
    with np.load(directory/row['resolution']/'fields.npz') as archive:
        fields = {key: archive[key] for key in archive.files}
    assert module.audit(case, row['config'], fields)['body_fitted']
    with pytest.raises(ValueError, match='require body-fitted'):
        module.audit(case, dict(row['config'], mesh_type='cartesian'), fields)
    fields['vertices_m'] = fields['vertices_m'].copy()
    fields['vertices_m'][0, 2, 0] += .001
    with pytest.raises(ValueError, match='Mesh generation must be repaired'):
        module.audit(case, row['config'], fields)