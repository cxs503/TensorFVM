"""Replay frozen input/source bindings and total native wall traction."""
from pathlib import Path
import hashlib
import json
import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D

root=Path(__file__).resolve().parent
repo=root.parents[1]
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
records=[];source_files=0;comparison_files=0
for directory in sorted(root.iterdir()):
    config_path=directory/'config.json'
    if not config_path.is_file():continue
    config=json.loads(config_path.read_text())
    if config.get('kind')!='native SST SUBOFF diagnostic':continue
    assert digest(directory/'input.msh')==config['mesh_sha256']
    for name,sha in config['source_sha256'].items():
        assert digest(directory/'source-snapshot'/name)==sha,(directory,name)
        source_files+=1
    result_path=directory/'result.json'
    if not result_path.exists():
        assert (directory/'termination.json').is_file()
        records.append(dict(run=directory.name,status='externally_terminated_without_final_fields'))
        continue
    result=json.loads(result_path.read_text())
    mesh=NativeMesh3D.from_gmsh(directory/'input.msh',triangulate_faces=True)
    fields=np.load(directory/'fields.npz',allow_pickle=False);wall=mesh.patch_labels=='wall'
    assert all(np.isfinite(fields[key]).all() for key in fields.files)
    force=(fields['pressure_faces'][wall,None]*mesh.area_vectors[wall]
        -fields['anisotropic_stress_traction'][wall]-fields['isotropic_reynolds_stress_traction'][wall]).sum(0)
    expected=np.asarray(result['forces_dimensionless']['total_force'])
    np.testing.assert_allclose(force,expected,rtol=1e-10,atol=1e-12)
    assert result['physical_validation_passed'] is False
    records.append(dict(run=directory.name,status=result.get('solution_state','original_initialization_failure'),
        source_files=len(config['source_sha256']),force_replay_max_error=float(abs(force-expected).max()),
        completed_outer_iterations=len(result['history']),numerically_converged=result['numerically_converged']))
for directory in sorted(root.iterdir()):
    path=directory/'comparison.json'
    if not path.is_file():continue
    compared=json.loads(path.read_text())
    if 'source_sha256' not in compared:continue
    for name,sha in compared['source_sha256'].items():
        assert digest(repo/name)==sha,name
        comparison_files+=1
    assert compared['physical_validation_passed'] is False
result=dict(evidence_checks_passed=True,physical_validation_passed=False,
    checked_producer_source_files=source_files,checked_comparison_input_bindings=comparison_files,cases=records)
(root/'evidence-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
