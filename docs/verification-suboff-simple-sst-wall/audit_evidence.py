"""Replay archived source bindings, native forces, mass balance and comparisons."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root=Path(__file__).resolve().parent
repo=root.parents[1]
sys.path.insert(0,str(repo/'src'))
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_operators3d import NativeOperators3D
from tensorfvm.suboff_validation import cp_comparison,sample_upper_meridian_pressure,drag_comparison

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
count=0;forces={}
for run in sorted(root.glob('*/config.json')):
    folder=run.parent;config=json.loads(run.read_text())
    assert sha(folder/'input.msh')==config['mesh_sha256']
    for relative,expected in config['source_sha256'].items():
        assert sha(folder/'source-snapshot'/relative)==expected,(folder.name,relative)
        count+=1
    result=json.loads((folder/'result.json').read_text())
    fields=np.load(folder/'fields.npz',allow_pickle=False)
    mesh=NativeMesh3D.from_gmsh(folder/'input.msh',triangulate_faces=True)
    wall=mesh.patch_labels=='wall'
    force=(fields['pressure_faces'][wall,None]*mesh.area_vectors[wall]
        -fields['anisotropic_stress_traction'][wall]-fields['isotropic_reynolds_stress_traction'][wall]).sum(0)
    np.testing.assert_allclose(force,result['forces_dimensionless']['total_force'],rtol=1e-10,atol=1e-12)
    divergence=np.max(np.abs(NativeOperators3D(mesh).integrate(fields['flux'])/mesh.cell_volumes))
    np.testing.assert_allclose(divergence,result['momentum_history'][-1]['native_divergence_max'],atol=1e-12)
    assert not result['numerically_converged'] and not result['physical_validation_passed']
    forces[folder.name]=force
for name in ('cp','drag'):
    compared=root/(name+'-comparison');run=root/(name+'-interleaved')
    summary=json.loads((compared/'comparison.json').read_text())
    for path,expected in summary['source_sha256'].items():assert sha(run/Path(path).name)==expected
    spec=json.loads((compared/'reference-specification.json').read_text())
    if name=='cp':
        case=spec['cases'][1];mesh=NativeMesh3D.from_gmsh(run/'input.msh',triangulate_faces=True)
        fields=np.load(run/'fields.npz',allow_pickle=False);points=case['reference']['corrected_candidate_points']
        sample=sample_upper_meridian_pressure(mesh,fields['pressure_faces'],[p['x_L'] for p in points])
        actual=cp_comparison(sample['Cp_calculated'],points)
        key='nominal_relative_L2_error'
    else:
        actual=drag_comparison(forces['drag-interleaved'],spec['cases'][0]);key='relative_error'
    np.testing.assert_allclose(actual[key],summary['metrics'][key],rtol=1e-13)
for path,expected in json.loads((root/'implementation-sha256.json').read_text()).items():assert sha(repo/path)==expected,path
print(json.dumps(dict(producer_source_hashes_checked=count,raw_force_and_continuity_replays=len(forces),experimental_metric_replays=2,evidence_checks_passed=True,physical_validation_passed=False),indent=2))
