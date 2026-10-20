"""Auditor rejects bad provenance and false convergence on a tiny real mesh."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from dataclasses import asdict
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_manufactured3d import abc_fields
from tensorfvm.native_face_geometry3d import merge_coplanar_faces

SCRIPT=Path(__file__).parents[1]/'scripts/audit_native_suboff_run.py'
spec=importlib.util.spec_from_file_location('audit_native_suboff_test',SCRIPT)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def require_gmsh():
    """Skip the test when the gmsh runtime cannot be loaded.

    ``pytest.importorskip`` only catches ImportError, but a pip-installed
    gmsh that lacks libGLU raises OSError while loading its shared object.
    Treat both as "the optional mesh runtime is unavailable".
    """
    try:
        import gmsh  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f'gmsh runtime unavailable: {exc}')


def fixture_run(tmp_path,quadratic=False,merged=False):
    # The fixture builds its mesh through NativeMesh3D.from_gmsh, which needs
    # the gmsh runtime.  Skip rather than fail when it is absent, so the suite
    # stays runnable without the optional 'mesh' extra.
    require_gmsh()
    run=tmp_path/'run';run.mkdir();n=3
    points=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])
    ix=lambda x,y,z:z*(n+1)**2+y*(n+1)+x
    cells=[]
    for z in range(n):
        for y in range(n):
            for x in range(n):cells.append([ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)])
    native=NativeMesh3D.from_cells(points,cells,['hexahedron']*len(cells))
    walls=[native.face_vertices[i] for i in np.flatnonzero(native.neighbor<0)]
    msh='$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n1\n2 1 "wall"\n$EndPhysicalNames\n$Nodes\n'+str(len(points))+'\n'
    msh+=''.join(f'{i+1} {p[0]} {p[1]} {p[2]}\n' for i,p in enumerate(points))+'$EndNodes\n$Elements\n'+str(len(cells)+len(walls))+'\n'
    msh+=''.join(f'{i+1} 5 0 '+ ' '.join(str(j+1) for j in c)+'\n' for i,c in enumerate(cells))
    msh+=''.join(f'{len(cells)+i+1} 3 2 1 1 '+ ' '.join(str(j+1) for j in c)+'\n' for i,c in enumerate(walls))+'$EndElements\n';(run/'input.msh').write_text(msh)
    mesh=NativeMesh3D.from_gmsh(run/'input.msh',triangulate_faces=True)
    if merged:mesh,_=merge_coplanar_faces(mesh)
    package=SCRIPT.parents[1]/'src/tensorfvm';snapshot=run/'source-snapshot/src/tensorfvm';shutil.copytree(package,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    sources={str(p.relative_to(run/'source-snapshot')):audit.digest(p) for p in snapshot.rglob('*.py')}
    config=dict(source_sha256=sources,mesh_sha256=audit.digest(run/'input.msh'),settings=asdict(NativeSteadySettings()),mode='manufactured',reconstruction='quadratic' if quadratic else 'linear',merge_coplanar_faces=merged,quadratic_linear_solver='assembled' if quadratic else 'matrix-free',initial_state=None)
    (run/'config.json').write_text(json.dumps(config))
    exact=abc_fields(mesh.cell_centers)
    np.savez(run/'fields.npz',velocity=np.zeros((len(mesh.cells),3)),pressure=np.zeros(len(mesh.cells)),flux=np.zeros(len(mesh.owner)),reference_velocity=exact['velocity'],reference_pressure=exact['pressure'])
    (run/'result.json').write_text(json.dumps(dict(converged=True,history=[dict(velocity_change=0.,pressure_change=0.,flux_change=0.)])))
    return run


def test_relative_norm_removes_only_constant_pressure_gauge():
    reference=np.array([1.,2.,4.]);weights=np.array([1.,2.,1.])
    assert audit.relative_l2(reference+10,reference,weights,True)==pytest.approx(0)
    assert audit.relative_l2(reference*2,reference,weights,True)==pytest.approx(1)


@pytest.mark.parametrize('target',['input.msh','source-snapshot/src/tensorfvm/native_collocated3d.py'])
def test_mutated_mesh_or_frozen_operator_rejected_before_replay(tmp_path,target):
    run=fixture_run(tmp_path);(run/target).write_text((run/target).read_text()+'\n# mutation\n')
    with pytest.raises(ValueError,match='hash mismatch'):audit.verify_bindings(run)


def test_initial_numerical_state_hash_is_required(tmp_path):
    run=fixture_run(tmp_path);config=json.loads((run/'config.json').read_text());np.savez(run/'initial-numerical-state.npz',velocity=np.zeros((27,3)),pressure=np.zeros(27));config['initial_state']=dict(sha256='bad');(run/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='initial numerical state hash'):audit.verify_bindings(run)


def test_initial_producer_failure_is_preserved_and_bound(tmp_path):
    run=fixture_run(tmp_path);config=json.loads((run/'config.json').read_text())
    seed=run/'initial-numerical-state.npz';np.savez(seed,velocity=np.zeros((27,3)),pressure=np.zeros(27))
    ledger=run/'initial-producer-result.json';ledger.write_text(json.dumps({'converged':False}))
    config['initial_state']=dict(sha256=audit.digest(seed),producer_converged=False,producer_result_sha256=audit.digest(ledger))
    (run/'config.json').write_text(json.dumps(config))
    assert not audit.verify_bindings(run)['initial_state']['producer_converged']
    ledger.write_text(json.dumps({'converged':True}))
    with pytest.raises(ValueError,match='initial producer result hash'):audit.verify_bindings(run)


@pytest.mark.parametrize('quadratic,merged',[(False,False),(True,True)])
def test_frozen_subprocess_replays_real_equations_and_rejects_false_convergence(tmp_path,quadratic,merged):
    run=fixture_run(tmp_path,quadratic,merged);out=tmp_path/'audit'
    completed=subprocess.run([sys.executable,str(SCRIPT),'--run',str(run),'--output',str(out)],capture_output=True,text=True)
    assert completed.returncode==2,completed.stdout+completed.stderr
    result=json.loads((out/'audit.json').read_text())
    assert not result.get('replay_failed',False),(out/'replay.log').read_text()
    assert not result['passed']
    assert result['current_equation_residuals']['momentum_residual']>1e-7
    assert result['relative_L2_errors']['velocity']==pytest.approx(1.)
    assert result['producer_inputs_unchanged_during_replay']
    assert result['face_policy']==('merged_coplanar_polygon_faces' if merged else 'triangulated_native_faces')
    assert result['frozen_imports']
    assert all('/source-snapshot/src/' in r['path'] and r['bound_by_producer'] for r in result['frozen_imports'].values())
    assert not result['experimental_accuracy_qualified']


def test_provenance_failure_is_archived_without_solver_replay(tmp_path):
    run=fixture_run(tmp_path);(run/'input.msh').write_text('tampered')
    out=tmp_path/'audit'
    completed=subprocess.run([sys.executable,str(SCRIPT),'--run',str(run),'--output',str(out)],capture_output=True,text=True)
    assert completed.returncode==2
    result=json.loads((out/'audit.json').read_text())
    assert not result['passed'] and not result['provenance_verified']
    assert 'mesh hash mismatch' in result['error']
    assert not (out/'worker-result.json').exists()
