import numpy as np
import pytest
pytest.importorskip('pyamg')
from tensorfvm.verification.native_hybrid_ns import potential_solver
from tensorfvm.verification.native_hybrid_flow import channel_solver
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh


@pytest.mark.parametrize('open_boundary',[False,True])
def test_block_solver_matches_direct_with_nonzero_gauge_or_traction(open_boundary):
    m=pressure_mesh(2);b=m.neighbor<0;fixed=b.copy()
    if open_boundary:fixed[np.flatnonzero(b)[::3]]=False
    g=np.array([.7,-.3,.2]);p=.4+m.cell_centers@g
    traction=-(.4+m.face_centers@g)[:,None]*m.area_vectors/np.linalg.norm(m.area_vectors,axis=1)[:,None]
    answers=[]
    for method in ('direct','block_amg'):
        s=NativeHybridStokes3D(m,fixed,body_force=g,boundary_traction=traction,gauge_value=p[0],coupled_linear_method=method)
        r=s.solve(algorithm='coupled',tolerance=1e-9)
        assert r['numerically_converged']
        np.testing.assert_allclose(r['pressure'],p,atol=1e-9)
        np.testing.assert_allclose(r['velocity'],0,atol=1e-9)
        answers.append(r)
    np.testing.assert_allclose(answers[0]['face_velocity'],answers[1]['face_velocity'],atol=1e-9)


def test_block_newton_matches_direct_without_changing_physics():
    answers=[]
    for method in ('direct','block_amg'):
        _,s,_=potential_solver(2,.1);s.coupled_linear_method=method
        s.initialize_stokes();r=s.solve_newton(tolerance=1e-9)
        assert r['numerically_converged']
        assert r['energy_balance_error']<1e-8
        answers.append(r)
    for key in ('velocity','pressure','face_velocity','flux'):
        np.testing.assert_allclose(answers[0][key],answers[1][key],atol=1e-8)


def test_bad_iterative_candidate_rejected_before_accepted_state_changes(monkeypatch):
    import tensorfvm.native_block_solver3d as module
    _,s,_=potential_solver(2,.1);s.coupled_linear_method='block_amg'
    old=s.state.copy();p=s.p.copy()
    monkeypatch.setattr(module,'gmres',lambda a,b,**kw:(np.zeros_like(b),0))
    with pytest.raises(RuntimeError,match='CURRENT true residual'):s.initialize_stokes()
    np.testing.assert_array_equal(old,s.state);np.testing.assert_array_equal(p,s.p)


@pytest.mark.parametrize('kind',['smooth','jump'])
def test_variable_viscosity_and_component_symmetry_match_direct(kind):
    from tensorfvm.verification.native_hybrid_shear import shear_solver
    answers=[]
    for method in ('direct','block_amg'):
        m,s,e=shear_solver(8,kind);s.coupled_linear_method=method
        r=s.solve(algorithm='coupled',tolerance=1e-9)
        assert r['numerically_converged']
        assert r['energy_balance_error']<1e-8
        answers.append(r)
    for key in ('velocity','pressure','total_traction_integrated'):
        np.testing.assert_allclose(answers[0][key],answers[1][key],atol=1e-8)


def test_failed_gmres_retains_candidate_and_keeps_accepted_state(monkeypatch):
    import tensorfvm.native_block_solver3d as module
    _,s,_=potential_solver(2,.1);s.coupled_linear_method='block_amg'
    old=s.state.copy();old_p=s.p.copy()
    monkeypatch.setattr(module,'gmres',lambda a,b,**kw:(np.zeros_like(b),30))
    with pytest.raises(module.NativeBlockSolveError) as error:s.initialize_stokes()
    failure=error.value
    assert failure.record['gmres_info']==30
    assert not failure.record['accepted']
    assert failure.candidate.shape==failure.defect.shape==(len(s.viscous.free)+s.N,)
    assert s.coupled_linear_history[-1]==failure.record
    assert np.linalg.norm(failure.defect)>0
    np.testing.assert_array_equal(s.state,old);np.testing.assert_array_equal(s.p,old_p)


def test_small_global_residual_cannot_hide_local_mass_defect(monkeypatch):
    import tensorfvm.native_block_solver3d as module
    from tensorfvm.native_mesh3d import NativeMesh3D
    from tensorfvm.verification.native_hybrid_flow import channel_mesh
    from scipy.sparse.linalg import spsolve
    base=channel_mesh(2,triangulate_faces=False)
    m=NativeMesh3D.from_cells(base.vertices*.01,base.cells,base.cell_types,triangulate_faces=False)
    s=NativeHybridStokes3D(m,m.neighbor<0,viscosity=1e-4,
        body_force=[.7,-.3,.2],gauge_value=.4,coupled_linear_method='block_amg')
    face=np.flatnonzero(m.neighbor>=0)[0];axis=np.argmax(abs(m.area_vectors[face]))
    position=np.flatnonzero(s.viscous.free==3*s.N+3*face+axis)[0]
    def inaccurate(matrix,rhs,**kwargs):
        candidate=spsolve(matrix,rhs);candidate[position]+=1e-9
        return candidate,0
    monkeypatch.setattr(module,'gmres',inaccurate)
    old=s.state.copy();old_p=s.p.copy()
    with pytest.raises(module.NativeBlockSolveError) as error:s.solve(algorithm='coupled')
    record=error.value.record
    assert record['gmres_info']==0 and record['true_relative_residual']<1e-9
    assert record['algebraic_continuity_defect_per_volume_max']>1e-8
    np.testing.assert_array_equal(s.state,old);np.testing.assert_array_equal(s.p,old_p)


def test_nonfinite_rejected_candidate_record_is_json_serializable(monkeypatch):
    import json
    import tensorfvm.native_block_solver3d as module
    _,s,_=potential_solver(2,.1);s.coupled_linear_method='block_amg'
    monkeypatch.setattr(module,'gmres',lambda a,b,**kw:(np.full_like(b,np.nan),0))
    with pytest.raises(module.NativeBlockSolveError) as error:s.initialize_stokes()
    assert not error.value.record['candidate_finite']
    assert error.value.record['true_relative_residual'] is None
    json.dumps(error.value.record,allow_nan=False)
