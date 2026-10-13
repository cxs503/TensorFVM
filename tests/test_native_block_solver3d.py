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
