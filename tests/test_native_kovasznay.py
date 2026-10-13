"""Unforced analytic checks and checks on nonlinear/linear solver safeguards."""
import numpy as np
import pytest
from tensorfvm.verification.native_hybrid_ns import kovasznay_fields, kovasznay_solver
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh


def test_kovasznay_satisfies_unforced_continuous_equations():
    points=np.random.default_rng(13).uniform(.1,.9,(20,3));h=1e-4
    for re in (10.,40.,100.):
        f=kovasznay_fields(points,re)
        plus=[kovasznay_fields(points+np.eye(3)[j]*h,re) for j in range(3)]
        minus=[kovasznay_fields(points-np.eye(3)[j]*h,re) for j in range(3)]
        grad=np.stack([(p['velocity']-m['velocity'])/(2*h) for p,m in zip(plus,minus)],axis=2)
        lap=sum((p['velocity']-2*f['velocity']+m['velocity'])/h**2 for p,m in zip(plus,minus))
        gp=np.column_stack([(p['pressure']-m['pressure'])/(2*h) for p,m in zip(plus,minus)])
        np.testing.assert_allclose(grad,f['gradient'],atol=5e-7)
        np.testing.assert_allclose(np.trace(grad,axis1=1,axis2=2),0,atol=3e-7)
        conv=np.einsum('cij,cj->ci',grad,f['velocity'])
        assert np.linalg.norm(conv)>1.
        np.testing.assert_allclose(conv+gp-lap/re,0,atol=5e-7)
        assert not f['force'].any()


@pytest.mark.parametrize('triangulate',[True,False])
def test_unforced_solution_reused_lu_matches_fresh_factorizations(triangulate):
    answers=[]
    for method in ('direct','reused_lu'):
        mesh,s,exact=kovasznay_solver(3,triangulate_faces=triangulate,coupled_linear_method=method)
        assert not s.body_force.any()
        assert not s.load.any()
        assert abs(s.mass_rhs.sum())<1e-12
        r=s.solve(algorithm='coupled',max_outer_iterations=100,tolerance=1e-8,outer_relaxation=.5)
        assert r['numerically_converged'],r['outer_history'][-1]
        assert r['energy_balance_error']<1e-7
        np.testing.assert_allclose(r['momentum_flux_integrated'][mesh.neighbor<0].sum(0),0,atol=1e-7)
        assert max(x['true_relative_residual'] for x in s.coupled_linear_history)<1e-9
        if method=='reused_lu':assert any(x['gmres_iterations']>0 for x in s.coupled_linear_history)
        answers.append(r)
    for key in ('velocity','face_velocity','pressure','flux'):
        np.testing.assert_allclose(answers[0][key],answers[1][key],atol=1e-7)


def test_reused_lu_failed_gmres_refreshes_against_current_matrix(monkeypatch):
    from scipy import sparse
    import tensorfvm.native_hybrid_stokes3d as module
    m=channel_mesh(1);s=NativeHybridStokes3D(m,m.neighbor<0,coupled_linear_method='reused_lu')
    a=sparse.diags([2.,3.,4.]).tocsc();rhs=np.array([1.,2.,3.])
    s._coupled_linear(a,rhs)
    # Simulate false success with a bad candidate: true residual must reject it.
    monkeypatch.setattr(module,'gmres',lambda *args,**kwargs:(np.zeros(3),0))
    b=sparse.diags([5.,7.,9.]).tocsc()
    x,residual=s._coupled_linear(b,rhs)
    np.testing.assert_allclose(b@x,rhs,atol=1e-14)
    assert s.coupled_linear_history[-1]['fallback_factorization']
    assert residual<1e-12


def test_outer_relaxation_validated_and_current_divergence_not_hidden():
    _,s,_=kovasznay_solver(2)
    for value in (0.,-1.,1.1,float('nan')):
        with pytest.raises(ValueError,match='outer_relaxation'):
            s.solve(outer_relaxation=value)
    r=s.solve(algorithm='coupled',outer_relaxation=.5,max_outer_iterations=1)
    assert r['outer_history'][-1]['frozen_inner_converged']
    assert not r['numerically_converged']
    assert r['outer_history'][-1]['divergence_max']>1e-3


@pytest.mark.parametrize('algorithm',['coupled','simple','simplec'])
def test_component_symmetry_preserves_pressure_gauge_and_affine_flow(algorithm):
    m=channel_mesh((2,3,1));b=m.neighbor<0;z=m.face_centers[:,2]
    planes=b & ((z<1e-10)|(z>1-1e-10))
    fixed=np.repeat(b[:,None],3,axis=1);fixed[planes,:2]=False
    gradient=np.diag([.2,-.2,0.])
    boundary=np.array([1.,0.,0.])+m.face_centers@gradient.T
    s=NativeHybridStokes3D(m,fixed,boundary,gauge_value=.7)
    assert s.gauge and not s.open_normal_boundary.any()
    r=s.solve(algorithm=algorithm,tolerance=1e-10,max_iterations=5000)
    assert r['numerically_converged']
    np.testing.assert_allclose(r['velocity'],np.array([1.,0.,0.])+m.cell_centers@gradient.T,atol=2e-7)
    np.testing.assert_allclose(r['pressure'],.7,atol=2e-7)
    np.testing.assert_allclose(r['face_velocity'][planes,2],0,atol=1e-13)
    np.testing.assert_allclose(r['total_traction_integrated'][planes,:2],0,atol=2e-9)
    assert r['energy_balance_error']<1e-8


def test_kovasznay_symmetry_simplec_matches_coupled():
    answers=[]
    for algorithm in ('coupled','simplec'):
        m,s,_=kovasznay_solver(3,symmetry_extrusion=True)
        r=s.solve(algorithm=algorithm,tolerance=1e-8,max_outer_iterations=80,outer_relaxation=.7)
        assert r['numerically_converged'],r['outer_history'][-1]
        assert s.gauge
        assert r['energy_balance_error']<1e-7
        answers.append(r)
    for k in ('velocity','pressure','face_velocity','flux'):
        np.testing.assert_allclose(answers[0][k],answers[1][k],atol=1e-7)
