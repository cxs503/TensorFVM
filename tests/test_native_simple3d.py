import numpy as np
import pytest
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_simple3d import NativeSimpleSolver3D
from tensorfvm.native_rans3d import NativeRansSettings,NativeSimpleSSTRansSolver3D
from tensorfvm.wall_functions import wall_traction
from tests.test_native_collocated3d import skew_mesh
from tests.test_native_rans3d import channel


@pytest.mark.parametrize('algorithm',['simple','simplec'])
@pytest.mark.parametrize('convection',[False,True])
def test_simple_affine_skew_mesh_exact_solution(convection,algorithm):
    m=skew_mesh(2);G=np.array([[.2,.4,.1],[-.3,-.1,.2],[.1,.1,-.1]])
    u=np.array([.3,-.1,.2])+m.cell_centers@G.T
    bd=np.array([.3,-.1,.2])+m.face_centers@G.T
    g=np.array([.7,-.3,.2]);p=m.cell_centers@g
    s=NativeSimpleSolver3D(m,m.neighbor<0,bd,body_force=np.tile(g,(len(u),1))+(u@G.T if convection else 0),gauge_value=p[0],
        settings=NativeSteadySettings(pressure_velocity_algorithm=algorithm,viscosity=.5,convection=convection,max_iterations=1200,relaxation=.5,nonlinear_tolerance=1e-9))
    r=s.solve();assert r['converged'],r['history'][-1]
    np.testing.assert_allclose(s.u,u,atol=1e-8);np.testing.assert_allclose(s.p,p,atol=1e-8)


def test_simple_wall_function_coupling_and_force_ledger():
    m=channel();fixed=(m.neighbor<0)&(m.patch_labels!='outlet')
    bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[m.patch_labels=='wall']=0.
    s=NativeSimpleSSTRansSolver3D(m,fixed,bd,m.patch_labels=='outlet',
        settings=NativeSteadySettings(viscosity=.001,max_iterations=30,relaxation=.4,convection_order=1),
        rans_settings=NativeRansSettings(max_outer_iterations=3,pseudo_time_step=.001,wall_treatment='spalding'))
    s.u[:,0]=1.
    r=s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.)
    assert len(r['history'])==3 and np.all(s.k>=0) and np.all(s.omega>0)
    assert np.max(np.abs(s.flux[s.wall]))==0
    assert r['history'][-1]['stored_native_divergence_max']<1e-9
    assert not s.transport.mask[s.wall].any()
    f=s.forces();m=s.mesh;faces=np.flatnonzero(s.wall);area=np.linalg.norm(m.area_vectors[faces],axis=1)
    n=m.area_vectors[faces]/area[:,None];u=s.u[m.owner[faces]]
    tangent=u-n*np.sum(u*n,axis=1)[:,None]
    expected=wall_traction(tangent,s.wall_distance[m.owner[faces]],1.,.001)['traction_pa']*area[:,None]
    actual=f['anisotropic_stress_traction'][faces];actual-=n*np.sum(actual*n,axis=1)[:,None]
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    np.testing.assert_allclose(f['total_force'],f['physical_pressure_force']+f['anisotropic_viscous_force']+f['isotropic_turbulent_force'],atol=1e-12)
    ledger=r['history'][-1]['frozen_transport_ledger']['omega']
    assert abs(ledger['global_residual'])<1e-9
    assert ledger['max_local_integrated_residual']<1e-9
    assert not r['numerically_converged']


@pytest.mark.parametrize('algorithm',['simple','simplec'])
@pytest.mark.parametrize('face_skew_correction',[True,False])
def test_simple_sst_wall_function_converges_current_equations(face_skew_correction,algorithm):
    m=channel();fixed=(m.neighbor<0)&(m.patch_labels!='outlet')
    bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[m.patch_labels=='wall']=0.
    s=NativeSimpleSSTRansSolver3D(m,fixed,bd,m.patch_labels=='outlet',
        settings=NativeSteadySettings(pressure_velocity_algorithm=algorithm,viscosity=.01,relaxation=.4,
            pressure_relaxation=1. if algorithm=='simplec' else .2,
            convection_order=1,max_iterations=10,face_skew_correction=face_skew_correction),
        rans_settings=NativeRansSettings(max_outer_iterations=80,pseudo_time_step=1.,
            turbulence_relaxation=.7,wall_treatment='spalding'))
    s.u[:,0]=1.
    r=s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.)
    assert r['numerically_converged'],r['history'][-1]
    residual=s.turbulence_residual()
    assert max(residual[name]['relative_residual'] for name in ('k','omega'))<1e-7
    block,target,_,corr,source=s._system(s.flux)
    assert np.linalg.norm((block@np.r_[s.u.ravel(),s.p]-target)[:3*s.N])<1e-7
    reconstructed=s.velocity_flux_matrix@s.u.ravel()+corr@s.p+source
    assert np.max(np.abs(s.viscous.op.integrate(reconstructed)/m.cell_volumes))<1e-7
    assert not r['physical_benchmark_passed']


@pytest.mark.parametrize('algorithm',['simple','simplec'])
@pytest.mark.parametrize('pressure_outlet',[False,True])
def test_one_full_pressure_correction_matches_reconstructed_flux(pressure_outlet,algorithm):
    """Frozen Stokes coefficients isolate correction consistency from iteration lag."""
    m=skew_mesh(3);boundary=m.neighbor<0
    outlet=boundary&(m.face_centers[:,0]>1.) if pressure_outlet else np.zeros(len(m.owner),bool)
    x,y,z=m.cell_centers.T
    force=np.column_stack((np.sin(3*y),np.cos(2*x),x*z))
    s=NativeSimpleSolver3D(m,boundary&~outlet,np.zeros((len(m.owner),3)),outlet,
        body_force=force,settings=NativeSteadySettings(pressure_velocity_algorithm=algorithm,viscosity=.1,convection=False,
            max_iterations=1,relaxation=.5,pressure_relaxation=1.))
    r=s.solve()
    assert np.linalg.norm(s.u)>1e-3 and np.linalg.norm(s.p)>1e-3
    # Pressure correction must conserve both the stored face flux and the flux
    # independently reconstructed from the corrected pressure and velocity.
    row=r['history'][-1]
    assert row['native_divergence_max']<1e-11
    assert row['current_reconstructed_divergence_max']<1e-11
    assert row['face_flux_consistency_max']<1e-11
    assert not r['converged']  # A projection is not a converged Stokes solution.


def test_interrupted_momentum_reports_current_wall_law_without_mutating_solver():
    m=channel();wall=m.patch_labels=='wall';bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[wall]=0.
    s=NativeSimpleSSTRansSolver3D(m,(m.neighbor<0)&(m.patch_labels!='outlet'),bd,m.patch_labels=='outlet',
        settings=NativeSteadySettings(viscosity=.001,max_iterations=1,convection_order=1,relaxation=.3),
        rans_settings=NativeRansSettings(wall_treatment='spalding'))
    s.u[:,0]=1.
    def stop(row):raise StopIteration('diagnostic guard')
    with pytest.raises(StopIteration):s.solve_rans(.001,1.,inlet_k=.001,inlet_omega=1.,momentum_callback=stop)
    matrix=s._diffusion.copy();velocity=s.u.copy();f=s.forces()
    area=np.linalg.norm(m.area_vectors[wall],axis=1);n=m.area_vectors[wall]/area[:,None]
    u=s.u[m.owner[wall]];tangent=u-n*np.sum(u*n,axis=1)[:,None]
    expected=wall_traction(tangent,s.wall_distance[m.owner[wall]],1.,.001)['traction_pa']*area[:,None]
    actual=f['anisotropic_stress_traction'][wall];actual=actual-n*np.sum(actual*n,axis=1)[:,None]
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    np.testing.assert_array_equal(s.u,velocity)
    assert (s._diffusion!=matrix).nnz==0
    assert np.linalg.norm(f['linearized_anisotropic_stress_traction']-f['anisotropic_stress_traction'])>1e-8
    np.testing.assert_allclose(f['total_force'],f['physical_pressure_force']+f['anisotropic_viscous_force']+f['isotropic_turbulent_force'],atol=1e-12)


def test_momentum_relaxation_does_not_change_converged_stokes_solution():
    m=skew_mesh(3);x,y,z=m.cell_centers.T
    force=np.column_stack((np.sin(3*y),np.cos(2*x),x*z));answers=[]
    for relaxation in (.3,.8):
        s=NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),body_force=force,
            settings=NativeSteadySettings(viscosity=.1,convection=False,relaxation=relaxation,
                max_iterations=1500,nonlinear_tolerance=1e-9))
        result=s.solve();assert result['converged']
        answers.append((s.u.copy(),s.p.copy()))
    np.testing.assert_allclose(answers[0][0],answers[1][0],atol=2e-8)
    np.testing.assert_allclose(answers[0][1],answers[1][1],atol=2e-8)


@pytest.mark.parametrize("bad_pressure",["inaccurate","nonfinite"])
def test_rejected_linear_solve_preserves_last_accepted_state(monkeypatch,bad_pressure):
    import tensorfvm.native_simple3d as module
    original=module.splu;m=skew_mesh(2)
    s=NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),
        body_force=np.tile([.3,.2,.1],(len(m.cells),1)),
        settings=NativeSteadySettings(viscosity=.1,convection=False,max_iterations=1))
    u=np.zeros((s.N,3));p=np.zeros(s.N);q=np.zeros(s.F)
    s.initialize_state(u,p,q)
    class PerturbedFactor:
        def __init__(self,matrix,**kwargs):self.factor=original(matrix,**kwargs);self.size=matrix.shape[0]
        def solve(self,rhs):
            answer=self.factor.solve(rhs)
            if self.size!=s.N:return answer
            return np.full(s.N,np.nan) if bad_pressure=="nonfinite" else answer+np.linspace(0.,1.,s.N)
    monkeypatch.setattr(module,'splu',PerturbedFactor)
    with pytest.raises(RuntimeError,match='linear solve'):s.solve()
    np.testing.assert_array_equal(s.u,u)
    np.testing.assert_array_equal(s.p,p)
    np.testing.assert_array_equal(s.flux,q)
    assert not s.history


def test_simplec_matches_simple_nonaffine_steady_solution():
    m=skew_mesh(3);x,y,z=m.cell_centers.T
    force=np.column_stack((np.sin(3*y),np.cos(2*x),x*z));answers=[]
    for algorithm in ('simple','simplec'):
        s=NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),body_force=force,
            settings=NativeSteadySettings(viscosity=.1,convection=False,relaxation=.5,
                pressure_velocity_algorithm=algorithm,pressure_relaxation=1.,
                max_iterations=1500,nonlinear_tolerance=1e-9))
        result=s.solve();assert result['converged'],result['history'][-1]
        assert result['history'][-1]['algorithm']==algorithm.upper()
        answers.append((s.u.copy(),s.p.copy(),s.flux.copy()))
    for left,right in zip(*answers):np.testing.assert_allclose(left,right,atol=2e-8)
