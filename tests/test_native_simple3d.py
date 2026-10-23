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


@pytest.mark.parametrize('damping',[1.,.3])
def test_simple_step_preserves_face_flux_ledger(damping):
    """Every accepted sweep must keep flux == B u + correction p + qsource.

    The legacy update under-relaxed the pressure alone (p += alpha_p dp) while
    the flux took the full response, injecting (1 - alpha_p) correction dp of
    ledger inconsistency per sweep.  On thin-layer external meshes that defect
    grew without bound.  Damping must now be a convex mix of the whole
    consistent step, which leaves the ledger exact for every damping factor.
    """
    m=channel();fixed=(m.neighbor<0)&(m.patch_labels!='outlet')
    bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[m.patch_labels=='wall']=0.
    s=NativeSimpleSSTRansSolver3D(m,fixed,bd,m.patch_labels=='outlet',
        settings=NativeSteadySettings(viscosity=.01,max_iterations=6,relaxation=.4,
            pressure_relaxation=damping,convection_order=1),
        rans_settings=NativeRansSettings(max_outer_iterations=1,pseudo_time_step=1.,wall_treatment='spalding'))
    s.u[:,0]=1.
    k0,omega0=.001,1.
    s.k=np.full(s.N,k0);s.omega=np.full(s.N,omega0)
    s.k_boundary=np.full(s.F,k0);s.omega_boundary=np.full(s.F,omega0)
    s._update_momentum_coefficients(s.current_closure())
    result=s.solve()
    assert len(result['history'])==6
    defects=[row['ledger_identity_defect_max'] for row in result['history']]
    if damping==1.:
        # A full consistent step puts the state on the ledger manifold at once.
        assert max(defects)<1e-12,defects
    else:
        # Uniform damping contracts the inherited off-manifold part by (1-w);
        # it never injects new inconsistency through the pressure update.
        assert np.all(np.diff(defects)<=1e-15) and defects[-1]<=defects[0]*(1-damping)**2,defects
    assert all(row['native_divergence_max']<1e-9 for row in result['history'])
    assert result['history'][-1]['pressure_increment_flux_response_max']>0.


def test_pressure_damping_does_not_change_converged_steady_solution():
    """Uniform damping is an iteration parameter, never a term in the equations."""
    m=skew_mesh(3);x,y,z=m.cell_centers.T
    force=np.column_stack((np.sin(3*y),np.cos(2*x),x*z));answers=[]
    for damping in (1.,.4):
        s=NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),body_force=force,
            settings=NativeSteadySettings(viscosity=.1,convection=False,relaxation=.5,
                pressure_relaxation=damping,max_iterations=1500,nonlinear_tolerance=1e-9))
        result=s.solve();assert result['converged'],result['history'][-1]
        answers.append((s.u.copy(),s.p.copy(),s.flux.copy()))
    for left,right in zip(*answers):np.testing.assert_allclose(left,right,atol=2e-8)


def test_channel_pressure_operator_is_admissible():
    """The pressure correction must stay a scaled Laplacian on clean meshes.

    Thin-layer and sliver cells can flip the sign of a correction row, which
    leaves dp only weakly constrained.  The channel is admissible and must
    report a fully signed, diagonally meaningful operator, so the diagnostic
    cannot silently rot.
    """
    m=channel();fixed=(m.neighbor<0)&(m.patch_labels!='outlet')
    bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[m.patch_labels=='wall']=0.
    s=NativeSimpleSSTRansSolver3D(m,fixed,bd,m.patch_labels=='outlet',
        settings=NativeSteadySettings(viscosity=.01,max_iterations=3,relaxation=.4,
            pressure_relaxation=1.,convection_order=1),
        rans_settings=NativeRansSettings(max_outer_iterations=1,pseudo_time_step=1.,wall_treatment='spalding'))
    s.u[:,0]=1.
    k0,omega0=.001,1.
    s.k=np.full(s.N,k0);s.omega=np.full(s.N,omega0)
    s.k_boundary=np.full(s.F,k0);s.omega_boundary=np.full(s.F,omega0)
    s._update_momentum_coefficients(s.current_closure())
    result=s.solve()
    for row in result['history']:
        assert row['pressure_operator_negative_diagonal_count']==0,row
        assert row['pressure_operator_minimum_diagonal_ratio']>1e-6,row
        assert row['maximum_increment']>0.


def test_increment_damping_is_a_pseudo_time_not_a_term():
    """The increment anchor must vanish at the fixed point.

    It is a Tikhonov term on dp only, so a converged answer cannot depend on
    it.  This pins that a damped run reaches the same discrete steady state.
    """
    m=skew_mesh(3);x,y,z=m.cell_centers.T
    force=np.column_stack((np.sin(2*y),np.cos(3*x),x*y));answers=[]
    for damping in (0.,.5):
        s=NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),body_force=force,
            settings=NativeSteadySettings(viscosity=.1,convection=False,relaxation=.5,
                pressure_relaxation=1.,pressure_increment_damping=damping,
                max_iterations=3000,nonlinear_tolerance=1e-9))
        result=s.solve();assert result['converged'],result['history'][-1]
        assert result['history'][-1]['pressure_increment_damping_scale']>=0.
        answers.append((s.u.copy(),s.p.copy(),s.flux.copy()))
    for left,right in zip(*answers):np.testing.assert_allclose(left,right,atol=2e-8)


def test_rejects_negative_increment_damping():
    m=channel()
    with pytest.raises(ValueError,match='pressure increment damping'):
        NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),
            settings=NativeSteadySettings(pressure_increment_damping=-1.)).solve()


@pytest.mark.parametrize('algorithm',['simple','simplec'])
def test_amg_pressure_solve_matches_direct(algorithm):
    """The AMG pressure solve is a preconditioner, not a different equation."""
    pytest.importorskip('pyamg')
    m=skew_mesh(3);x,y,z=m.cell_centers.T
    force=np.column_stack((np.sin(3*y),np.cos(2*x),x*z));answers=[]
    for method in ('direct','amg'):
        s=NativeSimpleSolver3D(m,m.neighbor<0,np.zeros((len(m.owner),3)),body_force=force,
            settings=NativeSteadySettings(viscosity=.1,convection=False,relaxation=.5,
                pressure_relaxation=1.,linear_method=method,pressure_velocity_algorithm=algorithm,
                max_iterations=1500,nonlinear_tolerance=1e-9))
        result=s.solve();assert result['converged'],result['history'][-1]
        assert result['history'][-1]['pressure_linear_solver']==method
        answers.append((s.u.copy(),s.p.copy(),s.flux.copy()))
    for left,right in zip(*answers):np.testing.assert_allclose(left,right,atol=2e-8)
