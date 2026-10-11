import numpy as np
import pytest
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_simple3d import NativeSimpleSolver3D
from tensorfvm.native_rans3d import NativeRansSettings,NativeSimpleSSTRansSolver3D
from tensorfvm.wall_functions import wall_traction
from tests.test_native_collocated3d import skew_mesh
from tests.test_native_rans3d import channel


@pytest.mark.parametrize('convection',[False,True])
def test_simple_affine_skew_mesh_exact_solution(convection):
    m=skew_mesh(2);G=np.array([[.2,.4,.1],[-.3,-.1,.2],[.1,.1,-.1]])
    u=np.array([.3,-.1,.2])+m.cell_centers@G.T
    bd=np.array([.3,-.1,.2])+m.face_centers@G.T
    g=np.array([.7,-.3,.2]);p=m.cell_centers@g
    s=NativeSimpleSolver3D(m,m.neighbor<0,bd,body_force=np.tile(g,(len(u),1))+(u@G.T if convection else 0),gauge_value=p[0],
        settings=NativeSteadySettings(viscosity=.5,convection=convection,max_iterations=1200,relaxation=.5,nonlinear_tolerance=1e-9))
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


@pytest.mark.parametrize('face_skew_correction',[True,False])
def test_simple_sst_wall_function_converges_current_equations(face_skew_correction):
    m=channel();fixed=(m.neighbor<0)&(m.patch_labels!='outlet')
    bd=np.tile([1.,0.,0.],(len(m.owner),1));bd[m.patch_labels=='wall']=0.
    s=NativeSimpleSSTRansSolver3D(m,fixed,bd,m.patch_labels=='outlet',
        settings=NativeSteadySettings(viscosity=.01,relaxation=.4,pressure_relaxation=.2,
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
