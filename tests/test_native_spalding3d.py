import numpy as np
import pytest
from tensorfvm.native_spalding3d import NativeSpaldingWall3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D


from tensorfvm.verification.native_spalding import spalding_couette_solver as couette


def test_cell_sampled_nonlinear_boundary_jacobian_and_force_assembly():
    m,s,*_=couette(2)
    rng=np.random.default_rng(24);state=rng.normal(size=s.state.shape)
    direction=rng.normal(size=state.shape);wall=s.wall_model
    matrix,source=wall.system(state);value=wall.evaluate(state)
    force=(matrix@state+source).reshape(-1,3)
    np.testing.assert_allclose(force[s.N:][wall.mask],-wall.area[:,None]*value['traction_pa'],atol=1e-15)
    assert np.max(abs(force[:s.N]))==0
    def residual(x):a,b=wall.system(x);return a@x+b
    h=1e-6
    np.testing.assert_allclose(wall.jacobian(state)@direction,(residual(state+h*direction)-residual(state-h*direction))/(2*h),rtol=2e-7,atol=2e-11)
    # Altering an auxiliary face trace must not alter cell-sampled stress.
    changed=state.copy();changed[3*s.N:]+=10
    np.testing.assert_array_equal(wall.evaluate(changed)['traction_pa'],value['traction_pa'])


@pytest.mark.parametrize('algorithm',['newton','simple','simplec'])
def test_unforced_couette_wall_stress_force_and_actual_boundary_work(algorithm):
    m,s,expected,tau,low,top=couette()
    assert np.count_nonzero(s.body_force)==0
    s.initialize_stokes()
    r=s.solve_newton(max_iterations=30,tolerance=1e-9) if algorithm=='newton' else s.solve(
        algorithm=algorithm,max_outer_iterations=80,max_iterations=3000,relaxation=.5,tolerance=1e-9)
    assert r['numerically_converged']
    np.testing.assert_allclose(r['velocity'],expected,atol=2e-7)
    wall=r['wall_model'];area=np.linalg.norm(m.area_vectors,axis=1)
    np.testing.assert_allclose(r['total_traction_integrated'][low,0],-tau*area[low],atol=1e-9)
    assert np.max(abs(r['flux'][low]))<1e-14
    assert wall['traction_defect_max']<1e-8
    assert wall['sample_dissipation']>0
    assert wall['work_decomposition_error']<1e-13
    assert abs(wall['sampling_transfer_power'])>1e-6
    assert r['energy_balance_error']<1e-8
    assert r['local_momentum_balance_max']<1e-8
    assert r['global_momentum_balance_max']<1e-8
    # Constitutive sample dissipation is not interchangeable with face work.
    assert abs(wall['boundary_power']+wall['sample_dissipation'])>1e-6


def test_wall_rejects_no_slip_or_open_normal_boundary():
    m,s,*_=couette(2);wall=s.wall_model
    for change in ('tangent','normal'):
        mask=s.viscous.dirichlet_components.copy()
        if change=='tangent':mask[wall.mask]=True
        else:mask[wall.mask]=False
        with pytest.raises(ValueError):NativeHybridNavierStokes3D(m,mask,viscosity=.02,
            backflow_velocity=[1,0,0],wall_model=wall)


def test_moving_wall_work_and_viscous_limit():
    for mu,nu,speed in ((.02,.001,.3),(100.,100.,0.)):
        m,s,expected,tau,low,top=couette(2,mu,nu,speed)
        s.initialize_stokes();r=s.solve_newton(tolerance=1e-9)
        assert r['numerically_converged']
        np.testing.assert_allclose(r['velocity'],expected,atol=2e-8)
        assert r['energy_balance_error']<1e-8
        assert r['wall_model']['wall_motion_power']==pytest.approx(-speed*tau,abs=1e-9)
        if speed==0:np.testing.assert_allclose(r['velocity'][:,0],m.cell_centers[:,1],atol=1e-8)


def test_cell_sampler_and_face_force_rotate_together():
    from scipy.spatial.transform import Rotation
    from tensorfvm.native_mesh3d import NativeMesh3D
    m,s,*_=couette(2);rng=np.random.default_rng(13)
    q=Rotation.from_rotvec([.4,-.7,.2]).as_matrix()
    rotated=NativeMesh3D.from_cells(m.vertices@q.T,m.cells,m.cell_types,triangulate_faces=True)
    np.testing.assert_array_equal(m.owner,rotated.owner)
    wall=NativeSpaldingWall3D(m,s.wall_model.mask,.25,molecular_viscosity=.001,wall_velocity=[.3,.2,-.1])
    other=NativeSpaldingWall3D(rotated,wall.mask,.25,molecular_viscosity=.001,wall_velocity=np.array([.3,.2,-.1])@q.T)
    state=rng.normal(size=(s.N+s.F,3));shift=state@q.T
    a,b=wall.system(state.ravel());c,d=other.system(shift.ravel())
    np.testing.assert_allclose((a@state.ravel()+b).reshape(-1,3)@q.T,(c@shift.ravel()+d).reshape(-1,3),atol=2e-16)
    direction=rng.normal(size=state.shape)
    np.testing.assert_allclose((wall.jacobian(state.ravel())@direction.ravel()).reshape(-1,3)@q.T,
        (other.jacobian(shift.ravel())@(direction@q.T).ravel()).reshape(-1,3),atol=2e-16)
