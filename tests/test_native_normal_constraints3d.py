import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.native_robin3d import tangential_friction
from tensorfvm.verification.native_robin import slip_couette_solver


def rotated_couette(algorithm,method='direct'):
    base,original,exact,*rest=slip_couette_solver(4);g=rest[-1]
    q=Rotation.from_rotvec([.43,-.61,.29]).as_matrix()
    mesh=NativeMesh3D.from_cells(base.vertices@q.T,base.cells,base.cell_types,triangulate_faces=True)
    np.testing.assert_array_equal(mesh.owner,base.owner);np.testing.assert_array_equal(mesh.neighbor,base.neighbor)
    normal=original.viscous.dirichlet_components.any(axis=1)
    c=np.einsum('ij,fjk,lk->fil',q,original.viscous.robin_coefficient,q)
    is_ns=algorithm=='newton';cls=NativeHybridNavierStokes3D if is_ns else NativeHybridStokes3D
    velocity=np.zeros((len(base.owner),3));velocity[:,0]=rest[0]*(base.face_centers[:,1]/.7+.5)
    extra=dict(backflow_velocity=velocity@q.T,convection_order=2) if is_ns else {}
    s=cls(mesh,np.zeros(len(mesh.owner),bool),viscosity=.7,boundary_traction=g@q.T,
        robin_coefficient=c,normal_velocity_faces=normal,coupled_linear_method=method,**extra)
    if is_ns:s.initialize_stokes();result=s.solve_newton(tolerance=1e-9)
    else:result=s.solve(algorithm=algorithm,relaxation=1.,tolerance=1e-9,max_iterations=5000)
    return mesh,s,result,exact@q.T,normal


@pytest.mark.parametrize('algorithm',['coupled','simplec','newton'])
def test_rotated_slip_channel_preserves_physics_and_zero_normal_flux(algorithm):
    m,s,r,exact,normal=rotated_couette(algorithm)
    assert r['numerically_converged']
    np.testing.assert_allclose(r['velocity'],exact,atol=2e-8)
    np.testing.assert_allclose(r['pressure'],0.,atol=2e-8)
    assert np.max(abs(r['flux'][normal]))<1e-14
    assert not s.open_normal_boundary[normal].any()
    assert r['energy_balance_error']<1e-8
    assert r['interface_traction_defect_max']<1e-8
    p=s.viscous.velocity_space.prolongation
    np.testing.assert_allclose((p.T@p).toarray(),np.eye(p.shape[1]),atol=5e-16)
    # Current pressure work remains exactly the transpose of reduced divergence.
    rng=np.random.default_rng(81);u=rng.normal(size=p.shape[1]);pressure=rng.normal(size=s.N)
    np.testing.assert_allclose(pressure@(s.divergence@(p@u)),u@(s.B.T@pressure),atol=1e-13)


def cylinder(triangulate=False):
    n=12;theta=np.arange(n)*2*np.pi/n
    plane=np.vstack(([0.,0.],np.column_stack((np.cos(theta),np.sin(theta)))))
    vertices=np.vstack([np.column_stack((plane,np.full(n+1,z))) for z in (0.,1.)])
    cells=[[0,i+1,(i+1)%n+1,n+1,n+2+i,n+2+(i+1)%n] for i in range(n)]
    return NativeMesh3D.from_cells(vertices,cells,['wedge']*n,triangulate_faces=triangulate)


@pytest.mark.parametrize('method',['direct','block_amg'])
def test_polygonal_cylinder_rotating_wall_has_no_penetration_or_spurious_stress(method):
    if method=='block_amg':pytest.importorskip('pyamg')
    m=cylinder();wall=m.neighbor<0
    def rotate(x):return np.column_stack((-x[:,1],x[:,0],np.zeros(len(x))))
    c=tangential_friction(m,wall,2.);wall_u=rotate(m.face_centers)
    g=np.einsum('fij,fj->fi',c,wall_u)
    s=NativeHybridStokes3D(m,np.zeros(len(wall),bool),robin_coefficient=c,
        normal_velocity_faces=wall,boundary_traction=g,coupled_linear_method=method)
    r=s.solve(algorithm='coupled',tolerance=1e-9)
    assert s.gauge and s.rigid_motion_constraints['rank']==6 and r['numerically_converged']
    np.testing.assert_allclose(r['velocity'],rotate(m.cell_centers),atol=2e-8)
    assert np.max(abs(r['flux'][wall]))<1e-14
    assert np.max(abs(r['total_traction_integrated']))<1e-9
    assert abs(r['bulk_viscous_energy'])<1e-9 and r['energy_balance_error']<1e-8


def test_affine_normal_data_is_preserved_in_viscous_face_equilibrium_and_rebuild():
    m=cylinder();wall=m.neighbor<0
    # Translation has a compatible nonzero pointwise normal boundary velocity.
    values=np.broadcast_to([.3,-.2,.1],(len(wall),3)).copy()
    c=tangential_friction(m,wall,2.);g=np.einsum('fij,fj->fi',c,values)
    op=NativeHybridViscous3D(m,dirichlet_mask=np.zeros(len(wall),bool),normal_velocity_faces=wall,robin_coefficient=c)
    r=op.equilibrate_faces(np.broadcast_to([.3,-.2,.1],(len(m.cells),3)).copy(),values,boundary_traction=g)
    np.testing.assert_allclose(r['face_values'],values,atol=1e-10)
    s=NativeHybridStokes3D(m,np.zeros(len(wall),bool),values,normal_velocity_faces=wall,robin_coefficient=c,boundary_traction=g)
    s.update_dynamic_viscosity(2.3);r=s.solve(algorithm='coupled')
    np.testing.assert_allclose(r['velocity'],np.tile([.3,-.2,.1],(s.N,1)),atol=1e-9)
    assert r['energy_balance_error']<1e-9


def test_conflicting_or_internal_normal_constraints_rejected():
    m=cylinder();wall=m.neighbor<0
    with pytest.raises(ValueError,match='overlap'):
        NativeHybridViscous3D(m,dirichlet_mask=wall,normal_velocity_faces=wall)
    with pytest.raises(ValueError,match='boundary'):
        NativeHybridViscous3D(m,dirichlet_mask=np.zeros(len(wall),bool),normal_velocity_faces=~wall)


def test_triangle_cylinder_stationary_faces_use_zero_flux_not_invalid_rigid_reference():
    m=cylinder(triangulate=True);wall=m.neighbor<0
    wall_u=np.column_stack((-m.face_centers[:,1],m.face_centers[:,0],np.zeros(len(wall))))
    # A rigid rotation is not tangent at the means of split polygon triangles.
    assert np.max(abs(np.sum(wall_u[wall]*m.area_vectors[wall],axis=1)))>1e-3
    c=tangential_friction(m,wall,2.);g=np.einsum('fij,fj->fi',c,wall_u)
    results=[]
    for method in ('direct','block_amg'):
        if method=='block_amg':pytest.importorskip('pyamg')
        s=NativeHybridStokes3D(m,np.zeros(len(wall),bool),normal_velocity_faces=wall,
            robin_coefficient=c,boundary_traction=g,coupled_linear_method=method)
        r=s.solve(algorithm='coupled',tolerance=1e-9)
        assert r['numerically_converged'] and np.max(abs(r['flux'][wall]))<1e-14
        assert r['energy_balance_error']<1e-8 and r['interface_traction_defect_max']<1e-9
        results.append(r)
    np.testing.assert_allclose(results[0]['velocity'],results[1]['velocity'],atol=1e-9)
    np.testing.assert_allclose(results[0]['pressure'],results[1]['pressure'],atol=1e-9)


def test_closed_normal_boundary_rejects_unbalanced_flux():
    m=cylinder();wall=m.neighbor<0
    normal=m.area_vectors/np.linalg.norm(m.area_vectors,axis=1)[:,None]
    c=tangential_friction(m,wall,1.)
    with pytest.raises(ValueError,match='unbalanced'):
        NativeHybridStokes3D(m,np.zeros(len(wall),bool),normal,normal_velocity_faces=wall,robin_coefficient=c)


def test_result_cannot_certify_a_mutated_velocity_violating_normal_boundary():
    m,s,r,_,selected=rotated_couette('coupled')
    assert r['numerically_converged']
    face=np.flatnonzero(selected)[0];normal=s.viscous.velocity_space.normal[face]
    s.state[3*(s.N+face):3*(s.N+face)+3]+=.01*normal
    r=s.result()
    assert r['boundary_velocity_error']>.009 and not r['numerically_converged']
