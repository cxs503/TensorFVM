import numpy as np
import pytest
from tensorfvm.native_hybrid_convection3d import NativeHybridConvection3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.verification.native_hybrid_ns import polynomial_solver, polynomial_fields
from tensorfvm.verification.native_energy import two_tetrahedra
from tests.test_native_hybrid_viscous3d import thin_mixed_mesh
from tests.test_native_rans3d import box


@pytest.mark.parametrize('factory', [two_tetrahedra, thin_mixed_mesh, lambda:box(2)])
def test_convection_global_conservation_and_full_energy_identity(factory):
    m = factory(); rng = np.random.default_rng(41)
    inflow = rng.normal(size=(len(m.owner),3)); q = rng.normal(size=len(m.owner))
    u = rng.normal(size=(len(m.cells),3))
    op = NativeHybridConvection3D(m, inflow, density=1.7)
    r = op.diagnostics(u,q)
    assert r['global_momentum_balance_error'] < 1e-12
    assert r['assembly_error'] < 1e-12
    assert r['energy_identity_error'] < 1e-12
    assert r['upwind_dissipation'] >= 0
    # This deliberately nonconservative q needs its divergence-power term.
    assert abs(r['divergence_power']) > 1e-4


def test_uniform_transport_on_skew_mesh_and_incoming_boundary_direction():
    m = two_tetrahedra(); velocity = np.array([.7,-.2,.4])
    q = m.area_vectors @ velocity
    op = NativeHybridConvection3D(m, velocity)
    r = op.diagnostics(np.tile(velocity,(len(m.cells),1)),q)
    np.testing.assert_allclose(r['integrated'],0,atol=1e-14)
    np.testing.assert_allclose(r['momentum_flux'],q[:,None]*velocity,atol=1e-14)
    matrix,source,_,_ = op.assemble(q)
    assert np.all(matrix.diagonal() >= 0)
    assert np.linalg.norm(source)>0
    np.testing.assert_allclose(op.assemble(np.zeros_like(q))[0].toarray(),0)


def test_polynomial_source_matches_independent_differential_equations():
    points=np.random.default_rng(11).uniform(.1,.9,(15,3)); h=1e-4
    f=polynomial_fields(points)
    gradient=np.stack([(polynomial_fields(points+np.eye(3)[j]*h)['velocity']-
                       polynomial_fields(points-np.eye(3)[j]*h)['velocity'])/(2*h) for j in range(3)],axis=2)
    laplacian=sum((polynomial_fields(points+np.eye(3)[j]*h)['velocity']-2*f['velocity']+
                   polynomial_fields(points-np.eye(3)[j]*h)['velocity'])/(h*h) for j in range(3))
    gp=np.column_stack([(polynomial_fields(points+np.eye(3)[j]*h)['pressure']-
                        polynomial_fields(points-np.eye(3)[j]*h)['pressure'])/(2*h) for j in range(3)])
    np.testing.assert_allclose(np.trace(gradient,axis1=1,axis2=2),0,atol=2e-12)
    conv=np.einsum('cij,cj->ci',gradient,f['velocity'])
    assert np.linalg.norm(conv) > .1
    np.testing.assert_allclose(conv+gp-.1*laplacian,f['force'],atol=1e-8)


@pytest.mark.parametrize('algorithm',['coupled','simple','simplec'])
def test_uniform_navier_stokes_and_explicit_backflow_boundary(algorithm):
    m=box(2); b=m.neighbor<0; outlet=b & (m.face_centers[:,0]>.99)
    velocity=np.array([1.,.1,-.05])
    solver=NativeHybridNavierStokes3D(m,b & ~outlet,velocity,viscosity=.1,backflow_velocity=velocity)
    r=solver.solve(algorithm=algorithm,tolerance=1e-9,max_outer_iterations=15)
    assert r['numerically_converged'],r['outer_history'][-1]
    np.testing.assert_allclose(r['velocity'],np.tile(velocity,(len(m.cells),1)),atol=1e-7)
    np.testing.assert_allclose(r['pressure'],0,atol=1e-7)
    assert r['energy_balance_error']<1e-8
    assert r['convection_energy_identity_error']<1e-12


def test_frozen_linear_convergence_is_not_nonlinear_acceptance():
    _,s,_=polynomial_solver(2)
    r=s.solve(algorithm='coupled',max_outer_iterations=1,tolerance=1e-9)
    assert r['outer_history'][-1]['frozen_inner_converged']
    assert r['outer_history'][-1]['current_momentum_residual']>1e-3
    assert not r['numerically_converged']
    assert not r['physical_benchmark_passed']


@pytest.mark.parametrize('order',[1,2])
def test_nonlinear_simple_variants_match_coupled_and_conserve_momentum(order):
    answers=[]
    for algorithm in ('coupled','simple','simplec'):
        m,s,_=polynomial_solver(3,convection_order=order)
        r=s.solve(algorithm=algorithm,tolerance=1e-9,max_outer_iterations=25)
        assert r['numerically_converged'],r['outer_history'][-1]
        assert r['energy_balance_error']<1e-8
        assert r['interface_traction_defect_max']<1e-8
        # Sum outward (advective momentum - total traction) equals body load.
        np.testing.assert_allclose(r['momentum_flux_integrated'][m.neighbor<0].sum(0),
                                   (m.cell_volumes[:,None]*s.body_force).sum(0),atol=1e-8)
        answers.append(r)
    for r in answers[1:]:
        for key in ('velocity','face_velocity','pressure','flux'):
            np.testing.assert_allclose(r[key],answers[0][key],atol=1e-7)


def test_missing_backflow_and_invalid_density_are_rejected():
    m=box(2); b=m.neighbor<0; outlet=b & (m.face_centers[:,0]>.99)
    with pytest.raises(ValueError,match='backflow_velocity'):
        NativeHybridNavierStokes3D(m,b & ~outlet,[1,0,0])
    with pytest.raises(ValueError,match='density'):
        NativeHybridNavierStokes3D(m,b,[1,0,0],density=-1)


def test_traction_boundary_actual_backflow_uses_explicit_ambient_velocity():
    m=box(2);b=m.neighbor<0;open_face=b & (m.face_centers[:,0]>.99)
    expected=np.array([-1.,.1,-.05])
    boundary=np.tile(expected,(len(m.owner),1));boundary[open_face]=[9,8,7]
    solver=NativeHybridNavierStokes3D(m,b & ~open_face,boundary,viscosity=.1,
                                    backflow_velocity=expected)
    r=solver.solve(algorithm='simplec',tolerance=1e-9,max_outer_iterations=20)
    assert r['numerically_converged'],r['outer_history'][-1]
    assert np.all(r['flux'][open_face]<0)
    np.testing.assert_allclose(r['velocity'],np.tile(expected,(len(m.cells),1)),atol=1e-7)


@pytest.mark.parametrize('factory',[two_tetrahedra,thin_mixed_mesh,lambda:box(2)])
@pytest.mark.parametrize('open_boundary',[False,True])
def test_pressure_gradient_force_balance_and_conservative_face_load(factory,open_boundary):
    m=factory(); b=m.neighbor<0; fixed=b.copy()
    if open_boundary: fixed[np.flatnonzero(b)[-1]]=False
    g=np.array([.7,-.3,.2]); pressure=.4+m.cell_centers @ g
    normal=m.area_vectors/np.linalg.norm(m.area_vectors,axis=1)[:,None]
    traction=-(.4+m.face_centers @ g)[:,None]*normal
    s=NativeHybridStokes3D(m,fixed,body_force=g,boundary_traction=traction,gauge_value=pressure[0])
    r=s.solve(algorithm='coupled',tolerance=1e-10)
    assert r['numerically_converged']
    np.testing.assert_allclose(r['velocity'],0,atol=2e-11)
    np.testing.assert_allclose(r['pressure'],pressure,atol=2e-11)
    np.testing.assert_allclose(r['total_traction_integrated'],
                               traction*np.linalg.norm(m.area_vectors,axis=1)[:,None],atol=2e-11)
    force=np.zeros((s.N,3))
    np.add.at(force,m.owner,s.owner_force_flux)
    i=m.neighbor>=0;np.add.at(force,m.neighbor[i],s.neighbor_force_flux[i])
    np.testing.assert_allclose(force,m.cell_volumes[:,None]*g,atol=1e-14)


def test_linear_upwind_affine_reconstruction_and_signed_energy_ledger():
    m=box(2);g=np.array([[.2,.4,.1],[-.3,-.1,.2],[.1,.1,-.1]])
    u=.4+m.cell_centers @ g.T;uf=.4+m.face_centers @ g.T
    q=np.sum(uf*m.area_vectors,axis=1)
    op=NativeHybridConvection3D(m,uf,order=2)
    r=op.diagnostics(u,q,np.broadcast_to(g,(len(u),3,3)))
    np.testing.assert_allclose(r['momentum_flux'],q[:,None]*uf,atol=1e-14)
    assert r['energy_identity_error']<1e-13
    assert r['global_momentum_balance_error']<1e-13
    assert abs(r['deferred_correction_power'])>1e-5
