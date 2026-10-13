"""Independent force, transport and energy checks for shared-face inertia."""
import numpy as np
import pytest
from tensorfvm.native_face_mean3d import NativeFaceMean3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.verification.native_hybrid_pressure import pressure_mesh
from tensorfvm.verification.native_hybrid_ns import polynomial_solver


def test_mean_constant_reproduction_duality_and_cell_force_balance():
    m=pressure_mesh(2);op=NativeFaceMean3D(m);rng=np.random.default_rng(18)
    constant=np.array([.7,-.4,.8]);state=np.tile(constant,(len(m.cells)+len(m.owner),1)).ravel()
    np.testing.assert_allclose((op.matrix@state).reshape(-1,3),np.tile(constant,(len(m.cells),1)),atol=2e-14)
    force=rng.normal(size=(len(m.cells),3));owner,neighbor=op.moments(force)
    dual=op.matrix.T@force.ravel()
    np.testing.assert_allclose(dual[:3*len(m.cells)],0,atol=1e-14)
    np.testing.assert_allclose(dual[3*len(m.cells):].reshape(-1,3),owner+neighbor,atol=2e-14)
    cell_sum=np.zeros_like(force);np.add.at(cell_sum,m.owner,owner)
    internal=m.neighbor>=0;np.add.at(cell_sum,m.neighbor[internal],neighbor[internal])
    np.testing.assert_allclose(cell_sum,force,atol=2e-14)


@pytest.mark.parametrize('order',[1,2])
def test_random_inertia_matrix_flux_energy_and_signed_corrections(order):
    m=pressure_mesh(2);rng=np.random.default_rng(123)
    s=NativeHybridNavierStokes3D(m,m.neighbor<0,rng.normal(size=(len(m.owner),3))*0,
                               convection_order=order,density=1.7)
    s.state=rng.normal(size=len(s.state))
    base=(s.face_mean.matrix@s.state).reshape(-1,3)
    u=s.state[:3*s.N].reshape(-1,3) if order==2 else base
    adv=s.convection.diagnostics(u,s.volume_flux(),s.velocity_gradient(),base_velocity=base)
    full,source=s._current_system()
    reaction=(full-s.viscous.full_matrix)@s.state+source
    np.testing.assert_allclose(reaction,s.face_mean.matrix.T@adv['integrated'].ravel(),atol=2e-13)
    np.testing.assert_allclose(s.state@reaction,adv['power'],atol=3e-13)
    assert adv['energy_identity_error']<3e-13
    assert adv['global_momentum_balance_error']<3e-13
    assert adv['upwind_dissipation']>=0
    assert abs(adv['divergence_power'])>1e-3
    if order==2:assert abs(adv['projection_correction_power'])>1e-3
    else:assert adv['projection_correction_power']==0


def test_order_two_restores_affine_face_transport_on_skew_mixed_cells():
    m=pressure_mesh(2);g=np.array([[.2,.4,.1],[-.3,-.1,.2],[.1,.1,-.1]])
    uf=.4+m.face_centers@g.T;u=.4+m.cell_centers@g.T
    s=NativeHybridNavierStokes3D(m,m.neighbor<0,uf,convection_order=2)
    s.state=np.r_[u.ravel(),uf.ravel()]
    base=(s.face_mean.matrix@s.state).reshape(-1,3)
    assert np.max(np.abs(base-u))>1e-3 # test needs non-affine-preserving R
    np.testing.assert_allclose(s.velocity_gradient(),np.broadcast_to(g,(s.N,3,3)),atol=1e-14)
    adv=s.convection.diagnostics(u,s.volume_flux(),s.velocity_gradient(),base_velocity=base)
    np.testing.assert_allclose(adv['momentum_flux'],s.volume_flux()[:,None]*uf,atol=2e-14)


@pytest.mark.parametrize('order',[1,2])
def test_converged_inertia_moments_are_required_for_local_and_interface_balance(order):
    m,s,_=polynomial_solver(3,convection_order=order)
    r=s.solve(algorithm='simplec',tolerance=1e-9,max_outer_iterations=40)
    assert r['numerically_converged']
    assert r['local_momentum_balance_max']<1e-8
    assert r['global_momentum_balance_max']<1e-8
    assert r['interface_traction_defect_max']<1e-8
    # Independently sum both outward tractions; the correction cannot merely
    # mask a defect by setting neighbor flux equal to negative owner flux.
    summed=np.zeros((s.N,3));np.add.at(summed,m.owner,r['total_traction_integrated'])
    i=m.neighbor>=0;np.add.at(summed,m.neighbor[i],r['neighbor_total_traction_integrated'])
    force=m.cell_volumes[:,None]*s.body_force
    adv=s.convection.D@(r['momentum_flux_integrated']+r['total_traction_integrated'])
    np.testing.assert_allclose(adv-summed,force,atol=1e-8)


def test_potential_flow_satisfies_unforced_equations_at_all_viscosities():
    from tensorfvm.verification.native_hybrid_ns import potential_fields
    x=np.random.default_rng(8).uniform(.1,.9,(13,3));h=1e-4;e=potential_fields(x)
    plus=[potential_fields(x+h*np.eye(3)[j]) for j in range(3)]
    minus=[potential_fields(x-h*np.eye(3)[j]) for j in range(3)]
    grad=np.stack([(p['velocity']-m['velocity'])/(2*h) for p,m in zip(plus,minus)],axis=2)
    gp=np.column_stack([(p['pressure']-m['pressure'])/(2*h) for p,m in zip(plus,minus)])
    lap=sum((p['velocity']-2*e['velocity']+m['velocity'])/h**2 for p,m in zip(plus,minus))
    np.testing.assert_allclose(grad,e['gradient'],atol=3e-12)
    np.testing.assert_allclose(np.trace(grad,axis1=1,axis2=2),0,atol=3e-12)
    conv=np.einsum('cij,cj->ci',grad,e['velocity'])
    assert np.all(np.linalg.norm(conv,axis=0)>.1)
    for mu in (1.,.1,.01,.001):np.testing.assert_allclose(conv+gp-mu*lap,0,atol=1e-7)
    assert not e['force'].any()
