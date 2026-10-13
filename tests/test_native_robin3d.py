import numpy as np
import pytest
from tensorfvm.native_robin3d import tangential_friction
from tensorfvm.native_hybrid_stokes3d import NativeHybridStokes3D
from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.verification.native_hybrid_flow import channel_mesh


from tensorfvm.verification.native_robin import slip_couette_solver as couette


@pytest.mark.parametrize('algorithm',['coupled','simple','simplec'])
def test_robin_couette_without_tangential_dirichlet_conserves_force_and_work(algorithm):
    m,s,expected,tau,low,high,g=couette()
    r=s.solve(algorithm=algorithm,max_iterations=6000,tolerance=1e-10)
    assert r['numerically_converged'] and s.rigid_motion_constraints['rank']==6
    np.testing.assert_allclose(r['velocity'],expected,atol=2e-8)
    np.testing.assert_allclose(r['pressure'],0.,atol=2e-8)
    area=np.linalg.norm(m.area_vectors,axis=1)
    np.testing.assert_allclose(r['total_traction_integrated'][low,0],-tau*area[low],atol=2e-9)
    np.testing.assert_allclose(r['total_traction_integrated'][high,0],tau*area[high],atol=2e-9)
    residual=r['total_traction_integrated']+r['robin_reaction_integrated']-area[:,None]*g
    assert np.max(abs(residual[s.viscous.traction_components]))<2e-8
    assert np.max(abs(r['total_traction_integrated'][m.neighbor<0].sum(0)))<2e-8
    assert r['energy_balance_error']<2e-8 and r['interface_traction_defect_max']<2e-8
    assert r['robin_power']>0 and r['bulk_viscous_energy']>0
    assert r['energy_form']==pytest.approx(r['robin_power']+r['bulk_viscous_energy'])
    # Moving-wall mechanical power balances bulk and relative-slip dissipation.
    slip=r['face_velocity'].copy();slip[high,0]-=1.
    slip_power=float(np.einsum('fi,fij,fj,f->',slip,s.viscous.robin_coefficient,slip,area))
    wall_power=float(r['total_traction_integrated'][high,0].sum())
    assert abs(wall_power-r['bulk_viscous_energy']-slip_power)<2e-8


@pytest.mark.parametrize('algorithm',['newton','simplec'])
def test_full_ns_robin_uses_current_convection_and_energy(algorithm):
    m,s,expected,*_=couette(ns=True)
    s.initialize_stokes()
    r=s.solve_newton(tolerance=1e-9) if algorithm=='newton' else s.solve(algorithm='simplec',tolerance=1e-9)
    assert r['numerically_converged']
    np.testing.assert_allclose(r['velocity'],expected,atol=1e-8)
    assert r['energy_balance_error']<1e-8 and r['local_momentum_balance_max']<1e-8


def test_viscosity_rebuild_preserves_robin_and_resets_solve_state():
    m,s,_,_,_,_,g=couette();s.solve(algorithm='coupled');c=s.viscous.robin_coefficient.copy()
    s.update_dynamic_viscosity(1.4)
    np.testing.assert_array_equal(c,s.viscous.robin_coefficient)
    assert not s.converged
    fresh=NativeHybridStokes3D(m,s.viscous.dirichlet_components,viscosity=1.4,boundary_traction=g,robin_coefficient=c)
    r=s.solve(algorithm='coupled');expected=fresh.solve(algorithm='coupled')
    for key in ('velocity','pressure','total_traction_integrated'):
        np.testing.assert_allclose(r[key],expected[key],atol=1e-9)


def test_full_tensor_robin_on_oblique_facets_has_correct_affine_solution():
    from tests.test_native_hybrid_viscous3d import thin_mixed_mesh
    m=thin_mixed_mesh();b=m.neighbor<0;c=np.zeros((len(b),3,3))
    matrix=np.array([[2.,.3,.2],[.3,1.,-.1],[.2,-.1,3.]])
    c[b]=matrix
    gradient=np.array([[.2,.3,-.1],[.4,-.2,.3],[-.1,.2,.0]])
    cells=.4+m.cell_centers@gradient.T;faces=.4+m.face_centers@gradient.T
    stress=gradient+gradient.T-2/3*np.trace(gradient)*np.eye(3)
    area=np.linalg.norm(m.area_vectors,axis=1)
    traction=m.area_vectors@stress.T/area[:,None]+np.einsum('fij,fj->fi',c,faces)
    s=NativeHybridViscous3D(m,dirichlet_mask=np.zeros(len(b),bool),robin_coefficient=c)
    r=s.solve(boundary_traction=traction)
    np.testing.assert_allclose(r['cell_values'],cells,atol=1e-9)
    np.testing.assert_allclose(r['face_flux'],m.area_vectors@stress.T,atol=1e-9)
    assert r['energy_balance_error']<1e-9


@pytest.mark.parametrize('invalid',['negative','asymmetric','internal','dirichlet','nonfinite'])
def test_invalid_robin_data_rejected(invalid):
    m=channel_mesh(2);b=m.neighbor<0;mask=np.zeros((len(b),3),bool);c=np.zeros((len(b),3,3));f=np.flatnonzero(b)[0]
    c[f]=np.eye(3)
    if invalid=='negative':c[f,0,0]=-1
    elif invalid=='asymmetric':c[f,0,1]=1
    elif invalid=='internal':c[np.flatnonzero(~b)[0]]=np.eye(3)
    elif invalid=='dirichlet':mask[f,0]=True
    else:c[f,0,0]=np.nan
    with pytest.raises(ValueError):NativeHybridViscous3D(m,dirichlet_mask=mask,robin_coefficient=c)


def test_block_amg_includes_robin_damping():
    pytest.importorskip('pyamg')
    m,s,expected,*_=couette(method='block_amg')
    r=s.solve(algorithm='coupled',tolerance=1e-9)
    assert r['numerically_converged']
    np.testing.assert_allclose(r['velocity'],expected,atol=1e-8)


def test_scalar_robin_reproduces_affine_flux_without_dirichlet_values():
    m=channel_mesh(2);b=m.neighbor<0
    c=np.zeros((len(b),1,1));c[b,0,0]=2.
    gradient=np.array([.2,-.1,.3]);face=1+m.face_centers@gradient
    area=np.linalg.norm(m.area_vectors,axis=1)
    g=(m.area_vectors@gradient/area+2*face)[:,None]
    s=NativeHybridViscous3D(m,components=1,dirichlet_mask=np.zeros(len(b),bool),robin_coefficient=c)
    r=s.solve(boundary_traction=g)
    np.testing.assert_allclose(r['cell_values'][:,0],1+m.cell_centers@gradient,atol=1e-11)
    np.testing.assert_allclose(r['face_flux'][:,0],m.area_vectors@gradient,atol=1e-11)
    np.testing.assert_allclose((r['face_flux']+r['robin_reaction_integrated']-area[:,None]*g)[b],0.,atol=1e-11)
