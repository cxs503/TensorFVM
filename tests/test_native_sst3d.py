import numpy as np
import pytest
from tensorfvm.native_sst3d import sst2003_closure,wall_omega,inflow_from_intensity,NativeSSTTransport3D
from tensorfvm.native_mesh3d import NativeMesh3D


def closure(G,k=.01,omega=2.,d=.001,nu=1e-5,gk=(0,0,0),gw=(0,0,0),rho=1.):
    return sst2003_closure(k,omega,d,nu,np.asarray(G),gk,gw,rho=rho)


def test_simple_shear_matches_independent_sst2003_values():
    G=np.zeros((3,3));G[0,1]=30.
    c=closure(G)
    # Near-wall F2=1; S=30. SST-2003 uses strain, and caps both productions.
    assert c['F1']==pytest.approx(1.) and c['F2']==pytest.approx(1.)
    nut=.31*.01/30
    assert c['strain_squared']==pytest.approx(900.)
    assert c['nu_t']==pytest.approx(nut)
    assert c['production_raw']==pytest.approx(nut*900)
    assert c['production_limited']==pytest.approx(10*.09*2*.01)
    assert c['omega_production']==pytest.approx((5/9)*(.018/nut))
    assert c['k_source']==pytest.approx(.018-.09*2*.01)


def test_pure_strain_and_solid_rotation_discriminate_2003_limiter():
    strain=closure(np.diag([4.,-4.,0]))
    rotation=closure([[0,-4,0],[4,0,0],[0,0,0]])
    assert strain['vorticity_squared']==0 and strain['strain_squared']==64
    assert strain['nu_t']==pytest.approx(.31*.01/8)
    assert rotation['strain_squared']==0
    assert rotation['nu_t']==pytest.approx(.01/2)
    assert rotation['production_limited']==0
    np.testing.assert_allclose(rotation['turbulent_stress'],-2/3*.01*np.eye(3))


def test_exact_production_is_stress_contraction_and_density_correct():
    G=np.array([[1,.2,-.4],[.3,-.7,.2],[.8,.3,.5]])
    c=closure(G,rho=997.)
    assert c['production_raw']==pytest.approx(np.sum(c['turbulent_stress']*G)/997)
    # Exact variant allows negative expansion production; it is not clipped away.
    assert closure(np.eye(3))['production_limited']<0


def test_rotational_objectivity_full_3d_gradients():
    G=np.array([[.2,1,.3],[-.4,-.1,.2],[.1,.5,-.1]])
    Q=np.array([[0,0,1],[1,0,0],[0,1,0]])
    gk=np.array([.1,.2,.3]);gw=np.array([.4,-.2,.1])
    a=closure(G,gk=gk,gw=gw,d=.2);b=closure(Q@G@Q.T,gk=Q@gk,gw=Q@gw,d=.2)
    for name in ['F1','F2','nu_t','production_raw','production_limited','cross_diffusion_source']:
        assert a[name]==pytest.approx(b[name])
    np.testing.assert_allclose(b['turbulent_stress'],Q@a['turbulent_stress']@Q.T)


def test_zero_k_limit_and_bad_states():
    c=closure([[0,3,0],[0,0,0],[0,0,0]],k=0)
    assert c['nu_t']==0 and c['production_limited']==0
    assert np.isfinite(c['omega_production'])
    for kwargs in [{'k':-.1},{'omega':0},{'d':0},{'nu':-1},{'rho':0}]:
        with pytest.raises(ValueError):closure(np.zeros((3,3)),**kwargs)


def test_wall_and_inlet_parameterization_documented_units():
    assert wall_omega(1e-5,.002)==pytest.approx(2000.)
    k,w=inflow_from_intensity(2.,.01,.1)
    assert k==pytest.approx(.0006)
    assert w==pytest.approx(np.sqrt(.0006)/(.09**.25*.1))
    with pytest.raises(ValueError):wall_omega(1e-5,0)


def mesh():
    return NativeMesh3D.from_cells([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],[[0,1,2,3],[0,2,1,4]],['tetra']*2)


def test_real_two_cell_transport_homogeneous_decay_conserves_source():
    m=mesh();op=NativeSSTTransport3D(m,np.zeros(len(m.owner),bool))
    k=np.full(2,.01);w=np.full(2,2.);dt=.01
    r=op.step(k,w,np.zeros(len(m.owner)),np.full(2,.1),1e-5,np.zeros((2,3,3)),dt,k_boundary=.01,omega_boundary=2.)
    np.testing.assert_allclose(r['k'],k/(1+dt*.09*w),atol=2e-15)
    np.testing.assert_allclose(r['omega'],w/(1+dt*r['closure']['beta']*w),atol=2e-15)
    for l in r['ledger'].values():
        assert l['max_local_integrated_residual']<1e-13
        assert abs(l['global_residual'])<1e-13
    assert r['physical_benchmark_passed'] is False


def test_shared_transport_flux_and_boundary_source_ledger():
    m=mesh();op=NativeSSTTransport3D(m,m.neighbor<0)
    # Shared-face advection and prescribed boundary inflow are present; ledger
    # balances actual source, boundary flux and pseudo-time storage.
    q=m.area_vectors@np.array([.3,.1,-.2])
    r=op.step(np.array([.01,.012]),np.array([2.,2.1]),q,np.full(2,.1),1e-5,np.zeros((2,3,3)),1e-4,k_boundary=.01,omega_boundary=2.)
    for l in r['ledger'].values():
        assert abs(l['global_residual'])<1e-11
        assert l['max_local_integrated_residual']<1e-11
