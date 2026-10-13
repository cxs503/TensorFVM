import numpy as np
import pytest
from tensorfvm.native_hybrid_sst3d import NativeHybridSSTTransport3D
from tests.test_native_hybrid_viscous3d import thin_mixed_mesh
from tests.test_native_hybrid_scalar3d import assert_ledger


def step(gradient, *, k=.1, omega=2., k_neumann=0., dt=.1):
    m=thin_mixed_mesh(); mask=np.zeros(len(m.owner),bool)
    op=NativeHybridSSTTransport3D(m,k_dirichlet_mask=mask,omega_dirichlet_mask=mask)
    return op.step(k,omega,k,omega,0.,.01,1e-5,gradient,dt,
                   k_boundary=0.,omega_boundary=2.,k_neumann=k_neumann)


def test_uniform_decay_matches_independent_local_ode():
    r=step(np.zeros((3,3)))
    np.testing.assert_allclose(r['k'],.1/(1+.1*.09*2),rtol=2e-12)
    np.testing.assert_allclose(r['omega'],2/(1+.1*.075*2),rtol=2e-12)
    for t in r['transport'].values():assert_ledger(t)
    assert not r['nonlinear_rans_converged'] and not r['physical_benchmark_passed']


def test_uniform_shear_production_and_implicit_destruction():
    G=np.zeros((3,3));G[0,1]=.1
    r=step(G)
    # F1,F2 saturate to 1; a1*omega dominates strain limiter.
    np.testing.assert_allclose(r['k'],(.1+.1*(.1/2)*.1**2)/(1+.1*.09*2),rtol=2e-12)
    np.testing.assert_allclose(r['omega'],(2+.1*(5/9)*.1**2)/(1+.1*.075*2),rtol=2e-12)
    for t in r['transport'].values():assert_ledger(t)


def test_negative_exact_production_is_preserved_as_sink():
    # Local frozen-source test: isotropic dilatation tau:S=-2*k*a.
    # This does not claim an incompressible momentum solution.
    r=step(.1*np.eye(3))
    np.testing.assert_allclose(r['transport']['k']['negative_production_sink'],.2,rtol=1e-12)
    np.testing.assert_allclose(r['k'],.1/(1+.1*(.09*2+.2)),rtol=2e-12)
    np.testing.assert_allclose(r['omega'],2/(1+.1*(.075*2+2*.1*5/9)),rtol=2e-12)
    for t in r['transport'].values():assert_ledger(t)


def test_zero_k_stays_zero_without_floor_and_invalid_update_rejected():
    r=step(np.zeros((3,3)),k=0.)
    np.testing.assert_array_equal(r['k'],0.)
    np.testing.assert_array_equal(r['k_faces'],0.)
    with pytest.raises(ValueError,match='positivity'):step(np.zeros((3,3)),omega=0.)
    with pytest.raises(RuntimeError,match='no clipping'):step(np.zeros((3,3)),k_neumann=-100.,dt=1.)


def test_nonuniform_shared_faces_drive_cross_diffusion_and_conservation():
    m=thin_mixed_mesh();x=m.cell_centers;xf=m.face_centers
    k=.2+.01*x[:,0];w=2-.02*x[:,0]
    kf=.2+.01*xf[:,0];wf=2-.02*xf[:,0]
    op=NativeHybridSSTTransport3D(m)
    q=m.area_vectors @ np.array([.1,-.03,.02])
    r=op.step(k,w,kf,wf,q,100.,1e-5,np.zeros((3,3)),.001,
              k_boundary=kf,omega_boundary=wf,k_inflow=kf,omega_inflow=wf)
    # Independent affine gradients have negative dot product -0.0002.
    closure=r['closure']
    exact=2*(1-closure['F1'])*.856*(-.0002)/w
    np.testing.assert_allclose(closure['cross_diffusion_source'],exact,rtol=1e-10)
    assert np.all(closure['cross_diffusion_source']<0)
    assert np.any(r['transport']['omega']['negative_production_sink']>0)
    for t in r['transport'].values():assert_ledger(t)
    with pytest.raises(ValueError,match='face k'):
        op.step(k,w,-kf,wf,q,100.,1e-5,np.zeros((3,3)),.001,
                k_boundary=kf,omega_boundary=wf,k_inflow=kf,omega_inflow=wf)
