import numpy as np
import pytest
from tensorfvm.native_hybrid_scalar3d import NativeHybridScalar3D
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means
from tests.test_native_hybrid_viscous3d import thin_mixed_mesh
from tests.test_native_rans3d import box


def assert_ledger(r, atol=2e-10):
    assert r['linear_relative_residual'] < 1e-11
    assert r['cell_balance_max'] < atol
    assert r['interface_flux_defect_max'] < atol
    assert abs(r['global_balance']) < atol
    assert abs(r['work_balance']) < atol


def test_affine_diffusion_mixed_boundary_and_independent_flux():
    m = thin_mixed_mesh(); bd = m.neighbor < 0
    mask = bd & (m.face_centers[:,0] < .8)
    gradient = np.array([.3,-.2,.7]); coefficient = 2.4
    values = 2+m.face_centers @ gradient
    flux = coefficient*m.area_vectors @ gradient
    gn = flux/np.linalg.norm(m.area_vectors,axis=1)
    op = NativeHybridScalar3D(m,coefficient,dirichlet_mask=mask)
    r = op.solve(boundary_values=values,boundary_diffusive_flux=gn)
    np.testing.assert_allclose(r['cell_values'],2+m.cell_centers @ gradient,atol=2e-11)
    np.testing.assert_allclose(r['diffusive_flux'],flux,atol=2e-11)
    np.testing.assert_allclose(r['gradient'],np.tile(gradient,(len(m.cells),1)),atol=2e-11)
    assert_ledger(r)


@pytest.mark.parametrize('sign',[-1,1])
def test_neumann_inflow_backflow_constant_and_storage(sign):
    m = box(2); op = NativeHybridScalar3D(m,.3,dirichlet_mask=np.zeros(len(m.owner),bool))
    q = m.area_vectors @ (sign*np.array([1.,-.4,.2]))
    # Source and implicit sink balance at exactly phi=2 despite old phi=3.
    r = op.solve(q,old_values=3.,dt=.5,sink=1.,source=0.,inflow_values=2.)
    np.testing.assert_allclose(r['cell_values'],2.,atol=1e-12)
    np.testing.assert_allclose(r['face_values'],2.,atol=1e-12)
    np.testing.assert_allclose(r['advective_flux'],2*q,atol=1e-12)
    assert_ledger(r)
    with pytest.raises(ValueError,match='inflow'):
        op.solve(q,old_values=3.,dt=.5)


def test_uniform_decay_matches_implicit_ode_on_thin_skew_mesh():
    m = thin_mixed_mesh(); op = NativeHybridScalar3D(m,.1,dirichlet_mask=np.zeros(len(m.owner),bool))
    r = op.solve(old_values=3.,dt=.2,sink=2.)
    np.testing.assert_allclose(r['cell_values'],3/1.4,atol=1e-11)
    np.testing.assert_allclose(r['face_values'],3/1.4,atol=1e-11)
    assert_ledger(r)


def test_independent_variable_diffusion_transport_refines():
    errors=[]
    U=np.array([.6,-.3,.2]); sink=.7
    for n in (2,4,8):
        m=box(n);x=m.cell_centers
        exact=1+np.sum(x*x,axis=1)
        bd=exact_boundary_velocity_means(m,lambda p: {'velocity': np.column_stack((1+np.sum(p*p,axis=1),np.zeros((len(p),2))))})[:,0]
        source=2*x @ U-(6+8*x[:,0])+sink*exact
        r=NativeHybridScalar3D(m,1+x[:,0]).solve(m.area_vectors @ U,
            source=source,sink=sink,boundary_values=bd,inflow_values=bd)
        errors.append(np.sqrt(np.sum(m.cell_volumes*(r['cell_values']-exact)**2)/np.sum(m.cell_volumes*exact**2)))
        assert_ledger(r,2e-9)
    assert errors[2] < errors[1] < errors[0], errors
    assert errors[-1] < .01, errors
    assert np.log2(errors[-2]/errors[-1]) > .8, errors


def test_invalid_and_signed_states_are_not_silently_repaired():
    op=NativeHybridScalar3D(box(1),dirichlet_mask=np.zeros(12,bool))
    with pytest.raises(ValueError,match='together'):op.solve(dt=.1)
    with pytest.raises(ValueError,match='nonnegative'):op.solve(sink=-1.)
    with pytest.raises(ValueError,match='positive'):op.solve(dt=0,old_values=1.)
    r=op.solve(dt=.1,old_values=0.,source=-1.)
    np.testing.assert_allclose(r['cell_values'],-.1,atol=1e-13)
    assert_ledger(r)
