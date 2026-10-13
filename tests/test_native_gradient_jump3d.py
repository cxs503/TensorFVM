import numpy as np
import pytest
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.verification.native_hybrid_pressure import pressure_solver
from tests.test_native_hybrid_viscous3d import thin_mixed_mesh


@pytest.mark.parametrize('factory',[two_tetrahedra,thin_mixed_mesh,lambda:channel_mesh(2,triangulate_faces=False)])
def test_jump_energy_and_independent_traction_replay(factory):
    mesh=factory();op=NativeHybridViscous3D(mesh)
    state=np.random.default_rng(1729).normal(size=op.full_matrix.shape[0])
    r=op.diagnostics(state)
    assert r['gradient_jump_energy']>0
    np.testing.assert_allclose(r['gradient_jump_energy'],state@op.gradient_jump.matrix@state,rtol=1e-11,atol=1e-10)
    np.testing.assert_allclose(r['energy_form'],state@op.full_matrix@state,rtol=1e-11,atol=1e-10)
    force=(op.full_matrix@state).reshape(-1,3)[op.N:]
    np.testing.assert_allclose(r['face_flux']+r['neighbor_face_flux'],force,rtol=1e-10,atol=1e-9)
    # Cell+face common translation is a null mode; no artificial net force.
    np.testing.assert_allclose((op.full_matrix@state).reshape(-1,3).sum(0),0,atol=1e-9)
    normal=mesh.area_vectors[op.gradient_jump.internal]
    np.testing.assert_allclose(np.einsum('fij,fj->fi',op.gradient_jump.covariance,normal),0,atol=1e-14)


def test_jump_preserves_arbitrary_affine_gradient_and_rigid_rotation():
    mesh=channel_mesh(2);op=NativeHybridViscous3D(mesh)
    g=np.array([[.1,-.7,.2],[.7,-.3,.5],[-.2,-.5,.2]])
    points=np.vstack((mesh.cell_centers,mesh.face_centers));state=(points@g.T+.3).ravel()
    response,energy=op.gradient_jump.gradient_reactions(state)
    assert energy<1e-25
    np.testing.assert_allclose(response,0,atol=1e-14)


def test_pressure_shear_jump_fix_reduces_error_and_preserves_force_energy():
    errors=[]
    for strength in (0.,1.):
        m,s,e=pressure_solver(3,kind='shear',stabilization=strength)
        r=s.solve(algorithm='coupled',tolerance=1e-10)
        assert r['numerically_converged']
        assert r['energy_balance_error']<1e-9
        assert r['interface_traction_defect_max']<1e-9
        np.testing.assert_allclose(r['total_traction_integrated'][m.neighbor<0].sum(0),
            -(s.body_force*m.cell_volumes[:,None]).sum(0),atol=1e-9)
        pressure=r['pressure']-np.average(r['pressure'],weights=m.cell_volumes)
        errors.append(np.sqrt(np.average(pressure**2,weights=m.cell_volumes)))
    assert errors[1]<.6*errors[0],errors
