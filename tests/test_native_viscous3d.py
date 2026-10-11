import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D


def skew_mesh():
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],dtype=float)
    vertices=vertices@np.array([[1,.3,.2],[0,1,.15],[0,0,1]]).T
    return NativeMesh3D.from_cells(vertices,[[0,1,2,3],[0,2,1,4]],['tetra']*2)


def mixed_mask(mesh):
    b=mesh.neighbor<0;mask=b.copy();mask[np.flatnonzero(b)[::2]]=False;return mask


def test_scalar_affine_mixed_boundary_flux_and_sources():
    m=skew_mesh();op=NativeViscous3D(m,mixed_mask(m),2.3)
    gradient=np.array([.7,-.3,.2]);values=1.2+m.cell_centers@gradient
    bd=1.2+m.face_centers@gradient;gn=op.normal@gradient
    computed=op.scalar_flux(values,gn,bd)
    np.testing.assert_allclose(computed,2.3*m.area_vectors@gradient,atol=2e-14)
    np.testing.assert_allclose(op.divergence_matrix@computed,0,atol=2e-14)
    assembled=op.scalar_diffusion_matrix@values+op.divergence_matrix@op.scalar_flux_source(gn,bd)
    np.testing.assert_allclose(assembled,0,atol=2e-14)
    np.testing.assert_allclose(computed[op.neumann],2.3*gn[op.neumann]*op.area[op.neumann],atol=2e-14)


def test_variable_mu_full_symmetric_affine_stress_and_divergence():
    m=skew_mesh();mu_gradient=np.array([.1,-.2,.3]);mu=2+m.face_centers@mu_gradient
    op=NativeViscous3D(m,mixed_mask(m),mu)
    G=np.array([[.7,.2,-.1],[-.3,.4,.6],[.15,-.25,.1]])
    base=np.array([1.2,.8,-.4]);u=base+m.cell_centers@G.T;bd=base+m.face_centers@G.T;gn=op.normal@G.T
    tau=G+G.T-2/3*np.trace(G)*np.eye(3)
    expected=mu[:,None]*(m.area_vectors@tau.T)
    actual=op.momentum_flux(u,gn,bd)
    np.testing.assert_allclose(actual,expected,atol=4e-14)
    # Analytic div[mu tau] = tau grad(mu), not a source produced by this operator.
    force=(op.vector_divergence_matrix@actual.ravel()).reshape(op.N,3)
    np.testing.assert_allclose(force/m.cell_volumes[:,None],np.tile(tau@mu_gradient,(op.N,1)),atol=2e-13)
    np.testing.assert_allclose(force.sum(0),actual[op.boundary].sum(0),atol=2e-14)
    assembled=(op.momentum_diffusion_matrix@u.ravel()).reshape(op.N,3)+(op.vector_divergence_matrix@op.momentum_flux_source(gn,bd).ravel()).reshape(op.N,3)
    np.testing.assert_allclose(assembled,force,atol=2e-14)


def test_solid_rotation_has_zero_symmetric_stress():
    m=skew_mesh();op=NativeViscous3D(m,m.neighbor<0)
    G=np.array([[0,-2,1],[2,0,-.3],[-1,.3,0]])
    np.testing.assert_allclose(op.momentum_flux(m.cell_centers@G.T,0,m.face_centers@G.T),0,atol=2e-14)


def test_zero_normal_gradient_is_not_zero_traction():
    m=skew_mesh();mask=m.neighbor<0;face=np.flatnonzero(mask)[0];mask[face]=False
    op=NativeViscous3D(m,mask)
    normal=op.normal[face];tangent=np.cross(normal,[0,0,1.])
    if np.linalg.norm(tangent)<.1:tangent=np.cross(normal,[0,1.,0])
    tangent/=np.linalg.norm(tangent)
    G=np.outer(normal,tangent)  # G n=0, but G^T n=tangent.
    u=m.cell_centers@G.T;bd=m.face_centers@G.T;gn=op.normal@G.T
    assert np.linalg.norm(gn[face])<1e-14
    traction=op.momentum_flux(u,gn,bd)
    np.testing.assert_allclose(traction[face]/op.area[face],tangent,atol=2e-14)


def test_warped_mixed_hex_facets_have_affine_response():
    v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1.2],[0,1,1]],float)
    m=NativeMesh3D.from_cells(v,[range(8)],['hexahedron'],triangulate_faces=True)
    assert len(m.owner)==12 and all(len(f)==3 for f in m.face_vertices)
    op=NativeViscous3D(m,m.neighbor<0)
    G=np.array([[1,.2,.3],[.4,2,.1],[.3,.8,-3.]])
    tau=G+G.T-2/3*np.trace(G)*np.eye(3)
    np.testing.assert_allclose(op.momentum_flux(m.cell_centers@G.T,0,m.face_centers@G.T),m.area_vectors@tau.T,atol=3e-14)


def test_invalid_viscosity_masks_and_data_rejected():
    m=skew_mesh()
    with pytest.raises(ValueError,match='positive'):NativeViscous3D(m,m.neighbor<0,-1)
    with pytest.raises(ValueError,match='boundary'):NativeViscous3D(m,np.ones(len(m.owner),bool))
    op=NativeViscous3D(m,m.neighbor<0)
    with pytest.raises(ValueError,match='finite'):op.momentum_flux(np.full((len(m.cells),3),np.nan))
