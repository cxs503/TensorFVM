import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_projection3d import NativeProjection3D


def mesh():
    # Two tetrahedra share a face; transform gives genuinely skew stencils.
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],dtype=float)
    vertices=vertices@np.array([[1,.3,.2],[0,1,.15],[0,0,1]]).T
    return NativeMesh3D.from_cells(vertices,[[0,1,2,3],[0,2,1,4]],['tetra']*2)


def test_mixed_boundary_affine_flux_and_projection_are_exact():
    m=mesh();boundary=m.neighbor<0;dirichlet=np.zeros(len(boundary),dtype=bool);dirichlet[np.flatnonzero(boundary)[0]]=True
    op=NativeProjection3D(m,dirichlet)
    gradient=np.array([.7,-.3,.2]);scalar=1.2+m.cell_centers@gradient
    target_faces=1.2+m.face_centers@gradient;neumann=op.normal@gradient
    np.testing.assert_allclose(op.gradient(scalar,neumann,target_faces),np.tile(gradient,(len(m.cells),1)),atol=2e-14)
    np.testing.assert_allclose(op.scalar_face_values(scalar,neumann,target_faces),target_faces,atol=2e-14)
    np.testing.assert_allclose(op.op.gauss_gradient(op.scalar_face_values(scalar,neumann,target_faces)),np.tile(gradient,(len(m.cells),1)),atol=3e-14)
    target_flux=m.area_vectors@gradient
    np.testing.assert_allclose(op.scalar_flux(scalar,neumann,target_faces),target_flux,atol=2e-14)
    potential,flux=op.project(target_flux,neumann,target_faces)
    np.testing.assert_allclose(potential,scalar,atol=3e-14)
    np.testing.assert_allclose(flux,0,atol=3e-14)


def test_projection_closes_random_native_flux_and_preserves_neumann_response():
    m=mesh();boundary=m.neighbor<0;dirichlet=boundary.copy();dirichlet[np.flatnonzero(boundary)[::2]]=False
    op=NativeProjection3D(m,dirichlet);rng=np.random.default_rng(31);q=rng.normal(size=len(boundary));g=rng.normal(size=len(boundary))
    potential,flux=op.project(q,g)
    np.testing.assert_allclose(op.op.integrate(flux),0,atol=2e-14)
    b=op.neumann
    np.testing.assert_allclose(flux[b],q[b]-g[b]*np.linalg.norm(m.area_vectors[b],axis=1),atol=2e-14)


def test_unanchored_or_internal_dirichlet_and_nonfinite_data_rejected():
    m=mesh();boundary=m.neighbor<0
    with pytest.raises(ValueError,match='anchored'):NativeProjection3D(m,np.zeros_like(boundary))
    with pytest.raises(ValueError,match='physical'):NativeProjection3D(m,np.ones_like(boundary))
    op=NativeProjection3D(m,boundary)
    with pytest.raises(ValueError,match='finite'):op.project(np.full(len(boundary),np.nan))
