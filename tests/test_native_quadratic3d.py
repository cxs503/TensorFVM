import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_quadratic3d import NativeQuadratic3D


def mesh(n=3, thin=False):
    points=np.array([[x,y,z] for z in np.linspace(0,1,n+1) for y in np.linspace(0,1,n+1) for x in np.linspace(0,1,n+1)])
    x,y,z=points.T.copy();points+=np.column_stack((.06*y*z,.04*x*z,.08*x*y))
    points=points@np.array([[1,.18,.12],[0,1,.2],[0,0,1]]).T
    if thin:
        rotation=np.linalg.qr(np.array([[1,.4,.2],[.3,1,.5],[.2,.6,1]]))[0]
        points=points@np.diag([1,.001,.7])@rotation.T
    def ix(x,y,z):return z*(n+1)**2+y*(n+1)+x
    cells=[]
    for z in range(n):
        for y in range(n):
            for x in range(n):cells.append([ix(x,y,z),ix(x+1,y,z),ix(x+1,y+1,z),ix(x,y+1,z),ix(x,y,z+1),ix(x+1,y,z+1),ix(x+1,y+1,z+1),ix(x,y+1,z+1)])
    return NativeMesh3D.from_cells(points,cells,['hexahedron']*len(cells),triangulate_faces=True)


def polynomial(points):
    g=np.array([.7,-.2,.3]);H=np.array([[.8,.2,-.1],[.2,-.6,.4],[-.1,.4,1.2]])
    return 1.3+points@g+.5*np.einsum('ci,ij,cj->c',points,H,points),g+points@H,H


@pytest.mark.parametrize('boundary_kind',['all_dirichlet','mixed','free_pressure'])
def test_quadratic_cell_derivatives_and_shared_flux_area_mean(boundary_kind):
    m=mesh();b=m.neighbor<0;mask=b.copy()
    if boundary_kind=='mixed':mask[np.flatnonzero(b)[::2]]=False
    if boundary_kind=='free_pressure':mask[:]=False
    op=NativeQuadratic3D(m,mask);p,g,H=polynomial(m.cell_centers);pf,gf,_=polynomial(m.face_centers)
    np.testing.assert_allclose(op.gradient(p,pf),g,atol=2e-12)
    np.testing.assert_allclose(op.hessian(p,pf),np.broadcast_to(H,(len(p),3,3)),atol=1e-11)
    np.testing.assert_allclose(op.face_mean(p,pf),pf+.5*np.einsum('ij,fij->f',H,op.face_covariance),atol=2e-12)
    np.testing.assert_allclose(op.face_gradient(p,pf),gf,atol=3e-12)
    np.testing.assert_allclose(op.face_gradient(p,pf,cell_gradients=g),gf,atol=3e-12)
    q=op.compact_normal_flux(p,pf);np.testing.assert_allclose(q,np.einsum('fi,fi->f',gf,m.area_vectors),atol=2e-12)
    residual=np.zeros(len(p));np.add.at(residual,m.owner,q);i=m.neighbor>=0;np.add.at(residual,m.neighbor[i],-q[i])
    np.testing.assert_allclose(residual,np.trace(H)*m.cell_volumes,atol=2e-12)
    np.testing.assert_allclose(residual.sum(),q[b].sum(),atol=2e-14)
    # Free pressure has nonzero wall normal derivative, and is shift invariant.
    if boundary_kind=='free_pressure':
        assert np.max(abs(q[b]))>.01
        np.testing.assert_allclose(op.face_mean(p+5)-op.face_mean(p),5,atol=3e-12)
        assert sum(r.nnz for r in op.boundary_response)==0


def test_all_affine_and_quadratic_monomials_vector_fields_and_sparse_sources():
    m=mesh();op=NativeQuadratic3D(m,m.neighbor<0)
    def basis(x):
        a,b,c=x.T;return np.column_stack((np.ones(len(x)),a,b,c,a*a,b*b,c*c,a*b,a*c,b*c))
    values=basis(m.cell_centers);boundary=basis(m.face_centers)
    G=np.zeros((len(values),10,3));H=np.zeros((len(values),10,3,3));G[:,1:4]=np.eye(3)
    for a in range(3):G[:,4+a,a]=2*m.cell_centers[:,a];H[:,4+a,a,a]=2
    for col,(a,b) in enumerate([(0,1),(0,2),(1,2)],7):G[:,col,a]=m.cell_centers[:,b];G[:,col,b]=m.cell_centers[:,a];H[:,col,a,b]=H[:,col,b,a]=1
    np.testing.assert_allclose(op.gradient(values,boundary),G,atol=2e-12)
    np.testing.assert_allclose(op.hessian(values,boundary),H,atol=1e-11)
    np.testing.assert_allclose(np.stack([r@values for r in op.gradient_matrices],axis=-1)+op.gradient_source(boundary),G,atol=2e-12)
    expected=basis(m.face_centers)+.5*np.einsum('fkij,fij->fk',H[m.owner],op.face_covariance)
    np.testing.assert_allclose(op.face_mean(values,boundary),expected,atol=2e-12)
    assert all(r.format=='csr' for r in op.gradient_matrices)
    assert op.stencil_diagnostics['normalized_condition_max']<1e8


def test_centroid_dirichlet_sample_is_distinct_from_prescribed_face_mean():
    m=mesh();b=m.neighbor<0;op=NativeQuadratic3D(m,b);p,_,_=polynomial(m.cell_centers);bp,_,H=polynomial(m.face_centers)
    means=op.face_mean(p,bp);assert np.max(abs(means[b]-bp[b]))>1e-4
    fixed=op.face_mean(p,bp,boundary_face_means=bp)
    np.testing.assert_allclose(fixed[b],bp[b],atol=0)
    np.testing.assert_array_equal(fixed[~b],means[~b])
    zeros=op.face_mean(p,bp,boundary_face_means=0.)
    np.testing.assert_array_equal(zeros[b],0.)
    # Explicit mean override has not changed the point-value reconstruction.
    np.testing.assert_allclose(op.hessian(p,bp),np.broadcast_to(H,(len(p),3,3)),atol=1e-11)


def test_directional_whitening_of_rotated_thin_warped_cells():
    m=mesh(thin=True);op=NativeQuadratic3D(m,m.neighbor<0);g=np.array([.1,-.2,.3]);p=m.cell_centers@g;bp=m.face_centers@g
    np.testing.assert_allclose(op.gradient(p,bp),np.broadcast_to(g,(len(p),3)),atol=1e-8)
    assert op.stencil_diagnostics['directional_scale_ratio_max']>100
    assert op.stencil_diagnostics['normalized_condition_max']<100
    # Physical-coordinate curvature remains identifiable under rotated anisotropy.
    p,grad,H=polynomial(m.cell_centers);bp,_,_=polynomial(m.face_centers)
    np.testing.assert_allclose(op.gradient(p,bp),grad,atol=3e-8)
    np.testing.assert_allclose(op.hessian(p,bp),np.broadcast_to(H,(len(p),3,3)),atol=1e-5)


def test_refuse_unidentifiable_free_pressure_and_invalid_geometry_contract():
    m=mesh(n=1)
    with pytest.raises(ValueError,match='full-rank'):NativeQuadratic3D(m)
    with pytest.raises(ValueError,match='Dirichlet mask'):NativeQuadratic3D(m,np.ones(len(m.owner)+1,bool))
    with pytest.raises(ValueError,match='condition'):NativeQuadratic3D(mesh(),condition_limit=1.)
