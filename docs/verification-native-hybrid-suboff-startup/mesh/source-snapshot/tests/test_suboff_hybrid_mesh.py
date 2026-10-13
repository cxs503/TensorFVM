import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_operators3d import NativeOperators3D
from tensorfvm.suboff_hybrid_mesh import structured_layers,generate_suboff_hybrid_mesh


def test_warped_hex_facets_are_affine_exact_and_share_identical_diagonal():
    # Two cells share a genuinely warped quad; a mismatched diagonal leaves a hole.
    vertices=np.array([[x,y,z] for x in [0,1,2] for y,z in [(0,0),(1,0),(1,1),(0,1)]],float)
    vertices[6,0]+=.12
    cells=[(0,4,5,1,3,7,6,2),(4,8,9,5,7,11,10,6)]
    m=NativeMesh3D.from_cells(vertices,cells,['hexahedron']*2,triangulate_faces=True)
    assert np.sum(m.neighbor>=0)==2
    assert m.cell_volumes.sum()==pytest.approx(2)
    assert m.audit()['maximum_relative_closure']<1e-14
    assert max(NativeOperators3D(m).audit_affine().values())<1e-12
    op=NativeOperators3D(m);f=np.zeros(len(m.owner));f[op.internal]=[.3,-1.7]
    assert op.integrate(f).sum()==0


def test_structured_normal_layers_close_caps_and_preserve_requested_spacing():
    p,c,t,wall,interface,s=structured_layers(axial=24,azimuthal=12,layers=3,first_height=.0003,growth=1.2)
    m=NativeMesh3D.from_cells(p,c,t,triangulate_faces=True)
    # Each layer has the same vertex ordering, including a single node at each cap.
    n=len(p)//4;displacement=np.linalg.norm(p[n:2*n]-p[:n],axis=1)
    np.testing.assert_allclose(displacement,.0003,rtol=1e-10,atol=1e-15)
    assert t.count('wedge')==2*12*3
    assert m.cell_volumes.min()>0
    assert len(wall)==len(interface)
    with pytest.raises(ValueError):structured_layers(first_height=.1)
    with pytest.raises(ValueError):structured_layers(growth=float('nan'))


@pytest.mark.parametrize('tail_size', [None, .01])
def test_actual_gmsh_exterior_shares_all_structured_facets(tmp_path, tail_size):
    pytest.importorskip('gmsh')
    m,s=generate_suboff_hybrid_mesh(tmp_path/'hybrid.msh',axial=12,azimuthal=8,layers=2,far_size=.6,tail_size=tail_size)
    if tail_size is not None:
        assert s['tail_refinement']['size_L']==tail_size
        # The background field must not move even one imposed structured node.
        from scipy.spatial import cKDTree
        points,*_=structured_layers(axial=12,azimuthal=8,layers=2)
        distance,_=cKDTree(m.vertices).query(points)
        assert distance.max()<1e-13
    assert s['exterior_tetrahedra']>0 and s['structured_hexes']>0
    assert s['verified_shared_interface_faces']==s['interface_triangles']
    assert set(m.patch_labels)=={'internal','wall','inlet','outlet','farfield'}
    assert m.audit()['maximum_relative_closure']<1e-12
    assert max(NativeOperators3D(m).audit_affine().values())<1e-8
    # The generic mixed-BC scalar projection also consumes faceted mixed cells.
    from tensorfvm.native_projection3d import NativeProjection3D
    projection=NativeProjection3D(m,m.patch_labels=='outlet')
    g=np.array([.7,-.3,.2]);exact=1.2+m.cell_centers@g
    boundary_scalar=1.2+m.face_centers@g;normal_derivative=projection.normal@g
    potential,remainder=projection.project(m.area_vectors@g,normal_derivative,boundary_scalar)
    np.testing.assert_allclose(potential,exact,atol=1e-8,rtol=0)
    np.testing.assert_allclose(remainder,0,atol=1e-9,rtol=0)
    # Round-trip records actual hexes/wedges, not a tetrahedral substitute.
    again=NativeMesh3D.from_gmsh(tmp_path/'hybrid.msh',triangulate_faces=True)
    np.testing.assert_array_equal(m.owner,again.owner)
    np.testing.assert_allclose(m.cell_volumes,again.cell_volumes,atol=0,rtol=0)
