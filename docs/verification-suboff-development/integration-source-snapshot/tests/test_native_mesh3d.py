import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_operators3d import NativeOperators3D

@pytest.mark.parametrize('kind,points,volume',[
 ('tetra',[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],1/6),
 ('hexahedron',[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],1),
 ('wedge',[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,0,1],[0,1,1]],.5),
 ('pyramid',[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,1]],1/3),
])
def test_linear_cells_have_real_volume_and_affine_response(kind,points,volume):
    # Shearing coordinates prevents Cartesian-only gradient formulas passing.
    transform=np.array([[1,.21,.17],[0,1,.32],[0,0,1]])
    m=NativeMesh3D.from_cells(np.asarray(points)@transform.T,[range(len(points))],[kind])
    assert m.cell_volumes[0]==pytest.approx(volume)
    assert m.audit()['maximum_relative_closure']<1e-14
    a=NativeOperators3D(m).audit_affine()
    assert max(a.values())<1e-12

def test_shared_faces_cancel_independent_of_field():
    vertices=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]]
    m=NativeMesh3D.from_cells(vertices,[[0,1,2,3],[0,2,1,4]],['tetra']*2)
    op=NativeOperators3D(m); assert op.internal.sum()==1
    values=np.zeros((len(m.owner),3));values[op.internal]=[4,-7,2]
    result=op.integrate(values)
    np.testing.assert_allclose(result.sum(axis=0),0,atol=0)
    np.testing.assert_allclose(result[0],-result[1],atol=0)

def test_exact_boundary_tags_required_and_unsupported_rejected():
    vertices=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    with pytest.raises(ValueError,match='untagged'):
        NativeMesh3D.from_cells(vertices,[[0,1,2,3]],['tetra'],{})
    with pytest.raises(ValueError,match='unsupported'):
        NativeMesh3D.from_cells(vertices,[[0,1,2,3]],['tetra10'])
    with pytest.raises(ValueError,match='nonconvex|degenerate'):
        NativeMesh3D.from_cells([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],[[0,1,2,3]],['tetra'])

def test_nonplanar_hex_is_explicitly_rejected():
    vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1.2],[0,1,1]]
    with pytest.raises(ValueError,match='nonplanar'):
        NativeMesh3D.from_cells(vertices,[range(8)],['hexahedron'])

def test_gmsh_volume_and_physical_patches(tmp_path):
    gmsh=pytest.importorskip('gmsh');gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
    try:
        box=gmsh.model.occ.addBox(0,0,0,1,1,1);gmsh.model.occ.synchronize()
        surfaces=[tag for dim,tag in gmsh.model.getBoundary([(3,box)],oriented=False)]
        patch=gmsh.model.addPhysicalGroup(2,surfaces);gmsh.model.setPhysicalName(2,patch,'outer')
        gmsh.model.addPhysicalGroup(3,[box]);gmsh.option.setNumber('Mesh.MeshSizeMax',.7);gmsh.model.mesh.generate(3)
        path=tmp_path/'cube.msh';gmsh.write(str(path))
    finally:gmsh.finalize()
    m=NativeMesh3D.from_gmsh(path)
    assert set(m.patch_labels)=={'internal','outer'}
    assert m.cell_volumes.sum()==pytest.approx(1)
    assert max(NativeOperators3D(m).audit_affine().values())<1e-10
