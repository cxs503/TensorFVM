"""Core mesh contract (MeshData) and the five mesh-class adapters."""
import numpy as np
import pytest
import torch

from tensorfvm.core.mesh import (
    MeshData,
    adapt_mesh,
    from_body_fitted3d,
    from_mesh2d,
    from_native_mesh3d,
    from_overset_grid,
    from_polymesh,
    validate_mesh_data,
)
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.openfoam_polymesh import (read_polymesh, structured_box_primitives,
                                         write_polymesh)
from tensorfvm.overset import cartesian_grid


def _two_tet_mesh():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
    return NativeMesh3D.from_cells(vertices, [[0, 1, 2, 3], [0, 2, 1, 4]], ["tetra"] * 2)


class TestMeshDataValidator:
    def test_valid_3d_mesh_passes_with_closure(self):
        mesh = validate_mesh_data(from_native_mesh3d(_two_tet_mesh()), check_closure=True)
        assert mesh.n_cells == 2
        assert mesh.dim == 3

    def test_bad_dim_rejected(self):
        mesh = from_native_mesh3d(_two_tet_mesh())
        bad = MeshData(**{**mesh.__dict__, "dim": 4})
        with pytest.raises(ValueError, match="dim must be 2 or 3"):
            validate_mesh_data(bad)

    def test_nonpositive_volume_rejected(self):
        mesh = from_native_mesh3d(_two_tet_mesh())
        bad = MeshData(**{**mesh.__dict__,
                          "cell_volumes": np.array([mesh.cell_volumes[0], -1.0])})
        with pytest.raises(ValueError, match="strictly positive"):
            validate_mesh_data(bad)

    def test_mask_on_interior_face_rejected(self):
        mesh = from_native_mesh3d(_two_tet_mesh())
        interior = mesh.interior
        bad = MeshData(**{**mesh.__dict__, "masks": {"bad": interior}})
        with pytest.raises(ValueError, match="only mark boundary faces"):
            validate_mesh_data(bad)

    def test_out_of_range_neighbor_rejected(self):
        mesh = from_native_mesh3d(_two_tet_mesh())
        neighbor = mesh.neighbor.copy()
        neighbor[0] = mesh.n_cells
        bad = MeshData(**{**mesh.__dict__, "neighbor": neighbor})
        with pytest.raises(ValueError, match="neighbor indices"):
            validate_mesh_data(bad)


class TestMeshAdapters:
    def test_native_mesh3d(self):
        mesh = from_native_mesh3d(_two_tet_mesh())
        assert mesh.n_faces == len(_two_tet_mesh().owner)
        assert "internal" in set(mesh.patch_labels)
        assert set(mesh.masks) == set(l for l in np.unique(mesh.patch_labels) if l != "internal")

    def test_polymesh(self, tmp_path):
        pts, faces, owner, neighbour, patches = structured_box_primitives(
            4, 4, 4, size=(1.0, 1.0, 1.0))
        write_polymesh(tmp_path / "case", pts, faces, owner, neighbour, patches)
        pm = read_polymesh(tmp_path / "case")
        mesh = from_polymesh(pm)
        validate_mesh_data(mesh, check_closure=True)
        assert mesh.n_cells == 64
        assert (mesh.boundary.sum() > 0) and (mesh.interior.sum() > 0)

    def test_overset_grid(self):
        grid = cartesian_grid((0.0, 1.0, 0.0, 1.0), 4, 4)
        mesh = from_overset_grid(grid)
        validate_mesh_data(mesh, check_closure=True)
        assert mesh.n_cells == 16
        # 4x4 grid: 24 interior shared edges + 16 boundary edges = 40 faces
        assert mesh.n_faces == 40
        assert "outer" in mesh.masks

    def test_mesh2d_seed(self):
        from tensorfvm.body_fitted import BodyFittedMesh
        from tensorfvm.solver import SolverConfig
        mesh2d = BodyFittedMesh(SolverConfig(nx=8, ny=4, mesh_type="body-fitted"))
        mesh = from_mesh2d(mesh2d)
        validate_mesh_data(mesh)
        assert mesh.dim == 2
        assert mesh.n_cells == 32
        assert np.allclose(mesh.area_vectors, _np(mesh2d.face_area_vectors))

    def test_body_fitted3d(self):
        import torch
        from tensorfvm.body_fitted3d import BodyFittedCylinderMesh3D
        from tensorfvm.solver3d import Cylinder3DConfig
        config = Cylinder3DConfig(nx=8, ny=4, nz=4, mesh_type="body-fitted",
                              body_fitted_stretching=2.0)
        slab = BodyFittedCylinderMesh3D.build(config, torch.device("cpu"))
        mesh = from_body_fitted3d(slab)
        validate_mesh_data(mesh)
        assert mesh.n_cells == 8 * 4 * 4
        assert "spanwise_lower" in mesh.masks and "spanwise_upper" in mesh.masks
        # spanwise face areas equal the cross-section cell volumes
        sw = mesh.masks["spanwise_lower"]
        assert np.allclose(mesh.face_lengths[sw],
                            _np(slab.cell_volumes).reshape(-1))

    def test_adapt_mesh_dispatch(self):
        mesh = adapt_mesh(_two_tet_mesh())
        assert isinstance(mesh, MeshData)
        with pytest.raises(TypeError, match="no MeshData adapter"):
            adapt_mesh(object())


def _np(t):
    return t.detach().cpu().numpy()
