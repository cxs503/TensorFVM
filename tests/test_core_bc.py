"""Core BC registry: three adapters over the existing boundary modules."""
import numpy as np
import pytest

from tensorfvm.core.bc import BoundaryCondition, get_bc, list_bc_types
from tensorfvm.free_surface_bc import FreeSurfaceBC


def _box_mesh():
    """4x4x4 hex box as a NativeMesh3D (enough neighbours for the stencils)."""
    import pathlib
    import tempfile
    from tensorfvm.openfoam_polymesh import (read_polymesh,
                                             structured_box_primitives,
                                             to_native_mesh3d, write_polymesh)
    pts, faces, owner, neighbour, patches = structured_box_primitives(
        4, 4, 4, size=(1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        case = pathlib.Path(tmp) / "box"
        write_polymesh(case, pts, faces, owner, neighbour, patches)
        return to_native_mesh3d(read_polymesh(case))


def test_three_bc_types_registered():
    assert set(list_bc_types()) == {"free_surface_open", "pressure_trace",
                                    "rigid_motion_constraint"}


def test_unknown_bc_type_raises():
    with pytest.raises(KeyError):
        get_bc("no_such_bc")


def test_free_surface_open_adapter():
    bc = FreeSurfaceBC(ny=4, nx=8, dx=0.1, dy=0.1)
    adapter = get_bc("free_surface_open", bc)
    assert isinstance(adapter, BoundaryCondition)
    import torch
    p = torch.ones(4, 8)
    u = torch.ones(4, 9)
    v = torch.ones(5, 8)
    p_new, u_new, v_new = adapter.apply(p, u, v)
    assert p_new[-1].sum() == 0.0            # top row is atmospheric
    assert v_new[0].sum() == 0.0             # bottom wall no-slip
    assert torch.allclose(v_new[-1], v[-2])  # top: zero gradient


def test_pressure_trace_adapter():
    from tensorfvm.native_boundary_pressure3d import NativeBoundaryPressureTrace3D
    mesh = _box_mesh()
    trace = NativeBoundaryPressureTrace3D(mesh, mesh.neighbor < 0)
    adapter = get_bc("pressure_trace", trace)
    result = adapter.apply(np.arange(len(mesh.cells), dtype=float))
    assert result["pressure"].shape == (int((mesh.neighbor < 0).sum()),)
    assert result["changes_momentum_equations"] is False


def test_rigid_motion_constraint_adapter():
    mesh = _box_mesh()
    dirichlet = np.zeros((len(mesh.owner), 3), dtype=bool)
    dirichlet[mesh.neighbor < 0] = True
    adapter = get_bc("rigid_motion_constraint", mesh)
    result = adapter.apply(dirichlet)
    assert result["rank"] == 6
    assert result["necessary_uniqueness_condition_passed"] is True
