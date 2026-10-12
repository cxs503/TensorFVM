from pathlib import Path
import re

import numpy as np
import pytest

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.openfoam_case import export_native_mesh, write_rans_case


def test_export_retains_shared_face_and_outward_closed_cells(tmp_path):
    points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0,0,-1.]])
    mesh = NativeMesh3D.from_cells(points, [[0,1,2,3], [0,2,1,4]], ['tetra']*2)
    export_native_mesh(mesh, tmp_path)
    folder = tmp_path/'constant/polyMesh'
    order = np.load(tmp_path/'native-face-order.npy')
    faces = [[int(x) for x in row.split()] for row in
             re.findall(r'^3\(([^)]+)\)$', (folder/'faces').read_text(), re.M)]
    assert len(faces) == 7
    assert mesh.neighbor[order[0]] == 1
    areas = np.array([np.cross(points[f[1]]-points[f[0]], points[f[2]]-points[f[0]])/2 for f in faces])
    np.testing.assert_allclose(areas, mesh.area_vectors[order], atol=1e-14)
    for c in range(2):
        flux = areas[mesh.owner[order] == c].sum(0) - areas[mesh.neighbor[order] == c].sum(0)
        np.testing.assert_allclose(flux, 0, atol=1e-14)
    with pytest.raises(FileExistsError):
        export_native_mesh(mesh, tmp_path)


def test_fresh_meshing_case_and_invalid_input_do_not_overwrite(tmp_path):
    case = tmp_path/'case'
    with pytest.raises(ValueError):
        write_rans_case(None, case, reynolds=-1)
    assert not case.exists()
    write_rans_case(None, case)
    assert not (case/'constant/polyMesh').exists()
    original = (case/'0/U').read_bytes()
    with pytest.raises(ValueError):
        write_rans_case(None, case)
    assert (case/'0/U').read_bytes() == original
