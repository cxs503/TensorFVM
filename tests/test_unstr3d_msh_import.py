"""Gmsh ASCII v2 tetrahedral import and mesh invariants (FEAT-UNSTR3D-MESH)."""
import numpy as np
import pytest

from tensorfvm.gmsh_msh3d import (PHYSICAL_WHITELIST, parse_msh_v2_ascii,
                                 read_msh_v2_ascii, write_msh_v2_ascii)
from tensorfvm.unstructured_mesh3d import (cube_tet_msh, mesh_invariants,
                                           tet_mesh_from_msh)

SINGLE_TET = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
2 1 "inlet"
2 2 "outlet"
2 3 "wall"
3 99 "fluid"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
$EndNodes
$Elements
5
1 4 2 99 1 1 2 3 4
2 2 2 1 1 1 2 3
3 2 2 2 1 1 2 4
4 2 2 3 1 1 3 4
5 2 2 3 1 2 3 4
$EndElements
"""


def test_single_tet_round_trip(tmp_path):
    msh = parse_msh_v2_ascii(SINGLE_TET)
    assert msh.cells == ((0, 1, 2, 3),)
    assert msh.physical_names[(3, 99)] == "fluid"
    assert sorted(msh.patches.values()) == ["inlet", "outlet", "wall", "wall"]
    mesh = tet_mesh_from_msh(text=SINGLE_TET)
    assert len(mesh.cells) == 1 and len(mesh.owner) == 4
    assert len(mesh.neighbor) == 4 and np.all(mesh.neighbor < 0)
    assert mesh.cell_volumes[0] == pytest.approx(1.0 / 6.0, rel=1e-12)
    # writer -> reader must reproduce the same topology
    path = write_msh_v2_ascii(tmp_path / "mesh.msh", msh.vertices, msh.cells, msh.patches)
    again = read_msh_v2_ascii(path)
    assert again.cells == msh.cells and again.patches == msh.patches


def test_physical_name_whitelist_is_enforced():
    text = SINGLE_TET.replace('"wall"', '"porous-jump"')
    with pytest.raises(ValueError, match="unknown 3-D physical name"):
        parse_msh_v2_ascii(text)


def test_whitelist_names_are_the_agreed_set():
    assert PHYSICAL_WHITELIST == ("inlet", "outlet", "wall", "far-field", "fluid")


@pytest.mark.parametrize("replacement, message", [
    ("2.2 1 8", "binary mesh unsupported"),
    ("4.1 0 8", "ASCII v2 required"),
])
def test_format_guards(replacement, message):
    with pytest.raises(ValueError, match=message):
        parse_msh_v2_ascii(SINGLE_TET.replace("2.2 0 8", replacement))


def test_high_order_element_is_rejected():
    text = SINGLE_TET.replace("1 4 2 99 1 1 2 3 4",
                              "1 11 2 99 1 1 2 3 4 1 2 3 4 1 2 3")
    with pytest.raises(ValueError, match="unsupported Gmsh 3-D element type"):
        parse_msh_v2_ascii(text)


def test_untagged_boundary_face_is_rejected():
    text = SINGLE_TET.replace("$Elements\n5\n", "$Elements\n4\n").replace(
        "5 2 2 3 1 2 3 4\n", "")
    with pytest.raises(ValueError):
        tet_mesh_from_msh(text=text)


def test_cube_tet_mesh_is_a_conforming_boundary_carrying_tet_mesh():
    mesh = tet_mesh_from_msh(text=cube_tet_msh(3, warp=0.25))
    invariants = mesh_invariants(mesh)
    assert invariants["cells"] == 6 * 3 ** 3
    assert invariants["positive_volume"]
    assert invariants["internal_normal_points_to_neighbour"]
    assert invariants["face_normal_consistency_error"] < 1e-12
    assert invariants["closure_error"] < 1e-12
    assert set(invariants["patches"]) == {"inlet", "outlet", "wall", "far-field", "internal"}
    # Inlet/outlet are one cube face each, walls two, far-field two.
    assert (invariants["patches"]["inlet"] == invariants["patches"]["outlet"]
            == 2 * 3 ** 2)
    assert invariants["patches"]["wall"] == 4 * 3 ** 2


def test_warped_mesh_is_non_orthogonal_and_still_valid():
    straight = mesh_invariants(tet_mesh_from_msh(text=cube_tet_msh(3)))
    warped = mesh_invariants(tet_mesh_from_msh(text=cube_tet_msh(3, warp=0.25)))
    assert warped["positive_volume"] and warped["minimum_volume"] > 0.0
    assert warped["maximum_nonorthogonality_degrees"] > straight["maximum_nonorthogonality_degrees"]
    assert warped["maximum_nonorthogonality_degrees"] > 30.0
