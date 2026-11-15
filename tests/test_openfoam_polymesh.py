"""Round-trip / connectivity / accuracy gates for the polyMesh reader channel."""
import numpy as np
import pytest

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.openfoam_case import export_native_mesh
from tensorfvm.openfoam_polymesh import (compare_meshes, l2_error, read_polymesh,
                                        roundtrip_gate, solve_poisson,
                                        structured_box_primitives, to_native_mesh3d,
                                        write_polymesh)


def _cavity_case(tmp_path, nx=4, ny=4, nz=2, **kw):
    """Externally authored (i.e. not produced by our exporter) polyMesh."""
    sides = {'xmin': 'fixedWalls', 'xmax': 'fixedWalls', 'ymin': 'fixedWalls',
             'ymax': 'movingWall', 'zmin': 'frontAndBack', 'zmax': 'frontAndBack'}
    pts, faces, owner, neighbour, patches = structured_box_primitives(
        nx, ny, nz, size=(1.0, 1.0, 0.1), sides=sides, **kw)
    case = tmp_path / 'cavity'
    write_polymesh(case, pts, faces, owner, neighbour, patches)
    return case


# --------------------------------------------------------------------------
# round trip: native export -> read back
# --------------------------------------------------------------------------
def test_exported_polymesh_reads_back_bit_exactly(tmp_path):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1.0]])
    mesh = NativeMesh3D.from_cells(pts, [[0, 1, 2, 3], [0, 2, 1, 4]], ['tetra'] * 2)
    parsed, report = roundtrip_gate(mesh, tmp_path / 'case')
    assert report['passed'], report
    assert report['points']['bitwise'] and report['points']['max_abs_deviation'] == 0.0
    assert report['face_vertices_oriented']['equal']
    assert report['owner']['equal'] and report['neighbour']['equal']
    assert report['patch_labels']['equal']
    assert parsed.n_cells == 2 and parsed.n_faces == 7 and parsed.n_internal_faces == 1


def test_hex_box_roundtrip_keeps_geometry(tmp_path):
    pts, faces, owner, neighbour, patches = structured_box_primitives(
        4, 3, 2, size=(2.0, 1.0, 1.0), shear=0.25)
    write_polymesh(tmp_path / 'authored', pts, faces, owner, neighbour, patches)
    authored = read_polymesh(tmp_path / 'authored')
    assert authored.audit()['cell_type_counts'] == {'hexahedron': 24}
    np.testing.assert_allclose(authored.cell_volumes.sum(), 2.0, rtol=1e-14)

    native = to_native_mesh3d(authored)
    export_native_mesh(native, tmp_path / 'exported')
    again = read_polymesh(tmp_path / 'exported')
    report = compare_meshes(authored, again)
    assert report['passed'], report
    np.testing.assert_allclose(again.cell_volumes.sum(), 2.0, rtol=1e-14)
    np.testing.assert_allclose(again.cell_centers.sum(axis=0),
                               authored.cell_centers.sum(axis=0), rtol=1e-14)


# --------------------------------------------------------------------------
# reader robustness
# --------------------------------------------------------------------------
def test_reader_handles_comments_and_preamble(tmp_path):
    case = _cavity_case(tmp_path, nx=2, ny=2, nz=1)
    folder = case / 'constant' / 'polyMesh'
    points = folder / 'points'
    text = points.read_text()
    cut = text.index('}\n') + 2                     # right after the FoamFile dict
    points.write_text('/* block comment */\n' + text[:cut]
                      + 'convertToMeters 1; // metres\n' + text[cut:])
    mesh = read_polymesh(case)
    assert mesh.n_cells == 4 and mesh.n_faces == 20


def test_binary_format_is_rejected(tmp_path):
    case = _cavity_case(tmp_path, nx=2, ny=2, nz=1)
    owner = case / 'constant' / 'polyMesh' / 'owner'
    owner.write_text(owner.read_text().replace('format ascii;', 'format binary;'))
    with pytest.raises(ValueError, match='ASCII'):
        read_polymesh(case)


def test_broken_boundary_tiling_is_rejected(tmp_path):
    case = _cavity_case(tmp_path, nx=2, ny=2, nz=1)
    boundary = case / 'constant' / 'polyMesh' / 'boundary'
    import re
    text = boundary.read_text()
    start = int(re.search(r'startFace (\d+)', text).group(1))
    boundary.write_text(text.replace(f'startFace {start}', f'startFace {start + 1}', 1))
    with pytest.raises(ValueError, match='contiguous'):
        read_polymesh(case)


def test_inconsistent_face_orientation_is_rejected(tmp_path):
    pts, faces, owner, neighbour, patches = structured_box_primitives(2, 2, 2)
    faces = list(faces)
    faces[0] = tuple(reversed(faces[0]))          # flip one internal face
    case = tmp_path / 'flipped'
    write_polymesh(case, pts, faces, owner, neighbour, patches)
    # either the cell-tetra check or the owner/neighbour contract fires first
    with pytest.raises(ValueError, match='oriented|neighbour to owner'):
        read_polymesh(case)


def test_patch_face_counts_and_connectivity(tmp_path):
    case = _cavity_case(tmp_path, nx=4, ny=4, nz=2)
    mesh = read_polymesh(case)
    audit = mesh.audit()
    assert audit['cells'] == 32
    assert audit['patch_face_counts'] == {'fixedWalls': 24, 'frontAndBack': 32,
                                          'movingWall': 8}
    assert audit['maximum_relative_closure'] < 1e-12
    assert audit['minimum_face_projection'] > 0.0
    assert all(len(lst) == 6 for lst in mesh.cell_faces)
    np.testing.assert_allclose(mesh.cell_volumes.sum(), 0.1, rtol=1e-14)
    # every internal face is claimed by exactly two cells
    for f in range(mesh.n_internal_faces):
        assert f in mesh.cell_faces[mesh.owner[f]]
        assert f in mesh.cell_faces[mesh.neighbour[f]]


# --------------------------------------------------------------------------
# Poisson cross-check
# --------------------------------------------------------------------------
def _manufactured(x):
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 2])


@pytest.mark.parametrize('kwargs', [dict(), dict(jitter=0.2, seed=3)])
def test_poisson_second_order_on_parsed_mesh(tmp_path, kwargs):
    errors = []
    for nx in (8, 16):
        pts, faces, owner, neighbour, patches = structured_box_primitives(nx, nx, nx,
                                                                         **kwargs)
        case = tmp_path / f'box{nx}'
        write_polymesh(case, pts, faces, owner, neighbour, patches)
        mesh = read_polymesh(case)
        bc = {pg['name']: ('fixedValue', 0.0) for pg in mesh.patches}
        exact = _manufactured(mesh.cell_centers)
        result = solve_poisson(mesh, -3 * np.pi ** 2 * exact, bc, n_correctors=3)
        errors.append(l2_error(mesh, result['phi'], exact))
        assert result['conservation_budget'] < 1e-6
        assert result['residual'] < 1e-10
    order = float(np.log2(errors[0] / errors[-1]))
    assert order > 1.8, (order, errors)
    assert errors[-1] < 0.03, errors


def test_poisson_needs_a_dirichlet_patch(tmp_path):
    case = _cavity_case(tmp_path, nx=2, ny=2, nz=1)
    mesh = read_polymesh(case)
    bc = {pg['name']: ('zeroGradient',) for pg in mesh.patches}
    with pytest.raises(ValueError, match='fixedValue'):
        solve_poisson(mesh, np.zeros(mesh.n_cells), bc)


def test_poisson_zero_gradient_patch_is_flux_free(tmp_path):
    sides = {'xmin': 'inlet', 'xmax': 'outlet', 'ymin': 'wall', 'ymax': 'wall',
             'zmin': 'empty', 'zmax': 'empty'}
    pts, faces, owner, neighbour, patches = structured_box_primitives(
        6, 6, 1, sides=sides)
    case = tmp_path / 'chan'
    write_polymesh(case, pts, faces, owner, neighbour, patches)
    mesh = read_polymesh(case)
    bc = {'inlet': ('fixedValue', 0.0), 'outlet': ('fixedValue', 1.0),
          'wall': ('zeroGradient',), 'empty': ('zeroGradient',)}
    # no source, pure conduction between two fixed walls -> linear profile 0..1 in x
    result = solve_poisson(mesh, np.zeros(mesh.n_cells), bc, n_correctors=2)
    np.testing.assert_allclose(result['phi'], mesh.cell_centers[:, 0], atol=1e-12)
    assert abs(result['boundary_flux']) < 1e-12