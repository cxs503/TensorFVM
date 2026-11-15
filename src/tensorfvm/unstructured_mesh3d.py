"""3-D unstructured tetrahedral mesh construction and invariants.

``tet_mesh_from_msh`` closes the import path (Gmsh ASCII v2 -> ``NativeMesh3D``
owner/neighbour faces), and ``cube_tet_msh`` emits a reproducible warped
tetrahedral mesh of the unit cube as Gmsh 2.2 ASCII so that every verification
gate runs through the *same* import path a user mesh takes.  No Gmsh binary and
no logical (i,j,k) indexing is involved anywhere downstream.
"""
import numpy as np

from .gmsh_msh3d import PHYSICAL_WHITELIST, parse_msh_v2_ascii, read_msh_v2_ascii
from .native_mesh3d import NativeMesh3D


#: Kuhn (Freudenthal) six-tetrahedron split of the reference cube, written with
#: local node bit codes p = 4*ix + 2*iy + iz (node 0 is (0,0,0), node 7 is
#: (1,1,1)).  Every tet is {000, e_pi1, e_pi1+e_pi2, 111} for one permutation
#: of the three axes, which is the only subdivision whose face diagonals agree
#: across a shared cube face -- an arbitrary six-tet list leaves hanging nodes
#: and the mesh then has non-conforming (single-counted) internal faces.
_KUHN = ((0, 4, 6, 7), (0, 4, 5, 7), (0, 2, 6, 7),
         (0, 2, 3, 7), (0, 1, 5, 7), (0, 1, 3, 7))

#: Default face tags of the unit cube, chosen so the whitelist is exercised:
#: flow enters at x=0, leaves at x=1, the sides are walls and the top/bottom
#: are far-field.  ``fluid`` names the volume group in ``$PhysicalNames``.
CUBE_PATCHES = {(0, "x"): "inlet", (1, "x"): "outlet",
                (0, "y"): "wall", (1, "y"): "wall",
                (0, "z"): "far-field", (1, "z"): "far-field"}


def tet_mesh_from_msh(path=None, *, text=None, allowed=None):
    """Build a :class:`NativeMesh3D` from a Gmsh 2.x ASCII tet mesh."""
    if (path is None) == (text is None):
        raise ValueError("pass exactly one of path= or text=")
    if path is not None:
        if allowed is None:
            msh = read_msh_v2_ascii(path)
        else:
            msh = read_msh_v2_ascii(path, allowed=allowed)
    else:
        msh = parse_msh_v2_ascii(text, allowed=PHYSICAL_WHITELIST if allowed is None else allowed)
    return NativeMesh3D.from_cells(msh.vertices, msh.cells, msh.cell_types, msh.patches)


def cube_tet_msh(n, *, warp=0.0, length=1.0, patches=None):
    """Gmsh 2.2 ASCII text of a warped ``n``-cell-per-axis tetrahedral cube.

    Each hexahedral cell of a uniform lattice is split by the Kuhn six-tet
    decomposition, which is conforming because every face diagonal is taken
    from global vertex ids.  ``warp`` displaces the lattice by a smooth
    sinusoidal field of amplitude ``warp`` times the cell size; the field
    vanishes on the cell faces it is normal to, so the outer boundary stays
    planar and the mapping stays a diffeomorphism for ``warp < 0.5``.  The
    result is a genuinely irregular tetrahedral mesh: no logical index survives
    downstream and face normals are not aligned with centroid differences.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive number of cells per axis")
    h = length / n
    axis = np.arange(n + 1) * h
    ijk = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    vertices = ijk.copy()
    if warp:
        if not 0.0 <= warp < 0.5:
            raise ValueError("warp must satisfy 0 <= warp < 0.5")
        phase = (2.0 * np.pi / length) * ijk
        amplitude = warp * h
        vertices[:, 0] += amplitude * np.sin(phase[:, 0]) * np.sin(phase[:, 1] + 1.1) * np.sin(phase[:, 2] + 0.7)
        vertices[:, 1] += amplitude * np.sin(phase[:, 1]) * np.sin(phase[:, 2] + 0.4) * np.sin(phase[:, 0] + 1.9)
        vertices[:, 2] += amplitude * np.sin(phase[:, 2]) * np.sin(phase[:, 0] + 2.3) * np.sin(phase[:, 1] + 0.9)

    def node(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                corner = [node(i + (p >> 2 & 1), j + (p >> 1 & 1), k + (p & 1))
                          for p in range(8)]
                cells.extend(tuple(corner[l] for l in tet) for tet in _KUHN)

    # Boundary triangles are the tet faces seen exactly once; tag them by the
    # plane they lie on.  Vertices stay exactly on the x=0/length planes, so
    # the comparison is exact.
    counts, record = {}, {}
    for cell in cells:
        for local in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)):
            face = tuple(cell[index] for index in local)
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
            record.setdefault(key, face)
    table = CUBE_PATCHES if patches is None else patches
    boundary = {}
    for key, count in counts.items():
        if count != 1:
            continue
        face = np.asarray(vertices[list(key)])
        for coordinate in range(3):
            if np.all(face[:, coordinate] == 0.0):
                boundary[record[key]] = table[(0, "xyz"[coordinate])]
                break
            if np.all(face[:, coordinate] == length):
                boundary[record[key]] = table[(1, "xyz"[coordinate])]
                break
        else:
            raise ValueError("generated boundary face is not on a cube plane")

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    names = sorted({str(name) for name in boundary.values()})
    tags = {name: number + 1 for number, name in enumerate(names)}
    lines += [str(len(names) + 1)]
    lines += ['2 {} "{}"'.format(tags[name], name) for name in names]
    lines += ['3 99 "fluid"']
    lines += ["$EndPhysicalNames", "$Nodes", str(len(vertices))]
    lines += ["{} {} {} {}".format(number + 1, *[repr(float(value)) for value in xyz])
              for number, xyz in enumerate(vertices)]
    lines += ["$EndNodes", "$Elements", str(len(cells) + len(boundary))]
    number = 0
    for cell in cells:
        number += 1
        lines.append("{} 4 2 99 1 {}".format(
            number, " ".join(str(int(node_) + 1) for node_ in cell)))
    for face, name in boundary.items():
        number += 1
        lines.append("{} 2 2 {} 1 {}".format(
            number, tags[name], " ".join(str(int(node_) + 1) for node_ in face)))
    lines.append("$EndElements")
    return "\n".join(lines) + "\n"


def mesh_invariants(mesh):
    """Geometric invariants a tet mesh must satisfy for the FVM operators.

    * ``positive_volume`` -- every cell volume is positive.
    * ``face_normal_consistency_error`` -- the stored area vector equals the
      one rebuilt from the stored face-vertex ordering (fan of triangles), so
      the owner-outward normal really belongs to that cyclic node order.
    * ``internal_normal_points_to_neighbour`` -- on every internal face the
      area vector satisfies ``s . (x_n - x_o) > 0``, i.e. the owner-side normal
      points at the neighbour; the neighbour's own vector is ``-s`` by
      construction, and the projection minimum quantifies the margin.
    * ``closure_error`` -- the outward area vectors of each cell sum to zero.
    """
    if not isinstance(mesh, NativeMesh3D):
        raise TypeError("mesh_invariants expects a NativeMesh3D")
    owner, neighbor = mesh.owner, mesh.neighbor
    internal = neighbor >= 0
    areas = np.asarray(mesh.area_vectors, dtype=float)
    centroids = np.asarray(mesh.cell_centers, dtype=float)
    d = np.asarray(mesh.face_centers, dtype=float) - centroids[owner]
    d[internal] = centroids[neighbor[internal]] - centroids[owner[internal]]
    projection = np.einsum("ij,ij->i", areas, d)

    rebuilt = np.zeros_like(areas)
    vertices = np.asarray(mesh.vertices, dtype=float)
    for index, face in enumerate(mesh.face_vertices):
        points = vertices[list(face)]
        fan = 0.5 * np.cross(points[1:-1] - points[0], points[2:] - points[0])
        rebuilt[index] = fan.sum(axis=0)
    consistency = np.linalg.norm(rebuilt - areas, axis=1) / np.linalg.norm(areas, axis=1)

    closure = np.zeros_like(centroids)
    magnitude = np.zeros(len(mesh.cells))
    np.add.at(closure, owner, areas)
    np.add.at(magnitude, owner, np.linalg.norm(areas, axis=1))
    np.add.at(closure, neighbor[internal], -areas[internal])
    np.add.at(magnitude, neighbor[internal], np.linalg.norm(areas[internal], axis=1))

    nonorthogonality = np.degrees(np.arccos(np.clip(
        projection / (np.linalg.norm(areas, axis=1) * np.linalg.norm(d, axis=1)), -1.0, 1.0)))
    labels = [str(label) for label in mesh.patch_labels]
    return dict(
        cells=int(len(mesh.cells)),
        faces=int(len(owner)),
        positive_volume=bool(np.all(np.asarray(mesh.cell_volumes) > 0)),
        minimum_volume=float(np.min(mesh.cell_volumes)),
        face_normal_consistency_error=float(consistency.max()),
        internal_normal_points_to_neighbour=bool(np.all(projection[internal] > 0)),
        minimum_projection=float(projection.min()),
        closure_error=float(np.max(np.linalg.norm(closure, axis=1) / magnitude)),
        maximum_nonorthogonality_degrees=float(nonorthogonality.max()),
        patches={name: labels.count(name) for name in sorted(set(labels))},
    )