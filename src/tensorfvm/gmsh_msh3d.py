"""Gmsh 2.x ASCII tetrahedral meshes for the 3-D unstructured FVM core.

Reading is pure Python: Gmsh is needed only to *produce* a ``.msh`` file, never
to read one or to solve on it.  The 2-D external path
(:class:`tensorfvm.gmsh_external.GmshExternalMesh`) fixes the contract that is
reused here: ASCII v2 only, linear elements only, and every boundary face must
carry a physical name from an explicit whitelist.  An unknown physical name is
a hard error, never a silently untagged face.

The reader returns node-indexed connectivity plus a ``{sorted face key: name}``
patch map; :func:`tensorfvm.unstructured_mesh3d.tet_mesh_from_msh` turns that
into a :class:`tensorfvm.native_mesh3d.NativeMesh3D` owner/neighbour mesh.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import shlex

import numpy as np


#: Physical names accepted on a 3-D tet mesh.  ``inlet``/``outlet``/``wall``/
#: ``far-field`` are the 2-D body-fitted whitelist; ``fluid`` names the volume
#: group of the same mesh (dimension 3) and is never a boundary patch.
PHYSICAL_WHITELIST = ("inlet", "outlet", "wall", "far-field", "fluid")

#: Gmsh element type ids: linear triangle (2) and linear tetrahedron (4).
_TRI, _TET = 2, 4
_NODES_PER_TYPE = {_TRI: 3, _TET: 4}


@dataclass(frozen=True)
class Msh3D:
    """Linear tetrahedral mesh parsed from a Gmsh 2.x ASCII file."""

    vertices: np.ndarray
    cells: tuple
    cell_types: tuple
    patches: dict
    physical_names: dict
    version: str

    @property
    def boundary_faces(self):
        return tuple(self.patches)


def _normalise(name):
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def _sections(text):
    lines = text.splitlines()
    sections, index = {}, 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("$") or line.startswith("$End"):
            index += 1
            continue
        name, end = line[1:], "$End" + line[1:]
        index += 1
        body = []
        while index < len(lines) and lines[index].strip() != end:
            stripped = lines[index].strip()
            if stripped:
                body.append(stripped)
            index += 1
        if index == len(lines):
            raise ValueError("unterminated Gmsh section $" + name)
        sections[name] = body
        index += 1
    return sections


def _physical_names(lines, allowed):
    names = {}
    if not lines:
        return names
    try:
        count = int(lines[0])
    except ValueError as exc:
        raise ValueError("malformed Gmsh $PhysicalNames header") from exc
    if len(lines) - 1 != count:
        raise ValueError("incomplete Gmsh $PhysicalNames list")
    for line in lines[1:]:
        try:
            dimension, tag, raw = shlex.split(line)
            dimension, tag = int(dimension), int(tag)
        except (ValueError, IndexError) as exc:
            raise ValueError("malformed Gmsh $PhysicalNames line") from exc
        if dimension not in (2, 3):
            raise ValueError("3-D tet mesh physical names must be faces (2) or volumes (3)")
        name = _normalise(raw)
        if name not in allowed:
            raise ValueError("unknown 3-D physical name: " + raw)
        if (dimension, tag) in names:
            raise ValueError("duplicate Gmsh physical tag")
        names[dimension, tag] = name
    return names


def _nodes(lines):
    if not lines:
        raise ValueError("Gmsh mesh contains no $Nodes section")
    try:
        count = int(lines[0])
    except ValueError as exc:
        raise ValueError("malformed Gmsh $Nodes header") from exc
    if len(lines) - 1 != count:
        raise ValueError("incomplete Gmsh node list")
    ids, coordinates = [], []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("Gmsh 3-D nodes need exactly three coordinates")
        node_id = int(fields[0])
        xyz = tuple(float(value) for value in fields[1:])
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("nonfinite node coordinate")
        ids.append(node_id)
        coordinates.append(xyz)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Gmsh node id")
    return ids, np.asarray(coordinates, dtype=float)


def _elements(lines, names, node_index):
    if not lines:
        raise ValueError("Gmsh mesh contains no $Elements section")
    try:
        count = int(lines[0])
    except ValueError as exc:
        raise ValueError("malformed Gmsh $Elements header") from exc
    if len(lines) - 1 != count:
        raise ValueError("incomplete Gmsh element list")
    cells, patches, patch_names = [], {}, {}
    for line in lines[1:]:
        try:
            fields = [int(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError("malformed Gmsh element line") from exc
        if len(fields) < 3:
            raise ValueError("malformed Gmsh element line")
        etype, tag_count = fields[1], fields[2]
        tags, connectivity = fields[3:3 + tag_count], fields[3 + tag_count:]
        if etype not in _NODES_PER_TYPE:
            raise ValueError("unsupported Gmsh 3-D element type {} "
                             "(linear tetrahedra and boundary triangles only)".format(etype))
        if len(connectivity) != _NODES_PER_TYPE[etype]:
            raise ValueError("Gmsh element type {} must have {} nodes".format(
                etype, _NODES_PER_TYPE[etype]))
        try:
            nodes = tuple(node_index[node] for node in connectivity)
        except KeyError as exc:
            raise ValueError("Gmsh element references an undefined node") from exc
        if etype == _TET:
            cells.append(nodes)
            continue
        if not tags:
            raise ValueError("Gmsh boundary triangle has no physical tag")
        name = names.get((2, tags[0]))
        if name is None:
            raise ValueError("Gmsh boundary triangle has an unknown physical name")
        key = tuple(sorted(nodes))
        previous = patch_names.get(key)
        if previous is not None and previous != name:
            raise ValueError("boundary triangle belongs to conflicting physical groups")
        patch_names[key] = name
        patches[key] = name
    if not cells:
        raise ValueError("Gmsh 3-D mesh contains no tetrahedral cells")
    return tuple(cells), patches


def parse_msh_v2_ascii(text, *, allowed=PHYSICAL_WHITELIST):
    """Parse the text of a Gmsh 2.x ASCII file into a :class:`Msh3D`."""
    allowed = {_normalise(name) for name in allowed}
    sections = _sections(text)
    for required in ("MeshFormat", "Nodes", "Elements"):
        if required not in sections:
            raise ValueError("Gmsh mesh is missing the ${} section".format(required))
    fields = sections["MeshFormat"][0].split()
    if len(fields) < 3:
        raise ValueError("malformed Gmsh $MeshFormat")
    version, file_type = fields[0], fields[1]
    if not version.startswith("2."):
        raise ValueError("Gmsh ASCII v2 required, got version " + version)
    if file_type != "0":
        raise ValueError("Gmsh binary mesh unsupported: ASCII (=0) required")
    names = _physical_names(sections.get("PhysicalNames", []), allowed)
    ids, vertices = _nodes(sections["Nodes"])
    index = {node: position for position, node in enumerate(ids)}
    cells, patches = _elements(sections["Elements"], names, index)
    return Msh3D(vertices=vertices, cells=cells,
                 cell_types=tuple("tetra" for _ in cells), patches=patches,
                 physical_names=names, version=version)


def read_msh_v2_ascii(path, *, allowed=PHYSICAL_WHITELIST):
    """Read a Gmsh 2.x ASCII tet mesh from ``path``."""
    return parse_msh_v2_ascii(Path(path).read_text(encoding="utf-8", errors="strict"),
                              allowed=allowed)


def write_msh_v2_ascii(path, vertices, cells, patches):
    """Write a minimal Gmsh 2.2 ASCII tet mesh (1-based ids, physical tags)."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (n,3)")
    names = sorted({str(name) for name in patches.values()})
    tags = {name: number + 1 for number, name in enumerate(names)}
    key_to_name = {tuple(sorted(int(node) for node in face)): str(name)
                   for face, name in patches.items()}
    # Boundary triangles are emitted in the orientation they appear in the cells.
    boundary = []
    seen = set()
    for cell in cells:
        for local in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)):
            face = tuple(int(cell[index]) for index in local)
            key = tuple(sorted(face))
            if key in seen:
                continue
            if key in key_to_name:
                seen.add(key)
                boundary.append((face, key_to_name[key]))
    missing = set(key_to_name) - seen
    if missing:
        raise ValueError("{} boundary faces are not present in the cell list".format(len(missing)))
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    lines += ['2 {} "{}"'.format(tags[name], name) for name in names]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(vertices))]
    lines += ["{} {} {} {}".format(number + 1, *[repr(float(value)) for value in xyz])
              for number, xyz in enumerate(vertices)]
    lines += ["$EndNodes", "$Elements", str(len(cells) + len(boundary))]
    number = 0
    for cell in cells:
        number += 1
        lines.append("{} 4 2 0 1 {}".format(number, " ".join(str(int(n) + 1) for n in cell)))
    for face, name in boundary:
        number += 1
        lines.append("{} 2 2 {} 1 {}".format(number, tags[name],
                                             " ".join(str(int(n) + 1) for n in face)))
    lines.append("$EndElements")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)