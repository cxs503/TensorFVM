"""Read an OpenFOAM polyMesh (ASCII) into 3-D unstructured finite-volume data.

``tensorfvm.openfoam_case`` writes ``constant/polyMesh`` from a
:class:`~tensorfvm.native_mesh3d.NativeMesh3D`.  This module is the inverse
channel: it parses the four polyMesh primitives (``points``/``faces``/``owner``/
``neighbour``) plus the ``boundary`` polyBoundaryMesh, reconstructs the
cell<->face connectivity that the file *implies*, and rebuilds the geometry
(face centres, owner-outward area vectors, cell centres, cell volumes) with the
OpenFOAM sign contract (``Sf`` points from ``owner`` towards ``neighbour`` and
out of the domain for boundary faces).

What is deliberately *not* claimed here:

* only ASCII ``polyMesh`` is supported (``format binary`` raises);
* faces must be simple polygons; non-convex/folded faces are reported, not
  silently repaired;
* geometry formulas are polygon-centroid exact for planar faces.  For warped
  (non-planar) faces the area vector is the same triangle fan OpenFOAM uses,
  but the centre is our own area-weighted triangle decomposition -- that is
  documented as a convention and validated by the closure identities
  (sum of owner-outward face area vectors per cell = 0, divergence-theorem
  volume).

Everything downstream (``solve_poisson``) exists so an imported mesh can be
*cross-checked against a known answer*; it is not presented as a replacement
for OpenFOAM's own solvers.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .registry import capability


# --------------------------------------------------------------------------
# tokeniser / low level parser
# --------------------------------------------------------------------------
_TOKEN = re.compile(
    r"""
      /\*.*?\*/            # block comment
    | //[^\n]*             # line comment
    | "(?:[^"\\]|\\.)*"    # quoted string
    | [(){}\;]             # structural
    | [^\s(){}\;"]+        # bare word / number
    """,
    re.S | re.X,
)
_INT = re.compile(r'^[+-]?\d+$')
_FLOAT = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _tokenise(text):
    out = []
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        if tok.startswith('//') or tok.startswith('/*'):
            continue
        out.append(tok)
    return out


class _Parser:
    def __init__(self, tokens, where='<polyMesh>'):
        self.t = tokens
        self.i = 0
        self.where = where

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def next(self):
        tok = self.peek()
        if tok is None:
            raise ValueError(f'{self.where}: unexpected end of file')
        self.i += 1
        return tok

    def expect(self, tok):
        got = self.next()
        if got != tok:
            raise ValueError(f'{self.where}: expected {tok!r}, got {got!r}')
        return got

    def skip_dict(self):
        """Consume a ``{ ... }`` block, honouring nesting."""
        self.expect('{')
        depth = 1
        while depth:
            tok = self.next()
            if tok == '{':
                depth += 1
            elif tok == '}':
                depth -= 1

    def read_header(self):
        """Parse the optional ``FoamFile {...}`` dictionary; return its fields."""
        if self.peek() != 'FoamFile':
            return {}
        self.next()
        self.expect('{')
        fields = {}
        while self.peek() != '}':
            key = self.next()
            vals = []
            while self.peek() not in (';', None):
                if self.peek() == '{':
                    self.skip_dict()
                    vals.append('<dict>')
                    continue
                vals.append(self.next())
            self.expect(';')
            fields[key] = ' '.join(vals)
        self.expect('}')
        fmt = fields.get('format', 'ascii').strip()
        if fmt != 'ascii':
            raise ValueError(
                f'{self.where}: format {fmt!r} not supported, only ASCII polyMesh '
                'can be read (binary polyMesh needs the OpenFOAM library)')
        return fields

    # -- typed list readers ------------------------------------------------
    def _count_list(self, item_reader):
        """``N ( item item ... )`` optionally preceded by ``key val;`` lines."""
        while True:
            tok = self.peek()
            if tok is None:
                raise ValueError(f'{self.where}: unexpected end of file')
            if _INT.match(tok) and self.peek(1) == '(':
                n = int(self.next())
                break
            while True:
                if self.next() == ';':
                    break
        self.expect('(')
        items = [item_reader(self) for _ in range(n)]
        self.expect(')')
        return items

    def read_scalar_list(self):
        def item(p):
            tok = p.next()
            if not _FLOAT.match(tok):
                raise ValueError(f'{p.where}: expected number, got {tok!r}')
            return float(tok)
        return self._count_list(item)

    def read_points(self):
        def item(p):
            p.expect('(')
            xyz = []
            for _ in range(3):
                tok = p.next()
                if not _FLOAT.match(tok):
                    raise ValueError(f'{p.where}: point coordinate {tok!r} is not a number')
                xyz.append(float(tok))
            p.expect(')')
            return tuple(xyz)
        return self._count_list(item)

    def read_faces(self):
        def item(p):
            tok = p.next()
            if not _INT.match(tok):
                raise ValueError(
                    f'{p.where}: expected face size integer, got {tok!r} '
                    '(ASCII faceList must use the compact form "n(v0 v1 ...)")')
            n = int(tok)
            if n < 3:
                raise ValueError(f'{p.where}: face with {n} vertices is degenerate')
            p.expect('(')
            ids = []
            for _ in range(n):
                tok = p.next()
                if not _INT.match(tok):
                    raise ValueError(f'{p.where}: face vertex label {tok!r} is not an integer')
                ids.append(int(tok))
            p.expect(')')
            return tuple(ids)
        return self._count_list(item)

    def read_boundary(self):
        """``N ( name { type ..; nFaces ..; startFace ..; } ... )``."""
        n = None
        while True:
            tok = self.peek()
            if tok is None:
                raise ValueError(f'{self.where}: unexpected end of file in boundary')
            if _INT.match(tok) and self.peek(1) == '(':
                n = int(self.next())
                break
            while True:
                if self.next() == ';':
                    break
        self.expect('(')
        patches = []
        for _ in range(n):
            name = self.next()
            self.expect('{')
            fields = {}
            while self.peek() != '}':
                key = self.next()
                vals = []
                while self.peek() not in (';', None):
                    if self.peek() == '{':
                        self.skip_dict()
                        vals.append('<dict>')
                        continue
                    vals.append(self.next())
                self.expect(';')
                fields[key] = vals
            self.expect('}')
            try:
                patches.append(dict(name=name, type=fields['type'][0],
                                    nFaces=int(fields['nFaces'][0]),
                                    startFace=int(fields['startFace'][0])))
            except KeyError as exc:
                raise ValueError(f'{self.where}: boundary patch {name!r} lacks {exc}') from None
        self.expect(')')
        return patches


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _face_geometry(points, face):
    """Owner-convention face geometry: area vector ``Sf`` and centroid."""
    x = points[list(face)]
    fan = np.cross(x[1:-1] - x[0], x[2:] - x[0])       # 2 * triangle area vectors
    sf = 0.5 * fan.sum(axis=0)
    mag = 0.5 * np.linalg.norm(fan, axis=1)
    total = mag.sum()
    if not np.isfinite(mag).all() or total <= 0.0:
        raise ValueError('degenerate face: zero area')
    centroid = ((x[0] + x[1:-1] + x[2:]) / 3.0 * mag[:, None]).sum(axis=0) / total
    return centroid, sf


def _cell_geometry(points, face_vertices, faces, signs, face_centers):
    """Volume and centroid by a tetrahedral fan from an interior reference point."""
    ref = face_centers[faces].mean(axis=0)
    volume = 0.0
    moment = np.zeros(3)
    for f, s in zip(faces, signs):
        x = points[list(face_vertices[f])]
        p0 = x[0] - ref
        v = (np.cross(x[1:-1] - ref, x[2:] - ref) @ p0) / 6.0
        v = s * v
        if np.any(v <= 0.0):
            raise ValueError(
                f'face {f} is not consistently oriented towards the outside of its cell '
                '(negative tetrahedron volume); the polyMesh face ordering is invalid')
        volume += v.sum()
        moment += (v[:, None] * ((x[0] + x[1:-1] + x[2:] + ref) / 4.0)).sum(axis=0)
    if volume <= 0.0:
        raise ValueError('nonpositive cell volume')
    return volume, moment / volume


def _infer_cell_type(n_faces, sizes):
    sizes = sorted(sizes)
    if n_faces == 4 and sizes == [3, 3, 3, 3]:
        return 'tetra'
    if n_faces == 6 and sizes == [3, 3, 3, 3, 4, 4]:
        return 'wedge'
    if n_faces == 6 and sizes == [4] * 6:
        return 'hexahedron'
    if n_faces == 5 and sizes == [3, 3, 3, 3, 4]:
        return 'pyramid'
    return 'polyhedron'


# --------------------------------------------------------------------------
# the mesh object
# --------------------------------------------------------------------------
@dataclass(frozen=True)
@capability('fvm.mesh.openfoam_polymesh',
            kind='mesh',
            maturity='developing',
            test_path='tests/test_openfoam_polymesh.py',
            wired=False,
            tested=True,
            verified=False,
            evidence={'grade': 'MECHANICS', 'docs': 0})
class PolyMesh:
    """3-D unstructured finite-volume mesh read from an OpenFOAM polyMesh."""

    points: np.ndarray                 # (nPoints, 3)
    face_vertices: tuple               # per face: vertex labels in file order
    owner: np.ndarray                  # (nFaces,) owner cell per face
    neighbour: np.ndarray              # (nInternalFaces,) neighbour cell per face
    patches: tuple                     # tuple of dicts (name/type/nFaces/startFace)
    face_centers: np.ndarray           # (nFaces, 3)
    face_area_vectors: np.ndarray      # (nFaces, 3), owner -> neighbour / outward
    cell_faces: tuple                  # per cell: tuple of face labels
    cell_vertices: tuple               # per cell: tuple of vertex labels
    cell_types: tuple                  # per cell: tetra/hexahedron/wedge/pyramid/polyhedron
    cell_centers: np.ndarray           # (nCells, 3)
    cell_volumes: np.ndarray           # (nCells,)
    patch_labels: np.ndarray           # per face: patch name, 'internal' inside
    case_dir: str = ''
    n_internal_faces: int = 0

    # -- derived conveniences ---------------------------------------------
    @property
    def n_cells(self):
        return len(self.cell_volumes)

    @property
    def n_faces(self):
        return len(self.owner)

    @property
    def internal(self):
        return np.arange(self.n_internal_faces)

    def neighbour_full(self):
        """Per-face neighbour label, ``-1`` on boundary faces (internal faces first)."""
        full = np.full(self.n_faces, -1, dtype=int)
        full[:self.n_internal_faces] = self.neighbour
        return full

    def patch_faces(self, name):
        for pg in self.patches:
            if pg['name'] == name:
                return np.arange(pg['startFace'], pg['startFace'] + pg['nFaces'])
        raise KeyError(f'no patch named {name!r}')

    def audit(self):
        """Closure/quality metrics -- the same identities ``NativeMesh3D.audit`` uses."""
        sf = self.face_area_vectors
        ii = np.arange(self.n_internal_faces)
        closure = np.zeros((self.n_cells, 3))
        area = np.zeros(self.n_cells)
        np.add.at(closure, self.owner, sf)
        np.add.at(area, self.owner, np.linalg.norm(sf, axis=1))
        np.add.at(closure, self.neighbour, -sf[ii])
        np.add.at(area, self.neighbour, np.linalg.norm(sf[ii], axis=1))
        d = self.face_centers - self.cell_centers[self.owner]
        d[ii] = self.cell_centers[self.neighbour] - self.cell_centers[self.owner[ii]]
        projection = np.einsum('ij,ij->i', sf, d)
        cosang = projection / (np.linalg.norm(sf, axis=1) * np.linalg.norm(d, axis=1))
        # skewness: face centre offset from the cell-centre line, normalised by half its length
        cf = self.face_centers[ii]
        a = self.cell_centers[self.owner[ii]]
        b = self.cell_centers[self.neighbour]
        skew = np.linalg.norm(cf - (a + b) * 0.5, axis=1) / np.linalg.norm(d[ii], axis=1)
        return dict(
            cells=int(self.n_cells),
            faces=int(self.n_faces),
            internal_faces=int(self.n_internal_faces),
            boundary_faces=int(self.n_faces - self.n_internal_faces),
            minimum_cell_volume=float(self.cell_volumes.min()),
            maximum_relative_closure=float((np.linalg.norm(closure, axis=1) / area).max()),
            minimum_face_projection=float(projection.min()),
            maximum_nonorthogonality_degrees=float(
                np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))).max()),
            maximum_skewness=float(skew.max()) if skew.size else 0.0,
            patch_face_counts={str(pg['name']): int(pg['nFaces']) for pg in self.patches},
            cell_type_counts={str(k): int(v) for k, v in
                              zip(*np.unique(np.array(self.cell_types, dtype=object),
                                             return_counts=True))},
        )


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------
def _locate(root):
    root = Path(root)
    candidate = root / 'constant' / 'polyMesh'
    if candidate.is_dir():
        return candidate
    if (root / 'points').is_file() and (root / 'faces').is_file():
        return root
    raise FileNotFoundError(f'{root}: neither a case dir nor a polyMesh dir')


def _read_all(folder):
    folder = Path(folder)
    out = {}
    for name in ('points', 'faces', 'owner', 'neighbour', 'boundary'):
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f'{path}: required polyMesh file missing')
        parser = _Parser(_tokenise(path.read_text()), where=str(path))
        parser.read_header()
        reader = {'points': parser.read_points, 'faces': parser.read_faces,
                  'owner': parser.read_scalar_list, 'neighbour': parser.read_scalar_list,
                  'boundary': parser.read_boundary}[name]
        out[name] = reader()
    return out


def _validate(raw, where):
    points = np.array(raw['points'], dtype=float)
    faces = tuple(raw['faces'])
    owner = np.array([int(x) for x in raw['owner']], dtype=int)
    neighbour = np.array([int(x) for x in raw['neighbour']], dtype=int)
    patches = tuple(raw['boundary'])
    n_faces = len(faces)
    if n_faces == 0:
        raise ValueError(f'{where}: empty face list')
    if len(owner) != n_faces:
        raise ValueError(f'{where}: owner has {len(owner)} entries but there are {n_faces} faces')
    if len(neighbour) > n_faces:
        raise ValueError(f'{where}: neighbour longer than the face list')
    if (owner < 0).any() or (neighbour < 0).any():
        raise ValueError(f'{where}: negative cell labels in owner/neighbour')
    for f, face in enumerate(faces):
        if len(set(face)) != len(face):
            raise ValueError(f'{where}: face {f} repeats a vertex label')
        if min(face) < 0 or max(face) >= len(points):
            raise ValueError(f'{where}: face {f} references a vertex outside 0..{len(points) - 1}')
    # boundary patches must tile the tail of the face list contiguously
    if not patches:
        raise ValueError(f'{where}: boundary list is empty')
    start = len(neighbour)
    for pg in patches:
        if not _IDENT.match(pg['name']):
            raise ValueError(f'{where}: patch name {pg["name"]!r} is not an identifier')
        if pg['nFaces'] < 0:
            raise ValueError(f'{where}: patch {pg["name"]!r} has negative nFaces')
        if pg['startFace'] != start:
            raise ValueError(
                f'{where}: patch {pg["name"]!r} startFace {pg["startFace"]} breaks '
                f'contiguous boundary ordering (expected {start})')
        start += pg['nFaces']
    if start != n_faces:
        raise ValueError(
            f'{where}: boundary patches cover {start - len(neighbour)} faces but the mesh has '
            f'{n_faces - len(neighbour)} boundary faces')
    return dict(points=points, faces=faces, owner=owner, neighbour=neighbour, patches=patches)


def read_polymesh(root):
    """Read ``<root>[/constant/polyMesh]/`` into a :class:`PolyMesh`.

    ``root`` may be the case directory or the ``polyMesh`` directory itself.
    Connectivity implied by the file is reconstructed and geometry rebuilt;
    inconsistent face orientation, non-positive volumes and a broken boundary
    tiling all raise ``ValueError`` instead of being silently patched.
    """
    folder = _locate(root)
    raw = _read_all(folder)
    data = _validate(raw, str(folder))
    points, faces = data['points'], data['faces']
    owner, neighbour, patches = data['owner'], data['neighbour'], data['patches']
    n_faces, n_internal = len(faces), len(neighbour)
    n_cells = int(max(owner.max(), neighbour.max() if n_internal else -1)) + 1

    # face geometry
    centers = np.zeros((n_faces, 3))
    areas = np.zeros((n_faces, 3))
    for f, face in enumerate(faces):
        centers[f], areas[f] = _face_geometry(points, face)

    # cell <-> face connectivity
    cell_faces = [[] for _ in range(n_cells)]
    for f in range(n_faces):
        cell_faces[int(owner[f])].append(f)
    for f in range(n_internal):
        cell_faces[int(neighbour[f])].append(f)
    empty = [c for c, lst in enumerate(cell_faces) if len(lst) < 4]
    if empty:
        raise ValueError(f'{folder}: cell {empty[0]} is bounded by fewer than 4 faces '
                         '(labels are not dense or connectivity is broken)')

    # cell geometry
    cell_centers = np.zeros((n_cells, 3))
    cell_volumes = np.zeros(n_cells)
    for c, lst in enumerate(cell_faces):
        signs = [1.0 if owner[f] == c else -1.0 for f in lst]
        cell_volumes[c], cell_centers[c] = _cell_geometry(
            points, faces, lst, signs, centers)

    # orientation contract: Sf points from owner to neighbour
    bad = []
    for f in range(n_internal):
        d = cell_centers[neighbour[f]] - cell_centers[owner[f]]
        if float(np.dot(areas[f], d)) <= 0.0:
            bad.append(f)
    if bad:
        raise ValueError(
            f'{folder}: {len(bad)} internal face(s) have an area vector pointing from '
            f'neighbour to owner (first face {bad[0]}); the polyMesh face ordering does '
            'not follow the OpenFOAM owner-outward convention')

    labels = np.empty(n_faces, dtype=object)
    labels[:n_internal] = 'internal'
    for pg in patches:
        labels[pg['startFace']:pg['startFace'] + pg['nFaces']] = pg['name']

    cell_vertices = []
    cell_types = []
    for lst in cell_faces:
        verts = sorted({v for f in lst for v in faces[f]})
        cell_vertices.append(tuple(verts))
        cell_types.append(_infer_cell_type(len(lst), [len(faces[f]) for f in lst]))

    return PolyMesh(
        points=points, face_vertices=faces, owner=owner, neighbour=neighbour,
        patches=patches, face_centers=centers, face_area_vectors=areas,
        cell_faces=tuple(tuple(sorted(lst)) for lst in cell_faces),
        cell_vertices=tuple(cell_vertices), cell_types=tuple(cell_types),
        cell_centers=cell_centers, cell_volumes=cell_volumes,
        patch_labels=labels, case_dir=str(folder), n_internal_faces=n_internal)


# --------------------------------------------------------------------------
# writer (independent of openfoam_case.export_native_mesh on purpose)
# --------------------------------------------------------------------------
def write_polymesh(root, points, faces, owner, neighbour, patches, *, location=None):
    """Write an ASCII polyMesh from raw primitives, without re-ordering anything.

    This is the *test-fixture* writer: it keeps the caller's face order,
    orientation and patch layout, so it can author meshes the way OpenFOAM
    utilities (blockMesh, snappyHexMesh, fluentMeshToFoam) would.  It exists so
    the reader can be exercised on input that our own exporter did not produce.
    """
    root = Path(root)
    folder = root / 'constant' / 'polyMesh'
    folder.mkdir(parents=True, exist_ok=False)
    loc = location or 'constant/polyMesh'

    def header(name, cls):
        return (f'FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n'
                f'    location "{loc}";\n    object {name};\n}}\n')

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be (n,3)')
    (folder / 'points').write_text(
        header('points', 'vectorField') + f'{len(points)}\n(\n'
        + '\n'.join('(' + ' '.join(f'{v:.17g}' for v in p) + ')' for p in points) + '\n)\n')
    faces = tuple(tuple(int(v) for v in f) for f in faces)
    (folder / 'faces').write_text(
        header('faces', 'faceList') + f'{len(faces)}\n(\n'
        + '\n'.join(f'{len(f)}(' + ' '.join(map(str, f)) + ')' for f in faces) + '\n)\n')
    owner = np.asarray(owner, dtype=int)
    neighbour = np.asarray(neighbour, dtype=int)
    (folder / 'owner').write_text(
        header('owner', 'labelList') + f'{len(owner)}\n(\n'
        + '\n'.join(map(str, owner)) + '\n)\n')
    (folder / 'neighbour').write_text(
        header('neighbour', 'labelList') + f'{len(neighbour)}\n(\n'
        + '\n'.join(map(str, neighbour)) + '\n)\n')
    entries = []
    for pg in patches:
        entries.append(f'    {pg["name"]}\n    {{\n        type {pg.get("type", "patch")};\n'
                       f'        nFaces {int(pg["nFaces"])};\n'
                       f'        startFace {int(pg["startFace"])};\n    }}')
    (folder / 'boundary').write_text(
        header('boundary', 'polyBoundaryMesh') + f'{len(patches)}\n(\n'
        + '\n'.join(entries) + '\n)\n')
    return folder


# --------------------------------------------------------------------------
# interop with NativeMesh3D / openfoam_case
# --------------------------------------------------------------------------
def to_native_mesh3d(mesh, *, allow_polyhedra=True):
    """Wrap a parsed :class:`PolyMesh` as a ``NativeMesh3D`` (for re-export).

    ``NativeMesh3D`` assumes tetra/hexahedron/wedge/pyramid cells; polyhedral
    cells are accepted only with ``allow_polyhedra=True`` and are tagged
    ``'polyhedron'``, which the native *construction* path never produces.
    """
    from tensorfvm.native_mesh3d import NativeMesh3D
    if not allow_polyhedra and 'polyhedron' in mesh.cell_types:
        raise ValueError('polyhedral cells present; only linear cell types supported')
    return NativeMesh3D(
        vertices=np.array(mesh.points, dtype=float),
        cells=tuple(mesh.cell_vertices),
        cell_types=tuple(mesh.cell_types),
        face_vertices=tuple(mesh.face_vertices),
        owner=np.array(mesh.owner, dtype=int),
        neighbor=mesh.neighbour_full(),          # NativeMesh3D keeps -1 on boundary
        face_centers=np.array(mesh.face_centers, dtype=float),
        area_vectors=np.array(mesh.face_area_vectors, dtype=float),
        cell_centers=np.array(mesh.cell_centers, dtype=float),
        cell_volumes=np.array(mesh.cell_volumes, dtype=float),
        patch_labels=np.array(mesh.patch_labels, dtype=object),
    )


def compare_meshes(a, b, *, atol_factor=1e-12):
    """Compare two :class:`PolyMesh` objects: connectivity exact, geometry tight.

    Faces are matched by their vertex *set*, so face ordering may differ; every
    face-indexed quantity (centre, area vector, owner/neighbour, patch label) is
    compared through that mapping.  Cell-indexed quantities are compared by cell
    label.  Connectivity must match bit-exactly; derived geometry is compared
    against ``atol_factor * L`` (``L^3`` for volumes), ``L`` being the largest
    bounding-box diagonal, because recomputation changes only the accumulation
    order.  ``bitwise`` flags are reported for information.
    """
    report = {}
    pts_a = np.asarray(a.points, dtype=float)
    pts_b = np.asarray(b.points, dtype=float)
    scale = max(float(np.linalg.norm(pts_a.max(axis=0) - pts_a.min(axis=0)) or 1.0),
                float(np.linalg.norm(pts_b.max(axis=0) - pts_b.min(axis=0)) or 1.0))

    def compare(name, x, y, power=1):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            report[name] = dict(equal=False, reason=f'shape {x.shape} != {y.shape}')
            return None
        dev = float(np.max(np.abs(x - y))) if x.size else 0.0
        report[name] = dict(equal=bool(dev <= atol_factor * scale ** power),
                            bitwise=bool(np.array_equal(x, y)), max_abs_deviation=dev,
                            tolerance=atol_factor * scale ** power)
        return dev

    compare('points', pts_a, pts_b)
    compare('cell_centers', a.cell_centers, b.cell_centers)
    compare('cell_volumes', a.cell_volumes, b.cell_volumes, power=3)

    def key(mesh):
        out = {}
        for f, face in enumerate(mesh.face_vertices):
            k = tuple(sorted(face))
            if k in out:
                report['face_keys_unique'] = dict(equal=False, reason=f'duplicate face {f}')
            out[k] = f
        return out

    ka, kb = key(a), key(b)
    same_keys = set(ka) == set(kb)
    report['face_sets'] = dict(equal=same_keys, count_a=len(ka), count_b=len(kb))
    nb_a, nb_b = a.neighbour_full(), b.neighbour_full()
    matched = same_keys and a.n_faces == b.n_faces
    verts_ok = own_ok = nei_ok = labels_ok = True
    face_dev = {'face_centers': 0.0, 'face_area_vectors': 0.0}
    face_exact = {'face_centers': True, 'face_area_vectors': True}
    if matched:
        for k, fa in ka.items():
            fb = kb[k]
            if a.face_vertices[fa] != b.face_vertices[fb]:
                verts_ok = False
            if int(a.owner[fa]) != int(b.owner[fb]):
                own_ok = False
            if int(nb_a[fa]) != int(nb_b[fb]):
                nei_ok = False
            if str(a.patch_labels[fa]) != str(b.patch_labels[fb]):
                labels_ok = False
            for name, arr_a, arr_b in (
                    ('face_centers', a.face_centers, b.face_centers),
                    ('face_area_vectors', a.face_area_vectors, b.face_area_vectors)):
                x, y = np.asarray(arr_a[fa], float), np.asarray(arr_b[fb], float)
                face_dev[name] = max(face_dev[name], float(np.max(np.abs(x - y))))
                face_exact[name] = face_exact[name] and bool(np.array_equal(x, y))
    else:
        verts_ok = own_ok = nei_ok = labels_ok = False
    for name, power in (('face_centers', 1), ('face_area_vectors', 2)):
        thresh = atol_factor * scale ** power
        report[name] = dict(equal=bool(matched and face_dev[name] <= thresh),
                            bitwise=bool(face_exact[name]),
                            max_abs_deviation=face_dev[name], tolerance=thresh,
                            compared_by='face vertex-set mapping')
    report['face_vertices_oriented'] = dict(equal=verts_ok)
    report['owner'] = dict(equal=own_ok)
    report['neighbour'] = dict(equal=nei_ok)
    report['patch_labels'] = dict(equal=labels_ok)
    report['patches'] = dict(
        equal=[(p['name'], p['nFaces']) for p in a.patches] ==
              [(p['name'], p['nFaces']) for p in b.patches],
        a=[(p['name'], p['type'], p['nFaces'], p['startFace']) for p in a.patches],
        b=[(p['name'], p['type'], p['nFaces'], p['startFace']) for p in b.patches])
    report['connectivity_bitwise'] = all(
        report[k]['equal'] for k in ('face_sets', 'face_vertices_oriented', 'owner',
                                     'neighbour', 'patch_labels', 'patches'))
    report['passed'] = report['connectivity_bitwise'] and all(
        report[k]['equal'] for k in ('points', 'face_centers', 'face_area_vectors',
                                     'cell_centers', 'cell_volumes'))
    return report


def roundtrip_gate(mesh, case_dir):
    """native mesh -> ``openfoam_case.export_native_mesh`` -> read back -> compare.

    Returns ``(parsed_mesh, report)``; ``report['passed']`` is the gate.
    """
    from tensorfvm.openfoam_case import export_native_mesh
    case_dir = Path(case_dir)
    export_native_mesh(mesh, case_dir)
    parsed = read_polymesh(case_dir)
    original = native_to_polymesh(mesh)
    return parsed, compare_meshes(original, parsed)


def native_to_polymesh(mesh, *, prefix=''):
    """Re-label a ``NativeMesh3D`` in :class:`PolyMesh` terms (no file round trip).

    Faces are re-ordered to the OpenFOAM convention (internal faces first, then
    contiguous boundary patches) so the result is directly comparable with what
    ``read_polymesh`` produces from ``export_native_mesh`` output.
    """
    labels = np.asarray([str(s) for s in mesh.patch_labels], dtype=object)
    neighbour = np.asarray(mesh.neighbor, dtype=int)
    internal = np.flatnonzero(neighbour >= 0)
    boundary = np.flatnonzero(neighbour < 0)
    names = sorted(set(str(s) for s in labels[boundary]))
    order = list(internal)
    patches = []
    for name in names:
        ids = np.flatnonzero((neighbour < 0) & (labels == name))
        patches.append(dict(name=prefix + name, type='patch', nFaces=int(len(ids)),
                            startFace=len(order)))
        order.extend(int(i) for i in ids)
    return PolyMesh(
        points=np.array(mesh.vertices, dtype=float),
        face_vertices=tuple(tuple(int(v) for v in mesh.face_vertices[f]) for f in order),
        owner=np.array([int(mesh.owner[f]) for f in order], dtype=int),
        neighbour=np.array([int(neighbour[f]) for f in internal], dtype=int),
        patches=tuple(patches),
        face_centers=np.array([mesh.face_centers[f] for f in order], dtype=float),
        face_area_vectors=np.array([mesh.area_vectors[f] for f in order], dtype=float),
        cell_faces=tuple(), cell_vertices=tuple(),
        cell_types=tuple(['unknown'] * len(mesh.cells)),
        cell_centers=np.array(mesh.cell_centers, dtype=float),
        cell_volumes=np.array(mesh.cell_volumes, dtype=float),
        patch_labels=np.array([labels[f] for f in order], dtype=object),
        case_dir='', n_internal_faces=int(len(internal)))


def _face_data(mesh):
    """Precompute the geometric factors of the FV laplacian."""
    n_faces = mesh.n_faces
    internal = mesh.n_internal_faces
    owner, neighbour = mesh.owner, mesh.neighbour
    C = mesh.cell_centers
    Cf = mesh.face_centers
    Sf = mesh.face_area_vectors
    d = np.empty((n_faces, 3))
    d[:internal] = C[neighbour] - C[owner[:internal]]
    d[internal:] = Cf[internal:] - C[owner[internal:]]
    Sfd = np.einsum('ij,ij->i', Sf, d)
    if np.any(Sfd <= 0.0):
        raise ValueError('non-positive Sf.d: mesh violates convexity/orientation assumptions')
    area = np.linalg.norm(Sf, axis=1)
    orth = area ** 2 / Sfd                       # over-relaxed orthogonal coefficient
    correction = Sf - orth[:, None] * d          # non-orthogonal correction direction
    # linear interpolation weight of the neighbour / boundary value at the face
    w = np.ones(n_faces)
    if internal:
        dn = np.linalg.norm(Cf[:internal] - C[neighbour], axis=1)
        dp = np.linalg.norm(Cf[:internal] - C[owner[:internal]], axis=1)
        w[:internal] = dn / (dp + dn)
    return dict(d=d, Sfd=Sfd, area=area, orth=orth, correction=correction, w=w)


def _patch_kind(bc, name):
    if name not in bc:
        raise KeyError(f'boundary condition missing for patch {name!r}')
    kind = bc[name][0]
    if kind not in ('fixedValue', 'zeroGradient'):
        raise ValueError(f'unsupported bc type {kind!r} for patch {name!r}')
    return kind


def _face_values(mesh, phi, bc, fd):
    """Face values by linear interpolation; boundary faces take the patch value."""
    internal, w = mesh.n_internal_faces, fd['w']
    pf = np.empty(mesh.n_faces)
    pf[:internal] = (1.0 - w[:internal]) * phi[mesh.owner[:internal]] \
        + w[:internal] * phi[mesh.neighbour]
    pf[internal:] = phi[mesh.owner[internal:]]
    for pg in mesh.patches:
        if _patch_kind(bc, pg['name']) == 'fixedValue':
            pf[pg['startFace']:pg['startFace'] + pg['nFaces']] = float(bc[pg['name']][1])
    return pf


def cell_gradient(mesh, phi, bc, face_data=None):
    """Gauss gradient ``(1/V) sum_f Sf phi_f`` with linear face interpolation."""
    fd = face_data or _face_data(mesh)
    phi = np.asarray(phi, dtype=float)
    if phi.shape != (mesh.n_cells,):
        raise ValueError(f'phi must have shape ({mesh.n_cells},)')
    pf = _face_values(mesh, phi, bc, fd)
    weighted = mesh.face_area_vectors * pf[:, None]
    ii = np.arange(mesh.n_internal_faces)
    n = mesh.n_cells
    grad = np.empty((n, 3))
    for k in range(3):
        grad[:, k] = (np.bincount(mesh.owner, weights=weighted[:, k], minlength=n)
                      - np.bincount(mesh.neighbour, weights=weighted[ii, k], minlength=n))
    return grad / mesh.cell_volumes[:, None]


def solve_poisson(mesh, source, bc, *, gamma=1.0, n_correctors=3, tol=1e-12,
                  phi0=None):
    """Solve ``div(gamma grad phi) = source`` on ``mesh``.

    Finite-volume discretisation in OpenFOAM's laplacian form: an implicit
    over-relaxed orthogonal part plus an explicit non-orthogonal correction that
    is refreshed by ``n_correctors`` Picard passes (the analogue of
    ``nNonOrthogonalCorrectors``).  ``source`` is the per-cell *volume-averaged*
    right-hand side; ``bc`` maps patch name to ``('fixedValue', value)`` or
    ``('zeroGradient',)``.

    Returns ``phi`` together with diagnostics that make the cross-check
    auditable: ``residual`` (linear-system residual of the final corrector),
    ``operator_residual`` (max |discrete laplacian - source| normalised, i.e.
    how well the *corrected* equation is satisfied), ``conservation_budget``
    (|outward boundary flux - sum of sources| / |sum of sources|) and the
    maximum non-orthogonality the operator had to handle.
    """
    source = np.asarray(source, dtype=float)
    n_cells, n_faces, internal = mesh.n_cells, mesh.n_faces, mesh.n_internal_faces
    if source.shape != (n_cells,):
        raise ValueError(f'source must have shape ({n_cells},)')
    if not any(_patch_kind(bc, pg['name']) == 'fixedValue' for pg in mesh.patches):
        raise ValueError('Poisson problem needs at least one fixedValue patch')
    fd = _face_data(mesh)
    owner, neighbour = mesh.owner, mesh.neighbour
    ii = np.arange(internal)
    volume, orth, correction, w = (mesh.cell_volumes, fd['orth'], fd['correction'],
                                   fd['w'])
    fixed = [(slice(pg['startFace'], pg['startFace'] + pg['nFaces']),
              float(bc[pg['name']][1])) for pg in mesh.patches
             if _patch_kind(bc, pg['name']) == 'fixedValue']
    zero = [slice(pg['startFace'], pg['startFace'] + pg['nFaces']) for pg in mesh.patches
            if _patch_kind(bc, pg['name']) == 'zeroGradient']

    zero_mask = np.zeros(n_faces, dtype=bool)
    for sl in zero:
        zero_mask[sl] = True

    # implicit matrix: (A phi)_P = sum_f gamma*orth*(phi_P - phi_other)
    diag = np.zeros(n_cells)
    np.add.at(diag, owner[ii], gamma * orth[ii])
    np.add.at(diag, neighbour, gamma * orth[ii])
    for sl, _ in fixed:
        np.add.at(diag, owner[sl], gamma * orth[sl])
    a = sp.csc_matrix(
        (np.concatenate([diag, -gamma * orth[ii], -gamma * orth[ii]]),
         (np.concatenate([np.arange(n_cells), owner[ii], neighbour]),
          np.concatenate([np.arange(n_cells), neighbour, owner[ii]]))),
        shape=(n_cells, n_cells))
    solve = spla.splu(a).solve

    def face_gradient(phi):
        g = cell_gradient(mesh, phi, bc, face_data=fd)
        gf = np.empty((n_faces, 3))
        gf[:internal] = (1.0 - w[:internal])[:, None] * g[owner[:internal]] \
            + w[:internal, None] * g[neighbour]
        gf[internal:] = g[owner[internal:]]
        return gf

    def laplacian(phi):
        """Discrete div(gamma grad phi) per cell (positive = net outflow)."""
        gf = face_gradient(phi)
        f = gamma * np.einsum('ij,ij->i', correction, gf)
        f[ii] += gamma * orth[ii] * (np.take(phi, neighbour) - np.take(phi, owner[ii]))
        for sl, value in fixed:
            f[sl] += gamma * orth[sl] * (value - np.take(phi, owner[sl]))
        if zero_mask.any():
            f[zero_mask] = 0.0
        out = np.bincount(owner, weights=f, minlength=n_cells) \
            - np.bincount(neighbour, weights=f[ii], minlength=n_cells)
        return out, f

    phi = np.full(n_cells, np.mean([v for _, v in fixed])) if phi0 is None \
        else np.array(phi0, dtype=float)
    rhs_source = -source * volume            # A = -L, hence the sign
    residual = np.inf
    for _ in range(n_correctors + 1):
        corr = gamma * np.einsum('ij,ij->i', correction, face_gradient(phi))
        rhs = np.array(rhs_source)
        np.add.at(rhs, owner[ii], corr[ii])
        np.add.at(rhs, neighbour, -corr[ii])
        for sl, value in fixed:
            np.add.at(rhs, owner[sl], gamma * orth[sl] * value + corr[sl])
        for sl in zero:
            np.add.at(rhs, owner[sl], corr[sl])
        new_phi = solve(rhs)
        if not np.isfinite(new_phi).all():
            raise RuntimeError('linear solve produced non-finite values')
        residual = float(np.linalg.norm(a @ new_phi - rhs) / max(np.linalg.norm(rhs), 1e-300))
        delta = float(np.max(np.abs(new_phi - phi)))
        phi = new_phi
        if delta < tol * max(1.0, float(np.max(np.abs(phi)))):
            break

    lap, fflux = laplacian(phi)
    op_resid = float(np.max(np.abs(lap - source * volume))
                     / max(np.max(np.abs(source * volume)), 1e-300))
    boundary_flux = float(sum(fflux[sl].sum() for sl, _ in fixed))
    imbalance = float(abs(boundary_flux - (source * volume).sum())
                      / max(abs((source * volume).sum()), 1e-300))
    nonorth = np.degrees(np.arccos(np.clip(
        fd['Sfd'] / (fd['area'] * np.linalg.norm(fd['d'], axis=1)), -1.0, 1.0)))
    return dict(phi=phi, residual=residual, operator_residual=op_resid,
                conservation_budget=imbalance, boundary_flux=boundary_flux,
                correctors=n_correctors,
                maximum_nonorthogonality_degrees=float(nonorth.max()))


def l2_error(mesh, phi, exact):
    """Volume-weighted relative L2 error of a cell-centred field."""
    exact = np.asarray(exact, dtype=float)
    num = float(np.sum(mesh.cell_volumes * (phi - exact) ** 2))
    den = float(np.sum(mesh.cell_volumes * exact ** 2))
    return float(np.sqrt(num / den))


def linf_error(mesh, phi, exact):
    return float(np.max(np.abs(np.asarray(phi) - np.asarray(exact))))


# --------------------------------------------------------------------------
# structured-box primitives (self-contained generator for fixtures/studies)
# --------------------------------------------------------------------------
_SIDE_FACES = {
    'xmin': lambda i, j, k: [(i, j, k), (i, j + 1, k), (i, j + 1, k + 1), (i, j, k + 1)],
    'xmax': lambda i, j, k: [(i + 1, j, k), (i + 1, j, k + 1), (i + 1, j + 1, k + 1),
                             (i + 1, j + 1, k)],
    'ymin': lambda i, j, k: [(i, j, k), (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j, k)],
    'ymax': lambda i, j, k: [(i, j + 1, k), (i + 1, j + 1, k), (i + 1, j + 1, k + 1),
                             (i, j + 1, k + 1)],
    'zmin': lambda i, j, k: [(i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k)],
    'zmax': lambda i, j, k: [(i, j, k + 1), (i, j + 1, k + 1), (i + 1, j + 1, k + 1),
                             (i + 1, j, k + 1)],
}
_SIDE_OUTWARD = {'xmin': (-1, 0, 0), 'xmax': (1, 0, 0), 'ymin': (0, -1, 0),
                 'ymax': (0, 1, 0), 'zmin': (0, 0, -1), 'zmax': (0, 0, 1)}


def structured_box_primitives(nx, ny, nz, *, origin=(0.0, 0.0, 0.0),
                              size=(1.0, 1.0, 1.0), shear=0.0, jitter=0.0,
                              seed=0, sides=None):
    """Raw polyMesh primitives for a (optionally sheared/jittered) hex box.

    Self-contained: no dependency on :class:`PolyMesh` or on
    ``openfoam_case.export_native_mesh``, so meshes built here are *independent*
    input for the reader -- the same role ``blockMesh`` plays.  ``shear`` adds
    ``x += shear*y`` (a volume-preserving, genuinely non-orthogonal map);
    ``jitter`` randomly perturbs the *interior* points by ``jitter*h`` so the
    grid becomes irregular while the domain stays an exact box.  ``sides`` maps
    ``{'xmin','xmax','ymin','ymax','zmin','zmax'}`` to patch names, several
    sides may share one name.

    Returns ``(points, faces, owner, neighbour, patches)`` ready for
    :func:`write_polymesh`.
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) < 1:
        raise ValueError('at least one cell per direction is required')
    origin = np.asarray(origin, dtype=float)
    size = np.asarray(size, dtype=float)
    sides = sides or {}

    order = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nz + 1, ny + 1, nx + 1)

    def vid(i, j, k):
        return int(order[k, j, i])

    pts = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                pts[vid(i, j, k)] = origin + size * np.array(
                    [i / nx, j / ny, k / nz], dtype=float)
    pts[:, 0] += shear * (pts[:, 1] - origin[1])
    if jitter:
        rng = np.random.default_rng(seed)
        h = size / np.array([nx, ny, nz], dtype=float)
        interior = np.zeros(len(pts), dtype=bool)
        for k in range(1, nz):
            for j in range(1, ny):
                for i in range(1, nx):
                    interior[vid(i, j, k)] = True
        if interior.any():
            pts[interior] += (rng.random((int(interior.sum()), 3)) - 0.5) * h * jitter

    faces, owner, neighbour = [], [], []
    lookup = {}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = i + nx * (j + ny * k)
                for side, maker in _SIDE_FACES.items():
                    quad = [vid(*g) for g in maker(i, j, k)]
                    x = pts[quad]
                    sf = 0.5 * np.cross(x[1] - x[0], x[2] - x[0]) \
                        + 0.5 * np.cross(x[2] - x[0], x[3] - x[0])
                    if np.dot(sf, np.array(_SIDE_OUTWARD[side], dtype=float)) < 0.0:
                        quad.reverse()
                    key = tuple(sorted(quad))
                    if key in lookup:
                        f = lookup.pop(key)
                        neighbour[f] = cell
                    else:
                        lookup[key] = len(faces)
                        faces.append(tuple(quad))
                        owner.append(cell)
                        neighbour.append(-1)
    internal = [f for f in range(len(faces)) if neighbour[f] >= 0]
    boundary = {}
    for key, f in lookup.items():
        boundary.setdefault(_patch_of(faces[f], pts, origin, size, shear, sides), []).append(f)
    # internal faces sorted by (owner, neighbour) as OpenFOAM's own ordering is
    order_int = sorted(internal, key=lambda f: (owner[f], neighbour[f]))
    order_bnd = []
    patches = []
    start = len(order_int)
    for name in sorted(boundary):
        ids = sorted(boundary[name])
        patches.append(dict(name=name, type='wall' if name == 'wall' else 'patch',
                            nFaces=len(ids), startFace=start))
        start += len(ids)
        order_bnd.extend(ids)
    perm = order_int + order_bnd
    return (pts, [faces[f] for f in perm], [owner[f] for f in perm],
            [neighbour[f] for f in internal], patches)


def _patch_of(face, pts, origin, size, shear, sides):
    """Classify a boundary quad by the box side it lies on."""
    x = pts[list(face)]
    rel = (x - origin)
    tol = 1e-12 * max(size.max(), 1.0)
    on = {}
    on['xmin'] = np.allclose(rel[:, 0] - shear * rel[:, 1], 0.0, atol=tol)
    on['xmax'] = np.allclose(rel[:, 0] - shear * rel[:, 1], size[0], atol=tol)
    on['ymin'] = np.allclose(rel[:, 1], 0.0, atol=tol)
    on['ymax'] = np.allclose(rel[:, 1], size[1], atol=tol)
    on['zmin'] = np.allclose(rel[:, 2], 0.0, atol=tol)
    on['zmax'] = np.allclose(rel[:, 2], size[2], atol=tol)
    hits = [k for k, v in on.items() if v]
    if len(hits) != 1:
        raise ValueError(f'boundary face is not on exactly one box side: {hits}')
    default = {'xmin': 'inlet', 'xmax': 'outlet'}.get(hits[0], 'wall')
    return sides.get(hits[0], default)
