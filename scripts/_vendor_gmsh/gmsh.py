"""Pure-Python subset of the Gmsh Python API, enough for NativeMesh3D.from_gmsh.

The real ``gmsh`` wheel needs libGLU at import time; this module parses Gmsh
MSH 4.1 ASCII files directly and implements exactly the API surface used by
``tensorfvm.native_mesh3d.NativeMesh3D.from_gmsh``.  Used only when the real
Gmsh runtime is unavailable (diagnostic environments); production runs should
install the official ``gmsh`` package.

Supports: initialize/isInitialized/open/finalize, option.setNumber (ignored),
model.mesh.getNodes, model.mesh.getElements(dim[, entity]),
model.getPhysicalGroups, model.getPhysicalName,
model.getEntitiesForPhysicalGroup.  Only MSH 4.1 ASCII.
"""
from pathlib import Path

import numpy as np

_INITIALIZED = False


class option:
    @staticmethod
    def setNumber(*_args, **_kwargs):
        return None


class _Mesh:
    def __init__(self, state):
        self._state = state

    def getNodes(self):
        s = self._state
        return s.node_tags.copy(), s.node_xyz.ravel().copy(), np.zeros(0)

    def getElements(self, dim, entity=None):
        types, tags, nodes = [], [], []
        for (d, e, et), block in self._state.element_blocks.items():
            if d != dim or (entity is not None and e != entity):
                continue
            types.append(et)
            tags.append(block['tags'])
            nodes.append(block['nodes'])
        return types, tags, nodes


class _Model:
    def __init__(self, state):
        self.mesh = _Mesh(state)
        self._names = state.physical_names
        self._entities = state.physical_entities

    def getPhysicalGroups(self, dim):
        return [(dim, tag) for (d, tag) in self._names if d == dim]

    def getPhysicalName(self, dim, tag):
        return self._names.get((dim, tag), '')

    def getEntitiesForPhysicalGroup(self, dim, tag):
        return list(self._entities.get((dim, tag), []))


class _NullModel:
    def __init__(self):
        self.mesh = _Mesh(type('Empty', (), dict(
            node_tags=np.zeros(0, dtype=np.int64),
            node_xyz=np.zeros((0, 3)),
            element_blocks={}))())
        self._names = {}
        self._entities = {}

    def getPhysicalGroups(self, dim):
        return []

    def getPhysicalName(self, dim, tag):
        return ''

    def getEntitiesForPhysicalGroup(self, dim, tag):
        return []


model = _NullModel()


def isInitialized():
    return _INITIALIZED


def initialize():
    global _INITIALIZED, model
    if _INITIALIZED:
        raise RuntimeError('gmsh stub already initialized')
    _INITIALIZED = True
    model = _NullModel()


def finalize():
    global _INITIALIZED, model
    _INITIALIZED = False
    model = _NullModel()


def open(path):  # noqa: A001 - mirrors the Gmsh API name
    global model
    if not _INITIALIZED:
        raise RuntimeError('initialize() the gmsh stub before open()')
    lines = Path(path).read_text().splitlines()

    def section(name):
        start = lines.index(f'${name}') + 1
        return start, lines.index(f'$End{name}')

    def numbers(start, end):
        for line in lines[start:end]:
            for token in line.split():
                yield token

    fmt = lines[lines.index('$MeshFormat') + 1].split()
    if fmt[:2] != ['4.1', '0']:
        raise ValueError(f'unsupported MSH format: {fmt}')

    names = {}
    if '$PhysicalNames' in lines:
        start, end = section('PhysicalNames')
        for line in lines[start + 1:start + 1 + int(lines[start])]:
            dim, tag, name = line.split(None, 2)
            names[(int(dim), int(tag))] = name.strip().strip('"')

    # Entities: bbox width 3 for points, 6 for curves/surfaces/volumes.
    entity_physical = {}
    physical_entities = {}
    start, end = section('Entities')
    counts = [int(v) for v in lines[start].split()]
    cursor = start + 1
    for dim, count in enumerate(counts):
        for _ in range(count):
            parts = lines[cursor].split()
            cursor += 1
            tag = int(parts[0])
            num_phys = int(parts[4 if dim == 0 else 7])
            phys = [int(parts[(4 if dim == 0 else 7) + 1 + k]) for k in range(num_phys)]
            entity_physical[(dim, tag)] = phys
            for p in phys:
                physical_entities.setdefault((dim, p), []).append(tag)

    start, end = section('Nodes')
    header = [int(v) for v in numbers(start, start + 1)]
    cursor = start + 1
    num_blocks, num_nodes = header[0], header[1]
    node_tags = np.empty(num_nodes, dtype=np.int64)
    node_xyz = np.empty((num_nodes, 3), dtype=float)
    offset = 0
    for _ in range(num_blocks):
        parts = [int(v) for v in numbers(cursor, cursor + 1)]
        cursor += 1
        num_in_block = parts[3]
        tags = [int(v) for v in numbers(cursor, cursor + num_in_block)]
        cursor += num_in_block
        coords = [float(v) for v in numbers(cursor, cursor + num_in_block)]
        cursor += num_in_block
        node_tags[offset:offset + num_in_block] = tags
        node_xyz[offset:offset + num_in_block] = np.asarray(coords).reshape(-1, 3)
        offset += num_in_block

    widths = {2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5}
    element_blocks = {}
    start, end = section('Elements')
    header = [int(v) for v in numbers(start, start + 1)]
    cursor = start + 1
    for _ in range(header[0]):
        parts = [int(v) for v in numbers(cursor, cursor + 1)]
        cursor += 1
        dim, entity, et, count = parts[:4]
        if et not in widths:
            raise ValueError(f'unsupported Gmsh element type {et}')
        width = widths[et]
        values = [int(v) for v in numbers(cursor, cursor + count)]
        cursor += count
        rows = np.asarray(values, dtype=np.int64).reshape(-1, 1 + width)
        element_blocks[(dim, entity, et)] = dict(tags=rows[:, 0], nodes=rows[:, 1:])

    state = type('GmshState', (), {})()
    state.node_tags = node_tags
    state.node_xyz = node_xyz
    state.element_blocks = element_blocks
    state.physical_names = names
    state.physical_entities = physical_entities
    model = _Model(state)
