"""Gmsh v2 mesh reader for the 30P30N multi-element benchmark.

The mesh generator is an optional preprocessing tool; the solver itself needs
only PyTorch.  Boundary face names in the Gmsh file are ``inlet``, ``outlet``,
``far-field``, ``slat``, ``main`` and ``flap``.
"""

from pathlib import Path
import math
import shlex

import torch


_BOUNDARY_ALIASES = {
    "inlet": "inlet",
    "outlet": "outlet",
    "far-field": "far-field",
    "farfield": "far-field",
    "top": "far-field",
    "bottom": "far-field",
    "slat": "slat",
    "slat-wall": "slat",
    "main": "main",
    "main-wall": "main",
    "flap": "flap",
    "flap-wall": "flap",
}


class ThreeElementMesh:
    """Unstructured triangle/quad mesh with separately tagged 30P30N elements."""

    def __init__(self, config):
        path = Path(config.mesh_file)
        if not path.is_file():
            raise ValueError(f"three-element mesh file does not exist: {path}")
        sections = self._read_sections(path)
        if "MeshFormat" not in sections:
            raise ValueError("not a Gmsh mesh: missing $MeshFormat section")
        mesh_format = sections["MeshFormat"][0].split()
        if len(mesh_format) < 2 or not mesh_format[0].startswith("2.") or mesh_format[1] != "0":
            raise ValueError("three-element solver requires an ASCII Gmsh v2 mesh")
        physical_names = self._physical_names(sections.get("PhysicalNames", []))
        node_ids, nodes = self._nodes(sections.get("Nodes", []))
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}
        line_groups, polygons = self._elements(sections.get("Elements", []),
                                               physical_names, node_index)
        if not polygons:
            raise ValueError("Gmsh mesh contains no linear triangle or quadrilateral cells")
        missing = {"slat", "main", "flap", "inlet", "outlet", "far-field"} - set(
            line_groups.values()
        )
        if missing:
            raise ValueError("Gmsh mesh is missing physical boundary groups: "
                             + ", ".join(sorted(missing)))

        coordinates = [nodes[index] for index in range(len(nodes))]
        cell_centers, cell_areas, cell_nodes = [], [], []
        for cell in polygons:
            points = [coordinates[index] for index in cell]
            signed_area = self._signed_area(points)
            if abs(signed_area) <= 1e-18:
                raise ValueError("Gmsh mesh contains a zero-area cell")
            if signed_area < 0:
                cell = list(reversed(cell))
                points = list(reversed(points))
                signed_area = -signed_area
            centroid, area = self._centroid(points)
            if area <= 0:
                raise ValueError("Gmsh mesh contains an invalid cell orientation")
            cell_nodes.append(cell)
            cell_centers.append(centroid)
            cell_areas.append(area)

        owners, neighbors, face_node_pairs = [], [], []
        known_faces = {}
        for cell_id, cell in enumerate(cell_nodes):
            for start, stop in zip(cell, cell[1:] + cell[:1]):
                key = (min(start, stop), max(start, stop))
                if key in known_faces:
                    face_id = known_faces[key]
                    if neighbors[face_id] >= 0:
                        raise ValueError("Gmsh mesh has a non-manifold face shared by >2 cells")
                    neighbors[face_id] = cell_id
                else:
                    known_faces[key] = len(owners)
                    owners.append(cell_id)
                    neighbors.append(-1)
                    # The CCW owner-cell edge gives an outward area vector.
                    face_node_pairs.append((start, stop))

        labels = []
        for face_id, (start, stop) in enumerate(face_node_pairs):
            key = (min(start, stop), max(start, stop))
            if neighbors[face_id] >= 0:
                labels.append("interior")
                continue
            label = line_groups.get(key)
            if label is None:
                raise ValueError("Gmsh mesh has an untagged boundary edge")
            labels.append(label)

        device = torch.device(config.device)
        opts = {"dtype": torch.float64, "device": device}
        self.vertices = torch.tensor(coordinates, **opts)
        self.cell_nodes = tuple(tuple(cell) for cell in cell_nodes)
        self.cells = self.cell_nodes
        self.centers = torch.tensor(cell_centers, **opts).unsqueeze(0)
        self.volumes = torch.tensor(cell_areas, **opts).unsqueeze(0)
        self.field_shape = (1, len(cell_nodes))
        self.owner = torch.tensor(owners, dtype=torch.long, device=device)
        self.neighbor = torch.tensor(neighbors, dtype=torch.long, device=device)
        self.boundary_labels = tuple(labels)
        endpoints = torch.tensor(face_node_pairs, dtype=torch.long, device=device)
        self.face_vertices = self.vertices[endpoints]
        self.face_centers = self.face_vertices.mean(1)
        edge = self.face_vertices[:, 1] - self.face_vertices[:, 0]
        self.face_area_vectors = torch.stack((edge[:, 1], -edge[:, 0]), -1)
        self.face_lengths = torch.linalg.vector_norm(self.face_area_vectors, dim=-1)
        if not bool((self.face_lengths > 0).all()):
            raise ValueError("Gmsh mesh contains a zero-length face")
        self.face_normals = self.face_area_vectors / self.face_lengths[:, None]
        self.interior = self.neighbor >= 0
        self.boundary = ~self.interior
        names = ("inlet", "outlet", "far-field", "slat", "main", "flap")
        self.masks = {
            name: torch.tensor([label == name for label in labels],
                               dtype=torch.bool, device=device)
            for name in names
        }
        self.masks["wall"] = self.masks["slat"] | self.masks["main"] | self.masks["flap"]
        self.masks["airfoil"] = self.masks["wall"]
        self.masks["cylinder"] = torch.zeros_like(self.boundary)

    @staticmethod
    def _read_sections(path: Path) -> dict[str, list[str]]:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
        sections: dict[str, list[str]] = {}
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line.startswith("$") or line.startswith("$End"):
                index += 1
                continue
            name = line[1:]
            end = f"$End{name}"
            index += 1
            content = []
            while index < len(lines) and lines[index].strip() != end:
                stripped = lines[index].strip()
                if stripped:
                    content.append(stripped)
                index += 1
            if index == len(lines):
                raise ValueError(f"unterminated Gmsh section ${name}")
            sections[name] = content
            index += 1
        return sections

    @staticmethod
    def _physical_names(lines: list[str]) -> dict[tuple[int, int], str]:
        if not lines:
            return {}
        try:
            count = int(lines[0])
            result = {}
            for line in lines[1:1 + count]:
                fields = shlex.split(line)
                dimension, tag = int(fields[0]), int(fields[1])
                raw_name = fields[2].strip().lower().replace("_", "-").replace(" ", "-")
                name = _BOUNDARY_ALIASES.get(raw_name)
                if name is not None:
                    result[(dimension, tag)] = name
            return result
        except (ValueError, IndexError) as exc:
            raise ValueError("malformed Gmsh $PhysicalNames section") from exc

    @staticmethod
    def _nodes(lines: list[str]) -> tuple[list[int], list[tuple[float, float]]]:
        if not lines:
            raise ValueError("Gmsh mesh contains no $Nodes section")
        try:
            count = int(lines[0])
            ids, nodes = [], []
            for line in lines[1:1 + count]:
                fields = line.split()
                node_id = int(fields[0])
                x, y = float(fields[1]), float(fields[2])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError("nonfinite node coordinate")
                ids.append(node_id)
                nodes.append((x, y))
            if len(ids) != count or len(set(ids)) != count:
                raise ValueError("incomplete or duplicate Gmsh node list")
            return ids, nodes
        except (ValueError, IndexError) as exc:
            raise ValueError("malformed Gmsh $Nodes section") from exc

    @staticmethod
    def _elements(lines: list[str], physical_names: dict[tuple[int, int], str],
                  node_index: dict[int, int]):
        if not lines:
            raise ValueError("Gmsh mesh contains no $Elements section")
        try:
            count = int(lines[0])
            line_groups, polygons = {}, []
            for line in lines[1:1 + count]:
                fields = [int(value) for value in line.split()]
                element_type, tag_count = fields[1], fields[2]
                tags = fields[3:3 + tag_count]
                raw_nodes = fields[3 + tag_count:]
                if element_type == 1:
                    if len(raw_nodes) != 2:
                        raise ValueError("only linear Gmsh boundary edges are supported")
                    if not tags:
                        raise ValueError("Gmsh boundary edge has no physical tag")
                    name = physical_names.get((1, tags[0]))
                    if name is None:
                        raise ValueError("Gmsh boundary edge has an unknown physical name")
                    edge = tuple(node_index[node] for node in raw_nodes)
                    key = (min(edge), max(edge))
                    previous = line_groups.get(key)
                    if previous is not None and previous != name:
                        raise ValueError("Gmsh boundary edge belongs to conflicting physical groups")
                    line_groups[key] = name
                elif element_type in (2, 3):
                    expected = 3 if element_type == 2 else 4
                    if len(raw_nodes) != expected:
                        raise ValueError("only linear Gmsh triangles and quadrilaterals are supported")
                    polygons.append([node_index[node] for node in raw_nodes])
            if len(lines[1:1 + count]) != count:
                raise ValueError("incomplete Gmsh element list")
            return line_groups, polygons
        except (ValueError, IndexError, KeyError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("only "):
                raise
            raise ValueError("malformed or unsupported Gmsh $Elements section") from exc

    @staticmethod
    def _signed_area(points: list[tuple[float, float]]) -> float:
        origin=points[0]
        local=[(x-origin[0],y-origin[1]) for x,y in points]
        return 0.5 * math.fsum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(local, local[1:] + local[:1])
        )

    @classmethod
    def _centroid(cls, points: list[tuple[float, float]]):
        # Thin first-layer fans can have area ~1e-13 at x~1. Global shoelace
        # sums lose digits and can put their centroid outside the triangle.
        # Translate before products; restore the origin only after division.
        origin=points[0]
        points=[(x-origin[0],y-origin[1]) for x,y in points]
        cross = [x0 * y1 - x1 * y0
                 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])]
        area = 0.5 * math.fsum(cross)
        center = (
            origin[0]+math.fsum((p[0] + q[0]) * cross_value
                for p, q, cross_value in zip(points, points[1:] + points[:1], cross))
            / (6 * area),
            origin[1]+math.fsum((p[1] + q[1]) * cross_value
                for p, q, cross_value in zip(points, points[1:] + points[:1], cross))
            / (6 * area),
        )
        return center, area
