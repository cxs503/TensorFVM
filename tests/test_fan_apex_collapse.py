"""Gmsh boundary-layer fan apex triangles are collapsed into wall fillets.

A 2-D Gmsh fan at a convex wall corner starts with a ring of triangles whose
apex is the corner node.  They touch the wall at one vertex only and own no
wall face, so no boundary condition ever reaches them (HLPW4 30P30N runs showed
|u| up to 2.6 U inside them).  ``ThreeElementMesh`` removes them and promotes
the exposed first-ring edges to wall faces.
"""

from pathlib import Path
from types import SimpleNamespace
import math
import tempfile
import unittest

from tensorfvm.multi_element import ThreeElementMesh


def _fan_mesh(scale):
    h = 1e-5 * scale
    s = 5e-5 * scale
    c = h / math.sqrt(2)
    nodes = [(-s, 0), (0, 0), (0, -s), (0, h), (c, c), (h, 0), (-s, h), (h, -s),
             (0, 2 * h), (2 * c, 2 * c), (2 * h, 0)]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "7",
             '1 1 "inlet"', '1 2 "outlet"', '1 3 "far-field"', '1 4 "slat"',
             '1 5 "main"', '1 6 "flap"', '2 7 "fluid"', "$EndPhysicalNames",
             "$Nodes", str(len(nodes))]
    lines += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes)]
    lines += ["$EndNodes", "$Elements", "16",
              "1 1 2 5 1 1 2", "2 1 2 5 2 2 3", "3 1 2 1 3 1 7", "4 1 2 3 4 7 4",
              "5 1 2 3 5 4 9", "6 1 2 4 6 9 10", "7 1 2 6 7 10 11", "8 1 2 2 8 11 6",
              "9 1 2 2 9 6 8", "10 1 2 2 10 8 3",
              "11 3 2 7 11 1 2 4 7", "12 3 2 7 12 2 3 8 6", "13 2 2 7 13 2 4 5",
              "14 2 2 7 14 2 5 6", "15 3 2 7 15 4 5 10 9", "16 3 2 7 16 5 6 11 10",
              "$EndElements"]
    return "\n".join(lines) + "\n"


class FanApexCollapseTests(unittest.TestCase):
    def _load(self, scale, **options):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fan.msh"
            path.write_text(_fan_mesh(scale), encoding="utf-8")
            return ThreeElementMesh(SimpleNamespace(mesh_file=path, device="cpu", **options))

    def test_apex_triangles_become_wall_fillet(self):
        raw = self._load(1.0, collapse_fan_apex_cells=False)
        mesh = self._load(1.0)
        self.assertEqual(raw.collapsed_fan_apex_cells, 0)
        self.assertEqual(len(raw.cells), 6)
        self.assertEqual(int(raw.masks["main"].sum()), 2)
        self.assertEqual(mesh.collapsed_fan_apex_cells, 2)
        self.assertEqual(len(mesh.cells), 4)
        self.assertTrue(all(len(cell) == 4 for cell in mesh.cells))
        self.assertEqual(int(mesh.masks["main"].sum()), 6)
        self.assertEqual(int(mesh.masks["wall"].sum()), 8)
        self.assertEqual(mesh.fillet_wall_nodes, (3, 4, 5))
        # Solid grows by the two apex triangles only.
        removed = float(raw.volumes.sum() - mesh.volumes.sum())
        self.assertAlmostEqual(removed, 2 * 0.5 * 1e-5 * 1e-5 * math.sin(math.pi / 4), delta=1e-24)
        # Every cell that touches the corner now owns a wall face.
        corner = 1
        touching = [i for i, cell in enumerate(mesh.cells) if corner in cell]
        owners = set(mesh.owner[mesh.masks["wall"]].tolist())
        self.assertEqual(len(touching), 2)
        self.assertTrue(set(touching) <= owners)
        # The fillet arc faces (both end points at radius h) face the corner:
        # their outward normal points out of the fluid, towards the origin.
        radii = mesh.face_vertices.norm(dim=-1)
        arc = mesh.masks["main"] & ((radii - 1e-5).abs() < 1e-12).all(-1)
        self.assertEqual(int(arc.sum()), 2)
        for face in arc.nonzero().flatten().tolist():
            center = mesh.face_centers[face]
            self.assertLess(float((mesh.face_normals[face] * center).sum()), 0.0)
        # Other boundary groups are untouched.
        for name, count in (("inlet", 1), ("outlet", 3), ("far-field", 2), ("slat", 1), ("flap", 1)):
            self.assertEqual(int(mesh.masks[name].sum()), count, name)

    def test_large_vertex_touching_triangles_are_kept(self):
        mesh = self._load(20.0)
        self.assertEqual(mesh.collapsed_fan_apex_cells, 0)
        self.assertEqual(len(mesh.cells), 6)
        self.assertEqual(mesh.fillet_wall_nodes, ())


if __name__ == "__main__":
    unittest.main()
