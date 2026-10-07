"""Regression tests for the 30P30N multi-element mesh and case helpers."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tensorfvm.benchmark_30p30n import compare_30p30n, write_gmsh_geo
from tensorfvm.multi_element import ThreeElementMesh
from tensorfvm.solver import SimpleSolver, SolverConfig


_TINY_GMSH = '''$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
7
1 1 "inlet"
1 2 "outlet"
1 3 "far-field"
1 4 "slat"
1 5 "main"
1 6 "flap"
2 7 "fluid"
$EndPhysicalNames
$Nodes
8
1 0 0 0
2 1 0 0
3 2 0 0
4 3 0 0
5 0 1 0
6 1 1 0
7 2 1 0
8 3 1 0
$EndNodes
$Elements
11
1 1 2 4 1 1 2
2 1 2 5 2 2 3
3 1 2 6 3 3 4
4 1 2 3 4 5 6
5 1 2 3 5 6 7
6 1 2 3 6 7 8
7 1 2 1 7 1 5
8 1 2 2 8 4 8
9 3 2 7 9 1 2 6 5
10 3 2 7 10 2 3 7 6
11 3 2 7 11 3 4 8 7
$EndElements
'''


class ThreeElementMeshTests(unittest.TestCase):
    def _mesh_file(self, directory):
        path = Path(directory) / "tiny.msh"
        path.write_text(_TINY_GMSH, encoding="utf-8")
        return path

    def test_gmsh_mesh_builds_conservative_face_topology_and_named_surfaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            mesh = ThreeElementMesh(SimpleNamespace(
                mesh_file=self._mesh_file(temporary), device="cpu"
            ))
        self.assertEqual(mesh.centers.shape, (1, 3, 2))
        self.assertEqual(mesh.volumes.shape, (1, 3))
        self.assertTrue((mesh.volumes > 0).all().item())
        self.assertEqual(int(mesh.interior.sum()), 2)
        self.assertEqual(int(mesh.masks["slat"].sum()), 1)
        self.assertEqual(int(mesh.masks["main"].sum()), 1)
        self.assertEqual(int(mesh.masks["flap"].sum()), 1)
        self.assertEqual(int(mesh.masks["inlet"].sum()), 1)
        self.assertEqual(int(mesh.masks["outlet"].sum()), 1)
        self.assertEqual(int(mesh.masks["far-field"].sum()), 3)
        self.assertEqual(int(mesh.masks["wall"].sum()), 3)

    def test_solver_factory_accepts_unstructured_three_element_mesh(self):
        with tempfile.TemporaryDirectory() as temporary:
            mesh_file = self._mesh_file(temporary)
            config = SolverConfig(nx=1, ny=1, length=3, height=1,
                                  reynolds=100, cylinder_radius=None,
                                  mesh_type="three-element", mesh_file=str(mesh_file))
            solver = SimpleSolver(config)
        self.assertEqual(solver.count, 3)
        self.assertEqual(tuple(solver.u.shape), (1, 3))
        self.assertAlmostEqual(config.reference_length, config.airfoil_chord)
        coefficients = solver._aerodynamic_coefficients()
        self.assertEqual(set(coefficients), {
            "drag", "lift", "slat_drag", "slat_lift", "main_drag", "main_lift",
            "flap_drag", "flap_lift",
        })

    def test_three_element_configuration_requires_mesh_and_no_cylinder(self):
        with self.assertRaisesRegex(ValueError, "mesh_file"):
            SolverConfig(mesh_type="three-element", cylinder_radius=None)
        with tempfile.TemporaryDirectory() as temporary:
            mesh_file = self._mesh_file(temporary)
            with self.assertRaisesRegex(ValueError, "does not use cylinder_radius"):
                SolverConfig(mesh_type="three-element", mesh_file=str(mesh_file))

    def test_generated_geo_has_three_physical_walls_and_farfield(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_gmsh_geo(Path(temporary) / "case.geo")
            text = path.read_text(encoding="utf-8")
        for name in ("slat", "main", "flap", "inlet", "outlet", "far-field"):
            self.assertIn(f'Physical Curve("{name}")', text)
        self.assertIn("BoundaryLayer Field = 3;", text)
        self.assertIn("Plane Surface(1)", text)

    def test_reference_comparison_is_diagnostic_and_reports_element_force_fields(self):
        result = SimpleNamespace(
            aerodynamic_coefficients={"lift": 0.04, "drag": 0.02},
        )
        comparison = compare_30p30n(result)
        self.assertAlmostEqual(comparison["computed_coefficients"]["drag_x100"], 2.0)
        self.assertAlmostEqual(comparison["relative_errors"]["lift"],
                               abs(0.04 - 0.033243) / 0.033243)
        self.assertIn("Diagnostic comparison only", comparison["note"])


if __name__ == "__main__":
    unittest.main()
