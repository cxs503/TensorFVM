"""Public extension-point regressions for solver and mesh backends."""

import unittest

import torch

from tensorfvm.backend_registry import (
    SolverBackend,
    get_backend,
    register_backend,
    registered_backends,
)
from tensorfvm.mesh_api import validate_mesh2d
from tensorfvm.body_fitted import BodyFittedMesh
from tensorfvm.solver import SimpleSolver, SolverConfig


class BackendRegistryTests(unittest.TestCase):
    def test_builtin_registry_is_complete_and_keeps_lazy_factories(self):
        self.assertTrue({
            "cartesian", "body-fitted", "c-grid", "flat-plate", "three-element",
        }.issubset(registered_backends()))
        self.assertIsNone(get_backend("cartesian").solver_factory)
        self.assertIsNotNone(get_backend("body-fitted").solver_factory)
        self.assertTrue(get_backend("three-element").requires_mesh_file)

    def test_custom_solver_factory_is_routed_by_the_registry(self):
        name = "test-registry-custom-solver"
        result = object()
        spec = SolverBackend(
            name=name,
            solver_factory=lambda config: result,
            mesh_factory=lambda config: object(),
            structured=False,
            min_nx=1,
            min_ny=1,
            reference_length=lambda config: 2.5,
        )
        register_backend(spec)
        config = SolverConfig(mesh_type=name, cylinder_radius=None, nx=1, ny=1)

        self.assertIs(SimpleSolver(config), result)
        self.assertEqual(config.reference_length, 2.5)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_backend(spec)

    def test_backend_capabilities_drive_configuration_validation(self):
        name = "test-registry-capabilities"
        register_backend(SolverBackend(
            name=name,
            solver_factory=lambda config: object(),
            mesh_factory=lambda config: object(),
            structured=True,
            min_nx=8,
            min_ny=4,
            nx_multiple=4,
            supports_transient=True,
            forbids_cylinder=True,
            reference_length="length",
        ))
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            SolverConfig(mesh_type=name, nx=10, ny=4, cylinder_radius=None)
        with self.assertRaisesRegex(ValueError, "does not use cylinder_radius"):
            SolverConfig(mesh_type=name, cylinder_radius=0.2)
        config = SolverConfig(
            mesh_type=name, nx=8, ny=4, cylinder_radius=None, time_step=0.1,
        )
        self.assertEqual(config.reference_length, config.length)

    def test_duplicate_registration_can_be_explicitly_replaced(self):
        name = "test-registry-replacement"
        first = SolverBackend(
            name=name, solver_factory=lambda config: "first",
            mesh_factory=lambda config: None, structured=False,
        )
        second = SolverBackend(
            name=name, solver_factory=lambda config: "second",
            mesh_factory=lambda config: None, structured=False,
        )
        register_backend(first)
        register_backend(second, replace=True)
        self.assertIs(get_backend(name), second)

    def test_mesh_contract_accepts_builtin_mesh_and_rejects_bad_topology(self):
        config = SolverConfig(mesh_type="body-fitted", nx=8, ny=4)
        mesh = BodyFittedMesh(config)
        self.assertIs(validate_mesh2d(
            mesh, required_masks=("inlet", "outlet", "wall", "far-field"),
        ), mesh)

        mesh.boundary = torch.zeros_like(mesh.boundary)
        with self.assertRaisesRegex(ValueError, "flags disagree"):
            validate_mesh2d(mesh)


if __name__ == "__main__":
    unittest.main()
