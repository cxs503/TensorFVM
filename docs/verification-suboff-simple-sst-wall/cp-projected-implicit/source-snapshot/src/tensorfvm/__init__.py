"""PyTorch finite-volume flow solvers."""

from .solver import SimpleSolver, SolverConfig, SolverResult
from .body_fitted import BodyFittedMesh, BodyFittedSolver, CGridMesh, FlatPlateMesh
from .solver3d import Cylinder3DConfig, Cylinder3DResult, Cylinder3DSolver, PressureSolveInfo
from .runtime import DistributedRuntime, SlabPartition, partition_slab
from .body_fitted3d import BodyFittedCylinderMesh3D
from .multi_element import ThreeElementMesh
from .backend_registry import (
    SolverBackend, build_mesh, get_backend, register_backend, registered_backends,
)
from .mesh_api import Mesh2D, validate_mesh2d

__all__ = ["SimpleSolver", "SolverConfig", "SolverResult", "ThreeElementMesh",
           "SolverBackend", "build_mesh", "get_backend", "register_backend",
           "registered_backends",
           "Mesh2D", "validate_mesh2d",
           "BodyFittedMesh", "BodyFittedSolver", "CGridMesh", "FlatPlateMesh",
           "Cylinder3DConfig", "Cylinder3DResult", "Cylinder3DSolver", "PressureSolveInfo",
           "DistributedRuntime", "SlabPartition", "partition_slab",
           "BodyFittedCylinderMesh3D"]
