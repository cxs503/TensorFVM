"""Mesh contract (``MeshData``) and adapters for the existing mesh classes."""

from .adapters import (
    MESH_ADAPTERS,
    adapt_mesh,
    from_body_fitted3d,
    from_mesh2d,
    from_native_mesh3d,
    from_overset_grid,
    from_polymesh,
)
from .contract import MeshData, validate_mesh_data

__all__ = [
    "MeshData",
    "validate_mesh_data",
    "adapt_mesh",
    "MESH_ADAPTERS",
    "from_mesh2d",
    "from_native_mesh3d",
    "from_polymesh",
    "from_overset_grid",
    "from_body_fitted3d",
]
