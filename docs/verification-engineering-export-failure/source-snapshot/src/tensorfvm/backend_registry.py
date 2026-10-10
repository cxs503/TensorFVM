"""Explicit solver-backend registry for extensible CFD configurations.

Built-in mesh/solver implementations are loaded lazily.  A downstream package
can register a new backend without editing ``SimpleSolver`` or the built-in
mesh selection logic.
"""

from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any, Callable


Factory = Callable[[Any], Any]
ReferenceLengthRule = str | Callable[[Any], float]
_LENGTH_RULES = frozenset({
    "airfoil_chord", "domain_length", "length", "cylinder_or_height", "height",
})


@dataclass(frozen=True)
class SolverBackend:
    """Capabilities and factories for one configured numerical backend.

    ``mesh_factory`` returns an object implementing :class:`Mesh2D` for
    collocated finite-volume solvers.  Staggered backends may leave it unset.
    Metadata is used by configuration validation, not inferred from class
    names or scattered ``mesh_type`` conditionals.
    """

    name: str
    solver_factory: Factory | None
    mesh_factory: Factory | None
    structured: bool
    min_nx: int = 1
    min_ny: int = 1
    nx_multiple: int = 1
    supports_spalart_allmaras: bool = False
    supports_transient: bool = False
    supports_parabolic_inlet: bool = False
    supports_far_field_option: bool = False
    requires_mesh_file: bool = False
    requires_cylinder: bool = False
    forbids_cylinder: bool = False
    uses_cylinder: bool = False
    cylinder_clearance_cells: bool = False
    requires_rasterized_cylinder: bool = False
    reference_length: ReferenceLengthRule = "cylinder_or_height"
    residual_length: ReferenceLengthRule | None = None
    required_mesh_masks: tuple[str, ...] = (
        "inlet", "outlet", "far-field", "wall",
    )


def _lazy_factory(module_name: str, class_name: str) -> Factory:
    """Return a factory that imports its implementation only when requested."""
    def factory(config):
        implementation = getattr(import_module(module_name), class_name)
        return implementation(config)
    factory.__name__ = f"lazy_{class_name.lower()}_factory"
    return factory


_BACKENDS: dict[str, SolverBackend] = {}
_BACKENDS_LOCK = RLock()


def register_backend(backend: SolverBackend, *, replace: bool = False) -> None:
    """Register a solver backend, rejecting invalid metadata and name collisions."""
    if not isinstance(backend, SolverBackend):
        raise TypeError("backend must be a SolverBackend instance")
    if not isinstance(backend.name, str) or not backend.name.strip():
        raise ValueError("backend name must be nonempty")
    for name in ("min_nx", "min_ny", "nx_multiple"):
        value = getattr(backend, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("backend grid-size constraints must be positive integers")
    for name in (
        "structured", "supports_spalart_allmaras", "supports_transient",
        "supports_parabolic_inlet", "supports_far_field_option", "requires_mesh_file",
        "requires_cylinder", "forbids_cylinder", "uses_cylinder",
        "cylinder_clearance_cells", "requires_rasterized_cylinder",
    ):
        if not isinstance(getattr(backend, name), bool):
            raise TypeError(f"backend capability {name!r} must be bool")
    if backend.solver_factory is not None and not callable(backend.solver_factory):
        raise TypeError("backend solver_factory must be callable or None")
    if backend.mesh_factory is not None and not callable(backend.mesh_factory):
        raise TypeError("backend mesh_factory must be callable or None")
    if backend.name != "cartesian" and backend.solver_factory is None:
        raise ValueError("non-cartesian backends must provide a solver_factory")
    if backend.name != "cartesian" and backend.mesh_factory is None:
        raise ValueError("non-cartesian backends must provide a mesh_factory")
    if backend.name == "cartesian" and (
            (backend.solver_factory is None) != (backend.mesh_factory is None)):
        raise ValueError("a custom cartesian backend must provide both solver and mesh factories")
    if backend.requires_mesh_file and backend.mesh_factory is None:
        raise ValueError("a backend requiring mesh_file must provide a mesh_factory")
    if ((backend.requires_cylinder or backend.cylinder_clearance_cells
         or backend.requires_rasterized_cylinder) and not backend.uses_cylinder):
        raise ValueError("cylinder constraints require uses_cylinder=True")
    if backend.requires_cylinder and backend.forbids_cylinder:
        raise ValueError("backend cannot both require and forbid a cylinder")
    if backend.reference_length is None:
        raise TypeError("reference_length rule cannot be None")
    for rule in (backend.reference_length, backend.residual_length):
        if rule is None:
            continue
        if not (isinstance(rule, str) or callable(rule)):
            raise TypeError("length rules must be strings or callables")
        if isinstance(rule, str) and rule not in _LENGTH_RULES:
            raise ValueError(f"unsupported characteristic-length rule: {rule!r}")
    if (not isinstance(backend.required_mesh_masks, tuple)
            or any(not isinstance(name, str) or not name for name in backend.required_mesh_masks)):
        raise ValueError("required_mesh_masks must be a tuple of nonempty strings")
    with _BACKENDS_LOCK:
        if backend.name in _BACKENDS and not replace:
            raise ValueError(f"solver backend {backend.name!r} is already registered")
        _BACKENDS[backend.name] = backend


def get_backend(name: str) -> SolverBackend:
    """Return a registered backend or raise a configuration-friendly error."""
    try:
        with _BACKENDS_LOCK:
            return _BACKENDS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown mesh/solver backend: {name!r}") from exc


def registered_backends() -> tuple[str, ...]:
    """Return registered backend names in deterministic order."""
    with _BACKENDS_LOCK:
        return tuple(sorted(_BACKENDS))


def build_mesh(config):
    """Build and validate the mesh for a collocated backend."""
    backend = get_backend(config.mesh_type)
    if backend.mesh_factory is None:
        raise ValueError(f"backend {backend.name!r} does not provide a collocated mesh")
    mesh = backend.mesh_factory(config)
    from .mesh_api import validate_mesh2d

    validate_mesh2d(mesh, required_masks=backend.required_mesh_masks)
    return mesh


# Built-in registrations use lazy imports to avoid a module cycle between the
# compatibility-facing SolverConfig, the solver implementations, and meshes.
register_backend(SolverBackend(
    name="cartesian",
    solver_factory=None,
    mesh_factory=None,
    structured=True,
    min_nx=4,
    min_ny=4,
    uses_cylinder=True,
    cylinder_clearance_cells=True,
    requires_rasterized_cylinder=True,
    reference_length="cylinder_or_height",
))
register_backend(SolverBackend(
    name="body-fitted",
    solver_factory=_lazy_factory("tensorfvm.body_fitted", "BodyFittedSolver"),
    mesh_factory=_lazy_factory("tensorfvm.body_fitted", "BodyFittedMesh"),
    structured=True,
    min_nx=8,
    min_ny=4,
    nx_multiple=4,
    supports_spalart_allmaras=True,
    supports_transient=True,
    supports_parabolic_inlet=True,
    supports_far_field_option=True,
    requires_cylinder=True,
    uses_cylinder=True,
    reference_length="cylinder_or_height",
    residual_length="height",
))
register_backend(SolverBackend(
    name="c-grid",
    solver_factory=_lazy_factory("tensorfvm.body_fitted", "BodyFittedSolver"),
    mesh_factory=_lazy_factory("tensorfvm.body_fitted", "CGridMesh"),
    structured=True,
    min_nx=16,
    min_ny=4,
    nx_multiple=4,
    supports_spalart_allmaras=True,
    supports_transient=True,
    reference_length="airfoil_chord",
    residual_length="airfoil_chord",
))
register_backend(SolverBackend(
    name="flat-plate",
    solver_factory=_lazy_factory("tensorfvm.body_fitted", "BodyFittedSolver"),
    mesh_factory=_lazy_factory("tensorfvm.body_fitted", "FlatPlateMesh"),
    structured=True,
    min_nx=4,
    min_ny=4,
    supports_spalart_allmaras=True,
    supports_transient=True,
    forbids_cylinder=True,
    reference_length="domain_length",
    residual_length="domain_length",
))
register_backend(SolverBackend(
    name="three-element",
    solver_factory=_lazy_factory("tensorfvm.body_fitted", "BodyFittedSolver"),
    mesh_factory=_lazy_factory("tensorfvm.multi_element", "ThreeElementMesh"),
    structured=False,
    min_nx=1,
    min_ny=1,
    supports_spalart_allmaras=True,
    supports_transient=True,
    requires_mesh_file=True,
    forbids_cylinder=True,
    reference_length="airfoil_chord",
    residual_length="airfoil_chord",
    required_mesh_masks=("inlet", "outlet", "far-field", "slat", "main", "flap"),
))
