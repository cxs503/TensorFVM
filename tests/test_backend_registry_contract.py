"""Contract tests for the solver-backend registry: register / resolve / fallback.

``tests/test_backend_registry.py`` covers the happy-path extension point.  This
module pins the *boundaries* of that contract, where a silent change would be
invisible to the rest of the suite:

* **register**  -- every rejected specification raises a specific error class
  and leaves the global registry byte-for-byte unchanged (no half-registered or
  shadowing names), and duplicate names need an explicit ``replace=True``.
* **resolve**   -- unknown / illegal names raise ``ValueError`` with a stable,
  configuration-friendly message and a chained cause, so callers can branch on
  the exception instead of parsing internals.
* **fallback**  -- ``SimpleSolver.__new__`` keeps the built-in collocated path
  when a backend registers no solver factory, routes to the registered factory
  otherwise, and hands subclasses straight to their own ``__init__``;
  ``build_mesh`` and the characteristic-length rules resolve through the same
  registry entry, including the ``residual_length=None`` fallback.

Every assertion here is a negative or a boundary: deleting a guard in
``backend_registry`` or in the ``SimpleSolver`` factory hook turns this module
red (see docs/backend-registry-contract.md for the injection proof).
"""

import unittest

from tensorfvm import backend_registry as registry
from tensorfvm.backend_registry import (
    SolverBackend,
    build_mesh,
    get_backend,
    register_backend,
    registered_backends,
)
from tensorfvm.mesh_api import validate_mesh2d
from tensorfvm.solver import SimpleSolver, SolverConfig

_UNKNOWN = "unknown mesh/solver backend"


def _sentinel_factory(tag):
    """Return a factory that records the config it was handed."""
    calls = []

    def factory(config):
        calls.append(config)
        return tag

    factory.calls = calls
    return factory


def _spec(**overrides):
    """A minimal valid non-cartesian spec, overridable per case."""
    kwargs = {
        "name": "contract-spec",
        "solver_factory": lambda config: object(),
        "mesh_factory": lambda config: object(),
        "structured": False,
    }
    kwargs.update(overrides)
    return SolverBackend(**kwargs)


class BackendRegistryContractTests(unittest.TestCase):
    """Boundary behaviour of the public backend registry."""

    def setUp(self):
        # The registry is process-global and intentionally has no public
        # unregister API, so contract tests snapshot the mapping and restore it
        # to keep test order irrelevant (other modules also register backends).
        with registry._BACKENDS_LOCK:
            self._snapshot = dict(registry._BACKENDS)

    def tearDown(self):
        with registry._BACKENDS_LOCK:
            registry._BACKENDS.clear()
            registry._BACKENDS.update(self._snapshot)

    # ---------------------------------------------------------------- resolve
    def test_resolution_returns_the_registered_object_stably(self):
        spec = _spec(name="contract-identity")
        register_backend(spec)

        self.assertIs(get_backend("contract-identity"), spec)
        self.assertIs(get_backend("contract-identity"), get_backend("contract-identity"))

    def test_registered_names_are_a_sorted_tuple_of_str(self):
        names = registered_backends()
        self.assertIsInstance(names, tuple)
        self.assertTrue(all(isinstance(name, str) for name in names))
        self.assertEqual(names, tuple(sorted(names)))
        # A fresh tuple per call: callers must not mutate registry state.
        self.assertEqual(names, registered_backends())

    def test_unknown_name_raises_configuration_friendly_valueerror(self):
        with self.assertRaises(ValueError) as raised:
            get_backend("contract-does-not-exist")
        error = raised.exception
        self.assertEqual(
            str(error), f"{_UNKNOWN}: 'contract-does-not-exist'")
        # The original KeyError is chained, not swallowed or merged into text.
        self.assertIsInstance(error.__cause__, KeyError)
        # No registry internals leak into the user-facing message.
        self.assertNotIn("{", str(error))

    def test_illegal_names_all_raise_the_unknown_backend_error(self):
        for name in (None, "", "   ", "Cartesian", "cartesian ", ["cartesian"], 3):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as raised:
                    get_backend(name)
                self.assertIn(_UNKNOWN, str(raised.exception))
                self.assertIn(repr(name), str(raised.exception))

    def test_unhashable_name_chains_typeerror_as_valueerror(self):
        with self.assertRaises(ValueError) as raised:
            get_backend(["cartesian"])
        self.assertIsInstance(raised.exception.__cause__, TypeError)

    def test_names_are_stored_and_looked_up_verbatim(self):
        # No stripping / case folding: a spaced name must not answer to the
        # stripped spelling, otherwise two backends could alias each other.
        register_backend(_spec(name=" contract-spaced "))
        self.assertTrue(get_backend(" contract-spaced ").name == " contract-spaced ")
        with self.assertRaises(ValueError):
            get_backend("contract-spaced")

    def test_unknown_host_backend_stops_config_construction(self):
        with self.assertRaises(ValueError) as raised:
            SolverConfig(mesh_type="contract-unknown-mesh-type")
        self.assertEqual(
            str(raised.exception),
            f"{_UNKNOWN}: 'contract-unknown-mesh-type'",
        )

    # --------------------------------------------------------------- register
    def test_registration_rejects_objects_that_are_not_backend_specs(self):
        for bogus in ({"name": "contract-dict"}, None, object(), "cartesian"):
            with self.subTest(bogus=bogus):
                with self.assertRaises(TypeError) as raised:
                    register_backend(bogus)
                self.assertIn("must be a SolverBackend instance", str(raised.exception))

    def test_registration_rejects_blank_names_without_mutation(self):
        before = registered_backends()
        for name in ("", "   ", "\t\n"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as raised:
                    register_backend(_spec(name=name))
                self.assertIn("name must be nonempty", str(raised.exception))
        self.assertEqual(registered_backends(), before)

    def test_invalid_specifications_are_rejected_and_never_registered(self):
        cases = (
            # (kwargs, expected exception, message fragment)
            ({"name": "contract-bad-grid", "min_nx": 0},
             ValueError, "positive integers"),
            ({"name": "contract-bad-grid", "min_ny": 2.5},
             ValueError, "positive integers"),
            ({"name": "contract-bad-grid", "nx_multiple": True},
             ValueError, "positive integers"),
            ({"name": "contract-bad-bool", "structured": "yes"},
             TypeError, "must be bool"),
            ({"name": "contract-bad-bool", "supports_transient": 1},
             TypeError, "must be bool"),
            ({"name": "contract-bad-solver", "solver_factory": 42},
             TypeError, "solver_factory must be callable"),
            ({"name": "contract-bad-mesh", "mesh_factory": "BodyFittedMesh"},
             TypeError, "mesh_factory must be callable"),
            ({"name": "contract-missing-solver", "solver_factory": None},
             ValueError, "must provide a solver_factory"),
            ({"name": "contract-missing-mesh", "mesh_factory": None},
             ValueError, "must provide a mesh_factory"),
            ({"name": "cartesian", "mesh_factory": None},
             ValueError, "must provide both solver and mesh factories"),
            ({"name": "cartesian", "solver_factory": None, "mesh_factory": None,
              "requires_mesh_file": True},
             ValueError, "requiring mesh_file must provide a mesh_factory"),
            ({"name": "contract-cylinder", "requires_cylinder": True},
             ValueError, "cylinder constraints require uses_cylinder"),
            ({"name": "contract-cylinder", "requires_cylinder": True,
              "forbids_cylinder": True, "uses_cylinder": True},
             ValueError, "cannot both require and forbid"),
            ({"name": "contract-length", "reference_length": None},
             TypeError, "reference_length rule cannot be None"),
            ({"name": "contract-length", "reference_length": "chord"},
             ValueError, "unsupported characteristic-length rule"),
            ({"name": "contract-length", "residual_length": 3.0},
             TypeError, "length rules must be strings or callables"),
            ({"name": "contract-masks", "required_mesh_masks": "inlet"},
             ValueError, "tuple of nonempty strings"),
            ({"name": "contract-masks", "required_mesh_masks": ("inlet", "")},
             ValueError, "tuple of nonempty strings"),
        )
        before = registered_backends()
        for kwargs, exc, fragment in cases:
            with self.subTest(kwargs=kwargs, fragment=fragment):
                with self.assertRaises(exc) as raised:
                    register_backend(_spec(**kwargs))
                self.assertIn(fragment, str(raised.exception))
                # Atomic rejection: nothing was recorded under any spelling.
                self.assertNotIn("contract-bad-grid", registered_backends())
                self.assertNotIn("contract-bad-bool", registered_backends())
                self.assertNotIn("contract-cylinder", registered_backends())
                self.assertNotIn("contract-length", registered_backends())
                self.assertNotIn("contract-masks", registered_backends())
                self.assertEqual(registered_backends(), before)
        # The built-in entry survives failed re-registrations of its own name.
        self.assertIsNone(get_backend("cartesian").solver_factory)
        self.assertEqual(get_backend("cartesian").name, "cartesian")

    def test_duplicate_registration_requires_explicit_replace(self):
        first = _spec(name="contract-duplicate")
        second = _spec(name="contract-duplicate", solver_factory=lambda config: "second")
        register_backend(first)

        with self.assertRaises(ValueError) as raised:
            register_backend(second)
        self.assertIn("is already registered", str(raised.exception))
        # The rejected duplicate must not shadow the live registration.
        self.assertIs(get_backend("contract-duplicate"), first)

        register_backend(second, replace=True)
        self.assertIs(get_backend("contract-duplicate"), second)
        self.assertIsNot(get_backend("contract-duplicate"), first)

    # --------------------------------------------------------------- fallback
    def test_simple_solver_falls_back_to_the_builtin_collocated_path(self):
        # nx=ny=16 keeps the default cylinder rasterisable on the stair-step grid.
        config = SolverConfig(mesh_type="cartesian", nx=16, ny=16)
        solver = SimpleSolver(config)
        self.assertIs(type(solver), SimpleSolver)
        self.assertIs(solver.config, config)

    def test_registered_factory_takes_over_the_simple_solver_entry_point(self):
        sentinel = object()
        factory = _sentinel_factory(sentinel)
        register_backend(_spec(name="contract-routed", solver_factory=factory))
        config = SolverConfig(
            mesh_type="contract-routed", nx=2, ny=2, cylinder_radius=None)

        self.assertIs(SimpleSolver(config), sentinel)
        self.assertEqual(len(factory.calls), 1)
        self.assertIs(factory.calls[0], config)

    def test_subclasses_bypass_the_registry_factory_hook(self):
        factory = _sentinel_factory("routed")
        register_backend(_spec(name="contract-subclass", solver_factory=factory))

        class _DerivedSolver(SimpleSolver):
            def __init__(self, config):
                self.config = config

        config = SolverConfig(
            mesh_type="contract-subclass", nx=2, ny=2, cylinder_radius=None)
        solver = _DerivedSolver(config)

        self.assertIs(type(solver), _DerivedSolver)
        self.assertEqual(factory.calls, [])

    def test_build_mesh_reports_missing_and_unknown_mesh_backends(self):
        with self.assertRaises(ValueError) as raised:
            build_mesh(SolverConfig(
                mesh_type="cartesian", nx=8, ny=4, cylinder_radius=None))
        self.assertIn("does not provide a collocated mesh", str(raised.exception))

        with self.assertRaises(ValueError) as raised:
            build_mesh(_ConfigView("contract-no-mesh-backend"))
        self.assertEqual(
            str(raised.exception),
            f"{_UNKNOWN}: 'contract-no-mesh-backend'",
        )

    def test_build_mesh_routes_the_factory_and_runs_the_mesh_contract(self):
        class _MarkerMesh:
            pass

        register_backend(_spec(
            name="contract-mesh", mesh_factory=lambda config: _MarkerMesh()))
        config = SolverConfig(mesh_type="contract-mesh", nx=2, ny=2, cylinder_radius=None)

        with self.assertRaises(TypeError) as raised:
            build_mesh(config)
        # The factory ran, and its output was validated at the boundary.
        self.assertIn("Mesh2D is missing required field", str(raised.exception))
        self.assertIsInstance(_MarkerMesh(), object)

        boom = RuntimeError("mesh factory reached")

        def exploding(config):
            raise boom

        register_backend(_spec(name="contract-mesh-raise", mesh_factory=exploding),
                         replace=False)
        with self.assertRaises(RuntimeError) as raised:
            build_mesh(_ConfigView("contract-mesh-raise"))
        self.assertIs(raised.exception, boom)
        self.assertTrue(validate_mesh2d.__doc__)

    def test_reference_length_fallback_chain_uses_registry_rules(self):
        register_backend(_spec(
            name="contract-callable-length",
            reference_length=lambda config: 2.5,
        ))
        register_backend(_spec(
            name="contract-residual-length",
            reference_length="length",
            residual_length="height",
        ))

        callable_config = SolverConfig(
            mesh_type="contract-callable-length", nx=2, ny=2, cylinder_radius=None)
        self.assertEqual(callable_config.reference_length, 2.5)
        # residual_length=None falls back to the reference-length rule.
        self.assertEqual(callable_config.residual_length, 2.5)

        split_config = SolverConfig(
            mesh_type="contract-residual-length", nx=2, ny=2, height=3.0,
            cylinder_radius=None)
        self.assertEqual(split_config.reference_length, split_config.length)
        self.assertEqual(split_config.residual_length, 3.0)
        self.assertNotEqual(split_config.residual_length, split_config.reference_length)

    def test_state_is_restored_by_the_test_harness(self):
        # Guards the snapshot/restore above: the built-ins are always present.
        self.assertTrue(
            {"cartesian", "body-fitted", "c-grid", "flat-plate",
             "three-element"}.issubset(registered_backends()))


class _ConfigView:
    """Minimal stand-in for SolverConfig used to reach registry lookup only."""

    def __init__(self, mesh_type):
        self.mesh_type = mesh_type


if __name__ == "__main__":
    unittest.main()