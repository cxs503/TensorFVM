#!/usr/bin/env python3
"""Mutation proof for ``tests/test_backend_registry_contract.py``.

Each entry below breaks exactly one guard of the backend-registry contract in
the *working tree*, runs the contract test that is supposed to notice, and
restores the file before the next mutation.  A mutation that leaves the suite
green means the contract test does not actually pin that guard, so the script
exits non-zero and the gate is reported as failed.

Usage (from a checkout root)::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/verify_backend_registry_contract_negatives.py

The full transcript is written to stdout so it can be archived as evidence.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "src/tensorfvm/backend_registry.py"
SOLVER = ROOT / "src/tensorfvm/solver.py"
CONTRACT = "tests/test_backend_registry_contract.py"
NODE = f"{CONTRACT}::BackendRegistryContractTests"

MUTATIONS = (
    (
        "duplicate-name guard removed (register_backend)",
        REGISTRY,
        "if backend.name in _BACKENDS and not replace:",
        "if False and not replace:",
        f"{NODE}::test_duplicate_registration_requires_explicit_replace",
    ),
    (
        "duplicate-name guard ignores replace=False (register_backend)",
        REGISTRY,
        "if backend.name in _BACKENDS and not replace:",
        "if backend.name in _BACKENDS and replace:",
        f"{NODE}::test_duplicate_registration_requires_explicit_replace",
    ),
    (
        "unknown-name error downgraded to KeyError (get_backend)",
        REGISTRY,
        "        raise ValueError(f\"unknown mesh/solver backend: {name!r}\") from exc",
        "        raise KeyError(name) from exc",
        f"{NODE}::test_unknown_name_raises_configuration_friendly_valueerror",
    ),
    (
        "blank-name guard weakened to a non-str check (register_backend)",
        REGISTRY,
        "    if not isinstance(backend.name, str) or not backend.name.strip():",
        "    if not isinstance(backend.name, str):",
        f"{NODE}::test_registration_rejects_blank_names_without_mutation",
    ),
    (
        "required_mesh_masks guard removed (register_backend)",
        REGISTRY,
        "    if (not isinstance(backend.required_mesh_masks, tuple)",
        "    if False and (not isinstance(backend.required_mesh_masks, tuple)",
        f"{NODE}::test_invalid_specifications_are_rejected_and_never_registered",
    ),
    (
        "cylinder/uses_cylinder consistency guard removed (register_backend)",
        REGISTRY,
        "    if ((backend.requires_cylinder or backend.cylinder_clearance_cells\n"
        "         or backend.requires_rasterized_cylinder) and not backend.uses_cylinder):",
        "    if False and ((backend.requires_cylinder \n"
        "         or backend.requires_rasterized_cylinder) and not backend.uses_cylinder):",
        f"{NODE}::test_invalid_specifications_are_rejected_and_never_registered",
    ),
    (
        "backend names silently normalised on write (register_backend)",
        REGISTRY,
        "        _BACKENDS[backend.name] = backend",
        "        _BACKENDS[backend.name.strip()] = backend",
        f"{NODE}::test_names_are_stored_and_looked_up_verbatim",
    ),
    (
        "SimpleSolver factory hook disabled (solver.__new__)",
        SOLVER,
        "        if cls is SimpleSolver:",
        "        if False:",
        f"{NODE}::test_registered_factory_takes_over_the_simple_solver_entry_point",
    ),
    (
        "SimpleSolver factory hook ignores subclasses (solver.__new__)",
        SOLVER,
        "        if cls is SimpleSolver:",
        "        if True:",
        f"{NODE}::test_subclasses_bypass_the_registry_factory_hook",
    ),
    (
        "build_mesh skips the missing-factory guard",
        REGISTRY,
        "    if backend.mesh_factory is None:\n"
        "        raise ValueError(f\"backend {backend.name!r} does not provide "
        "a collocated mesh\")",
        "    if False:\n"
        "        raise ValueError(f\"backend {backend.name!r} does not provide "
        "a collocated mesh\")",
        f"{NODE}::test_build_mesh_reports_missing_and_unknown_mesh_backends",
    ),
    (
        "residual_length fallback to reference_length dropped (SolverConfig)",
        SOLVER,
        "            backend.reference_length if rule is None else rule",
        "            backend.reference_length if rule is None else None",
        f"{NODE}::test_reference_length_fallback_chain_uses_registry_rules",
    ),
)


def run(selector: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", selector, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    failures = []
    for index, (label, path, old, new, selector) in enumerate(MUTATIONS, start=1):
        original = path.read_text()
        if original.count(old) != 1:
            print(f"[{index}] SETUP-ERROR {label}: anchor found "
                  f"{original.count(old)}x in {path.name}")
            failures.append(label)
            continue
        try:
            path.write_text(original.replace(old, new, 1))
            code, output = run(selector)
        finally:
            path.write_text(original)
        summary = [line for line in output.splitlines()
                   if line.startswith(("FAILED", "ERROR", "="))
                   or "passed" in line or "failed" in line]
        verdict = "RED (good)" if code != 0 else "GREEN (mutation NOT caught)"
        print(f"[{index}] {verdict} :: {label}")
        print(f"    selector: {selector}")
        print(f"    pytest rc={code}")
        for line in summary[-4:]:
            print(f"    | {line}")
        if code == 0:
            failures.append(label)
    # Baseline: the unmutated tree must be green, otherwise the proof is void.
    code, output = run(NODE)
    print(f"[baseline] contract suite rc={code} "
          f"{'GREEN (good)' if code == 0 else 'RED (bad)'}")
    if code != 0:
        failures.append("baseline")
        print(output[-2000:])
    if failures:
        print(f"\nMUTATION PROOF FAILED for: {failures}")
        return 1
    print(f"\nMUTATION PROOF PASSED: {len(MUTATIONS)} guards, all negative cases red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())