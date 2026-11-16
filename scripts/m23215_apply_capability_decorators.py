#!/usr/bin/env python
"""Insert ``@capability`` decorators into the TensorFVM kernel entry points.

The table below is the *declared* inventory; the two facts that make it honest
are computed, not typed:

``wired``
    true only when some other module under ``src/tensorfvm`` imports the module
    (the same mechanical rule the audit script uses).  "Exported from
    ``__init__``" does not count -- nothing dispatches to it.
``tested`` / ``test_path``
    true/first when a ``tests/test_*.py`` names the dotted module and that file
    defines at least one ``test_*`` function.

``verified`` is *not* derived: it is typed next to a number that
``scripts/m23215_error_bands.py`` actually measures (``max_error_pct``), so a
``verified=True`` without a reproduction path fails the registry discipline.

Run with ``--check`` to see the table without touching the tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "tensorfvm"
TESTS_ROOT = REPO_ROOT / "tests"
BANDS = REPO_ROOT / "docs" / "m23215" / "error_bands.json"

#: (id, relative module file, entry, kind, extra evidence overrides)
TABLE: list[tuple[str, str, str, str, dict[str, Any]]] = [
    # ---- 2-D incompressible core ------------------------------------------ #
    ("fvm.core.ins2d_collocated", "solver.py", "SimpleSolver", "solver", {}),
    ("fvm.core.ins3d_cylinder", "solver3d.py", "Cylinder3DSolver", "solver", {}),
    ("fvm.core.annular_axial", "annular.py", "solve", "solver",
     {"band": "annular"}),
    # ---- linear algebra / mesh ------------------------------------------- #
    ("fvm.linear.sparse_body_fitted", "sparse_simple.py", "SparseBodyFittedSolver",
     "linear_solver", {}),
    ("fvm.linear.splu_direct", "external_linear.py", "ExternalFlowSolver",
     "linear_solver", {}),
    ("fvm.linear.diagnostics", "native_linear_diagnostics.py",
     "linear_system_diagnostics", "linear_solver", {"band": "linear_diagnostics"}),
    ("fvm.linear.block_amg3d", "native_block_solver3d.py", "NativeHybridBlockSolver3D",
     "linear_solver", {"note": "needs the optional tensorfvm[amg] extra (pyamg); "
                               "not importable in the default environment"}),
    ("fvm.linear.overset_poisson", "overset_poisson.py", "solve_overset_poisson",
     "linear_solver", {}),
    ("fvm.mesh.contract2d", "mesh_api.py", "validate_mesh2d", "mesh", {}),
    ("fvm.mesh.backend_registry", "backend_registry.py", "build_mesh", "mesh", {}),
    ("fvm.mesh.unstructured3d", "unstructured_fvm3d.py", "UnstructuredFVM3D", "mesh", {}),
    ("fvm.mesh.openfoam_polymesh", "openfoam_polymesh.py", "PolyMesh", "mesh", {}),
    ("fvm.mesh.overset", "overset.py", "OversetGrid", "mesh", {}),
    ("fvm.flux.overset_transfer", "overset_flux.py", "PlanarFluxTransfer", "flux",
     {"band": "overset"}),
    # ---- free surface ----------------------------------------------------- #
    ("fvm.freesurface.plic_geometry", "vof_plic.py", "reconstruct", "free_surface",
     {"band": "plic"}),
    ("fvm.freesurface.plic_advect", "vof_plic.py", "advect_plic", "free_surface", {}),
    ("fvm.freesurface.cicsam", "vof_cicsam.py", "CICSAMVOF", "free_surface", {}),
    ("fvm.freesurface.fct", "vof_fct.py", "LimitedVOFTransport", "free_surface", {}),
    ("fvm.freesurface.thinc", "vof/thinc.py", "thinc_flux", "free_surface", {}),
    # ---- multiphase ------------------------------------------------------- #
    ("fvm.multiphase.solver", "two_phase_solver.py", "TwoPhaseSolver", "multiphase", {}),
    ("fvm.multiphase.drift_flux", "two_phase/drift_flux.py", "wallis_drift_velocity",
     "multiphase", {}),
    ("fvm.multiphase.rayleigh_plesset", "two_phase/cavitation_transport.py",
     "plesset_solution", "multiphase", {}),
    ("fvm.multiphase.stefan", "two_phase/stefan.py", "stefan_lambda", "multiphase", {}),
    # ---- compressible / turbulence --------------------------------------- #
    ("fvm.compressible.euler", "compressible/solver.py", "EulerSolver", "compressible", {}),
    ("fvm.compressible.slope_limiter", "compressible/reconstruct.py", "minmod",
     "compressible", {}),
    ("fvm.turbulence.hybrid_rans3d", "native_hybrid_rans3d.py", "NativeHybridRans3D",
     "turbulence", {}),
    # ---- ship-performance domains ---------------------------------------- #
    ("fvm.cavitation.inception", "cavitation/inception.py", "cavitation_number",
     "cavitation", {}),
    ("fvm.thermal.conjugate_energy", "cht/energy.py", "EnergyLedger", "thermal", {}),
    ("fvm.seakeeping.wave_theory", "wave_tank/theory.py", "dispersion_k", "seakeeping", {}),
    ("fvm.seakeeping.strip_theory", "wave_tank_td/seakeeping.py", "hydrostatic_stiffness",
     "seakeeping", {}),
    ("fvm.maneuvering.dynamics", "maneuvering/dynamics.py", "ManoeuvreModel",
     "maneuvering", {}),
    ("fvm.differentiable.shape_adjoint", "adjoint_shape/adjoint.py", "adjoint_gradient",
     "differentiable", {}),
    ("fvm.pmm.section_ns2d", "pmm_solver.py", "SectionNS2D", "solver", {}),
    # ---- remaining subpackages (coverage: one honest entry each) ---------- #
    ("fvm.ice.load_path_closure", "ice/benchmark.py", "row_g4_load_path_closure", "ice", {}),
    ("fvm.immersed.control_box", "immersed/forces.py", "ControlBox", "module", {}),
    ("fvm.internal_flow.backward_step", "internal_flow/backward_step.py",
     "BackwardStepCase", "solver", {}),
    ("fvm.resistance.speed_sweep", "rans_ship_resistance/speed_sweep.py",
     "speed_points_kn", "module", {}),
    ("fvm.turbulence.validation_matrix", "rans_validation/matrix.py", "solve_case",
     "turbulence", {}),
    ("fvm.shallow_water.ritter_exact", "shallow_water/exact.py", "ritter_state",
     "verification", {}),
    ("fvm.ship_freesurface.wave_elevation", "ship_freesurface/cfd_verification.py",
     "compute_wave_elevation_from_alpha", "seakeeping", {}),
    ("fvm.freesurface.algebraic_vof", "vof_phase0/advection.py", "AlgebraicVOF",
     "free_surface", {}),
    ("fvm.multiphase.bench_campaign", "twophase_bench/campaign.py", "run_campaign",
     "multiphase", {}),
    ("fvm.multiphase.dambreak_gate", "twophase_gap/dambreak_gate.py",
     "decompose_dambreak_error", "verification", {}),
    ("fvm.freesurface.threecase_campaign", "vof_threecase/campaign.py",
     "zalesak_sweep", "free_surface", {}),
    ("fvm.verification.gci", "vof_verification/metrics.py", "gci", "verification", {}),
    ("fvm.inc_bench.armaly_experiment", "inc_bench/armaly.py", "experiment_point",
     "verification", {}),
    ("fvm.verification.rate_fit", "verification/convergence_xval_round3.py",
     "QuadraticRateFit", "verification", {}),
]


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


def _dotted(rel: Path) -> str:
    parts = ["tensorfvm", *rel.with_suffix("").parts]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _load_audit():
    """Import the audit script so both tools use the *same* edge resolution."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m23215_fvm_audit", REPO_ROOT / "scripts" / "m23215_fvm_audit.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _importers() -> dict[str, list[str]]:
    """Repo-relative module file -> module files that import it (src only)."""
    audit = _load_audit()
    mods, _shadowed = audit.scan_modules()
    users: dict[str, list[str]] = {rel: [] for rel in mods}
    for rel, rec in mods.items():
        for target in rec["resolved_imports"]:
            if target != rel:
                users[target].append(rel)
    return users


def _test_binding(dotted: str) -> tuple[bool, str | None]:
    """First test that *directly* imports the module, preferring a matching name."""
    direct = re.compile(rf"(?:from\s+{re.escape(dotted)}\s+import|import\s+{re.escape(dotted)}\b)")
    mentions = re.compile(rf"\b{re.escape(dotted)}\b")
    short = dotted.split(".", 1)[1].split(".")[-1]
    cands: list[tuple[int, str]] = []
    for path in sorted(TESTS_ROOT.glob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not mentions.search(text):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        if not any(isinstance(n, ast.FunctionDef) and n.name.startswith("test")
                   for n in ast.walk(tree)):
            continue
        score = 0
        if direct.search(text):
            score += 2
        if short.replace("_", "") in path.name.replace("_", ""):
            score += 3
        cands.append((-score, str(path.relative_to(REPO_ROOT))))
    if not cands:
        return False, None
    cands.sort()
    return True, cands[0][1]


def _docs_count(rel: Path) -> int:
    pat = re.compile(rf"(?<![\w/]){re.escape(str(rel.with_suffix('')))}(?:\.py)?(?![\w])")
    return sum(1 for p in sorted((REPO_ROOT / "docs").glob("*.md"))
               if pat.search(p.read_text(encoding="utf-8", errors="replace")))


def _bands() -> dict[str, Any]:
    if not BANDS.is_file():
        return {}
    raw = json.loads(BANDS.read_text(encoding="utf-8"))
    return {rec["capability"]: rec for rec in raw.values() if "capability" in rec}


# --------------------------------------------------------------------------- #
# Decorator text + insertion
# --------------------------------------------------------------------------- #


def _fmt(value: Any) -> str:
    return repr(value)


def decorator_source(cap_id: str, kind: str, maturity: str, entry: str,
                     wired: bool, tested: bool, test_path: str | None,
                     evidence: dict[str, Any], verified: bool) -> str:
    lines = [
        f"@capability({cap_id!r},",
        f"            kind={kind!r},",
        f"            maturity={maturity!r},",
    ]
    if test_path:
        lines.append(f"            test_path={test_path!r},")
    lines += [
        f"            wired={wired!r},",
        f"            tested={tested!r},",
        f"            verified={verified!r},",
        "            evidence=" + _pretty_dict(evidence) + ")",
    ]
    return "\n".join(lines)


def _pretty_dict(evidence: dict[str, Any], indent: int = 13) -> str:
    if not evidence:
        return "{}"
    pad = " " * indent
    inner = ", ".join(f"{k!r}: {v!r}" for k, v in evidence.items())
    if len(inner) + indent < 96:
        return "{" + inner + "}"
    body = ",\n".join(f"{pad}{k!r}: {v!r}" for k, v in evidence.items())
    return "{\n" + body + ",\n" + " " * (indent - 4) + "}"


def _insert(path: Path, entry: str, block: str, import_line: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if f'@capability(' in text and f"'\n" not in text:
        pass
    if "from .registry import capability" in text or "from ..registry import capability" in text:
        return False
    lines = text.splitlines()
    # locate the target definition (top level or one level of class nesting)
    target = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"def {entry}(") or stripped.startswith(f"class {entry}(") \
                or stripped.startswith(f"class {entry}:"):
            target = i
            break
    if target is None:
        raise SystemExit(f"{path}: entry {entry!r} not found")
    indent = lines[target][: len(lines[target]) - len(lines[target].lstrip())]
    block_lines = [(indent + ln if ln else ln) for ln in block.splitlines()]
    # insert the import right after the last top-level import line
    last_import = None
    for i, line in enumerate(lines[:target]):
        if line.startswith(("import ", "from ")):
            last_import = i
    if last_import is None:
        raise SystemExit(f"{path}: no import block to attach the registry import to")
    out = lines[: last_import + 1] + [import_line] + lines[last_import + 1: target] \
        + block_lines + lines[target:]
    path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return True


def _import_line(rel: Path) -> str:
    depth = len(rel.parts) - 1  # 0 for a top-level module
    dots = "." * (depth + 1)
    return f"from {dots}registry import capability"


def build_table() -> list[dict[str, Any]]:
    users = _importers()
    bands = _bands()
    rows = []
    for cap_id, rel_str, entry, kind, extra in TABLE:
        rel = Path(rel_str)
        dotted = _dotted(rel)
        wired = len(users.get(rel_str, [])) > 0
        tested, test_path = _test_binding(dotted)
        band = bands.get(cap_id) if extra.get("band") else None
        evidence: dict[str, Any] = {}
        verified = False
        if band and "max_error_pct" in band:
            verified = True
            evidence = {
                "grade": band["grade"],
                "max_error_pct": band["max_error_pct"],
                "reference": band["reference"],
                "band_script": "scripts/m23215_error_bands.py",
            }
            if extra.get("note"):
                evidence["note"] = extra["note"]
            if band.get("coarse_max_error_pct"):
                evidence["coarse_max_error_pct"] = band["coarse_max_error_pct"]
        else:
            evidence = {"grade": "MECHANICS" if tested else "NONE"}
            if extra.get("note"):
                evidence["note"] = extra["note"]
            evidence["docs"] = _docs_count(rel)
        maturity = "mature" if (verified and wired) else "developing"
        rows.append({
            "id": cap_id, "file": rel_str, "dotted": dotted, "entry": entry,
            "kind": kind, "wired": wired, "importers": users.get(dotted, []),
            "tested": tested, "test_path": test_path, "verified": verified,
            "maturity": maturity, "evidence": evidence,
        })
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="print the table, change nothing")
    ap.add_argument("--apply", action="store_true", help="insert the decorators")
    args = ap.parse_args(list(argv) if argv is not None else None)

    rows = build_table()
    print(f"{'capability':34s} {'maturity':10s} {'wired':5s} {'tested':6s} {'verified':8s} tests")
    for row in rows:
        print(f"{row['id']:34s} {row['maturity']:10s} {str(row['wired']):5s} "
              f"{str(row['tested']):6s} {str(row['verified']):8s} {row['test_path'] or '-'}")
    if args.check or not args.apply:
        return 0

    changes = 0
    for row in rows:
        path = PKG_ROOT / row["file"]
        block = decorator_source(row["id"], row["kind"], row["maturity"], row["entry"],
                                 row["wired"], row["tested"], row["test_path"],
                                 row["evidence"], row["verified"])
        if _insert(path, row["entry"], block, _import_line(Path(row["file"]))):
            changes += 1
            print(f"decorated {row['file']}:{row['entry']} -> {row['id']}")
    print(f"\n{changes} files decorated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())