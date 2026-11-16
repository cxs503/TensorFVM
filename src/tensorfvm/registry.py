"""Self-registering capability registry for TensorFVM.

Motivation
----------
TensorFVM grew vertically: 24 subpackages and ~195 top-level modules, four
generations of VOF campaign code, three of two-phase, two wave-tank stacks, and
no single mesh / flux / boundary-condition / linear-solver / time-integration
interface layer.  The inventory of "what numerics actually exist and what backs
them" lived in prose (README tables, per-case ``docs/CASE-*`` reports), which
cannot be checked: a kernel can be renamed, mocked or orphaned while the prose
keeps claiming it works.

This module turns the inventory into code that registers itself at import time,
the way a router registers its routes:

.. code-block:: python

    from .registry import capability

    @capability(
        "fvm.freesurface.plic_reconstruct",
        kind="free_surface",
        maturity="mature",
        wired=True,
        tested=True,
        verified=True,
        test_path="tests/test_vof_plic.py",
        evidence={"grade": "ANALYTIC", "max_error_pct": 1e-9,
                  "note": "A(n, c(alpha)) == alpha exact to the bisection floor"},
    )
    def reconstruct(alpha, dx=1.0, dy=1.0):
        ...

Nothing else is needed: importing the module populates the process-wide
:data:`REGISTRY` singleton.  Two dump modes (the CLI contract is *stdout is one
JSON object* with at least ``source_commit`` and ``capabilities``)::

    # import mode: needs the environment (torch), sees runtime state
    python -m tensorfvm.registry

    # static AST mode: imports nothing from the package, needs no torch
    python src/tensorfvm/registry.py --static

The static mode is deliberately runnable *by file path*: ``python -m
tensorfvm.registry`` imports the package ``__init__`` (which imports torch), so a
torch-free host must call the file directly.  This module imports only the
standard library and is therefore safe for modules deep in the package to
import.

Honesty rules encoded here (see :func:`validate_capability`)
------------------------------------------------------------
``mature``
    requires ``wired and verified`` and is *warned about* when ``tested`` /
    ``test_path`` are missing.  A false "mature" fails at import time
    (:class:`MaturityDisciplineError`) instead of reaching a dashboard.
``planned``
    requires an evidence anchor naming the roadmap row
    (``evidence["feature_ledger"]``), because "we plan to build it" only means
    something if it is booked somewhere, and may not be ``verified``.
``endpoint``
    may only be declared by a wired capability and must use a recognised
    scheme (``cli:`` / ``route:`` / ``api:``).

Pointer resolution (both modes): every ``module_path``, ``entry`` and
``test_path`` is resolved against the repository.  A capability naming a module,
function or test file that does not exist is reported as a *problem*, not
silently dropped -- that is exactly the drift prose cannot detect.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping

__all__ = [
    "CAPABILITY_KINDS",
    "Capability",
    "DuplicateCapabilityError",
    "EVIDENCE_GRADES",
    "MATURITIES",
    "MaturityDisciplineError",
    "REGISTRY",
    "Registry",
    "capability",
    "collect_problems",
    "dump",
    "import_registry",
    "main",
    "scan_static",
    "validate_capability",
]

PKG_NAME = "tensorfvm"
REPO_NAME = "TensorFVM"

#: Maturity ladder (mirrors the platform column
#: ``ck_capreg_maturity CHECK (maturity = ANY ('mature','developing','planned'))``).
MATURITIES: tuple[str, ...] = ("mature", "developing", "planned")

#: Coarse capability families: the five missing interfaces plus the physics
#: domains TensorFVM actually ships.
CAPABILITY_KINDS: tuple[str, ...] = (
    "solver",            # a time-stepping production solver
    "mesh",              # mesh construction / mesh contract
    "flux",              # flux / interface transfer operator
    "free_surface",      # VOF / interface reconstruction & advection
    "multiphase",        # two-phase, cavitation, drift flux, Stefan
    "turbulence",        # RANS/LES closures and wall treatment
    "compressible",      # compressible / shock-capturing
    "linear_solver",     # linear algebra backend
    "time_integration",  # time marching / coupling
    "boundary",          # boundary conditions
    "thermal",           # conjugate heat transfer
    "cavitation",        # cavitation inception / erosion
    "ice",               # ice / freezing
    "seakeeping",        # wave tank / hydrodynamics
    "maneuvering",       # manoeuvring / added mass
    "differentiable",    # adjoint / shape sensitivity
    "verification",      # verification & uncertainty machinery
    "backend",           # device / partition backend
    "module",            # anything else
)

#: Evidence grades used in ``evidence["grade"]``.
#:
#: ``ANALYTIC``      closed form / manufactured solution (gives an error band)
#: ``XVALUE``        external reference (benchmark, experiment, literature)
#: ``CONSERVATION``  exact conservation / algebraic identity, machine-precision
#: ``MECHANICS``     qualitative behaviour only (symmetry, boundedness, limit)
#: ``MOCK``          hard-coded or random stand-in
#: ``NONE``         no evidence
EVIDENCE_GRADES: tuple[str, ...] = (
    "ANALYTIC", "XVALUE", "CONSERVATION", "MECHANICS", "MOCK", "NONE",
)

#: Grades that may back ``verified=True``.  ``MECHANICS`` may not: a symmetry
#: check does not tell you how far the number is from the reference.
VERIFYING_GRADES: tuple[str, ...] = ("ANALYTIC", "XVALUE", "CONSERVATION")

ENDPOINT_SCHEMES: tuple[str, ...] = ("cli:", "route:", "api:")

#: Platform accuracy standard: a reference error above this is not acceptable.
MAX_ACCEPTABLE_ERROR_PCT = 15.0

_SEVERITY_ERROR = "error"
_SEVERITY_WARNING = "warning"


class MaturityDisciplineError(RuntimeError):
    """Raised when a registration would put a false maturity on the board."""


class DuplicateCapabilityError(RuntimeError):
    """Raised when one capability id is registered by two different objects."""


# --------------------------------------------------------------------------- #
# Capability record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Capability:
    id: str
    kind: str = "module"
    maturity: str = "developing"
    module_path: str = ""
    entry: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    wired: bool = False
    tested: bool = False
    verified: bool = False
    test_path: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    endpoint: str | None = None
    source: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "maturity": self.maturity,
            "module_path": self.module_path,
            "entry": self.entry,
            "params": dict(self.params),
            "wired": bool(self.wired),
            "tested": bool(self.tested),
            "verified": bool(self.verified),
            "test_path": self.test_path,
            "evidence": dict(self.evidence),
            "endpoint": self.endpoint,
            "source": self.source,
        }


def validate_capability(cap: Capability) -> list[dict[str, Any]]:
    """Return the discipline problems for one capability (empty == clean)."""
    problems: list[dict[str, Any]] = []

    def add(severity: str, code: str, detail: str) -> None:
        problems.append({"severity": severity, "code": code,
                         "capability_id": cap.id, "detail": detail})

    if cap.maturity not in MATURITIES:
        add(_SEVERITY_ERROR, "UNKNOWN_MATURITY",
            f"maturity={cap.maturity!r} not one of {MATURITIES}")
    if cap.kind not in CAPABILITY_KINDS:
        add(_SEVERITY_WARNING, "UNKNOWN_KIND",
            f"kind={cap.kind!r} not one of the declared families {CAPABILITY_KINDS}")
    if not cap.module_path:
        add(_SEVERITY_ERROR, "NO_MODULE_PATH", "module_path is required")
    if not cap.id or cap.id.count(".") < 1:
        add(_SEVERITY_ERROR, "BAD_ID", f"id={cap.id!r} must be dotted, e.g. 'fvm.kind.name'")
    if cap.id and not cap.id.startswith(f"fvm."):
        add(_SEVERITY_WARNING, "ID_PREFIX",
            f"id={cap.id!r} should start with 'fvm.' so the platform can group it")

    grade = str(cap.evidence.get("grade", "")).upper() or None
    if grade is not None and grade not in EVIDENCE_GRADES:
        add(_SEVERITY_ERROR, "UNKNOWN_GRADE", f"evidence.grade={grade!r} unknown")

    # --- anti-overclaim: mature needs wiring and evidence ------------------ #
    if cap.maturity == "mature":
        if not cap.wired:
            add(_SEVERITY_ERROR, "MATURE_WITHOUT_WIRING",
                "maturity='mature' requires wired=True (nothing imports this entry)")
        if not cap.verified:
            add(_SEVERITY_ERROR, "MATURE_WITHOUT_VERIFICATION",
                "maturity='mature' requires verified=True with a reference error band")
        if not cap.tested:
            add(_SEVERITY_WARNING, "MATURE_UNTESTED",
                "maturity='mature' without tested=True")
        if not cap.test_path:
            add(_SEVERITY_WARNING, "MATURE_WITHOUT_TEST_PATH",
                "maturity='mature' without test_path -- the gate cannot be re-run")

    if cap.maturity == "planned":
        if cap.verified:
            add(_SEVERITY_ERROR, "PLANNED_VERIFIED",
                "maturity='planned' cannot be verified=True; promote to developing/mature")
        if not cap.evidence.get("feature_ledger"):
            add(_SEVERITY_ERROR, "PLANNED_WITHOUT_LEDGER",
                "maturity='planned' requires evidence.feature_ledger (a booked roadmap row)")
        if cap.wired:
            add(_SEVERITY_WARNING, "PLANNED_WIRED",
                "maturity='planned' but wired=True -- promote it")

    if cap.verified and grade and grade not in VERIFYING_GRADES:
        add(_SEVERITY_ERROR, "VERIFIED_ON_WEAK_EVIDENCE",
            f"verified=True with evidence.grade={grade!r} (needs one of {VERIFYING_GRADES})")
    if cap.verified and not grade:
        add(_SEVERITY_ERROR, "VERIFIED_WITHOUT_GRADE",
            "verified=True but evidence.grade absent")
    if cap.verified and grade in VERIFYING_GRADES:
        if "max_error_pct" not in cap.evidence:
            add(_SEVERITY_WARNING, "VERIFIED_WITHOUT_MAX_ERROR",
                "verified=True but evidence.max_error_pct absent -- band undisclosed")
        elif float(cap.evidence["max_error_pct"]) > MAX_ACCEPTABLE_ERROR_PCT:
            add(_SEVERITY_ERROR, "VERIFIED_OVER_ACCEPTANCE",
                f"max_error_pct={cap.evidence['max_error_pct']} exceeds the "
                f"{MAX_ACCEPTABLE_ERROR_PCT}% acceptance line")

    if cap.endpoint:
        if not any(cap.endpoint.startswith(s) for s in ENDPOINT_SCHEMES):
            add(_SEVERITY_ERROR, "ENDPOINT_BAD_SCHEME",
                f"endpoint={cap.endpoint!r} must start with one of {ENDPOINT_SCHEMES}")
        if not cap.wired:
            add(_SEVERITY_ERROR, "ENDPOINT_WITHOUT_WIRING",
                "endpoint declared for a capability that is not wired")
    elif cap.wired and cap.kind == "solver":
        add(_SEVERITY_WARNING, "WIRED_SOLVER_WITHOUT_ENDPOINT",
            "wired solver capability without an endpoint a consumer could call")

    if cap.tested and not cap.test_path:
        add(_SEVERITY_WARNING, "TESTED_WITHOUT_TEST_PATH",
            "tested=True but no test_path to re-run")
    return problems


# --------------------------------------------------------------------------- #
# Registry singleton
# --------------------------------------------------------------------------- #


class Registry:
    """Process-wide capability table (self-registration target)."""

    def __init__(self) -> None:
        self._caps: MutableMapping[str, Capability] = {}
        self.problems: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []

    def register(self, cap: Capability, *, strict: bool = True) -> Capability:
        problems = validate_capability(cap)
        self.problems.extend(problems)
        blocking = [p for p in problems if p["severity"] == _SEVERITY_ERROR]
        if blocking:
            codes = ", ".join(sorted({p["code"] for p in blocking}))
            if strict:
                raise MaturityDisciplineError(
                    f"capability {cap.id!r} violates registry discipline: {codes}")
            self.skipped.append({"id": cap.id,
                                 "codes": sorted({p["code"] for p in blocking})})
            return cap
        if cap.id in self._caps:
            raise DuplicateCapabilityError(
                f"capability id {cap.id!r} registered twice "
                f"({self._caps[cap.id].source} and {cap.source})")
        self._caps[cap.id] = cap
        return cap

    def __contains__(self, cid: object) -> bool:
        return cid in self._caps

    def __len__(self) -> int:
        return len(self._caps)

    def __iter__(self) -> Iterator[Capability]:
        return iter(self.capabilities())

    def get(self, cid: str) -> Capability:
        return self._caps[cid]

    def capabilities(self) -> list[Capability]:
        return [self._caps[k] for k in sorted(self._caps)]

    def ids(self) -> list[str]:
        return sorted(self._caps)

    def clear(self) -> None:
        """Drop every registration (test helper; production code never calls it)."""
        self._caps.clear()
        self.problems.clear()
        self.skipped.clear()

    def summary(self) -> dict[str, Any]:
        by_maturity = {m: 0 for m in MATURITIES}
        by_kind: dict[str, int] = {}
        for cap in self._caps.values():
            by_maturity[cap.maturity] = by_maturity.get(cap.maturity, 0) + 1
            by_kind[cap.kind] = by_kind.get(cap.kind, 0) + 1
        return {
            "count": len(self._caps),
            "by_maturity": by_maturity,
            "by_kind": dict(sorted(by_kind.items())),
            "wired": sum(1 for c in self._caps.values() if c.wired),
            "tested": sum(1 for c in self._caps.values() if c.tested),
            "verified": sum(1 for c in self._caps.values() if c.verified),
            "with_endpoint": sum(1 for c in self._caps.values() if c.endpoint),
        }

    def dump(self, *, source_commit: str | None = None, mode: str = "import") -> dict[str, Any]:
        caps = [c.to_payload() for c in self.capabilities()]
        return {
            "repo": REPO_NAME,
            "mode": mode,
            "source_commit": source_commit or "unknown",
            "count": len(caps),
            "summary": self.summary(),
            "capabilities": caps,
            "problems": list(self.problems),
            "skipped": list(self.skipped),
        }


#: The singleton every ``@capability`` decorator feeds.
REGISTRY = Registry()


def capability(
    id: str,
    kind: str = "module",
    maturity: str = "developing",
    module_path: str | None = None,
    entry: str | None = None,
    params: Mapping[str, Any] | None = None,
    wired: bool = False,
    tested: bool = False,
    verified: bool = False,
    test_path: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    endpoint: str | None = None,
    *,
    registry: Registry | None = None,
    strict: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the decorated function/class as a capability at import time.

    ``module_path`` and ``entry`` default to the decorated object's own module
    and qualified name, so the common case needs no boilerplate -- and the static
    AST scanner derives exactly the same two values from the source file, which
    keeps the import-mode and static-mode dumps comparable field by field.
    """
    registry = REGISTRY if registry is None else registry

    def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
        mod = module_path or getattr(obj, "__module__", "") or ""
        name = entry or getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
        cap = Capability(
            id=id,
            kind=kind,
            maturity=maturity,
            module_path=mod,
            entry=name,
            params=dict(params or {}),
            wired=wired,
            tested=tested,
            verified=verified,
            test_path=test_path,
            evidence=dict(evidence or {}),
            endpoint=endpoint,
            source=f"{mod}:{name}",
        )
        registry.register(cap, strict=strict)
        # The registry stores the entry name; keep the object itself reachable
        # from the decorator so tools can call it without re-importing by name.
        try:
            obj.__capability_id__ = id  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        return obj

    return decorator


def collect_problems(caps: Iterable[Capability]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cap in caps:
        out.extend(validate_capability(cap))
    return out


# --------------------------------------------------------------------------- #
# Pointer resolution (shared vocabulary for both modes)
# --------------------------------------------------------------------------- #


def _repo_root_from(pkg_root: Path) -> Path:
    """``<repo>/src/tensorfvm`` -> ``<repo>`` (falling back to the parent)."""
    for parent in [pkg_root, *pkg_root.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return pkg_root.parent.parent if len(pkg_root.parents) >= 2 else pkg_root


def _module_file(pkg_root: Path, module_path: str) -> Path | None:
    if not module_path.startswith(PKG_NAME):
        return None
    rel = module_path.split(".")
    rel = rel[1:]
    if not rel:
        return None
    candidate = pkg_root.joinpath(*rel).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = pkg_root.joinpath(*rel) / "__init__.py"
    return package if package.is_file() else None


def _defines(tree: ast.AST, qualified: str) -> bool:
    """Is ``qualified`` (possibly ``Class.method``) defined in this module?"""
    parts = [p for p in qualified.split(".") if p]
    node: Any = tree
    for depth, part in enumerate(parts):
        body = getattr(node, "body", [])
        found = None
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and stmt.name == part:
                found = stmt
                break
        if found is None:
            return False
        node = found
        if depth < len(parts) - 1 and not isinstance(node, ast.ClassDef):
            return False
    return True


def _test_functions(tree: ast.AST) -> set[str]:
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test")}


def resolve_pointers(cap: Capability, *, repo_root: Path, pkg_root: Path) -> list[dict[str, Any]]:
    """Check that module_path / entry / test_path exist in this working tree."""
    problems: list[dict[str, Any]] = []
    where = {"capability_id": cap.id, "file": cap.source}
    module_file = _module_file(pkg_root, cap.module_path)
    if module_file is None:
        problems.append({"severity": _SEVERITY_ERROR, "code": "MODULE_UNRESOLVED",
                         "detail": f"module_path={cap.module_path!r} is not a module "
                                   f"under {pkg_root.name}/", **where})
    elif cap.entry:
        try:
            tree = ast.parse(module_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            tree = None
        if tree is not None and not _defines(tree, cap.entry):
            problems.append({"severity": _SEVERITY_ERROR, "code": "ENTRY_UNRESOLVED",
                             "detail": f"entry={cap.entry!r} not defined in "
                                       f"{module_file.relative_to(repo_root)}", **where})
    if cap.test_path:
        test_file = repo_root / cap.test_path
        if not test_file.is_file():
            problems.append({"severity": _SEVERITY_ERROR, "code": "TEST_FILE_MISSING",
                             "detail": f"test_path={cap.test_path!r} does not exist", **where})
        else:
            try:
                tree = ast.parse(test_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                tree = None
            if tree is not None and not _test_functions(tree):
                problems.append({"severity": _SEVERITY_ERROR, "code": "TEST_FILE_EMPTY",
                                 "detail": f"test_path={cap.test_path!r} defines no test_*",
                                 **where})
    return problems


# --------------------------------------------------------------------------- #
# Static (AST) mode
# --------------------------------------------------------------------------- #


def _literal(node: ast.AST, default: Any = None) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return default


def _decorator_capability(dec: ast.expr) -> ast.Call | None:
    func = dec.func if isinstance(dec, ast.Call) else dec
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name != "capability" or not isinstance(dec, ast.Call):
        return None
    return dec


def scan_static(pkg_root: Path | str | None = None) -> tuple[list[Capability], list[dict[str, Any]]]:
    """Build the capability list from the source without importing the package."""
    root_pkg = Path(pkg_root) if pkg_root is not None else Path(__file__).resolve().parent
    repo_root = _repo_root_from(root_pkg)
    caps: list[Capability] = []
    problems: list[dict[str, Any]] = []
    seen: dict[str, str] = {}

    for path in sorted(root_pkg.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        dotted = ".".join([PKG_NAME, *path.relative_to(root_pkg).with_suffix("").parts])
        dotted = dotted.removesuffix(".__init__")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for dec in node.decorator_list:
                call = _decorator_capability(dec)
                if call is None:
                    continue
                where = {"file": str(path.relative_to(repo_root)), "line": node.lineno}
                args = [_literal(a) for a in call.args]
                kwargs = {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg}
                if not args or not isinstance(args[0], str):
                    problems.append({"severity": _SEVERITY_ERROR, "code": "BAD_ID",
                                     "capability_id": None,
                                     "detail": "@capability needs a literal id", **where})
                    continue
                cap = Capability(
                    id=args[0],
                    kind=kwargs.get("kind", "module"),
                    maturity=kwargs.get("maturity", "developing"),
                    module_path=kwargs.get("module_path") or dotted,
                    entry=kwargs.get("entry") or node.name,
                    params=kwargs.get("params") or {},
                    wired=bool(kwargs.get("wired", False)),
                    tested=bool(kwargs.get("tested", False)),
                    verified=bool(kwargs.get("verified", False)),
                    test_path=kwargs.get("test_path"),
                    evidence=kwargs.get("evidence") or {},
                    endpoint=kwargs.get("endpoint"),
                    source=f"{dotted}:{node.name}",
                )
                for prob in validate_capability(cap):
                    prob.setdefault("file", where["file"])
                    prob.setdefault("line", where["line"])
                    problems.append(prob)
                problems.extend(resolve_pointers(cap, repo_root=repo_root, pkg_root=root_pkg))
                if cap.id in seen:
                    problems.append({"severity": _SEVERITY_ERROR,
                                     "code": "STATIC_DUPLICATE_ID",
                                     "capability_id": cap.id,
                                     "detail": f"id already declared at {seen[cap.id]}",
                                     **where})
                    continue
                seen[cap.id] = f"{where['file']}:{where['line']}"
                caps.append(cap)
    caps.sort(key=lambda c: c.id)
    return caps, problems


# --------------------------------------------------------------------------- #
# Import mode
# --------------------------------------------------------------------------- #


def _is_capability_decorator(dec: ast.expr) -> bool:
    return _decorator_capability(dec) is not None


def _modules_with_decorators(pkg_root: Path) -> list[str]:
    """Dotted modules containing at least one ``@capability`` decorator."""
    mods: list[str] = []
    for path in sorted(pkg_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        decorated = any(
            _is_capability_decorator(dec)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for dec in node.decorator_list
        )
        if not decorated:
            continue
        parts = [PKG_NAME, *path.relative_to(pkg_root).with_suffix("").parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods.append(".".join(parts))
    return mods


def import_registry(
    modules: Iterable[str] | None = None,
    *,
    pkg_root: Path | None = None,
    registry: Registry | None = None,
) -> tuple[Registry, dict[str, Any]]:
    """Import the decorated modules and return the populated registry.

    Requires a runnable environment (torch etc.).  Import failures are recorded
    instead of raised, so one broken optional backend cannot blank the inventory.
    """
    import importlib

    root = Path(pkg_root) if pkg_root is not None else Path(__file__).resolve().parent
    parent = str(root.parent)
    if sys.path[:1] != [parent]:
        sys.path.insert(0, parent)
    shadow = sys.modules.get(PKG_NAME)
    shadow_path = getattr(shadow, "__file__", None) if shadow is not None else None
    if shadow is not None and shadow_path and Path(shadow_path).resolve() != (root / "__init__.py"):
        # An editable install from a sibling checkout would register *its*
        # inventory; drop it so this working tree wins, and say so.
        for name in [m for m in sys.modules if m == PKG_NAME or m.startswith(PKG_NAME + ".")]:
            del sys.modules[name]

    # The decorators run inside ``tensorfvm.registry`` as imported from the
    # package.  When *this* file was executed as a script its ``REGISTRY`` is a
    # different object from the one the package modules feed, so resolve the
    # authoritative module first and read the singleton from there.
    authoritative = importlib.import_module(f"{PKG_NAME}.registry")
    target = registry if registry is not None else authoritative.REGISTRY

    wanted = list(modules) if modules is not None else _modules_with_decorators(root)
    status = {"imported": [], "failed": [], "shadowed_package": shadow_path}
    for name in wanted:
        try:
            importlib.import_module(name)
            status["imported"].append(name)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            status["failed"].append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    return target, status


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _source_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("TENSORFVM_SOURCE_COMMIT", "unknown")


def dump(
    *,
    static: bool = False,
    strict: bool = False,
    pkg_root: Path | None = None,
) -> dict[str, Any]:
    """Produce the JSON payload in either mode (static never imports the package)."""
    root = Path(pkg_root) if pkg_root is not None else Path(__file__).resolve().parent
    repo_root = _repo_root_from(root)
    if static:
        caps, problems = scan_static(root)
        payload = {
            "repo": REPO_NAME,
            "mode": "static",
            "source_commit": _source_commit(repo_root),
            "count": len(caps),
            "summary": _caps_summary(caps),
            "capabilities": [c.to_payload() for c in caps],
            "problems": problems,
        }
        if strict:
            payload["errors"] = [p for p in problems if p["severity"] == _SEVERITY_ERROR]
        return payload

    registry, status = import_registry(pkg_root=root)
    payload = registry.dump(source_commit=_source_commit(repo_root), mode="import")
    payload["imports"] = status
    if strict:
        payload["errors"] = [p for p in payload["problems"]
                             if p["severity"] == _SEVERITY_ERROR]
    return payload


def _caps_summary(caps: Iterable[Capability]) -> dict[str, Any]:
    caps = list(caps)
    by_maturity = {m: 0 for m in MATURITIES}
    by_kind: dict[str, int] = {}
    for cap in caps:
        by_maturity[cap.maturity] = by_maturity.get(cap.maturity, 0) + 1
        by_kind[cap.kind] = by_kind.get(cap.kind, 0) + 1
    return {
        "count": len(caps),
        "by_maturity": by_maturity,
        "by_kind": dict(sorted(by_kind.items())),
        "wired": sum(1 for c in caps if c.wired),
        "tested": sum(1 for c in caps if c.tested),
        "verified": sum(1 for c in caps if c.verified),
        "with_endpoint": sum(1 for c in caps if c.endpoint),
    }


def _parity(static_payload: Mapping[str, Any], import_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the two modes field by field; they must agree."""
    left = {c["id"]: c for c in static_payload["capabilities"]}
    right = {c["id"]: c for c in import_payload["capabilities"]}
    mismatched = []
    for cid in sorted(set(left) | set(right)):
        if cid not in left or cid not in right:
            mismatched.append({"id": cid, "reason": "missing in one mode"})
            continue
        for key in ("kind", "maturity", "module_path", "entry", "wired", "tested",
                    "verified", "test_path", "endpoint"):
            if left[cid][key] != right[cid][key]:
                mismatched.append({"id": cid, "field": key,
                                   "static": left[cid][key], "import": right[cid][key]})
    return {"ok": not mismatched, "static_count": len(left),
            "import_count": len(right), "mismatched": mismatched,
            "import_failures": import_payload.get("imports", {}).get("failed", [])}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TensorFVM capability registry dump")
    ap.add_argument("--static", action="store_true",
                    help="AST mode: parse decorators, import nothing (no torch needed)")
    ap.add_argument("--strict", action="store_true",
                    help="also list the error-severity problems under 'errors'")
    ap.add_argument("--strict-exit", action="store_true",
                    help="exit 1 when a problem has error severity")
    ap.add_argument("--parity", action="store_true",
                    help="compare static and import dumps field by field")
    ap.add_argument("--pkg-root", type=Path, default=None,
                    help="override the package root (default: this file's directory)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    payload = dump(static=args.static, strict=args.strict, pkg_root=args.pkg_root)
    if args.parity:
        static_payload = dump(static=True, pkg_root=args.pkg_root)
        import_payload = dump(static=False, pkg_root=args.pkg_root)
        payload = {"repo": REPO_NAME, "mode": "parity",
                   "parity": _parity(static_payload, import_payload),
                   "source_commit": static_payload["source_commit"]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict_exit:
        errors = [p for p in payload.get("problems", []) if p["severity"] == _SEVERITY_ERROR]
        return 1 if errors else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())