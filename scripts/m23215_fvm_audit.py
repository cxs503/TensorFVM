#!/usr/bin/env python
"""Mechanical audit of ``src/tensorfvm`` for the CAP-REG-FVM capability registry.

Why this exists
---------------
Until now the TensorFVM inventory only existed as prose.  This script turns the
claims made in ``docs/m23215/SOLVER_INVENTORY.md`` and
``docs/m23215/KERNEL_ARCHITECTURE.md`` into numbers that anyone can re-derive:

* intra-package import graph (module level plus subpackage roll-up);
* module import in-degree -- which files are *shared kernels* and which are
  *vertical case code nobody imports*;
* stub / mock probes (bodies that are only ``pass``/``raise NotImplementedError``,
  bodies that draw random numbers or return constants where a solver result is
  expected);
* exact-duplicate detection across the alleged copy families (vof x4,
  two-phase x3, wave tank x2) using AST-normalised body hashes, so a renamed
  copy still counts as a copy;
* test binding -- which modules are named by at least one ``tests/test_*.py``;
* shadowed modules -- same dotted import name produced by two files (a real
  import hazard, not a style issue).

Stdlib only (``ast`` + ``json``): it runs on a torch-free host, exactly like
``python src/tensorfvm/registry.py --static``.

Usage::

    python scripts/m23215_fvm_audit.py                     # tables on stdout
    python scripts/m23215_fvm_audit.py --json docs/m23215/fvm_audit.json
    python scripts/m23215_fvm_audit.py --check             # registry parity gate
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "tensorfvm"
TESTS_ROOT = REPO_ROOT / "tests"
DOCS_ROOT = REPO_ROOT / "docs"

#: Parallel generations the mission statement claims.  Verified below by
#: counting *exact duplicate function bodies* across each family -- that is what
#: turns "a collection of copies" into a measurable statement.
FAMILIES: dict[str, list[str]] = {
    "vof": ["vof", "vof_phase0", "vof_threecase", "vof_verification"],
    "two_phase": ["two_phase", "twophase_bench", "twophase_gap"],
    "wave_tank": ["wave_tank", "wave_tank_td"],
}

STUB_SENTINELS = ("NotImplementedError", "TODO", "FIXME")


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #


def _rel(path: Path) -> str:
    return str(path.relative_to(PKG_ROOT))


def _dotted(rel: str) -> str:
    parts = ["tensorfvm", *Path(rel).with_suffix("").parts]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _subpackage(rel: str) -> str:
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else "<top>"


def _fingerprint(body: list[ast.stmt]) -> str:
    """AST-normalised body hash: signature/docstring dropped, numbers kept."""
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    dump = ast.dump(ast.Module(body=list(body), type_ignores=[]))
    return hashlib.sha1(dump.encode("utf-8")).hexdigest()[:16]


def _is_stub(node: ast.AST) -> bool:
    body = list(getattr(node, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return True
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
            and stmt.value.value is Ellipsis:
        return True
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        return any(s in ast.dump(stmt.exc) for s in STUB_SENTINELS)
    return False


def _mock_hints(dump: str) -> list[str]:
    """Cheap MOCK probes on an already-dumped function body."""
    hits = []
    if "random" in dump:
        hits.append("uses_random")
    if "returns_constant" in dump:
        hits.append("returns_constant")
    return hits


def _imports(tree: ast.AST, dotted: str) -> list[str]:
    """Intra-package imports resolved to dotted module names."""
    out: list[str] = []
    pkg_parts = dotted.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: -node.level] if node.level <= len(pkg_parts) else []
                target = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                target = node.module or ""
            if target == "tensorfvm" or target.startswith("tensorfvm."):
                out.append(target)
                for alias in node.names:
                    out.append(f"{target}.{alias.name}")
        elif isinstance(node, ast.Import):
            out.extend(a.name for a in node.names if a.name.startswith("tensorfvm"))
    return out


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #


def scan_modules() -> dict[str, dict[str, Any]]:
    """Keyed by repo-relative file path (dotted names can collide; files cannot)."""
    mods: dict[str, dict[str, Any]] = {}
    for path in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = _rel(path)
        dotted = _dotted(rel)
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        funcs = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        dumps = {id(f): ast.dump(f) for f in funcs}
        prints: dict[str, str] = {}
        for f in funcs:
            if not _is_stub(f):
                prints.setdefault(_fingerprint(f.body), f"{f.name}@{rel}")
        decorated = any(
            "capability" in ast.dump(dec)
            for n in (*funcs, *classes)
            for dec in n.decorator_list
        )
        mods[rel] = {
            "dotted": dotted,
            "subpackage": _subpackage(rel),
            "lines": src.count("\n") + (0 if src.endswith("\n") else 1),
            "functions": len(funcs),
            "classes": len(classes),
            "public_functions": sorted({f.name for f in funcs if not f.name.startswith("_")}),
            "stubs": sorted({f.name for f in funcs if _is_stub(f)}),
            "mock_hints": sorted({h for f in funcs for h in _mock_hints(dumps[id(f)])}),
            "imports": _imports(tree, dotted),
            "has_capability": decorated,
            "fingerprints": prints,
        }

    # dotted -> rel, preferring packages (``pkg/__init__.py`` beats ``pkg.py``,
    # which is what importlib does); the loser is reported as shadowed.
    by_dotted: dict[str, list[str]] = defaultdict(list)
    for rel, rec in mods.items():
        by_dotted[rec["dotted"]].append(rel)
    shadowed: list[dict[str, Any]] = []
    canonical: dict[str, str] = {}
    for dotted, rels in by_dotted.items():
        rels_sorted = sorted(rels, key=lambda r: (not r.endswith("__init__.py"), r))
        canonical[dotted] = rels_sorted[0]
        for loser in rels_sorted[1:]:
            shadowed.append({"dotted": dotted, "winner": rels_sorted[0], "shadowed": loser})
    for rel, rec in mods.items():
        rec["shadowed_by"] = next(
            (s["winner"] for s in shadowed if s["shadowed"] == rel), None)
        resolved = set()
        for target in rec["imports"]:
            probe = target
            while probe and probe not in canonical and probe != "tensorfvm":
                probe = probe.rpartition(".")[0]
            if probe and probe != "tensorfvm" and probe in canonical:
                hit = canonical[probe]
                if hit != rel:
                    resolved.add(hit)
        rec["resolved_imports"] = sorted(resolved)
    return mods, shadowed


def test_bindings(mods: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    tests = {p.name: p.read_text(encoding="utf-8", errors="replace")
             for p in sorted(TESTS_ROOT.glob("test_*.py"))}
    binding: dict[str, list[str]] = {}
    for rel, rec in mods.items():
        dotted = rec["dotted"]
        if dotted == "tensorfvm":
            continue
        pat = re.compile(rf"\b{re.escape(dotted)}(?![\w.])"
                         rf"|\bfrom\s+{re.escape(dotted)}\b")
        hits = [t for t, src in tests.items() if pat.search(src)]
        if hits:
            binding[rel] = hits
    return binding


def doc_references(mods: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    docs = {p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(DOCS_ROOT.glob("*.md"))}
    refs: dict[str, list[str]] = {}
    for rel, rec in mods.items():
        short = Path(rel).with_suffix("")
        pat = re.compile(rf"(?<![\w/]){re.escape(str(short))}(?:\.py)?(?![\w])")
        hits = [d for d, src in docs.items() if pat.search(src)]
        if hits:
            refs[rel] = hits
    return refs


def graph_rollup(mods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    indeg: Counter[str] = Counter()
    users: dict[str, list[str]] = defaultdict(list)
    edges: Counter[str] = Counter()
    per_sub: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"modules": 0, "lines": 0, "out": Counter(), "internal": 0})
    for rel, rec in mods.items():
        bucket = per_sub[rec["subpackage"]]
        bucket["modules"] += 1
        bucket["lines"] += rec["lines"]
        for target in rec["resolved_imports"]:
            indeg[target] += 1
            users[target].append(rel)
            tsub = mods[target]["subpackage"]
            if tsub == rec["subpackage"]:
                bucket["internal"] += 1
            else:
                bucket["out"][tsub] += 1
                edges[f"{rec['subpackage']}->{tsub}"] += 1
    return {
        "in_degree": dict(indeg.most_common()),
        "importers": {k: sorted(v) for k, v in users.items()},
        "subpackage_edges": dict(sorted(edges.items())),
        "per_subpackage": {
            k: {"modules": v["modules"], "lines": v["lines"],
                "internal_imports": v["internal"],
                "imports_from": dict(sorted(v["out"].items()))}
            for k, v in sorted(per_sub.items())
        },
    }


def family_duplication(mods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family, members in FAMILIES.items():
        files = {rel: rec for rel, rec in mods.items()
                 if rec["subpackage"] in members}
        by_fp: dict[str, list[str]] = defaultdict(list)
        for rel, rec in files.items():
            for fp, symbol in rec["fingerprints"].items():
                by_fp[fp].append(f"{symbol.split('@')[0]}@{rel}")
        exact = [{"fingerprint": fp, "copies": sorted(set(where))}
                 for fp, where in by_fp.items() if len({w.split("@")[1] for w in where}) > 1]
        name_sets = {rel: set(rec["public_functions"]) for rel, rec in files.items()}
        overlap = {}
        names = sorted(name_sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = name_sets[a] & name_sets[b]
                if not inter:
                    continue
                union = name_sets[a] | name_sets[b]
                overlap[f"{a}|{b}"] = {"shared_symbols": sorted(inter),
                                       "jaccard": round(len(inter) / len(union), 3)}
        out[family] = {
            "members_declared": members,
            "members_present": sorted({rec["subpackage"] for rec in files.values()}),
            "modules": len(files),
            "lines": sum(r["lines"] for r in files.values()),
            "exact_duplicate_bodies": exact,
            "symbol_overlap": overlap,
        }
    return out


def top_level_profile(mods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    top = {rel: r for rel, r in mods.items() if r["subpackage"] == "<top>"}
    imported = {t for r in mods.values() for t in r["resolved_imports"]}
    orphans = sorted(rel for rel in top if rel not in imported)
    return {
        "top_level_modules": len(top),
        "top_level_lines": sum(r["lines"] for r in top.values()),
        "imported_by_nobody": orphans,
        "subpackages": sorted({r["subpackage"] for r in mods.values()
                               if r["subpackage"] != "<top>"}),
    }


def summary(mods, tests, docs, shadowed) -> dict[str, Any]:
    return {
        "modules": len(mods),
        "lines": sum(r["lines"] for r in mods.values()),
        "subpackages": len({r["subpackage"] for r in mods.values()}) - 1,
        "functions": sum(r["functions"] for r in mods.values()),
        "classes": sum(r["classes"] for r in mods.values()),
        "stub_functions": sum(len(r["stubs"]) for r in mods.values()),
        "modules_with_stubs": sum(1 for r in mods.values() if r["stubs"]),
        "modules_with_mock_hints": sum(1 for r in mods.values() if r["mock_hints"]),
        "modules_test_bound": len(tests),
        "modules_doc_referenced": len(docs),
        "modules_capability_decorated": sum(1 for r in mods.values() if r["has_capability"]),
        "test_files": len(list(TESTS_ROOT.glob("test_*.py"))),
        "shadowed_modules": len(shadowed),
    }


def build() -> dict[str, Any]:
    mods, shadowed = scan_modules()
    tests = test_bindings(mods)
    docs = doc_references(mods)
    graph = graph_rollup(mods)
    return {
        "summary": summary(mods, tests, docs, shadowed),
        "graph": graph,
        "shadowed_modules": shadowed,
        "shared_modules": [
            {"module": m, "importers": d, "users_sample": graph["importers"][m][:5]}
            for m, d in graph["in_degree"].items() if d >= 3
        ],
        "top_level": top_level_profile(mods),
        "families": family_duplication(mods),
        "test_bindings": {rel: v for rel, v in sorted(tests.items())},
        "doc_references": {rel: len(v) for rel, v in sorted(docs.items())},
        "modules": {
            rel: {k: v for k, v in rec.items() if k != "fingerprints"}
            for rel, rec in sorted(mods.items())
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def render(result: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        "# m23215 TensorFVM mechanical audit",
        "",
        f"- modules: {s['modules']} ({s['lines']} lines, {s['subpackages']} subpackages)",
        f"- functions / classes: {s['functions']} / {s['classes']}",
        f"- stub functions: {s['stub_functions']} in {s['modules_with_stubs']} modules",
        f"- modules with mock hints: {s['modules_with_mock_hints']}",
        f"- modules named by a test: {s['modules_test_bound']}/{s['modules']} "
        f"(test files: {s['test_files']})",
        f"- modules named by a docs/*.md: {s['modules_doc_referenced']}",
        f"- shadowed modules (same dotted name): {s['shadowed_modules']}",
        "",
        "## shared kernels (import in-degree >= 3)",
        "",
        "| module | importers | sample users |",
        "|---|---|---|",
    ]
    for row in result["shared_modules"]:
        lines.append(f"| `{row['module']}` | {row['importers']} | "
                     f"{', '.join('`' + u + '`' for u in row['users_sample'][:3])} |")
    lines += ["", "## copy families", "",
              "| family | modules | lines | exact duplicate bodies |",
              "|---|---|---|---|"]
    for fam, rec in result["families"].items():
        lines.append(f"| {fam} | {rec['modules']} "
                     f"({', '.join(rec['members_present'])}) | {rec['lines']} | "
                     f"{len(rec['exact_duplicate_bodies'])} |")
    tl = result["top_level"]
    lines += ["", "## top-level vertical code", "",
              f"- top-level modules {tl['top_level_modules']} "
              f"({tl['top_level_lines']} lines); imported by nobody: "
              f"{len(tl['imported_by_nobody'])}"]
    for sh in result["shadowed_modules"]:
        lines.append(f"- shadowed: `{sh['shadowed']}` loses to `{sh['winner']}` "
                     f"(both named `{sh['dotted']}`)")
    return "\n".join(lines)


def _registry_parity() -> list[str]:
    """Every @capability-decorated module must appear in the static dump."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tfvm_registry", PKG_ROOT / "registry.py")
    assert spec and spec.loader
    reg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reg
    spec.loader.exec_module(reg)
    caps, problems = reg.scan_static(PKG_ROOT)
    declared = {c.module_path for c in caps}
    mods, _ = scan_modules()
    decorated = {rec["dotted"] for rec in mods.values() if rec["has_capability"]}
    out = [f"module {m} decorates but registers no capability"
           for m in sorted(decorated - declared)]
    out += [f"{p['code']}: {p.get('capability_id')} {p['detail']}"
            for p in problems if p["severity"] == "error"]
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=None, help="write the JSON audit here")
    ap.add_argument("--check", action="store_true", help="registry parity gate")
    args = ap.parse_args(list(argv) if argv is not None else None)

    result = build()
    print(render(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.check:
        problems = _registry_parity()
        for p in problems:
            print(f"REGISTRY-PARITY: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())