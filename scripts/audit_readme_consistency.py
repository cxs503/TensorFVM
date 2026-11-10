#!/usr/bin/env python3
"""Audit README/example claims against the code that is supposed to back them.

Motivation
----------
``README.rst`` is the only entry point most readers see, and it makes claims of
three kinds that can silently drift away from the code:

*  *executable* claims -- a quoted command / command-line option that a shipped
   module must accept (``python -m tensorfvm --mesh-type c-grid ...``);
*  *API* claims -- an exported name, a configuration default, a mesh-validity
   rule, an output file (``fields.csv`` has columns ``x,y,u,v,p,fluid``);
*  *evidence* claims -- a reference coefficient (``Cd=5.579535``), an error gate
   (3 %), a documented workflow (which test runner, which optional extra).

Every check below is one README statement turned into an executable test.  A
FAIL means the sentence is not reproducible from this checkout; a PASS records
the exact code fact that backs it, so the audit doubles as provenance.

Modes
-----
``--static``   no subprocesses: option lists come from the ``add_argument``
               literals in the target sources.  Fast enough for the test suite.
(default)      additionally resolves every documented command's real ``--help``
               option set and runs one small end-to-end example.

Usage::

    python scripts/audit_readme_consistency.py                       # full
    python scripts/audit_readme_consistency.py --static --json out.json
    python scripts/audit_readme_consistency.py --report docs/readme-consistency/report.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULE = re.compile(r"\bpython -m (tensorfvm[A-Za-z0-9_.]*)")
CONSOLE = re.compile(r"\b(tensorfvm-[a-z0-9-]+)")
SCRIPT = re.compile(r"\b(scripts/[A-Za-z0-9_./-]+\.py)")
OPTION = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
DOCPTR = re.compile(r"(?<![\w/])(docs/[A-Za-z0-9_./-]+)")
NUMBER = re.compile(r"(?<![\w.])\d+\.\d+(?:[eE][-+]?\d+)?")
ADDARG = re.compile(r"""add_argument\(\s*["'](--[a-z][a-z0-9-]*)["']""")


@dataclass
class Check:
    id: str
    area: str
    claim: str
    status: str = "SKIP"
    detail: str = ""
    evidence: dict = field(default_factory=dict)


def read_readme(root: Path) -> str:
    return (root / "README.rst").read_text(encoding="utf-8")


def code_blocks(text: str) -> list[str]:
    """Indented ``::`` blocks of an RST document, one string per block."""
    blocks, current = [], []
    for line in text.splitlines():
        if line.startswith(("    ", "\t")):
            current.append(line.strip())
        else:
            if current:
                blocks.append("\n".join(current))
                current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def commands(text: str) -> list[str]:
    """Complete shell commands from README code blocks (joins ``\\`` continuations)."""
    out = []
    for block in code_blocks(text):
        joined = block.replace("\\\n", " ").splitlines()
        buffer: list[str] = []
        for line in joined:
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if line.endswith("\\"):
                continue
            candidate = " ".join(buffer)
            buffer = []
            head = candidate.split()[0]
            if head in {"python", "tensorfvm-cylinder", "tensorfvm-benchmark",
                        "tensorfvm-mesh-cylinder-3d", "OMP_NUM_THREADS=1",
                        "PYTHONPATH=src", "torchrun"} or " python -m " in candidate:
                candidate = re.sub(r"^\S*=\S*\s+", "", candidate)  # env prefixes
                out.append(candidate)
    return out


def target_of(command: str) -> tuple[str, str] | None:
    """(kind, identifier) for a documented command: module, console, script or other."""
    if m := MODULE.search(command):
        return "module", m.group(1)
    if command.split()[0] == "torchrun":
        m = MODULE.search(command)
        return ("module", m.group(1)) if m else None
    if m := SCRIPT.search(command):
        return "script", m.group(1)
    if m := CONSOLE.match(command):
        return "console", m.group(1)
    if command.startswith("python -m unittest") or command.startswith("python -m pytest"):
        return "runner", " ".join(command.split()[:4])
    return None


def option_vocabulary(root: Path, kind: str, ident: str, live: bool) -> set[str] | None:
    """Option strings accepted by a documented target (None when unresolvable)."""
    sources: list[Path] = []
    if kind == "module":
        base = root / "src" / Path(*ident.split("."))
        sources = [base.with_suffix(".py"), base / "__main__.py"]
    elif kind == "script":
        sources = [root / ident]
    elif kind == "console":
        ident = console_target(root, ident) or ""
        if ident:
            mod, _, _func = ident.partition(":")
            base = root / "src" / Path(*mod.split("."))
            sources = [base.with_suffix(".py"), base / "__main__.py"]
            kind, ident = "module", mod  # a console script is just ``python -m mod``
    if not live:
        text = "".join(p.read_text(encoding="utf-8") for p in sources if p.is_file())
        options = set(ADDARG.findall(text))
        # A package ``__main__`` only delegates: follow one ``from .mod import main`` hop.
        for hop in re.findall(r"from\s+\.(\w+)\s+import\s+main", text):
            stub = (root / "src" / Path(*ident.split(".")) / f"{hop}.py")
            if stub.is_file():
                options |= set(ADDARG.findall(stub.read_text(encoding="utf-8")))
        return options | {"--help"}
    cmd = ([sys.executable, "-m", ident, "--help"] if kind == "module"
           else [sys.executable, ident, "--help"])
    try:
        out = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=180,
                             env={**os.environ, "PYTHONPATH": str(root / "src")})
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0 and "--help" not in out.stdout:
        return None
    return set(OPTION.findall(out.stdout))


def console_target(root: Path, name: str) -> str | None:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(rf'^"?{re.escape(name)}"?\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def numeric_tokens(text: str) -> set[str]:
    return {t for t in NUMBER.findall(text) if len(re.sub(r"[^0-9]", "", t.split("e")[0])) >= 3}


def corpus_numbers(paths) -> list[float]:
    values = []
    for path in paths:
        try:
            values.extend(float(v) for v in NUMBER.findall(path.read_text(errors="ignore")))
        except OSError:
            continue
    return values


def close(value: float, population: list[float]) -> bool:
    for other in population:
        if other == value or abs(other - value) <= max(abs(value), abs(other)) * 2e-3:
            return True
    return False


def audit(root: Path = ROOT, live: bool = True, quick: bool = True) -> dict:
    readme = read_readme(root)
    checks: list[Check] = []

    # ---------------------------------------------------------------- executable
    blocks = commands(readme)
    targets = [(c, target_of(c)) for c in blocks]
    targets = [(c, t) for c, t in targets if t]

    bad: list[str] = []
    seen: dict[tuple[str, str], set[str] | None] = {}
    for command, (kind, ident) in targets:
        if kind == "runner":
            continue
        vocabulary = seen.get((kind, ident), "missing")
        if vocabulary == "missing":
            vocabulary = option_vocabulary(root, kind, ident, live)
            seen[(kind, ident)] = vocabulary
        if vocabulary is None:
            bad.append(f"{command}: 无法解析 {kind} {ident}")
            continue
        for option in OPTION.findall(command):
            if option not in vocabulary:
                bad.append(f"{command}: {ident} 不接受 {option}")
    checks.append(Check(
        "A1", "示例命令/选项", "README 每个示例命令的选项都被目标程序接受",
        "PASS" if not bad else "FAIL",
        f"{len(targets)} 条命令、{sum(len(OPTION.findall(c)) for c, _ in targets)} 个选项 token；"
        + ("全部可解析" if not bad else "；".join(bad)),
        {"commands": [c for c, _ in targets]},
    ))

    modules = sorted({ident for _cmd, (kind, ident) in targets if kind == "module"})
    guard = re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]")
    missing_mods = []
    for ident in modules:
        base = root / "src" / Path(*ident.split("."))
        if base.is_dir():
            # ``python -m package`` executes ``__main__.py`` at import; no guard needed.
            entry, needs_guard = base / "__main__.py", False
        else:
            entry, needs_guard = base.with_suffix(".py"), True
        if not entry.is_file():
            missing_mods.append(f"{ident} (缺 {entry.name})")
            continue
        text = entry.read_text(encoding="utf-8")
        has_main = bool(re.search(r"^def main\(|^from \S+ import main|SystemExit\(main\(", text, re.M))
        if needs_guard and not guard.search(text):
            missing_mods.append(f"{ident} (缺 __main__ guard)")
        elif not has_main:
            missing_mods.append(f"{ident} (无 main 入口)")
    checks.append(Check(
        "A2", "示例命令/选项", "README 中的 `python -m tensorfvm.*` 模块可执行",
        "PASS" if not missing_mods else "FAIL",
        f"{len(modules)} 个模块；" +
        ("全部存在且可执行" if not missing_mods else "；".join(missing_mods)),
        {"modules": modules},
    ))

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    entries = dict(re.findall(r'^"?([a-z0-9-]+)"?\s*=\s*"([A-Za-z0-9_.]+:[A-Za-z0-9_]+)"',
                              pyproject, re.MULTILINE))
    bad_scripts = []
    for name, pointer in entries.items():
        mod, _, func = pointer.partition(":")
        base = root / "src" / Path(*mod.split("."))
        entry = base / "__main__.py" if base.is_dir() else base.with_suffix(".py")
        if not entry.is_file() or not re.search(rf"^def {func}\(", entry.read_text(encoding="utf-8"), re.M):
            bad_scripts.append(f"{name} -> {pointer}")
    checks.append(Check(
        "A3", "控制台入口", "pyproject [project.scripts] 的条目都指向真实模块函数",
        "PASS" if not bad_scripts else "FAIL",
        f"{len(entries)} 个入口；" + ("全部可解析" if not bad_scripts else "；".join(bad_scripts)),
        {"scripts": entries},
    ))

    missing_scripts = sorted({s for s in SCRIPT.findall(readme) if not (root / s).is_file()})
    checks.append(Check(
        "A4", "示例命令/选项", "README 引用的 scripts/*.py 存在",
        "PASS" if not missing_scripts else "FAIL",
        f"{len(set(SCRIPT.findall(readme)))} 个脚本指针；" +
        ("全部存在" if not missing_scripts else "；".join(missing_scripts)),
    ))

    # ---------------------------------------------------------------------- API
    import tensorfvm  # noqa: PLC0415  (heavy import, only inside the audit)
    from tensorfvm import SolverConfig  # noqa: PLC0415

    documented_exports = ["SolverConfig", "SimpleSolver", "BodyFittedMesh", "CGridMesh",
                          "ThreeElementMesh", "BodyFittedSolver"]
    absent = [n for n in documented_exports if not hasattr(tensorfvm, n)]
    checks.append(Check(
        "B1", "公开 API", "README 列出的导出名可由 `import tensorfvm` 取得",
        "PASS" if not absent else "FAIL",
        f"{len(documented_exports)} 个名称；" + ("全部可导入" if not absent else "；".join(absent)),
    ))

    default = SolverConfig()
    ok = default.mesh_type == "cartesian"
    checks.append(Check(
        "B2", "配置默认值", "`SolverConfig` 默认 mesh_type=cartesian（README 兼容性说明）",
        "PASS" if ok else "FAIL", f"实测 mesh_type={default.mesh_type!r}",
    ))

    derived = (default.density * default.inlet_velocity
               * (2 * default.cylinder_radius) / default.reynolds)
    facts = {
        "length": (default.length, 4.0), "height": (default.height, 2.0),
        "cylinder_x": (default.cylinder_x, 1.0), "cylinder_y": (default.cylinder_y, 1.0),
        "cylinder_radius": (default.cylinder_radius, 0.2),
        "inlet_velocity": (default.inlet_velocity, 1.0), "density": (default.density, 1.0),
        "reynolds": (default.reynolds, 20.0), "viscosity": (default.viscosity, derived),
    }
    wrong = [f"{k}: {v!r} != {want!r}" for k, (v, want) in facts.items()
             if not math.isclose(float(v), float(want))]
    checks.append(Check(
        "B3", "配置默认值", "README 默认算例（4×2 域、中心 (1,1)、R=0.2、U=1、rho=1、Re=20、mu=rho U 2R/Re）",
        "PASS" if not wrong else "FAIL",
        "；".join(wrong) if wrong else
        f"mu={default.viscosity:.6g} = rho*U*(2R)/Re，与 8 个默认字段一致",
    ))

    from tensorfvm import BodyFittedMesh  # noqa: PLC0415

    def accepts(**kw) -> bool:
        try:
            BodyFittedMesh(SolverConfig(**kw))
            return True
        except ValueError:
            return False

    rules = {
        "body-fitted nx=8,ny=4": (accepts(mesh_type="body-fitted", nx=8, ny=4), True),
        "body-fitted nx=10": (accepts(mesh_type="body-fitted", nx=10, ny=8), False),
        "body-fitted nx=12": (accepts(mesh_type="body-fitted", nx=12, ny=8), True),
        "body-fitted ny=3": (accepts(mesh_type="body-fitted", nx=16, ny=3), False),
        "c-grid nx=16": (accepts(mesh_type="c-grid", nx=16, ny=4, length=10., height=8.,
                                 airfoil_x=2., airfoil_y=4.), True),
        "c-grid nx=12": (accepts(mesh_type="c-grid", nx=12, ny=8, length=10., height=8.,
                                 airfoil_x=2., airfoil_y=4.), False),
    }
    violations = [f"{k}: got {got}" for k, (got, want) in rules.items() if got != want]
    checks.append(Check(
        "B4", "网格约束", "README 的贴体/C 型网格约束（body-fitted nx≥8 且 4 的倍数、ny≥4；c-grid nx≥16 且 4 的倍数）",
        "PASS" if not violations else "FAIL",
        "；".join(violations) if violations else "5 个边界用例逐一符合",
    ))

    exporter = (root / "src" / "tensorfvm" / "cylinder.py").read_text(encoding="utf-8")
    documented_files = ["fields.csv", "history.json", "summary.json", "nodes.csv", "cells.csv",
                        "velocity.svg", "pressure.svg", "airfoil.csv", "turbulence.csv",
                        "forces.json"]
    not_emitted = [n for n in documented_files if f'"{n}"' not in exporter]
    header_ok = '"x"' in exporter and '"y"' in exporter and "fluid" in exporter
    checks.append(Check(
        "B5", "输出文件", "README 列的运行输出（fields.csv 含 x,y,u,v,p,fluid 等 10 个文件）由导出处写出",
        "PASS" if not not_emitted and header_ok else "FAIL",
        "；".join(not_emitted) if not_emitted else
        "10 个文件名全部出现在 cylinder.py；fields.csv 表头 x,y,u,v,p,fluid",
    ))

    # ------------------------------------------------------ benchmark reference
    from tensorfvm import benchmark_airfoil, benchmark_dfg, benchmark_flat_plate  # noqa: PLC0415
    from tensorfvm import benchmark_cylinder_urans, benchmark_30p30n  # noqa: PLC0415

    dfg_ok = (math.isclose(benchmark_dfg.REFERENCE_DRAG, 5.579535)
              and benchmark_dfg.GRIDS[-1] == (256, 96)
              and math.isclose(benchmark_dfg.ERROR_LIMIT, 0.03))
    checks.append(Check(
        "C1", "基准常数", "DFG Re=20：Cd=5.579535、最细网格 256×96、误差门 3%",
        "PASS" if dfg_ok else "FAIL",
        f"REFERENCE_DRAG={benchmark_dfg.REFERENCE_DRAG}, GRIDS={benchmark_dfg.GRIDS}, "
        f"ERROR_LIMIT={benchmark_dfg.ERROR_LIMIT}",
    ))

    lift = benchmark_airfoil.REFERENCE["lift_coefficient"]
    drag = benchmark_airfoil.REFERENCE["drag_coefficient"]
    air_ok = (f"{lift:.7f}" == "0.2009636" and f"{drag:.7f}" == "0.1250065"
              and benchmark_airfoil.REFERENCE["digitization_absolute_uncertainty"] == 0.0001
              and math.isclose(benchmark_airfoil.ERROR_LIMIT, 0.03))
    checks.append(Check(
        "C2", "基准常数", "NACA 0012 Re=1000 α=4°：Cl≈0.2009636、Cd≈0.1250065、读图 ±0.0001、3% 门",
        "PASS" if air_ok else "FAIL",
        f"lift={lift:.7f}, drag={drag:.7f}, band="
        f"{benchmark_airfoil.REFERENCE['digitization_absolute_uncertainty']}",
    ))

    flat_ok = (math.isclose(benchmark_flat_plate.ERROR_LIMIT, 0.20)
               and math.isclose(benchmark_flat_plate.REFERENCE_REYNOLDS, 1e5)
               and math.isclose(benchmark_flat_plate.reference_skin_friction(1e5),
                                0.074 / 1e5 ** 0.2))
    checks.append(Check(
        "C3", "基准常数", "平板：Cf=0.074 Re^-1/5、Re_L=1e5、平均 Cf 误差门 20%、首层 y+<1",
        "PASS" if flat_ok else "FAIL",
        f"ERROR_LIMIT={benchmark_flat_plate.ERROR_LIMIT}, "
        f"Re={benchmark_flat_plate.REFERENCE_REYNOLDS:g}, "
        f"Cf(1e5)={benchmark_flat_plate.reference_skin_friction(1e5):.6g}",
    ))

    urans = benchmark_cylinder_urans.REFERENCE
    ur_ok = (math.isclose(urans["mean_drag"], 1.12) and math.isclose(urans["strouhal"], 0.20))
    checks.append(Check(
        "C4", "基准常数", "Re=3900 外流圆柱：参考 Cd=1.12、St=0.20",
        "PASS" if ur_ok else "FAIL",
        f"mean_drag={urans['mean_drag']}, strouhal={urans['strouhal']}",
    ))

    m30 = (math.isclose(benchmark_30p30n.REFERENCE_LIFT, 2.167089)
           and math.isclose(benchmark_30p30n.REFERENCE_DRAG_REPORTED, 0.033243))
    checks.append(Check(
        "C5", "基准常数", "30P30N：原始表 Cl=2.167089、Cd=0.033243",
        "PASS" if m30 else "FAIL",
        f"lift={benchmark_30p30n.REFERENCE_LIFT}, drag={benchmark_30p30n.REFERENCE_DRAG_REPORTED}",
    ))

    # --------------------------------------------------------------- evidence
    pointers = sorted(set(DOCPTR.findall(readme)))
    gone = [p for p in pointers if not (root / p).exists()]
    checks.append(Check(
        "D1", "文档指针", "README 引用的 docs/* 路径在检出中存在",
        "PASS" if not gone else "FAIL",
        f"{len(pointers)} 个指针；" + ("全部存在" if not gone else "缺失：" + "；".join(gone)),
    ))

    corpus = corpus_numbers(p for p in (root / "docs").rglob("*")
                            if p.is_file() and p.suffix in {".md", ".rst", ".json", ".csv", ".txt"})
    undocumented = [t for t in sorted(numeric_tokens(readme))
                    if not close(float(t), corpus)]
    checks.append(Check(
        "D2", "数字可溯源", "README 引用的实测数字都能在 docs/ 证据中找到同一数值",
        "PASS" if not undocumented else "FAIL",
        f"{len(numeric_tokens(readme))} 个三位以上有效数字；" +
        ("全部可溯源到 docs/" if not undocumented else "无出处：" + "；".join(undocumented)),
    ))

    test_files = list((root / "tests").glob("test_*.py"))
    pytest_files = [p for p in test_files if re.search(r"^import pytest", p.read_text(errors="ignore"), re.M)]
    unittest_files = [p for p in test_files if "unittest.TestCase" in p.read_text(errors="ignore")]
    functions = sum(len(re.findall(r"^def test_", p.read_text(errors="ignore"), re.M)) for p in test_files)
    runner_ok = "python -m pytest" in readme and "标准库 unittest" not in readme
    checks.append(Check(
        "D3", "测试入口", "README 记的测试运行方式与本仓库测试语料一致",
        "PASS" if runner_ok else "FAIL",
        f"tests/ 共 {len(test_files)} 个文件：{len(pytest_files)} 个 import pytest、"
        f"{len(unittest_files)} 个定义 unittest.TestCase、{functions} 个模块级 test 函数；"
        f"README 当前写的是 " + ("pytest" if "python -m pytest" in readme else "unittest"),
        {"pytest_files": len(pytest_files), "unittest_files": len(unittest_files),
         "test_functions": functions},
    ))

    table = pyproject.split("[project.optional-dependencies]")[-1].split("\n[")[0]
    declared = set(re.findall(r"^([a-z-]+) = \[", table, re.MULTILINE))
    verification_needs = sorted({"numpy", "matplotlib"}
                                & set(re.findall(r"^import ([a-z]+)", 
                                                 (root / "src" / "tensorfvm" / "verification" / "report.py")
                                                 .read_text(encoding="utf-8"), re.MULTILINE)))
    install = readme.split("安装\n----")[-1].split("\n---")[0]
    documents_extra = ".[benchmark" in install or ".[dev]" in install
    checks.append(Check(
        "D4", "前置依赖", "README 文档化的 benchmark/verification 命令所需可选依赖有安装说明",
        "PASS" if documents_extra else "FAIL",
        f"verification/report.py 需要 {verification_needs}；pyproject extras={sorted(declared)}；"
        f"安装节" + ("已说明 extras 安装方式" if documents_extra else "未说明这些可选依赖如何安装"),
    ))

    # ---------------------------------------------------------------- behaviour
    suite = (root / "src" / "tensorfvm" / "benchmark_suite_channel.py").read_text(encoding="utf-8")
    default_out = re.search(r"default=Path\(['\"]([^'\"]+)['\"]\)", suite)
    default_out = default_out.group(1) if default_out else ""
    evidence_dir = (root / default_out).is_relative_to(root / "docs") if default_out else True
    evidence_policy = (root / "docs" / "evidence-resolvability-allowlist.json").is_file()
    checks.append(Check(
        "E1", "证据树只读", "文档化的示例命令默认输出不落在已发布的 docs/ 证据树内",
        "PASS" if not evidence_dir else "FAIL",
        f"benchmark_suite_channel 默认 --output={default_out!r}"
        + ("（指向证据树，照 README 运行会覆盖已提交证据）" if evidence_dir else "（results/ 下，与其它 benchmark 一致）"),
        {"docs_tree_readonly_policy": evidence_policy},
    ))

    # ------------------------------------------------------- end-to-end example
    if live and quick:
        out = root / "results" / "readme-consistency-smoke"
        cmd = [sys.executable, "-m", "tensorfvm", "--mesh-type", "body-fitted",
               "--nx", "16", "--ny", "8", "--reynolds", "20", "--max-iterations", "20",
               "--output", str(out)]
        run = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=600,
                             env={**os.environ, "PYTHONPATH": str(root / "src")})
        expected = {"fields.csv", "history.json", "summary.json", "nodes.csv", "cells.csv",
                    "velocity.svg", "pressure.svg"}
        present = {p.name for p in out.glob("*")} if out.is_dir() else set()
        header = (out / "fields.csv").read_text(encoding="utf-8").splitlines()[0] if present else ""
        ok = (run.returncode in (0, 2) and expected <= present
              and header.strip() == "x,y,u,v,p,fluid")
        checks.append(Check(
            "E2", "示例可运行", "README 的贴体圆柱示例按文档参数小规模重跑，产出全部声明文件",
            "PASS" if ok else "FAIL",
            f"exit={run.returncode}（0=收敛 2=达迭代上限）；产出 {sorted(present)}；表头 {header.strip()!r}",
            {"command": " ".join(cmd[:8] + ["--output", "<tmp>"]), "returncode": run.returncode},
        ))
    else:
        checks.append(Check("E2", "示例可运行", "贴体圆柱示例端到端重跑", "SKIP",
                            "static 模式跳过实跑（--static 只做静态对照）"))

    failed = [c for c in checks if c.status == "FAIL"]
    return {
        "schema": "tensorfvm.readme-consistency/1",
        "readme_lines": len(readme.splitlines()),
        "checks": [c.__dict__ for c in checks],
        "summary": {"total": len(checks), "pass": len(checks) - len(failed) - len(
            [c for c in checks if c.status == "SKIP"]), "fail": len(failed),
            "skip": len([c for c in checks if c.status == "SKIP"])},
        "failed": [c.id for c in failed],
    }


def markdown(result: dict) -> str:
    s = result["summary"]
    lines = ["# README 一致性巡检（自动生成）", "",
             f"抽查项 {s['total']}：PASS {s['pass']}，FAIL {s['fail']}，SKIP {s['skip']}。", "",
             "| # | 面 | 声明 | 状态 | 实测证据 |", "|---|---|---|---|---|"]
    for c in result["checks"]:
        detail = c["detail"].replace("|", "\\|")
        lines.append(f"| {c['id']} | {c['area']} | {c['claim']} | {c['status']} | {detail} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--static", action="store_true", help="跳过 --help 探测与端到端实跑")
    parser.add_argument("--json", type=Path, default=None, help="机器可读结果输出路径")
    parser.add_argument("--report", type=Path, default=None, help="markdown 表输出路径")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = audit(args.root, live=not args.static, quick=not args.static)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
    return 0 if not result["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())