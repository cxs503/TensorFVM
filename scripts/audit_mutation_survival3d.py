#!/usr/bin/env python3
"""Mutation-survival audit of the **3-D native** finite-volume core.

Same protocol as ``scripts/audit_mutation_survival.py`` (which covers the 2-D
kernels): apply one catalogue defect to the real source tree, run the targeted
verification tests, restore the source and verify the sha256.  A mutant that
makes the suite fail is *killed*; a mutant the suite accepts is a measured hole
in the 3-D verification coverage -- not a statement about the production code.

Everything published in ``docs/mutation-survival3d/report.md`` is generated
here, so no number is transcribed by hand.

Usage: ``PYTHONPATH=src python scripts/audit_mutation_survival3d.py``
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "scripts", REPO):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tensorfvm.verification.mutation_catalogue3d import CATALOGUE3D  # noqa: E402
from audit_mutation_survival import (  # noqa: E402
    _version,
    apply_mutation,
    first_failure,
    run_tests,
    sha256,
    status_of,
)


def audit(entries, timeout):
    rows = []
    for entry in entries:
        path = REPO / entry["file"]
        before = sha256(path)
        row = dict(
            id=entry["id"],
            file=entry["file"],
            description=entry["description"],
            note=entry.get("note"),
            rationale=entry["rationale"],
            expected=entry["expected"],
            tests=entry["tests"],
        )
        started = time.perf_counter()
        try:
            target, mutated = apply_mutation(entry)
            target.write_text(mutated)
            if sha256(target) == before:
                raise RuntimeError("source hash unchanged after write")
            proc, elapsed = run_tests(entry, timeout)
            row["status"] = status_of(proc)
            row["returncode"] = proc.returncode
            row["first_failure"] = first_failure(proc)
            tail = [l for l in proc.stdout.strip().splitlines() if l.strip()][-1:] or [
                ""
            ]
            row["summary_line"] = tail[-1][:200]
        except Exception as exc:
            row["status"] = "invalid(exception)"
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            subprocess.run(
                ["git", "checkout", "--", entry["file"]],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            row["source_restored"] = bool(sha256(path) == before)
            row["source_sha256"] = before
        row["duration_seconds"] = round(time.perf_counter() - started, 2)
        print(
            f"{entry['id']}: {row['status']} ({row['duration_seconds']}s) "
            f"{row.get('first_failure') or ''}",
            flush=True,
        )
        rows.append(row)
    return rows


def write_figure(path, rows):
    ids = [r["id"] for r in rows]
    killed = [1 if r["status"] == "killed" else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(ids) + 1.8), constrained_layout=True)
    ax.barh(ids, killed, color=["#2b7a3d" if k else "#b03030" for k in killed])
    for i, (row, k) in enumerate(zip(rows, killed)):
        ax.text(
            k + 0.02,
            i,
            f"{row['status']} ({row['duration_seconds']}s)",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 1], ["survived", "killed"])
    ax.set_title(
        "TensorFVM mutation survival: 3-D native core "
        f"(killed {sum(killed)}/{len(ids)})"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path, rows, environment):
    killed = [r for r in rows if r["status"] == "killed"]
    survived = [r for r in rows if r["status"] == "survived"]
    invalid = [r for r in rows if r["status"].startswith("invalid")]
    lines = [
        "# 三维原生有限体积核的变异存活审计（3-D mutation coverage）",
        "",
        "## 1. 案例介绍",
        "",
        "本仓已有的变异审计 `src/tensorfvm/verification/mutation_catalogue.py` 自述只覆盖"
        "**二维**生产核（`body_fitted` / `solver`）；承担 SUBOFF、HLPW 类验收的三维原生路径"
        "（面/算子几何、相容面通量投影、同位网格 Rhie–Chow 型压力-速度耦合）从未做过故障注入。"
        "本审计对该缺口做一次可复现的测量，并回答一个问题：现有测试**是否约束了三维核的离散算子**，"
        "而不是它们算得准不准。",
        "",
        "每个变异体只改一行语义（符号、删项、权重互换、漏除体积），`find` 串必须在被改文件中"
        "**恰好出现一次**，否则判 `invalid`；改后编译、跑定向测试、再从 git 恢复源码并以 sha256 校验。",
        "",
        "## 2. 怎么做的",
        "",
        f"工具链：Python {environment['python']}、numpy {environment['numpy']}、"
        f"torch {environment['torch']}、scipy {environment['scipy']}；"
        "`scripts/audit_mutation_survival3d.py` 复用二维审计脚本的注入/恢复/判定函数，"
        "只换目录与输出，保证两次审计判据一致（exit=1 → killed，exit=0 → survived，其余 → invalid）。"
        "测试运行时 `OMP_NUM_THREADS=MKL_NUM_THREADS=1`，逐个变异体串行执行。",
        "",
        f"本次纳入 {len(rows)} 个变异体，覆盖 2 个模块："
        "`src/tensorfvm/native_collocated3d.py`（压力-速度耦合、面偏斜修正）与 "
        "`src/tensorfvm/native_operators3d.py`（仿射精确重构、Gauss 梯度）。",
        "",
        "## 3. 计算结果",
        "",
        "| 变异体 | 文件 | 缺陷 | 预声明 | 实测 | 首个失败测试 | 用时/s |",
        "|---|---|---|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {:.1f} |".format(
                r["id"],
                r["file"],
                r["description"],
                r["expected"],
                r["status"],
                r.get("first_failure") or "-",
                r["duration_seconds"],
            )
        )
    lines += [
        "",
        f"汇总：killed {len(killed)}，survived {len(survived)}，invalid {len(invalid)}，共 {len(rows)} 个。",
        "源码恢复校验：{} / {} 个变异体恢复了原始 sha256。".format(
            sum(1 for r in rows if r["source_restored"]), len(rows)
        ),
        "",
        "每个变异体的处置：",
        "",
    ]
    for r in rows:
        lines.append(f"- **{r['id']}** ({r['status']})：{r['rationale']}")
        if r.get("summary_line"):
            lines.append(f"  - pytest 末行：`{r['summary_line']}`")
        if r.get("note"):
            lines.append(f"  - 补测记录：{r['note']}")
    lines += [
        "",
        "## 4. 结果图",
        "",
        "`mutation-survival-3d.png`：逐变异体的 killed/survived 与用时（本审计程序化生成）。",
        "",
        "## 5. 注意点",
        "",
        "* 变异体是**人工挑选**的语义缺陷，不是随机缺陷分布，killed 比例不能当成缺陷检出概率的估计；",
        "* 判定依赖被指向的测试文件本身是否够格：定向测试若只断言「有限值」，本审计只会把它记为 survived；",
        "* 一个变异体 killed 只说明**某处**有约束，不说明这条断言正是为它写的；",
        "* 结论只覆盖被改的两行语义与所列测试文件，不外推到全部三维核；",
        "* 审计过程会临时改写工作树源码，必须串行、禁止与其它写源任务并发。",
        "",
        "## 6. 误差判定",
        "",
        "本审计没有连续量的参考值，判定是布尔量（测试红/绿），因此不存在「是否 <3%」的误差问题："
        "判据是退出码，逐项列于上表「实测」列。若以「killed 比例」作为覆盖度量，"
        f"本次为 {len(killed)}/{len(rows)}，其不确定性来源是变异体的选取，而不是数值误差。",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPO / "docs/mutation-survival3d"
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    entries = [e for e in CATALOGUE3D if not args.only or e["id"] in args.only]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = audit(entries, args.timeout)
    environment = dict(
        python=sys.version.split()[0],
        numpy=_version("numpy"),
        torch=_version("torch"),
        scipy=_version("scipy"),
        omp_threads=1,
    )
    record = dict(
        tool="scripts/audit_mutation_survival3d.py",
        environment=environment,
        scope="3-D native finite-volume core",
        mutants=rows,
        killed=sum(r["status"] == "killed" for r in rows),
        survived=sum(r["status"] == "survived" for r in rows),
        invalid=sum(r["status"].startswith("invalid") for r in rows),
    )
    (args.output / "mutation-survival-3d.json").write_text(
        json.dumps(record, indent=1) + "\n"
    )
    write_figure(args.output / "mutation-survival-3d.png", rows)
    write_report(args.output / "report.md", rows, environment)
    print(
        f"killed={record['killed']} survived={record['survived']} invalid={record['invalid']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
