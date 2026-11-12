#!/usr/bin/env python3
"""m15790 · TensorFVM 生态巡更：上游差集 + 最近 15 提交主题演化盘点（零手抄，全程序化）。

输出（默认写入本脚本所在目录，可用 A2_OUT 覆盖；测试用临时目录）:
  diffset.json    —— 结构化事实（报告数字全部来自此文件）
用法: python3 docs/audit-m15790-upstream-diffset/analyze_diffset.py
环境变量: A2_OUT=<dir> 输出目录 | A2_SKIP_REMOTE=1 跳过 ls-remote | A2_SKIP_FETCH=1 跳过 fetch
退出码 0 = 分析完成（不代表"无风险"）。
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("A2_OUT") or Path(__file__).resolve().parent)
N_COMMITS = 15

BLOB_LINES = 10000          # 单文件新增行数 > 1e4 → data/blob 膨胀
BIG_COMMIT_INS = 20000      # 单提交新增 > 2e4 行
DUP_JACCARD = 0.45          # 函数名集合 Jaccard 相似度阈值
MIN_FUNCS_DUP = 5           # 参与相似度比较的最小函数数（过滤小脚本噪声）
GIT_TIMEOUT = 45            # 单条 git 命令上限，避免网络抖动挂死

THEME_KEYWORDS = [
    ("audit-verdict", ("audit", "verdict", "board_audit", "board-audit")),
    ("reproducibility", ("repro", "reproduce")),
    ("case-cavity", ("cavity",)),
    ("case-cylinder", ("cylinder",)),
    ("case-airfoil", ("airfoil",)),
    ("model-mixing-layer", ("mixing_layer", "mixing-layer")),
    ("vof-two-phase", ("vof", "two-phase", "twophase")),
    ("benchmark", ("benchmark",)),
    ("report-pipeline", ("report", "publish")),
    ("geometry-verify", ("geometry", "cb-", "body-fitted")),
]

FAILED: list[str] = []


def sh(*args: str) -> str:
    """执行 git 命令；超时/异常返回 "" 并登记，保证脚本不因网络抖动挂死。"""
    try:
        return subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                              check=False, timeout=GIT_TIMEOUT).stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as exc:  # pragma: no cover - 环境相关
        FAILED.append(f"{' '.join(args)} :: {type(exc).__name__}")
        return ""


def category(path: str) -> str:
    for pre, name in (("src/", "src"), ("tests/", "tests"), ("docs/", "docs"),
                      ("scripts/", "scripts")):
        if path.startswith(pre):
            return name
    return "other"


def numstat(base: str, tip: str) -> list[tuple[str, int, int]]:
    rows = []
    for line in sh("git", "diff", "--numstat", base, tip).splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, d, p = parts
        rows.append((p, int(a) if a.isdigit() else 0, int(d) if d.isdigit() else 0))
    return rows


def themes_of(paths: list[str]) -> list[str]:
    blob = " ".join(paths).lower()
    hits = [name for name, kws in THEME_KEYWORDS if any(k in blob for k in kws)]
    return hits or ["misc"]


def mission_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"m(\d{4,6})", text)))


def commit_record(sha: str) -> dict:
    """单个提交（相对其第一个父）的结构化记录 + 风险标注。"""
    meta = sh("git", "log", "-1", "--pretty=%H|%ad|%an|%s", "--date=iso", sha)
    full, date, author, subject = meta.split("|", 3)
    parents = sh("git", "log", "-1", "--pretty=%P", sha).split()
    parent = parents[0] if parents else ""
    rows = numstat(parent, sha) if parent else []
    cat_ins: Counter = Counter()
    cat_files: Counter = Counter()
    for p, a, d in rows:
        cat_ins[category(p)] += a
        cat_files[category(p)] += 1
    paths = [r[0] for r in rows]
    ins = sum(r[1] for r in rows)
    m_subject = mission_ids(subject)
    doc_m = sorted({m for p in paths if p.startswith("docs/") for m in mission_ids(p)})
    referenced_other = [m for m in doc_m if m not in m_subject]
    has_src = cat_files.get("src", 0) > 0
    snap = [p for p in paths if p.startswith("docs/") and "/src/tensorfvm/" in p]
    risks = []
    if has_src and cat_files.get("tests", 0) == 0:
        risks.append("R1:src-change-without-test-delta")
    if ins > BIG_COMMIT_INS or any(r[1] > BLOB_LINES for r in rows):
        risks.append("R2:data-blob-inflation")
    if not rows:
        risks.append("R3:no-tree-change")
    if rows and not has_src:
        risks.append("R4:no-src-change(证据/文档型)")
    if referenced_other:
        risks.append("R5:meta-audit-chain(审计对象为既往任务号)")
    if snap:
        risks.append(f"R6:docs-embedded-source-snapshot({len(snap)} files)")
    if subject.startswith("Merge") and len(parents) < 2:
        risks.append("R7:merge-subject-without-merge-topology")
    return {
        "sha": sha[:8], "date": date, "author": author, "subject": subject,
        "n_parents": len(parents),
        "files": len(rows), "insertions": ins, "deletions": sum(r[2] for r in rows),
        "files_by_cat": dict(cat_files), "ins_by_cat": dict(cat_ins),
        "top_files": [dict(path=p, ins=a, del_=d)
                      for p, a, d in sorted(rows, key=lambda r: -r[1])[:3]],
        "themes": themes_of(paths),
        "mission_in_subject": m_subject,
        "missions_referenced_in_paths": doc_m,
        "docs_src_snapshot_files": len(snap),
        "risks": risks,
    }


def py_funcs(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main() -> int:
    facts: dict = {}
    skip_remote = os.environ.get("A2_SKIP_REMOTE") == "1"
    skip_fetch = os.environ.get("A2_SKIP_FETCH") == "1"

    # ---------- A. 上游同步状态 ----------
    head = sh("git", "rev-parse", "HEAD")
    remotes = [l for l in sh("git", "remote", "-v").splitlines() if l]
    ls_remote = "" if skip_remote else sh("git", "ls-remote", "origin",
                                          "refs/heads/main", "refs/heads/master")
    ls_remote_ok = bool(ls_remote)
    if ls_remote_ok:
        upstream_main, origin_main_source = ls_remote.split()[0], "ls-remote(远端实测)"
    else:
        upstream_main = sh("git", "rev-parse", "origin/main") or None
        origin_main_source = ("remote-tracking-ref(本地快照,A2_SKIP_REMOTE=1 跳过远端核验)"
                              if skip_remote else
                              "remote-tracking-ref(本地快照,ls-remote 不可用)")
    # 远端已前进但本地 tracking ref 陈旧 → 尽力 fetch（只取不推），并记录是否成功
    local_ref = sh("git", "rev-parse", "origin/main")
    fetched = False
    if (upstream_main and local_ref and upstream_main != local_ref and not skip_fetch):
        fetched = bool(sh("git", "fetch", "--no-tags", "origin", "main"))
        local_ref = sh("git", "rev-parse", "origin/main")
    counts = sh("git", "rev-list", "--left-right", "--count", "origin/main...HEAD")
    behind, ahead = (int(x) for x in counts.split()) if counts else (None, None)
    merged_refs = set(sh("git", "branch", "-r", "--merged", "origin/main").split())
    all_refs = set(sh("git", "branch", "-r").split())
    unmerged = sorted(r for r in (all_refs - merged_refs)
                      if "->" not in r and not r.endswith("/HEAD"))
    facts["upstream"] = {
        "remotes": remotes,
        "local_head": head,
        "origin_main_sha": upstream_main,
        "origin_main_source": origin_main_source,
        "ls_remote_ok": ls_remote_ok,
        "fetched_to_sync_tracking_ref": fetched,
        "head_equals_origin_main": upstream_main == head,
        "commits_ahead_of_origin_main": ahead,
        "commits_behind_origin_main": behind,
        "diffset_pairs": (ahead + behind) if (ahead is not None and behind is not None) else None,
        "remote_branch_total": len([r for r in all_refs if "->" not in r]),
        "remote_branches_not_merged_into_main": len(unmerged),
        "unmerged_examples": [
            {"ref": r, "commits_ahead_of_main": int(sh("git", "rev-list", "--count", r,
                                                       "^origin/main") or 0)}
            for r in unmerged[:8]
        ],
        "git_cmd_failures": FAILED,
    }

    # ---------- B. 差集内容（上游有、本地无 / 本地有、上游无）----------
    up_only = [s for s in sh("git", "rev-list", "HEAD..origin/main").splitlines() if s]
    local_only = [s for s in sh("git", "rev-list", "origin/main..HEAD").splitlines() if s]
    fp_chain = [s for s in sh("git", "rev-list", "--first-parent",
                              "HEAD..origin/main").splitlines() if s]
    facts["diffset"] = {
        "upstream_only_commits": up_only,
        "upstream_only_first_parent_chain": fp_chain,
        "local_only_commits": local_only,
        "upstream_only_records": [commit_record(s) for s in up_only],
    }

    # ---------- C. 最近 N 个主线提交（主题演化基线）----------
    log = sh("git", "log", "--first-parent", "--pretty=%H", f"-{N_COMMITS}", "HEAD")
    commits = [commit_record(s) for s in log.splitlines() if s]
    facts["commits"] = commits

    theme_stats: dict = defaultdict(lambda: {"commits": [], "files": 0, "insertions": 0})
    for c in commits:
        for t in c["themes"]:
            theme_stats[t]["commits"].append(c["sha"])
            theme_stats[t]["files"] += c["files"]
            theme_stats[t]["insertions"] += c["insertions"]
    facts["theme_summary"] = [
        {"theme": t, "n_commits": len(v["commits"]), "commits": v["commits"],
         "files": v["files"], "insertions": v["insertions"]}
        for t, v in sorted(theme_stats.items(), key=lambda kv: -len(kv[1]["commits"]))
    ]
    facts["risk_summary"] = dict(Counter(r.split(":")[0] for c in commits for r in c["risks"]))
    facts["diffset_risk_summary"] = dict(
        Counter(r.split(":")[0] for c in facts["diffset"]["upstream_only_records"]
                for r in c["risks"]))

    # ---------- D. 并行/重复实现扫描 ----------
    added_py: dict[str, str] = {}
    base = commits[-1]["sha"] + "^1" if commits else None
    if base:
        for line in sh("git", "diff", "--diff-filter=A", "--name-only", base, "HEAD").splitlines():
            if line.endswith(".py"):
                blob = sh("git", "show", f"HEAD:{line}")
                if blob:
                    added_py[line] = blob
    funcs = {p: py_funcs(s) for p, s in added_py.items()}
    funcs = {p: fset for p, fset in funcs.items() if len(fset) >= MIN_FUNCS_DUP}
    dup_pairs = []
    for (p1, f1), (p2, f2) in combinations(funcs.items(), 2):
        j = len(f1 & f2) / len(f1 | f2)
        if j >= DUP_JACCARD:
            dup_pairs.append({"a": p1, "b": p2, "jaccard": round(j, 3),
                              "shared": len(f1 & f2), "sizes": [len(f1), len(f2)]})
    facts["duplicate_scan"] = {
        "added_py_files": len(added_py),
        "added_py_with_ge5_funcs": len(funcs),
        "threshold_jaccard": DUP_JACCARD,
        "min_funcs": MIN_FUNCS_DUP,
        "pairs": sorted(dup_pairs, key=lambda d: -d["jaccard"])[:12],
    }

    # ---------- E. docs 内嵌源码副本 ----------
    vendored = []
    for fpath in (REPO / "docs").rglob("src/tensorfvm/**/*.py"):
        rel = fpath.relative_to(REPO).as_posix()
        marker = "src/tensorfvm/"
        twin_rel = marker + rel.split(marker, 1)[1]
        twin = REPO / twin_rel
        vendored.append({
            "path": rel, "twin": twin_rel, "twin_exists": twin.exists(),
            "identical": (twin.exists() and twin.read_text() == fpath.read_text()),
        })
    twin_dir = REPO / "src/tensorfvm/compressible"
    facts["vendored_copies"] = {
        "n_py_under_docs": len(vendored),
        "doc_roots": sorted({v["path"].split("/")[1] for v in vendored}),
        "n_twin_missing": sum(1 for v in vendored if not v["twin_exists"]),
        "n_twin_diverged": sum(1 for v in vendored if v["twin_exists"] and not v["identical"]),
        "files": vendored[:12],
        "src_files_in_twin_package": len(list(twin_dir.rglob("*.py"))) if twin_dir.exists() else 0,
    }

    # ---------- F. 仓库形态 ----------
    docs_bytes = sum(fp.stat().st_size for fp in (REPO / "docs").rglob("*")
                     if fp.is_file() and OUT not in fp.parents)
    src_bytes = sum(fp.stat().st_size for fp in (REPO / "src").rglob("*") if fp.is_file())
    facts["repo_shape"] = {
        "docs_bytes": docs_bytes,
        "src_bytes": src_bytes,
        "docs_over_src_ratio": round(docs_bytes / src_bytes, 2) if src_bytes else None,
        # 排除本分析自身输出目录，避免"输出改变输入"的自指偏差
        "json_files_total": len([fp for fp in REPO.rglob("*.json") if OUT not in fp.parents]),
        "docs_entries": len(list((REPO / "docs").iterdir())),
        "src_py": len(list((REPO / "src").rglob("*.py"))),
        "tests_py": len(list((REPO / "tests").rglob("*.py"))),
        "scripts_py": len(list((REPO / "scripts").rglob("*.py"))),
        "authors_last_window": dict(Counter(c["author"] for c in commits)),
        "commit_window": [commits[-1]["date"], commits[0]["date"]] if commits else [],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diffset.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in facts[k].items()
                          if kk in ("origin_main_sha", "head_equals_origin_main",
                                    "commits_ahead_of_origin_main", "commits_behind_origin_main",
                                    "diffset_pairs", "remote_branches_not_merged_into_main",
                                    "origin_main_source")}
                      for k in ("upstream",)}, ensure_ascii=False, indent=2))
    print("diffset upstream-only:", len(up_only), "local-only:", len(local_only),
          "first-parent chain:", len(fp_chain))
    print("themes:", [(t["theme"], t["n_commits"]) for t in facts["theme_summary"]])
    print("risk:", facts["risk_summary"], "diffset risk:", facts["diffset_risk_summary"])
    print("dup pairs:", len(facts["duplicate_scan"]["pairs"]),
          "added_py:", facts["duplicate_scan"]["added_py_files"])
    return 0


if __name__ == "__main__":
    sys.exit(main())