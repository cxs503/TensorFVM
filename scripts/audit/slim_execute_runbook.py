#!/usr/bin/env python3
"""FVM-REPO-SLIM 执行 runbook（方案 docs/REPO-SLIM-PLAN.md 的执行阶段单入口）。

设计原则（对应方案 §4/§5 协调面与回滚保全）
------------------------------------------
1. **只读优先**：缺省 ``--mode preflight`` 对源仓库零写入，随时可跑；
   输出 JSON 审计 + 人类可读摘要，供 owner 审批前复核。
2. **改写只发生在一次性 scratch 克隆上**：execute 模式在 ``--scratch``
   目录里 ``git clone --mirror`` 出「回滚镜像」与「改写目标」两份克隆，
   源仓库（含全部 worktree）在 execute 模式里**只被读、不被写**。
3. **永不自动 force-push**：execute 模式止步于「改写完成 + 校验通过 +
   打印精确推送命令」，推送由 owner 在冻结窗口内手工执行（方案 §4）。
4. **场景规则单一事实源**：删除规则直接 import 自
   ``measure_rewrite_pack.py``（与实测 pack 体积同一套规则），
   避免方案文档与执行脚本规则漂移。
5. **全程审计**：每步命令、耗时、退出码、关键输出落 ``--out`` JSON。

用法
----
    # 审批前复核（只读，安全）
    python3 scripts/audit/slim_execute_runbook.py --mode preflight \
        --out docs/FVM-REPO-SLIM-RUNBOOK/preflight.json

    # owner 批准决策点后执行（仍在 scratch 内，不碰源仓库、不推送）
    python3 scripts/audit/slim_execute_runbook.py --mode execute \
        --confirm FVM-REPO-SLIM --scenario R2c \
        --decisions "D1=B;D2=prune;D3=freeze;D4=reclone;D5=90d;D6=done" \
        --out docs/FVM-REPO-SLIM-RUNBOOK/execute.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from measure_rewrite_pack import SCENARIOS, SIZE_RULES, TEXT_EXTS, _ext, _under_docs
except ImportError:  # 允许从任意 CWD 调用
    from measure_rewrite_pack import SCENARIOS, SIZE_RULES, TEXT_EXTS, _ext, _under_docs  # type: ignore

MiB = 1024 * 1024
CONFIRM_TOKEN = "FVM-REPO-SLIM"


def rule_hit_local(path: str, rule) -> bool:
    try:
        return bool(rule(path, 1))
    except TypeError:
        return bool(rule(path))


# ---------------------------------------------------------------- git 辅助
def git(repo: str, *args: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, timeout=timeout,
    )


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Runbook:
    def __init__(self, repo: str, scratch: str, out: str, scenario: str,
                 decisions: str, skip_fsck: bool):
        self.repo = os.path.abspath(repo)
        self.scratch = os.path.abspath(scratch)
        self.out = out
        self.scenario = scenario
        self.decisions = decisions
        self.skip_fsck = skip_fsck
        self.log: list[dict] = []
        self.hard_fail = False

    # -- 审计 ----------------------------------------------------------
    def step(self, name: str, cmd: list[str], cp: subprocess.CompletedProcess,
             ok: bool, note: str = "") -> None:
        entry = {
            "step": name,
            "cmd": " ".join(cmd),
            "rc": cp.returncode,
            "ok": ok,
            "elapsed_s": round(getattr(cp, "_elapsed", 0.0), 1),
            "note": note,
            "stderr_tail": cp.stderr.decode("utf-8", "replace")[-400:],
        }
        self.log.append(entry)
        mark = "OK " if ok else "FAIL"
        print(f"[{mark}] {name} (rc={cp.returncode}) {note}")

    def run(self, name: str, cmd: list[str], ok_if_zero: bool = True,
            note: str = "", timeout: int = 3600,
            fatal: bool = True) -> subprocess.CompletedProcess:
        t0 = time.time()
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False, timeout=timeout)
        cp._elapsed = time.time() - t0  # type: ignore[attr-defined]
        ok = (cp.returncode == 0) if ok_if_zero else True
        self.step(name, cmd, cp, ok, note=note)
        if not ok and fatal:
            self.hard_fail = True
        return cp

    # -- 只读普查 -------------------------------------------------------
    def census(self) -> dict:
        head = git(self.repo, "rev-parse", "HEAD").stdout.decode().strip()
        co = git(self.repo, "count-objects", "-vH").stdout.decode()
        packs_mib = 0.0
        for line in co.splitlines():
            if line.startswith("size-pack:"):
                v = line.split(":")[1].strip()
                num, unit = v.split()
                packs_mib = float(num) * (1024 if unit == "GiB" else 1)
        tmp_packs = [f for f in os.listdir(os.path.join(self.repo, ".git", "objects", "pack"))
                     if f.startswith("tmp_pack")]
        br = git(self.repo, "branch", "-r", "--format=%(refname:short)")
        branches = [l.strip() for l in br.stdout.decode().splitlines() if l.strip()]
        agent = [b for b in branches if b.startswith("agent/")]
        maint = [b for b in branches if b.startswith("maintenance/")]
        wt = git(self.repo, "worktree", "list", "--porcelain").stdout.decode().splitlines()
        n_worktrees = sum(1 for l in wt if l.startswith("worktree "))
        # 扩展名普查：必须覆盖**全部 refs**（filter-repo 默认重写所有 ref，
        # 实测场景规则同样基于 --all 的出现集）；只按 main 会漏掉仅存在于
        # 分支历史中的扩展名（如 provenance.sqlite）。
        from measure_rewrite_pack import occurrence_map
        exts: dict[str, int] = {}
        for _oid, paths in occurrence_map(self.repo).items():
            for path in paths:
                if path.startswith(("src/", "tests/")):
                    continue
                e = _ext(path)
                if e:
                    exts[e] = exts.get(e, 0) + 1
        return {
            "head": head,
            "packs_MiB": round(packs_mib, 2),
            "tmp_pack_files": len(tmp_packs),
            "remote_branches_total": len(branches),
            "remote_branches_agent": len(agent),
            "remote_branches_maintenance": len(maint),
            "maintenance_list": maint,
            "worktrees": n_worktrees,
            "doc_exts_present": sorted(exts),
        }

    def preflight(self) -> dict:
        c = self.census()
        checks: dict = {"census": c}

        # 1) fsck（可跳过）。已知坑：共享 .git 的 commit-graph 若过期，
        #    fsck --full 会以 rc=16 报 "commit-graph parent list ... too long"。
        #    该文件是本地元数据（非历史对象），`git commit-graph write` 可重建；
        #    镜像克隆不继承它，故此处降级为「可修复告警」而非硬失败（§5.5 门禁
        #    作用于镜像，execute 阶段对镜像单独 fsck 把关）。
        if self.skip_fsck:
            checks["fsck"] = "skipped"
        else:
            cp = self.run("fsck", ["git", "-C", self.repo, "fsck", "--full"],
                       note="源仓库完整性", fatal=False)
            if cp.returncode == 0:
                checks["fsck"] = "pass"
            else:
                cp2 = self.run("fsck-no-graph",
                    ["git", "-C", self.repo, "-c", "core.commitGraph=false",
                     "fsck", "--full"],
                    note="排除 commit-graph 后复测")
                if cp2.returncode == 0:
                    checks["fsck"] = ("stale_commit_graph: 仅 commit-graph 过期"
                                      "（git commit-graph write 可修复，决策点 D7）")
                else:
                    checks["fsck"] = "FAIL"
                    self.hard_fail = True

        # 2) 磁盘余量：镜像 + 改写克隆 + 归档 ≈ 3×pack
        mirror_fs = self.scratch if os.path.isdir(self.scratch) else os.path.dirname(self.scratch) or "/"
        usage = shutil.disk_usage(mirror_fs)
        need = 3 * c["packs_MiB"] * MiB
        checks["disk"] = {
            "free_MiB": round(usage.free / MiB, 1),
            "need_MiB": round(need / MiB, 1),
            "ok": usage.free >= need,
        }

        # 3) git-filter-repo 可用性
        cp = subprocess.run([sys.executable, "-c",
                             "import git_filter_repo; print(git_filter_repo.__version__ if hasattr(git_filter_repo,'__version__') else 'ok')"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        checks["filter_repo"] = ("ok:" + cp.stdout.decode().strip()) if cp.returncode == 0 \
            else "MISSING (pip install git-filter-repo)"
        if cp.returncode != 0:
            self.hard_fail = True

        # 4) 场景规则可解析
        noext = self.noext_paths_for(self.scenario, self.repo)
        checks["scenario"] = {
            "tag": self.scenario,
            "desc": SCENARIOS[self.scenario][0],
            "globs": self.globs_for(self.scenario, c["doc_exts_present"]),
            "noext_paths": noext,
        }

        # 5) 决策点门禁（execute 模式必须带齐 D1-D6）
        checks["decisions"] = self.decisions
        checks["preflight_ok"] = (not self.hard_fail) and checks["disk"]["ok"]
        return checks

    def noext_paths_for(self, tag: str, repo: str) -> list[str]:
        """R3 的无扩展名文件（glob 无法表达「无扩展名」，用精确 --path 补）。"""
        if tag != "R3":
            return []
        from measure_rewrite_pack import occurrence_map
        rule = SCENARIOS[tag][1]
        out = set()
        for _oid, paths in occurrence_map(repo).items():
            for p in paths:
                if _under_docs(p) and _ext(p) == "" and rule_hit_local(p, rule):
                    out.add(p)
        return sorted(out)

    def globs_for(self, tag: str, exts_present: list[str]) -> list[str]:
        """把场景规则翻译成 filter-repo --path-glob 集（含 docs/*.ext 零层形式）。"""
        rule = SCENARIOS[tag][1]
        if tag in SIZE_RULES:
            raise SystemExit(
                f"scenario {tag} 是体积阈值规则，无法用 path-glob 表达；"
                "执行档请用 R2c（推荐）或 R3（见方案 §8.2）。")
        # 用「仓库实际出现的扩展名」探测规则，保证 glob 集与实测场景一致
        def _hit(path: str) -> bool:
            try:
                return bool(rule(path, 1))
            except TypeError:
                return bool(rule(path))
        hit = {e for e in exts_present if _hit(f"docs/a{e}")}
        globs: list[str] = []
        for e in sorted(hit):
            globs += [f"docs/*{e}", f"docs/**/*{e}"]
        if tag in ("R2b", "R2c", "R3"):  # 全仓 pyc
            globs += ["**/*.pyc"]
        return globs

    # -- execute --------------------------------------------------------
    def execute(self) -> dict:
        if self.decisions.count(";") < 5:
            print("FATAL: --decisions 必须覆盖 D1..D6（分号分隔），owner 裁决留痕。")
            sys.exit(2)
        pre = self.preflight()
        if not pre["preflight_ok"]:
            print("FATAL: preflight 未通过，拒绝执行。")
            sys.exit(2)

        mirror = os.path.join(self.scratch, "mirror.git")
        rewrite = os.path.join(self.scratch, "rewrite.git")
        os.makedirs(self.scratch, exist_ok=True)

        # §5.1 回滚镜像
        self.run("mirror-clone", ["git", "clone", "--mirror", self.repo, mirror])
        self.run("mirror-fsck", ["git", "-C", mirror, "fsck", "--full"],
                 note="镜像完整性（§5.5 门禁）")
        archive = os.path.join(self.scratch, f"tfvm-pre-slim-{time.strftime('%Y%m%d')}.tar.gz")
        self.run("mirror-archive", ["tar", "czf", archive, "-C", self.scratch,
                                   os.path.basename(mirror)])
        checks = {"archive": archive, "archive_sha256": sha256_of(archive)}

        # 源仓库 src/tests 树指纹（不变量基准）。
        # 注意：R2b/R2c/R3 会删全仓 *.pyc（含 src/、tests/ 下的 22 个），
        # 故不变量口径 = 「src/tests 树去掉 *.pyc 后逐字节不变」。
        def _tree_hash(cp: subprocess.CompletedProcess) -> str:
            lines = [l for l in cp.stdout.decode().splitlines()
                     if not l.split("\t", 1)[-1].endswith(".pyc")]
            return hashlib.sha256("\n".join(lines).encode()).hexdigest()
        pre_tree = git(self.repo, "ls-tree", "-r", "main", "--", "src", "tests")
        pre_hash = _tree_hash(pre_tree)

        # §2 改写（在 scratch 克隆上）
        self.run("rewrite-clone", ["git", "clone", "--mirror", self.repo, rewrite])
        globs = pre["scenario"]["globs"]
        noext = pre["scenario"]["noext_paths"]
        cmd = ["git", "-C", rewrite, "filter-repo", "--invert-paths"]
        for g in globs:
            cmd += ["--path-glob", g]
        for pth in noext:
            cmd += ["--path", pth]
        self.run("filter-repo", cmd,
                 note=f"scenario={self.scenario} globs={len(globs)} noext={len(noext)}")
        self.run("gc", ["git", "-C", rewrite, "gc", "--aggressive", "--prune=now"])
        self.run("rewrite-fsck", ["git", "-C", rewrite, "fsck", "--full"])

        # 不变量：src/tests 树（去 pyc）必须逐字节不变
        post_tree = git(rewrite, "ls-tree", "-r", "main", "--", "src", "tests")
        post_hash = _tree_hash(post_tree)
        checks["src_tests_tree_sha256"] = {"pre": pre_hash, "post": post_hash,
                                           "invariant_holds": pre_hash == post_hash}
        if pre_hash != post_hash:
            print("FATAL: src/tests 树在改写后发生变化——中止，不得推送。")
            sys.exit(3)

        # 体积对比
        def pack_mib(r: str) -> float:
            co = git(r, "count-objects", "-vH").stdout.decode()
            for line in co.splitlines():
                if line.startswith("size-pack:"):
                    num, unit = line.split(":")[1].strip().split()
                    return float(num) * (1024 if unit == "GiB" else 1)
            return 0.0
        checks["pack_MiB"] = {"pre": round(pack_mib(self.repo), 1),
                             "post": round(pack_mib(rewrite), 1)}
        checks["commit_map"] = os.path.join(rewrite, "commit-map")
        checks["push_commands"] = [
            f"# owner 在冻结窗口内手工执行（runbook 不自动推送）：",
            f"git -C {rewrite} push --force origin main",
            f"# 下游 worktree 全部重新 clone（D4）；SHA 映射见 commit-map",
        ]
        return checks

    # -- 输出 ----------------------------------------------------------
    def finish(self, payload: dict) -> None:
        os.makedirs(os.path.dirname(self.out) or ".", exist_ok=True)
        doc = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "repo": self.repo,
            "scenario": self.scenario,
            "owner_decisions": self.decisions,
            "checks": payload,
            "steps": self.log,
            "verdict": "PASS" if not self.hard_fail else "FAIL",
        }
        with open(self.out, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"[i] audit -> {self.out}  verdict={doc['verdict']}")
        sys.exit(0 if doc["verdict"] == "PASS" else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="/root/tensor-ecosystem/TensorFVM")
    ap.add_argument("--mode", choices=["preflight", "execute"], default="preflight")
    ap.add_argument("--scenario", default="R2c", choices=sorted(SCENARIOS))
    ap.add_argument("--scratch", default="/tmp/tfvm-slim-scratch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm", default="",
                    help=f"execute 模式必须等于 {CONFIRM_TOKEN}")
    ap.add_argument("--decisions", default="",
                    help="owner 裁决留痕，如 'D1=B;D2=prune;D3=freeze;D4=reclone;D5=90d;D6=done'")
    ap.add_argument("--skip-fsck", action="store_true")
    args = ap.parse_args()

    if args.mode == "execute" and args.confirm != CONFIRM_TOKEN:
        print(f"FATAL: execute 模式需要 --confirm {CONFIRM_TOKEN}")
        sys.exit(2)

    rb = Runbook(args.repo, args.scratch, args.out, args.scenario,
                 args.decisions, args.skip_fsck)
    if args.mode == "preflight":
        rb.finish(rb.preflight())
    else:
        rb.finish(rb.execute())


if __name__ == "__main__":
    main()
