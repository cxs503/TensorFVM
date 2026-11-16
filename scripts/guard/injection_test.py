#!/usr/bin/env python3
"""FVM-SLIM-GUARD injection test (D6, R4): verify real interception.

Injects a large blob into a throwaway repo that has the *actual* guard
wired up (real .githooks/pre-commit + scripts/guard/repo_guard.py +
.gitattributes, core.hooksPath set), then runs `git commit` and checks
the hook really fires.  Records results into docs/m24008/injection-test.md.

This is an end-to-end probe of the *wiring*, not just the pure logic:
it proves the pre-commit hook (as shipped) blocks a >512 KiB blob, lets a
small blob through, and lets a whitelisted (`guard-allow-large`) large
artifact through.  Stdlib + git only; no torch.

Usage:  python3 scripts/guard/injection_test.py [--out docs/m24008]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "guard" / "repo_guard.py"
HOOK = REPO_ROOT / ".githooks" / "pre-commit"
GITATTR = REPO_ROOT / ".gitattributes"

BLOB_LIMIT = 512 * 1024
LARGE = 600 * 1024          # > 512 KiB -> must be blocked
SMALL = 100                 # << 512 KiB -> must pass


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=check,
    )


def _setup_guard(repo: Path) -> None:
    """Wire the real guard into the throwaway repo (as a fresh clone would).

    The hook/CI wiring files are tracked in git but may not be materialized
    in this sparse worktree, so we read their content from HEAD (git show)
    rather than the working tree.
    """
    def _tracked(path: str) -> str:
        return subprocess.run(
            ["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout

    # Copy the hook + guard + gitattributes, set core.hooksPath.
    hooks = repo / ".githooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text(_tracked(".githooks/pre-commit"))
    os.chmod(hooks / "pre-commit", 0o755)
    guard_dir = repo / "scripts" / "guard"
    guard_dir.mkdir(parents=True)
    shutil.copy2(GUARD, guard_dir / "repo_guard.py")
    shutil.copy2(GITATTR, repo / ".gitattributes")
    _git(repo, "config", "core.hooksPath", ".githooks")


def _run_case(repo: Path, name: str, blob_path: str, blob_bytes: int,
              expect_block: bool) -> dict:
    """Inject a blob, attempt a commit, and report whether the hook fired."""
    p = repo / blob_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * blob_bytes)
    _git(repo, "add", blob_path)
    proc = _git(repo, "commit", "-qm", f"inject {name}", check=False)
    blocked = proc.returncode != 0
    ok = (blocked == expect_block)
    return {
        "case": name,
        "blob_path": blob_path,
        "blob_bytes": blob_bytes,
        "blob_human": f"{blob_bytes/1024:.1f} KiB",
        "commit_rc": proc.returncode,
        "hook_fired": blocked,
        "expect_block": expect_block,
        "pass": ok,
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-3:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "m24008"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "inject"
        repo.mkdir(parents=True, exist_ok=True)
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "config", "user.email", "guard@test")
        _git(repo, "config", "user.name", "guard-test")
        (repo / "seed.txt").write_text("seed\n")
        _git(repo, "add", "seed.txt")
        _git(repo, "commit", "-qm", "seed")
        _setup_guard(repo)

        # Each case is independent: reset the index after every commit
        # attempt so a blocked blob from a previous case does not carry over
        # and contaminate the next one.
        def _reset() -> None:
            _git(repo, "reset", "-q", "--hard", "HEAD", check=False)

        # Case 1: large non-whitelisted blob -> hook must block.
        results.append(_run_case(
            repo, "large-unlisted", "big.bin", LARGE, expect_block=True))
        _reset()
        # Case 2: small blob -> must pass.
        results.append(_run_case(
            repo, "small", "small.txt", SMALL, expect_block=False))
        _reset()
        # Case 3: large whitelisted npz (guard-allow-large) -> must pass.
        results.append(_run_case(
            repo, "large-whitelisted-npz",
            "docs/verification-x/f/fields.npz", LARGE, expect_block=False))
        _reset()
        # Case 4: large unlisted under docs (not whitelisted) -> must block.
        results.append(_run_case(
            repo, "large-unlisted-docs",
            "docs/verification-x/f/fields.bin", LARGE, expect_block=True))
        _reset()

    # Live volume gate probe on the real repo (read-only).
    live_proc = subprocess.run(
        [sys.executable, str(GUARD), "live"], cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    live_json = json.loads(live_proc.stdout) if live_proc.returncode == 0 else {}

    all_pass = all(r["pass"] for r in results)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blob_limit_bytes": BLOB_LIMIT,
        "cases": results,
        "all_pass": all_pass,
        "live_gate": {
            "rc": live_proc.returncode,
            "live_bytes": live_json.get("live_bytes"),
            "live_human": live_json.get("live_human"),
            "live_status": live_json.get("live_status"),
        },
    }

    # Write the machine-readable record.
    (out_dir / "injection-test.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # Write the human-readable doc.
    rows = "\n".join(
        f"| {r['case']} | {r['blob_path']} | {r['blob_human']} | "
        f"{'blocked' if r['hook_fired'] else 'passed'} | "
        f"{'block' if r['expect_block'] else 'pass'} | "
        f"{'PASS' if r['pass'] else 'FAIL'} |"
        for r in results
    )
    doc = f"""# FVM-SLIM-GUARD 注入试验（D6, R4）：实测拦截有效性

> 生成：{summary['generated_at']} ｜ 脚本 `scripts/guard/injection_test.py`
> 方法：在**临时一次性 git 仓**上接线**真实**守卫（`.githooks/pre-commit` +
> `scripts/guard/repo_guard.py` + `.gitattributes`，`core.hooksPath=.githooks`），
> 注入大/小/白名单 blob 后跑 `git commit`，看钩子是否真的拦截。
> 机器记录：`docs/m24008/injection-test.json`。

## 1. 判据（冻结）

| 判据 | 值 |
|---|---|
| 512 KiB 拦截阈值 | {BLOB_LIMIT} B（{BLOB_LIMIT/1024:.0f} KiB） |
| 注入大 blob | {LARGE/1024:.0f} KiB（> 阈值，应拦截） |
| 注入小 blob | {SMALL} B（< 阈值，应放行） |
| 白名单属性 | `guard-allow-large`（`.gitattributes`） |

## 2. 结果（程序化注入，零手抄）

| 用例 | 路径 | 大小 | 钩子行为 | 期望 | 判定 |
|---|---|---|---|---|---|
{rows}

## 3. live 体积门（真实仓，只读）

| 量 | 值 |
|---|---|
| live 体积（unique HEAD 引用 blob） | {live_json.get('live_human', 'n/a')}（{live_json.get('live_bytes', 'n/a')} B） |
| live 门状态 | {live_json.get('live_status', 'n/a')} |

## 4. 结论

- **全部 {len(results)} 用例 {'通过' if all_pass else '未通过'}**：真实钩子接线
  在注入大 blob 时确实拦截（rc≠0），小 blob 放行，白名单大工件放行，
  未列入白名单的大 blob 仍拦截。
- 拦截有效性 = **{sum(1 for r in results if r['hook_fired'])}/{len(results)}**
  用例按预期触发钩子。
- 本试验**不触碰真实仓历史**（全部在临时仓完成）；真实仓只做只读 `live` 探测。
"""
    (out_dir / "injection-test.md").write_text(doc)

    print(json.dumps(summary, indent=2))
    print(f"\nrecorded -> {out_dir / 'injection-test.md'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
