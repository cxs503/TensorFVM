"""m24430：slim_execute_runbook preflight 两处实测缺陷的回归测试。

缺陷 1（分支计数）：``git branch -r --format=%(refname:short)`` 的短名自带
远端名前缀（``origin/agent/m1``），原实现直接 ``startswith("agent/")`` /
``startswith("maintenance/")`` 匹配，导致 agent/maintenance 计数恒为 0——
D2 协调面（maintenance 旧平行分支处置）被低估为 0（实际 2 个）。

缺陷 2（repo 探测）：``--repo`` 历史缺省 ``/srv/tensor/workspace/TensorFVM``
在当前 harness 已失效；且从 worktree 调用时 ``.git`` 是文件而非目录，
``<repo>/.git/objects/pack`` 不存在，preflight 直接 FileNotFoundError。
修复为 ``git rev-parse --path-format=absolute --git-common-dir`` 自动探测。
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "audit"))

from slim_execute_runbook import classify_remote_branches, detect_repo_root  # noqa: E402


# ---------------------------------------------------------------- 分类
def test_classify_origin_prefixed_short_names():
    """真实形态：短名带 origin/ 前缀，maintenance 必须被数出来（回归缺陷 1）。"""
    branches = [
        "origin/main",
        "origin/HEAD",
        "origin/agent/m1",
        "origin/agent/m2",
        "origin/maintenance/artifact-hygiene",
        "origin/maintenance/remove-overset-embedded-svg",
    ]
    agent, maint = classify_remote_branches(branches)
    assert agent == ["origin/agent/m1", "origin/agent/m2"]
    assert maint == [
        "origin/maintenance/artifact-hygiene",
        "origin/maintenance/remove-overset-embedded-svg",
    ]


def test_classify_bare_names_defensive():
    """防御形态：无远端前缀的裸名也能分类（不因修复而破坏旧输入）。"""
    agent, maint = classify_remote_branches(
        ["agent/x", "maintenance/y", "main", "feature/z"])
    assert agent == ["agent/x"]
    assert maint == ["maintenance/y"]


def test_classify_empty():
    assert classify_remote_branches([]) == ([], [])


def test_classify_keeps_original_names_for_traceability():
    """列表保留原始短名（带 origin/），便于审计追溯。"""
    _, maint = classify_remote_branches(["upstream/maintenance/keep"])
    assert maint == ["upstream/maintenance/keep"]


# ---------------------------------------------------------------- 探测
def _git(cwd, *args):
    return subprocess.run(["git", "-C", cwd, *args],
                          check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


@pytest.fixture(name="repo_with_worktree")
def _repo_with_worktree(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    _git(str(main), "init", "-q", "--initial-branch=main")
    _git(str(main), "config", "user.email", "t@example.com")
    _git(str(main), "config", "user.name", "t")
    (main / "f.txt").write_text("hi")
    _git(str(main), "add", "-A")
    _git(str(main), "commit", "-qm", "init")
    wt = tmp_path / "wt"
    _git(str(main), "worktree", "add", "-q", str(wt), "-b", "agent/wt")
    return main, wt


def test_detect_repo_root_from_worktree(repo_with_worktree):
    """从 worktree（.git 是文件）探测 → 返回共享仓库根（回归缺陷 2）。"""
    main, wt = repo_with_worktree
    assert detect_repo_root(cwd=str(wt)) == str(main)


def test_detect_repo_root_from_main_checkout(repo_with_worktree):
    main, _wt = repo_with_worktree
    assert detect_repo_root(cwd=str(main)) == str(main)


def test_detect_repo_root_fallback_outside_repo(tmp_path):
    """非 git 目录 → 回退历史缺省值（不抛异常）。"""
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    assert detect_repo_root(cwd=str(outside)) == "/srv/tensor/workspace/TensorFVM"


def test_detect_repo_root_matches_real_checkout():
    """在本仓 worktree 内探测：结果必须是含 .git/objects/pack 的真实根。"""
    root = detect_repo_root()
    assert os.path.isdir(os.path.join(root, ".git", "objects", "pack")), root


@pytest.mark.parametrize("obj", [classify_remote_branches, detect_repo_root])
def test_helpers_callable(obj):
    assert callable(obj)
