"""slim_execute_runbook 的 glob 翻译正确性测试。

核心主张：runbook 把场景规则（measure_rewrite_pack.SCENARIOS，blob 级
「全部历史路径都命中才删」）翻译成 filter-repo 的 --path-glob 集后，
两者对同一对象库判定的**被删 blob 集必须完全一致**——否则 runbook 的
实测收益（docs/FVM-REPO-SLIM-M23832/measure/）就不成立。

filter-repo 的 glob 语义（上游文档）：``**`` 跨目录（正则 ``.*``），
``*``/``?`` 不跨 ``/``。本测试按该语义实现匹配器，与 runbook 生成的
glob 集做 blob 级对账。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "audit"))
from measure_rewrite_pack import SCENARIOS, SIZE_RULES  # noqa: E402
from slim_execute_runbook import Runbook  # noqa: E402

REPO = "/srv/tensor/workspace/TensorFVM"


def glob_to_regex(g: str) -> re.Pattern:
    out, i = [], 0
    while i < len(g):
        c = g[i]
        if c == "*":
            if g[i:i + 2] == "**":
                out.append(".*")
                i += 2
            else:
                out.append(r"[^/]*")
                i += 1
        elif c == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(path: str, globs: list[str]) -> bool:
    return any(rx.match(path) for rx in (glob_to_regex(g) for g in globs))


def rule_hit(rule, path: str) -> bool:
    try:
        return bool(rule(path, 1))
    except TypeError:
        return bool(rule(path))


def blob_deletion_set_by_rule(repo: str, tag: str) -> set[str]:
    """场景规则口径：blob 的全部历史路径都命中 → 删。"""
    from measure_rewrite_pack import occurrence_map
    rule = SCENARIOS[tag][1]
    out = set()
    for oid, paths in occurrence_map(repo).items():
        if all(rule_hit(rule, p) for p in paths):
            out.add(oid)
    return out


def blob_deletion_set_by_globs(repo: str, tag: str) -> set[str]:
    """glob 口径：blob 的全部历史路径都被某个 glob 命中 → 删。"""
    from measure_rewrite_pack import occurrence_map
    rb = Runbook.__new__(Runbook)
    # 用全 refs 的扩展名集合生成 glob（与 runbook preflight 同路径）
    from measure_rewrite_pack import _ext
    exts = set()
    for _oid, paths in occurrence_map(repo).items():
        for path in paths:
            if path.startswith(("src/", "tests/")):
                continue
            e = _ext(path)
            if e:
                exts.add(e)
    globs = rb.globs_for(tag, sorted(exts))
    noext = set(rb.noext_paths_for(tag, repo))
    out = set()
    for oid, paths in occurrence_map(repo).items():
        if all(glob_match(p, globs) or p in noext for p in paths):
            out.add(oid)
    return out


@pytest.mark.parametrize("tag", ["R2c", "R3"])
def test_glob_translation_matches_scenario_rule(tag):
    """真实仓库对账：glob 集与场景规则判定的被删 blob 集一致。"""
    if not os.path.isdir(os.path.join(REPO, ".git")):
        pytest.skip("TensorFVM 主仓不在本机")
    by_rule = blob_deletion_set_by_rule(REPO, tag)
    by_glob = blob_deletion_set_by_globs(REPO, tag)
    only_rule = by_rule - by_glob
    only_glob = by_glob - by_rule
    assert not only_rule, f"规则删而 glob 漏删 {len(only_rule)} 个 blob（收益高估风险）"
    assert not only_glob, f"glob 多删 {len(only_glob)} 个 blob（误删风险）"
    assert len(by_rule) > 0


def test_size_scenarios_rejected():
    """体积阈值场景无法用 path-glob 表达，runbook 必须显式拒绝。"""
    rb = Runbook.__new__(Runbook)
    for tag in sorted(SIZE_RULES):
        with pytest.raises(SystemExit):
            rb.globs_for(tag, [".npz", ".md"])


def test_globs_never_touch_src_tests_non_pyc():
    """glob 集不得触及 src/tests 的非 pyc 路径（源码不变量的前提）。"""
    rb = Runbook.__new__(Runbook)
    for tag in ("R2c", "R3"):
        globs = rb.globs_for(tag, [".npz", ".md", ".py"])
        for g in globs:
            assert g.startswith("docs/") or g == "**/*.pyc", (tag, g)
