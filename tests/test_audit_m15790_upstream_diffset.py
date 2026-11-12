"""m15790 巡更产物的可复算门：diffset.json 结构完整 + 分析脚本幂等。

幂等性用 A2_SKIP_REMOTE=1 / A2_SKIP_FETCH=1 的离线模式隔离网络抖动：
同一代码、同一 git 状态两次运行必须产出逐字节相同的 JSON。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "audit-m15790-upstream-diffset"
SCRIPT = DOC / "analyze_diffset.py"


@pytest.fixture(scope="module")
def facts():
    path = DOC / "diffset.json"
    if not path.exists():
        pytest.skip("diffset.json 未生成，跳过巡更产物校验")
    return json.loads(path.read_text())


def test_upstream_section_present(facts):
    up = facts["upstream"]
    for key in ("local_head", "origin_main_sha", "origin_main_source", "ls_remote_ok",
                "commits_ahead_of_origin_main", "commits_behind_origin_main",
                "diffset_pairs", "remote_branches_not_merged_into_main"):
        assert key in up, key
    assert len(up["local_head"]) == 40
    assert up["diffset_pairs"] == (up["commits_ahead_of_origin_main"]
                                   + up["commits_behind_origin_main"])


def test_diffset_records_consistent(facts):
    ds = facts["diffset"]
    assert len(ds["upstream_only_records"]) == len(ds["upstream_only_commits"])
    assert len(ds["local_only_commits"]) == facts["upstream"]["commits_ahead_of_origin_main"]
    assert len(ds["upstream_only_commits"]) == facts["upstream"]["commits_behind_origin_main"]
    for rec in ds["upstream_only_records"]:
        assert len(rec["sha"]) == 8
        assert rec["files"] == sum(rec["files_by_cat"].values())
        assert rec["insertions"] == sum(rec["ins_by_cat"].values())
        assert rec["themes"]
        assert all(r.split(":")[0] in {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}
                   for r in rec["risks"])


def test_window_and_themes(facts):
    commits = facts["commits"]
    assert len(commits) == 15, "窗口定义 = 最近 15 个 first-parent 提交"
    assert facts["theme_summary"], "主题清单不能为空"
    assert sum(t["n_commits"] for t in facts["theme_summary"]) >= len(commits)
    assert set(facts["risk_summary"]) <= {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}


def test_report_cites_current_digest(facts):
    digest = hashlib.sha256((DOC / "diffset.json").read_bytes()).hexdigest()
    report = (DOC / "REPORT.md").read_text()
    assert digest[:8] in report, "REPORT.md 的 dh256 引用与 diffset.json 不一致（产物过期）"
    assert "## ③ 计算结果" in report and "## ⑥ 误差判定" in report


def test_analyzer_deterministic(tmp_path):
    """离线模式连跑两次，JSON 必须逐字节一致（幂等 + 无隐藏时间戳）。"""
    env = dict(os.environ, A2_OUT=str(tmp_path),
               A2_SKIP_REMOTE="1", A2_SKIP_FETCH="1")
    digests = []
    for run in range(2):
        out = tmp_path / f"run{run}"
        env["A2_OUT"] = str(out)
        proc = subprocess.run([sys.executable, str(SCRIPT)], env=env,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-800:]
        digests.append(hashlib.sha256((out / "diffset.json").read_bytes()).hexdigest())
    assert digests[0] == digests[1], "同一 git 状态下两次分析结果不一致"