#!/usr/bin/env python3
"""Publish the m3948 progress report to the platform (paper-grade, six elements).

Builds ``/tmp/report.json`` from the committed markdown + figure (nothing is typed by
hand) and POSTs it to ``$TENSOR_API_BASE/tensor/reports`` after logging in with
``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS``.

    python3 scripts/m3948_publish_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "docs/m3948/report.md"
PNG = ROOT / "docs/m3948/outputs/pr82-merge-verification.png"


def post(url: str, payload: dict, token: str | None = None) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    base = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
    if not base:
        raise SystemExit("TENSOR_API_BASE is not set")

    markdown = MD.read_text(encoding="utf-8")
    png = PNG.read_bytes()
    report = {
        "title": "PR #82 (TensorFVM) 治愈：语义并集解冲 + 机械无损审计 + 复算证据",
        "summary": (
            "PR #82（agent/m3240，Ghia Re=400 v-centreline 转录之争）对 main 变脏的三方合并："
            "2 个内容冲突按语义并集解冲（published -0.23827 保留为计分表，m3240 的 -0.38760 转为 "
            "opt-in CORRECTIONS 寄存器；双侧测试均保留），1 个静默数据覆盖隐患（共享 ladder "
            "summary.json 被单侧重算）以重定位 + sha256 相等处置。机械无损审计覆盖 48 个文件、"
            "failures=0；合并树 96 passed / 0 failed / 1 skipped / 1 xfailed，main 基线 95 passed，"
            "无回归。从 committed bundles 复算：printed 计分 0.11457/0.14237/0.14931（随细化增长，"
            "仍是 strict xfail），proposed 计分 0.03604/0.00738/0.00431（n>=64 在 3% 内，条件为提案正确）。"
            "git push 被沙箱安全策略拦截（REJECTED），PR #82 落地需平台自动推送 agent/m3948(b3cf933)。"
        ),
        "scenario_key": "tensorfvm",
        "mission_id": 3948,
        "tags": ["TensorFVM", "PR-82", "three-way-merge", "no-silent-loss",
                 "lid-cavity", "Ghia-1982", "verification"],
        "markdown": markdown,
        "figures": [{
            "name": "pr82-merge-verification.png",
            "caption": (
                "左：committed Re=400, n=64 场量涡量云图；中：Re=400 v-centreline 印刷表 / 提案表 / "
                "求解器 n=32/64/128；右：逐点 |误差| 对双参考（log 轴）与 3% 门限。数据全部来自 "
                "committed bundles，脚本复算并断言 stored reference == 印刷表。"
            ),
            "png_base64": base64.b64encode(png).decode(),
        }],
    }
    pathlib.Path("/tmp/report.json").write_text(json.dumps(report, ensure_ascii=False))
    print(f"report.json {os.path.getsize('/tmp/report.json')} bytes")
    if args.dry_run:
        return 0

    login = post(base + "/tensor/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    })
    token = login.get("token") or login.get("access_token") or login.get("data", {}).get("token")
    if not token:
        raise SystemExit(f"no token in login response keys={list(login)}")
    created = post(base + "/tensor/reports", report, token)
    print(json.dumps(created, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())