#!/usr/bin/env python3
"""Publish the m15554 audit report to the platform (paper-grade six sections).

The markdown body is ``docs/audit-m15554/REPORT.md`` verbatim (already generated from
``cavity_audit.json``/``platform_state.json`` by ``build_audit_report.py``), so the
published text and the in-repo artefact cannot drift; the three figures are attached
as base64 PNGs.  Credentials come from /root/tp-agent/agent.env.

Usage: python3 docs/audit-m15554/publish_report.py
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(l.split("=", 1) for l in open("/root/tp-agent/agent.env")
           if "=" in l and not l.startswith("#"))


def post(path, payload, token):
    req = urllib.request.Request(BASE + path, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, data=json.dumps(payload).encode(), timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> None:
    req = urllib.request.Request(BASE + "/auth/login", method="POST")
    req.add_header("Content-Type", "application/json")
    token = json.loads(urllib.request.urlopen(req, data=json.dumps({
        "username": ENV["TENSOR_AGENT_USER"].strip(),
        "password": ENV["TENSOR_AGENT_PASS"].strip()}).encode()).read())["access_token"]

    md = open(os.path.join(HERE, "REPORT.md"), encoding="utf-8").read()
    figs = []
    for name, cap in (("fig_audit_centrelines.png",
                       "图1 从存档场独立重读的两条中心线 (n=32/64/128) 对标 Ghia 1982 表 I/II"),
                      ("fig_audit_fields.png",
                       "图2 n=128 存档场涡量与 u/U，主涡中心由 ψ 路径积分独立定位"),
                      ("fig_audit_convergence.png",
                       "图3 误差 vs 分辨率：内点 8.00%→1.89%→0.53%，与壁面端点外推口径对照")):
        with open(os.path.join(HERE, name), "rb") as fh:
            figs.append({"name": name, "caption": cap,
                         "png_base64": base64.b64encode(fh.read()).decode()})
    payload = {
        "title": "m15458 交差审计 (board-audit)：CASE-CAVITY-RE1000-FVM · TensorFVM 方腔 Re=1000 独立重算与 L4 裁定",
        "summary": "不采信交付数字：从存档速度场独立重读中心线/主涡中心/散度/边界，"
                   "并重走一投影步复算 1e-7 定常门槛；n=128 全项 <1%（u_min 0.58%、v_max 0.74%、"
                   "内点 max|Δu| 0.53%）。裁定 L4（verdict id=338）；发现并修复：交付报告落后于 "
                   "bundle 导致自带测试失败（重生成后 5 passed）、基准锚点未登记（补 id=18）；"
                   "另指出 headline 的 2.93% 实为壁面外推残差、平台 L4 账本口径 CLAIMED_NOT_PROVEN 的成因。",
        "scenario_key": "CASE-CAVITY-RE1000-FVM",
        "mission_id": 15554,
        "tags": ["FVM", "lid-driven-cavity", "Ghia1982", "board-audit", "L4", "TensorFVM", "Re1000"],
        "markdown": md,
        "figures": figs,
    }
    st, d = post("/tensor/reports", payload, token)
    print("POST /tensor/reports ->", st, {k: v for k, v in d.items() if k != "markdown"})


if __name__ == "__main__":
    main()