#!/usr/bin/env python3
"""Publish the m15190 board-audit report (six-element, figure embedded)."""
import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"]
OUT = Path("docs/board-audit-m15026")

req = urllib.request.Request(
    BASE + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"})
TOKEN = json.load(urllib.request.urlopen(req))["access_token"]

md = Path(OUT / "REPORT.md").read_text()
png = base64.b64encode((OUT / "fig_audit_m15026.png").read_bytes()).decode()

payload = {
    "title": "交差审计 m15026/GAP-FVM-TWOPHASE：判 L3——Zalesak 侧零水分复现（3.95%），官方基准 Hysing 末速半侧缺失、溃坝 84.8% FAIL",
    "summary": (
        "独立复算 m15026 交付：8 tests 通过；--summary-only 12 s 复算 gates 8/10；单 leg 重跑逐位复现"
        "（Zalesak PLIC N=64 = 0.09302325581395349）。Zalesak 4 leg 报告值=审计复算=自建解析参考（差 0.00%），"
        "溃坝报告值同样逐位可复算。但唯一裁判 tensor_benchmark_cases(id=6) 要求 Zalesak 形状 + Hysing 末速<10%："
        "后者全仓无数据（campaign 全 2d_reference=None，算例 Eo=1.23 非 Hysing）→ L4 不过；附加溃坝 84.8%≫15%。"
        "判 L3（真代码+入口+工件+集成调用方 solver.py:130），残余差距已列返修建议。"),
    "scenario_key": "tensorfvm/gap-fvm-twophase/board-audit",
    "mission_id": 15190,
    "tags": ["TensorFVM", "VOF", "two-phase", "audit", "board-audit",
             "GAP-FVM-TWOPHASE", "m15026", "maturity-verdict", "Zalesak", "Hysing"],
    "markdown": md,
    "figures": [{
        "name": "fig_audit_m15026.png",
        "caption": "审计复算三视图：(a) Zalesak 几何误差 vs 分辨率（PLIC/CICSAM，5% 自设判据线，N=128 声明分辨率）；"
                   "(b) 溃坝前锋 x*(t*) 数值 vs Martin & Moyce（系统性滞后，honest FAIL）；"
                   "(c) 滞后量时间历程：近似常数平移（−0.78/−0.67），误差主体为零点/定义平移而非扩散累积。",
        "png_base64": png,
    }],
}
req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + TOKEN})
try:
    r = json.load(urllib.request.urlopen(req))
    print("published id=", r.get("id"))
except urllib.error.HTTPError as e:
    print("ERR", e.code, e.read().decode()[:600])