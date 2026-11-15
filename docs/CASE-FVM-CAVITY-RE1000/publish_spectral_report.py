#!/usr/bin/env python3
"""Publish the SPECTRAL.md progress report to the platform (mission 21494).

Credentials come from the environment (``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``); nothing is written to the repo.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

CASE = Path("docs/CASE-FVM-CAVITY-RE1000")
MARKDOWN = (CASE / "SPECTRAL.md").read_text()
FIGURES = [
    ("spectral_centrelines.png",
     "图 S-1 Re=1000 中心线剖面：谱解 N=48/64 与 Ghia(1982) 表 I/II 采样点"),
    ("spectral_convergence.png",
     "图 S-2 u_min/v_max 随谱阶 N 收敛；虚线=谱外推极限，点线=生产 FVM Richardson 极限，点划线=Ghia 表值"),
    ("spectral_fields.png", "图 S-3 流函数与涡量场（N=48，Re=1000）"),
]

SUMMARY = (
    "高阶（Chebyshev 谱配置 + 定常 Newton）独立参考解：u_min=-0.388443、v_max=+0.376778，"
    "与生产二阶 FVM Richardson 极限(-0.389450/+0.377626)一致到 0.26%/0.22%，"
    "与 Ghia(1982) 相距 1.45%/1.57% → 否定「两套二阶码共享二阶偏差」这一解释；"
    "顺带修掉漏斜率项导致 ψ=ω=0 平凡解的缺陷（回归测试守住）"
)


def _post(base: str, path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    auth = _post(base, "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                      "password": os.environ["TENSOR_AGENT_PASS"]})
    token = auth.get("access_token") or auth["token"]
    figures = [{"name": n, "caption": c,
                "png_base64": base64.b64encode((CASE / "figures" / n).read_bytes()).decode()}
               for n, c in FIGURES]
    out = _post(base, "/tensor/reports", {
        "title": "方腔流 Re=1000：Chebyshev 谱参考解切断「二阶族共同偏差」未决命题",
        "summary": SUMMARY,
        "scenario_key": "CASE-FVM-CAVITY-RE1000",
        "mission_id": 21494,
        "tags": ["TensorFVM", "cavity", "Re1000", "spectral", "chebyshev",
                 "verification", "Ghia1982", "newton"],
        "markdown": MARKDOWN,
        "figures": figures,
    }, token)
    print(json.dumps(out)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())