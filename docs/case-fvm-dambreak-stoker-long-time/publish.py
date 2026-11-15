#!/usr/bin/env python
"""Build and publish the platform report for the long-time dam-break study.

Reads the graded artifact (``study.json``) and the rendered ``REPORT.md``,
embeds the two PNGs, logs in with the agent credentials from the environment
and POSTs to ``/tensor/reports``.  All numbers in the markdown come from the
artifact; nothing is retyped here.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(ROOT, "src", "tensorfvm", "verification", "artifacts",
                   "shallow-water-long-time", "study.json")
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")
PAYLOAD = "/tmp/report.json"


def pct(value, digits=4):
    if value is None or value != value:
        return "n/a"
    return f"{100.0 * value:.{digits}f} %"


def request(url, body=None, token=None, timeout=60.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if body else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode()


def login():
    creds = {"username": USER, "password": PASSWORD}
    for path in ("/auth/login", "/login", "/tensor/login", "/users/login"):
        try:
            status, text = request(f"{BASE}{path}", creds)
        except Exception as exc:                       # noqa: BLE001 - probe
            print(f"  {path}: {type(exc).__name__} {exc}")
            continue
        if status < 300:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            token = (data.get("token") or data.get("access_token")
                     or data.get("data", {}).get("token"))
            if token:
                print(f"  logged in via {path}")
                return token
    return None


def build():
    payload = json.loads(open(ART).read())
    crit = payload["criteria"]
    report_md = open(os.path.join(HERE, "REPORT.md")).read()
    rows = [(k, r) for k, case in payload["cases"].items() for r in case["rows"]]
    worst_pos = max(rows, key=lambda kr: kr[1]["front_series_max_error"])
    worst_speed = max(rows, key=lambda kr: kr[1]["x_over_t_error"])
    all_pass = all(c["verdict"]["pass"] for c in payload["cases"].values())
    summary = (
        f"锚点 CASE-FVM-DAMBREAK-STOKER 的时间维度补账：dx={crit['dx']} m 固定、域随 t 增长，"
        f"把 t* 从 2 推到 10，逐点判 波位置/波速 x_f/t/静水区水位阶跃/自相似坍缩；"
        f"最差波位置 {pct(worst_pos[1]['front_series_max_error'])}、最差波速 "
        f"{pct(worst_speed[1]['x_over_t_error'])}（"
        f"{'全部判据通过' if all_pass else '存在不通过项，如实记录'}）")
    figures = []
    for name, caption in (
            ("long_time_errors.png",
             "波位置与波速相对误差随 t* 的变化（5 % 验收带）"),
            ("long_time_collapse.png",
             "自相似坐标 xi=x/t 下的剖面坍缩：各 t* 数值解 vs 精确解")):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            figures.append({"name": name, "caption": caption,
                            "png_base64": base64.b64encode(
                                open(path, "rb").read()).decode()})
    return {
        "title": "CASE-FVM-DAMBREAK-STOKER 时域有效性验证：t* 到 10（锚点 5 倍） 的波速/水位阶跃/自相似坍缩",
        "summary": summary,
        "scenario_key": "case-fvm-dambreak-stoker-long-time",
        "mission_id": 22089,
        "tags": ["TensorFVM", "shallow-water", "dam-break", "Ritter", "Stoker",
                 "self-similar", "verification", "CASE-FVM-DAMBREAK-STOKER"],
        "markdown": report_md,
        "figures": figures,
    }


def main(argv=None) -> int:
    payload = build()
    json.dump(payload, open(PAYLOAD, "w"), ensure_ascii=False)
    with open(os.path.join(HERE, "REPORT-platform.md"), "w",
              encoding="utf-8") as fh:
        fh.write(f"# {payload['title']}\n\n{payload['summary']}\n\n{payload['markdown']}\n")
    print(f"payload: {PAYLOAD} ({len(payload['markdown'])} chars, "
          f"{len(payload['figures'])} figures)")
    token = login()
    if not token:
        print("login failed; payload kept on disk")
        return 2
    status, text = request(f"{BASE}/tensor/reports", payload, token=token)
    print(f"POST /tensor/reports -> HTTP {status}: {text[:400]}")
    return 0 if status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())