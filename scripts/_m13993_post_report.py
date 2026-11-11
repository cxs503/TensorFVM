#!/usr/bin/env python
"""Publish the node report to the platform report API (figures embedded)."""
import base64
import json
import os
import glob

import urllib.request

API = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
SECRET = os.environ["TENSOR_AGENT_PASS"]

HERE = "/root/tp-agent/work/m13993/docs/P447W2-N01-vof-three-case"


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main():
    tok = post("/auth/login", {"username": USER, "password": SECRET})
    token = tok.get("access_token") or tok.get("token")
    if not token:
        raise SystemExit("login failed: %s" % list(tok)[:5])

    markdown = open(os.path.join(HERE, "report.md")).read()
    summary_path = os.path.join(HERE, "artifacts", "vof_threecase.json")
    summary = json.load(open(summary_path))
    crit = summary["criteria"]
    verdicts = ", ".join(f"{k.split('_')[0].upper()}={'PASS' if v.get('pass') else 'FAIL'}"
                         for k, v in crit.items())

    captions = {
        "fig1_static_fields.png": "静止气泡终态 α 与 |u| 场（balanced CSF + height-function 曲率）",
        "fig2_static_current.png": "归一化虚假流速时程与分辨率依赖性，含 legacy/κ=div(n) 对照与 0.01 判据线",
        "fig3_zalesak_fields.png": "Zalesak 终态 α 场（N=128）：白线为初值等值面，青虚线为共转缺口掩膜",
        "fig4_zalesak_notch.png": "缺口面积误差与周长误差随分辨率变化（360°/1000°）",
        "fig5_dam_front.png": "溃坝波前 x*–t* 曲线叠 Martin & Moyce 实验点，右为判据时刻误差带",
        "fig6_dam_fields.png": "溃坝终态 α 场与自由面位置",
        "fig7_dam_mass.png": "溃坝全局质量账本与 0.1 % 判据线",
    }
    figures = []
    for path in sorted(glob.glob(os.path.join(HERE, "artifacts", "figures", "*.png"))):
        name = os.path.basename(path)
        figures.append({
            "name": name,
            "caption": captions.get(name, name),
            "png_base64": base64.b64encode(open(path, "rb").read()).decode(),
        })

    payload = {
        "title": "P447W2-N01 — FVM 两相 VOF 三算例验证面（溃坝波前/静止气泡/Zalesak 缺口）",
        "summary": ("新建 src/tensorfvm/vof_threecase 三算例验证面：静止气泡归一化虚假流速、"
                    "Zalesak 缺口面积（360°/1000° 共转掩膜）、Martin & Moyce 耦合溃坝。"
                    "机器裁决：" + verdicts + "。FAIL 项（波前 -40%@t=0.1 s、静止气泡长时发散、"
                    "网格非单调）逐条附证据，未覆盖。"),
        "scenario_key": "fvm_vof_two_phase_verification",
        "mission_id": 13993,
        "tags": ["TensorFVM", "VOF", "two-phase", "dam-break", "Martin-Moyce",
                 "Zalesak", "spurious-current", "verification", "P447W2-N01"],
        "markdown": markdown,
        "figures": figures,
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out)[:400])


if __name__ == "__main__":
    main()