"""Publish the order-audit report to the platform (numbers injected, zero copying).

Usage:  PYTHONPATH=src python docs/case-fvm-dambreak-order-audit/publish_report.py
Needs TENSOR_API_BASE / TENSOR_AGENT_USER / TENSOR_AGENT_PASS in the environment.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACT = (HERE.parents[1] / "src" / "tensorfvm" / "verification" / "artifacts"
            / "shallow-water-order-audit" / "summary.json")

FIGURES = [
    ("fig_profiles.png",
     "t*=2 depth profile vs the Ritter/Stoker closed forms at n=800; the shaded band is "
     "the analytically derived fan-interior window in which the order is graded (the dry "
     "front, the bore and both fan kinks are outside it)."),
    ("fig_convergence.png",
     "log-log interior L1 error vs dx for the four studies plus the smooth-bump "
     "self-convergence; the dam-step runs follow slope 1, the dry-bed restart follows "
     "slope 2, the wet-bed restart is non-monotone and is reported ungradable."),
    ("fig_odx_collapse.png",
     "Ritter dam break: (h_num - h_exact)/dx against the similarity coordinate xi for "
     "n=400/800/1600. The three curves coincide, so the interior error is an O(dx) "
     "self-similar limit and not an estimator artefact."),
]


def _token() -> str:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{base}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data.get("access_token") or data["token"]


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    token = _token()
    rec = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    markdown = (HERE / "REPORT.md").read_text(encoding="utf-8")
    figures = []
    for name, caption in FIGURES:
        png = base64.b64encode((HERE / name).read_bytes()).decode()
        figures.append({"name": name, "caption": caption, "png_base64": png})
    payload = {
        "title": ("CASE-FVM-DAMBREAK-STOKER 收敛阶审计：算例池 order=2 的证据与适用范围"
                  f"（光滑区 p={rec['measured']['smooth_regime']['order_asymptotic']:.2f}，"
                  f"干床 step p={rec['measured']['dam_break']['ritter-dry/step']['order_asymptotic']:.2f}，"
                  f"湿床 step p={rec['measured']['dam_break']['stoker-wet/step']['order_asymptotic']:.2f}）"),
        "summary": ("为锚点算例池条目里无证据的 solver.order=2 补上可失败实验：光滑隆起自收敛 "
                    f"p={rec['measured']['smooth_regime']['order_asymptotic']:.2f}（算子确为二阶）；"
                    f"干床间断初值 p={rec['measured']['dam_break']['ritter-dry/step']['order_asymptotic']:.2f}，"
                    f"同解连续初值重启 p={rec['measured']['dam_break']['ritter-dry/smooth']['order_asymptotic']:.2f}"
                    "→ 失阶属于初值数据而非格式；湿床 p="
                    f"{rec['measured']['dam_break']['stoker-wet/step']['order_asymptotic']:.2f} 且重启序列非单调"
                    "（如实 WITHHELD，不引用阶）；err/dx 塌缩 "
                    f"{rec['fan']['ritter-dry/step']['collapse_finest_relative']*100:.1f}% 证明是一阶自相似极限。"
                    "5 % 验收不变，条目 order 声明得到适用范围绑定。"),
        "scenario_key": "dam_break_order_audit",
        "mission_id": 21759,
        "tags": ["TensorFVM", "shallow-water", "dam-break", "Ritter", "Stoker",
                 "order-of-accuracy", "verification-debt", "case-pool",
                 "CASE-FVM-DAMBREAK-STOKER"],
        "markdown": markdown,
        "figures": figures,
    }
    req = urllib.request.Request(
        f"{base}/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.load(resp)
    print(json.dumps({"report_id": out.get("id") or out.get("report_id"),
                      "title": payload["title"][:60],
                      "figures": len(figures)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())