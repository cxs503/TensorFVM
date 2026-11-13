"""Publish the Sod(1978) survey reproduction report to the platform (six elements)."""
import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main():
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token")
    md = open("docs/repro-fvm-sod1978-survey/REPORT.md").read()
    caps = {
        "survey_profiles.png": "Sod 激波管五种 surveyed 格式 + 精确解（rho/u/p 三面板，nx=100，t=0.2）",
        "survey_error_contact.png": "左：L1(rho) 与接触 10-90% 厚度；右：接触与激波局部放大",
        "survey_order.png": "精确熵波上 L1 误差 vs h 双对数（实测设计阶）",
    }
    figs = []
    for name, cap in caps.items():
        p = os.path.join("docs/repro-fvm-sod1978-survey", name)
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(open(p, "rb").read()).decode()})
    body = {
        "title": "REPRO-FVM-SOD1978 第5轮：Sod(1978)综述的五种经典FD格式横向复现",
        "summary": ("按论文描述原样实现 LF/Rusanov/Godunov/LW/MacCormack 五格式，同网格同CFL 对齐精确解："
                    "S1 通量预算闭合 1e-16；S2 实测阶 一阶0.93-0.99、二阶2.00；S3 激波<=1.61%(<3%)、"
                    "接触<=4.21%(<6%)；S4 一阶密度不越界、LW 越界指数0.343；S5 LF 16.4 cells vs LW 5.1 cells；"
                    "纯 MacCormack t=0.0216 失正性=honest FAIL 入账。"),
        "scenario_key": "repro-fvm-sod1978-survey",
        "mission_id": 18399,
        "tags": ["TensorFVM", "REPRO-FVM-SOD1978", "verification", "Sod1978"],
        "markdown": md,
        "figures": figs,
    }
    out = post("/tensor/reports", body, token)
    print(json.dumps({k: out[k] for k in out if k in ("id", "status", "title")}, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
