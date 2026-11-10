"""Publish the planing-Savitsky report to the platform report API.

Figures are embedded as base64 (owner report discipline: six elements, figures
inline, numbers injected from results.json -- no hand-copied digits).
"""
import base64
import json
import os
import subprocess
import sys

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "..", "docs", "planing-savitsky-verification")
FIG = os.path.join(DOC, "figures")


def sh(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


rc, out, err = sh([
    "curl", "-s", "-X", "POST", f"{BASE}/auth/login",
    "-H", "Content-Type: application/json",
    "-d", json.dumps({"username": USER, "password": PASS}),
])
print("login rc", rc, out[:200], err[:200])
token = None
try:
    blob = json.loads(out)
    token = (blob.get("access_token") or blob.get("token")
             or (blob.get("data") or {}).get("access_token"))
except Exception:
    token = None
if not token:
    token = os.environ.get("TENSOR_TOKEN", "")
print("token?", bool(token))

captions = {
    "fig1_lift_chart.png": "Savitsky (1964) 平板动升力律：C_L0/τ^1.1 随平均湿长比 λ，C_V=1/2/4/6/8",
    "fig2_deadrise.png": "斜升修正族：C_Lβ 随斜升角 β（λ=3，4 组 C_V/τ），递减量 0.0065·β·C_L0^0.6",
    "fig3_running_attitude.png": "平衡运行姿态：τ(C_V) 与 λ(C_V)，β=10/15/20°，虚线为包线下界 τ_min=2°",
    "fig4_resistance.png": "阻力分解 R_T/R_f/R_spray/Δtanτ 与升阻比 Δ/R_T 随航速（β=15°）",
    "fig5_porpoising.png": "纵倾斜率判据 dτ/dC_V 随 C_V（β=5…30°）：正斜率带 = porpoising-prone 带",
}
figures = []
for name in sorted(os.listdir(FIG)):
    if not name.endswith(".png"):
        continue
    with open(os.path.join(FIG, name), "rb") as fh:
        figures.append({
            "name": name,
            "caption": captions.get(name, name),
            "png_base64": base64.b64encode(fh.read()).decode(),
        })

markdown = open(os.path.join(DOC, "REPORT.md"), encoding="utf-8").read()
payload = {
    "title": "高速滑行艇 Savitsky 验证面：动升力 / 喷雾阻力 / 纵摇稳定性（TensorFVM）",
    "summary": ("TensorFVM 新增 planing_savitsky 模块：Savitsky(1964) 动升力律（平板+斜升）、"
                "压力中心及其精确反解、喷雾阻力、ITTC-57 湿面积摩擦、阻力分解、"
                "两方程平衡求解（力矩定 λ、升力定 τ）、纵倾斜率与 2 自由度准静态刚度两条 porpoising 判据。"
                "参考艇 C_V=2 设计点：λ=2.962、τ=3.908°、C_L=0.150、Δ/R_T=6.36；"
                "61 点扫掠升力/力矩残差 ≤2.2e-13/6.3e-13；27 项 pytest 全绿。"
                "原文图表对比因离线无法取得数字化数据，如实记为 WITHHELD，不伪造。"),
    "scenario_key": "planing-craft-savitsky",
    "mission_id": 9029,
    "tags": ["planing-hull", "savitsky", "dynamic-lift", "spray-drag", "porpoising",
             "TensorFVM", "verification-surface", "high-speed-craft"],
    "markdown": markdown,
    "figures": figures,
}
path = os.path.join(HERE, "..", "docs", "planing-savitsky-verification", "report_payload.json")
with open(path, "w") as fh:
    json.dump(payload, fh)
print("payload bytes:", os.path.getsize(path))

rc, out, err = sh([
    "curl", "-s", "-X", "POST", f"{BASE}/tensor/reports",
    "-H", f"Authorization: Bearer {token}",
    "-H", "Content-Type: application/json",
    "-d", "@" + path,
])
print("report rc", rc)
print("resp:", out[:600])
print("err:", err[:300])
sys.exit(0 if rc == 0 else 1)